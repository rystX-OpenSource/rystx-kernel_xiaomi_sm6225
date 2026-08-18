/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Multi-level feedback queue classification for the fair class.
 *
 * Ported from scx_mlfq, a sched_ext scheduler by galpt:
 *   https://github.com/galpt/scx_mlfq
 *
 * scx_mlfq keeps three per-CPU dispatch queues, each ordered by virtual
 * time, and serves them with a quota so that the interactive queue is
 * drained first while the batch queue keeps a guaranteed share of every
 * dispatch batch. A task's queue is chosen by a burst gauge: running time
 * accumulated since the task last slept long enough to be refunded for it,
 * bounded above, which is a direct measure of how long a turn the task takes
 * before giving the CPU back.
 *
 * The fair class has a single virtual-time ordered tree per runqueue, so
 * the three queues are not reproduced as three trees. Instead the queue
 * level selects the task's EEVDF request size r_i:
 *
 *	Q1 -> 1ms	Q2 -> 2ms	Q3 -> 4ms
 *
 * EEVDF derives a task's virtual deadline from its request size,
 * vd_i = ve_i + r_i/w_i, so a task in a lower queue carries an earlier
 * deadline and is picked sooner and more often, for shorter turns. That is
 * precisely the latency ordering the dispatch quota produced, and it comes
 * with two properties the quota did not have: the service each task
 * receives still follows its weight rather than its queue, and the deadline
 * order itself bounds how long any task waits. A task that runs advances its
 * deadline by r_i/w_i at the end of every request, while a task that waits
 * keeps the deadline it was given, so the tasks running ahead of it push their
 * own deadlines past its, and it is picked. That is the starvation bound the
 * guaranteed batch-queue share of each dispatch batch provided in scx_mlfq, so
 * no quota is carried over.
 *
 * A level's request size is also the budget of a CBS server, and a task that
 * blocks before its request runs out leaves part of that budget unspent. The
 * remainder is reclaimed for the level that owned it, so the next task placed at
 * that level is granted the leftover on top of its own request; see struct
 * mlfq_bonus. That is the only thing besides the classification that changes a
 * request, and it changes it by at most the level's own size.
 *
 * Everything below the classifier is unchanged EEVDF: placement, lag,
 * the tree, throttling, group scheduling and load balancing.
 *
 * The port is split to mirror the upstream sources, so a change made
 * upstream lands in the file here that corresponds to it:
 *
 *	mlfq.h		 <- src/bpf/intf.h	 constants, state, the math
 *	mlfq_classify.c	 <- src/bpf/classify.bpf.c
 *					 the classification state machines
 *	mlfq_main.c	 <- src/bpf/main.bpf.c	 the system-wide state
 *	mlfq_stats.c	 <- src/stats.rs, src/webui.rs, ui/index.html
 *					 the /proc dashboard
 *
 * The hooks themselves stay in fair.c, because they have to sit inside the
 * EEVDF paths they observe; each is a single call whose policy lives here.
 */
#ifndef _KERNEL_SCHED_MLFQ_H
#define _KERNEL_SCHED_MLFQ_H

#include <linux/compiler.h>
#include <linux/kernel.h>
#include <linux/limits.h>
#include <linux/log2.h>
#include <linux/percpu.h>
#include <linux/sched.h>
#include <linux/time64.h>
#include <linux/types.h>

/*
 * Request size per queue level, in nanoseconds. The values are exact powers
 * of two, nominally 1ms, 2ms and 4ms, so that the gauge decay below divides
 * by a queue's period with a shift instead of a divide.
 */
#define MLFQ_SLICE_Q1_NS		(1ULL << 20)
#define MLFQ_SLICE_Q2_NS		(1ULL << 21)
#define MLFQ_SLICE_Q3_NS		(1ULL << 22)

/*
 * Ceiling of the burst gauge. A task that has run for this long without
 * being refunded for a sleep is as CPU-bound as the gauge can express. It is
 * four times the CPU-bound edge, so a saturated gauge sits well past it.
 */
#define MLFQ_GAUGE_MAX_NS		(1ULL << 23)

/*
 * Classification thresholds on the gauge. At or below the low threshold a
 * task is interactive, at or above the high threshold it is CPU-bound, and in
 * between it is left in the default queue.
 *
 * The high edge is the batch request size, which is what makes the gauge and
 * the edges commensurable: a task is CPU-bound once it has accumulated a
 * whole batch-level request of unrefunded running time.
 */
#define MLFQ_THRESH_LOW_NS		250000ULL
#define MLFQ_THRESH_HIGH_NS		4000000ULL

/*
 * Period multiplier for the refund below. A level's server is (Q_i, P_i) with
 * P_i = MLFQ_CBS_PERIOD_MULT * Q_i, so half of every period is reserved,
 * and the period is a power of two because the request size is.
 */
#define MLFQ_CBS_PERIOD_MULT		2

/*
 * A sleep shorter than this is taken as a task waiting on something
 * rather than being done, and boosts it into the interactive queue.
 */
#define MLFQ_SHORT_SLEEP_NS		32000000ULL

/* At most one short-sleep boost per task per this interval. */
#define MLFQ_BOOST_INTERVAL_NS		2000000ULL

/*
 * A sleep at most this long counts towards the consecutive-short-sleep
 * counter that gates promotion. It is much tighter than the boost window
 * above: the boost only has to survive until the task's next request runs
 * out, while a promotion changes the task's level until something demotes it
 * again, so it asks for a clearer pattern of brief sleeps.
 */
#define MLFQ_HYSTERESIS_SLEEP_NS	4000000ULL

/*
 * A stay of this long in Q2 or Q3 elevates the task to Q1. EEVDF already
 * bounds the wait through the deadline order, so this only catches a task
 * whose classification has gone stale.
 */
#define MLFQ_AGING_PERIOD_NS		1000000000ULL

/* Consecutive short sleeps required before promoting a level. */
#define MLFQ_PROMOTE_WAKES		2
/* Consecutive request exhaustions required before demoting a level. */
#define MLFQ_DEMOTE_EXHAUSTIONS		8

/*
 * The request a wakeup gets when it is owed a preemption, in place of the
 * request its level would have given it. It is a bounded burst: the wakeup
 * displaces what is running, does the small piece of work it woke up to do,
 * and hands the CPU back at the end of the burst rather than holding it for a
 * full level's request. Its level's request governs the continuation, from
 * the next request that update_deadline() issues.
 *
 * scx_mlfq also rate limits the same-level case behind a minimum run of the
 * task being displaced, MLFQ_SAMEQ_PREEMPT_MIN_RUN_NS. That is not carried
 * over: upstream fixes it at zero and never derives anything else from it, so
 * the guard cannot block a preemption there, and a constant zero here would
 * only add a comparison that no value can fail. See mlfq_preempt_owed().
 */
#define MLFQ_PREEMPT_SLICE_NS		150000ULL

/**
 * mlfq_time_before - wrap-safe comparison of two rq clock readings
 * @a: first timestamp
 * @b: second timestamp
 *
 * Returns %true when @a precedes @b. The rq clock is monotonic within a
 * runqueue but a task that migrated may compare readings taken on
 * different CPUs, so treat the difference as signed and let a backwards
 * delta read as "not before" rather than as a very long interval.
 */
static inline bool mlfq_time_before(u64 a, u64 b)
{
	return (s64)(a - b) < 0;
}

/**
 * mlfq_elapsed - interval between two rq clock readings, clamped at zero
 * @now: the later reading
 * @then: the earlier reading
 *
 * Returns the number of nanoseconds from @then to @now, or 0 if @then is
 * unset or does not precede @now.
 */
static inline u64 mlfq_elapsed(u64 now, u64 then)
{
	if (!then || !mlfq_time_before(then, now))
		return 0;

	return now - then;
}

/**
 * mlfq_queue_slice - EEVDF request size for a queue level
 * @queue: the queue level, 1..3
 *
 * Anything outside the interactive and batch levels, including the zero a
 * freshly allocated task_struct carries, maps to the default request.
 */
static inline u64 mlfq_queue_slice(u8 queue)
{
	if (queue == MLFQ_Q_INTERACTIVE)
		return MLFQ_SLICE_Q1_NS;
	if (queue == MLFQ_Q_BATCH)
		return MLFQ_SLICE_Q3_NS;

	return MLFQ_SLICE_Q2_NS;
}

/**
 * mlfq_queue_period - CBS server period for a queue level
 * @queue: the queue level, 1..3
 *
 * Twice the level's request size, so half of every period is reserved. Like
 * the request size it is an exact power of two, which is what lets
 * mlfq_gauge_decay() divide by it with a shift.
 */
static inline u64 mlfq_queue_period(u8 queue)
{
	return mlfq_queue_slice(queue) * MLFQ_CBS_PERIOD_MULT;
}

/**
 * mlfq_gauge_climb - add a stretch of running time to the gauge
 * @g: current gauge value
 * @delta: nanoseconds of running time to account for
 *
 * The gauge is running time, so the climb is an addition, saturating at
 * MLFQ_GAUGE_MAX_NS. Where the exponential gauge this replaced approached its
 * ceiling asymptotically and took a divide to advance, this is a bounded add,
 * and what it measures is directly comparable with the thresholds: both are
 * durations, and the CPU-bound edge is a batch-level request.
 *
 * Being an addition also makes it exactly indifferent to how the running time
 * arrives. min(min(g + a, MAX) + b, MAX) is min(g + a + b, MAX) for every g, a
 * and b, so accounting a run segment in pieces gives the same gauge as
 * accounting it whole. scx_mlfq climbs once per run segment, from
 * ops.stopping(); the fair class climbs per update_curr() delta, which sums to
 * the same segment, and now does so with no error at all rather than to first
 * order.
 *
 * Returns the updated gauge.
 */
static inline u64 mlfq_gauge_climb(u64 g, u64 delta)
{
	if (g >= MLFQ_GAUGE_MAX_NS || delta > MLFQ_GAUGE_MAX_NS - g)
		return MLFQ_GAUGE_MAX_NS;

	return g + delta;
}

/**
 * mlfq_gauge_decay - refund the gauge for a sleep
 * @g: current gauge value
 * @sleep_ns: nanoseconds slept
 * @q_i: the level's request size, its CBS budget
 * @p_i: the level's CBS period, MLFQ_CBS_PERIOD_MULT * @q_i
 *
 * Every whole server period spent asleep refunds one whole budget:
 *
 *	periods = sleep_ns / p_i
 *	g -= min(periods * q_i, g)
 *
 * The divide is a shift because @p_i is a power of two. The refund is
 * therefore quantised to multiples of @q_i, so a sleep shorter than one period
 * refunds nothing, and a sleep of two gauge ceilings refunds the whole gauge
 * at any level.
 *
 * That quantisation is the point rather than a rounding artefact: it is what
 * makes the gauge a measure of the task's turn length. A task that keeps its
 * level is refunded exactly what a task of that level is entitled to sleep
 * off, so only running time it did not sleep off accumulates.
 *
 * The budget and the period are passed in because they belong to the level the
 * task was last served by, which is the caller's business to know; see
 * mlfq_classify_wakeup(), which reads them before any promotion moves the task.
 *
 * Returns the updated gauge.
 */
static inline u64 mlfq_gauge_decay(u64 g, u64 sleep_ns, u64 q_i, u64 p_i)
{
	u64 periods, refund;

	if (!p_i)
		return g;

	/*
	 * ilog2() of a power of two is its shift count. It is the kernel's own
	 * spelling of the __builtin_ctzll() scx_mlfq uses here, and lowers to
	 * the same single instruction on arm64, with no divide and no libcall.
	 */
	periods = sleep_ns >> ilog2(p_i);

	/*
	 * A sleep long enough for the product below to overflow has already
	 * refunded far more than any gauge can hold, so a period count past
	 * the ceiling itself empties the gauge without multiplying anything:
	 * at that count the refund exceeds MLFQ_GAUGE_MAX_NS whatever the
	 * level, since the smallest budget is larger than one nanosecond.
	 */
	if (periods > MLFQ_GAUGE_MAX_NS)
		return 0;

	refund = periods * q_i;

	return g > refund ? g - refund : 0;
}

/**
 * mlfq_reenq_cnt_step - saturating increment of the exhaustion counter
 * @cnt: the current consecutive-exhaustion count
 *
 * Saturates at MLFQ_DEMOTE_EXHAUSTIONS rather than at the width of the field,
 * so a task that has nowhere left to be demoted to keeps a count that still
 * reads as "sustained" instead of wrapping through zero and losing its
 * history.
 */
static inline u8 mlfq_reenq_cnt_step(u8 cnt)
{
	return cnt < MLFQ_DEMOTE_EXHAUSTIONS ? cnt + 1 : MLFQ_DEMOTE_EXHAUSTIONS;
}

/**
 * struct mlfq_bonus - a level's reclaim pool
 * @ns:		request time granted to tasks of this level and never used,
 *		because they blocked before their request ran out. Bounded by
 *		the level's own request size.
 * @since:	the rq clock at the last deposit, which is what the pool decays
 *		from. Zero exactly when @ns is zero.
 *
 * Fair-share CBS reclaim: a level's server is (Q_i, P_i), and a task that
 * blocks early hands back what it did not use so that another task of the same
 * level can have it. The pool is only ever the unused part of budget the level
 * had already been granted, so reclaiming it hands out no bandwidth the level
 * did not have, and it is never shared across levels: what an interactive task
 * gives back can only go to another interactive task, so no latency budget ever
 * reaches a CPU-bound one.
 *
 * scx_mlfq keeps one pool per level for the whole machine, inside the queue_ctx
 * its dispatch queues are described by, and reaches it with relaxed atomics.
 * Here there is a pool per level per runqueue, in struct mlfq_pcpu, for two
 * reasons. The slack is idle capacity on one CPU, and a task queued on another
 * CPU never had that capacity available to it, so a machine-wide pool would
 * hand it somewhere it cannot be spent. And both ends of the transfer happen
 * under one runqueue's lock, which a per-runqueue pool is therefore already
 * serialised by, so the atomics upstream needs are not needed here.
 */
struct mlfq_bonus {
	u64	ns;
	u64	since;
};

/**
 * mlfq_fcbs_slack - the unused part of a request
 * @grant_end: the sum_exec_runtime at which the request runs out
 * @used: the task's sum_exec_runtime now
 *
 * A task that gives the CPU back before its request runs out leaves the
 * remainder unused, and that remainder is what the pool is made of.
 *
 * scx_mlfq compares the grant against the length of the run segment, because a
 * sched_ext task is dispatched with a slice grant and runs until it is used up
 * or the task blocks, so the two are the same measurement. Under EEVDF a task
 * can be preempted in the middle of a request and come back to the rest of it,
 * so a run segment is not a request; what is compared here instead is the
 * task's own sum_exec_runtime against the value it would have reached had the
 * request been used in full. That is the same subtraction against a total that
 * survives preemption and migration, and it needs no clock.
 *
 * A zero @grant_end means the task has no request to give anything back from,
 * which is how the preemption burst declines to donate; the guarded subtraction
 * returns zero for it without a case of its own. See mlfq_preempt_burst().
 *
 * Returns the unused nanoseconds.
 */
static inline u64 mlfq_fcbs_slack(u64 grant_end, u64 used)
{
	return grant_end > used ? grant_end - used : 0;
}

/**
 * mlfq_fcbs_deposit - give a level's pool the unused part of a request
 * @b: the level's pool
 * @slack: the unused nanoseconds
 * @q_i: the level's request size, which caps the pool
 * @now: current rq clock
 *
 * The cap is what bounds everything downstream: a pool that holds at most one
 * request means a reclaimed request is at most twice a level's own, so the
 * bound is structural and no consumer has to clamp.
 *
 * @since is restamped on every deposit, so the decay in mlfq_fcbs_consume()
 * measures how long the pool has gone unclaimed rather than how long ago it
 * first held anything. scx_mlfq only stamps it when it is unset and never
 * clears it, which anchors the decay at the first deposit after boot and so
 * subtracts the whole uptime from every later pool; its own comment calls the
 * field "the scx_bpf_now() of the last deposit", which is what this does.
 */
static inline void mlfq_fcbs_deposit(struct mlfq_bonus *b, u64 slack, u64 q_i,
				     u64 now)
{
	b->ns = min(b->ns + slack, q_i);
	b->since = now;
}

/**
 * mlfq_fcbs_consume - take a level's pool into a request
 * @b: the level's pool
 * @slice: the level's own request size
 * @now: current rq clock
 *
 * The pool is worth less the longer it goes unclaimed, because what it stands
 * for is capacity that was free at the moment it was given back. It is decayed
 * by the time since the last deposit and emptied, so it is granted once and to
 * one task; nothing accumulates across levels and nothing is granted twice.
 *
 * Returns the request to grant, which is at most twice @slice.
 */
static inline u64 mlfq_fcbs_consume(struct mlfq_bonus *b, u64 slice, u64 now)
{
	u64 idle, bonus;

	if (!b->ns)
		return slice;

	idle = mlfq_elapsed(now, b->since);
	bonus = b->ns > idle ? b->ns - idle : 0;

	b->ns = 0;
	b->since = 0;

	return slice + bonus;
}

/**
 * mlfq_boost_pending - does this wakeup earn the interactive boost
 * @ctx: the task's classification state
 * @sleep_ns: how long the task slept, 0 if unknown
 * @io_wait: the task was sleeping on I/O
 * @now: current rq clock
 *
 * A wakeup out of I/O, or out of a sleep short enough to look like waiting on
 * a peer rather than being finished, is treated as interactive. This stands
 * in for the wakeup fast paths a scheduler would otherwise need to recognise
 * one at a time, so it deliberately does not care what the task was waiting
 * on. It is rate limited to one boost per MLFQ_BOOST_INTERVAL_NS per task so
 * a burst of wakeups cannot chain boosts.
 */
static inline bool mlfq_boost_pending(const struct mlfq_ctx *ctx, u64 sleep_ns,
				      bool io_wait, u64 now)
{
	if (!io_wait && !(sleep_ns && sleep_ns <= MLFQ_SHORT_SLEEP_NS))
		return false;

	if (!ctx->last_boost_at)
		return true;

	return mlfq_time_before(ctx->last_boost_at + MLFQ_BOOST_INTERVAL_NS,
				now);
}

/**
 * mlfq_promote_on_wakeup - hysteresis promotion at wakeup
 * @ctx: the task's classification state
 * @sleep_ns: how long the task slept
 * @t_l: effective low edge
 * @t_h: effective high edge
 *
 * A single wakeup does not move a task up a level. The gauge has to be well
 * inside the band below, at half the edge that would have placed the task
 * there, and the task has to have slept briefly MLFQ_PROMOTE_WAKES times in a
 * row. Requiring the crossing of a band rather than a point is what keeps a
 * task whose gauge sits near an edge from changing level on every wakeup.
 *
 * The counter saturates where scx_mlfq's increments without a bound. Both
 * behave the same up to the comparison, which only asks for two, but a task
 * whose gauge keeps it below the promotion gate can sleep briefly indefinitely,
 * and an unbounded u8 wraps through zero and throws the run away.
 *
 * Returns %true if the task moved up a level.
 */
static inline bool mlfq_promote_on_wakeup(struct mlfq_ctx *ctx, u64 sleep_ns,
					  u64 t_l, u64 t_h)
{
	bool promoted = false;

	if (sleep_ns > MLFQ_HYSTERESIS_SLEEP_NS)
		ctx->wake_cnt = 0;
	else if (ctx->wake_cnt < U8_MAX)
		ctx->wake_cnt++;

	if (ctx->wake_cnt >= MLFQ_PROMOTE_WAKES) {
		if (ctx->queue == MLFQ_Q_DEFAULT && ctx->g < t_l / 2) {
			ctx->queue = MLFQ_Q_INTERACTIVE;
			promoted = true;
		} else if (ctx->queue == MLFQ_Q_BATCH && ctx->g < t_h / 2) {
			ctx->queue = MLFQ_Q_DEFAULT;
			promoted = true;
		}
	}

	if (promoted)
		ctx->wake_cnt = 0;

	return promoted;
}

/**
 * mlfq_demote_on_runout - hysteresis demotion on request exhaustion
 * @ctx: the task's classification state
 * @t_h: effective high edge
 *
 * Demotion needs a sustained run rather than a single exhausted request:
 * MLFQ_DEMOTE_EXHAUSTIONS exhaustions in a row, roughly 8ms at the interactive
 * request size, while the gauge stays at or above the CPU-bound edge. A task
 * that sleeps in between is boosted at its wakeup, which clears the counter, so
 * something bursty like a decoder keeps its level for the whole burst, while a
 * task that simply never sleeps accumulates the counter and is demoted.
 *
 * Returns %true if the task moved down a level.
 */
static inline bool mlfq_demote_on_runout(struct mlfq_ctx *ctx, u64 t_h)
{
	ctx->reenq_cnt = mlfq_reenq_cnt_step(ctx->reenq_cnt);

	if (ctx->queue != MLFQ_Q_INTERACTIVE && ctx->queue != MLFQ_Q_DEFAULT)
		return false;
	if (ctx->g < t_h || ctx->reenq_cnt < MLFQ_DEMOTE_EXHAUSTIONS)
		return false;

	ctx->queue++;
	ctx->reenq_cnt = 0;

	return true;
}

/**
 * mlfq_wakeup_pending - is this enqueue the one a wakeup arrived on
 * @ctx: the task's classification state
 *
 * The marker scx_mlfq kept as MLFQ_TF_ENQ_WAKEUP, set in ops.enqueue() when
 * sched_ext handed it a wakeup and cleared once the task reached a CPU.
 *
 * Upstream dropped the flag once nothing but ops.enqueue() consulted it, since
 * there enq_flags carries SCX_ENQ_WAKEUP directly. It is kept here because the
 * flags do not survive every path into placement: requeue_delayed_entity()
 * places a task that woke up, with no flags at all. The mark is set once per
 * wakeup, on the way in, so it is true wherever that wakeup ends up being
 * placed.
 */
static inline bool mlfq_wakeup_pending(const struct mlfq_ctx *ctx)
{
	return ctx->wake_pending;
}

/**
 * mlfq_preempt_owed - may this wakeup displace what is running
 * @p: the waking task
 * @running: the task it would displace
 * @earlier: @p's virtual deadline is ahead of @running's
 *
 * scx_mlfq asks this in ops.enqueue(), of the task running on the CPU the
 * wakeup is heading for, and answers it in two parts.
 *
 * A wakeup from a lower-numbered level preempts unconditionally. That is the
 * whole point of the levels: a queue that is drained first is one whose tasks
 * do not wait behind a task from a queue that is drained later.
 *
 * Within one level there is no such ordering to appeal to, so the levels
 * defer to virtual time, and the answer is the one a dispatch pass over the
 * level would have given: the wakeup preempts when its deadline is the earlier
 * of the two. The interactive level is the exception, and preempts a peer
 * unconditionally -- everything there has been classified as doing short pieces
 * of work between sleeps, so whichever of them runs next will hand the CPU back
 * shortly either way, and taking the wakeup first is what the level exists for.
 *
 * A wakeup from a higher-numbered level never preempts.
 *
 * Returns %true if the preemption is owed.
 */
static inline bool mlfq_preempt_owed(struct task_struct *p,
				     struct task_struct *running, bool earlier)
{
	if (p->mlfq.queue != running->mlfq.queue)
		return p->mlfq.queue < running->mlfq.queue;

	return p->mlfq.queue == MLFQ_Q_INTERACTIVE || earlier;
}

/**
 * mlfq_reset_classification - start a task's classification from scratch
 * @ctx: the task's classification state
 *
 * A task with no history lands in the default queue with an empty gauge, so
 * its first few stretches of running time decide where it belongs.
 *
 * @last_qid is deliberately not touched: it is not classification state but a
 * record of a decrement still owed to a runqueue counter, and one caller,
 * switched_to_fair(), can run with the task already queued and counted.
 */
static inline void mlfq_reset_classification(struct mlfq_ctx *ctx)
{
	ctx->g = 0;
	ctx->last_sleep_at = 0;
	ctx->queued_at = 0;
	ctx->last_boost_at = 0;
	ctx->grant_end_ns = 0;
	ctx->wake_pending = 0;
	ctx->queue = MLFQ_Q_DEFAULT;
	ctx->reenq_cnt = 0;
	ctx->wake_cnt = 0;
}

/*
 * Classification entry points, out of line in kernel/sched/mlfq_classify.c.
 * @now is the rq clock the caller already holds.
 */
void mlfq_classify_enqueue(struct task_struct *p, u64 now, bool wakeup);
void mlfq_classify_runout(struct task_struct *p, u64 now);

/**
 * enum mlfq_stat_item - the events the classifier counts
 *
 * The port of scx_mlfq's struct mlfq_stats, restricted to the events that
 * exist here. The counters scx_mlfq keeps for its dispatch queues, its BPF
 * enqueue early-returns, its callback latency histogram and its realtime
 * takeover hook have no counterpart in the fair class and are not carried
 * over; kernel/sched/mlfq_stats.c says so where a reader would look for them.
 */
enum mlfq_stat_item {
	MLFQ_STAT_ON_CPU,		/* tasks given a CPU */
	MLFQ_STAT_TOTAL_RUNTIME,	/* nanoseconds of service, summed */
	MLFQ_STAT_Q1_PLACEMENTS,
	MLFQ_STAT_Q2_PLACEMENTS,
	MLFQ_STAT_Q3_PLACEMENTS,
	MLFQ_STAT_PROMOTIONS,
	MLFQ_STAT_DEMOTIONS,
	MLFQ_STAT_AGING_BOOSTS,
	MLFQ_STAT_SHORT_SLEEP_BOOSTS,
	MLFQ_STAT_PREEMPTION_KICKS,
	MLFQ_STAT_FCBS_GRANTS,		/* requests enlarged from a pool */
	MLFQ_STAT_FCBS_SLACK_EVENTS,	/* deposits into a pool */
	MLFQ_NR_STATS,
};

/**
 * struct mlfq_pcpu - per-CPU classifier state
 * @stat:	counters, summed over CPUs by the reader. Accumulated on the
 *		CPU that runs the event, so no counter is ever a shared line.
 * @q_runnable:	tasks of each level currently queued on this CPU's runqueue,
 *		indexed by level, so slot 0 is unused. Written only under this
 *		runqueue's lock, by the enqueue and dequeue hooks.
 * @bonus:	the reclaim pool of each level on this runqueue, indexed by
 *		level, so slot 0 is unused. Written only under this runqueue's
 *		lock, which is what serialises it; see struct mlfq_bonus.
 * @wake_total:	wakeup enqueues seen on this CPU since boot, for the reader.
 *
 * scx_mlfq keeps the same split: one per-CPU array for the counters, and its
 * runnable occupancy derived from per-task ownership records rather than from
 * a shared total. Its wakeup total is per-CPU for the same reason -- nothing on
 * a scheduling path should have to touch a shared line to count an event. The
 * reclaim pools are per-CPU here where upstream shares them; struct mlfq_bonus
 * says why.
 */
struct mlfq_pcpu {
	u64			stat[MLFQ_NR_STATS];
	u32			q_runnable[MLFQ_NR_QUEUES + 1];
	struct mlfq_bonus	bonus[MLFQ_NR_QUEUES + 1];
	u64			wake_total;
};

DECLARE_PER_CPU(struct mlfq_pcpu, mlfq_pcpu);

/**
 * mlfq_bonus_of - the reclaim pool of a level on a runqueue
 * @cpu: the runqueue, whose lock the caller holds
 * @queue: the queue level, 1..3
 *
 * Returns %NULL for a level outside 1..3, including the zero a freshly
 * allocated task_struct carries, which is scx_mlfq's failed queue lookup and is
 * what its callers fall back to the plain request size on.
 */
static inline struct mlfq_bonus *mlfq_bonus_of(int cpu, u8 queue)
{
	if (!queue || queue > MLFQ_NR_QUEUES)
		return NULL;

	return &per_cpu(mlfq_pcpu, cpu).bonus[queue];
}

/*
 * Both of these are called from the scheduling paths with a runqueue lock
 * held, so preemption is already off and the cheaper form is the correct one.
 * The count lands on the CPU that ran the event, which is not necessarily the
 * CPU whose runqueue the event was about; that only matters to a reader, and
 * the reader sums over every CPU.
 */
static inline void mlfq_stat_add(enum mlfq_stat_item item, u64 val)
{
	__this_cpu_add(mlfq_pcpu.stat[item], val);
}

static inline void mlfq_stat_inc(enum mlfq_stat_item item)
{
	mlfq_stat_add(item, 1);
}

/* Placement counter for a level, in the order of enum mlfq_stat_item. */
static inline enum mlfq_stat_item mlfq_placement_stat(u8 queue)
{
	if (queue == MLFQ_Q_INTERACTIVE)
		return MLFQ_STAT_Q1_PLACEMENTS;
	if (queue == MLFQ_Q_BATCH)
		return MLFQ_STAT_Q3_PLACEMENTS;

	return MLFQ_STAT_Q2_PLACEMENTS;
}

/**
 * mlfq_runnable_enter - count a task into its level on a runqueue
 * @cpu: the runqueue the task is being queued on
 * @ctx: the task's classification state, already reclassified
 *
 * Records which level was counted in @ctx->last_qid, so the matching exit
 * decrements the same slot even if the task is reclassified while it waits.
 * That makes the occupancy the level the task was *placed* at, which is what
 * scx_mlfq's runnable gauges report and what the request sizes were chosen
 * from. A task can only be dequeued from the runqueue it was queued on, so
 * the paired exit always runs on this same @cpu.
 *
 * Also the placement counter: one placement per enqueue, at the level the
 * classifier just picked.
 */
static inline void mlfq_runnable_enter(int cpu, struct mlfq_ctx *ctx)
{
	struct mlfq_pcpu *pc = &per_cpu(mlfq_pcpu, cpu);
	u8 queue = ctx->queue;

	if (!queue || queue > MLFQ_NR_QUEUES)
		return;

	mlfq_stat_inc(mlfq_placement_stat(queue));

	/* Already counted: a hook was missed, so do not double count. */
	if (ctx->last_qid)
		return;

	pc->q_runnable[queue]++;
	ctx->last_qid = queue;
}

/**
 * mlfq_runnable_exit - take a task back out of its level on a runqueue
 * @cpu: the runqueue the task is leaving
 * @ctx: the task's classification state
 *
 * The counterpart of mlfq_runnable_enter(). A no-op for a task that was never
 * counted, so the pair is self-correcting rather than able to underflow.
 */
static inline void mlfq_runnable_exit(int cpu, struct mlfq_ctx *ctx)
{
	struct mlfq_pcpu *pc = &per_cpu(mlfq_pcpu, cpu);

	if (!ctx->last_qid || ctx->last_qid > MLFQ_NR_QUEUES)
		return;

	if (pc->q_runnable[ctx->last_qid])
		pc->q_runnable[ctx->last_qid]--;
	ctx->last_qid = 0;
}

/**
 * mlfq_wakeup_mark - record that this enqueue is a wakeup
 * @ctx: the waking task's classification state
 *
 * The stand-in for scx_mlfq's MLFQ_TF_ENQ_WAKEUP, set on the way in and read
 * once placement happens; see mlfq_wakeup_pending() for why the caller's
 * enqueue flags will not do.
 *
 * Also the arrival counter the dashboard reports, which is why every wakeup is
 * counted here rather than only the ones that go on to be classified.
 */
static inline void mlfq_wakeup_mark(struct mlfq_ctx *ctx)
{
	ctx->wake_pending = 1;

	__this_cpu_inc(mlfq_pcpu.wake_total);
}

/**
 * mlfq_wakeup_clear - the wakeup has been placed, or was never one
 * @ctx: the task's classification state
 *
 * Called both when the task is given a CPU, which is where scx_mlfq clears the
 * flag in ops.running(), and from a re-enqueue that is not a wakeup, matching
 * the paths where it clears the flag without ever having set it. Clearing on
 * the second kind is the point: a task requeued by load balancing or by a nice
 * change has not woken up, and leaving a stale mark on it would let the next
 * placement treat it as one.
 */
static inline void mlfq_wakeup_clear(struct mlfq_ctx *ctx)
{
	ctx->wake_pending = 0;
}

#endif /* _KERNEL_SCHED_MLFQ_H */
