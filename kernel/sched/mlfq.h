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
 * dispatch batch. A task's queue is chosen by an interactivity gauge: a
 * saturating exponentially-weighted average of running time that climbs
 * while the task runs and decays while it sleeps.
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
 * receives still follows its weight rather than its queue, and the
 * eligibility criterion bounds how long any task waits. Eligibility is
 * therefore the starvation bound that the guaranteed batch-queue share of
 * each dispatch batch provided in scx_mlfq, so no quota is carried over.
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
 *	mlfq_adapt.c	 <- src/bpf/main.bpf.c	 system gauges, band tuning
 *	mlfq_stats.c	 <- src/stats.rs, src/webui.rs, ui/index.html
 *					 the /proc dashboard
 *
 * The hooks themselves stay in fair.c, because they have to sit inside the
 * EEVDF paths they observe; each is a single call whose policy lives here.
 */
#ifndef _KERNEL_SCHED_MLFQ_H
#define _KERNEL_SCHED_MLFQ_H

#include <linux/limits.h>
#include <linux/math64.h>
#include <linux/sched.h>
#include <linux/types.h>

/* Fixed-point shift for the gauge coefficients. */
#define MLFQ_FP_SHIFT			8
#define MLFQ_FP_ONE			(1UL << MLFQ_FP_SHIFT)

/* Request size per queue level, in nanoseconds. */
#define MLFQ_SLICE_Q1_NS		1000000ULL
#define MLFQ_SLICE_Q2_NS		2000000ULL
#define MLFQ_SLICE_Q3_NS		4000000ULL

/*
 * Ceiling of the interactivity gauge. A task that runs for this long
 * without sleeping is as CPU-bound as the gauge can express.
 */
#define MLFQ_BUDGET_MAX_NS		6000000ULL

/*
 * Classification thresholds on the gauge. Below the low threshold a task
 * is interactive, at or above the high threshold it is CPU-bound, and in
 * between it is left in the default queue.
 */
#define MLFQ_THRESH_LOW_NS		250000ULL
#define MLFQ_THRESH_HIGH_NS		2000000ULL

/*
 * Climb coefficient. The gauge closes the gap to MLFQ_BUDGET_MAX_NS with a
 * time constant of MLFQ_BUDGET_MAX_NS * MLFQ_FP_ONE / MLFQ_EMA_ALPHA_FP,
 * which is 500us of running time.
 */
#define MLFQ_EMA_ALPHA_FP		3072UL

/* Half-life of the gauge while the task sleeps. */
#define MLFQ_EMA_HALF_LIFE_NS		24000000ULL

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
 * A sleep this long makes the task's history stale, so the gauge alone
 * decides the queue again and the hysteresis counters are dropped.
 */
#define MLFQ_LONG_SLEEP_NS		120000000ULL

/*
 * A stay of this long in Q2 or Q3 elevates the task to Q1. EEVDF already
 * bounds the wait through eligibility, so this only catches a task whose
 * classification has gone stale.
 */
#define MLFQ_AGING_PERIOD_NS		1000000000ULL

/* Consecutive short sleeps required before promoting a level. */
#define MLFQ_PROMOTE_WAKES		2
/* Consecutive request exhaustions required before demoting a level. */
#define MLFQ_DEMOTE_REENQS		8

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
 * mlfq_ema_climb - advance the gauge over a stretch of running time
 * @ema: current gauge value
 * @delta: nanoseconds of running time to account for
 *
 * Saturating exponential climb towards the ceiling:
 *
 *	step = (MLFQ_BUDGET_MAX_NS - ema) * delta * alpha / (BUDGET_MAX * FP_ONE)
 *	ema += min(step, MLFQ_BUDGET_MAX_NS - ema)
 *
 * The step is proportional to the remaining gap, so the gauge approaches the
 * ceiling with a time constant of BUDGET_MAX * FP_ONE / alpha and never
 * passes it. @delta is clamped to the ceiling first, which also keeps the
 * product below 2^63.
 *
 * The step is linear in @delta to first order, so accounting one long stretch
 * or the same stretch split across several calls gives the same result to
 * within the rounding of the divide. The fair class can therefore climb the
 * gauge per update_curr() delta rather than per run segment.
 *
 * Returns the updated gauge.
 */
static inline u64 mlfq_ema_climb(u64 ema, u64 delta)
{
	u64 gap, step;

	if (delta > MLFQ_BUDGET_MAX_NS)
		delta = MLFQ_BUDGET_MAX_NS;
	if (!delta)
		return ema;

	gap = MLFQ_BUDGET_MAX_NS - ema;
	if (!gap)
		return ema;

	/*
	 * gap and delta are both at most MLFQ_BUDGET_MAX_NS, so the product
	 * with the coefficient stays well inside 64 bits, and the divisor is
	 * a compile-time constant below 2^32.
	 */
	step = div_u64(gap * delta * MLFQ_EMA_ALPHA_FP,
		       MLFQ_BUDGET_MAX_NS * MLFQ_FP_ONE);
	if (step > gap)
		step = gap;

	return ema + step;
}

/**
 * mlfq_ema_decay - decay the gauge over a stretch of sleep
 * @ema: current gauge value
 * @sleep_ns: nanoseconds slept
 *
 * Whole half-lives are applied as a right shift. The remaining fraction x of
 * a half-life uses the second-order expansion of 2^-x,
 *
 *	2^-x ~= 1 - x*ln2 + (x*ln2)^2 / 2
 *
 * evaluated in MLFQ_FP_ONE fixed point, where 177 approximates
 * MLFQ_FP_ONE * ln(2). The relative error stays under 10% over x in [0,1),
 * which is far finer than the gap between the classification thresholds. A
 * sleep of 64 half-lives or more zeroes the gauge outright, which also keeps
 * the shift count in range.
 *
 * Returns the updated gauge.
 */
static inline u64 mlfq_ema_decay(u64 ema, u64 sleep_ns)
{
	u64 periods, decayed, x_fp, a_fp, factor_fp;
	u32 sub;

	if (sleep_ns >= (MLFQ_EMA_HALF_LIFE_NS << 6))
		return 0;

	periods = div_u64_rem(sleep_ns, MLFQ_EMA_HALF_LIFE_NS, &sub);
	decayed = ema >> periods;

	if (!sub || !decayed)
		return decayed;

	x_fp = div_u64((u64)sub * MLFQ_FP_ONE, MLFQ_EMA_HALF_LIFE_NS);
	a_fp = (x_fp * 177) >> MLFQ_FP_SHIFT;
	factor_fp = MLFQ_FP_ONE - a_fp + ((a_fp * a_fp) >> (MLFQ_FP_SHIFT + 1));

	return (decayed * factor_fp) >> MLFQ_FP_SHIFT;
}

/**
 * mlfq_queue_from_ema - the queue the gauge alone asks for
 * @ema: the gauge value
 *
 * The base mapping, without any hysteresis: at or below the low threshold the
 * task is interactive, at or above the high threshold it is CPU-bound, and in
 * between it stays in the default queue.
 */
static inline u8 mlfq_queue_from_ema(u64 ema)
{
	if (ema <= MLFQ_THRESH_LOW_NS)
		return MLFQ_Q_INTERACTIVE;
	if (ema >= MLFQ_THRESH_HIGH_NS)
		return MLFQ_Q_BATCH;

	return MLFQ_Q_DEFAULT;
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
 *
 * A single wakeup does not move a task up a level. The gauge has to be well
 * inside the band below, at half the threshold that would have placed the
 * task there, and the task has to have slept briefly MLFQ_PROMOTE_WAKES times
 * in a row. Requiring the crossing of a band rather than a point is what
 * keeps a task whose gauge sits near a threshold from changing level on every
 * wakeup.
 *
 * Returns %true if the task moved up a level.
 */
static inline bool mlfq_promote_on_wakeup(struct mlfq_ctx *ctx, u64 sleep_ns)
{
	bool promoted = false;

	if (sleep_ns > MLFQ_HYSTERESIS_SLEEP_NS)
		ctx->wake_cnt = 0;
	else if (ctx->wake_cnt < U8_MAX)
		ctx->wake_cnt++;

	if (ctx->wake_cnt >= MLFQ_PROMOTE_WAKES) {
		if (ctx->queue == MLFQ_Q_DEFAULT &&
		    ctx->ema < MLFQ_THRESH_LOW_NS / 2) {
			ctx->queue = MLFQ_Q_INTERACTIVE;
			promoted = true;
		} else if (ctx->queue == MLFQ_Q_BATCH &&
			   ctx->ema < MLFQ_THRESH_HIGH_NS / 2) {
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
 *
 * Demotion needs a sustained run rather than a single exhausted request:
 * MLFQ_DEMOTE_REENQS exhaustions in a row, roughly 8ms at the interactive
 * request size, while the gauge stays above the CPU-bound threshold. A task
 * that sleeps in between is boosted at its wakeup, which clears the counter,
 * so something bursty like a decoder keeps its level for the whole burst,
 * while a task that simply never sleeps accumulates the counter and is
 * demoted.
 *
 * Returns %true if the task moved down a level.
 */
static inline bool mlfq_demote_on_runout(struct mlfq_ctx *ctx)
{
	if (ctx->reenq_cnt < U8_MAX)
		ctx->reenq_cnt++;

	if (ctx->queue != MLFQ_Q_INTERACTIVE && ctx->queue != MLFQ_Q_DEFAULT)
		return false;
	if (ctx->ema <= MLFQ_THRESH_HIGH_NS ||
	    ctx->reenq_cnt < MLFQ_DEMOTE_REENQS)
		return false;

	ctx->queue++;
	ctx->reenq_cnt = 0;

	return true;
}

/**
 * mlfq_reset_classification - start a task's classification from scratch
 * @ctx: the task's classification state
 *
 * A task with no history lands in the default queue with an empty gauge, so
 * its first few stretches of running time decide where it belongs.
 */
static inline void mlfq_reset_classification(struct mlfq_ctx *ctx)
{
	ctx->ema = 0;
	ctx->last_sleep_at = 0;
	ctx->queued_at = 0;
	ctx->last_boost_at = 0;
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

#endif /* _KERNEL_SCHED_MLFQ_H */
