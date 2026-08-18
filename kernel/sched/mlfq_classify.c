// SPDX-License-Identifier: GPL-2.0
/*
 * Multi-level feedback queue classification for the fair class.
 *
 * Ported from scx_mlfq, a sched_ext scheduler by galpt:
 *   https://github.com/galpt/scx_mlfq
 *
 * The gauge in mlfq.h is a continuous measure of how interactive a task
 * is; this file maps it onto the three queues. A level change requires
 * crossing a band rather than a threshold, so the two entry points here drive
 * the consecutive-event counters that implement the hysteresis, and the
 * classification is only ever consulted for the task's EEVDF request size.
 *
 * scx_mlfq reaches all of this from ops.enqueue(), which sched_ext calls both
 * for wakeups and, with no flags at all, for a task that exhausted its slice.
 * The fair class splits those two events across different call sites:
 *
 *	wakeup			enqueue_task_fair()
 *	request exhausted	update_deadline(), through update_curr()
 *
 * so there is one entry point for each. Together they cover the same task
 * states scx_mlfq classified in, and no other path changes a task's level.
 */
#include "sched.h"
#include "mlfq.h"

/*
 * A task asking for a minimum utilization is telling the scheduler it is
 * latency-sensitive, which is the same claim the interactive queue encodes.
 * Take it at its word and leave its level alone rather than demoting it for
 * consuming the CPU it asked to be given.
 */
static bool mlfq_demotion_blocked(struct task_struct *p)
{
#ifdef CONFIG_UCLAMP_TASK
	return p->uclamp_req[UCLAMP_MIN].value > 0;
#else
	return false;
#endif
}

/*
 * SCHED_IDLE tasks run in the batch queue unconditionally: the policy already
 * says they should only get the CPU when nothing else wants it, so there is
 * nothing for the classifier to decide. Returns %true if the policy applied,
 * which also exempts the task from aging.
 */
static bool mlfq_apply_idle_policy(struct task_struct *p, struct mlfq_ctx *ctx)
{
	if (!task_has_idle_policy(p))
		return false;

	ctx->queue = MLFQ_Q_BATCH;

	return true;
}

/*
 * Stay bookkeeping for the aging pass below. Only a stay in a non-interactive
 * queue is timed, so a task already in Q1 carries no stay at all.
 */
static void mlfq_rearm_stay(struct mlfq_ctx *ctx, u64 now)
{
	ctx->queued_at = ctx->queue == MLFQ_Q_INTERACTIVE ? 0 : now;
}

/*
 * Aging: a task that has sat in Q2 or Q3 for MLFQ_AGING_PERIOD_NS of
 * uninterrupted wall-clock time is elevated to Q1.
 *
 * The stay re-arms at every wakeup, every level change and every exhausted
 * request, so a task that is running, or sleeping and waking, never
 * accumulates aging time; only one that has been classified non-interactive
 * and has since neither run nor slept keeps a stale stay. In scx_mlfq that
 * meant a task waiting behind others in its queue, because a waiting task
 * still passed through ops.enqueue() on the requeue paths. Under EEVDF a
 * waiting task passes through no hook at all, so this fires rarely, and it is
 * not what keeps such a task served: the deadline order already bounds its
 * wait. It is kept because it still does the one thing that order cannot,
 * which is to revisit a classification that has gone stale.
 */
static void mlfq_age_stay(struct mlfq_ctx *ctx, u64 now)
{
	if (ctx->queue == MLFQ_Q_INTERACTIVE || !ctx->queued_at)
		return;

	if (mlfq_time_before(now, ctx->queued_at + MLFQ_AGING_PERIOD_NS))
		return;

	ctx->queue = MLFQ_Q_INTERACTIVE;
	ctx->queued_at = 0;
	ctx->reenq_cnt = 0;
	mlfq_stat_inc(MLFQ_STAT_AGING_BOOSTS);
}

/*
 * mlfq_classify_wakeup - reclassify a task that is waking up
 *
 * Applies, in order: the decay over the sleep just ended, the rate-limited
 * boost for a wakeup that looks like it was waiting on something, the
 * band-hysteresis promotion, and finally the base mapping again if the sleep
 * was long enough to make the task's history meaningless.
 */
static void mlfq_classify_wakeup(struct task_struct *p, struct mlfq_ctx *ctx,
				 u64 now)
{
	u64 sleep_ns = mlfq_elapsed(now, ctx->last_sleep_at);
	u8 base_queue;

	if (sleep_ns)
		ctx->ema = mlfq_ema_decay(ctx->ema, sleep_ns,
					  MLFQ_EMA_HALF_LIFE_NS);

	ctx->last_sleep_at = 0;
	ctx->reenq_cnt = 0;

	/*
	 * The boost stands in for the wakeup fast paths a scheduler would
	 * otherwise have to recognise one kind at a time, so it deliberately
	 * does not care what the task was waiting on. SCHED_IDLE tasks are
	 * forced to Q3 below, so boosting one would only spend its rate-limit
	 * budget to no effect.
	 */
	if (!task_has_idle_policy(p) &&
	    mlfq_boost_pending(ctx, sleep_ns, p->in_iowait, now)) {
		ctx->queue = MLFQ_Q_INTERACTIVE;
		ctx->last_boost_at = now;
		mlfq_stat_inc(MLFQ_STAT_SHORT_SLEEP_BOOSTS);
	}

	if (mlfq_promote_on_wakeup(ctx, sleep_ns, MLFQ_THRESH_LOW_NS,
				   MLFQ_THRESH_HIGH_NS))
		mlfq_stat_inc(MLFQ_STAT_PROMOTIONS);

	/*
	 * A sleep this long has decayed the gauge to near nothing, so whatever
	 * put the task in a higher-numbered queue no longer describes it. Let
	 * the gauge alone speak again, but only to promote: a task that has
	 * been idle for a tenth of a second should not be demoted for it.
	 */
	if (sleep_ns > MLFQ_LONG_SLEEP_NS) {
		base_queue = mlfq_queue_from_ema(ctx->ema, MLFQ_THRESH_LOW_NS,
						 MLFQ_THRESH_HIGH_NS);
		if (base_queue < ctx->queue) {
			ctx->queue = base_queue;
			mlfq_stat_inc(MLFQ_STAT_PROMOTIONS);
		}
	}
}

/**
 * mlfq_classify_enqueue - reclassify a task being enqueued on a runqueue
 * @p: the task
 * @now: the rq clock the caller already holds
 * @wakeup: the task is waking from a sleep rather than being requeued
 *
 * The counterpart of scx_mlfq's ops.enqueue() for everything except the
 * exhausted-request path, which arrives through mlfq_classify_runout()
 * instead. A wakeup runs the full wakeup classification; every other enqueue
 * only re-examines the stay, since nothing about being moved between
 * runqueues or requeued says anything about how the task behaves.
 *
 * scx_mlfq did clear the hysteresis counters on its non-wakeup enqueues, but
 * the only ones it saw were a fork and a task entering or leaving the
 * scheduler. Here the same call site is also reached by load balancing and by
 * every requeue that a nice change, a cgroup move or a uclamp update
 * performs, none of which interrupt a run of consecutive sleeps or exhausted
 * requests. The two events scx_mlfq was reacting to reset the classification
 * outright instead, in __sched_fork() and switched_to_fair().
 *
 * On return p->mlfq.queue is the level the task will be placed at, which
 * place_entity() reads for its request size.
 */
void mlfq_classify_enqueue(struct task_struct *p, u64 now, bool wakeup)
{
	struct mlfq_ctx *ctx = &p->mlfq;
	u8 old_queue = ctx->queue;

	if (wakeup) {
		mlfq_wakeup_mark(ctx);
		mlfq_classify_wakeup(p, ctx, now);
	} else {
		mlfq_wakeup_clear(ctx);
	}

	if (mlfq_apply_idle_policy(p, ctx)) {
		ctx->queued_at = 0;
		return;
	}

	/*
	 * A level change starts a fresh stay and voids the counter that was
	 * accumulating towards the level just left. A wakeup starts a fresh
	 * stay too, since time spent asleep is not time spent waiting.
	 */
	if (ctx->queue != old_queue) {
		mlfq_rearm_stay(ctx, now);
		ctx->reenq_cnt = 0;
	} else if (wakeup) {
		mlfq_rearm_stay(ctx, now);
	}

	mlfq_age_stay(ctx, now);
}

/**
 * mlfq_classify_runout - reclassify a task that has exhausted its request
 * @p: the task
 * @now: the rq clock the caller already holds
 *
 * The counterpart of scx_mlfq's flags-less ops.enqueue(), which sched_ext
 * delivered from put_prev_task_scx() once a task had consumed its slice
 * grant. Under EEVDF the same event is update_deadline() finding the task's
 * virtual time has reached its deadline, so this is called from there, before
 * the next request is issued.
 *
 * Reaching the end of a request without sleeping is evidence against being
 * interactive, and enough of it in a row demotes the task. Having run a full
 * request also ends the current stay, so the task is not credited with aging
 * time for the period it spent on the CPU.
 *
 * sched_yield() also lands here, because yield_task_fair() forfeits the rest
 * of the request by pushing the task's virtual time up to its deadline. That
 * is deliberate and matches sched_ext, where a yield zeroed the slice grant
 * and sent the task back through the same flag-less ops.enqueue(): a task
 * spinning on sched_yield() is burning the CPU whether or not it waits out
 * each request in full, and once it does block, the decay and the boost put
 * it back where it belongs.
 */
void mlfq_classify_runout(struct task_struct *p, u64 now)
{
	struct mlfq_ctx *ctx = &p->mlfq;

	ctx->wake_cnt = 0;

	if (!mlfq_demotion_blocked(p) &&
	    mlfq_demote_on_runout(ctx, MLFQ_THRESH_HIGH_NS))
		mlfq_stat_inc(MLFQ_STAT_DEMOTIONS);

	if (mlfq_apply_idle_policy(p, ctx)) {
		ctx->queued_at = 0;
		return;
	}

	mlfq_rearm_stay(ctx, now);
}
