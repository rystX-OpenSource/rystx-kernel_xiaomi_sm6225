// SPDX-License-Identifier: GPL-2.0
/*
 * System-wide state for the multi-level feedback queue classifier.
 *
 * Ported from scx_mlfq, a sched_ext scheduler by galpt:
 *   https://github.com/galpt/scx_mlfq
 *
 * Everything in mlfq.h and mlfq_classify.c decides one task's level from that
 * task's own history. This file holds the state that spans tasks: the event
 * counters, the per-runqueue record of how many tasks of each level are
 * waiting, the two gauges that measure what the machine as a whole is doing,
 * and the once-a-second step that turns the first of those gauges into the band
 * edges every classification compares against. scx_mlfq keeps the same split,
 * with its per-CPU state and its system gauges in main.bpf.c and only its
 * classification in classify.bpf.c.
 *
 * The per-CPU state is written from the CPU that owns it, under that runqueue's
 * lock, so none of it costs a shared cacheline on any scheduling path. Readers
 * sum across CPUs and are not serialised against the writers;
 * kernel/sched/mlfq_stats.c is the only reader and says what that means for
 * what it prints.
 */
#include "sched.h"
#include "mlfq.h"

DEFINE_PER_CPU(struct mlfq_pcpu, mlfq_pcpu);

struct mlfq_sys_gauge mlfq_sys_gauge;

/*
 * The bands start at their base values rather than at zero, so that the very
 * first classification, which can happen before any step has run, compares
 * against the thresholds the classifier was designed around. scx_mlfq copies
 * the same constants in from its read-only section in ops.init() for the same
 * reason.
 */
struct mlfq_adapt_state mlfq_adapt_state = {
	.shift_fp	= 0,
	.t_l_eff_ns	= MLFQ_THRESH_LOW_NS,
	.t_h_eff_ns	= MLFQ_THRESH_HIGH_NS,
};

/*
 * Folding a window means reading each CPU's accumulator and zeroing it, from
 * whichever CPU won the step, while the CPUs that own them carry on counting.
 * There is no attempt to serialise the two: an increment landing between the
 * read and the store is lost, which costs at most one event per CPU per window,
 * out of the thousands a busy second holds, and on a quiet second there is
 * nothing to race with. That is far below the resolution of a gauge that exists
 * to show a trend, and scx_mlfq's read-modify-write of the same accumulators
 * from unsynchronised CPUs has exactly the same property.
 */
static u32 mlfq_wakeup_window_fold(void)
{
	u32 total = 0;
	int cpu;

	for_each_possible_cpu(cpu) {
		struct mlfq_pcpu *pc = &per_cpu(mlfq_pcpu, cpu);

		total += READ_ONCE(pc->wake_window);
		WRITE_ONCE(pc->wake_window, 0);
	}

	return total;
}

static void mlfq_wait_window_fold(u64 *total_ns, u64 *count)
{
	u64 ns = 0, n = 0;
	int cpu;

	for_each_possible_cpu(cpu) {
		struct mlfq_pcpu *pc = &per_cpu(mlfq_pcpu, cpu);

		ns += READ_ONCE(pc->wait_total);
		WRITE_ONCE(pc->wait_total, 0);
		n += READ_ONCE(pc->wait_count);
		WRITE_ONCE(pc->wait_count, 0);
	}

	*total_ns = ns;
	*count = n;
}

/**
 * mlfq_adapt_step - fold one window into the gauges and re-derive the bands
 *
 * scx_mlfq drives this from both ops.stopping() and ops.dispatch(), so that a
 * busy CPU keeps the cadence going whichever of the two it reaches first. The
 * fair class has a hook with strictly better coverage: the tick fires on every
 * CPU that has a runnable fair task, including one running a single CPU-bound
 * task that never context switches, which is a case ops.stopping() misses for
 * as long as the task keeps the CPU. An idle CPU neither ticks nor dispatches,
 * so both scx_mlfq and this port stop stepping on an idle system and hold the
 * gauges where they were, rather than decaying them towards zero against a
 * window in which nothing happened.
 *
 * Every caller is a tick, so the cadence gate is the common path and has to be
 * a single unsynchronised read. Only the CPU that then wins the compare and
 * exchange does the work; the rest return immediately, and nothing here ever
 * waits on anything.
 */
void mlfq_adapt_step(void)
{
	struct mlfq_sys_gauge *g = &mlfq_sys_gauge;
	u64 now = local_clock();
	u64 prev, elapsed, wait_total, wait_count, lat, rate;
	s64 shift;
	u32 wakeups;

	/*
	 * A machine-wide cadence needs a machine-wide clock, so this is
	 * the one place in the port that does not use the runqueue clock:
	 * rq_clock_task() leaves out each CPU's own interrupt time and so
	 * advances at a different rate on every CPU, which is what makes
	 * it the right clock for measuring one task and the wrong one for
	 * anchoring an interval across all of them.
	 */
	prev = READ_ONCE(g->step_at);
	if (mlfq_time_before(now, prev + MLFQ_ADAPT_MIN_INTERVAL_NS))
		return;

	if (cmpxchg64(&g->step_at, prev, now) != prev)
		return;

	/*
	 * The gate has already established that @now is at least a second
	 * past @prev, so this is both safe and the true length of the
	 * window just accumulated: at the first step @prev is zero and the
	 * window is the uptime, which is the period the counters cover.
	 */
	elapsed = now - prev;

	/*
	 * The latency gauge is folded whether or not the controller is
	 * enabled, because it is also what the dashboard reports, and
	 * reporting it is what makes enabling the controller an informed
	 * choice. Both windows are consumed either way, so a disabled
	 * controller cannot leave an uptime of arrivals sitting in the
	 * accumulators for the moment one is enabled.
	 */
	mlfq_wait_window_fold(&wait_total, &wait_count);
	lat = mlfq_sys_lat_fold(READ_ONCE(g->lat_ema), wait_total,
				wait_count, elapsed);
	WRITE_ONCE(g->lat_ema, lat);

	wakeups = mlfq_wakeup_window_fold();

	if (!sched_feat(MLFQ_ADAPT))
		return;

	rate = mlfq_sys_rate_step(READ_ONCE(g->rate_ema), wakeups, elapsed);
	WRITE_ONCE(g->rate_ema, rate);
	WRITE_ONCE(g->adapt_steps, READ_ONCE(g->adapt_steps) + 1);

	shift = mlfq_adapt_slew(READ_ONCE(mlfq_adapt_state.shift_fp),
				mlfq_adapt_shift_target(lat, rate));

	/*
	 * A classification that reads the edges while these stores are in
	 * flight sees one of them from the previous step, which the
	 * disjoint bounds on the two make harmless rather than merely
	 * unlikely; see mlfq_read_bands().
	 */
	WRITE_ONCE(mlfq_adapt_state.t_l_eff_ns,
		   mlfq_adapt_band(MLFQ_THRESH_LOW_NS, shift,
				   MLFQ_T_L_FLOOR_NS, MLFQ_T_L_CEIL_NS));
	WRITE_ONCE(mlfq_adapt_state.t_h_eff_ns,
		   mlfq_adapt_band(MLFQ_THRESH_HIGH_NS, shift,
				   MLFQ_T_H_FLOOR_NS, MLFQ_T_H_CEIL_NS));
	WRITE_ONCE(mlfq_adapt_state.shift_fp, shift);
}
