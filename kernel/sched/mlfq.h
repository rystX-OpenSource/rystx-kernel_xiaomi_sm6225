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
 *	mlfq_adapt.c	 <- src/bpf/main.bpf.c	 system-wide state and gauges
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
#include <linux/math64.h>
#include <linux/percpu.h>
#include <linux/sched.h>
#include <linux/time64.h>
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

/*
 * The classification thresholds above are the *base* edges. What the
 * classifier actually compares against is an effective edge that a slow
 * controller widens when the machine's wakeup latency runs above target; see
 * struct mlfq_adapt_state and mlfq_adapt_step() in mlfq_adapt.c. The
 * constants below size that controller.
 *
 * Cadence of the controller, and of both gauges it reads. One second is also
 * the gauges' half-life, so each step replaces roughly half of what the gauge
 * held with what the last second measured.
 */
#define MLFQ_ADAPT_MIN_INTERVAL_NS	1000000000ULL
#define MLFQ_SYS_GAUGE_HALF_LIFE_NS	1000000000ULL

/* Ceiling of the wakeup-latency gauge. */
#define MLFQ_SYS_LAT_MAX_NS		16000000ULL

/*
 * Bounds on the wakeup-rate gauge: the widest window a single step will
 * believe, so a long idle gap cannot inflate the rate beyond what one busy
 * minute would produce, and a ceiling on the rate itself.
 */
#define MLFQ_SYS_RATE_WINDOW_MAX_NS	60000000000ULL
#define MLFQ_SYS_RATE_MAX		1000000ULL

/*
 * Target for the wakeup-latency gauge. Deliberately equal to the interactive
 * request size: 1ms is what a task the classifier calls interactive is being
 * promised, so it is also the latency the bands are tuned to deliver.
 */
#define MLFQ_ADAPT_TARGET_LAT_NS	1000000ULL

/*
 * Proportional gain, and the range of the shift it produces. A gauge error of
 * one whole target maps to MLFQ_ADAPT_K, so the shift saturates at half the
 * base once the machine settles at twice the target latency: 250us becomes
 * 375us and 2ms becomes 3ms.
 */
#define MLFQ_ADAPT_K			(MLFQ_FP_ONE / 2)
#define MLFQ_ADAPT_MAX_SHIFT		((s64)MLFQ_FP_ONE / 2)

/*
 * Slew limit, in shift units per step. FP_ONE/10 is 25 by integer division,
 * so about 9.8% of the base per second, and a full swing to the maximum shift
 * takes six steps. Together with the gauges' own one-second half-life this
 * makes the loop a cascaded first-order limit: no single bad measurement can
 * move a band edge more than one step.
 */
#define MLFQ_ADAPT_MAX_STEP		((s64)MLFQ_FP_ONE / 10)

/*
 * Wakeup-storm gate. Above this rate the machine is an order of magnitude
 * past a normal interactive cadence, the latency gauge is measuring queueing
 * rather than classification, and the bands must stop chasing it so far.
 */
#define MLFQ_ADAPT_RATE_GATE_HIGH	(200000ULL << MLFQ_FP_SHIFT)
#define MLFQ_ADAPT_RATE_GATE_SHIFT	((s64)MLFQ_FP_ONE / 4)

/*
 * Hard bounds on each effective edge. The two ranges are disjoint, with the
 * low edge's ceiling well under the high edge's floor, so no admissible shift
 * can collapse the default band to nothing or invert the queue order: the
 * effective band is at least 700us wide whatever the controller does.
 */
#define MLFQ_T_L_FLOOR_NS		150000ULL
#define MLFQ_T_L_CEIL_NS		500000ULL
#define MLFQ_T_H_FLOOR_NS		1200000ULL
#define MLFQ_T_H_CEIL_NS		3200000ULL

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
 * mlfq_ema_decay - decay a gauge over a stretch of elapsed time
 * @ema: current gauge value
 * @sleep_ns: nanoseconds elapsed
 * @half_life: nanoseconds in which the gauge halves
 *
 * Whole half-lives are applied as a right shift. The remaining fraction x of
 * a half-life uses the second-order expansion of 2^-x,
 *
 *	2^-x ~= 1 - x*ln2 + (x*ln2)^2 / 2
 *
 * evaluated in MLFQ_FP_ONE fixed point, where 177 approximates
 * MLFQ_FP_ONE * ln(2). The relative error stays under 10% over x in [0,1),
 * which is far finer than the gap between the classification thresholds. An
 * elapsed time of 64 half-lives or more zeroes the gauge outright, which also
 * keeps the shift count in range.
 *
 * The half-life is a parameter because this is used at three different rates:
 * MLFQ_EMA_HALF_LIFE_NS for a task's interactivity gauge, and
 * MLFQ_SYS_GAUGE_HALF_LIFE_NS for the two system gauges, which also call it as
 * mlfq_ema_decay(MLFQ_FP_ONE, elapsed, half_life) purely to obtain the
 * coefficient 2^-(elapsed/half_life) in fixed point.
 *
 * Returns the updated gauge.
 */
static inline u64 mlfq_ema_decay(u64 ema, u64 sleep_ns, u64 half_life)
{
	u64 periods, decayed, sub, x_fp, a_fp, factor_fp;

	if (!half_life || sleep_ns >= (half_life << 6))
		return 0;

	periods = div64_u64_rem(sleep_ns, half_life, &sub);
	decayed = ema >> periods;

	if (!sub || !decayed)
		return decayed;

	x_fp = div64_u64(sub * MLFQ_FP_ONE, half_life);
	a_fp = (x_fp * 177) >> MLFQ_FP_SHIFT;
	factor_fp = MLFQ_FP_ONE - a_fp + ((a_fp * a_fp) >> (MLFQ_FP_SHIFT + 1));

	return (decayed * factor_fp) >> MLFQ_FP_SHIFT;
}

/**
 * mlfq_sys_lat_fold - fold one window of wakeup waits into the latency gauge
 * @lat_ema: current gauge value, in nanoseconds
 * @wait_total: sum of the waits measured in the window
 * @wait_count: number of waits measured in the window
 * @elapsed: length of the window
 *
 * The gauge is an exponential average over the window *average* wait,
 *
 *	lat_ema' = lat_ema * 2^-(elapsed/T) + avg * (1 - 2^-(elapsed/T))
 *
 * with T = MLFQ_SYS_GAUGE_HALF_LIFE_NS, capped at MLFQ_SYS_LAT_MAX_NS. Folding
 * the average rather than the total is what makes the equilibrium the average
 * wait regardless of how many wakeups produced it, so a busy machine and a
 * quiet one with the same per-wakeup wait settle on the same gauge and get the
 * same bands. A window with no wakeups in it leaves the gauge alone rather
 * than pulling it towards zero.
 *
 * Returns the updated gauge.
 */
static inline u64 mlfq_sys_lat_fold(u64 lat_ema, u64 wait_total, u64 wait_count,
				    u64 elapsed)
{
	u64 avg, factor, decayed, rise;

	if (!wait_count)
		return lat_ema;

	avg = div64_u64(wait_total, wait_count);
	if (avg > MLFQ_SYS_LAT_MAX_NS)
		avg = MLFQ_SYS_LAT_MAX_NS;

	factor = mlfq_ema_decay(MLFQ_FP_ONE, elapsed,
				MLFQ_SYS_GAUGE_HALF_LIFE_NS);
	decayed = mlfq_ema_decay(lat_ema, elapsed,
				 MLFQ_SYS_GAUGE_HALF_LIFE_NS);
	rise = (avg * (MLFQ_FP_ONE - factor)) >> MLFQ_FP_SHIFT;

	return min_t(u64, decayed + rise, MLFQ_SYS_LAT_MAX_NS);
}

/**
 * mlfq_sys_rate_step - fold one window of wakeup arrivals into the rate gauge
 * @rate_ema: current gauge value, wakeups per second in MLFQ_FP_ONE fixed point
 * @wakeup_cnt: arrivals counted in the window
 * @elapsed: length of the window
 *
 * The same one-second exponential average as the latency gauge, over the
 * window's arrival rate. The window is clamped at both ends: the lower bound
 * because the cadence gate already guarantees a full interval, so a shorter
 * one can only come from a clock going backwards, and the upper bound so that
 * a long idle gap is charged at the rate of one busy minute rather than
 * appearing as a burst.
 *
 * Returns the updated gauge, still in fixed point.
 */
static inline u64 mlfq_sys_rate_step(u64 rate_ema, u32 wakeup_cnt, u64 elapsed)
{
	u64 rate_i, rate_i_fp, factor;

	if (elapsed < MLFQ_ADAPT_MIN_INTERVAL_NS)
		elapsed = MLFQ_ADAPT_MIN_INTERVAL_NS;
	if (elapsed > MLFQ_SYS_RATE_WINDOW_MAX_NS)
		elapsed = MLFQ_SYS_RATE_WINDOW_MAX_NS;

	rate_i = div64_u64((u64)wakeup_cnt * NSEC_PER_SEC, elapsed);
	if (rate_i > MLFQ_SYS_RATE_MAX)
		rate_i = MLFQ_SYS_RATE_MAX;
	rate_i_fp = rate_i << MLFQ_FP_SHIFT;

	factor = mlfq_ema_decay(MLFQ_FP_ONE, elapsed,
				MLFQ_SYS_GAUGE_HALF_LIFE_NS);

	return ((rate_ema * factor) >> MLFQ_FP_SHIFT) +
	       ((rate_i_fp * (MLFQ_FP_ONE - factor)) >> MLFQ_FP_SHIFT);
}

/**
 * mlfq_adapt_shift_target - where the controller wants the band shift to be
 * @lat_ema: the wakeup-latency gauge
 * @rate_ema: the wakeup-rate gauge
 *
 * Proportional control on the latency gauge's error against
 * MLFQ_ADAPT_TARGET_LAT_NS:
 *
 *	target = clamp((lat_ema - target_lat) * K / target_lat, 0, MAX_SHIFT)
 *
 * One-sided on purpose. A gauge below target produces a shift of zero, which
 * is the base bands exactly, and never a negative shift: narrowing the bands
 * below the base was measured to make the wakeup-latency tail worse under
 * load, not better, so only the widening half of the law is kept. Above a
 * storm of wakeups the shift is additionally capped, because there the gauge
 * is measuring how many tasks want a CPU rather than how well they are being
 * classified.
 *
 * Returns the target shift in MLFQ_FP_ONE fixed point, in [0, MAX_SHIFT].
 */
static inline s64 mlfq_adapt_shift_target(u64 lat_ema, u64 rate_ema)
{
	s64 err = (s64)lat_ema - (s64)MLFQ_ADAPT_TARGET_LAT_NS;
	s64 target;
	u64 mag;

	mag = err < 0 ? 0 : (u64)err;
	target = (s64)div64_u64(mag * MLFQ_ADAPT_K, MLFQ_ADAPT_TARGET_LAT_NS);

	if (target > MLFQ_ADAPT_MAX_SHIFT)
		target = MLFQ_ADAPT_MAX_SHIFT;
	if (rate_ema > MLFQ_ADAPT_RATE_GATE_HIGH &&
	    target > MLFQ_ADAPT_RATE_GATE_SHIFT)
		target = MLFQ_ADAPT_RATE_GATE_SHIFT;

	return target;
}

/**
 * mlfq_adapt_slew - move the shift one step towards its target
 * @prev: the shift in force
 * @target: where mlfq_adapt_shift_target() wants it
 *
 * Returns @prev moved towards @target by at most MLFQ_ADAPT_MAX_STEP.
 */
static inline s64 mlfq_adapt_slew(s64 prev, s64 target)
{
	s64 step = target - prev;

	if (step > MLFQ_ADAPT_MAX_STEP)
		step = MLFQ_ADAPT_MAX_STEP;
	if (step < -MLFQ_ADAPT_MAX_STEP)
		step = -MLFQ_ADAPT_MAX_STEP;

	return prev + step;
}

/**
 * mlfq_adapt_band - the effective edge for a base threshold
 * @base: the base threshold
 * @shift_fp: the shift in force, in MLFQ_FP_ONE fixed point
 * @floor: lowest effective value permitted
 * @ceil: highest effective value permitted
 *
 * effective = clamp(base + base * shift_fp / FP_ONE, floor, ceil)
 *
 * The clamp is on the result, never on the shift, so both edges of a band see
 * the same shift and the band is scaled rather than sheared. A zero shift
 * returns @base exactly, which is what makes a disabled controller
 * indistinguishable from fixed thresholds.
 */
static inline u64 mlfq_adapt_band(u64 base, s64 shift_fp, u64 floor, u64 ceil)
{
	s64 v = (s64)base;
	u64 mag = shift_fp < 0 ? (u64)-shift_fp : (u64)shift_fp;
	s64 scaled = (s64)((base * mag) >> MLFQ_FP_SHIFT);

	v += shift_fp < 0 ? -scaled : scaled;

	if (v < (s64)floor)
		v = (s64)floor;
	if (v > (s64)ceil)
		v = (s64)ceil;

	return (u64)v;
}

/**
 * struct mlfq_sys_gauge - what the machine as a whole is doing
 * @lat_ema:		wakeup-latency gauge, in nanoseconds, at most
 *			MLFQ_SYS_LAT_MAX_NS. An average over the average, so
 *			its equilibrium is the per-wakeup wait and not the
 *			number of wakeups. Folded at every step, even with the
 *			controller disabled.
 * @rate_ema:		wakeup-rate gauge, in wakeups per second in
 *			MLFQ_FP_ONE fixed point. Folded only when the
 *			controller is enabled, and frozen otherwise, because
 *			its only consumer is the storm gate.
 * @step_at:		the cadence gate, and the anchor the single winner of
 *			each step claims. Not an rq clock: the steps are
 *			machine-wide, so this is a local_clock() reading.
 * @adapt_steps:	steps taken since boot.
 *
 * One global instance, written only by mlfq_adapt_step(). scx_mlfq keeps the
 * two window accumulators in here as well; this port keeps them per-CPU in
 * struct mlfq_pcpu instead, for the reason scx_mlfq gives for having moved its
 * wakeup arrival counters out: nothing on a scheduling path should have to
 * touch this line.
 */
struct mlfq_sys_gauge {
	u64	lat_ema;
	u64	rate_ema;
	u64	step_at;
	u32	adapt_steps;
};

/**
 * struct mlfq_adapt_state - the band edges the classifier compares against
 * @shift_fp:		the relative widening in force, MLFQ_FP_ONE fixed
 *			point, within [0, MLFQ_ADAPT_MAX_SHIFT].
 * @t_l_eff_ns:		effective low edge, MLFQ_THRESH_LOW_NS shifted.
 * @t_h_eff_ns:		effective high edge, MLFQ_THRESH_HIGH_NS shifted.
 *
 * One global instance, written only by mlfq_adapt_step() and read locklessly
 * by every classification. It starts at the base thresholds rather than at
 * zero, so the classifier never sees a degenerate band, and with the
 * controller disabled it simply stays there.
 */
struct mlfq_adapt_state {
	s64	shift_fp;
	u64	t_l_eff_ns;
	u64	t_h_eff_ns;
};

extern struct mlfq_sys_gauge mlfq_sys_gauge;
extern struct mlfq_adapt_state mlfq_adapt_state;

/**
 * struct mlfq_bands - one classification pass's view of the effective edges
 * @t_l: effective low edge
 * @t_h: effective high edge
 */
struct mlfq_bands {
	u64	t_l;
	u64	t_h;
};

/**
 * mlfq_read_bands - take the effective edges for one classification pass
 *
 * Read once and passed down, so that every predicate in a single pass compares
 * against the same pair. A step landing between the two loads is harmless
 * rather than merely unlikely: the hard bounds on the two edges are disjoint,
 * so an old low edge with a new high edge, or the reverse, is still an ordered
 * band at least 700us wide.
 */
static inline struct mlfq_bands mlfq_read_bands(void)
{
	struct mlfq_bands bands = {
		.t_l = READ_ONCE(mlfq_adapt_state.t_l_eff_ns),
		.t_h = READ_ONCE(mlfq_adapt_state.t_h_eff_ns),
	};

	return bands;
}

/* The 1Hz controller step, out of line in kernel/sched/mlfq_adapt.c. */
void mlfq_adapt_step(void);

/**
 * mlfq_queue_from_ema - the queue the gauge alone asks for
 * @ema: the gauge value
 * @t_l: effective low edge
 * @t_h: effective high edge
 *
 * The base mapping, without any hysteresis: at or below the low edge the task
 * is interactive, at or above the high edge it is CPU-bound, and in between it
 * stays in the default queue. The edges are passed in rather than read here so
 * that one classification pass sees one consistent pair; see
 * mlfq_read_bands().
 */
static inline u8 mlfq_queue_from_ema(u64 ema, u64 t_l, u64 t_h)
{
	if (ema <= t_l)
		return MLFQ_Q_INTERACTIVE;
	if (ema >= t_h)
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
 * @t_l: effective low edge
 * @t_h: effective high edge
 *
 * A single wakeup does not move a task up a level. The gauge has to be well
 * inside the band below, at half the edge that would have placed the task
 * there, and the task has to have slept briefly MLFQ_PROMOTE_WAKES times in a
 * row. Requiring the crossing of a band rather than a point is what keeps a
 * task whose gauge sits near an edge from changing level on every wakeup.
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
		if (ctx->queue == MLFQ_Q_DEFAULT && ctx->ema < t_l / 2) {
			ctx->queue = MLFQ_Q_INTERACTIVE;
			promoted = true;
		} else if (ctx->queue == MLFQ_Q_BATCH && ctx->ema < t_h / 2) {
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
 * MLFQ_DEMOTE_REENQS exhaustions in a row, roughly 8ms at the interactive
 * request size, while the gauge stays above the CPU-bound edge. A task that
 * sleeps in between is boosted at its wakeup, which clears the counter, so
 * something bursty like a decoder keeps its level for the whole burst, while a
 * task that simply never sleeps accumulates the counter and is demoted.
 *
 * Returns %true if the task moved down a level.
 */
static inline bool mlfq_demote_on_runout(struct mlfq_ctx *ctx, u64 t_h)
{
	if (ctx->reenq_cnt < U8_MAX)
		ctx->reenq_cnt++;

	if (ctx->queue != MLFQ_Q_INTERACTIVE && ctx->queue != MLFQ_Q_DEFAULT)
		return false;
	if (ctx->ema <= t_h || ctx->reenq_cnt < MLFQ_DEMOTE_REENQS)
		return false;

	ctx->queue++;
	ctx->reenq_cnt = 0;

	return true;
}

/**
 * mlfq_wakeup_pending - is this enqueue the one a wakeup arrived on
 * @ctx: the task's classification state
 *
 * The marker scx_mlfq keeps as MLFQ_TF_ENQ_WAKEUP, which it sets in
 * ops.enqueue() when sched_ext hands it a wakeup and clears once the task
 * reaches a CPU. Here the timestamp doubles as the mark, so this is one read of
 * the field mlfq_wakeup_episode_begin() stamped.
 *
 * It is asked rather than the caller's enqueue flags because the flags do not
 * survive every path into placement: requeue_delayed_entity() places a task
 * that woke up, with no flags at all. The mark is set once per wakeup, on the
 * way in, so it is true wherever that wakeup ends up being placed.
 */
static inline bool mlfq_wakeup_pending(const struct mlfq_ctx *ctx)
{
	return ctx->wake_enq_at != 0;
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
	ctx->ema = 0;
	ctx->last_sleep_at = 0;
	ctx->queued_at = 0;
	ctx->last_boost_at = 0;
	ctx->wake_enq_at = 0;
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
	MLFQ_NR_STATS,
};

/**
 * struct mlfq_pcpu - per-CPU classifier state
 * @stat:	counters, summed over CPUs by the reader. Accumulated on the
 *		CPU that runs the event, so no counter is ever a shared line.
 * @q_runnable:	tasks of each level currently queued on this CPU's runqueue,
 *		indexed by level, so slot 0 is unused. Written only under this
 *		runqueue's lock, by the enqueue and dequeue hooks.
 * @wait_total:	nanoseconds of wakeup-to-CPU wait, over the episodes that
 *		ended on this CPU since the last controller step.
 * @wait_count:	episodes making up @wait_total.
 * @wake_total:	wakeup enqueues seen on this CPU since boot, for the reader.
 * @wake_window:	wakeup enqueues since the last controller step.
 *
 * scx_mlfq keeps the same split: one per-CPU array for the counters, and its
 * runnable occupancy derived from per-task ownership records rather than from
 * a shared total. The two window accumulators are per-CPU here where scx_mlfq
 * has the wait pair in its global gauge; see struct mlfq_sys_gauge.
 */
struct mlfq_pcpu {
	u64	stat[MLFQ_NR_STATS];
	u32	q_runnable[MLFQ_NR_QUEUES + 1];
	u64	wait_total;
	u32	wait_count;
	u64	wake_total;
	u32	wake_window;
};

DECLARE_PER_CPU(struct mlfq_pcpu, mlfq_pcpu);

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
 * mlfq_wakeup_episode_begin - start timing a wakeup's wait for a CPU
 * @ctx: the waking task's classification state
 * @now: the rq clock the caller already holds
 *
 * scx_mlfq marks the task with MLFQ_TF_ENQ_WAKEUP and stamps the time; here a
 * non-zero @wake_enq_at *is* that mark, so there is one field instead of two
 * and no way for them to disagree. Zero means no episode is in flight, hence
 * the sentinel-avoiding assignment; an rq clock is never actually zero, but
 * nothing here needs to rely on that.
 *
 * Also the arrival counter that the rate gauge folds. It is bumped for every
 * wakeup whether or not the controller is enabled, so that the dashboard shows
 * a real arrival rate on a system that is only being observed.
 */
static inline void mlfq_wakeup_episode_begin(struct mlfq_ctx *ctx, u64 now)
{
	ctx->wake_enq_at = now ? now : 1;

	__this_cpu_inc(mlfq_pcpu.wake_total);
	__this_cpu_inc(mlfq_pcpu.wake_window);
}

/**
 * mlfq_wakeup_episode_end - close a wakeup episode by reaching a CPU
 * @ctx: the task's classification state
 * @now: the rq clock the caller already holds
 *
 * Called when the task is given a CPU, which is where scx_mlfq takes the same
 * measurement, in ops.running(). A zero wait is recorded like any other: a
 * wakeup onto an idle CPU is exactly the case the gauge exists to distinguish
 * from a wakeup that had to queue, so it belongs in the average.
 */
static inline void mlfq_wakeup_episode_end(struct mlfq_ctx *ctx, u64 now)
{
	if (!ctx->wake_enq_at)
		return;

	__this_cpu_add(mlfq_pcpu.wait_total,
		       mlfq_elapsed(now, ctx->wake_enq_at));
	__this_cpu_inc(mlfq_pcpu.wait_count);
	ctx->wake_enq_at = 0;
}

/**
 * mlfq_wakeup_episode_drop - close a wakeup episode without measuring it
 * @ctx: the task's classification state
 *
 * Called from a re-enqueue that is not a wakeup, matching the point where
 * scx_mlfq clears MLFQ_TF_ENQ_WAKEUP on the same paths. Dropping it is the
 * point: a task requeued by load balancing or by a nice change has not been
 * waiting on a wakeup, and timing it to whenever it next runs would report a
 * queueing delay as a wakeup latency.
 */
static inline void mlfq_wakeup_episode_drop(struct mlfq_ctx *ctx)
{
	ctx->wake_enq_at = 0;
}

#endif /* _KERNEL_SCHED_MLFQ_H */
