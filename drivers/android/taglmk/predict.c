// SPDX-License-Identifier: GPL-2.0
/*
 * TAGLMK - memory cache load prediction.
 *
 * The killer reacts to what the machine looks like now, which on a small
 * device is already too late: by the time the file cache has collapsed the
 * pages that would have saved it are gone.  This file gives the core a short
 * view of the future instead, so a reclaim pass can be sized for the pressure
 * that is coming rather than the pressure that has already arrived.
 *
 * Two numbers come out of it.
 *
 * Burstiness is how violently the cache load moves relative to how large it is
 * - the mean absolute step between samples over the mean sample.  A steady load
 * sits near zero; a load that halves and doubles between passes approaches and
 * exceeds one.  It is a measure of how much the trend below can be trusted, not
 * of the trend itself.  The Q4.2 byte sysfs reports is the published form of it
 * and nothing more: the factor is built from the full precision ratio, because
 * a quarter step is a coarse thing to hand to two further multiplications.
 *
 * The factor (intfp32) is the multiplier the core applies to its base reclaim
 * budget.  It is built from a least squares fit of the window carried
 * TAGLMK_HORIZON passes past its end, how far that lands short of
 * free_file_limit, the burstiness as a confidence margin, and the per-profile
 * gain that makes a 2GB device react sooner than a 4GB one.  It is never below
 * one, so prediction can only ever ask for more work than the static budget,
 * never for less.
 *
 * Locking: the window is private to taglmk_predict_sample(), which only ever
 * runs from the pass work item under taglmk.lock, so it needs no protection of
 * its own.  The two results are published with WRITE_ONCE() because sysfs and
 * the reclaim path read them without any lock; a reader that catches the
 * previous pass' value simply acts on slightly older advice, which is exactly
 * what it would have done had it asked a moment earlier.
 *
 * Copyright (C) 2026 iDeadXS <datarafi43@gmail.com>
 */

#include <linux/compiler.h>
#include <linux/kernel.h>
#include <linux/math64.h>
#include <linux/string.h>
#include <linux/types.h>

#include "taglmk.h"

/*
 * Samples kept, and how far ahead the extrapolation looks.  The window is a
 * multiple of the accelerated kernels' wide step, so on a full window their
 * plain sums have no tail to walk, and short enough that sixteen passes of
 * history is still recent history.
 */
#define TAGLMK_WINDOW		16
#define TAGLMK_HORIZON		4

/* The factor is capped rather than left to saturate, so a pathological run of
 * samples can quadruple the reclaim budget but never turn one pass into an
 * unbounded walk of every task on the system.
 */
#define TAGLMK_FACTOR_MAX	(4U * TAGLMK_FP_ONE)

/*
 * Ceiling on the burstiness the margin below is formed from.  Two things want
 * one.  A load moving by four times its own mean between passes is already as
 * untrustworthy as the factor clamp can express, so nothing above this changes
 * an outcome; and taglmk_ratio_fp() saturates at U32_MAX when the mean of the
 * window has collapsed to almost nothing, where one plus the burstiness would
 * wrap round to a margin of nearly zero - the exact opposite of what a wildly
 * moving load ought to mean.
 */
#define TAGLMK_BURST_MAX	(4U * TAGLMK_FP_ONE)

/*
 * Oldest sample at [0], newest at [taglmk_nr_samples - 1].  Shifting down by
 * one on overflow costs a 60 byte memmove per pass and buys a window that is
 * always contiguous and always in order, which is what both the extrapolation
 * and the NEON loads want.  A ring would save the copy and pay for it in every
 * reader.
 */
static u32 taglmk_samples[TAGLMK_WINDOW];
static unsigned int taglmk_nr_samples;

static u8 taglmk_burst;
static u32 taglmk_factor = TAGLMK_FP_ONE;

/*
 * The three sums of struct taglmk_window, in plain C.  Every accumulator is an
 * exact 64 bit integer, exactly as in the NEON twin, so the two agree bit for
 * bit; the accelerator is chosen for speed alone and never changes an answer.
 * An empty window yields three zeros, which is what the twin returns too.
 */
static void taglmk_window_sums_scalar(const u32 *x, unsigned int n,
				      struct taglmk_window *w)
{
	u64 sum = 0;
	u64 weighted = 0;
	u64 absdiff = 0;
	unsigned int i;

	for (i = 0; i < n; i++) {
		sum += x[i];
		weighted += (u64)i * x[i];
	}

	for (i = 1; i < n; i++)
		absdiff += x[i] > x[i - 1] ? x[i] - x[i - 1]
					   : x[i - 1] - x[i];

	w->sum = sum;
	w->absdiff = absdiff;
	w->weighted = weighted;
}

static void taglmk_window_sums(const u32 *x, unsigned int n,
			       struct taglmk_window *w)
{
	if (taglmk_neon_ok()) {
		taglmk_neon_window_sums(x, n, w);
		return;
	}

	taglmk_window_sums_scalar(x, n, w);
}

/*
 * Sum of 0 .. n - 1, and n * (sum of i squared) - (sum of i) squared over the
 * same range.  Both are closed forms in @n, which is what lets a least squares
 * slope against the sample index come out of a single walk of the data: only
 * the cross term sum of i * x[i] depends on the samples at all.
 *
 * The variance term is n^2 * (n^2 - 1) / 12, which factors as n^2 * (n - 1) *
 * (n + 1) and is therefore an exact integer for every n: one of the three
 * consecutive numbers n - 1, n, n + 1 is divisible by three, and four divides
 * n^2 when n is even and (n - 1) * (n + 1) when it is odd.  At n = 2 it comes
 * out as one, where a fit over two points correctly degenerates to the
 * difference between them.  Both helpers are only ever called with the window
 * length, so the fourth power is nowhere near a 64 bit limit.
 */
static inline s64 taglmk_index_sum(unsigned int n)
{
	return (s64)n * (n - 1) / 2;
}

static inline s64 taglmk_index_var(unsigned int n)
{
	return (s64)n * n * ((s64)n * n - 1) / 12;
}

/*
 * Where the cache load is heading, TAGLMK_HORIZON passes out.
 *
 * The gradient is the least squares slope of the whole window against the
 * sample index.  The difference between the window's two ends is not that, and
 * is not robust to an outlier either: it rests on two samples out of sixteen,
 * and on the two sitting where a transient has had the least chance of being
 * averaged away, so one bad pass at either end tilts the prediction by the full
 * width of the outlier.  A fit gives every sample a say, weighted by how far
 * from the middle of the window it sits - in both directions, not just the
 * newest.
 *
 * Signed throughout.  The denominator grows as the fourth power of the window
 * and so does not fit the s32 divisor div_s64() takes, hence div64_s64(); both
 * are available on the 32 bit trees this driver is meant to backport to.
 */
static u32 taglmk_extrapolate(const struct taglmk_window *w, const u32 *x,
			      unsigned int n)
{
	s64 num;
	s64 slope;
	s64 predicted;

	if (n < 2)
		return x[0];

	num = (s64)n * (s64)w->weighted - taglmk_index_sum(n) * (s64)w->sum;
	slope = div64_s64(num, taglmk_index_var(n));
	predicted = (s64)x[n - 1] + slope * TAGLMK_HORIZON;

	if (predicted < 0)
		return 0;
	if (predicted > U32_MAX)
		return U32_MAX;

	return (u32)predicted;
}

/*
 * How much more than the base budget this pass should try to reclaim.
 *
 * Everything is in intfp32 and every term is at least zero, so the result can
 * only ever be at or above one:
 *
 *   urgency  how far the predicted cache load falls short of free_file_limit,
 *            as a fraction of that limit - zero when the prediction is safe,
 *            approaching one as it approaches nothing left
 *   margin   one plus the burstiness, so a load that is jumping around is
 *            treated as closer to the limit than its mean suggests
 *   gain     the profile's compensation, above one on devices with less RAM
 *
 * @burst_fp is the intfp32 ratio, not the Q4.2 byte sysfs publishes.  Rounding
 * to a quarter before multiplying by two further fractions would throw away
 * precision the ratio already carried and would flatten every burstiness below
 * a quarter to none at all, which is most of the useful range on a load that is
 * merely drifting.  It is clamped instead, at %TAGLMK_BURST_MAX.
 */
static u32 taglmk_factor_from(u32 predicted, u32 burst_fp)
{
	unsigned long limit = taglmk.free_file_limit;
	u32 urgency;
	u32 margin;
	u32 gain;
	u32 factor;

	if (!limit || predicted >= limit)
		return TAGLMK_FP_ONE;

	urgency = taglmk_ratio_fp(limit - predicted, limit);
	margin = TAGLMK_FP_ONE + min(burst_fp, TAGLMK_BURST_MAX);
	gain = taglmk_q42_to_fp(taglmk.profile->burst_gain);

	factor = TAGLMK_FP_ONE +
		 taglmk_fp_mul(taglmk_fp_mul(urgency, margin), gain);

	return min(factor, TAGLMK_FACTOR_MAX);
}

/**
 * taglmk_predict_sample - record this pass' cache load and re-predict
 *
 * Called once at the top of every pass, before any decision is taken, so the
 * burstiness and factor the rest of the pass reads describe the machine as it
 * is right now.  Must only be called from the pass work item.
 */
void taglmk_predict_sample(void)
{
	struct taglmk_window w;
	unsigned long load;
	u32 mean;
	u32 absdiff;
	u32 burst_fp;

	/*
	 * The cache load is both file LRUs together.  Only the active list
	 * decides whether the situation is critical, but the inactive list is
	 * where pages about to be dropped sit, so a cache that is collapsing
	 * shows up here a pass or two before it shows up in the ladder.
	 */
	load = taglmk_active_file_pages() + taglmk_inactive_file_pages();

	if (taglmk_nr_samples == TAGLMK_WINDOW) {
		memmove(taglmk_samples, taglmk_samples + 1,
			(TAGLMK_WINDOW - 1) * sizeof(*taglmk_samples));
		taglmk_nr_samples--;
	}

	taglmk_samples[taglmk_nr_samples++] =
		(u32)min_t(unsigned long, load, U32_MAX);

	taglmk_window_sums(taglmk_samples, taglmk_nr_samples, &w);

	/*
	 * The two means are taken here rather than in the kernels because a
	 * division has no business happening with softirqs off, and because the
	 * two do not share a denominator: a window of n samples holds n - 1
	 * steps between them.
	 *
	 * Burstiness is a ratio of two page counts, so it has to be formed with
	 * taglmk_ratio_fp(): converting either operand to intfp32 first would
	 * saturate it and the answer would always be one.
	 */
	mean = (u32)div_u64(w.sum, taglmk_nr_samples);
	absdiff = taglmk_nr_samples > 1 ?
		  (u32)div_u64(w.absdiff, taglmk_nr_samples - 1) : 0;
	burst_fp = taglmk_ratio_fp(absdiff, mean);

	/*
	 * The byte is published, the ratio is used.  taglmk_fp_to_q42() rounds
	 * down and saturates at 15.75: right for a number a human reads out of
	 * sysfs, wrong for one that two more multiplications depend on.
	 */
	WRITE_ONCE(taglmk_burst, taglmk_fp_to_q42(burst_fp));
	WRITE_ONCE(taglmk_factor,
		   taglmk_factor_from(taglmk_extrapolate(&w, taglmk_samples,
							 taglmk_nr_samples),
				      burst_fp));
}

/**
 * taglmk_predict_burstiness - how noisy the cache load has been, Q4.2
 *
 * Reported through sysfs so a device evaluation can see what the predictor
 * thinks it is looking at.
 */
u8 taglmk_predict_burstiness(void)
{
	return READ_ONCE(taglmk_burst);
}

/**
 * taglmk_predict_factor - the current budget multiplier, intfp32
 *
 * Never below %TAGLMK_FP_ONE and never above %TAGLMK_FACTOR_MAX.
 */
u32 taglmk_predict_factor(void)
{
	return READ_ONCE(taglmk_factor);
}

/**
 * taglmk_predict_budget - scale a base page budget by the prediction
 * @base: Pages the profile would reclaim with no prediction at all.
 *
 * Returns @base at rest and up to four times @base when the cache is
 * predicted to fall through free_file_limit.  Clamped at both ends so a
 * truncated multiply can never hand back less work than was asked for, and no
 * factor can ever hand back an unbounded amount.
 */
unsigned int taglmk_predict_budget(unsigned int base)
{
	unsigned long scaled;

	if (!base)
		return 0;

	scaled = taglmk_fp_int(taglmk_fp_mul(taglmk_fp(base),
					     READ_ONCE(taglmk_factor)));

	return clamp_t(unsigned long, scaled, base, (unsigned long)base * 4);
}
