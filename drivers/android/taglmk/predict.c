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
 * Burstiness (Q4.2) is how violently the cache load moves relative to how
 * large it is - the mean absolute step between samples over the mean sample.
 * A steady load sits near zero; a load that halves and doubles between passes
 * approaches and exceeds one.  It is a measure of how much the trend below can
 * be trusted, not of the trend itself.
 *
 * The factor (intfp32) is the multiplier the core applies to its base reclaim
 * budget.  It is built from a linear extrapolation of the window, how far that
 * extrapolation falls short of free_file_limit, the burstiness as a confidence
 * margin, and the per-profile gain that makes a 2GB device react sooner than a
 * 4GB one.  It is never below one, so prediction can only ever ask for more
 * work than the static budget, never for less.
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
 * multiple of four so the NEON kernels have no tail to walk on a full window,
 * and short enough that sixteen passes of history is still recent history.
 */
#define TAGLMK_WINDOW		16
#define TAGLMK_HORIZON		4

/* The factor is capped rather than left to saturate, so a pathological run of
 * samples can quadruple the reclaim budget but never turn one pass into an
 * unbounded walk of every task on the system.
 */
#define TAGLMK_FACTOR_MAX	(4U * TAGLMK_FP_ONE)

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
 * Mean of the window and mean absolute step across it.  Both are exact 64 bit
 * sums divided once at the end, so this and its NEON twin agree bit for bit;
 * the accelerator is chosen for speed alone and never changes an answer.
 */
static void taglmk_stats_scalar(const u32 *x, unsigned int n, u32 *out_mean,
				u32 *out_absdiff)
{
	u64 sum = 0;
	u64 absdiff = 0;
	unsigned int i;

	for (i = 0; i < n; i++)
		sum += x[i];

	for (i = 1; i < n; i++)
		absdiff += x[i] > x[i - 1] ? x[i] - x[i - 1]
					   : x[i - 1] - x[i];

	*out_mean = (u32)div_u64(sum, n);
	*out_absdiff = n > 1 ? (u32)div_u64(absdiff, n - 1) : 0;
}

static void taglmk_window_stats(const u32 *x, unsigned int n, u32 *out_mean,
				u32 *out_absdiff)
{
	if (taglmk_neon_ok()) {
		taglmk_neon_window_stats(x, n, out_mean, out_absdiff);
		return;
	}

	taglmk_stats_scalar(x, n, out_mean, out_absdiff);
}

/*
 * Where the cache load is heading, TAGLMK_HORIZON passes out.  The gradient is
 * taken across the whole window rather than between the last two samples so a
 * single outlying pass cannot swing the prediction; burstiness is what carries
 * the information that the samples are noisy.
 *
 * Signed throughout, and divided with div_s64() so this stays correct on the
 * 32 bit trees this driver is meant to backport to.
 */
static u32 taglmk_extrapolate(const u32 *x, unsigned int n)
{
	s64 slope;
	s64 predicted;

	if (n < 2)
		return x[0];

	slope = div_s64((s64)x[n - 1] - (s64)x[0], (s32)(n - 1));
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
 */
static u32 taglmk_factor_from(u32 predicted, u8 burst)
{
	unsigned long limit = taglmk.free_file_limit;
	u32 urgency;
	u32 margin;
	u32 gain;
	u32 factor;

	if (!limit || predicted >= limit)
		return TAGLMK_FP_ONE;

	urgency = taglmk_ratio_fp(limit - predicted, limit);
	margin = TAGLMK_FP_ONE + taglmk_q42_to_fp(burst);
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
	unsigned long load;
	u32 mean;
	u32 absdiff;
	u8 burst;

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

	taglmk_window_stats(taglmk_samples, taglmk_nr_samples, &mean, &absdiff);

	/*
	 * Burstiness is a ratio of two page counts, so it has to be formed
	 * with taglmk_ratio_fp(): converting either operand to intfp32 first
	 * would saturate it and the answer would always be one.
	 */
	burst = taglmk_fp_to_q42(taglmk_ratio_fp(absdiff, mean));

	WRITE_ONCE(taglmk_burst, burst);
	WRITE_ONCE(taglmk_factor,
		   taglmk_factor_from(taglmk_extrapolate(taglmk_samples,
							 taglmk_nr_samples),
				      burst));
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
