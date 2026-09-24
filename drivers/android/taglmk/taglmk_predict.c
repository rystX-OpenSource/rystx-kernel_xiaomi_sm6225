// SPDX-License-Identifier: GPL-2.0
/*
 * TAGLMK - memory-cache-load prediction.
 *
 * A short window of Q4.4 cache-load samples is reduced into a smoothed mean
 * (EWMA), a burstiness estimate (mean absolute deviation) and a short-horizon
 * forecast, all carried in intfp (Q16.16).  The window reductions are done by
 * either the NEON accelerator or an equivalent scalar intfp-32 path; both
 * paths compute identical integer sums, so the resulting forecast - and every
 * decision derived from it - is the same with or without NEON.
 */
#include <linux/moduleparam.h>
#include <linux/spinlock.h>
#include <linux/string.h>

#include "taglmk.h"

/* EWMA smoothing factor, percent (0..100). */
static int pred_ewma_alpha_pct = 25;
module_param(pred_ewma_alpha_pct, int, 0644);

/* How strongly burstiness inflates the forecast headroom, percent. */
static int pred_burst_gain_pct = 50;
module_param(pred_burst_gain_pct, int, 0644);

/* Forecast horizon, in samples. */
static int pred_horizon = 4;
module_param(pred_horizon, int, 0644);

/* ------------------------------------------------------------------ */
/* Scalar (intfp-32) reductions.  Reference semantics for the NEON path. */

s32 taglmk_scalar_sum_q44(const q4_4_t *s, int n)
{
	s32 sum = 0;
	int i;

	for (i = 0; i < n; i++)
		sum += s[i];
	return sum;
}

s32 taglmk_scalar_absdev_q44(const q4_4_t *s, int n, q4_4_t mean)
{
	s32 acc = 0;
	int i;

	for (i = 0; i < n; i++) {
		s32 d = (s32)s[i] - mean;

		acc += (d < 0) ? -d : d;
	}
	return acc;
}

/* ------------------------------------------------------------------ */
/* Dispatch: prefer NEON when it is compiled in and usable right now. */

static s32 reduce_sum(const q4_4_t *s, int n)
{
	if (taglmk_neon_usable())
		return taglmk_neon_sum_q44(s, n);
	return taglmk_scalar_sum_q44(s, n);
}

static s32 reduce_absdev(const q4_4_t *s, int n, q4_4_t mean)
{
	if (taglmk_neon_usable())
		return taglmk_neon_absdev_q44(s, n, mean);
	return taglmk_scalar_absdev_q44(s, n, mean);
}

/* ------------------------------------------------------------------ */

void taglmk_predict_init(struct taglmk_predictor *p)
{
	memset(p, 0, sizeof(*p));
	spin_lock_init(&p->lock);
}

void taglmk_predict_push(struct taglmk_predictor *p, q4_4_t sample)
{
	unsigned long flags;

	spin_lock_irqsave(&p->lock, flags);
	p->samples[p->head] = sample;
	p->head = (p->head + 1) % TAGLMK_PRED_WINDOW;
	if (p->count < TAGLMK_PRED_WINDOW)
		p->count++;
	spin_unlock_irqrestore(&p->lock, flags);
}

/* Mean of a Q4.4 sample slice returned in Q16.16. */
static intfp_t slice_mean_fp(const q4_4_t *s, int n)
{
	s32 sum;

	if (n <= 0)
		return 0;
	sum = reduce_sum(s, n);
	/* sum is Q4.4; (sum << 12)/n converts the mean to Q16.16. */
	return (intfp_t)(((s64)sum << (TAGLMK_FP_FBITS - TAGLMK_Q44_FBITS)) / n);
}

static intfp_t pct_to_fp(int pct)
{
	if (pct < 0)
		pct = 0;
	return (intfp_t)(((s64)pct * TAGLMK_FP_ONE) / 100);
}

intfp_t taglmk_predict_eval(struct taglmk_predictor *p)
{
	q4_4_t buf[TAGLMK_PRED_WINDOW];
	unsigned long flags;
	unsigned int oldest, i, n;
	s32 sum, absdev;
	q4_4_t mean_raw;
	intfp_t mean_fp, burst_fp, slope_fp, forecast_fp, alpha_fp;

	/* Snapshot the ring in chronological order, then drop the lock so the
	 * (possibly NEON) reduction never runs under it. */
	spin_lock_irqsave(&p->lock, flags);
	n = p->count;
	oldest = (p->head + TAGLMK_PRED_WINDOW - n) % TAGLMK_PRED_WINDOW;
	for (i = 0; i < n; i++)
		buf[i] = p->samples[(oldest + i) % TAGLMK_PRED_WINDOW];
	spin_unlock_irqrestore(&p->lock, flags);

	if (n == 0)
		return 0;

	sum = reduce_sum(buf, n);
	mean_raw = (q4_4_t)(sum / (s32)n);
	mean_fp = (intfp_t)(((s64)sum << (TAGLMK_FP_FBITS - TAGLMK_Q44_FBITS)) / n);

	absdev = reduce_absdev(buf, n, mean_raw);
	burst_fp = (intfp_t)(((s64)absdev << (TAGLMK_FP_FBITS - TAGLMK_Q44_FBITS)) / n);

	/* Trend across the two halves of the window, per-sample slope. */
	slope_fp = 0;
	if (n >= 2) {
		unsigned int half = n / 2;
		intfp_t old_fp = slice_mean_fp(buf, half);
		intfp_t new_fp = slice_mean_fp(buf + (n - half), half);

		slope_fp = (new_fp - old_fp) / (intfp_t)half;
	}

	/* forecast = mean + slope*horizon + burst_gain*burstiness (headroom). */
	forecast_fp = mean_fp;
	forecast_fp += taglmk_fp_mul(slope_fp, (intfp_t)pred_horizon * TAGLMK_FP_ONE);
	forecast_fp += taglmk_fp_mul(burst_fp, pct_to_fp(pred_burst_gain_pct));
	if (forecast_fp < 0)
		forecast_fp = 0;

	alpha_fp = pct_to_fp(pred_ewma_alpha_pct);

	spin_lock_irqsave(&p->lock, flags);
	if (p->ewma == 0 && p->count <= 1)
		p->ewma = mean_fp;
	else
		p->ewma += taglmk_fp_mul(alpha_fp, mean_fp - p->ewma);
	p->burst = burst_fp;
	p->forecast = forecast_fp;
	spin_unlock_irqrestore(&p->lock, flags);

	return forecast_fp;
}
