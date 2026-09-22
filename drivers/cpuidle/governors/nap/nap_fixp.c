// SPDX-License-Identifier: GPL-2.0
/*
 * nap_fixp.c -- integer fixed-point inference for the NAP governor (arm64)
 *
 * This file implements the ->select() inference path for arm64.  Unlike x86,
 * arm64's ->select() runs with hard IRQs disabled (do_idle() disables IRQs
 * before cpuidle_idle_call()), where kernel_neon_begin() would
 * BUG_ON(!may_use_simd()); and the kernel is built -mgeneral-regs-only, so
 * even scalar float is unavailable.  Everything here is therefore integer
 * Q16.16 fixed-point using general-purpose registers only, and this TU is
 * compiled with the normal kernel flags (no FPU/NEON, no flag removal).
 *
 * It reads the fixed-point shadow weights (d->weights_fx) that ->reflect()
 * produces by quantizing the float master after each learning step
 * (nap_nn_neon.c).  The forward pass and decision layer mirror the float
 * versions in nap_fpu.c; small numeric differences vs the float path are
 * expected and harmless (the decision is a >= confidence comparison).
 */

#include <linux/bitops.h>
#include <linux/cpuidle.h>
#include <linux/math64.h>
#include <linux/percpu.h>
#include <linux/pm_qos.h>
#include <linux/sched/clock.h>
#include <linux/tick.h>

#include "nap.h"

#define NAP_FLOOR_WIN  256
#define NAP_PRIOR_K    16

/* ================================================================
 * Fixed-point math (Q16.16)
 * ================================================================
 */

/* log2(x) for x >= 1, result in Q16.16.  Same minimax as fast_log2f(). */
static fx_t fx_log2_u64(u64 x)
{
	int b;
	u64 mant_fx;
	fx_t y, t;
	/* Q16.16 minimax coefficients: 1.4425, 0.7213, 0.4808 */
	const fx_t C1 = 94556, C2 = 47277, C3 = 31513;

	if (x <= 1)
		return 0;

	b = fls64(x) - 1;			/* floor(log2(x)) */

	if (b >= NAP_FX_SHIFT)
		mant_fx = x >> (b - NAP_FX_SHIFT);
	else
		mant_fx = x << (NAP_FX_SHIFT - b);
	/* mant_fx now in [1.0, 2.0) as Q16.16 */

	y = (fx_t)mant_fx - NAP_FX_ONE;		/* fractional mantissa in [0,1) */

	t = C3;
	t = fx_mul(y, t);
	t = C2 - t;
	t = fx_mul(y, t);
	t = C1 - t;
	t = fx_mul(y, t);

	return fx_from_int(b) + t;
}

/* 2^x in Q16.16 for x in Q16.16.  Same cubic as fast_exp2f(). */
static fx_t fx_exp2(fx_t x)
{
	int xi;
	fx_t f, poly, base;
	/* Q16.16: 0.6931472, 0.2402265, 0.0555041 */
	const fx_t A = 45426, B = 15743, C = 3638;
	const fx_t LIMIT = 30 * NAP_FX_ONE;

	if (x > LIMIT)
		x = LIMIT;
	else if (x < -LIMIT)
		x = -LIMIT;

	xi = x >> NAP_FX_SHIFT;			/* arithmetic shift floors */
	f = x - fx_from_int(xi);		/* fractional part in [0,1) */

	poly = C;
	poly = B + fx_mul(f, poly);
	poly = A + fx_mul(f, poly);
	poly = NAP_FX_ONE + fx_mul(f, poly);

	/* base = 2^xi in Q16.16 */
	if (xi >= 0) {
		if (xi > 45)			/* saturate well before overflow */
			return (fx_t)0x7fffffff;
		base = NAP_FX_ONE << xi;
	} else {
		if (xi < -31)
			return 0;
		base = NAP_FX_ONE >> (-xi);
	}

	return fx_mul(base, poly);
}

/* sigmoid(x) = 1 / (1 + 2^(-x*log2e)), result in Q16.16. */
static fx_t fx_sigmoid(fx_t x)
{
	const fx_t LOG2E = 94548;		/* 1.4426950 in Q16.16 */
	fx_t e = fx_exp2(fx_mul(-x, LOG2E));
	fx_t denom = NAP_FX_ONE + e;

	if (denom <= 0)
		return 0;
	return (fx_t)(((s64)NAP_FX_ONE << NAP_FX_SHIFT) / denom);
}

static inline fx_t fx_min(fx_t a, fx_t b) { return a < b ? a : b; }
static inline fx_t fx_max(fx_t a, fx_t b) { return a > b ? a : b; }

/* ================================================================
 * Feature extraction (fixed-point)
 * ================================================================
 */

struct fx_logring {
	fx_t avg;
	fx_t min;
	fx_t max;
};

/*
 * Recompute log2 statistics directly from the raw us history ring.  Cheaper
 * than caching a fixed-point log ring, and hist_count <= NAP_HISTORY_SIZE.
 */
static void fx_logring_compute(const struct nap_cpu_data *d,
			       struct fx_logring *s)
{
	int i, n = d->hist_count;
	s64 sum = 0;
	fx_t l;

	if (n == 0) {
		s->avg = s->min = s->max = 0;
		return;
	}

	l = fx_log2_u64(d->history[0] ? d->history[0] : 1);
	s->min = s->max = l;
	sum = l;

	for (i = 1; i < n; i++) {
		l = fx_log2_u64(d->history[i] ? d->history[i] : 1);
		sum += l;
		s->min = fx_min(s->min, l);
		s->max = fx_max(s->max, l);
	}

	s->avg = (fx_t)div_s64(sum, n);
}

static void fx_extract_features(struct cpuidle_driver *drv,
				struct cpuidle_device *dev,
				struct nap_cpu_data *d,
				s64 latency_req_us)
{
	struct fx_logring lr;
	ktime_t delta_tick;
	u64 sleep_us, busy_ns, busy_us, last_us;
	s64 err_us;
	fx_t *out = d->features_fx;

	sleep_us = ktime_to_us(tick_nohz_get_sleep_length(&delta_tick));
	busy_ns = local_clock() - d->prev_idle_exit;
	busy_us = div_u64(busy_ns, NSEC_PER_USEC);
	last_us = (dev->last_residency > 0) ? (u64)dev->last_residency : 0;

	out[0] = fx_log2_u64(sleep_us ? sleep_us : 1);
	out[1] = fx_log2_u64(last_us ? last_us : 1);

	fx_logring_compute(d, &lr);
	out[2] = lr.avg;
	out[3] = lr.min;
	out[4] = lr.max;

	/* out[5]: sign-preserving log2(|err_us| + 1) */
	err_us = d->last_prediction_error;
	{
		u64 mag = (err_us >= 0) ? (u64)err_us : (u64)(-err_us);
		fx_t l = fx_log2_u64(mag + 1);

		out[5] = (err_us < 0) ? -l : l;
	}

	out[6] = fx_log2_u64(busy_us ? busy_us : 1);

	/* out[7]: log2(latency_req) - log2(deepest_lat), 0 if unconstrained */
	{
		u64 deepest_lat = drv->states[drv->state_count - 1].exit_latency;
		bool lat_valid = (latency_req_us < PM_QOS_LATENCY_ANY &&
				  deepest_lat > 0);

		if (lat_valid)
			out[7] = fx_log2_u64((u64)latency_req_us ?: 1) -
				 fx_log2_u64(deepest_lat);
		else
			out[7] = 0;
	}

	d->last_predicted_us = (s64)sleep_us;
}

/* ================================================================
 * Forward pass (fixed-point)
 * ================================================================
 */

static fx_t fx_forward(struct nap_cpu_data *d)
{
	const struct nap_weights_fx *w = &d->weights_fx;
	const fx_t *in = d->features_fx;
	fx_t hidden[NAP_HIDDEN_SIZE];
	s64 acc;
	int i, j;

	/* Hidden layer: hidden[i] = ReLU(b_h1[i] + sum_j w_h1[j][i]*in[j]) */
	for (i = 0; i < NAP_HIDDEN_SIZE; i++) {
		acc = 0;
		for (j = 0; j < NAP_INPUT_SIZE; j++)
			acc += (s64)w->w_h1[j][i] * (s64)in[j];
		hidden[i] = w->b_h1[i] + (fx_t)(acc >> NAP_FX_SHIFT);
		if (hidden[i] < 0)
			hidden[i] = 0;			/* ReLU */
	}

	/* Score: s = b_out + sum_i w_out[i]*hidden[i] */
	acc = 0;
	for (i = 0; i < NAP_HIDDEN_SIZE; i++)
		acc += (s64)w->w_out[i] * (s64)hidden[i];

	return w->b_out + (fx_t)(acc >> NAP_FX_SHIFT);
}

/* ================================================================
 * Decision layer + entry point
 * ================================================================
 */

int nap_fixp_select(struct cpuidle_driver *drv,
		    struct cpuidle_device *dev,
		    struct nap_cpu_data *d, s64 latency_req_us)
{
	fx_t conf_fx, s, sleep_log2, qmin, total, suffix[CPUIDLE_STATE_MAX];
	int k, m = 0, idx = 0;

	fx_extract_features(drv, dev, d, latency_req_us);
	s = fx_forward(d);

	conf_fx = (fx_t)(((s64)d->conf_millths << NAP_FX_SHIFT) / 1000);
	sleep_log2 = d->features_fx[0];

	total = 0;
	for (k = 0; k < drv->state_count; k++)
		total += d->bin_count_fx[k];

	suffix[drv->state_count - 1] = d->bin_count_fx[drv->state_count - 1];
	for (k = drv->state_count - 2; k >= 0; k--)
		suffix[k] = suffix[k + 1] + d->bin_count_fx[k];

	qmin = NAP_FX_ONE;
	for (k = 1; k < drv->state_count; k++) {
		fx_t q_nn = fx_sigmoid(s - d->weights_fx.thr_ord[k - 1]);
		/* q = (K*q_nn + suffix[k]) / (K + total), Beta-Binomial shrinkage */
		s64 num = (s64)NAP_PRIOR_K * q_nn + suffix[k];
		s64 den = ((s64)NAP_PRIOR_K << NAP_FX_SHIFT) + total;
		fx_t q = (fx_t)((num << NAP_FX_SHIFT) / den);

		if (d->log2_tres_fx[k] > sleep_log2)
			q = 0;			/* cannot idle past next timer */
		if (q < qmin)
			qmin = q;
		q = qmin;

		if (q >= conf_fx)
			m = k;
		else
			break;
	}

	for (k = m; k >= 1; k--) {
		if (dev->states_usage[k].disable)
			continue;
		if (drv->states[k].exit_latency > latency_req_us)
			continue;
		idx = k;
		break;
	}
	return idx;
}
