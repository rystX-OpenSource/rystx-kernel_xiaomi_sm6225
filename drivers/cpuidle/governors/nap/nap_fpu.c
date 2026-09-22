// SPDX-License-Identifier: GPL-2.0
/*
 * nap_fpu.c -- FPU/SIMD code for the NAP cpuidle governor (x86_64)
 *
 * Compiled with FPU/SSE flags enabled (see the Makefile).  ALL functions here
 * MUST be called only from within kernel_fpu_begin()/kernel_fpu_end() blocks.
 * Keeping FPU code in a separate TU ensures the compiler cannot emit SSE/x87
 * in non-FPU paths (nap.c), which would corrupt userspace FPU state.
 *
 * 4.19 backport: this tree's cpuidle core is microsecond-based.  Every
 * residency/latency value here is US, and the log2 feature space is
 * log2(microseconds) so it lines up with the ordinal thresholds seeded from
 * log2(target_residency_us) in nap_init_log2_tres().  (The upstream 6.x patch
 * used *_ns fields and PM_QOS_LATENCY_ANY_NS; neither is correct here.)
 */

#include <linux/cpuidle.h>
#include <linux/math64.h>
#include <linux/percpu.h>
#include <linux/pm_qos.h>
#include <linux/sched/clock.h>
#include <linux/string.h>
#include <linux/tick.h>

#include "nap.h"

/* Clang lacks __builtin_ia32_movhlps; emulate with __builtin_shufflevector */
#ifdef __clang__
#define __builtin_ia32_movhlps(a, b) \
	__builtin_shufflevector(b, a, 2, 3, 6, 7)
#endif

/* ================================================================
 * Float math helpers
 * ================================================================ */

static inline float float_min(float a, float b) { return a < b ? a : b; }
static inline float float_max(float a, float b) { return a > b ? a : b; }

static inline float nap_sqrtf(float x)
{
	asm("sqrtss %1, %0" : "=x"(x) : "x"(x));
	return x;
}

static inline float fast_log2f(float x)
{
	union { float f; u32 i; } u = { .f = x };
	int exp = (int)((u.i >> 23) & 0xFFu) - 127;
	float e = (float)exp;
	float m, p;

	u.i = (u.i & 0x7FFFFFu) | (127u << 23);
	m = u.f - 1.0f;

	p = m * 0.4808f;
	p = 0.7213f - p;
	p = m * p;
	p = 1.4425f - p;
	p = m * p;

	return e + p;
}

static inline float fast_exp2f(float x)
{
	union { u32 i; float f; } v;
	int xi;
	float f;

	if (x > 60.0f)
		x = 60.0f;
	else if (x < -60.0f)
		x = -60.0f;

	xi = (int)x;
	if (x < (float)xi)
		xi--;			/* floor toward negative infinity */
	f = x - (float)xi;

	v.i = (u32)((xi + 127) << 23);	/* 2^xi */
	return v.f * (1.0f + f * (0.6931472f +
			f * (0.2402265f + f * 0.0555041f)));
}

static inline float nap_sigmoidf(float x)
{
	return 1.0f / (1.0f + fast_exp2f(-1.4426950f * x));
}

#define NAP_FLOOR_WIN  256
#define NAP_PRIOR_K    16

/* ================================================================
 * Deterministic PRNG for weight initialization (LCG)
 * ================================================================ */

#define NAP_PRNG_SEED 42u

static inline float nap_prng_float(u32 *state)
{
	*state = *state * 1664525u + 1013904223u;
	return (float)(s32)*state * (1.0f / 2147483648.0f);
}

/* ================================================================
 * ISA dispatch via static keys
 * ================================================================ */

static inline void nap_nn_forward(const float *input, float *output,
				  float *hidden_save,
				  const struct nap_weights *w)
{
	if (static_branch_unlikely(&nap_use_avx2))
		nap_nn_forward_avx2(input, output, hidden_save, w);
	else
		nap_nn_forward_sse2(input, output, hidden_save, w);
}

static inline void nap_nn_learn(struct nap_cpu_data *d)
{
	if (static_branch_unlikely(&nap_use_avx2))
		nap_nn_learn_avx2(d);
	else
		nap_nn_learn_sse2(d);
}

/* ================================================================
 * Weight initialization
 * ================================================================ */

static void nap_init_weights(struct nap_weights *w)
{
	u32 rng = NAP_PRNG_SEED;
	float scale_h1, scale_out;
	int i, j;

	scale_h1  = nap_sqrtf(6.0f / (float)(NAP_INPUT_SIZE + NAP_HIDDEN_SIZE));
	scale_out = 0.01f;

	for (i = 0; i < NAP_INPUT_SIZE; i++)
		for (j = 0; j < NAP_HIDDEN_SIZE; j++)
			w->w_h1[i][j] = nap_prng_float(&rng) * scale_h1;

	memset(w->b_h1, 0, sizeof(w->b_h1));

	for (j = 0; j < NAP_HIDDEN_SIZE; j++)
		w->w_out[j] = nap_prng_float(&rng) * scale_out;

	w->b_out = 0.0f;

	/*
	 * Neuron 0: pass-through for feature[0] = log2(sleep_length).
	 * Override the random init so initial output ~= input[0].
	 */
	for (i = 0; i < NAP_INPUT_SIZE; i++)
		w->w_h1[i][0] = 0.0f;
	w->w_h1[0][0] = 1.0f;
	w->b_h1[0] = 0.0f;
	w->w_out[0] = 1.0f;
}

static void nap_init_log2_tres(struct nap_cpu_data *d,
			       struct cpuidle_driver *drv)
{
	int i;

	for (i = 0; i < drv->state_count; i++) {
		/* 4.19 backport: target_residency is US -> log2(microseconds). */
		float tres = float_max(
			(float)drv->states[i].target_residency, 1.0f);

		d->log2_tres[i] = fast_log2f(tres);
	}

	for (i = 1; i < drv->state_count; i++)
		d->weights.thr_ord[i - 1] = d->log2_tres[i];
}

/* ================================================================
 * Feature extraction helpers
 * ================================================================ */

struct logring_stats {
	float avg;
	float min;
	float max;
};

static void logring_compute(const struct nap_cpu_data *d,
			    struct logring_stats *s)
{
	int i, n = d->hist_count;
	float sum;

	if (n == 0) {
		*s = (struct logring_stats){ 0 };
		return;
	}

	if (n == NAP_HISTORY_SIZE) {
		v4sf v0 = *(const v4sf *)&d->log_history[0];
		v4sf v1 = *(const v4sf *)&d->log_history[4];
		v4sf pmin, pmax, psum, t;

		pmin = __builtin_ia32_minps(v0, v1);
		pmax = __builtin_ia32_maxps(v0, v1);
		psum = v0 + v1;

		t = __builtin_ia32_movhlps(pmin, pmin);
		pmin = __builtin_ia32_minps(pmin, t);
		t = __builtin_ia32_movhlps(pmax, pmax);
		pmax = __builtin_ia32_maxps(pmax, t);
		t = __builtin_ia32_movhlps(psum, psum);
		psum = psum + t;

		t = __builtin_ia32_shufps(pmin, pmin, 0x55);
		pmin = __builtin_ia32_minps(pmin, t);
		t = __builtin_ia32_shufps(pmax, pmax, 0x55);
		pmax = __builtin_ia32_maxps(pmax, t);
		t = __builtin_ia32_shufps(psum, psum, 0x55);
		psum = psum + t;

		sum = psum[0];
		s->min = pmin[0];
		s->max = pmax[0];
	} else {
		float val;

		sum = d->log_history[0];
		s->min = sum;
		s->max = sum;

		for (i = 1; i < n; i++) {
			val = d->log_history[i];
			sum += val;
			s->min = float_min(s->min, val);
			s->max = float_max(s->max, val);
		}
	}

	s->avg = sum / (float)n;
}

static void nap_extract_features(struct cpuidle_driver *drv,
				 struct cpuidle_device *dev,
				 float out[NAP_INPUT_SIZE],
				 s64 latency_req_us)
{
	struct nap_cpu_data *d = this_cpu_ptr(&nap_data);
	struct logring_stats lr;
	ktime_t sleep_length, delta_tick;
	u64 busy_ns, busy_us, sleep_us, last_us;
	float log_inputs[4] __aligned(16);
	float log_results[4] __aligned(16);

	sleep_length = tick_nohz_get_sleep_length(&delta_tick);
	sleep_us = ktime_to_us(sleep_length);
	busy_ns = local_clock() - d->prev_idle_exit;
	busy_us = div_u64(busy_ns, NSEC_PER_USEC);
	last_us = (dev->last_residency > 0) ? (u64)dev->last_residency : 0;

	/*
	 * SSE log2 batch (4 values), all in US so the feature space matches
	 * the log2(target_residency_us) thresholds:
	 *   [0] sleep_length_us   -> out[0]
	 *   [1] last_residency_us -> out[1], also into log_history
	 *   [2] busy_us           -> out[6]
	 *   [3] |pred_error_us|+1 -> out[5] (sign restored after)
	 */
	{
		float err_f = (float)d->last_prediction_error;	/* already US */
		float abs_err = (err_f >= 0.0f) ? err_f : -err_f;

		log_inputs[0] = float_max((float)sleep_us, 1.0f);
		log_inputs[1] = float_max((float)last_us, 1.0f);
		log_inputs[2] = float_max((float)busy_us, 1.0f);
		log_inputs[3] = abs_err + 1.0f;

		{
			v4sf log_in  = *(const v4sf *)log_inputs;
			v4sf log_out = fast_log2f_sse(log_in);
			*(v4sf *)log_results = log_out;
		}

		out[0] = log_results[0];
		out[1] = log_results[1];
		out[6] = log_results[2];

		{
			union { float f; u32 i; } res = { .f = log_results[3] };
			union { float f; u32 i; } sgn = { .f = err_f };

			res.i |= sgn.i & 0x80000000u;
			out[5] = res.f;
		}
	}

	{
		int prev = (d->hist_idx - 1 + NAP_HISTORY_SIZE) % NAP_HISTORY_SIZE;

		d->log_history[prev] = log_results[1];
	}

	logring_compute(d, &lr);
	out[2] = lr.avg;
	out[3] = lr.min;
	out[4] = lr.max;

	/* out[7]: log2(latency_req) - log2(deepest_lat), 0 if unconstrained */
	{
		u64 deepest_lat = drv->states[drv->state_count - 1].exit_latency;
		bool lat_valid = (latency_req_us < PM_QOS_LATENCY_ANY &&
				  deepest_lat > 0);

		if (lat_valid)
			out[7] = fast_log2f(float_max((float)latency_req_us, 1.0f))
			       - fast_log2f(float_max((float)deepest_lat, 1.0f));
		else
			out[7] = 0.0f;
	}

	d->last_predicted_us = (s64)sleep_us;
}

/* ================================================================
 * FPU entry point for nap_select
 * Called within kernel_fpu_begin()/kernel_fpu_end().
 * Returns selected idle-state index (>= 0), or -1 to fall back.
 * ================================================================ */

int nap_fpu_select(struct cpuidle_driver *drv,
		   struct cpuidle_device *dev,
		   struct nap_cpu_data *d)
{
	s64 latency_req_us = cpuidle_governor_latency_req(dev->cpu);

	if (unlikely(d->reset_pending)) {
		nap_init_weights(&d->weights);
		nap_init_log2_tres(d, drv);
		memset(d->bin_count, 0, sizeof(d->bin_count));
		d->have_sample = false;
		d->stats.learn_count = 0;
		d->needs_learn = false;
		d->reset_pending = false;
	}

	if (d->have_sample) {
		float decay = (float)(NAP_FLOOR_WIN - 1) / (float)NAP_FLOOR_WIN;
		int k, label_bin = 0;

		if (d->needs_learn) {
			float base_lr = (float)d->learning_rate_millths / 1000.0f;
			float clamp_val = (float)d->max_grad_norm_millths / 1000.0f;
			float s = d->nn_output;
			float g = 0.0f;

			for (k = 1; k < drv->state_count; k++) {
				float th = d->active_w->thr_ord[k - 1];
				float q = nap_sigmoidf(s - th);
				float y = (d->learn_actual_us >=
					   drv->states[k].target_residency)
					  ? 1.0f : 0.0f;
				float err = q - y;
				float lo = d->log2_tres[k] - 6.0f;
				float hi = d->log2_tres[k] + 6.0f;

				g += err;
				d->active_w->thr_ord[k - 1] =
					fclampf(th + fclampf(base_lr * err,
							     -clamp_val, clamp_val),
						lo, hi);
			}
			d->learn_d_out = g;
			d->learn_lr = base_lr;
			d->stats.learn_count++;
			nap_nn_learn(d);
			d->needs_learn = false;
		}

		for (k = 1; k < drv->state_count; k++)
			if (d->learn_actual_us >=
			    drv->states[k].target_residency)
				label_bin = k;
		for (k = 0; k < drv->state_count; k++)
			d->bin_count[k] *= decay;
		d->bin_count[label_bin] += 1.0f;

		d->have_sample = false;
	}

	nap_extract_features(drv, dev, d->features_f32, latency_req_us);

	d->active_w = &d->weights;

	nap_nn_forward(d->features_f32, &d->nn_output, d->hidden_out,
		       d->active_w);

	{
		float conf = (float)d->conf_millths / 1000.0f;
		float s = d->nn_output;
		float sleep_log2 = d->features_f32[0];
		float suffix[CPUIDLE_STATE_MAX];
		float total = 0.0f;
		float qmin = 1.0f;
		int k, m = 0, idx = 0;

		for (k = 0; k < drv->state_count; k++)
			total += d->bin_count[k];

		suffix[drv->state_count - 1] =
			d->bin_count[drv->state_count - 1];
		for (k = drv->state_count - 2; k >= 0; k--)
			suffix[k] = suffix[k + 1] + d->bin_count[k];

		for (k = 1; k < drv->state_count; k++) {
			float q_nn = nap_sigmoidf(s - d->active_w->thr_ord[k - 1]);
			float q = ((float)NAP_PRIOR_K * q_nn + suffix[k]) /
				  ((float)NAP_PRIOR_K + total);

			if (d->log2_tres[k] > sleep_log2)
				q = 0.0f;	/* cannot idle past next timer */
			if (q < qmin)
				qmin = q;
			q = qmin;

			if (q >= conf)
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
}
