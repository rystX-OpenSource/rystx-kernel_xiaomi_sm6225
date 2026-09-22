// SPDX-License-Identifier: GPL-2.0
/*
 * nap_nn_neon.c -- NEON float weight init + backprop for the NAP governor (arm64)
 *
 * On arm64 the float MLP cannot run in ->select() (IRQs disabled there; see
 * nap.h).  Instead this file runs from ->reflect(), where IRQs are enabled and
 * kernel_neon_begin() is legal.  It owns the float master weights: it
 * initializes them (Xavier), runs the online forward+backprop for the just-
 * finished idle, maintains the decayed floor histogram, and quantizes the
 * result into the Q16.16 shadow (d->weights_fx / bin_count_fx / log2_tres_fx)
 * that the fixed-point ->select() path (nap_fixp.c) reads.
 *
 * ALL functions here MUST be called only within kernel_neon_begin()/
 * kernel_neon_end().  This TU is built with -mgeneral-regs-only removed (see
 * the Makefile), the arm64 analogue of the x86 FPU-flag strip.
 *
 * NEON/ASIMD is a mandatory AArch64 baseline, so there is no runtime ISA
 * detection or dispatch key (unlike x86's optional AVX2 tier).  FMA (vfmaq_f32)
 * is likewise unconditional on AArch64; the fused rounding differs slightly
 * from the x86 SSE2 mul+add path, which only affects the last ULP of training
 * updates and is immaterial to the >= confidence decision.
 */

#include <linux/cpuidle.h>
#include <linux/kernel.h>
#include <linux/string.h>
#include <linux/types.h>

#include <asm/neon.h>
#include <asm/neon-intrinsics.h>

#include "nap.h"

#define NAP_FLOOR_WIN  256
#define NAP_PRIOR_K    16
#define NAP_PRNG_SEED  42u

/* ================================================================
 * Scalar float helpers (no libm; kernel has no float runtime)
 * ================================================================
 */

static inline float nap_sqrtf(float x)
{
	/* AArch64 has a native fsqrt; use the ASIMD scalar form explicitly. */
	return vget_lane_f32(vsqrt_f32(vdup_n_f32(x)), 0);
}

static inline float fclampf(float v, float lo, float hi)
{
	if (v < lo)
		return lo;
	if (v > hi)
		return hi;
	return v;
}

static inline float float_max(float a, float b) { return a > b ? a : b; }

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
		xi--;
	f = x - (float)xi;

	v.i = (u32)((xi + 127) << 23);
	return v.f * (1.0f + f * (0.6931472f +
			f * (0.2402265f + f * 0.0555041f)));
}

static inline float nap_sigmoidf(float x)
{
	return 1.0f / (1.0f + fast_exp2f(-1.4426950f * x));
}

static inline u32 nap_prng_next(u32 *state)
{
	*state = *state * 1664525u + 1013904223u;
	return *state;
}

static inline float nap_prng_float(u32 *state)
{
	return (float)(s32)nap_prng_next(state) * (1.0f / 2147483648.0f);
}

/* ================================================================
 * Fixed-point quantization of the float master into the shadow
 * ================================================================
 */

static inline fx_t to_fx(float v)
{
	float scaled = v * (float)NAP_FX_ONE;

	if (scaled >= 2147483520.0f)
		return (fx_t)0x7fffffff;
	if (scaled <= -2147483520.0f)
		return (fx_t)0x80000000;
	return (fx_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
}

static void nap_quantize_weights(struct nap_cpu_data *d)
{
	const struct nap_weights *w = &d->weights;
	struct nap_weights_fx *wf = &d->weights_fx;
	int i, j;

	for (j = 0; j < NAP_INPUT_SIZE; j++)
		for (i = 0; i < NAP_HIDDEN_SIZE; i++)
			wf->w_h1[j][i] = to_fx(w->w_h1[j][i]);
	for (i = 0; i < NAP_HIDDEN_SIZE; i++) {
		wf->b_h1[i]  = to_fx(w->b_h1[i]);
		wf->w_out[i] = to_fx(w->w_out[i]);
	}
	wf->b_out = to_fx(w->b_out);
	for (i = 0; i < NAP_NUM_CUTS; i++)
		wf->thr_ord[i] = to_fx(w->thr_ord[i]);
}

static void nap_quantize_bins(struct nap_cpu_data *d)
{
	int k;

	for (k = 0; k < CPUIDLE_STATE_MAX; k++)
		d->bin_count_fx[k] = to_fx(d->bin_count[k]);
}

/* ================================================================
 * Weight / threshold initialization (Xavier)
 * ================================================================
 */

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

	/* Neuron 0: pass-through for feature[0] = log2(sleep_length). */
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
		float tres = float_max((float)drv->states[i].target_residency,
				       1.0f);

		d->log2_tres[i] = fast_log2f(tres);
		d->log2_tres_fx[i] = to_fx(d->log2_tres[i]);
	}

	for (i = 1; i < drv->state_count; i++)
		d->weights.thr_ord[i - 1] = d->log2_tres[i];
}

void nap_neon_init(struct cpuidle_driver *drv, struct nap_cpu_data *d)
{
	nap_init_weights(&d->weights);
	d->active_w = &d->weights;
	nap_init_log2_tres(d, drv);

	memset(d->bin_count, 0, sizeof(d->bin_count));
	memset(d->bin_count_fx, 0, sizeof(d->bin_count_fx));

	nap_quantize_weights(d);

	d->have_sample = false;
	d->needs_learn = false;
	d->stats.learn_count = 0;
	d->reset_pending = false;
	d->fx_ready = true;
}

/* ================================================================
 * NEON forward pass (recompute hidden + score from stored features)
 * ================================================================
 */

static void nap_neon_forward(struct nap_cpu_data *d, float *score_out)
{
	const struct nap_weights *w = &d->weights;
	const float *in = d->features_f32;
	float32x4_t zero = vdupq_n_f32(0.0f);
	float32x4_t acc0 = vld1q_f32(&w->b_h1[0]);
	float32x4_t acc1 = vld1q_f32(&w->b_h1[4]);
	float32x4_t p0, p1;
	int j;

	for (j = 0; j < NAP_INPUT_SIZE; j++) {
		float32x4_t x = vdupq_n_f32(in[j]);

		acc0 = vfmaq_f32(acc0, vld1q_f32(&w->w_h1[j][0]), x);
		acc1 = vfmaq_f32(acc1, vld1q_f32(&w->w_h1[j][4]), x);
	}

	acc0 = vmaxq_f32(acc0, zero);		/* ReLU */
	acc1 = vmaxq_f32(acc1, zero);

	vst1q_f32(&d->hidden_out[0], acc0);
	vst1q_f32(&d->hidden_out[4], acc1);

	p0 = vmulq_f32(vld1q_f32(&w->w_out[0]), acc0);
	p1 = vmulq_f32(vld1q_f32(&w->w_out[4]), acc1);

	*score_out = vaddvq_f32(vaddq_f32(p0, p1)) + w->b_out;
}

/* ================================================================
 * NEON backpropagation (SGD, one sample)
 * ================================================================
 */

static void nap_neon_backprop(struct nap_cpu_data *d)
{
	float d_out_scalar = d->learn_d_out;
	float lr = d->learn_lr;
	float clamp_val = (float)d->max_grad_norm_millths / 1000.0f;
	float32x4_t vd = vdupq_n_f32(d_out_scalar);
	float32x4_t v_lr = vdupq_n_f32(lr);
	float32x4_t v_cl_hi = vdupq_n_f32(clamp_val);
	float32x4_t v_cl_lo = vdupq_n_f32(-clamp_val);
	float32x4_t zero = vdupq_n_f32(0.0f);
	float32x4_t dh0, dh1, h, g, w0, w1;
	uint32x4_t mask;
	int i;

	/* d_hid[j] = relu'(h[j]) * w_out[j] * d_out (branchless mask) */
	h = vld1q_f32(&d->hidden_out[0]);
	g = vmulq_f32(vld1q_f32(&d->weights.w_out[0]), vd);
	mask = vcgtq_f32(h, zero);
	dh0 = vbslq_f32(mask, g, zero);
	vst1q_f32(&d->learn_d_hid[0], dh0);

	h = vld1q_f32(&d->hidden_out[4]);
	g = vmulq_f32(vld1q_f32(&d->weights.w_out[4]), vd);
	mask = vcgtq_f32(h, zero);
	dh1 = vbslq_f32(mask, g, zero);
	vst1q_f32(&d->learn_d_hid[4], dh1);

	/* w_out[j] -= lr * clamp(h[j] * d_out) */
	w0 = vld1q_f32(&d->weights.w_out[0]);
	w1 = vld1q_f32(&d->weights.w_out[4]);
	w0 = vsubq_f32(w0, vmulq_f32(v_lr,
		vmaxq_f32(vminq_f32(vmulq_f32(vld1q_f32(&d->hidden_out[0]), vd),
				    v_cl_hi), v_cl_lo)));
	w1 = vsubq_f32(w1, vmulq_f32(v_lr,
		vmaxq_f32(vminq_f32(vmulq_f32(vld1q_f32(&d->hidden_out[4]), vd),
				    v_cl_hi), v_cl_lo)));
	vst1q_f32(&d->weights.w_out[0], w0);
	vst1q_f32(&d->weights.w_out[4], w1);

	/* b_out -= lr * clamp(d_out) */
	d->weights.b_out -= lr * fclampf(d_out_scalar, -clamp_val, clamp_val);

	/* w_h1[i][j] -= lr * clamp(feat[i] * d_hid[j]) */
	for (i = 0; i < NAP_INPUT_SIZE; i++) {
		float32x4_t vf = vdupq_n_f32(d->features_f32[i]);
		float32x4_t a0 = vld1q_f32(&d->weights.w_h1[i][0]);
		float32x4_t a1 = vld1q_f32(&d->weights.w_h1[i][4]);

		a0 = vsubq_f32(a0, vmulq_f32(v_lr,
			vmaxq_f32(vminq_f32(vmulq_f32(vf, dh0), v_cl_hi), v_cl_lo)));
		a1 = vsubq_f32(a1, vmulq_f32(v_lr,
			vmaxq_f32(vminq_f32(vmulq_f32(vf, dh1), v_cl_hi), v_cl_lo)));
		vst1q_f32(&d->weights.w_h1[i][0], a0);
		vst1q_f32(&d->weights.w_h1[i][4], a1);
	}

	/* b_h1[j] -= lr * clamp(d_hid[j]) */
	{
		float32x4_t b0 = vld1q_f32(&d->weights.b_h1[0]);
		float32x4_t b1 = vld1q_f32(&d->weights.b_h1[4]);

		b0 = vsubq_f32(b0, vmulq_f32(v_lr,
			vmaxq_f32(vminq_f32(dh0, v_cl_hi), v_cl_lo)));
		b1 = vsubq_f32(b1, vmulq_f32(v_lr,
			vmaxq_f32(vminq_f32(dh1, v_cl_hi), v_cl_lo)));
		vst1q_f32(&d->weights.b_h1[0], b0);
		vst1q_f32(&d->weights.b_h1[4], b1);
	}
}

/* ================================================================
 * Learning step + floor histogram + re-quantize shadow
 * ================================================================
 */

void nap_neon_learn(struct cpuidle_driver *drv, struct nap_cpu_data *d)
{
	float decay = (float)(NAP_FLOOR_WIN - 1) / (float)NAP_FLOOR_WIN;
	int k, label_bin = 0;
	bool learned = false;
	float s;

	if (!d->have_sample)
		return;

	d->active_w = &d->weights;

	/* Dequantize the features the fixed-point select() used, then forward. */
	for (k = 0; k < NAP_INPUT_SIZE; k++)
		d->features_f32[k] = (float)d->features_fx[k] / (float)NAP_FX_ONE;

	nap_neon_forward(d, &s);
	d->nn_output = s;

	if (d->needs_learn) {
		float base_lr = (float)d->learning_rate_millths / 1000.0f;
		float clamp_val = (float)d->max_grad_norm_millths / 1000.0f;
		float g = 0.0f;

		for (k = 1; k < drv->state_count; k++) {
			float th = d->weights.thr_ord[k - 1];
			float q = nap_sigmoidf(s - th);
			float y = (d->learn_actual_us >=
				   drv->states[k].target_residency) ? 1.0f : 0.0f;
			float err = q - y;
			float lo = d->log2_tres[k] - 6.0f;
			float hi = d->log2_tres[k] + 6.0f;

			g += err;
			d->weights.thr_ord[k - 1] =
				fclampf(th + fclampf(base_lr * err,
						     -clamp_val, clamp_val),
					lo, hi);
		}
		d->learn_d_out = g;
		d->learn_lr = base_lr;
		d->stats.learn_count++;
		nap_neon_backprop(d);
		d->needs_learn = false;
		learned = true;
	}

	/* Floor histogram update, every idle */
	for (k = 1; k < drv->state_count; k++)
		if (d->learn_actual_us >= drv->states[k].target_residency)
			label_bin = k;
	for (k = 0; k < drv->state_count; k++)
		d->bin_count[k] *= decay;
	d->bin_count[label_bin] += 1.0f;

	d->have_sample = false;

	/* Publish the updated shadow for the fixed-point select() path. */
	if (learned)
		nap_quantize_weights(d);
	nap_quantize_bins(d);
}
