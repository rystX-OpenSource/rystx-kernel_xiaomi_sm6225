// SPDX-License-Identifier: GPL-2.0
/*
 * nap_nn_sse2.c -- SSE2 forward pass and backpropagation for the nap MLP
 *
 * 8->8 trunk + scalar score s feeding the ordinal survival head.
 * Baseline implementation using SSE2, which is always available on x86_64.
 * No FMA -- uses separate mul + add (2 instructions per MAC).
 *
 * Must be called within kernel_fpu_begin/end.
 * Compiled with: CFLAGS += -msse2
 *
 * 4.19 backport: unchanged from upstream -- this file operates purely on the
 * float weight/feature arrays and does not touch any cpuidle unit-bearing
 * field, so the us/ns reconciliation does not apply here.
 */

#include "nap.h"

/* Aligned load/store */
static inline v4sf v4sf_load(const float *p)   { return *(const v4sf *)p; }
static inline void v4sf_store(float *p, v4sf v) { *(v4sf *)p = v; }

/* ReLU helper */
static inline v4sf v4sf_max(v4sf a, v4sf b)
{
	return __builtin_ia32_maxps(a, b);
}

void nap_nn_forward_sse2(const float *input,
			 float *output,
			 float *hidden_save,
			 const struct nap_weights *w)
{
	int j;

	/* === Hidden layer: 8 outputs = 2xxmm === */
	v4sf acc0 = v4sf_load(&w->b_h1[0]);
	v4sf acc1 = v4sf_load(&w->b_h1[4]);

	for (j = 0; j < NAP_INPUT_SIZE; j++) {
		v4sf x = V4SF_SET1(input[j]);

		acc0 += v4sf_load(&w->w_h1[j][0]) * x;
		acc1 += v4sf_load(&w->w_h1[j][4]) * x;
	}

	/* ReLU */
	{
		v4sf zero = V4SF_SET1(0.0f);

		acc0 = v4sf_max(acc0, zero);
		acc1 = v4sf_max(acc1, zero);
	}
	v4sf_store(&hidden_save[0], acc0);
	v4sf_store(&hidden_save[4], acc1);

	/* === Output layer: dot(hidden[8], w_out[8]) + b_out -> 1 scalar === */
	{
		v4sf p0 = v4sf_load(&w->w_out[0]) * acc0;
		v4sf p1 = v4sf_load(&w->w_out[4]) * acc1;
		v4sf sum = p0 + p1;

		*output = sum[0] + sum[1] + sum[2] + sum[3] + w->b_out;
	}
}

/*
 * Online learning (backpropagation) -- SSE2
 *
 * Output: scalar d_out (pre-computed by caller)
 * Hidden layer: 8 neurons = 2xxmm
 */
void nap_nn_learn_sse2(struct nap_cpu_data *d)
{
	int i;
	float d_out_scalar = d->learn_d_out;
	float *d_hid = d->learn_d_hid;
	float lr = d->learn_lr;
	float clamp_val = (float)d->max_grad_norm_millths / 1000.0f;
	v4sf v_lr    = V4SF_SET1(lr);
	v4sf v_cl_hi = V4SF_SET1(clamp_val);
	v4sf v_cl_lo = V4SF_SET1(-clamp_val);

	/*
	 * Hidden gradient: d_hid[j] = relu'(h[j]) * w_out[j] * d_out.
	 * Must be computed before the output weight update to use pre-update
	 * w_out.
	 */
	{
		v4sf vd = V4SF_SET1(d_out_scalar);
		v4sf zero = V4SF_SET1(0.0f);
		v4sf h, g;
		v4si m;

		h = v4sf_load(&d->hidden_out[0]);
		g = v4sf_load(&d->active_w->w_out[0]) * vd;
		m = (v4si)(h > zero);
		v4sf_store(&d_hid[0], v4si_as_v4sf(v4sf_as_v4si(g) & m));

		h = v4sf_load(&d->hidden_out[4]);
		g = v4sf_load(&d->active_w->w_out[4]) * vd;
		m = (v4si)(h > zero);
		v4sf_store(&d_hid[4], v4si_as_v4sf(v4sf_as_v4si(g) & m));
	}

	/* Output weight update: w_out[j] -= lr * clamp(h[j] * d_out) */
	{
		v4sf vd = V4SF_SET1(d_out_scalar);
		v4sf *w = (v4sf *)&d->active_w->w_out[0];

		w[0] -= v_lr * v4sf_clamp(v4sf_load(&d->hidden_out[0]) * vd,
					  v_cl_lo, v_cl_hi);
		w[1] -= v_lr * v4sf_clamp(v4sf_load(&d->hidden_out[4]) * vd,
					  v_cl_lo, v_cl_hi);
	}

	/* Output bias update: b_out -= lr * clamp(d_out) */
	d->active_w->b_out -= lr * fclampf(d_out_scalar, -clamp_val, clamp_val);

	/* Hidden weight update: w_h1[i][j] -= lr * clamp(feat[i] * d_hid[j]) */
	{
		v4sf dh0 = *(const v4sf *)&d_hid[0];
		v4sf dh1 = *(const v4sf *)&d_hid[4];

		for (i = 0; i < NAP_INPUT_SIZE; i++) {
			v4sf vf = V4SF_SET1(d->features_f32[i]);
			v4sf *w = (v4sf *)&d->active_w->w_h1[i][0];

			w[0] -= v_lr * v4sf_clamp(vf * dh0, v_cl_lo, v_cl_hi);
			w[1] -= v_lr * v4sf_clamp(vf * dh1, v_cl_lo, v_cl_hi);
		}

		/* Hidden bias update: b_h1[j] -= lr * clamp(d_hid[j]) */
		{
			v4sf *b = (v4sf *)&d->active_w->b_h1[0];

			b[0] -= v_lr * v4sf_clamp(dh0, v_cl_lo, v_cl_hi);
			b[1] -= v_lr * v4sf_clamp(dh1, v_cl_lo, v_cl_hi);
		}
	}
}
