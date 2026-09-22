// SPDX-License-Identifier: GPL-2.0
/*
 * nap_nn_avx2.c -- AVX2+FMA forward pass and backpropagation for the nap MLP
 *
 * 8->8 trunk + scalar score s feeding the ordinal survival head.
 * Uses 256-bit ymm registers: 8 hidden neurons = 1 ymm.
 * FMA via vfmadd231ps for fused multiply-add.
 *
 * Must be called within kernel_fpu_begin/end.
 * Compiled with: CFLAGS += -mavx2 -mfma
 *
 * 4.19 backport: unchanged from upstream -- pure float math, no cpuidle
 * unit-bearing field is touched here.
 */

#include "nap.h"

/* Aligned load/store: GCC translates v8sf* dereference to vmovaps */
static inline v8sf v8sf_load(const float *p)   { return *(const v8sf *)p; }
static inline void v8sf_store(float *p, v8sf v) { *(v8sf *)p = v; }

/* FMA: a*b+c -- vfmadd231ps: dest = src1 * src2 + dest */
static inline v8sf v8sf_fmadd(v8sf a, v8sf b, v8sf c)
{
	asm("vfmadd231ps %2, %1, %0" : "+x"(c) : "x"(a), "xm"(b));
	return c;
}

/* ymm clamp: max(min(v, hi), lo) */
static inline v8sf v8sf_clamp(v8sf v, v8sf lo, v8sf hi)
{
	return __builtin_ia32_maxps256(__builtin_ia32_minps256(v, hi), lo);
}

void nap_nn_forward_avx2(const float *input,
			 float *output,
			 float *hidden_save,
			 const struct nap_weights *w)
{
	int j;

	/* === Hidden layer: 8 outputs = 1xymm, 2-way accumulator === */
	v8sf acc0 = v8sf_load(&w->b_h1[0]);
	v8sf acc1 = V8SF_ZERO;

	for (j = 0; j < NAP_INPUT_SIZE; j += 2) {
		v8sf x0 = V8SF_SET1(input[j]);
		v8sf x1 = V8SF_SET1(input[j + 1]);

		acc0 = v8sf_fmadd(v8sf_load(&w->w_h1[j][0]),     x0, acc0);
		acc1 = v8sf_fmadd(v8sf_load(&w->w_h1[j + 1][0]), x1, acc1);
	}

	/* Merge accumulators + ReLU */
	{
		v8sf h = __builtin_ia32_maxps256(acc0 + acc1, V8SF_ZERO);

		v8sf_store(hidden_save, h);

		/* === Output layer: dot(hidden[8], w_out[8]) + b_out === */
		{
			v8sf p = v8sf_load(&w->w_out[0]) * h;

			/* Horizontal reduce: 8 -> 4 -> scalar */
			v4sf lo = __builtin_ia32_vextractf128_ps256(p, 0);
			v4sf hi = __builtin_ia32_vextractf128_ps256(p, 1);
			v4sf s4 = lo + hi;

			*output = s4[0] + s4[1] + s4[2] + s4[3] + w->b_out;
		}
	}
}

/*
 * Online learning (backpropagation) -- AVX2+FMA
 *
 * Output: scalar d_out (pre-computed by caller)
 * Hidden layer: 8 neurons = 1xymm
 */
void nap_nn_learn_avx2(struct nap_cpu_data *d)
{
	int i;
	float d_out_scalar = d->learn_d_out;
	float *d_hid = d->learn_d_hid;
	float lr = d->learn_lr;
	float clamp_val = (float)d->max_grad_norm_millths / 1000.0f;
	v8sf v_neg_lr = V8SF_SET1(-lr);
	v8sf v_cl_hi  = V8SF_SET1(clamp_val);
	v8sf v_cl_lo  = V8SF_SET1(-clamp_val);

	/*
	 * Hidden gradient: d_hid[j] = relu'(h[j]) * w_out[j] * d_out.
	 * vcmpps + vandps: branchless SIMD mask (1xymm = 8 neurons).
	 */
	v8sf dh;
	{
		v8sf vd = V8SF_SET1(d_out_scalar);
		v8sf g = v8sf_load(&d->active_w->w_out[0]) * vd;
		v8sf mask = __builtin_ia32_cmpps256(
				v8sf_load(&d->hidden_out[0]), V8SF_ZERO, 14);

		asm("vandps %2, %1, %0" : "=x"(dh) : "x"(g), "xm"(mask));
		v8sf_store(d_hid, dh);
	}

	/* Output weight update: w_out[j] -= lr * clamp(h[j] * d_out) */
	{
		v8sf vd = V8SF_SET1(d_out_scalar);
		v8sf *w = (v8sf *)&d->active_w->w_out[0];

		*w = v8sf_fmadd(v_neg_lr,
				v8sf_clamp(v8sf_load(&d->hidden_out[0]) * vd,
					   v_cl_lo, v_cl_hi),
				*w);
	}

	/* Output bias update (scalar) */
	d->active_w->b_out -= lr * fclampf(d_out_scalar, -clamp_val, clamp_val);

	/* Hidden weight update: w_h1[i][j] -= lr * clamp(feat[i] * d_hid[j]) */
	for (i = 0; i < NAP_INPUT_SIZE; i++) {
		v8sf vf = V8SF_SET1(d->features_f32[i]);
		v8sf *w = (v8sf *)&d->active_w->w_h1[i][0];

		*w = v8sf_fmadd(v_neg_lr,
				v8sf_clamp(vf * dh, v_cl_lo, v_cl_hi),
				*w);
	}

	/* Hidden bias update */
	{
		v8sf *b = (v8sf *)&d->active_w->b_h1[0];

		*b = v8sf_fmadd(v_neg_lr,
				v8sf_clamp(dh, v_cl_lo, v_cl_hi),
				*b);
	}
}
