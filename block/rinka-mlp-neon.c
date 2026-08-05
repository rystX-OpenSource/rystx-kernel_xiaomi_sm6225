// SPDX-License-Identifier: GPL-2.0
/*
 * RINKA MLP - NEON implementation (arm64)
 *
 * Vectorized inference for the quantized MLP. Must be called from a context
 * where may_use_simd() is true, wrapped in kernel_neon_begin()/end() by the
 * caller (see rinka-iosched.c: the inference kthread).
 *
 * Built with -mgeneral-regs-only removed; see block/Makefile.
 */

#include <asm/neon-intrinsics.h>

#include "rinka-mlp.h"

/*
 * Hidden layer: h[i] = ReLU((W1[i] . in + b1[i]<<Q) >> Q)
 *
 * RINKA_MLP_IN is 4, so each hidden unit's dot product is exactly one
 * 4-lane s16 multiply-widen into 4 s32 lanes, reduced horizontally.
 * With RINKA_MLP_HIDDEN == 8 we walk 8 rows; the weight matrix is small
 * enough that this stays in L1.
 */
static inline void rinka_mlp_hidden_neon(const struct rinka_mlp_weights *w,
					 int16x4_t vin,
					 s32 hidden[RINKA_MLP_HIDDEN])
{
	int i;

	for (i = 0; i < RINKA_MLP_HIDDEN; i++) {
		int16x4_t vw = vld1_s16(w->w1[i]);
		int32x4_t vprod = vmull_s16(vw, vin);
		s32 acc;

		/* Horizontal add of the 4 lanes. */
		int32x2_t vsum = vadd_s32(vget_low_s32(vprod),
					  vget_high_s32(vprod));
		vsum = vpadd_s32(vsum, vsum);
		acc = vget_lane_s32(vsum, 0);

		acc += (s32)w->b1[i] << RINKA_Q_SHIFT;
		acc >>= RINKA_Q_SHIFT;

		/* ReLU + saturate to the s16 range the next layer expects. */
		if (acc < 0)
			acc = 0;
		else if (acc > 32767)
			acc = 32767;

		hidden[i] = acc;
	}
}

s32 rinka_mlp_infer_neon(const struct rinka_mlp_weights *w,
			 const s16 in[RINKA_MLP_IN])
{
	s32 hidden[RINKA_MLP_HIDDEN];
	int16x4_t vin;
	int32x4_t vacc;
	int32x2_t vsum;
	s32 out;
	int i;

	vin = vld1_s16(in);
	rinka_mlp_hidden_neon(w, vin, hidden);

	/*
	 * Output layer: out = W2[0] . hidden + b2[0]<<Q
	 *
	 * hidden[] is s32 (post-ReLU, clamped to s16 range) while W2 is s16,
	 * so widen the weights to s32 and use a 4-lane s32 multiply, two
	 * iterations for the 8 hidden units.
	 */
	vacc = vdupq_n_s32(0);
	for (i = 0; i < RINKA_MLP_HIDDEN; i += 4) {
		int16x4_t vw16 = vld1_s16(&w->w2[0][i]);
		int32x4_t vw = vmovl_s16(vw16);
		int32x4_t vh = vld1q_s32(&hidden[i]);

		vacc = vmlaq_s32(vacc, vw, vh);
	}

	vsum = vadd_s32(vget_low_s32(vacc), vget_high_s32(vacc));
	vsum = vpadd_s32(vsum, vsum);
	out = vget_lane_s32(vsum, 0);

	out += (s32)w->b2[0] << RINKA_Q_SHIFT;
	out >>= RINKA_Q_SHIFT;

	if (out < RINKA_RATIO_MIN)
		out = RINKA_RATIO_MIN;
	else if (out > RINKA_RATIO_MAX)
		out = RINKA_RATIO_MAX;

	return out;
}
