// SPDX-License-Identifier: GPL-2.0
/*
 * RINKA MLP - integer reference implementation
 *
 * Portable C implementation of the quantized MLP inference.
 * Always built; used on non-arm64 and as fallback when NEON unavailable.
 */

#include "rinka-mlp.h"

/* ReLU activation in Q8.8 */
static inline s16 relu_q88(s32 x)
{
	if (x < 0)
		return 0;
	if (x > 32767)
		return 32767;
	return (s16)x;
}

/* Q8.8 × Q8.8 → Q8.8 with saturation */
static inline s16 mul_q88(s16 a, s16 b)
{
	s32 prod = ((s32)a * (s32)b) >> RINKA_Q_SHIFT;
	if (prod > 32767)
		return 32767;
	if (prod < -32768)
		return -32768;
	return (s16)prod;
}

/*
 * MLP forward pass: 4 → 8 → 1 with ReLU hidden activation.
 * Returns Q8.8 correction ratio clamped to [0.25x, 4x].
 */
s32 rinka_mlp_infer_int(const struct rinka_mlp_weights *w,
			const s16 in[RINKA_MLP_IN])
{
	s32 hidden[RINKA_MLP_HIDDEN];
	s32 out;
	int i, j;

	/* Hidden layer: h = ReLU(W1 * in + b1) */
	for (i = 0; i < RINKA_MLP_HIDDEN; i++) {
		s32 acc = (s32)w->b1[i] << RINKA_Q_SHIFT;

		for (j = 0; j < RINKA_MLP_IN; j++)
			acc += (s32)w->w1[i][j] * (s32)in[j];

		hidden[i] = relu_q88(acc >> RINKA_Q_SHIFT);
	}

	/* Output layer: out = W2 * hidden + b2 */
	out = (s32)w->b2[0] << RINKA_Q_SHIFT;
	for (i = 0; i < RINKA_MLP_HIDDEN; i++)
		out += (s32)w->w2[0][i] * hidden[i];

	out >>= RINKA_Q_SHIFT;

	/* Clamp to [0.25x, 4x] in Q8.8 */
	if (out < RINKA_RATIO_MIN)
		out = RINKA_RATIO_MIN;
	if (out > RINKA_RATIO_MAX)
		out = RINKA_RATIO_MAX;

	return out;
}

/*
 * Default weights: neutral network that outputs 1.0 for any input.
 * This reproduces the Phase-2 linear model exactly.
 */
const struct rinka_mlp_weights rinka_mlp_default_weights = {
	/* w1: all zeros */
	.w1 = {{0}},
	/* b1: small positive values to keep hidden units active */
	.b1 = {
		RINKA_Q_ONE / 8, RINKA_Q_ONE / 8,
		RINKA_Q_ONE / 8, RINKA_Q_ONE / 8,
		RINKA_Q_ONE / 8, RINKA_Q_ONE / 8,
		RINKA_Q_ONE / 8, RINKA_Q_ONE / 8,
	},
	/* w2: all zeros */
	.w2 = {{0}},
	/* b2: 1.0 in Q8.8 */
	.b2 = {RINKA_Q_ONE},
};
