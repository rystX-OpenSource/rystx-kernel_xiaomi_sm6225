/* SPDX-License-Identifier: GPL-2.0 */
/*
 * RINKA I/O Scheduler - MLP latency predictor
 *
 * Phase 3: A small quantized multi-layer perceptron that refines the
 * Phase-2 linear latency model.
 *
 * Design constraints that shape this code:
 *
 *  - The block layer completion and dispatch paths run with the queue lock
 *    held and IRQs disabled. may_use_simd() is false there, so NEON can
 *    never be used inline. Inference therefore runs in a kthread (process
 *    context, IRQs enabled) which publishes its results into an
 *    RCU-protected lookup table. The hot path only does a table lookup.
 *
 *  - The network predicts a *correction ratio* applied to the linear
 *    (base + slope*size) prediction rather than an absolute latency. This
 *    bounds the output range, keeps the fixed-point arithmetic well
 *    conditioned, and means neutral weights reproduce Phase-2 behaviour
 *    exactly.
 *
 * Arithmetic: Q8.8 fixed point in s16, accumulated in s32.
 */

#ifndef RINKA_MLP_H
#define RINKA_MLP_H

#include <linux/types.h>

/* Q8.8 fixed point */
#define RINKA_Q_SHIFT		8
#define RINKA_Q_ONE		(1 << RINKA_Q_SHIFT)	/* 1.0 */

/* Network geometry. Kept tiny: this must be cheap enough to evaluate over
 * the whole input grid every update period without being noticeable. */
#define RINKA_MLP_IN		4
#define RINKA_MLP_HIDDEN	8
#define RINKA_MLP_OUT		1

/*
 * Size buckets for the published lookup table. Index is derived from
 * blk_rq_bytes() by rinka_size_bucket().
 */
#define RINKA_LUT_SIZE_BUCKETS	8

/* Congestion buckets: in_flight scaled against target_depth. */
#define RINKA_LUT_DEPTH_BUCKETS	4

#define RINKA_LUT_ENTRIES \
	(RINKA_LUT_SIZE_BUCKETS * RINKA_LUT_DEPTH_BUCKETS)

/*
 * Clamp on the correction ratio the network may apply, in Q8.8.
 * A misbehaving or badly trained network cannot push the prediction
 * further than 0.25x .. 4x of the linear model.
 */
#define RINKA_RATIO_MIN		(RINKA_Q_ONE / 4)
#define RINKA_RATIO_MAX		(RINKA_Q_ONE * 4)

/* Quantized, offline-trained weights. */
struct rinka_mlp_weights {
	s16 w1[RINKA_MLP_HIDDEN][RINKA_MLP_IN];
	s16 b1[RINKA_MLP_HIDDEN];
	s16 w2[RINKA_MLP_OUT][RINKA_MLP_HIDDEN];
	s16 b2[RINKA_MLP_OUT];
};

/*
 * Published prediction table. Written by the inference kthread, read by the
 * dispatch/completion paths under rcu_read_lock().
 *
 * ratio[] holds the Q8.8 correction factor for each (size, depth) bucket.
 */
struct rinka_pred_lut {
	s16 ratio[RINKA_LUT_ENTRIES];
	struct rcu_head rcu;
};

static inline unsigned int rinka_lut_index(unsigned int size_bucket,
					   unsigned int depth_bucket)
{
	return size_bucket * RINKA_LUT_DEPTH_BUCKETS + depth_bucket;
}

/*
 * Map a request size to a bucket index. Buckets are power-of-two spaced
 * starting at 4K, which is where the Phase-2 model switches from the flat
 * base term to the size-dependent slope term.
 */
static inline unsigned int rinka_size_bucket(u32 bytes)
{
	unsigned int b;

	if (bytes <= 4096)
		return 0;

	/* 8K->1, 16K->2, ... saturating at the last bucket */
	b = ilog2((bytes - 1) >> 12) + 1;
	if (b >= RINKA_LUT_SIZE_BUCKETS)
		b = RINKA_LUT_SIZE_BUCKETS - 1;

	return b;
}

static inline unsigned int rinka_depth_bucket(unsigned int in_flight,
					      unsigned int target_depth)
{
	unsigned int b;

	if (!target_depth)
		return 0;

	b = (in_flight * RINKA_LUT_DEPTH_BUCKETS) / target_depth;
	if (b >= RINKA_LUT_DEPTH_BUCKETS)
		b = RINKA_LUT_DEPTH_BUCKETS - 1;

	return b;
}

/*
 * Portable integer reference implementation. Always built; used directly on
 * non-arm64 and as the fallback when kernel-mode NEON is unavailable.
 *
 * in[] and the returned value are Q8.8.
 */
s32 rinka_mlp_infer_int(const struct rinka_mlp_weights *w,
			const s16 in[RINKA_MLP_IN]);

#ifdef CONFIG_IOSCHED_RINKA_MLP_NEON
/*
 * NEON implementation. Caller must have verified may_use_simd() and wrapped
 * the call in kernel_neon_begin()/kernel_neon_end().
 */
s32 rinka_mlp_infer_neon(const struct rinka_mlp_weights *w,
			 const s16 in[RINKA_MLP_IN]);
#endif

/* Default weights: neutral, i.e. they reproduce the linear model. */
extern const struct rinka_mlp_weights rinka_mlp_default_weights;

#endif /* RINKA_MLP_H */
