// SPDX-License-Identifier: GPL-2.0
/*
 * TAGLMK - ARM64 NEON accelerated kernels.
 *
 * The predictor and the ZRAM balancer both boil a small array of per-pass or
 * per-task page counts down to a handful of sums.  The arrays are short, so
 * this is not about throughput; it is about how long a pass spends working
 * while the machine is already short of memory.
 *
 * Two things buy that back on such short arrays.  A reduction is turned into
 * two independent accumulators over eight lanes a step, because a sixteen
 * element window is otherwise only four accumulate instructions long and every
 * one of them waits on the last - latency, not width, is what a short reduction
 * spends its time on.  And every widening multiply is issued in place, low half
 * and high half, so a loop body that needs two products issues four multiply
 * longs back to back with no lane extracted and nothing to fold in between.
 *
 * Three rules hold everywhere in this file.
 *
 * Exactness.  Every accumulator is 64 bit and every operation is an integer add
 * or multiply, so each kernel produces bit for bit the same answer as the
 * scalar twin it stands in for.  Turning the accelerator on or off changes how
 * long a pass takes and nothing else, which is what makes the fallback path
 * trustworthy and the two paths comparable during a device evaluation.
 *
 * Context.  kernel_neon_begin() will BUG_ON(!may_use_simd()) and then disables
 * softirqs for the duration, so every caller has to have asked taglmk_neon_ok()
 * immediately beforehand, and nothing between begin and end may sleep, fault,
 * or take a lock.  These kernels only read the arrays they are handed, write
 * one caller supplied buffer, and do integer arithmetic.
 *
 * Separation.  All of the intrinsics live in the __taglmk_*() inner functions,
 * which are noinline, and the outer functions that bracket them with
 * kernel_neon_begin()/kernel_neon_end() mention no vector type at all.  That is
 * not style: a vector local declared in the outer function could be
 * materialised, spilled, or reloaded by the compiler on either side of the
 * bracket, which would touch the FPU with userspace's registers still live in
 * it.  Keeping the two apart makes the hazard structurally impossible rather
 * than merely unlikely, and is why lib/raid6 splits its NEON code the same way.
 *
 * Copyright (C) 2026 iDeadXS <datarafi43@gmail.com>
 */

/*
 * Before anything else: this header repairs the 64 bit type definitions gcc
 * would otherwise hand <arm_neon.h>, which it then includes.  It has to come
 * first so no other header can pull the intrinsics in behind its back.
 */
#include <asm/neon-intrinsics.h>

#include <linux/compiler.h>
#include <linux/kernel.h>
#include <linux/math64.h>
#include <linux/types.h>

#include <asm/neon.h>
#include <asm/simd.h>

#include "taglmk.h"

/*
 * Lanes per vector, and the wide step the reductions run at.  Every loop below
 * is written as "while a whole step still fits", so the bound is the length
 * itself and no rounding is needed; a step that no longer fits falls through to
 * the narrower one, and whatever is left is finished by a scalar tail.
 */
#define TAGLMK_LANES		4
#define TAGLMK_LANES_WIDE	(2 * TAGLMK_LANES)

/* The two lanes of a 64 bit accumulator, added up. */
static inline u64 taglmk_reduce_u64(uint64x2_t v)
{
	return vgetq_lane_u64(v, 0) + vgetq_lane_u64(v, 1);
}

/*
 * acc += a * b, widening every lane to 64 bits first.  Two u32 factors need 64
 * bits to hold their product, so this is the only form that cannot overflow
 * however large the page counts get.
 *
 * UMULL reads the low half of both operands and UMULL2 the high half of both in
 * place, so neither vector has to be pulled apart first.  The two products are
 * then folded into each other before they reach @acc, which leaves one step of
 * the carried dependency per call rather than two; unsigned 64 bit addition is
 * associative modulo 2^64, so regrouping it costs no exactness.
 */
static inline uint64x2_t taglmk_mla_widen(uint64x2_t acc, uint32x4_t a,
					  uint32x4_t b)
{
	uint64x2_t lo = vmull_u32(vget_low_u32(a), vget_low_u32(b));
	uint64x2_t hi = vmull_high_u32(a, b);

	return vaddq_u64(acc, vaddq_u64(lo, hi));
}

/*
 * (v * scale) >> TAGLMK_FP_SHIFT across four lanes, each product formed in 64
 * bits.  SHRN shifts and narrows to 32 bits in one instruction, which is the
 * same truncation the scalar twin's cast performs and one instruction fewer
 * than shifting and moving down separately.
 */
static inline uint32x4_t taglmk_scale_q(uint32x4_t v, u32 scale)
{
	uint64x2_t lo = vmull_n_u32(vget_low_u32(v), scale);
	uint64x2_t hi = vmull_high_n_u32(v, scale);

	return vcombine_u32(vshrn_n_u64(lo, TAGLMK_FP_SHIFT),
			    vshrn_n_u64(hi, TAGLMK_FP_SHIFT));
}

/*
 * acc += |x[i] - x[i-1]| across four lanes.  UABD is one instruction where the
 * scalar twin needs a branch per element, and UADALP widens and accumulates
 * without a separate add - which is the whole reason this kernel exists.
 */
static inline uint64x2_t taglmk_step_acc(uint64x2_t acc, const u32 *x)
{
	return vpadalq_u32(acc, vabdq_u32(vld1q_u32(x), vld1q_u32(x - 1)));
}

/**
 * taglmk_neon_ok - may the accelerated kernels be used from here, right now?
 *
 * Both halves matter and both have to be asked again at every call site.  The
 * CPU half never changes after boot; the context half is a property of the
 * caller, and a path that is fine to accelerate on one call may not be on the
 * next.  Caching this answer would be a way to turn a policy decision into a
 * BUG_ON inside kernel_neon_begin().
 */
bool taglmk_neon_ok(void)
{
	return cpu_has_neon() && may_use_simd();
}

/*
 * Sum of @x, and sum of |x[i] - x[i-1]| across it.  Raw sums: the caller
 * divides, outside the NEON region.
 *
 * Both reductions run two accumulators over eight lanes a step and fold them
 * together once at the end.  On a full sixteen element window that is two
 * iterations of two independent chains rather than four of one, and the fold
 * costs a single add.
 */
static noinline void __taglmk_window_sums(const u32 *x, unsigned int n,
					  u64 *out_sum, u64 *out_absdiff)
{
	uint64x2_t vsum0 = vdupq_n_u64(0);
	uint64x2_t vsum1 = vdupq_n_u64(0);
	uint64x2_t vdiff0 = vdupq_n_u64(0);
	uint64x2_t vdiff1 = vdupq_n_u64(0);
	u64 sum;
	u64 absdiff;
	unsigned int i;

	for (i = 0; i + TAGLMK_LANES_WIDE <= n; i += TAGLMK_LANES_WIDE) {
		vsum0 = vpadalq_u32(vsum0, vld1q_u32(x + i));
		vsum1 = vpadalq_u32(vsum1, vld1q_u32(x + i + TAGLMK_LANES));
	}

	for (; i + TAGLMK_LANES <= n; i += TAGLMK_LANES)
		vsum0 = vpadalq_u32(vsum0, vld1q_u32(x + i));

	sum = taglmk_reduce_u64(vaddq_u64(vsum0, vsum1));
	for (; i < n; i++)
		sum += x[i];

	/*
	 * Steps start at one, so the body works on the overlapping pair
	 * (x + i, x + i - 1).  On a full window that leaves eight steps to the
	 * wide loop, four to the narrow one and three to the tail, which is
	 * cheaper than realigning the loads would be.
	 */
	for (i = 1; i + TAGLMK_LANES_WIDE <= n; i += TAGLMK_LANES_WIDE) {
		vdiff0 = taglmk_step_acc(vdiff0, x + i);
		vdiff1 = taglmk_step_acc(vdiff1, x + i + TAGLMK_LANES);
	}

	for (; i + TAGLMK_LANES <= n; i += TAGLMK_LANES)
		vdiff0 = taglmk_step_acc(vdiff0, x + i);

	absdiff = taglmk_reduce_u64(vaddq_u64(vdiff0, vdiff1));
	for (; i < n; i++)
		absdiff += x[i] > x[i - 1] ? x[i] - x[i - 1]
					   : x[i - 1] - x[i];

	*out_sum = sum;
	*out_absdiff = absdiff;
}

/**
 * taglmk_neon_window_stats - mean and mean absolute step of a sample window
 * @x: Samples, oldest first.
 * @n: How many.
 * @out_mean: Arithmetic mean of @x, or zero for an empty window.
 * @out_absdiff: Mean of |x[i] - x[i-1]| over the @n - 1 steps, or zero if
 *	there are no steps yet.
 */
void taglmk_neon_window_stats(const u32 *x, unsigned int n, u32 *out_mean,
			      u32 *out_absdiff)
{
	u64 sum;
	u64 absdiff;

	if (!n) {
		*out_mean = 0;
		*out_absdiff = 0;
		return;
	}

	kernel_neon_begin();
	__taglmk_window_sums(x, n, &sum, &absdiff);
	kernel_neon_end();

	*out_mean = (u32)div_u64(sum, n);
	*out_absdiff = n > 1 ? (u32)div_u64(absdiff, n - 1) : 0;
}

/*
 * The four sums, accumulated four elements at a time.  Four elements rather
 * than eight is deliberate here: each iteration already issues four multiply
 * longs - the low and high halves of x*x and of x*y - which is as many as the
 * two product accumulators can absorb before one of them has to wait, and the
 * two plain sums ride along in a pairwise accumulate that needs no add of its
 * own.
 */
static noinline void __taglmk_regress_sums(const u32 *x, const u32 *y,
					   unsigned int n, u64 *out_sx,
					   u64 *out_sy, u64 *out_sxx,
					   u64 *out_sxy)
{
	uint64x2_t vsx = vdupq_n_u64(0);
	uint64x2_t vsy = vdupq_n_u64(0);
	uint64x2_t vsxx = vdupq_n_u64(0);
	uint64x2_t vsxy = vdupq_n_u64(0);
	u64 sx;
	u64 sy;
	u64 sxx;
	u64 sxy;
	unsigned int i;

	for (i = 0; i + TAGLMK_LANES <= n; i += TAGLMK_LANES) {
		uint32x4_t vx = vld1q_u32(x + i);
		uint32x4_t vy = vld1q_u32(y + i);

		vsx = vpadalq_u32(vsx, vx);
		vsy = vpadalq_u32(vsy, vy);
		vsxx = taglmk_mla_widen(vsxx, vx, vx);
		vsxy = taglmk_mla_widen(vsxy, vx, vy);
	}

	sx = taglmk_reduce_u64(vsx);
	sy = taglmk_reduce_u64(vsy);
	sxx = taglmk_reduce_u64(vsxx);
	sxy = taglmk_reduce_u64(vsxy);

	for (; i < n; i++) {
		sx += x[i];
		sy += y[i];
		sxx += (u64)x[i] * x[i];
		sxy += (u64)x[i] * y[i];
	}

	*out_sx = sx;
	*out_sy = sy;
	*out_sxx = sxx;
	*out_sxy = sxy;
}

/**
 * taglmk_neon_regress - the four sums a least squares fit of y on x needs
 * @x: Independent samples.
 * @y: Dependent samples, same length.
 * @n: How many.
 * @out_sx: Sum of x.
 * @out_sy: Sum of y.
 * @out_sxx: Sum of x squared.
 * @out_sxy: Sum of x times y.
 *
 * Only the sums are produced.  Turning them into a slope needs a signed
 * division that has no business happening with softirqs off, and the balancer
 * folds them into a running total before it divides anything anyway.
 */
void taglmk_neon_regress(const u32 *x, const u32 *y, unsigned int n,
			 u64 *out_sx, u64 *out_sy, u64 *out_sxx, u64 *out_sxy)
{
	if (!n) {
		*out_sx = 0;
		*out_sy = 0;
		*out_sxx = 0;
		*out_sxy = 0;
		return;
	}

	kernel_neon_begin();
	__taglmk_regress_sums(x, y, n, out_sx, out_sy, out_sxx, out_sxy);
	kernel_neon_end();
}

/*
 * out[i] = (anon[i] * scale) >> 16, through a 64 bit product.  This is a map
 * with no carried dependency at all, so the wide step is pure issue width: two
 * vectors a time is four multiply longs and four narrowing shifts back to back.
 */
static noinline void __taglmk_share(const u32 *anon, u32 *out, unsigned int n,
				    u32 scale)
{
	unsigned int i;

	for (i = 0; i + TAGLMK_LANES_WIDE <= n; i += TAGLMK_LANES_WIDE) {
		uint32x4_t v0 = vld1q_u32(anon + i);
		uint32x4_t v1 = vld1q_u32(anon + i + TAGLMK_LANES);

		vst1q_u32(out + i, taglmk_scale_q(v0, scale));
		vst1q_u32(out + i + TAGLMK_LANES, taglmk_scale_q(v1, scale));
	}

	for (; i + TAGLMK_LANES <= n; i += TAGLMK_LANES)
		vst1q_u32(out + i, taglmk_scale_q(vld1q_u32(anon + i), scale));

	for (; i < n; i++)
		out[i] = (u32)(((u64)anon[i] * scale) >> TAGLMK_FP_SHIFT);
}

/**
 * taglmk_neon_share - split a budget across tasks in proportion to their size
 * @anon: Per task anonymous page counts.
 * @out: Per task page budgets, @n entries, every one written.
 * @n: How many tasks.
 * @scale: Q16.16 fraction of each task's pages to ask for.
 *
 * The caller has already worked @scale out from the total, which is why this is
 * a pure map and needs no reduction.  Both paths truncate the shifted product
 * to 32 bits, so an implausibly large @scale degrades identically either way.
 */
void taglmk_neon_share(const u32 *anon, u32 *out, unsigned int n, u32 scale)
{
	if (!n)
		return;

	kernel_neon_begin();
	__taglmk_share(anon, out, n, scale);
	kernel_neon_end();
}
