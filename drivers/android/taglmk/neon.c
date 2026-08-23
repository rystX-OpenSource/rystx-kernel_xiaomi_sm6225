// SPDX-License-Identifier: GPL-2.0
/*
 * TAGLMK - ARM64 NEON accelerated kernels.
 *
 * The predictor and the ZRAM balancer both boil a small array of per-pass or
 * per-task page counts down to a handful of sums.  The arrays are short, so
 * this is not about throughput; it is about how long a pass spends working
 * while the machine is already short of memory.  Four lanes at a time turns a
 * sixteen element reduction into four instructions.
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
 * Lanes per vector.  Every loop below is written as "while a whole vector still
 * fits", so the bound is the length itself and no rounding is needed; what is
 * left over is finished by a scalar tail.
 */
#define TAGLMK_LANES		4

/* The two lanes of a 64 bit accumulator, added up. */
static inline u64 taglmk_reduce_u64(uint64x2_t v)
{
	return vgetq_lane_u64(v, 0) + vgetq_lane_u64(v, 1);
}

/*
 * acc += a * b, widening every lane to 64 bits first.  Two u32 factors need 64
 * bits to hold their product, so this is the only form that cannot overflow
 * however large the page counts get.
 */
static inline uint64x2_t taglmk_mla_widen(uint64x2_t acc, uint32x4_t a,
					  uint32x4_t b)
{
	acc = vaddq_u64(acc, vmull_u32(vget_low_u32(a), vget_low_u32(b)));
	acc = vaddq_u64(acc, vmull_u32(vget_high_u32(a), vget_high_u32(b)));

	return acc;
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
 * The unsigned absolute difference is a single instruction per vector, which is
 * the whole reason this kernel exists - the scalar twin needs a branch per
 * element to get the same answer.
 */
static noinline void __taglmk_window_sums(const u32 *x, unsigned int n,
					  u64 *out_sum, u64 *out_absdiff)
{
	uint64x2_t vsum = vdupq_n_u64(0);
	uint64x2_t vdiff = vdupq_n_u64(0);
	u64 sum;
	u64 absdiff;
	unsigned int i;

	for (i = 0; i + TAGLMK_LANES <= n; i += TAGLMK_LANES)
		vsum = vaddq_u64(vsum, vpaddlq_u32(vld1q_u32(x + i)));

	sum = taglmk_reduce_u64(vsum);
	for (; i < n; i++)
		sum += x[i];

	/*
	 * Steps start at one, so the body works on the overlapping pair
	 * (x + i, x + i - 1); with n a multiple of four there is always a
	 * three element tail, which is cheaper than realigning would be.
	 */
	for (i = 1; i + TAGLMK_LANES <= n; i += TAGLMK_LANES)
		vdiff = vaddq_u64(vdiff,
				  vpaddlq_u32(vabdq_u32(vld1q_u32(x + i),
							vld1q_u32(x + i - 1))));

	absdiff = taglmk_reduce_u64(vdiff);
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

/* The four sums, accumulated four elements at a time. */
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

		vsx = vaddq_u64(vsx, vpaddlq_u32(vx));
		vsy = vaddq_u64(vsy, vpaddlq_u32(vy));
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

/* out[i] = (anon[i] * scale) >> 16, through a 64 bit product. */
static noinline void __taglmk_share(const u32 *anon, u32 *out, unsigned int n,
				    u32 scale)
{
	unsigned int i;

	for (i = 0; i + TAGLMK_LANES <= n; i += TAGLMK_LANES) {
		uint32x4_t v = vld1q_u32(anon + i);
		uint64x2_t lo = vmull_n_u32(vget_low_u32(v), scale);
		uint64x2_t hi = vmull_n_u32(vget_high_u32(v), scale);

		lo = vshrq_n_u64(lo, TAGLMK_FP_SHIFT);
		hi = vshrq_n_u64(hi, TAGLMK_FP_SHIFT);

		vst1q_u32(out + i, vcombine_u32(vmovn_u64(lo), vmovn_u64(hi)));
	}

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
