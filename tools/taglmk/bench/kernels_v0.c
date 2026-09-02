// SPDX-License-Identifier: GPL-2.0
/*
 * taglmk_bench - the NEON kernels as they stood before the widening change.
 *
 * Transcribed from drivers/android/taglmk/neon.c at 61b438c9d7d0, the commit
 * immediately before "switch the advisory format to Q4.2 and widen the
 * kernels".  Four lanes a step, one accumulator per sum, and a widening
 * multiply that pulls the high half of each operand out with vget_high_u32()
 * and folds each product into the accumulator separately.
 *
 * This lives in its own translation unit and must stay in one.  Compiled
 * alongside the v1 bodies the compiler would be free to notice that the two
 * differ only in an inline helper, unify them, and report that the change made
 * no difference - which is exactly the question being asked.  build.sh compiles
 * each file separately and does not link with LTO for the same reason.
 *
 * Copyright (C) 2026 iDeadXS <datarafi43@gmail.com>
 */
#include "kernels.h"

/*
 * Guarded whole, rather than left out of the build, so that build.sh needs no
 * per-architecture logic and a build for anything but arm64 quietly produces a
 * scalar only binary instead of a compile error.
 */
#if BENCH_HAVE_NEON

#include <arm_neon.h>

/* The two lanes of a 64 bit accumulator, added up. */
static inline uint64_t bench_reduce_u64(uint64x2_t v)
{
	return vgetq_lane_u64(v, 0) + vgetq_lane_u64(v, 1);
}

/*
 * acc += a * b, widening every lane to 64 bits first.  The high half of each
 * operand is extracted into its own register before its product can be formed,
 * and each of the two products is added to @acc in turn, so the carried
 * dependency advances twice per call.
 */
static inline uint64x2_t bench_mla_widen_v0(uint64x2_t acc, uint32x4_t a,
					    uint32x4_t b)
{
	acc = vaddq_u64(acc, vmull_u32(vget_low_u32(a), vget_low_u32(b)));
	acc = vaddq_u64(acc, vmull_u32(vget_high_u32(a), vget_high_u32(b)));

	return acc;
}

/*
 * __taglmk_window_sums() as it was: sum, and sum of |x[i] - x[i-1]|.  A single
 * accumulator per sum, four lanes a step, and a separate widening pairwise add
 * followed by an add into the accumulator.
 */
void bench_window2_v0(const uint32_t *x, unsigned int n,
		      struct bench_window *w)
{
	uint64x2_t vsum = vdupq_n_u64(0);
	uint64x2_t vdiff = vdupq_n_u64(0);
	uint64_t sum;
	uint64_t absdiff;
	unsigned int i;

	for (i = 0; i + BENCH_LANES <= n; i += BENCH_LANES)
		vsum = vaddq_u64(vsum, vpaddlq_u32(vld1q_u32(x + i)));

	sum = bench_reduce_u64(vsum);
	for (; i < n; i++)
		sum += x[i];

	for (i = 1; i + BENCH_LANES <= n; i += BENCH_LANES)
		vdiff = vaddq_u64(vdiff,
				  vpaddlq_u32(vabdq_u32(vld1q_u32(x + i),
							vld1q_u32(x + i - 1))));

	absdiff = bench_reduce_u64(vdiff);
	for (; i < n; i++)
		absdiff += x[i] > x[i - 1] ? x[i] - x[i - 1]
					   : x[i - 1] - x[i];

	w->sum = sum;
	w->absdiff = absdiff;
	w->weighted = 0;
}

/* __taglmk_regress_sums() as it was.  The loop is unchanged in v1; only the
 * body of the widening multiply above is.
 */
void bench_regress_v0(const uint32_t *x, const uint32_t *y, unsigned int n,
		      struct bench_sums *s)
{
	uint64x2_t vsx = vdupq_n_u64(0);
	uint64x2_t vsy = vdupq_n_u64(0);
	uint64x2_t vsxx = vdupq_n_u64(0);
	uint64x2_t vsxy = vdupq_n_u64(0);
	uint64_t sx;
	uint64_t sy;
	uint64_t sxx;
	uint64_t sxy;
	unsigned int i;

	for (i = 0; i + BENCH_LANES <= n; i += BENCH_LANES) {
		uint32x4_t vx = vld1q_u32(x + i);
		uint32x4_t vy = vld1q_u32(y + i);

		vsx = vaddq_u64(vsx, vpaddlq_u32(vx));
		vsy = vaddq_u64(vsy, vpaddlq_u32(vy));
		vsxx = bench_mla_widen_v0(vsxx, vx, vx);
		vsxy = bench_mla_widen_v0(vsxy, vx, vy);
	}

	sx = bench_reduce_u64(vsx);
	sy = bench_reduce_u64(vsy);
	sxx = bench_reduce_u64(vsxx);
	sxy = bench_reduce_u64(vsxy);

	for (; i < n; i++) {
		sx += x[i];
		sy += y[i];
		sxx += (uint64_t)x[i] * x[i];
		sxy += (uint64_t)x[i] * y[i];
	}

	s->sx = sx;
	s->sy = sy;
	s->sxx = sxx;
	s->sxy = sxy;
}

/*
 * __taglmk_share() as it was: four lanes a step, the high half extracted, and
 * the shift and the narrowing done as two instructions.
 */
void bench_share_v0(const uint32_t *anon, uint32_t *out, unsigned int n,
		    uint32_t scale)
{
	unsigned int i;

	for (i = 0; i + BENCH_LANES <= n; i += BENCH_LANES) {
		uint32x4_t v = vld1q_u32(anon + i);
		uint64x2_t lo = vmull_n_u32(vget_low_u32(v), scale);
		uint64x2_t hi = vmull_n_u32(vget_high_u32(v), scale);

		lo = vshrq_n_u64(lo, BENCH_FP_SHIFT);
		hi = vshrq_n_u64(hi, BENCH_FP_SHIFT);

		vst1q_u32(out + i, vcombine_u32(vmovn_u64(lo), vmovn_u64(hi)));
	}

	for (; i < n; i++)
		out[i] = (uint32_t)(((uint64_t)anon[i] * scale) >>
				    BENCH_FP_SHIFT);
}

#endif /* BENCH_HAVE_NEON */
