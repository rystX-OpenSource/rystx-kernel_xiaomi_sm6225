// SPDX-License-Identifier: GPL-2.0
/*
 * taglmk_bench - the NEON kernels as they stand now.
 *
 * Transcribed from drivers/android/taglmk/neon.c at "switch the advisory format
 * to Q4.2 and widen the kernels".  Two changes from v0, and the cases in
 * cpu_suite.c are arranged so that each can be seen on its own.
 *
 * The widening multiply issues both halves in place.  UMULL takes the low half
 * of each operand and UMULL2 the high half of each, so neither vector has to be
 * pulled apart first, and the two products are folded into each other before
 * they reach the accumulator - one step of the carried dependency per call
 * instead of two.  regress is the case that isolates this: its loop is
 * character for character v0's.
 *
 * The reductions run two accumulators over eight lanes a step.  A sixteen
 * element window is only four accumulate instructions long at four lanes, and
 * each waits on the one before it, so the cost is latency rather than width;
 * two independent chains halve the depth and the fold at the end costs one add.
 * vpadalq_u32() then widens and accumulates in a single instruction where v0
 * needed vpaddlq_u32() and an add.
 *
 * Separate translation unit, for the reason kernels_v0.c gives.
 *
 * Copyright (C) 2026 iDeadXS <datarafi43@gmail.com>
 */
#include "kernels.h"

#if BENCH_HAVE_NEON

#include <arm_neon.h>

/* The two lanes of a 64 bit accumulator, added up. */
static inline uint64_t bench_reduce_u64(uint64x2_t v)
{
	return vgetq_lane_u64(v, 0) + vgetq_lane_u64(v, 1);
}

/*
 * acc += a * b, widening every lane to 64 bits first, both halves issued in
 * place.  Unsigned 64 bit addition is associative modulo 2^64, so folding the
 * two products together before the accumulator sees them regroups the sum
 * without changing it.
 */
static inline uint64x2_t bench_mla_widen_v1(uint64x2_t acc, uint32x4_t a,
					    uint32x4_t b)
{
	uint64x2_t lo = vmull_u32(vget_low_u32(a), vget_low_u32(b));
	uint64x2_t hi = vmull_high_u32(a, b);

	return vaddq_u64(acc, vaddq_u64(lo, hi));
}

/*
 * (v * scale) >> BENCH_FP_SHIFT across four lanes.  SHRN shifts and narrows in
 * one instruction, and the truncation it performs is the same one the scalar
 * twin's cast performs.
 */
static inline uint32x4_t bench_scale_q(uint32x4_t v, uint32_t scale)
{
	uint64x2_t lo = vmull_n_u32(vget_low_u32(v), scale);
	uint64x2_t hi = vmull_high_n_u32(v, scale);

	return vcombine_u32(vshrn_n_u64(lo, BENCH_FP_SHIFT),
			    vshrn_n_u64(hi, BENCH_FP_SHIFT));
}

/* acc += |x[i] - x[i-1]| across four lanes, widened and accumulated in one. */
static inline uint64x2_t bench_step_acc(uint64x2_t acc, const uint32_t *x)
{
	return vpadalq_u32(acc, vabdq_u32(vld1q_u32(x), vld1q_u32(x - 1)));
}

/* The sample index in lane order, for the weighted sum. */
static const uint32_t bench_lane_index[BENCH_LANES_WIDE] = {
	0, 1, 2, 3, 4, 5, 6, 7,
};

/*
 * v1's reduction technique over v0's two outputs.
 *
 * This is the one kernel here that has no exact counterpart in the tree, and it
 * exists so that the comparison is fair rather than flattering.  The tree's v1
 * window kernel also produces the weighted sum, which v0 never computed; timing
 * the two against each other would charge v1 for work its predecessor did not
 * do and then report the difference as a regression, or - if the extra sum were
 * quietly ignored - credit the vectorization with having removed it.  Taking
 * v1's dual accumulator, eight lane, accumulate-in-place reduction and pointing
 * it at exactly v0's two outputs answers the question actually being asked:
 * what did the wider reduction do to the work that was already there?
 *
 * The tree kernel itself is bench_window3_v1() below, timed against its own
 * scalar twin and never against v0.
 */
void bench_window2_v1(const uint32_t *x, unsigned int n,
		      struct bench_window *w)
{
	uint64x2_t vsum0 = vdupq_n_u64(0);
	uint64x2_t vsum1 = vdupq_n_u64(0);
	uint64x2_t vdiff0 = vdupq_n_u64(0);
	uint64x2_t vdiff1 = vdupq_n_u64(0);
	uint64_t sum;
	uint64_t absdiff;
	unsigned int i;

	for (i = 0; i + BENCH_LANES_WIDE <= n; i += BENCH_LANES_WIDE) {
		vsum0 = vpadalq_u32(vsum0, vld1q_u32(x + i));
		vsum1 = vpadalq_u32(vsum1, vld1q_u32(x + i + BENCH_LANES));
	}

	for (; i + BENCH_LANES <= n; i += BENCH_LANES)
		vsum0 = vpadalq_u32(vsum0, vld1q_u32(x + i));

	sum = bench_reduce_u64(vaddq_u64(vsum0, vsum1));
	for (; i < n; i++)
		sum += x[i];

	for (i = 1; i + BENCH_LANES_WIDE <= n; i += BENCH_LANES_WIDE) {
		vdiff0 = bench_step_acc(vdiff0, x + i);
		vdiff1 = bench_step_acc(vdiff1, x + i + BENCH_LANES);
	}

	for (; i + BENCH_LANES <= n; i += BENCH_LANES)
		vdiff0 = bench_step_acc(vdiff0, x + i);

	absdiff = bench_reduce_u64(vaddq_u64(vdiff0, vdiff1));
	for (; i < n; i++)
		absdiff += x[i] > x[i - 1] ? x[i] - x[i - 1]
					   : x[i - 1] - x[i];

	w->sum = sum;
	w->absdiff = absdiff;
	w->weighted = 0;
}

/* __taglmk_window_sums() exactly as it stands: all three sums. */
void bench_window3_v1(const uint32_t *x, unsigned int n,
		      struct bench_window *w)
{
	const uint32x4_t vstep_wide = vdupq_n_u32(BENCH_LANES_WIDE);
	const uint32x4_t vstep_narrow = vdupq_n_u32(BENCH_LANES);
	uint32x4_t vidx0 = vld1q_u32(bench_lane_index);
	uint32x4_t vidx1 = vld1q_u32(bench_lane_index + BENCH_LANES);
	uint64x2_t vsum0 = vdupq_n_u64(0);
	uint64x2_t vsum1 = vdupq_n_u64(0);
	uint64x2_t vwgt0 = vdupq_n_u64(0);
	uint64x2_t vwgt1 = vdupq_n_u64(0);
	uint64x2_t vdiff0 = vdupq_n_u64(0);
	uint64x2_t vdiff1 = vdupq_n_u64(0);
	uint64_t sum;
	uint64_t weighted;
	uint64_t absdiff;
	unsigned int i;

	for (i = 0; i + BENCH_LANES_WIDE <= n; i += BENCH_LANES_WIDE) {
		uint32x4_t v0 = vld1q_u32(x + i);
		uint32x4_t v1 = vld1q_u32(x + i + BENCH_LANES);

		vsum0 = vpadalq_u32(vsum0, v0);
		vsum1 = vpadalq_u32(vsum1, v1);
		vwgt0 = bench_mla_widen_v1(vwgt0, vidx0, v0);
		vwgt1 = bench_mla_widen_v1(vwgt1, vidx1, v1);
		vidx0 = vaddq_u32(vidx0, vstep_wide);
		vidx1 = vaddq_u32(vidx1, vstep_wide);
	}

	for (; i + BENCH_LANES <= n; i += BENCH_LANES) {
		uint32x4_t v = vld1q_u32(x + i);

		vsum0 = vpadalq_u32(vsum0, v);
		vwgt0 = bench_mla_widen_v1(vwgt0, vidx0, v);
		vidx0 = vaddq_u32(vidx0, vstep_narrow);
	}

	sum = bench_reduce_u64(vaddq_u64(vsum0, vsum1));
	weighted = bench_reduce_u64(vaddq_u64(vwgt0, vwgt1));
	for (; i < n; i++) {
		sum += x[i];
		weighted += (uint64_t)i * x[i];
	}

	for (i = 1; i + BENCH_LANES_WIDE <= n; i += BENCH_LANES_WIDE) {
		vdiff0 = bench_step_acc(vdiff0, x + i);
		vdiff1 = bench_step_acc(vdiff1, x + i + BENCH_LANES);
	}

	for (; i + BENCH_LANES <= n; i += BENCH_LANES)
		vdiff0 = bench_step_acc(vdiff0, x + i);

	absdiff = bench_reduce_u64(vaddq_u64(vdiff0, vdiff1));
	for (; i < n; i++)
		absdiff += x[i] > x[i - 1] ? x[i] - x[i - 1]
					   : x[i - 1] - x[i];

	w->sum = sum;
	w->absdiff = absdiff;
	w->weighted = weighted;
}

/*
 * __taglmk_regress_sums() as it stands.  Set this against bench_regress_v0()
 * and the only difference between the two is the widening multiply: the loop,
 * the lane count, the accumulator count and the tail are all the same.
 */
void bench_regress_v1(const uint32_t *x, const uint32_t *y, unsigned int n,
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

		vsx = vpadalq_u32(vsx, vx);
		vsy = vpadalq_u32(vsy, vy);
		vsxx = bench_mla_widen_v1(vsxx, vx, vx);
		vsxy = bench_mla_widen_v1(vsxy, vx, vy);
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
 * __taglmk_share() as it stands: eight lanes a step, both halves multiplied in
 * place, and one instruction to shift and narrow.
 */
void bench_share_v1(const uint32_t *anon, uint32_t *out, unsigned int n,
		    uint32_t scale)
{
	unsigned int i;

	for (i = 0; i + BENCH_LANES_WIDE <= n; i += BENCH_LANES_WIDE) {
		uint32x4_t v0 = vld1q_u32(anon + i);
		uint32x4_t v1 = vld1q_u32(anon + i + BENCH_LANES);

		vst1q_u32(out + i, bench_scale_q(v0, scale));
		vst1q_u32(out + i + BENCH_LANES, bench_scale_q(v1, scale));
	}

	for (; i + BENCH_LANES <= n; i += BENCH_LANES)
		vst1q_u32(out + i, bench_scale_q(vld1q_u32(anon + i), scale));

	for (; i < n; i++)
		out[i] = (uint32_t)(((uint64_t)anon[i] * scale) >>
				    BENCH_FP_SHIFT);
}

#endif /* BENCH_HAVE_NEON */
