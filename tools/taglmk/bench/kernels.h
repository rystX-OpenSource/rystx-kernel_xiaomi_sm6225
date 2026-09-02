/* SPDX-License-Identifier: GPL-2.0 */
/*
 * taglmk_bench - the three kernels, in the three forms being compared.
 *
 * Each family appears as a plain C twin and, on arm64, as the NEON kernel
 * before and after the widening change.  The bodies are transcribed from
 * drivers/android/taglmk/, not reimplemented: the point of the exercise is to
 * time the code that runs on the device, so anything rewritten here would be
 * measuring something else.  Only the types differ - u32/u64 become
 * uint32_t/uint64_t - because that is all that stands between kernel and
 * userspace for integer arithmetic.
 *
 * The three families are chosen for what they isolate.
 *
 *   regress  Four sums, two of them products.  v0 and v1 do identical work in
 *            identical loop structure and differ only in how the widening
 *            multiply is issued, so this case and this case alone carries the
 *            multiply-long claim.
 *
 *   share    A pure map with no carried dependency, so it isolates issue width
 *            from latency: whatever v1 gains here it gains by doing more per
 *            step, not by breaking a dependency chain.
 *
 *   window2  Sum and mean absolute step, the two outputs v0 produced, computed
 *            by both.  v1's tree kernel also produces a third sum, so timing
 *            it against v0 would credit the vectorization with work v0 never
 *            did; window2 applies v1's reduction technique to v0's outputs so
 *            that the two are answering the same question.
 *
 *   window3  The tree kernel as it actually stands, all three sums, against
 *            its scalar twin.  Reported on its own and never against v0.
 *
 * Copyright (C) 2026 iDeadXS <datarafi43@gmail.com>
 */
#ifndef TAGLMK_BENCH_KERNELS_H
#define TAGLMK_BENCH_KERNELS_H

#include <stdint.h>

/* The driver's own constants, so the twins here cannot drift from it. */
#define BENCH_FP_SHIFT		16
#define BENCH_FP_ONE		(1u << BENCH_FP_SHIFT)
#define BENCH_Q42_SHIFT		2
#define BENCH_Q42_ONE		(1u << BENCH_Q42_SHIFT)
#define BENCH_Q42_MAX		0x3fu
#define BENCH_LANES		4
#define BENCH_LANES_WIDE	(2 * BENCH_LANES)

/* The window the predictor actually keeps. */
#define BENCH_WINDOW		16

/*
 * NEON on arm64 is architectural, so there is nothing to probe: if this is an
 * arm64 build the kernels are present.  On arm32 they are not, because
 * vmull_high_u32() and its relatives exist only in the 64 bit intrinsic set,
 * and rewriting them into the portable form would make the "after" kernel a
 * different kernel from the one on the device.  A build for any other
 * architecture gets the scalar twins alone, which is enough to check the file
 * format and the zram suite.
 */
#if defined(__aarch64__)
#define BENCH_HAVE_NEON		1
#else
#define BENCH_HAVE_NEON		0
#endif

/**
 * struct bench_window - the sums a sample window reduces to
 * @sum: Sum of the samples.
 * @absdiff: Sum of |x[i] - x[i-1]| over the n - 1 steps.
 * @weighted: Sum of i * x[i].  Left zero by the window2 kernels, which is what
 *	makes it visible that they never computed it.
 */
struct bench_window {
	uint64_t	sum;
	uint64_t	absdiff;
	uint64_t	weighted;
};

/**
 * struct bench_sums - the four sums a least squares fit needs
 */
struct bench_sums {
	uint64_t	sx;
	uint64_t	sy;
	uint64_t	sxx;
	uint64_t	sxy;
};

/* --------------------------------------------------------- kernels_scalar.c */

void bench_window2_scalar(const uint32_t *x, unsigned int n,
			  struct bench_window *w);
void bench_window3_scalar(const uint32_t *x, unsigned int n,
			  struct bench_window *w);
void bench_regress_scalar(const uint32_t *x, const uint32_t *y, unsigned int n,
			  struct bench_sums *s);
void bench_share_scalar(const uint32_t *anon, uint32_t *out, unsigned int n,
			uint32_t scale);

#if BENCH_HAVE_NEON

/* ------------------------------------------------------------- kernels_v0.c */

void bench_window2_v0(const uint32_t *x, unsigned int n,
		      struct bench_window *w);
void bench_regress_v0(const uint32_t *x, const uint32_t *y, unsigned int n,
		      struct bench_sums *s);
void bench_share_v0(const uint32_t *anon, uint32_t *out, unsigned int n,
		    uint32_t scale);

/* ------------------------------------------------------------- kernels_v1.c */

void bench_window2_v1(const uint32_t *x, unsigned int n,
		      struct bench_window *w);
void bench_window3_v1(const uint32_t *x, unsigned int n,
		      struct bench_window *w);
void bench_regress_v1(const uint32_t *x, const uint32_t *y, unsigned int n,
		      struct bench_sums *s);
void bench_share_v1(const uint32_t *anon, uint32_t *out, unsigned int n,
		    uint32_t scale);

#endif /* BENCH_HAVE_NEON */

#endif /* TAGLMK_BENCH_KERNELS_H */
