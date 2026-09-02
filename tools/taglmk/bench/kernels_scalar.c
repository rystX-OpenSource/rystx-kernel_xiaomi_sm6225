// SPDX-License-Identifier: GPL-2.0
/*
 * taglmk_bench - the plain C twins.
 *
 * These are the driver's fallback paths, transcribed.  They serve two purposes
 * and it is worth being clear about which is which.
 *
 * They are the reference.  Every accelerated kernel is checked against the twin
 * for a bit exact match before any of its timings are reported, which is the
 * property drivers/android/taglmk/neon.c calls Exactness and the only reason a
 * device may run either path.  A variant that disagrees by one is reported as
 * having failed and is never scored.
 *
 * They are context, not the baseline.  The changelog claim is about one NEON
 * kernel against its predecessor, so the number that backs it is v0 against v1.
 * The scalar row says what a build without the accelerator pays, and it is
 * compiled exactly as the kernel compiles predict.c - plain -O2, no pragmas -
 * so if the compiler chooses to vectorize a loop here it is choosing the same
 * thing in the kernel, and the row stays truthful.
 *
 * Copyright (C) 2026 iDeadXS <datarafi43@gmail.com>
 */
#include "kernels.h"

/*
 * taglmk_window_sums_scalar(), with the weighted sum left out.  Everything
 * about the two loops is otherwise identical, so this is what v0's outputs cost
 * in plain C.
 */
void bench_window2_scalar(const uint32_t *x, unsigned int n,
			  struct bench_window *w)
{
	uint64_t sum = 0;
	uint64_t absdiff = 0;
	unsigned int i;

	for (i = 0; i < n; i++)
		sum += x[i];

	for (i = 1; i < n; i++)
		absdiff += x[i] > x[i - 1] ? x[i] - x[i - 1]
					   : x[i - 1] - x[i];

	w->sum = sum;
	w->absdiff = absdiff;
	w->weighted = 0;
}

/* taglmk_window_sums_scalar() as it stands in the tree, all three sums. */
void bench_window3_scalar(const uint32_t *x, unsigned int n,
			  struct bench_window *w)
{
	uint64_t sum = 0;
	uint64_t weighted = 0;
	uint64_t absdiff = 0;
	unsigned int i;

	for (i = 0; i < n; i++) {
		sum += x[i];
		weighted += (uint64_t)i * x[i];
	}

	for (i = 1; i < n; i++)
		absdiff += x[i] > x[i - 1] ? x[i] - x[i - 1]
					   : x[i - 1] - x[i];

	w->sum = sum;
	w->absdiff = absdiff;
	w->weighted = weighted;
}

/* The tail of __taglmk_regress_sums(), which is the whole of the fallback. */
void bench_regress_scalar(const uint32_t *x, const uint32_t *y, unsigned int n,
			  struct bench_sums *s)
{
	uint64_t sx = 0;
	uint64_t sy = 0;
	uint64_t sxx = 0;
	uint64_t sxy = 0;
	unsigned int i;

	for (i = 0; i < n; i++) {
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

/* The tail of __taglmk_share(): a 64 bit product, truncated after the shift. */
void bench_share_scalar(const uint32_t *anon, uint32_t *out, unsigned int n,
			uint32_t scale)
{
	unsigned int i;

	for (i = 0; i < n; i++)
		out[i] = (uint32_t)(((uint64_t)anon[i] * scale) >>
				    BENCH_FP_SHIFT);
}
