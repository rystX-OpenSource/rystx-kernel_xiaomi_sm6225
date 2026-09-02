// SPDX-License-Identifier: GPL-2.0
/*
 * taglmk_bench - the CPU suite.
 *
 * Four cases, and the reason there are four rather than three is that a fair
 * comparison and a complete one are not the same thing.  kernels.h sets out
 * which case isolates what; this file's job is to make sure nothing is timed
 * before it has been proved correct, and that every variant of a case is fed
 * byte for byte the same input.
 *
 * The inputs are generated here, from the run's seed, and every array is
 * generated once and shared by every variant of every case.  They are shaped
 * like what the driver actually sees - a file cache that drifts with occasional
 * steps, a request-and-outcome pair per pass, a spread of task sizes - because
 * the scalar twin's absolute difference loop branches per element and would
 * otherwise be measured on data no phone produces.  The NEON kernels are
 * branch free and do not care.
 *
 * Verification comes first and is not optional.  Each variant is run once, its
 * output compared field by field against the scalar twin's, and a variant that
 * disagrees anywhere publishes verify = 0 and is skipped by the comparison for
 * good.  That is the property neon.c calls Exactness, and checking it here is
 * what stops a wrong kernel from being reported as a fast one.
 *
 * Copyright (C) 2026 iDeadXS <datarafi43@gmail.com>
 */
#define _GNU_SOURCE

#include <stdio.h>
#include <string.h>

#include "bench.h"
#include "kernels.h"

#define BENCH_CPU_SUITE		"cpu"

/* The sizes the driver uses, and the only sizes measured. */
#define BENCH_N_WINDOW		BENCH_WINDOW	/* predict.c's sample window */
#define BENCH_N_REGRESS		16		/* zram.c's window */
#define BENCH_N_SHARE		128		/* TAGLMK_MAX_VICTIMS */

/* A quarter, in the Q16.16 the share kernel takes. */
#define BENCH_SHARE_SCALE	(BENCH_FP_ONE / 4)

/*
 * splitmix64, for inputs that are identical on every device and in every order.
 * A generator is wanted here rather than a fixed table because the table would
 * have to be long enough for the share case and would then be the thing under
 * test; a seed is recorded in the saved file instead, and a comparison refuses
 * two runs that disagree about it.
 */
static uint64_t bench_mix(uint64_t *state)
{
	uint64_t z = (*state += 0x9e3779b97f4a7c15ull);

	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;

	return z ^ (z >> 31);
}

/* Uniform on [0, span). */
static uint32_t bench_pick(uint64_t *state, uint32_t span)
{
	return span ? (uint32_t)(bench_mix(state) % span) : 0;
}

/* Sixteen aligned, so every variant loads from the same alignment. */
static uint32_t bench_win[BENCH_N_WINDOW] __attribute__((aligned(16)));
static uint32_t bench_asked[BENCH_N_REGRESS] __attribute__((aligned(16)));
static uint32_t bench_got[BENCH_N_REGRESS] __attribute__((aligned(16)));
static uint32_t bench_anon[BENCH_N_SHARE] __attribute__((aligned(16)));
static uint32_t bench_out[BENCH_N_SHARE] __attribute__((aligned(16)));

/*
 * A file cache load, as the predictor sees it: a couple of hundred thousand
 * pages, drifting a little each pass, with an occasional step of a few per cent
 * where something was launched or killed.  The absolute values do not matter to
 * any kernel here, but the number of sign changes does matter to the scalar
 * twin, so it is worth them being plausible.
 */
static void bench_gen_window(uint64_t *state)
{
	uint32_t load = 180000 + bench_pick(state, 40000);

	for (unsigned int i = 0; i < BENCH_N_WINDOW; i++) {
		uint32_t drift = bench_pick(state, 4000);

		if (bench_pick(state, 8) == 0)
			drift += bench_pick(state, 20000);

		load = bench_pick(state, 2) ? load + drift
					    : (load > drift ? load - drift : 0);
		bench_win[i] = load;
	}
}

/*
 * Pages asked for, and pages the reclaimer came back with, one pair per pass.
 * The fit the balancer runs is of the second on the first, so the two have to
 * be correlated for the products to be of a realistic magnitude.
 */
static void bench_gen_regress(uint64_t *state)
{
	for (unsigned int i = 0; i < BENCH_N_REGRESS; i++) {
		uint32_t asked = 256 + bench_pick(state, 3840);
		uint32_t got = asked / 2 + bench_pick(state, asked / 2 + 1);

		bench_asked[i] = asked;
		bench_got[i] = got;
	}
}

/*
 * Task sizes, as a scan finds them: a long tail of small background tasks and a
 * handful of large ones.  The share kernel is a pure map, so the distribution
 * changes nothing it does, but it is what the numbers are quoted against.
 */
static void bench_gen_share(uint64_t *state)
{
	for (unsigned int i = 0; i < BENCH_N_SHARE; i++) {
		uint32_t pages = 2000 + bench_pick(state, 30000);

		if (bench_pick(state, 12) == 0)
			pages += bench_pick(state, 200000);

		bench_anon[i] = pages;
	}

	memset(bench_out, 0, sizeof(bench_out));
}

/* ------------------------------------------------------------ the harnesses */

typedef void (*bench_window_fn)(const uint32_t *x, unsigned int n,
				struct bench_window *w);
typedef void (*bench_regress_fn)(const uint32_t *x, const uint32_t *y,
				 unsigned int n, struct bench_sums *s);
typedef void (*bench_share_fn)(const uint32_t *anon, uint32_t *out,
			       unsigned int n, uint32_t scale);

struct bench_window_ctx {
	bench_window_fn		fn;
	const uint32_t		*x;
	unsigned int		n;
	struct bench_window	w;
};

struct bench_regress_ctx {
	bench_regress_fn	fn;
	unsigned int		n;
	struct bench_sums	s;
};

struct bench_share_ctx {
	bench_share_fn		fn;
	unsigned int		n;
	uint32_t		scale;
};

/*
 * Each trampoline folds the kernel's output into a value it returns, and
 * bench_measure() sinks that value after the clock has been read.  Without it
 * the compiler would be entitled to notice that nothing reads the output struct
 * and delete the whole loop, which is the classic way a microbenchmark comes
 * back reporting a hundredfold improvement.
 */
static uint64_t bench_run_window(void *p, uint64_t reps)
{
	struct bench_window_ctx *c = p;
	uint64_t acc = 0;

	for (uint64_t i = 0; i < reps; i++) {
		c->fn(c->x, c->n, &c->w);
		acc ^= c->w.sum ^ c->w.absdiff ^ c->w.weighted;
	}

	return acc;
}

static uint64_t bench_run_regress(void *p, uint64_t reps)
{
	struct bench_regress_ctx *c = p;
	uint64_t acc = 0;

	for (uint64_t i = 0; i < reps; i++) {
		c->fn(bench_asked, bench_got, c->n, &c->s);
		acc ^= c->s.sx ^ c->s.sy ^ c->s.sxx ^ c->s.sxy;
	}

	return acc;
}

static uint64_t bench_run_share(void *p, uint64_t reps)
{
	struct bench_share_ctx *c = p;
	uint64_t acc = 0;

	for (uint64_t i = 0; i < reps; i++) {
		c->fn(bench_anon, bench_out, c->n, c->scale);
		acc ^= bench_out[0] ^ bench_out[c->n - 1];
	}

	return acc;
}

/*
 * Record whether a variant produced the right answer, and say so out loud when
 * it did not.  A zero here is a correctness failure in a kernel that a phone
 * would otherwise be running, so it is worth more than a quiet row in a table.
 */
static void bench_verify(struct bench_report *rep, const char *tcase,
			 const char *variant, bool ok)
{
	bench_add(rep, BENCH_CPU_SUITE, tcase, variant, "verify", "bool",
		  BENCH_HIGHER_BETTER, ok ? 1.0 : 0.0);

	if (!ok)
		bench_err("%s/%s disagrees with the scalar twin; not timed",
			  tcase, variant);
}

/* ---------------------------------------------------------------- the cases */

struct bench_window_variant {
	const char	*name;
	bench_window_fn	fn;
	bool		weighted;	/* does it compute the third sum? */
};

static int bench_window_case(struct bench_report *rep,
			     const struct bench_opts *o, const char *tcase,
			     const struct bench_window_variant *var,
			     unsigned int nr)
{
	struct bench_window ref;
	unsigned int failed = 0;

	/* Variant zero is the scalar twin, and is therefore the reference. */
	var[0].fn(bench_win, BENCH_N_WINDOW, &ref);

	for (unsigned int i = 0; i < nr; i++) {
		struct bench_window_ctx ctx;
		struct bench_stats st;
		struct bench_window w;
		bool ok;

		memset(&w, 0, sizeof(w));
		var[i].fn(bench_win, BENCH_N_WINDOW, &w);

		ok = w.sum == ref.sum && w.absdiff == ref.absdiff;
		if (var[i].weighted)
			ok = ok && w.weighted == ref.weighted;
		else
			ok = ok && w.weighted == 0;

		bench_verify(rep, tcase, var[i].name, ok);
		if (!ok) {
			failed++;
			continue;
		}

		ctx.fn = var[i].fn;
		ctx.x = bench_win;
		ctx.n = BENCH_N_WINDOW;
		memset(&ctx.w, 0, sizeof(ctx.w));

		if (bench_measure(bench_run_window, &ctx, &o->timing, &st))
			continue;

		bench_publish(rep, BENCH_CPU_SUITE, tcase, var[i].name, &st);
	}

	return failed ? -1 : 0;
}

struct bench_regress_variant {
	const char		*name;
	bench_regress_fn	fn;
};

static int bench_regress_case(struct bench_report *rep,
			      const struct bench_opts *o,
			      const struct bench_regress_variant *var,
			      unsigned int nr)
{
	struct bench_sums ref;
	unsigned int failed = 0;

	var[0].fn(bench_asked, bench_got, BENCH_N_REGRESS, &ref);

	for (unsigned int i = 0; i < nr; i++) {
		struct bench_regress_ctx ctx;
		struct bench_stats st;
		struct bench_sums s;
		bool ok;

		memset(&s, 0, sizeof(s));
		var[i].fn(bench_asked, bench_got, BENCH_N_REGRESS, &s);

		ok = s.sx == ref.sx && s.sy == ref.sy && s.sxx == ref.sxx &&
		     s.sxy == ref.sxy;

		bench_verify(rep, "regress", var[i].name, ok);
		if (!ok) {
			failed++;
			continue;
		}

		ctx.fn = var[i].fn;
		ctx.n = BENCH_N_REGRESS;
		memset(&ctx.s, 0, sizeof(ctx.s));

		if (bench_measure(bench_run_regress, &ctx, &o->timing, &st))
			continue;

		bench_publish(rep, BENCH_CPU_SUITE, "regress", var[i].name,
			      &st);
	}

	return failed ? -1 : 0;
}

struct bench_share_variant {
	const char	*name;
	bench_share_fn	fn;
};

static int bench_share_case(struct bench_report *rep,
			    const struct bench_opts *o,
			    const struct bench_share_variant *var,
			    unsigned int nr)
{
	static uint32_t ref[BENCH_N_SHARE];
	unsigned int failed = 0;

	var[0].fn(bench_anon, ref, BENCH_N_SHARE, BENCH_SHARE_SCALE);

	for (unsigned int i = 0; i < nr; i++) {
		struct bench_share_ctx ctx;
		struct bench_stats st;
		bool ok;

		memset(bench_out, 0, sizeof(bench_out));
		var[i].fn(bench_anon, bench_out, BENCH_N_SHARE,
			  BENCH_SHARE_SCALE);

		ok = !memcmp(bench_out, ref, sizeof(ref));

		bench_verify(rep, "share", var[i].name, ok);
		if (!ok) {
			failed++;
			continue;
		}

		ctx.fn = var[i].fn;
		ctx.n = BENCH_N_SHARE;
		ctx.scale = BENCH_SHARE_SCALE;

		if (bench_measure(bench_run_share, &ctx, &o->timing, &st))
			continue;

		bench_publish(rep, BENCH_CPU_SUITE, "share", var[i].name, &st);
	}

	return failed ? -1 : 0;
}

/*
 * The Q4.2 case, which deliberately reports no timing at all.
 *
 * Narrowing the advisory format does not change how the factor is computed - it
 * is the same two multiplications either way - so there is no cycle count to
 * quote for it and inventing one would be dishonest.  What the format buys is a
 * bound: the widest product the predictor can form from a margin and a gain
 * fits sixteen bits, which is a lane rather than a widening multiply.  So that
 * bound is what is checked, exhaustively over all 64 x 64 pairs the format can
 * represent, and the Q4.4 counterexample is recorded beside it.
 *
 * Note what is *not* claimed: Q4.2 is not the widest format that fits.  Q4.3
 * would too, at 135 * 127 = 17145.  Q4.2 was chosen because a quarter is as
 * fine as an advisory term read by eye needs to be, not because anything wider
 * would overflow.
 */
static void bench_q42_case(struct bench_report *rep)
{
	unsigned int worst = 0;
	bool ok = true;

	for (unsigned int burst = 0; burst <= BENCH_Q42_MAX; burst++) {
		unsigned int margin = BENCH_Q42_ONE + burst;

		for (unsigned int gain = 0; gain <= BENCH_Q42_MAX; gain++) {
			unsigned int prod = margin * gain;

			if (prod > worst)
				worst = prod;
			if (prod > 0xffffu)
				ok = false;
		}
	}

	bench_verify(rep, "q42", "width", ok);
	bench_add(rep, BENCH_CPU_SUITE, "q42", "width", "product.max", "u16",
		  BENCH_LOWER_BETTER, (double)worst);

	/*
	 * Q4.4: sixteen is one, so the product reaches (16 + 255) * 255.  A
	 * case of its own rather than a second variant of q42, because these
	 * two numbers are a bound and a counterexample; scoring one against
	 * the other as a percentage would read like a measurement.
	 */
	bench_add(rep, BENCH_CPU_SUITE, "q44", "width", "product.max", "u16",
		  BENCH_LOWER_BETTER, (double)((16 + 255) * 255));

	bench_info("Q4.2 margin x gain peaks at %u, inside sixteen bits",
		   worst);
}

/* -------------------------------------------------------------------------- */

int bench_cpu_suite(struct bench_report *rep, const struct bench_opts *o)
{
	static const struct bench_window_variant window2[] = {
		{ "scalar", bench_window2_scalar, false },
#if BENCH_HAVE_NEON
		{ "v0", bench_window2_v0, false },
		{ "v1", bench_window2_v1, false },
#endif
	};
	static const struct bench_window_variant window3[] = {
		{ "scalar", bench_window3_scalar, true },
#if BENCH_HAVE_NEON
		{ "v1", bench_window3_v1, true },
#endif
	};
	static const struct bench_regress_variant regress[] = {
		{ "scalar", bench_regress_scalar },
#if BENCH_HAVE_NEON
		{ "v0", bench_regress_v0 },
		{ "v1", bench_regress_v1 },
#endif
	};
	static const struct bench_share_variant share[] = {
		{ "scalar", bench_share_scalar },
#if BENCH_HAVE_NEON
		{ "v0", bench_share_v0 },
		{ "v1", bench_share_v1 },
#endif
	};
	uint64_t state = o->corpus_seed;
	int ret = 0;

	bench_add_meta(rep, "cpu.sizes", "window=%d,regress=%d,share=%d",
		       BENCH_N_WINDOW, BENCH_N_REGRESS, BENCH_N_SHARE);
	bench_add_meta(rep, "cpu.neon", "%s",
		       BENCH_HAVE_NEON ? "yes" : "no (scalar only build)");
	bench_add_meta(rep, "cpu.counters", "%s",
		       bench_pmu_ready() ? "cycles+insns" : "wall time only");

	if (!BENCH_HAVE_NEON)
		bench_warn("not an arm64 build: only the scalar twins exist, "
			   "so there is nothing here to compare");

	bench_gen_window(&state);
	bench_gen_regress(&state);
	bench_gen_share(&state);

	bench_info("cpu suite: batches of at least %llu ns, %u rounds each",
		   (unsigned long long)o->timing.target_ns, o->timing.rounds);

	if (bench_window_case(rep, o, "window2", window2,
			      sizeof(window2) / sizeof(window2[0])))
		ret = -1;
	if (bench_window_case(rep, o, "window3", window3,
			      sizeof(window3) / sizeof(window3[0])))
		ret = -1;
	if (bench_regress_case(rep, o, regress,
			       sizeof(regress) / sizeof(regress[0])))
		ret = -1;
	if (bench_share_case(rep, o, share,
			     sizeof(share) / sizeof(share[0])))
		ret = -1;

	bench_q42_case(rep);

	return ret;
}
