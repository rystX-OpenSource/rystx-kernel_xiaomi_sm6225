/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * Userspace unit tests for the FLD decision math (fld_math.h).
 * Build and run:  cc -O2 -Wall -Wextra -o test_fld_math test_fld_math.c && ./test_fld_math
 */

#include <stdio.h>
#include <stdlib.h>

#include "../fld_math.h"

static int failures;

#define CHECK(cond, ...) do {						\
	if (!(cond)) {							\
		failures++;						\
		printf("FAIL %s:%d: ", __func__, __LINE__);		\
		printf(__VA_ARGS__);					\
		printf("\n");						\
	}								\
} while (0)

static void test_ewma_rise_fast(void)
{
	/* 0 -> full load in one tick with up_weight 3/4 */
	CHECK(fld_ewma(0, 1024, 75, 12) == 768, "up 75%% of 1024 from 0");
	CHECK(fld_ewma(768, 1024, 75, 12) == 960, "second rise tick");
	/* Full load in ~3 ticks */
	CHECK(fld_ewma(fld_ewma(fld_ewma(0, 1024, 75, 12), 1024, 75, 12),
			1024, 75, 12) >= 1000, "converges upward");
}

static void test_ewma_decay_slow(void)
{
	/* 1/8 decay: 1024 -> 902 in one tick with down_weight 12% */
	CHECK(fld_ewma(1024, 0, 75, 12) == 902, "down 12%% of 1024");
	/* Zero load does not move a zero estimate */
	CHECK(fld_ewma(0, 0, 75, 12) == 0, "stable at zero");
	/* Symmetric point: load == est keeps est unchanged */
	CHECK(fld_ewma(500, 500, 75, 12) == 500, "no change on equal load");
}

static void test_blend(void)
{
	/* EWMA dominates when feeder is quiet */
	CHECK(fld_blend(800, 100, 1000) == 800, "ewma wins");
	/* Feeder dominates with gain 1.0: 700 util -> 700 */
	CHECK(fld_blend(100, 700, 1000) == 700, "feeder wins at K=1.0");
	/* Gain scales the feeder term: K=2.0 doubles it */
	CHECK(fld_blend(100, 700, 2000) == 1400, "feeder gain 2.0");
	/* Gain zero disables feed-forward entirely */
	CHECK(fld_blend(100, 900, 0) == 100, "feeder gain 0");
	/* Feed-forward saturates at FLD_LOAD_MAX */
	CHECK(fld_blend(0, 1024, 8000) == FLD_LOAD_MAX, "feeder saturates");
}

static void test_target_freq(void)
{
	/* est = 1.0, target_load 80%: need 1.25x the current frequency */
	CHECK(fld_target_freq(1000000000ULL, 1024, 80) == 1250000000ULL,
		"1.0 load at 80%% -> 1.25x");
	/* Half load asks for ~0.625x */
	CHECK(fld_target_freq(1000000000ULL, 512, 80) == 625000000ULL,
		"0.5 load at 80%% -> 0.625x");
	/* Zero estimate asks for nothing */
	CHECK(fld_target_freq(1000000000ULL, 0, 80) == 0, "zero est");
	/* Divide-by-zero guard: target_load 0 falls back to 1% */
	CHECK(fld_target_freq(1000000000ULL, 1024, 0) == 100000000000ULL,
		"target_load 0 does not crash");
}

static void test_snap_level(void)
{
	/* Descending table like KGSL pwrlevels[] (index 0 = fastest) */
	const unsigned int freqs[] = { 1000000, 800000, 500000, 200000 };
	int n = 4;

	/* Exact hit */
	CHECK(fld_snap_level(freqs, n, 800000, 0, 3) == 1, "exact snap");
	/* Nearest wins: 550k is closer to 500k than 800k */
	CHECK(fld_snap_level(freqs, n, 550000, 0, 3) == 2, "nearest snap");
	/* Equidistant picks the faster level (safe side: more headroom) */
	CHECK(fld_snap_level(freqs, n, 650000, 0, 3) == 1, "tie -> faster");
	/* Clamp to fastest allowed (lo_idx) */
	CHECK(fld_snap_level(freqs, n, 999999, 1, 3) == 1, "clamp fastest");
	/* Clamp to slowest allowed (hi_idx) */
	CHECK(fld_snap_level(freqs, n, 100, 0, 2) == 2, "clamp slowest");
	/* Degenerate table */
	CHECK(fld_snap_level(freqs, 0, 500000, 0, 0) == 0, "empty table");
}

static void test_decide_ramp_up_immediate(void)
{
	/* Dense table like a real OPP list (index 0 = fastest) */
	const unsigned int freqs[] = { 1000000, 900000, 800000, 700000,
					600000, 500000, 400000, 300000,
					200000 };
	struct fld_ramp r = { .est = 850, .calm = 0 };
	int level;

	/*
	 * Recent heavy history (est 850) plus a full-load frame pushes the
	 * estimate past target_load and up-ramps with no hysteresis delay.
	 */
	level = fld_decide(&r, 1024, 0, 75, 12, 1000, 80, 4, 5,
			freqs, 9, 0, 8);
	CHECK(level == 4, "up ramp is immediate (got %d)", level);
	CHECK(r.calm == 0, "up ramp clears calm");

	/* Feed-forward alone must trigger the same immediate up ramp */
	r.est = 0;
	level = fld_decide(&r, 0, 1024, 75, 12, 1000, 80, 4, 5,
			freqs, 9, 0, 8);
	CHECK(level == 4, "feed-forward up ramp (got %d)", level);
	CHECK(r.calm == 0, "feed-forward up clears calm");
}

static void test_decide_ramp_down_hysteresis(void)
{
	const unsigned int freqs[] = { 1000000, 900000, 800000, 700000,
					600000, 500000, 400000, 300000,
					200000 };
	struct fld_ramp r = { .est = 1024, .calm = 0 };
	int i, level;

	/*
	 * Idle GPU at the fastest level: every tick is calm, but the ramp
	 * down is held for down_frames and then steps one snapped level.
	 */
	for (i = 0; i < 4; i++) {
		level = fld_decide(&r, 0, 0, 75, 12, 1000, 80, 4, 0,
				freqs, 9, 0, 8);
		if (i < 3)
			CHECK(level == -1, "tick %d held back (calm=%d)",
				i, r.calm);
		else
			CHECK(level == 2, "tick %d ramps down to %d", i, level);
	}
	/* After the drop, a further idle tick keeps the (now current) level */
	level = fld_decide(&r, 0, 0, 75, 12, 1000, 80, 4, 2, freqs, 9, 0, 8);
	CHECK(level == -1, "no flapping right after a down ramp");
}

static void test_decide_up_interrupts_down(void)
{
	const unsigned int freqs[] = { 1000000, 900000, 800000, 700000,
					600000, 500000, 400000, 300000,
					200000 };
	struct fld_ramp r = { .est = 1024, .calm = 0 };
	int level;

	/* Two calm ticks accumulate */
	fld_decide(&r, 0, 0, 75, 12, 1000, 80, 4, 0, freqs, 9, 0, 8);
	level = fld_decide(&r, 0, 0, 75, 12, 1000, 80, 4, 0, freqs, 9, 0, 8);
	CHECK(level == -1 && r.calm == 2, "calm accumulates");

	/* A heavy frame from a mid-table level boosts immediately */
	r.est = 300;
	r.calm = 3;
	level = fld_decide(&r, 0, 2048, 75, 12, 1000, 80, 4, 5,
			freqs, 9, 0, 8);
	CHECK(level == 0, "heavy frame ramps up (got %d)", level);
	CHECK(r.calm == 0, "heavy frame clears calm");
}

static void test_clamp(void)
{
	CHECK(fld_clamp_u32(5, 1, 10) == 5, "in range");
	CHECK(fld_clamp_u32(0, 1, 10) == 1, "below range");
	CHECK(fld_clamp_u32(99, 1, 10) == 10, "above range");
}

int main(void)
{
	test_ewma_rise_fast();
	test_ewma_decay_slow();
	test_blend();
	test_target_freq();
	test_snap_level();
	test_decide_ramp_up_immediate();
	test_decide_ramp_down_hysteresis();
	test_decide_up_interrupts_down();
	test_clamp();

	if (failures) {
		printf("%d check(s) FAILED\n", failures);
		return EXIT_FAILURE;
	}
	printf("all fld_math tests passed\n");
	return EXIT_SUCCESS;
}
