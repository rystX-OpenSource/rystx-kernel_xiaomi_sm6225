/* SPDX-License-Identifier: GPL-2.0 */
/*
 * taglmk_bench - shared contract.
 *
 * Everything the suites, the harness and the report agree on lives here: what a
 * measurement is, how one is identified, and the handful of helpers that talk
 * to sysfs.  Nothing here allocates on behalf of a caller except the result
 * table, which owns its own storage for the whole run.
 *
 * Copyright (C) 2026 iDeadXS <datarafi43@gmail.com>
 */
#ifndef TAGLMK_BENCH_H
#define TAGLMK_BENCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define BENCH_FORMAT_NAME	"taglmk-bench"
#define BENCH_FORMAT_VERSION	1

/*
 * Bounds.  Every one of them is enforced on both the writing and the reading
 * side: a saved file is an input like any other, and a run that compares
 * against one must not be able to read past the end of a field because the file
 * said it could.
 */
#define BENCH_NAME_MAX		48	/* one identifier field, no spaces */
#define BENCH_LINE_MAX		1024	/* one line of a saved file */
#define BENCH_RESULTS_MAX	4096
#define BENCH_META_MAX		64
#define BENCH_META_VALUE_MAX	192

/*
 * Which way is better.  Carried in the record rather than looked up from a
 * table at compare time, so a metric added to one side of a comparison cannot
 * be scored backwards by a reader that has never heard of it.
 */
enum bench_dir {
	BENCH_LOWER_BETTER,
	BENCH_HIGHER_BETTER,
};

/**
 * struct bench_result - one measured number
 * @suite: Which family of tests produced it, e.g. "cpu".
 * @tcase: What was measured, e.g. "regress".
 * @variant: Which implementation, e.g. "v1".
 * @metric: Which number, e.g. "cycles.min".
 * @unit: What it is counted in, for the report only.
 * @dir: Whether more of it is better.
 * @value: The number itself.
 *
 * The first four fields together identify a result, and that tuple is what a
 * comparison matches on.  All five strings are constrained to
 * %BENCH_NAME_MAX - 1 printable non-space characters so a record always
 * survives a round trip through the saved file unambiguously.
 */
struct bench_result {
	char		suite[BENCH_NAME_MAX];
	char		tcase[BENCH_NAME_MAX];
	char		variant[BENCH_NAME_MAX];
	char		metric[BENCH_NAME_MAX];
	char		unit[BENCH_NAME_MAX];
	enum bench_dir	dir;
	double		value;
};

/**
 * struct bench_meta - one descriptive fact about a run
 * @key: Identifier, constrained like the fields above.
 * @value: Free text, printable, no newline.
 *
 * Metadata never scores.  Some keys do decide whether two runs may be compared
 * at all; bench_meta_is_binding() is the single place that says which.
 */
struct bench_meta {
	char	key[BENCH_NAME_MAX];
	char	value[BENCH_META_VALUE_MAX];
};

/**
 * struct bench_report - everything one run produced
 * @meta: Descriptive facts, @nr_meta of them.
 * @results: Measurements, @nr_results of them.
 */
struct bench_report {
	struct bench_meta	meta[BENCH_META_MAX];
	unsigned int		nr_meta;
	struct bench_result	results[BENCH_RESULTS_MAX];
	unsigned int		nr_results;
};

/* ---------------------------------------------------------------- report.c */

void bench_report_init(struct bench_report *rep);

/*
 * Both of these validate their arguments and return false rather than store
 * anything questionable: an over-long field, an unprintable character, a
 * non-finite value, or a table that is already full.  A caller that ignores
 * the result loses a row from the report and nothing else.
 */
bool bench_add_meta(struct bench_report *rep, const char *key,
		    const char *fmt, ...)
	__attribute__((format(printf, 3, 4)));
bool bench_add(struct bench_report *rep, const char *suite, const char *tcase,
	       const char *variant, const char *metric, const char *unit,
	       enum bench_dir dir, double value);

const char *bench_get_meta(const struct bench_report *rep, const char *key);
const struct bench_result *bench_find(const struct bench_report *rep,
				      const char *suite, const char *tcase,
				      const char *variant, const char *metric);

/* Whether a mismatch in @key makes two runs incomparable rather than merely
 * interesting.  The corpus and the device configuration are binding; the kernel
 * release deliberately is not, since comparing two kernels is the point.
 */
bool bench_meta_is_binding(const char *key);

int bench_report_save(const struct bench_report *rep, const char *path);
int bench_report_load(struct bench_report *rep, const char *path);

void bench_report_print(const struct bench_report *rep, bool verbose);

/*
 * The variants of each case scored against each other inside one run.  This is
 * where the vectorisation numbers come from: the old kernel and the new one are
 * measured in the same process on the same core, so the difference between them
 * is not carrying any difference between two boots.
 */
void bench_report_ladder(const struct bench_report *rep);
void bench_report_compare(const struct bench_report *base,
			  const struct bench_report *now);

/* ------------------------------------------------------------------ util.c */

extern bool bench_verbose;

void bench_info(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void bench_warn(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void bench_err(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

void *bench_xalloc(size_t n);
void *bench_xalloc_aligned(size_t align, size_t n);

/*
 * Whole file reads and writes, both bounded and both reporting every failure.
 * bench_write_file() insists the write was accepted in full, because a short
 * write to a sysfs attribute means the value was rejected, not that it was
 * partly applied.
 */
int bench_read_file(const char *path, char *buf, size_t len);
int bench_write_file(const char *path, const char *value);
int bench_read_u64(const char *path, uint64_t *out);
bool bench_path_exists(const char *path);

/*
 * Strict conversions.  Both reject trailing rubbish, empty input and overflow.
 */
bool bench_parse_u64(const char *s, uint64_t *out);
bool bench_parse_double(const char *s, double *out);

/* True when every character is printable and none is a space. */
bool bench_name_ok(const char *s);

/* ---------------------------------------------------------------- timing.c */

/**
 * struct bench_clock - one measurement of one repetition batch
 * @ns: Elapsed wall time, CLOCK_MONOTONIC_RAW.
 * @cycles: CPU cycles, or zero when the counter is unavailable.
 * @insns: Instructions retired, or zero when unavailable.
 */
struct bench_clock {
	uint64_t	ns;
	uint64_t	cycles;
	uint64_t	insns;
};

/**
 * struct bench_timing - how a case should be measured
 * @target_ns: Grow the repetition count until a batch takes at least this long.
 * @rounds: Measured batches, after one discarded warmup.
 * @max_reps: Ceiling on the repetition count, so a kernel that is optimised
 *	away cannot spin forever.
 */
struct bench_timing {
	uint64_t	target_ns;
	unsigned int	rounds;
	uint64_t	max_reps;
};

/**
 * struct bench_stats - what the rounds came to, per repetition
 * @reps: Repetitions in each batch.
 * @rounds: Batches measured.
 * @ns_min, @ns_med, @ns_sd: Wall time per repetition.
 * @cyc_min, @cyc_med, @cyc_sd: Cycles per repetition, zero if uncounted.
 * @insn_min: Instructions per repetition, zero if uncounted.
 * @have_pmu: Whether the two counter fields mean anything.
 */
struct bench_stats {
	uint64_t	reps;
	unsigned int	rounds;
	double		ns_min, ns_med, ns_sd;
	double		cyc_min, cyc_med, cyc_sd;
	double		insn_min;
	bool		have_pmu;
};

/* The thing being measured: run @reps repetitions, return anything the caller
 * should keep so the compiler cannot decide the work was pointless.
 */
typedef uint64_t (*bench_fn)(void *ctx, uint64_t reps);

/* Bounded so the per-round arrays can live on the stack; the command line
 * validates against this rather than letting a large --rounds be silently
 * clamped.
 */
#define BENCH_ROUNDS_MAX	64

uint64_t bench_now_ns(void);

int bench_pmu_open(void);
void bench_pmu_close(void);
bool bench_pmu_ready(void);

int bench_pin_cpu(int cpu);
int bench_pick_fast_cpu(void);
void bench_try_realtime(void);
void bench_drop_realtime(void);

/*
 * Calibrate, warm up, then measure.  Returns 0 and fills @out, or -1 when the
 * function could not be made to take @target_ns within @max_reps repetitions,
 * which means the measurement would have been noise.
 */
int bench_measure(bench_fn fn, void *ctx, const struct bench_timing *t,
		  struct bench_stats *out);

/*
 * Publish one struct bench_stats as report rows.  Wall time always, the two
 * counter derived numbers only when they were actually counted, so a run
 * without a PMU is short of rows rather than full of zeroes.
 */
void bench_publish(struct bench_report *rep, const char *suite,
		   const char *tcase, const char *variant,
		   const struct bench_stats *st);

/* Keep the optimiser honest: make @v observably used. */
static inline void bench_sink(uint64_t v)
{
	__asm__ __volatile__("" : : "r"(v) : "memory");
}

/* ------------------------------------------------------------- suites */

/**
 * struct bench_opts - everything the command line settled
 */
/* What @cpu means when it is not a core number. */
#define BENCH_CPU_AUTO	(-1)	/* the fastest online core */
#define BENCH_CPU_NONE	(-2)	/* do not pin at all */

struct bench_opts {
	const char	*output;
	const char	*baseline;
	bool		run_cpu;
	bool		run_zram;
	int		cpu;		/* core, or one of the two above */
	struct bench_timing timing;

	/* zram suite */
	int		zram_dev;	/* -1: hot-add a scratch device */
	bool		zram_force;
	uint64_t	zram_size;
	const char	*zram_comp;	/* comma separated, priority order */
	uint64_t	zram_pages;
	uint64_t	zram_io_pages;	/* pages per write() */
	const char	*ir_levels;	/* comma separated */
	uint32_t	corpus_seed;
};

int bench_cpu_suite(struct bench_report *rep, const struct bench_opts *o);
int bench_zram_suite(struct bench_report *rep, const struct bench_opts *o);

/* ---------------------------------------------------------------- corpus.c */

/**
 * enum bench_page_kind - the shapes of page the corpus is built from
 *
 * A compressor's throughput depends far more on what it is fed than on how fast
 * it is called, so a run is only comparable to another that was fed the same
 * mixture.  The mixture is recorded in the saved file and a comparison refuses
 * to score two runs that disagree about it.
 */
enum bench_page_kind {
	BENCH_PAGE_ZERO,	/* deduplicated by zram, never compressed */
	BENCH_PAGE_TEXT,	/* prose-like, compresses hard */
	BENCH_PAGE_HEAP,	/* pointers and small repeated records */
	BENCH_PAGE_MIXED,	/* structure with incompressible islands */
	BENCH_PAGE_RANDOM,	/* incompressible: this is what turns huge */
	BENCH_PAGE_KINDS
};

/**
 * struct bench_corpus - a deterministic sequence of page images
 * @page_size: Bytes per page.
 * @nr_pages: Pages in the sequence.
 * @seed: What the sequence was generated from.
 * @weight: Parts per hundred of each kind; sums to 100.
 */
struct bench_corpus {
	size_t		page_size;
	uint64_t	nr_pages;
	uint32_t	seed;
	unsigned int	weight[BENCH_PAGE_KINDS];
};

void bench_corpus_init(struct bench_corpus *c, size_t page_size,
		       uint64_t nr_pages, uint32_t seed);

/* Fill @out with page @index of the sequence.  Pure: the same corpus and index
 * always give the same bytes, on any device, in any order.
 */
void bench_corpus_page(const struct bench_corpus *c, uint64_t index,
		       unsigned char *out);

/* "z=5,t=30,h=35,m=20,r=10", for the saved file. */
void bench_corpus_describe(const struct bench_corpus *c, char *buf, size_t len);

#endif /* TAGLMK_BENCH_H */
