// SPDX-License-Identifier: GPL-2.0
/*
 * taglmk_bench - the command line.
 *
 * Two jobs, in this order, every time: run the suites and save what they
 * produced, then, if a previous saved file was named, compare against it.  The
 * saving is not conditional on the comparing.  A run that cannot be compared is
 * still a run worth keeping, and a run that is thrown away because the
 * comparison it was made for turned out to be impossible is a wasted minute on
 * a phone that is now warm.
 *
 * The comparison is deliberately unforgiving.  It refuses to score two runs
 * that were fed different data, given a different device, or built for a
 * different architecture, and it refuses to score a case whose output failed
 * verification on either side.  A benchmark that always prints a percentage is
 * worse than one that sometimes says the question was wrong, because the
 * percentage ends up in a changelog either way.
 *
 * Nothing here decides that a failure was unimportant.  A suite that fails is
 * named, the exit status is non-zero, and the saved file records that it is
 * partial - but it is still saved, and a later comparison against it accounts
 * for the rows that are missing rather than quietly scoring around them.
 *
 * Copyright (C) 2026 iDeadXS <datarafi43@gmail.com>
 */
#define _GNU_SOURCE

#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <sys/utsname.h>

#include "bench.h"

#define BENCH_PROG		"taglmk_bench"

/* Where a run goes when nobody says otherwise. */
#define BENCH_OUTPUT_DEFAULT	"saved.txt"

/*
 * Long enough that the timing overhead is divided away and the governor has
 * settled, short enough that the whole cpu suite is a handful of seconds.
 */
#define BENCH_MS_DEFAULT	30u
#define BENCH_MS_MIN		1u
#define BENCH_MS_MAX		10000u

/* Odd, so the median is an observation rather than an average of two. */
#define BENCH_ROUNDS_DEFAULT	9u
#define BENCH_ROUNDS_MIN	3u

/* A kernel that got optimised away must not spin forever. */
#define BENCH_REPS_MAX		(1ull << 32)

/* The bench in the changelog: 4 GiB of disksize, the ladder, level 1. */
#define BENCH_ZRAM_SIZE_DEFAULT	(4ull << 30)
#define BENCH_ZRAM_COMP_DEFAULT	"lz4kdr,zstd:3"
#define BENCH_IR_DEFAULT	"1"

/* 128 MiB of traffic in 128 KiB requests, at a 4 KiB page. */
#define BENCH_ZRAM_PAGES_DEFAULT	32768ull
#define BENCH_ZRAM_IO_PAGES_DEFAULT	32ull

/* Bounds on the sizes, so a typo cannot ask for a terabyte of scratch. */
#define BENCH_ZRAM_SIZE_MIN	(16ull << 20)
#define BENCH_ZRAM_SIZE_MAX	(64ull << 30)
#define BENCH_ZRAM_PAGES_MIN	64ull
#define BENCH_ZRAM_PAGES_MAX	(16ull << 20)
#define BENCH_ZRAM_IO_PAGES_MIN	1ull
#define BENCH_ZRAM_IO_PAGES_MAX	512ull

/* The architecture the numbers describe.  Binding: a cycle count from one is
 * meaningless against a cycle count from another.
 */
static const char *bench_abi(void)
{
#if defined(__aarch64__)
	return "aarch64";
#elif defined(__arm__)
	return "arm";
#elif defined(__x86_64__)
	return "x86_64";
#elif defined(__i386__)
	return "i386";
#else
	return "unknown";
#endif
}

static void bench_usage(FILE *out)
{
	fprintf(out,
"usage: %s [options]\n"
"\n"
"Runs the taglmk comparison suites, saves what they measured, and\n"
"optionally compares that against a run saved earlier.\n"
"\n"
"output\n"
"  -o, --output FILE     save this run here (default: %s)\n"
"  -b, --baseline FILE   compare this run against a saved one\n"
"  -q, --quiet           do not print this run's own table\n"
"  -v, --verbose         explain what is happening and print every row\n"
"  -h, --help            this text\n"
"\n"
"what to run\n"
"  -s, --suite LIST      cpu, zram, or all (default: all)\n"
"      --rounds N        measured batches per case (%u..%u, default %u)\n"
"      --duration MS     grow each batch to at least this long\n"
"                        (%u..%u, default %u)\n"
"      --cpu N           pin to core N (default: the fastest online core)\n"
"      --no-pin          do not pin, and expect a wider spread\n"
"      --seed N          corpus seed (default 1); binding on a comparison\n"
"\n"
"the zram suite (needs root; skipped without it)\n"
"      --zram-dev N      use zram N instead of adding a scratch device\n"
"      --force           allow an adopted device that is already initialised\n"
"      --zram-size N     disksize, K/M/G accepted (default 4G)\n"
"      --zram-comp LIST  compressors in priority order (default %s)\n"
"      --zram-pages N    pages to write (default %llu)\n"
"      --zram-io-pages N pages per request (default %llu)\n"
"      --ir-levels LIST  zram_recomp_immediate levels to measure\n"
"                        (default %s)\n"
"\n"
"A comparison refuses to score two runs that disagree about anything\n"
"binding: the architecture, the page size, the seed, the input sizes, or\n"
"the zram device and traffic.  Everything else is reported and scored.\n",
		BENCH_PROG, BENCH_OUTPUT_DEFAULT,
		BENCH_ROUNDS_MIN, BENCH_ROUNDS_MAX, BENCH_ROUNDS_DEFAULT,
		BENCH_MS_MIN, BENCH_MS_MAX, BENCH_MS_DEFAULT,
		BENCH_ZRAM_COMP_DEFAULT,
		(unsigned long long)BENCH_ZRAM_PAGES_DEFAULT,
		(unsigned long long)BENCH_ZRAM_IO_PAGES_DEFAULT,
		BENCH_IR_DEFAULT);
}

static bool bench_range_ok(const char *name, uint64_t v, uint64_t lo,
			   uint64_t hi)
{
	if (v >= lo && v <= hi)
		return true;

	bench_err("%s: %" PRIu64 " is outside %" PRIu64 "..%" PRIu64, name, v,
		  lo, hi);

	return false;
}

static bool bench_arg_u64(const char *name, const char *s, uint64_t lo,
			  uint64_t hi, uint64_t *out)
{
	uint64_t v;

	if (!bench_parse_u64(s, &v)) {
		bench_err("%s: '%s' is not a whole number", name, s);
		return false;
	}

	if (!bench_range_ok(name, v, lo, hi))
		return false;

	*out = v;

	return true;
}

/*
 * The same, with a K, M or G suffix.  The shift is applied only after the
 * overflow check, so no size on the command line can wrap into a small one.
 */
static bool bench_arg_size(const char *name, const char *s, uint64_t lo,
			   uint64_t hi, uint64_t *out)
{
	char digits[32];
	unsigned int shift = 0;
	size_t len = strlen(s);
	uint64_t v;

	if (!len || len >= sizeof(digits)) {
		bench_err("%s: '%s' is not a size", name, s);
		return false;
	}

	memcpy(digits, s, len + 1);

	switch (digits[len - 1]) {
	case 'k':
	case 'K':
		shift = 10;
		break;
	case 'm':
	case 'M':
		shift = 20;
		break;
	case 'g':
	case 'G':
		shift = 30;
		break;
	default:
		break;
	}

	if (shift) {
		digits[len - 1] = '\0';
		if (!digits[0]) {
			bench_err("%s: '%s' is a suffix with no number", name,
				  s);
			return false;
		}
	}

	if (!bench_parse_u64(digits, &v)) {
		bench_err("%s: '%s' is not a size", name, s);
		return false;
	}

	if (v > (UINT64_MAX >> shift)) {
		bench_err("%s: '%s' overflows", name, s);
		return false;
	}

	v <<= shift;

	if (!bench_range_ok(name, v, lo, hi))
		return false;

	*out = v;

	return true;
}

static bool bench_arg_int(const char *name, const char *s, int *out)
{
	uint64_t v;

	if (!bench_arg_u64(name, s, 0, INT_MAX, &v))
		return false;

	*out = (int)v;

	return true;
}

/*
 * "cpu", "zram", "all" or "none", comma separated.  An unrecognised name is an
 * error rather than a warning: a mistyped suite that silently ran nothing would
 * save an empty file and report success.
 */
static bool bench_parse_suites(const char *list, struct bench_opts *o)
{
	char buf[BENCH_LINE_MAX];
	char *save = NULL, *tok;

	if (strlen(list) >= sizeof(buf)) {
		bench_err("--suite: the list is too long");
		return false;
	}

	snprintf(buf, sizeof(buf), "%s", list);

	o->run_cpu = false;
	o->run_zram = false;

	for (tok = strtok_r(buf, ",", &save); tok;
	     tok = strtok_r(NULL, ",", &save)) {
		if (!strcmp(tok, "all")) {
			o->run_cpu = true;
			o->run_zram = true;
		} else if (!strcmp(tok, "cpu")) {
			o->run_cpu = true;
		} else if (!strcmp(tok, "zram")) {
			o->run_zram = true;
		} else if (!strcmp(tok, "none")) {
			/* Explicitly nothing: useful with --baseline alone. */
		} else {
			bench_err("--suite: '%s' is not a suite", tok);
			return false;
		}
	}

	return true;
}

/* ISO 8601 local time, or the empty string if the clock will not say. */
static void bench_stamp(char *buf, size_t len)
{
	struct tm tm;
	time_t now = time(NULL);

	buf[0] = '\0';

	if (now == (time_t)-1 || !localtime_r(&now, &tm))
		return;

	if (!strftime(buf, len, "%Y-%m-%dT%H:%M:%S%z", &tm))
		buf[0] = '\0';
}

/*
 * The facts about the machine, recorded before anything is measured.  Which of
 * these block a comparison is bench_meta_is_binding()'s decision, not this
 * function's: everything known is written down, and the reader of a saved file
 * gets the whole picture even for the parts that never score.
 */
static void bench_add_environment(struct bench_report *rep,
				  const struct bench_opts *o)
{
	char stamp[64];
	struct utsname u;
	long page = sysconf(_SC_PAGESIZE);

	bench_add_meta(rep, "tool", "%s %d", BENCH_PROG, BENCH_FORMAT_VERSION);
	bench_add_meta(rep, "abi", "%s", bench_abi());
	bench_add_meta(rep, "pagesize", "%ld", page > 0 ? page : 0);
	bench_add_meta(rep, "seed", "%" PRIu32, o->corpus_seed);

	if (!uname(&u)) {
		bench_add_meta(rep, "kernel", "%s", u.release);
		bench_add_meta(rep, "kernel.build", "%s", u.version);
		bench_add_meta(rep, "machine", "%s", u.machine);
	}

	bench_stamp(stamp, sizeof(stamp));
	if (stamp[0])
		bench_add_meta(rep, "date", "%s", stamp);

	bench_add_meta(rep, "root", "%s", geteuid() ? "no" : "yes");
	bench_add_meta(rep, "timing", "%llu ms x %u rounds",
		       (unsigned long long)(o->timing.target_ns / 1000000),
		       o->timing.rounds);
}

/*
 * Pin and open the counters, in that order and both of them best effort.  Each
 * can be refused on a locked down phone, and neither is worth abandoning a run
 * over: what they change is how much the numbers can be trusted, which is why
 * what happened is written down.  Priority is left to bench_run_suites(),
 * because the two suites do not want the same scheduler.
 */
static void bench_setup_machine(struct bench_report *rep,
				const struct bench_opts *o)
{
	int cpu = BENCH_CPU_NONE;

	if (o->cpu == BENCH_CPU_AUTO)
		cpu = bench_pick_fast_cpu();
	else if (o->cpu != BENCH_CPU_NONE)
		cpu = o->cpu;

	/* bench_pin_cpu() has already said why, if it refused. */
	if (cpu >= 0 && bench_pin_cpu(cpu))
		cpu = -1;

	if (cpu >= 0)
		bench_add_meta(rep, "cpu", "%d", cpu);
	else
		bench_add_meta(rep, "cpu", "not pinned");

	if (bench_pmu_open() && o->run_cpu)
		bench_warn("no hardware counters, so this run has wall time "
			   "but no cycle count; run as root, or lower "
			   "kernel.perf_event_paranoid");
}

/*
 * Runs what was asked for and returns the number of suites that failed.  A
 * suite that fails does not stop the next one: the cpu numbers are worth having
 * even on a kernel whose zram refuses to be configured, and the reverse.
 */
static unsigned int bench_run_suites(struct bench_report *rep,
				     const struct bench_opts *o)
{
	char ran[BENCH_META_VALUE_MAX];
	unsigned int failed = 0;

	snprintf(ran, sizeof(ran), "%s%s%s",
		 o->run_cpu ? "cpu" : "",
		 o->run_cpu && o->run_zram ? "," : "",
		 o->run_zram ? "zram" : "");
	bench_add_meta(rep, "suites", "%s", ran[0] ? ran : "none");

	if (o->run_cpu) {
		/*
		 * The FIFO band is for the cpu kernels alone.  Each of their
		 * batches is a few tens of milliseconds and the score is the
		 * minimum of many, so the once a second stall the kernel's RT
		 * throttle inserts lands in some batches and not in the best
		 * one.  A zram pass is a single long timed region and has no
		 * such escape, so it runs at ordinary priority - it spends its
		 * time in the compressor and in writeback anyway, where being
		 * at the front of the queue buys very little.
		 */
		bench_try_realtime();

		if (bench_cpu_suite(rep, o)) {
			bench_err("the cpu suite did not finish");
			failed++;
		}

		bench_drop_realtime();
	}

	if (o->run_zram && bench_zram_suite(rep, o)) {
		bench_err("the zram suite did not finish");
		failed++;
	}

	return failed;
}

/*
 * A baseline that was itself incomplete is still usable - the rows it does hold
 * are real - but a reader comparing against it deserves to be told, because a
 * missing case looks exactly like a case that was never asked for.
 */
static void bench_warn_partial_baseline(const struct bench_report *base,
					const char *path)
{
	const char *status = bench_get_meta(base, "status");

	if (status && strcmp(status, "complete"))
		bench_warn("the baseline %s is marked '%s': some of what it "
			   "was asked to measure is missing", path, status);
}

/*
 * Fail fast on an unwritable destination.  Measuring for three minutes and then
 * discovering the results have nowhere to go is the one avoidable way to lose a
 * run, and the directory can be asked without disturbing an existing file.
 */
static bool bench_output_writable(const char *path)
{
	char dir[BENCH_LINE_MAX];
	const char *slash = strrchr(path, '/');
	size_t len;

	if (!slash)
		return !access(".", W_OK);

	len = (size_t)(slash - path);
	if (!len)		/* "/name": the root directory */
		len = 1;

	if (len >= sizeof(dir)) {
		errno = ENAMETOOLONG;
		return false;
	}

	memcpy(dir, path, len);
	dir[len] = '\0';

	return !access(dir, W_OK);
}

enum {
	BENCH_OPT_ROUNDS = 1000,
	BENCH_OPT_DURATION,
	BENCH_OPT_CPU,
	BENCH_OPT_NO_PIN,
	BENCH_OPT_SEED,
	BENCH_OPT_ZRAM_DEV,
	BENCH_OPT_FORCE,
	BENCH_OPT_ZRAM_SIZE,
	BENCH_OPT_ZRAM_COMP,
	BENCH_OPT_ZRAM_PAGES,
	BENCH_OPT_ZRAM_IO_PAGES,
	BENCH_OPT_IR_LEVELS,
};

static const struct option bench_long_opts[] = {
	{ "output",		required_argument, NULL, 'o' },
	{ "baseline",		required_argument, NULL, 'b' },
	{ "suite",		required_argument, NULL, 's' },
	{ "quiet",		no_argument,	   NULL, 'q' },
	{ "verbose",		no_argument,	   NULL, 'v' },
	{ "help",		no_argument,	   NULL, 'h' },
	{ "rounds",		required_argument, NULL, BENCH_OPT_ROUNDS },
	{ "duration",		required_argument, NULL, BENCH_OPT_DURATION },
	{ "cpu",		required_argument, NULL, BENCH_OPT_CPU },
	{ "no-pin",		no_argument,	   NULL, BENCH_OPT_NO_PIN },
	{ "seed",		required_argument, NULL, BENCH_OPT_SEED },
	{ "zram-dev",		required_argument, NULL, BENCH_OPT_ZRAM_DEV },
	{ "force",		no_argument,	   NULL, BENCH_OPT_FORCE },
	{ "zram-size",		required_argument, NULL, BENCH_OPT_ZRAM_SIZE },
	{ "zram-comp",		required_argument, NULL, BENCH_OPT_ZRAM_COMP },
	{ "zram-pages",		required_argument, NULL, BENCH_OPT_ZRAM_PAGES },
	{ "zram-io-pages",	required_argument, NULL,
						   BENCH_OPT_ZRAM_IO_PAGES },
	{ "ir-levels",		required_argument, NULL, BENCH_OPT_IR_LEVELS },
	{ NULL,			0,		   NULL, 0 },
};

int main(int argc, char **argv)
{
	struct bench_report *now, *base = NULL;
	struct bench_opts o;
	uint64_t ms = BENCH_MS_DEFAULT;
	uint64_t rounds = BENCH_ROUNDS_DEFAULT;
	uint64_t v;
	unsigned int failed;
	bool suite_named = false;
	bool quiet = false;
	int c, ret = 0;

	memset(&o, 0, sizeof(o));
	o.output = BENCH_OUTPUT_DEFAULT;
	o.run_cpu = true;
	o.run_zram = true;
	o.cpu = BENCH_CPU_AUTO;
	o.zram_dev = -1;
	o.zram_size = BENCH_ZRAM_SIZE_DEFAULT;
	o.zram_comp = BENCH_ZRAM_COMP_DEFAULT;
	o.zram_pages = BENCH_ZRAM_PAGES_DEFAULT;
	o.zram_io_pages = BENCH_ZRAM_IO_PAGES_DEFAULT;
	o.ir_levels = BENCH_IR_DEFAULT;
	o.corpus_seed = 1;
	o.timing.max_reps = BENCH_REPS_MAX;

	while ((c = getopt_long(argc, argv, "o:b:s:qvh", bench_long_opts,
				NULL)) != -1) {
		switch (c) {
		case 'o':
			o.output = optarg;
			break;
		case 'b':
			o.baseline = optarg;
			break;
		case 's':
			if (!bench_parse_suites(optarg, &o))
				return 2;
			suite_named = true;
			break;
		case 'q':
			quiet = true;
			break;
		case 'v':
			bench_verbose = true;
			break;
		case 'h':
			bench_usage(stdout);
			return 0;
		case BENCH_OPT_ROUNDS:
			if (!bench_arg_u64("--rounds", optarg,
					   BENCH_ROUNDS_MIN, BENCH_ROUNDS_MAX,
					   &rounds))
				return 2;
			break;
		case BENCH_OPT_DURATION:
			if (!bench_arg_u64("--duration", optarg, BENCH_MS_MIN,
					   BENCH_MS_MAX, &ms))
				return 2;
			break;
		case BENCH_OPT_CPU:
			if (!bench_arg_int("--cpu", optarg, &o.cpu))
				return 2;
			break;
		case BENCH_OPT_NO_PIN:
			o.cpu = BENCH_CPU_NONE;
			break;
		case BENCH_OPT_SEED:
			if (!bench_arg_u64("--seed", optarg, 0, UINT32_MAX, &v))
				return 2;
			o.corpus_seed = (uint32_t)v;
			break;
		case BENCH_OPT_ZRAM_DEV:
			if (!bench_arg_int("--zram-dev", optarg, &o.zram_dev))
				return 2;
			break;
		case BENCH_OPT_FORCE:
			o.zram_force = true;
			break;
		case BENCH_OPT_ZRAM_SIZE:
			if (!bench_arg_size("--zram-size", optarg,
					    BENCH_ZRAM_SIZE_MIN,
					    BENCH_ZRAM_SIZE_MAX, &o.zram_size))
				return 2;
			break;
		case BENCH_OPT_ZRAM_COMP:
			o.zram_comp = optarg;
			break;
		case BENCH_OPT_ZRAM_PAGES:
			if (!bench_arg_u64("--zram-pages", optarg,
					   BENCH_ZRAM_PAGES_MIN,
					   BENCH_ZRAM_PAGES_MAX,
					   &o.zram_pages))
				return 2;
			break;
		case BENCH_OPT_ZRAM_IO_PAGES:
			if (!bench_arg_u64("--zram-io-pages", optarg,
					   BENCH_ZRAM_IO_PAGES_MIN,
					   BENCH_ZRAM_IO_PAGES_MAX,
					   &o.zram_io_pages))
				return 2;
			break;
		case BENCH_OPT_IR_LEVELS:
			o.ir_levels = optarg;
			break;
		default:
			bench_usage(stderr);
			return 2;
		}
	}

	/*
	 * A leftover word is almost always a filename that was meant to follow
	 * -o.  Taking it as decoration would save the run to the default name
	 * while the caller believed it went somewhere else.
	 */
	if (optind != argc) {
		bench_err("unexpected argument '%s'", argv[optind]);
		bench_usage(stderr);
		return 2;
	}

	o.timing.target_ns = ms * 1000000ull;
	o.timing.rounds = (unsigned int)rounds;

	if (!o.run_cpu && !o.run_zram && !o.baseline) {
		bench_err("nothing to run and nothing to compare");
		return 2;
	}

	if (!bench_output_writable(o.output)) {
		bench_err("cannot write to %s: %s", o.output,
			  strerror(errno));
		return 2;
	}

	if (o.baseline && !strcmp(o.baseline, o.output))
		bench_warn("--output and --baseline are the same file, so this "
			   "run replaces the baseline it is compared against");

	/*
	 * The zram suite has to add or reset a block device and write to sysfs,
	 * none of which is available to an unprivileged caller.  Asked for by
	 * name it is an error, because a run that measured nothing must not
	 * look like a run that measured everything; arrived at by default it is
	 * a skip, so the cpu numbers are still worth having.
	 */
	if (o.run_zram && geteuid()) {
		if (suite_named) {
			bench_err("the zram suite has to run as root");
			return 2;
		}

		bench_warn("not root: skipping the zram suite (name it with "
			   "--suite=zram to make that an error instead)");
		o.run_zram = false;
	}

	/*
	 * The baseline is read before anything is measured.  A path that does
	 * not exist, or a file this build cannot parse, is worth finding out
	 * about now rather than after the phone has spent a minute warming up.
	 */
	if (o.baseline) {
		base = bench_xalloc(sizeof(*base));

		if (bench_report_load(base, o.baseline)) {
			free(base);
			return 2;
		}

		bench_warn_partial_baseline(base, o.baseline);
		bench_info("baseline %s: %u results, %u facts", o.baseline,
			   base->nr_results, base->nr_meta);
	}

	now = bench_xalloc(sizeof(*now));
	bench_report_init(now);

	bench_add_environment(now, &o);
	bench_setup_machine(now, &o);

	failed = bench_run_suites(now, &o);

	bench_pmu_close();

	bench_add_meta(now, "status", "%s", failed ? "partial" : "complete");

	if (failed) {
		bench_warn("%u suite%s did not finish; %s records what was "
			   "measured and is marked partial", failed,
			   failed == 1 ? "" : "s", o.output);
		ret = 1;
	}

	/*
	 * Saved whatever happened.  A partial run is still evidence, and the
	 * saved file says so in its own metadata, so a later comparison against
	 * it accounts for the rows that are missing.
	 */
	if (bench_report_save(now, o.output)) {
		bench_err("cannot save to %s: %s", o.output, strerror(errno));
		ret = 1;
	} else {
		bench_info("saved %u results to %s", now->nr_results,
			   o.output);
	}

	if (!quiet) {
		bench_report_print(now, bench_verbose);
		bench_report_ladder(now);
	}

	if (base) {
		bench_report_compare(base, now);
		free(base);
	} else if (!quiet) {
		printf("\nPass this file back with -b %s to compare a later "
		       "run against it.\n", o.output);
	}

	free(now);

	return ret;
}
