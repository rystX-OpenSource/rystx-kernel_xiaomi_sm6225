// SPDX-License-Identifier: GPL-2.0
/*
 * taglmk_bench - the stopwatch.
 *
 * A phone is a hostile place to measure a few hundred cycles.  The frequency
 * moves, the scheduler moves the task, and the cores are not alike.  Four
 * things are done about it, in the order they matter.
 *
 * Pin.  One core, chosen for the highest cpuinfo_max_freq, for the whole run,
 * so a kernel is never timed on a little core and its rival on a big one.
 *
 * Count, do not time.  Wall time is a frequency measurement as much as a work
 * measurement.  Cycles retired, from the PMU via perf_event_open(), are the
 * number a changelog can honestly quote; wall time is recorded alongside so a
 * reader can see when the two disagree and distrust the run.
 *
 * Batch, then take the minimum.  Each kernel is called in a batch sized to run
 * for a target duration, so the timing overhead is divided away, and the run is
 * repeated.  The minimum of the repeats is the figure of merit: interference
 * only ever adds, so the smallest observation is the closest to the truth.  The
 * spread is published too, since a large one means the minimum was luck.
 *
 * Ask, never insist.  Real time priority and the PMU are both attempts.  If
 * either is refused the run continues and says what it lost, because a
 * benchmark that only works as root is a benchmark nobody runs.
 *
 * Copyright (C) 2026 iDeadXS <datarafi43@gmail.com>
 */
#define _GNU_SOURCE

#include <errno.h>
#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>

#include "bench.h"

/* How far the repetition count may be scaled in one calibration step. */
#define BENCH_CALIBRATE_STEP_MAX	64u

/* Calibration gives up after this many steps even if the target is not met. */
#define BENCH_CALIBRATE_TRIES	32

/* Where the topology lives, so the path snprintfs stay readable. */
#define BENCH_CPU_DIR		"/sys/devices/system/cpu"

static int bench_pmu_cycles_fd = -1;
static int bench_pmu_insns_fd = -1;

/* Whether this process actually made it into the FIFO band. */
static bool bench_realtime_on;

uint64_t bench_now_ns(void)
{
	struct timespec ts;

	/*
	 * RAW, so that an NTP correction landing mid-run cannot appear as a
	 * kernel that got faster.
	 */
	if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts)) {
		/* Nothing sensible to fall back to; a zero clock is visible. */
		bench_warn("clock_gettime failed: %s", strerror(errno));
		return 0;
	}

	return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* bionic has no wrapper for this one, so make the call directly. */
static int bench_perf_open(struct perf_event_attr *attr, int group_fd)
{
	return (int)syscall(__NR_perf_event_open, attr, /* pid: self */ 0,
			    /* cpu: any */ -1, group_fd, PERF_FLAG_FD_CLOEXEC);
}

int bench_pmu_open(void)
{
	struct perf_event_attr attr;

	if (bench_pmu_cycles_fd >= 0)
		return 0;

	memset(&attr, 0, sizeof(attr));
	attr.size = sizeof(attr);
	attr.type = PERF_TYPE_HARDWARE;
	attr.config = PERF_COUNT_HW_CPU_CYCLES;
	attr.disabled = 1;
	attr.inherit = 0;
	/*
	 * Userspace only.  An interrupt taken during a batch would otherwise be
	 * charged to the kernel being measured, and interrupts do not arrive
	 * evenly between two variants of the same loop.
	 */
	attr.exclude_kernel = 1;
	attr.exclude_hv = 1;
	/* One read returns the whole group, so the two counters cannot skew. */
	attr.read_format = PERF_FORMAT_GROUP;

	bench_pmu_cycles_fd = bench_perf_open(&attr, -1);
	if (bench_pmu_cycles_fd < 0) {
		bench_info("no cycle counter (%s); reporting wall time only",
			   strerror(errno));
		if (errno == EACCES || errno == EPERM)
			bench_info("  run as root, or lower "
				   "kernel.perf_event_paranoid");
		return -1;
	}

	/* Instructions are a bonus: they say whether a difference in cycles
	 * came from doing less work or from doing the same work better.
	 */
	memset(&attr, 0, sizeof(attr));
	attr.size = sizeof(attr);
	attr.type = PERF_TYPE_HARDWARE;
	attr.config = PERF_COUNT_HW_INSTRUCTIONS;
	attr.disabled = 1;
	attr.inherit = 0;
	attr.exclude_kernel = 1;
	attr.exclude_hv = 1;

	bench_pmu_insns_fd = bench_perf_open(&attr, bench_pmu_cycles_fd);
	if (bench_pmu_insns_fd < 0)
		bench_info("no instruction counter (%s)", strerror(errno));

	return 0;
}

void bench_pmu_close(void)
{
	if (bench_pmu_insns_fd >= 0) {
		close(bench_pmu_insns_fd);
		bench_pmu_insns_fd = -1;
	}
	if (bench_pmu_cycles_fd >= 0) {
		close(bench_pmu_cycles_fd);
		bench_pmu_cycles_fd = -1;
	}
}

bool bench_pmu_ready(void)
{
	return bench_pmu_cycles_fd >= 0;
}

static void bench_pmu_start(void)
{
	if (bench_pmu_cycles_fd < 0)
		return;

	ioctl(bench_pmu_cycles_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
	ioctl(bench_pmu_cycles_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
}

/*
 * Stop the group and read it.  Anything unexpected - a short read, a group that
 * came back with a different number of members than were opened - discards the
 * counters for this batch rather than reporting a number built from part of a
 * buffer.  Wall time is unaffected and the batch still counts.
 */
static void bench_pmu_stop(struct bench_clock *c)
{
	uint64_t buf[3] = { 0, 0, 0 };
	ssize_t got;

	c->cycles = 0;
	c->insns = 0;

	if (bench_pmu_cycles_fd < 0)
		return;

	ioctl(bench_pmu_cycles_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);

	got = read(bench_pmu_cycles_fd, buf, sizeof(buf));
	if (got < (ssize_t)(2 * sizeof(uint64_t)))
		return;

	/* buf[0] is the member count, then one value per member. */
	if (buf[0] < 1 || buf[0] > 2)
		return;
	if ((size_t)got < (size_t)(buf[0] + 1) * sizeof(uint64_t))
		return;

	c->cycles = buf[1];
	if (buf[0] == 2)
		c->insns = buf[2];
}

int bench_pin_cpu(int cpu)
{
	cpu_set_t set;

	if (cpu < 0 || cpu >= CPU_SETSIZE)
		return -1;

	CPU_ZERO(&set);
	CPU_SET(cpu, &set);

	if (sched_setaffinity(0, sizeof(set), &set)) {
		bench_warn("cannot pin to cpu%d: %s", cpu, strerror(errno));
		return -1;
	}

	bench_info("pinned to cpu%d", cpu);

	return 0;
}

/*
 * The fastest core that is online, by cpuinfo_max_freq.  Ties go to the highest
 * numbered core, which on every big.LITTLE part in this family is the one least
 * likely to be handling interrupts.  Returns -1 when the topology cannot be
 * read, and the caller then simply does not pin.
 */
int bench_pick_fast_cpu(void)
{
	uint64_t best_khz = 0;
	int best = -1;

	for (int cpu = 0; cpu < CPU_SETSIZE && cpu < 64; cpu++) {
		char path[128];
		uint64_t khz, online;

		/*
		 * cpu0 has no "online" file on most kernels because it cannot
		 * be taken down; absence therefore means online, not offline.
		 */
		snprintf(path, sizeof(path), "%s/cpu%d/online",
			 BENCH_CPU_DIR, cpu);
		if (bench_path_exists(path) &&
		    (bench_read_u64(path, &online) || !online))
			continue;

		/* A core with no cpufreq node has no maximum to compare. */
		snprintf(path, sizeof(path),
			 "%s/cpu%d/cpufreq/cpuinfo_max_freq", BENCH_CPU_DIR,
			 cpu);
		if (bench_read_u64(path, &khz))
			continue;

		if (khz >= best_khz) {
			best_khz = khz;
			best = cpu;
		}
	}

	if (best >= 0)
		bench_info("cpu%d is the fastest online core (%llu kHz)", best,
			   (unsigned long long)best_khz);

	return best;
}

/*
 * Try for real time priority, at the very bottom of the FIFO band: enough to
 * keep an ordinary background task from preempting a batch, not enough to be
 * worth a second thought if the run is killed.  The kernel's own RT throttle
 * still applies, so a runaway loop cannot take the core away entirely.
 */
void bench_try_realtime(void)
{
	struct sched_param p;
	int prio;

	prio = sched_get_priority_min(SCHED_FIFO);
	if (prio < 0) {
		bench_info("SCHED_FIFO unavailable; running at normal "
			   "priority");
		return;
	}

	memset(&p, 0, sizeof(p));
	p.sched_priority = prio;

	if (sched_setscheduler(0, SCHED_FIFO, &p)) {
		bench_info("staying at normal priority (%s); expect more "
			   "spread", strerror(errno));
		return;
	}

	bench_realtime_on = true;
	bench_info("running SCHED_FIFO at priority %d", prio);
}

/*
 * Hand the band back.  Worth doing between suites: a long timed region at
 * SCHED_FIFO eventually meets the kernel's RT throttle, which stops the task
 * for tens of milliseconds once a second.  A suite that takes the minimum of
 * many short batches shrugs that off; one that times a single long pass wears
 * it as a slowdown.
 */
void bench_drop_realtime(void)
{
	struct sched_param p;

	if (!bench_realtime_on)
		return;

	memset(&p, 0, sizeof(p));
	p.sched_priority = 0;

	if (sched_setscheduler(0, SCHED_OTHER, &p)) {
		bench_warn("cannot leave the FIFO band (%s); a throttling "
			   "stall may sit inside the next timed pass",
			   strerror(errno));
		return;
	}

	bench_realtime_on = false;
	bench_info("back to normal priority");
}

/* One batch: @reps repetitions of @fn, timed and counted. */
static void bench_one(bench_fn fn, void *ctx, uint64_t reps,
		      struct bench_clock *c)
{
	uint64_t t0, t1, keep;

	bench_pmu_start();
	t0 = bench_now_ns();

	keep = fn(ctx, reps);

	t1 = bench_now_ns();
	bench_pmu_stop(c);

	/* Make the result observably used, after the clock has been read. */
	bench_sink(keep);

	c->ns = t1 > t0 ? t1 - t0 : 0;
}

static int bench_cmp_double(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;

	if (x < y)
		return -1;
	if (x > y)
		return 1;

	return 0;
}

/* Median of @n values, which sorts @v in place. */
static double bench_median(double *v, unsigned int n)
{
	if (!n)
		return 0.0;

	qsort(v, n, sizeof(*v), bench_cmp_double);

	if (n & 1)
		return v[n / 2];

	return (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

/* Sample standard deviation of @n values. */
static double bench_stddev(const double *v, unsigned int n)
{
	double mean = 0.0, sum = 0.0;

	if (n < 2)
		return 0.0;

	for (unsigned int i = 0; i < n; i++)
		mean += v[i];
	mean /= n;

	for (unsigned int i = 0; i < n; i++) {
		double d = v[i] - mean;

		sum += d * d;
	}

	return sqrt(sum / (n - 1));
}

static double bench_min(const double *v, unsigned int n)
{
	double m;

	if (!n)
		return 0.0;

	m = v[0];
	for (unsigned int i = 1; i < n; i++)
		if (v[i] < m)
			m = v[i];

	return m;
}

/*
 * Grow the repetition count until one batch takes at least @target_ns.  The
 * step is derived from how far short the last attempt fell, but clamped, so a
 * first batch that happened to be interrupted cannot ask for a billion
 * repetitions on the next try.
 */
static uint64_t bench_calibrate(bench_fn fn, void *ctx,
				const struct bench_timing *t)
{
	uint64_t reps = 1;

	for (unsigned int try = 0; try < BENCH_CALIBRATE_TRIES; try++) {
		struct bench_clock c;
		uint64_t want;

		bench_one(fn, ctx, reps, &c);

		if (c.ns >= t->target_ns)
			return reps;

		if (reps >= t->max_reps)
			return 0;

		/*
		 * A batch too short to have measured at all says nothing about
		 * the right size, so just double.  Otherwise scale by the
		 * shortfall, with a little headroom.
		 */
		if (c.ns < 1000) {
			want = reps * 2;
		} else {
			uint64_t scale = t->target_ns / c.ns + 1;

			if (scale > BENCH_CALIBRATE_STEP_MAX)
				scale = BENCH_CALIBRATE_STEP_MAX;
			want = reps * scale;
		}

		/* Overflow, or past the ceiling: stop at the ceiling. */
		if (want <= reps || want > t->max_reps)
			want = t->max_reps;

		reps = want;
	}

	return 0;
}

int bench_measure(bench_fn fn, void *ctx, const struct bench_timing *t,
		  struct bench_stats *out)
{
	double ns[BENCH_ROUNDS_MAX], cyc[BENCH_ROUNDS_MAX];
	double insn[BENCH_ROUNDS_MAX];
	unsigned int rounds, counted = 0;
	struct bench_clock c;
	uint64_t reps;

	memset(out, 0, sizeof(*out));

	rounds = t->rounds ? t->rounds : 1;
	if (rounds > BENCH_ROUNDS_MAX)
		rounds = BENCH_ROUNDS_MAX;

	reps = bench_calibrate(fn, ctx, t);
	if (!reps) {
		bench_warn("cannot reach %llu ns within %llu repetitions; "
			   "not measuring",
			   (unsigned long long)t->target_ns,
			   (unsigned long long)t->max_reps);
		return -1;
	}

	/* One batch thrown away: caches, branch predictors, and the governor
	 * all need to have seen this loop before the first kept number.
	 */
	bench_one(fn, ctx, reps, &c);

	for (unsigned int i = 0; i < rounds; i++) {
		bench_one(fn, ctx, reps, &c);

		ns[i] = (double)c.ns / (double)reps;

		/*
		 * The counters are all or nothing per batch.  Keeping a partial
		 * set would mean averaging some rounds that were counted with
		 * some that were not, so a batch whose counters were discarded
		 * contributes wall time only.
		 */
		if (c.cycles) {
			cyc[counted] = (double)c.cycles / (double)reps;
			insn[counted] = (double)c.insns / (double)reps;
			counted++;
		}
	}

	out->reps = reps;
	out->rounds = rounds;

	out->ns_min = bench_min(ns, rounds);
	out->ns_sd = bench_stddev(ns, rounds);
	out->ns_med = bench_median(ns, rounds);

	/* Every round counted, or none of them: a mixture is not a measurement.
	 */
	out->have_pmu = counted == rounds && counted > 0;
	if (out->have_pmu) {
		out->cyc_min = bench_min(cyc, counted);
		out->cyc_sd = bench_stddev(cyc, counted);
		out->insn_min = bench_min(insn, counted);
		out->cyc_med = bench_median(cyc, counted);
	} else if (counted) {
		bench_info("counters missed %u of %u rounds; wall time only",
			   rounds - counted, rounds);
	}

	return 0;
}

void bench_publish(struct bench_report *rep, const char *suite,
		   const char *tcase, const char *variant,
		   const struct bench_stats *st)
{
	bench_add(rep, suite, tcase, variant, "ns.min", "ns",
		  BENCH_LOWER_BETTER, st->ns_min);
	bench_add(rep, suite, tcase, variant, "ns.med", "ns",
		  BENCH_LOWER_BETTER, st->ns_med);
	bench_add(rep, suite, tcase, variant, "ns.sd", "ns",
		  BENCH_LOWER_BETTER, st->ns_sd);

	if (!st->have_pmu)
		return;

	bench_add(rep, suite, tcase, variant, "cycles.min", "cyc",
		  BENCH_LOWER_BETTER, st->cyc_min);
	bench_add(rep, suite, tcase, variant, "cycles.med", "cyc",
		  BENCH_LOWER_BETTER, st->cyc_med);
	bench_add(rep, suite, tcase, variant, "cycles.sd", "cyc",
		  BENCH_LOWER_BETTER, st->cyc_sd);
	/*
	 * The instruction counter is the one member of the group that is
	 * allowed to be absent; a zero here means it never opened, not that
	 * the kernel retired nothing.
	 */
	if (st->insn_min > 0.0)
		bench_add(rep, suite, tcase, variant, "insns.min", "insn",
			  BENCH_LOWER_BETTER, st->insn_min);
}
