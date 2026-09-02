// SPDX-License-Identifier: GPL-2.0
/*
 * taglmk_bench - the result table, the file it is saved to, and the comparison.
 *
 * The saved file is the whole point of the tool, so it is treated as an
 * interface rather than as a dump.  It is line oriented, one record per line,
 * every field bounded, and it carries the format version first so a reader can
 * refuse a file it does not understand instead of misreading it.
 *
 *   taglmk-bench 1
 *   M kernel 4.19.325-rystx
 *   R cpu regress v1 cycles.min - cyc 214.5
 *
 * A comparison is where a benchmark can most easily lie, so three rules hold.
 * A metric carries its own direction, so a reader that has never heard of it
 * still scores it the right way round.  A case that failed verification on
 * either side is reported as failed and never scored.  And a run whose binding
 * metadata differs from the baseline's - a different corpus, a different set of
 * compressors - is refused outright rather than scored across the difference.
 *
 * Copyright (C) 2026 iDeadXS <datarafi43@gmail.com>
 */
#define _GNU_SOURCE

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench.h"

/* The metric every case publishes to say whether its output was correct. */
#define BENCH_VERIFY_METRIC	"verify"

void bench_report_init(struct bench_report *rep)
{
	memset(rep, 0, sizeof(*rep));
}

bool bench_add_meta(struct bench_report *rep, const char *key,
		    const char *fmt, ...)
{
	struct bench_meta *m;
	va_list ap;
	int n;

	if (!bench_name_ok(key)) {
		bench_warn("dropping metadata with an unusable key");
		return false;
	}

	if (rep->nr_meta >= BENCH_META_MAX) {
		bench_warn("metadata table full, dropping '%s'", key);
		return false;
	}

	m = &rep->meta[rep->nr_meta];

	va_start(ap, fmt);
	n = vsnprintf(m->value, sizeof(m->value), fmt, ap);
	va_end(ap);

	if (n < 0 || (size_t)n >= sizeof(m->value)) {
		bench_warn("metadata '%s' too long, dropped", key);
		return false;
	}

	/*
	 * The value is free text but it still has to survive a line based
	 * format, so anything that could end a line early or start a new field
	 * where none was meant becomes a space.
	 */
	for (char *p = m->value; *p; p++) {
		unsigned char c = (unsigned char)*p;

		if (c < ' ' || c >= 0x7f)
			*p = ' ';
	}

	snprintf(m->key, sizeof(m->key), "%s", key);
	rep->nr_meta++;

	return true;
}

bool bench_add(struct bench_report *rep, const char *suite, const char *tcase,
	       const char *variant, const char *metric, const char *unit,
	       enum bench_dir dir, double value)
{
	struct bench_result *r;

	if (!bench_name_ok(suite) || !bench_name_ok(tcase) ||
	    !bench_name_ok(variant) || !bench_name_ok(metric) ||
	    !bench_name_ok(unit)) {
		bench_warn("dropping a result with an unusable identifier");
		return false;
	}

	if (!isfinite(value)) {
		bench_warn("dropping %s/%s/%s/%s: value is not finite",
			   suite, tcase, variant, metric);
		return false;
	}

	if (rep->nr_results >= BENCH_RESULTS_MAX) {
		bench_warn("result table full, dropping %s/%s/%s/%s",
			   suite, tcase, variant, metric);
		return false;
	}

	r = &rep->results[rep->nr_results++];
	snprintf(r->suite, sizeof(r->suite), "%s", suite);
	snprintf(r->tcase, sizeof(r->tcase), "%s", tcase);
	snprintf(r->variant, sizeof(r->variant), "%s", variant);
	snprintf(r->metric, sizeof(r->metric), "%s", metric);
	snprintf(r->unit, sizeof(r->unit), "%s", unit);
	r->dir = dir;
	r->value = value;

	return true;
}

const char *bench_get_meta(const struct bench_report *rep, const char *key)
{
	for (unsigned int i = 0; i < rep->nr_meta; i++)
		if (!strcmp(rep->meta[i].key, key))
			return rep->meta[i].value;

	return NULL;
}

const struct bench_result *bench_find(const struct bench_report *rep,
				      const char *suite, const char *tcase,
				      const char *variant, const char *metric)
{
	for (unsigned int i = 0; i < rep->nr_results; i++) {
		const struct bench_result *r = &rep->results[i];

		if (!strcmp(r->suite, suite) && !strcmp(r->tcase, tcase) &&
		    !strcmp(r->variant, variant) && !strcmp(r->metric, metric))
			return r;
	}

	return NULL;
}

/*
 * What has to match before two runs may be scored against each other.  These
 * are the things that change what the numbers mean rather than how good they
 * are: the width of a page, the shape of the data fed to the compressor, which
 * compressors were in the ladder.  Everything else - kernel release, governor,
 * which core, the time of day - is recorded and reported but never blocking,
 * because a difference in those is very often the thing being measured.
 */
bool bench_meta_is_binding(const char *key)
{
	static const char * const binding[] = {
		/* Global: the machine the numbers describe. */
		"abi",
		"pagesize",
		"seed",

		/* cpu: the input lengths the kernels were given. */
		"cpu.sizes",

		/* zram: the ladder, the device, and the traffic. */
		"zram.comp",
		"zram.disksize",
		"zram.fill",
		"zram.io",
		"zram.pages",
		"zram.corpus",
		"zram.direct",
	};

	for (size_t i = 0; i < sizeof(binding) / sizeof(binding[0]); i++)
		if (!strcmp(key, binding[i]))
			return true;

	return false;
}

/*
 * The suite a metadata key belongs to: the part before the first dot, or ""
 * for a global key.  Only used to ask whether a key's suite ran at all.
 */
static void bench_meta_suite(const char *key, char *buf, size_t len)
{
	const char *dot = strchr(key, '.');
	size_t n;

	if (!dot) {
		buf[0] = '\0';
		return;
	}

	n = (size_t)(dot - key);
	if (n >= len)
		n = len - 1;

	memcpy(buf, key, n);
	buf[n] = '\0';
}

/* Whether @rep holds any result from @suite. */
static bool bench_has_suite(const struct bench_report *rep, const char *suite)
{
	for (unsigned int i = 0; i < rep->nr_results; i++)
		if (!strcmp(rep->results[i].suite, suite))
			return true;

	return false;
}

/*
 * A binding key the baseline recorded and this run did not is only a conflict
 * if this run actually measured that key's suite.  Running just --suite=cpu
 * against a baseline that also holds zram numbers is a narrower run, not an
 * incomparable one: the cpu cases still mean the same thing, and the zram
 * settings the baseline names simply were not exercised.
 */
static bool bench_absence_blocks(const struct bench_report *now,
				 const char *key)
{
	char suite[BENCH_NAME_MAX];

	bench_meta_suite(key, suite, sizeof(suite));

	/* A global key must always be present; its absence is a real gap. */
	if (!suite[0])
		return true;

	return bench_has_suite(now, suite);
}

int bench_report_save(const struct bench_report *rep, const char *path)
{
	FILE *f;

	f = fopen(path, "we");
	if (!f) {
		bench_err("cannot write '%s': %s", path, strerror(errno));
		return -1;
	}

	fprintf(f, "%s %d\n", BENCH_FORMAT_NAME, BENCH_FORMAT_VERSION);

	for (unsigned int i = 0; i < rep->nr_meta; i++)
		fprintf(f, "M %s %s\n", rep->meta[i].key, rep->meta[i].value);

	for (unsigned int i = 0; i < rep->nr_results; i++) {
		const struct bench_result *r = &rep->results[i];

		/*
		 * Seventeen significant digits is what it takes for a double to
		 * survive being printed and read back unchanged, which is what
		 * keeps "compared against itself" showing exactly zero.
		 */
		fprintf(f, "R %s %s %s %s %c %s %.17g\n",
			r->suite, r->tcase, r->variant, r->metric,
			r->dir == BENCH_HIGHER_BETTER ? '+' : '-',
			r->unit, r->value);
	}

	if (fflush(f) || ferror(f)) {
		bench_err("cannot write '%s': %s", path, strerror(errno));
		fclose(f);
		return -1;
	}

	if (fclose(f)) {
		bench_err("cannot close '%s': %s", path, strerror(errno));
		return -1;
	}

	return 0;
}

/* Split @line on single spaces into at most @max fields.  No allocation, no
 * copying: the line is punctuated in place and the pieces pointed at.
 */
static unsigned int bench_split(char *line, char **field, unsigned int max)
{
	unsigned int n = 0;

	while (*line && n < max) {
		while (*line == ' ')
			line++;
		if (!*line)
			break;

		field[n++] = line;

		while (*line && *line != ' ')
			line++;
		if (*line)
			*line++ = '\0';
	}

	return n;
}

int bench_report_load(struct bench_report *rep, const char *path)
{
	char line[BENCH_LINE_MAX];
	unsigned long lineno = 0;
	bool have_header = false;
	FILE *f;

	bench_report_init(rep);

	f = fopen(path, "re");
	if (!f) {
		bench_err("cannot read '%s': %s", path, strerror(errno));
		return -1;
	}

	while (fgets(line, sizeof(line), f)) {
		char *field[8];
		unsigned int nr;
		size_t len;

		lineno++;

		len = strlen(line);
		if (len == sizeof(line) - 1 && line[len - 1] != '\n') {
			bench_err("%s:%lu: line too long", path, lineno);
			goto bad;
		}
		while (len && (line[len - 1] == '\n' || line[len - 1] == '\r'))
			line[--len] = '\0';

		if (!len || line[0] == '#')
			continue;

		if (!have_header) {
			nr = bench_split(line, field, 8);
			if (nr != 2 || strcmp(field[0], BENCH_FORMAT_NAME)) {
				bench_err("%s: not a %s file", path,
					  BENCH_FORMAT_NAME);
				goto bad;
			}
			if (strcmp(field[1], "1")) {
				bench_err("%s: format version '%s', this "
					  "build understands %d", path,
					  field[1], BENCH_FORMAT_VERSION);
				goto bad;
			}
			have_header = true;
			continue;
		}

		if (line[0] == 'M' && line[1] == ' ') {
			char *key = line + 2;
			char *val;

			while (*key == ' ')
				key++;

			val = strchr(key, ' ');
			if (!val) {
				bench_err("%s:%lu: metadata has no value",
					  path, lineno);
				goto bad;
			}
			*val++ = '\0';
			while (*val == ' ')
				val++;

			if (!bench_add_meta(rep, key, "%s", val)) {
				bench_err("%s:%lu: unusable metadata", path,
					  lineno);
				goto bad;
			}
			continue;
		}

		if (line[0] == 'R' && line[1] == ' ') {
			enum bench_dir dir;
			double value;

			nr = bench_split(line, field, 8);
			if (nr != 8) {
				bench_err("%s:%lu: a result needs seven "
					  "fields, found %u", path, lineno,
					  nr ? nr - 1 : 0);
				goto bad;
			}

			if (!strcmp(field[5], "+")) {
				dir = BENCH_HIGHER_BETTER;
			} else if (!strcmp(field[5], "-")) {
				dir = BENCH_LOWER_BETTER;
			} else {
				bench_err("%s:%lu: direction must be + or -",
					  path, lineno);
				goto bad;
			}

			if (!bench_parse_double(field[7], &value)) {
				bench_err("%s:%lu: '%s' is not a number",
					  path, lineno, field[7]);
				goto bad;
			}

			if (!bench_add(rep, field[1], field[2], field[3],
				       field[4], field[6], dir, value)) {
				bench_err("%s:%lu: unusable result", path,
					  lineno);
				goto bad;
			}
			continue;
		}

		bench_err("%s:%lu: unrecognised record", path, lineno);
		goto bad;
	}

	if (ferror(f)) {
		bench_err("%s: read failed: %s", path, strerror(errno));
		goto bad;
	}

	if (!have_header) {
		bench_err("%s: empty, or not a %s file", path,
			  BENCH_FORMAT_NAME);
		goto bad;
	}

	fclose(f);
	bench_info("loaded %u results and %u facts from %s", rep->nr_results,
		   rep->nr_meta, path);

	return 0;

bad:
	fclose(f);
	bench_report_init(rep);

	return -1;
}

/*
 * A case is only ever reported once, and only from the row that says whether it
 * worked.  Anything a failed case measured is arithmetic on a wrong answer.
 */
static bool bench_case_ok(const struct bench_report *rep, const char *suite,
			  const char *tcase, const char *variant)
{
	const struct bench_result *v;

	v = bench_find(rep, suite, tcase, variant, BENCH_VERIFY_METRIC);

	/* No verification row at all means the case does not claim one. */
	return !v || v->value != 0.0;
}

static void bench_print_group(const struct bench_report *rep,
			      const char *metric, const char *heading)
{
	bool printed = false;

	for (unsigned int i = 0; i < rep->nr_results; i++) {
		const struct bench_result *r = &rep->results[i];

		if (strcmp(r->metric, metric))
			continue;

		if (!printed) {
			printf("\n%s\n", heading);
			printed = true;
		}

		if (!bench_case_ok(rep, r->suite, r->tcase, r->variant)) {
			printf("  %-10s %-14s %-10s   FAILED VERIFICATION\n",
			       r->suite, r->tcase, r->variant);
			continue;
		}

		printf("  %-10s %-14s %-10s %14.2f %s\n", r->suite, r->tcase,
		       r->variant, r->value, r->unit);
	}
}

void bench_report_print(const struct bench_report *rep, bool verbose)
{
	printf("\n== run ==\n");
	for (unsigned int i = 0; i < rep->nr_meta; i++)
		printf("  %-18s %s\n", rep->meta[i].key, rep->meta[i].value);

	bench_print_group(rep, "cycles.min",
			  "== cycles per call (lower is better) ==");
	bench_print_group(rep, "ns.min",
			  "== nanoseconds per call (lower is better) ==");
	bench_print_group(rep, "write.mib_s", "== zram write throughput ==");
	bench_print_group(rep, "read.mib_s", "== zram read throughput ==");
	bench_print_group(rep, "ratio.pct", "== zram compression ratio ==");

	if (!verbose)
		return;

	printf("\n== every metric ==\n");
	for (unsigned int i = 0; i < rep->nr_results; i++) {
		const struct bench_result *r = &rep->results[i];

		printf("  %-8s %-14s %-10s %-16s %14.4f %s\n", r->suite,
		       r->tcase, r->variant, r->metric, r->value, r->unit);
	}
}

/*
 * Percentage change from @from to @to, signed so that positive always means
 * better whichever way the metric runs.  A zero baseline has no percentage, and
 * saying so is the only honest answer.
 */
static bool bench_delta_pct(double from, double to, enum bench_dir dir,
			    double *out)
{
	double pct;

	if (from == 0.0)
		return false;

	pct = (to - from) / (from < 0 ? -from : from) * 100.0;

	*out = dir == BENCH_HIGHER_BETTER ? pct : -pct;

	return isfinite(*out);
}

/*
 * Variants of one case, in the order they were measured, each scored against
 * the first that verified and against the one before it.
 *
 * This is where the headline number lives, and it is a within-run comparison
 * rather than a before-and-after one.  The cpu suite measures the scalar twin,
 * the old NEON kernel and the new one in the same process, on the same core,
 * against the same inputs, so "vs prev" on the v1 row is the change from v0 to
 * v1 with nothing else moved.  Taking that from two separate runs would fold in
 * every difference between the two boots as well.
 *
 * Metrics that describe the spread rather than the result are left out: a
 * standard deviation that got 40% better is not an improvement anybody claimed.
 */
#define BENCH_LADDER_MAX	16
#define BENCH_LADDER_FMT	"  %-6s %-10s %-16s %-7s %11.2f %9s %9s\n"
#define BENCH_LADDER_HDR	"  %-6s %-10s %-16s %-7s %11s %9s %9s\n"

static bool bench_ladder_metric(const char *metric)
{
	size_t n = strlen(metric);

	if (!strcmp(metric, BENCH_VERIFY_METRIC))
		return false;

	if (n > 3 && !strcmp(metric + n - 3, ".sd"))
		return false;

	if (n > 4 && !strcmp(metric + n - 4, ".med"))
		return false;

	return true;
}

/* Whether this suite, case and metric was already laid out from an earlier row.
 */
static bool bench_ladder_done(const struct bench_report *rep, unsigned int upto,
			      const struct bench_result *r)
{
	for (unsigned int i = 0; i < upto; i++) {
		const struct bench_result *p = &rep->results[i];

		if (!strcmp(p->suite, r->suite) &&
		    !strcmp(p->tcase, r->tcase) &&
		    !strcmp(p->metric, r->metric))
			return true;
	}

	return false;
}

/*
 * One row.  Returns whether it verified, so the caller can keep "the one before
 * it" pointing at the last variant whose number means anything.
 */
static bool bench_ladder_row(const struct bench_report *rep,
			     const struct bench_result *r,
			     const struct bench_result *ref,
			     const struct bench_result *prev)
{
	double pct;
	char a[16], b[16];

	if (!bench_case_ok(rep, r->suite, r->tcase, r->variant)) {
		printf("  %-6s %-10s %-16s %-7s   FAILED VERIFICATION\n",
		       r->suite, r->tcase, r->metric, r->variant);
		return false;
	}

	if (ref == r)
		snprintf(a, sizeof(a), "%s", "ref");
	else if (bench_delta_pct(ref->value, r->value, r->dir, &pct))
		snprintf(a, sizeof(a), "%+.2f%%", pct);
	else
		snprintf(a, sizeof(a), "%s", "n/a");

	if (prev == r)
		snprintf(b, sizeof(b), "%s", "-");
	else if (bench_delta_pct(prev->value, r->value, r->dir, &pct))
		snprintf(b, sizeof(b), "%+.2f%%", pct);
	else
		snprintf(b, sizeof(b), "%s", "n/a");

	printf(BENCH_LADDER_FMT, r->suite, r->tcase, r->metric, r->variant,
	       r->value, a, b);

	return true;
}

void bench_report_ladder(const struct bench_report *rep)
{
	bool printed = false;

	for (unsigned int i = 0; i < rep->nr_results; i++) {
		const struct bench_result *r = &rep->results[i];
		const struct bench_result *set[BENCH_LADDER_MAX];
		const struct bench_result *ref = NULL;
		const struct bench_result *prev;
		unsigned int nr = 0;

		if (!bench_ladder_metric(r->metric))
			continue;

		if (bench_ladder_done(rep, i, r))
			continue;

		for (unsigned int j = i; j < rep->nr_results; j++) {
			const struct bench_result *q = &rep->results[j];

			if (strcmp(q->suite, r->suite) ||
			    strcmp(q->tcase, r->tcase) ||
			    strcmp(q->metric, r->metric))
				continue;

			if (nr == BENCH_LADDER_MAX) {
				bench_warn("more than %d variants of %s %s "
					   "%s; the rest are in the saved "
					   "file only", BENCH_LADDER_MAX,
					   r->suite, r->tcase, r->metric);
				break;
			}

			set[nr++] = q;
		}

		/* A case with one variant compares against nothing. */
		if (nr < 2)
			continue;

		/*
		 * The reference is the first variant that verified.  Scoring
		 * against one that did not would put a wrong answer in the
		 * denominator of every row below it.
		 */
		for (unsigned int k = 0; k < nr && !ref; k++)
			if (bench_case_ok(rep, set[k]->suite, set[k]->tcase,
					  set[k]->variant))
				ref = set[k];

		if (!printed) {
			printf("\n== variants of one case, same run "
			       "(positive is better) ==\n");
			printf(BENCH_LADDER_HDR, "suite", "case", "metric",
			       "variant", "value", "vs ref", "vs prev");
			printed = true;
		}

		if (!ref) {
			printf("  %-6s %-10s %-16s   every variant failed "
			       "verification\n", r->suite, r->tcase,
			       r->metric);
			continue;
		}

		prev = ref;
		for (unsigned int k = 0; k < nr; k++)
			if (bench_ladder_row(rep, set[k], ref, prev))
				prev = set[k];
	}
}

void bench_report_compare(const struct bench_report *base,
			  const struct bench_report *now)
{
	unsigned int scored = 0, failed = 0, missing = 0, added = 0;
	unsigned int blocking = 0, unmeasured = 0;

	printf("\n== comparability ==\n");

	for (unsigned int i = 0; i < base->nr_meta; i++) {
		const char *key = base->meta[i].key;
		const char *was = base->meta[i].value;
		const char *is = bench_get_meta(now, key);
		bool bind = bench_meta_is_binding(key);

		if (is && !strcmp(is, was))
			continue;

		/*
		 * Absent because that suite was not run this time.  Reported so
		 * the reader knows the baseline is wider, but not blocking: the
		 * cases that did run are still comparable.
		 */
		if (!is && !bench_absence_blocks(now, key)) {
			if (bind || bench_verbose)
				printf("  not run %-14s baseline '%s'\n", key,
				       was);
			unmeasured++;
			continue;
		}

		if (bind) {
			printf("  BLOCKING %-14s baseline '%s', "
			       "this run '%s'\n",
			       key, was, is ? is : "(absent)");
			blocking++;
		} else if (bench_verbose || !is) {
			printf("  differs  %-14s baseline '%s', "
			       "this run '%s'\n",
			       key, was, is ? is : "(absent)");
		}
	}

	if (blocking) {
		printf("\n  %u binding fact%s differ, so these runs do\n"
		       "  not measure the same thing and no comparison\n"
		       "  is printed.  Re-run with the baseline's\n"
		       "  settings, or start a new baseline.\n",
		       blocking, blocking == 1 ? "" : "s");
		return;
	}

	if (unmeasured)
		printf("  %u baseline fact%s not measured this run\n",
		       unmeasured, unmeasured == 1 ? "" : "s");

	printf("  nothing binding differs; the runs are comparable\n");

	printf("\n== change against baseline (positive is better) ==\n");
	printf("  %-8s %-14s %-10s %-14s %12s %12s %9s\n", "suite", "case",
	       "variant", "metric", "baseline", "this run", "change");

	for (unsigned int i = 0; i < now->nr_results; i++) {
		const struct bench_result *r = &now->results[i];
		const struct bench_result *b;
		double pct;

		if (!strcmp(r->metric, BENCH_VERIFY_METRIC))
			continue;

		b = bench_find(base, r->suite, r->tcase, r->variant,
			       r->metric);
		if (!b) {
			added++;
			continue;
		}

		/*
		 * Both sides have to have been right.  A case that failed
		 * verification in either run is named and skipped: the whole
		 * reason to check the output is so that a wrong kernel cannot
		 * come back as a fast one.
		 */
		if (!bench_case_ok(now, r->suite, r->tcase, r->variant) ||
		    !bench_case_ok(base, r->suite, r->tcase, r->variant)) {
			failed++;
			continue;
		}

		/*
		 * Directions must agree.  If they do not, one of the two files
		 * was written by a different idea of what the metric means, and
		 * scoring it would invent a result.
		 */
		if (b->dir != r->dir) {
			printf("  %-8s %-14s %-10s %-14s   direction "
			       "disagrees, not scored\n",
			       r->suite, r->tcase, r->variant,
			       r->metric);
			continue;
		}

		if (!bench_delta_pct(b->value, r->value, r->dir, &pct)) {
			printf("  %-8s %-14s %-10s %-14s %12.2f %12.2f %9s\n",
			       r->suite, r->tcase, r->variant, r->metric,
			       b->value, r->value, "n/a");
			continue;
		}

		printf("  %-8s %-14s %-10s %-14s %12.2f %12.2f %+8.2f%%\n",
		       r->suite, r->tcase, r->variant, r->metric, b->value,
		       r->value, pct);
		scored++;
	}

	for (unsigned int i = 0; i < base->nr_results; i++) {
		const struct bench_result *b = &base->results[i];

		if (!strcmp(b->metric, BENCH_VERIFY_METRIC))
			continue;

		if (!bench_find(now, b->suite, b->tcase, b->variant,
				b->metric))
			missing++;
	}

	printf("\n  %u metric%s scored", scored, scored == 1 ? "" : "s");
	if (failed)
		printf(", %u skipped for failing verification", failed);
	if (missing)
		printf(", %u in the baseline not measured this time", missing);
	if (added)
		printf(", %u new", added);
	printf("\n");
}
