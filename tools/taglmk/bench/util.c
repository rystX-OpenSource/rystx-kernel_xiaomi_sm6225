// SPDX-License-Identifier: GPL-2.0
/*
 * taglmk_bench - diagnostics, allocation, and the sysfs conversation.
 *
 * The device side of this benchmark is a series of small text writes to sysfs
 * attributes, and every one of them can be refused: an algorithm the kernel was
 * not built with, a disksize on a device that is already initialised, a
 * recompression on a device nobody has written to.  A refusal arrives as a
 * short write or an errno on close, so both are checked here, once, rather
 * than at forty call sites.
 *
 * Copyright (C) 2026 iDeadXS <datarafi43@gmail.com>
 */
#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "bench.h"

bool bench_verbose;

static void bench_vmsg(FILE *out, const char *tag, const char *fmt,
		       va_list ap)
{
	if (tag)
		fprintf(out, "%s: ", tag);
	vfprintf(out, fmt, ap);
	fputc('\n', out);
}

void bench_info(const char *fmt, ...)
{
	va_list ap;

	if (!bench_verbose)
		return;

	va_start(ap, fmt);
	bench_vmsg(stderr, NULL, fmt, ap);
	va_end(ap);
}

void bench_warn(const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	bench_vmsg(stderr, "warning", fmt, ap);
	va_end(ap);
}

void bench_err(const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	bench_vmsg(stderr, "error", fmt, ap);
	va_end(ap);
}

/*
 * Allocation failure is not a case worth threading through a benchmark: there
 * is no partial answer to fall back on and nothing to clean up that the kernel
 * will not clean up on exit.  Say so and stop.
 */
void *bench_xalloc(size_t n)
{
	void *p = calloc(1, n ? n : 1);

	if (!p) {
		bench_err("out of memory allocating %zu bytes", n);
		exit(1);
	}

	return p;
}

void *bench_xalloc_aligned(size_t align, size_t n)
{
	void *p = NULL;

	if (posix_memalign(&p, align, n ? n : align) || !p) {
		bench_err("out of memory allocating %zu bytes aligned to %zu",
			  n, align);
		exit(1);
	}

	memset(p, 0, n ? n : align);

	return p;
}

int bench_read_file(const char *path, char *buf, size_t len)
{
	ssize_t got;
	int fd;

	if (len < 2) {
		errno = EINVAL;
		return -1;
	}

	fd = open(path, O_RDONLY | O_CLOEXEC);
	if (fd < 0)
		return -1;

	got = read(fd, buf, len - 1);
	if (got < 0) {
		int err = errno;

		close(fd);
		errno = err;
		return -1;
	}
	close(fd);

	buf[got] = '\0';

	/* sysfs hands back one trailing newline; nothing below wants it. */
	while (got > 0 && (buf[got - 1] == '\n' || buf[got - 1] == '\r'))
		buf[--got] = '\0';

	return 0;
}

/*
 * One write, and it has to be accepted whole.  A sysfs store either consumes
 * the entire buffer or returns an error; a short count therefore means the
 * value was rejected, and treating that as success is how a benchmark ends up
 * reporting a configuration it never actually applied.
 */
int bench_write_file(const char *path, const char *value)
{
	size_t len = strlen(value);
	ssize_t put;
	int fd, err;

	fd = open(path, O_WRONLY | O_CLOEXEC);
	if (fd < 0)
		return -1;

	put = write(fd, value, len);
	err = errno;

	if (close(fd) && put >= 0) {
		errno = errno ? errno : EIO;
		return -1;
	}

	if (put < 0) {
		errno = err;
		return -1;
	}

	if ((size_t)put != len) {
		errno = EIO;
		return -1;
	}

	return 0;
}

int bench_read_u64(const char *path, uint64_t *out)
{
	char buf[64];

	if (bench_read_file(path, buf, sizeof(buf)))
		return -1;

	if (!bench_parse_u64(buf, out)) {
		errno = EINVAL;
		return -1;
	}

	return 0;
}

bool bench_path_exists(const char *path)
{
	return access(path, F_OK) == 0;
}

bool bench_parse_u64(const char *s, uint64_t *out)
{
	unsigned long long v;
	char *end;

	if (!s || !*s)
		return false;

	/*
	 * strtoull happily accepts a leading minus and wraps it, which would
	 * turn a corrupt field into a plausible enormous count.  Refuse the
	 * sign outright, and refuse leading space so a field can never span
	 * what was meant to be two.
	 */
	if (*s == '-' || *s == '+' || *s == ' ' || *s == '\t')
		return false;

	errno = 0;
	v = strtoull(s, &end, 10);
	if (errno || end == s || *end)
		return false;

	*out = (uint64_t)v;

	return true;
}

bool bench_parse_double(const char *s, double *out)
{
	double v;
	char *end;

	if (!s || !*s || *s == ' ' || *s == '\t')
		return false;

	errno = 0;
	v = strtod(s, &end);
	if (errno || end == s || *end)
		return false;

	/* NaN and the infinities parse cleanly and score nonsensically. */
	if (!(v == v) || v > 1e300 || v < -1e300)
		return false;

	*out = v;

	return true;
}

bool bench_name_ok(const char *s)
{
	size_t i;

	if (!s || !*s)
		return false;

	for (i = 0; s[i]; i++) {
		unsigned char c = (unsigned char)s[i];

		if (i >= BENCH_NAME_MAX - 1)
			return false;
		if (c <= ' ' || c >= 0x7f)
			return false;
	}

	return true;
}
