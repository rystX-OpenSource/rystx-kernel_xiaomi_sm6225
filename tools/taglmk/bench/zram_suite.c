// SPDX-License-Identifier: GPL-2.0
/*
 * taglmk_bench - the zram suite.
 *
 * The ladder is driven through the block device rather than through swap,
 * because swap decides for itself what to write and when, and a number that
 * depends on the kernel's own reclaim decisions cannot be attributed to a
 * change in the compressor path.  A scratch device, a fixed corpus and O_DIRECT
 * give a fill that is the same work every time it is asked for.
 *
 * Immediate recompression is a write time decision - the store path walks the
 * ladder as far as the current depth allows before it commits anything - so a
 * level is only measurable by filling the device again with that level in
 * force. Each level therefore gets its own fill, and levels that come out at
 * the same effective depth are measured once and reported once: with two
 * algorithms configured min(level + 1, nr_comps) is two whichever of 1, 2 or 3
 * the sysctl says, and printing one measurement three times under three names
 * would be an invented result.
 *
 * The two sweep cases are the userspace visible form of what the recompression
 * sweep change was about.  A sweep narrowed to type=huge can only select slots
 * that are not already flagged incompressible, and once the write time depth
 * has reached the end of the ladder every huge slot carries that flag; the scan
 * then selects nothing and still walks every slot on the disk.  The unnarrowed
 * sweep over the same device state is the comparison, and the device is
 * refilled between the two so that both act on identical state.
 *
 * Safety.  By default the suite adds its own device and takes it away again on
 * every exit path.  An existing device may be named instead, but never one that
 * /proc/swaps is using, and never one that is already initialised unless
 * --force says so.  The fill is clamped to what MemAvailable can absorb even if
 * nothing compresses at all, and mem_limit is set as a backstop so that zram
 * refuses a write rather than pushing the machine into OOM.  A signal sets a
 * flag the IO loops check between requests, so teardown always runs on the main
 * path; a SIGKILL is the one case that leaves the scratch device behind, which
 * is why its name is printed as soon as it exists.
 *
 * Copyright (C) 2026 iDeadXS <datarafi43@gmail.com>
 */
#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/sysmacros.h>

#include "bench.h"

#define BENCH_ZRAM_SUITE	"zram"

#define BENCH_ZRAM_CONTROL	"/sys/class/zram-control"
#define BENCH_ZRAM_IR_SYSCTL	"/proc/sys/vm/zram_recomp_immediate"

/* ZRAM_MAX_COMPS in the tree: one primary and three recompression rungs. */
#define BENCH_ZRAM_COMPS_MAX	4u

/*
 * The sysctl is a u8 clamped to [1, 3] by the kernel, so there is no level zero
 * to ask for and no point offering one.
 */
#define BENCH_ZRAM_LEVEL_MIN	1u
#define BENCH_ZRAM_LEVEL_MAX	3u

#define BENCH_ZRAM_PATH		128
#define BENCH_ZRAM_RING_PAGES	2048u	/* distinct images: 8MiB at 4K */
#define BENCH_ZRAM_IO_PAGES_MAX	512u
#define BENCH_ZRAM_IO_ROUNDS	3u
#define BENCH_ZRAM_SWEEP_PAGES	4096u
#define BENCH_ZRAM_HEADROOM_KB	(256u * 1024u)
#define BENCH_ZRAM_BUSY_TRIES	20u
#define BENCH_ZRAM_BUSY_MS	50u

/*
 * How long to wait for the block device node to turn up.  Nothing in the
 * kernel creates it: hot_add publishes a uevent and returns, and on Android it
 * is ueventd that answers, so the node appears some milliseconds after the
 * write completes.  Checking once, immediately, is a race that the tool loses.
 */
#define BENCH_ZRAM_NODE_TRIES	40u
#define BENCH_ZRAM_NODE_MS	50u
#define BENCH_ZRAM_MEMINFO_MAX	8192

/**
 * struct bench_comp - one rung of the configured ladder
 * @name: Backend name, exactly as the kernel registered it.
 * @level: Compression level, or -1 to leave the backend's own default alone.
 *	The two go to different attributes - the name to comp_algorithm or
 *	recomp_algorithm, the level to algorithm_params - because the kernel has
 *	no name:level syntax, however often it is written that way.
 */
struct bench_comp {
	char	name[BENCH_NAME_MAX];
	long	level;
};

/**
 * struct bench_zram - the device under test and everything held for it
 * @id: Device number.
 * @added: We created it, so we are the ones who take it away.
 * @sys: /sys/block/zramN.
 * @node: The block device node.
 * @node_made: We had to create @node ourselves, so we unlink it again.
 * @fd: Open handle, or -1.  Closed before every reset, because a device with an
 *	opener refuses to reset.
 * @probed: Whether the device was ever opened, so that @direct means anything.
 * @direct: Whether O_DIRECT was accepted.  When it was not, the read pass is
 *	measuring the page cache as much as the compressor, so it is recorded as
 *	a binding fact rather than quietly tolerated.
 * @comp: The ladder, in priority order.
 * @nr_comp: Rungs in @comp.
 * @page_size: What the running kernel uses.
 * @disksize: Bytes the device is configured for.
 * @ring_pages: Distinct page images held in @ring.
 * @fill_pages: Pages one pass writes, a multiple of @io_pages.
 * @io_pages: Pages per request.
 * @ring: The corpus, page aligned so that O_DIRECT will take it.
 * @io: Landing buffer for the read passes, page aligned for the same reason.
 * @mix: The corpus mixture, for the saved file.
 * @have_sysctl: Whether the immediate recompression knob is present.
 * @saved_level: What that knob said when we arrived, restored on the way out.
 * @multi_comp: Whether this kernel has the recompression attributes at all.
 */
struct bench_zram {
	int			id;
	bool			added;
	char			sys[BENCH_ZRAM_PATH];
	char			node[BENCH_ZRAM_PATH];
	bool			node_made;
	int			fd;
	bool			probed;
	bool			direct;

	struct bench_comp	comp[BENCH_ZRAM_COMPS_MAX];
	unsigned int		nr_comp;

	uint64_t		page_size;
	uint64_t		disksize;
	uint64_t		ring_pages;
	uint64_t		fill_pages;
	uint64_t		io_pages;
	unsigned char		*ring;
	unsigned char		*io;
	char			mix[BENCH_META_VALUE_MAX];

	bool			have_sysctl;
	uint64_t		saved_level;
	bool			multi_comp;
};

/**
 * struct bench_mm - the nine fields of mm_stat
 *
 * Read whole and parsed whole: a partial parse would silently give a ratio
 * built from one run's numerator and another's denominator.
 */
struct bench_mm {
	uint64_t	orig;
	uint64_t	compr;
	uint64_t	mem_used;
	uint64_t	mem_limit;
	uint64_t	mem_max;
	uint64_t	same;
	uint64_t	compacted;
	uint64_t	huge;
	uint64_t	huge_since;
};

/*
 * Set from a signal handler and read from the IO loops, which is the only thing
 * sig_atomic_t promises and the only thing wanted here.  Nothing is torn down
 * from the handler: teardown talks to sysfs and frees memory, neither of which
 * belongs in a signal context, so the handler raises a flag and the loops
 * return an error through the ordinary path.
 */
static volatile sig_atomic_t bench_zram_stop;

static void bench_zram_catch(int sig)
{
	(void)sig;
	bench_zram_stop = 1;
}

static void bench_msleep(unsigned int ms)
{
	struct timespec ts = {
		.tv_sec = ms / 1000,
		.tv_nsec = (long)(ms % 1000) * 1000000L,
	};

	nanosleep(&ts, NULL);
}

/* ----------------------------------------------------------- sysfs plumbing */

static void bench_zram_attr(const struct bench_zram *z, const char *attr,
			    char *buf, size_t len)
{
	snprintf(buf, len, "%s/%s", z->sys, attr);
}

static bool bench_zram_has(const struct bench_zram *z, const char *attr)
{
	char path[BENCH_ZRAM_PATH];

	bench_zram_attr(z, attr, path, sizeof(path));

	return bench_path_exists(path);
}

/* Write @value to one attribute, naming the attribute if it is refused. */
static int bench_zram_set(const struct bench_zram *z, const char *attr,
			  const char *value)
{
	char path[BENCH_ZRAM_PATH];

	bench_zram_attr(z, attr, path, sizeof(path));

	/*
	 * Drain first, so that anything the log holds afterwards belongs to
	 * this store and not to the one before it.  A store that refuses a
	 * value has already written down why by the time it returns, and the
	 * errno userspace gets does not carry that reasoning.
	 */
	bench_kmsg_drain();

	if (bench_write_file(path, value)) {
		bench_err("zram%d: %s = '%s' refused: %s", z->id, attr, value,
			  strerror(errno));
		bench_kmsg_dump();
		return -1;
	}

	bench_info("zram%d: %s = %s", z->id, attr, value);

	return 0;
}

/*
 * The same, retried while the answer is EBUSY.  reset and hot_remove both fail
 * that way while anything still holds the device open, and something briefly
 * does after a disksize write on a system with a udev.
 */
static int bench_zram_set_busy(const char *path, const char *value,
			       const char *what)
{
	unsigned int try;

	for (try = 0; try < BENCH_ZRAM_BUSY_TRIES; try++) {
		bench_kmsg_drain();

		if (!bench_write_file(path, value))
			return 0;

		if (errno != EBUSY) {
			bench_err("%s: %s", what, strerror(errno));
			bench_kmsg_dump();
			return -1;
		}

		bench_msleep(BENCH_ZRAM_BUSY_MS);
	}

	bench_err("%s: still busy after %ums", what,
		  BENCH_ZRAM_BUSY_TRIES * BENCH_ZRAM_BUSY_MS);
	bench_kmsg_dump();

	return -1;
}

static void bench_zram_close(struct bench_zram *z)
{
	if (z->fd >= 0) {
		close(z->fd);
		z->fd = -1;
	}
}

static int bench_zram_reset(const struct bench_zram *z)
{
	char path[BENCH_ZRAM_PATH];
	char what[64];

	bench_zram_attr(z, "reset", path, sizeof(path));
	snprintf(what, sizeof(what), "zram%d: reset", z->id);

	return bench_zram_set_busy(path, "1", what);
}

static int bench_zram_mm(const struct bench_zram *z, struct bench_mm *mm)
{
	char buf[256];
	char path[BENCH_ZRAM_PATH];

	bench_zram_attr(z, "mm_stat", path, sizeof(path));

	if (bench_read_file(path, buf, sizeof(buf))) {
		bench_err("zram%d: cannot read mm_stat: %s", z->id,
			  strerror(errno));
		return -1;
	}

	if (sscanf(buf,
		   "%" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64
		   " %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64,
		   &mm->orig, &mm->compr, &mm->mem_used, &mm->mem_limit,
		   &mm->mem_max, &mm->same, &mm->compacted, &mm->huge,
		   &mm->huge_since) != 9) {
		bench_err("zram%d: mm_stat has an unexpected shape: '%s'",
			  z->id, buf);
		return -1;
	}

	return 0;
}

/* ----------------------------------------------------------- device custody */

static int bench_zram_hot_add(void)
{
	char buf[32];
	uint64_t id;

	bench_kmsg_drain();

	if (bench_read_file(BENCH_ZRAM_CONTROL "/hot_add", buf, sizeof(buf))) {
		/* Held, because everything below reports on it in turn. */
		int err = errno;

		bench_err("cannot add a zram device: %s", strerror(err));
		if (err == EACCES || err == EPERM)
			bench_err("the zram suite has to run as root");
		else if (err == ENOENT)
			bench_err("this kernel has no zram, or it is a "
				  "module that is not loaded");
		bench_kmsg_dump();
		return -1;
	}

	if (!bench_parse_u64(buf, &id) || id > INT_MAX) {
		bench_err("zram-control answered '%s', which is not a "
			  "device id", buf);
		return -1;
	}

	return (int)id;
}

static void bench_zram_hot_remove(int id)
{
	char value[32];
	char what[64];

	snprintf(value, sizeof(value), "%d", id);
	snprintf(what, sizeof(what), "cannot remove zram%d", id);

	if (bench_zram_set_busy(BENCH_ZRAM_CONTROL "/hot_remove", value, what))
		bench_warn("zram%d is still present; remove it by hand: "
			   "echo %d > %s/hot_remove",
			   id, id, BENCH_ZRAM_CONTROL);
	else
		bench_info("removed zram%d", id);
}

/*
 * Whether swap is using this device.  Fails closed: a /proc/swaps that cannot
 * be read is treated as one that says yes, because the cost of being wrong in
 * the other direction is a device the system is swapping to being reset
 * underneath it.
 */
static bool bench_zram_in_swap(int id)
{
	char want[32];
	char line[512];
	bool hit = false;
	FILE *f;

	snprintf(want, sizeof(want), "zram%d", id);

	f = fopen("/proc/swaps", "re");
	if (!f) {
		bench_warn("cannot read /proc/swaps (%s), so zram%d is "
			   "assumed to be in use", strerror(errno), id);
		return true;
	}

	while (fgets(line, sizeof(line), f)) {
		char *base = line;
		char *p;

		while (*base == ' ' || *base == '\t')
			base++;

		for (p = base; *p && *p != ' ' && *p != '\t' && *p != '\n'; p++)
			;
		*p = '\0';

		p = strrchr(base, '/');
		if (p)
			base = p + 1;

		if (!strcmp(base, want)) {
			hit = true;
			break;
		}
	}

	fclose(f);

	return hit;
}

/* Where the node lives, in the order the platforms put it. */
static const char * const bench_zram_node_shapes[] = {
	"/dev/block/zram%d",
	"/dev/zram%d",
};

/* One sweep of the candidates; sets @node and returns true on a hit. */
static bool bench_zram_node_here(struct bench_zram *z)
{
	size_t n = sizeof(bench_zram_node_shapes) /
		   sizeof(bench_zram_node_shapes[0]);

	for (size_t i = 0; i < n; i++) {
		snprintf(z->node, sizeof(z->node),
			 bench_zram_node_shapes[i], z->id);
		if (bench_path_exists(z->node))
			return true;
	}

	z->node[0] = '\0';

	return false;
}

/*
 * Say how the lookup failed.  A node that is absent and a directory the process
 * may not search look identical through access(F_OK), and the difference
 * decides whether the answer is to wait, to create, or to fix the caller's
 * privileges - so print the errno for each candidate rather than a verdict.
 */
static void bench_zram_node_complain(const struct bench_zram *z)
{
	size_t n = sizeof(bench_zram_node_shapes) /
		   sizeof(bench_zram_node_shapes[0]);
	char path[BENCH_ZRAM_PATH];

	for (size_t i = 0; i < n; i++) {
		snprintf(path, sizeof(path), bench_zram_node_shapes[i], z->id);

		if (access(path, F_OK))
			bench_err("zram%d: %s: %s", z->id, path,
				  strerror(errno));
	}
}

/*
 * Last resort: make the node.  /sys/block/zramN/dev holds the numbers, so this
 * needs no guessing, and it is recorded so that teardown takes it away again.
 * Reached on a system with no ueventd at all - a recovery ramdisk, a chroot -
 * where waiting would only ever time out.
 */
static int bench_zram_make_node(struct bench_zram *z)
{
	char path[BENCH_ZRAM_PATH];
	char buf[64];
	unsigned long maj, min;
	char *end;

	bench_zram_attr(z, "dev", path, sizeof(path));
	if (bench_read_file(path, buf, sizeof(buf))) {
		bench_err("zram%d: cannot read %s: %s", z->id, path,
			  strerror(errno));
		return -1;
	}

	errno = 0;
	maj = strtoul(buf, &end, 10);
	if (errno || *end != ':')
		goto malformed;

	errno = 0;
	min = strtoul(end + 1, &end, 10);
	if (errno || (*end && *end != '\n'))
		goto malformed;

	/* The kernel's own split: twelve bits of major, twenty of minor. */
	if (maj > 0xfff || min > 0xfffff)
		goto malformed;

	snprintf(z->node, sizeof(z->node), bench_zram_node_shapes[1], z->id);

	if (mknod(z->node, S_IFBLK | 0600, makedev((unsigned int)maj,
						   (unsigned int)min))) {
		bench_err("zram%d: cannot create %s: %s", z->id, z->node,
			  strerror(errno));
		z->node[0] = '\0';
		return -1;
	}

	z->node_made = true;
	bench_info("created %s for %lu:%lu", z->node, maj, min);

	return 0;

malformed:
	bench_err("zram%d: %s answered '%s', which is not major:minor", z->id,
		  path, buf);

	return -1;
}

/*
 * Find the node, waiting for it if it is merely late.  hot_add returns as soon
 * as the disk is registered, so on Android the node is still on its way: the
 * uevent has to reach ueventd and be acted on.  The wait is what makes a
 * freshly added device usable at all.
 */
static int bench_zram_find_node(struct bench_zram *z)
{
	unsigned int try;

	for (try = 0; try < BENCH_ZRAM_NODE_TRIES; try++) {
		if (bench_zram_node_here(z)) {
			if (try)
				bench_info("%s appeared after %ums", z->node,
					   try * BENCH_ZRAM_NODE_MS);
			return 0;
		}

		/*
		 * A device we adopted is not being created right now, so there
		 * is nothing to wait for; only a hot_add has a node in flight.
		 */
		if (!z->added)
			break;

		if (bench_zram_stop) {
			bench_err("interrupted while waiting for the zram%d "
				  "device node", z->id);
			return -1;
		}

		bench_msleep(BENCH_ZRAM_NODE_MS);
	}

	if (z->added) {
		bench_zram_node_complain(z);
		bench_warn("zram%d has no device node after %ums; creating it",
			   z->id, BENCH_ZRAM_NODE_TRIES * BENCH_ZRAM_NODE_MS);
		return bench_zram_make_node(z);
	}

	bench_zram_node_complain(z);
	bench_err("zram%d exists but has no usable device node", z->id);

	return -1;
}

static int bench_zram_acquire(struct bench_zram *z, const struct bench_opts *o)
{
	uint64_t initstate;
	char path[BENCH_ZRAM_PATH];

	if (o->zram_dev < 0) {
		z->id = bench_zram_hot_add();
		if (z->id < 0)
			return -1;
		z->added = true;
		bench_info("added scratch device zram%d", z->id);
	} else {
		z->id = o->zram_dev;
	}

	snprintf(z->sys, sizeof(z->sys), "/sys/block/zram%d", z->id);

	if (!bench_path_exists(z->sys)) {
		bench_err("no such device: %s", z->sys);
		return -1;
	}

	if (bench_zram_find_node(z))
		return -1;

	if (z->added)
		return 0;

	/*
	 * An adopted device is somebody else's.  Two questions have to come
	 * back the right way before it is touched, and --force only answers
	 * the second one: no amount of forcing makes resetting live swap safe.
	 */
	if (bench_zram_in_swap(z->id)) {
		bench_err("zram%d is in use as swap; the suite will not "
			  "reset it", z->id);
		return -1;
	}

	bench_zram_attr(z, "initstate", path, sizeof(path));
	if (bench_read_u64(path, &initstate)) {
		bench_err("zram%d: cannot read initstate: %s", z->id,
			  strerror(errno));
		return -1;
	}

	if (initstate && !o->zram_force) {
		bench_err("zram%d is already initialised and may hold data; "
			  "pass --force to reset it anyway",
			  z->id);
		return -1;
	}

	if (initstate)
		bench_warn("zram%d is initialised and --force was given: its "
			   "contents are about to be destroyed",
			   z->id);

	return 0;
}

/* ------------------------------------------------------------ configuration */

/*
 * "lz4kdr,zstd:3" into rungs.  The level is split off here because the kernel
 * has nowhere to put it in a name: comp_algorithm and recomp_algorithm both
 * look the whole string up as a backend, so a name carrying a suffix is simply
 * not found.
 */
static int bench_zram_parse_comp(struct bench_zram *z, const char *spec)
{
	const char *p = spec;

	z->nr_comp = 0;

	while (*p) {
		struct bench_comp *c;
		const char *end;
		const char *colon;
		size_t len;

		while (*p == ',' || *p == ' ')
			p++;
		if (!*p)
			break;

		if (z->nr_comp >= BENCH_ZRAM_COMPS_MAX) {
			bench_err("at most %u compressors can be configured",
				  BENCH_ZRAM_COMPS_MAX);
			return -1;
		}

		for (end = p; *end && *end != ','; end++)
			;

		c = &z->comp[z->nr_comp];
		c->level = -1;

		colon = memchr(p, ':', (size_t)(end - p));
		if (colon) {
			char digits[16];
			uint64_t level;

			len = (size_t)(end - colon - 1);
			if (!len || len >= sizeof(digits)) {
				bench_err("'%.*s' has no usable level",
					  (int)(end - p), p);
				return -1;
			}

			memcpy(digits, colon + 1, len);
			digits[len] = '\0';

			if (!bench_parse_u64(digits, &level) || level > 32767) {
				bench_err("'%s' is not a compression level",
					  digits);
				return -1;
			}

			c->level = (long)level;
			end = colon;
		}

		len = (size_t)(end - p);
		if (!len || len >= sizeof(c->name)) {
			bench_err("'%s' does not name a compressor", spec);
			return -1;
		}

		memcpy(c->name, p, len);
		c->name[len] = '\0';

		if (!bench_name_ok(c->name)) {
			bench_err("'%s' is not a usable backend name", c->name);
			return -1;
		}

		z->nr_comp++;
		p = colon ? colon : end;
		while (*p && *p != ',')
			p++;
	}

	if (!z->nr_comp) {
		bench_err("no compressor was named");
		return -1;
	}

	return 0;
}

/* The parsed ladder printed back, so two runs agree on spelling and spacing. */
static void bench_zram_comp_str(const struct bench_zram *z, char *buf,
				size_t len)
{
	size_t used = 0;

	buf[0] = '\0';

	for (unsigned int i = 0; i < z->nr_comp && used < len; i++) {
		int n;

		if (z->comp[i].level < 0)
			n = snprintf(buf + used, len - used, "%s%s",
				     i ? "," : "", z->comp[i].name);
		else
			n = snprintf(buf + used, len - used, "%s%s:%ld",
				     i ? "," : "", z->comp[i].name,
				     z->comp[i].level);

		if (n < 0 || (size_t)n >= len - used)
			break;

		used += (size_t)n;
	}
}

static int bench_zram_configure(struct bench_zram *z, unsigned int level)
{
	char val[BENCH_META_VALUE_MAX];
	uint64_t fill_bytes = z->fill_pages * z->page_size;

	/* A device with an opener refuses to reset, and this always resets. */
	bench_zram_close(z);

	if (bench_zram_reset(z))
		return -1;

	if (bench_zram_set(z, "comp_algorithm", z->comp[0].name))
		return -1;

	for (unsigned int i = 1; i < z->nr_comp; i++) {
		snprintf(val, sizeof(val), "algo=%s priority=%u",
			 z->comp[i].name, i);
		if (bench_zram_set(z, "recomp_algorithm", val))
			return -1;
	}

	/*
	 * Levels go in after the names, because algorithm_params looks a
	 * priority up against the names that are already there, and before
	 * disksize, because it is refused outright once the device is live.
	 */
	for (unsigned int i = 0; i < z->nr_comp; i++) {
		if (z->comp[i].level < 0)
			continue;

		snprintf(val, sizeof(val), "priority=%u level=%ld", i,
			 z->comp[i].level);
		if (bench_zram_set(z, "algorithm_params", val))
			return -1;
	}

	/*
	 * A backstop, not a target.  If every page turned out incompressible
	 * the device would want one page of memory per page written plus
	 * allocator overhead, so an eighth over the fill is generous; past it
	 * zram refuses the write and the suite reports a failure, a far better
	 * outcome than the machine going out to the OOM killer.
	 */
	snprintf(val, sizeof(val), "%" PRIu64, fill_bytes + fill_bytes / 8);
	if (bench_zram_set(z, "mem_limit", val))
		return -1;

	snprintf(val, sizeof(val), "%" PRIu64, z->disksize);
	if (bench_zram_set(z, "disksize", val))
		return -1;

	if (z->have_sysctl) {
		snprintf(val, sizeof(val), "%u", level);
		bench_kmsg_drain();
		if (bench_write_file(BENCH_ZRAM_IR_SYSCTL, val)) {
			bench_err("cannot set %s to %u: %s",
				  BENCH_ZRAM_IR_SYSCTL, level,
				  strerror(errno));
			bench_kmsg_dump();
			return -1;
		}
		bench_info("%s = %u", BENCH_ZRAM_IR_SYSCTL, level);
	}

	return 0;
}

/* ----------------------------------------------------------------- the work */

static int bench_zram_open(struct bench_zram *z)
{
	bench_zram_close(z);

	z->fd = open(z->node, O_RDWR | O_DIRECT | O_CLOEXEC);
	if (z->fd >= 0) {
		z->probed = true;
		z->direct = true;
		return 0;
	}

	if (errno != EINVAL && errno != EOPNOTSUPP) {
		bench_err("cannot open %s: %s", z->node, strerror(errno));
		return -1;
	}

	z->fd = open(z->node, O_RDWR | O_CLOEXEC);
	if (z->fd < 0) {
		bench_err("cannot open %s: %s", z->node, strerror(errno));
		return -1;
	}

	z->probed = true;
	z->direct = false;
	bench_warn("%s will not take O_DIRECT: the read pass measures the "
		   "page cache as much as the compressor", z->node);

	return 0;
}

/*
 * One sequential pass over the filled region, @chunk bytes at a time, timed as
 * a whole.  The data comes from a ring of pre-generated pages rather than being
 * generated as it goes, so what is timed is the device and not the generator;
 * the ring is large enough that the mixture it presents is the mixture the
 * corpus describes, and zram compresses every slot independently, so repeating
 * page images cannot flatter the ratio.
 */
static int bench_zram_pass(struct bench_zram *z, bool writing, uint64_t *ns)
{
	uint64_t chunk = z->io_pages * z->page_size;
	uint64_t total = z->fill_pages * z->page_size;
	uint64_t ring = z->ring_pages * z->page_size;
	uint64_t off;
	uint64_t start;

	start = bench_now_ns();

	for (off = 0; off < total; off += chunk) {
		ssize_t done;

		if (bench_zram_stop) {
			bench_err("interrupted during a %s pass",
				  writing ? "write" : "read");
			return -1;
		}

		if (writing)
			done = pwrite(z->fd, z->ring + (off % ring), chunk,
				      (off_t)off);
		else
			done = pread(z->fd, z->io, chunk, (off_t)off);

		if (done != (ssize_t)chunk) {
			bench_err("zram%d: %s of %" PRIu64 " bytes at %"
				  PRIu64 " returned %zd: %s", z->id,
				  writing ? "write" : "read", chunk, off,
				  done, strerror(errno));
			return -1;
		}
	}

	if (writing && fdatasync(z->fd)) {
		bench_err("zram%d: fdatasync failed: %s", z->id,
			  strerror(errno));
		return -1;
	}

	*ns = bench_now_ns() - start;

	return 0;
}

/* Best of @rounds passes, in MiB/s.  Best rather than mean: a pass that was
 * interrupted by something else on the machine measured that something else.
 */
static int bench_zram_throughput(struct bench_zram *z, bool writing,
				 unsigned int rounds, double *mib_s)
{
	double total_mib;
	double best = 0.0;

	total_mib = (double)(z->fill_pages * z->page_size) / (1024.0 * 1024.0);

	for (unsigned int i = 0; i < rounds; i++) {
		double rate;
		uint64_t ns;

		if (bench_zram_pass(z, writing, &ns))
			return -1;

		if (!ns)
			continue;

		rate = total_mib / ((double)ns / 1e9);
		if (rate > best)
			best = rate;
	}

	if (best <= 0.0) {
		bench_err("zram%d: no %s pass produced a usable time", z->id,
			  writing ? "write" : "read");
		return -1;
	}

	*mib_s = best;

	return 0;
}

/*
 * Read everything back and compare it against what was written.  Untimed and
 * unconditional: it is the answer to "was it fast because it was wrong", and it
 * is the only check that the recompression path preserves what it rewrites.
 * Returns the number of mismatching chunks, or -1 if the read itself failed.
 */
static int64_t bench_zram_check(struct bench_zram *z)
{
	uint64_t chunk = z->io_pages * z->page_size;
	uint64_t total = z->fill_pages * z->page_size;
	uint64_t ring = z->ring_pages * z->page_size;
	uint64_t off;
	int64_t bad = 0;

	for (off = 0; off < total; off += chunk) {
		ssize_t done;

		if (bench_zram_stop) {
			bench_err("interrupted during verification");
			return -1;
		}

		done = pread(z->fd, z->io, chunk, (off_t)off);
		if (done != (ssize_t)chunk) {
			bench_err("zram%d: verification read at %" PRIu64
				  " returned %zd: %s", z->id, off, done,
				  strerror(errno));
			return -1;
		}

		if (memcmp(z->io, z->ring + (off % ring), chunk))
			bad++;
	}

	if (bad)
		bench_err("zram%d: %" PRId64 " of %" PRIu64 " chunks came "
			  "back wrong", z->id, bad, total / chunk);

	return bad;
}

/*
 * One recompression sweep, timed, with the saving taken from zram's own
 * accounting either side of it.  The difference is signed and reported as it
 * stands: a sweep that made the data larger is something to see rather than
 * something to clamp to zero.
 */
static int bench_zram_sweep(struct bench_zram *z, const char *args,
			    uint64_t *ns, double *saved_kib)
{
	struct bench_mm before, after;
	char path[BENCH_ZRAM_PATH];
	uint64_t start;

	if (bench_zram_mm(z, &before))
		return -1;

	bench_zram_attr(z, "recompress", path, sizeof(path));

	/* Before the clock starts: draining is not part of what is measured. */
	bench_kmsg_drain();

	start = bench_now_ns();
	if (bench_write_file(path, args)) {
		bench_err("zram%d: recompress '%s' refused: %s", z->id, args,
			  strerror(errno));
		bench_kmsg_dump();
		return -1;
	}
	*ns = bench_now_ns() - start;

	if (bench_zram_mm(z, &after))
		return -1;

	*saved_kib = ((double)before.compr - (double)after.compr) / 1024.0;

	return 0;
}

/* --------------------------------------------------------------- publishing */

static void bench_zram_add(struct bench_report *rep, const char *tcase,
			   const char *variant, const char *metric,
			   const char *unit, enum bench_dir dir, double value)
{
	bench_add(rep, BENCH_ZRAM_SUITE, tcase, variant, metric, unit, dir,
		  value);
}

static void bench_zram_verdict(struct bench_report *rep, const char *tcase,
			       const char *variant, bool ok)
{
	bench_zram_add(rep, tcase, variant, "verify", "bool",
		       BENCH_HIGHER_BETTER, ok ? 1.0 : 0.0);
}

static void bench_zram_publish_mm(struct bench_report *rep,
				  const struct bench_zram *z,
				  const char *variant,
				  const struct bench_mm *mm)
{
	double page = (double)z->page_size;
	double orig = (double)mm->orig;

	if (orig > 0.0) {
		bench_zram_add(rep, "fill", variant, "ratio.pct", "pct",
			       BENCH_LOWER_BETTER,
			       (double)mm->compr / orig * 100.0);
		bench_zram_add(rep, "fill", variant, "huge.pct", "pct",
			       BENCH_LOWER_BETTER,
			       (double)mm->huge * page / orig * 100.0);
		bench_zram_add(rep, "fill", variant, "same.pct", "pct",
			       BENCH_HIGHER_BETTER,
			       (double)mm->same * page / orig * 100.0);
	}

	bench_zram_add(rep, "fill", variant, "mem.mib", "MiB",
		       BENCH_LOWER_BETTER,
		       (double)mm->mem_used / (1024.0 * 1024.0));
}

/* --------------------------------------------------------------- one level */

/*
 * A sweep case: refill so that it acts on the same state its sibling did, run
 * it, then read everything back to be sure recompression did not lose anything.
 */
static int bench_zram_sweep_case(struct bench_report *rep,
				 struct bench_zram *z, const char *tcase,
				 const char *variant, unsigned int level,
				 const char *args, bool refill)
{
	double saved_kib;
	uint64_t ns;
	int64_t bad;

	if (refill) {
		uint64_t ignored;

		if (bench_zram_configure(z, level) || bench_zram_open(z))
			return -1;
		if (bench_zram_pass(z, true, &ignored))
			return -1;
	}

	if (bench_zram_sweep(z, args, &ns, &saved_kib)) {
		bench_zram_verdict(rep, tcase, variant, false);
		return -1;
	}

	bad = bench_zram_check(z);
	if (bad < 0)
		return -1;

	bench_zram_verdict(rep, tcase, variant, bad == 0);
	bench_zram_add(rep, tcase, variant, "recomp.ms", "ms",
		       BENCH_LOWER_BETTER, (double)ns / 1e6);
	bench_zram_add(rep, tcase, variant, "recomp.saved_kib", "KiB",
		       BENCH_HIGHER_BETTER, saved_kib);

	return 0;
}

static int bench_zram_level(struct bench_report *rep, struct bench_zram *z,
			    unsigned int level)
{
	char variant[BENCH_NAME_MAX];
	char args[128];
	struct bench_mm mm;
	double mib_s;
	int64_t bad;

	snprintf(variant, sizeof(variant), "ir%u", level);

	if (bench_zram_configure(z, level) || bench_zram_open(z))
		return -1;

	if (bench_zram_throughput(z, true, BENCH_ZRAM_IO_ROUNDS, &mib_s))
		return -1;
	bench_zram_add(rep, "fill", variant, "write.mib_s", "MiB/s",
		       BENCH_HIGHER_BETTER, mib_s);

	if (bench_zram_mm(z, &mm))
		return -1;
	bench_zram_publish_mm(rep, z, variant, &mm);

	if (bench_zram_throughput(z, false, BENCH_ZRAM_IO_ROUNDS, &mib_s))
		return -1;
	bench_zram_add(rep, "read", variant, "read.mib_s", "MiB/s",
		       BENCH_HIGHER_BETTER, mib_s);

	bad = bench_zram_check(z);
	if (bad < 0)
		return -1;

	bench_zram_verdict(rep, "fill", variant, bad == 0);
	bench_zram_verdict(rep, "read", variant, bad == 0);

	if (!z->multi_comp || z->nr_comp < 2) {
		bench_info("no recompression sweep: %s",
			   z->multi_comp ? "fewer than two compressors" :
					   "no recompress attribute");
		return 0;
	}

	/*
	 * The narrowed sweep first, on the state the fill just left, then the
	 * unnarrowed one on a freshly identical device.  Refilling between them
	 * is what makes the two numbers comparable: the first sweep may have
	 * changed what the second would have found.
	 */
	snprintf(args, sizeof(args), "type=huge max_pages=%u",
		 BENCH_ZRAM_SWEEP_PAGES);
	if (bench_zram_sweep_case(rep, z, "sweep.huge", variant, level, args,
				  false))
		return -1;

	snprintf(args, sizeof(args), "max_pages=%u", BENCH_ZRAM_SWEEP_PAGES);
	if (bench_zram_sweep_case(rep, z, "sweep.all", variant, level, args,
				  true))
		return -1;

	return 0;
}

/* ------------------------------------------------------------------- set up */

static int bench_meminfo_kb(const char *key, uint64_t *out)
{
	char buf[BENCH_ZRAM_MEMINFO_MAX];
	char *p;

	if (bench_read_file("/proc/meminfo", buf, sizeof(buf)))
		return -1;

	p = strstr(buf, key);
	if (!p)
		return -1;

	p += strlen(key);
	while (*p == ' ' || *p == '\t' || *p == ':')
		p++;

	*out = 0;
	if (*p < '0' || *p > '9')
		return -1;

	for (; *p >= '0' && *p <= '9'; p++) {
		if (*out > (UINT64_MAX - (uint64_t)(*p - '0')) / 10)
			return -1;
		*out = *out * 10 + (uint64_t)(*p - '0');
	}

	return 0;
}

/*
 * How many pages this machine can absorb right now assuming the worst, which is
 * that nothing compresses at all and every page written costs a page of memory.
 * The clamp is loud because it changes a binding fact: a run that had to write
 * less than it was asked to is not comparable with one that did not, and the
 * comparison will say so rather than scoring across the difference.
 */
static void bench_zram_clamp_fill(struct bench_zram *z)
{
	uint64_t available_kb;
	uint64_t budget_pages;

	if (bench_meminfo_kb("MemAvailable", &available_kb)) {
		bench_warn("cannot read MemAvailable; the fill is not "
			   "being clamped");
		return;
	}

	if (available_kb <= BENCH_ZRAM_HEADROOM_KB) {
		bench_warn("only %" PRIu64 " KiB available, inside the %u KiB "
			   "headroom; the fill is left as asked and may fail",
			   available_kb, BENCH_ZRAM_HEADROOM_KB);
		return;
	}

	budget_pages = (available_kb - BENCH_ZRAM_HEADROOM_KB) * 1024 /
		       z->page_size;
	budget_pages -= budget_pages % z->io_pages;

	if (!budget_pages) {
		bench_warn("there is no room for even one request; the fill "
			   "is left as asked and may fail");
		return;
	}

	if (z->fill_pages <= budget_pages)
		return;

	bench_warn("clamping the fill from %" PRIu64 " to %" PRIu64 " pages: "
		   "%" PRIu64 " KiB available less %u KiB headroom",
		   z->fill_pages, budget_pages, available_kb,
		   BENCH_ZRAM_HEADROOM_KB);
	bench_warn("this is a binding fact, so this run will not compare "
		   "against a baseline that wrote more");

	z->fill_pages = budget_pages;
}

static int bench_zram_sizes(struct bench_zram *z, const struct bench_opts *o)
{
	uint64_t disk_pages;

	z->page_size = (uint64_t)sysconf(_SC_PAGESIZE);
	if (!z->page_size) {
		bench_err("cannot determine the page size");
		return -1;
	}

	z->io_pages = o->zram_io_pages;
	if (!z->io_pages || z->io_pages > BENCH_ZRAM_IO_PAGES_MAX) {
		bench_err("pages per request must be between 1 and %u",
			  BENCH_ZRAM_IO_PAGES_MAX);
		return -1;
	}

	/*
	 * The ring has to hold a whole number of requests, because a request is
	 * served from one offset in it and must not run off the end.
	 */
	z->ring_pages = BENCH_ZRAM_RING_PAGES;
	if (z->ring_pages % z->io_pages)
		z->ring_pages += z->io_pages - z->ring_pages % z->io_pages;

	z->disksize = o->zram_size;
	if (z->disksize < z->page_size) {
		bench_err("the disksize must be at least one page");
		return -1;
	}

	z->fill_pages = o->zram_pages;
	z->fill_pages -= z->fill_pages % z->io_pages;
	if (!z->fill_pages) {
		bench_err("the fill must be at least one request of %"
			  PRIu64 " pages", z->io_pages);
		return -1;
	}

	disk_pages = z->disksize / z->page_size;
	disk_pages -= disk_pages % z->io_pages;
	if (z->fill_pages > disk_pages) {
		bench_warn("clamping the fill from %" PRIu64 " to %" PRIu64
			   " pages, which is all the disksize holds",
			   z->fill_pages, disk_pages);
		z->fill_pages = disk_pages;
	}

	if (!z->fill_pages) {
		bench_err("the disksize does not hold one request");
		return -1;
	}

	bench_zram_clamp_fill(z);

	return 0;
}

static int bench_zram_buffers(struct bench_zram *z, uint32_t seed)
{
	struct bench_corpus corpus;
	size_t page = (size_t)z->page_size;

	bench_corpus_init(&corpus, page, z->ring_pages, seed);
	bench_corpus_describe(&corpus, z->mix, sizeof(z->mix));

	/*
	 * Page aligned because O_DIRECT will not take anything else, and
	 * allocated before the device is acquired: bench_xalloc_aligned() exits
	 * on failure, and it must not be able to do that with a scratch device
	 * outstanding.
	 */
	z->ring = bench_xalloc_aligned(page, (size_t)z->ring_pages * page);
	z->io = bench_xalloc_aligned(page, (size_t)z->io_pages * page);

	for (uint64_t i = 0; i < z->ring_pages; i++)
		bench_corpus_page(&corpus, i, z->ring + (size_t)i * page);

	return 0;
}

/*
 * The levels to measure, deduplicated by the depth they actually produce.  The
 * sysctl names a depth of level + 1, and the write path takes the smaller of
 * that and the number of configured compressors, so on a two rung ladder all
 * three levels are the same run of code and only one of them is worth
 * measuring.
 */
static int bench_zram_levels(const struct bench_zram *z, const char *spec,
			     unsigned int *out, unsigned int *nr_out)
{
	unsigned int seen_depth[BENCH_ZRAM_LEVEL_MAX + 2] = { 0 };
	const char *p = spec;
	unsigned int nr = 0;

	while (*p) {
		char digits[8];
		uint64_t level;
		unsigned int depth;
		const char *end;
		size_t len;

		while (*p == ',' || *p == ' ')
			p++;
		if (!*p)
			break;

		for (end = p; *end && *end != ','; end++)
			;

		len = (size_t)(end - p);
		if (!len || len >= sizeof(digits)) {
			bench_err("'%s' is not a list of levels", spec);
			return -1;
		}

		memcpy(digits, p, len);
		digits[len] = '\0';
		p = end;

		if (!bench_parse_u64(digits, &level) ||
		    level < BENCH_ZRAM_LEVEL_MIN ||
		    level > BENCH_ZRAM_LEVEL_MAX) {
			bench_err("level '%s' is outside the %u..%u the "
				  "kernel accepts", digits,
				  BENCH_ZRAM_LEVEL_MIN,
				  BENCH_ZRAM_LEVEL_MAX);
			return -1;
		}

		depth = (unsigned int)level + 1;
		if (depth > z->nr_comp)
			depth = z->nr_comp;

		if (seen_depth[depth]) {
			bench_info("level %" PRIu64 " reaches depth %u, which "
				   "level %u already measured; skipping it",
				   level, depth, seen_depth[depth] - 1);
			continue;
		}

		if (nr >= BENCH_ZRAM_LEVEL_MAX) {
			bench_err("more distinct depths than there are levels");
			return -1;
		}

		seen_depth[depth] = (unsigned int)level + 1;
		out[nr++] = (unsigned int)level;
	}

	if (!nr) {
		bench_err("no level was named");
		return -1;
	}

	*nr_out = nr;

	return 0;
}

static void bench_zram_teardown(struct bench_zram *z)
{
	bench_zram_close(z);

	if (z->sys[0] && bench_path_exists(z->sys))
		bench_zram_reset(z);

	if (z->have_sysctl) {
		char val[32];

		snprintf(val, sizeof(val), "%" PRIu64, z->saved_level);
		if (bench_write_file(BENCH_ZRAM_IR_SYSCTL, val))
			bench_warn("cannot restore %s to %" PRIu64 ": %s",
				   BENCH_ZRAM_IR_SYSCTL, z->saved_level,
				   strerror(errno));
		else
			bench_info("restored %s to %" PRIu64,
				   BENCH_ZRAM_IR_SYSCTL, z->saved_level);
	}

	if (z->added)
		bench_zram_hot_remove(z->id);

	/*
	 * Only ever our own: z->node_made is set nowhere else, so an unlink
	 * here cannot reach a node that ueventd or the platform put there.
	 */
	if (z->node_made) {
		if (unlink(z->node))
			bench_warn("cannot remove %s: %s", z->node,
				   strerror(errno));
		else
			bench_info("removed %s", z->node);

		z->node_made = false;
	}

	free(z->ring);
	free(z->io);
	z->ring = NULL;
	z->io = NULL;

	bench_kmsg_close();
}

/* ------------------------------------------------------------------- driver */

int bench_zram_suite(struct bench_report *rep, const struct bench_opts *o)
{
	unsigned int levels[BENCH_ZRAM_LEVEL_MAX];
	unsigned int nr_levels = 0;
	struct sigaction act, old_int, old_term;
	struct bench_zram z;
	char comps[BENCH_META_VALUE_MAX];
	char measured[BENCH_META_VALUE_MAX];
	bool installed = false;
	int ret = -1;

	memset(&z, 0, sizeof(z));
	memset(&old_int, 0, sizeof(old_int));
	memset(&old_term, 0, sizeof(old_term));
	z.fd = -1;

	if (bench_zram_parse_comp(&z, o->zram_comp))
		return -1;

	if (bench_zram_levels(&z, o->ir_levels, levels, &nr_levels))
		return -1;

	if (bench_zram_sizes(&z, o))
		return -1;

	if (bench_zram_buffers(&z, o->corpus_seed))
		goto out;

	/*
	 * From here on there is state to give back, so every exit goes through
	 * the teardown.  The handlers are installed before the device exists so
	 * that a signal arriving in the middle of acquiring it still leaves the
	 * flag set for the first loop that looks.
	 */
	memset(&act, 0, sizeof(act));
	act.sa_handler = bench_zram_catch;
	act.sa_flags = SA_RESTART;
	sigemptyset(&act.sa_mask);

	if (!sigaction(SIGINT, &act, &old_int)) {
		if (sigaction(SIGTERM, &act, &old_term)) {
			sigaction(SIGINT, &old_int, NULL);
		} else {
			installed = true;
		}
	}
	if (!installed)
		bench_warn("cannot install signal handlers; an interrupt "
			   "will leave the device behind");

	/*
	 * From here to the teardown every refusal can be explained, which
	 * starts with the very first one: hot_add is a sysfs store like the
	 * rest and fails for reasons it only writes to the log.
	 */
	bench_kmsg_open();

	if (bench_zram_acquire(&z, o))
		goto out;

	if (bench_path_exists(BENCH_ZRAM_IR_SYSCTL)) {
		if (bench_read_u64(BENCH_ZRAM_IR_SYSCTL, &z.saved_level)) {
			bench_err("cannot read %s: %s", BENCH_ZRAM_IR_SYSCTL,
				  strerror(errno));
			goto out;
		}
		z.have_sysctl = true;
	} else {
		bench_warn("%s is absent: this kernel has no immediate "
			   "recompression, so every level below measures the "
			   "same thing", BENCH_ZRAM_IR_SYSCTL);
	}

	z.multi_comp = bench_zram_has(&z, "recompress") &&
		       bench_zram_has(&z, "recomp_algorithm");
	if (!z.multi_comp && z.nr_comp > 1) {
		bench_warn("this kernel has no recompression attributes; "
			   "only %s will be configured", z.comp[0].name);
		z.nr_comp = 1;
	}

	bench_zram_comp_str(&z, comps, sizeof(comps));

	measured[0] = '\0';
	for (unsigned int i = 0; i < nr_levels; i++) {
		char one[8];

		snprintf(one, sizeof(one), "%s%u", i ? "," : "", levels[i]);
		strncat(measured, one, sizeof(measured) - strlen(measured) - 1);
	}

	bench_add_meta(rep, "zram.dev", "zram%d (%s)", z.id,
		       z.added ? "scratch" : "adopted");
	bench_add_meta(rep, "zram.comp", "%s", comps);
	bench_add_meta(rep, "zram.disksize", "%" PRIu64, z.disksize);
	bench_add_meta(rep, "zram.fill", "%" PRIu64, z.fill_pages);
	bench_add_meta(rep, "zram.io", "%" PRIu64, z.io_pages);
	bench_add_meta(rep, "zram.pages", "%" PRIu64, z.ring_pages);
	bench_add_meta(rep, "zram.corpus", "%s", z.mix);
	bench_add_meta(rep, "zram.rounds", "%u", BENCH_ZRAM_IO_ROUNDS);
	bench_add_meta(rep, "zram.sweep", "%u pages", BENCH_ZRAM_SWEEP_PAGES);
	bench_add_meta(rep, "zram.ir", "%s", measured);

	bench_info("zram%d: %s, %" PRIu64 " MiB disk, filling %" PRIu64
		   " MiB in %" PRIu64 " KiB requests",
		   z.id, comps, z.disksize / (1024 * 1024),
		   z.fill_pages * z.page_size / (1024 * 1024),
		   z.io_pages * z.page_size / 1024);

	ret = 0;
	for (unsigned int i = 0; i < nr_levels; i++) {
		if (bench_zram_level(rep, &z, levels[i])) {
			ret = -1;
			break;
		}
	}

	/*
	 * Recorded after the fact: whether O_DIRECT was accepted is only known
	 * once the device has been opened, and it decides whether the read
	 * number describes the compressor or the page cache.
	 */
	bench_add_meta(rep, "zram.direct", "%s",
		       !z.probed ? "unknown" : z.direct ? "yes" : "no");

out:
	bench_zram_teardown(&z);

	if (installed) {
		sigaction(SIGINT, &old_int, NULL);
		sigaction(SIGTERM, &old_term, NULL);
	}

	if (bench_zram_stop)
		bench_err("the zram suite was interrupted; results from it "
			  "are incomplete");

	return ret;
}
