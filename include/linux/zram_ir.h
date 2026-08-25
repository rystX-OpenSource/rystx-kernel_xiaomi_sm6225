/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * zram_ir.h - in-kernel interface to zram immediate recompression (IR)
 *
 * zram exposes recompression and its accounting to userspace only, through
 * sysfs.  An in-kernel reclaimer that already knows which pages it is about
 * to swap out, and how valuable the owning task is, can make better
 * decisions than a periodic userspace sweep - but it needs two things that
 * sysfs cannot give it:
 *
 *   - a way to tell the *store* path how hard to try for the page it is
 *     writing right now (zram_ir_set_depth()), and
 *   - a bounded, allocation-free way to run a recompression sweep from a
 *     memory-pressure context (zram_ir_recompress()).
 *
 * Copyright (C) 2026 iDeadXS <datarafi43@gmail.com>
 */
#ifndef _LINUX_ZRAM_IR_H
#define _LINUX_ZRAM_IR_H

#include <linux/kconfig.h>
#include <linux/sched.h>
#include <linux/types.h>

/*
 * Recompression is only meaningful when there is more than one algorithm
 * configured, and the symbols only link when zram is reachable from the
 * caller's link unit.  CONFIG_ZRAM_MULTI_COMP is a bool, so it cannot carry
 * the =m case on its own; pair it with IS_REACHABLE(CONFIG_ZRAM).
 */
#define ZRAM_IR_REACHABLE \
	(IS_REACHABLE(CONFIG_ZRAM) && IS_ENABLED(CONFIG_ZRAM_MULTI_COMP))

/**
 * struct zram_ir_stats - accounting aggregated over all initialised devices
 * @compr_bytes: compressed data currently stored, in bytes
 * @stored_pages: slots currently holding data, including same-filled and huge
 * @mem_used_pages: physical pages zsmalloc holds, including fragmentation
 * @huge_pages: slots stored uncompressed because nothing beat the huge class
 * @same_pages: slots that are a repeated pattern and cost no memory at all
 * @nr_devices: initialised devices folded into these numbers
 * @nr_comps: lowest number of configured algorithms across those devices
 * @ir_depth: priorities the store path may currently walk by default
 *
 * @compr_bytes is in bytes rather than pages because compressed objects are
 * sub-page sized; rounding each device to pages would swamp the signal a
 * caller is trying to measure.
 *
 * @huge_pages is only a fair proxy for "genuinely incompressible residue"
 * when @ir_depth >= @nr_comps: a shallower ladder leaves pages uncompressed
 * that a deeper one would have compressed, and those are not residue.
 * Callers that reason about residue must check that themselves.
 */
struct zram_ir_stats {
	u64			compr_bytes;
	unsigned long		stored_pages;
	unsigned long		mem_used_pages;
	unsigned long		huge_pages;
	unsigned long		same_pages;
	unsigned int		nr_devices;
	unsigned int		nr_comps;
	unsigned int		ir_depth;
};

/*
 * Depth values for zram_ir_set_depth().  ZRAM_IR_DEPTH_FULL is deliberately
 * larger than any possible number of algorithms: the store path clamps the
 * hint against the device's configured count, so "more than exists" reads as
 * "as deep as this device goes" without callers having to know the count.
 */
#define ZRAM_IR_DEPTH_DEFAULT	0u
#define ZRAM_IR_DEPTH_MIN	1u
#define ZRAM_IR_DEPTH_FULL	0xffu

/* Slot selection filters, mirroring RECOMPRESS_* in zram_drv.c. */
#define ZRAM_IR_IDLE		(1u << 0)
#define ZRAM_IR_HUGE		(1u << 1)

/**
 * struct zram_ir_req - one bounded recompression sweep
 * @scratch: caller-owned page used to stage decompressed data
 * @threshold: leave objects already smaller than this alone, in bytes
 * @max_pages: stop after attempting this many slots, 0 for no limit
 * @max_slots: never track more than this many candidate slots at once
 * @mode: ZRAM_IR_* filter applied while selecting slots
 *
 * @scratch lets a pressure-path caller pre-allocate once at init instead of
 * allocating a page while trying to free memory; it must be a single page
 * the caller keeps alive across the call.
 *
 * @max_slots bounds the only unbounded cost in the sysfs path, which tracks
 * every eligible slot on the device at once - on a 4G zram that is a
 * million slots and tens of megabytes of transient allocation, which is
 * exactly the wrong thing to do while reclaiming.
 */
struct zram_ir_req {
	struct page		*scratch;
	u32			threshold;
	unsigned long		max_pages;
	unsigned long		max_slots;
	u32			mode;
};

#if ZRAM_IR_REACHABLE

int zram_ir_get_stats(struct zram_ir_stats *stats);
long zram_ir_recompress(const struct zram_ir_req *req);

#else /* !ZRAM_IR_REACHABLE */

static inline int zram_ir_get_stats(struct zram_ir_stats *stats)
{
	return -ENODEV;
}

static inline long zram_ir_recompress(const struct zram_ir_req *req)
{
	return -ENODEV;
}

#endif /* ZRAM_IR_REACHABLE */

#ifdef CONFIG_ZRAM_MULTI_COMP

/**
 * zram_ir_set_depth - ask the store path to walk @depth priorities
 * @depth: number of compression priorities to allow, 1 based
 *
 * Applies to every zram store this task performs until reset, and is
 * clamped by the number of algorithms the device actually has configured.
 * @depth of 1 means "primary algorithm only", so a caller that wants to
 * spend as little CPU as possible can say so; 0 restores the default.
 *
 * The state lives on task_struct rather than a per-CPU slot because the
 * reclaim path between the hint and the store sleeps, and the reclaimer can
 * migrate CPUs in between.  It is inherited across fork() like the rest of
 * task_struct, so callers must pair this with zram_ir_reset_depth() and keep
 * the region free of fork(), exactly as the scoped memcg fields do.
 */
static inline void zram_ir_set_depth(u8 depth)
{
	current->zram_ir_depth = depth;
}

static inline void zram_ir_reset_depth(void)
{
	current->zram_ir_depth = 0;
}

#else /* !CONFIG_ZRAM_MULTI_COMP */

static inline void zram_ir_set_depth(u8 depth) { }
static inline void zram_ir_reset_depth(void) { }

#endif /* CONFIG_ZRAM_MULTI_COMP */

#endif /* _LINUX_ZRAM_IR_H */
