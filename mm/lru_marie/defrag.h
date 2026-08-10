/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _MM_LRU_MARIE_DEFRAG_H
#define _MM_LRU_MARIE_DEFRAG_H

/*
 * Marie defragmentation -- Step 1: per-pageblock occupancy
 * histogram (the sufficient-statistic input to the future cost-scored block
 * selector). See mm/lru_marie/defrag_design.md.
 *
 * This header exposes only the two hot-path maintenance hooks
 * (marie_defrag_hist_inc/dec). They are tapped from Marie's single gen-occupancy
 * choke-point marie_gen_occ_inc/dec (state.h), so every per-PFN gen
 * transition that moves marie_gen_occupied moves the per-pageblock mirror in
 * lockstep -- the completeness invariant being
 *
 *     Sigma_blocks count[gen][type] == marie_gen_occupied[gen][type]
 *
 * which defrag.c verifies via the read-only /sys/kernel/mm/lru_marie/marie_defrag node.
 *
 * Step 1 is observability ONLY: no migration, no reclaim, no policy change.
 * Gated by CONFIG_LRU_MARIE_DEFRAG; when n the hooks are empty inlines
 * and the kernel image is byte-identical to plain Marie.
 */

#include <linux/types.h>

#ifdef CONFIG_LRU_MARIE_DEFRAG

#include <linux/atomic.h>
#include <linux/mmzone.h>
#include <linux/pageblock-flags.h>	/* pageblock_order */

struct kobject;

/*
 * One atomic count per (gen, type) class, per order-9 pageblock.
 * MARIE_DEFRAG_NGENS x MARIE_DEFRAG_NTYPES = 8 x 2 = 16 classes = 64 bytes = exactly one
 * cache line per block. class index = gen * MARIE_DEFRAG_NTYPES + type. The
 * MARIE_PFN_NR_GENS == MARIE_DEFRAG_NGENS coupling is asserted in defrag.c
 * (BUILD_BUG_ON), so this header needs no state.h dependency.
 */
#define MARIE_DEFRAG_NGENS	8
#define MARIE_DEFRAG_NTYPES	2
#define MARIE_DEFRAG_NCLASS	(MARIE_DEFRAG_NGENS * MARIE_DEFRAG_NTYPES)

struct marie_defrag_block_hist {
	atomic_t count[MARIE_DEFRAG_NCLASS];
};

extern struct marie_defrag_block_hist *marie_defrag_hist;	/* NULL until marie_defrag_init() */
extern unsigned long marie_defrag_nr_blocks;

static inline void marie_defrag_hist_inc(unsigned long pfn, int gen, int type)
{
	unsigned long blk;

	if (!marie_defrag_hist)
		return;
	blk = pfn >> pageblock_order;
	if (blk < marie_defrag_nr_blocks)
		atomic_inc(&marie_defrag_hist[blk].count[gen * MARIE_DEFRAG_NTYPES + type]);
}

static inline void marie_defrag_hist_dec(unsigned long pfn, int gen, int type)
{
	unsigned long blk;

	if (!marie_defrag_hist)
		return;
	blk = pfn >> pageblock_order;
	if (blk < marie_defrag_nr_blocks)
		atomic_dec(&marie_defrag_hist[blk].count[gen * MARIE_DEFRAG_NTYPES + type]);
}

/* One-shot init from marie_init(): allocate the array + create the sysfs node. */
int marie_defrag_init(struct kobject *parent);

#else /* !CONFIG_LRU_MARIE_DEFRAG */

static inline void marie_defrag_hist_inc(unsigned long pfn, int gen, int type) { }
static inline void marie_defrag_hist_dec(unsigned long pfn, int gen, int type) { }

#endif /* CONFIG_LRU_MARIE_DEFRAG */

#endif /* _MM_LRU_MARIE_DEFRAG_H */
