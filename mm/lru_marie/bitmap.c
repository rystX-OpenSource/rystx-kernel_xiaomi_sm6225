// SPDX-License-Identifier: GPL-2.0
/*
 * Hierarchical PFN bitmap operations. See bitmap.h for the design
 * overview. Used by the global per-(type, gen, tier) plane.
 */

#include <linux/atomic.h>
#include <linux/bitmap.h>
#include <linux/bitops.h>
#include <linux/cache.h>
#include <linux/export.h>
#include <linux/kernel.h>
#include <linux/mm.h>
#include <linux/mmzone.h>
#include <linux/slab.h>
#include <linux/spinlock.h>

#include "bitmap.h"
#include "state.h"	/* max_pfn, marie_l2_shift, marie_pfn_to_l2_bit */

/*
 * 512 cacheline-aligned spinlocks, one per L2 bit. Each lock makes
 * one concurrent scanner the exclusive owner of the PFN range
 * covered by that L2 bit -- collisions never produce wasted
 * candidate scan work, only a single try_lock failure that costs
 * one atomic op. ____cacheline_aligned_in_smp prevents false sharing
 * between adjacent locks while keeping UP-build footprint flat.
 *
 * 32 KiB total on SMP (64 B x 512). Shared across every marie_bitmap
 * instance: the lock is over the PFN address space, not per-bitmap.
 * Two scanners walking different (type, gen, tier) bitmaps in the
 * same L2 range still serialise via the same lock, avoiding wasted
 * parallel L1 fetches of the same physical cachelines.
 *
 * Trylock / unlock are static inline in bitmap.h; this file only
 * holds the storage and the boot-time init.
 */
struct marie_bm_range_lock marie_bm_range_locks[MARIE_L2_BITS];

void marie_bm_range_locks_init(void)
{
	int i;

	for (i = 0; i < MARIE_L2_BITS; i++)
		spin_lock_init(&marie_bm_range_locks[i].lock);
}

int marie_bm_init(struct marie_bitmap *bm)
{
	unsigned long bytes;

	if (!max_pfn)
		return 0;
	bytes = BITS_TO_LONGS(max_pfn) * sizeof(unsigned long);
	bm->l1 = kvmalloc(bytes, GFP_KERNEL | __GFP_ZERO);
	if (!bm->l1)
		return -ENOMEM;
	return 0;
}

void marie_bm_free(struct marie_bitmap *bm)
{
	if (!bm)
		return;
	kvfree(bm->l1);
	bm->l1 = NULL;
}

/*
 * marie_bm_set / marie_bm_clear / marie_bm_test are static inline in
 * bitmap.h -- they sit on the install / del / promote hot path and
 * out-of-lining costs measurable cycles per fault.
 */

/*
 * Inclusive [start_word, end_word) covering one L2 bit's worth of L1 words.
 * Clipped to the actual l1 storage extent.
 */
static void marie_bm_l1_word_range(unsigned int l2bit,
				   unsigned long *start_word,
				   unsigned long *end_word)
{
	unsigned long pfns_per_l2 = 1UL << marie_l2_shift;
	unsigned long start_pfn = (unsigned long)l2bit << marie_l2_shift;
	unsigned long end_pfn = start_pfn + pfns_per_l2;
	unsigned long max_words = BITS_TO_LONGS(max_pfn);

	*start_word = start_pfn / BITS_PER_LONG;
	*end_word = DIV_ROUND_UP(end_pfn, BITS_PER_LONG);
	if (*end_word > max_words)
		*end_word = max_words;
}

void marie_bm_drop_l2_range(struct marie_bitmap *bm, unsigned int l2bit)
{
	unsigned long start_word, end_word, wi;

	if (!bm->l1)
		return;
	marie_bm_l1_word_range(l2bit, &start_word, &end_word);
	for (wi = start_word; wi < end_word; wi++)
		bm->l1[wi] = 0;
	atomic_set(&bm->l2_count[l2bit], 0);
	clear_bit(l2bit, bm->l2);
}

void marie_bm_reset(struct marie_bitmap *bm)
{
	int i;

	if (!bm->l1)
		return;
	if (max_pfn)
		bitmap_zero(bm->l1, max_pfn);
	bitmap_zero(bm->l2, MARIE_L2_BITS);
	for (i = 0; i < MARIE_L2_BITS; i++)
		atomic_set(&bm->l2_count[i], 0);
}
