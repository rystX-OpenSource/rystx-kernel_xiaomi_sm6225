/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Hierarchical PFN bitmap backing Marie's global per-(type, gen, tier)
 * planes.
 *
 * Two layers held by one struct:
 *   L1: per-PFN bit, sized BITS_TO_LONGS(max_pfn). set_bit()/clear_bit()
 *       (atomic). One word covers 64 PFNs.
 *
 *   L2: 512-bit summary, each bit covers (max_pfn / 512) PFNs (one
 *       "L2 range", typically 32 MiB on an 8 GiB system). A companion
 *       per-cell atomic_t refcount tracks how many L1 bits are set in
 *       that range. The L2 bit transitions on the 0 <-> 1 counter
 *       boundary, performed inside the same atomic_*_return path,
 *       so concurrent set/clear cannot desynchronise the bit from
 *       the counter.
 *
 * Consumer:
 *   - Global plane:  one struct per (type, gen, tier), 16 instances
 *                    total (marie_track_bm[type][gen][tier]).
 *
 * No internal lock; producers serialise via the existing Marie lock
 * hierarchy (lru_lock on the install/del side, marie_l2_locks[bit]
 * trylock on the scanner side).
 */
#ifndef _MM_LRU_MARIE_BITMAP_H
#define _MM_LRU_MARIE_BITMAP_H

#include <linux/atomic.h>
#include <linux/bitmap.h>
#include <linux/bitops.h>
#include <linux/cache.h>
#include <linux/mm.h>		/* upstream's spelling; see max_pfn note below */
#include <linux/spinlock.h>
#include <linux/types.h>
#include <linux/version.h>

/*
 * max_pfn.  The upstream patch marks the <linux/mm.h> include above as the
 * source of it, but that is mislabelled even on 6.x: max_pfn is declared in
 * <linux/memblock.h> (6.19.8 memblock.h:21) and no include path from mm.h
 * reaches that header.  Upstream gets away with it because every .c that
 * pulls in this header also includes <linux/memblock.h> itself -- so the
 * declaration arrives by luck of translation-unit ordering rather than from
 * this file.
 *
 * That luck does not survive the backport: 4.19 predates the move and
 * declares max_pfn in <linux/bootmem.h> instead (bootmem.h:24, alongside
 * max_low_pfn / min_low_pfn), while its <linux/memblock.h> carries no
 * max_pfn at all -- so the .c files' memblock.h include supplies nothing and
 * the three bounds checks below fail to compile.
 *
 * Include the declaring header here explicitly on each target, so this
 * header is self-contained rather than depending on its includers.
 * bootmem.h itself includes <linux/memblock.h>, making the 4.19 arm a
 * superset of the 6.x spelling rather than a divergent one.
 */
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 8, 0)
#include <linux/bootmem.h>	/* max_pfn */
#else
#include <linux/memblock.h>	/* max_pfn */
#endif

/*
 * MARIE_L2_BITS sizes the L2 summary plane. Placed here so the
 * struct can lay out its inline arrays without pulling in state.h.
 */
#define MARIE_L2_BITS		512

/*
 * PFN -> L2 bit shift, set at marie_state_init time so that
 * (1 << marie_l2_shift) PFNs map to one L2 bit and 512 L2 bits cover
 * the full max_pfn range. shift = ceil(log2(max_pfn / 512)).
 */
extern unsigned int marie_l2_shift;

static inline unsigned int marie_pfn_to_l2_bit(unsigned long pfn)
{
	unsigned int b = pfn >> marie_l2_shift;

	return b < MARIE_L2_BITS ? b : MARIE_L2_BITS - 1;
}

static inline unsigned long marie_l2_bit_pfn_start(unsigned int bit)
{
	return (unsigned long)bit << marie_l2_shift;
}

static inline unsigned long marie_l2_bit_pfn_end(unsigned int bit)
{
	return ((unsigned long)bit + 1) << marie_l2_shift;
}

struct marie_bitmap {
	unsigned long	*l1;					/* BITS_TO_LONGS(max_pfn) words */
	unsigned long	 l2[BITS_TO_LONGS(MARIE_L2_BITS)];	/* 64 B inline */
	atomic_t	 l2_count[MARIE_L2_BITS];		/* 2 KiB inline */
};

/*
 * marie_bm_init - allocate @bm->l1 sized for the system max_pfn.
 * @bm->l2 and @bm->l2_count are zero-initialised by the caller (the
 * struct itself is typically zero-allocated). Returns 0 on success,
 * -ENOMEM on allocation failure.
 */
int marie_bm_init(struct marie_bitmap *bm);

/* marie_bm_free - release @bm->l1 (no-op when never initialised). */
void marie_bm_free(struct marie_bitmap *bm);

/*
 * marie_bm_set - mark @pfn tracked.
 *
 * Atomically sets the L1 bit at @pfn via test_and_set_bit; the per-cell
 * refcount is incremented ONLY on the real 0 -> 1 transition, and on that
 * 0 -> 1 the L2 summary bit for @pfn's range is set. Counting the actual L1
 * transition (rather than incrementing unconditionally) keeps
 * l2_count == popcount(L1 in range) an exact invariant: a double-set is a
 * no-op on the count and -- paired with the symmetric marie_bm_clear() --
 * a concurrent double-clear can no longer drive the count negative.
 *
 * Returns true iff this call performed the 0 -> 1 transition. Callers gate
 * their paired marie_gen_occ_inc() on it, so the L1 bit is the exactly-once
 * ownership token for gen_occupied too (same discipline as l2_count): whoever
 * wins the atomic set owns the +1, keeping gen_occupied == popcount(bits in
 * gen) >= 0 even when installs/moves race a reclaim isolate on one page.
 *
 * static inline because this is a hot-path operation invoked at
 * every install / promote / move; out-of-lining would add a function
 * call + bound-check overhead per call.
 */
static inline bool marie_bm_set(struct marie_bitmap *bm, unsigned long pfn)
{
	unsigned int l2bit;

	if (!bm->l1 || pfn >= max_pfn)
		return false;
	if (test_and_set_bit(pfn, bm->l1))
		return false;		/* already set: keep l2_count == popcount(L1) */
	l2bit = marie_pfn_to_l2_bit(pfn);
	if (atomic_inc_return(&bm->l2_count[l2bit]) == 1)
		set_bit(l2bit, bm->l2);
	return true;			/* this call set the bit (0 -> 1) */
}

/*
 * marie_bm_clear - mark @pfn untracked.
 *
 * Atomically clears the L1 bit at @pfn via test_and_clear_bit; the per-cell
 * refcount is decremented ONLY on the real 1 -> 0 transition, and on that
 * 1 -> 0 the L2 summary bit for @pfn's range is cleared. Decrementing only
 * when this call actually cleared a set bit makes a double-clear -- e.g. a
 * reclaim isolate (marie_evict_counters_only) racing the lock-free tier/gen
 * mutators, both retiring the same (gen, tier) coordinate for one page --
 * harmless, instead of underflowing l2_count and desyncing the L2 summary.
 */
static inline bool marie_bm_clear(struct marie_bitmap *bm, unsigned long pfn)
{
	unsigned int l2bit;

	if (!bm->l1 || pfn >= max_pfn)
		return false;
	if (!test_and_clear_bit(pfn, bm->l1))
		return false;		/* already clear: do not underflow l2_count */
	l2bit = marie_pfn_to_l2_bit(pfn);
	if (atomic_dec_return(&bm->l2_count[l2bit]) == 0)
		clear_bit(l2bit, bm->l2);
	return true;			/* this call cleared the bit (1 -> 0) */
}

/*
 * marie_bm_test - is @pfn tracked? Lock-free single-word read.
 * Returns false when @bm->l1 is unallocated.
 */
static inline bool marie_bm_test(const struct marie_bitmap *bm,
				 unsigned long pfn)
{
	if (!bm->l1 || pfn >= max_pfn)
		return false;
	return test_bit(pfn, bm->l1);
}

/*
 * marie_bm_drop_l2_range - bulk-clear all L1 / L2 / counter state
 * for the L2 range identified by @l2bit. Used when recycling one
 * range of a bitmap (precise, touches the L1 words covered by the
 * range as well).
 *
 * Caller must guarantee no concurrent set/clear on @bm for the
 * affected PFN range (try_advance_head fences via head_gen cmpxchg).
 */
void marie_bm_drop_l2_range(struct marie_bitmap *bm, unsigned int l2bit);

/*
 * marie_bm_reset - reset @bm to fully empty: L1 cleared, L2 cleared,
 * all l2_count cells zeroed.
 *
 * L1 must be cleared too: leaving stale L1 bits and resetting only
 * L2 + l2_count would let a subsequent marie_bm_set(@pfn) on a
 * different PFN in a stale-set L1 word leave that stale bit visible
 * to the scanner (which now sees the just-set L2 bit and enters the
 * range). Worse, a later marie_bm_clear() on the stale PFN would
 * dec the l2_count below zero, corrupting the refcount invariant.
 *
 * Used by try_advance_head when recycling a (type, gen, tier) slot
 * for the next ring cycle. Caller must fence subsequent installs
 * (head_gen cmpxchg in try_advance_head's case) so no install can
 * target @bm until the reset is visible.
 */
void marie_bm_reset(struct marie_bitmap *bm);

/*
 * L2 range coordination locks: 512 spinlocks (one per L2 bit, ~32 KiB
 * total), used by scanners to claim exclusive ownership of a PFN
 * range for the duration of their L1 walk in that range.
 *
 * Shared by ALL marie_bitmap instances: the lock is over the PFN
 * address space, not the bitmap instance. Two scanners walking
 * different (type, gen, tier) bitmaps in the same L2 range still
 * serialise via the same lock, avoiding wasted parallel L1 fetches
 * of the same physical cachelines.
 *
 * The storage is exposed (rather than wrapped in opaque accessors)
 * so the per-bit trylock / unlock can be static inline in this
 * header -- scanners take them once per processed L2 bit, which is
 * hot enough that a function-call wrapper costs measurable cycles.
 * Callers must guarantee @l2bit < MARIE_L2_BITS (always true for
 * indices produced by __ffs on an L2 word).
 */
struct marie_bm_range_lock {
	spinlock_t lock;
} ____cacheline_aligned_in_smp;

extern struct marie_bm_range_lock marie_bm_range_locks[MARIE_L2_BITS];

void marie_bm_range_locks_init(void);

static inline bool marie_bm_range_trylock(unsigned int l2bit)
{
	return spin_trylock(&marie_bm_range_locks[l2bit].lock);
}

static inline void marie_bm_range_unlock(unsigned int l2bit)
{
	spin_unlock(&marie_bm_range_locks[l2bit].lock);
}

#endif	/* _MM_LRU_MARIE_BITMAP_H */
