/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _MM_LRU_MARIE_SIMD_H
#define _MM_LRU_MARIE_SIMD_H

/*
 * Marie SIMD-accelerated PTE scan.
 *
 * Primary entry point:
 *
 *      lru_marie_simd_young_pte_mask(pte_table, bitmap);
 *
 * On x86 it wraps one PMD's scan in kernel_fpu_begin/end; per-PMD
 * begin/end is effectively free (kthreads skip the FPU save, the restore
 * is deferred to userspace return), so there is nothing to amortise
 * across PMDs. On arm64 the scan is NEON/ASIMD but the bracket cannot be
 * opened at this call site (the walker holds the PTE spinlock here and
 * kernel_neon_begin() may run softirqs) -- see the second entry point
 * below and the header of mm/lru_marie/simd_arm64.c. On every other arch
 * the scan is a plain scalar pte_young loop with no FPU state.
 */

#include <linux/types.h>
#include <linux/bits.h>
#include <linux/jump_label.h>

#if defined(CONFIG_X86) || defined(CONFIG_ARM64)
/*
 * Runtime kill-switch for the boot-detected SIMD walker, exposed via
 * /sys/kernel/mm/lru_marie/simd. Default true: the walker uses the
 * widest SIMD kernel available for the arch -- on x86 whatever
 * arch_initcall could pick (AVX-512F > AVX2 > SSE2), on arm64 the single
 * mandatory ASIMD kernel. Writing 0 to the sysfs file flips the static
 * branch so the walker falls back to a pure scalar pte_young loop in the
 * same translation unit.
 *
 * Other arches use the generic scalar fallback already, so the toggle
 * does not need to exist there and the sysfs attribute is hidden.
 */
DECLARE_STATIC_KEY_TRUE(marie_simd_enabled_key);

static inline bool marie_simd_enabled(void)
{
	return static_branch_likely(&marie_simd_enabled_key);
}
#else
static inline bool marie_simd_enabled(void) { return false; }
#endif

#ifdef CONFIG_X86
/*
 * SIMD ISA cap (simd_max) accessors. Implemented in simd_x86.c, consumed by
 * the /sys/kernel/mm/lru_marie/simd_max knob in core.c. marie_simd_max_name()
 * returns the current cap ("avx512"/"avx2"/"sse2"); marie_simd_max_store()
 * parses a name, applies it (re-patching the scan static call), and returns 0
 * or -EINVAL.
 *
 * x86-only: arm64 has no ISA ladder to cap (ASIMD is architecturally
 * mandatory whenever FPSIMD is present), so core.c hides the knob there.
 */
const char *marie_simd_max_name(void);
int marie_simd_max_store(const char *buf);
#endif

/*
 * Number of unsigned longs needed to hold the young-bit bitmap for one
 * PMD's worth of PTEs (PTRS_PER_PTE = 512 on x86_64 and on 4K-page
 * arm64; the value is pulled from the arch's pgtable headers via the
 * caller's includes).
 */
#define MARIE_SIMD_PTE_BITMAP_LONGS	((512 + BITS_PER_LONG - 1) / BITS_PER_LONG)

/**
 * lru_marie_simd_young_pte_mask - scan one PMD's PTE array for young bits.
 * @table:  pointer to the first pte_t in the PMD's PTE array (512 entries)
 * @bitmap: output, MARIE_SIMD_PTE_BITMAP_LONGS unsigned longs.
 *
 * Called by the walker with the PTE spinlock held. On x86 it self-brackets
 * the scan in kernel_fpu_begin/end (or runs the scalar loop when the simd
 * knob is off); on arm64 it is the scalar path (the NEON bracket is opened
 * by the pre-lock hook below instead); on other arches it is a plain
 * scalar pte_young loop.
 */
void lru_marie_simd_young_pte_mask(const void *table, unsigned long *bitmap);

/**
 * lru_marie_simd_young_pte_mask_prelock - optional pre-PTL young-bit harvest.
 * @table:  pointer to the first pte_t in the PMD's PTE array (512 entries)
 * @bitmap: output, MARIE_SIMD_PTE_BITMAP_LONGS unsigned longs.
 *
 * Called by the walker BEFORE it takes the PTE spinlock, for arches whose
 * SIMD bracket is not permissible under a spinlock. Returns true if it
 * filled @bitmap (the caller must then NOT call
 * lru_marie_simd_young_pte_mask() for this PMD), false if it declined and
 * the caller should do the ordinary under-lock scan.
 *
 * The resulting bitmap is advisory: the walker re-validates every
 * candidate PTE under the PTL before acting on it, so a PTE that changes
 * between this scan and the lock is filtered there, exactly as it already
 * is on the under-lock path.
 *
 * Implemented on arm64 (NEON); a no-op returning false everywhere else,
 * which leaves the x86 and generic paths byte-identical to upstream.
 */
#ifdef CONFIG_ARM64
bool lru_marie_simd_young_pte_mask_prelock(const void *table,
					   unsigned long *bitmap);

/*
 * Does this arch want the walker to run the pre-PTL protocol at all?
 *
 * Compile-time constant so that on x86 and the generic build the entire
 * pre-lock block in marie_walk_pmd_range() -- including the extra
 * pte_offset_map()/pte_unmap() pair it needs to obtain the page-table base
 * before the lock -- is dead-code-eliminated. Those arches therefore emit
 * exactly the upstream instruction sequence: take the PTL, scan under it.
 */
static inline bool marie_simd_has_prelock(void) { return true; }
#else
static inline bool lru_marie_simd_young_pte_mask_prelock(const void *table,
							 unsigned long *bitmap)
{
	return false;
}

static inline bool marie_simd_has_prelock(void) { return false; }
#endif

#endif /* _MM_LRU_MARIE_SIMD_H */
