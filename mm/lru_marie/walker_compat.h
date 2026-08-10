/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _MM_LRU_MARIE_WALKER_COMPAT_H
#define _MM_LRU_MARIE_WALKER_COMPAT_H

/*
 * Per-kernel-version adaptation layer for the page-table walker (walker.c).
 *
 * Marie's core sources are meant to be byte-identical across every kernel it is
 * ported to (currently 6.12 / 6.18 / 7.0 / 7.1).  The only core code that
 * genuinely has to differ per version is a handful of in-tree mm APIs that
 * changed signature; the walker's share of those is isolated here behind
 * uniform names (the reclaim side lives in state_compat.h), so producing a
 * per-version patch re-touches just these small headers plus the unavoidable
 * context lines of the integration hunks -- never the bulk of the core.
 *
 * Include AFTER "../internal.h" (and the usual mm headers): the wrappers are
 * static inline and need the underlying declarations + struct types complete.
 *
 * Version boundaries below match the four supported targets exactly; revisit
 * them when adding a new target kernel.
 */

#include <linux/version.h>
#include <linux/mmu_notifier.h>	/* ptep_clear_young_notify (young-ptes shim) */

/*
 * PMD stability guard before mapping a PTE table.
 *
 * pte_offset_map_lock() only gained its "return NULL when @pmd is no longer a
 * page table" contract in 6.5, and Marie's walker leans on it as its THP /
 * migration-entry / cleared-pmd guard:
 *
 *	pte_table = pte_offset_map_lock(...);
 *	if (!pte_table)
 *		return 0;
 *
 * On targets predating that contract the call cannot fail, and this tree's
 * walk_pmd_range() hands ->pmd_entry() every non-none pmd *including* huge
 * ones (it only splits when ->pte_entry is also set, which Marie does not
 * provide).  Mapping a huge pmd as a 512-entry PTE array would make the
 * walker read the THP's own data as page-table entries and then run
 * ptep_test_and_clear_young() over it, so the check has to be made explicitly
 * here instead.
 *
 * pmd_none_or_trans_huge_or_clear_bad() (via pmd_trans_unstable) is the
 * era-correct spelling: it re-reads the pmd atomically, rejects none / huge /
 * migration-entry pmds, and clears-and-rejects a bad one.  It compiles to 0
 * when THP is off, where a populated pmd cannot turn huge underneath us, so
 * the explicit pmd_none() is kept alongside for that configuration.
 *
 * Once this returns false the pmd is a real page table and stays one for the
 * duration of the walk: the walker holds mmap_read_lock, and every path that
 * retracts a page table (khugepaged collapse, munmap) needs mmap_write_lock.
 * A fault can only install a THP over a *none* pmd, which this has excluded.
 */
#if LINUX_VERSION_CODE < KERNEL_VERSION(6, 5, 0)
static inline bool marie_pmd_scan_unstable(pmd_t *pmd)
{
	return pmd_none(*pmd) || pmd_trans_unstable(pmd);
}
#else
static inline bool marie_pmd_scan_unstable(pmd_t *pmd)
{
	return false;	/* pte_offset_map_lock()'s NULL return covers this */
}
#endif

/*
 * Look-around neighbour batching.  6.12 has folio_pte_batch() with the long
 * (max_nr, FPB flags, out-params) signature; 6.18+ replaced it with
 * folio_pte_batch_flags(page, vma, ptep, &pte, max_nr, FPB_*).  Both collapse
 * a run of present PTEs mapping @page while ignoring young/dirty differences:
 *   - 6.12: FPB_MERGE_YOUNG_DIRTY does not exist, but folio_pte_batch()
 *     pte_mkold()s before comparing, so FPB_IGNORE_DIRTY alone gives the same
 *     young/dirty-agnostic batching.
 *   - 6.18+: FPB_MERGE_YOUNG_DIRTY merges across young/dirty directly.
 * 4.19 predates the helper entirely; page_pte_batch() in mm/internal.h is the
 * 6.19.8 implementation rewritten in struct page terms, so it takes the 6.18+
 * argument set.  The uniform wrapper takes the 6.12 argument set (@addr is
 * unused on 4.19 and 6.18+).
 */
static inline int
marie_page_pte_batch(struct page *page, unsigned long addr, pte_t *ptep,
		      pte_t pte, unsigned int max_nr)
{
#if LINUX_VERSION_CODE < KERNEL_VERSION(6, 0, 0)
	return page_pte_batch(page, ptep, &pte, max_nr, FPB_MERGE_YOUNG_DIRTY);
#elif LINUX_VERSION_CODE >= KERNEL_VERSION(6, 18, 0)
	return folio_pte_batch_flags(page, NULL, ptep, &pte, max_nr,
				     FPB_MERGE_YOUNG_DIRTY);
#else
	return folio_pte_batch(page, addr, ptep, pte, max_nr, FPB_IGNORE_DIRTY,
			       NULL, NULL, NULL);
#endif
}

/*
 * arch_enter/leave_lazy_mmu_mode() were renamed to
 * lazy_mmu_mode_enable/disable() in 7.0.  Provide the new names on the older
 * targets so the walker can use one spelling.
 */
#if LINUX_VERSION_CODE < KERNEL_VERSION(7, 0, 0)
#define lazy_mmu_mode_enable()	arch_enter_lazy_mmu_mode()
#define lazy_mmu_mode_disable()	arch_leave_lazy_mmu_mode()
#endif

/*
 * test_and_clear_young_ptes_notify() -- the batched young-clear with
 * mmu-notifier callback -- landed in 7.1.  On older targets emulate it as a
 * per-PTE ptep_clear_young_notify() loop; functionally equivalent, losing only
 * the batched-notifier amortisation.
 */
#if LINUX_VERSION_CODE < KERNEL_VERSION(7, 1, 0)
static inline int marie_test_and_clear_young_ptes(struct vm_area_struct *vma,
						  unsigned long addr,
						  pte_t *pte, unsigned int nr)
{
	int young = 0;
	unsigned int i;

	for (i = 0; i < nr; i++)
		young |= ptep_clear_young_notify(vma, addr + i * PAGE_SIZE,
						 pte + i);
	return young;
}
#define test_and_clear_young_ptes_notify marie_test_and_clear_young_ptes
#endif

#endif /* _MM_LRU_MARIE_WALKER_COMPAT_H */
