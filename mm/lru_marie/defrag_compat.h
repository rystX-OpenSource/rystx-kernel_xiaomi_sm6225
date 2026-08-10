/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _MM_LRU_MARIE_DEFRAG_COMPAT_H
#define _MM_LRU_MARIE_DEFRAG_COMPAT_H

/*
 * Per-kernel-version adaptation layer for the defragmenter (defrag.c).
 *
 * Marie's core sources are meant to be byte-identical across every kernel it is
 * ported to (currently 6.12 / 6.18 / 7.0 / 7.1 / 7.2).  The only defrag code
 * that genuinely has to differ per version is the free-page allocation
 * contract that changed in 6.14; it is isolated here behind a uniform name so a
 * per-version patch never has to touch defrag.c itself.
 *
 * Include AFTER "../internal.h" (and the usual mm headers): the wrappers are
 * static inline and need post_alloc_hook()/set_page_refcounted() +
 * __GFP_MOVABLE declared.
 *
 * Version boundary below matches the supported targets; revisit when adding a
 * new target kernel.
 */

#include <linux/version.h>

/*
 * marie_defrag_prep_allocated - turn a harvested raw free page (refcount 0,
 * removed from the buddy allocator via __isolate_free_page()) into a
 * fully-allocated, refcounted page.  This is the free-pool analogue of stock
 * compaction's mark_allocated() and the post_alloc_hook() sequence in
 * compaction_alloc(): both the migration-target handout (alloc_target) and the
 * leftover-return-to-buddy path (freectx_release) need it.
 *
 * Up to 6.13 post_alloc_hook() itself set the refcount to 1.  6.14 moved that
 * to the callers (upstream: set_page_refcounted() split out of
 * post_alloc_hook(); compaction_alloc()/mark_allocated() gained an explicit
 * call).  On >= 6.14 we must therefore add set_page_refcounted(); on <= 6.13
 * doing so would double-set it and trip VM_BUG_ON_PAGE(page_ref_count) under
 * CONFIG_DEBUG_VM -- hence the gate.
 *
 * A refcount-0 page escaping this helper is not benign: as a migration target
 * it is freed mid-migration by lru_cache_add() (bad_page on a still-locked /
 * just-remapped page, refcount underflow, live-anon corruption); on the
 * release path __free_pages()' put_page_testzero() never fires, leaking the
 * pool back-pressure.
 */
static inline void marie_defrag_prep_allocated(struct page *page, unsigned int order)
{
	post_alloc_hook(page, order, __GFP_MOVABLE);
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 14, 0)
	set_page_refcounted(page);
#endif
}

/*
 * pgdat->kcompactd_highest_zoneidx was named kcompactd_classzone_idx before
 * 5.9 (upstream commit 97a225e69a1f "mm/page_alloc: integrate classzone_idx
 * and high_zoneidx", which renamed the field along with the rest of the
 * classzone_idx -> highest_zoneidx sweep).  Only the spelling changed: it is
 * the same enum zone_type carrying the highest zone index kcompactd was asked
 * to compact for, written under pgdat->kcompactd_wait's lock in both eras
 * (this tree: mm/compaction.c:2557/2617/2629/2663).
 *
 * Marie reads it unsynchronised in marie_defrag_drop_need(), exactly as
 * kcompactd_do_work() does at mm/compaction.c:2509 -- a stale index only
 * mis-sizes one DROP budget estimate for one pass, which the next pass
 * recomputes.
 */
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 9, 0)
#define kcompactd_highest_zoneidx	kcompactd_classzone_idx
#endif

#endif /* _MM_LRU_MARIE_DEFRAG_COMPAT_H */
