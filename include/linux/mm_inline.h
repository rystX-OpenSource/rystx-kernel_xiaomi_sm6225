/* SPDX-License-Identifier: GPL-2.0 */
#ifndef LINUX_MM_INLINE_H
#define LINUX_MM_INLINE_H

#include <linux/huge_mm.h>
#include <linux/lru_marie.h>
#include <linux/swap.h>

/**
 * page_is_file_cache - should the page be on a file LRU or anon LRU?
 * @page: the page to test
 *
 * Returns 1 if @page is page cache page backed by a regular filesystem,
 * or 0 if @page is anonymous, tmpfs or otherwise ram or swap backed.
 * Used by functions that manipulate the LRU lists, to sort a page
 * onto the right LRU list.
 *
 * We would like to get this info without a page flag, but the state
 * needs to survive until the page is last deleted from the LRU, which
 * could be as far down as __page_cache_release.
 */
static inline int page_is_file_cache(struct page *page)
{
	return !PageSwapBacked(page);
}

static __always_inline void __update_lru_size(struct lruvec *lruvec,
				enum lru_list lru, enum zone_type zid,
				int nr_pages)
{
	struct pglist_data *pgdat = lruvec_pgdat(lruvec);

	/*
	 * Upstream relaxes a lockdep_assert_held(&lruvec->lru_lock) here when
	 * Marie is enabled: Marie's reclaim isolate path
	 * (marie_evict_counters_only) and its deferred post-reclaim teardown
	 * (marie_state_drop_pfn_after_reclaim via marie_state_shrink_lruvec)
	 * intentionally run this without lru_lock -- install/evict serialise
	 * via marie_state[pfn]'s TRACKED bit and TestClearPageLRU, and the
	 * per-CPU vmstat helpers below are preempt-off-safe on their own.
	 *
	 * This tree's __update_lru_size carries no such assertion (the
	 * assertion was added with the per-lruvec lru_lock in 5.11, commit
	 * 6168d0da2b47), so there is nothing to relax and the hunk collapses
	 * to nothing.
	 */
	__mod_node_page_state(pgdat, NR_LRU_BASE + lru, nr_pages);
	__mod_zone_page_state(&pgdat->node_zones[zid],
				NR_ZONE_LRU_BASE + lru, nr_pages);
}

static __always_inline void update_lru_size(struct lruvec *lruvec,
				enum lru_list lru, enum zone_type zid,
				int nr_pages)
{
	__update_lru_size(lruvec, lru, zid, nr_pages);
#ifdef CONFIG_MEMCG
	mem_cgroup_update_lru_size(lruvec, lru, zid, nr_pages);
#endif
}

static __always_inline void add_page_to_lru_list(struct page *page,
				struct lruvec *lruvec, enum lru_list lru)
{
#ifdef CONFIG_LRU_MARIE
	if (lru_marie_add_page(lruvec, page, false))
		return;

	/*
	 * If Marie is enabled, lru_marie_add_page failed only due to the
	 * per-PFN state array being absent for this PFN (hotplug racing the
	 * array grow).  Fall through to the legacy LRU lists: shrink_lruvec
	 * runs legacy reclaim alongside Marie specifically to drain these
	 * orphans.  This tree has no MGLRU, so the upstream hunk's
	 * lru_gen_add_folio() fallback has no analogue here.
	 */
#endif
	update_lru_size(lruvec, lru, page_zonenum(page), hpage_nr_pages(page));
	list_add(&page->lru, &lruvec->lists[lru]);
}

static __always_inline void add_page_to_lru_list_tail(struct page *page,
				struct lruvec *lruvec, enum lru_list lru)
{
#ifdef CONFIG_LRU_MARIE
	/* See add_page_to_lru_list() — Marie failure falls to legacy. */
	if (lru_marie_add_page(lruvec, page, true))
		return;
#endif
	update_lru_size(lruvec, lru, page_zonenum(page), hpage_nr_pages(page));
	list_add_tail(&page->lru, &lruvec->lists[lru]);
}

static __always_inline void del_page_from_lru_list(struct page *page,
				struct lruvec *lruvec, enum lru_list lru)
{
#ifdef CONFIG_LRU_MARIE
	if (lru_marie_del_page(lruvec, page, false))
		return;
#endif
	list_del(&page->lru);
	update_lru_size(lruvec, lru, page_zonenum(page), -hpage_nr_pages(page));
}

/**
 * page_lru_base_type - which LRU list type should a page be on?
 * @page: the page to test
 *
 * Used for LRU list index arithmetic.
 *
 * Returns the base LRU type - file or anon - @page should be on.
 */
static inline enum lru_list page_lru_base_type(struct page *page)
{
	if (page_is_file_cache(page))
		return LRU_INACTIVE_FILE;
	return LRU_INACTIVE_ANON;
}

/**
 * page_off_lru - which LRU list was page on? clearing its lru flags.
 * @page: the page to test
 *
 * Returns the LRU list a page was on, as an index into the array of LRU
 * lists; and clears its Unevictable or Active flags, ready for freeing.
 */
static __always_inline enum lru_list page_off_lru(struct page *page)
{
	enum lru_list lru;

	if (PageUnevictable(page)) {
		__ClearPageUnevictable(page);
		lru = LRU_UNEVICTABLE;
	} else {
		lru = page_lru_base_type(page);
		if (PageActive(page)) {
			__ClearPageActive(page);
			lru += LRU_ACTIVE;
		}
	}
	return lru;
}

/**
 * page_lru - which LRU list should a page be on?
 * @page: the page to test
 *
 * Returns the LRU list a page should be on, as an index
 * into the array of LRU lists.
 */
static __always_inline enum lru_list page_lru(struct page *page)
{
	enum lru_list lru;

	if (PageUnevictable(page))
		lru = LRU_UNEVICTABLE;
	else {
		lru = page_lru_base_type(page);
		if (PageActive(page))
			lru += LRU_ACTIVE;
	}
	return lru;
}

/*
 * Backported from mainline __folio_clear_lru_flags(), rewritten in struct page
 * terms for this tree. 4.19 folds the flag clearing into page_off_lru(), which
 * also returns the lru index; Marie needs the clear as a separate step because
 * its callers have already obtained (or already consumed) the index.
 */
static __always_inline void __clear_page_lru_flags(struct page *page)
{
	__ClearPageLRU(page);

	/* this shouldn't happen, so leave the flags to bad_page() */
	if (PageActive(page) && PageUnevictable(page))
		return;

	__ClearPageActive(page);
	__ClearPageUnevictable(page);
}

#define lru_to_page(head) (list_entry((head)->prev, struct page, lru))

#endif
