/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _MM_LRU_MARIE_PFN_INSTALL_H
#define _MM_LRU_MARIE_PFN_INSTALL_H

#include <linux/atomic.h>
#include <linux/memcontrol.h>
#include <linux/mm.h>
#include <linux/mm_inline.h>

#include "bitmap.h"
#include "state.h"

/*
 * Marie's "publish a PFN as TRACKED" primitive, factored out of the
 * install/split paths.
 *
 * What it writes (the single source of truth for "Marie owns this PFN"):
 *   - marie_state[pfn]: TRACKED | (gen) | (tier) | (type) | (zone)
 *   - marie_track_bm[type][gen][tier]: scan bit for this PFN (a single
 *     global per-(type,gen,tier) plane)
 *   - marie_gen_occ_inc(pfn, gen, type): the single gen-occupancy choke-point
 *     (state.h), shared with every other transition site; @pfn also drives
 *     the Marie defrag per-pageblock occupancy mirror
 *
 * What it deliberately does NOT touch:
 *   - page->flags (PG_active / PG_lru) -- the install path flips these
 *     in one atomic mask write after publish; the split path's caller
 *     sets PG_lru later.
 *   - page->lru list pointers -- INIT_LIST_HEAD vs list_add_tail differs
 *     between install and split.
 *   - marie_nr_pages and vmstat lru_size -- accounted by the caller (or
 *     by marie_page_install for the fresh-install path).
 *   - marie_gen_installs -- the global install "throttle" counter that
 *     drives gen advance; split intentionally does NOT bump it because
 *     the split tail inherits its parent's install budget (the parent
 *     was already counted at fault-install).
 *
 * Caller context: lru_lock held with IRQs off. The publish is a plain
 * non-atomic byte write because lru_lock serialises every install on the
 * same PFN, and the "already TRACKED" early-out in marie_page_install
 * catches concurrent re-install attempts.
 */
static inline void marie_pfn_publish_inherit(struct page *f, int type,
					     u8 gen, u8 tier, int zone)
{
	unsigned long pfn = page_to_pfn(f);

	marie_state[pfn] = MARIE_PFN_TRACKED |
		(gen << MARIE_PFN_GEN_SHIFT) |
		(tier << MARIE_PFN_TIER_SHIFT) |
		(type ? MARIE_PFN_TYPE_FILE : 0) |
		marie_pfn_zone_bits(zone);
	if (marie_bm_set(&marie_track_bm[type][gen][tier], pfn))
		marie_gen_occ_inc(pfn, gen, type);
}

/*
 * marie_pfn_publish_isolated - publish a PFN as TRACKED in the *isolated*
 * state: the byte only, with NO scan bit and NO gen_occupied increment.
 *
 * This mirrors exactly the residual state marie_evict_counters_only leaves
 * on an isolated page (TRACKED byte retained for the install-race early-out;
 * scan bit + gen_occupied already retired; nr_pages + lru_size already
 * debited). It is the correct publish for a reclaim-split THP tail.
 *
 * The parent THP was isolated via marie_evict_counters_only (occ + bit
 * retired, nr_pages/lru_size debited) BEFORE shrink_page_list split it, so
 * each child must enter the isolated state, NOT a fresh install:
 *   - a child that is reclaimed frees with no bit -> nothing to retire
 *     (free/uncharge are gated on the bit), net-zero, matching the parent's
 *     already-debited aggregate;
 *   - a child that survives is balanced EXACTLY ONCE by the putback path
 *     (marie_state_publish_at_gen sets bit + occ; marie_account_install_isolate
 *     adds nr_pages + lru_size) -- the same balance as a non-split isolated
 *     survivor.
 *
 * Using marie_pfn_publish_inherit here instead publishes the tail in the
 * fresh-install state (bit + occ + nr_pages), so a surviving tail is
 * counted twice (split inc + putback inc) but retired once at free, leaking
 * +1 gen_occupied (and +1 nr_pages) per surviving tail. After the tails
 * free, that residue is phantom gen_occupied>0 / bit=0 occupancy: find_oldest
 * keeps returning the phantom gen, the scanner finds no bit there, anon
 * reclaim makes zero progress, and head-advance is wedged (its gate is
 * gen_occupied[next]==0). THP-only (order-0 never splits) and persists across
 * OOM until a reboot zeroes the global gen ring -- the "first tail run
 * reclaims, every retry stalls at swapout onset" freeze.
 *
 * Caller context: identical to marie_pfn_publish_inherit (split path holds
 * the per-type lock; the tail is exclusively owned, off-LRU, PG_lru clear).
 */
static inline void marie_pfn_publish_isolated(struct page *f, int type,
					      u8 gen, u8 tier, int zone)
{
	unsigned long pfn = page_to_pfn(f);

	marie_state[pfn] = MARIE_PFN_TRACKED |
		(gen << MARIE_PFN_GEN_SHIFT) |
		(tier << MARIE_PFN_TIER_SHIFT) |
		(type ? MARIE_PFN_TYPE_FILE : 0) |
		marie_pfn_zone_bits(zone);
}

/*
 * marie_page_install - the unified fresh-install path.
 *
 * Single entry point that replaces the former marie_install_local /
 * marie_install_locked pair. Both call sites (lru_marie_add_page for THP
 * via per-type lock + small page direct, and marie_change_state_lruvec
 * during gate-on fill) now route here. The per-type lock context that
 * used to distinguish "locked" from "local" is the caller's concern, not
 * this function's: the body only requires lru_lock + IRQs off and uses
 * the same publish + flag flip + account sequence in both cases.
 *
 * Sequence:
 *   1. TRACKED early-out (returns false). Defends against gate-flip race
 *      and reclaim-survivor re-install (TRACKED is preserved across
 *      isolate by design; marie_state_publish_at_gen handles the
 *      survivor putback separately, never this function).
 *   2. Capture (PG_active, PG_workingset) -> 2-bit tier signal.
 *   3. Clear PG_active early; the final flag write is still a single
 *      atomic set_mask_bits, but capturing was_active before the clear
 *      keeps the tier value coherent with the byte we publish below.
 *   4. INIT_LIST_HEAD(&f->lru) -- a recycled page arrives with
 *      LIST_POISON{1,2} that would later fault list_del_init.
 *   5. Publish per-PFN state via marie_pfn_publish_inherit.
 *   6. Bump marie_gen_installs[type] and advance the head at
 *      marie_gen_growth_live[type]. Split path skips this bump
 *      (publish_inherit only).
 *   7. set_mask_bits(PG_active->0, PG_lru->1) -- one atomic flag write.
 *      Ordered AFTER step 5 so a concurrent __page_cache_release
 *      observing PG_lru=1 also observes TRACKED=1.
 *   8. Account (marie_nr_pages + vmstat lru_size, i.e. NR_LRU_BASE /
 *      NR_ZONE_LRU_BASE) via marie_account_install.
 *
 * Returns true on success, false on TRACKED early-out.
 */
bool marie_page_install(struct page *f);

#endif /* _MM_LRU_MARIE_PFN_INSTALL_H */
