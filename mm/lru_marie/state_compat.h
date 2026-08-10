/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _MM_LRU_MARIE_STATE_COMPAT_H
#define _MM_LRU_MARIE_STATE_COMPAT_H

/*
 * Per-kernel-version adaptation layer for the reclaim state machine (state.c).
 *
 * Marie's core sources are meant to be byte-identical across every kernel it is
 * ported to (currently 4.19 / 6.12 / 6.18 / 7.0 / 7.1).  The only core code that
 * genuinely has to differ per version is a handful of in-tree mm APIs that
 * changed signature; the reclaim side's share of those is isolated here behind
 * uniform names (the walker side lives in walker_compat.h), so producing a
 * per-version patch re-touches just these small headers plus the unavoidable
 * context lines of the integration hunks -- never the bulk of the core.
 *
 * Include AFTER "../internal.h" and the usual mm headers (in particular
 * <linux/vmstat.h> and <linux/memcontrol.h>): the wrappers are static inline
 * and need the underlying declarations + struct types complete.
 *
 * Version boundaries below match the five supported targets exactly; revisit
 * them when adding a new target kernel.
 *
 * The 4.19 branches carry the bulk of the adaptation: 4.19 predates the folio
 * conversion entirely, so the whole core is written in struct page terms (see
 * the port notes in include/linux/lru_marie.h) and the handful of reclaim
 * helpers that only exist in folio spelling are reproduced here.
 */

#include <linux/version.h>

#if LINUX_VERSION_CODE < KERNEL_VERSION(6, 0, 0)
#include <linux/backing-dev.h>	/* congestion_wait */
#include <linux/gfp.h>
#include <linux/memcontrol.h>
#include <linux/mm_inline.h>
#include <linux/swap.h>
#endif

/*
 * page->flags became the typed memdesc_flags_t -- struct { unsigned long f; }
 * -- in 6.18 (upstream series "Add and use memdesc_flags_t", first commit
 * 53fbef56e07d; <6.18 is a plain unsigned long, so this boundary is exact, not
 * just target-derived).  The macro yields the raw "unsigned long" lvalue on
 * every version, so both a read (`MARIE_FOLIO_FLAGS(f) & MASK`) and a bit op
 * (`set_mask_bits(&MARIE_FOLIO_FLAGS(f), ...)`) are version-agnostic.
 *
 * Marie touches the raw word in only the two narrow spots the page-flag
 * accessors do not cover: clearing stale LRU_GEN/LRU_REFS residue and the
 * single atomic PG_active->0 + PG_lru->1 publish.
 */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 18, 0)
#define MARIE_FOLIO_FLAGS(page)	((page)->flags.f)
#else
#define MARIE_FOLIO_FLAGS(page)	((page)->flags)
#endif

/*
 * shrink_page_list() gained a trailing @memcg parameter in 6.18 (the scan is
 * told which memcg it is reclaiming for).  Marie always has the memcg in hand
 * at the call site, so the uniform wrapper takes it unconditionally and simply
 * does not forward it on the pre-6.18 signature.
 *
 * Symmetrically, kernels this old still carry a leading-side @ttu_flags
 * argument (folded into the try_to_unmap() flags at the unmap site), which
 * upstream dropped once TTU_IGNORE_ACCESS became unconditional and the
 * parameter lost its last non-zero caller.  Only
 * reclaim_clean_pages_from_list() ever passed something non-zero; this tree's
 * reclaim path (shrink_inactive_list) passes 0, so pass 0 here too and Marie's
 * isolate->shrink loop unmaps exactly the way the legacy reclaim it replaces
 * does.
 *
 * This tree also returns unsigned long rather than upstream's unsigned int.
 * The wrapper keeps the upstream return type -- the value is the reclaimed
 * count of one isolate batch (bounded by the isolation limit, far below
 * 2^32), so the narrowing cannot lose anything, and state.c's caller stays
 * byte-identical across every target.
 */
static inline unsigned int
marie_shrink_page_list(struct list_head *page_list, struct pglist_data *pgdat,
			struct scan_control *sc, struct reclaim_stat *stat,
			bool ignore_references, struct mem_cgroup *memcg)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 18, 0)
	return shrink_page_list(page_list, pgdat, sc, stat, ignore_references,
				 memcg);
#elif LINUX_VERSION_CODE < KERNEL_VERSION(5, 11, 0)
	return shrink_page_list(page_list, pgdat, sc, 0, stat,
				ignore_references);
#else
	return shrink_page_list(page_list, pgdat, sc, stat, ignore_references);
#endif
}

/*
 * mem_cgroup_lruvec() swapped its two arguments in 5.11 (upstream commit
 * a984226f457f "mm: memcontrol: remove the pgdat parameter of
 * mem_cgroup_page_lruvec" and its neighbours reordered the pair to
 * (memcg, pgdat) so the memcg reads first).  4.19 still spells it
 * (pgdat, memcg) -- include/linux/memcontrol.h:369 with CONFIG_MEMCG=y and
 * :904 with =n, both orders identical -- and every in-tree caller here passes
 * it that way, so the flip cannot be papered over by touching the callee.
 *
 * Marie has exactly one call site (lru_marie_uncharge_backstop), so the
 * wrapper keeps the 6.x argument order and reverses it on the older
 * signature, leaving state.c byte-identical across targets.
 */
static inline struct lruvec *marie_mem_cgroup_lruvec(struct mem_cgroup *memcg,
						     struct pglist_data *pgdat)
{
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 11, 0)
	return mem_cgroup_lruvec(pgdat, memcg);
#else
	return mem_cgroup_lruvec(memcg, pgdat);
#endif
}

/*
 * WORKINGSET_REFAULT split into per-type WORKINGSET_REFAULT_ANON /
 * WORKINGSET_REFAULT_FILE in 5.9 (upstream commit 170b04b7ae49 "mm/workingset:
 * prepare the workingset detection infrastructure for anon LRU").  Before that
 * there is a single unsplit WORKINGSET_REFAULT node stat (4.19
 * include/linux/mmzone.h:190) which counts exactly the file refaults -- 4.19
 * has no anon workingset detection at all, so nothing else was ever folded
 * into it and the old counter is a precise stand-in for the FILE half rather
 * than an approximation of it.
 *
 * Marie reads only the FILE counter (the file-refault term of its aging
 * feedback), so the ANON spelling is deliberately not mapped: it has no
 * meaning on this tree and a stand-in would silently feed anon numbers that
 * 4.19 never collects.
 */
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 9, 0)
#define WORKINGSET_REFAULT_FILE	WORKINGSET_REFAULT
#endif

/*
 * Reclaim-counter accounting.
 *
 * 7.1 relocated the PGSTEAL, PGSCAN, PGDEMOTE and PGREFILL counters out of
 * enum vm_event_item into enum node_stat_item (they are per-memcg lruvec stats
 * now).  Route the post-isolation (PGSCAN) and post-reclaim (PGSTEAL) bumps
 * through whichever API the building kernel exposes.  @base is the
 * PG{SCAN,STEAL}_KSWAPD reclaimer-offset base, @per_type the PG{SCAN,STEAL}_ANON
 * counter indexed by the anon/file @type; both are passed as int so the call
 * site is identical whichever enum they live in.
 */
static inline void marie_account_reclaim(struct lruvec *lruvec,
					 struct scan_control *sc,
					 int base, int per_type,
					 int type, unsigned long nr)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(7, 1, 0)
	/* node_stat_item: one lruvec update folds node vmstat + memcg stat. */
	mod_lruvec_state(lruvec, base + vmscan_reclaimer_offset(sc), nr);
	mod_lruvec_state(lruvec, per_type + type, nr);
#else
	/* vm_event_item: global (skipped for cgroup reclaim) + memcg + type. */
	int item = base + vmscan_reclaimer_offset(sc);

	if (!sc_cgroup_reclaim(sc))
		count_vm_events(item, nr);
	count_memcg_events(lruvec_memcg(lruvec), item, nr);
	count_vm_events(per_type + type, nr);
#endif
}

/*
 * ---------------------------------------------------------------------------
 * 4.19-only shims.
 *
 * Each of these is either a rename that happened between 4.19 and 6.12 or a
 * helper introduced after 4.19; the bodies are taken from 6.19.8 and expressed
 * in this tree's primitives.  Guarded < 6.0 so the 6.x/7.x targets keep using
 * the in-tree versions untouched.
 * ---------------------------------------------------------------------------
 */
#if LINUX_VERSION_CODE < KERNEL_VERSION(6, 0, 0)

/*
 * gfp_has_io_fs() -- from 6.19.8 include/linux/gfp.h (upstream commit
 * f5e64e1b0c4a).  Verbatim; the __GFP_IO / __GFP_FS flags predate 4.19.
 */
#ifndef gfp_has_io_fs
static inline bool gfp_has_io_fs(gfp_t gfp)
{
	return (gfp & (__GFP_IO | __GFP_FS)) == (__GFP_IO | __GFP_FS);
}
#endif

/*
 * can_reclaim_anon_pages() -- 6.19.8 mm/vmscan.c:386, static there and so not
 * reachable from this subdirectory.  Unlike the other shims in this file it is
 * NOT reproduced here: it has to live in vmscan.c on this target.
 *
 * Upstream's body only asks the two swap-space questions plus can_demote(),
 * all of which are reachable from here.  But 4.19 puts a third condition in
 * front of them.  Upstream splits the anon-reclaimability test across two
 * sites -- get_scan_count() checks sc->may_swap separately (cachy
 * vmscan.c:2569, "!sc->may_swap || !can_reclaim_anon_pages(...)"), so the
 * helper itself does not repeat it -- whereas 4.19's get_scan_count() folds
 * both into one condition (mm/vmscan.c:2573, "!sc->may_swap ||
 * mem_cgroup_get_nr_swap_pages(memcg) <= 0").  Marie calls the helper alone,
 * with no separate may_swap check at its call site, so on this tree the
 * may_swap arm must be inside the helper or it is lost entirely -- and
 * scan_control is private to vmscan.c, so only a definition there can read it.
 *
 * Hence the sole provider is vmscan.c's non-static
 * vmscan_can_reclaim_anon_pages() (declared in ../internal.h), synthesized
 * alongside reclaimer_offset() and cgroup_reclaim() under CONFIG_LRU_MARIE.
 * A static inline copy here would additionally collide with that declaration.
 *
 * The demotion tail is dropped in that definition: 4.19 has no tiered-memory
 * demotion, so an unswappable page is simply unreclaimable, which is exactly
 * what can_demote() returns on a single-tier node upstream.
 */

/*
 * reclaim_throttle() -- 6.x's typed reclaim stall (upstream commit
 * d818fca1cac3 "mm/vmscan: throttle reclaim when no progress is being made")
 * replaced the untyped congestion_wait()/wait_iff_congested() pair that 4.19
 * uses.  Marie only ever raises VMSCAN_THROTTLE_ISOLATED, which is precisely
 * the site 4.19's too_many_isolated() caller answers with
 * congestion_wait(BLK_RW_ASYNC, HZ/10) in mm/vmscan.c:2919 -- so that is what
 * the shim does, and the reason enum vmscan_throttle_state is not backported.
 *
 * congestion_wait() is an uninterruptible sleep, matching upstream: the caller
 * re-checks marie_too_many_isolated() and its own fatal_signal_pending() after
 * returning.
 */
enum { VMSCAN_THROTTLE_ISOLATED };

static inline void reclaim_throttle(struct pglist_data *pgdat, int reason)
{
	congestion_wait(BLK_RW_ASYNC, HZ / 10);
}

/*
 * The remaining folio-era reclaim helpers Marie leans on are not shimmed here:
 * they belong to the wider tree rather than to reclaim, so they were backported
 * into their upstream homes under this tree's page spelling --
 * page_isolate_lru() in mm/internal.h (upstream mm/internal.h:538), and
 * page_memcg() / page_lruvec() / page_lruvec_relock_irqsave() in
 * <linux/memcontrol.h> (upstream :447 / :739 / :1507).
 */

#endif /* < 6.0 */

#endif /* _MM_LRU_MARIE_STATE_COMPAT_H */
