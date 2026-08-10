// SPDX-License-Identifier: GPL-2.0
/*
 * mm/lru_marie/core.c — Marie LRU.
 *
 * Multi-graded Adaptive Reclaim & Independent Eviction (Marie)
 *
 * Architecture in one paragraph (desktop/global-only -- no per-memcg
 * reclaim, ZERO per-memcg/per-lruvec state):
 *   - per-PFN state byte (marie_state[pfn]) as the single source of
 *     truth for every page's (TRACKED, type, zone, gen, tier) tuple;
 *     install / del / aging are single byte writes with no allocation
 *     in any fault-path operation
 *   - global per-type locks (marie_type_locks[]) guard the type's
 *     aging clock; accounting rides the node-global NR_LRU_BASE /
 *     NR_ZONE_LRU_BASE vmstats, so there is no per-lruvec carrier and
 *     no per-(lru, zone) shadow counter to bounce
 *   - per-pgdat walker driven from kswapd, with rmap-fed bloom
 *     feedback so PMD scans concentrate on hot regions
 *   - single global cycling per-type gen clock (MARIE_PFN_NR_GENS = 8)
 *     encoded in the per-PFN byte, advanced by global install cadence
 *     (marie_page_install counts installs onto the head gen and
 *     advances at marie_gen_growth_live[type])
 *   - SIMD PTE young-bit batch scan with boot-time AVX-512F / AVX2 /
 *     SSE2 dispatch on x86; scalar fallback on arm64 and elsewhere
 *
 * Core state (all global; no per-lruvec/per-memcg objects):
 *
 *   marie_state[]        — global per-PFN byte array (state.{h,c})
 *   marie_track_bm       — single global per-(type, gen, tier) track
 *                          bitmap plane the reclaim scan walks
 *   marie_head_gen[type] / marie_gen_occupied[gen][type] /
 *   marie_gen_installs[type] / marie_recycle_epoch
 *                        — the global per-type aging clock
 *   marie_type_locks[]   — global per-type lock (anon / file)
 *
 * Marie holds no per-memcg state: there is no lruvec carrier to look
 * up, allocate, free, or reparent, and nothing happens at any css
 * transition. cgroup charging stays on the stock path.
 *
 * Recommended userspace configuration:
 *   - vm.swappiness: nothing to set.  Marie clamps its effective
 *     swappiness to at most 1 in-kernel by default via low_swappiness_mode
 *     (a /sys/kernel/mm/lru_marie/ knob that defaults on), so the higher
 *     values udev rules, tuning daemons, or distro defaults install in
 *     vm.swappiness are ignored by the reclaim pick driver.  (The clamp
 *     only lowers, so a deliberate vm.swappiness=0 "never swap" is still
 *     honoured.)  This is why CONFIG_LRU_MARIE=y no longer overrides the
 *     vm_swappiness default.  The rationale for swappiness = 1 follows.
 *
 *     swappiness historically encoded the relative IO cost of swap
 *     vs. filesystem paging, on the assumption that file cache and
 *     anon working set carry comparable "hotness" distributions and
 *     comparable refault costs. That assumption was authored against
 *     spinning-disk-era hardware and no longer matches modern
 *     systems:
 *
 *       Storage type           File cache cost    Recommended
 *       -------------------    ----------------   -------------
 *       SSD+ZRAM (Modern)          Low            1 (Marie default)
 *       HDD (Slow,Unresponsive)    High           Higher (60+)
 *
 *     On modern desktops with NVMe-class file storage, lost file
 *     cache refaults in microseconds and is largely transparent to
 *     the user. ZRAM-backed swap, by contrast, is "free in RAM" only
 *     on the surface: every swapout/swapin pays compression CPU,
 *     L1/L2/L3 cache pollution from the codec working set, and
 *     blocks the calling context -- costs that are systematically
 *     hidden in IO accounting but ergonomically very visible as UI
 *     stutter and jank.
 *
 *     Worse, on a ZRAM-equipped system in normal steady state the
 *     pagecache typically fills physical memory. Any proportional
 *     anon eviction at that point disturbs the anon working set just
 *     to make room for what is mostly cold pagecache anyway --
 *     trading a transparent SSD refault on the file side for a
 *     visible ZRAM hit on the anon side. The cart goes before the
 *     horse.
 *
 *     swappiness = 1 captures the resulting policy precisely: anon
 *     is fully protected until the file pagecache falls below the
 *     clean_min_ratio floor, at which point swap engages as a true
 *     last resort. Marie's per-PFN reclaim driver maps this onto
 *     MARIE_PICK_FILE_THEN_ANON -- FILE scanned first, ANON engaged
 *     ONLY when skip_file is set inside marie_state_shrink_lruvec
 *     (i.e. the floor has been breached). Per-call transient FILE
 *     failures (empty oldest gen, all shrink_page_list rejects,
 *     etc.) do NOT leak into ANON -- the clean_min_ratio floor is
 *     the single depletion signal. The bias controller stays at
 *     zero throughout, because swappiness=1 short-circuits the
 *     proportional update path.
 *
 *     With low_swappiness_mode cleared, higher values (s = 2..199)
 *     remain useful on slower-storage systems where the file refault
 *     cost is no longer negligible; Marie honours them via the stubborn
 *     proportional controller in marie_swap_bias_update. s = 0 is a
 *     hard "never swap" -- reach OOM rather than touch anon (and, being
 *     the low extreme, is honoured even with low_swappiness_mode on).
 *     s = 200 is the symmetric "anon only" override, clamped to 1 while
 *     low_swappiness_mode is on. All are intentional user policy
 *     overrides; clean_min_ratio does not punch through them.
 *   - systemd-oomd OFF
 *     systemd-oomd reacts to PSI before Marie's clean_min_ratio
 *     floor + no-progress OOM path has a chance to stabilise
 *     reclaim. With Marie engaged the kernel-side OOM gate is more
 *     accurate, and userspace OOMD ends up killing tasks Marie
 *     would have rescued. Disable it (or leave its swap thresholds
 *     at 100%) for predictable behaviour.
 */

#define pr_fmt(fmt) "lru_marie: " fmt

#include <linux/cpu.h>
#include <linux/cpuhotplug.h>
#include <linux/hash.h>
#include <linux/init.h>
#include <linux/jiffies.h>
#include <linux/jump_label.h>
#include <linux/kobject.h>
#include <linux/list.h>
#include <linux/llist.h>
#include <linux/memblock.h>
#include <linux/memcontrol.h>
#include <linux/mm.h>
#include <linux/mm_inline.h>
#include <linux/mmzone.h>
#include <linux/pagewalk.h>
#include <linux/percpu.h>
#include <linux/printk.h>
#include <linux/rmap.h>
#include <linux/rwsem.h>
#include <linux/sched.h>
#include <linux/sched/mm.h>
#include <linux/sched/signal.h>
#include <linux/lru_marie.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/swap.h>
#include <linux/sysfs.h>
#include <linux/vmstat.h>
#include <linux/workqueue.h>
#include <linux/writeback.h>
#include <linux/xarray.h>

#include "../internal.h"	/* struct scan_control, shrink_page_list */
#include "defrag.h"
#include "pfn_install.h"
#include "simd.h"
#include "state.h"
#include "version.h"

#ifdef CONFIG_LRU_MARIE_DEFAULT_ON
DEFINE_STATIC_KEY_TRUE(lru_marie_enabled_key);
#else
DEFINE_STATIC_KEY_FALSE(lru_marie_enabled_key);
#endif
EXPORT_SYMBOL_GPL(lru_marie_enabled_key);

/*
 * Marie indexes its per-PFN state array by raw PFN. The implementation
 * caps max_pfn at 2^32 (= 4 KiB pages × 2^32 = 16 TiB of physical
 * address space, holes included): the per-PFN byte array would be at
 * most 4 GiB under that cap, and several other internal helpers
 * assume the PFN fits in 32 bits. On a box that violates this Marie
 * refuses to enable; Legacy / MGLRU continue to run unchanged.
 *
 * marie_pfn_unsupported is latched at subsys_initcall once max_pfn is
 * stable (set during setup_arch / memblock init) and read-only after
 * that, so the runtime cost is a single __read_mostly load.
 */
#define MARIE_MAX_SUPPORTED_PFN	(1UL << 32)
static bool marie_pfn_unsupported __read_mostly;

/*
 * ---------------------------------------------------------------------
 *  install path -- fully synchronous, no staging or pending queues
 * ---------------------------------------------------------------------
 *
 * lru_marie_add_page dispatches into marie_page_install (under the
 * per-type lock for THP, lock-free for small pages). That helper
 * publishes the per-PFN state byte, sets the global track bitmap,
 * bumps the global counters (marie_nr_pages + the node-global
 * NR_LRU_BASE / NR_ZONE_LRU_BASE vmstats), and sets PG_lru -- all under
 * the caller's lru_lock irqsave. No per-CPU staging, no session-end
 * flush hook, no wrapper allocation, no async drain, no kworker
 * dispatch.
 *
 * Walker tier promotion is similarly synchronous: when a young PTE
 * references a page whose tier is already saturated, the walker calls
 * marie_state_move_to_gen() directly on the per-PFN byte (no queueing).
 */

/*
 * ---------------------------------------------------------------------
 *  data structures (struct definitions live in mm/lru_marie/state.h)
 * ---------------------------------------------------------------------
 *
 * The global per-type aging clock (marie_head_gen, marie_gen_occupied,
 * marie_gen_installs, marie_recycle_epoch), the global track bitmap,
 * and the MARIE_PFN_NR_GENS / MARIE_NR_TIERS / MARIE_TIER_MAX /
 * MARIE_ISOLATE_BATCH constants live in state.{h,c}. Runtime-tunable
 * knobs (clean_min_ratio, walker intervals, etc.) live in the sysfs section at
 * the bottom of this file. Desktop/global-only: there are no per-lruvec
 * objects, so no lookup / alloc / free / memcg-teardown lifecycle.
 */

/*
 * Exported via mm/lru_marie/state.h for the install / evict helpers to
 * update during TRACKED 0<->1 transitions. percpu_counter
 * so per-page writes hit the local CPU's diff and only flush to the
 * global on every percpu_counter_batch ops; reads use
 * percpu_counter_sum (accurate, slower) in stats_show and
 * percpu_counter_read_positive (approximate, fast) where a hot
 * heuristic is good enough.
 */
struct percpu_counter marie_nr_pages;

/*
 * ---------------------------------------------------------------------
 *  generation lifecycle (helpers in mm/lru_marie/state.c)
 * ---------------------------------------------------------------------
 *
 * Marie keeps a cycling per-type gen ring of MARIE_PFN_NR_GENS (= 8)
 * slots, encoded directly in the per-PFN state byte. Aging is driven
 * by three signals:
 *
 *   - lru_marie_add_page always lands on the current head gen
 *     (atomic_read(&marie_head_gen[type])).
 *   - marie_state_isolate_scan_l2lock always pulls from the oldest
 *     occupied gen (marie_find_oldest_occupied).
 *   - shrink_page_list classifies each isolated page:
 *       FOLIOREF_RECLAIM  → freed (this page truly was cold)
 *       FOLIOREF_KEEP     → returned in page_list, putback re-routes
 *       FOLIOREF_ACTIVATE → ditto, with PG_active set
 *     putback re-installs the survivor at (oldest+1)&3 with
 *     target_tier = max(prev_tier, w_tier) via
 *     marie_install_at_gen.
 *   - head_gen advances per type via marie_try_advance_head_mlv (the
 *     _mlv suffix is historical -- the clock is a single global one per
 *     type), fired by global install cadence alone (marie_page_install
 *     checks marie_gen_installs[type] against marie_gen_growth_live[type];
 *     the old reclaim-time occupied<2 trigger was removed as it thrashed
 *     the ring). Advance is gated: the next slot must be fully empty
 *     (marie_gen_occupied[next][type] == 0).
 *
 * The reclaim cycle alone does not surface every hot page; a per-pgdat
 * SIMD walker (mm/lru_marie/walker.c) clears young PTEs and bumps
 * marie_state_inc_tier on tracked pages. When tier saturates, the
 * walker calls marie_state_move_to_gen() synchronously to move the
 * page into the head gen at tier 0 (no pending queue). The rmap path
 * (lru_marie_look_around) feeds the walker via a per-pgdat bloom
 * filter so PMD scans concentrate on regions the rmap recently
 * flagged hot.
 */

/*
 * ---------------------------------------------------------------------
 *  page add / del
 * ---------------------------------------------------------------------
 */

/*
 * lru_marie_add_page: per-page synchronous install.
 *
 * All pages are installed by marie_page_install under the caller's
 * lru_lock irqsave: per-PFN state byte (TRACKED + initial tier + type +
 * zone + head_gen), global track bitmap, global counters (marie_nr_pages
 * + node-global NR_LRU_BASE / NR_ZONE_LRU_BASE), and PG_lru are all
 * published in one synchronous call. No per-CPU staging and no
 * session-end flush hook -- every
 * install is self-contained, so no carry-over state can leak across
 * calls or across lruvecs.
 *
 * Skipped page classes:
 *
 *   - Unevictable pages: struct page overlays page->lru with
 *     page->mlock_count via union. mm/mlock.c writes mlock_count
 *     directly while the page is "owned" by an lruvec but NOT on a
 *     list. Marie keeps unevictable pages on the legacy path so
 *     mlock_count stays addressable.
 *
 * THP pages are routed through the per-type lock at the dispatcher
 * level so the install is ordered against concurrent operations on
 * the THP's lifetime. The per-type lock is purely a caller concern;
 * marie_page_install's body is identical for both branches.
 */
bool lru_marie_add_page(struct lruvec *lv, struct page *page, bool reclaiming)
{
	lockdep_assert_held(marie_lruvec_lock(lv));
	lockdep_assert_irqs_disabled();
	WARN_ON_ONCE(in_hardirq());

	if (!lru_marie_enabled())
		return false;
	if (PageUnevictable(page))
		return false;

	/*
	 * Contract (single-source): past this point Marie is enabled and the
	 * page is evictable, so it MUST become Marie-tracked -- the reader
	 * (lru_marie_zone_size_read) trusts the Marie counter alone for
	 * evictable buckets and will not size any page that escapes to the
	 * legacy list. Desktop/global-only: there is no per-memcg carrier to
	 * be missing, so every evictable page installs unconditionally.
	 *
	 * Large pages (THP) take the per-type lock on the way in so the
	 * install is ordered against the split-tail path on the same type;
	 * small pages run lock-free with only lru_lock. Both branches
	 * route to marie_page_install; the per-type lock is purely a
	 * caller concern.
	 */
	if (PageCompound(page)) {
		bool ok;
		int type = page_is_file_cache(page);

		scoped_guard(marie_type_lock, &marie_type_locks[type])
			ok = marie_page_install(page);
		return ok;
	}

	return marie_page_install(page);
}
EXPORT_SYMBOL_GPL(lru_marie_add_page);

/*
 * Non-adopting legacy LRU add for an untracked orphan inside a del+add
 * move_fn (swap.c: lru_activate / lru_deactivate{,_file} / lru_lazyfree) or
 * the legacy reclaim putback (vmscan.c: move_pages_to_lru).
 *
 * Those paths run del_page_from_lru_list() (legacy del, mz -nr for an untracked
 * page) and then add the page back. Routing that add through
 * add_page_to_lru_list() -> lru_marie_add_page() would ADOPT the page into
 * Marie: the install credits Marie's own accounting, but the original -nr was
 * a legacy debit, so mz->lru_zone_size drifts and a later legacy/Marie del
 * underflows ("marie underflow-del" / mem_cgroup_update_lru_size lru_size -1).
 * Do a pure legacy add (the +nr leg) instead.
 *
 * Callers MUST first bail on lru_marie_test_tracked() pages -- a tracked
 * page is Marie-owned and must never touch a legacy list. Shared by swap.c's
 * move_fns and vmscan.c's putback; see the header doc in lru_marie.h.
 */
void lru_marie_orphan_add(struct lruvec *lruvec, struct page *page, bool tail)
{
	enum lru_list lru = page_lru(page);

	update_lru_size(lruvec, lru, page_zonenum(page),
			compound_nr(page));
	if (tail)
		list_add_tail(&page->lru, &lruvec->lists[lru]);
	else
		list_add(&page->lru, &lruvec->lists[lru]);
}
EXPORT_SYMBOL_GPL(lru_marie_orphan_add);

/**
 * lru_marie_split_page - install a freshly-split tail page under Marie.
 * @lv:        head page's lruvec (caller holds lru_lock)
 * @head:      THP head page currently RESIDENT in Marie
 * @new_page: tail page created by __split_huge_page
 *
 * Mirrors mm/huge_memory.c::lru_add_split_page's
 * "list_add_tail(&new_page->lru, &page->lru)" for the Marie case so
 * that @new_page:
 *
 *   - inherits @head's tier 0 install at the current head_gen
 *     (a freshly-split tail page has no independent hotness signal
 *     yet -- subsequent walker passes promote it on young hits)
 *   - has its TRACKED bit set in marie_state[pfn] so dispatcher del
 *     routes through Marie (without TRACKED, dispatcher del would
 *     fall through to legacy update_lru_size and double-debit the
 *     node-global NR_LRU_BASE / NR_ZONE_LRU_BASE accounting)
 *
 * Accounting note: the node-global lru_size is NOT incremented for
 * @new_page. The original head install +N covered the
 * full pre-split compound, and each sub-page's eventual del decrements
 * by its own compound_nr; the sum balances. marie_nr_pages IS
 * incremented because it is a page count, not a page count, and the
 * post-split state has 1 + ntails pages where there was 1 before.
 *
 * Caller MUST hold @lv->lru_lock and have established that @head is
 * Marie-tracked (page_marie_test_tracked) before invoking this. The
 * helper takes the per-type lock internally via
 * scoped_guard(marie_type_lock, ...).
 *
 * No-op (and returns) if @head is unevictable -- legacy
 * lru_add_split_page handles that branch separately, before calling
 * here.
 */
void lru_marie_split_page(struct lruvec *lv, struct page *head,
			 struct page *new_page)
{
	lockdep_assert_held(marie_lruvec_lock(lv));
	lockdep_assert_irqs_disabled();
	WARN_ON_ONCE(in_hardirq());

	/*
	 * Caller already checked lru_marie_enabled() via the static branch,
	 * but @head may not be Marie-tracked (e.g. THP added to legacy LRU
	 * because Marie alloc failed at the original add). Fall back to
	 * plain list_add_tail in that case so @new_page joins @head's
	 * neighbour link on the legacy LRU list as it would have without
	 * Marie.
	 */
	if (!page_marie_test_tracked(head)) {
		list_add_tail(&new_page->lru, &head->lru);
		return;
	}

	if (PageUnevictable(head))
		return;

	/* Marie's invariant: clear PG_active before publishing TRACKED. */
	if (PageActive(new_page))
		ClearPageActive(new_page);

	/* Head and new_page share the same Marie type (page split does
	 * not change LRU category), so head's per-type lock guards both. */
	scoped_guard(marie_type_lock, &marie_type_locks[page_is_file_cache(head)]) {
		int type = page_is_file_cache(head);
		int zone = page_zonenum(new_page);
		u8 head_gen = (u8)atomic_read(&marie_head_gen[type]);

		/*
		 * Inherit head's tier-0 install at the global head_gen (a
		 * freshly-split tail page has no independent hotness signal).
		 * marie_pfn_publish_inherit writes the state byte, the global
		 * (type, gen, tier) track bitmap, and the global gen_occupied++.
		 * It deliberately skips gen_installs because the tail inherits
		 * the parent's install budget.
		 *
		 * page->lru is initialised to a self-loop, exactly as
		 * marie_page_install() does for a fresh install: every Marie
		 * page is OFF the legacy lruvec lists and tracked purely by the
		 * per-PFN state + bitmap. The old list_add_tail onto @head's link
		 * instead built a multi-element ring from @head and all of its
		 * split tails; the reclaim isolate path
		 * (marie_evict_counters_only + list_add, state.c) and every other
		 * self-loop-assuming site then corrupted that ring, abandoning
		 * neighbours that still pointed at the moved page -- pages
		 * orphaned off mz accounting (the mz->lru_zone_size underflow) and
		 * list_del corruption / use-after-free of a reused page (the
		 * userspace SEGV). PG_lru is set by the caller (lru_add_split_page)
		 * after this returns, after the per-PFN state is published, so a
		 * concurrent __page_cache_release observing PG_lru=1 also observes
		 * marie_state[pfn] & MARIE_PFN_TRACKED.
		 */
		INIT_LIST_HEAD(&new_page->lru);
		marie_pfn_publish_inherit(new_page, type, head_gen, 0, zone);
		marie_pc_add(&marie_nr_pages, 1);
	}
}
EXPORT_SYMBOL_GPL(lru_marie_split_page);

/*
 * lru_marie_split_page_isolated - reclaim-split counterpart of
 * lru_marie_split_page. See the header doc in <linux/lru_marie.h>.
 *
 * Symmetry with the on-LRU split is the whole point: lru_add_split_page()'s
 * on-LRU branch (head still on the LRU) routes tails through
 * lru_marie_split_page + SetPageLRU, so a normal split keeps every tail
 * TRACKED. The reclaim branch (head frozen off the LRU, tails threaded onto
 * the reclaim list) historically did a bare list_add_tail, leaving the tail
 * UNTRACKED -- the source of the THP "lru_size -1" escapee. This restores the
 * symmetry: publish TRACKED for the tail, but leave PG_lru clear and the list
 * linkage to the caller, because the tail lives on the reclaim list off-LRU
 * exactly like its isolated head.
 */
void lru_marie_split_page_isolated(struct page *head, struct page *new_page)
{
	if (!page_marie_test_tracked(head))
		return;
	if (PageUnevictable(head))
		return;

	/* Marie's invariant: clear PG_active before publishing TRACKED. */
	if (PageActive(new_page))
		ClearPageActive(new_page);

	scoped_guard(marie_type_lock, &marie_type_locks[page_is_file_cache(head)]) {
		int type = page_is_file_cache(head);
		int zone = page_zonenum(new_page);
		u8 head_gen = (u8)atomic_read(&marie_head_gen[type]);

		/*
		 * Publish the tail in the ISOLATED state (TRACKED byte only --
		 * no scan bit, no gen_occupied, no nr_pages), NOT a fresh
		 * install. The parent THP was already isolated
		 * (marie_evict_counters_only) before this split, so each child
		 * inherits that state and is balanced exactly once downstream:
		 * by free if reclaimed (bit-gated, net-zero) or by the putback
		 * path if it survives. marie_pfn_publish_inherit here would
		 * double-count gen_occupied (split inc + putback inc, freed
		 * once), leaving phantom gen_occupied>0/bit=0 occupancy that
		 * wedges find_oldest + head-advance after the tails free -- the
		 * THP-only, reboot-persistent anon-reclaim stall. See
		 * marie_pfn_publish_isolated.
		 */
		marie_pfn_publish_isolated(new_page, type, head_gen, 0, zone);
	}
}
EXPORT_SYMBOL_GPL(lru_marie_split_page_isolated);

bool lru_marie_del_page(struct lruvec *lv, struct page *page, bool reclaiming)
{
	lockdep_assert_held(marie_lruvec_lock(lv));
	lockdep_assert_irqs_disabled();
	WARN_ON_ONCE(in_hardirq());

	/*
	 * TRACKED takes priority over the lru_marie_enabled() gate.  Trusting
	 * TRACKED is safe regardless of the gate: TRACKED is only ever set
	 * under Marie's install helpers (marie_page_install on every install
	 * path, marie_pfn_publish_inherit on split's tail,
	 * marie_state_publish_at_gen on the reclaim survivor putback) and only
	 * ever cleared by Marie's evict paths. A page on a Marie self-loop
	 * must be handled by Marie; falling through to legacy del_page_from_lru_list
	 * would list_del(&page->lru) on the self-loop and corrupt the page.
	 */
	if (!page_marie_test_tracked(page))
		return false;

	/*
	 * External-removal entry runs without acquiring the per-type
	 * lock. The caller (del_page_from_lru_list reaching here from
	 * compaction / put_page -> __page_cache_release) holds
	 * lruvec->lru_lock, which serialises every other path that could
	 * clear MARIE_TRACKED. The eviction's list_del_init is
	 * unconditional (page->lru is either a self-loop or on legacy
	 * lruvec->lists[lru], whose mutation is already covered by the
	 * caller's lru_lock).
	 *
	 * marie_del_page_locked -> marie_evict_locked -> marie_account_evict
	 * owns the full counter wind-down, including the single
	 * marie_nr_pages -1. Do NOT decrement it again here (the old
	 * caller-side -1 predated the account.h funnel and double-counted
	 * every generic del of a tracked page).
	 */
	return marie_del_page_locked(page);
}
EXPORT_SYMBOL_GPL(lru_marie_del_page);

/*
 * Outer-level release entry called from __page_cache_release when the
 * caller has determined that TRACKED is set. See the contract in
 * <linux/lru_marie.h>.
 *
 * Why a TRACKED outer gate (rather than the legacy PageLRU
 * gate) matters: a Marie-installed page is on a self-loop
 * (page->lru points at itself), not on a legacy lruvec list. If the
 * legacy gate let such a page reach mm_inline.h::del_page_from_lru_list,
 * its list_del(&page->lru) would operate on the self-loop instead of
 * a real list and corrupt Marie's bookkeeping. With TRACKED as the
 * outer gate, Marie pages are routed here, which unlinks the
 * self-loop and debits the node-global NR_LRU_BASE / NR_ZONE_LRU_BASE
 * accounting (marie_update_lru_size is unified with legacy
 * update_lru_size, so the +nr at install and the -nr here balance
 * structurally).
 */
void lru_marie_release_page(struct page *page, struct lruvec **lruvecp,
			     unsigned long *flagsp)
{
	page_lruvec_relock_irqsave(page, lruvecp, flagsp);

	lockdep_assert_held(marie_lruvec_lock(*lruvecp));
	lockdep_assert_irqs_disabled();

	/*
	 * lru_marie_del_page re-tests TRACKED under the lock. A Marie page
	 * is always handled by Marie (desktop/global-only: no runtime drain
	 * to flip ownership). Returns true on Marie ownership; false means
	 * the page was never Marie-tracked -- the not-tracked legacy
	 * fall-through path.
	 */
	if (lru_marie_del_page(*lruvecp, page, false)) {
		/*
		 * marie_evict_locked already did list_del + cleared PG_lru and
		 * PG_active (it must, for its isolate-race defence and lru-index
		 * ordering -- see its body). The stock __clear_page_lru_flags()
		 * would re-assert PG_lru is still set (VM_BUG_ON_PAGE under
		 * DEBUG_VM, now false) and redundantly re-clear lru/active; only
		 * its PG_unevictable clear is still owed before the buddy handoff
		 * (PAGE_FLAGS_CHECK_AT_FREE). Clear just that.
		 */
		__ClearPageUnevictable(page);
		return;
	}

	/*
	 * Not Marie-tracked: this page lives on a legacy lruvec list with
	 * the node accounting credited the legacy way. Run the legacy del to
	 * keep PG_lru, the list membership, and accounting consistent.
	 */
	if (TestClearPageLRU(page))
		del_page_from_lru_list(page, *lruvecp, page_lru(page));
	__clear_page_lru_flags(page);
}
EXPORT_SYMBOL_GPL(lru_marie_release_page);

/*
 * No memcg lifecycle hooks. Marie is desktop/global-only: a single global
 * aging clock + global track bitmap, with ZERO per-memcg state. There is no
 * per-lruvec carrier to allocate at css_alloc, drop at css_offline/css_free,
 * or reparent -- the per-PFN state byte + global bitmap are independent of
 * which memcg a page is charged to. cgroup charging (memory.current, PSI)
 * stays on the stock path; Marie does nothing at any css transition.
 */

/*
 * Invoked from the vm.swappiness sysctl handler and memcg's
 * memory.swappiness writer when a swappiness value has changed. Resets the
 * single global swap_bias to neutral so the proportional controller restarts
 * from zero under the new weight ratio (desktop/global-only: one node-wide
 * controller, so one reset).
 */
void lru_marie_swappiness_changed(void)
{
	if (!lru_marie_enabled())
		return;

	atomic64_set(&marie_swap_bias, 0);
}
EXPORT_SYMBOL_GPL(lru_marie_swappiness_changed);

/*
 * marie_lruvec_zone_size - Marie's (lru, zone) bucket count.
 *
 * Desktop/global-only: there is no per-mlv shadow counter, so this returns
 * the node-global NR_ZONE_LRU_BASE total that marie_update_lru_size keeps
 * in lockstep with every install/evict. Every @lv query returns the same
 * node total. Callers go through lru_marie_zone_size_read(), which decides
 * marie-vs-stock.
 */
static unsigned long marie_lruvec_zone_size(struct lruvec *lv, enum lru_list lru,
					    int zone)
{
	struct pglist_data *pgdat = lruvec_pgdat(lv);

	/*
	 * Desktop/global-only: the per-(lru, zone) size reclaim reads is the
	 * NODE total, which marie_update_lru_size already maintains in the
	 * NR_ZONE_LRU_BASE vmstat (in lockstep with every install/evict). No
	 * per-mlv shadow counter needed. Every memcg's query returns the node
	 * total -- correct for global reclaim, best-effort for any (now
	 * global) memcg-targeted shrink.
	 */
	return zone_page_state(&pgdat->node_zones[zone], NR_ZONE_LRU_BASE + lru);
}

/*
 * lru_marie_zone_size_read - single-source SWITCH for mem_cgroup_get_zone_lru_size.
 *
 * Contract: while Marie is enabled, EVERY evictable page is Marie-tracked
 * (lru_marie_add_page admits all evictable pages; the only non-Marie pages
 * are the LRU_UNEVICTABLE list, which Marie never tracks). So the Marie
 * counter is the SOLE truth for evictable buckets and stock holds nothing
 * there -- return Marie alone, do NOT add stock. Summing the two was the
 * design flaw: it made stock a second, independently drifting source whose
 * boundary leaks underflowed mem_cgroup_update_lru_size.
 *
 *   - Marie enabled + evictable lru  -> Marie counter (sole source).
 *   - LRU_UNEVICTABLE                 -> @stock (unevictable pages live on
 *                                        the legacy list, counted in stock).
 *   - Marie disabled                  -> @stock (legacy / MGLRU regime).
 *
 * The contract is checked, not merely asserted: lru_marie_add_page WARNs if
 * an evictable page is ever turned away to legacy, and the
 * mem_cgroup_update_lru_size underflow WARN fires iff such an escapee's del
 * breaches it. If both stay silent, marie-only is exact by construction.
 */
unsigned long lru_marie_zone_size_read(struct lruvec *lv, enum lru_list lru,
				       int zone, unsigned long stock)
{
	if (lru_marie_enabled() && lru != LRU_UNEVICTABLE)
		return marie_lruvec_zone_size(lv, lru, zone);
	return stock;
}
EXPORT_SYMBOL_GPL(lru_marie_zone_size_read);




/* boot param: lru_marie=0 / lru_marie=1. Marie is selected ONCE here, at boot,
 * before the cgroup tree is populated and before any lruvec carries pages, so
 * a plain static-key toggle is sufficient -- there is nothing tracked to drain
 * or fill. Runtime toggling is not supported (desktop/global-only). */
static int __init marie_setup(char *str)
{
	int v;

	if (!str || kstrtoint(str, 0, &v))
		return 0;
	if (v)
		static_branch_enable(&lru_marie_enabled_key);
	else
		static_branch_disable(&lru_marie_enabled_key);
	return 1;
}
__setup("lru_marie=", marie_setup);


unsigned int lru_marie_shrink_lruvec(struct lruvec *lruvec, struct scan_control *sc)
{
	WARN_ON_ONCE(!sc);

	/*
	 * Per-PFN bitmap scan is the sole reclaim driver in Marie. The
	 * returned MARIE_DRAIN_* mask tells shrink_lruvec which orphan type(s)
	 * its legacy drain may reclaim (exactly the type(s) Marie scanned).
	 */
	return marie_state_shrink_lruvec(lruvec, sc);
}

/*
 * ---------------------------------------------------------------------
 *  /sys/kernel/mm/lru_marie/
 * ---------------------------------------------------------------------
 */

static ssize_t enabled_show(struct kobject *kobj, struct kobj_attribute *attr,
			    char *buf)
{
	return sysfs_emit(buf, "%d\n",
			  static_branch_likely(&lru_marie_enabled_key) ? 1 : 0);
}

/*
 * enabled is READ-ONLY: Marie is selected once at boot via lru_marie=0/1
 * (before any page is tracked). Runtime toggling was removed with the whole
 * drain/fill/change_state/transition_sem migration machinery -- desktop/
 * global-only Marie needs no runtime enable/disable, and dropping it deletes
 * the toggle race bug class and the per-reclaim-pass transition_sem read lock.
 */
static struct kobj_attribute marie_enabled_attr = __ATTR_RO(enabled);

/*
 * /sys/kernel/mm/lru_marie/version
 *
 * Read-only. Exposes MARIE_VERSION so userspace tooling (benchmark
 * scripts, sysadmins, support pastes) can identify which Marie build
 * is running without parsing dmesg.
 */
static ssize_t version_show(struct kobject *kobj, struct kobj_attribute *attr,
			    char *buf)
{
	return sysfs_emit(buf, "%s\n", MARIE_VERSION);
}

static struct kobj_attribute marie_version_attr = __ATTR_RO(version);

static ssize_t stats_show(struct kobject *kobj, struct kobj_attribute *attr,
			  char *buf)
{
	return sysfs_emit(buf,
			  "nr_pages %lld\n"
			  "pick file_strict %ld anon_strict %ld file_then_anon %ld anon_first %ld file_first %ld\n"
			  "reclaimed anon %ld file %ld\n"
			  "gen_growth_live anon %lu file %lu\n"
			  "concede floor %ld free %ld refault %ld memcg %ld\n"
			  "orphan_bit anon %ld file %ld\n",
			  percpu_counter_sum(&marie_nr_pages),
			  atomic_long_read(&marie_dbg_pick[0]),
			  atomic_long_read(&marie_dbg_pick[1]),
			  atomic_long_read(&marie_dbg_pick[2]),
			  atomic_long_read(&marie_dbg_pick[3]),
			  atomic_long_read(&marie_dbg_pick[4]),
			  atomic_long_read(&marie_dbg_reclaimed[0]),
			  atomic_long_read(&marie_dbg_reclaimed[1]),
			  READ_ONCE(marie_gen_growth_live[0]),
			  READ_ONCE(marie_gen_growth_live[1]),
			  atomic_long_read(&marie_dbg_concede[0]),
			  atomic_long_read(&marie_dbg_concede[1]),
			  atomic_long_read(&marie_dbg_concede[2]),
			  atomic_long_read(&marie_dbg_concede[3]),
			  atomic_long_read(&marie_dbg_orphan_bit[0]),
			  atomic_long_read(&marie_dbg_orphan_bit[1]));
}

static struct kobj_attribute marie_stats_attr = __ATTR_RO(stats);

/*
 * clean_min_ratio sysfs knob.
 * Range 0..100 (percentage of node_present_pages).
 */
static ssize_t clean_min_ratio_show(struct kobject *kobj,
				    struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%u\n", READ_ONCE(marie_clean_min_ratio));
}

static ssize_t clean_min_ratio_store(struct kobject *kobj,
				     struct kobj_attribute *attr,
				     const char *buf, size_t count)
{
	unsigned int v;
	int err = kstrtouint(buf, 10, &v);

	if (err)
		return err;
	if (v > 100)
		return -EINVAL;
	WRITE_ONCE(marie_clean_min_ratio, v);
	/* The file reserve/NR_GENS threshold floor tracks this ratio. */
	marie_recompute_growth_threshold(1);
	return count;
}

static struct kobj_attribute marie_clean_min_ratio_attr =
	__ATTR_RW(clean_min_ratio);

/*
 * low_swappiness_mode — clamp Marie's effective swappiness to at most 1
 * (bool, default on). When set, the reclaim pick driver caps swappiness at 1
 * (MARIE_PICK_FILE_THEN_ANON) regardless of the higher vm.swappiness /
 * memory.swappiness values udev rules, tuning daemons, or distro defaults
 * install. That is Marie's recommended policy (see the storage-tier rationale
 * at the top of this file), so enforcing it in-kernel spares the operator from
 * chasing down every writer. Default on => CONFIG_LRU_MARIE=y needs no
 * swappiness config.
 *
 * It only LOWERS swappiness, never raises it, so the special "never swap"
 * value 0 is preserved: an operator who set vm.swappiness=0 (OOM rather than
 * touch anon) still gets 0 even with this mode on. Clear the knob to honour
 * vm.swappiness verbatim (e.g. a higher proportional value on slow-storage
 * systems where file refault cost is not negligible).
 */
unsigned int marie_low_swappiness_mode = 1;

static ssize_t low_swappiness_mode_show(struct kobject *kobj,
				       struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%u\n", READ_ONCE(marie_low_swappiness_mode));
}

static ssize_t low_swappiness_mode_store(struct kobject *kobj,
					struct kobj_attribute *attr,
					const char *buf, size_t count)
{
	unsigned int v;
	int err = kstrtouint(buf, 10, &v);

	if (err)
		return err;
	if (v > 1)
		return -EINVAL;
	WRITE_ONCE(marie_low_swappiness_mode, v);
	/* Effective swappiness changed: restart the proportional controller. */
	lru_marie_swappiness_changed();
	return count;
}

static struct kobj_attribute marie_low_swappiness_mode_attr =
	__ATTR_RW(low_swappiness_mode);

#ifdef CONFIG_SWAP
/*
 * kcompressd sysfs knob: signed -100..+100, default +24.
 *
 *   0           — disabled. kcompressd_store short-circuits to false
 *                 and swap_writeout falls straight through to inline
 *                 zswap_store / __swap_writepage.
 *   +1..+100    — Marie-gated. Queue length = |v|. The kfifo backing
 *                 storage is sized at KCOMPRESSD_FIFO_SIZE (the max);
 *                 |v| is the soft depth at which the producer treats
 *                 the queue as full and falls back to sync writeout.
 *                 Tracks lru_marie_enabled() so disabling Marie at
 *                 runtime also quiesces kcompressd without a second
 *                 sysfs write.
 *   -1..-100    — force mode. Queue length = |v|. Runs even when
 *                 Marie is off, for users who want the async-compress
 *                 helper independently of the Marie reclaim path.
 *
 * Default +24 mirrors the queue length kcompressd-unofficial proved
 * sound under sustained anon pressure. Use -24 to force kcompressd on
 * even with Marie off; use 0 to disable entirely.
 *
 * Encoded as two static branches (kcompressd_enabled_key and
 * kcompressd_force_key declared in <linux/lru_marie.h>) so the hot path
 * costs a single predicted jump in the common (enabled, Marie-gated) case.
 */
DEFINE_STATIC_KEY_TRUE(kcompressd_enabled_key);
EXPORT_SYMBOL_GPL(kcompressd_enabled_key);
DEFINE_STATIC_KEY_FALSE(kcompressd_force_key);
EXPORT_SYMBOL_GPL(kcompressd_force_key);

int vm_kcompressd = 24;
EXPORT_SYMBOL_GPL(vm_kcompressd);

static ssize_t kcompressd_show(struct kobject *kobj,
			      struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%d\n", READ_ONCE(vm_kcompressd));
}

static ssize_t kcompressd_store(struct kobject *kobj,
			       struct kobj_attribute *attr,
			       const char *buf, size_t count)
{
	int v;
	int err = kstrtoint(buf, 10, &v);

	if (err)
		return err;
	if (v < -100 || v > 100)
		return -EINVAL;
	WRITE_ONCE(vm_kcompressd, v);

	if (v != 0)
		static_branch_enable(&kcompressd_enabled_key);
	else
		static_branch_disable(&kcompressd_enabled_key);
	if (v < 0)
		static_branch_enable(&kcompressd_force_key);
	else
		static_branch_disable(&kcompressd_force_key);

	return count;
}

static struct kobj_attribute marie_kcompressd_attr = __ATTR_RW(kcompressd);
#endif /* CONFIG_SWAP */

#ifdef CONFIG_X86
/*
 * SIMD walker kill-switch: /sys/kernel/mm/lru_marie/simd
 *
 * Default 1: walker uses the boot-detected SIMD wrapper (AVX-512F /
 * AVX2 / SSE2). Writing 0 flips marie_simd_enabled_key so the walker
 * falls through to a scalar pte_young loop in mm/lru_marie/simd_x86.c.
 * Also settable at boot via `lru_marie.simd=0|1`.
 */
static ssize_t simd_show(struct kobject *kobj,
			 struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%d\n",
			  static_branch_likely(&marie_simd_enabled_key) ? 1 : 0);
}

static ssize_t simd_store(struct kobject *kobj,
			  struct kobj_attribute *attr,
			  const char *buf, size_t count)
{
	bool v;
	int err = kstrtobool(buf, &v);

	if (err)
		return err;
	if (v)
		static_branch_enable(&marie_simd_enabled_key);
	else
		static_branch_disable(&marie_simd_enabled_key);
	return count;
}

static struct kobj_attribute marie_simd_attr = __ATTR_RW(simd);

/*
 * SIMD ISA cap: /sys/kernel/mm/lru_marie/simd_max
 *
 * Caps how wide a SIMD kernel the walker uses -- "avx512" (default, no cap),
 * "avx2", or "sse2" -- even on a CPU that supports wider. Writing "avx2"
 * avoids AVX-512 (e.g. for Intel license-based downclocking; harmless on AMD
 * Zen 4/5) while keeping AVX2. Orthogonal to the `simd` 0/1 switch, which
 * still forces the pure-scalar fallback. Also settable at boot via
 * `lru_marie.simd_max=`.
 */
static ssize_t simd_max_show(struct kobject *kobj,
			     struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%s\n", marie_simd_max_name());
}

static ssize_t simd_max_store(struct kobject *kobj,
			      struct kobj_attribute *attr,
			      const char *buf, size_t count)
{
	int err = marie_simd_max_store(buf);

	return err ? err : count;
}

static struct kobj_attribute marie_simd_max_attr = __ATTR_RW(simd_max);
#endif /* CONFIG_X86 */

/*
 * ---------------------------------------------------------------------
 *  Reclaim / walker tunables (runtime-adjustable via sysfs)
 * ---------------------------------------------------------------------
 *
 * Each variable is read with READ_ONCE on its hot path. Reclaim-loop
 * snapshots take the value at the top of each pass, so concurrent
 * sysfs writes take effect on the next pass without locking.
 */

/*
 * marie_clean_min_ratio — file-pagecache floor as a percentage of
 * node_present_pages. marie_state_shrink_lruvec diverts file reclaim
 * to anon when the node's NR_*_FILE total drops below this fraction,
 * preserving a working set of clean cache for codepaths that depend
 * on it (executable text, mapped data files, etc.) instead of
 * letting unbounded anon pressure flush it. 0 disables the floor
 * (legacy behaviour); 100 caps every file fault as protected.
 * Range 0..100; default 10.
 */
unsigned int marie_clean_min_ratio = 10;

/*
 * marie_walker_interval_* — adaptive walker pass deadline per pgdat,
 * stored in jiffies. marie_walker_interval() picks one based on the
 * zone's free-page state relative to its watermarks:
 *
 *   free < min      -> critical
 *   free < low      -> low
 *   free < high     -> normal
 *   free >= high    -> idle
 *
 * Defaults mirror the original literal cadence (HZ/30, HZ/10, HZ/4,
 * HZ — ~33 ms, 100 ms, 250 ms, 1 s on HZ=1000).  Hot writers see the
 * value via READ_ONCE inside marie_walker_interval(); the sysfs
 * helpers convert to and from ms for user friendliness.
 */
unsigned long marie_walker_interval_critical = HZ / 30;
unsigned long marie_walker_interval_low      = HZ / 10;
unsigned long marie_walker_interval_normal   = HZ / 4;
unsigned long marie_walker_interval_idle     = HZ;

/*
 * Walker-interval knob factory: every stage uses the same show/store
 * shape (ms in, jiffies stored, clamped to >= 1 jiffy).  Range is
 * 1..60000 ms — anything shorter than a jiffy is meaningless on
 * commodity HZ, anything longer than a minute defeats the adaptive
 * gating.
 */
#define MARIE_WALKER_INTERVAL_KNOB(name, var)				\
static ssize_t name##_show(struct kobject *kobj,			\
			   struct kobj_attribute *attr, char *buf)	\
{									\
	return sysfs_emit(buf, "%u\n",					\
			  jiffies_to_msecs(READ_ONCE(var)));		\
}									\
static ssize_t name##_store(struct kobject *kobj,			\
			    struct kobj_attribute *attr,		\
			    const char *buf, size_t count)		\
{									\
	unsigned int ms;						\
	unsigned long j;						\
	int err = kstrtouint(buf, 10, &ms);				\
									\
	if (err)							\
		return err;						\
	if (ms < 1 || ms > 60000)					\
		return -EINVAL;						\
	j = msecs_to_jiffies(ms);					\
	if (j < 1)							\
		j = 1;							\
	WRITE_ONCE(var, j);						\
	return count;							\
}									\
static struct kobj_attribute marie_##name##_attr = __ATTR_RW(name)

MARIE_WALKER_INTERVAL_KNOB(walker_interval_critical_ms,
			   marie_walker_interval_critical);
MARIE_WALKER_INTERVAL_KNOB(walker_interval_low_ms,
			   marie_walker_interval_low);
MARIE_WALKER_INTERVAL_KNOB(walker_interval_normal_ms,
			   marie_walker_interval_normal);
MARIE_WALKER_INTERVAL_KNOB(walker_interval_idle_ms,
			   marie_walker_interval_idle);

static struct attribute *marie_attrs[] = {
	&marie_enabled_attr.attr,
	&marie_version_attr.attr,
	&marie_stats_attr.attr,
	&marie_clean_min_ratio_attr.attr,
	&marie_low_swappiness_mode_attr.attr,
#ifdef CONFIG_SWAP
	&marie_kcompressd_attr.attr,
#endif
#ifdef CONFIG_X86
	&marie_simd_attr.attr,
	&marie_simd_max_attr.attr,
#endif
	&marie_walker_interval_critical_ms_attr.attr,
	&marie_walker_interval_low_ms_attr.attr,
	&marie_walker_interval_normal_ms_attr.attr,
	&marie_walker_interval_idle_ms_attr.attr,
	NULL,
};

static const struct attribute_group marie_attr_group = {
	.attrs = marie_attrs,
};

static int __init marie_init(void)
{
	struct kobject *marie_kobj;
	int err;

	printk(KERN_INFO "%s %s by %s\n",
	       MARIE_PROGNAME, MARIE_VERSION, MARIE_AUTHOR);

	marie_prefetch_params_init();

	/*
	 * Latch the 32-bit PFN gate. max_pfn is established by setup_arch /
	 * memblock init well before subsys_initcall, so this single read is
	 * authoritative for the lifetime of the system. If the box overflows
	 * the 32-bit PFN window we disable Marie up front, regardless of
	 * lru_marie= boot param or the static-key default. (enabled is
	 * read-only at runtime, so there are no later sysfs enables to gate.)
	 */
	if (max_pfn > MARIE_MAX_SUPPORTED_PFN) {
		marie_pfn_unsupported = true;
		if (static_branch_likely(&lru_marie_enabled_key))
			static_branch_disable(&lru_marie_enabled_key);
		pr_warn("disabled: max_pfn %lu exceeds 32-bit limit (%lu); Marie requires physical address space <= 16 TiB\n",
			max_pfn, MARIE_MAX_SUPPORTED_PFN);
	} else {
		/*
		 * Allocate the per-PFN state array now that the gate has
		 * been verified. If this fails we cannot run, so disable
		 * Marie and continue boot with the in-tree LRU paths.
		 */
		err = marie_state_init();
		if (err) {
			marie_pfn_unsupported = true;
			if (static_branch_likely(&lru_marie_enabled_key))
				static_branch_disable(&lru_marie_enabled_key);
			pr_warn("disabled: marie_state_init failed (%d)\n",
				err);
		}
	}

	/*
	 * Initialise the global marie_nr_pages percpu_counter. (Earlier
	 * revisions also set up slab caches and per-CPU pools here; the
	 * per-PFN paradigm has none of that to allocate.)
	 */
	err = marie_counters_init();
	if (err < 0)
		return err;

	marie_walker_init();

	marie_kobj = kobject_create_and_add("lru_marie", mm_kobj);
	if (!marie_kobj) {
		pr_err("failed to create /sys/kernel/mm/lru_marie\n");
		return -ENOMEM;
	}

	err = sysfs_create_group(marie_kobj, &marie_attr_group);
	if (err) {
		pr_err("failed to create /sys/kernel/mm/lru_marie attributes: %d\n", err);
		kobject_put(marie_kobj);
		return err;
	}

#ifdef CONFIG_LRU_MARIE_DEFRAG
	/*
	 * Allocate the Marie defrag per-pageblock occupancy histogram and expose its
	 * completeness-check node. Non-fatal: a failure here disables Marie defrag
	 * observability but leaves Marie fully functional (the marie_defrag_hist_inc/dec
	 * hooks no-op on a NULL array).
	 */
	if (marie_defrag_init(marie_kobj))
		pr_warn("Marie defrag histogram init failed; compaction observability off\n");
#endif

	pr_info("currently %s\n",
		static_branch_likely(&lru_marie_enabled_key) ? "enabled" : "disabled");
	return 0;
}
subsys_initcall(marie_init);
