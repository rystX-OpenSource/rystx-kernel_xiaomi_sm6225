/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _MM_LRU_MARIE_ACCOUNT_H
#define _MM_LRU_MARIE_ACCOUNT_H

#include <linux/lockdep.h>
#include <linux/mm.h>
#include <linux/mm_inline.h>

#include "state.h"

/*
 * Marie's counter+vmstat updates that move together on every
 * install/evict.  Marie holds zero per-lruvec state, so the only counters
 * are global: the percpu marie_nr_pages (+-1 per page) and the vmstat
 * lru_size pair via marie_update_lru_size.  The drift hazard that
 * cost us 9c6a93782 was: each site picked its own IRQ-state discipline
 * because no helper enforced it.
 *
 *   marie_pc_add(&marie_nr_pages,        +-1 )
 *   marie_update_lru_size(lv, lru, zone,  +-nr)
 *
 * marie_update_lru_size credits/debits NR_LRU_BASE (via mod_lruvec_state,
 * which also feeds the per-memcg memory.stat breakdown) plus the node-
 * global NR_ZONE_LRU_BASE total, so a Marie page is counted in vmstat
 * from install to evict exactly like a legacy/MGLRU page.  It does NOT
 * write any per-memcg mz->lru_zone_size; lruvec_lru_size() reads the
 * NR_LRU_BASE/NR_ZONE_LRU_BASE state directly.
 *
 * Two contexts:
 *
 *   LOCKED   - caller holds lv->lru_lock with IRQs off.  Used by the
 *              install / evict_locked / del_page_locked /
 *              fill / drain hot paths.  Helpers assert both held
 *              conditions via lockdep.
 *
 *   ISOLATE  - caller holds NOTHING (no lru_lock, IRQs on).  Used by
 *              the reclaim isolate path and the survivor putback.
 *              Helpers own local_irq_save/restore so the marie_pc_add
 *              fast path and __mod_zone_page_state inside
 *              marie_update_lru_size are safe against same-CPU
 *              softirq reentrancy (the very property 9c6a93782
 *              introduced).
 *
 * The helpers do NOT touch page flags, the per-PFN state byte, or the
 * scan bitmap.  Those belong to pfn_install.h
 * (marie_pfn_publish_inherit) and to marie_state_publish_at_gen /
 * marie_state_drop_pfn.  Each layer keeps its own invariant.
 */

static inline void marie_account_install(struct lruvec *lv,
					 struct page *f,
					 enum lru_list lru, int zone)
{
	long nr = compound_nr(f);

	lockdep_assert_held(marie_lruvec_lock(lv));
	lockdep_assert_irqs_disabled();

	marie_pc_add(&marie_nr_pages, 1);
	marie_update_lru_size(lv, lru, zone, nr);
}

static inline void marie_account_evict(struct lruvec *lv,
				       struct page *f,
				       enum lru_list lru, int zone)
{
	long nr = compound_nr(f);

	lockdep_assert_held(marie_lruvec_lock(lv));
	lockdep_assert_irqs_disabled();

	marie_pc_add(&marie_nr_pages, -1);
	marie_update_lru_size(lv, lru, zone, -nr);
}

static inline void marie_account_install_isolate(struct lruvec *lv,
						 struct page *f,
						 enum lru_list lru, int zone)
{
	long nr = compound_nr(f);
	unsigned long flags;

	WARN_ON_ONCE(irqs_disabled());

	local_irq_save(flags);
	marie_pc_add(&marie_nr_pages, 1);
	__update_lru_size(lv, lru, zone, nr);
	local_irq_restore(flags);
}

static inline void marie_account_evict_isolate(struct lruvec *lv,
					       struct page *f,
					       enum lru_list lru, int zone)
{
	long nr = compound_nr(f);
	unsigned long flags;

	WARN_ON_ONCE(irqs_disabled());

	local_irq_save(flags);
	marie_pc_add(&marie_nr_pages, -1);
	__update_lru_size(lv, lru, zone, -nr);
	local_irq_restore(flags);
}

#endif /* _MM_LRU_MARIE_ACCOUNT_H */
