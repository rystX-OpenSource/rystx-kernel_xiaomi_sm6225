/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _LINUX_PSR_LMK_H
#define _LINUX_PSR_LMK_H

#include <linux/types.h>
#include <linux/gfp.h>
#include <linux/vmpressure.h>

struct page;
struct pglist_data;

/*
 * PSR-LMK core hook API.
 *
 * These are the mm/fs call sites that feed PSR-LMK's regression
 * engine and let it short-circuit further reclaim once it has
 * already dispatched a bypass kill. All of them are cheap (counter
 * bumps / conditional checks) except the actual kill dispatch, which
 * PSR-LMK itself defers to a workqueue rather than doing inline from
 * these (often atomic / lock-held) call sites -- see
 * drivers/android/psr_lmk.c.
 */

#ifdef CONFIG_ANDROID_PSR_LMK

/* mm/swap.c: an anon (swap-backed) page was just reactivated from the
 * inactive to the active LRU -- memory that was a swap-out candidate
 * is being pulled back into active use. Core "protected-swap
 * regression" signal, hence the driver's name. */
void psr_lmk_note_anon_reactivation(struct page *page);

/* mm/vmpressure.c: kernel-native scan/reclaim efficiency pressure,
 * computed once per vmpressure work cycle. Replaces userspace polling
 * of /proc/pressure/memory with the same signal the kernel itself
 * already computes. */
void psr_lmk_note_pressure(enum vmpressure_levels level,
			    unsigned long pressure,
			    unsigned long scanned,
			    unsigned long reclaimed);

/* mm/vmscan.c: per-priority-pass scan/reclaim progress for a node. */
void psr_lmk_note_scan_progress(struct pglist_data *pgdat,
				 unsigned long nr_scanned,
				 unsigned long nr_reclaimed);

/* mm/vmscan.c: checked at the top of each reclaim priority pass in
 * shrink_node(). Returns true if PSR-LMK already dispatched a bypass
 * kill this cycle, so the caller can stop scanning early instead of
 * grinding through priorities a kill likely already made unnecessary. */
bool psr_lmk_should_abort_reclaim(void);

/* mm/page_alloc.c: direct reclaim finished but the allocation still
 * failed -- the same "failed allocation stuck in the slow path"
 * signal simple_lmk uses to trigger killing. */
void psr_lmk_note_alloc_failure(unsigned int order, gfp_t gfp_mask);

/* mm/workingset.c: a page just refaulted after being evicted.
 * is_swap distinguishes an anon/swap-backed refault (real swap
 * thrash -- what PSR-LMK cares about) from a file-cache refault
 * (page-cache churn, tracked but weighted differently). */
void psr_lmk_note_refault(struct page *page, bool is_swap,
			   unsigned long refault_distance,
			   unsigned long active_size);

#else /* !CONFIG_ANDROID_PSR_LMK */

static inline void psr_lmk_note_anon_reactivation(struct page *page) {}

static inline void psr_lmk_note_pressure(enum vmpressure_levels level,
					  unsigned long pressure,
					  unsigned long scanned,
					  unsigned long reclaimed) {}

static inline void psr_lmk_note_scan_progress(struct pglist_data *pgdat,
					       unsigned long nr_scanned,
					       unsigned long nr_reclaimed) {}

static inline bool psr_lmk_should_abort_reclaim(void)
{
	return false;
}

static inline void psr_lmk_note_alloc_failure(unsigned int order,
					       gfp_t gfp_mask) {}

static inline void psr_lmk_note_refault(struct page *page, bool is_swap,
					 unsigned long refault_distance,
					 unsigned long active_size) {}

#endif /* CONFIG_ANDROID_PSR_LMK */

#endif /* _LINUX_PSR_LMK_H */
