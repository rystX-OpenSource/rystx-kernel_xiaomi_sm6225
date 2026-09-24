// SPDX-License-Identifier: GPL-2.0
/*
 * TAGLMK - LRU / aging signal (MGLRU generation-based adaptation).
 *
 * This is the MGLRU-specific counterpart of the baseline taglmk_lru.c.  It
 * keeps the exact same interface (taglmk_lru_load_sample / _file_critical) so
 * the rest of the driver is identical to the baseline branch, but it sources
 * its cache-load and file-critical signals from the multi-gen LRU's per-node
 * generation state instead of the classic active/inactive counters.
 *
 * All reads are lockless: only the eventually-consistent nr_pages[] counts,
 * the per-type seq numbers and the refault EMAs are read (via READ_ONCE), and
 * the generation *lists* are never traversed.  Nothing here takes
 * lruvec->lru_lock, so it can neither block nor race the aging (max_seq bump)
 * or eviction (min_seq bump) paths.
 *
 * When MGLRU is compiled in but disabled at runtime the classic counters are
 * still maintained (lru_gen_update_size() keeps them), so the code falls back
 * to them and behaves like the baseline.
 */
#include <linux/mm.h>
#include <linux/mm_inline.h>
#include <linux/mmzone.h>
#include <linux/swap.h>
#include <linux/vmstat.h>

#include "taglmk.h"

/* Swap utilisation is LRU-agnostic; shared by both signal sources. Q4.4. */
static unsigned long taglmk_swap_x16(void)
{
	long total_swap = total_swap_pages;
	long free_swap = get_nr_swap_pages();

	if (free_swap < 0)
		free_swap = 0;

	if (total_swap > 0) {
		unsigned long used = (total_swap > free_swap) ?
				     (unsigned long)(total_swap - free_swap) : 0;

		return (used * TAGLMK_Q44_ONE) / (unsigned long)total_swap;
	}

	/* No swap: anon share of the working set as a swap-demand proxy. */
	{
		unsigned long anon = global_node_page_state(NR_ACTIVE_ANON) +
				     global_node_page_state(NR_INACTIVE_ANON);
		unsigned long file = global_node_page_state(NR_ACTIVE_FILE) +
				     global_node_page_state(NR_INACTIVE_FILE);

		return (anon + file) ?
		       (anon * TAGLMK_Q44_ONE) / (anon + file) : 0;
	}
}

#ifdef CONFIG_LRU_GEN
/*
 * Summarise the file-type multi-gen LRU across all online nodes:
 *   @totalp   - file pages held in every generation
 *   @youngp   - file pages in the two youngest (active-equivalent) gens
 *   @refx16p  - Q4.4 refault ratio (avg_refaulted / avg_total) averaged over
 *               tiers: MGLRU's own thrashing signal, 0 == no refaults.
 *
 * Only the eventually-consistent counters and seq numbers are read; the gen
 * lists are never walked and no lock is taken, so this cannot race the aging
 * (max_seq bump) or eviction (min_seq bump) fast paths.
 */
static void taglmk_mglru_file(unsigned long *totalp, unsigned long *youngp,
			      unsigned long *refx16p)
{
	unsigned long total = 0, young = 0;
	unsigned long refaulted = 0, evicted = 0;
	int nid;

	for_each_online_node(nid) {
		struct lruvec *lruvec = node_lruvec(NODE_DATA(nid));
		struct lru_gen_struct *lrugen = &lruvec->lrugen;
		unsigned long max_seq = READ_ONCE(lrugen->max_seq);
		unsigned long min_seq = READ_ONCE(lrugen->min_seq[LRU_GEN_FILE]);
		unsigned long seq;
		int tier;

		/* At most MAX_NR_GENS (4) live generations, oldest to youngest. */
		for (seq = min_seq; seq <= max_seq; seq++) {
			int gen = lru_gen_from_seq(seq);
			unsigned long n = 0;
			int zone;

			for (zone = 0; zone < MAX_NR_ZONES; zone++) {
				long v = READ_ONCE(lrugen->nr_pages[gen][LRU_GEN_FILE][zone]);

				if (v > 0)
					n += (unsigned long)v;
			}
			total += n;
			/* seq == max_seq or max_seq-1 are the "active" gens; the
			 * unsigned form avoids underflow at max_seq == 0. */
			if (seq + 2 > max_seq)
				young += n;
		}

		for (tier = 0; tier < MAX_NR_TIERS; tier++) {
			refaulted += READ_ONCE(lrugen->avg_refaulted[LRU_GEN_FILE][tier]);
			evicted += READ_ONCE(lrugen->avg_total[LRU_GEN_FILE][tier]);
		}
	}

	*totalp = total;
	*youngp = young;
	*refx16p = evicted ? (refaulted * TAGLMK_Q44_ONE) / evicted : 0;
}

/* Q4.4 cache-load from the multi-gen LRU. Returns false when MGLRU is off. */
static bool taglmk_mglru_load(unsigned long *loadp)
{
	unsigned long total, young, refx16, target, file_x16 = 0;

	if (!lru_gen_enabled())
		return false;

	taglmk_mglru_file(&total, &young, &refx16);

	/* Young (active-equivalent) file cache below an eighth of RAM signals
	 * pressure, mirroring the baseline's active-file floor. */
	target = totalram_pages / 8;
	if (target && young < target)
		file_x16 = ((target - young) * TAGLMK_Q44_ONE) / target;

	*loadp = max(file_x16, refx16);
	return true;
}
#else
static inline bool taglmk_mglru_load(unsigned long *loadp) { return false; }
#endif /* CONFIG_LRU_GEN */

/* Classic active-file shortfall (Q4.4), used when MGLRU is unavailable/off. */
static unsigned long taglmk_classic_file_x16(void)
{
	unsigned long active_file = global_node_page_state(NR_ACTIVE_FILE);
	unsigned long target = totalram_pages / 8;

	if (target && active_file < target)
		return ((target - active_file) * TAGLMK_Q44_ONE) / target;
	return 0;
}

/*
 * Return a Q4.4 sample (raw value, i.e. load << 4, clamped to 0..255) that
 * rises as the page cache comes under pressure.  Swap utilisation is common
 * to both signal sources; the file-cache term is generation-based when the
 * multi-gen LRU is active and falls back to the classic active-file counter
 * otherwise.  Both forms keep the same 0..16 scale so the driver's forecast
 * thresholds are unchanged across the baseline and MGLRU branches.
 */
q4_4_t taglmk_lru_load_sample(void)
{
	unsigned long swap_x16 = taglmk_swap_x16();
	unsigned long file_x16, load;

	if (!taglmk_mglru_load(&file_x16))
		file_x16 = taglmk_classic_file_x16();

	load = max(swap_x16, file_x16);
	if (load > 255)
		load = 255;
	return (q4_4_t)load;
}

bool taglmk_lru_file_critical(unsigned long free_file_limit)
{
#ifdef CONFIG_LRU_GEN
	if (lru_gen_enabled()) {
		unsigned long total, young, refx16;

		/* Compare the active-equivalent (young) file cache, the MGLRU
		 * analogue of NR_ACTIVE_FILE, against the limit. */
		taglmk_mglru_file(&total, &young, &refx16);
		return young < free_file_limit;
	}
#endif
	return global_node_page_state(NR_ACTIVE_FILE) < free_file_limit;
}
