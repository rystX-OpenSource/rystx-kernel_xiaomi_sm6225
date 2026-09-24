// SPDX-License-Identifier: GPL-2.0
/*
 * TAGLMK - LRU / aging signal (classic active-inactive LRU baseline).
 *
 * This translation unit is the single point where TAGLMK reads the page
 * reclaim subsystem's aging state.  On this baseline branch it uses the
 * classic active/inactive LRU counters plus swap utilisation to derive a
 * normalised cache-load / aging-pressure sample.  The MGLRU adaptation
 * replaces the body of these functions with generation-based reads while
 * keeping the same interface, so the rest of the driver is unaffected.
 */
#include <linux/mm.h>
#include <linux/mmzone.h>
#include <linux/swap.h>
#include <linux/vmstat.h>

#include "taglmk.h"

/*
 * Return a Q4.4 sample (raw value, i.e. load << 4, clamped to 0..255) that
 * rises as the page cache comes under pressure: high swap utilisation or a
 * shrinking file cache both push the value up.
 */
q4_4_t taglmk_lru_load_sample(void)
{
	long total_swap = total_swap_pages;
	long free_swap = get_nr_swap_pages();
	unsigned long active_file, file_target;
	unsigned long swap_x16 = 0, file_x16 = 0, load;

	if (free_swap < 0)
		free_swap = 0;

	if (total_swap > 0) {
		unsigned long used = (total_swap > free_swap) ?
				     (unsigned long)(total_swap - free_swap) : 0;

		swap_x16 = (used * TAGLMK_Q44_ONE) / (unsigned long)total_swap;
	} else {
		/* No swap: use anon share of the working set as a proxy for the
		 * swap demand that would exist. */
		unsigned long anon = global_node_page_state(NR_ACTIVE_ANON) +
				     global_node_page_state(NR_INACTIVE_ANON);
		unsigned long file = global_node_page_state(NR_ACTIVE_FILE) +
				     global_node_page_state(NR_INACTIVE_FILE);

		if (anon + file)
			swap_x16 = (anon * TAGLMK_Q44_ONE) / (anon + file);
	}

	/* File-cache shortfall: healthy target is an eighth of RAM. */
	active_file = global_node_page_state(NR_ACTIVE_FILE);
	file_target = totalram_pages / 8;
	if (file_target && active_file < file_target)
		file_x16 = ((file_target - active_file) * TAGLMK_Q44_ONE) /
			   file_target;

	load = max(swap_x16, file_x16);
	if (load > 255)
		load = 255;
	return (q4_4_t)load;
}

bool taglmk_lru_file_critical(unsigned long free_file_limit)
{
	return global_node_page_state(NR_ACTIVE_FILE) < free_file_limit;
}
