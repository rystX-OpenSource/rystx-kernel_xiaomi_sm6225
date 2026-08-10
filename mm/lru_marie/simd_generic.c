// SPDX-License-Identifier: GPL-2.0
/*
 * mm/lru_marie/simd_generic.c -- fallback PTE young-bit scan for arches
 *                              without a SIMD path.
 *
 * Uses the arch-provided pte_young() helper so we don't need to know
 * the accessed-bit name on every architecture.
 *
 * There is no FPU bracket here because the scan is scalar (no FPU state
 * to preserve). Every arch other than x86 currently lands on this file
 * (including arm64, where a future NEON variant could be slotted in once
 * its FPSIMD save/restore cost has been profiled against the per-pmd
 * gain).
 */

#include <linux/bitmap.h>
#include <linux/mm.h>		/* pte_young */
#include <asm/pgtable.h>

#include "simd.h"

#define PTES_PER_PMD	512

void lru_marie_simd_young_pte_mask(const void *table, unsigned long *bitmap)
{
	const pte_t *pte = (const pte_t *)table;
	int i;

	for (i = 0; i < PTES_PER_PMD; i++) {
		if (pte_young(pte[i]))
			__set_bit(i, bitmap);
	}
}
EXPORT_SYMBOL_GPL(lru_marie_simd_young_pte_mask);
