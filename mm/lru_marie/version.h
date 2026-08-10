/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _MM_LRU_MARIE_VERSION_H
#define _MM_LRU_MARIE_VERSION_H

/*
 * Marie LRU — version identifiers.
 *
 * Kept in mm/ rather than include/linux/lru_marie.h so that bumping
 * MARIE_VERSION (the only string that changes from one release to the
 * next) does not invalidate the ccache entry for every translation
 * unit that includes <linux/lru_marie.h> (mm/mm.h, mm/mm_inline.h,
 * mm/vmscan.c, mm/swap.c, mm/rmap.c, mm/memcontrol.c, etc.). Only
 * Marie's own .c files include this header, so a version bump rebuilds
 * just mm/lru_marie*.o.
 */

#define MARIE_PROGNAME	"Marie LRU"
#define MARIE_AUTHOR	"Masahito Suzuki"

#define MARIE_VERSION	"0.9.3"

#endif /* _MM_LRU_MARIE_VERSION_H */
