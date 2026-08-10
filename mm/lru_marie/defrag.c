// SPDX-License-Identifier: GPL-2.0
/*
 * mm/lru_marie/defrag.c -- Marie defragmentation, Step 1.
 *
 * Maintains a per-pageblock (gen, type) occupancy histogram off Marie's
 * single gen-occupancy choke-point (marie_gen_occ_inc/dec, state.h). The
 * histogram is the sufficient-statistic input to the future cost-scored
 * block selector (see defrag_design.md); this step is observability
 * ONLY -- no migration, no reclaim, no policy change.
 *
 * Correctness is checkable by construction: because the only writers are the
 * choke-point hooks, summing the histogram over all blocks must reproduce the
 * global marie_gen_occupied counter for every (gen, type). The read-only
 * sysfs node verifies this. (gen_occupied and its per-block mirror are bumped
 * by two separate atomics, so a sum taken under active churn can differ by the
 * number of in-flight transitions; the invariant is exact at quiescence --
 * read it after the load settles.)
 */
#define pr_fmt(fmt) "lru_marie_defrag: " fmt

#include <linux/atomic.h>
#include <linux/build_bug.h>
#include <linux/kobject.h>
#include <linux/migrate.h>		/* migrate_pages, alloc_migration_target */
#include <linux/migrate_mode.h>		/* MIGRATE_SYNC_LIGHT */
#include <linux/mm.h>			/* max_pfn */
#include <linux/mmzone.h>
#include <linux/numa.h>			/* NUMA_NO_NODE */
#include <linux/pageblock-flags.h>
#include <linux/printk.h>
#include <linux/sched.h>		/* cond_resched */
#include <linux/slab.h>
#include <linux/sort.h>
#include <linux/sysfs.h>
#include <linux/vmalloc.h>
#include <linux/vmstat.h>		/* node_page_state */
#include <linux/lru_marie.h>		/* lru_marie_enabled */

#include "../internal.h"		/* page_isolate_lru, migration_target_control */
#include "defrag_compat.h"		/* marie_defrag_prep_allocated (6.14 refcount split) */
#include "defrag.h"
#include "state.h"			/* marie_gen_occupied, MARIE_PFN_NR_GENS */

struct marie_defrag_block_hist *marie_defrag_hist;
unsigned long marie_defrag_nr_blocks;

/* Max source blocks selected per run (also the candidate cap of the read-only node). */
#define MARIE_DEFRAG_TOPK	16

/*
 * Age-neutrality scratch (see the design comment above marie_defrag_freectx).
 * Two pre-allocated (pfn, gen) tables plus the single-owner trylock guarding
 * them. Declared here because marie_defrag_scratch_alloc() below sizes them.
 */
#define MARIE_DEFRAG_GEN_NONE	0xff	/* srcmap miss -> fall back to oldest */

struct marie_defrag_pg {
	u32	pfn;
	u8	gen;
};

static struct marie_defrag_pg	*marie_defrag_srcmap;	/* src pfn -> gen (isolation) */
static struct marie_defrag_pg	*marie_defrag_restamp;	/* dst pfn + intended gen */
static unsigned long		 marie_defrag_scratch_cap;	/* entries per table */
static atomic_t			 marie_defrag_busy = ATOMIC_INIT(0);

/*
 * Allocate the per-pageblock histogram covering [0, max_pfn) rounded up to a
 * whole pageblock. ~1 MiB on a 30 GiB box (≈15k blocks x 64 B). kvmalloc:
 * accessed by block index only, so physical contiguity is not required. Lives
 * for the kernel's lifetime, like marie_state.
 */
static int __init marie_defrag_hist_alloc(void)
{
	unsigned long bytes, pb_pages = 1UL << pageblock_order;

	BUILD_BUG_ON(MARIE_PFN_NR_GENS != MARIE_DEFRAG_NGENS);

	marie_defrag_nr_blocks = (max_pfn + pb_pages - 1) >> pageblock_order;
	if (!marie_defrag_nr_blocks)
		return -EINVAL;

	bytes = marie_defrag_nr_blocks * sizeof(struct marie_defrag_block_hist);
	marie_defrag_hist = kvmalloc(bytes, GFP_KERNEL | __GFP_ZERO);
	if (!marie_defrag_hist)
		return -ENOMEM;

	pr_info("per-pageblock histogram: %lu blocks (order %u), %lu KiB\n",
		marie_defrag_nr_blocks, pageblock_order, bytes >> 10);
	return 0;
}

/*
 * Pre-allocate the two age-neutrality scratch tables (srcmap, restamp) once, at
 * boot, so the under-pressure defrag path never allocates. Sized to the worst
 * case run: MARIE_DEFRAG_TOPK fully-occupied source blocks. Non-fatal on failure
 * -- defrag then runs without age correction (plain head install), like a per-run
 * alloc that failed used to. ~64 KiB per table on an order-9 pageblock.
 */
static void __init marie_defrag_scratch_alloc(void)
{
	unsigned long cap = (unsigned long)MARIE_DEFRAG_TOPK << pageblock_order;
	size_t bytes = cap * sizeof(struct marie_defrag_pg);

	marie_defrag_srcmap = kvmalloc(bytes, GFP_KERNEL);
	marie_defrag_restamp = kvmalloc(bytes, GFP_KERNEL);
	if (!marie_defrag_srcmap || !marie_defrag_restamp) {
		kvfree(marie_defrag_srcmap);
		kvfree(marie_defrag_restamp);
		marie_defrag_srcmap = marie_defrag_restamp = NULL;
		marie_defrag_scratch_cap = 0;
		pr_warn("age-neutrality scratch alloc failed; defrag runs without age correction\n");
		return;
	}
	marie_defrag_scratch_cap = cap;
	pr_info("age-neutrality scratch: %lu entries x2 (%zu KiB)\n",
		cap, (bytes * 2) >> 10);
}

/*
 * Completeness invariant. For each (gen, type), the histogram summed over all
 * blocks must equal the global marie_gen_occupied -- otherwise a gen
 * transition escaped the choke-point. O(nr_blocks x NCLASS); debug-on-read.
 */
static int marie_defrag_invariant_report(char *buf, int len, int cap)
{
	long hist_sum[MARIE_DEFRAG_NGENS][MARIE_DEFRAG_NTYPES] = {};
	unsigned long b;
	int g, t, mism = 0;

	for (b = 0; b < marie_defrag_nr_blocks; b++) {
		struct marie_defrag_block_hist *h = &marie_defrag_hist[b];

		for (g = 0; g < MARIE_DEFRAG_NGENS; g++)
			for (t = 0; t < MARIE_DEFRAG_NTYPES; t++)
				hist_sum[g][t] +=
					atomic_read(&h->count[g * MARIE_DEFRAG_NTYPES + t]);
	}

	for (g = 0; g < MARIE_DEFRAG_NGENS; g++) {
		for (t = 0; t < MARIE_DEFRAG_NTYPES; t++) {
			long occ = atomic_long_read(&marie_gen_occupied[g][t]);

			if (hist_sum[g][t] != occ) {
				mism++;
				len += scnprintf(buf + len, cap - len,
					"MISMATCH gen %d type %d: hist %ld occ %ld\n",
					g, t, hist_sum[g][t], occ);
			}
		}
	}
	len += scnprintf(buf + len, cap - len, "invariant %s (%d mismatch)\n",
			 mism ? "FAIL" : "OK", mism);
	return len;
}

/* marie_defrag_invariant_report() is consumed by the defrag_stats read below. */

/*
 * ---------------------------------------------------------------------
 * Static cost scorer (Step 2) -- read-only block_cost ranking.
 * ---------------------------------------------------------------------
 *
 * block_cost(B) = sum over a block's occupants of a per-(age, type) cost
 * weight: the sufficient-statistic dot product of defrag_design.md §6.2.
 * Coefficients are STATIC relative units encoding the §5 evacuation ladder
 * (Step 5 replaces them with boot-measured / learned ns). Read-only: this
 * ranks and reports candidate source blocks; it migrates nothing.
 */

/* Relative per-occupant evacuation cost (abstract units). */
#define MARIE_DEFRAG_C_DROP	 1u	/* cold, clean, unmapped file: ~unlink (near free) */
#define MARIE_DEFRAG_C_MOVE_FILE	 3u	/* warm clean unmapped file: copy + xarray, no TLB */
#define MARIE_DEFRAG_C_MOVE_ANON	16u	/* anon: always mapped -> copy + rmap + TLB shootdown */

/*
 * Per-occupant cost. Drop-eligibility is K=1: only the single gen
 * marie_find_oldest_occupied_mlv(FILE) reports as the oldest still-occupied
 * file gen is drop-eligible (near free); anon is TLB-bound regardless of age
 * because it is always mapped. @oldest_file is that gen (or -1 if no file gen
 * is occupied at all, e.g. immediately post-boot -- nothing is droppable
 * then). marie_find_oldest_occupied_mlv scans forward from head+1, so it can
 * never return head itself: the head's own (youngest) gen is structurally
 * excluded, not just numerically unlikely. Volatile mapped/dirty refinement
 * is left to phase 2 / the later learned coeffs; phase 1 scores on the stable
 * (gen, type) class.
 */
static inline u32 marie_defrag_coeff(int gen, int type, int oldest_file)
{
	if (type)	/* FILE */
		return (oldest_file >= 0 && gen == oldest_file) ? MARIE_DEFRAG_C_DROP
								 : MARIE_DEFRAG_C_MOVE_FILE;
	return MARIE_DEFRAG_C_MOVE_ANON;	/* ANON */
}

static long marie_defrag_block_occupancy(const struct marie_defrag_block_hist *h)
{
	long occ = 0;
	int c;

	for (c = 0; c < MARIE_DEFRAG_NCLASS; c++)
		occ += atomic_read(&h->count[c]);
	return occ;
}

static u64 marie_defrag_block_cost(const struct marie_defrag_block_hist *h, int oldest_file)
{
	u64 cost = 0;
	int g;

	for (g = 0; g < MARIE_DEFRAG_NGENS; g++) {
		long ca = atomic_read(&h->count[g * MARIE_DEFRAG_NTYPES + 0]);
		long cf = atomic_read(&h->count[g * MARIE_DEFRAG_NTYPES + 1]);

		cost += (u64)ca * marie_defrag_coeff(g, 0, oldest_file);
		cost += (u64)cf * marie_defrag_coeff(g, 1, oldest_file);
	}
	return cost;
}

struct marie_defrag_cand {
	u64		cost;
	unsigned long	blk;
	long		occ;
};

/* Insert (cost, blk, occ) into the ascending top-K array holding *n entries. */
static void marie_defrag_topk_insert(struct marie_defrag_cand *top, int *n, u64 cost,
			    unsigned long blk, long occ)
{
	int i;

	if (*n == MARIE_DEFRAG_TOPK && cost >= top[MARIE_DEFRAG_TOPK - 1].cost)
		return;
	i = (*n < MARIE_DEFRAG_TOPK) ? (*n)++ : MARIE_DEFRAG_TOPK - 1;
	for (; i > 0 && top[i - 1].cost > cost; i--)
		top[i] = top[i - 1];
	top[i].cost = cost;
	top[i].blk = blk;
	top[i].occ = occ;
}

/*
 * Phase 1: rank every non-empty MIGRATE_MOVABLE pageblock by block_cost into
 * the ascending top-K array. Shared by the read-only candidate listing and the
 * compaction driver. Returns the count collected (<= MARIE_DEFRAG_TOPK).
 */
static int marie_defrag_collect(struct marie_defrag_cand *top, unsigned long *scanned_out,
		       unsigned long *movable_out)
{
	int oldest_file = marie_find_oldest_occupied_mlv(1);
	unsigned long b, scanned = 0, movable = 0;
	int n = 0;

	for (b = 0; b < marie_defrag_nr_blocks; b++) {
		struct marie_defrag_block_hist *h = &marie_defrag_hist[b];
		unsigned long pfn = b << pageblock_order;
		long occ = marie_defrag_block_occupancy(h);

		if (occ == 0)
			continue;
		scanned++;
		if (!pfn_valid(pfn))
			continue;
		if (get_pageblock_migratetype(pfn_to_page(pfn)) != MIGRATE_MOVABLE)
			continue;
		movable++;
		marie_defrag_topk_insert(top, &n, marie_defrag_block_cost(h, oldest_file),
				b, occ);
	}
	if (scanned_out)
		*scanned_out = scanned;
	if (movable_out)
		*movable_out = movable;
	return n;
}

/* marie_defrag_collect() is the selection used by marie_defrag_topn (scanned/
 * movable out-params unused now; pass NULL). */

/*
 * ---------------------------------------------------------------------
 * Migration front-end (Step 3 + Step 6 target selection).
 * ---------------------------------------------------------------------
 *
 * Evacuate the Marie-tracked (movable LRU) pages of the cheapest source
 * pageblocks via the core migrate_pages() machinery -- design P5: Marie defrag
 * contributes the SELECTION, not the migration. page_isolate_lru() routes
 * through Marie's del (untracking the source page, decrementing the histogram
 * at the source block); migrate_pages() copies + remaps and re-adds each
 * destination via lru_cache_add() -> lru_marie_add_page() (re-installing it,
 * incrementing the histogram at the destination block). The move is therefore
 * histogram-coherent for free.
 *
 * Step 6 fixes step 3's net-yield problem: rather than letting the buddy place
 * targets (alloc_migration_target, which splits high-order free blocks to find
 * order-0 target pages and so fragments them), Marie defrag pre-harvests target pages
 * from the sub-pageblock free HOLES of OTHER partial movable blocks
 * (marie_defrag_harvest_block via __isolate_free_page, capped at MARIE_DEFRAG_HARVEST_MAX_ORDER
 * so a near-free order-9 block is never broken) and hands them out
 * (marie_defrag_alloc_target / marie_defrag_free_target, mirroring compaction_alloc/free).
 * Evacuees thus fill existing holes -- the source blocks free into order-9
 * blocks, the harvested-from blocks get denser, and no high-order block is
 * split. DROP of cold-dead clean file is still deferred to step 4; this step
 * MOVES every occupant.
 */

/*
 * Cumulative diagnostic counters, exposed ONLY via the /sys .../defrag_stats
 * read while the debug flag (marie_defrag_stats) is on -- for an A/B test vs
 * stock compaction (read totals before and after a window, diff). No per-run
 * dmesg output. Maintained unconditionally (a few atomics per run) so the A/B
 * window does not depend on when the flag was flipped.
 */
static unsigned int marie_defrag_stats;		/* debug flag: gates the stats read */
static atomic_long_t marie_defrag_tot_runs, marie_defrag_tot_move,
		     marie_defrag_tot_migrated, marie_defrag_tot_dropped,
		     marie_defrag_tot_freed, marie_defrag_tot_restamped;
/*
 * Master switch (sysfs /sys/kernel/mm/lru_marie/defrag): 1 = Marie defrag
 * REPLACES stock compaction on both kcompactd paths (DEFAULT -- the build
 * already opted in via CONFIG_LRU_MARIE_DEFRAG, so an installed defrag kernel
 * is active out of the box); 0 = stock kernel compaction, Marie dormant.
 */
static unsigned int marie_defrag_enabled = 1;
static atomic_long_t marie_defrag_fires;	/* proactive replacements done (stat) */

/*
 * Safety killswitch (sysfs /sys/kernel/mm/lru_marie/defrag_drop): 1 (DEFAULT)
 * = urgent/direct compaction may DROP cold-dead clean file as designed; 0 =
 * never DROP, even when the caller's may_drop says the path is urgent --
 * compaction falls back to MOVE-only, identical to the proactive path. Lets
 * an operator disable the drop-and-refault-risk rung alone if it proves
 * harmful on a given workload, without losing Marie defrag's MOVE-based
 * compaction (that would require the coarser .../defrag master switch).
 */
static unsigned int marie_defrag_drop_enabled = 1;

/* Cap harvested hole order so we never break a near-pageblock free block. */
#define MARIE_DEFRAG_HARVEST_MAX_ORDER	3

/*
 * Age neutrality across migration. The generic migrate path re-adds the
 * destination via lru_cache_add() -> marie_page_install(), which installs at
 * the HEAD (youngest) gen -- rejuvenating a page that defrag merely relocated.
 * To undo that, after migration we move each dst off head back to the SOURCE
 * page's own gen, so a cold source stays cold and a warm source stays warm:
 * the age *distribution* survives the relocation, not just its coldest end.
 *
 * The source gen is absolute, but head advances while the (possibly long)
 * MIGRATE_SYNC runs (the run's own dst installs drive the install-cadence
 * clock). Because gens are a mod-N ring, a source gen the advancing head has
 * LAPPED past ("beyond tail" -- its slot no longer exists) is indistinguishable
 * by number from a still-valid old gen, and replaying it could land at/ahead of
 * head and pin the head-advance gate (gen_occupied[next] != 0) into an
 * OOM-livelock. So the restamp keeps the absolute source gen ONLY while it
 * still lies within the current occupied arc [oldest .. head] (age <= oldest's
 * age); once lapped it falls back to the current oldest -- frame-relative, so it
 * can never pin the gate. See marie_defrag_restamp_target().
 *
 * TIER is preserved from the dst's own byte (migration copies PG_active /
 * PG_workingset, which install reads); only the GEN is corrected. Capturing the
 * source gen needs a side channel: page_isolate_lru clears the source byte, and
 * the src<->dst pairing is first visible in alloc_target -- so isolation records
 * (src pfn, gen) into @srcmap (sorted, bsearch'd by alloc_target) and alloc_target
 * records (dst pfn, that gen) into @restamp for the tail to apply. pfn fits in
 * u32: Marie requires max_pfn < 2^32 (MARIE_MAX_SUPPORTED_PFN).
 *
 * Both tables are PRE-ALLOCATED once at init (marie_defrag_scratch_alloc), never
 * on the defrag path: defrag runs under memory pressure, where a per-run kvmalloc
 * could fail and silently drop the age correction. They are shared, so a single
 * owner runs at a time (marie_defrag_busy trylock in lru_marie_defrag_pgdat); a
 * concurrent caller skips (best-effort).
 */

/* Target free-page pool, harvested from partial blocks (compaction-style). */
struct marie_defrag_freectx {
	struct list_head freepages[NR_PAGE_ORDERS];
	unsigned long	 nr;		/* order-0-equivalent pages available now */
	unsigned long	 scan_cursor;	/* next block to harvest holes from */
	unsigned long	 harvested;	/* cumulative pages harvested (stats) */
	/*
	 * Age-neutrality scratch: borrowed pointers into the pre-allocated
	 * global tables (NULL = init alloc failed -> fall back to head install).
	 * @srcmap: (src pfn, gen) captured at isolation, sorted by pfn.
	 * @restamp: (dst pfn, intended gen) recorded at alloc_target.
	 * Both bounded by @cap (the worst-case run occupancy).
	 */
	struct marie_defrag_pg	*srcmap;
	unsigned int		 srcmap_n;
	struct marie_defrag_pg	*restamp;
	unsigned int		 restamp_n;
	unsigned int		 cap;
};

/* A refill grabs at least this many pages per scan, to amortise the walk. */
#define MARIE_DEFRAG_HARVEST_BATCH	64

static void marie_defrag_harvest_refill(struct marie_defrag_freectx *fc, unsigned long min_pages);

static void marie_defrag_freectx_init(struct marie_defrag_freectx *fc)
{
	int o;

	for (o = 0; o < NR_PAGE_ORDERS; o++)
		INIT_LIST_HEAD(&fc->freepages[o]);
	fc->nr = 0;
	fc->scan_cursor = 0;
	fc->harvested = 0;
	/*
	 * Borrow the pre-allocated scratch (single-owner via marie_defrag_busy).
	 * If the init alloc failed both stay NULL and the run degrades to the
	 * plain head install -- no age correction, but no per-run alloc either.
	 */
	fc->srcmap = marie_defrag_srcmap;
	fc->restamp = marie_defrag_restamp;
	fc->cap = (marie_defrag_srcmap && marie_defrag_restamp) ?
			(unsigned int)marie_defrag_scratch_cap : 0;
	fc->srcmap_n = fc->restamp_n = 0;
}

/* Sort key for @srcmap so alloc_target can binary-search a migrating src's gen. */
static int marie_defrag_pg_cmp(const void *a, const void *b)
{
	u32 pa = ((const struct marie_defrag_pg *)a)->pfn;
	u32 pb = ((const struct marie_defrag_pg *)b)->pfn;

	return (pa > pb) - (pa < pb);
}

/*
 * Look up the gen captured at isolation for source @pfn. @srcmap is sorted by
 * pfn (marie_defrag_topn sorts it before migration starts). Returns the gen, or
 * MARIE_DEFRAG_GEN_NONE if not found (the caller then falls back to oldest).
 */
static u8 marie_defrag_srcmap_gen(struct marie_defrag_freectx *fc, unsigned long pfn)
{
	int lo = 0, hi = (int)fc->srcmap_n - 1;
	u32 key = (u32)pfn;

	while (lo <= hi) {
		int mid = (lo + hi) >> 1;
		u32 mp = fc->srcmap[mid].pfn;

		if (mp == key)
			return fc->srcmap[mid].gen;
		if (mp < key)
			lo = mid + 1;
		else
			hi = mid - 1;
	}
	return MARIE_DEFRAG_GEN_NONE;
}

/*
 * Where to restamp a migrated dst, given its captured source gen and the CURRENT
 * ring (@head, @oldest for the dst's type). Keep the absolute source gen while
 * it still lies within the occupied arc [oldest .. head] -- i.e. its head-relative
 * age has not passed the oldest's age. Once the advancing head has lapped it
 * ("beyond tail", or an unknown source), promote to @oldest: frame-relative, so
 * it tracks head and can never re-populate the head-advance gate slot. Worst case
 * is thus either a benign install at head (a genuinely young source) or at oldest
 * (== the old always-oldest behaviour) -- never a gate pin.
 */
static int marie_defrag_restamp_target(u8 src_gen, u8 head, int oldest)
{
	unsigned int mask = MARIE_PFN_NR_GENS - 1;
	unsigned int age_src, age_oldest;

	if (src_gen == MARIE_DEFRAG_GEN_NONE)
		return oldest;
	src_gen &= mask;
	age_src = (head - src_gen) & mask;
	age_oldest = (head - (u8)oldest) & mask;
	return age_src <= age_oldest ? (int)src_gen : oldest;
}

/*
 * get_new_page for migrate_pages: hand out a target from the harvested pool,
 * split down to the source page's order. Mirrors compaction_alloc(): when the
 * pool runs dry it refills by scanning FORWARD for more partial blocks to
 * harvest holes from (marie_defrag_harvest_refill), so a drained pool selects new
 * destinations and continues instead of failing the rest of the batch. NULL is
 * returned only when no holes remain anywhere -- migrate_pages then puts the
 * source page back.
 */
static struct page *marie_defrag_alloc_target(struct page *src, unsigned long data)
{
	struct marie_defrag_freectx *fc = (struct marie_defrag_freectx *)data;
	int order = compound_order(src);
	struct page *freepage;
	unsigned long size;
	int so;

	for (so = order; so < NR_PAGE_ORDERS; so++)
		if (!list_empty(&fc->freepages[so]))
			break;
	if (so == NR_PAGE_ORDERS) {
		marie_defrag_harvest_refill(fc, max_t(unsigned long, 1UL << order,
					     MARIE_DEFRAG_HARVEST_BATCH));
		for (so = order; so < NR_PAGE_ORDERS; so++)
			if (!list_empty(&fc->freepages[so]))
				break;
		if (so == NR_PAGE_ORDERS)
			return NULL;
	}

	freepage = list_first_entry(&fc->freepages[so], struct page, lru);
	size = 1UL << so;
	list_del(&freepage->lru);
	while (so > order) {
		so--;
		size >>= 1;
		list_add(&freepage[size].lru, &fc->freepages[so]);
		set_page_private(&freepage[size], so);
	}
	fc->nr -= 1UL << order;

	/*
	 * Turn the harvested raw free page into a refcounted allocation. Since
	 * 6.14 post_alloc_hook() no longer sets the refcount, so this must be
	 * paired with set_page_refcounted() (marie_defrag_prep_allocated); a
	 * refcount-0 dst here would be freed mid-migration by lru_cache_add()
	 * (bad_page + live-anon corruption). See defrag_compat.h.
	 */
	marie_defrag_prep_allocated(freepage, order);
	if (order)
		prep_compound_page(freepage, order);

	/*
	 * Age neutrality: pair this dst with its source's gen (looked up from
	 * @srcmap by the src pfn). After migration installs the dst at head,
	 * marie_defrag_topn moves it back to that gen (see restamp_target), so
	 * the relocation does not rejuvenate it. A dst whose migration later
	 * fails is freed without ever being installed, so its byte is not TRACKED
	 * and the restamp simply skips it; a source missing from @srcmap yields
	 * MARIE_DEFRAG_GEN_NONE and falls back to oldest.
	 */
	if (fc->restamp && fc->restamp_n < fc->cap) {
		fc->restamp[fc->restamp_n].pfn = (u32)page_to_pfn(freepage);
		fc->restamp[fc->restamp_n].gen =
			marie_defrag_srcmap_gen(fc, page_to_pfn(src));
		fc->restamp_n++;
	}

	return page_rmappable_page(freepage);
}

/*
 * put_new_page: return an unused target to the pool. Mirrors compaction_free().
 *
 * Upstream unwraps the folio (@dst->page) and calls free_pages_prepare() before
 * re-listing.  Neither survives the backport:
 *
 *  - @dst is already a struct page here, so the unwrap is dropped.
 *
 *  - free_pages_prepare() is exported via mm/internal.h on 6.x, but in this
 *    tree it is static __always_inline in mm/page_alloc.c (page_alloc.c:1144)
 *    and takes a third @check_free argument -- it is unreachable from here in
 *    any spelling.  It is also unnecessary: it is the buddy-allocator free
 *    path's teardown (poisoning, debug checks, page-owner), and these pages
 *    are not going back to the buddy here -- they return to our own free pool
 *    to be handed out again by marie_defrag_alloc_target(), which re-preps
 *    them via marie_defrag_prep_allocated().  This tree's own
 *    compaction_free() (mm/compaction.c:1575) does exactly the same thing:
 *    list_add() straight onto the free list, no prepare step.  The teardown
 *    that does matter happens in marie_defrag_freectx_release(), which preps
 *    and __free_pages() each leftover.
 *
 * The refcount drop is kept: migrate.c calls put_new_page() on a target it
 * never used, holding the reference marie_defrag_prep_allocated() set.
 */
static void marie_defrag_free_target(struct page *dst, unsigned long data)
{
	struct marie_defrag_freectx *fc = (struct marie_defrag_freectx *)data;
	int order = compound_order(dst);

	if (put_page_testzero(dst)) {
		list_add(&dst->lru, &fc->freepages[order]);
		fc->nr += 1UL << order;
	}
}

/* Return leftover (unused) pool pages to the buddy. Mirrors release_free_list(). */
static void marie_defrag_freectx_release(struct marie_defrag_freectx *fc)
{
	int order;

	for (order = 0; order < NR_PAGE_ORDERS; order++) {
		struct page *page, *next;

		list_for_each_entry_safe(page, next, &fc->freepages[order], lru) {
			list_del(&page->lru);
			/*
			 * Mirror release_free_list()/mark_allocated(): __free_pages()'s
			 * put_page_testzero() needs the refcount at 1 to actually free,
			 * so prep the page the same way as a handed-out target (else the
			 * pool leaks on 6.14+). See defrag_compat.h.
			 */
			marie_defrag_prep_allocated(page, order);
			__free_pages(page, order);
		}
	}
	fc->nr = 0;
	/* Scratch is pre-allocated global state; nothing to free here. */
}

/*
 * Harvest sub-pageblock free holes from one block into the pool (until the pool
 * reaches @want). Mirrors isolate_freepages_block: under zone->lock, small
 * PageBuddy free runs are removed via __isolate_free_page. Free runs larger
 * than MARIE_DEFRAG_HARVEST_MAX_ORDER are skipped, so a near-free order-9 block is never
 * broken -- the whole point of the smart-target step.
 */
static void marie_defrag_harvest_block(struct marie_defrag_freectx *fc, unsigned long blk,
			      unsigned long want)
{
	unsigned long start = blk << pageblock_order;
	unsigned long pfn, end = start + (1UL << pageblock_order);
	struct zone *zone;
	unsigned long flags;

	if (fc->nr >= want || !pfn_valid(start))
		return;
	zone = page_zone(pfn_to_page(start));
	/*
	 * Gate the whole pageblock exactly as stock isolate_freepages() does via
	 * pageblock_pfn_to_page(): it rejects an offline/invalid start or end
	 * and -- the corruption guard -- a block that STRADDLES a zone boundary
	 * (a zone may end mid-pageblock). __isolate_free_page() below derives the
	 * target from page_zone(page) INTERNALLY, so every page we hand it under
	 * @zone->lock must belong to @zone; a straddling page would corrupt the
	 * neighbouring zone's free_area unlocked. @zone is only a tentative guess
	 * here (an offline start could make it stale), but pageblock_pfn_to_page()
	 * re-derives via pfn_to_online_page() and bails on any mismatch.
	 */
	if (!pageblock_pfn_to_page(start, end, zone))
		return;

	spin_lock_irqsave(&zone->lock, flags);
	for (pfn = start; pfn < end && fc->nr < want; pfn++) {
		struct page *page = pfn_to_page(pfn);
		unsigned int order;

		if (!PageBuddy(page))
			continue;
		order = page_order(page);
		if (order > MARIE_DEFRAG_HARVEST_MAX_ORDER ||
		    !__isolate_free_page(page, order)) {
			pfn += (1UL << order) - 1;	/* skip this free run */
			continue;
		}
		set_page_private(page, order);
		list_add_tail(&page->lru, &fc->freepages[order]);
		fc->nr += 1UL << order;
		fc->harvested += 1UL << order;
		pfn += (1UL << order) - 1;
	}
	spin_unlock_irqrestore(&zone->lock, flags);
}

/*
 * Refill the target pool by scanning FORWARD from fc->scan_cursor for more
 * partial movable blocks to harvest holes from (compaction's isolate_freepages
 * analogue) until the pool holds @min_pages or the blocks are exhausted. occ==0
 * blocks are skipped: that covers genuinely-free blocks AND just-emptied source
 * blocks, whose own free pages must stay free to coalesce into the freed
 * order-9 rather than be re-consumed as targets.
 */
static void marie_defrag_harvest_refill(struct marie_defrag_freectx *fc, unsigned long min_pages)
{
	while (fc->nr < min_pages && fc->scan_cursor < marie_defrag_nr_blocks) {
		unsigned long b = fc->scan_cursor++;
		unsigned long pfn = b << pageblock_order;

		if (marie_defrag_block_occupancy(&marie_defrag_hist[b]) == 0)
			continue;
		if (!pfn_valid(pfn) ||
		    get_pageblock_migratetype(pfn_to_page(pfn)) != MIGRATE_MOVABLE)
			continue;
		marie_defrag_harvest_block(fc, b, min_pages);
	}
}

/*
 * Isolate every tracked movable page of one block into @movelist (the standard
 * lock-free get_page_unless_zero + page_isolate_lru; isolate fails -- skipped -- for a
 * page another path already claimed, incl. a Marie page mid-reclaim). Returns
 * the count isolated.
 */
/*
 * A cold-dead, clean, unmapped FILE page is DROPPED (reclaimed) rather than
 * moved: near-free -- no target page, no copy, no TLB (design §5 rung 1). @s is
 * the per-PFN byte read just before isolation (so it still carries the gen).
 * Conservative gate: only the single oldest still-occupied file gen
 * (@oldest_file, K=1 -- marie_find_oldest_occupied_mlv(FILE), NOT a static
 * head-relative age band: that formula assumed a fully-populated ring and
 * under-fired -- never dropping anything -- whenever file occupancy was
 * sparse), and only clean + unmapped + file -- exactly the pages Marie's own
 * swappiness=1 reclaim would drop first, so the refault risk is the normal
 * cold-cache one. @oldest_file < 0 means no file gen is occupied at all,
 * so nothing is droppable. marie_find_oldest_occupied_mlv scans forward from
 * head+1 and so can never return head itself -- the head's own (youngest)
 * gen is structurally excluded from this test, never just a numeric
 * near-miss. anon is never dropped (that would need swap, defeating
 * "near-free"); the per-gen learned p_refault crossover (design §8.2) is a
 * later refinement of this K=1 rule.
 */
static bool marie_defrag_droppable(struct page *page, u8 s, int oldest_file)
{
	int gen;

	if (oldest_file < 0)
		return false;
	if (!(s & MARIE_PFN_TYPE_FILE))
		return false;
	if (PageDirty(page) || PageWriteback(page))
		return false;
	if (page_mapped(page) || PageUnevictable(page))
		return false;
	gen = (s & MARIE_PFN_GEN_MASK) >> MARIE_PFN_GEN_SHIFT;
	return gen == oldest_file;
}

/*
 * How many pages of clean file cache Marie defrag may DROP before breaching the
 * clean_min_ratio floor -- the same reserve Marie's reclaim protects via
 * marie_file_floor_protect (NR_FILE_DIRTY excluded: dirty pages are not
 * droppable and must not count toward the reserve). LONG_MAX when the floor is
 * disabled (ratio 0). Used as a running budget so Marie defrag drops cold file down to,
 * but never below, the floor; pages past the budget fall through to MOVE.
 *
 * Desktop/global-only: computed for the first online node. (NUMA-per-node
 * budgeting is a refinement; Marie's own floor is likewise node-scoped.)
 */
static long marie_defrag_drop_budget(struct pglist_data *pgdat)
{
	unsigned int ratio = READ_ONCE(marie_clean_min_ratio);
	unsigned long file, dirty, file_min;

	if (!ratio)
		return LONG_MAX;
	file = node_page_state(pgdat, NR_ACTIVE_FILE) +
	       node_page_state(pgdat, NR_INACTIVE_FILE);
	dirty = node_page_state(pgdat, NR_FILE_DIRTY);
	file = file > dirty ? file - dirty : 0;
	file_min = pgdat->node_present_pages * ratio / 100;
	return file > file_min ? (long)(file - file_min) : 0;
}

/*
 * Isolate every tracked movable page of one block, classifying each as a DROP
 * (cold-dead clean file -> @drop_list) or a MOVE (-> @move_list). The standard
 * lock-free get_page_unless_zero + page_isolate_lru; isolate fails (skipped) for a
 * page another path already claimed. The droppable test reads the per-PFN byte
 * BEFORE page_isolate_lru clears it.
 */
static void marie_defrag_isolate_block(struct marie_defrag_freectx *fc, unsigned long blk,
			      struct list_head *move_list,
			      struct list_head *drop_list, int oldest_file,
			      bool may_drop, long *drop_budget,
			      int *n_move, int *n_drop)
{
	unsigned long start = blk << pageblock_order;
	unsigned long pfn, end = start + (1UL << pageblock_order);

	if (end > marie_state_size)
		end = marie_state_size;

	for (pfn = start; pfn < end; pfn++) {
		struct page *page;
		long nr;
		bool drop;
		u8 s = READ_ONCE(marie_state[pfn]);

		if (!(s & MARIE_PFN_TRACKED) || !pfn_valid(pfn))
			continue;
		page = pfn_to_page(pfn);
		nr = compound_nr(page);
		if (PageCompound(page) && nr >= (1UL << pageblock_order))
			continue;	/* compound fills block: already contiguous */
		if (!get_page_unless_zero(page))
			continue;
		/*
		 * @may_drop reflects the request's urgency. Urgent/direct
		 * compaction (an allocation is blocked) drops cold-dead clean
		 * file to free a block with no target -- a refault is cheaper
		 * than the allocation failing. Background/proactive compaction
		 * has no blocked requester, so it MOVES everything and sacrifices
		 * no cache. @drop_budget caps drops at the clean_min_ratio floor;
		 * a droppable page past the budget falls through to MOVE, so the
		 * protected clean-file reserve is never breached.
		 */
		drop = may_drop && *drop_budget >= nr &&
		       marie_defrag_droppable(page, s, oldest_file);
		if (page_isolate_lru(page)) {
			if (drop) {
				list_add(&page->lru, drop_list);
				(*n_drop)++;
				*drop_budget -= nr;
			} else {
				list_add(&page->lru, move_list);
				(*n_move)++;
				/*
				 * Age neutrality: record this source's (pfn, gen)
				 * so its migrated dst is restamped back to this
				 * gen. @s was read above, before page_isolate_lru
				 * cleared the byte. page_to_pfn (the head pfn) is the
				 * key alloc_target will look up via page_to_pfn(src).
				 */
				if (fc->srcmap && fc->srcmap_n < fc->cap) {
					fc->srcmap[fc->srcmap_n].pfn =
						(u32)page_to_pfn(page);
					fc->srcmap[fc->srcmap_n].gen =
						(s & MARIE_PFN_GEN_MASK) >>
							MARIE_PFN_GEN_SHIFT;
					fc->srcmap_n++;
				}
			}
		}
		put_page(page);
	}
}

/*
 * Compact the @n cheapest movable source blocks (capped at MARIE_DEFRAG_TOPK): isolate
 * all their pages, harvest an equal number of target holes from OTHER partial
 * movable blocks, then migrate into the harvested pool.
 *
 * @drop_need is the caller's marie_defrag_drop_need() result: 0 means DROP is
 * off for this run (MOVE only), otherwise the number of pages this specific
 * blocked allocation needs. The actual drop_budget is capped at that need --
 * see marie_defrag_drop_need() -- and separately at the clean_min_ratio floor
 * (marie_defrag_drop_budget()), whichever is smaller.
 */
static void marie_defrag_topn(unsigned int n, unsigned long drop_need)
{
	struct marie_defrag_cand top[MARIE_DEFRAG_TOPK];
	struct marie_defrag_freectx fc;
	LIST_HEAD(move_list);
	LIST_HEAD(drop_list);
	int oldest_file;
	bool may_drop = drop_need > 0;
	long drop_budget;
	unsigned int blocks = 0, migrated = 0, dropped = 0, freed = 0, restamped = 0;
	int collected, i, n_move = 0, n_drop = 0, nr_failed_move = 0;
	struct page *mpage;

	if (!marie_defrag_hist)
		return;
	collected = marie_defrag_collect(top, NULL, NULL);
	if (n > (unsigned int)collected)
		n = collected;
	if (!n)
		return;
	atomic_long_inc(&marie_defrag_tot_runs);

	/*
	 * Borrow the pre-allocated age-neutrality scratch (no per-run alloc on
	 * this under-pressure path). isolation fills @srcmap, alloc_target reads
	 * it and fills @restamp; the tail applies each restamp.
	 */
	marie_defrag_freectx_init(&fc);

	/*
	 * Isolate every source page, splitting into DROP (cold-dead clean file)
	 * and MOVE. This already decremented the source histograms, so the source
	 * blocks read occupancy 0 below.
	 */
	oldest_file = marie_find_oldest_occupied_mlv(1);
	drop_budget = may_drop ?
		min_t(long, (long)drop_need,
		      marie_defrag_drop_budget(NODE_DATA(first_online_node))) : 0;
	for (i = 0; i < (int)n; i++) {
		marie_defrag_isolate_block(&fc, top[i].blk, &move_list, &drop_list,
				  oldest_file, may_drop, &drop_budget, &n_move, &n_drop);
		blocks++;
		cond_resched();
	}
	if (!n_move && !n_drop) {
		marie_defrag_freectx_release(&fc);
		return;
	}

	/* Sort @srcmap by pfn so alloc_target can binary-search each src's gen. */
	if (fc.srcmap && fc.srcmap_n)
		sort(fc.srcmap, fc.srcmap_n, sizeof(*fc.srcmap),
		     marie_defrag_pg_cmp, NULL);

	/*
	 * Only MOVE pages need a target hole; DROP pages are reclaimed in
	 * place. marie_defrag_alloc_target harvests holes ON DEMAND from OTHER partial
	 * movable blocks as the pool drains (the cursor scans forward), so a
	 * drained pool selects new destinations and continues instead of failing
	 * the rest of the batch. MIGRATE_SYNC (not _LIGHT) so file pages with
	 * buffer_heads and pages under writeback are actually moved, not skipped.
	 */
	if (!list_empty(&move_list)) {
		/*
		 * migrate_pages() grew its trailing @ret_succeeded out-param in
		 * 5.17; this tree's signature ends at @reason (migrate.h:68) and
		 * returns nr_failed, discarding its internal nr_succeeded.
		 *
		 * Recover the count from the list instead of the return value.
		 * migrate_pages() removes every page it migrates successfully
		 * from @from and leaves exactly the failures on it (the -EAGAIN
		 * retries are folded into nr_failed on the final pass), so the
		 * surviving length is the failure count and n_move minus it is
		 * the number migrated -- the same quantity @ret_succeeded
		 * reports upstream.
		 *
		 * Counting the remainder is also why this is done before the
		 * putback below, which empties the list.
		 */
		migrate_pages(&move_list, marie_defrag_alloc_target, marie_defrag_free_target,
			      (unsigned long)&fc, MIGRATE_SYNC,
			      MR_COMPACTION);
		list_for_each_entry(mpage, &move_list, lru)
			nr_failed_move++;
		migrated = (unsigned int)max(0, n_move - nr_failed_move);
		if (!list_empty(&move_list))
			putback_movable_pages(&move_list);
	}

	/*
	 * Age neutrality: move each migrated dst off the head gen (where the
	 * generic lru_cache_add install landed it) back to its SOURCE page's gen
	 * (marie_defrag_restamp_target: keep the absolute gen while it is still in
	 * the occupied arc; fall back to the current oldest once head has lapped
	 * it), preserving tier -- migration is a relocation, not an access, so it
	 * must not shift the page's age. Done BEFORE freectx_release: a dst whose
	 * migration FAILED was returned to the pool (marie_defrag_free_target),
	 * never installed, so its byte is not TRACKED and the read below skips it.
	 * Successful dsts are TRACKED at head and get relocated cleanly (the move
	 * updates byte + bitmap + gen_occupied + per-block histogram coherently, so
	 * the Sigma_blocks hist == gen_occupied invariant holds).
	 *
	 * head/oldest are snapshot once per type here (post-migration): the arc
	 * test needs one consistent frame, and our moves only add into [oldest ..
	 * head] so they never create a new oldest mid-loop.
	 */
	if (fc.restamp) {
		u8 head[ANON_AND_FILE];
		int oldest[ANON_AND_FILE];
		unsigned int j;

		for (i = 0; i < ANON_AND_FILE; i++) {
			head[i] = (u8)atomic_read(&marie_head_gen[i]);
			oldest[i] = marie_find_oldest_occupied_mlv(i);
		}

		for (j = 0; j < fc.restamp_n; j++) {
			unsigned long dpfn = fc.restamp[j].pfn;
			int type, target;
			u8 s;

			if (dpfn >= marie_state_size)
				continue;
			s = READ_ONCE(marie_state[dpfn]);
			if (!(s & MARIE_PFN_TRACKED))
				continue;	/* failed dst: never installed */
			type = (s & MARIE_PFN_TYPE_FILE) ? 1 : 0;
			if (oldest[type] < 0)
				continue;	/* only head occupied: nowhere older */
			target = marie_defrag_restamp_target(fc.restamp[j].gen,
							     head[type], oldest[type]);
			marie_state_move_to_gen(dpfn, (u8)target,
				(s & MARIE_PFN_TIER_MASK) >> MARIE_PFN_TIER_SHIFT);
			restamped++;
		}
	}

	marie_defrag_freectx_release(&fc);

	/*
	 * DROP: reclaim the cold-dead clean file pages. reclaim_pages() drops the
	 * clean ones (no I/O, no target) and puts back any that resisted. The
	 * source-histogram decrement already happened at isolation, so a dropped
	 * page simply never re-installs.
	 */
	if (!list_empty(&drop_list))
		dropped = reclaim_pages(&drop_list);

	for (i = 0; i < (int)n; i++)
		if (marie_defrag_block_occupancy(&marie_defrag_hist[top[i].blk]) == 0)
			freed++;

	/* Cumulative diagnostics (read via .../defrag_stats when its flag is on). */
	atomic_long_add(n_move, &marie_defrag_tot_move);
	atomic_long_add(migrated, &marie_defrag_tot_migrated);
	atomic_long_add(dropped, &marie_defrag_tot_dropped);
	atomic_long_add(freed, &marie_defrag_tot_freed);
	atomic_long_add(restamped, &marie_defrag_tot_restamped);
	(void)blocks;
}

/*
 * /sys/kernel/mm/lru_marie/defrag_stats -- A/B / debug node.
 *   write 0/1  : the debug flag (default 0). 1 enables the diagnostic read.
 *   read       : while the flag is on, the cumulative defrag counters (runs,
 *                pages moved/migrated/dropped, blocks freed, dsts restamped,
 *                kcompactd replace fires) plus the histogram completeness
 *                invariant (Sigma_blocks hist == gen_occupied). Off => "disabled".
 *
 * For an A/B vs stock compaction: enable, read totals, run the workload window,
 * read again, diff -- compare migrated/freed against the kernel's compact_*
 * vmstat from a defrag=0 run. No per-run dmesg, no manual trigger.
 */
static ssize_t marie_defrag_stats_show(struct kobject *kobj, struct kobj_attribute *attr,
			    char *buf)
{
	int len;

	if (!READ_ONCE(marie_defrag_stats))
		return sysfs_emit(buf, "disabled\n");

	len = sysfs_emit(buf,
		"runs %ld move %ld migrated %ld dropped %ld freed %ld restamped %ld fires %ld\n"
		"nr_blocks %lu\n",
		atomic_long_read(&marie_defrag_tot_runs),
		atomic_long_read(&marie_defrag_tot_move),
		atomic_long_read(&marie_defrag_tot_migrated),
		atomic_long_read(&marie_defrag_tot_dropped),
		atomic_long_read(&marie_defrag_tot_freed),
		atomic_long_read(&marie_defrag_tot_restamped),
		atomic_long_read(&marie_defrag_fires),
		marie_defrag_nr_blocks);
	if (marie_defrag_hist)
		len = marie_defrag_invariant_report(buf, len, PAGE_SIZE);
	return len;
}

static ssize_t marie_defrag_stats_store(struct kobject *kobj, struct kobj_attribute *attr,
			     const char *buf, size_t count)
{
	unsigned int v;
	int err = kstrtouint(buf, 10, &v);

	if (err)
		return err;
	WRITE_ONCE(marie_defrag_stats, !!v);
	return count;
}

static struct kobj_attribute marie_defrag_stats_attr =
	__ATTR(defrag_stats, 0644, marie_defrag_stats_show, marie_defrag_stats_store);

/*
 * ---------------------------------------------------------------------
 * Master switch + kcompactd replacement.
 * ---------------------------------------------------------------------
 *
 * /sys/kernel/mm/lru_marie/defrag selects WHO performs kcompactd's defrag:
 *   0           -- stock kernel compaction; Marie dormant.
 *   1 (default) -- Marie defrag REPLACES stock compaction on BOTH kcompactd
 *                  paths. mm/compaction.c gates each swap on
 *                  lru_marie_defrag_active():
 *                    - demand (kcompactd_do_work; a high-order allocation is
 *                      blocked) -> lru_marie_defrag_pgdat(pgdat, may_drop=true):
 *                      URGENT, so MOVE + DROP cold-dead clean file -- Marie's
 *                      cheapest, highest-value rung (the clean_min_ratio floor
 *                      still bounds DROP).
 *                    - proactive (background timer) -> may_drop=false: MOVE
 *                      only, no DROP -- no allocation is waiting, so sacrificing
 *                      cache would be pure loss. The proactive loop's
 *                      fragmentation_score-progress defer applies to Marie's
 *                      result identically.
 *
 * may_drop is decided by the PATH (urgency), per the design (commit 95e36ac58f):
 * urgent/direct compaction may sacrifice cold cache to free a block target-free;
 * background must not. Runs in the kcompactd kthread context (sleeping allowed,
 * off the allocation hot path). The .../defrag_run node still triggers a manual
 * one-shot (+N urgent / -N background) regardless of the switch, for testing.
 *
 * /sys/kernel/mm/lru_marie/defrag_drop is a separate, finer-grained killswitch:
 * 0 forces every DROP decision to fall back to MOVE regardless of may_drop,
 * without touching the master defrag switch above (see marie_defrag_drop_enabled).
 *
 * DROP additionally requires marie_defrag_drop_need(pgdat) > 0: may_drop=true
 * alone only means the demand path is order-blocked (fragmented), not that
 * free memory is actually short, so it is gated on a WMARK_LOW check at the
 * blocked allocation's own order before anything is dropped. When it does
 * fire, the DROP budget is capped at that allocation's own size (1 << order),
 * not "drop everything droppable up to the clean_min_ratio floor" -- see
 * marie_defrag_drop_need() for why headroom already free elsewhere isn't
 * credited against the budget.
 */
#define MARIE_DEFRAG_BATCH	16		/* source blocks per kcompactd wake */

/*
 * lru_marie_defrag_active - is the .../defrag switch on, i.e. should Marie
 * defrag replace stock compaction in kcompactd? Read by both compaction.c
 * paths. True only when Marie is enabled, the switch is on, and the histogram
 * exists.
 */
bool lru_marie_defrag_active(void)
{
	return lru_marie_enabled() && READ_ONCE(marie_defrag_enabled) &&
	       marie_defrag_hist;
}
EXPORT_SYMBOL_GPL(lru_marie_defrag_active);

/*
 * Is @pgdat under real reclaim pressure right now, as opposed to merely
 * fragmented, for the actual allocation kcompactd is trying to unblock? A
 * demand-path may_drop=true fires purely on allocation ORDER (a high-order
 * block is unavailable) and says nothing about how much free memory actually
 * exists. Checking at order 0 would answer the wrong question too -- order 0
 * passes almost everywhere and says nothing about whether THIS high-order
 * request can be met -- so the check runs at kcompactd's own
 * pgdat->kcompactd_max_order (read here before kcompactd_do_work resets it),
 * the same loop shape as kcompactd_node_suitable() a few lines up in
 * mm/compaction.c. This is deliberately NOT shared with
 * thrash_wd_mem_pressured() (mm/oom_kill.c): that watchdog answers a
 * different, order/zone-agnostic question (is the whole system livelocked)
 * with an empirically-tuned raw global free-vs-high-watermark ratio: mixing
 * in zone_watermark_ok()'s per-zone lowmem_reserve/CMA/highatomic exclusions
 * would silently shift that already-incident-tuned threshold.
 *
 * Returns the DROP budget in pages: 0 when every zone clears the check
 * (nothing to drop), else the size of the block this allocation actually
 * needs (1 << order). Deliberately NOT "block size minus existing
 * watermark headroom": a DROPped page is freed wherever it happened to be
 * cold in whichever candidate block Marie picked, with no guaranteed
 * spatial relation to wherever that headroom already sits, so crediting it
 * against the budget would assume a contiguity DROP cannot promise. The
 * order's own size is the defensible, self-contained amount -- enough to
 * plausibly cover this one allocation, no more.
 */
static unsigned long marie_defrag_drop_need(struct pglist_data *pgdat)
{
	enum zone_type highest_zoneidx = pgdat->kcompactd_highest_zoneidx;
	unsigned int order = pgdat->kcompactd_max_order;
	int zoneid;

	for (zoneid = 0; zoneid <= highest_zoneidx; zoneid++) {
		struct zone *zone = &pgdat->node_zones[zoneid];

		if (!populated_zone(zone))
			continue;
		if (!zone_watermark_ok(zone, order, low_wmark_pages(zone),
					highest_zoneidx, 0))
			return 1UL << order;
	}
	return 0;
}

/*
 * Run one Marie defrag pass for @pgdat. @may_drop is set by the caller from the
 * kcompactd path's urgency: demand=true (MOVE + DROP), proactive=false (MOVE only).
 */
void lru_marie_defrag_pgdat(struct pglist_data *pgdat, bool may_drop)
{
	unsigned long drop_need;

	if (!marie_defrag_hist)
		return;
	/*
	 * Single owner: the age-neutrality scratch (srcmap/restamp) is shared
	 * pre-allocated state, and topn walks the global hist/gen ring. A
	 * concurrent caller -- NUMA multi-node kcompactd, or a future manual
	 * trigger -- skips; defrag is best-effort and overlapping runs would only
	 * duplicate work. Also caps the latent NUMA double-run.
	 */
	if (atomic_cmpxchg(&marie_defrag_busy, 0, 1) != 0)
		return;
	/*
	 * Global scan (desktop/single-node). On NUMA this over-triggers across
	 * nodes; per-node Marie defrag is a refinement matching Marie's global posture.
	 */
	atomic_long_inc(&marie_defrag_fires);
	drop_need = (may_drop && READ_ONCE(marie_defrag_drop_enabled)) ?
		marie_defrag_drop_need(pgdat) : 0;
	marie_defrag_topn(MARIE_DEFRAG_BATCH, drop_need);
	atomic_set(&marie_defrag_busy, 0);
}

static ssize_t marie_defrag_enabled_show(struct kobject *kobj, struct kobj_attribute *attr,
			     char *buf)
{
	return sysfs_emit(buf, "%u\n", READ_ONCE(marie_defrag_enabled));
}

static ssize_t marie_defrag_enabled_store(struct kobject *kobj, struct kobj_attribute *attr,
			      const char *buf, size_t count)
{
	unsigned int v;
	int err = kstrtouint(buf, 10, &v);

	if (err)
		return err;
	WRITE_ONCE(marie_defrag_enabled, !!v);
	return count;
}

static struct kobj_attribute marie_defrag_switch_attr =
	__ATTR(defrag, 0644, marie_defrag_enabled_show, marie_defrag_enabled_store);

static ssize_t marie_defrag_drop_enabled_show(struct kobject *kobj, struct kobj_attribute *attr,
				  char *buf)
{
	return sysfs_emit(buf, "%u\n", READ_ONCE(marie_defrag_drop_enabled));
}

static ssize_t marie_defrag_drop_enabled_store(struct kobject *kobj, struct kobj_attribute *attr,
				   const char *buf, size_t count)
{
	unsigned int v;
	int err = kstrtouint(buf, 10, &v);

	if (err)
		return err;
	WRITE_ONCE(marie_defrag_drop_enabled, !!v);
	return count;
}

static struct kobj_attribute marie_defrag_drop_attr =
	__ATTR(defrag_drop, 0644, marie_defrag_drop_enabled_show, marie_defrag_drop_enabled_store);

int __init marie_defrag_init(struct kobject *parent)
{
	int err = marie_defrag_hist_alloc();

	if (err)
		return err;

	marie_defrag_scratch_alloc();

	err = sysfs_create_file(parent, &marie_defrag_switch_attr.attr);
	if (err)
		pr_warn("failed to create sysfs defrag node: %d\n", err);
	err = sysfs_create_file(parent, &marie_defrag_stats_attr.attr);
	if (err)
		pr_warn("failed to create sysfs defrag_stats node: %d\n", err);
	err = sysfs_create_file(parent, &marie_defrag_drop_attr.attr);
	if (err)
		pr_warn("failed to create sysfs defrag_drop node: %d\n", err);
	return 0;
}
