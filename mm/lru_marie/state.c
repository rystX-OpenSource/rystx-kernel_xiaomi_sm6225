// SPDX-License-Identifier: GPL-2.0
/*
 * Marie per-PFN state array — allocation, init, and global counters.
 *
 * Implements the public storage declared in state.h: the flat
 * marie_state[] array indexed by PFN, the cycling head-gen counter,
 * and the per-(gen, type) install counters that drive aging. All of
 * these are allocated once at subsys_initcall time and never freed
 * for the lifetime of the kernel.
 *
 * Sizing rule: the array covers PFNs [0, max_pfn). max_pfn is bounded
 * by MARIE_MAX_SUPPORTED_PFN (the 32-bit PFN gate latched in
 * marie_init), so worst-case footprint is 4 GiB. Realistic configs
 * are 4-64 MiB. NUMA holes and reserved regions read as zero
 * (untracked) and incur only sequential-read cost during scans.
 */

#define pr_fmt(fmt) "marie_state: " fmt

#include <linux/atomic.h>
#include <linux/bitmap.h>
#include <linux/bitops.h>
#include <linux/init.h>
#include <linux/jiffies.h>
#include <linux/log2.h>
#include <linux/jump_label.h>
#include <linux/spinlock.h>
#include <linux/list.h>
#include <linux/lru_marie.h>
#include <linux/memblock.h>
#include <linux/memcontrol.h>
#include <linux/mm.h>
#include <linux/moduleparam.h>
#include <linux/mm_inline.h>
#include <linux/mmzone.h>
#include <linux/oom.h>
#include <linux/printk.h>
#include <linux/sched/signal.h>
#include <linux/swap.h>
#include <linux/vmalloc.h>
#include <linux/vm_event_item.h>
#include <linux/vmstat.h>

#ifdef CONFIG_X86
#include <asm/cpufeature.h>
#include <asm/processor.h>
#endif

#include "../internal.h"	/* struct scan_control, shrink_page_list */
#include "state_compat.h"	/* MARIE_FOLIO_FLAGS, marie_shrink_page_list, marie_account_reclaim */
#include "account.h"
#include "pfn_install.h"
#include "prefetch.h"
#include "state.h"

/*
 * Runtime prefetch-ring parameters, set once at boot by
 * marie_prefetch_params_init() based on CPUID. All values are
 * powers of 2 so the hot path can use & marie_l3_mask instead of
 * % marie_l3_ahead. Defaults are conservative (Silvermont / non-x86).
 */
static unsigned int marie_l3_ahead __read_mostly = 8;
static unsigned int marie_l3_mask  __read_mostly = 7;
static unsigned int marie_l1_ahead __read_mostly = 2;

void __init marie_prefetch_params_init(void)
{
	unsigned int l3 = 8, l1 = 2;

#ifdef CONFIG_X86
	if (!boot_cpu_has(X86_FEATURE_AVX2))
		goto done;

	if (boot_cpu_has(X86_FEATURE_AVX512F)) {
		/* Zen 4/5, Sapphire Rapids: L2 MSHR ~32 */
		l3 = 32; l1 = 8;
		goto done;
	}

	/* AVX2 present but no AVX-512 */
	switch (boot_cpu_data.x86_vendor) {
	case X86_VENDOR_AMD:
		if (boot_cpu_data.x86 >= 0x1A) {
			/* Zen 5+ mobile without AVX-512 */
			l3 = 32; l1 = 8;
		} else if (boot_cpu_data.x86 == 0x19) {
			/* Zen 3 (family 0x19): L2 MSHR ~24 */
			l3 = 24; l1 = 8;
		} else if (boot_cpu_data.x86 == 0x17) {
			/* Zen 1/2 (family 0x17): L2 MSHR ~20 */
			l3 = 20; l1 = 8;
		} else {
			/* Excavator era (family 0x15): L2 MSHR ~12 */
			l3 = 16; l1 = 6;
		}
		break;
	case X86_VENDOR_INTEL:
		/*
		 * CLFLUSHOPT as a Skylake proxy: Haswell and Broadwell
		 * (all models) predate it; Skylake introduced it.
		 */
		if (boot_cpu_has(X86_FEATURE_CLFLUSHOPT)) {
			/* Skylake and newer: L2 MSHR ~20-32 */
			l3 = 24; l1 = 8;
		} else {
			/* Haswell / Broadwell: L2 MSHR ~16 */
			l3 = 16; l1 = 6;
		}
		break;
	default:
		/* Unknown vendor with AVX2: conservative v3 baseline */
		l3 = 16; l1 = 6;
	}
done:
#endif
	marie_l3_ahead = l3;
	marie_l3_mask  = l3 - 1;
	marie_l1_ahead = l1;
	pr_info("prefetch ring: l3_ahead=%u l1_ahead=%u\n", l3, l1);
}

u8 *marie_state;
unsigned long marie_state_size;

/*
 * Latches true once marie_state[] is allocated (first enable) and never
 * flips back -- the array lives for the kernel's lifetime. Gates the
 * page-free hook so stale TRACKED bits are wiped at the buddy handoff
 * even across a Marie disable transition (when lru_marie_enabled() is
 * already false but the drain walk is still in flight). See
 * marie_state_ready() in <linux/lru_marie.h>.
 */
DEFINE_STATIC_KEY_FALSE(marie_state_ready_key);
EXPORT_SYMBOL_GPL(marie_state_ready_key);

/*
 * Per-(gen, type) live page population, maintained node-wide by the
 * marie_gen_occ_inc/dec abstraction. Desktop/global-only: a single global
 * per-(gen, type) counter drives both the oldest-gen scan target
 * (marie_find_oldest_occupied_mlv) and the "is any anon tracked?" signal read
 * by marie_file_floor_protect.
 */
atomic_long_t marie_gen_occupied[MARIE_PFN_NR_GENS][2];

/*
 * Global aging epoch, bumped once per walker pass per type. The global aging
 * clock stamps marie_recycle_epoch[gen] = this value when it recycles @gen at
 * a head advance, and reclaim sets ignore_references for the oldest gen once
 * the epoch has moved past that stamp (the walker has since swept the gen).
 */
atomic_t marie_aging_epoch[ANON_AND_FILE];

/*
 * GLOBAL aging clock. Desktop/global-only Marie has no per-memcg reclaim, so a
 * single ring serves the whole node (replacing the retired per-mlv head/clock):
 *   marie_head_gen      youngest/install gen (atomic; cmpxchg on advance)
 *   marie_recycle_epoch marie_aging_epoch captured when the head recycled @gen
 *                       (ignore_references gate; WRITE_ONCE by the advance winner)
 *   marie_gen_installs  install-cadence counter (pages onto the head gen)
 * All shared across lruvecs (installs run under different per-lruvec lru_locks),
 * hence atomic; the u32s are racy heuristics (READ_ONCE/WRITE_ONCE is enough).
 */
atomic_t marie_head_gen[ANON_AND_FILE];
u32 marie_recycle_epoch[MARIE_PFN_NR_GENS][ANON_AND_FILE];
atomic_long_t marie_gen_installs[ANON_AND_FILE];


struct marie_bitmap marie_track_bm[2][MARIE_PFN_NR_GENS][MARIE_PFN_NR_TIERS];
unsigned int marie_l2_shift;

/*
 * Per-CPU shrink scratch buffer, pre-allocated at boot. Reclaim path
 * cannot kmalloc / kvmalloc on the hot path (allocation under memory
 * pressure is what we are trying to relieve), so the isolate batch
 * lives in a fixed per-CPU buffer claimed via an atomic in_use flag.
 * On contention (preempted reclaimer on the same CPU holds the buf
 * across a shrink_page_list sleep) marie_state_shrink_lruvec falls
 * back to a 160-entry stack array.
 *
 * Sizing: 8192 entries = SWAP_CLUSTER_MAX << 8. Doubled from the
 * MGLRU MAX_LRU_BATCH (4096) reference after boot testing showed
 * 4096-cap reclaim falling behind tail /dev/zero alloc rate. 32 MiB
 * per shrink_page_list flush at peak amortises lock + IPI overhead
 * twice as well. Per-CPU memory cost:
 *   batch:       8192 * 8 B = 64 KiB
 *   atomic:                = ~4 B
 *   ~= 64 KiB / CPU. 16 CPUs = ~1 MiB system-wide static.
 *
 * Neither PFN nor prev_tier needs its own array at putback: PFN is
 * recovered via page_to_pfn(batch[i]), and prev_tier is read back from the
 * per-PFN state byte (counters_only preserves it across isolate).
 */
#define MARIE_PFN_SHRINK_BATCH	(SWAP_CLUSTER_MAX << 8)	/* 8192 */
#define MARIE_PFN_BATCH_FLOOR	(SWAP_CLUSTER_MAX * 8)	/* 256, matches
							 * legacy
							 * MARIE_BATCH_FLOOR */
/*
 * Fallback batch size when the per-CPU buf is contended. 5 *
 * SWAP_CLUSTER_MAX = 160 entries occupy 160 * 8 = 1280 B on the
 * stack; combined with the surrounding ~464 B of non-array locals
 * in shrink_lruvec the frame lands at ~1744 B, staying under the
 * gcc -Wframe-larger-than=2048 threshold without restructuring.
 * 5x SWAP_CLUSTER_MAX.
 */
#define MARIE_PFN_FALLBACK_BATCH (SWAP_CLUSTER_MAX * 5)	/* 160 */

struct marie_shrink_buf {
	atomic_t in_use;
	struct page *batch[MARIE_PFN_SHRINK_BATCH];
};
static DEFINE_PER_CPU(struct marie_shrink_buf, marie_shrink_buf);

/*
 * Per-PFN adaptive batch threshold, in PAGES (not pages).
 *
 *   priority = DEF_PRIORITY -> floor (MARIE_PFN_BATCH_FLOOR = 256)
 *   priority = 0            -> cap   (MARIE_PFN_SHRINK_BATCH = 8192)
 *
 * This bounds how many pages of isolated-but-not-yet-reclaimed
 * exposure one tier-loop pass may accumulate before calling
 * shrink_page_list. It must be a PAGE budget, not a page-count
 * budget: an anon THP is a single page worth up to HPAGE_PMD_NR
 * (512 on x86-64) pages, so a page-count cap lets a THP-heavy scan
 * isolate two-plus orders of magnitude more memory than the cap
 * implies. The array capacity (MARIE_PFN_SHRINK_BATCH /
 * MARIE_PFN_FALLBACK_BATCH slots) remains a separate, harder
 * page-count ceiling enforced at the call site -- this value only
 * gates on n_taken_pages.
 */
static unsigned long marie_pfn_batch_threshold(struct scan_control *sc)
{
	unsigned long floor = MARIE_PFN_BATCH_FLOOR;
	unsigned long cap = MARIE_PFN_SHRINK_BATCH;
	unsigned long pressure;

	pressure = DEF_PRIORITY + 1 -
		   clamp(sc_priority(sc), 0, DEF_PRIORITY);
	return floor + (cap - floor) * (pressure - 1) / DEF_PRIORITY;
}

/*
 * Equivalent of mm/vmscan.c's too_many_isolated() for Marie's own
 * isolate path, which (unlike shrink_inactive_list) never consulted
 * it. Concurrent direct reclaimers isolating from the same node can
 * otherwise pile up unboundedly: each isolated batch is in flight
 * (off the LRU, not yet freed, not counted as free memory) for the
 * duration of shrink_page_list, which can itself sleep -- e.g. the
 * swap-table cluster allocator's blocking GFP_KERNEL fallback
 * (swap_cluster_alloc_table() in mm/swapfile.c). Without this check
 * that pile-up is unbounded and can drive a zone below its min
 * watermark purely from in-flight isolation, starving even small
 * in-reclaim allocations.
 *
 * kswapd is exempt, matching too_many_isolated(): it is the sole
 * global reclaimer, so throttling it on isolated pages contributed
 * by OTHER (possibly stalled) reclaimers risks stalling the one
 * task relied on to make forward progress.
 */
static bool marie_too_many_isolated(struct pglist_data *pgdat, int type,
				     struct scan_control *sc)
{
	unsigned long inactive, isolated;

	if (current_is_kswapd())
		return false;

	inactive = node_page_state(pgdat, NR_INACTIVE_ANON + type);
	isolated = node_page_state(pgdat, NR_ISOLATED_ANON + type);

	/*
	 * GFP_NOIO/GFP_NOFS callers cannot recurse into the IO/FS paths
	 * that would let them get unstuck, so give them a lower bar to
	 * clear (matches too_many_isolated()'s rationale: avoids a
	 * circular wait against normal, IO-capable reclaimers who ARE
	 * throttled here).
	 */
	if (gfp_has_io_fs(sc_gfp_mask(sc)))
		inactive >>= 3;

	return isolated > inactive;
}

/*
 * Allocate the per-PFN state array. Called from marie_init() after
 * the 32-bit PFN gate is latched, so max_pfn is guaranteed to fit
 * in the supported range.
 *
 * kvmalloc lets the array fall back to vmalloc on systems where a
 * physically contiguous allocation is unavailable; the array is
 * accessed strictly by PFN index and does not require contiguity.
 * GFP_KERNEL is safe here — initcall context can sleep.
 */
int __init marie_state_init(void)
{
	unsigned long bytes;
	int g, t, ty;

	bytes = max_pfn * sizeof(u8);
	if (!bytes) {
		pr_err("max_pfn is zero; refusing to initialise\n");
		return -EINVAL;
	}

	marie_state = kvmalloc(bytes, GFP_KERNEL | __GFP_ZERO);
	if (!marie_state) {
		pr_err("failed to allocate %lu-byte per-PFN state array\n",
		       bytes);
		return -ENOMEM;
	}
	marie_state_size = max_pfn;

	/*
	 * L2 bitmap shift: (1 << shift) PFNs map to one L2 bit so 512
	 * L2 bits cover the full max_pfn range. Round up to the next
	 * power of two so the index is a simple right shift in the hot
	 * path. Floor at shift 0 for tiny VMs where max_pfn < 512.
	 * Must be set before any marie_bm_* call so marie_pfn_to_l2_bit
	 * works correctly.
	 */
	{
		unsigned long ppb = max_pfn / MARIE_L2_BITS;

		if (ppb < 1)
			ppb = 1;
		marie_l2_shift = order_base_2(ppb);
	}

	marie_bm_range_locks_init();

	/* Per-(type, gen, tier) L1 bitmaps: 16 total. */
	for (ty = 0; ty < 2; ty++) {
		for (g = 0; g < MARIE_PFN_NR_GENS; g++) {
			for (t = 0; t < MARIE_PFN_NR_TIERS; t++) {
				if (marie_bm_init(&marie_track_bm[ty][g][t]))
					goto bm_oom;
			}
		}
	}

	/*
	 * Latch the page-free hook on now that marie_state[] exists. Never
	 * disabled -- the array is never freed, and TRACKED bits can persist
	 * into a disable transition, so the hook must keep wiping them.
	 */
	static_branch_enable(&marie_state_ready_key);

	/*
	 * Seed the dynamic install-cadence thresholds before the first install
	 * so the cadence never compares against a 0 live value (which would
	 * advance the head on every install). At init occ==0 and adv==0, so
	 * live = memtotal/8 (anon warm-up decay) / max(reserve/NR_GENS, memtotal/256)
	 * (file).
	 */
	marie_recompute_growth_threshold(0);
	marie_recompute_growth_threshold(1);

	pr_info("allocated state %lu B + 16 tracking bitmaps (max_pfn=%lu, l2_shift=%u)\n",
		bytes, max_pfn, marie_l2_shift);
	return 0;

bm_oom:
	for (ty = 0; ty < 2; ty++)
		for (g = 0; g < MARIE_PFN_NR_GENS; g++)
			for (t = 0; t < MARIE_PFN_NR_TIERS; t++)
				marie_bm_free(&marie_track_bm[ty][g][t]);
	kvfree(marie_state);
	marie_state = NULL;
	return -ENOMEM;
}

/*
 * Global generation frame primitives (desktop/global-only). A single aging
 * clock per type (marie_head_gen[type]) serves the whole node: the per-PFN
 * byte stores a gen VALUE, and its age is that value read against the global
 * head_gen. The _mlv suffix is historical (the per-lruvec frame is gone).
 */

/*
 * Oldest gen (walking out from the global head) where the node has live pages.
 */
int marie_find_oldest_occupied_mlv(int type)
{
	int head = atomic_read(&marie_head_gen[type]);
	int i;

	for (i = 1; i < MARIE_PFN_NR_GENS; i++) {
		int slot = (head + i) & (MARIE_PFN_NR_GENS - 1);

		if (atomic_long_read(&marie_gen_occupied[slot][type]) > 0)
			return slot;
	}
	return -1;
}

/*
 * Dynamic per-type install-cadence threshold. marie_gen_growth_live[type] is the
 * LIVE value the install cadence compares marie_gen_installs[type] against.
 * Fully automatic -- there is no sysfs knob:
 *
 *   live[anon] = max(occ/NR_GENS, (memtotal - occ)/NR_GENS, memtotal/256)
 *   live[file] = max(occ/NR_GENS,  reserve/NR_GENS,         memtotal/256)
 *
 *   occ/NR_GENS = scale-invariant steady state: the ring spans the type's
 *              working set, so the oldest gen holds ~1/NR_GENS of it and a page
 *              ages over ~one set-turnover of installs. PAGES (node LRU stat),
 *              not pages, to match the page-counted install cadence so THP does
 *              not skew it.
 *   (memtotal - occ)/NR_GENS = ANON ONLY, stateless coverage/slack warm-up.
 *              max()'d with occ/NR_GENS the effective anon gen size becomes
 *              max(anon, memtotal - anon)/NR_GENS, so the ring ALWAYS spans the
 *              full anon set (never under-strata) and gains head-ahead slack
 *              while anon is below half of RAM -- coarse gens defer stratifying a
 *              still-filling set. It tightens to a full NR_GENS-deep ring as anon
 *              approaches memtotal and hands off to occ/NR_GENS past half of RAM,
 *              exactly the under-pressure regime where fine cold-page selection
 *              matters (under swappiness=1 anon is tapped only when it already
 *              fills the non-file remainder, so "anon large" ~= "anon reclaimed").
 *              No latch, no advance counter: pressure-reactive and self-recovering.
 *              File has no such term -- pagecache churns continuously (no
 *              fill-then-settle phase), so immediate occ/NR_GENS stratification is
 *              correct.
 *   reserve/NR_GENS = FILE ONLY: the clean_min_ratio reserve / NR_GENS. When file
 *              is squeezed to its protected reserve the ring still stratifies that
 *              reserve across all gens; occ/NR_GENS takes over once file grows
 *              past the reserve. Anon has no clean_min_ratio reserve (it is a file
 *              concept), so no such term.
 *   memtotal/256 = absolute thrash floor for the occ~0 / high-churn corner (and
 *              for file when clean_min_ratio == 0 zeroes its reserve term).
 *
 * Recomputed only at a head advance (rare) and on a clean_min_ratio write -- the
 * per-install hot path is one READ_ONCE(marie_gen_growth_live[type]). Both live
 * terms track current occupancy, so an advance-time recompute keeps them fresh.
 */
unsigned long marie_gen_growth_live[2];

void marie_recompute_growth_threshold(int type)
{
	unsigned long total = totalram_pages();
	unsigned long occ_pages, occ, thr;

	/* Absolute thrash floor; also covers clean_min_ratio==0 for file. */
	thr = total >> 8;			/* memtotal/256 */

	/* Scale-invariant steady state: occupancy pages / NR_GENS. */
	if (type == 0)
		occ_pages = global_node_page_state(NR_ACTIVE_ANON) +
			    global_node_page_state(NR_INACTIVE_ANON);
	else
		occ_pages = global_node_page_state(NR_ACTIVE_FILE) +
			    global_node_page_state(NR_INACTIVE_FILE);
	occ = occ_pages / MARIE_PFN_NR_GENS;
	if (occ > thr)
		thr = occ;

	if (type == 1) {
		/* FILE policy floor: clean_min_ratio reserve / NR_GENS. */
		unsigned long resv = total * READ_ONCE(marie_clean_min_ratio)
				     / 100 / MARIE_PFN_NR_GENS;
		if (resv > thr)
			thr = resv;
	} else {
		/*
		 * ANON coverage/slack warm-up (stateless): (memtotal - anon)/NR_GENS.
		 * max()'d with occ/NR_GENS -> effective gen = max(anon, memtotal-anon)
		 * /NR_GENS: ring always spans the anon set, slack while anon < half RAM,
		 * full NR_GENS-deep once anon fills. See the header block.
		 */
		unsigned long warm = (total > occ_pages)
				     ? (total - occ_pages) / MARIE_PFN_NR_GENS : 0;
		if (warm > thr)
			thr = warm;
	}

	WRITE_ONCE(marie_gen_growth_live[type], thr);
}

/*
 * Advance the global head if its next slot is empty. The gate
 * (gen_occupied[next]==0) guarantees nothing live sits at the slot the head
 * recycles, so it never aliases old pages. Stamp recycle_epoch so
 * ignore_references waits for a fresh walker sweep before force-reclaiming the
 * recycled slot once it ages back to oldest.
 *
 * Returns true iff it advanced (the install-cadence caller resets its install
 * counter only on a real advance, so a blocked attempt -- next slot still
 * draining -- retries on the following install).
 *
 * Driven solely by install cadence under the installing lruvec's lru_lock (see
 * marie_page_install). The former reclaim-time "occupied < 2" trigger was
 * removed: under concurrent global reclaim it fired on every shrink entry and
 * raced the head around the ring (~10^6 advances/run vs ~10^1 aging ticks),
 * destroying age stratification so the oldest gen held mixed-age (incl. hot)
 * pages -> rotation -> ~50% reclaim efficiency and OOM with swap free.
 */
static bool marie_try_advance_head_mlv(int type)
{
	u8 head = (u8)atomic_read(&marie_head_gen[type]);
	u8 next = (head + 1) & (MARIE_PFN_NR_GENS - 1);

	if (atomic_long_read(&marie_gen_occupied[next][type]) != 0)
		return false;
	if (atomic_cmpxchg(&marie_head_gen[type], head, next) != head)
		return false;
	WRITE_ONCE(marie_recycle_epoch[next][type],
		   atomic_read(&marie_aging_epoch[type]));
	/*
	 * Real advance: recompute this type's live install-cadence threshold from
	 * current occupancy (both the occ/NR_GENS and the anon warm-up terms track
	 * live occ_pages). Runs for install-cadence AND demand-pull advances.
	 */
	marie_recompute_growth_threshold(type);
	return true;
}

/*
 * marie_gen_occ_inc/dec -- the SINGLE gen-occupancy interface -- now live as
 * static inline in state.h so pfn_install.h's install-side publisher
 * (marie_pfn_publish_inherit) routes through the same hook instead of bumping
 * marie_gen_occupied directly. See the contract above their definition there.
 */

/*
 * marie_state_isolate_scan_l2lock - L2-bitmap pre-filtered scan with
 * 512-way parallel exclusion via try_lock on per-L2-bit locks.
 *
 * Walks the L2 bitmap (1 cacheline) for the oldest (gen, type). For
 * each set L2 bit it try_locks the matching L2 lock; on success it
 * holds exclusive ownership of that PFN range and walks the L1
 * bitmap within it, applying the same (mask, target) byte filter as
 * the cursor scan. On try_lock failure another scanner already owns
 * the range -- skip and try the next L2 bit. No wasted candidate
 * scan work, no per-CPU cursor, no overlap-arbitration via
 * TestClearPageLRU collisions.
 *
 * Loop exits when batch_size is reached, nr_to_scan is exhausted,
 * or every L2 bit in the pgdat's PFN range has been visited (locked
 * or skipped).
 */
unsigned long marie_state_isolate_scan_l2lock(struct pglist_data *pgdat,
					      int type, int max_zone,
					      unsigned int tier,
					      struct page **batch,
					      unsigned long batch_size,
					      unsigned long nr_to_scan,
					      int oldest_in)
{
	unsigned long *l1, *l2;
	u8 oldest_gen, mask, target;
	unsigned long start_pfn, end_pfn;
	unsigned int start_l2, end_l2;
	unsigned int l2_word, l2_word_end;
	unsigned long n_batch = 0;

	if (!marie_state)
		return 0;

	/* Gen to scan is the global oldest (already validated >= 0). */
	if (oldest_in < 0)
		return 0;
	oldest_gen = (u8)oldest_in;
	{
		struct marie_bitmap *bm =
			&marie_track_bm[type][oldest_gen][tier & 0x3];

		l1 = bm->l1;
		l2 = bm->l2;
	}
	if (!l1)
		return 0;

	mask = MARIE_PFN_TRACKED | MARIE_PFN_GEN_MASK |
	       MARIE_PFN_TIER_MASK | MARIE_PFN_TYPE_MASK;
	target = MARIE_PFN_TRACKED |
		 (oldest_gen << MARIE_PFN_GEN_SHIFT) |
		 ((tier & MARIE_PFN_TIER_MAX) << MARIE_PFN_TIER_SHIFT) |
		 (type ? MARIE_PFN_TYPE_FILE : 0);

	start_pfn = pgdat->node_start_pfn;
	end_pfn   = pgdat_end_pfn(pgdat);
	if (end_pfn > marie_state_size)
		end_pfn = marie_state_size;
	if (start_pfn >= end_pfn)
		return 0;

	start_l2 = marie_pfn_to_l2_bit(start_pfn);
	end_l2 = marie_pfn_to_l2_bit(end_pfn - 1) + 1;
	if (end_l2 > MARIE_L2_BITS)
		end_l2 = MARIE_L2_BITS;
	l2_word = start_l2 / BITS_PER_LONG;
	l2_word_end = DIV_ROUND_UP(end_l2, BITS_PER_LONG);

	/*
	 * Outer L2 loop is word-level: the inner __ffs/blsr extraction
	 * visits only set L2 bits of the global (type, gen, tier) plane.
	 * 512 L2 bits collapse to 8 u64 word iterations; empty words skip
	 * at one cycle each.
	 */
	for (; l2_word < l2_word_end; l2_word++) {
		unsigned long l2w = l2[l2_word];

		/* Mask off pre-start_l2 / post-end_l2 bits in edge words. */
		if (l2_word == start_l2 / BITS_PER_LONG &&
		    (start_l2 % BITS_PER_LONG))
			l2w &= ~((1UL << (start_l2 % BITS_PER_LONG)) - 1);
		if (l2_word + 1 == l2_word_end &&
		    (end_l2 % BITS_PER_LONG))
			l2w &= (1UL << (end_l2 % BITS_PER_LONG)) - 1;

	while (l2w && n_batch < batch_size && nr_to_scan > 0) {
		unsigned int bit = l2_word * BITS_PER_LONG + __ffs(l2w);
		unsigned long lo, hi;
		unsigned long ring[MARIE_L3_AHEAD_MAX];
		int rh = 0, rt = 0, rc = 0;
		unsigned long word_rem;
		unsigned long word_base;
		unsigned long word_i, end_word;
		bool producer_done = false;
		int i, n;
		/*
		 * Local copies of the runtime ring parameters. Declaring them
		 * here as loop-scope constants lets the compiler see them as
		 * truly invariant within this L2 lock window and allocate
		 * registers for them, rather than spilling the file-static
		 * globals to the stack under register pressure.
		 */
		const unsigned int r_l3_ahead = marie_l3_ahead;
		const unsigned int r_l3_mask  = marie_l3_mask;
		const unsigned int r_l1_ahead = marie_l1_ahead;
		/*
		 * Per-L2-range cache-line cursors for marie_state[] prefetch.
		 * PFNs within one L2 range are monotonically increasing, so
		 * the cursor only advances; resetting per L2 range avoids
		 * stale comparisons when the next range starts at a lower
		 * cache line than the previous one ended at.
		 */
		unsigned long state_cl_cursor_l3 = 0;
		unsigned long state_cl_cursor_l1 = 0;
		/*
		 * Per-L2-range cache-line cursor for the L1 bitmap. word_i is
		 * monotonically increasing within the range so it only advances.
		 */
		unsigned long l1_cl_cursor = 0;

		l2w &= l2w - 1;

		if (!marie_bm_range_trylock(bit))
			continue;

		lo = marie_l2_bit_pfn_start(bit);
		hi = marie_l2_bit_pfn_end(bit);
		if (lo < start_pfn)
			lo = start_pfn;
		if (hi > end_pfn)
			hi = end_pfn;

		/*
		 * Inline bit producer state with optional word-level mbm
		 * AND: word_rem is the live remainder of l1[word_i] with
		 * mbm[word_i] AND-ed in (when memcg-targeted). Persists
		 * across Phase 1 fill and Phase 3 refill so we never
		 * re-scan a cleared word and never pay find_next_bit's
		 * call overhead. The AND narrows iteration to
		 * (type, gen, tier) ∩ memcg at source -- per-candidate
		 * mbm post-filter falls away.
		 */
		word_i = lo / BITS_PER_LONG;
		end_word = BITS_TO_LONGS(hi);
		word_base = word_i * BITS_PER_LONG;
		word_rem = (word_i < end_word) ? l1[word_i] : 0;
		/* Mask off pre-lo bits in the first word. */
		if (lo > word_base)
			word_rem &= ~((1UL << (lo - word_base)) - 1);
		word_i++;

		/*
		 * Two-stage prefetch ring within this L2 lock window:
		 *
		 *   Phase 1: fill the ring (up to marie_l3_ahead candidate
		 *     PFNs via inline __ffs/blsr), firing prefetcht2 on
		 *     each struct page + state byte -- DRAM fetch in
		 *     flight by the time the iterator pulls the entry.
		 *
		 *   Phase 2: L1-escalate the first marie_l1_ahead entries
		 *     with prefetcht0 so they land in L1 before processing.
		 *
		 *   Phase 3: drain. Per pulled entry, refill the head (one
		 *     more L3 prefetch) and L1-escalate the entry now
		 *     marie_l1_ahead ahead of the new tail. State byte
		 *     confirm and pfn_to_page() both hit cache.
		 *
		 * Ring is local to this L2 lock acquisition; struct page +
		 * state byte are vmemmap/contiguous so prefetches incur no
		 * locking cost.
		 */
	/*
	 * Cache-line cursor prefetch for bitmap arrays. Issued at each word
	 * refill; the cursor only advances so a dense word transition does
	 * not re-prefetch the same cache line.
	 */
#define MARIE_PREFETCH_BMWORD_L3(arr, cursor) do {				\
		unsigned long _bi = word_i + MARIE_BM_L3_AHEAD_WORDS;		\
		if (_bi < end_word) {						\
			unsigned long _cl = (unsigned long)&(arr)[_bi]		\
					    & ~63UL;				\
			if (_cl != (cursor)) {					\
				marie_prefetch_l3((void *)_cl);			\
				(cursor) = _cl;					\
			}							\
		}								\
	} while (0)

#define MARIE_RING_PRODUCE(out_pfn, done_label) do {			\
		while (!word_rem) {					\
			if (word_i >= end_word) {			\
				producer_done = true;			\
				goto done_label;			\
			}						\
			word_rem = l1[word_i];				\
			MARIE_PREFETCH_BMWORD_L3(l1, l1_cl_cursor);	\
			word_base = word_i * BITS_PER_LONG;		\
			word_i++;					\
		}							\
		(out_pfn) = word_base + __ffs(word_rem);		\
		word_rem &= word_rem - 1;				\
		if ((out_pfn) >= hi) {					\
			producer_done = true;				\
			goto done_label;				\
		}							\
	} while (0)

	/*
	 * Cache-line cursor prefetch for marie_state[]. AHEAD_PFN pushes the
	 * prefetched cache line N PFN ahead of the current producer position
	 * so DRAM (L3-tier) and L3->L1 latencies are hidden even when the
	 * consumer's fast-skip iter (mask filter early-continue) burns only
	 * a few cycles per PFN. struct page is per-PFN = per-cache-line
	 * already, so its prefetches stay per-PFN unchanged.
	 */
#define MARIE_PREFETCH_STATE_L3(pfn) do {					\
		unsigned long _ah = (pfn) + MARIE_STATE_L3_AHEAD_PFN;		\
		if (_ah < marie_state_size) {					\
			unsigned long _cl = (unsigned long)&marie_state[_ah]	\
					    & ~63UL;				\
			if (_cl != state_cl_cursor_l3) {			\
				marie_prefetch_l3((void *)_cl);			\
				state_cl_cursor_l3 = _cl;			\
			}							\
		}								\
	} while (0)
#define MARIE_PREFETCH_STATE_L1(pfn) do {					\
		unsigned long _ah = (pfn) + MARIE_STATE_L1_AHEAD_PFN;		\
		if (_ah < marie_state_size) {					\
			unsigned long _cl = (unsigned long)&marie_state[_ah]	\
					    & ~63UL;				\
			if (_cl != state_cl_cursor_l1) {			\
				marie_prefetch_l1((void *)_cl);			\
				state_cl_cursor_l1 = _cl;			\
			}							\
		}								\
	} while (0)

		while (rc < r_l3_ahead) {
			unsigned long p;

			MARIE_RING_PRODUCE(p, phase1_done);
			ring[rh] = p;
			rh = (rh + 1) & r_l3_mask;
			rc++;
			MARIE_PREFETCH_STATE_L3(p);
			marie_prefetch_l3(pfn_to_page(p));
		}
phase1_done:

		n = rc < r_l1_ahead ? rc : r_l1_ahead;
		for (i = 0; i < n; i++) {
			unsigned long p = ring[(rt + i) & r_l3_mask];

			MARIE_PREFETCH_STATE_L1(p);
			marie_prefetch_l1(pfn_to_page(p));
		}

		while (rc > 0 && n_batch < batch_size && nr_to_scan > 0) {
			unsigned long pfn = ring[rt];
			u8 s;
			unsigned int z;
			struct page *f;

			rt = (rt + 1) & r_l3_mask;
			rc--;
			nr_to_scan--;

			if (!producer_done) {
				unsigned long np;

				MARIE_RING_PRODUCE(np, refill_done);
				ring[rh] = np;
				rh = (rh + 1) & r_l3_mask;
				rc++;
				MARIE_PREFETCH_STATE_L3(np);
				marie_prefetch_l3(pfn_to_page(np));
			}
refill_done:

			if (rc > r_l1_ahead) {
				int idx = (rt + r_l1_ahead - 1) &
					  r_l3_mask;
				unsigned long lp = ring[idx];

				MARIE_PREFETCH_STATE_L1(lp);
				marie_prefetch_l1(pfn_to_page(lp));
			}

			s = READ_ONCE(marie_state[pfn]);
			if ((s & mask) != target) {
				/*
				 * Orphaned L1 bit: set at (type, oldest_gen, tier)
				 * but this PFN's current state byte no longer
				 * encodes that coordinate -- it moved (inc_tier /
				 * move_to_gen / publish_at_gen) after this bit was
				 * set, or the page was dropped and the PFN reused.
				 * The mutators are lock-free best-effort (the
				 * bitmap move is not atomic with the byte CAS that
				 * commits it -- see marie_state_move_to_gen /
				 * __marie_state_inc_tier), so a bit can be left
				 * behind pointing at a coordinate the byte has
				 * since moved past.
				 *
				 * Previously this was a silent `continue`: the bit
				 * stayed set and gen_occupied stayed incremented
				 * for it forever, since nothing else ever revisits
				 * a bit once its gen ages past reclaim. Under a
				 * hot, fast, single-type burst (walker inc_tier and
				 * reclaim isolate racing the same PFNs at maximum
				 * rate -- exactly a `tail /dev/zero`-style anon
				 * flood) enough of these accumulate in one gen that
				 * marie_find_oldest_occupied_mlv() keeps returning
				 * a gen that is now 100% orphaned bits: isolate
				 * finds candidates, every one fails this check, and
				 * the sweep returns 0 reclaimed forever -- reclaim
				 * for that type wedges permanently despite
				 * gen_occupied reading > 0 (confirmed live: pick
				 * anon_strict climbed by millions during a burst
				 * while reclaimed-anon and pswpout stayed exactly
				 * flat).
				 *
				 * We hold this bit's L2 range trylock exclusively
				 * (marie_bm_range_trylock above), so no other
				 * scanner can be racing us to clear this same bit.
				 * marie_bm_clear only returns true on a genuine
				 * 1->0 transition, so gating the gen_occ_dec on it
				 * cannot double-decrement a slot some other path
				 * already retired -- same discipline as every other
				 * gen_occupied mutation site. A concurrent mutator
				 * winning a fresh, legitimate transition into this
				 * exact coordinate between our read of @s and this
				 * clear is the narrow residual case; it strands
				 * that one page invisibly (bounded, self-limiting,
				 * no counter underflow) rather than the unbounded
				 * whole-gen wedge this closes.
				 */
				if (marie_bm_clear(&marie_track_bm[type][oldest_gen]
						    [tier & 0x3], pfn)) {
					marie_gen_occ_dec(pfn, oldest_gen, type);
					atomic_long_inc(&marie_dbg_orphan_bit[type]);
				}
				continue;
			}
			z = (s & MARIE_PFN_ZONE_MASK)
				>> MARIE_PFN_ZONE_SHIFT;
			if ((int)z > max_zone)
				continue;

			f = pfn_to_page(pfn);
			batch[n_batch++] = f;
		}

		marie_bm_range_unlock(bit);
	}	/* while (l2w) -- next set bit in this L2 word */
	}	/* for (l2_word) -- next L2 word */
#undef MARIE_RING_PRODUCE
#undef MARIE_PREFETCH_BMWORD_L3
#undef MARIE_PREFETCH_STATE_L3
#undef MARIE_PREFETCH_STATE_L1

	return n_batch;
}

/*
 * marie_state_drop_pfn - zero out every per-PFN tracking artifact
 * for one page (state byte, (type, gen, tier) L1 bit, and the
 * global per-(gen, type) occupancy counter).
 *
 * Called from:
 *   marie_evict_locked      -- normal evict path
 *   marie_drain_pfn_locked  -- enable=0 sysfs flip; page gets
 *                              returned to legacy LRU, the per-PFN
 *                              artifacts MUST be wiped or they
 *                              survive across the disabled window
 *                              as ghosts that wedge counters on
 *                              re-enable.
 *
 * No-op when the state byte is not TRACKED (defensive against
 * double-drop). Reads the (gen, tier, type) tuple from the byte
 * BEFORE zeroing it so the per-(type, gen, tier) bitmap and
 * occupancy counter are decremented at the same coordinate the
 * install incremented.
 */
void marie_state_drop_pfn(struct page *page)
{
	unsigned long pfn;
	u8 s, g, tier, type_bit;

	if (!marie_state || !page)
		return;

	pfn = page_to_pfn(page);
	if (pfn >= marie_state_size)
		return;

	s = marie_state[pfn];
	marie_state[pfn] = 0;
	if (!(s & MARIE_PFN_TRACKED))
		return;

	g = (s & MARIE_PFN_GEN_MASK) >> MARIE_PFN_GEN_SHIFT;
	tier = (s & MARIE_PFN_TIER_MASK) >> MARIE_PFN_TIER_SHIFT;
	type_bit = (s & MARIE_PFN_TYPE_MASK) ? 1 : 0;

	if (marie_bm_clear(&marie_track_bm[type_bit][g][tier], pfn))
		marie_gen_occ_dec(pfn, g, type_bit);
}
EXPORT_SYMBOL_GPL(marie_state_drop_pfn);

/*
 * marie_state_drop_pfn_at_free - canonical buddy-handoff cleanup.
 *
 * Invoked from mm/page_alloc.c::free_pages_prepare for every page about
 * to enter the buddy allocator. Eliminates the deferred-cleanup race
 * between marie_evict_counters_only (counters -1, TRACKED preserved) and
 * the next allocation at the same PFN: the moment the page is destined
 * for buddy, we wipe Marie's per-PFN bookkeeping so a subsequent
 * install_local starts from a clean state byte.
 *
 * Counters are NOT touched here -- they were either already balanced
 * by marie_evict_locked (the normal Marie del path) or pre-decremented
 * by marie_evict_counters_only (the reclaim isolate path), and the
 * page-free hook runs once per page regardless of which del path was
 * taken upstream.
 *
 * memcg_bitmap is intentionally untouched. page_memcg is unsafe to
 * dereference at free time (the page is mid-uncharge); the stale bit
 * is harmless because the next install at this PFN under a different
 * memcg will re-set the new memcg's bitmap bit, and a memcg teardown
 * will free the bitmap wholesale.
 *
 * Lock-free: byte write, bitmap atomic-bit-clear, atomic_long_dec --
 * safe from any context including IRQ.
 */
void marie_state_drop_pfn_at_free(unsigned long pfn)
{
	u8 s, g, tier, type_bit;

	if (!marie_state || pfn >= marie_state_size)
		return;

	s = marie_state[pfn];
	if (!(s & MARIE_PFN_TRACKED))
		return;

	/*
	 * A TRACKED page reaching the buddy free path still carrying PG_lru
	 * bypassed Marie's evict (which clears both TRACKED and PG_lru under
	 * the TestClearPageLRU claim). Leaving PG_lru set trips the
	 * "Bad page state |lru|" PAGE_FLAGS_CHECK_AT_FREE oops. Clear it
	 * here as the canonical last-resort: the page is being freed
	 * (refcount 0) and Marie pages keep page->lru as a self-loop
	 * (never linked onto a real lruvec list), so dropping PG_lru cannot
	 * corrupt any list. This is a mitigation for a residual reclaim
	 * accounting race (a Marie page reaching free with TRACKED still
	 * set); the per-page vmstat that install +nr'd is not undone here,
	 * a minor drift accepted in exchange for not oopsing.
	 */
	{
		struct page *f = pfn_to_page(pfn);

		/*
		 * Invariant: a TRACKED page must never reach the buddy free
		 * path still carrying PG_lru. Marie's evict clears both under
		 * the TestClearPageLRU claim, and folio_batch_move_lru no
		 * longer re-stamps PG_lru onto a tracked page (the mm/swap.c
		 * fix). VM_WARN_ON_ONCE flags a regression of that invariant in
		 * DEBUG_VM builds; it compiles to nothing in production, so the
		 * PageLRU below costs only a predicted-not-taken branch
		 * on an already-hot page->flags. The trailing clear is the
		 * production last resort -- it degrades any future regression
		 * to a counter blip instead of a PAGE_FLAGS_CHECK_AT_FREE oops.
		 * Marie pages keep page->lru detached from real lruvec lists,
		 * so clearing PG_lru here cannot corrupt a list.
		 */
		if (unlikely(PageLRU(f))) {
			VM_WARN_ON_ONCE_PAGE(1, f);
			ClearPageLRU(f);
		}
		/*
		 * shrink_page_list can re-set PG_active on a page whose
		 * PG_lru is clear (Marie isolated it). PG_active is in
		 * PAGE_FLAGS_CHECK_AT_FREE; if still set here it would
		 * trigger bad_page in free_pages_prepare. Clear it
		 * unconditionally as a last-resort safety net.
		 */
		if (unlikely(PageActive(f)))
			ClearPageActive(f);

#ifdef CONFIG_LRU_GEN
		/*
		 * Scrub MGLRU gen/refs residue. LRU_GEN_MASK is in
		 * PAGE_FLAGS_CHECK_AT_FREE, so a leftover gen counter trips
		 * "Bad page state" in free_pages_prepare. With Marie masking
		 * lru_gen_enabled() off (see lru_gen_enabled()), no MGLRU
		 * writer stamps these onto a tracked page, so this is the
		 * structural last resort that keeps any future regression a
		 * counter blip rather than a buddy-path oops -- independent of
		 * whether every lru_gen_enabled() reader stays correctly gated.
		 *
		 * PG_workingset is deliberately NOT cleared: Marie's eviction
		 * relies on the legacy workingset_eviction shadow encoding,
		 * which reads PG_workingset, and the bit is not in
		 * PAGE_FLAGS_CHECK_AT_FREE.
		 */
		if (unlikely(MARIE_FOLIO_FLAGS(f) & (LRU_GEN_MASK | LRU_REFS_MASK))) {
			VM_WARN_ON_ONCE_PAGE(1, f);
			set_mask_bits(&MARIE_FOLIO_FLAGS(f), LRU_GEN_MASK | LRU_REFS_MASK, 0);
		}
#endif
	}

	marie_state[pfn] = 0;

	g = (s & MARIE_PFN_GEN_MASK) >> MARIE_PFN_GEN_SHIFT;
	tier = (s & MARIE_PFN_TIER_MASK) >> MARIE_PFN_TIER_SHIFT;
	type_bit = (s & MARIE_PFN_TYPE_MASK) ? 1 : 0;

	/*
	 * The reclaim isolate path (marie_evict_counters_only) already retired
	 * this PFN's scan-bitmap slot + gen_occupied at isolate, leaving only
	 * the TRACKED byte (wiped just above). The atomic clear returns false on
	 * the common path (evict already retired the slot) so gen_occupied is NOT
	 * double-decremented below zero; it returns true only for the residual
	 * case -- a TRACKED page reaching free without having gone through isolate
	 * -- whose scan slot is genuinely still live, gating the decrement on it.
	 */
	if (marie_bm_clear(&marie_track_bm[type_bit][g][tier], pfn)) {
		marie_gen_occ_dec(pfn, g, type_bit);
		/*
		 * A TRACKED page reached buddy free with its scan slot still
		 * live -- it never went through Marie's evict. The common escape
		 * is settled one step earlier in lru_marie_uncharge_backstop,
		 * which runs while page->memcg_data is still valid, debits the
		 * global counters (marie_nr_pages + vmstat lru_size), and clears
		 * this bit -- so for any charged page this branch does not fire
		 * (verified: the free-hook escape counter is 0 once the backstop
		 * is in place). What is left here is the residual corner where
		 * uncharge_page never runs (memcg-disabled builds, never-charged
		 * pages): retiring the scan slot + occupied count is the whole
		 * obligation. The nr_pages / vmstat counters are not touched at
		 * free time -- a bounded corner accepted in exchange for not
		 * dereferencing a page that has no live charge context.
		 */
	}
}

/*
 * lru_marie_uncharge_backstop - debit an escaped tracked page at the last
 *                               point its owning memcg is still live.
 *
 * Called from mm/memcontrol.c::uncharge_page, immediately before
 * page->memcg_data is zeroed. This is the universal confluence point we
 * unify the counter debit on: every charged LRU page is uncharged before
 * it reaches the buddy allocator (per the allocator's contract), and the
 * memcg is still readable here -- unlike the page-free hook, which runs
 * after the uncharge and therefore has no memcg to debit.
 *
 * The per-(type,gen,tier) scan bit is the exactly-once token: the normal
 * evict/isolate path clears it when it debits the counter. If it is already
 * clear, the debit happened upstream and nothing is owed. If it is still
 * set, this page escaped evict entirely (its install +nr was never undone)
 * and we settle it here.
 *
 * The debit bucket is reconstructed PURELY from the per-PFN state byte:
 *   - zone   : the byte's ZONE field
 *   - lru    : fixed by TYPE alone -- install always credits INACTIVE_*
 *              (it clears PG_active before computing page_lru), so we
 *              must NOT read page_lru() here; the reclaim shrinker may
 *              have re-stamped PG_active on this isolated page.
 * No new per-PFN storage is needed: the byte says which bucket, the page's
 * still-live memcg says whose counter.
 *
 * IRQ-tolerant: uncharge can run from softirq (put_page on bio completion),
 * so local_irq_save guards the percpu_counter + vmstat updates rather than
 * asserting IRQs-on like the isolate funnel.
 */
void lru_marie_uncharge_backstop(struct page *page, struct mem_cgroup *memcg)
{
	unsigned long pfn, flags;
	u8 s, g, tier, type_bit, zone;
	enum lru_list lru;
	struct lruvec *lv;
	long nr;

	if (!marie_state_ready() || !marie_state)
		return;
	pfn = page_to_pfn(page);
	if (pfn >= marie_state_size)
		return;
	s = marie_state[pfn];
	if (!(s & MARIE_PFN_TRACKED))
		return;

	g = (s & MARIE_PFN_GEN_MASK) >> MARIE_PFN_GEN_SHIFT;
	tier = (s & MARIE_PFN_TIER_MASK) >> MARIE_PFN_TIER_SHIFT;
	type_bit = (s & MARIE_PFN_TYPE_MASK) ? 1 : 0;

	/*
	 * exactly-once gate: atomically claim (retire) the scan slot. bm_clear
	 * returns false when the bit is already clear -- evict debited upstream,
	 * so nothing is owed and the whole debit below is skipped. Winning the
	 * atomic clear owns the entire debit (gen_occupied + nr_pages + lru_size),
	 * so it can never double-debit even against a concurrent retirer.
	 */
	if (!marie_bm_clear(&marie_track_bm[type_bit][g][tier], pfn))
		return;
	marie_gen_occ_dec(pfn, g, type_bit);

	zone = (s & MARIE_PFN_ZONE_MASK) >> MARIE_PFN_ZONE_SHIFT;
	lru = type_bit ? LRU_INACTIVE_FILE : LRU_INACTIVE_ANON;
	nr = compound_nr(page);

	lv = marie_mem_cgroup_lruvec(memcg, page_pgdat(page));

	/* same wind-down as marie_account_evict_isolate, IRQ-context tolerant. */
	local_irq_save(flags);
	marie_pc_add(&marie_nr_pages, -1);
	__update_lru_size(lv, lru, zone, -nr);
	local_irq_restore(flags);
}

/*
 * marie_state_move_to_gen - relocate a tracked PFN's encoding to
 * (@target_gen, @target_tier) with matched (gen, type) bitmap +
 * occupied-counter updates.
 *
 * Step 1: CAS the state byte. Defeats races against del (cur becomes
 * 0) and against another concurrent move (cur changes). Retry on
 * mismatch.
 *
 * Step 2: shuffle the bitmaps / counters. Order is "new first, then
 * old" so the page is visible on at least one (gen, type) plane
 * throughout the transition. Skipped entirely when old_gen ==
 * target_gen (only the tier changed, no slot movement needed).
 *
 * Skipped if the page is no longer tracked, or the byte already
 * encodes (target_gen, target_tier).
 *
 * Called from:
 *   marie_state_inc_tier saturate path (target_gen=head, target_tier=0)
 *   shrink_lruvec residue putback (target_gen=(head+2)&3,
 *                                  target_tier=max(prev, w_tier))
 */
void marie_state_move_to_gen(unsigned long pfn,
			     u8 target_gen, u8 target_tier)
{
	u8 cur, type, old_gen, old_tier, new_byte;
	bool set_new, clr_old;

	if (pfn >= marie_state_size)
		return;
	target_gen &= MARIE_PFN_NR_GENS - 1;
	target_tier &= MARIE_PFN_TIER_MAX;

retry:
	cur = READ_ONCE(marie_state[pfn]);
	if (!(cur & MARIE_PFN_TRACKED))
		return;

	new_byte = (cur & ~(MARIE_PFN_GEN_MASK | MARIE_PFN_TIER_MASK)) |
		   ((u8)target_gen << MARIE_PFN_GEN_SHIFT) |
		   ((u8)target_tier << MARIE_PFN_TIER_SHIFT);
	if (new_byte == cur)
		return;

	if (cmpxchg(&marie_state[pfn], cur, new_byte) != cur)
		goto retry;

	type = (cur & MARIE_PFN_TYPE_MASK) ? 1 : 0;
	old_gen = (cur & MARIE_PFN_GEN_MASK) >> MARIE_PFN_GEN_SHIFT;
	old_tier = (cur & MARIE_PFN_TIER_MASK) >> MARIE_PFN_TIER_SHIFT;
	if (old_gen == target_gen && old_tier == target_tier)
		return;

	/*
	 * publish on new (type, gen, tier) first, then un-publish old. Gate each
	 * gen_occupied update on the atomic bit transition it owns (bm_set 0->1 /
	 * bm_clear 1->0), so a concurrent reclaim isolate retiring the old slot on
	 * this same page can never make both paths dec the old gen -- gen_occupied
	 * stays == popcount(bits in gen) >= 0.
	 */
	set_new = marie_bm_set(&marie_track_bm[type][target_gen][target_tier], pfn);
	clr_old = marie_bm_clear(&marie_track_bm[type][old_gen][old_tier], pfn);
	if (old_gen != target_gen) {
		if (set_new)
			marie_gen_occ_inc(pfn, target_gen, type);
		if (clr_old)
			marie_gen_occ_dec(pfn, old_gen, type);
	}
}
EXPORT_SYMBOL_GPL(marie_state_move_to_gen);

/*
 * marie_state_publish_at_gen - (re)publish an already-TRACKED PFN's scan
 * slot at (@target_gen, @target_tier), PUBLISH-ONLY (no un-publish of an
 * old slot).
 *
 * This is the putback counterpart to marie_evict_counters_only: isolate
 * already retired the old (gen, tier) bitmap bit + gen_occupied slot, so a
 * surviving page has NO old slot to clear -- only the new one to set.
 * Unlike marie_state_move_to_gen (set-new + clear-old), this never touches
 * the old coordinate, so it cannot double-decrement the l2_count / occupied
 * accounting that isolate already balanced.
 *
 * The byte stays TRACKED throughout (counters_only preserves it); here we
 * only rewrite its (gen, tier) field and set the matching bitmap bit +
 * occupied counter. Always sets the bitmap bit, even when the byte's
 * (gen, tier) is unchanged, because the bit itself was cleared at isolate.
 *
 * Caller context: putback, where the page is exclusively owned (PG_lru
 * cleared at claim, not yet republished; the dropped scan bit keeps the
 * walker away), so the cmpxchg cannot lose a race in practice -- it is
 * kept only to preserve the byte's TRACKED/TYPE/ZONE bits cleanly.
 */
static void marie_state_publish_at_gen(unsigned long pfn, u8 target_gen,
				       u8 target_tier)
{
	u8 cur, type, new_byte;

	if (pfn >= marie_state_size)
		return;
	target_gen &= MARIE_PFN_NR_GENS - 1;
	target_tier &= MARIE_PFN_TIER_MAX;

retry:
	cur = READ_ONCE(marie_state[pfn]);
	if (!(cur & MARIE_PFN_TRACKED))
		return;

	new_byte = (cur & ~(MARIE_PFN_GEN_MASK | MARIE_PFN_TIER_MASK)) |
		   ((u8)target_gen << MARIE_PFN_GEN_SHIFT) |
		   ((u8)target_tier << MARIE_PFN_TIER_SHIFT);
	if (new_byte != cur &&
	    cmpxchg(&marie_state[pfn], cur, new_byte) != cur)
		goto retry;

	type = (cur & MARIE_PFN_TYPE_MASK) ? 1 : 0;
	if (marie_bm_set(&marie_track_bm[type][target_gen][target_tier], pfn))
		marie_gen_occ_inc(pfn, target_gen, type);
}

/*
 * marie_state_inc_tier - saturating tier bump on the per-PFN byte.
 *
 * Runs from mark_page_accessed() WITHOUT lru_lock, so the state byte
 * is committed with try_cmpxchg to avoid losing a concurrent lock-free
 * drop_pfn / install publish (see the loop comment below).
 *
 * Non-saturated (tier < MAX): bump the tier field in place.
 *
 * Saturated (tier == MAX): in-place promote -- roll to head gen at
 * tier 0 (inlined marie_state_move_to_gen). The "already on head"
 * early exit avoids the CAS round-trip when the page cannot be
 * promoted further.
 */
/*
 * __marie_state_inc_tier - tier bump on the per-PFN byte from an
 * already-read state byte @cur. The caller guarantees pfn <
 * marie_state_size.
 *
 * mark_page_accessed() reaches the wrappers below from the fault /
 * pagecache-hit path WITHOUT lru_lock, racing the lock-free reclaim
 * isolate (marie_state_drop_pfn) and the lru_lock-held install publish.
 * All three RMW the same non-atomic state byte, so a plain READ/WRITE_ONCE
 * loses updates -- e.g. resurrecting a TRACKED bit drop_pfn just cleared.
 * Commit with try_cmpxchg; a concurrent writer forces a reload + recheck
 * (so a stale @cur seed only costs one extra loop iteration), and if
 * drop_pfn cleared TRACKED we bail.
 */
static void __marie_state_inc_tier(unsigned long pfn, u8 cur)
{
	u8 new, t, type, gen, head = 0, old_gen, new_tier = 0;
	bool roll;

	do {
		if (!(cur & MARIE_PFN_TRACKED))
			return;
		t = (cur & MARIE_PFN_TIER_MASK) >> MARIE_PFN_TIER_SHIFT;
		type = (cur & MARIE_PFN_TYPE_MASK) ? 1 : 0;
		if (t < MARIE_PFN_TIER_MAX) {
			new_tier = t + 1;
			new = (cur & ~MARIE_PFN_TIER_MASK) |
			      ((new_tier << MARIE_PFN_TIER_SHIFT) &
			       MARIE_PFN_TIER_MASK);
			roll = false;
		} else {
			/*
			 * Saturated promote rolls the page to the global
			 * head gen at tier 0 (desktop/global-only: one clock).
			 */
			head = (u8)atomic_read(&marie_head_gen[type]);
			old_gen = (cur & MARIE_PFN_GEN_MASK) >> MARIE_PFN_GEN_SHIFT;
			if (head == old_gen)
				return;
			new = (cur & ~(MARIE_PFN_GEN_MASK | MARIE_PFN_TIER_MASK)) |
			      (head << MARIE_PFN_GEN_SHIFT);
			roll = true;
		}
	} while (!try_cmpxchg(&marie_state[pfn], &cur, new));

	/*
	 * State byte committed. The bitmap moves are best-effort (the scanner
	 * re-validates each bit against the byte), but each gen_occupied update is
	 * gated on the atomic bit transition it owns (bm_set 0->1 / bm_clear 1->0):
	 * if a concurrent reclaim isolate retires the old (gen, tier) slot first,
	 * our bm_clear returns false and we skip the dec, so gen_occupied stays
	 * == popcount(bits in gen) >= 0 -- the double-dec underflow cannot happen.
	 */
	gen = (cur & MARIE_PFN_GEN_MASK) >> MARIE_PFN_GEN_SHIFT;
	if (!roll) {
		marie_bm_set(&marie_track_bm[type][gen][new_tier], pfn);
		marie_bm_clear(&marie_track_bm[type][gen][t], pfn);
	} else {
		if (marie_bm_set(&marie_track_bm[type][head][0], pfn))
			marie_gen_occ_inc(pfn, head, type);
		if (marie_bm_clear(&marie_track_bm[type][gen][t], pfn))
			marie_gen_occ_dec(pfn, gen, type);
	}
}

void marie_state_inc_tier(unsigned long pfn)
{
	if (pfn >= marie_state_size)
		return;
	__marie_state_inc_tier(pfn, READ_ONCE(marie_state[pfn]));
}
EXPORT_SYMBOL_GPL(marie_state_inc_tier);

/*
 * marie_state_inc_tier_seeded - tier bump when the caller already holds the
 * state byte and has bounds-checked @pfn (the walker, which gated on the
 * TRACKED bit just before clearing the young bit). Skips the redundant
 * reload + bound check; the cmpxchg loop still self-corrects @cur.
 */
void marie_state_inc_tier_seeded(unsigned long pfn, u8 cur)
{
	__marie_state_inc_tier(pfn, cur);
}
EXPORT_SYMBOL_GPL(marie_state_inc_tier_seeded);

/*
 * --------------------------------------------------------------------
 *  Anon/file swap-bias controller (stubborn proportional)
 * --------------------------------------------------------------------
 *
 * A single signed counter per marie_lruvec drives the anon-vs-file
 * pick under proportional swappiness (2..199). Granularity rule:
 * EXACTLY ONE type is scanned per shrink_lruvec call in the
 * proportional regime -- the bias sign selects which. Scanning both
 * sides in the same call would dissolve the s:(MAX-s) ratio because
 * every call would contribute pages from both. The caller's priority
 * loop re-enters shrink_lruvec for the next pick, and the bias
 * (updated from this call's outcome) may flip the selection in
 * between -- yielding "fine-grained" type switching at call
 * granularity, which matches the user-visible reclaim cadence.
 *
 *   SUCCESS (nr_reclaimed > 0):
 *     bias += sign * nr_reclaimed * weight
 *     -- page-flow proportional. Long-run pages(anon):pages(file)
 *        converges to s:(MAX_SWAPPINESS-s) even when per-pick batch
 *        sizes differ systematically between types.
 *
 *   FAILURE (nr_reclaimed == 0):
 *     bias unchanged (no-op).
 *     -- The picked side stays the picked side. Failure carries no
 *        back-pressure -- not even a unit nudge -- so the favored
 *        side remains favored indefinitely under sustained failure.
 *        This is the entire point of low-swappiness on modern ZRAM
 *        systems: file should be the eviction target even when it
 *        transiently (or persistently) produces nothing, and anon
 *        must NOT be touched as a consequence of file being stuck on
 *        dirty / locked / writeback / depleted state. If file truly
 *        cannot be reclaimed, the caller escalates priority or OOM
 *        kicks in -- the controller does not surrender protection.
 *
 *   sign = -1 for picked=ANON (push bias toward FILE)
 *          +1 for picked=FILE (push bias toward ANON)
 *   weight = MAX_SWAPPINESS - s   for picked=ANON
 *          = s                    for picked=FILE
 *
 * Special-value swappiness short-circuits the controller:
 *   s=0   FILE only, no fallback (caller proceeds to OOM if depleted)
 *   s=1   FILE first; ANON engages on EITHER of two depletion
 *         signals (see the FILE_THEN_ANON tail gate):
 *           - file < clean_min_ratio floor (skip_file true), or
 *           - file >= floor but the FILE pass FAILED TO MEET this
 *             call's reclaim target = file reclaim is not keeping
 *             pace right now.
 *         Throughput is empirical -- a tracked file page may be
 *         hot/dirty/mapped, and how much frees is knowable only by
 *         trying -- so the FILE pass's own outcome, not occupancy, is
 *         the signal. Sufficiency (target met) rather than exact-zero
 *         is what keeps reclaim file-first: a positive-but-insufficient
 *         file trickle must not pin reclaim file-only while swappable
 *         anon OOMs with swap free. The fallback fires on the first
 *         call file cannot satisfy -- it does NOT wait for sc->priority
 *         to decay -- and a transient file stall costs at most one
 *         early anon batch; preferred over OOM with swap free.
 *   s=MAX ANON only, no fallback (symmetric to s=0)
 *
 * clean_min_ratio override: when the floor diverts reclaim to
 * anon-only (skip_file in marie_state_shrink_lruvec), the caller
 * does NOT invoke marie_swap_bias_update for that call. The
 * controller stays frozen at its pre-override value so that, when
 * file recovers above the floor, the proportional regime resumes
 * from where it left off -- no post-recovery overshoot from anon
 * reclaim that was driven by external policy, not swappiness.
 *
 * Sysctl writes invoke lru_marie_swappiness_changed() which walks
 * the xarray and resets every swap_bias to zero, so the controller
 * restarts cleanly under the new weight ratio.
 *
 * No CAP: per-cycle delta is bounded by batch_max (~8192) *
 * MAX_SWAPPINESS (200) ~ 1.6e6, far below S64_MAX in any realistic
 * running time. The sysctl-write reset is the only reset mechanism.
 */

/*
 * Swappiness/pick diagnostics (read via /sys/kernel/mm/lru_marie/stats):
 * which pick regime runs, and how many pages Marie's own shrinker reclaims
 * per type. A large reclaimed[anon] vs pswpout means Marie's gate; a small
 * one means the anon comes from elsewhere (legacy orphan drain / non-Marie).
 */
atomic_long_t marie_dbg_pick[5];
atomic_long_t marie_dbg_reclaimed[2];

/* Concede-trigger attribution: [0]=floor [1]=free-pressure [2]=refault [3]=memcg. */
atomic_long_t marie_dbg_concede[4];

/*
 * Orphaned-bit self-heal count, per type [0]=anon [1]=file. See the
 * byte-mismatch branch in marie_state_isolate_scan_l2lock: counts L1 bits
 * the scanner found set at a (type, gen, tier) the PFN's current state byte
 * no longer encodes, and therefore cleared + retired from gen_occupied on
 * the spot. A steadily climbing count during a hot single-type burst is the
 * signature of the walker/isolate/putback race that produces these orphans;
 * see the fix comment at the clear site for the full mechanism.
 */
atomic_long_t marie_dbg_orphan_bit[2];

/*
 * GLOBAL anon-vs-file proportional pick bias (desktop/global-only). Signed:
 * < 0 favours FILE, >= 0 favours ANON. A single node-wide controller drives
 * every reclaim pass.
 */
atomic64_t marie_swap_bias;

enum marie_pick_kind marie_swap_pick_type(u8 swappiness)
{
	if (swappiness == 0)
		return MARIE_PICK_FILE_STRICT;
	if (swappiness == 1)
		return MARIE_PICK_FILE_THEN_ANON;
	if (swappiness >= MAX_SWAPPINESS)
		return MARIE_PICK_ANON_STRICT;

	return (atomic64_read(&marie_swap_bias) < 0)
		? MARIE_PICK_FILE_FIRST
		: MARIE_PICK_ANON_FIRST;
}

void marie_swap_bias_update(int picked_type,
			    unsigned long nr_reclaimed,
			    u8 swappiness)
{
	s64 delta;

	/*
	 * Special values bypass the controller. The pick path does not
	 * read swap_bias under {0, 1, MAX_SWAPPINESS}, so the value
	 * here is irrelevant to observable behaviour; skipping the
	 * write also avoids gratuitous cache-line bouncing.
	 */
	if (swappiness <= 1 || swappiness >= MAX_SWAPPINESS)
		return;

	/*
	 * Failure carries no back-pressure: when nr_reclaimed is zero,
	 * the bias is left untouched. The picked side stays the picked
	 * side -- truly stubborn protection of the favored type. See
	 * the top of this section for the failsafe semantics.
	 */
	if (!nr_reclaimed)
		return;

	if (picked_type == 0)
		delta = -(s64)nr_reclaimed *
			(s64)(MAX_SWAPPINESS - swappiness);
	else
		delta = +(s64)nr_reclaimed * (s64)swappiness;

	atomic64_add(delta, &marie_swap_bias);
}

/*
 * marie_file_floor_protect - is the clean_min_ratio file floor in force?
 *
 * Returns true when this node's clean file pagecache has fallen TO OR BELOW
 * marie_clean_min_ratio (% of node_present_pages) and Marie still has
 * anon to absorb the pressure, so file reclaim must be withheld. The pick
 * driver diverts file -> anon on this signal (skip_file) and folds the
 * result into the MARIE_DRAIN_* mask it returns, so shrink_lruvec's legacy
 * orphan drain spares file too. No reclaim path may evict file below the
 * floor -- le9uo's single-path floor invariant applied across Marie's paths.
 *
 * Only CLEAN file counts toward the floor (NR_FILE_DIRTY subtracted):
 * dirty pages cannot be reclaimed without writeback, so counting them
 * would let the floor be satisfied by unreclaimable pages and strand the
 * clean working set.
 *
 * If anon is empty Marie has no reserve to protect anyway, so the floor
 * yields and file scan proceeds as a last resort. An OOM victim bypasses
 * the floor entirely (its file is fair game; see the oom_victim handling
 * in marie_state_shrink_lruvec).
 */
static bool marie_file_floor_protect(struct pglist_data *pgdat)
{
	unsigned int min_ratio = READ_ONCE(marie_clean_min_ratio);
	unsigned long file_pages, dirty, file_min;
	long anon_occupied = 0;
	int g;

	if (!min_ratio || unlikely(tsk_is_oom_victim(current)))
		return false;

	file_pages = node_page_state(pgdat, NR_ACTIVE_FILE) +
		     node_page_state(pgdat, NR_INACTIVE_FILE);
	dirty = node_page_state(pgdat, NR_FILE_DIRTY);
	file_pages = (file_pages > dirty) ? file_pages - dirty : 0;
	file_min = pgdat->node_present_pages * min_ratio / 100;

	/*
	 * Strict '>' (not '>='): clean file AT EXACTLY the floor must be
	 * protected, not reclaimed. Under swappiness=1 the FILE_THEN_ANON tail
	 * concedes to anon iff this returns true (concede == floor_protect for
	 * global reclaim); with '>=', file held at the floor by pagecache refill
	 * never crosses STRICTLY below it, so concede never fires and GBs of cold
	 * anon are stranded behind a file refill treadmill -- the allocator
	 * livelocks (anon still counts as reclaimable, so the OOM gate never
	 * trips) instead of swapping. Conceding at the floor is the intended
	 * "file first, until the floor, then anon".
	 */
	if (file_pages > file_min)
		return false;

	for (g = 0; g < MARIE_PFN_NR_GENS; g++)
		anon_occupied += atomic_long_read(&marie_gen_occupied[g][0]);

	return anon_occupied > 0;
}

/*
 * marie_node_under_pressure - is this node failing to hold free above its
 * watermarks?
 *
 * The FILE_THEN_ANON tail (swappiness=1, Marie's default) concedes to anon
 * only once clean file has drained BELOW the clean_min_ratio floor. That DEFER
 * assumes successive file-only calls DRAIN clean file to the floor. A workload
 * that refills clean file above the floor faster than the batch-capped sweep
 * drains it -- a compile streaming its object cache, dozens of mmap'd binaries
 * whose text re-faults the moment it is dropped -- holds the floor forever
 * unreached: the aged FILE ring keeps depleting (fresh reads land in the young
 * head gen), the FILE pass falls short every time it drains it, yet the DEFER
 * fires because the total file LEVEL is still above the floor. Free stays
 * pinned at the watermarks while GBs of swappable anon -- including swap-backed
 * shmem (tmpfs, memfd/ZGC heaps) -- are never offered to swap: the box thrashes
 * on the file refill treadmill and OOMs with swap free.
 *
 * When free has actually fallen to the watermarks the file level is moot: the
 * allocator cannot build headroom, so offer anon to swap now regardless of the
 * floor. Same free <= 2*high gate the thrash watchdog uses
 * (thrash_wd_mem_pressured in oom_kill.c), applied here one layer earlier so
 * anon reaches swap as pressure builds rather than only after the watchdog's
 * multi-second stall. A busy-but-healthy large-RAM box streaming page cache
 * keeps free well above the watermarks (the FILE pass also meets its target
 * from the abundant cache and never reaches this tail), so this stays false
 * and file-first is preserved.
 */
static bool marie_node_under_pressure(struct pglist_data *pgdat)
{
	unsigned long free = 0, high = 0;
	int z;

	for (z = 0; z < MAX_NR_ZONES; z++) {
		struct zone *zone = &pgdat->node_zones[z];

		if (!managed_zone(zone))
			continue;
		free += zone_page_state(zone, NR_FREE_PAGES);
		high += high_wmark_pages(zone);
	}

	return free <= high * 2;
}

/*
 * marie_file_refaulting - is reclaim evicting FILE that comes straight back?
 *
 * The refault-feedback pressure signal (concede_pressure_mode & REFAULT).
 * WORKINGSET_REFAULT_FILE counts file pages faulted back in that carried an
 * eviction shadow -- pages we dropped and that returned. When that runs at
 * >= half of Marie's own file reclaim rate over the sample window, the
 * "clean" file we keep dropping IS the working set (hot; a refill/refault
 * treadmill), so file-first is futile -- concede to anon regardless of the
 * clean_min_ratio floor or the free level. Unlike free<=2*high this measures
 * the pathology (are the evicted pages actually cold?) directly, and fires as
 * file goes hot rather than only after free has cratered.
 *
 * Sampled globally (Marie is desktop/global-only) at most ~4x/s under a
 * trylock; concurrent reclaimers read the cached boolean. First refresh only
 * seeds the baseline (no bogus cumulative delta). The 2x weight mirrors the
 * thrash watchdog's refault:steal ratio in mm/oom_kill.c.
 */
static DEFINE_SPINLOCK(marie_rf_lock);
static unsigned long marie_rf_next;			/* jiffies of next refresh */
static unsigned long marie_rf_last_rf, marie_rf_last_st;	/* file refault / reclaim */
static bool marie_rf_primed;
static bool marie_rf_file_hot;

static bool marie_file_refaulting(void)
{
	unsigned long now = jiffies, rf, st, drf, dst;

	if (time_before(now, READ_ONCE(marie_rf_next)))
		return READ_ONCE(marie_rf_file_hot);
	if (!spin_trylock(&marie_rf_lock))
		return READ_ONCE(marie_rf_file_hot);
	if (time_before(now, marie_rf_next)) {		/* lost the refresh race */
		spin_unlock(&marie_rf_lock);
		return READ_ONCE(marie_rf_file_hot);
	}

	rf = global_node_page_state(WORKINGSET_REFAULT_FILE);
	st = atomic_long_read(&marie_dbg_reclaimed[1]);		/* type 1 == file */
	WRITE_ONCE(marie_rf_next, now + HZ / 4);

	if (unlikely(!marie_rf_primed)) {
		marie_rf_last_rf = rf;
		marie_rf_last_st = st;
		marie_rf_primed = true;
		spin_unlock(&marie_rf_lock);
		return false;
	}

	drf = rf - marie_rf_last_rf;
	dst = st - marie_rf_last_st;
	marie_rf_last_rf = rf;
	marie_rf_last_st = st;
	/* >= half of the file reclaimed this window faulted straight back. */
	WRITE_ONCE(marie_rf_file_hot, dst && drf * 2 >= dst);
	spin_unlock(&marie_rf_lock);

	return READ_ONCE(marie_rf_file_hot);
}

/*
 * marie_state_shrink_lruvec - per-PFN paradigm reclaim driver.
 *
 * Aging has two head-advance triggers. SUPPLY-PUSH: install cadence
 * advances the head every marie_gen_growth_live[type] installs
 * (marie_page_install). DEMAND-PULL: when a type's sweep finds the aged
 * ring exhausted (find_oldest < 0) but cold pages are parked in the head
 * gen, it seals the head so they age into reclaim range -- the
 * reclaim-driven trigger, fired ONLY on true exhaustion (the retired
 * unconditional "occupied < 2 at entry" form thrashed the ring). Without
 * demand-pull a workload that stops installing a type strands its
 * head-gen pages, since marie_find_oldest_occupied skips head (the
 * install destination).
 *
 * Per (type, tier) the scan walks the per-(gen, type) bitmap, claims
 * each candidate via get_page_unless_zero + TestClearPageLRU, then calls
 * marie_evict_counters_only: counters decremented and the scan-bitmap
 * slot + gen_occupied retired at isolate (so other CPUs stop re-finding
 * the in-flight page), but the per-PFN TRACKED byte is KEPT so
 * install_local's early-out blocks a concurrent install from re-setting
 * PG_lru while shrink_page_list reclaims it.
 *
 * Teardown of the TRACKED byte is deferred: a reclaimed page is wiped
 * at its buddy handoff (marie_state_drop_pfn_at_free via the
 * free_pages_prepare hook), which finds the scan bit already clear and
 * so does not double-decrement l2_count / gen_occupied. Survivors of
 * shrink_page_list keep TRACKED and are re-published at the putback gen
 * via marie_state_publish_at_gen (set-only: no clear-old, because isolate
 * already retired the old slot), seeding tier from max(prev_tier,
 * PG_active/PG_workingset).
 */

unsigned int marie_state_shrink_lruvec(struct lruvec *lruvec, struct scan_control *sc)
{
	struct pglist_data *pgdat = lruvec_pgdat(lruvec);
	struct mem_cgroup *memcg = lruvec_memcg(lruvec);
	/*
	 * Desktop/global-only Marie: the scan is ALWAYS global -- there is no
	 * per-memcg reclaim, so even a cgroup-targeted shrink scans the whole
	 * node's oldest gen (best-effort; memory.max is not enforced via
	 * reclaim).
	 */
	/*
	 * @swappiness is captured once per call; subsequent sysctl
	 * writes that reset the global bias to zero are seen on the NEXT
	 * call. mem_cgroup_swappiness returns the effective value (memcg own
	 * value on cgroup v1 non-root, vm_swappiness otherwise) and is
	 * a plain READ_ONCE under the hood.
	 *
	 * low_swappiness_mode (default on) clamps the effective value to at
	 * most 1 (MARIE_PICK_FILE_THEN_ANON), Marie's recommended policy,
	 * regardless of the higher values vm.swappiness / memory.swappiness
	 * udev rules, tuning daemons, or distro defaults have installed. It
	 * only ever LOWERS swappiness, so the special "never swap" value 0
	 * (MARIE_PICK_FILE_STRICT: OOM rather than touch anon) is preserved --
	 * an operator who deliberately set 0 still gets 0. Clear the knob to
	 * honour the configured value verbatim. See the rationale at the top
	 * of core.c.
	 */
	u8 configured = (u8)mem_cgroup_swappiness(memcg);
	u8 swappiness = (READ_ONCE(marie_low_swappiness_mode) && configured > 1) ?
			1 : configured;
	enum marie_pick_kind pick_kind;
	int type_order[2];
	int type_count;
	int idx;
	bool skip_file = false;
	unsigned int drain_mask;
	/*
	 * When anon cannot be reclaimed at all (no free swap slots,
	 * cgroup swap limit hit, no demotion target), swappiness is by
	 * definition meaningless -- it expresses the anon:file reclaim
	 * ratio, and one side of that ratio no longer exists. Every ANON
	 * pick would reclaim nothing, and because the bias controller
	 * takes no back-pressure from a zero-reclaim pick
	 * (marie_swap_bias_update bails on !nr_reclaimed), the bias never
	 * flips to FILE: reclaimable file cache is stranded until OOM.
	 * Drop the stubborn swappiness preference and force FILE only,
	 * mirroring get_scan_count()'s "!can_reclaim_anon_pages ->
	 * SCAN_FILE". The clean_min_ratio floor below still applies, so
	 * file is reclaimed only down to the protected floor; once file is
	 * at the floor and anon is unreclaimable this pass reclaims nothing,
	 * and the stock no_progress_loops path in should_reclaim_retry()
	 * reaches the OOM killer.
	 */
	bool anon_unreclaimable =
		!vmscan_can_reclaim_anon_pages(memcg, pgdat->node_id, sc);
	/*
	 * An OOM victim's own direct reclaim runs FILE-only, with no holds
	 * barred on the file side: scan FILE ignoring the swappiness/bias
	 * pick, the clean_min_ratio floor, the FILE_THEN_ANON tail gate and
	 * the bias controller. The task has been selected for death and the
	 * OOM reaper frees its anon, so swapping anon here would only add
	 * I/O thrash for no benefit -- reclaim just the cheap, no-I/O file
	 * side (clean_min_ratio is bypassed below, so all file is fair
	 * game). If file is exhausted the victim falls back on the reaper,
	 * which is the normal OOM mechanism. kswapd is never an OOM victim,
	 * so background reclaim is unaffected.
	 */
	bool oom_victim = tsk_is_oom_victim(current);
	int type;

	/*
	 * No head advance here. Aging is driven by install cadence
	 * (marie_page_install advances the head every
	 * marie_gen_growth_live[type] installs, under lru_lock). The old
	 * reclaim-time "occupied < 2" advance thrashed the ring under
	 * concurrent reclaim; see marie_try_advance_head_mlv.
	 */

	/*
	 * clean_min_ratio hard floor. True when this node's clean file
	 * pagecache is below the configured percentage of node_present_pages
	 * (and anon remains, and we are not an OOM victim). The same predicate
	 * masks the legacy drain's file scan in shrink_lruvec, so no path
	 * evicts file below the floor (le9uo's single-path floor invariant).
	 */
	skip_file = marie_file_floor_protect(pgdat);

	/*
	 * Choose the type(s) to scan as a strict priority cascade:
	 *
	 *   oom_victim         -> FILE only. The victim's anon is reaped by the
	 *                         OOM reaper, so swapping anon is pure I/O thrash;
	 *                         reclaim the cheap no-I/O file side. The floor is
	 *                         bypassed for victims (skip_file is false), so
	 *                         file scans freely.
	 *   anon_unreclaimable -> FILE only. No free swap slots / no demotion
	 *                         target: swappiness is meaningless and every ANON
	 *                         pick would free nothing. If file is also at the
	 *                         floor the per-iteration gate no-ops the file
	 *                         scan and the stock no_progress_loops path OOMs.
	 *   swappiness == 0     -> FILE only. Hard "never swap" user policy: the
	 *                         clean_min_ratio floor must NOT punch through it
	 *                         (core.c). At the floor file is blocked too, so
	 *                         this OOMs rather than swapping -- the contract.
	 *   skip_file          -> ANON only. The floor is in force and file is
	 *                         protected, so divert all reclaim to anon
	 *                         regardless of the swappiness/bias pick. This
	 *                         outranks the proportional controller: a
	 *                         FILE_FIRST pick would otherwise scan the
	 *                         floor-blocked file side, free nothing, and --
	 *                         the bias being frozen during skip_file -- stay
	 *                         pinned on FILE while anon is never picked,
	 *                         stalling reclaim under pressure at high swappiness.
	 *   otherwise          -> the swappiness / swap_bias proportional pick.
	 */
	if (oom_victim)
		pick_kind = MARIE_PICK_FILE_STRICT;
	else if (anon_unreclaimable)
		pick_kind = MARIE_PICK_FILE_STRICT;
	else if (swappiness == 0)
		pick_kind = MARIE_PICK_FILE_STRICT;
	else if (skip_file)
		pick_kind = MARIE_PICK_ANON_STRICT;
	else
		pick_kind = marie_swap_pick_type(swappiness);

	if ((unsigned int)pick_kind < ARRAY_SIZE(marie_dbg_pick))
		atomic_long_inc(&marie_dbg_pick[pick_kind]);

	switch (pick_kind) {
	case MARIE_PICK_FILE_STRICT:
		type_order[0] = 1;
		type_count = 1;
		break;
	case MARIE_PICK_ANON_STRICT:
		type_order[0] = 0;
		type_count = 1;
		break;
	case MARIE_PICK_FILE_THEN_ANON:
		/*
		 * swappiness=1: FILE first, ANON as the depletion fallback
		 * the moment FILE fails to satisfy this call's reclaim
		 * target (not only when FILE returns exactly zero).
		 * type_count=2 with the sufficiency gate at the tail.
		 */
		type_order[0] = 1;
		type_order[1] = 0;
		type_count = 2;
		break;
	case MARIE_PICK_FILE_FIRST:
		/*
		 * Proportional regime, bias picks FILE. SINGLE type per
		 * call: scanning the other side in the same call would
		 * dissolve the s:(MAX-s) ratio because both sides would
		 * contribute pages on every invocation. The caller
		 * (vmscan priority loop) re-enters shrink_lruvec for
		 * the next pick; bias may flip in between via the
		 * proportional update from this call's outcome.
		 */
		type_order[0] = 1;
		type_count = 1;
		break;
	case MARIE_PICK_ANON_FIRST:
	default:
		/* Symmetric: proportional regime, bias picks ANON. */
		type_order[0] = 0;
		type_count = 1;
		break;
	}

	/*
	 * Tell shrink_lruvec which orphan type(s) its legacy drain may
	 * reclaim: exactly the type this call scans. type_order[0] is the
	 * primary (and, in the single-type regime, only) type. A file pick
	 * blocked by skip_file (FILE_STRICT under the clean_min_ratio floor)
	 * scans nothing, so it grants no drain -- preserving the
	 * no-progress -> OOM path.
	 */
	if (type_order[0] == 1)
		drain_mask = skip_file ? 0 : MARIE_DRAIN_FILE;
	else
		drain_mask = MARIE_DRAIN_ANON;

	{
		/*
		 * Claim this CPU's pre-allocated shrink buffer. If the
		 * cmpxchg fails (preempted reclaimer on the same CPU
		 * holds it across a shrink_page_list sleep), fall back
		 * to a small stack batch.
		 */
		struct marie_shrink_buf *buf;
		/*
		 * Fallback uses MARIE_PFN_FALLBACK_BATCH-sized stack
		 * arrays. Sized to stay under gcc -Wframe-larger-than=2048
		 * given the ~464 B baseline frame; see MARIE_PFN_FALLBACK_
		 * BATCH comment.
		 */
		struct page *small_batch[MARIE_PFN_FALLBACK_BATCH];
		struct page **scratch_batch;
		/* Hard page-count ceiling: the scratch array's actual slot
		 * count. Never exceeded regardless of page order. */
		unsigned long array_cap;
		/* Soft PAGE-count budget: see marie_pfn_batch_threshold(). */
		unsigned long batch_max;
		bool using_percpu;

		buf = per_cpu_ptr(&marie_shrink_buf, raw_smp_processor_id());
		if (atomic_cmpxchg(&buf->in_use, 0, 1) == 0) {
			scratch_batch = buf->batch;
			array_cap = MARIE_PFN_SHRINK_BATCH;
			batch_max = marie_pfn_batch_threshold(sc);
			using_percpu = true;
		} else {
			scratch_batch = small_batch;
			array_cap = MARIE_PFN_FALLBACK_BATCH;
			batch_max = MARIE_PFN_FALLBACK_BATCH;
			using_percpu = false;
		}

		for (idx = 0; idx < type_count; idx++) {
			unsigned int tier;
			int oldest;
			bool ignore_refs = false;
			LIST_HEAD(page_list);
			struct reclaim_stat stat = {};
			unsigned long n_taken = 0;
			/* PAGE count of isolated pages (n_taken counts pages; a
			 * THP is compound_nr pages). NR_ISOLATED_* and PGSCAN_*
			 * are page counters by kernel convention, so they must use
			 * this, not n_taken -- else THP undercounts NR_ISOLATED and
			 * too_many_isolated() under-throttles concurrent reclaim. */
			unsigned long n_taken_pages = 0;
			unsigned int n_reclaimed = 0;
			int oldest_for_putback;
			u8 putback_gen;
			struct page *f, *tmp;
			/* Per-sweep reclaim accumulator across the aged gen ring. */
			unsigned long total_reclaimed = 0;
			int sweep_i;
			/*
			 * Tracks whether this iteration actually attempted
			 * to pick the type. An external override
			 * (skip_file from clean_min_ratio) clears this so
			 * the bias controller is NOT updated for a pick
			 * that never ran -- the bias must reflect actual
			 * picking policy, not blocked intentions.
			 */
			bool attempted_pick = true;

			type = type_order[idx];

			/*
			 * SWEEP the aged gen ring (oldest -> head) for this type
			 * in ONE call: reclaim until the target is met (goto
			 * done) or every aged gen has been visited and reclaim
			 * still fell short. Bounded by MARIE_PFN_NR_GENS
			 * iterations -- there are only NR_GENS-1 non-head gens,
			 * and survivors re-publish ahead of the oldest pointer
			 * (hot -> head, cold -> oldest+1) so find_oldest advances
			 * monotonically toward head. This makes "swept the whole
			 * ring and still short" a direct, content-based failure
			 * signal for the s1 gate below, replacing the former
			 * one-gen-per-call scan that leaned on vmscan's priority
			 * loop (and decayed priority) to drive the sweep.
			 *
			 * A plain `break` (skip_file, ring empty, nothing
			 * isolatable) drops to the per-type tail; `goto done`
			 * (target reached) bypasses the tail entirely.
			 */
			for (sweep_i = 0; sweep_i < MARIE_PFN_NR_GENS; sweep_i++) {

			if (type == 1 && skip_file) {
				attempted_pick = false;
				break;
			}

			/* Reset per-gen scratch for this sweep step. */
			INIT_LIST_HEAD(&page_list);
			stat = (struct reclaim_stat){};
			n_taken = 0;
			n_taken_pages = 0;
			n_reclaimed = 0;
			ignore_refs = false;

			oldest = marie_find_oldest_occupied_mlv(type);
			if (oldest < 0) {
				u8 cur_head;
				/*
				 * Demand-pull aging. The aged ring is out of
				 * reclaimable pages of this type, but cold/clean
				 * pages may be PARKED in the head gen: supply-push
				 * (install cadence) advances the head only as new
				 * pages of this type are installed, so a workload
				 * that stops installing it (a static file cache; a
				 * cold anon burst touched once then idle) strands
				 * its head-gen pages, which find_oldest -- skipping
				 * the head by design -- cannot reach. Seal the head
				 * so they age into reclaim range, then retry the
				 * sweep.
				 *
				 * This is the reclaim-driven counterpart to
				 * supply-push -- the retired occupied<2 trigger,
				 * restored in demand-pull form: it fires ONLY on
				 * true aged exhaustion (find_oldest<0), not at every
				 * shrink entry (the unconditional firing is what
				 * thrashed the ring), and is bounded by the
				 * MARIE_PFN_NR_GENS sweep cap. Needed for ANON too:
				 * without it a cold anon burst parked at the head
				 * could never be swapped under high swappiness. For
				 * FILE the clean_min_ratio floor gates it so the
				 * reserve is not breached; anon has no such reserve.
				 */
				bool may_advance = (type != 1) ||
					!marie_file_floor_protect(pgdat);

				if (may_advance) {
					cur_head = (u8)atomic_read(
						&marie_head_gen[type]);
					/*
					 * Only seal a NON-EMPTY head. Advancing an
					 * empty head would recycle it into a sealed
					 * empty gen -- a hole that find_oldest skips
					 * and that nothing fills until the head laps
					 * the ring. The other advance site (install
					 * cadence in marie_page_install) is likewise
					 * non-empty: it fires right after the publish
					 * that bumped gen_occupied[head]. So no
					 * caller of marie_try_advance_head_mlv ever
					 * advances an empty head.
					 */
					if (atomic_long_read(
						&marie_gen_occupied[cur_head][type]) > 0 &&
					    marie_try_advance_head_mlv(type))
						continue;
				}
				break;
			}
			/*
			 * Force-reclaim referenced pages at the oldest gen once
			 * the walker has swept it since the head recycled it: the
			 * aging epoch has moved past the recycle stamp.
			 */
			ignore_refs =
				(s32)(atomic_read(&marie_aging_epoch[type]) -
				      READ_ONCE(marie_recycle_epoch[oldest][type])) > 0;

			/*
			 * File-preference policy contract (swappiness <= 1):
			 * while clean file is ABOVE the clean_min_ratio floor,
			 * reclaim it in PREFERENCE to swapping anon -- including
			 * REFERENCED clean file. Under a workload that keeps
			 * reading its pagecache (PG_referenced set), the
			 * walker-gated ignore_refs above is mostly false, so
			 * page_check_references rotates that file and the FILE
			 * pass falls short -> concedes to anon while GBs of
			 * reclaimable clean file sit far above the floor (the
			 * nibble; measured file ~5x the floor with pgscan_file ~=
			 * 2x pgsteal_file). Re-reading rotated file from disk is
			 * the accepted cost of a file-preference policy, so FORCE
			 * ignore_refs. The live floor check stops it the moment
			 * file reaches the reserve, protecting the working-set
			 * reserve below the floor.
			 *
			 * swappiness <= 1 captures both file-only POLICY regimes:
			 *   0 -> never swap anon (FILE_STRICT)
			 *   1 -> drain file to the floor before anon (FILE_THEN_ANON)
			 * Both are the user's explicit choice to evict file over
			 * swapping anon, so the referenced-file rotation cost is
			 * accepted. swappiness >= 2 is NOT forced: the proportional
			 * bias already routes pressure to anon when file is hard to
			 * reclaim, and force-evicting the working set there would
			 * thrash (refault) instead. anon_unreclaimable is likewise
			 * NOT forced -- it is a circumstance, not a policy; the
			 * default reference-respecting reclaim + epoch backstop are
			 * the right (gentle) handling near OOM, and forcing would
			 * only add refault thrash to a doomed system.
			 */
			if (type == 1 && swappiness <= 1 &&
			    !marie_file_floor_protect(pgdat))
				ignore_refs = true;

			/*
			 * Throttle before isolating: if other concurrent
			 * reclaimers already have more isolated than the
			 * inactive list holds, wait for them to catch up
			 * instead of piling more pages into flight on top.
			 * One retry (mirrors shrink_inactive_list's
			 * `stalled` bool) -- but unlike an earlier version of
			 * this check, if it is STILL too many after that one
			 * wait, give up on this type for the rest of the
			 * sweep (drop to the per-type tail, same as the
			 * skip_file/ring-empty `break` above) rather than
			 * isolating another batch_max-sized batch (up to
			 * MARIE_PFN_SHRINK_BATCH pages) regardless.
			 * batch_max scales up to 256x SWAP_CLUSTER_MAX under
			 * heavy pressure, so "proceed anyway" lets every
			 * stalled reclaimer pile a full batch on top of an
			 * already-excessive isolated count instead of
			 * bailing out the way shrink_inactive_list's
			 * `stalled` path does (it returns 0 rather than
			 * isolating unconditionally). That mismatch is what
			 * let isolated_anon balloon into the tens-of-GB range
			 * (GitHub issue #6) even after the batch was already
			 * page-budget-capped: each concurrent reclaimer got
			 * one grace wait, then isolated up to its full batch
			 * regardless of whether the pileup had cleared.
			 * kswapd never blocks here (see
			 * marie_too_many_isolated).
			 */
			if (unlikely(marie_too_many_isolated(pgdat, type, sc))) {
				reclaim_throttle(pgdat, VMSCAN_THROTTLE_ISOLATED);
				if (fatal_signal_pending(current))
					goto done;
				if (unlikely(marie_too_many_isolated(pgdat, type, sc)))
					break;
			}

			/*
			 * Accumulate across all tiers of this type into one
			 * page_list up to batch_max, then call
			 * shrink_page_list once.
			 *
			 * Scan writes candidate pages directly into
			 * scratch_batch[n_taken..] in a SINGLE call per
			 * tier. The previous SWAP_CLUSTER_MAX-bounded
			 * tmp_batch did 128 scan invocations per type at
			 * batch_max=4096, re-initialising the prefetch
			 * ring each time -- now one invocation per tier
			 * (4 per type) lets the ring amortise across the
			 * full bitmap walk.
			 *
			 * Failed claims (try_get / test_clear_lru) leave
			 * the corresponding scratch_batch slot to be
			 * overwritten by the next successful claim --
			 * in-place compaction via accept_idx.
			 */
			for (tier = 0; tier < MARIE_PFN_NR_TIERS; tier++) {
				unsigned long nr_isolated, i;
				unsigned long room;
				unsigned long accept_idx = n_taken;

				if (sc_reclaim_target_reached(sc))
					goto done;
				/* Hard ceiling: never overrun the scratch array. */
				if (n_taken >= array_cap)
					break;
				/* Soft ceiling: cap PAGE exposure, not page
				 * count -- THP pages must not let one pass
				 * isolate far more memory than batch_max
				 * implies. */
				if (n_taken_pages >= batch_max)
					break;

				room = array_cap - n_taken;
				nr_isolated = marie_state_isolate_scan_l2lock(
					pgdat, type, sc_reclaim_idx(sc),
					tier,
					&scratch_batch[n_taken], room,
					ULONG_MAX, oldest);
				if (!nr_isolated)
					continue;

				for (i = 0; i < nr_isolated; i++) {
					f = scratch_batch[n_taken + i];
					if (!get_page_unless_zero(f))
						continue;
					if (!TestClearPageLRU(f)) {
						put_page(f);
						continue;
					}

					scratch_batch[accept_idx] = f;

					/*
					 * marie_evict_counters_only decrements
					 * counters AND retires the scan-bitmap
					 * slot (so other CPUs stop re-finding
					 * this in-flight page), but KEEPS the
					 * TRACKED byte so install_local's early-
					 * out blocks any concurrent install from
					 * re-setting PG_lru while shrink_page_-
					 * list reclaims it. The TRACKED byte is
					 * wiped at the buddy handoff via
					 * marie_state_drop_pfn_at_free() (called
					 * from free_pages_prepare). Survivors
					 * keep TRACKED and re-publish a fresh
					 * scan slot + PG_lru in the putback loop
					 * below.
					 */
					marie_evict_counters_only(f);

					list_add(&f->lru, &page_list);
					n_taken_pages += compound_nr(f);
					accept_idx++;
				}
				n_taken = accept_idx;
			}

			if (!n_taken)
				break;

			/*
			 * PGSCAN accounting, mirroring upstream MGLRU's
			 * post-isolation bump (mm/vmscan.c evict_pages).
			 * n_taken is the count actually pulled off the LRU
			 * (the equivalent of MGLRU's `isolated`); upstream
			 * PGSCAN_* tracks isolated, not bitmap-scanned bits.
			 *
			 * NR_ISOLATED_ANON / _FILE must be bumped here so
			 * reclaim throttling and writeback congestion
			 * checks see Marie's in-flight isolation; the
			 * counter is decremented after shrink_page_list
			 * finishes (whether the page was reclaimed or put
			 * back).
			 */
			{
				mod_node_page_state(pgdat,
						    NR_ISOLATED_ANON + type,
						    n_taken_pages);
				marie_account_reclaim(lruvec, sc,
						PGSCAN_KSWAPD, PGSCAN_ANON,
						type, n_taken_pages);
			}

			n_reclaimed = marie_shrink_page_list(&page_list, pgdat,
							sc, &stat, ignore_refs,
							memcg);
			sc_add_reclaimed(sc, n_reclaimed);
			total_reclaimed += n_reclaimed;
			if (type < 2)
				atomic_long_add(n_reclaimed,
						&marie_dbg_reclaimed[type]);

			/*
			 * PGSTEAL accounting + matched NR_ISOLATED decrement.
			 * shrink_page_list has either freed each page or
			 * left it on @page_list for putback; either way the
			 * isolation window for these n_taken pages is over.
			 */
			{
				mod_node_page_state(pgdat,
						    NR_ISOLATED_ANON + type,
						    -n_taken_pages);
				marie_account_reclaim(lruvec, sc,
						PGSTEAL_KSWAPD, PGSTEAL_ANON,
						type, n_reclaimed);
			}

			oldest_for_putback =
				marie_find_oldest_occupied_mlv(type);
			if (oldest_for_putback >= 0)
				putback_gen = (u8)((oldest_for_putback + 1)
					& (MARIE_PFN_NR_GENS - 1));
			else
				putback_gen = (u8)atomic_read(
					&marie_head_gen[type]);

			list_for_each_entry_safe(f, tmp, &page_list, lru) {
				u8 prev, w, target_tier, gen;
				struct lruvec *lv;
				unsigned long pfn;
				int zone;
				enum lru_list inst_lru;

				pfn = page_to_pfn(f);
				/*
				 * prev_tier comes straight from the per-PFN
				 * byte: counters_only preserved it across
				 * isolate and the publish below has not run
				 * yet, so the byte still encodes the tier this
				 * page carried when it was isolated. (Replaces
				 * the old scratch_prev_tier[] capture + O(n^2)
				 * linear search back into scratch_batch.)
				 */
				if (pfn < marie_state_size)
					prev = (READ_ONCE(marie_state[pfn]) &
						MARIE_PFN_TIER_MASK) >>
					       MARIE_PFN_TIER_SHIFT;
				else
					prev = 0;
				/*
				 * Survivor placement signal: PG_active (hot) and
				 * PG_referenced (accessed since the last scan --
				 * the FOLIO_KEEP pages). NOT PG_workingset: that
				 * is the refault-on-ENTRY signal, consumed at
				 * install (marie_page_install) to give a
				 * refaulted page a protected re-entry; at putback
				 * the live access signal is referenced/active.
				 *
				 * A hot survivor (w != 0) is promoted to the head
				 * gen so it LEAVES the oldest (reclaim) gen rather
				 * than being re-isolated every round. A referenced
				 * file page stuck at the frontier is exactly what
				 * produces a perpetual non-IO FILE shortfall and
				 * the swappiness=1 anon nibble; head promotion
				 * drains the frontier so the FILE pass reaches the
				 * genuinely reclaimable (cold) file instead of
				 * conceding to anon. The 2nd-visit ignore_refs
				 * force-reclaim remains the backstop against a hot
				 * page escaping reclaim forever. A cold survivor
				 * stays at oldest+1 (putback_gen).
				 */
				/*
				 * (strong=active, weak=referenced) -> seed tier.
				 * seed is non-zero iff (active||referenced), so the
				 * gen = w ? head : putback_gen decision below is
				 * preserved unchanged across both tier-count splits.
				 */
				w = marie_seed_tier(PageActive(f),
						    PageReferenced(f));
				target_tier = prev > w ? prev : w;
				gen = w
					? (u8)atomic_read(&marie_head_gen[type])
					: putback_gen;

				list_del_init(&f->lru);
				lv = page_lruvec(f);
				zone = page_zonenum(f);
				/*
				 * Normalize PG_active->0 BEFORE computing inst_lru, mirroring
				 * marie_page_install() and marie_evict_locked(). The active hotness
				 * was already folded into target_tier (w) above, so nothing is lost.
				 * shrink_page_list's activate_locked path can leave PG_active set on a
				 * Marie-isolated page; crediting page_lru() with it still set
				 * lands the survivor's +nr in ACTIVE_*, but every debit path
				 * (marie_evict_locked / marie_evict_counters_only) clears PG_active
				 * first and debits INACTIVE_*. That producer/consumer bucket split is
				 * what underflows mz->lru_zone_size at the eventual free
				 * ("mem_cgroup_update_lru_size: lru_size -1").
				 */
				if (PageActive(f))
					ClearPageActive(f);
				inst_lru = page_lru(f);

				/*
				 * Survivor putback -- UNIFIED, global-only.
				 * marie_state[pfn] still has TRACKED set from
				 * before isolate (counters_only preserves it),
				 * so re-publish a FRESH scan slot at
				 * (putback_gen, target_tier) -- publish-only, no
				 * clear-old (isolate already retired the old
				 * slot). The page stays a Marie page; we do
				 * NOT route it back through putback_lru_page /
				 * lru_cache_add. That generic path re-enters the
				 * per-cpu folio_batch pipeline, which assumes
				 * legacy-LRU invariants (page on a real list,
				 * counted in mz->lru_zone_size) that Marie pages
				 * break -- under heavy pressure it freed
				 * still-dirty swapbacked pages out of the batch
				 * drain ("Bad page state").
				 */
				marie_state_publish_at_gen(pfn, gen,
							   target_tier);

				/* Account the survivor's re-installation. */
				marie_account_install_isolate(lv, f,
							      inst_lru, zone);

				if (!put_page_testzero(f)) {
					/*
					 * Isolation ref dropped, page still alive.
					 * Set PG_lru so the next scan can re-isolate
					 * it via TestClearPageLRU.
					 */
					SetPageLRU(f);
				} else {
					/*
					 * Isolation ref was the last one -- page is
					 * being freed now. PG_lru is clear (was cleared
					 * at isolation), so __put_page's
					 * __page_cache_release will not call
					 * del_page_from_lru_list and will not debit
					 * mz->lru_zone_size a second time -- isolation
					 * already debited it (the install +nr is settled
					 * by the isolate path), so a free-time debit here
					 * would underflow.
					 *
					 * shrink_page_list's activate_locked path may
					 * set PG_active on a page whose PG_lru is
					 * already clear (Marie isolated it). Normally
					 * PAGE_FLAGS_CHECK_AT_FREE is satisfied because
					 * activate_page() checks PG_lru and is a no-op
					 * when it is clear -- but some stock paths set
					 * PG_active directly (e.g. SetPageActive in
					 * the deactivate batch). Clear it here; the
					 * page has no live references and is not on any
					 * LRU list, so clearing PG_active is safe.
					 *
					 * Undo the putback counter increments before
					 * completing the free. Bitmaps and TRACKED are
					 * cleaned at buddy handoff by
					 * marie_state_drop_pfn_at_free.
					 */
					ClearPageActive(f);
					marie_account_evict_isolate(lv, f,
								    inst_lru,
								    zone);
					__put_page(f);
				}
			}

			/*
			 * No deferred drop pass: the scan-bitmap slot was
			 * retired at isolate (counters_only), and the TRACKED
			 * byte of a reclaimed page is wiped at its buddy
			 * handoff (marie_state_drop_pfn_at_free via the
			 * free_pages_prepare hook). Folios still alive in
			 * page_list went through the survivor putback above,
			 * which re-published a fresh scan slot via
			 * marie_state_publish_at_gen.
			 */

			/*
			 * Gen visited; loop to the next-oldest. The sweep ends
			 * by `break` (ring empty / nothing isolatable) or by the
			 * MARIE_PFN_NR_GENS bound, then falls to the per-type
			 * tail. `goto done` already left on target-reached.
			 */
			}

			/*
			 * Per-iteration tail.
			 *
			 * Bias controller update is skipped when:
			 *   - !attempted_pick: external override (skip_file
			 *     from clean_min_ratio) blocked the scan. The
			 *     bias must track actual picking policy, not
			 *     policy preempted before it ran.
			 *   - skip_file is in effect for THIS call: even
			 *     the ANON pick that succeeds during a
			 *     skip_file regime is happening only because
			 *     file was forcibly removed from contention.
			 *     Freezing the controller during the override
			 *     keeps the bias at its pre-override value, so
			 *     when file recovers above clean_min_ratio the
			 *     proportional regime resumes without an
			 *     overshoot driven by anon-only reclaim that
			 *     was never about the swappiness ratio.
			 *
			 * swappiness=1 (FILE_THEN_ANON) depletion-fallback
			 * gate (see the tail `if` below). Two independent
			 * reasons divert reclaim to ANON, on separate layers:
			 *
			 *   1. file < clean_min_ratio floor: handled UPFRONT by
			 *      marie_file_floor_protect -> skip_file -> pick
			 *      ANON_STRICT. Protects a minimum clean-file
			 *      reserve and never reaches here (skip_file
			 *      short-circuits FILE_THEN_ANON).
			 *
			 *   2. file >= floor but file reclaim cannot keep pace:
			 *      detected HERE by the FILE pass FAILING TO MEET
			 *      this call's reclaim target. A target-meeting FILE
			 *      pass exits via the tier loop's
			 *      sc_reclaim_target_reached() -> `goto done`, PAST
			 *      this tail; so merely arriving here means file fell
			 *      short. Occupancy/tier cannot tell reclaimability
			 *      or throughput apart -- a tracked file page may be
			 *      hot/dirty/mapped, and how much actually frees is
			 *      known only by trying (shrink_page_list). The
			 *      earlier gate keyed on the FILE pass returning
			 *      EXACTLY zero, which conflates "no reclaimable
			 *      file" with "file frees a positive trickle that
			 *      cannot match the allocation rate": while any
			 *      recyclable clean pagecache keeps cycling (refault /
			 *      IO refill) the FILE pass returns >0 forever, anon
			 *      is never scanned, and GBs of swappable anon OOM
			 *      with swap free. Sufficiency, not exact-zero, is the
			 *      correct depletion signal.
			 *
			 *      Because the FILE pass now SWEEPS the whole aged
			 *      gen ring before arriving here, merely arriving
			 *      means file fell short after reclaiming everything
			 *      reclaimable this call. The remedy depends only on
			 *      the floor: while clean file is still ABOVE
			 *      clean_min_ratio, DEFER -- successive file-only
			 *      calls drain it (forced ignore_refs makes even
			 *      referenced clean file reclaimable). Once file is at
			 *      the floor, CONCEDE to anon -- the file reserve is
			 *      protected and swapping anon is the OOM-with-swap-
			 *      free safety; the pick flips to ANON_STRICT next
			 *      call.
			 *
			 * `goto done` (target reached inside the tier loop)
			 * jumps PAST this tail intentionally: we are winning,
			 * the controller does not need a back-pressure tick.
			 */
			/*
			 * anon_unreclaimable forced FILE_STRICT above,
			 * bypassing the proportional controller; do not let
			 * those forced-file picks drive the bias (matches the
			 * "special swappiness values bypass the controller"
			 * rule -- the bias must resume cleanly once swap
			 * capacity returns and can_reclaim_anon flips back).
			 *
			 * The FILE_THEN_ANON depletion fallback (idx==1 ANON,
			 * reached only because the FILE pass found nothing
			 * reclaimable) is likewise a forced pick driven by file
			 * depletion, not by the swappiness ratio, so it must not
			 * drive the bias either.
			 */
			if (attempted_pick && !skip_file && !anon_unreclaimable &&
			    !(pick_kind == MARIE_PICK_FILE_THEN_ANON && idx == 1) &&
			    likely(!oom_victim))
				marie_swap_bias_update(type,
						       total_reclaimed, swappiness);
			if (likely(!oom_victim) &&
			    pick_kind == MARIE_PICK_FILE_THEN_ANON &&
			    idx == 0 && !skip_file) {
				/*
				 * swappiness=1 depletion fallback -- the FILE pass
				 * SWEPT the whole aged gen ring this call and still
				 * fell short (a target-meeting pass left via the tier
				 * loop's sc_reclaim_target_reached() -> `goto done`,
				 * PAST this gate). Decide anon purely on the floor:
				 *
				 *   file still ABOVE the clean_min_ratio floor ->
				 *   DEFER. swappiness=1 drains file to the floor before
				 *   anon; the per-call sweep is batch-capped, so a
				 *   large target is met across successive file-only
				 *   calls, not by conceding to anon while reclaimable
				 *   file sits above the floor (forced ignore_refs makes
				 *   even referenced clean file reclaimable, so this is
				 *   not a rotation treadmill). memcg-targeted reclaim
				 *   that made no file progress is the exception -- it
				 *   concedes so the limited memcg makes progress.
				 *
				 *   file at/below the floor -> CONCEDE to anon. The
				 *   file reserve is protected at the floor and swapping
				 *   anon is the OOM-with-swap-free safety. Reaching
				 *   FILE_THEN_ANON proved anon is reclaimable, so swap
				 *   capacity exists by construction.
				 *
				 *   node under acute free pressure (free <= 2*high) ->
				 *   CONCEDE regardless of the file level. The DEFER above
				 *   assumes file-only calls drain clean file to the floor;
				 *   a workload refilling clean file above the floor faster
				 *   than the batch-capped sweep drains it holds the floor
				 *   forever unreached, so this tail fires every time the
				 *   aged FILE ring depletes while free stays pinned at the
				 *   watermarks and GBs of swappable anon (incl. swap-backed
				 *   shmem) never reach swap -> premature OOM with swap free.
				 *   Once free is at the watermarks the file level is moot;
				 *   offer anon now. See marie_node_under_pressure().
				 *
				 * Equivalent original terms retained: floor_protect, and the
				 * memcg-targeted no-file-progress concede; the pressure term
				 * is the added early treadmill escape.
				 */
				bool floor  = marie_file_floor_protect(pgdat);
				bool freep  = marie_node_under_pressure(pgdat);
				bool refp   = marie_file_refaulting();
				bool memcgp = sc_cgroup_reclaim(sc) &&
					      !total_reclaimed;
				bool concede = floor || freep || refp || memcgp;

				if (!concede)
					goto done;

				/* Trigger attribution (priority floor>free>refault>memcg). */
				atomic_long_inc(&marie_dbg_concede[
					floor ? 0 : freep ? 1 : refp ? 2 : 3]);

				/*
				 * Engage the ANON pass NOW. If a final FILE batch
				 * happened to tip the target, the idx==1 ANON pass
				 * self-aborts at its own sc_reclaim_target_reached()
				 * gate, so no anon is over-reclaimed.
				 */
				drain_mask |= MARIE_DRAIN_ANON;
			}
		}
done:
		if (using_percpu)
			atomic_set(&buf->in_use, 0);
	}

	return drain_mask;
}


/* --- install / evict implementations --- */


static DEFINE_PER_CPU(int[ANON_AND_FILE], marie_drain_depth);

void marie_drain_enter_type(int type)
{
	this_cpu_inc(marie_drain_depth[type]);
}
void marie_drain_exit_type(int type)
{
	this_cpu_dec(marie_drain_depth[type]);
}
bool marie_in_drain_type(int type)
{
	return this_cpu_read(marie_drain_depth[type]) > 0;
}

/*
 * ---------------------------------------------------------------------
 *  Install / evict — direct synchronous transitions under lru_lock
 * ---------------------------------------------------------------------
 *
 * The per-PFN paradigm reduces every Marie page's state to a single
 * bit (TRACKED in marie_state[pfn]). There are exactly two state
 * transitions:
 *
 *   marie_page_install:   TRACKED 0 -> 1   (writes gen, tier, type,
 *                          zone, sets PG_lru, bumps counters; defined
 *                          below, declared in pfn_install.h)
 *   marie_evict_locked:    TRACKED 1 -> 0   (counter decrements +
 *                          per-PFN state wipe via marie_state_drop_pfn)
 *
 * Both are called with the caller's lru_lock irqsave held, so the
 * per-PFN byte write, the bitmap mutations, and the counter updates all
 * run in the same atomic context. PG_active hygiene and other
 * cross-cutting concerns are concentrated here.
 */


/*
 * marie_page_install - unified fresh install (TRACKED 0 -> 1).
 *
 * Replaces the former marie_install_local / marie_install_locked pair.
 * The two used to differ only in the order of (publish, account, flag)
 * and in the PG_lru set method; this canonical form picks set_mask_bits
 * (atomic PG_active clear + PG_lru set in one mask write) and the
 * publish -> flag -> account order from install_local.
 *
 * Call site:
 *   - lru_marie_add_page (THP under per-type lock, small page direct)
 *
 * Per-type lock is a property of the caller, not of this function: the
 * body only requires lru_lock + IRQs off and behaves identically whether
 * or not the caller additionally holds the per-type lock.
 *
 * Returns true on success, false on the "already TRACKED" early-out.
 * See pfn_install.h for the contract documentation.
 */
bool marie_page_install(struct page *page)
{
	struct lruvec *lv = page_lruvec(page);
	bool was_active, was_workingset;
	unsigned int tier;
	int type, zone;
	u8 head;
	enum lru_list inst_lru;
	unsigned long pfn;

	lockdep_assert_held(marie_lruvec_lock(lv));
	lockdep_assert_irqs_disabled();

	/*
	 * "Already TRACKED" early-out. A page reaching install while its
	 * per-PFN byte is still TRACKED is a Marie-owned, reclaim-isolated
	 * page (the deferred-teardown design preserves TRACKED while PG_lru
	 * is cleared) being re-added through a path that lacks a TRACKED gate
	 * -- e.g. lru_cache_add()/putback_lru_page() on an anon page that
	 * reclaim isolated into the swap cache and a fault then swaps back in.
	 * Re-installing would re-set PG_lru and double-count Marie's counters;
	 * the resurrected PG_lru then survives onto the buddy free path and
	 * trips "Bad page state |lru|" PAGE_FLAGS_CHECK_AT_FREE. Bail so the
	 * in-flight reclaim retains ownership.
	 *
	 * Return TRUE, not false: returning false tells add_page_to_lru_list() to
	 * run its LEGACY fallback (update_lru_size(+nr) + list_add onto a real
	 * lruvec->lists[lru]) on a page that is STILL TRACKED and that Marie
	 * never credited to mz->lru_zone_size. That stray, never-debited mz
	 * credit + a page cross-linked onto a legacy list is exactly the
	 * mz->lru_zone_size underflow ("lru_size -1") we were chasing. TRUE
	 * means "Marie owns it, do not add anywhere" -- which is what "retain
	 * ownership" requires.
	 */
	pfn = page_to_pfn(page);
	if (pfn < marie_state_size &&
	    (READ_ONCE(marie_state[pfn]) & MARIE_PFN_TRACKED))
		return true;

	/*
	 * Workingset signal capture: (PG_active, PG_workingset) -> tier.
	 *   (0,0) tier 0  cold
	 *   (0,1) tier 1  workingset, distance too large
	 *   (1,0) tier 2  recent refault, never workingset before
	 *   (1,1) tier 3 = MARIE_PFN_TIER_MAX  established hot
	 * Read PG_active BEFORE clearing so the captured tier matches the
	 * byte we publish below. PG_workingset stays set:
	 * workingset_eviction's shadow encoding needs it at next eviction.
	 */
	was_active = PageActive(page);
	was_workingset = PageWorkingset(page);
	/* (strong=active, weak=workingset) -> seed tier; agnostic to tier count. */
	tier = marie_seed_tier(was_active, was_workingset);

	if (was_active)
		ClearPageActive(page);

	/*
	 * page->lru MUST be re-initialised here. A recycled page arrives
	 * with LIST_POISON{1,2} from the prior owner's list_del, and the
	 * eventual marie_evict_locked's list_del_init would walk the
	 * poison pointers and fault.
	 */
	INIT_LIST_HEAD(&page->lru);

	type = page_is_file_cache(page);
	zone = page_zonenum(page);
	/* Install into THIS lruvec's own youngest (head) gen. */
	head = (u8)atomic_read(&marie_head_gen[type]);

	/*
	 * Publish per-PFN state byte + scan bitmap + memcg L1 + the global
	 * gen_occupied++. See pfn_install.h::marie_pfn_publish_inherit.
	 */
	marie_pfn_publish_inherit(page, type, head, (u8)tier, zone);
	/*
	 * Install-cadence aging (global). Count installs onto the head gen and
	 * seal the generation once it has accumulated marie_gen_growth_live[type]
	 * pages, advancing the head so subsequent installs land in a fresh
	 * younger gen. This stratifies pages by age PROACTIVELY so the oldest
	 * gen holds genuinely-old pages and reclaim does not waste scans
	 * rotating hot ones. Counter is global/atomic (installs run under
	 * different per-lruvec lru_locks); reset only on a real advance, so a
	 * blocked attempt (next slot still draining) retries on the next install.
	 *
	 * Count PAGES, not pages: a large page (THP) deposits compound_nr
	 * pages onto the head gen in one install, so the head must advance in
	 * proportion. Incrementing by 1 per page advanced the head up to
	 * 512x too slowly under THP=always -- the entire anon set piled into the
	 * head gen, reclaim (which scans the aged gens) found nothing to isolate,
	 * and the node went all_unreclaimable with free swap: premature OOM and a
	 * multi-CPU lru_lock livelock (tens-of-seconds GUI freeze). order-0 is
	 * unaffected (compound_nr == 1).
	 */
	if (atomic_long_add_return(compound_nr(page),
				   &marie_gen_installs[type]) >=
	    READ_ONCE(marie_gen_growth_live[type])) {
		if (marie_try_advance_head_mlv(type))
			atomic_long_set(&marie_gen_installs[type], 0);
	}

	/*
	 * Atomic PG_active->0 + PG_lru->1 in one mask write. PG_active was
	 * cleared above when set; the mask write keeps the invariant
	 * against the defensive case where another path set PG_active
	 * between then and now. Ordered AFTER the state-byte publish so a
	 * concurrent __page_cache_release observing PG_lru=1 also observes
	 * marie_state[pfn] & MARIE_PFN_TRACKED.
	 */
	set_mask_bits(&MARIE_FOLIO_FLAGS(page), BIT(PG_active), BIT(PG_lru));
	inst_lru = page_lru(page);

	marie_account_install(lv, page, inst_lru, zone);

	return true;
}

bool marie_evict_locked(struct page *page)
{
	struct lruvec *lv = page_lruvec(page);
	int zone = page_zonenum(page);

	lockdep_assert_held(marie_lruvec_lock(lv));
	lockdep_assert_irqs_disabled();

	/*
	 * page->lru is either a self-loop (install/flush leave it that
	 * way, and the per-PFN paradigm never re-attaches it onto a
	 * Marie-owned list) or on legacy lruvec->lists[lru] after a
	 * drain handed it off. list_del_init is a no-op in the first
	 * case and a legacy-list removal in the second; the caller
	 * holds lruvec->lru_lock for the latter, so no extra Marie-side
	 * lock is required.
	 */
	list_del_init(&page->lru);

	/*
	 * PG_active hygiene MUST happen before page_lru() below.
	 * The install helper clears PG_active and then computes the lru
	 * index, so install always credits INACTIVE_*. If we read
	 * page_lru() here while PG_active is still set (e.g. via
	 * activate_page() on a tracked page between install and del),
	 * we would decrement ACTIVE_* -- an LRU index Marie's install
	 * never +1'd -- and trip the mz->lru_zone_size underflow WARN.
	 * Mirror install's order: clear PG_active, then compute lru.
	 *
	 * Also drops PG_active for shrink_page_list, which trips
	 * VM_BUG_ON_PAGE(PageActive) otherwise.
	 */
	if (PageActive(page))
		ClearPageActive(page);

	marie_account_evict(lv, page, page_lru(page), zone);

	/*
	 * Clear PG_lru BEFORE marie_state_drop_pfn so a concurrent
	 * del-side path gated on TestClearPageLRU cannot observe
	 * (state=TRACKED, PG_lru=1) -> Marie del again recursion.
	 * drop_pfn then wipes the per-PFN state (byte, bitmap,
	 * l2_range_count, memcg L1) which is the only Marie tracking
	 * for this page.
	 *
	 * Idempotent for callers that already cleared PG_lru via
	 * TestClearPageLRU before reaching evict
	 * (__page_cache_release, marie_state_shrink_lruvec claim loop).
	 */
	ClearPageLRU(page);
	marie_state_drop_pfn(page);

	return true;
}

/*
 * marie_evict_counters_only - reclaim-isolate per-page counter decrement
 * that also retires the scan-bitmap slot, but PRESERVES marie_state[]'s
 * TRACKED bit.
 *
 * The per-PFN state byte staying TRACKED throughout shrink_page_list is
 * the race defence: marie_page_install's "already TRACKED" early-out
 * makes a concurrent install on this PFN bail, so install cannot set
 * PG_lru on the page while shrink_page_list is reclaiming it. (The
 * earlier full marie_evict_isolated cleared TRACKED inline; a concurrent
 * install would then succeed, set PG_lru, and trip
 * PAGE_FLAGS_CHECK_AT_FREE at free_unref_page_list in the success path.)
 *
 * The global (type, gen, tier) bitmap bit + gen_occupied slot ARE dropped
 * here, at isolate. The bit is the scanner's candidate index, and an
 * isolated page is no longer a candidate: leaving it set lets every
 * other CPU's scanner re-find the same in-flight PFN for the whole
 * swap-out window (the claim fails on the already-cleared PG_lru, but the
 * re-scan / re-batch work is pure waste, and a page shrink_page_list
 * chose to KEEP can get re-isolated before its second chance is honoured
 * -> avoidable refaults). Retiring the scan slot here while keeping the
 * TRACKED byte separates "is a scan candidate" (bitmap) from "blocks a
 * concurrent install" (byte). l2_count / gen_occupied stay balanced 1:1:
 * the matching set is the install; the matching re-set, for a survivor,
 * is marie_state_publish_at_gen at putback; a reclaimed page's byte is
 * wiped at the buddy free hook, which finds the bit already clear.
 *
 * Caller-side gates that hold throughout this path:
 *   1. get_page_unless_zero()        - reference held, page cannot be freed.
 *   2. TestClearPageLRU() - PG_lru cleared atomically, gating
 *                                external del paths.
 *   3. install_local TRACKED early-out (above)
 *
 * memcg_bitmap is cleared here because the buddy free hook
 * (marie_state_drop_pfn_at_free) runs without a page reference and cannot
 * derive memcg later.
 *
 * Counters are decremented immediately so the in-flight page does not
 * inflate lruvec_lru_size() and skew reclaim pressure heuristics during
 * shrink_page_list. The scan bitmap + gen_occupied are torn down HERE so
 * the in-flight page leaves the candidate index immediately; only the
 * TRACKED byte teardown is deferred (to the buddy free hook for reclaimed
 * pages). Survivors go through the putback path, which re-publishes a
 * fresh scan slot via marie_state_publish_at_gen and re-sets PG_lru.
 */
void marie_evict_counters_only(struct page *page)
{
	struct lruvec *lv = page_lruvec(page);
	int zone = page_zonenum(page);
	enum lru_list del_lru;

	if (unlikely(!list_empty(&page->lru))) {
		/*
		 * Defensive: an mm/swap.c batch path lacking a Marie gate
		 * may have placed this page onto a legacy lruvec list via
		 * add_page_to_lru_list_tail. The caller's list_add(&f->lru, ...)
		 * would then corrupt that list. Detach under lru_lock first;
		 * DO NOT fall back to lru_marie_del_page (it would clear
		 * TRACKED via marie_state_drop_pfn, breaking the deferred-
		 * teardown invariant the putback path relies on).
		 */
		VM_WARN_ON_ONCE_PAGE(1, page);
		scoped_guard(spinlock_irq, marie_lruvec_lock(lv))
			list_del_init(&page->lru);
	}

	if (PageActive(page))
		ClearPageActive(page);

	del_lru = page_lru(page);

	/*
	 * marie_account_evict_isolate owns the local_irq_save/restore that
	 * the lock-free reclaim path needs against same-CPU softirq
	 * reentrancy on fbc->lock and the per-CPU vmstat diff (see the
	 * helper's contract in account.h, and 9c6a93782's lockup history).
	 */
	marie_account_evict_isolate(lv, page, del_lru, zone);

	/*
	 * Retire the scan-bitmap slot + gen_occupied at isolate (see the
	 * function comment). Read the still-TRACKED byte for its (gen, tier,
	 * type) coordinate; the byte itself is left TRACKED for the install-
	 * race early-out. These bit ops are atomic and need no IRQ-off
	 * window; the helper's local_irq_save/restore is scoped to the
	 * counters that actually need it.
	 */
	{
		unsigned long pfn = page_to_pfn(page);

		if (pfn < marie_state_size) {
			u8 s = READ_ONCE(marie_state[pfn]);

			if (s & MARIE_PFN_TRACKED) {
				u8 g = (s & MARIE_PFN_GEN_MASK) >>
				       MARIE_PFN_GEN_SHIFT;
				u8 tr = (s & MARIE_PFN_TIER_MASK) >>
					MARIE_PFN_TIER_SHIFT;
				u8 tb = (s & MARIE_PFN_TYPE_MASK) ? 1 : 0;

				if (marie_bm_clear(&marie_track_bm[tb][g][tr], pfn))
					marie_gen_occ_dec(pfn, g, tb);
			}
		}
	}

}

/*
 * Bumps the per-PFN tier; marie_state_inc_tier handles both the
 * non-saturated bump (WRITE_ONCE) and the saturated promote
 * (marie_state_move_to_gen to head_gen + tier 0) internally.
 */
void lru_marie_mark_accessed(struct page *page)
{
	unsigned long pfn = page_to_pfn(page);
	u8 state;

	if (!lru_marie_enabled() || !marie_state_ready())
		return;
	if (pfn >= marie_state_size)
		return;
	state = READ_ONCE(marie_state[pfn]);
	if (!(state & MARIE_PFN_TRACKED))
		return;

	/* Bump the access tier toward MAX (hotter). Lock-free fault/hit path;
	 * the saturated promote rolls to the global head gen. */
	marie_state_inc_tier(pfn);
	/* Mark the page as recently accessed for the workingset estimator. */
	if (TestClearPageReferenced(page))
		SetPageWorkingset(page);
}
EXPORT_SYMBOL_GPL(lru_marie_mark_accessed);

/*
 * Per-cpu folio_batch LRU-op hooks (declared in <linux/lru_marie.h>).
 * Each applies the op directly on the page's per-PFN state and returns
 * true so mm/swap.c skips the legacy folio_batch; false (Marie off / page
 * untracked) falls through to the legacy path. All run lock-free:
 * marie_state_move_to_gen is CAS-based and its bitmap ops are atomic,
 * matching the no-lru_lock contract of these entry points.
 */

/*
 * Demote: relocate to the oldest live gen at tier 0 so Marie's next scan
 * reclaims it promptly. Used for the EXPLICIT user "make cold" madvise
 * (MADV_COLD -> deactivate_page / deactivate_file_page). Reclaim-internal
 * hints (activate / rotate) deliberately do NOT demote -- see those hooks.
 */
static bool marie_page_demote(struct page *page)
{
	int type, oldest;

	if (!lru_marie_enabled() || !page_marie_test_tracked(page))
		return false;
	type = page_is_file_cache(page);
	oldest = marie_find_oldest_occupied_mlv(type);
	if (oldest >= 0)
		marie_state_move_to_gen(page_to_pfn(page), (u8)oldest, 0);
	return true;
}

bool lru_marie_deactivate(struct page *page)
{
	return marie_page_demote(page);
}
EXPORT_SYMBOL_GPL(lru_marie_deactivate);

/*
 * rotate: NO-OP for Marie pages (skip the legacy batch). Like activate
 * this is a reclaim-internal hint (rotate_reclaimable_page fires on
 * writeback completion of a PG_reclaim page). An actively reclaimed Marie
 * page is isolated (PG_lru cleared) so this is rarely reached, and Marie's
 * gen aging already orders reclaim -- no per-PFN state change is wanted.
 */
bool lru_marie_rotate(struct page *page)
{
	return lru_marie_enabled() && page_marie_test_tracked(page);
}
EXPORT_SYMBOL_GPL(lru_marie_rotate);

/*
 * activate: NO-OP for Marie pages (but skip the legacy batch by returning
 * true). activate_page is driven mostly by shrink_page_list's
 * FOLIOREF_ACTIVATE during reclaim, and Marie already decides retention
 * there via its tier vote in page_check_references. Promoting to the head
 * gen on top would pull referenced pages out of the oldest gen on every
 * reclaim pass; under an all-hot workload that starves reclaim entirely
 * (OOM with GBs of unreclaimable inactive_anon). The explicit-access
 * channel is mark_page_accessed -> lru_marie_mark_accessed (tier bump),
 * which must not be double-counted here.
 */
bool lru_marie_activate(struct page *page)
{
	return lru_marie_enabled() && page_marie_test_tracked(page);
}
EXPORT_SYMBOL_GPL(lru_marie_activate);

/*
 * MADV_FREE: make the anon page reclaim-without-writeback. Clear the
 * dirtiness signals synchronously (what the legacy lru_lazyfree move_fn
 * does) and demote so Marie frees it promptly without swap on the next
 * scan. type is read before clearing swapbacked (page_is_file_cache flips
 * once swapbacked is gone); the Marie byte keeps its anon TYPE, so demote
 * stays within the anon gen ring.
 */
bool lru_marie_lazyfree(struct page *page)
{
	int type, oldest;

	if (!lru_marie_enabled() || !page_marie_test_tracked(page))
		return false;
	type = page_is_file_cache(page);
	ClearPageActive(page);
	ClearPageReferenced(page);
	ClearPageSwapBacked(page);
	count_vm_events(PGLAZYFREE, compound_nr(page));
	oldest = marie_find_oldest_occupied_mlv(type);
	if (oldest >= 0)
		marie_state_move_to_gen(page_to_pfn(page), (u8)oldest, 0);
	return true;
}
EXPORT_SYMBOL_GPL(lru_marie_lazyfree);

/*
 * page_marie_get_tier (public API in <linux/lru_marie.h>): returns the
 * page's tier, or 0 when Marie is off, the PFN is out of range, or the
 * page is untracked.
 */
unsigned int page_marie_get_tier(const struct page *page)
{
	unsigned long pfn = page_to_pfn((struct page *)page);
	u8 state;

	if (!marie_state || pfn >= marie_state_size)
		return 0;
	state = READ_ONCE(marie_state[pfn]);
	if (!(state & MARIE_PFN_TRACKED))
		return 0;
	return (state & MARIE_PFN_TIER_MASK) >> MARIE_PFN_TIER_SHIFT;
}
EXPORT_SYMBOL_GPL(page_marie_get_tier);

/*
 * lru_marie_test_tracked (public API in <linux/lru_marie.h>).
 */
bool lru_marie_test_tracked(const struct page *page)
{
	return page_marie_test_tracked(page);
}
EXPORT_SYMBOL_GPL(lru_marie_test_tracked);

/*
 * lru_marie_free_page_hook (public API in <linux/lru_marie.h>).
 * Thin wrapper over marie_state_drop_pfn_at_free so the page allocator
 * can call the hook without including the private state.h.
 */
void lru_marie_free_page_hook(unsigned long pfn)
{
	marie_state_drop_pfn_at_free(pfn);
}
EXPORT_SYMBOL_GPL(lru_marie_free_page_hook);

/*
 * marie_del_page_locked - lru_marie_del_page body.
 *
 * External-removal entry: if the page is still Marie-tracked, do the
 * full evict via marie_evict_locked, which routes through
 * marie_account_evict and owns the ENTIRE counter wind-down -- including
 * the single marie_nr_pages -1. The caller does no accounting of its
 * own; an earlier caller-side -1 predated the account.h funnel and
 * double-counted marie_nr_pages on every generic del of a tracked page.
 *
 * Lock contract: caller holds lruvec->lru_lock. No Marie lock is taken
 * here -- the lru_lock invariant already serialises every Marie state
 * mutation. See the comment above the call site in lru_marie_del_page
 * for the full protection-model rationale.
 *
 * Returning true tells the dispatcher (del_page_from_lru_list in
 * include/linux/mm_inline.h) "Marie owns this page, do not fall
 * through to legacy".
 *
 * The not-tracked branch returns true defensively. Under the lru_lock
 * invariant it is unreachable -- the caller's TRACKED fast-path test
 * already gated entry here -- but returning true keeps the safe
 * behaviour if the invariant ever regresses: a stray legacy
 * update_lru_size on a page Marie already accounted would double-
 * decrement mz->lru_zone_size.
 */
bool marie_del_page_locked(struct page *page)
{
	lockdep_assert_held(marie_lruvec_lock(page_lruvec(page)));
	lockdep_assert_irqs_disabled();

	if (!page_marie_test_tracked(page))
		return true;
	return marie_evict_locked(page);
}

/*
 * ---------------------------------------------------------------------
 *  global init
 * ---------------------------------------------------------------------
 */

/*
 * Global per-type locks.  Marie is desktop/global-only: there is a single
 * aging clock and a single global track bitmap, so the per-type serialising
 * lock is one global instance per type (anon / file), not one per lruvec.
 * The only holders are the THP-install and split-tail paths in core.c.
 */
struct marie_type marie_type_locks[ANON_AND_FILE];

/* marie_type_init: caller-side scalar/lock initialisation only. */
static void marie_type_init(struct marie_type *t, int type)
{
	spin_lock_init(&t->type_lock);
	t->type = type;
}

int marie_counters_init(void)
{
	int t;

	for (t = 0; t < ANON_AND_FILE; t++)
		marie_type_init(&marie_type_locks[t], t);

	return percpu_counter_init(&marie_nr_pages, 0, GFP_KERNEL);
}
