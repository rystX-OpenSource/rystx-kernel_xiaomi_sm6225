/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _MM_LRU_MARIE_STATE_H
#define _MM_LRU_MARIE_STATE_H

#include <linux/percpu_counter.h>
#include "bitmap.h"	/* struct marie_bitmap, MARIE_L2_BITS, marie_bm_* */
#include "defrag.h"		/* marie_defrag_hist_inc/dec -- per-pageblock occupancy mirror */

/* Swappiness/pick diagnostics, surfaced in /sys/kernel/mm/lru_marie/stats. */
extern atomic_long_t marie_dbg_pick[5];
extern atomic_long_t marie_dbg_reclaimed[2];

/*
 * Concede-trigger breakdown (surfaced in stats): which term engaged the
 * FILE_THEN_ANON concede-to-anon -- the clean_min_ratio floor, free-level
 * pressure (marie_node_under_pressure), file-refault feedback
 * (marie_file_refaulting), or a memcg no-file-progress pass.
 */
extern atomic_long_t marie_dbg_concede[4];	/* [floor,free,refault,memcg] */

/* Orphaned scan-bit self-heal count, per type [0]=anon [1]=file. */
extern atomic_long_t marie_dbg_orphan_bit[2];

/*
 * Marie per-PFN state array — paradigm specification.
 * ======================================================
 *
 * Marie represents every page's reclaim state as a single byte in
 * a flat per-PFN array allocated once at boot. Each Marie operation
 * on a page is a single byte read or write at marie_state[pfn] —
 * there is no allocation anywhere in the fault / del / aging fast
 * paths, no linked-list traversal, no per-CPU staging.
 *
 * The array is sized once at boot to cover totalram_pages PFNs
 * (~4 MB on a 16 GiB box, ~16 MB on 64 GiB; the same scale as a
 * 1/64-th miniature struct page) and never grows or shrinks. The
 * 32-bit PFN gate (marie_init's MARIE_MAX_SUPPORTED_PFN check)
 * caps the worst-case array size at 4 GiB.
 *
 *
 * Byte layout
 * -----------
 *
 * Byte layout (8 gens x 2 tiers; see the field-width defines below for the
 * rationale):
 *
 *   bit 7     TRACKED      1 = page is owned by Marie; 0 = ignore byte
 *   bit 6     TYPE         1 = file LRU, 0 = anon LRU
 *   bit 5..4  ZONE         page_zonenum: 0=DMA, 1=DMA32, 2=NORMAL, 3=MOVABLE
 *   bit 3..1  GEN          relative-position 0..7 in the cycling ring
 *                          (0 = oldest, head = atomic_read(&marie_head_gen[type]))
 *   bit 0     TIER         0 = cold, 1 = referenced/active (any hotness signal)
 *
 * The 8 gens give a longer head->oldest descent (time-domain grace), which
 * measurably protects warm/marginally-hot pages: their hotness signal has
 * more time to re-promote them to the head before they reach reclaim range
 * (QEMU warm-set A/B: ~40% fewer refaults vs the former 4-gen x 4-tier split;
 * the 2-tier recency loss was measured neutral).
 *
 * The 8 bits saturate the byte. Bits are laid out in reclaim filter
 * hierarchy from MSB (root: existence) down to LSB (leaf: hotness),
 * so the isolate scan can extend its (s & mask) == target test by
 * widening @mask from the top:
 *
 *   (byte == 0)            -> untracked (single-cycle skip)
 *   (byte & 0x80)          -> TRACKED
 *   (byte & 0xC0)          -> TRACKED + type
 *   (byte & 0xF0)          -> TRACKED + type + zone
 *   (byte & 0xFE)          -> TRACKED + type + zone + gen
 *   (byte & 0xFF)          -> all five dimensions
 *
 * The whole filter is a pure byte mask + compare with no pfn_to_page()
 * dereference required to make a candidate / skip decision — the
 * inner loop scales to AVX-512 vpand+vpcmpeqb at 64 byte per cycle
 * and reserves the struct page touch for confirmed candidates only.
 *
 * The zone field truncates to 2 bits. ZONE_DEVICE (when enabled)
 * never reaches Marie because the dispatcher gates on regular LRU
 * pages; ZONE_HIGHMEM is 32-bit-only and excluded by the 32-bit
 * PFN gate. So the 4 zone codes cover every Marie-tracked page in
 * practice.
 *
 * Untracked PFNs read as 0. The TRACKED bit is the single source of
 * truth — no separate page->flags Marie bit is used.
 *
 *
 * Aging — gen ring as a cycling counter (per type)
 * ------------------------------------------------
 *
 *   atomic_t marie_head_gen[ANON_AND_FILE];           // 0..NR_GENS-1 cycling per type
 *   atomic_long_t marie_gen_installs[MARIE_PFN_NR_GENS][ANON_AND_FILE];
 *   atomic_long_t marie_gen_occupied[MARIE_PFN_NR_GENS][ANON_AND_FILE];
 *
 * install:
 *
 *   u8 gen = atomic_read(&marie_head_gen[type]);
 *   marie_state[pfn] = MARIE_PFN_TRACKED | (type<<6) | (zone<<4) |
 *                      (gen<<MARIE_PFN_GEN_SHIFT) | tier;
 *   set_bit(pfn, marie_gen_bitmap[gen][type]);
 *   atomic_long_inc(&marie_gen_installs[gen][type]);
 *   atomic_long_inc(&marie_gen_occupied[gen][type]);
 *
 * head_gen advance is global and per-type, drain-wait gated (next gen
 * empty for that type), and fired by install cadence alone: a single
 * global per-type counter (marie_gen_installs[type]) counts installs onto
 * the head gen and advances at marie_gen_growth_live[type]
 * (marie_page_install -> marie_try_advance_head_mlv, under lru_lock). The
 * former reclaim-time "occupied < 2 at shrink entry" trigger was removed --
 * under concurrent global reclaim it thrashed the ring and collapsed age
 * stratification.
 *
 *
 * Del — single byte zero
 * ----------------------
 *
 *   marie_state[pfn] = 0;
 *
 * No swap-pop, no list_del, no shard lock dance. External del
 * (lru_marie_del_page from compaction, put_page, munmap) is the
 * same single store.
 *
 *
 * Isolate — cursor + SIMD scan
 * ----------------------------
 *
 * Per-pgdat scan cursor walks the array; SIMD reads 64 byte / cycle
 * (AVX-512) and tests for (TRACKED && gen == oldest && tier == 0)
 * via a single AND + CMP mask. Cursor saves position across calls so
 * batch-32 isolate typically scans only a few hundred PFNs.
 *
 *   for (pfn = cursor; n_batch < batch; pfn = next_or_wrap(pfn)) {
 *       u8 s = marie_state[pfn];
 *       if ((s & MARIE_PFN_FILTER) != MARIE_PFN_TARGET)
 *           continue;
 *       batch[n_batch++] = pfn_to_page(pfn);
 *   }
 *   cursor = pfn;
 *
 * Worst-case (sparse) full sweep of the 4 MB array is ~0.5 ms at
 * DRAM bandwidth, ~50 µs in L3. Cursor amortises across many
 * batches, so typical batch cost is sub-µs.
 *
 *
 * memcg scope
 * -----------
 *
 * The array is global (single allocation system-wide), not per-memcg.
 * Marie is desktop/global-only: there is no per-memcg reclaim and the
 * scan covers every Marie page (no per-memcg filter). This trades
 * per-memcg locality for vastly simpler data structures — desktop and
 * small-server cgroup trees (where Marie targets) are dominated by the
 * root memcg anyway, so the locality loss is small in practice.
 *
 *
 * Walker integration
 * ------------------
 *
 * The PTE walker (marie_walker) inspects young bits as before but
 * commits tier bumps to marie_state[pfn] instead of page->flags.
 * The same SIMD young-pte machinery from the prior implementation
 * carries over unchanged.
 *
 *
 * Disable
 * -------
 *
 * Marie disable (boot-only configuration): write 0 to every TRACKED
 * byte via SIMD bulk store, put_page each one. An O(N) sweep but it
 * happens rarely. (Memcg reparent is gone -- desktop/global-only.)
 *
 *
 * Sizing & init
 * -------------
 *
 * marie_state is kvmalloc'd at subsys_initcall with size
 * `max_pfn` bytes. max_pfn is bounded by the 32-bit PFN gate
 * (marie_init's MARIE_MAX_SUPPORTED_PFN check), so the array is at
 * most 4 GiB on the maximum supported config. Realistic sizings:
 *
 *   16 GiB RAM  ->  4 MiB   (single kvmalloc, contiguous in vmalloc)
 *   64 GiB RAM  -> 16 MiB
 *  256 GiB RAM  -> 64 MiB
 *
 * The array is sparse-tolerant: NUMA holes and reserved regions read
 * as 0 (untracked) and incur only sequential-read cost during scan.
 */

/*
 * Field shifts and masks within each marie_state[] byte. Ordered
 * MSB -> LSB by reclaim filter hierarchy: TRACKED, TYPE, ZONE, GEN,
 * TIER. See the byte-layout block above for the rationale.
 */
/*
 * gen:tier split of the per-PFN byte's low nibble: 8 gens (3 bits) x 2 tiers
 * (1 bit). The 8 gens lengthen the head->oldest descent (time-domain grace) so a
 * warm page's hotness signal has time to re-promote it before it reaches reclaim
 * range; a QEMU warm-set A/B (cold filler + a working set re-accessed near the
 * grace period) measured ~40% fewer warm-page refaults than the former 4-gens-x-
 * 4-tiers split, and a 4-gens-x-2-tiers control did NOT improve -- isolating the
 * gain to the gen count, not the (coarser) 2-tier promotion threshold. GEN_SHIFT/
 * ZONE_SHIFT derive from the field widths; ZONE/TYPE/TRACKED are unaffected.
 */
#define MARIE_PFN_TIER_SHIFT		0
#define MARIE_PFN_TIER_BITS		1
#define MARIE_PFN_GEN_BITS		3
#define MARIE_PFN_TIER_MASK		(((1U << MARIE_PFN_TIER_BITS) - 1) << \
					 MARIE_PFN_TIER_SHIFT)
#define MARIE_PFN_NR_TIERS		(1U << MARIE_PFN_TIER_BITS)
#define MARIE_PFN_TIER_MAX		(MARIE_PFN_NR_TIERS - 1)

#define MARIE_PFN_GEN_SHIFT		(MARIE_PFN_TIER_SHIFT + MARIE_PFN_TIER_BITS)
#define MARIE_PFN_GEN_MASK		(((1U << MARIE_PFN_GEN_BITS) - 1) << \
					 MARIE_PFN_GEN_SHIFT)
#define MARIE_PFN_NR_GENS		(1U << MARIE_PFN_GEN_BITS)

#define MARIE_PFN_ZONE_SHIFT		(MARIE_PFN_GEN_SHIFT + MARIE_PFN_GEN_BITS)
#define MARIE_PFN_ZONE_BITS		2
#define MARIE_PFN_ZONE_MASK		(((1U << MARIE_PFN_ZONE_BITS) - 1) << \
					 MARIE_PFN_ZONE_SHIFT)
#define MARIE_PFN_NR_ZONES_ENCODED	(1U << MARIE_PFN_ZONE_BITS)

#define MARIE_PFN_TYPE_SHIFT		6
#define MARIE_PFN_TYPE_FILE		(1U << MARIE_PFN_TYPE_SHIFT)
#define MARIE_PFN_TYPE_MASK		MARIE_PFN_TYPE_FILE

#define MARIE_PFN_TRACKED_SHIFT		7
#define MARIE_PFN_TRACKED		(1U << MARIE_PFN_TRACKED_SHIFT)

/*
 * Encode @zone (page_zonenum result) into the byte's zone nibble.
 * Truncates to MARIE_PFN_NR_ZONES_ENCODED-1 so ZONE_DEVICE etc. do
 * not overflow the 2-bit field; in practice those zones do not
 * reach Marie's install path.
 */
static inline u8 marie_pfn_zone_bits(unsigned int zone)
{
	return (u8)((zone & (MARIE_PFN_NR_ZONES_ENCODED - 1)) <<
		    MARIE_PFN_ZONE_SHIFT);
}

/*
 * Map a (strong, weak) hotness signal pair to a seed tier (1-bit tier: any
 * hotness signal -> protected tier 1, none -> cold tier 0).
 *   install:  strong = PG_active,  weak = PG_workingset (refault-on-entry)
 *   putback:  strong = PG_active,  weak = PG_referenced (accessed-since-scan)
 * The result is non-zero iff (strong||weak), so the putback path reuses it for
 * its head-promotion decision (gen = seed ? head : putback_gen).
 */
static inline u8 marie_seed_tier(bool strong, bool weak)
{
	return (strong || weak) ? 1 : 0;
}

/* The base allocation (subsys_initcall) covers totalram_pages PFNs. */
extern u8 *marie_state;
extern unsigned long marie_state_size;

/*
 * Per-(gen, type) live page count, kept as a node-wide population
 * aggregate by the marie_gen_occ_inc/dec abstraction (state.c). Aging is
 * global now (single per-type clock: marie_head_gen / marie_gen_occupied);
 * this sum is the global gen occupancy and also serves as the "is any anon
 * tracked?" signal for marie_file_floor_protect.
 */
extern atomic_long_t marie_gen_occupied[MARIE_PFN_NR_GENS][2 /* ANON_AND_FILE */];

/*
 * gen_occupancy abstraction -- the SINGLE interface for "a page entered /
 * left gen @g of @type". EVERY per-PFN gen-transition site routes its
 * occupancy bookkeeping through here: the install/split publisher
 * (marie_pfn_publish_inherit, pfn_install.h) and the state.c transitions
 * (publish_at_gen, move_to_gen, inc_tier, drop_pfn, drop_pfn_at_free,
 * evict_counters_only, uncharge_backstop). Keeping it a single choke-point
 * means a new transition site cannot forget to update the counter -- and
 * gives the per-pageblock occupancy mirror (Marie defrag compaction, marie_defrag_hist_inc/dec)
 * one hook to tap, with Sigma_blocks(mirror) == Sigma(marie_gen_occupied) as
 * its completeness invariant. @pfn is threaded through purely so the mirror
 * can locate the page's pageblock; the marie_gen_occupied bump ignores it.
 * When CONFIG_LRU_MARIE_DEFRAG=n the marie_defrag hooks are empty inlines and the
 * emitted code is identical to a bare atomic_long_inc/dec.
 *
 * static inline in the header (was state.c-private) precisely so the
 * install-side publisher in pfn_install.h shares this path instead of
 * bumping marie_gen_occupied directly. Desktop/global-only: a single global
 * per-(gen, type) counter, so these take no carrier.
 */
static inline void marie_gen_occ_inc(unsigned long pfn, int gen, int type)
{
	atomic_long_inc(&marie_gen_occupied[gen][type]);
	marie_defrag_hist_inc(pfn, gen, type);
}

static inline void marie_gen_occ_dec(unsigned long pfn, int gen, int type)
{
	atomic_long_dec(&marie_gen_occupied[gen][type]);
	marie_defrag_hist_dec(pfn, gen, type);
}

/* Global aging epoch (bumped per walker pass per type); see state.c. */
extern atomic_t marie_aging_epoch[2 /* ANON_AND_FILE */];

/* Global aging clock (desktop/global-only); see state.c for the contract. */
extern atomic_t marie_head_gen[2 /* ANON_AND_FILE */];
extern u32 marie_recycle_epoch[MARIE_PFN_NR_GENS][2 /* ANON_AND_FILE */];
extern atomic_long_t marie_gen_installs[2 /* ANON_AND_FILE */];

/*
 * Dynamic per-type install-cadence threshold (see state.c). marie_gen_growth_live[]
 * is the live value the install cadence reads; marie_recompute_growth_threshold()
 * runs at each head advance and on a clean_min_ratio write. Fully automatic (no
 * sysfs knob): per-type gen size scales to occupancy (occ/NR_GENS) with an anon
 * coverage/slack warm-up and a file clean_min_ratio-reserve floor.
 */
extern unsigned long marie_gen_growth_live[2 /* ANON_AND_FILE */];
void marie_recompute_growth_threshold(int type);

/*
 * Oldest live gen for @type (walking out from the global head), or -1 if none.
 * The single placement primitive for "make this page cold now": demote /
 * lazyfree (state.c) and Marie defrag's post-migration restamp all target it so
 * a page lands at the current tail, frame-relative -- never a stale absolute
 * gen that head has since recycled. See marie_find_oldest_occupied_mlv.
 */
int marie_find_oldest_occupied_mlv(int type);

/* Global anon-vs-file proportional pick bias; see marie_swap_pick_type. */
extern atomic64_t marie_swap_bias;

/*
 * Per-(type, gen, tier) tracking bitmap. One struct marie_bitmap per
 * (type, gen, tier) tuple, each holding:
 *   - L1: per-PFN bit (BITS_TO_LONGS(max_pfn) words, ~256 KiB / bitmap
 *         on an 8 GiB system; 16 bitmaps = ~4 MiB total)
 *   - L2: 512-bit summary over the same PFN space (64 B / bitmap)
 *   - per-cell refcount: 512 atomic_t per bitmap (2 KiB / bitmap), so
 *     the L2 bit transitions track L1 occupancy exactly via the 0 <->
 *     1 refcount boundary.
 *
 * Scanners walk one (type, gen, tier) bitmap at a time; the L2 plane
 * provides a 512-way fast-skip over empty 32 MiB ranges.
 *
 * struct + operations are defined in mm/lru_marie/bitmap.{h,c}. This is
 * a single global per-(type, gen, tier) plane (marie_track_bm); there is
 * no per-memcg bitmap (desktop/global-only).
 */
extern struct marie_bitmap marie_track_bm[2 /* ANON_AND_FILE */]
					 [MARIE_PFN_NR_GENS]
					 [MARIE_PFN_NR_TIERS];

/*
 * clean_min_ratio: minimum file-pagecache reserve as percent of
 * node_present_pages. Sysfs-tunable in core.c, read by reclaim.
 * Default 15 (le9uo recommendation for desktop).
 */
extern unsigned int marie_clean_min_ratio;

/*
 * Per-PFN isolate scan uses an L2-pruned, range-locked walk rather
 * than a per-CPU cursor -- see marie_state_isolate_scan_l2lock below
 * for the parallelism model (try-lock on per-L2-bit range locks gives
 * 512-way exclusion across concurrent reclaimers).
 */
#include <linux/percpu.h>
/* One-shot init from marie_init(). Allocates marie_state with kvmalloc. */
int marie_state_init(void);
/* Detect CPUID-based prefetch ring parameters. Call before marie_state_init(). */
void marie_prefetch_params_init(void);

struct pglist_data;
struct page;
struct lruvec;
struct scan_control;
struct mem_cgroup;

/*
 * L2-lock parallel isolate scan: collapses the 512-bit outer L2
 * walk to an 8-word loop, word-ANDing the global (type, gen, tier)
 * L2 to skip empty PFN ranges in one cycle; surviving L2 bits are
 * taken under a try_lock for exclusive PFN-range ownership before the
 * inner producer extracts candidates via __ffs/blsr. Global-only: the
 * scan always covers every Marie page (no per-memcg filter).
 */
unsigned long marie_state_isolate_scan_l2lock(struct pglist_data *pgdat,
					      int type, int max_zone,
					      unsigned int tier,
					      struct page **batch,
					      unsigned long batch_size,
					      unsigned long nr_to_scan,
					      int oldest_in);

/*
 * Per-PFN-array reclaim driver. Walks (type, tier) via
 * marie_state_isolate_scan_l2lock, claims each candidate via
 * try_get + test_clear_lru, hands the resulting page_list to
 * shrink_page_list, and putbacks any survivors. Sole reclaim
 * driver in PFN-only Marie.
 */
unsigned int marie_state_shrink_lruvec(struct lruvec *lruvec,
				       struct scan_control *sc);

/*
 * Marie type-pick return codes for marie_swap_pick_type().
 *
 *   MARIE_PICK_FILE_STRICT  swappiness=0:   FILE only, no ANON fallback;
 *                                           caller proceeds to OOM if FILE
 *                                           is depleted.
 *   MARIE_PICK_ANON_STRICT  swappiness=MAX: ANON only, no FILE fallback.
 *   MARIE_PICK_FILE_THEN_ANON  swappiness=1: FILE first; ANON engages
 *                                            ONLY when skip_file is set
 *                                            (clean_min_ratio breached).
 *                                            Per-call transient FILE
 *                                            failures do not promote to
 *                                            ANON -- the floor itself is
 *                                            the sole depletion signal.
 *   MARIE_PICK_ANON_FIRST   Proportional regime (s=2..199), bias picks
 *                           ANON. SINGLE type per call -- scanning the
 *                           other side would dissolve the s:(MAX-s)
 *                           page-flow ratio. Bias gets updated from
 *                           this call's outcome, possibly flipping the
 *                           pick for the next shrink_lruvec call.
 *   MARIE_PICK_FILE_FIRST   Symmetric to ANON_FIRST: bias picks FILE,
 *                           single type per call.
 */
enum marie_pick_kind {
	MARIE_PICK_FILE_STRICT,
	MARIE_PICK_ANON_STRICT,
	MARIE_PICK_FILE_THEN_ANON,
	MARIE_PICK_ANON_FIRST,
	MARIE_PICK_FILE_FIRST,
};

/*
 * Resolve the type-pick policy for one shrink_lruvec invocation.
 *
 * Pure read of the controller state: looks at @swappiness to detect
 * the {0, 1, MAX_SWAPPINESS} special values, otherwise reads the global
 * marie_swap_bias sign to pick the primary type for the proportional
 * regime. Does not modify any state.
 */
enum marie_pick_kind marie_swap_pick_type(u8 swappiness);

/*
 * Apply the bias-controller update for one ATTEMPTED pick.
 *
 *   nr_reclaimed > 0  -> bias += sign * nr_reclaimed * weight
 *                        Page-flow proportional: long-run
 *                        pages(anon):pages(file) -> s:(MAX-s) even
 *                        when per-pick batches differ between types.
 *
 *   nr_reclaimed == 0 -> no-op (bias unchanged)
 *                        Failure carries no back-pressure. The
 *                        picked side stays the picked side
 *                        indefinitely under sustained failure;
 *                        anon is not surrendered just because file
 *                        is transiently or persistently stuck.
 *
 *   sign   = -1 for picked=ANON (push toward FILE)
 *            +1 for picked=FILE (push toward ANON)
 *   weight = MAX_SWAPPINESS - s   for picked=ANON
 *          = s                    for picked=FILE
 *
 * Bypassed entirely under special-value swappiness (0, 1, MAX),
 * where the pick is deterministic and the global bias is not consulted.
 *
 * Caller MUST only invoke when the pick was actually attempted;
 * do NOT call when an external override (skip_file from
 * clean_min_ratio) blocked the picked type before the scan ran.
 */
void marie_swap_bias_update(int picked_type,
			    unsigned long nr_reclaimed,
			    u8 swappiness);

/*
 * Saturating tier increment for a Marie-tracked page's per-PFN byte.
 *
 * Non-saturated bump (tier < MAX) is a best-effort race-tolerant
 * WRITE_ONCE; losing a bump to a concurrent racer is benign because
 * tier is a hotness hint, not a correctness primitive.
 *
 * Saturated bump (tier == MAX) is in-place promote: the page's GEN
 * field is CAS-moved to atomic_read(&marie_head_gen[type]) with TIER
 * reset to 0. The CAS guards against concurrent del and against
 * another walker promoting the same PFN.
 *
 * Skips quietly if the page is not (or no longer) tracked, or if a
 * saturated page is already encoded on the head gen.
 */
void marie_state_inc_tier(unsigned long pfn);

/*
 * marie_state_inc_tier_seeded - tier bump from an already-read state byte.
 * Walker-only fast path: the caller has bounds-checked @pfn and read the
 * byte for the TRACKED gate, so this skips the reload. See state.c.
 */
void marie_state_inc_tier_seeded(unsigned long pfn, u8 cur);

/*
 * marie_state_move_to_gen - relocate a tracked PFN to (@target_gen,
 * @target_tier) in the per-PFN byte, with matched bitmap / occupied
 * counter updates on both source and destination (gen, type) planes.
 *
 * Single point of policy for any operation that needs to move a page
 * between gens. Two callers in design.h:
 *   - walker tier saturate (section 7):
 *       marie_state_move_to_gen(pfn, head, 0)
 *   - shrink_page_list residue putback (section 13):
 *       marie_state_move_to_gen(pfn, (head + 2) & 3, max(prev, w))
 *
 * The state-byte cmpxchg defeats races against del / another
 * concurrent move. The bitmap / counter shuffle uses "new first, then
 * old" ordering so the page remains visible to scan on at least one
 * (gen, type) plane throughout the transition.
 *
 * No-op if the page is no longer tracked or already encodes the
 * target (gen, tier).
 */
void marie_state_move_to_gen(unsigned long pfn,
			     u8 target_gen, u8 target_tier);

struct page;
/*
 * marie_state_drop_pfn - wipe every per-PFN tracking artifact
 * (state byte, (type, gen, tier) L1 bit, occupancy counter, global
 * L2 range counter with bulk L2 clear on 0) for @page. Shared by the
 * normal evict path (marie_evict_locked) and the enable=0 drain path
 * (marie_drain_one_lruvec) so disable->enable cycles never leave
 * ghost per-PFN state behind. No-op when the byte is not TRACKED.
 */
void marie_state_drop_pfn(struct page *page);


/* --- per-page residency state and install/evict surface --- */
#ifdef CONFIG_LRU_MARIE

#include <linux/atomic.h>
#include <linux/cleanup.h>
#include <linux/version.h>
/*
 * <linux/gfp_types.h> is the GFP flag/typedef half of <linux/gfp.h>, split
 * out of it during the 5.19 header-dependency cleanup.  4.19 predates the
 * split and has no such header: the same flags are declared in the unsplit
 * <linux/gfp.h> (gfp.h:214 for __GFP_NOWARN), and gfp_t itself is a
 * <linux/types.h> typedef (types.h:162) on both trees.  So the unsplit
 * header is the era-correct spelling of exactly this dependency -- and on
 * 6.x+ it is a superset anyway, since gfp.h includes gfp_types.h (cachy
 * gfp.h:5).
 */
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 19, 0)
#include <linux/gfp.h>
#else
#include <linux/gfp_types.h>
#endif
#include <linux/hash.h>
#include <linux/irqflags.h>
#include <linux/list.h>
#include <linux/llist.h>
#include <linux/log2.h>
#include <linux/memcontrol.h>
#include <linux/mm.h>

#include <linux/mm_inline.h>
#include <linux/mmzone.h>
#include <linux/percpu.h>
#include <linux/xarray.h>
#include <linux/rcupdate.h>
#include <linux/workqueue.h>
#include <linux/spinlock.h>
#include <linux/swap.h>		/* SWAP_CLUSTER_MAX, ANON_AND_FILE */
#include <linux/types.h>
#include <linux/vmstat.h>

struct page;
struct lruvec;
struct mem_cgroup;
struct marie_gen;

/*
 * marie_lruvec_lock - the LRU lock guarding @lv.
 *
 * The LRU lock moved from the node to the lruvec in 5.11 (upstream commit
 * 6168d0da2b47 "mm/lru: replace pgdat lru_lock with lruvec lock"); before
 * that it is pgdat->lru_lock, reached here the same way 4.19's own
 * zone_lru_lock() reaches it (&zone->zone_pgdat->lru_lock, mmzone.h:763).
 *
 * A field was deliberately NOT added to 4.19's struct lruvec: that would
 * mint a second LRU lock which the rest of the tree -- vmscan, swap,
 * compaction, memcontrol -- never takes, so Marie would assert on, and
 * serialise against, a lock nobody else honours.  Routing through the
 * node's real lock keeps Marie under the same exclusion as every legacy
 * LRU caller, which is the whole point of the assertions below.
 *
 * The substitution is exact rather than approximate because Marie is
 * desktop/global-only and carries zero per-memcg reclaim state (see the
 * architecture notes at the head of core.c): every lruvec it is ever
 * handed is the node's global lruvec, whose 4.19 lock *is* pgdat->lru_lock.
 * On 5.11+ this collapses back to the plain field access the original
 * patch writes.
 *
 * (<linux/version.h> is already included with the header block above.)
 */
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 11, 0)
#define marie_lruvec_lock(lv)	(&lruvec_pgdat(lv)->lru_lock)
#else
#define marie_lruvec_lock(lv)	(&(lv)->lru_lock)
#endif

/*
 * ---------------------------------------------------------------------
 *  Per-page state inspection (internal)
 * ---------------------------------------------------------------------
 *
 * Reads of the per-PFN state byte are lock-free (READ_ONCE). Writes
 * go through state.c helpers (marie_state_inc_tier,
 * marie_state_move_to_gen, marie_state_drop_pfn); the byte is the
 * single source of truth for Marie's per-page state.
 *
 * page->lru is not interpreted as part of Marie's state -- pages
 * are never linked from a Marie-owned list. It exists only so legacy
 * LRU can attach drained pages via lruvec->lists[lru] (handed off by
 * marie_drain_pfn_locked when Marie is disabled).
 */

/**
 * page_marie_test_tracked - is @page claimed by Marie?
 *
 * Reads the per-PFN state byte (the single source of truth in the
 * per-PFN paradigm). page->flags carries no Marie state.
 */
static inline bool page_marie_test_tracked(const struct page *page)
{
	unsigned long pfn = page_to_pfn((struct page *)page);

	if (!marie_state || pfn >= marie_state_size)
		return false;
	return READ_ONCE(marie_state[pfn]) & MARIE_PFN_TRACKED;
}

/*
 * page_marie_get_tier is declared in <linux/lru_marie.h>
 * so callers outside mm/lru_marie/ (e.g. mm/vmscan.c
 * page_check_references) can read tier without including this
 * private header.
 *
 * Tier bumps go through marie_state_inc_tier (defined in state.c) -- the
 * per-PFN state byte is the only place tier lives.
 */

/*
 * marie_page_lruvec_rcu - RCU-bracketed page_lruvec() for Marie hot paths.
 *
 * page_lruvec() reaches obj_cgroup_memcg() which has a lockdep predicate
 * requiring rcu_read_lock or cgroup_mutex. Marie's drain and walker paths
 * run with preemption disabled (e.g. under lru_lock) but NOT under
 * rcu_read_lock(); preempt-disable does not satisfy the lockdep
 * predicate. The brief RCU bracket avoids the WARN trip; the returned
 * pointer is used only for equality comparison, never dereferenced
 * after rcu_read_unlock().
 */
static inline struct lruvec *marie_page_lruvec_rcu(struct page *page)
{
	struct lruvec *lv;

	rcu_read_lock();
	lv = page_lruvec(page);
	rcu_read_unlock();
	return lv;
}

/*
 * marie_update_lru_size - Marie counterpart to legacy update_lru_size().
 *
 * Maintains two counters for a Marie-tracked page:
 *
 *   - node NR_LRU_BASE via mod_lruvec_state(), which also folds into the
 *     per-memcg memory.stat breakdown, preserving it for Marie pages;
 *   - the node-global NR_ZONE_LRU_BASE zone counter via
 *     __mod_zone_page_state().
 *
 * Reclaim sizing reads the latter: lru_marie_zone_size_read ->
 * marie_lruvec_zone_size returns the NR_ZONE_LRU_BASE node total
 * directly -- no per-memcg summing, no Marie-private counter.
 *
 * This deliberately does NOT call mem_cgroup_update_lru_size and does NOT
 * write the per-memcg mz->lru_zone_size; that counter is left to
 * legacy/orphan pages only. So Marie<->legacy list transitions stay
 * mz-neutral and there is no stock RMW under lru_lock to underflow.
 *
 * Caller MUST hold lruvec->lru_lock. mod_lruvec_state's per-CPU fold
 * and __mod_zone_page_state's per-zone counter are documented as
 * lru_lock-protected against concurrent updaters of the same lruvec.
 */
static inline void marie_update_lru_size(struct lruvec *lruvec,
				       enum lru_list lru,
				       enum zone_type zid,
				       long nr_pages)
{
	struct pglist_data *pgdat = lruvec_pgdat(lruvec);

	/*
	 * Node NR_LRU_BASE (folds into the per-memcg memory.stat breakdown,
	 * preserving it for Marie pages).
	 */
	mod_lruvec_state(lruvec, NR_LRU_BASE + lru, nr_pages);
	/*
	 * Node-global NR_ZONE_LRU_BASE zone total -- this is what reclaim
	 * sizing reads (marie_lruvec_zone_size). We do NOT touch the per-memcg
	 * mz->lru_zone_size here: it holds only legacy/orphan pages, which
	 * avoids the non-atomic stock RMW under lru_lock and the
	 * "mem_cgroup_update_lru_size: lru_size -N" underflow it used to cause.
	 */
	__mod_zone_page_state(&pgdat->node_zones[zid],
			      NR_ZONE_LRU_BASE + lru, nr_pages);
}

/*
 * Global page counter, lives in mm/lru_marie/core.c for stats_show;
 * the install/evict helpers in state.c percpu_counter_add it during
 * Marie's TRACKED 0<->1 transitions.
 */
extern struct percpu_counter marie_nr_pages;

/*
 * marie_pc_add - Marie-private percpu_counter add that elides the
 * outer preempt_disable / preempt_enable bracket of
 * percpu_counter_add_batch() while preserving its IRQ safety.
 *
 * percpu_counter_add_batch() wraps the whole body in
 * preempt_disable/enable. Under DEBUG_PREEMPT that bracket shows up in
 * perf under 16-thread memhog as ~4 % of total CPU (preempt_count_add +
 * check_preemption_disabled). We drop it because the individual
 * this_cpu_* primitives used here are each self-contained: this_cpu_add
 * is a single atomic RMW (one instruction on x86), and the slow-path
 * fbc->lock section takes raw_spin_lock_irqsave, so correctness does
 * not depend on the caller's preempt or IRQ state.
 *
 * IRQ safety is MANDATORY, not optional: not every caller holds
 * lru_lock. The reclaim isolate path (marie_evict_counters_only) and
 * the survivor putback in marie_state_shrink_lruvec update the GLOBAL
 * marie_nr_pages counter with IRQs ENABLED (preempt_disable only).
 * The same counter is also bumped from IRQ/softirq context when a
 * Marie-tracked LRU page's last reference is dropped
 * (put_page -> __page_cache_release -> del_page_from_lru_list ->
 * lru_marie_del_page -> marie_evict_locked). If the flush path used a
 * plain raw_spin_lock, a softirq landing on the CPU that already holds
 * fbc->lock would spin forever on it -> hard lockup. Hence
 * raw_spin_lock_irqsave below, exactly as percpu_counter_add_batch does.
 *
 * The fast path uses this_cpu_add (atomic against same-CPU IRQ
 * reentrancy); the earlier __this_cpu_read + __this_cpu_write pair was
 * a non-atomic RMW that could lose an IRQ-context update.
 */
static inline void marie_pc_add(struct percpu_counter *fbc, s64 amount)
{
	s64 count = this_cpu_read(*fbc->counters) + amount;

	if (unlikely(abs(count) >= percpu_counter_batch)) {
		unsigned long flags;

		raw_spin_lock_irqsave(&fbc->lock, flags);
		count = __this_cpu_read(*fbc->counters) + amount;
		fbc->count += count;
		__this_cpu_sub(*fbc->counters, count - amount);
		raw_spin_unlock_irqrestore(&fbc->lock, flags);
	} else {
		this_cpu_add(*fbc->counters, amount);
	}
}

/*
 * ---------------------------------------------------------------------
 *  Install / evict — per-page TRACKED 0 <-> 1 with lru_lock held
 * ---------------------------------------------------------------------
 *
 * Marie's per-page state is one bit: TRACKED in marie_state[pfn].
 * Synchronous install/evict helpers own all the bookkeeping (per-PFN
 * state byte, global per-(type, gen, tier) bitmap, global
 * marie_nr_pages percpu_counter, lru_size mirror, PG_active /
 * PG_lru hygiene):
 *
 *   marie_page_install:            TRACKED 0 -> 1
 *                                   unified fresh install for both small
 *                                   pages and THP; declared in
 *                                   pfn_install.h
 *   marie_state_publish_at_gen:     TRACKED stays, (gen, tier) refreshed
 *                                   reclaim survivor putback
 *   marie_evict_locked:             TRACKED 1 -> 0
 *                                   called from marie_del_page_locked
 *
 * page_marie_test_tracked() is the lock-free state inspector: it
 * reads marie_state[pfn] & MARIE_PFN_TRACKED, returning whether
 * Marie owns @page. The binary state is checked directly at each
 * callsite -- no intermediate dispatch machinery.
 */
bool marie_evict_locked(struct page *page);

/*
 * Reclaim isolate path: counters-only decrement at claim time. The per-
 * PFN state byte's TRACKED bit intentionally stays set throughout
 * shrink_page_list so marie_page_install's TRACKED early-out blocks any
 * concurrent install from setting PG_lru on a page currently in the
 * reclaim list. The scan-bitmap slot + gen_occupied ARE retired here (an
 * isolated page is no longer a scan candidate); the TRACKED byte is
 * wiped later -- at the buddy free hook (marie_state_drop_pfn_at_free)
 * for a reclaimed page, or re-published by marie_state_publish_at_gen at
 * putback for a survivor. See state.c body for the full rationale.
 */
void marie_evict_counters_only(struct page *page);

/*
 * Canonical per-PFN state teardown invoked from
 * mm/page_alloc.c::free_pages_prepare at every page's buddy handoff.
 * Wipes the per-PFN state byte / bitmap / gen_occupied slot whenever
 * the byte still carries TRACKED. No-op on already-cleared state.
 * Counters are NOT touched (they were balanced upstream by Marie's
 * del path or by marie_evict_counters_only).
 *
 * Lock-free; safe from any context.
 */
void marie_state_drop_pfn_at_free(unsigned long pfn);

/* marie_page_install lives in pfn_install.h. */

/*
 * Adaptive batch threshold. Returns the per-call page accumulator cap,
 * lerped between MARIE_PFN_BATCH_FLOOR (low pressure,
 * sc->priority == DEF_PRIORITY) and MARIE_PFN_SHRINK_BATCH (max
 * pressure, sc->priority == 0). Defined in state.c as
 * marie_pfn_batch_threshold; this declaration is the public name.
 */
struct scan_control;
unsigned long marie_adaptive_batch_threshold(struct scan_control *sc);

/**
 * marie_del_page_locked - lru_marie_del_page body.
 * @page:            page to remove (any Marie-tracked state)
 *
 * Universal external-removal handler called from lru_marie_del_page when
 * del_page_from_lru_list fires from outside Marie (compaction, lru_activate
 * batch drain, __page_cache_release after the last put_page). If the
 * page is TRACKED, calls marie_evict_locked to run the full eviction
 * (per-PFN state wipe + counter decrements + lru_size mirror). If the
 * page is no longer TRACKED, returns true defensively (treated as
 * "Marie already removed it").
 *
 * Returns true iff @page was tracked (the caller can fall through to
 * its remaining bookkeeping). The full counter wind-down -- including
 * the single marie_nr_pages -1 -- is owned by marie_evict_locked via
 * marie_account_evict; the caller adds no decrement of its own.
 */
bool marie_del_page_locked(struct page *page);

/*
 * Tier count: alias to the per-PFN state byte's tier field width.
 * Tier lives entirely in marie_state[pfn]'s MARIE_PFN_TIER field
 * (see state.h: MARIE_PFN_NR_TIERS / MARIE_PFN_TIER_MAX). The
 * aliases keep call sites (overflow buffer sizing, tier-loop bounds)
 * readable without rewriting them all.
 *
 * Tier 0 = "never touched since added"; tier MARIE_TIER_MAX = saturated
 * (further young hits trigger a sync promote to head_gen via
 * marie_state_inc_tier).
 */
#define MARIE_NR_TIERS  MARIE_PFN_NR_TIERS
#define MARIE_TIER_MAX  MARIE_PFN_TIER_MAX

/*
 * Reclaim-side batch size — fallback compile-time constant used by
 * a few non-hot-path call sites. The per-PFN scan path uses
 * MARIE_PFN_FALLBACK_BATCH / MARIE_PFN_SHRINK_BATCH (see state.c).
 */
#define MARIE_ISOLATE_BATCH SWAP_CLUSTER_MAX

/*
 * Allocation-side aging trigger threshold (per head gen installs). Fully
 * automatic, per-type: marie_gen_growth_live[type] (state.c),
 * recomputed at each head advance and on a clean_min_ratio write. Global
 * install cadence: marie_page_install advances the head gen once the global
 * marie_gen_installs[type] counter reaches marie_gen_growth_live[type].
 */

/*
 * ---------------------------------------------------------------------
 *  data structures
 * ---------------------------------------------------------------------
 *
 * Per-type independence is fundamental: anon and file each have their
 * own per-type lock and their own slice of the global per-(type, gen,
 * tier) bitmap / counter arrays. vm.swappiness controls only the
 * eviction proportion between types; aging on one type never forces
 * work on the other.
 *
 * The per-PFN state byte carries the zone field, so per-zone filtering
 * is part of the scan mask -- no per-zone data structure is needed
 * (matching the existing NR_LRU_LISTS / zone semantics).
 */

struct marie_type {
	/*
	 * @type_lock serialises per-type operations that need to be
	 * mutually exclusive across CPUs (THP install, split-tail). Hot
	 * install/del do not take it -- they update the per-PFN state byte
	 * and the unified bitmap lock-free.
	 *
	 * @type: 0 = anon, 1 = file. Set once at marie_type_init time so
	 * scoped_guard(marie_type_lock, ...) can recover the type index
	 * (needed for the per-CPU drain-depth counter) from a bare
	 * struct marie_type * without an extra argument.
	 */
	spinlock_t		type_lock;
	int			type;
};

/*
 * Global per-type locks (one per anon/file).  Marie is desktop/global-only,
 * so the per-type serialising lock is a single global instance, not per
 * lruvec.  Defined in state.c, initialised in marie_counters_init.
 */
extern struct marie_type marie_type_locks[ANON_AND_FILE];

/*
 * Per-type re-entrant-drain detection. Caller (lru_marie_del_page in
 * mm/lru_marie/core.c) uses marie_in_drain_type(page's type) to detect "we
 * are already inside a per-type-locked drain for this page's type on
 * this CPU" and skip the scoped_guard re-acquire. The depth counters
 * are per-CPU statics inside the ADT, mutated by the scoped_guard
 * lock/unlock body (S5 / per-CPU encapsulation).
 */
bool marie_in_drain_type(int type);
void marie_drain_enter_type(int type);
void marie_drain_exit_type(int type);

/*
 * ---------------------------------------------------------------------
 *  drain helpers
 * ---------------------------------------------------------------------
 *
 * No promote-queue or per-CPU staging drain remains: every install /
 * evict / tier bump is synchronous (install_local / install_locked
 * publish per-PFN state inline, evict_locked wipes it inline,
 * marie_state_inc_tier handles saturation via marie_state_move_to_gen
 * directly).
 *
 * The remaining drain entry, marie_drain_one_lruvec (in core.c), is
 * only the enable/disable transition: it walks the per-(type, gen,
 * tier) bitmap and hands every TRACKED page back to the legacy LRU.
 */

/*
 * scoped_guard(marie_type_lock, &marie_type_locks[type]) — per-type lock
 * acquisition.
 *
 * Equivalent to the handwritten dance:
 *
 *   spin_lock_irqsave(&t->type_lock, flags);
 *   marie_drain_enter_type(t->type);
 *   ... critical section touching marie_type_locks[t->type] ...
 *   marie_drain_exit_type(t->type);
 *   spin_unlock_irqrestore(&t->type_lock, flags);
 *
 * The cleanup attribute on the guard variable makes the unlock +
 * depth-counter pair a structural property of the scope, not a
 * discipline the caller must remember on every early return / goto.
 *
 * Re-entry inside the scope is handled by the per-CPU per-type
 * marie_drain_depth contract — drain helpers' put_page recursion that
 * lands in lru_marie_del_page observes marie_in_drain_type(page's
 * type) > 0 and skips the spin_lock_irqsave for that type only. Recursion
 * involving the *other* type lands on a depth-0 counter and proceeds to
 * take the corresponding per-type lock as usual (the outer guard holds
 * only one type's lock, so this is not a self-deadlock).
 */
DEFINE_LOCK_GUARD_1(marie_type_lock, struct marie_type,
	/* lock */ ({
		spin_lock_irqsave(&_T->lock->type_lock, _T->flags);
		marie_drain_enter_type(_T->lock->type);
	}),
	/* unlock */ ({
		marie_drain_exit_type(_T->lock->type);
		spin_unlock_irqrestore(&_T->lock->type_lock, _T->flags);
	}),
	unsigned long flags
)

/**
 * marie_counters_init - one-shot init for Marie's global counters.
 *
 * Called from marie_init() (subsys_initcall in mm/lru_marie/core.c).
 * Initialises the global per-type locks (marie_type_locks) and the
 * global marie_nr_pages percpu_counter (the per-CPU bucket pool, slab
 * caches, and cpuhp callbacks that earlier revisions needed have all
 * been retired together with the staging machinery).
 *
 * Returns 0 on success, negative errno on failure (in which case the
 * caller propagates the error up to the initcall machinery).
 */
int marie_counters_init(void);

/*
 * ---------------------------------------------------------------------
 *  Cross-file glue (walker entry points)
 * ---------------------------------------------------------------------
 *
 * These declarations connect mm/lru_marie/core.c (dispatch / lifecycle) and
 * mm/lru_marie/walker.c (PTE walker). They live here to keep mm/
 * private headers down to a single file.
 *
 * Marie holds ZERO per-memcg/per-lruvec state (desktop/global-only): there
 * is no per-lruvec mlv carrier, no lifecycle xarray, no RCU side table.
 */

/*
 * marie_walk_pgdat - run one walker pass for @pgdat.
 *
 * Called from lru_marie_age_node() (kswapd hook) and
 * lru_marie_shrink_lruvec() (direct-reclaim hook). Internally
 * rate-limited per pgdat via a jiffies deadline so calling on every
 * reclaim/kswapd cycle is fine.
 */
void marie_walk_pgdat(struct pglist_data *pgdat);

/*
 * marie_walker_init - one-shot init for the walker subsystem.
 *
 * Initialises per-pgdat bloom-filter spinlocks. Bitmaps themselves
 * are lazily allocated on first Producer hit. Called from
 * marie_init().
 */
void marie_walker_init(void);


#endif /* CONFIG_LRU_MARIE */
#endif /* _MM_LRU_MARIE_STATE_H */
