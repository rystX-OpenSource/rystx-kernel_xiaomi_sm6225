/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _MM_LRU_MARIE_PREFETCH_H
#define _MM_LRU_MARIE_PREFETCH_H

/*
 * Two-stage software prefetch primitives used by Marie's per-PFN
 * array scan. The bitmap-driven isolate loop issues
 *
 *   marie_prefetch_l3(target_N_ahead);   // pull from DRAM into L3
 *   marie_prefetch_l1(target_K_ahead);   // pull from L3 into L1
 *
 * where N (~marie_l3_ahead) is sized to cover DRAM round-trip
 * (~200 cycles) and K (~marie_l1_ahead) is sized to cover the L3->
 * L1 round-trip (~30 cycles). Splitting lets the AGU fire the long-
 * haul prefetch as early as the bitmap walk can predict the next
 * candidate PFN, without keeping the L1 occupied with all the
 * pending lines at once.
 *
 * The kernel's generic prefetch() expands to PREFETCHNTA on x86
 * (L1 with bypass-LRU semantics), which is wrong for the L3-ahead
 * leg — NTA evicts quickly from L1 and never settles in L3, so by
 * the time the target should be in L3 it is gone. We therefore
 * drop to the bare instructions:
 *
 *   prefetcht0 -- T0 hint, fetched into all cache levels (L1+L2+L3)
 *   prefetcht2 -- T2 hint, fetched into L2/L3 but not L1
 *
 * Non-x86 builds get no-op stubs; the scan still works, just
 * without the prefetch acceleration (HW prefetcher alone).
 */

#ifdef CONFIG_X86
static __always_inline void marie_prefetch_l1(const void *addr)
{
	asm volatile("prefetcht0 %0" :: "m" (*(const char *)addr));
}

static __always_inline void marie_prefetch_l3(const void *addr)
{
	asm volatile("prefetcht2 %0" :: "m" (*(const char *)addr));
}
#else
static __always_inline void marie_prefetch_l1(const void *addr) { (void)addr; }
static __always_inline void marie_prefetch_l3(const void *addr) { (void)addr; }
#endif

/*
 * Ahead distances for the two-stage prefetch ring. Values are set at
 * boot by marie_prefetch_params_init() based on CPUID and stored in
 * the file-static variables in state.c. MARIE_L3_AHEAD_MAX is the
 * compile-time upper bound used to size the on-stack ring[] array;
 * the runtime value (marie_l3_ahead) may be smaller on MSHR-limited
 * microarchitectures.
 *
 * prefetcht2 requests are tracked by L2/L3 MSHRs (independent of L1
 * LFBs); prefetcht0 requests are tracked by L1 LFBs. Tiers chosen by
 * marie_prefetch_params_init():
 *
 *   AVX-512F (Zen 4/5, Sapphire Rapids): L2 MSHR ~32 → l3=32, l1=8
 *   AMD Zen 3 (fam 0x19):               L2 MSHR ~24 → l3=24, l1=8
 *   AMD Zen 1/2 (fam 0x17):             L2 MSHR ~20 → l3=20, l1=8
 *   AMD Excavator (fam 0x15):           L2 MSHR ~12 → l3=16, l1=6
 *   Intel Skylake+ (CLFLUSHOPT):        L2 MSHR ~24 → l3=24, l1=8
 *   Intel Haswell/Broadwell:            L2 MSHR ~16 → l3=16, l1=6
 *   x86_64-v2 or below / non-x86:      L2 MSHR  ~8 → l3= 8, l1=2
 *
 * marie_l3_mask = marie_l3_ahead - 1 (all values are powers of 2,
 * enabling bitwise-AND modulo in the hot path).
 *
 * These can be promoted to sysfs tunables in a later commit if
 * profiling shows different sweet spots per workload.
 */
#define MARIE_L3_AHEAD_MAX	32	/* on-stack ring[] sizing upper bound */

void marie_prefetch_params_init(void);

/*
 * Cache-line cursor look-ahead for marie_state[] (1 byte per PFN, 64 PFN
 * per cache line). Unlike the per-PFN struct page prefetch (where 1 PFN
 * = 1 cache line already gives ring-depth look-ahead), state[] is dense
 * — without an explicit cursor, the producer issues up to 64 prefetches
 * for the same cache line and gains zero look-ahead in cache-line space.
 *
 * Sized for the sparse-bitmap fast-skip case. On OOO x86 (~5 cycles/PFN,
 * DRAM ~200 cycles) we need ≥ 40 PFN on top of the runtime ring lag
 * (up to 32); on MSHR-limited in-order x86 (~20 cy/PFN, ring lag 8)
 * ~18 PFN suffices. 512 PFN (8 cache lines) covers all tiers and also
 * absorbs bitmap-density jumps within an L2 range.
 *
 * L1 distance is the L3→L1 analogue: shorter latency target, smaller
 * margin since L1d evicts aggressively.
 */
#define MARIE_STATE_L3_AHEAD_PFN	512
#define MARIE_STATE_L1_AHEAD_PFN	64

/*
 * Cache-line cursor look-ahead for the bitmap arrays (l1[], mbm[]) used
 * by the isolate producer. The arrays are u64 (8 words per cache line,
 * each word covering 64 PFN). The producer reads one word per "word_rem
 * exhausted" event; in the sparse-bitmap worst case (1 bit per word) the
 * word transition rate hits ~5-30 cycles per consumer iter, so the next
 * cache line must be on the way well before the cursor crosses it.
 *
 * 16 words = 2 cache lines ahead gives margin for the sparse case while
 * keeping the prefetch budget modest. Only L3 hint is needed — once a
 * bitmap cache line lands in L3, the L3->L1 promote (~30-40 cycles) is
 * easily hidden by the per-word consumer drain (64 PFN × 5+ cycles).
 */
#define MARIE_BM_L3_AHEAD_WORDS		16

#endif /* _MM_LRU_MARIE_PREFETCH_H */
