// SPDX-License-Identifier: GPL-2.0
/*
 * mm/lru_marie/simd_arm64.c -- ARM64 NEON PTE young-bit scan dispatch.
 *
 * One SIMD kernel is linked in: lru_marie_simd_arm64_neon.S (ASIMD, 8
 * PTEs/iter via CMTST + UZP1 narrowing + weighted ADDV).  ASIMD is
 * architecturally mandatory on arm64 whenever FPSIMD is present, so there
 * is no ISA ladder to walk as there is on x86 (AVX-512F > AVX2 > SSE2) and
 * hence no `simd_max` knob here -- the only choice is NEON vs scalar,
 * exposed through the same /sys/kernel/mm/lru_marie/simd 0/1 switch and
 * the same `lru_marie.simd=` boot parameter x86 uses.
 *
 * ---------------------------------------------------------------------
 * Why this file exists, and why it is not simply simd_x86.c with NEON
 * ---------------------------------------------------------------------
 *
 * The x86 entry point self-brackets its scan in kernel_fpu_begin/end at
 * the point of use -- which on x86 is legal under the walker's page-table
 * lock.  On arm64 it is NOT:
 *
 *     kernel_neon_begin()
 *       local_bh_disable();
 *       ... fpsimd_save(); fpsimd_flush_cpu_state();
 *       preempt_disable();
 *       local_bh_enable();          <-- __local_bh_enable_ip()
 *
 * and __local_bh_enable_ip() both asserts IRQs are enabled
 * (lockdep_assert_irqs_enabled()) and, if softirqs are pending, calls
 * do_softirq() inline.  Running softirq handlers while the caller holds
 * the PTE spinlock is a latency and lock-ordering hazard, so the NEON
 * bracket must be opened BEFORE the PTL is taken, not under it.  (This is
 * the FPSIMD save/restore context cost that upstream's status table refers
 * to when it lists arm64 as scalar-only "pending profiling"; the fix is
 * structural rather than a matter of cost.)
 *
 * Hence the two entry points below:
 *
 *   lru_marie_simd_young_pte_mask_prelock()  -- the fast path.  Called by
 *       the walker BEFORE pte_offset_map_lock().  Opens the NEON bracket
 *       in a context where IRQs are on and no spinlock is held, scans, and
 *       closes it.  Returns true when it filled @bitmap.
 *
 *   lru_marie_simd_young_pte_mask()  -- the uniform entry point every arch
 *       provides, called by the walker under the PTL.  On arm64 it is the
 *       scalar fallback: it runs only when the pre-lock attempt declined
 *       (see below), and never takes a NEON bracket.
 *
 * Scanning before the PTL is safe because the young bitmap is strictly an
 * ADVISORY HINT.  The walker re-reads every candidate PTE under the PTL
 * with ptep_get() and re-validates pte_present() / pte_special() /
 * pfn_valid() / the per-PFN TRACKED byte before acting on it, and the aging
 * itself (ptep_test_and_clear_young) is done under the lock.  So:
 *   - a false positive (PTE changed after the scan) is filtered by the
 *     re-validation under the lock, exactly as it already is for a PTE that
 *     changed between the x86 under-lock scan and the loop body;
 *   - a false negative (PTE became young just after the scan) merely defers
 *     that page's tier bump to the next walker pass, which is already the
 *     steady-state behaviour of a sampling walker.
 * The page table itself cannot be freed under us: the walker holds the
 * mmap_sem and the pmd is established, which is the same guarantee that
 * lets the generic pagewalk code dereference it.
 *
 * The pre-lock attempt declines (returns false, leaving @bitmap untouched)
 * whenever a NEON bracket is not permissible -- !may_use_simd(), which
 * covers hardirq/softirq-disabled contexts, or the simd knob being off.
 * The walker then falls through to the under-PTL scalar path, so a decline
 * costs correctness nothing.
 */

#include <linux/bitmap.h>
#include <linux/build_bug.h>
#include <linux/init.h>
#include <linux/jump_label.h>
#include <linux/kernel.h>
#include <linux/mm.h>			/* pte_young */
#include <linux/printk.h>
#include <asm/cpufeature.h>		/* system_supports_fpsimd */
#include <asm/neon.h>			/* kernel_neon_begin/end */
#include <asm/simd.h>			/* may_use_simd */
#include <asm/pgtable.h>		/* pte_young, PTRS_PER_PTE */
#include <asm/pgtable-hwdef.h>		/* PTE_AF */

#include "simd.h"

#define PTES_PER_PMD	512

/*
 * The .S kernel bakes in PTE_AF numerically so it needs no pgtable
 * headers.  Pin the two together: if the arch ever moves the Access Flag,
 * this fails the build rather than silently scanning the wrong bit.
 *
 * The kernel also assumes a 512-entry PMD (4 KiB of PTEs) and
 * little-endian lane/byte order; both are checked where the scan is
 * selected (marie_neon_usable) rather than here, since a 16K/64K-page or
 * big-endian build must fall back to scalar at runtime, not fail to build.
 */
static_assert(PTE_AF == (1 << 10),
	      "simd_arm64_neon.S hardcodes PTE_AF as bit 10");

/*
 * Default true: the walker uses the NEON scan.  Writing 0 to
 * /sys/kernel/mm/lru_marie/simd flips the static branch so the walker
 * falls back to the scalar pte_young loop below, for A/B benchmarking
 * against the NEON path without a rebuild.  Shared spelling with x86 so
 * mm/lru_marie/core.c's sysfs knob needs no arch conditional.
 */
DEFINE_STATIC_KEY_TRUE(marie_simd_enabled_key);
EXPORT_SYMBOL_GPL(marie_simd_enabled_key);

/*
 * Defined in mm/lru_marie/simd_arm64_neon.S.
 * Caller must hold kernel_neon_begin/end.
 */
asmlinkage void lru_marie_simd_scan_neon(const pte_t *pte_table,
					 unsigned long *bitmap);

/* ------------------------------------------------------------------ */
/* Scalar fallback                                                    */
/* ------------------------------------------------------------------ */

/*
 * Reference scalar implementation, and the arm64 under-PTL path.  Used
 * when the simd knob is off, when may_use_simd() says a NEON bracket is
 * not permissible, and on page-size / endianness configurations the NEON
 * kernel does not cover.  Uses the arch pte_young() helper rather than
 * open-coding PTE_AF.  No FPSIMD state -- safe to call under any lock.
 */
static void marie_simd_scan_scalar(const pte_t *pte, unsigned long *bitmap)
{
	int i;

	for (i = 0; i < PTES_PER_PMD; i++) {
		if (pte_young(pte[i]))
			__set_bit(i, bitmap);
	}
}

/* ------------------------------------------------------------------ */
/* Boot-time configuration                                            */
/* ------------------------------------------------------------------ */

/*
 * Boot param: lru_marie.simd=0|1 -- master SIMD switch, same spelling and
 * semantics as x86's.  Seeds the marie_simd_enabled_key state applied in
 * the arch_initcall below; 0 forces the scalar pte_young loop from the
 * very first walker pass.  The runtime /sys/kernel/mm/lru_marie/simd knob
 * overrides this afterwards.  Accepts the usual kstrtobool forms.
 */
static int marie_simd_boot __initdata = -1;	/* -1 unset, 0 off, 1 on */
static int __init marie_simd_setup(char *str)
{
	bool v;

	if (str && !kstrtobool(str, &v))
		marie_simd_boot = v;
	return 1;
}
__setup("lru_marie.simd=", marie_simd_setup);

/*
 * Is the NEON kernel applicable to this build at all?
 *
 * The .S kernel scans exactly 512 PTEs and maps lanes to bits in
 * little-endian order.  A 16K/64K-page build (PTRS_PER_PTE != 512) or a
 * big-endian build gets the scalar path; both resolve at compile time, so
 * the check folds away on the common 4K LE configuration.
 *
 * Endianness is spelled as !CONFIG_CPU_BIG_ENDIAN rather than
 * CONFIG_CPU_LITTLE_ENDIAN: arm64 defines only the big-endian symbol (see
 * arch/arm64/Kconfig and the #ifndef in asm/sysreg.h), so testing for a
 * LITTLE symbol would be permanently false and would silently disable the
 * NEON path on every build.
 *
 * CONFIG_KERNEL_MODE_NEON is def_bool y on arm64, so its check folds to
 * true; it is spelled out anyway because kernel_neon_begin() is only
 * *defined* under that symbol -- this keeps the dependency honest rather
 * than implicit.
 *
 * The FPSIMD-presence check is a separate, runtime concern:
 * system_supports_fpsimd() can be false on a (rare) CPU without FPSIMD,
 * where kernel_neon_begin() would WARN and return without a bracket.
 */
static inline bool marie_neon_supported(void)
{
	return IS_ENABLED(CONFIG_KERNEL_MODE_NEON) &&
	       !IS_ENABLED(CONFIG_CPU_BIG_ENDIAN) &&
	       PTRS_PER_PTE == PTES_PER_PMD;
}

static int __init marie_simd_arm64_init(void)
{
	if (marie_simd_boot == 0)
		static_branch_disable(&marie_simd_enabled_key);
	else if (marie_simd_boot == 1)
		static_branch_enable(&marie_simd_enabled_key);

	if (!marie_neon_supported()) {
		/*
		 * Not a build the NEON kernel covers.  Leave the knob alone
		 * (so its state still reads back as written) -- the usable
		 * check at each call site routes to scalar regardless.
		 */
		pr_info("SIMD PTE scan: scalar (NEON needs 4K pages + little-endian)\n");
		return 0;
	}

	if (static_branch_likely(&marie_simd_enabled_key))
		pr_info("SIMD PTE scan: NEON/ASIMD (8 PTEs/iter)\n");
	else
		pr_info("SIMD PTE scan: scalar (disabled via lru_marie.simd=0)\n");
	return 0;
}
/*
 * arch_initcall fires before subsys_initcall (marie_init), so the boot
 * seed is applied well before the walker first runs -- matching x86.
 */
arch_initcall(marie_simd_arm64_init);

/* ------------------------------------------------------------------ */
/* Public API                                                         */
/* ------------------------------------------------------------------ */

/*
 * May we open a NEON bracket right now, and should we?
 *
 * may_use_simd() is the arm64 gate: false in hardirq context, in a
 * softirq while FPSIMD is already in kernel use, and whenever
 * kernel_neon_busy is set.  It does NOT test for preemption or spinlocks,
 * which is exactly why the call site matters -- see the file header.
 */
static inline bool marie_neon_usable(void)
{
	return static_branch_likely(&marie_simd_enabled_key) &&
	       marie_neon_supported() &&
	       system_supports_fpsimd() &&
	       may_use_simd();
}

bool lru_marie_simd_young_pte_mask_prelock(const void *table,
					   unsigned long *bitmap)
{
	if (!marie_neon_usable())
		return false;

	/*
	 * Legal here and not under the PTL: IRQs are enabled and no
	 * spinlock is held, so the local_bh_enable() inside
	 * kernel_neon_begin() may run pending softirqs safely.  The bracket
	 * spans one PMD scan only -- ~600 instructions, no memory
	 * allocation, no sleeping call -- so the preempt-disabled window it
	 * opens stays bounded regardless of walk length.
	 */
	kernel_neon_begin();
	lru_marie_simd_scan_neon((const pte_t *)table, bitmap);
	kernel_neon_end();

	return true;
}
EXPORT_SYMBOL_GPL(lru_marie_simd_young_pte_mask_prelock);

void lru_marie_simd_young_pte_mask(const void *table, unsigned long *bitmap)
{
	/*
	 * The under-PTL entry point.  Always scalar on arm64: a NEON
	 * bracket is not permissible here (file header), and the walker has
	 * already had its chance at the NEON path via the _prelock hook
	 * above.  Reached when that hook declined, or from any caller that
	 * does not implement the pre-lock protocol.
	 */
	marie_simd_scan_scalar((const pte_t *)table, bitmap);
}
EXPORT_SYMBOL_GPL(lru_marie_simd_young_pte_mask);
