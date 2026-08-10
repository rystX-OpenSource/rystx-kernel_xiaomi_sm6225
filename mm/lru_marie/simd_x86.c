// SPDX-License-Identifier: GPL-2.0
/*
 * mm/lru_marie/simd_x86.c -- x86-64 PTE young-bit scan dispatch.
 *
 * Three SIMD .S kernels are linked in: lru_marie_simd_x86_{sse2,avx2,avx512}.S.
 * arch_initcall picks the widest available at boot:
 *   AVX-512F: 8 PTEs/iter via VPTESTMQ kmask
 *   AVX2:     4 PTEs/iter via VPCMPEQQ
 *   SSE2:     4 PTEs/iter via PSHUFD pack
 *
 * SSE2 is the floor -- x86-64 ABI-mandatory since 2003, always works,
 * no cpu_has() check needed. It's the default initial value of the
 * static call, so even if arch_initcall runs late the walker never
 * falls back to the slower scalar path.
 *
 * The single entry point lru_marie_simd_young_pte_mask() self-brackets
 * one PMD scan in kernel_fpu_begin/end, so the FPU bracket wraps only the
 * scan and never spans PMDs or the heavy per-young-PTE work. Per-PMD
 * begin/end is effectively free: kernel_fpu_begin() skips the FPU save
 * entirely for kthreads (kswapd, where the walker runs) and at most once
 * per kernel entry otherwise, and kernel_fpu_end() defers the restore to
 * the return to userspace -- so a per-PMD begin/end pair costs only
 * preempt + bookkeeping, no XSAVE/XRSTOR per scan, and there is nothing
 * to amortise by batching several scans under one bracket.
 */

#include <linux/init.h>
#include <linux/jump_label.h>
#include <linux/minmax.h>
#include <linux/printk.h>
#include <linux/static_call.h>
#include <linux/string.h>
#include <linux/kstrtox.h>
#include <linux/bitmap.h>
#include <asm/cpufeature.h>
#include <asm/fpu/api.h>
#include <asm/pgtable.h>
#include <asm/pgtable_types.h>

#include "simd.h"

#define PTES_PER_PMD	512

/*
 * Default true: walker uses the boot-detected SIMD wrapper. Flipped
 * by writes to /sys/kernel/mm/lru_marie/simd; a write of 0 routes
 * lru_marie_simd_young_pte_mask through the scalar pte_young loop
 * below for benchmark A/B comparisons.
 */
DEFINE_STATIC_KEY_TRUE(marie_simd_enabled_key);
EXPORT_SYMBOL_GPL(marie_simd_enabled_key);

/* Defined in mm/lru_marie/simd_x86_{sse2,avx2,avx512}.S.
 * Caller must hold kernel_fpu_begin/end. */
asmlinkage void lru_marie_simd_scan_sse2(const pte_t *pte_table,
				   unsigned long *bitmap);
asmlinkage void lru_marie_simd_scan_avx2(const pte_t *pte_table,
				   unsigned long *bitmap);
asmlinkage void lru_marie_simd_scan_avx512(const pte_t *pte_table,
				     unsigned long *bitmap);

/* ------------------------------------------------------------------ */
/* Scalar fallback                                                    */
/* ------------------------------------------------------------------ */

/*
 * Reference scalar implementation. Used as the SIMD off-path when
 * marie_simd_enabled_key is flipped via /sys/kernel/mm/lru_marie/simd
 * for A/B-comparing the SIMD walker against a scalar pte_young loop
 * without rebuilding the kernel. Also doubles as a correctness oracle
 * for future SIMD bug fixes. No FPU state -- safe to call regardless
 * of bracket state.
 */
static void marie_simd_scan_scalar(const pte_t *pte, unsigned long *bitmap)
{
	int i;

	for (i = 0; i < PTES_PER_PMD; i++) {
		if (pte_val(pte[i]) & _PAGE_ACCESSED)
			__set_bit(i, bitmap);
	}
}

/* ------------------------------------------------------------------ */
/* Boot-time dispatch                                                 */
/* ------------------------------------------------------------------ */

/*
 * Boot-patched direct call to the .S kernel. arch_initcall upgrades
 * from the SSE2 default to AVX2 / AVX-512F if those feature bits are
 * set. Each call site compiles to a single direct CALL instruction
 * (text-patched at static_call_update time), avoiding the indirect-
 * call retpoline tax in the per-PMD walker hot path.
 *
 * The static call points DIRECTLY at the .S kernel -- no FPU-bracket
 * wrapper. lru_marie_simd_young_pte_mask() supplies the enclosing
 * kernel_fpu_begin/end around the call.
 */
DEFINE_STATIC_CALL(marie_simd_scan, lru_marie_simd_scan_sse2);

/*
 * ISA cap for the SIMD PTE scan. marie_simd_pick() never selects a kernel
 * wider than this, even on a CPU that supports wider. Default AVX-512 = no cap
 * (use the widest available). Capping at AVX2 lets a user avoid AVX-512 where
 * they prefer not to -- e.g. Intel Skylake-X..Cascade-Lake license-based
 * downclocking (NOT an issue on AMD Zen 4/5, which run AVX-512 at full clock).
 * Orthogonal to the `simd` 0/1 master switch, which still forces pure scalar.
 *
 * Set at boot via `lru_marie.simd_max=avx512|avx2|sse2`, or at runtime via
 * /sys/kernel/mm/lru_marie/simd_max (re-patches the static call).
 */
enum marie_simd_isa {
	MARIE_SIMD_SSE2 = 0,
	MARIE_SIMD_AVX2,
	MARIE_SIMD_AVX512,
};
static enum marie_simd_isa marie_simd_max = MARIE_SIMD_AVX512;

/* Widest ISA the CPU supports, clamped to the marie_simd_max cap. */
static enum marie_simd_isa marie_simd_pick(void)
{
	enum marie_simd_isa isa = MARIE_SIMD_SSE2;

	if (boot_cpu_has(X86_FEATURE_AVX512F))
		isa = MARIE_SIMD_AVX512;
	else if (boot_cpu_has(X86_FEATURE_AVX2))
		isa = MARIE_SIMD_AVX2;

	return (isa < marie_simd_max) ? isa : marie_simd_max;
}

/* (Re)patch the static call to the picked kernel and log the choice. */
static void marie_simd_apply(void)
{
	switch (marie_simd_pick()) {
	case MARIE_SIMD_AVX512:
		static_call_update(marie_simd_scan, lru_marie_simd_scan_avx512);
		pr_info("SIMD PTE scan: AVX-512F (8 PTEs/iter)\n");
		break;
	case MARIE_SIMD_AVX2:
		static_call_update(marie_simd_scan, lru_marie_simd_scan_avx2);
		pr_info("SIMD PTE scan: AVX2 (4 PTEs/iter)\n");
		break;
	default:
		static_call_update(marie_simd_scan, lru_marie_simd_scan_sse2);
		pr_info("SIMD PTE scan: SSE2 (4 PTEs/iter, x86-64 baseline)\n");
		break;
	}
}

/* Boot param: lru_marie.simd_max=avx512|avx2|sse2 (read before arch_initcall). */
static int __init marie_simd_max_setup(char *str)
{
	if (str && !strcmp(str, "avx512"))
		marie_simd_max = MARIE_SIMD_AVX512;
	else if (str && !strcmp(str, "avx2"))
		marie_simd_max = MARIE_SIMD_AVX2;
	else if (str && !strcmp(str, "sse2"))
		marie_simd_max = MARIE_SIMD_SSE2;
	return 1;
}
__setup("lru_marie.simd_max=", marie_simd_max_setup);

/*
 * Boot param: lru_marie.simd=0|1 -- master SIMD switch. Seeds the
 * marie_simd_enabled_key state applied in arch_initcall below; 0 forces the
 * scalar pte_young loop from the very first walker pass, default (key TRUE)
 * uses the boot-detected SIMD kernel. Orthogonal to simd_max, which only
 * caps the ISA width. The runtime /sys/kernel/mm/lru_marie/simd knob
 * overrides this afterwards. Accepts the usual kstrtobool forms (0/1/on/off).
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

static int __init marie_simd_x86_init(void)
{
	marie_simd_apply();
	if (marie_simd_boot == 0)
		static_branch_disable(&marie_simd_enabled_key);
	else if (marie_simd_boot == 1)
		static_branch_enable(&marie_simd_enabled_key);
	return 0;
}
/*
 * arch_initcall fires before subsys_initcall (marie_init), so the
 * static call is patched well before the walker first runs.
 */
arch_initcall(marie_simd_x86_init);

/* ---- simd_max sysfs accessors (used by mm/lru_marie/core.c) ---- */

static const char * const marie_simd_isa_names[] = {
	[MARIE_SIMD_SSE2]   = "sse2",
	[MARIE_SIMD_AVX2]   = "avx2",
	[MARIE_SIMD_AVX512] = "avx512",
};

const char *marie_simd_max_name(void)
{
	return marie_simd_isa_names[marie_simd_max];
}

int marie_simd_max_store(const char *buf)
{
	enum marie_simd_isa v;

	if (sysfs_streq(buf, "avx512"))
		v = MARIE_SIMD_AVX512;
	else if (sysfs_streq(buf, "avx2"))
		v = MARIE_SIMD_AVX2;
	else if (sysfs_streq(buf, "sse2"))
		v = MARIE_SIMD_SSE2;
	else
		return -EINVAL;

	marie_simd_max = v;
	marie_simd_apply();
	return 0;
}

/* ------------------------------------------------------------------ */
/* Public API                                                         */
/* ------------------------------------------------------------------ */

void lru_marie_simd_young_pte_mask(const void *table, unsigned long *bitmap)
{
	if (static_branch_likely(&marie_simd_enabled_key)) {
		kernel_fpu_begin();
		static_call(marie_simd_scan)((const pte_t *)table, bitmap);
		kernel_fpu_end();
	} else {
		marie_simd_scan_scalar((const pte_t *)table, bitmap);
	}
}
EXPORT_SYMBOL_GPL(lru_marie_simd_young_pte_mask);
