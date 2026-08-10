/* SPDX-License-Identifier: GPL-2.0 */
/*
 * mm/lru_marie/simd_arm64_neon.S -- ARM64 NEON (ASIMD) PTE young-bit scan
 *                                   for the Marie walker.
 *
 * Processes 512 PTEs (one full PMD page table on 4K-page arm64) per call,
 * producing an 8-ulong (512-bit) young-bit bitmap.  8 PTEs are handled per
 * SIMD iteration; 8 iterations fill one output ulong; 8 ulongs total.
 *
 * Variant chosen: counter-based rolled loop (outer=8, inner=8), mirroring
 * the structure of the x86 SSE2 kernel (mm/lru_marie/simd_x86_sse2.S) --
 * a per-iteration byte mask is folded into a 64-bit GPR accumulator at a
 * running bit position.  A fully-unrolled variant would be ~1000
 * instructions and no faster on an out-of-order core: this kernel is
 * bounded by the 4 KiB of page-table loads (64 cache lines), not by loop
 * overhead.
 *
 * Young bit on arm64 is the hardware Access Flag, PTE_AF = bit 10
 * (arch/arm64/include/asm/pgtable-hwdef.h); pte_young() is
 * "!!(pte_val(pte) & PTE_AF)".  simd_arm64.c carries a static_assert that
 * pins PTE_AF to the value baked in below, so a future arch change fails
 * the build loudly instead of silently mis-scanning.
 *
 * The caller (lru_marie_simd_young_pte_mask / _prelock in
 * mm/lru_marie/simd_arm64.c) holds kernel_neon_begin/end around this.  Do
 * NOT call directly from C without a NEON context bracket.
 *
 * ABI (AAPCS64):
 *   x0 = const pte_t *pte_table  (512 PTEs, 4096 bytes, page-table base;
 *                                 page-aligned by the kernel page-table
 *                                 allocator.  LD1 requires only natural
 *                                 element alignment, so the 8-byte
 *                                 alignment of a pte_t array suffices.)
 *   x1 = unsigned long *bitmap   (8 ulongs = 64 bytes; fully overwritten,
 *                                 so the caller need not pre-clear)
 *
 * Register usage:
 *   v0-v3 = 8 PTEs under test / narrowing scratch
 *   v28   = PTE_AF broadcast into both 64-bit lanes
 *   v30   = byte weights { 1,2,4,8,16,32,64,128 }
 *   x0    = current PTE pointer (walks forward, 64 bytes per inner iter)
 *   x1    = bitmap write pointer (advances 8 bytes per outer iter)
 *   x2    = outer loop counter (8 -> 0)
 *   x3    = constant-pool pointer
 *   x4    = inner loop counter (8 -> 0)
 *   x5    = bit position within the current ulong (steps of 8, 0..56)
 *   x6    = extracted byte mask / shift scratch
 *   x7    = per-ulong accumulator
 *
 * Baseline ASIMD only (mandatory on arm64 whenever FPSIMD is present):
 *   ld1, cmtst, uzp1, xtn, and, addv, umov, movi.  No SVE, no dot-product,
 *   no crypto extensions.
 *
 * Little-endian only.  The byte/bit mapping below (byte j of the output
 * word holds PTEs 8j..8j+7) relies on LE lane and byte ordering; the C
 * dispatcher does not build or select this kernel on big-endian.
 */

#include <linux/linkage.h>

/*
 * PTE_AF, spelled numerically so this file needs no pgtable headers.
 * Kept honest by the BUILD_BUG_ON in simd_arm64.c.
 */
#define MARIE_PTE_AF	(1 << 10)

	.text

/*
 * void lru_marie_simd_scan_neon(const pte_t *pte_table, unsigned long *bitmap)
 *
 * pte_table: x0 -- 512 PTEs (4096 bytes), page-table base (8-byte aligned)
 * bitmap:    x1 -- 8 ulongs (64 bytes), fully overwritten
 */
SYM_FUNC_START(lru_marie_simd_scan_neon)

	/* Load the two vector constants from the pool below. */
	adrp	x3, .Lmarie_neon_consts
	add	x3, x3, :lo12:.Lmarie_neon_consts
	ld1	{v28.2d}, [x3], #16		/* AF broadcast, both lanes   */
	ld1	{v30.8b}, [x3]			/* byte weights 1,2,4,...,128 */

	mov	x2, #8				/* outer: 8 output ulongs     */

	/*
	 * Outer loop: one iteration per output ulong (64 PTEs / 8 words).
	 */
.Louter:
	mov	x7, xzr				/* accumulator = 0            */
	mov	x5, xzr				/* bit position = 0           */
	mov	x4, #8				/* inner: 8 groups of 8 PTEs  */

	/*
	 * Inner loop: one iteration processes 8 PTEs => 8 result bits.
	 *
	 * Each PTE is 8 bytes; 8 PTEs = 64 bytes = one LD1 of four Q
	 * registers, which is also exactly one cache line.
	 */
.Linner:
	ld1	{v0.2d, v1.2d, v2.2d, v3.2d}, [x0], #64

	/*
	 * CMTST lane-tests (Vn AND Vm) != 0, setting the lane to all-ones
	 * on a hit and all-zeros otherwise -- a one-instruction
	 * "is the Access Flag set" per 64-bit PTE, with no mask-and-compare
	 * pair and no shifting of the PFN bits into the test.
	 *
	 * Lane order after this point:
	 *   v0 = {PTE0, PTE1}   v1 = {PTE2, PTE3}
	 *   v2 = {PTE4, PTE5}   v3 = {PTE6, PTE7}
	 */
	cmtst	v0.2d, v0.2d, v28.2d
	cmtst	v1.2d, v1.2d, v28.2d
	cmtst	v2.2d, v2.2d, v28.2d
	cmtst	v3.2d, v3.2d, v28.2d

	/*
	 * Narrow 8 x 64-bit all-ones/all-zeros lanes down to 8 bytes,
	 * preserving PTE order.  UZP1 takes the even-indexed elements of
	 * its two sources, which at .4s granularity is precisely the low
	 * half of each 64-bit lane -- the truth value survives narrowing
	 * because the lane is uniform.
	 *
	 *   uzp1 .4s : {PTE0,PTE1,PTE2,PTE3} and {PTE4,PTE5,PTE6,PTE7}
	 *   uzp1 .8h : {PTE0..PTE7} as 16-bit lanes
	 *   xtn  .8b : {PTE0..PTE7} as bytes (0xFF set / 0x00 clear)
	 */
	uzp1	v0.4s, v0.4s, v1.4s
	uzp1	v2.4s, v2.4s, v3.4s
	uzp1	v0.8h, v0.8h, v2.8h
	xtn	v0.8b, v0.8h

	/*
	 * Weight each surviving byte by its bit position and horizontally
	 * add: byte i contributes 1<<i exactly when PTE i is young, so the
	 * 8-bit sum IS the young bitmap for this group.  Each lane is 0 or
	 * its distinct power of two, so the sum tops out at 255 and cannot
	 * overflow the byte ADDV produces.
	 */
	and	v0.8b, v0.8b, v30.8b
	addv	b0, v0.8b

	/*
	 * Fold the 8-bit group mask into the output ulong at the running
	 * bit position.  LSLV takes its shift count from a register, so the
	 * position lives in x5 and no unrolling is needed to satisfy an
	 * immediate-index constraint (the reason this kernel accumulates in
	 * a GPR rather than inserting bytes into a vector).
	 */
	umov	w6, v0.b[0]			/* w6 = 8 young bits         */
	lsl	x6, x6, x5			/* shift to bit position     */
	orr	x7, x7, x6			/* accumulate into ulong     */
	add	x5, x5, #8			/* advance by 8 PTEs         */

	subs	x4, x4, #1
	b.ne	.Linner

	/* Store the completed ulong and advance the bitmap pointer. */
	str	x7, [x1], #8

	subs	x2, x2, #1
	b.ne	.Louter

	ret
SYM_FUNC_END(lru_marie_simd_scan_neon)

	/* -------------------------------------------------------------- */
	/* Constant pool                                                  */
	/* -------------------------------------------------------------- */
	.section .rodata, "a"
	.align	4
.Lmarie_neon_consts:
	/* PTE_AF broadcast to both 64-bit lanes (CMTST operand). */
	.quad	MARIE_PTE_AF
	.quad	MARIE_PTE_AF
	/* Per-byte weights: byte i -> bit i of the group mask. */
	.byte	1, 2, 4, 8, 16, 32, 64, 128
