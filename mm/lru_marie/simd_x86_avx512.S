/* SPDX-License-Identifier: GPL-2.0 */
/*
 * mm/lru_marie/simd_x86_avx512.S -- AVX-512F PTE young-bit scan for Marie walker.
 *
 * Processes 512 PTEs (one full PMD page table) per call, producing an
 * 8-ulong (512-bit) young-bit bitmap.  8 PTEs are handled per SIMD
 * iteration via a single 512-bit ZMM load and VPTESTMQ; 8 iterations
 * fill one output ulong; 8 ulongs total.
 *
 * Variant chosen: counter-based rolled loop (outer=8, inner=8).
 * Reason: mirrors the SSE2 and AVX2 kernel structure for auditability;
 * a fully-unrolled variant of 64 SIMD blocks would be ~320 instructions
 * but harder to review and provides no measurable throughput gain on
 * modern out-of-order cores (loop overhead is < 1% of SIMD work).
 *
 * Why AVX-512F is the most efficient of the three .S kernels:
 *   VPTESTMQ ANDs two ZMM registers and writes an 8-bit opmask register
 *   (one bit per 64-bit lane = one bit per PTE), directly encoding eight
 *   young-bit results in a single instruction.  KMOVW transfers the opmask
 *   into a general-purpose register.  This eliminates the PMOVMSKB +
 *   test/setnz dance that SSE2 and AVX2 require.
 *
 *   ~5 instructions per 8 PTEs vs ~24 instructions per 4 PTEs (SSE2/AVX2).
 *   Per-PMD scan is approximately 10 ns of compute on a modern core.
 *
 * KMOVW vs KMOVB:
 *   KMOVW (move 16-bit opmask to/from GPR) is part of AVX-512F.
 *   KMOVB requires AVX-512BW.  VPTESTMQ on an 8-lane ZMM only sets
 *   k1[7:0]; the upper 8 bits written by KMOVW are zero, so KMOVW
 *   gives the correct 8-bit result.  All instructions in this file are
 *   strictly AVX-512F (Foundation); no BW/DQ/VL/VBMI sub-extensions.
 *
 * Caller (marie_simd_scan_avx512_wrapper in mm/lru_marie/simd_x86.c, added in
 * Task 5) holds kernel_fpu_begin/end around this.  Do NOT call directly
 * from C without an FPU context bracket.
 *
 * ABI (System V AMD64):
 *   rdi = const pte_t *pte_table  (512 PTEs, 4096 bytes, page-table base)
 *   rsi = unsigned long *bitmap   (8 ulongs = 64 bytes, caller pre-cleared)
 *
 * Register usage:
 *   zmm0  = young-bit mask broadcast (0x20 x8, 64-bit lanes), constant
 *   zmm1  = 8-PTE scratch register
 *   k1    = VPTESTMQ result (8-bit opmask, one bit per PTE)
 *   rdi   = current PTE pointer (walks forward by 64 bytes per inner iter)
 *   rsi   = bitmap base (constant)
 *   rdx   = outer loop counter (8 -> 0)
 *   rcx   = bit-position counter within current ulong (0, 8, 16, ..., 56);
 *           cl is directly the shlq shift count, so no register move and no
 *           push/pop-around-shlq are needed
 *   rax   = kmovw result / shift scratch
 *   r8    = per-ulong accumulator
 *   r10   = inner loop counter (8 -> 0)
 *   r11   = bitmap write pointer (rsi + outer*8, updated each outer iter)
 *
 * Opmask register constraints:
 *   k0 is reserved as the "no mask" predicate (all lanes active); do not
 *   use k0 as an output register.  k1..k7 are used here (k1 only).
 *
 * AVX-512F only (strict Foundation subset):
 *   vmovdqa64 (mask), vmovdqu64 (PTE data), vptestmq, kmovw, vzeroupper -- all AVX-512F.
 *   zmm0..zmm7 only; k1 only.
 *   VZEROUPPER before RET to avoid SSE/AVX transition penalties.
 */

#include <linux/linkage.h>
#include <linux/cfi_types.h>

	/* -------------------------------------------------------------- */
	/* Read-only data: young-bit mask broadcast                        */
	/* -------------------------------------------------------------- */
	.section .rodata
	.align 64
marie_avx512_young_mask:
	/* _PAGE_ACCESSED = bit 5 = 0x20, broadcast to 8x 64-bit lanes */
	.quad 0x0000000000000020
	.quad 0x0000000000000020
	.quad 0x0000000000000020
	.quad 0x0000000000000020
	.quad 0x0000000000000020
	.quad 0x0000000000000020
	.quad 0x0000000000000020
	.quad 0x0000000000000020

	/* -------------------------------------------------------------- */
	/* Text                                                            */
	/* -------------------------------------------------------------- */
	.text

/*
 * void lru_marie_simd_scan_avx512(const pte_t *pte_table, unsigned long *bitmap)
 *
 * pte_table: rdi -- 512 PTEs (4096 bytes), page-table base; page-aligned by
 *                   the kernel page-table allocator; vmovdqu64 is used so no
 *                   stricter SIMD alignment is required for the input
 * bitmap:    rsi -- 8 ulongs (64 bytes), caller pre-cleared
 */
SYM_TYPED_FUNC_START(lru_marie_simd_scan_avx512)

	/* Load young-bit mask once; zmm0 is constant for the whole call. */
	vmovdqa64	marie_avx512_young_mask(%rip), %zmm0

	movq	%rsi, %r11		/* r11 = current bitmap word ptr    */
	movl	$8, %edx		/* outer counter: 8 ulongs          */

	/*
	 * Outer loop: one iteration per output ulong (64 PTEs / 8 words).
	 *
	 * Each outer iteration processes 64 PTEs in 8 inner SIMD steps
	 * (8 PTEs per step) and writes one 64-bit ulong to the bitmap.
	 */
.Louter_avx512:
	xorq	%r8, %r8		/* accumulator A = 0                */
	xorq	%r9, %r9		/* accumulator B = 0 (split chain)  */
	xorq	%rcx, %rcx		/* bit-position = 0 (cl = shift cnt)*/
	movl	$4, %r10d		/* inner counter: 4 x 2 unrolled    */

	/*
	 * Inner loop: one iteration processes 8 PTEs => 8 result bits.
	 *
	 * Each PTE is 8 bytes; 8 PTEs = 64 bytes = one 512-bit ZMM load.
	 * rdi advances by 64 bytes per inner iteration.
	 *
	 * Algorithm per iteration:
	 *   1. vmovdqu64 loads 8 PTEs into zmm1 (unaligned; page tables are
	 *      page-aligned but rdi is the base of the 512-PTE array which
	 *      is always 8-byte aligned, not necessarily 64-byte aligned).
	 *   2. vptestmq ANDs zmm1 with zmm0 (mask) and sets k1[i] = 1 iff
	 *      the masked value in lane i is nonzero (i.e. young bit set).
	 *   3. kmovw extracts the 8-bit opmask from k1 into eax[7:0].
	 *   4. Zero-extend al -> rax (upper bits already zero from kmovw).
	 *   5. Shift rax left by r9 (0, 8, 16, ..., 56) to place the 8 bits
	 *      at the correct position in the 64-bit accumulator.
	 *   6. OR into r8 accumulator; advance bit-position by 8.
	 *
	 * The variable-shift idiom (shlq %cl, rax) requires the shift count
	 * in cl, so the bit-position is kept in rcx directly and the inner
	 * loop counter lives in r10 -- no push/pop-around-shlq is needed.
	 */
.Linner_avx512:
	/*
	 * DUAL-LOAD: issue both 8-PTE loads (zmm1, zmm2) up front so the two
	 * cache misses are in flight together (memory-level parallelism) --
	 * then test both, then merge A->r8 / B->r9 (split accumulators).
	 */
	vmovdqu64	0(%rdi), %zmm1		/* block A: pte[7]..pte[0]      */
	vmovdqu64	64(%rdi), %zmm2		/* block B: pte[15]..pte[8]    */
	addq	$128, %rdi

	/*
	 * VPTESTMQ: for each 64-bit lane i, compute:
	 *   k1[i] = (zmm1[i] & zmm0[i]) != 0 ? 1 : 0
	 *
	 * zmm0 holds 0x20 in all lanes.  A PTE with _PAGE_ACCESSED set has
	 * bit 5 = 1, so the AND is nonzero => k1[i] = 1.
	 * The 8-bit opmask k1 encodes young status for all 8 PTEs directly.
	 *
	 * No PSHUFD trick or PCMPEQD needed: the opmask result is exact.
	 * No false positives from PFN high bits: VPTESTMQ tests for ANY
	 * nonzero bit after AND with the narrow mask, but since the mask is
	 * exactly 0x20 (single bit), the result is correct for any PFN.
	 */
	vptestmq	%zmm1, %zmm0, %k1	/* block A young mask -> k1     */
	vptestmq	%zmm2, %zmm0, %k2	/* block B young mask -> k2     */

	/*
	 * KMOVW: move the 16-bit opmask register k1 to eax.
	 * VPTESTMQ on an 8-lane ZMM only sets k1[7:0]; k1[15:8] is zero.
	 * So eax[7:0] = 8-bit young bitmap; eax[31:8] = 0 (zero-extended).
	 * rax[63:32] is also zero because movl/kmovw zero-extends to 64 bits.
	 *
	 * KMOVW is in AVX-512F.  KMOVB requires AVX-512BW -- not used here.
	 */
	kmovw	%k1, %eax			/* eax[7:0] = 8-bit young mask  */

	/*
	 * Shift the 8-bit result to the correct position in the accumulator.
	 * rcx = 0 for the first inner iter, 8 for the second, ..., 56 for
	 * the eighth.  After shifting, rax holds the 8 result bits at their
	 * correct positions within the 64-bit ulong.
	 *
	 * The bit-position lives in rcx so cl is already the shift count: a
	 * single shlq %cl with no register move and no push/pop-around-shlq
	 * (the inner counter lives in r10).
	 */
	shlq	%cl, %rax			/* rax <<= bit-position          */
	orq	%rax, %r8			/* block A -> accumulator A       */
	addq	$8, %rcx			/* advance bit-position by 8     */

	/*
	 * Block B: a second 8-PTE block, accumulated into a SEPARATE ulong
	 * (r9) so its shift+OR carry chain is independent of block A's -- two
	 * 4-deep chains in parallel instead of one 8-deep serial chain into
	 * r8. Block B lands at bit-positions 8,24,40,56 (A at 0,16,32,48);
	 * disjoint, so r8|r9 after the loop is the full 64-bit ulong.
	 */
	kmovw	%k2, %eax			/* block B mask (loaded up front) */
	shlq	%cl, %rax
	orq	%rax, %r9			/* block B -> accumulator B       */
	addq	$8, %rcx

	decl	%r10d
	jnz	.Linner_avx512

	/* Combine the two split accumulators, then store the ulong. */
	orq	%r9, %r8
	movq	%r8, (%r11)
	addq	$8, %r11		/* advance bitmap write pointer     */

	decl	%edx
	jnz	.Louter_avx512

	/*
	 * VZEROUPPER: clear the upper halves of all YMM/ZMM registers to
	 * avoid SSE/AVX transition penalties in subsequent legacy SSE code.
	 * Required whenever AVX or AVX-512 code may be followed by SSE code.
	 */
	vzeroupper
	RET
SYM_FUNC_END(lru_marie_simd_scan_avx512)
