/* SPDX-License-Identifier: GPL-2.0 */
/*
 * mm/lru_marie/simd_x86_avx2.S -- AVX2 PTE young-bit scan for Marie walker.
 *
 * Processes 512 PTEs (one full PMD page table) per call, producing an
 * 8-ulong (512-bit) young-bit bitmap.  4 PTEs are handled per SIMD
 * iteration via a single 256-bit YMM load; 16 iterations fill one
 * output ulong; 8 ulongs total.
 *
 * Variant chosen: counter-based rolled loop (outer=8, inner=16).
 * Reason: mirrors the SSE2 kernel structure for auditability; 128
 * fully-unrolled SIMD blocks would exceed 3000 instructions with no
 * measurable throughput gain on modern out-of-order cores (loop
 * overhead is < 1% of SIMD work).
 *
 * Why AVX2 is cleaner than SSE2 here:
 *   AVX2 has 64-bit-lane ops (VPSLLQ shifts whole 64-bit PTEs), so the
 *   young bit (bit 5) can be shifted straight into the sign bit (bit 63)
 *   and read out -- no SSE2-style PSHUFD/SHUFPS low-dword packing to
 *   dodge the PFN false positive in the high 32 bits, and no broadcast
 *   mask constant at all.
 *
 *   VPSLLQ $58 moves bit 5 to bit 63; VMOVMSKPD then extracts the sign
 *   bit of each of the 4 64-bit lanes into eax[3:0] -- one bit per PTE,
 *   no per-PTE scalar work.  That 4-bit young mask is the GPR-side
 *   analogue of the AVX-512 kernel's kmovw(k1) and drops straight into
 *   the accumulator via a single variable shift, replacing both the
 *   VPAND + VPCMPEQQ compare and the byte-granular VPMOVMSKB + four
 *   per-PTE test/setnz/shift sequences of earlier revisions.
 *
 * Caller (marie_simd_scan_avx2_wrapper in mm/lru_marie/simd_x86.c, added in
 * Task 5) holds kernel_fpu_begin/end around this.  Do NOT call directly
 * from C without an FPU context bracket.
 *
 * ABI (System V AMD64):
 *   rdi = const pte_t *pte_table  (512 PTEs, 4096 bytes, page-table base;
 *                                  page-aligned by the kernel page-table
 *                                  allocator; vmovdqu is used so no stricter
 *                                  SIMD alignment is required for the input)
 *   rsi = unsigned long *bitmap   (8 ulongs = 64 bytes, caller pre-cleared)
 *
 * Register usage:
 *   ymm1  = 4-PTE scratch register (no mask-constant register needed)
 *   rdi   = current PTE pointer (walks forward by 32 bytes per inner iter)
 *   rsi   = bitmap base (constant)
 *   rdx   = outer loop counter (8 -> 0)
 *   rcx   = bit-position counter within current ulong (steps of 4, 0..60);
 *           cl is directly the shlq shift count, so no register move and no
 *           push/pop-around-shlq are needed
 *   rax   = vmovmskpd result (4-bit young mask) / shift scratch
 *   r8    = per-ulong accumulator
 *   r10   = inner loop counter (16 -> 0)
 *   r11   = bitmap write pointer (rsi + outer*8, updated each outer iter)
 *
 * AVX2 only (no AVX-512 zmm/kmask/opmask):
 *   vmovdqu, vpsllq, vmovmskpd, vzeroupper -- all AVX2.
 *   All VEX-encoded to avoid SSE/AVX transition penalties.
 *   VZEROUPPER before RET to clear upper YMM halves for subsequent
 *   legacy SSE code.
 */

#include <linux/linkage.h>
#include <linux/cfi_types.h>

	/* -------------------------------------------------------------- */
	/* Text  (no rodata: the young bit is shifted into the sign bit,  */
	/*        so there is no broadcast mask constant)                  */
	/* -------------------------------------------------------------- */
	.text

/*
 * void lru_marie_simd_scan_avx2(const pte_t *pte_table, unsigned long *bitmap)
 *
 * pte_table: rdi -- 512 PTEs (4096 bytes), page-table base (8-byte aligned)
 * bitmap:    rsi -- 8 ulongs (64 bytes), caller pre-cleared
 */
SYM_TYPED_FUNC_START(lru_marie_simd_scan_avx2)

	movq	%rsi, %r11		/* r11 = current bitmap word ptr    */
	movl	$8, %edx		/* outer counter: 8 ulongs          */

	/*
	 * Outer loop: one iteration per output ulong (64 PTEs / 8 words).
	 */
.Louter_avx2:
	xorq	%r8, %r8		/* accumulator A = 0                */
	xorq	%r9, %r9		/* accumulator B = 0 (split chain)  */
	xorq	%rcx, %rcx		/* bit-position = 0 (cl = shift cnt)*/
	movl	$8, %r10d		/* inner counter: 8 x 2 unrolled    */

	/*
	 * Inner loop: one iteration processes 4 PTEs => 4 result bits.
	 *
	 * Each PTE is 8 bytes; 4 PTEs = 32 bytes = one 256-bit YMM load.
	 * rdi advances by 32 bytes per inner iteration.
	 */
.Linner_avx2:
	/*
	 * DUAL-LOAD: both 4-PTE loads (ymm1, ymm2) up front for memory-level
	 * parallelism, then shift+extract both, then merge A->r8 / B->r9.
	 */
	vmovdqu	0(%rdi), %ymm1		/* block A: { pte3..pte0 } */
	vmovdqu	32(%rdi), %ymm2		/* block B: { pte7..pte4 } */
	addq	$64, %rdi

	/*
	 * Shift each 64-bit PTE lane left by 58 so the young bit (bit 5 =
	 * _PAGE_ACCESSED) lands in the sign bit (bit 63).  VMOVMSKPD reads
	 * only bit 63 of each lane, so this single op replaces VPAND +
	 * VPCMPEQQ and the broadcast mask entirely.  No PFN false positive
	 * is possible: only the original bit 5 can reach bit 63 (every other
	 * bit shifts past it or stays below it).
	 */
	vpsllq	$58, %ymm1, %ymm1	/* block A: bit5 -> bit63          */
	vpsllq	$58, %ymm2, %ymm2	/* block B: bit5 -> bit63          */

	/*
	 * VMOVMSKPD: extract the sign bit (bit 63) of each of the 4 packed
	 * doubles (64-bit lanes) into eax[3:0] -- exactly one result bit per
	 * PTE.  VPCMPEQQ set each young lane to all-ones (sign bit 1) and
	 * each non-young lane to all-zeros (sign bit 0), so eax[3:0] is the
	 * 4-bit young bitmap directly (bit i = PTE i young).  Writing eax
	 * (32-bit form) zero-extends, so rax = 0..0 | 4 bits.
	 *
	 * This mirrors the AVX-512 kernel's vptestmq -> kmovw(k1) -> eax: one
	 * mask extract per SIMD step then a single variable shift.  The old
	 * byte-granular vpmovmskb forced four per-PTE test/setnz/push-pop/
	 * shift sequences (512 per call); that scalar tail dominated runtime
	 * and left this kernel slower than the narrower SSE2 path even though
	 * its vector compare is cheaper.
	 */
	vmovmskpd %ymm1, %eax		/* eax[3:0] = 4 young bits, one/PTE  */

	/*
	 * Place the 4-bit group at bit-position rcx (0, 4, 8, ..., 60) in the
	 * 64-bit accumulator.  The bit-position lives in rcx precisely so cl
	 * is already the shift count: a single shlq %cl with no register move
	 * and no push/pop-around-shlq (the inner counter lives in r10).  Then
	 * advance the bit-position by the 4 PTEs consumed this iteration.
	 */
	shlq	%cl, %rax		/* rax <<= bit-position              */
	orq	%rax, %r8		/* block A -> accumulator A           */
	addq	$4, %rcx		/* advance bit-position by 4 PTEs    */

	/*
	 * Block B: a second 4-PTE block into a SEPARATE ulong (r9) so its
	 * shift+OR carry chain is independent of block A's. Block B lands at
	 * bit-positions 4,12,...,60 (A at 0,8,...,56); disjoint, so r8|r9
	 * after the loop is the full 64-bit ulong.
	 */
	vmovmskpd %ymm2, %eax		/* block B mask (loaded up front)     */
	shlq	%cl, %rax
	orq	%rax, %r9		/* block B -> accumulator B           */
	addq	$4, %rcx

	decl	%r10d
	jnz	.Linner_avx2

	/* Combine the two split accumulators, then store the ulong. */
	orq	%r9, %r8
	movq	%r8, (%r11)
	addq	$8, %r11		/* advance bitmap write pointer     */

	decl	%edx
	jnz	.Louter_avx2

	/* Clear upper YMM halves to avoid SSE/AVX transition penalties. */
	vzeroupper
	RET
SYM_FUNC_END(lru_marie_simd_scan_avx2)
