/* SPDX-License-Identifier: GPL-2.0 */
/*
 * mm/lru_marie/simd_x86_sse2.S -- SSE2 PTE young-bit scan for the Marie walker.
 *
 * Processes 512 PTEs (one full PMD page table) per call, producing an
 * 8-ulong (512-bit) young-bit bitmap.  4 PTEs are handled per SIMD
 * iteration; 16 iterations fill one output ulong; 8 ulongs total.
 *
 * Variant chosen: counter-based rolled loop (outer=8, inner=16).
 * Reason: keeps the file ~130 lines and fully auditable for SSE2
 * compliance; a fully-unrolled variant would be ~3000 instructions and
 * harder to review without meaningfully faster throughput on modern OOO
 * cores (loop overhead is < 1% of SIMD work).
 *
 * Caller (marie_simd_scan_sse2_wrapper in mm/lru_marie/simd_x86.c, added in
 * Task 5) holds kernel_fpu_begin/end around this.  Do NOT call directly
 * from C without an FPU context bracket.
 *
 * ABI (System V AMD64):
 *   rdi = const pte_t *pte_table  (512 PTEs, 4096 bytes, page-table base;
 *                                  page-aligned by the kernel page-table
 *                                  allocator; movdqu is used so no stricter
 *                                  SIMD alignment is required for the input)
 *   rsi = unsigned long *bitmap   (8 ulongs = 64 bytes, caller pre-cleared)
 *
 * Register usage:
 *   xmm1  = PTE pair 0-1 scratch (no mask-constant register needed)
 *   xmm2  = PTE pair 2-3 scratch
 *   rdi   = current PTE pointer (walks forward)
 *   rsi   = bitmap base (constant)
 *   rdx   = outer loop counter (8 -> 0)
 *   rcx   = bit-position counter within current ulong (steps of 4, 0..60);
 *           cl is directly the shlq shift count, so no register move and no
 *           push/pop-around-shlq are needed
 *   rax   = movmskps result (4-bit young mask) / shift scratch
 *   r8    = per-ulong accumulator
 *   r10   = inner loop counter (16 -> 0)
 *   r11   = bitmap write pointer (rsi + outer*8, updated each outer iter)
 *
 * SSE2 only (x86-64 ABI baseline since AMD Opteron 2003):
 *   movdqu, pslld -- all SSE2; shufps and movmskps are SSE1 (also
 *   x86-64 baseline).  No SSE3, SSSE3, SSE4.x, AVX, BMI1/BMI2
 *   instructions used.
 *
 * SHUFPS low-dword packing: the high 32 bits of an x86-64 PTE carry
 * PFN bits 32-51, pkey, and NX.  Working the young bit at dword
 * granularity, SHUFPS with imm8=0x88 gathers only the low dword of all
 * four 64-bit PTE lanes into one register (a single instruction, vs the
 * former pshufd x2 + punpcklqdq) so the high-dword PFN bits cannot reach
 * the young-bit position.  The packing is required because the SSE2 path
 * shifts at dword granularity (pslld); the AVX2 kernel shifts whole
 * 64-bit lanes (vpsllq) and needs no packing.
 */

#include <linux/linkage.h>
#include <linux/cfi_types.h>

	/* -------------------------------------------------------------- */
	/* Text  (no rodata: the young bit is shifted into the sign bit,  */
	/*        so there is no broadcast mask constant)                  */
	/* -------------------------------------------------------------- */
	.text

/*
 * void lru_marie_simd_scan_sse2(const pte_t *pte_table, unsigned long *bitmap)
 *
 * pte_table: rdi -- 512 PTEs (4096 bytes), page-table base (8-byte aligned)
 * bitmap:    rsi -- 8 ulongs (64 bytes), caller pre-cleared
 */
SYM_TYPED_FUNC_START(lru_marie_simd_scan_sse2)

	movq	%rsi, %r11		/* r11 = current bitmap word ptr    */
	movl	$8, %edx		/* outer counter: 8 ulongs          */

	/*
	 * Outer loop: one iteration per output ulong (64 PTEs / 8 words).
	 */
.Louter:
	xorq	%r8, %r8		/* accumulator = 0                  */
	xorq	%rcx, %rcx		/* bit-position = 0 (cl = shift cnt)*/
	movl	$16, %r10d		/* inner counter: 16 SIMD iters     */

	/*
	 * Inner loop: one iteration processes 4 PTEs => 4 result bits.
	 *
	 * Each PTE is 8 bytes; 4 PTEs = 32 bytes = 2 x 16-byte XMM loads.
	 * rdi advances by 32 bytes per inner iteration.
	 */
.Linner:
	/* Load PTE[0..1] and PTE[2..3] (unaligned; 8-byte align guaranteed). */
	movdqu	(%rdi), %xmm1		/* xmm1 = { pte1_hi:pte1_lo | pte0_hi:pte0_lo } */
	movdqu	16(%rdi), %xmm2		/* xmm2 = { pte3_hi:pte3_lo | pte2_hi:pte2_lo } */
	addq	$32, %rdi

	/*
	 * Pack the low 32-bit dword of each PTE into one XMM register with a
	 * single SHUFPS (SSE1).  shufps imm8=0x88 = 0b_10_00_10_00 takes the
	 * result's low two dwords from the destination (xmm1) and the high
	 * two from the source (xmm2):
	 *   dst[0] = xmm1[0] = lo0    (imm[1:0] = 0)
	 *   dst[1] = xmm1[2] = lo1    (imm[3:2] = 2)
	 *   dst[2] = xmm2[0] = lo2    (imm[5:4] = 0)
	 *   dst[3] = xmm2[2] = lo3    (imm[7:6] = 2)
	 * => xmm1 = { lo3 | lo2 | lo1 | lo0 }
	 *
	 * Identical dword layout to the former pshufd x2 + punpcklqdq (3
	 * instructions), folded into one.
	 */
	shufps	$0x88, %xmm2, %xmm1	/* xmm1 = { lo3, lo2, lo1, lo0 }   */

	/*
	 * Shift each dword left by 26 so the young bit (bit 5) lands in the
	 * sign bit (bit 31).  MOVMSKPS reads only bit 31 of each dword, so
	 * this single op replaces PAND + PCMPEQD and the broadcast mask.
	 * The packed lanes are PTE low dwords, so only the original bit 5
	 * reaches bit 31.
	 */
	pslld	$26, %xmm1		/* bit5 -> bit31 (sign) per dword   */

	/*
	 * MOVMSKPS: extract the sign bit (bit 31) of each of the 4 packed
	 * singles (32-bit lanes) into eax[3:0] -- exactly one result bit per
	 * PTE.  PSLLD put PTE i's young bit in dword i's sign bit, so eax[3:0]
	 * is the 4-bit young bitmap directly (bit i = PTE i young).  Writing
	 * eax (32-bit form) zero-extends, so rax = 0..0 | 4 bits.
	 *
	 * MOVMSKPS is SSE1 (x86-64 baseline) -- even more basic than the
	 * SSE2 PMOVMSKB it replaces.  Mirrors the AVX2 kernel's vmovmskpd:
	 * one mask extract per SIMD step then a single variable shift,
	 * instead of the byte-granular PMOVMSKB + four per-PTE
	 * test/setnz/push-pop/shift sequences (512 per call) whose scalar
	 * tail dominated this kernel's runtime.
	 */
	movmskps %xmm1, %eax		/* eax[3:0] = 4 young bits, one/PTE  */

	/*
	 * Place the 4-bit group at bit-position rcx (0, 4, 8, ..., 60) in the
	 * 64-bit accumulator.  The bit-position lives in rcx precisely so cl
	 * is already the shift count: a single shlq %cl with no register move
	 * and no push/pop-around-shlq (the inner counter lives in r10).  Then
	 * advance the bit-position by the 4 PTEs consumed this iteration.
	 */
	shlq	%cl, %rax		/* rax <<= bit-position (0..60)      */
	orq	%rax, %r8		/* accumulate into output ulong      */
	addq	$4, %rcx		/* advance bit-position by 4 PTEs    */

	decl	%r10d
	jnz	.Linner

	/* Store the completed ulong into bitmap. */
	movq	%r8, (%r11)
	addq	$8, %r11		/* advance bitmap write pointer     */

	decl	%edx
	jnz	.Louter

	RET
SYM_FUNC_END(lru_marie_simd_scan_sse2)
