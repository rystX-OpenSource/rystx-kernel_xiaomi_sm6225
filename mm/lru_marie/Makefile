# SPDX-License-Identifier: GPL-2.0
#
# Marie LRU. Reached from mm/Makefile via obj-$(CONFIG_LRU_MARIE) += lru_marie/.
# Public dispatch surface for the rest of the kernel is in
# include/linux/lru_marie.h; everything below is Marie-private.

obj-y	+= bitmap.o
obj-y	+= core.o
obj-y	+= state.o
obj-y	+= walker.o

obj-$(CONFIG_LRU_MARIE_DEFRAG)	+= defrag.o

ifdef CONFIG_X86
obj-y	+= simd_x86.o
obj-y	+= simd_x86_sse2.o
obj-y	+= simd_x86_avx2.o
obj-y	+= simd_x86_avx512.o
else ifdef CONFIG_ARM64
obj-y	+= simd_arm64.o
obj-y	+= simd_arm64_neon.o
#
# No per-object flags are needed for the NEON kernel here:
#
#  - arm64's -mgeneral-regs-only lives in KBUILD_CFLAGS, not KBUILD_AFLAGS, so
#    it never reaches a .S file.  (arch/arm64/lib/Makefile has to strip it for
#    xor-neon.c precisely because that one is C with NEON intrinsics; ours is
#    hand-written assembly and simd_arm64.c itself touches no vector type.)
#  - the assembler's default arm64 target already permits baseline ASIMD, so
#    no -march/.arch is required -- only the crypto extensions need one, which
#    is why arch/arm64/crypto/{chacha-neon-core,aes-neon}.S carry no .arch
#    directive either while the *-ce-*.S files do.
else
obj-y	+= simd_generic.o
endif
