# SPDX-License-Identifier: GPL-2.0-only
#
# Makefile for the NAP cpuidle governor
#
# 4.19 backport: this tree defines neither CC_FLAGS_FPU (x86) nor a shared
# FPU-flag helper, so the ISA flags are spelled out here.  Object selection
# is arch-conditional (see the design note in nap.h):
#
#   nap.o        always built, compiled with normal kernel flags (no FPU/SIMD).
#                Governor callbacks run at the WFI boundary; they must never
#                emit FPU/SIMD instructions.
#   x86_64       float MLP: nap_fpu.o (feature extraction + decision, run under
#                kernel_fpu_begin() in ->select()) plus the SSE2/AVX2 kernels.
#   arm64        fixed-point inference in ->select() (nap_fixp.o, FPU-free) and
#                NEON float backprop in ->reflect() (nap_nn_neon.o).
#

obj-$(CONFIG_CPU_IDLE_GOV_NAP) += cpuidle_gov_nap.o

cpuidle_gov_nap-y := nap.o

# ======================================================================
# x86_64: SSE2 baseline, AVX2+FMA when available.
#
# The kernel is built with FPU/SSE disabled; FPU-using translation units
# need those flags stripped and the ISA flags added.  nap.o is deliberately
# NOT listed here — it must stay FPU-free so the compiler cannot emit SSE in
# the governor callbacks (which would corrupt userspace FPU state).
# ======================================================================
ifeq ($(CONFIG_X86_64),y)
cpuidle_gov_nap-y += nap_fpu.o nap_nn_sse2.o nap_nn_avx2.o

FPU_KILL_FLAGS := -mno-sse -mno-sse2 -mno-mmx -mno-avx -mno-3dnow \
                  -mno-sse4a -msoft-float -mno-80387 -mno-fp-ret-in-387

CFLAGS_REMOVE_nap_fpu.o     += $(FPU_KILL_FLAGS)
CFLAGS_REMOVE_nap_nn_sse2.o += $(FPU_KILL_FLAGS)
CFLAGS_REMOVE_nap_nn_avx2.o += $(FPU_KILL_FLAGS)

CFLAGS_nap_fpu.o     += -msse2
CFLAGS_nap_nn_sse2.o += -msse2
CFLAGS_nap_nn_avx2.o += -msse2 -mavx -mavx2 -mfma
endif

# ======================================================================
# arm64: NEON/ASIMD is a mandatory AArch64 baseline (no runtime detection).
#
# ->select() runs with hard IRQs disabled (do_idle() -> local_irq_disable()
# -> cpuidle_idle_call() -> cpuidle_select()), where kernel_neon_begin()
# would BUG_ON(!may_use_simd()).  So the forward pass and decision are
# integer fixed-point (nap_fixp.o, normal kernel flags, GP registers only).
#
# ->reflect() runs with IRQs enabled, where NEON is usable, so the float
# backprop lives in nap_nn_neon.o with -mgeneral-regs-only removed (the
# arm64 analogue of the x86 FPU-flag strip; see arch/arm64/lib/Makefile's
# xor-neon.o).
# ======================================================================
ifeq ($(CONFIG_ARM64),y)
cpuidle_gov_nap-y += nap_fixp.o nap_nn_neon.o

CFLAGS_REMOVE_nap_nn_neon.o += -mgeneral-regs-only
endif
