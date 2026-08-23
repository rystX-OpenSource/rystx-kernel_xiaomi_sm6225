# SPDX-License-Identifier: GPL-2.0
#
# Task-aware Android Guided Low Memory Killer
#
obj-$(CONFIG_ANDROID_TAGLMK)	+= taglmk.o

taglmk-y			:= core.o task.o predict.o zram.o sysfs.o
taglmk-$(CONFIG_ANDROID_TAGLMK_ARM64_NEON) += neon.o

# neon.c is the only file here that speaks intrinsics, and it needs two things
# the rest of the kernel is built without: -ffreestanding, because <arm_neon.h>
# is not compilable in a C99-incomplete environment, and the vector registers,
# which -mgeneral-regs-only otherwise puts out of reach.  Both are scoped to the
# single object, so nothing else in the driver can accidentally emit FPU code.
# The option this is guarded by already depends on ARM64, so there is no other
# architecture to spell out.  See lib/raid6/Makefile, which does the same.
CFLAGS_neon.o			+= -ffreestanding
CFLAGS_REMOVE_neon.o		+= -mgeneral-regs-only
