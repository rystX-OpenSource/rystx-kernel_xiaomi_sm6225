# SPDX-License-Identifier: GPL-2.0
#
# Task-aware Android Guided Low Memory Killer
#
obj-$(CONFIG_ANDROID_TAGLMK)		+= taglmk.o

taglmk-y := taglmk_core.o taglmk_pin.o taglmk_predict.o taglmk_lru.o
taglmk-$(CONFIG_ANDROID_TAGLMK_ARM64_NEON) += taglmk_neon.o

# The NEON accelerator needs the vector registers the rest of the kernel is
# built without.  Drop -mgeneral-regs-only for that single object only, the
# same way arch/arm64/lib/xor-neon.o is built.
CFLAGS_REMOVE_taglmk_neon.o += -mgeneral-regs-only
CFLAGS_taglmk_neon.o += -ffreestanding
