/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * KGSL <-> FLD (Frame Load Decision GPU scheduler) interface.
 *
 * Two tiny hooks; both compile to nothing when CONFIG_FLD_GPU=n so a
 * stock kernel is byte-for-byte unaffected.  The hook implementations and
 * the rest of FLD live in drivers/gpu/msm/fld/; see docs/fld-kgsl-hooks.md.
 */

#ifndef __KGSL_FLD_H
#define __KGSL_FLD_H

#include <linux/types.h>

struct kgsl_device;

#ifdef CONFIG_FLD_GPU

/* Called from the submit path (kgsl_ioctl_gpu_command) once per command
 * batch: FLD's frame tick. */
void kgsl_fld_submit_tick(struct kgsl_device *device);

/* True while FLD owns the GPU frequency policy; the devfreq target
 * callback consults this to stand aside. */
bool kgsl_fld_own_freq(struct kgsl_device *device);

#else

static inline void kgsl_fld_submit_tick(struct kgsl_device *device) {}
static inline bool kgsl_fld_own_freq(struct kgsl_device *device)
{
	return false;
}

#endif /* CONFIG_FLD_GPU */

#endif /* __KGSL_FLD_H */
