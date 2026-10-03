/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * FLD - Frame Load Decision GPU scheduler
 *
 * fld.h - Shared types and the interface between the KGSL hooks, the core
 * scheduler (fld_core.c) and the PELT feeder (fld_feeder.c).
 */

#ifndef _FLD_H
#define _FLD_H

#include <linux/spinlock.h>
#include <linux/types.h>
#include <linux/workqueue.h>

#include "fld_math.h"

struct kgsl_device;
struct task_struct;

/*
 * KGSL-facing hooks (implemented in fld_core.c, called from the KGSL
 * integration patches; no-op compile-outs live in kgsl_fld.h).
 */
void kgsl_fld_submit_tick(struct kgsl_device *device);
bool kgsl_fld_own_freq(struct kgsl_device *device);

/* Feeder interface (fld_feeder.c). */
void fld_feeder_observe(struct task_struct *t);
unsigned int fld_feeder_load(void);
int fld_feeder_count(void);
void fld_feeder_flush(void);

/*
 * Tunables.  Written from sysfs with WRITE_ONCE after range validation,
 * read from the tick path with READ_ONCE.  No lock needed: all fields are
 * u32 and order-independent.
 */
struct fld_tunables {
	unsigned int up_pct;		/* EWMA rise weight, %          */
	unsigned int down_pct;		/* EWMA decay weight, %         */
	unsigned int target_load_pct;	/* OPP utilization target, %    */
	unsigned int feeder_k_milli;	/* feed-forward gain, x1000     */
	unsigned int down_frames;	/* calm frames before ramp-down */
	unsigned int idle_timeout_ms;	/* no-submit time before stand-down */
	unsigned int max_idle_ms;	/* idle-timer lever ceiling, 0 = off */
	unsigned int enabled;		/* master switch                */
};

extern struct fld_tunables fld_tun;

/*
 * Per-GPU state.  Phones have a single GPU; FLD_MAX_DEVICES slots keep the
 * design generic.  Lock order: fld_dev.lock is a short, leaf spinlock used
 * by the submit hook; the tick work takes it briefly and ALWAYS releases
 * it before taking device->mutex.  Nothing may take device->mutex while
 * holding fld_dev.lock.
 */
#define FLD_MAX_DEVICES 2

struct fld_dev {
	struct kgsl_device *device;
	struct delayed_work work;
	spinlock_t lock;

	/* guarded by lock */
	u64 last_submit_ns;
	u64 last_tick_ns;
	u32 period_us;		/* EWMA of submit spacing */
	bool stood_down;

	/* tick-work private (work is serialized, wq max_active == 1) */
	struct fld_ramp ramp;
	unsigned int last_load;
	unsigned int last_feeder;
	unsigned long stock_interval_j;
	bool interval_touched;
};

#ifdef CONFIG_FLD_GPU_DEBUG
#define fld_dbg(fmt, ...)	pr_info_ratelimited(fmt, ##__VA_ARGS__)
#else
#define fld_dbg(fmt, ...)	pr_debug(fmt, ##__VA_ARGS__)
#endif

#endif /* _FLD_H */
