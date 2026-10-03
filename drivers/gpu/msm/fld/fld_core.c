// SPDX-License-Identifier: GPL-2.0-only
/*
 * FLD - Frame Load Decision GPU scheduler
 *
 * fld_core.c - Per-frame decision engine.  FLD runs its own frame timer:
 * every KGSL submit (kgsl_ioctl_gpu_command) hits kgsl_fld_submit_tick(),
 * which timestamps the frame and schedules this work.  The tick combines
 *
 *   - GPU feedback:  busy fraction of the last window, read straight from
 *     the hardware GPU-busy counter via device->ftbl->power_stats(),
 *   - CPU feed-forward: PELT utilization of the submitting threads
 *     (fld_feeder.c), normalized by arch_scale_cpu_capacity(),
 *
 * into an asymmetric-EWMA estimate, snaps it to a KGSL power level with
 * schedutil-style math, and applies it through kgsl_pwrctrl_pwrlevel_change()
 * - which keeps every stock constraint (thermal, user min/max, POPP,
 * context constraints) intact.  It also shapes the KGSL idle timer so the
 * GPU does not power-collapse between frames that are still arriving.
 *
 * While FLD owns the policy (kgsl_fld_own_freq()), the devfreq governor's
 * frequency requests are ignored by the KGSL integration patch; the
 * devfreq machinery itself keeps running untouched.
 *
 * Lock order: fld_dev.lock (leaf) is never held across device->mutex;
 * device->mutex is taken only in the tick work and the sysfs paths.
 */

#define pr_fmt(fmt) "fld: " fmt

#include <linux/atomic.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/kobject.h>
#include <linux/ktime.h>
#include <linux/sched.h>
#include <linux/sysfs.h>
#include <linux/time.h>
#include <linux/workqueue.h>

#include "../kgsl_device.h"
#include "../kgsl_pwrctrl.h"

#include "fld.h"
#include "fld_compat.h"

/* Bounds on the tick window we are willing to reason about. */
#define FLD_MIN_WINDOW_US	500
#define FLD_MAX_WINDOW_US	1000000

/* Watchdog cadence bounds: keeps idle decay/stand-down running without submits. */
#define FLD_WATCHDOG_MIN_MS	8
#define FLD_WATCHDOG_MAX_MS	40

/* Ignore submit-spacing samples that are clearly not consecutive frames. */
#define FLD_PERIOD_MAX_US	200000

struct fld_tunables fld_tun = {
	.up_pct		= 75,
	.down_pct	= 12,
	.target_load_pct = 80,
	.feeder_k_milli	= 1000,
	.down_frames	= 4,
	.idle_timeout_ms = 200,
	.max_idle_ms	= 120,
	.enabled	= 1,
};

static struct workqueue_struct *fld_wq;

static void fld_tick(struct work_struct *w);

static struct fld_dev fld_slots[FLD_MAX_DEVICES] = {
	[0] = {
		.lock	= __SPIN_LOCK_UNLOCKED(fld_slots[0].lock),
	},
	[1] = {
		.lock	= __SPIN_LOCK_UNLOCKED(fld_slots[1].lock),
	},
};

/*
 * Find (or lazily create) the slot for @device.  The first submit of each
 * GPU attaches it; two GPUs racing attach through cmpxchg.  Runs in the
 * submitter's process context.
 */
static struct fld_dev *fld_attach(struct kgsl_device *device)
{
	int i;

	for (i = 0; i < FLD_MAX_DEVICES; i++)
		if (READ_ONCE(fld_slots[i].device) == device)
			return &fld_slots[i];

	for (i = 0; i < FLD_MAX_DEVICES; i++) {
		if (cmpxchg(&fld_slots[i].device, NULL, device) == NULL) {
			pr_info("attached to %s\n", dev_name(device->dev));
			return &fld_slots[i];
		}
	}

	pr_err_ratelimited("no free slot, GPU %s unsupported\n",
			dev_name(device->dev));
	return NULL;
}

bool kgsl_fld_own_freq(struct kgsl_device *device)
{
	int i;

	if (!READ_ONCE(fld_tun.enabled))
		return false;

	for (i = 0; i < FLD_MAX_DEVICES; i++)
		if (READ_ONCE(fld_slots[i].device) == device)
			return true;

	return false;
}
EXPORT_SYMBOL_GPL(kgsl_fld_own_freq);

/* Submit hook: timestamp the frame, record the feeder, schedule a tick. */
void kgsl_fld_submit_tick(struct kgsl_device *device)
{
	struct fld_dev *slot;
	unsigned long flags;
	u64 now, prev;

	if (!READ_ONCE(fld_tun.enabled))
		return;

	slot = fld_attach(device);
	if (!slot)
		return;

	now = ktime_get_ns();

	spin_lock_irqsave(&slot->lock, flags);
	prev = slot->last_submit_ns;
	slot->last_submit_ns = now;
	if (prev) {
		u64 delta_us = (now - prev) / NSEC_PER_USEC;

		if (delta_us && delta_us < FLD_PERIOD_MAX_US) {
			/* Bounded arithmetic EWMA, w = 1/4. */
			u32 sample = delta_us;

			if (sample > slot->period_us)
				slot->period_us += (sample - slot->period_us) / 4;
			else
				slot->period_us -= (slot->period_us - sample) / 4;
		}
	}
	slot->stood_down = false;
	spin_unlock_irqrestore(&slot->lock, flags);

	fld_feeder_observe(current);

	mod_delayed_work(fld_wq, &slot->work, 0);
}
EXPORT_SYMBOL_GPL(kgsl_fld_submit_tick);

/*
 * Restore every lever to its stock position and decay FLD's state.
 * Called with device->mutex held.
 */
static void fld_stand_down(struct fld_dev *slot, struct kgsl_device *device,
			u64 idle_ms)
{
	struct kgsl_pwrctrl *pwr = &device->pwrctrl;
	unsigned long flags;

	if (slot->interval_touched) {
		pwr->interval_timeout = slot->stock_interval_j;
		slot->interval_touched = false;
	}

	slot->ramp.est = 0;
	slot->ramp.calm = 0;

	fld_feeder_flush();

	spin_lock_irqsave(&slot->lock, flags);
	slot->stood_down = true;
	spin_unlock_irqrestore(&slot->lock, flags);

	/*
	 * Drop to the slowest user level - but only if the GPU is truly
	 * quiet.  An ACTIVE GPU with no submits for the timeout is either
	 * already asleep (clocks off, level applies at next wake) or
	 * finishing an unusually long frame.
	 */
	if (device->state == KGSL_STATE_ACTIVE && device->ftbl->power_stats) {
		struct kgsl_power_stats stats;
		u64 window_us = idle_ms * USEC_PER_MSEC;

		device->ftbl->power_stats(device, &stats);
		if (stats.busy_time * 4 < window_us)
			kgsl_pwrctrl_pwrlevel_change(device, pwr->min_pwrlevel);
	}

	fld_dbg("stand down\n");
}

/*
 * Idle-timer lever: while frames are arriving, wait (up to max_idle_ms)
 * for the predicted next frame before letting the GPU power-collapse.
 * Never shortens the stock timeout.  Caller holds device->mutex.
 */
static void fld_shape_idle_timer(struct fld_dev *slot,
				struct kgsl_device *device)
{
	struct kgsl_pwrctrl *pwr = &device->pwrctrl;
	unsigned long flags, stock, want, max_j;
	unsigned int max_idle, period_us;

	max_idle = READ_ONCE(fld_tun.max_idle_ms);

	spin_lock_irqsave(&slot->lock, flags);
	period_us = slot->period_us;
	spin_unlock_irqrestore(&slot->lock, flags);

	if (!max_idle || !period_us || device->state != KGSL_STATE_ACTIVE)
		goto restore;

	stock = slot->stock_interval_j;
	if (!stock)
		goto restore;

	max_j = msecs_to_jiffies(max_idle);
	want = msecs_to_jiffies((period_us / USEC_PER_MSEC) * 2);

	/* Only ever wait longer than stock, and never beyond the ceiling. */
	if (want < stock)
		want = stock;
	if (want > max_j)
		want = max_j;

	if (pwr->interval_timeout != want) {
		pwr->interval_timeout = want;
		slot->interval_touched = (want != stock);
		fld_dbg("idle timer %ums\n", jiffies_to_msecs(want));
	}
	return;

restore:
	if (slot->interval_touched) {
		pwr->interval_timeout = slot->stock_interval_j;
		slot->interval_touched = false;
	}
}

static void fld_tick(struct work_struct *w)
{
	struct fld_dev *slot = container_of(to_delayed_work(w),
					struct fld_dev, work);
	struct kgsl_device *device = READ_ONCE(slot->device);
	struct kgsl_pwrctrl *pwr;
	unsigned long flags;
	u64 now, idle_ns;
	u32 period_us;
	unsigned int wdog_ms = FLD_WATCHDOG_MIN_MS;
	bool active;

	if (!device)
		return;

	mutex_lock(&device->mutex);

	if (!READ_ONCE(fld_tun.enabled)) {
		mutex_unlock(&device->mutex);
		return;
	}

	pwr = &device->pwrctrl;
	now = ktime_get_ns();

	spin_lock_irqsave(&slot->lock, flags);
	idle_ns = now - slot->last_submit_ns;
	period_us = slot->period_us;
	spin_unlock_irqrestore(&slot->lock, flags);

	/* No frames for idle_timeout_ms: restore stock behavior and park. */
	if (idle_ns > (u64)READ_ONCE(fld_tun.idle_timeout_ms) * NSEC_PER_MSEC) {
		fld_stand_down(slot, device,
				READ_ONCE(fld_tun.idle_timeout_ms));
		mutex_unlock(&device->mutex);
		return;
	}

	/* First tick after attach: remember the stock idle timer. */
	if (!slot->stock_interval_j)
		slot->stock_interval_j = pwr->interval_timeout;

	active = (device->state == KGSL_STATE_ACTIVE);

	if (active && device->ftbl->power_stats) {
		struct kgsl_power_stats stats;
		unsigned int freqs[KGSL_MAX_PWRLEVELS];
		u64 win_us, busy_us;
		unsigned int load, feeder;
		int n, i, level;

		/* Hardware busy counter delta; clocks are on while ACTIVE. */
		device->ftbl->power_stats(device, &stats);
		busy_us = stats.busy_time;

		spin_lock_irqsave(&slot->lock, flags);
		win_us = (now - slot->last_tick_ns) / NSEC_PER_USEC;
		slot->last_tick_ns = now;
		spin_unlock_irqrestore(&slot->lock, flags);

		if (win_us >= FLD_MIN_WINDOW_US && win_us <= FLD_MAX_WINDOW_US) {
			load = (busy_us >= win_us) ? FLD_LOAD_SCALE :
				(unsigned int)(busy_us * FLD_LOAD_SCALE / win_us);
			feeder = fld_feeder_load();

			/* Snapshot the runtime table (index 0 = fastest). */
			n = pwr->num_pwrlevels;
			if (n > KGSL_MAX_PWRLEVELS)
				n = KGSL_MAX_PWRLEVELS;
			for (i = 0; i < n; i++)
				freqs[i] = pwr->pwrlevels[i].gpu_freq;

			level = fld_decide(&slot->ramp, load, feeder,
					READ_ONCE(fld_tun.up_pct),
					READ_ONCE(fld_tun.down_pct),
					READ_ONCE(fld_tun.feeder_k_milli),
					READ_ONCE(fld_tun.target_load_pct),
					READ_ONCE(fld_tun.down_frames),
					pwr->active_pwrlevel,
					freqs, n,
					pwr->max_pwrlevel, pwr->min_pwrlevel);

			if (level >= 0 && level != pwr->active_pwrlevel) {
				fld_dbg("load %u feeder %u -> level %d\n",
					load, feeder, level);
				kgsl_pwrctrl_pwrlevel_change(device, level);
			}

			slot->last_load = load;
			slot->last_feeder = feeder;

			fld_shape_idle_timer(slot, device);
		}
	} else {
		/* GPU asleep: decay the estimate toward zero. */
		slot->ramp.est = fld_ewma(slot->ramp.est, 0,
					READ_ONCE(fld_tun.up_pct),
					READ_ONCE(fld_tun.down_pct));
	}

	mutex_unlock(&device->mutex);

	/*
	 * Keep the watchdog alive so idle decay and stand-down run even
	 * when submits stop.  A queued submit tick (immediate) wins over
	 * the periodic re-arm.
	 */
	if (period_us / USEC_PER_MSEC / 2 > wdog_ms) {
		wdog_ms = period_us / USEC_PER_MSEC / 2;
		if (wdog_ms > FLD_WATCHDOG_MAX_MS)
			wdog_ms = FLD_WATCHDOG_MAX_MS;
	}

	if (!delayed_work_pending(&slot->work))
		queue_delayed_work(fld_wq, &slot->work,
				msecs_to_jiffies(wdog_ms));
}

/* ---- sysfs: /sys/kernel/fld ---- */

static ssize_t fld_tun_show(unsigned int val, char *buf)
{
	return scnprintf(buf, PAGE_SIZE, "%u\n", val);
}

static ssize_t fld_tun_store(const char *buf, unsigned int *field,
				unsigned int min, unsigned int max)
{
	unsigned int val;
	int ret;

	ret = kstrtouint(buf, 0, &val);
	if (ret)
		return ret;
	if (val < min || val > max)
		return -EINVAL;

	WRITE_ONCE(*field, val);
	return 0;
}

#define FLD_TUN_ATTR(_name, _field, _min, _max)				\
static ssize_t _name##_show(struct kobject *kobj,			\
			struct kobj_attribute *attr, char *buf)		\
{									\
	return fld_tun_show(READ_ONCE(fld_tun._field), buf);		\
}									\
static ssize_t _name##_store(struct kobject *kobj,			\
			struct kobj_attribute *attr, const char *buf,	\
			size_t count)					\
{									\
	int ret = fld_tun_store(buf, &fld_tun._field, _min, _max);	\
	return ret ? ret : count;					\
}									\
static struct kobj_attribute dev_attr_##_name =			\
	__ATTR(_name, 0644, _name##_show, _name##_store)

FLD_TUN_ATTR(up_weight, up_pct, 1, 100);
FLD_TUN_ATTR(down_weight, down_pct, 1, 100);
FLD_TUN_ATTR(target_load, target_load_pct, 20, 100);
FLD_TUN_ATTR(feeder_k, feeder_k_milli, 0, 4000);
FLD_TUN_ATTR(down_frames, down_frames, 1, 64);
FLD_TUN_ATTR(idle_timeout_ms, idle_timeout_ms, 20, 10000);
FLD_TUN_ATTR(max_idle_ms, max_idle_ms, 0, 1000);

static ssize_t enabled_show(struct kobject *kobj, struct kobj_attribute *attr,
			char *buf)
{
	return fld_tun_show(READ_ONCE(fld_tun.enabled), buf);
}

static ssize_t enabled_store(struct kobject *kobj, struct kobj_attribute *attr,
			const char *buf, size_t count)
{
	unsigned int val;
	int ret, i;

	ret = kstrtouint(buf, 0, &val);
	if (ret)
		return ret;
	if (val > 1)
		return -EINVAL;

	WRITE_ONCE(fld_tun.enabled, val);

	if (!val) {
		/*
		 * Hand the policy back: park the ticks and restore any
		 * shaped idle timer so stock KGSL is exactly as it was.
		 */
		for (i = 0; i < FLD_MAX_DEVICES; i++) {
			struct fld_dev *slot = &fld_slots[i];
			struct kgsl_device *device = READ_ONCE(slot->device);

			cancel_delayed_work_sync(&slot->work);

			if (device && slot->interval_touched) {
				mutex_lock(&device->mutex);
				device->pwrctrl.interval_timeout =
						slot->stock_interval_j;
				slot->interval_touched = false;
				mutex_unlock(&device->mutex);
			}

			fld_feeder_flush();
		}
	}

	return count;
}

static struct kobj_attribute dev_attr_enabled =
	__ATTR(enabled, 0644, enabled_show, enabled_store);

static ssize_t status_show(struct kobject *kobj, struct kobj_attribute *attr,
			char *buf)
{
	struct fld_dev *slot = &fld_slots[0];
	struct kgsl_device *device = READ_ONCE(slot->device);
	unsigned long flags;
	u64 last_submit;
	u32 period_us;
	bool stood_down;
	unsigned int est, load, feeder;

	spin_lock_irqsave(&slot->lock, flags);
	last_submit = slot->last_submit_ns;
	period_us = slot->period_us;
	stood_down = slot->stood_down;
	spin_unlock_irqrestore(&slot->lock, flags);

	est = READ_ONCE(slot->ramp.est);
	load = READ_ONCE(slot->last_load);
	feeder = READ_ONCE(slot->last_feeder);

	if (!device)
		return scnprintf(buf, PAGE_SIZE, "attached: no\n");

	/* pwrctrl fields are read locklessly: status is a hint, not a vote. */
	return scnprintf(buf, PAGE_SIZE,
		"attached: yes\n"
		"state: %s\n"
		"own_freq: %s\n"
		"active_pwrlevel: %u (%u Hz)\n"
		"min/max levels: %u/%u\n"
		"thermal_pwrlevel: %u\n"
		"est: %u\n"
		"last_load: %u\n"
		"feeder: %u (n=%d)\n"
		"period_us: %u\n"
		"age_us: %llu\n"
		"idle_timer: %ums (stock %ums, touched %d)\n"
		"stood_down: %d\n",
		(device->state == KGSL_STATE_ACTIVE) ? "active" : "idle",
		kgsl_fld_own_freq(device) ? "yes" : "no",
		device->pwrctrl.active_pwrlevel,
		device->pwrctrl.pwrlevels[device->pwrctrl.active_pwrlevel].gpu_freq,
		device->pwrctrl.max_pwrlevel, device->pwrctrl.min_pwrlevel,
		device->pwrctrl.thermal_pwrlevel,
		est, load, feeder, fld_feeder_count(),
		period_us,
		(last_submit ? (ktime_get_ns() - last_submit) / NSEC_PER_USEC
				: 0),
		jiffies_to_msecs(device->pwrctrl.interval_timeout),
		jiffies_to_msecs(slot->stock_interval_j),
		(int)slot->interval_touched,
		(int)stood_down);
}

static struct kobj_attribute dev_attr_status =
	__ATTR_RO(status);

static struct attribute *fld_attrs[] = {
	&dev_attr_enabled.attr,
	&dev_attr_up_weight.attr,
	&dev_attr_down_weight.attr,
	&dev_attr_target_load.attr,
	&dev_attr_feeder_k.attr,
	&dev_attr_down_frames.attr,
	&dev_attr_idle_timeout_ms.attr,
	&dev_attr_max_idle_ms.attr,
	&dev_attr_status.attr,
	NULL,
};

static const struct attribute_group fld_attr_group = {
	.attrs = fld_attrs,
};

static int __init fld_init(void)
{
	struct kobject *kobj;
	int i, ret;

	/*
	 * Single-threaded, high priority: tick decisions stay ordered and
	 * frame-latency sensitive.
	 */
	fld_wq = alloc_workqueue("kgsl_fld", WQ_HIGHPRI | WQ_UNBOUND, 1);
	if (!fld_wq)
		return -ENOMEM;

	for (i = 0; i < FLD_MAX_DEVICES; i++)
		INIT_DELAYED_WORK(&fld_slots[i].work, fld_tick);

	kobj = kobject_create_and_add("fld", kernel_kobj);
	if (!kobj) {
		destroy_workqueue(fld_wq);
		return -ENOMEM;
	}

	ret = sysfs_create_group(kobj, &fld_attr_group);
	if (ret) {
		kobject_put(kobj);
		destroy_workqueue(fld_wq);
		return ret;
	}

	pr_info("FLD GPU scheduler ready (up=%u%% down=%u%% target=%u%%)\n",
		fld_tun.up_pct, fld_tun.down_pct, fld_tun.target_load_pct);
	return 0;
}
subsys_initcall(fld_init);
