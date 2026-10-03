// SPDX-License-Identifier: GPL-2.0-only
/*
 * FLD - Frame Load Decision GPU scheduler
 *
 * fld_feeder.c - CPU feed-forward.  FLD tracks the tasks that submit GPU
 * work (identified at the KGSL submit hook as `current`, never by name or
 * cgroup) and polls their PELT utilization from the frame tick.  This is
 * "pull, not hook": no scheduler hooks of any kind are used.
 */

#include <linux/jiffies.h>
#include <linux/sched.h>
#include <linux/sched/task.h>
#include <linux/spinlock.h>

#include "fld.h"
#include "fld_compat.h"

#define FLD_FEEDER_MAX		16
#define FLD_FEEDER_STALE_MS	200

static struct task_struct *feeder_task[FLD_FEEDER_MAX];
static unsigned long feeder_seen[FLD_FEEDER_MAX];
static int feeder_nr;
static DEFINE_SPINLOCK(fld_feeder_lock);

/*
 * Record @t as a recent GPU-feeding thread.  Callable from the submit hook
 * (process context); the task pointer is pinned with a reference for as
 * long as it stays in the ring, so the tick can read PELT state safely
 * even if the thread exits.
 */
void fld_feeder_observe(struct task_struct *t)
{
	struct task_struct *evict = NULL;
	unsigned long flags;
	bool stored = false;
	int i, oldest = 0;

	get_task_struct(t);

	spin_lock_irqsave(&fld_feeder_lock, flags);

	for (i = 0; i < feeder_nr; i++) {
		if (feeder_task[i] == t) {
			feeder_seen[i] = jiffies;
			break;
		}
		if (time_before(feeder_seen[i], feeder_seen[oldest]))
			oldest = i;
	}

	if (i == feeder_nr) {
		if (feeder_nr < FLD_FEEDER_MAX) {
			feeder_task[feeder_nr] = t;
			feeder_seen[feeder_nr] = jiffies;
			feeder_nr++;
			stored = true;
		} else {
			evict = feeder_task[oldest];
			feeder_task[oldest] = t;
			feeder_seen[oldest] = jiffies;
			stored = true;
		}
	}

	spin_unlock_irqrestore(&fld_feeder_lock, flags);

	if (evict)
		put_task_struct(evict);
	if (!stored)
		put_task_struct(t);
}

/* PELT utilization of one feeder, normalized to its running CPU's capacity. */
static unsigned int fld_feeder_util_scaled(struct task_struct *t)
{
	unsigned long cap;
	unsigned int u;

	u = fld_task_util(t);
	if (u == 0)
		return 0;

	cap = fld_cpu_capacity(task_cpu(t));
	if (cap == 0)
		return 0;

	if (u > FLD_LOAD_SCALE)
		u = FLD_LOAD_SCALE;

	return (u * FLD_LOAD_SCALE) / cap;
}

/*
 * Capacity-normalized sum of the current feeder set, in FLD_LOAD_SCALE
 * fixed point (1024 == one saturated core at whatever capacity it has).
 * Called only from the tick work.  Stale feeders are evicted here, so the
 * ring self-cleans as soon as the GPU goes quiet.
 */
unsigned int fld_feeder_load(void)
{
	struct task_struct *evict[FLD_FEEDER_MAX];
	unsigned long flags, total = 0;
	int n_evict = 0, i, w;

	spin_lock_irqsave(&fld_feeder_lock, flags);

	w = 0;
	for (i = 0; i < feeder_nr; i++) {
		struct task_struct *t = feeder_task[i];

		if (time_after(jiffies, feeder_seen[i] +
				msecs_to_jiffies(FLD_FEEDER_STALE_MS))) {
			evict[n_evict++] = t;
			continue;
		}

		total += fld_feeder_util_scaled(t);

		feeder_task[w] = feeder_task[i];
		feeder_seen[w] = feeder_seen[i];
		w++;
	}
	feeder_nr = w;

	spin_unlock_irqrestore(&fld_feeder_lock, flags);

	while (n_evict)
		put_task_struct(evict[--n_evict]);

	return (total > FLD_LOAD_MAX) ? FLD_LOAD_MAX : total;
}

int fld_feeder_count(void)
{
	return feeder_nr;
}

/* Drop every tracked feeder (idle stand-down, FLD disable). */
void fld_feeder_flush(void)
{
	struct task_struct *evict[FLD_FEEDER_MAX];
	unsigned long flags;
	int n_evict = 0;

	spin_lock_irqsave(&fld_feeder_lock, flags);
	while (feeder_nr)
		evict[n_evict++] = feeder_task[--feeder_nr];
	spin_unlock_irqrestore(&fld_feeder_lock, flags);

	while (n_evict)
		put_task_struct(evict[--n_evict]);
}
