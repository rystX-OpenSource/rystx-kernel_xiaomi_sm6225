/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * FLD - Frame Load Decision GPU scheduler
 *
 * fld_compat.h - Kernel-version compatibility layer.  Every
 * kernel-version-specific access FLD makes goes through here, so the core
 * scheduler code has a single code path from 4.9 through 6.x.  Support
 * claims follow what has been verified in source:
 *
 *   4.19.325 (this tree, Adreno 610) - verified by build and code audit.
 *   4.17+   - util_est path enabled by version check.
 *   4.10+   - arch_scale_cpu_capacity() via <linux/sched/topology.h>.
 *   4.9     - builds with util_est and capacity fallbacks; UNTESTED.
 */

#ifndef _FLD_COMPAT_H
#define _FLD_COMPAT_H

#include <linux/sched.h>
#include <linux/sched/topology.h>
#include <linux/version.h>

/*
 * arm64 defines arch_scale_cpu_capacity to topology_get_cpu_scale()
 * (include/linux/arch_topology.h since 4.14); pull that in so the
 * expansion resolves.  Older and other arches fall back to the generic
 * helpers in <linux/sched/topology.h>.
 */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 14, 0)
#include <linux/arch_topology.h>
#endif

/*
 * Top bit of util_est.enqueued: set while the enqueued value has not
 * changed since the last PELT decay.  Defined inside struct util_est on
 * some trees and in fair.c on others, so FLD carries its own constant.
 * 4.19 value: include/linux/sched.h (struct util_est).
 */
#define FLD_UTIL_AVG_UNCHANGED	0x80000000u

/* PELT fixed-point scale: SCHED_CAPACITY_SCALE == 1 << 10 on every tree. */
#define FLD_CPU_CAP_MAX		1024u

#ifdef CONFIG_SMP

/*
 * Read a fair task's PELT utilization hint locklessly.  The caller must
 * hold a reference on @t (or otherwise guarantee the struct is alive);
 * values may be slightly stale by design.
 *
 * PELT (se.avg) is only maintained for fair-class tasks, so RT/DEADLINE
 * tasks are reported as zero rather than reading stale numbers.
 */
static inline unsigned int fld_task_util(const struct task_struct *t)
{
	unsigned int u;

	if (t->policy != SCHED_NORMAL && t->policy != SCHED_BATCH &&
	    t->policy != SCHED_IDLE)
		return 0;

	u = READ_ONCE(t->se.avg.util_avg);

#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 17, 0)
	/*
	 * util_est.enqueued (mainline 4.17+) reacts sooner to bursts than
	 * the decaying util_avg; prefer it when it is ahead.  Some vendor
	 * 4.9/4.14 trees backport it, but the version gate stays
	 * conservative - see docs/fld-kgsl-hooks.md.
	 */
	{
		unsigned int ue = READ_ONCE(t->se.avg.util_est.enqueued) &
						~FLD_UTIL_AVG_UNCHANGED;

		if (ue > u)
			u = ue;
	}
#endif

	return u;
}

/*
 * Capacity of @cpu in PELT scale (FLD_CPU_CAP_MAX == fastest CPU).  This
 * is what normalizes feeder utilization across the asymmetric clusters
 * (e.g. SM6225: A73-class 1024 vs A53-class 448).
 */
static inline unsigned long fld_cpu_capacity(int cpu)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 10, 0)
	return arch_scale_cpu_capacity(cpu);
#else
	/*
	 * 4.9: the arch override is not uniformly visible outside
	 * kernel/sched/.  Treat every CPU as full capacity - feed-forward
	 * stays functional but loses cross-cluster normalization.
	 * UNTESTED.
	 */
	return FLD_CPU_CAP_MAX;
#endif
}

#else /* !CONFIG_SMP */

/* No PELT on uniprocessor: feed-forward is disabled, GPU feedback alone
 * drives the scheduler. */
static inline unsigned int fld_task_util(const struct task_struct *t)
{
	return 0;
}

static inline unsigned long fld_cpu_capacity(int cpu)
{
	return FLD_CPU_CAP_MAX;
}

#endif /* CONFIG_SMP */

#endif /* _FLD_COMPAT_H */
