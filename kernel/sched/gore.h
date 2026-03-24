/* SPDX-License-Identifier: GPL-2.0 */
/*
 * GoreScheduler internal header — Linux 4.19 backport
 * Combines BORE + CacULE + TT + ECHO algorithms.
 */
#ifndef _SCHED_GORE_H
#define _SCHED_GORE_H

#ifdef CONFIG_GORE_SCHED

#include <linux/sched.h>
#include "sched.h"

/* ---- Task-Type constants (TT-derived) ---- */
#define GORE_TT_REALTIME	0
#define GORE_TT_INTERACTIVE	1
#define GORE_TT_NO_TYPE		2
#define GORE_TT_CPU_BOUND	3
#define GORE_TT_BATCH		4

/* ---- Type detection thresholds (TT) ---- */
#define GORE_RT_WAIT_DELTA	800000ULL	/* 0.8 ms */
#define GORE_RT_BURST_DELTA	2000000ULL	/* 2 ms   */
#define GORE_RT_BURST_MAX	4000000ULL	/* 4 ms   */
#define GORE_RT_STICKY_TICKS	4
#define GORE_INTERACTIVE_HRRN	2U

/* ---- BORE burst constants ---- */
#define GORE_MAX_BURST_PENALTY		((40U << 8) - 1)

/* ---- ECHO EST constants ---- */
#define GORE_EST_SHIFT			12	/* scale EST into hrrn range */

/* ---- Score composition constants (tuneable via sysctl) ---- */
#define GORE_BURST_WEIGHT_DEFAULT	25
#define GORE_STARVE_DIV_DEFAULT		3000000ULL	/* 3 ms  */
#define GORE_EST_ALPHA_DEFAULT		250		/* 0.25 */

/* ---- Yield mark (CacULE) ---- */
#define GORE_YIELD_MARK		0x8000000000000000ULL
#define GORE_YIELD_UNMARK	0x7FFFFFFFFFFFFFFFULL

/* ---- Sysctl-exported tunables ---- */
extern int  __read_mostly gore_enabled;
extern unsigned int __read_mostly gore_max_lifetime_ms;
extern unsigned int __read_mostly gore_burst_penalty_offset;
extern unsigned int __read_mostly gore_burst_penalty_scale;
extern unsigned int __read_mostly gore_burst_smoothness;
extern unsigned int __read_mostly gore_est_alpha;
extern unsigned int __read_mostly gore_burst_weight;
extern unsigned int __read_mostly gore_starve_div;
extern int  __read_mostly gore_lat_sens_enabled;
extern int  __read_mostly gore_dedicated_cpu_enabled;

DECLARE_PER_CPU(int, gore_nr_lat_sensitive);

/* ---- Functions exported to fair.c ---- */
extern void gore_init_entity(struct sched_entity *se, u64 now);
extern void gore_task_fork(struct task_struct *p);
extern void gore_update_curr(struct cfs_rq *cfs_rq,
			     struct sched_entity *curr, u64 delta_exec, u64 now);
extern void gore_enqueue_entity(struct cfs_rq *cfs_rq,
				struct sched_entity *se, int flags, u64 now);
extern void gore_dequeue_entity(struct cfs_rq *cfs_rq,
				struct sched_entity *se, int flags, u64 now);
extern struct sched_entity *gore_pick_next_entity(struct cfs_rq *cfs_rq,
						   struct sched_entity *curr);
extern bool gore_check_preempt(struct cfs_rq *cfs_rq,
			       struct sched_entity *curr,
			       struct sched_entity *p);
extern void gore_yield_entity(struct sched_entity *se);
extern void gore_init_cfs_rq(struct cfs_rq *cfs_rq);
extern void __init sched_init_gore(void);
extern int  gore_effective_prio(struct task_struct *p);

/* ---- Called from core.c ---- */
extern void gore_inc_nr_lat_sensitive(int cpu);
extern void gore_dec_nr_lat_sensitive(int cpu);

static inline bool task_is_gore_lat_sensitive(struct task_struct *p)
{
	struct gore_node *gn = &p->se.gore_node;
	return gn->task_type <= GORE_TT_INTERACTIVE;
}

static inline struct sched_entity *gore_se_of(struct gore_node *gn)
{
	return container_of(gn, struct sched_entity, gore_node);
}

#endif /* CONFIG_GORE_SCHED */
#endif /* _SCHED_GORE_H */