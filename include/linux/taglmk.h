/* SPDX-License-Identifier: GPL-2.0 */
/*
 * TAGLMK - Task-aware Android Guided Low Memory Killer
 *
 * Public interface between the TAGLMK driver (drivers/android/taglmk/), the
 * reclaim path (mm/vmscan.c), the OOM path (mm/oom_kill.c) and the per-task
 * memory inspection/reclaim helpers hosted in fs/proc/task_mmu.c.
 *
 * TAGLMK is an independent Android low-memory reclaim-and-kill driver.  It is
 * not derived from and does not depend on any other in-kernel LMK.
 */
#ifndef _LINUX_TAGLMK_H
#define _LINUX_TAGLMK_H

#include <linux/types.h>
#include <linux/errno.h>

struct task_struct;
struct mm_struct;

/*
 * Snapshot of a task's memory footprint, produced by the per-task inspection
 * logic in fs/proc/task_mmu.c and consumed by the driver's policy engine.
 * All counts are in pages.
 */
struct taglmk_mm_stat {
	unsigned long rss_anon;		/* resident anonymous pages */
	unsigned long rss_file;		/* resident file-backed pages */
	unsigned long swap_ents;	/* anon pages already swapped out */
	unsigned long anon_cold;	/* cold anon pages eligible for pageout */
};

#ifdef CONFIG_ANDROID_TAGLMK

/*
 * Latch a memory-pressure event from the reclaim path.  Called from
 * shrink_node() when scanning priority climbs; must be cheap and must not
 * sleep or touch the FPU/NEON (it can run with IRQs disabled).
 */
void taglmk_note_pressure(int order, bool direct_reclaim);

/*
 * True when TAGLMK is the active low-memory killer, i.e. it is enabled and its
 * worker is running.  The OOM path uses this to hand the kill decision to
 * TAGLMK instead of the in-kernel OOM killer; when TAGLMK is disabled at
 * runtime the kernel OOM killer is restored.
 */
bool taglmk_oom_active(void);

/*
 * Per-task memory inspection and guided per-task reclaim.  Implemented in
 * fs/proc/task_mmu.c because they walk the task page tables the same way the
 * smaps/clear_refs machinery there does.  Both must be called from process
 * context (they take mmap_read_lock and may sleep).
 */
int taglmk_mm_inspect(struct task_struct *tsk, struct taglmk_mm_stat *stat);
unsigned long taglmk_reclaim_task_anon(struct task_struct *tsk,
				       unsigned long nr_to_reclaim);

#else /* !CONFIG_ANDROID_TAGLMK */

static inline void taglmk_note_pressure(int order, bool direct_reclaim) { }

static inline bool taglmk_oom_active(void)
{
	return false;
}

static inline int taglmk_mm_inspect(struct task_struct *tsk,
				    struct taglmk_mm_stat *stat)
{
	return -ENOSYS;
}

static inline unsigned long taglmk_reclaim_task_anon(struct task_struct *tsk,
						     unsigned long nr_to_reclaim)
{
	return 0;
}

#endif /* CONFIG_ANDROID_TAGLMK */

#endif /* _LINUX_TAGLMK_H */
