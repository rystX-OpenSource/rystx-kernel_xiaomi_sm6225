/* SPDX-License-Identifier: GPL-2.0 */
/*
 * TAGLMK - Task-aware Android Guided Low Memory Killer
 *
 * Interface shared between the independent Android reclaim driver, which
 * lives in fs/proc/task_mmu.c next to the rest of the page table walkers,
 * and the guided killer core in drivers/android/taglmk/.
 *
 * Copyright (C) 2026 iDeadXS <datarafi43@gmail.com>
 */
#ifndef _LINUX_TAGLMK_H
#define _LINUX_TAGLMK_H

#include <linux/errno.h>
#include <linux/types.h>

struct task_struct;

/**
 * enum taglmk_task_type - guided classification of an Android task
 * @TAGLMK_TYPE_APP: An ordinary, user launchable application.  Cheapest to
 *	kill and the only class considered while memory pressure is merely low.
 * @TAGLMK_TYPE_PINNED: An application whose package name the user placed on
 *	the pin list.  Identical to %TAGLMK_TYPE_APP in every respect except
 *	survivability: it is only considered once no plain application is left.
 * @TAGLMK_TYPE_SYSTEM_APP: A platform component - system_server, persistent
 *	apps, anything running as a platform uid.  Considered only once the
 *	situation is memory critical.
 * @TAGLMK_TYPE_CRITICAL: An Android system core.  Never killed by TAGLMK.
 * @TAGLMK_TYPE_COUNT: Number of classes, not a class itself.
 *
 * The numeric order is load bearing: a task is eligible for the killer only
 * while its type compares strictly below the cut-off derived from the current
 * pressure level, so the enum doubles as the escalation ladder.  Keep the
 * values ordered from cheapest to most expensive to kill.
 */
enum taglmk_task_type {
	TAGLMK_TYPE_APP,
	TAGLMK_TYPE_PINNED,
	TAGLMK_TYPE_SYSTEM_APP,
	TAGLMK_TYPE_CRITICAL,

	TAGLMK_TYPE_COUNT
};

/**
 * enum taglmk_reclaim_type - which pages the reclaim driver should act upon
 * @TAGLMK_RECLAIM_FILE: Only file backed mappings.  File pages are pushed to
 *	the inactive list rather than discarded, so the file LRU stays large
 *	and neither vmscan nor lmkd sees an artificial cliff.
 * @TAGLMK_RECLAIM_ANON: Only anonymous mappings.  Those are swapped out, which
 *	on Android means they are compressed into ZRAM.
 * @TAGLMK_RECLAIM_ALL: Both of the above, each handled as described.
 */
enum taglmk_reclaim_type {
	TAGLMK_RECLAIM_FILE,
	TAGLMK_RECLAIM_ANON,
	TAGLMK_RECLAIM_ALL,
};

/**
 * struct taglmk_reclaim_stat - outcome of one pass over an address space
 * @nr_scanned: Pages examined by the walker and found to be candidates.
 * @nr_reclaimed: Anonymous pages actually handed back to the allocator.
 * @nr_deactivated: File pages moved from the active to the inactive list.
 */
struct taglmk_reclaim_stat {
	unsigned long nr_scanned;
	unsigned long nr_reclaimed;
	unsigned long nr_deactivated;
};

#ifdef CONFIG_ANDROID_TAGLMK

/**
 * taglmk_reclaim_mm - reclaim from a task's address space
 * @tsk: Task to reclaim from.  The caller must hold a reference to it.
 * @type: Which mappings to act upon.
 * @nr_to_reclaim: Stop once this many pages have been reclaimed, or walk the
 *	whole address space when zero.
 * @stat: Filled in with the outcome.  May be %NULL.
 *
 * Return: 0 on success, or a negative errno.  A task that has already exited
 * is not an error and simply reclaims nothing.
 */
int taglmk_reclaim_mm(struct task_struct *tsk, enum taglmk_reclaim_type type,
		      unsigned long nr_to_reclaim,
		      struct taglmk_reclaim_stat *stat);

#else /* !CONFIG_ANDROID_TAGLMK */

static inline int taglmk_reclaim_mm(struct task_struct *tsk,
				    enum taglmk_reclaim_type type,
				    unsigned long nr_to_reclaim,
				    struct taglmk_reclaim_stat *stat)
{
	return -ENOSYS;
}

#endif /* CONFIG_ANDROID_TAGLMK */

#endif /* _LINUX_TAGLMK_H */
