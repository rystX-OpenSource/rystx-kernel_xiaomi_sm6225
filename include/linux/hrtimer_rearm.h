/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Lazy hrtimer rearm interface.
 *
 * MuQSS arms its high resolution preemption timer (hrexpiry) from inside
 * __schedule() and then immediately cancels or reprograms it again once it
 * knows which task it picked. On the -ck kernels this is paired with a
 * hrtimer core extension which lets a HRTIMER_MODE_LAZY_REARM timer skip
 * reprogramming the clock event device until the deferred rearm point,
 * turning that arm/cancel/arm sequence into a single hardware program.
 *
 * That core extension is not part of the MuQSS patch itself, so on this tree
 * the interface degrades to a no-op: MuQSS still defers its own start/cancel
 * decisions until hrexpiry_schedule_exit(), it just does not get the extra
 * clockevent reprogramming elision. Behaviour is unchanged, only the number
 * of clockevent programs differs.
 */
#ifndef _LINUX_HRTIMER_REARM_H
#define _LINUX_HRTIMER_REARM_H

#include <linux/hrtimer.h>

/*
 * Mode bit requesting deferred rearm. Zero here so that
 * HRTIMER_MODE_REL_HARD | HRTIMER_MODE_LAZY_REARM stays a valid mode.
 */
#define HRTIMER_MODE_LAZY_REARM	((enum hrtimer_mode)0)

static inline bool hrtimer_test_and_clear_rearm_deferred(void)
{
	return false;
}

static inline void __hrtimer_rearm_deferred(void)
{
}

static inline void hrtimer_rearm_deferred(void)
{
}

#endif /* _LINUX_HRTIMER_REARM_H */
