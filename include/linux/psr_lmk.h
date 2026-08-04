/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _LINUX_PSR_LMK_H
#define _LINUX_PSR_LMK_H

#include <linux/types.h>

struct mm_struct;

/*
 * PSR-LMK core hook API.
 *
 * Everything here is a corroboration signal only: a per-CPU counter bump
 * behind a static key. None of these hooks take a lock, read a timer,
 * walk the task list, or influence what the reclaimer does -- they are
 * pure statistics. The actual regression evaluation, victim selection
 * and kill dispatch all happen on the psr_lmkd kthread, woken by the
 * existing global vmpressure notifier chain (see drivers/android/psr_lmk.c).
 *
 * Because these sit in genuinely hot mm paths (__activate_page() runs
 * under the LRU lock, workingset_refault() runs on every refault), the
 * cost of the disabled case has to be *zero*, not "small". Hence the
 * static key: until the driver is up, each hook is a patched-out NOP
 * rather than a load-and-test. Once up, it is a single non-atomic
 * per-CPU increment -- no shared cacheline, so no cross-CPU bouncing.
 */

#ifdef CONFIG_ANDROID_PSR_LMK

#include <linux/jump_label.h>

DECLARE_STATIC_KEY_FALSE(psr_lmk_key);

void __psr_lmk_note_anon_reactivation(void);
void __psr_lmk_note_refault(bool is_swap);
void __psr_lmk_note_alloc_failure(void);
void __psr_lmk_mm_freed(struct mm_struct *mm);

/*
 * mm/swap.c: an anon (swap-backed) page was just reactivated from the
 * inactive to the active LRU -- memory that was a swap-out candidate is
 * being pulled back into active use. Core "protected-swap regression"
 * signal, hence the driver's name.
 */
static inline void psr_lmk_note_anon_reactivation(void)
{
	if (static_branch_unlikely(&psr_lmk_key))
		__psr_lmk_note_anon_reactivation();
}

/*
 * mm/workingset.c: a page just refaulted after being evicted. @is_swap
 * distinguishes an anon/swap-backed refault (real swap thrash -- what
 * PSR-LMK cares about) from a file-cache refault (page-cache churn,
 * tracked but weighted differently).
 */
static inline void psr_lmk_note_refault(bool is_swap)
{
	if (static_branch_unlikely(&psr_lmk_key))
		__psr_lmk_note_refault(is_swap);
}

/*
 * mm/page_alloc.c: direct reclaim finished but the allocation still
 * failed. Slow path only, so this one is never hot.
 */
static inline void psr_lmk_note_alloc_failure(void)
{
	if (static_branch_unlikely(&psr_lmk_key))
		__psr_lmk_note_alloc_failure();
}

/*
 * kernel/fork.c (__mmput): a victim's address space has finished
 * exit_mmap(), i.e. its memory is actually back. This is what closes
 * PSR-LMK's pending-kill gate; without it the gate can only be closed by
 * a timeout, which either releases it too early (overkill) or holds it
 * too long (under-kill). Cheap: an unlocked bit test on the dying mm,
 * which is false for every non-victim process exit.
 */
static inline void psr_lmk_mm_freed(struct mm_struct *mm)
{
	if (static_branch_unlikely(&psr_lmk_key))
		__psr_lmk_mm_freed(mm);
}

#else /* !CONFIG_ANDROID_PSR_LMK */

static inline void psr_lmk_note_anon_reactivation(void) {}
static inline void psr_lmk_note_refault(bool is_swap) {}
static inline void psr_lmk_note_alloc_failure(void) {}
static inline void psr_lmk_mm_freed(struct mm_struct *mm) {}

#endif /* CONFIG_ANDROID_PSR_LMK */

#endif /* _LINUX_PSR_LMK_H */
