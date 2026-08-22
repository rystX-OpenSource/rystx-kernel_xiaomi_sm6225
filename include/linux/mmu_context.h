/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _LINUX_MMU_CONTEXT_H
#define _LINUX_MMU_CONTEXT_H

#include <asm/mmu_context.h>

struct mm_struct;

void use_mm(struct mm_struct *mm);
void unuse_mm(struct mm_struct *mm);

/* Architectures that care about IRQ state in switch_mm can override this. */
#ifndef switch_mm_irqs_off
# define switch_mm_irqs_off switch_mm
#endif

#ifdef CONFIG_SCHED_MUQSS
/*
 * CPUs that are capable of running task @p. Must contain at least one active
 * CPU. It is assumed that the kernel can run on all CPUs, so calling this for
 * a kernel thread is pointless.
 *
 * By default, we assume a sane, homogeneous system.
 *
 * Backported from mainline b90ca8badbd1 as MuQSS asks for it; no 4.19 arch
 * defines task_cpu_possible_mask, so the default arm is the one that is used.
 * Mainline later grew a task_cpu_fallback_mask() as well - MuQSS does not use
 * it and 4.19's housekeeping_cpumask() spells its flags differently, so it is
 * left out rather than guessed at.
 */
#ifndef task_cpu_possible_mask
# define task_cpu_possible_mask(p)	cpu_possible_mask
# define task_cpu_possible(cpu, p)	true
#else
# define task_cpu_possible(cpu, p)	cpumask_test_cpu((cpu), task_cpu_possible_mask(p))
#endif
#endif /* CONFIG_SCHED_MUQSS */

#endif
