// SPDX-License-Identifier: GPL-2.0
/*
 * System-wide state for the multi-level feedback queue classifier.
 *
 * Ported from scx_mlfq, a sched_ext scheduler by galpt:
 *   https://github.com/galpt/scx_mlfq
 *
 * Everything in mlfq.h and mlfq_classify.c decides one task's level from that
 * task's own history. This file holds the state that spans tasks: the event
 * counters, the per-runqueue record of how many tasks of each level are
 * waiting, and the per-runqueue pool each level reclaims unused budget through.
 * scx_mlfq keeps the same split, with its per-CPU state in main.bpf.c and only
 * its classification in classify.bpf.c.
 *
 * The per-CPU state is written from the CPU that owns it, under that runqueue's
 * lock, so none of it costs a shared cacheline on any scheduling path. Readers
 * sum across CPUs and are not serialised against the writers;
 * kernel/sched/mlfq_stats.c is the only reader and says what that means for
 * what it prints. The reclaim pools are the one part of this that is read back
 * on a scheduling path rather than only by a reader, and the runqueue lock that
 * keeps the rest cheap is what makes those exact.
 */
#include "sched.h"
#include "mlfq.h"

DEFINE_PER_CPU(struct mlfq_pcpu, mlfq_pcpu);
