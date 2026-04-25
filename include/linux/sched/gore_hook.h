/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Gore Hook API — external interface for GoreScheduler v1.2
 * include/linux/sched/gore_hook.h
 *
 * Allows kernel subsystems to push scheduling hints into a task's
 * gore_node without modifying fair.c.
 *
 * v1.2 scope: Android binder and futex hooks ONLY.
 * All functions are no-ops when CONFIG_GORE_SCHED=n — callers need
 * no #ifdef guards of their own.
 *
 * Thread safety: all public functions are safe from any context.
 * Internally uses atomic_set / READ_ONCE / WRITE_ONCE — no locks.
 */
#ifndef _LINUX_GORE_HOOK_H
#define _LINUX_GORE_HOOK_H

#include <linux/sched.h>
#include <linux/types.h>
#include <linux/atomic.h>

#ifdef CONFIG_GORE_SCHED

/* ================================================================
 * Hint flags (written to gore_node.hint_flags)
 * ================================================================ */

/*
 * GORE_HINT_BINDER_SERVER
 * Task is handling an incoming binder transaction.
 * Temporarily promotes to INTERACTIVE tier for the IPC duration.
 */
#define GORE_HINT_BINDER_SERVER        0x01U

/*
 * GORE_HINT_BINDER_REPLY
 * Task just sent a binder reply and will likely sleep.
 * Clears the binder promotion on the next classify pass.
 */
#define GORE_HINT_BINDER_REPLY     0x02U

/*
 * GORE_HINT_FUTEX_WAKEUP
 * Task was woken via futex_wake() by a high-priority task.
 * Receives a brief score_bias boost as wakeup inheritance.
 */
#define GORE_HINT_FUTEX_WAKEUP     0x04U

/* ================================================================
 * Score bias constants
 *
 * Subtracted from gore_score() result for the boosted task.
 * All values < GORE_TIER_SCALE (100000) so tier ordering is
 * never overridden by a boost.
 * ================================================================ */
#define GORE_BOOST_BINDER      40000   /* binder server transaction    */
#define GORE_BOOST_FUTEX       25000   /* futex wakeup inheritance     */

/* ================================================================
 * Boost duration constants (nanoseconds)
 * ================================================================ */
#define GORE_BOOST_DUR_BINDER      2000000ULL  /* 2 ms — IPC RTT    */
#define GORE_BOOST_DUR_FUTEX       1000000ULL  /* 1 ms — one frame  */

/* ================================================================
 * Public API — implementations in kernel/sched/fair.c
 * ================================================================ */

/**
 * gore_apply_boost — apply a timed score bonus to a task
 *
 * @task:        target task_struct (must not be NULL)
 * @boost:       points to subtract from gore_score() (capped internally)
 * @duration_ns: nanoseconds the boost remains active
 *
 * The bias decays to zero after duration_ns elapsed.  Decay is
 * checked lazily in gore_score() — no timer overhead.
 */
void gore_apply_boost(struct task_struct *task, int boost, u64 duration_ns);

/**
 * gore_lock_type — pin task classification, bypass auto-detect
 *
 * @task:      target task_struct
 * @task_type: GORE_REALTIME / GORE_INTERACTIVE / GORE_NO_TYPE /
 * GORE_CPU_BOUND / GORE_BATCH
 *
 * Sets gore_node.type_locked = true.  gore_detect_type() skips this
 * task until gore_unlock_type() is called.
 */
void gore_lock_type(struct task_struct *task, unsigned int task_type);

/**
 * gore_unlock_type — resume auto-classification
 * @task: task previously passed to gore_lock_type()
 */
void gore_unlock_type(struct task_struct *task);

/**
 * gore_binder_wakeup — called when a binder server thread is selected
 *
 * @server: the binder server task being woken to handle a transaction
 * @client: the calling task (informational, for future inheritance)
 *
 * Applies GORE_HINT_BINDER_SERVER: temporarily locks server to
 * GORE_INTERACTIVE and gives it GORE_BOOST_BINDER so it can
 * preempt BATCH/CPU_BOUND work while serving the transaction.
 */
void gore_binder_wakeup(struct task_struct *server,
           struct task_struct *client);

/**
 * gore_binder_done — called when a binder transaction reply completes
 *
 * @server: the binder server task that finished handling
 *
 * Clears type lock and score bias so the task re-classifies naturally
 * on its next sleep/wake cycle.
 */
void gore_binder_done(struct task_struct *server);

/**
 * gore_futex_wake_boost — called from futex_wake() for each woken task
 *
 * @waker: the task calling futex_wake() (current)
 * @wakee: the task being woken from futex wait queue
 *
 * If waker is GORE_REALTIME or GORE_INTERACTIVE, applies a brief
 * GORE_BOOST_FUTEX to wakee so it can resume quickly on the
 * assumption it is on the same critical path as the waker.
 *
 * Does nothing if waker is GORE_CPU_BOUND or GORE_BATCH — those
 * wakeups are not considered latency-critical.
 */
void gore_futex_wake_boost(struct task_struct *waker,
              struct task_struct *wakee);

/* ================================================================
 * Stubs when CONFIG_GORE_SCHED=n — zero overhead
 * ================================================================ */

#else /* !CONFIG_GORE_SCHED */

static inline void gore_apply_boost(struct task_struct *t,
                   int b, u64 d)       {}
static inline void gore_lock_type(struct task_struct *t,
                 unsigned int ty)      {}
static inline void gore_unlock_type(struct task_struct *t) {}
static inline void gore_binder_wakeup(struct task_struct *s,
                     struct task_struct *c)    {}
static inline void gore_binder_done(struct task_struct *s) {}
static inline void gore_futex_wake_boost(struct task_struct *waker,
                    struct task_struct *wakee) {}

#endif /* CONFIG_GORE_SCHED */
#endif /* _LINUX_GORE_HOOK_H */