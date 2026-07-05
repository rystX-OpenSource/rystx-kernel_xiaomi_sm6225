/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (c) 2026 Galih Tama <galpt@v.recipes>
 *
 * infinity_sched.h — Infinity scheduler API (v4.5).
 *
 * Architecture:
 *
 *   fair.c (Linux scheduler)         infinity_sched.c (Infinity algorithm)
 *   ──────────────────────────       ─────────────────────────────────────
 *   update_deadline()        ──call──► infinity_update_weight() — EMA weight
 *   update_curr()            ──call──► infinity_consume()       — EMA budget
 *   enqueue_task_fair()      ──call──► infinity_wakeup()        — EMA decay
 *   place_entity()           ──check──► futex_waiting            — halve vslice on futex wakeup
 *   dequeue_task_fair()      ──call──► (records last_sleep_ns)  — sleep tracking
 *   update_curr_rt()         ──call──► infinity_rt_consume()    — RT EMA climb
 *   enqueue_task_rt()        ──call──► infinity_rt_wakeup()     — RT EMA decay
 *   dequeue_task_rt()        ──call──► (records rt_last_sleep_ns)
 *   task_tick_rt()           ──call──► infinity_rr_timeslice()  — adaptive RR slice
 *   task_fork_fair()         ──call──► infinity_fork_init()     — fork init
 *   init/init_task.c         ──init──► infinity.{}              — static init
 *
 * Weight-based modulation: the task's EEVDF weight is modulated by EMA.
 * EEVDF natively computes a shorter slice and later deadline from a lower
 * weight — no second level of fairness logic needed.
 *
 * The base weight is always derived from the task's static priority (nice),
 * never from the live weight, so the modulation is idempotent and the nice
 * value is always honoured.
 *
 * Tunables:
 *   kernel.infinity_smt_divisor   — SMT secondary slice divisor (default 2)
 *   kernel.infinity_running       — read-only flag, 1 if active
 *
 * Self-stabilizing by construction: the EMA naturally converges between
 * 0 and BUDGET_MAX without any clamps or external feedback loop.
 * Higher EMA → lower effective weight → later deadline.
 */
#ifndef __INFINITY_SCHED_H
#define __INFINITY_SCHED_H
#include <linux/sched.h>
/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */
/** Maximum budget ceiling (6ms). */
#define INFINITY_BUDGET_MAX_NS		6000000ULL
/**
 * EMA time constant: step = (BUDGET_MAX - ema) × runtime × ALPHA / (...)
 * α = 3072 gives ~500µs continuous runtime to reach full EMA penalty.
 */
#define INFINITY_EMA_ALPHA		3072
/** Fixed-point shift for fractional precision (8 bits). */
#define INFINITY_FP_SHIFT		8
#define INFINITY_FP_ONE			(1 << INFINITY_FP_SHIFT)
/**
 * Weight reduction slope: effective = base × (100 - pct × 98/100) / 100.
 * At EMA=100%, denom = 2, weight = base × 2/100 = base × 2% (50× reduction).
 * With 32 storm threads at 2% = 640 < 1 interactive thread at 1024.
 */
#define INFINITY_WEIGHT_SLOPE_NUM	98
#define INFINITY_WEIGHT_SLOPE_DEN	100
/* ------------------------------------------------------------------ */
/* SMT divisor bounds                                                  */
/* ------------------------------------------------------------------ */
#define INFINITY_SMT_DIVISOR_DEFAULT	2
#define INFINITY_SMT_DIVISOR_MIN	1
#define INFINITY_SMT_DIVISOR_MAX	16
/* ------------------------------------------------------------------ */
/* RT EMA constants                                                    */
/* ------------------------------------------------------------------ */
/** RT budget ceiling (10ms). */
#define INFINITY_RT_BUDGET_NS		10000000ULL
/** RT alpha. */
#define INFINITY_RT_ALPHA		4
/* ------------------------------------------------------------------ */
/* External sysctl tunables                                            */
/* ------------------------------------------------------------------ */
extern unsigned long infinity_tune_smt_divisor;
/* ------------------------------------------------------------------ */
/* API — called from fair.c and rt.c                                   */
/* ------------------------------------------------------------------ */
u32 infinity_calc_weight(struct task_struct *p, u64 ema);
void infinity_consume(struct infinity_ctx *ctx, u64 delta_ns);
void infinity_wakeup(struct infinity_ctx *ctx, u64 sleep_ns);
void infinity_fork_init(struct infinity_ctx *ctx, u64 now);
void infinity_rt_consume(struct infinity_ctx *ctx, u64 delta_ns);
void infinity_rt_wakeup(struct infinity_ctx *ctx, u64 sleep_ns);
unsigned int infinity_rr_timeslice(struct task_struct *p,
				   unsigned int rr_default);
/*
 * Infinity scheduler per-task context is heap-allocated rather than
 * embedded in task_struct. Embedding it directly shifts the byte
 * offset of every field declared after it (dl_server, scx, and
 * everything below), which silently breaks any vendor blob or
 * prebuilt kernel module that expects the stock ANDROID_KABI-frozen
 * layout. Using a pointer stored in a reserved ANDROID_KABI_USE()
 * slot keeps task_struct's layout stable; only this pointer's
 * target is new.
 *
 * Allocated in dup_task_struct() (fork) and freed in free_task().
 * init_task uses a static instance instead, since it is never
 * allocated or freed through the normal task lifecycle.
 */
struct infinity_ctx *infinity_ctx_alloc(void);
void infinity_ctx_free(struct infinity_ctx *ctx);
#endif /* __INFINITY_SCHED_H */