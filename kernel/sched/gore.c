// SPDX-License-Identifier: GPL-2.0
/*
 * GoreScheduler (GoreSched) v1.0 — Linux 4.19 backport
 *
 * A unified CPU scheduler combining:
 * BORE  (C) 2021-2025 Masahito Suzuki <firelzrd@gmail.com>
 * CacULE (C) 2020 Hamad Al Marri <hamad.s.almarri@gmail.com>
 * TT    (C) 2023 Hamad Al Marri <hamad.s.almarri@gmail.com>
 * ECHO  (C) 2024 Hamad Al Marri <hamad.s.almarri@gmail.com>
 *
 * Combined into GoreSched by the GoreScheduler Unified Project, 2026.
 * Backported to Linux 4.19 CFS kernel structure.
 */

#include <linux/sched.h>
#include <linux/sched/task.h>
#include <linux/sched/signal.h>
#include <linux/sched/cputime.h>
#include <linux/sysctl.h>
#include <linux/init.h>
#include <linux/percpu.h>
#include <linux/spinlock.h>
#include <linux/math64.h>
#include "sched.h"
#include "gore.h"

#ifdef CONFIG_GORE_SCHED

/* ====================================================================
 * Tunables
 * ==================================================================== */

int  __read_mostly gore_enabled             = 1;
unsigned int __read_mostly gore_max_lifetime_ms    = 22000;
unsigned int __read_mostly gore_burst_penalty_offset = 24;
unsigned int __read_mostly gore_burst_penalty_scale  = 1536;
unsigned int __read_mostly gore_burst_smoothness     = 1;
unsigned int __read_mostly gore_est_alpha            = 250; /* 0.25 × 1000 */
unsigned int __read_mostly gore_burst_weight         = GORE_BURST_WEIGHT_DEFAULT;
unsigned int __read_mostly gore_starve_div           = GORE_STARVE_DIV_DEFAULT;
int  __read_mostly gore_lat_sens_enabled    = 1;
int  __read_mostly gore_dedicated_cpu_enabled = 1;

DEFINE_PER_CPU(int, gore_nr_lat_sensitive);

/* ====================================================================
 * BORE: Burst-Oriented Response Enhancer helpers
 * (adapted from BORE 6.6.2, backported to 4.19 CFS)
 * ==================================================================== */

/*
 * gore_log2p1_u64 — fixed-point log₂(v+1) with 8 fractional bits.
 * Used to compute a logarithmic burst penalty (BORE algorithm).
 */
static inline u32 gore_log2p1_u64(u64 v)
{
   int clz, exponent;
   u32 mantissa;

   if (unlikely(!v))
       return 0;
   clz      = __builtin_clzll(v);
   exponent = 64 - clz;
   mantissa = (u32)((v << clz) << 1 >> (64 - 8));
   return (exponent << 8) | mantissa;
}

/*
 * gore_calc_burst_penalty — map burst_time → penalty value [0, MAX_BURST_PENALTY].
 *
 * The penalty is zero for short bursts (< offset threshold) and grows
 * logarithmically beyond that.  Scale stretches the slope.
 */
static u32 gore_calc_burst_penalty(u64 burst_time)
{
   u32 greed, tolerance;
   s32 diff;
   u32 penalty, scaled;
   s32 overflow;

   greed     = gore_log2p1_u64(burst_time);
   tolerance = gore_burst_penalty_offset << 8;
   diff      = (s32)(greed - tolerance);
   penalty   = diff & ~(diff >> 31);       /* max(0, diff) */
   scaled    = penalty * gore_burst_penalty_scale >> 10;
   overflow  = scaled - GORE_MAX_BURST_PENALTY;
   return scaled - (overflow & ~(overflow >> 31)); /* clamp */
}

/*
 * gore_binary_smooth — one-sided exponential smoothing.
 *
 * When new > old the value rises slowly (shifted by smoothness bits).
 * When new <= old it drops instantly.  This means burst penalties can
 * only grow slowly but fall immediately when a task starts sleeping.
 */
static inline u32 gore_binary_smooth(u32 new_val, u32 old_val)
{
   u32 growing  = (new_val > old_val);
   u32 incr     = (new_val - old_val) * growing;
   u32 shift    = gore_burst_smoothness;
   u32 smoothed = old_val + ((incr + (1U << shift) - 1) >> shift);
   return (new_val & ~(-(int)growing)) | (smoothed & (-(int)growing));
}

/*
 * gore_effective_prio — compute the BORE-adjusted nice index [0..39].
 *
 * bore_score is added to the CFS nice index.  Tasks with high burst
 * scores get a lower scheduling weight (as if niced positively).
 * Kernel threads are exempt.
 */
int gore_effective_prio(struct task_struct *p)
{
   int prio;
   s32 diff;

   if (!gore_enabled || (p->flags & PF_KTHREAD))
       return p->static_prio - MAX_RT_PRIO;

   prio = (p->static_prio - MAX_RT_PRIO) + (int)p->se.gore_node.bore_score;
   /* clamp to [0, 39] */
   prio = prio & ~(prio >> 31);          /* max(0, prio)  */
   diff = prio - 39;
   prio -= (diff & ~(diff >> 31));        /* min(prio, 39) */
   return prio;
}

/*
 * gore_update_bore — called every delta_exec from update_curr().
 *
 * Accumulates raw burst time and recomputes the burst penalty and
 * bore_score.  bore_score feeds back into gore_effective_prio().
 */
static void gore_update_bore(struct gore_node *gn, u64 delta_exec)
{
   u32 curr_penalty;

   if (gn->bore_stop_update)
       return;

   gn->bore_burst_time += delta_exec;
   curr_penalty = gore_calc_burst_penalty(gn->bore_burst_time);
   gn->bore_curr_penalty = curr_penalty;

   if (curr_penalty > gn->bore_prev_penalty) {
       /*
        * Penalty grew: derive bore_score from the
        * max(curr, prev) penalty, scaled to [0..39].
        * Kernel threads are always score=0.
        */
       u32 max_pen = curr_penalty;
       /* score = max_pen >> 8, clamped to 39 */
       u32 raw = max_pen >> 8;
       gn->bore_score = (u8)min_t(u32, raw, 39);
   }
}

/*
 * gore_restart_burst — called when a task goes to sleep.
 *
 * The burst time is reset.  The previous penalty is updated via
 * binary smoothing so the penalty decays gracefully.
 */
static void gore_restart_burst(struct gore_node *gn)
{
   u32 smoothed = gore_binary_smooth(gn->bore_curr_penalty,
                     gn->bore_prev_penalty);
   gn->bore_prev_penalty = smoothed;
   gn->bore_curr_penalty = 0;
   gn->bore_burst_time   = 0;

   /* recompute score from smoothed previous penalty */
   {
       u32 raw = smoothed >> 8;
       gn->bore_score = (u8)min_t(u32, raw, 39);
   }
}

/* ====================================================================
 * ECHO: Exponential Smoothing Estimate helpers
 * (adapted from ECHO bs.c, backported to 4.19)
 * ==================================================================== */

/*
 * gore_update_est — update exponential smoothing estimate (ECHO).
 *
 * EST = alpha * vburst + (1 - alpha) * prev_est
 * with alpha = gore_est_alpha / 1000.
 *
 * vburst is the virtual burst since the last wakeup.  EST tracks
 * historical burstiness smoothly, damping short-lived spikes.
 */
static void gore_update_est(struct gore_node *gn)
{
   u64 vburst   = gn->vburst;
   u64 prev_est = gn->est;
   u32 alpha    = gore_est_alpha;        /* 0..1000 */

   gn->est = (alpha * vburst + (1000 - alpha) * prev_est) / 1000;
}

/* ====================================================================
 * CacULE: Interactivity score & lifetime normalisation
 * (adapted from CacULE 5.10-r3, backported to 4.19)
 * ==================================================================== */

/*
 * gore_normalize_lifetime — prevent integer overflow for long-lived tasks.
 *
 * When a task has been alive longer than gore_max_lifetime_ms, rescale
 * both gore_start_time and vruntime so the HRRN ratio is preserved but
 * the raw timestamps fit comfortably in u64 arithmetic.
 */
static void gore_normalize_lifetime(u64 now, struct gore_node *gn)
{
   u64 max_life_ns = (u64)gore_max_lifetime_ms << 20; /* ~ms in ns */
   u64 life_time   = now - gn->start_time;
   s64 diff        = (s64)(life_time - max_life_ns);
   u64 old_ratio_x, new_vrt;

   if (diff <= 0)
       return;

   /* Unmark yield flag during normalisation */
   gn->vruntime &= GORE_YIELD_UNMARK;

   /* Preserve the hrrn ratio: vrt / life_time */
   old_ratio_x = (life_time << 7) / ((gn->vruntime >> 3) | 1);
   if (!old_ratio_x)
       old_ratio_x = 1;

   /* Reset life to half max */
   gn->start_time = now - (max_life_ns >> 1);
   new_vrt        = (max_life_ns << 9) / old_ratio_x;
   gn->vruntime   = new_vrt;
}

/* ====================================================================
 * Task-Type detection (TT-derived)
 * ==================================================================== */

void gore_inc_nr_lat_sensitive(int cpu)
{
   if (gore_lat_sens_enabled)
       per_cpu(gore_nr_lat_sensitive, cpu)++;
}

void gore_dec_nr_lat_sensitive(int cpu)
{
   if (per_cpu(gore_nr_lat_sensitive, cpu))
       per_cpu(gore_nr_lat_sensitive, cpu) >>= 1;
}

#define GORE_GEQ(a, b) ((s64)((a) - (b)) >= 0)
#define GORE_LEQ(a, b) ((s64)((a) - (b)) <= 0)
#define GORE_LES(a, b) ((s64)((a) - (b)) <  0)
#define GORE_EQ_D(a, b, d) (GORE_LEQ((a), (b) + (d)) && GORE_GEQ((a), (b) - (d)))

static inline bool gore_is_realtime(struct gore_node *gn, u64 now, int flags)
{
   struct sched_entity *se = gore_se_of(gn);
   struct task_struct  *p  = task_of(se);
   u64 life_time, wait;

   if (!gn->wait_time)
       return false;

   /* Must have been alive at least 0.5 s */
   life_time = now - p->start_time;
   if (GORE_LES(life_time, 500000000ULL))
       return false;

   if (!(flags & ENQUEUE_MIGRATED)) {
       wait = now - se->exec_start;
       if (wait && !GORE_EQ_D(wait, gn->prev_wait_time, GORE_RT_WAIT_DELTA))
           return false;
   }

   if (!GORE_EQ_D(gn->burst, gn->prev_burst, GORE_RT_BURST_DELTA))
       return false;

   if (GORE_LEQ(gn->burst, GORE_RT_BURST_MAX) &&
       GORE_LEQ(gn->curr_burst, GORE_RT_BURST_MAX))
       return true;

   return false;
}

static inline bool gore_is_interactive(struct gore_node *gn, u64 now, u64 hrrn)
{
   struct sched_entity *se = gore_se_of(gn);
   u64 wait;

   /* HRRN >= 2 means vrt * 1000 / lifetime >= 2 → sleep >= run */
   if (GORE_LES(hrrn, (u64)GORE_INTERACTIVE_HRRN))
       return false;

   wait = now - se->exec_start;
   if (wait && GORE_EQ_D(wait, gn->prev_wait_time, GORE_RT_WAIT_DELTA))
       return false;

   return true;
}

static inline bool gore_is_cpu_bound(struct gore_node *gn)
{
   u64 total, pct;

   total = gn->wait_time + gn->vruntime;
   if (!total)
       return false;

   pct = gn->vruntime * 100ULL;
   do_div(pct, total);
   return GORE_GEQ(pct, 90ULL);
}

static inline bool gore_is_batch(u64 hrrn)
{
   return GORE_LES(hrrn, 2ULL);
}

/*
 * gore_detect_type — classify a task into one of the five GORE_TT_* types.
 *
 * Called from gore_enqueue_entity() on every wakeup.  Updates
 * gore_nr_lat_sensitive on type transitions.
 */
static void gore_detect_type(struct cfs_rq *cfs_rq,
                struct gore_node *gn, u64 now, int flags)
{
   unsigned int old_type = gn->task_type;
   unsigned int new_type = GORE_TT_NO_TYPE;
   int cpu;
   u64 hrrn;

   if (gn->vruntime <= 1) {
       gn->task_type = GORE_TT_NO_TYPE;
       return;
   }

   hrrn = (gn->wait_time + gn->vruntime) / gn->vruntime;

   if (gore_is_realtime(gn, now, flags))
       new_type = GORE_TT_REALTIME;
   else if (gore_is_interactive(gn, now, hrrn))
       new_type = GORE_TT_INTERACTIVE;
   else if (gore_is_cpu_bound(gn))
       new_type = GORE_TT_CPU_BOUND;
   else if (gore_is_batch(hrrn))
       new_type = GORE_TT_BATCH;

   /* RT sticky: keep REALTIME type for a few ticks */
   if (new_type == GORE_TT_REALTIME) {
       gn->rt_sticky = GORE_RT_STICKY_TICKS;
   } else if (old_type == GORE_TT_REALTIME && gn->rt_sticky) {
       gn->rt_sticky--;
       return; /* retain REALTIME for now */
   }

   if (new_type != old_type) {
       cpu = task_cpu(task_of(gore_se_of(gn)));

       /* Maintain latency-sensitive counter */
       if (new_type <= GORE_TT_INTERACTIVE && old_type > GORE_TT_INTERACTIVE)
           gore_inc_nr_lat_sensitive(cpu);
       else if (old_type <= GORE_TT_INTERACTIVE && new_type > GORE_TT_INTERACTIVE)
           gore_dec_nr_lat_sensitive(cpu);

       /* Dedicated CPU-bound slot */
       if (gore_dedicated_cpu_enabled) {
           if (!cfs_rq->gore_dedicated_cpu && new_type == GORE_TT_CPU_BOUND)
               cfs_rq->gore_dedicated_cpu = gn;
           else if (cfs_rq->gore_dedicated_cpu == gn &&
                old_type == GORE_TT_CPU_BOUND)
               cfs_rq->gore_dedicated_cpu = NULL;
       }
   }

   gn->task_type = new_type;
}

/* ====================================================================
 * Combined Gore Score
 * ==================================================================== */

/*
 * gore_calc_score — unified scheduling score.
 *
 * Combines HRRN (CacULE/TT), EST (ECHO), and BORE burst penalty into a
 * single comparable value.  Lower score = higher scheduling priority.
 *
 * Formula:
 * hrrn_pct  = vruntime * 1000 / (now - start_time)      [CacULE / TT]
 * est_part  = est >> GORE_EST_SHIFT                       [ECHO]
 * bore_part = bore_score * gore_burst_weight              [BORE]
 * base      = (hrrn_pct*2 + est_part) / 3 + bore_part
 * score     = base * 1000 / (1000 + starvation_ticks)    [CacULE anti-starve]
 */
static u64 gore_calc_score(u64 now, struct gore_node *gn)
{
   u64 lifetime, vrt, hrrn_pct;
   u64 est_part, bore_part, base, stale, score;

   lifetime  = (now - gn->start_time) | 1;
   vrt       = (gn->vruntime & GORE_YIELD_UNMARK) | 1;
   hrrn_pct  = (vrt * 1000ULL) / lifetime;
   est_part  = gn->est >> GORE_EST_SHIFT;
   bore_part = (u64)gn->bore_score * gore_burst_weight;
   base      = (hrrn_pct * 2 + est_part) / 3 + bore_part;

   /* Anti-starvation: task that hasn't run in a while gets score reduced */
   stale = (now - gn->last_run) / gore_starve_div;
   if (stale) {
       score = base * 1000ULL;
       do_div(score, 1000 + (u32)min_t(u64, stale, 0xFFFFFFULL));
   } else {
       score = base;
   }

   return score;
}

/*
 * gore_entity_before — compare two entities for scheduling order.
 *
 * Returns true if entity 'a' should run before entity 'b'.
 * Primary key: task type (lower = higher priority tier).
 * Secondary key: gore_score (lower = more interactive).
 */
static bool gore_entity_before(u64 now,
               struct gore_node *a,
               struct gore_node *b)
{
   if (a->task_type != b->task_type)
       return a->task_type < b->task_type;

   return (s64)(gore_calc_score(now, a) - gore_calc_score(now, b)) < 0;
}

/* ====================================================================
 * Linked-list runqueue operations
 * ==================================================================== */

/*
 * gore_ll_enqueue — insert 'gn' at the head of the linked list.
 * The linked list is unsorted at enqueue (O(1)); sort happens at pick time.
 */
static void gore_ll_enqueue(struct cfs_rq *cfs_rq, struct gore_node *gn)
{
   gn->next = NULL;
   gn->prev = NULL;

   if (cfs_rq->gore_head) {
       gn->next             = cfs_rq->gore_head;
       cfs_rq->gore_head->prev = gn;
       cfs_rq->gore_head    = gn;
   } else {
       cfs_rq->gore_head = gn;
       cfs_rq->gore_tail = gn;
   }
}

/*
 * gore_ll_dequeue — remove 'gn' from the linked list.
 */
static void gore_ll_dequeue(struct cfs_rq *cfs_rq, struct gore_node *gn)
{
   if (cfs_rq->gore_head == cfs_rq->gore_tail) {
       /* sole element */
       cfs_rq->gore_head = NULL;
       cfs_rq->gore_tail = NULL;
   } else if (gn == cfs_rq->gore_head) {
       cfs_rq->gore_head       = gn->next;
       if (cfs_rq->gore_head)
           cfs_rq->gore_head->prev = NULL;
   } else if (gn == cfs_rq->gore_tail) {
       cfs_rq->gore_tail       = gn->prev;
       if (cfs_rq->gore_tail)
           cfs_rq->gore_tail->next = NULL;
   } else {
       struct gore_node *prev = gn->prev;
       struct gore_node *next = gn->next;
       if (prev) prev->next = next;
       if (next) next->prev = prev;
   }
   gn->next = NULL;
   gn->prev = NULL;
}

/* ====================================================================
 * Public API called from fair.c
 * ==================================================================== */

/*
 * gore_init_entity — zero-initialise a new gore_node.
 */
void gore_init_entity(struct sched_entity *se, u64 now)
{
   struct gore_node *gn = &se->gore_node;
   memset(gn, 0, sizeof(*gn));
   gn->start_time  = now;
   gn->last_run    = now;
   gn->task_type   = GORE_TT_NO_TYPE;
}

/*
 * gore_task_fork — inherit burst penalty from parent (simplified BORE
 * inheritance for 4.19, no RCU children traversal needed).
 */
void gore_task_fork(struct task_struct *p)
{
   struct gore_node *gn     = &p->se.gore_node;
   struct task_struct *par  = current;
   struct gore_node *par_gn = &par->se.gore_node;

   if (!gore_enabled)
       return;

   /* Inherit parent's smoothed penalty as a starting point */
   gn->bore_prev_penalty = par_gn->bore_prev_penalty;
   gn->bore_score        = par_gn->bore_score;
   gn->est               = par_gn->est >> 1; /* halved — child starts fresh */
   gn->task_type         = GORE_TT_NO_TYPE;
}

/*
 * gore_update_curr — main per-tick accounting (called from update_curr).
 *
 * This is the central accounting function that advances all four
 * algorithm's per-task state simultaneously:
 * - BORE burst penalty update
 * - CacULE vruntime + lifetime normalisation
 * - ECHO EST vburst accumulation
 * - TT wait/burst tracking
 */
void gore_update_curr(struct cfs_rq *cfs_rq,
             struct sched_entity *curr, u64 delta_exec, u64 now)
{
   struct gore_node *gn;
   u64 delta_fair;

   if (!gore_enabled || !entity_is_task(curr))
       return;

   gn = &curr->gore_node;

   /* ---- BORE burst accumulation ---- */
   gore_update_bore(gn, delta_exec);

   /* ---- CacULE: advance virtual runtime and track last_run ---- */
   delta_fair = calc_delta_fair(delta_exec, curr);
   gn->vruntime += delta_fair;
   gn->last_run  = now;

   /* ---- ECHO: accumulate vburst (virtual burst since wakeup) ---- */
   gn->vburst += delta_fair;

   /* ---- TT: track curr_burst ---- */
   gn->curr_burst += delta_exec;

   /* ---- CacULE: lifetime normalisation (prevent overflow) ---- */
   gore_normalize_lifetime(now, gn);
}

/*
 * gore_enqueue_entity — called when a task is enqueued (wakeup or fork).
 *
 * Updates wait_time, resets vburst for ECHO, runs type detection,
 * and inserts into the linked list.
 */
void gore_enqueue_entity(struct cfs_rq *cfs_rq,
            struct sched_entity *se, int flags, u64 now)
{
   struct gore_node *gn = &se->gore_node;

   if (!gore_enabled)
       return;

   if (flags & ENQUEUE_WAKEUP) {
       u64 wait = now - se->exec_start;

       /* TT: update wait/burst history */
       gn->prev_wait_time = gn->wait_time;
       gn->wait_time      = wait;
       gn->prev_burst     = gn->burst;
       gn->burst          = gn->curr_burst;
       gn->curr_burst     = 0;

       /* ECHO: update EST with last vburst, then reset vburst */
       gore_update_est(gn);
       gn->vburst = 0;

       /* Type detection on every wakeup */
       gore_detect_type(cfs_rq, gn, now, flags);
   }

   gore_ll_enqueue(cfs_rq, gn);
}

/*
 * gore_dequeue_entity — called when a task is dequeued (sleep or preempt).
 *
 * On sleep: restart BORE burst, final CacULE vruntime accounting.
 * Always removes from linked list.
 */
void gore_dequeue_entity(struct cfs_rq *cfs_rq,
            struct sched_entity *se, int flags, u64 now)
{
   struct gore_node *gn = &se->gore_node;

   if (!gore_enabled)
       return;

   gore_ll_dequeue(cfs_rq, gn);

   if (flags & DEQUEUE_SLEEP) {
       /* BORE: restart burst penalty smoothing */
       gore_restart_burst(gn);
       /* ECHO: EST update on final vburst before sleep */
       gore_update_est(gn);
       gn->vburst = 0;
   }

   /* Clear yield mark on dequeue */
   gn->vruntime &= GORE_YIELD_UNMARK;
   gn->yielded  = false;
}

/*
 * gore_pick_next_entity — select the best entity from the linked list.
 *
 * O(n) scan: compare each entity against the current best using
 * gore_entity_before().  Returns the entity with the lowest gore_score
 * in the highest priority task-type tier.
 *
 * If curr is still runnable and beats all queued entities it is returned.
 */
struct sched_entity *gore_pick_next_entity(struct cfs_rq *cfs_rq,
                      struct sched_entity *curr)
{
   struct gore_node *best, *iter;
   u64 now = sched_clock();

   best = cfs_rq->gore_head;
   if (!best)
       return curr;

   for (iter = best->next; iter; iter = iter->next) {
       /* Skip yielded entities unless they are the only option */
       if (iter->yielded && best)
           continue;
       if (gore_entity_before(now, iter, best))
           best = iter;
   }

   /* Does curr beat the best queued candidate? */
   if (curr && !curr->gore_node.yielded) {
       if (!best || gore_entity_before(now, &curr->gore_node, best))
           return curr;
   }

   return best ? gore_se_of(best) : curr;
}

/*
 * gore_check_preempt — return true if new entity 'p' should preempt 'curr'.
 */
bool gore_check_preempt(struct cfs_rq *cfs_rq,
           struct sched_entity *curr,
           struct sched_entity *p)
{
   u64 now;

   if (!gore_enabled)
       return false;

   now = sched_clock();
   return gore_entity_before(now, &p->gore_node, &curr->gore_node);
}

/*
 * gore_yield_entity — mark entity as yielded (CacULE yield mechanism).
 */
void gore_yield_entity(struct sched_entity *se)
{
   if (!gore_enabled)
       return;
   se->gore_node.vruntime |= GORE_YIELD_MARK;
   se->gore_node.yielded   = true;
}

/*
 * gore_init_cfs_rq — initialise per-rq GoreSched fields.
 */
void gore_init_cfs_rq(struct cfs_rq *cfs_rq)
{
   cfs_rq->gore_head         = NULL;
   cfs_rq->gore_tail         = NULL;
   cfs_rq->gore_dedicated_cpu = NULL;
}

/* ====================================================================
 * Scheduler initialisation
 * ==================================================================== */

void __init sched_init_gore(void)
{
   int i;

   printk(KERN_INFO
          "GoreScheduler v1.0: unified BORE+CacULE+TT+ECHO scheduler\n");

   for_each_possible_cpu(i)
       per_cpu(gore_nr_lat_sensitive, i) = 0;
}

/* ====================================================================
 * Sysctl registration (late_initcall, 4.19 style)
 * ==================================================================== */

#ifdef CONFIG_SYSCTL
static int gore_sysctl_zero     = 0;
static int gore_sysctl_one      = 1;
static int gore_sysctl_three    = 3;
static unsigned int gore_uint_zero = 0;
static unsigned int gore_uint_max  = 0xFFFFFFFFU;

static struct ctl_table gore_sysctls[] = {
   {
       .procname     = "sched_gore_enabled",
       .data         = &gore_enabled,
       .maxlen       = sizeof(int),
       .mode         = 0644,
       .proc_handler = proc_dointvec_minmax,
       .extra1       = &gore_sysctl_zero,
       .extra2       = &gore_sysctl_one,
   },
   {
       .procname     = "sched_gore_max_lifetime_ms",
       .data         = &gore_max_lifetime_ms,
       .maxlen       = sizeof(unsigned int),
       .mode         = 0644,
       .proc_handler = proc_douintvec,
   },
   {
       .procname     = "sched_gore_burst_penalty_offset",
       .data         = &gore_burst_penalty_offset,
       .maxlen       = sizeof(unsigned int),
       .mode         = 0644,
       .proc_handler = proc_douintvec_minmax,
       .extra1       = &gore_uint_zero,
       .extra2       = &gore_uint_max,
   },
   {
       .procname     = "sched_gore_burst_penalty_scale",
       .data         = &gore_burst_penalty_scale,
       .maxlen       = sizeof(unsigned int),
       .mode         = 0644,
       .proc_handler = proc_douintvec_minmax,
       .extra1       = &gore_uint_zero,
       .extra2       = &gore_uint_max,
   },
   {
       .procname     = "sched_gore_burst_smoothness",
       .data         = &gore_burst_smoothness,
       .maxlen       = sizeof(unsigned int),
       .mode         = 0644,
       .proc_handler = proc_douintvec_minmax,
       .extra1       = &gore_uint_zero,
       .extra2       = &gore_sysctl_three, /* reuse int 3 */
   },
   {
       .procname     = "sched_gore_est_alpha",
       .data         = &gore_est_alpha,
       .maxlen       = sizeof(unsigned int),
       .mode         = 0644,
       .proc_handler = proc_douintvec_minmax,
       .extra1       = &gore_uint_zero,
       .extra2       = &gore_uint_max,
   },
   {
       .procname     = "sched_gore_burst_weight",
       .data         = &gore_burst_weight,
       .maxlen       = sizeof(unsigned int),
       .mode         = 0644,
       .proc_handler = proc_douintvec,
   },
   {
       .procname     = "sched_gore_starve_div",
       .data         = &gore_starve_div,
       .maxlen       = sizeof(unsigned int),
       .mode         = 0644,
       .proc_handler = proc_douintvec,
   },
   {
       .procname     = "sched_gore_lat_sens_enabled",
       .data         = &gore_lat_sens_enabled,
       .maxlen       = sizeof(int),
       .mode         = 0644,
       .proc_handler = proc_dointvec_minmax,
       .extra1       = &gore_sysctl_zero,
       .extra2       = &gore_sysctl_one,
   },
   {
       .procname     = "sched_gore_dedicated_cpu_enabled",
       .data         = &gore_dedicated_cpu_enabled,
       .maxlen       = sizeof(int),
       .mode         = 0644,
       .proc_handler = proc_dointvec_minmax,
       .extra1       = &gore_sysctl_zero,
       .extra2       = &gore_sysctl_one,
   },
   { }
};

static int __init gore_sysctl_init(void)
{
   register_sysctl("kernel", gore_sysctls);
   return 0;
}
late_initcall(gore_sysctl_init);
#endif /* CONFIG_SYSCTL */

#endif /* CONFIG_GORE_SCHED */