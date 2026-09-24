// SPDX-License-Identifier: GPL-2.0
/*
 * TAGLMK - Task-aware Android Guided Low Memory Killer (core).
 *
 * An independent Android low-memory manager.  While free swap stays above
 * free_swap_limit it proactively pages out cold anonymous memory of background
 * apps (guided by a memory-cache-load forecast).  Once free swap drops below
 * free_swap_limit it collects kill candidates, sorts them so the lowest
 * cumulative-CPU-time apps go first, and kills just enough of them to bring
 * free swap back above the limit.  A shrinking active file cache escalates
 * into more aggressive killing.  Android system core processes are never
 * killed; pinned packages are sacrificed last.
 *
 * This driver shares no code with, and does not depend on, simple_lmk.
 */
#define pr_fmt(fmt) "taglmk: " fmt

#include <linux/delay.h>
#include <linux/kobject.h>
#include <linux/kthread.h>
#include <linux/mm.h>
#include <linux/math64.h>
#include <linux/moduleparam.h>
#include <linux/oom.h>
#include <linux/sched.h>
#include <linux/sched/mm.h>
#include <linux/sched/signal.h>
#include <linux/signal.h>
#include <linux/slab.h>
#include <linux/sort.h>
#include <linux/spinlock.h>
#include <linux/swap.h>
#include <linux/taglmk.h>
#include <linux/wait.h>

#include "taglmk.h"

/* ---- tunables (all live-tunable via /sys/module/taglmk/parameters) ------- */

static bool taglmk_enabled = true;
module_param_named(enabled, taglmk_enabled, bool, 0644);

/* Reclaim while free swap stays above this; start killing once below it. */
static unsigned int free_swap_limit_mb = 256;
module_param(free_swap_limit_mb, uint, 0644);

/* Active file cache below this is treated as memory-critical (escalation). */
static unsigned int free_file_limit_mb = 100;
module_param(free_file_limit_mb, uint, 0644);

/* Cold anon pages paged out per task in one reclaim step. */
static unsigned int reclaim_batch = 512;
module_param(reclaim_batch, uint, 0644);

/* Only reclaim from apps at or above this oom_score_adj (leave foreground). */
static int reclaim_min_adj = 200;
module_param(reclaim_min_adj, int, 0644);

/* oom_score_adj below this (but >= 0) is a SYSTEM_APP: killed on escalation. */
static int system_adj_max = 100;
module_param(system_adj_max, int, 0644);

/* Settle time after a kill before re-checking free swap. */
static unsigned int kill_settle_ms = 20;
module_param(kill_settle_ms, uint, 0644);

/* Forecast cache-load (0..16) at or above which proactive reclaim kicks in. */
static int reclaim_load_threshold = 8;
module_param(reclaim_load_threshold, int, 0644);

/* Low-RAM (<=4GB) devices reclaim earlier by this many load units. */
static int low_ram_bias = 3;
module_param(low_ram_bias, int, 0644);

/* ---- state --------------------------------------------------------------- */

#define TAGLMK_MAX_VICTIMS	256
#define TAGLMK_REAPER_QLEN	64
#define TAGLMK_REAP_RETRIES	10
#define TAGLMK_CYCLE_MAX	64	/* bound on state-machine iterations */

static struct taglmk_predictor predictor;
static struct kobject *taglmk_kobj;
static struct task_struct *main_thread;
static struct task_struct *reaper_thread;

static DECLARE_WAIT_QUEUE_HEAD(main_wait);
static bool pressure_latched;

static struct mm_struct *reaper_ring[TAGLMK_REAPER_QLEN];
static unsigned int reaper_head, reaper_tail;
static DEFINE_SPINLOCK(reaper_lock);
static DECLARE_WAIT_QUEUE_HEAD(reaper_wait);

/* ---- unit helpers -------------------------------------------------------- */

static inline unsigned long mb_to_pages(unsigned int mb)
{
	return (unsigned long)mb << (20 - PAGE_SHIFT);
}

static inline unsigned long swap_limit_pages(void)
{
	return mb_to_pages(free_swap_limit_mb);
}

static inline unsigned long file_limit_pages(void)
{
	return mb_to_pages(free_file_limit_mb);
}

static inline unsigned long free_swap_now(void)
{
	long s = get_nr_swap_pages();

	return s < 0 ? 0UL : (unsigned long)s;
}

bool taglmk_is_low_ram(void)
{
	return totalram_pages <= (4UL << (30 - PAGE_SHIFT));
}

/* ---- victim reaper ------------------------------------------------------- */

static bool taglmk_reap_once(struct mm_struct *mm)
{
	bool ret = true;

	if (!mmap_read_trylock(mm))
		return false;
	if (test_bit(MMF_OOM_SKIP, &mm->flags))
		goto unlock;
	ret = __oom_reap_task_mm(mm);
unlock:
	mmap_read_unlock(mm);
	return ret;
}

static void taglmk_reap_mm(struct mm_struct *mm)
{
	int attempts = 0;

	while (attempts++ < TAGLMK_REAP_RETRIES && !taglmk_reap_once(mm))
		schedule_timeout_idle(HZ / 10);
	/* Hide this mm from any further reaping attempts. */
	set_bit(MMF_OOM_SKIP, &mm->flags);
}

/* Consumes one mmgrab() reference on @mm. */
static void taglmk_reaper_enqueue(struct mm_struct *mm)
{
	unsigned long flags;
	bool queued = false;

	spin_lock_irqsave(&reaper_lock, flags);
	if ((reaper_head + 1) % TAGLMK_REAPER_QLEN != reaper_tail) {
		reaper_ring[reaper_head] = mm;
		reaper_head = (reaper_head + 1) % TAGLMK_REAPER_QLEN;
		queued = true;
	}
	spin_unlock_irqrestore(&reaper_lock, flags);

	if (queued) {
		wake_up(&reaper_wait);
	} else {
		/* Queue full: reap inline so the reference is never leaked. */
		taglmk_reap_mm(mm);
		mmdrop(mm);
	}
}

static int taglmk_reaper_fn(void *unused)
{
	while (!kthread_should_stop()) {
		struct mm_struct *mm = NULL;
		unsigned long flags;

		wait_event_interruptible(reaper_wait,
				reaper_head != reaper_tail ||
				kthread_should_stop());
		if (kthread_should_stop())
			break;

		spin_lock_irqsave(&reaper_lock, flags);
		if (reaper_head != reaper_tail) {
			mm = reaper_ring[reaper_tail];
			reaper_tail = (reaper_tail + 1) % TAGLMK_REAPER_QLEN;
		}
		spin_unlock_irqrestore(&reaper_lock, flags);

		if (mm) {
			taglmk_reap_mm(mm);
			mmdrop(mm);
		}
	}
	return 0;
}

/* ---- task collection & classification ------------------------------------ */

#define K(x) ((x) << (PAGE_SHIFT - 10))

struct taglmk_victim {
	struct task_struct	*tsk;
	u64			usage;		/* cumulative CPU time (ns) */
	unsigned long		rss_anon;
	int			tier;		/* 0 APP, 1 SYSTEM_APP, 2 PINNED */
};

/* Scratch, only ever touched by the single main kthread. */
static struct task_struct	*cand_arr[TAGLMK_MAX_VICTIMS];
static struct taglmk_victim	victim_arr[TAGLMK_MAX_VICTIMS];
static bool			low_ram_mode;

/*
 * Pin every non-core group leader whose oom_score_adj is >= @adj_min.  Runs
 * under rcu_read_lock and applies only non-sleeping filters; the caller does
 * the sleeping classification (cmdline/pin match, inspection) after unlock.
 * System core tasks (kthreads, adj < 0) are never collected.  Returns the
 * count stored, each holding a task reference the caller must drop.
 */
static int taglmk_collect(struct task_struct **arr, int max, int adj_min)
{
	struct task_struct *tsk;
	int n = 0;

	if (adj_min < 0)
		adj_min = 0;

	rcu_read_lock();
	for_each_process(tsk) {
		short adj;

		if (tsk->flags & PF_KTHREAD)
			continue;
		if (tsk->exit_state || !tsk->mm)
			continue;
		adj = READ_ONCE(tsk->signal->oom_score_adj);
		if (adj < adj_min)		/* adj < 0 == system core */
			continue;
		if (n >= max)
			break;
		get_task_struct(tsk);
		arr[n++] = tsk;
	}
	rcu_read_unlock();
	return n;
}

static u64 task_cpu_usage(struct task_struct *tsk)
{
	u64 usage = READ_ONCE(tsk->utime) + READ_ONCE(tsk->stime);

	/* Include CPU time accumulated from already-exited threads. */
	usage += READ_ONCE(tsk->signal->utime) + READ_ONCE(tsk->signal->stime);
	return usage;
}

static int taglmk_victim_tier(struct task_struct *tsk)
{
	short adj = READ_ONCE(tsk->signal->oom_score_adj);

	if (taglmk_is_pinned(tsk))
		return 2;			/* PINNED: sacrificed last */
	if (adj < system_adj_max)
		return 1;			/* SYSTEM_APP: escalation only */
	return 0;				/* APP: primary target */
}

static int victim_cmp(const void *a, const void *b)
{
	const struct taglmk_victim *va = a, *vb = b;
	u64 ka, kb;

	if (va->tier != vb->tier)
		return va->tier - vb->tier;	/* exhaust APP before the rest */

	if (low_ram_mode) {
		/*
		 * Low-RAM variant: rank by cumulative CPU time per resident
		 * anon page, so big, cold, lightly-used apps go first and each
		 * kill frees the most memory.
		 */
		ka = div64_u64(va->usage, va->rss_anon + 1);
		kb = div64_u64(vb->usage, vb->rss_anon + 1);
	} else {
		ka = va->usage;			/* lowest CPU time first */
		kb = vb->usage;
	}
	if (ka < kb)
		return -1;
	return ka > kb ? 1 : 0;
}

/* Mark, signal and queue one victim's whole thread group for reaping. */
static bool taglmk_kill_one(struct task_struct *victim)
{
	struct task_struct *p;
	struct mm_struct *mm;

	p = find_lock_task_mm(victim);
	if (!p)
		return false;
	mm = p->mm;
	if (test_bit(MMF_OOM_SKIP, &mm->flags) ||
	    test_and_set_bit(MMF_OOM_VICTIM, &mm->flags)) {
		task_unlock(p);	/* already dying / being reaped */
		return false;
	}
	mmgrab(mm);
	task_unlock(p);

	do_send_sig_info(SIGKILL, SEND_SIG_FORCED, victim, PIDTYPE_TGID);
	pr_info("killed %d (%s) anon:%lukB\n", task_pid_nr(victim),
		victim->comm, K(get_mm_counter(mm, MM_ANONPAGES)));

	taglmk_reaper_enqueue(mm);	/* consumes the mmgrab reference */
	return true;
}

/* Kill just enough of the lowest-usage apps to relieve the pressure. */
static void taglmk_kill_pass(bool file_critical)
{
	unsigned long swap_limit = swap_limit_pages();
	int n, m = 0, i, max_tier;

	low_ram_mode = taglmk_is_low_ram();

	/*
	 * Escalation ladder.  Normally only ordinary apps are eligible.  A
	 * critically small file cache also exposes system apps; if free swap
	 * is critically low as well, pinned apps become eligible - but always
	 * last, and system core is never eligible.
	 */
	if (file_critical)
		max_tier = (free_swap_now() < (swap_limit >> 2)) ? 2 : 1;
	else
		max_tier = 0;

	n = taglmk_collect(cand_arr, TAGLMK_MAX_VICTIMS, 0);
	for (i = 0; i < n; i++) {
		struct task_struct *t = cand_arr[i];
		struct taglmk_mm_stat st;
		int tier = taglmk_victim_tier(t);

		if (tier > max_tier || taglmk_mm_inspect(t, &st) < 0) {
			put_task_struct(t);
			continue;
		}
		victim_arr[m].tsk = t;
		victim_arr[m].tier = tier;
		victim_arr[m].usage = task_cpu_usage(t);
		victim_arr[m].rss_anon = st.rss_anon;
		m++;
	}

	sort(victim_arr, m, sizeof(victim_arr[0]), victim_cmp, NULL);

	for (i = 0; i < m; i++) {
		if (free_swap_now() > swap_limit &&
		    !taglmk_lru_file_critical(file_limit_pages()))
			break;		/* pressure relieved */
		if (taglmk_kill_one(victim_arr[i].tsk))
			msleep(kill_settle_ms);
	}

	for (i = 0; i < m; i++)
		put_task_struct(victim_arr[i].tsk);
}

/* ---- proactive reclaim --------------------------------------------------- */

/* Page out cold anon from background apps; returns pages reclaimed. */
static unsigned long taglmk_reclaim_pass(void)
{
	unsigned long swap_limit = swap_limit_pages();
	unsigned long reclaimed = 0;
	int n, i;

	n = taglmk_collect(cand_arr, TAGLMK_MAX_VICTIMS, reclaim_min_adj);
	for (i = 0; i < n; i++) {
		struct task_struct *t = cand_arr[i];

		/* Stop reclaiming once we have crossed into kill territory. */
		if (free_swap_now() > swap_limit)
			reclaimed += taglmk_reclaim_task_anon(t, reclaim_batch);
		put_task_struct(t);
		cond_resched();
	}
	return reclaimed;
}

static bool should_reclaim(intfp_t forecast)
{
	int thresh = reclaim_load_threshold;

	if (taglmk_is_low_ram())
		thresh -= low_ram_bias;		/* reclaim earlier on <=4GB */
	if (thresh < 0)
		thresh = 0;

	/* Forecast is Q16.16 load units (0..16). */
	if (forecast >= ((intfp_t)thresh << TAGLMK_FP_FBITS))
		return true;
	/* Or when free swap is already closing on the limit. */
	return free_swap_now() < swap_limit_pages() * 2;
}

/* ---- state machine ------------------------------------------------------- */

static void taglmk_run_cycle(intfp_t forecast)
{
	bool reclaim_ok = should_reclaim(forecast);
	int iter = 0;

	do {
		unsigned long swap_limit = swap_limit_pages();
		bool file_crit = taglmk_lru_file_critical(file_limit_pages());

		if (free_swap_now() > swap_limit && !file_crit) {
			/* Healthy: only page out when pressure is building. */
			if (!reclaim_ok || !taglmk_reclaim_pass())
				break;
		} else {
			/* Under the swap limit or file-critical: kill. */
			taglmk_kill_pass(file_crit);
		}
		cond_resched();
	} while (++iter < TAGLMK_CYCLE_MAX && !kthread_should_stop());
}

static int taglmk_main_fn(void *unused)
{
	while (!kthread_should_stop()) {
		wait_event_interruptible(main_wait,
				READ_ONCE(pressure_latched) ||
				kthread_should_stop());
		if (kthread_should_stop())
			break;
		WRITE_ONCE(pressure_latched, false);
		if (!READ_ONCE(taglmk_enabled))
			continue;

		taglmk_predict_push(&predictor, taglmk_lru_load_sample());
		taglmk_run_cycle(taglmk_predict_eval(&predictor));
	}
	return 0;
}

/* ---- reclaim-path hook --------------------------------------------------- */

void taglmk_note_pressure(int order, bool direct_reclaim)
{
	if (!READ_ONCE(taglmk_enabled) || !READ_ONCE(main_thread))
		return;
	WRITE_ONCE(pressure_latched, true);
	wake_up(&main_wait);
}

/* ---- OOM-path hook ------------------------------------------------------- */

bool taglmk_oom_active(void)
{
	return READ_ONCE(taglmk_enabled) && READ_ONCE(main_thread);
}

/* ---- init ---------------------------------------------------------------- */

static int __init taglmk_init(void)
{
	taglmk_predict_init(&predictor);

	taglmk_kobj = kobject_create_and_add("taglmk", kernel_kobj);
	if (taglmk_kobj)
		taglmk_pin_init(taglmk_kobj);

	reaper_thread = kthread_run(taglmk_reaper_fn, NULL, "taglmk_reaper");
	if (IS_ERR(reaper_thread)) {
		pr_err("failed to start reaper (%ld)\n", PTR_ERR(reaper_thread));
		reaper_thread = NULL;
		return -ENOMEM;
	}

	main_thread = kthread_run(taglmk_main_fn, NULL, "taglmk");
	if (IS_ERR(main_thread)) {
		pr_err("failed to start main thread (%ld)\n",
		       PTR_ERR(main_thread));
		main_thread = NULL;
		kthread_stop(reaper_thread);
		reaper_thread = NULL;
		return -ENOMEM;
	}

	pr_info("initialised (low_ram=%d)\n", taglmk_is_low_ram());
	return 0;
}
late_initcall(taglmk_init);

/* ---- Android lowmemorykiller compatibility shim -------------------------- */
/*
 * Android userspace (lmkd / ActivityManager) probes for the module parameter
 * "lowmemorykiller.minfree" to decide whether an in-kernel low-memory killer
 * is present.  If that parameter is absent under exactly that name the
 * framework concludes there is no LMK and reboots the device, regardless of
 * which driver is actually reclaiming underneath.  TAGLMK therefore exposes
 * that exact parameter surface (same prefix, same name, same write-only mode)
 * even though it shares no internals with the simple_lmk being removed.  This
 * is a live userspace contract, not vestigial code - do not delete it.
 *
 * The handler is TAGLMK's own.  lmkd writes the classic ascending list of
 * free-memory page thresholds; its largest entry is the earliest (highest
 * free) trigger, so we adopt it as TAGLMK's file-cache-critical floor
 * (free_file_limit), mapping the framework's LMK intent onto this driver's
 * own escalation knob rather than reusing simple_lmk's logic.
 */
static int taglmk_minfree_set(const char *val, const struct kernel_param *kp)
{
	int ints[9];			/* ints[0] = count, then the thresholds */
	unsigned long max_pages = 0;
	int i;

	get_options(val, ARRAY_SIZE(ints), ints);
	for (i = 1; i <= ints[0] && i < (int)ARRAY_SIZE(ints); i++) {
		if (ints[i] > 0 && (unsigned long)ints[i] > max_pages)
			max_pages = ints[i];
	}

	if (max_pages) {
		unsigned int mb = max_pages >> (20 - PAGE_SHIFT);

		WRITE_ONCE(free_file_limit_mb, mb ? mb : 1);
	}
	return 0;
}

static const struct kernel_param_ops taglmk_minfree_ops = {
	.set = taglmk_minfree_set,
};

/* Preserve the Android-visible parameter name/prefix; see comment above. */
#undef MODULE_PARAM_PREFIX
#define MODULE_PARAM_PREFIX "lowmemorykiller."
module_param_cb(minfree, &taglmk_minfree_ops, NULL, 0200);
