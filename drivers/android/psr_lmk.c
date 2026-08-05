// SPDX-License-Identifier: GPL-2.0
/*
 * drivers/android/psr_lmk.c
 *
 * PSR-LMK: Protected-Swap Regression Low Memory Killer
 * ------------------------------------------------------------------
 * A self-contained Android LMK driver. Inspired by le9uo's working-set
 * protection concept (anon_min_ratio/clean_min_ratio, enforced in
 * mm/vmscan.c via a core reclaim scan-balance rewrite), PSR-LMK reuses
 * the same class of signals -- swap-in trend, protected anon ratio,
 * protected clean-file ratio -- but keeps them out of the reclaim path
 * entirely: they're read-only inputs to an independent kill decision
 * here, and PSR-LMK acts by SIGKILLing a victim task instead of
 * rewriting LRU scan balance.
 *
 * Threading model
 * ---------------
 * Everything expensive runs on a dedicated RT kthread (psr_lmkd), woken
 * by the *global* vmpressure notifier chain -- the same trigger
 * simple_lmk used, and the one that works regardless of CONFIG_MEMCG.
 * Nothing in a reclaim path ever walks the task list, takes a mutex,
 * reads a clock, or sends a signal.
 *
 * The mm hooks (mm/swap.c, mm/workingset.c, mm/page_alloc.c) are
 * per-CPU counter bumps behind a static key: patched-out NOPs until the
 * driver is up, and a single non-atomic per-CPU increment afterwards.
 * They contribute no shared cachelines to the reclaim path.
 *
 * kernel/fork.c's __mmput() reports when a victim's address space is
 * actually gone, which is what releases the pending-kill gate. That is
 * the only reliable "the memory came back" signal available; a timeout
 * alone either fires too early (overkill) or too late (under-kill).
 */

#define pr_fmt(fmt) "psr_lmk: " fmt

#include <linux/atomic.h>
#include <linux/cpumask.h>
#include <linux/freezer.h>
#include <linux/gfp.h>
#include <linux/jiffies.h>
#include <linux/jump_label.h>
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/mm.h>
#include <linux/mmzone.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/notifier.h>
#include <linux/oom.h>
#include <linux/percpu.h>
#include <linux/proc_fs.h>
#include <linux/psr_lmk.h>
#include <linux/sched.h>
#include <linux/sched/coredump.h>
#include <linux/sched/mm.h>
#include <linux/sched/signal.h>
#include <linux/sched/task.h>
#include <linux/seq_file.h>
#include <linux/signal.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/swap.h>
#include <linux/time.h>
#include <linux/timekeeping.h>
#include <linux/vmpressure.h>
#include <linux/wait.h>
#include <uapi/linux/sched/types.h>

#define PSR_LMK_WINDOW 8

DEFINE_STATIC_KEY_FALSE(psr_lmk_key);
EXPORT_SYMBOL_GPL(psr_lmk_key);

/* ------------------------------------------------------------------
 * Hook counters.
 *
 * Per-CPU and non-atomic on purpose. The previous revision used shared
 * atomic_t counters bumped from __activate_page() and
 * workingset_refault(); on an 8-core phone under any UI scroll that is a
 * globally-contended cacheline taking a LOCK-prefixed RMW from every CPU
 * on every reactivation and every refault, inside the LRU lock. That
 * alone is enough to produce the frame drops this driver was reported to
 * cause, and it produced them whether or not memory was actually tight.
 *
 * These are statistics feeding a threshold comparison, so a lost
 * increment from a preemption race is irrelevant -- accuracy here is
 * worth nothing and contention costs everything.
 * ------------------------------------------------------------------ */
struct psr_counters {
	unsigned long anon_reactivations;
	unsigned long alloc_failures;
	unsigned long swap_refaults;
	unsigned long file_refaults;
};

static DEFINE_PER_CPU(struct psr_counters, psr_counters);

void __psr_lmk_note_anon_reactivation(void)
{
	raw_cpu_inc(psr_counters.anon_reactivations);
}
EXPORT_SYMBOL_GPL(__psr_lmk_note_anon_reactivation);

void __psr_lmk_note_refault(bool is_swap)
{
	if (is_swap)
		raw_cpu_inc(psr_counters.swap_refaults);
	else
		raw_cpu_inc(psr_counters.file_refaults);
}
EXPORT_SYMBOL_GPL(__psr_lmk_note_refault);

/*
 * Only order-0, non-__GFP_NORETRY failures count as a memory regression.
 *
 * This hook sits in __alloc_pages_direct_reclaim(), which is reached by
 * plenty of allocations that are *designed* to fail and fall back: THP
 * faults, SLUB's high-order attempts, zsmalloc. Those callers pass
 * __GFP_NORETRY (and/or order > 0) precisely because failure is an
 * acceptable outcome they handle by dropping to order-0. Counting them
 * made a single benign fallback sufficient to declare a regression,
 * since psr_evaluate() tests alloc_failures > 0.
 *
 * An order-0 failure that survived direct reclaim is the real signal --
 * the same one simple_lmk triggers on.
 */
void __psr_lmk_note_alloc_failure(unsigned int order, gfp_t gfp_mask)
{
	if (order > 0 || (gfp_mask & (__GFP_NORETRY | __GFP_NOWAIT)))
		return;

	raw_cpu_inc(psr_counters.alloc_failures);
}
EXPORT_SYMBOL_GPL(__psr_lmk_note_alloc_failure);

/* Drain every CPU's counters into @out and reset them. Called only from
 * the psr_lmkd kthread, once per pressure event. */
static void psr_drain_counters(struct psr_counters *out)
{
	int cpu;

	memset(out, 0, sizeof(*out));

	for_each_possible_cpu(cpu) {
		struct psr_counters *c = per_cpu_ptr(&psr_counters, cpu);

		out->anon_reactivations += xchg(&c->anon_reactivations, 0);
		out->alloc_failures     += xchg(&c->alloc_failures, 0);
		out->swap_refaults      += xchg(&c->swap_refaults, 0);
		out->file_refaults      += xchg(&c->file_refaults, 0);
	}
}

/* ------------------------------------------------------------------
 * Tunables (module params; live view in /proc/psr_lmk/status)
 *
 * reserved_swap_floor_kb  - hard floor on free swap (KB); breach is an
 *                            automatic trigger regardless of trend.
 * anon_min_ratio          - percent of memory below which anon pages are
 *                            considered already-starved. Mirrors le9uo's
 *                            vm.anon_min_ratio semantics, consulted as a
 *                            signal here, not enforced against the LRU.
 * clean_min_ratio         - percent of memory below which clean file
 *                            pages are considered starved. Mirrors
 *                            le9uo's vm.clean_min_ratio.
 * regression_slope_thresh - pressure trend (per sample) above which we
 *                            call it a regression.
 * thrash_hard_limit       - absolute vmpressure level (0-100) that
 *                            triggers regardless of slope.
 * min_oom_score_adj       - only consider victims at/above this
 *                            oom_score_adj (background-ish tasks).
 * dry_run                 - if 1, log the decision but do not SIGKILL.
 * ------------------------------------------------------------------ */
static unsigned long reserved_swap_floor_kb  = 64UL * 1024;
static unsigned int  anon_min_ratio          = 15;  /* percent */
static unsigned int  clean_min_ratio         = 15;  /* percent */
static unsigned int  regression_slope_thresh = 50;
static unsigned int  thrash_hard_limit       = 90;  /* vmpressure 0-100 */
static int           min_oom_score_adj       = 200;
static int           escalated_min_oom_score_adj = 100;
static unsigned int  usage_time_weight_pct   = 30;
static int           dry_run                 = 1;

module_param(reserved_swap_floor_kb, ulong, 0644);
module_param(anon_min_ratio, uint, 0644);
module_param(clean_min_ratio, uint, 0644);
module_param(regression_slope_thresh, uint, 0644);
module_param(thrash_hard_limit, uint, 0644);
module_param(min_oom_score_adj, int, 0644);
module_param(escalated_min_oom_score_adj, int, 0644);
module_param(usage_time_weight_pct, uint, 0644);
module_param(dry_run, int, 0644);

/*
 * Pressure level at which the global vmpressure notifier wakes us.
 *
 * This MUST stay below thrash_hard_limit. The kthread only ever runs
 * because the notifier saw pressure >= wakeup_pressure, so if that value
 * is >= thrash_hard_limit then "pressure >= thrash_hard_limit" is
 * tautologically true on every evaluation and every wakeup produces a
 * kill decision before any corroborating signal is consulted. The
 * previous revision had wakeup_pressure=95 against a 4GB-profile hard
 * limit of 85 and did exactly that.
 *
 * It also has to leave the sliding window some dynamic range: if every
 * sample that reaches psr_push_sample_and_slope() is >= 95, the fit runs
 * over the top 5 points of a 100-point scale and no slope can ever
 * approach regression_slope_thresh. Waking lower is what makes the trend
 * signal -- the actual premise of this driver -- computable at all.
 *
 * Profile-tuned; overridable at runtime, clamped in psr_sanity_check().
 */
static unsigned int wakeup_pressure = 75;
module_param(wakeup_pressure, uint, 0644);

/*
 * Number of consecutive evaluations that must agree before pressure
 * alone is allowed to trip the hard limit.
 *
 * vmpressure_prio() synthesizes pressure=100 (critical, scanned=0,
 * reclaimed=0) whenever vmscan's scan priority dips past
 * vmpressure_level_critical_prio. That is a scan-*depth* signal, not a
 * reclaim-*efficiency* one, and it fires routinely during ordinary app
 * launches on a healthy system -- observed firing on facebook/instagram
 * cold starts with every starvation flag clear. Requiring the condition
 * to persist across several evaluations distinguishes a launch blip from
 * sustained thrash.
 */
static unsigned int sustain_evals = 3;
module_param(sustain_evals, uint, 0644);

/*
 * Grace period after a task starts during which it will not be selected
 * as a victim, in ms.
 *
 * An app being launched is simultaneously the most likely *cause* of a
 * pressure spike and the worst possible victim: killing it makes the
 * launch the user just requested fail visibly. Without this, PSR-LMK
 * will happily pick a process belonging to the app currently starting --
 * which is precisely what the first dry-run trace showed it doing.
 */
static unsigned int launch_grace_ms = 8000;
module_param(launch_grace_ms, uint, 0644);

/*
 * Floor on how often the kthread will do real work, in ms. The notifier
 * can fire far faster than a kill can possibly help; without this, a
 * sustained pressure event would have us re-walking the task list
 * continuously, which is exactly the kind of background CPU burn that
 * shows up as UI jank.
 */
static unsigned int min_eval_interval_ms = 100;
module_param(min_eval_interval_ms, uint, 0644);

/* ------------------------------------------------------------------
 * Auto-tuning profiles, applied once at init based on total device RAM.
 * Concept borrowed from prlmk's RAM-based auto-detection, extended
 * across PSR-LMK's full tunable set and split into three tiers.
 *
 * Direction: less RAM = protect harder, kill sooner. LOW has the
 * smallest absolute swap floor but the tightest thresholds and the
 * widest victim pool; HIGH is the opposite.
 * ------------------------------------------------------------------ */
static unsigned int psr_lmk_ram_tier_low_mb = 4096;
static unsigned int psr_lmk_ram_tier_mid_mb = 8192;
static int          auto_tune               = 1;

module_param(psr_lmk_ram_tier_low_mb, uint, 0444);
module_param(psr_lmk_ram_tier_mid_mb, uint, 0444);
module_param(auto_tune, int, 0444);

struct psr_lmk_profile {
	const char    *name;
	unsigned long  reserved_swap_floor_kb;
	unsigned int   anon_min_ratio;
	unsigned int   clean_min_ratio;
	unsigned int   regression_slope_thresh;
	unsigned int   thrash_hard_limit;
	unsigned int   wakeup_pressure;
	int            min_oom_score_adj;
	int            escalated_min_oom_score_adj;
	unsigned int   usage_time_weight_pct;
};

static const struct psr_lmk_profile psr_lmk_profile_low = {
	.name                        = "3-4GB",
	.reserved_swap_floor_kb      = 32UL * 1024,
	.anon_min_ratio              = 10,
	.clean_min_ratio             = 10,
	.regression_slope_thresh     = 30,
	.thrash_hard_limit           = 85,
	.wakeup_pressure             = 70,
	.min_oom_score_adj           = 150,
	.escalated_min_oom_score_adj = 50,
	.usage_time_weight_pct       = 20,
};

static const struct psr_lmk_profile psr_lmk_profile_mid = {
	.name                        = "6-8GB",
	.reserved_swap_floor_kb      = 64UL * 1024,
	.anon_min_ratio              = 15,
	.clean_min_ratio             = 15,
	.regression_slope_thresh     = 50,
	.thrash_hard_limit           = 90,
	.wakeup_pressure             = 75,
	.min_oom_score_adj           = 200,
	.escalated_min_oom_score_adj = 100,
	.usage_time_weight_pct       = 30,
};

static const struct psr_lmk_profile psr_lmk_profile_high = {
	.name                        = "12GB+",
	.reserved_swap_floor_kb      = 128UL * 1024,
	.anon_min_ratio              = 20,
	.clean_min_ratio             = 20,
	.regression_slope_thresh     = 80,
	.thrash_hard_limit           = 95,
	.wakeup_pressure             = 80,
	.min_oom_score_adj           = 300,
	.escalated_min_oom_score_adj = 150,
	.usage_time_weight_pct       = 40,
};

static const char *psr_lmk_active_profile = "unset";

static void psr_lmk_auto_tune(void)
{
	const struct psr_lmk_profile *profile;
	unsigned long total_mb;

	if (!auto_tune) {
		pr_info("auto_tune=0, using module param defaults\n");
		return;
	}

	/* totalram_pages is a plain global in this tree (include/linux/mm.h);
	 * on trees that converted it to an atomic counter this becomes
	 * totalram_pages(). */
	total_mb = totalram_pages >> (20 - PAGE_SHIFT);

	if (total_mb <= psr_lmk_ram_tier_low_mb)
		profile = &psr_lmk_profile_low;
	else if (total_mb <= psr_lmk_ram_tier_mid_mb)
		profile = &psr_lmk_profile_mid;
	else
		profile = &psr_lmk_profile_high;

	reserved_swap_floor_kb      = profile->reserved_swap_floor_kb;
	anon_min_ratio              = profile->anon_min_ratio;
	clean_min_ratio             = profile->clean_min_ratio;
	regression_slope_thresh     = profile->regression_slope_thresh;
	thrash_hard_limit           = profile->thrash_hard_limit;
	wakeup_pressure             = profile->wakeup_pressure;
	min_oom_score_adj           = profile->min_oom_score_adj;
	escalated_min_oom_score_adj = profile->escalated_min_oom_score_adj;
	usage_time_weight_pct       = profile->usage_time_weight_pct;
	psr_lmk_active_profile      = profile->name;

	pr_info("detected %lu MB RAM, applying \"%s\" profile (swap_floor=%luKB anon_min=%u%% clean_min=%u%% slope=%u hard=%u wakeup=%u min_adj=%d esc_adj=%d usage_weight=%u%%)\n",
		total_mb, profile->name, reserved_swap_floor_kb,
		anon_min_ratio, clean_min_ratio, regression_slope_thresh,
		thrash_hard_limit, wakeup_pressure, min_oom_score_adj,
		escalated_min_oom_score_adj, usage_time_weight_pct);
}

/*
 * Enforce wakeup_pressure < thrash_hard_limit.
 *
 * Both are writable at runtime, and getting the ordering wrong silently
 * turns every wakeup into a guaranteed regression verdict -- a failure
 * mode with no symptom other than a flood of kills, so it is worth
 * refusing rather than trusting. Called at start and after either value
 * could have been written.
 */
static void psr_sanity_check(void)
{
	if (wakeup_pressure >= thrash_hard_limit) {
		unsigned int fixed = thrash_hard_limit > 15
					? thrash_hard_limit - 15 : 1;

		pr_warn("wakeup_pressure=%u >= thrash_hard_limit=%u would make the hard limit unconditional; clamping wakeup to %u\n",
			wakeup_pressure, thrash_hard_limit, fixed);
		wakeup_pressure = fixed;
	}
}

/* ------------------------------------------------------------------
 * Regression engine state.
 *
 * Only ever touched by the psr_lmkd kthread, so it needs no lock at all.
 * The previous revision took a mutex here from what it believed was
 * process context; it is now single-threaded by construction.
 * ------------------------------------------------------------------ */
static struct {
	unsigned long window[PSR_LMK_WINDOW];
	int head;
	int count;
} psr_engine;

/*
 * Consecutive evaluations with pressure >= thrash_hard_limit. Reset on
 * any sample below the limit and after a kill is dispatched. kthread-only,
 * like the rest of the engine state, so it needs no synchronisation.
 */
static unsigned int psr_high_streak;

/*
 * Push a pressure sample and return the least-squares slope over the
 * sliding window.
 *
 * Two bugs are fixed relative to the previous revision:
 *
 *  - It read the window in array order rather than chronological order,
 *    so once the ring wrapped, the x-axis no longer corresponded to
 *    time. The slope past the first 8 samples was fitted against
 *    shuffled data, i.e. noise.
 *
 *  - It computed mean_x/mean_y with integer division and then used them
 *    in the centred-sums formula, which is only valid for exact means.
 *    With n=8 and rounded means, the truncation error is applied n
 *    times over and lands directly in the numerator.
 *
 * Both are avoided by scaling the centred sums by n instead of dividing:
 * for x = 0..n-1 this is exact integer arithmetic. Sizes are tiny
 * (n <= 8, y <= 100), so nothing here can overflow a long.
 */
static long psr_push_sample_and_slope(unsigned long rate)
{
	long sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
	long num, den;
	int n, i, idx;

	psr_engine.window[psr_engine.head] = rate;
	psr_engine.head = (psr_engine.head + 1) % PSR_LMK_WINDOW;
	if (psr_engine.count < PSR_LMK_WINDOW)
		psr_engine.count++;

	n = psr_engine.count;
	if (n < 2)
		return 0;

	/* Walk oldest -> newest. head points one past the newest sample,
	 * so the oldest is head - count (mod window). */
	idx = (psr_engine.head - n + PSR_LMK_WINDOW) % PSR_LMK_WINDOW;
	for (i = 0; i < n; i++) {
		long y = psr_engine.window[idx];

		sum_x  += i;
		sum_y  += y;
		sum_xy += (long)i * y;
		sum_xx += (long)i * i;

		idx = (idx + 1) % PSR_LMK_WINDOW;
	}

	/* n * (Sxy - n*mean_x*mean_y) and n * (Sxx - n*mean_x^2), which
	 * share the factor n and so cancel in the ratio. */
	num = (long)n * sum_xy - sum_x * sum_y;
	den = (long)n * sum_xx - sum_x * sum_x;

	return den ? num / den : 0;
}

/* ------------------------------------------------------------------
 * Ratio-based confirmation signals (concept from le9uo's
 * anon_min_ratio / clean_min_ratio; read-only here, used to qualify
 * whether a pressure trend is worth acting on)
 * ------------------------------------------------------------------ */
struct psr_node_state {
	bool anon_below_min;
	bool clean_below_min;
	bool swap_floor_breached;
	bool escalated;
	unsigned long pressure;

	/* Filled in by psr_evaluate() purely so the BYPASS log line can
	 * report which signals justified the decision. The first dry-run
	 * trace showed only the node flags, all of them zero, which said
	 * that nothing agreed but not what had fired instead. */
	long          slope;
	unsigned int  streak;
	unsigned long swap_refaults;
	unsigned long alloc_failures;
	unsigned long anon_reactivations;
};

static void psr_compute_node_state(struct psr_node_state *st)
{
	pg_data_t *pgdat = NODE_DATA(first_online_node);
	unsigned long total_kb, anon_kb, file_kb, dirty_kb, clean_kb;
	unsigned long free_swap_kb;

	total_kb = totalram_pages << (PAGE_SHIFT - 10);

	anon_kb = (node_page_state(pgdat, NR_ACTIVE_ANON) +
		   node_page_state(pgdat, NR_INACTIVE_ANON)) << (PAGE_SHIFT - 10);

	file_kb = (node_page_state(pgdat, NR_ACTIVE_FILE) +
		   node_page_state(pgdat, NR_INACTIVE_FILE)) << (PAGE_SHIFT - 10);
	dirty_kb = node_page_state(pgdat, NR_FILE_DIRTY) << (PAGE_SHIFT - 10);
	clean_kb = (file_kb > dirty_kb) ? (file_kb - dirty_kb) : 0;

	st->anon_below_min  = anon_kb  < total_kb * anon_min_ratio  / 100;
	st->clean_below_min = clean_kb < total_kb * clean_min_ratio / 100;

	/*
	 * get_nr_swap_pages() instead of si_meminfo()'s freeswap: same
	 * number, without walking every swap device under swap_lock. It
	 * returns 0 with CONFIG_SWAP=n, in which case a swap floor is
	 * meaningless -- don't let that read as permanently breached.
	 */
	free_swap_kb = (unsigned long)get_nr_swap_pages() << (PAGE_SHIFT - 10);
	st->swap_floor_breached = total_swap_pages &&
				  free_swap_kb < reserved_swap_floor_kb;

	/* Escalation tier, concept from prlmk: crossing the clean-file
	 * floor is a harder signal than trend alone -- widen the victim
	 * pool in psr_select_victim(). */
	st->escalated = st->clean_below_min;
}

/* ------------------------------------------------------------------
 * Pending-kill gate.
 *
 * Don't dispatch a second kill until the previous victim's memory has
 * actually come back -- concept from simple_lmk. The mm is reported
 * released by psr_lmk_mm_freed() from __mmput(); the timeout is only a
 * backstop for a victim wedged in D-state.
 *
 * The previous revision polled find_task_by_vpid() on a raw pid and gave
 * up after a fixed 200ms. Two problems: a pid can be recycled onto an
 * unrelated new task within that window (so the gate would keep blocking
 * on a stranger), and task exit is not the same event as memory being
 * freed -- the mm can outlive the task struct. Holding an mm reference
 * and waiting for __mmput() removes both.
 * ------------------------------------------------------------------ */
static struct mm_struct *psr_victim_mm;	/* kthread-owned, holds mmgrab ref */
static unsigned long psr_victim_deadline;
static atomic_t psr_victim_freed = ATOMIC_INIT(0);

#define PSR_KILL_TIMEOUT_MS 2000

void __psr_lmk_mm_freed(struct mm_struct *mm)
{
	/*
	 * Called from __mmput() for every dying address space. Reading
	 * psr_victim_mm unlocked is fine: it is only ever written by the
	 * kthread, and a stale read can only make us miss a match, which
	 * degrades to the timeout path. Comparing pointers -- not pids --
	 * so a recycled pid can never produce a false match.
	 */
	if (READ_ONCE(psr_victim_mm) == mm)
		atomic_set(&psr_victim_freed, 1);
}
EXPORT_SYMBOL_GPL(__psr_lmk_mm_freed);

static bool psr_kill_in_flight(void)
{
	if (!psr_victim_mm)
		return false;

	if (atomic_read(&psr_victim_freed))
		goto release;

	if (time_after(jiffies, psr_victim_deadline)) {
		pr_warn("victim mm not released within %dms, unblocking\n",
			PSR_KILL_TIMEOUT_MS);
		goto release;
	}

	return true;

release:
	mmdrop(psr_victim_mm);
	WRITE_ONCE(psr_victim_mm, NULL);
	atomic_set(&psr_victim_freed, 0);
	return false;
}

/* ------------------------------------------------------------------
 * Victim selection: highest oom_score_adj, tie-broken by largest RSS,
 * discounted by accumulated CPU time -- an app you've actively used
 * scores lower than an idle one at the same oom_score_adj/RSS (concept
 * from prlmk's stime+utime sort).
 *
 * Fixes relative to the previous revision, which had three bugs that
 * would each have produced a wrong or fatal outcome:
 *
 *  1. It returned the winning task_struct but had called task_unlock()
 *     on it inside the loop and only did get_task_struct() at the very
 *     end -- the winner was unreferenced for the rest of the walk, so it
 *     could be freed underneath us. Now the reference is taken at the
 *     moment the task becomes the leader, and the previous leader's is
 *     dropped.
 *
 *  2. thread_group_cputime() walks every thread in the group and takes
 *     siglock. Running that for *every* process on every evaluation is
 *     both slow and a lock-ordering hazard under task_lock. Replaced
 *     with the group's already-summed signal->stime/utime plus the
 *     leader's own, which needs no walk and no extra lock.
 *
 *  3. It never skipped kthreads, dying tasks, or the global init
 *     process. find_lock_task_mm() filters kthreads, but a task already
 *     in the middle of exiting is a wasted kill, and current->mm-less
 *     checks alone don't cover SIGNAL_GROUP_EXIT.
 * ------------------------------------------------------------------ */
static struct task_struct *psr_select_victim(const struct psr_node_state *st)
{
	struct task_struct *p, *victim = NULL;
	long best_score = LONG_MIN;
	int score_floor = st->escalated ? escalated_min_oom_score_adj
					: min_oom_score_adj;
	u64 now_ns = ktime_get_ns();
	u64 grace_ns = (u64)launch_grace_ms * NSEC_PER_MSEC;

	rcu_read_lock();
	for_each_process(p) {
		struct signal_struct *sig = p->signal;
		struct task_struct *t;
		unsigned long rss_kb, cputime_s;
		long score, adj;

		/* Cheap filters first, before find_lock_task_mm() has to
		 * take task_lock on anything. */
		adj = READ_ONCE(sig->oom_score_adj);
		if (adj < score_floor)
			continue;

		/* Already dying: killing it again frees nothing sooner. */
		if (sig->flags & (SIGNAL_GROUP_EXIT | SIGNAL_GROUP_COREDUMP))
			continue;
		if (thread_group_empty(p) && (p->flags & PF_EXITING))
			continue;

		/*
		 * Launch grace. An app that started moments ago is the most
		 * likely cause of the pressure spike we are responding to and
		 * the worst possible victim -- killing it fails the launch the
		 * user just asked for, visibly. p->start_time is monotonic ns
		 * (kernel/fork.c), so this is a plain subtraction.
		 */
		if (grace_ns && now_ns - p->start_time < grace_ns)
			continue;

		t = find_lock_task_mm(p);
		if (!t)
			continue;

		if (unlikely(is_global_init(t))) {
			task_unlock(t);
			continue;
		}

		rss_kb = get_mm_rss(t->mm) << (PAGE_SHIFT - 10);

		/*
		 * Accumulated CPU time as a proxy for "how much this app has
		 * actually been used". signal->{s,u}time is the sum already
		 * accrued by exited threads; adding the leader's own covers
		 * the common single-heavy-thread case without walking the
		 * group.
		 */
		cputime_s = (unsigned long)((sig->stime + sig->utime +
					     t->stime + t->utime) / NSEC_PER_SEC);

		task_unlock(t);

		score = adj * 1000 + (long)rss_kb;

		/* Killing an anon-heavy task when anon is already starved
		 * doesn't relieve the actual pressure source. */
		if (st->anon_below_min)
			score -= (long)rss_kb / 2;

		if (usage_time_weight_pct)
			score -= (long)(cputime_s * usage_time_weight_pct) / 100;

		if (score > best_score) {
			struct task_struct *old = victim;

			/*
			 * Pin the new leader before dropping the old one, so
			 * the returned task is referenced continuously from
			 * the moment it wins.
			 */
			get_task_struct(t);
			victim = t;
			best_score = score;
			if (old)
				put_task_struct(old);
		}
	}
	rcu_read_unlock();

	return victim;
}

/*
 * Dispatch the kill.
 *
 * Beyond plain send_sig(), this mirrors simple_lmk's acceleration: mark
 * the mm as an OOM victim so exit_mmap() reaps anonymous memory first,
 * force the signal past any blocking, thaw the task if it's frozen (a
 * frozen task cannot act on a signal at all -- a real hang source in the
 * previous revision, which would then have blocked every subsequent kill
 * until its gate timed out), and briefly elevate the group to RT so the
 * memory actually comes back promptly instead of whenever a background
 * cgroup next gets scheduled.
 */
static void psr_kill_victim(struct task_struct *victim)
{
	static const struct sched_param rt_prio = { .sched_priority = 1 };
	struct task_struct *t;
	struct mm_struct *mm;

	task_lock(victim);
	mm = victim->mm;
	if (!mm) {
		task_unlock(victim);
		return;
	}
	/* Keep the mm alive so the pending-kill gate can watch it, and so
	 * __mmput()'s pointer comparison stays valid. */
	mmgrab(mm);
	set_bit(MMF_OOM_VICTIM, &mm->flags);
	task_unlock(victim);

	do_send_sig_info(SIGKILL, SEND_SIG_FORCED, victim, PIDTYPE_TGID);

	rcu_read_lock();
	for_each_thread(victim, t) {
		set_tsk_thread_flag(t, TIF_MEMDIE);
		sched_setscheduler_nocheck(t, SCHED_RR, &rt_prio);
		/* Signals can't wake frozen tasks; only a thaw can. */
		__thaw_task(t);
	}
	rcu_read_unlock();

	/* Let it run anywhere so exit_mmap() isn't stuck behind a busy
	 * little core. This doesn't schedule. */
	set_cpus_allowed_ptr(victim, cpu_all_mask);

	WRITE_ONCE(psr_victim_mm, mm);
	atomic_set(&psr_victim_freed, 0);
	psr_victim_deadline = jiffies + msecs_to_jiffies(PSR_KILL_TIMEOUT_MS);
}

static void psr_bypass_and_kill(const struct psr_node_state *st)
{
	struct task_struct *victim;

	victim = psr_select_victim(st);
	if (!victim) {
		pr_info("regression signaled, no eligible victim\n");
		return;
	}

	pr_warn("BYPASS pid=%d comm=%s adj=%d escalated=%d anon_below_min=%d clean_below_min=%d swap_floor_breached=%d pressure=%lu streak=%u slope=%ld swap_rf=%lu allocfail=%lu react=%lu%s\n",
		victim->pid, victim->comm, victim->signal->oom_score_adj,
		st->escalated, st->anon_below_min, st->clean_below_min,
		st->swap_floor_breached, st->pressure, st->streak, st->slope,
		st->swap_refaults, st->alloc_failures, st->anon_reactivations,
		dry_run ? " (dry-run)" : "");

	if (!dry_run)
		psr_kill_victim(victim);

	put_task_struct(victim);
}

/* ------------------------------------------------------------------
 * The decision engine, running on the psr_lmkd kthread.
 * ------------------------------------------------------------------ */
static void psr_evaluate(unsigned long pressure)
{
	struct psr_counters c;
	struct psr_node_state st;
	bool corroborated, sustained;
	bool regression = false;
	long slope;

	/* Cheapest gate first: if a kill is still outstanding, there is
	 * nothing useful to do and no reason to touch anything else. */
	if (psr_kill_in_flight())
		return;

	psr_drain_counters(&c);
	psr_compute_node_state(&st);
	st.pressure = pressure;

	slope = psr_push_sample_and_slope(pressure);

	st.slope              = slope;
	st.swap_refaults      = c.swap_refaults;
	st.alloc_failures     = c.alloc_failures;
	st.anon_reactivations = c.anon_reactivations;

	/*
	 * Sustain tracking. pressure >= thrash_hard_limit on a single
	 * sample means very little: vmpressure_prio() manufactures
	 * pressure=100 from a scan-depth threshold crossing, with no
	 * reference to whether reclaim is actually failing. Only a run of
	 * consecutive high samples indicates the condition persists.
	 */
	if (pressure >= thrash_hard_limit)
		psr_high_streak++;
	else
		psr_high_streak = 0;

	sustained = psr_high_streak >= sustain_evals;
	st.streak = psr_high_streak;

	/*
	 * Corroboration. Every input below is independent of vmpressure,
	 * so requiring one of them prevents a pure pressure artifact from
	 * being sufficient on its own.
	 *
	 * The first dry-run trace on a 4GB device showed ten consecutive
	 * BYPASS decisions with anon_below_min=0 clean_below_min=0
	 * swap_floor_breached=0 -- i.e. no independent signal agreed that
	 * memory was tight, and all ten were false positives driven by the
	 * hard limit alone. Hence the AND rather than an OR.
	 */
	corroborated = st.anon_below_min || st.clean_below_min ||
		       st.swap_floor_breached || c.alloc_failures > 0 ||
		       c.swap_refaults > 0;

	/* Sustained high pressure, agreed with by something that is not
	 * vmpressure. */
	if (sustained && corroborated)
		regression = true;

	/* A real upward trend in pressure. Independent of the hard limit,
	 * and only computable now that wakeup_pressure sits low enough to
	 * give the window some range. */
	if (!regression && slope >= (long)regression_slope_thresh &&
	    corroborated)
		regression = true;

	/*
	 * Corroboration (le9uo's ratio concept): real anon/swap refaults
	 * strengthen a borderline reading when the clean-file floor is
	 * also under pressure. A pure file-cache refault burst with no
	 * swap activity doesn't count -- that's ordinary page-cache churn,
	 * not swap thrash.
	 */
	if (!regression && st.clean_below_min && c.swap_refaults > 0)
		regression = true;

	/* Reactivation churn: memory being pulled straight back off the
	 * inactive list is thrash even when the pressure number itself
	 * hasn't peaked. */
	if (!regression && st.clean_below_min && c.anon_reactivations > 64)
		regression = true;

	pr_debug("pressure=%lu streak=%u slope=%ld react=%lu allocfail=%lu swap_rf=%lu file_rf=%lu corrob=%d regression=%d\n",
		 pressure, psr_high_streak, slope, c.anon_reactivations,
		 c.alloc_failures, c.swap_refaults, c.file_refaults,
		 corroborated, regression);

	if (regression || st.swap_floor_breached) {
		psr_high_streak = 0;
		psr_bypass_and_kill(&st);
	}
}

/* ------------------------------------------------------------------
 * Trigger: the global vmpressure notifier chain.
 *
 * The previous revision hooked vmpressure_work_fn() directly and ran the
 * whole decision -- task-list walk included -- inline on the system
 * workqueue. That call site is also MEMCG-only, and this defconfig had
 * to turn MEMCG on to reach it, which costs page-counter accounting on
 * every charge/uncharge kernel-wide. That combination is the main reason
 * the UI stuttered regardless of what was running.
 *
 * The notifier chain is the same trigger simple_lmk used, it exists
 * independently of CONFIG_MEMCG, and the callback here does nothing but
 * a comparison and a wakeup.
 * ------------------------------------------------------------------ */
static DECLARE_WAIT_QUEUE_HEAD(psr_waitq);
static atomic_t psr_needs_eval = ATOMIC_INIT(0);
static unsigned long psr_last_pressure;

static int psr_vmpressure_cb(struct notifier_block *nb, unsigned long pressure,
			     void *data)
{
	if (pressure >= wakeup_pressure) {
		WRITE_ONCE(psr_last_pressure, pressure);
		atomic_set(&psr_needs_eval, 1);
		smp_mb__after_atomic();
		if (waitqueue_active(&psr_waitq))
			wake_up(&psr_waitq);
	}

	return NOTIFY_OK;
}

static struct notifier_block psr_vmpressure_nb = {
	.notifier_call = psr_vmpressure_cb,
	.priority = INT_MAX,
};

static int psr_lmkd_thread(void *data)
{
	static const struct sched_param rt_prio = {
		.sched_priority = MAX_RT_PRIO - 1
	};

	sched_setscheduler_nocheck(current, SCHED_RR, &rt_prio);
	set_freezable();

	while (!kthread_should_stop()) {
		wait_event_freezable(psr_waitq,
				     atomic_read(&psr_needs_eval) ||
				     kthread_should_stop());
		if (kthread_should_stop())
			break;

		psr_evaluate(READ_ONCE(psr_last_pressure));

		/*
		 * Rate limit. A kill takes real time to pay off, and the
		 * notifier can fire far faster than that -- re-walking the
		 * task list in a tight loop would burn CPU that the UI
		 * needs, which is precisely the failure being fixed here.
		 */
		if (min_eval_interval_ms)
			schedule_timeout_interruptible(
				msecs_to_jiffies(min_eval_interval_ms));

		/*
		 * Clear the flag *after* sleeping, not before evaluating.
		 * The notifier re-arms it on every vmpressure window, so
		 * clearing first meant any notification landing during the
		 * sleep guaranteed an immediate second evaluation of the
		 * same event -- visible in the first dry-run trace as every
		 * BYPASS line appearing twice, ~100ms apart. Coalescing here
		 * makes min_eval_interval_ms drop redundant work instead of
		 * merely delaying it.
		 */
		atomic_set(&psr_needs_eval, 0);
	}

	return 0;
}

/* ------------------------------------------------------------------
 * proc interface: /proc/psr_lmk/status
 * ------------------------------------------------------------------ */
static int psr_status_show(struct seq_file *m, void *v)
{
	struct psr_node_state st;

	psr_compute_node_state(&st);

	seq_printf(m,
		   "active_profile: %s\n"
		   "dry_run: %d\n"
		   "reserved_swap_floor_kb: %lu\n"
		   "anon_min_ratio: %u\n"
		   "clean_min_ratio: %u\n"
		   "regression_slope_thresh: %u\n"
		   "thrash_hard_limit: %u\n"
		   "wakeup_pressure: %u\n"
		   "sustain_evals: %u\n"
		   "launch_grace_ms: %u\n"
		   "min_eval_interval_ms: %u\n"
		   "min_oom_score_adj: %d\n"
		   "escalated_min_oom_score_adj: %d\n"
		   "usage_time_weight_pct: %u\n"
		   "---\n"
		   "anon_below_min: %d\n"
		   "clean_below_min: %d\n"
		   "swap_floor_breached: %d\n"
		   "escalated: %d\n"
		   "kill_in_flight: %d\n"
		   "high_streak: %u\n"
		   "last_pressure: %lu\n",
		   psr_lmk_active_profile, dry_run, reserved_swap_floor_kb,
		   anon_min_ratio, clean_min_ratio, regression_slope_thresh,
		   thrash_hard_limit, wakeup_pressure, sustain_evals,
		   launch_grace_ms, min_eval_interval_ms,
		   min_oom_score_adj, escalated_min_oom_score_adj,
		   usage_time_weight_pct, st.anon_below_min, st.clean_below_min,
		   st.swap_floor_breached, st.escalated,
		   READ_ONCE(psr_victim_mm) != NULL,
		   READ_ONCE(psr_high_streak),
		   READ_ONCE(psr_last_pressure));

	return 0;
}

static int psr_status_open(struct inode *inode, struct file *file)
{
	return single_open(file, psr_status_show, NULL);
}

static const struct file_operations psr_status_fops = {
	.open    = psr_status_open,
	.read    = seq_read,
	.llseek  = seq_lseek,
	.release = single_release,
};

/* ------------------------------------------------------------------
 * Bring-up.
 *
 * Deferred to the first write to the legacy lowmemorykiller.minfree
 * node, the same way simple_lmk did it. That write comes from
 * init/lmkd once userspace is far enough along to want an LMK, which
 * means the kthread doesn't exist and -- more importantly -- the static
 * key stays off, so the mm hooks are patched-out NOPs, for the whole of
 * boot.
 *
 * That node has to exist for a second reason: some Android init/vendor
 * scripts and some lmkd branches treat its absence as "no LMK present"
 * and reboot. Note this requires CONFIG_ANDROID_PSR_LMK=y, not =m -- the
 * dot in "lowmemorykiller.minfree" is only split into a synthesized
 * /sys/module/lowmemorykiller/ directory for built-in params.
 *
 * MODULE_PARAM_PREFIX is a plain preprocessor macro affecting every
 * module_param() that follows it in this file, so this block stays at
 * the very end, after every real tunable is registered, and the prefix
 * is restored immediately. Do not add module_param() calls between the
 * #define and #undef below.
 * ------------------------------------------------------------------ */
static struct proc_dir_entry *psr_proc_dir;
static struct task_struct *psr_lmkd_task;

static int psr_lmk_start(void)
{
	struct task_struct *thread;
	int ret;

	psr_lmk_auto_tune();
	psr_sanity_check();

	thread = kthread_run(psr_lmkd_thread, NULL, "psr_lmkd");
	if (IS_ERR(thread)) {
		pr_err("failed to start kthread: %ld\n", PTR_ERR(thread));
		return PTR_ERR(thread);
	}
	psr_lmkd_task = thread;

	ret = vmpressure_notifier_register(&psr_vmpressure_nb);
	if (ret) {
		pr_err("failed to register vmpressure notifier: %d\n", ret);
		kthread_stop(psr_lmkd_task);
		psr_lmkd_task = NULL;
		return ret;
	}

	psr_proc_dir = proc_mkdir("psr_lmk", NULL);
	if (psr_proc_dir)
		proc_create("status", 0444, psr_proc_dir, &psr_status_fops);

	/* Last: only now do the mm hooks start costing anything. */
	static_branch_enable(&psr_lmk_key);

	pr_info("started (dry_run=%d wakeup_pressure=%u thrash_hard_limit=%u sustain_evals=%u launch_grace_ms=%u)\n",
		dry_run, wakeup_pressure, thrash_hard_limit, sustain_evals,
		launch_grace_ms);
	return 0;
}

static int psr_lmk_compat_minfree_set(const char *val,
				      const struct kernel_param *kp)
{
	static atomic_t init_done = ATOMIC_INIT(0);

	if (!atomic_cmpxchg(&init_done, 0, 1))
		psr_lmk_start();

	/* The legacy minfree values themselves are ignored -- PSR-LMK's
	 * thresholds come from its own params and RAM profile. */
	return 0;
}

static const struct kernel_param_ops psr_lmk_compat_minfree_ops = {
	.set = psr_lmk_compat_minfree_set,
};

#undef MODULE_PARAM_PREFIX
#define MODULE_PARAM_PREFIX "lowmemorykiller."
module_param_cb(minfree, &psr_lmk_compat_minfree_ops, NULL, 0200);
#undef MODULE_PARAM_PREFIX
#define MODULE_PARAM_PREFIX ""

MODULE_LICENSE("GPL v2");
MODULE_DESCRIPTION("PSR-LMK: Protected-Swap Regression Low Memory Killer");
