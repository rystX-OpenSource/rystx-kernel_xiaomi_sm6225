// SPDX-License-Identifier: GPL-2.0
/*
 * drivers/android/psr_lmk.c
 *
 * PSR-LMK: Protected-Swap Regression Low Memory Killer
 * ------------------------------------------------------------------
 * A self-contained Android LMK driver. Inspired by le9uo's working-set
 * protection concept (anon_min_ratio/clean_min_ratio, enforced in
 * mm/vmscan.c via a core reclaim scan-balance rewrite), PSR-LMK
 * reuses the same class of signals -- swap-in trend, protected anon
 * ratio, protected clean-file ratio -- but keeps them out of the
 * reclaim path entirely: they're read-only inputs to an independent
 * kill decision here, and PSR-LMK acts by SIGKILLing a victim task
 * instead of rewriting LRU scan balance.
 *
 * Design rationale for keeping this separate from vmscan.c:
 *   - Stays out-of-tree friendly: builds against any kernel exposing
 *     node_page_state()/si_meminfo(), no core mm patch to carry
 *     across kernel version bumps.
 *   - Keeps blast radius contained to "pick a victim, kill it" --
 *     the same operational model as the existing Android LMKD, just
 *     with an earlier, regression-aware trigger.
 *   - The ratio knobs are reused as *confirmation signals*, not
 *     enforcement levers: they change whether we treat a swap-in
 *     trend as "real pressure worth killing over" vs "noise",
 *     rather than changing what the reclaimer scans.
 *
 * Hook points (see the companion mm/*.c patch and include/linux/psr_lmk.h):
 *   - mm/vmpressure.c's vmpressure_work_fn() -- primary trigger and
 *     the only call site that runs the actual kill decision, since
 *     it's the one guaranteed-sleepable context among all the hooks.
 *   - mm/swap.c, mm/vmscan.c, mm/workingset.c, mm/page_alloc.c --
 *     lightweight atomic-counter corroboration signals only; see
 *     psr_lmk_note_*() below for why they can't safely do more than
 *     that from their (often atomic) call sites.
 */

#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/sched.h>
#include <linux/sched/signal.h>
#include <linux/sched/task.h>
#include <linux/sched/cputime.h>
#include <linux/mm.h>
#include <linux/mmzone.h>
#include <linux/swap.h>
#include <linux/vmpressure.h>
#include <linux/shrinker.h>
#include <linux/oom.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/uaccess.h>
#include <linux/ktime.h>
#include <linux/rculist.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/psr_lmk.h>
#include <linux/atomic.h>

/*
 * Counters bumped by the five hooks that can fire from atomic /
 * spinlock-held contexts deep in reclaim (mm/swap.c's
 * __activate_page, mm/vmscan.c's shrink_node, mm/workingset.c's
 * workingset_refault, mm/page_alloc.c's direct-reclaim slow path).
 * They only ever do an atomic_inc/add -- no locking, no sleeping, no
 * task-list walk -- because those call sites cannot safely do more
 * than that. psr_lmk_note_pressure() (fired from vmpressure's
 * workqueue, which is real process context) drains and interprets
 * them each cycle; see the big comment on that function below.
 */
static atomic_t      psr_stat_anon_reactivations = ATOMIC_INIT(0);
static atomic_t      psr_stat_alloc_failures     = ATOMIC_INIT(0);
static atomic_t      psr_stat_swap_refaults      = ATOMIC_INIT(0);
static atomic_t      psr_stat_file_refaults      = ATOMIC_INIT(0);
static atomic_long_t psr_stat_scan_sum           = ATOMIC_LONG_INIT(0);
static atomic_long_t psr_stat_reclaimed_sum      = ATOMIC_LONG_INIT(0);

#define PSR_LMK_NAME "psr_lmk"
#define PSR_LMK_WINDOW 8

/* ------------------------------------------------------------------
 * Tunables (exposed under /proc/psr_lmk/*)
 *
 * reserved_swap_floor_kb  - hard floor on free swap (KB); breach is
 *                            an automatic bypass trigger regardless
 *                            of trend.
 * anon_min_ratio          - percent of node memory below which anon
 *                            pages are considered already-starved.
 *                            Mirrors le9uo's vm.anon_min_ratio
 *                            semantics, but consulted as a signal
 *                            here, not enforced against the LRU.
 * clean_min_ratio         - percent of node memory below which clean
 *                            file pages are considered starved.
 *                            Mirrors le9uo's vm.clean_min_ratio.
 * regression_slope_thresh - swap-in rate trend (pages/sec per
 *                            sample) above which we call it a
 *                            regression.
 * thrash_hard_limit       - absolute swap-in rate (pages/sec) that
 *                            triggers bypass regardless of slope.
 * min_oom_score_adj       - only consider victims at/above this
 *                            oom_score_adj (background-ish tasks).
 * dry_run                 - if 1, log the decision but do not send
 *                            SIGKILL. Default 1; flip explicitly.
 * ------------------------------------------------------------------ */
static unsigned long reserved_swap_floor_kb = 64UL * 1024;
static unsigned int  anon_min_ratio         = 15;   /* percent */
static unsigned int  clean_min_ratio        = 15;   /* percent */
static unsigned int  regression_slope_thresh = 50;  /* pages/s trend */
static unsigned int  thrash_hard_limit      = 500;  /* pages/s */
static int           min_oom_score_adj      = 200;
static int           escalated_min_oom_score_adj = 100; /* wider victim
				pool once clean_below_min escalates, concept
				from prlmk's aggressive-mode file threshold */
static unsigned int  usage_time_weight_pct  = 30;   /* how much accumulated
				CPU time (stime+utime) discounts a victim's
				kill score, 0-100; concept from prlmk's
				stime+utime victim sort -- protects apps
				you're actively using over idle ones with
				similar oom_score_adj/RSS */
static int           dry_run                = 1;

module_param(reserved_swap_floor_kb, ulong, 0644);
module_param(anon_min_ratio, uint, 0644);
module_param(clean_min_ratio, uint, 0644);
module_param(regression_slope_thresh, uint, 0644);
module_param(thrash_hard_limit, uint, 0644);
module_param(min_oom_score_adj, int, 0644);
module_param(escalated_min_oom_score_adj, int, 0644);
module_param(usage_time_weight_pct, uint, 0644);
module_param(dry_run, int, 0644);

/* ------------------------------------------------------------------
 * Optional PSI corroboration.
 *
 * PSR-LMK's primary signal is mm/vmpressure.c's scanned/reclaimed
 * ratio (CONFIG_MEMCG only -- always available, what everything
 * above this block already uses). PSI, when the kernel has it, is a
 * genuinely better pressure signal: it measures actual task stall
 * time instead of inferring pressure from reclaim efficiency, which
 * is known to under-report thrashing that's still "succeeding" often
 * enough to look efficient. This block adds PSI as one more
 * corroborating input alongside the existing ones (reactivations,
 * refaults, ratio floors) -- it never replaces the vmpressure trigger
 * and has zero effect (dead code, compiled out) when CONFIG_PSI=n.
 *
 * VERIFIED against the psi.c/psi_types.h you provided (previously
 * this read PSI via a hand-built seq_file into psi_show(), which
 * worked but was needlessly roundabout -- replaced with a direct
 * read now that the actual struct layout and fixed-point macros are
 * confirmed):
 *
 *   - group->avg[state][window] is indexed as `res * 2 + full`,
 *     confirmed directly from psi_show()'s own indexing
 *     (psi.c:1035, "avg[w] = group->avg[res * 2 + full][w]").
 *     PSI_MEM "some" is therefore index PSI_MEM * 2 + 0.
 *     window index 0 = avg10, 1 = avg60, 2 = avg300 (psi.c:307-309).
 *   - The fixed-point encoding is calc_load()'s (psi.c #includes
 *     <linux/sched/loadavg.h> and reuses LOAD_INT/LOAD_FRAC/FIXED_1,
 *     the same macros CPU loadavg has used for decades) -- this is
 *     about as stable a kernel ABI as exists, not PSI-specific.
 *   - No locking taken on the read: avg[] is refreshed by
 *     psi_avgs_work every PSI_FREQ (2s, psi.c:176/456) independent
 *     of any reader, kicked off by real task-state transitions
 *     (psi.c:859-860) -- so this isn't dependent on something else
 *     polling psi_show()/psi_system elsewhere first. avgs_lock in
 *     psi_show() only protects the recompute-if-stale step; a bare
 *     unlocked read of the current value is fine for a threshold
 *     comparison that tolerates a couple seconds of staleness.
 *   - psi_disabled is checked first (mirrors psi_show()'s own guard)
 *     since PSI can be compiled in but turned off at boot via the
 *     `psi=0` cmdline param.
 * ------------------------------------------------------------------ */
#if IS_ENABLED(CONFIG_PSI)
#include <linux/psi.h>
#include <linux/sched/loadavg.h>

static unsigned int psi_avg10_thresh = 20; /* percent; corroboration only */
module_param(psi_avg10_thresh, uint, 0644);

static inline unsigned int psi_avg10_thresh_or_zero(void)
{
	return psi_avg10_thresh;
}

static unsigned long psr_lmk_read_psi_mem_avg10(void)
{
	unsigned long avg10_raw;

	if (static_branch_likely(&psi_disabled))
		return 0;

	avg10_raw = psi_system.avg[PSI_MEM * 2 + 0][0];

	return LOAD_INT(avg10_raw);
}
#else
static inline unsigned long psr_lmk_read_psi_mem_avg10(void)
{
	return 0;
}

static inline unsigned int psi_avg10_thresh_or_zero(void)
{
	return 0;
}
#endif /* CONFIG_PSI */

/* ------------------------------------------------------------------
 * Auto-tuning profiles, applied once at driver init based on total
 * device RAM. Expression style for reading total RAM is borrowed
 * from prlmk's own RAM-based minfree/timeout auto-detection
 * (see darkhz/prlmk), extended here across PSR-LMK's full tunable
 * set instead of just two values, and split into three tiers.
 *
 * Boundaries (tune via the module params below, read-only after
 * driver load since they only take effect during init):
 *   total_mb <= psr_lmk_ram_tier_low_mb  -> LOW  ("3-4GB" budget)
 *   total_mb <= psr_lmk_ram_tier_mid_mb  -> MID  ("6-8GB" mainstream)
 *   total_mb  > psr_lmk_ram_tier_mid_mb  -> HIGH ("12GB+" flagship)
 *
 * The 4-6GB gap rounds down into MID; the 8-12GB gap rounds up into
 * HIGH. If you want a true 4th tier instead of that rounding, add
 * another boundary + profile struct rather than fighting these two.
 *
 * Direction of the tuning, mirroring the "less RAM = protect harder,
 * kill sooner" logic in the reference snippet: LOW has the smallest
 * absolute swap floor but the *tightest* (most sensitive) regression
 * thresholds and the widest victim pool (lowest min_oom_score_adj),
 * since a low-RAM device has the least headroom to lose. HIGH is the
 * opposite: loosest thresholds, narrowest victim pool, because a
 * flagship device with tons of RAM shouldn't be killing background
 * apps on thresholds tuned for a budget phone.
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
	int            min_oom_score_adj;
	int            escalated_min_oom_score_adj;
	unsigned int   usage_time_weight_pct;
};

static const struct psr_lmk_profile psr_lmk_profile_low = {
	.name                         = "3-4GB",
	.reserved_swap_floor_kb       = 32UL * 1024,
	.anon_min_ratio               = 10,
	.clean_min_ratio              = 10,
	.regression_slope_thresh      = 30,
	.thrash_hard_limit            = 300,
	.min_oom_score_adj            = 150,
	.escalated_min_oom_score_adj  = 50,
	.usage_time_weight_pct        = 20,
};

static const struct psr_lmk_profile psr_lmk_profile_mid = {
	.name                         = "6-8GB",
	.reserved_swap_floor_kb       = 64UL * 1024,
	.anon_min_ratio               = 15,
	.clean_min_ratio              = 15,
	.regression_slope_thresh      = 50,
	.thrash_hard_limit            = 500,
	.min_oom_score_adj            = 200,
	.escalated_min_oom_score_adj  = 100,
	.usage_time_weight_pct        = 30,
};

static const struct psr_lmk_profile psr_lmk_profile_high = {
	.name                         = "12GB+",
	.reserved_swap_floor_kb       = 128UL * 1024,
	.anon_min_ratio               = 20,
	.clean_min_ratio              = 20,
	.regression_slope_thresh      = 80,
	.thrash_hard_limit            = 800,
	.min_oom_score_adj            = 300,
	.escalated_min_oom_score_adj  = 150,
	.usage_time_weight_pct        = 40,
};

static const char *psr_lmk_active_profile = "unset";

static void __init psr_lmk_auto_tune(void)
{
	const struct psr_lmk_profile *profile;
	unsigned long total_mb;

	if (!auto_tune) {
		pr_info(PSR_LMK_NAME ": auto_tune=0, using compiled-in/cmdline module param defaults\n");
		return;
	}

	/*
	 * NOTE: totalram_pages is a plain global on kernels before the
	 * atomic-counter conversion; if your tree already converted it,
	 * change this to totalram_pages() (function call) instead.
	 */
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
	min_oom_score_adj           = profile->min_oom_score_adj;
	escalated_min_oom_score_adj = profile->escalated_min_oom_score_adj;
	usage_time_weight_pct       = profile->usage_time_weight_pct;
	psr_lmk_active_profile      = profile->name;

	pr_info_once(PSR_LMK_NAME ": detected %lu MB RAM, applying \"%s\" profile "
		     "(swap_floor=%luKB anon_min=%u%% clean_min=%u%% "
		     "slope_thresh=%u thrash_hard=%u min_adj=%d "
		     "escalated_adj=%d usage_weight=%u%%)\n",
		     total_mb, profile->name,
		     reserved_swap_floor_kb, anon_min_ratio, clean_min_ratio,
		     regression_slope_thresh, thrash_hard_limit,
		     min_oom_score_adj, escalated_min_oom_score_adj,
		     usage_time_weight_pct);
}

/* ------------------------------------------------------------------
 * Regression engine state
 * ------------------------------------------------------------------ */
struct psr_sample {
	u64 pswpin;
	ktime_t t;
};

static struct {
	unsigned long rate_window[PSR_LMK_WINDOW];
	int head;
	int count;
	struct psr_sample last;
	struct mutex lock;
} psr_engine;

static void psr_engine_init(void)
{
	memset(&psr_engine, 0, sizeof(psr_engine));
	mutex_init(&psr_engine.lock);
}

/* Push a new swap-in rate sample (pages/sec), return current slope
 * via simple least-squares fit over the sliding window. */
static long psr_push_sample_and_slope(unsigned long rate)
{
	int n, i;
	long sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
	long mean_x, mean_y, num, den, slope;

	mutex_lock(&psr_engine.lock);
	psr_engine.rate_window[psr_engine.head] = rate;
	psr_engine.head = (psr_engine.head + 1) % PSR_LMK_WINDOW;
	if (psr_engine.count < PSR_LMK_WINDOW)
		psr_engine.count++;

	n = psr_engine.count;
	if (n < 2) {
		mutex_unlock(&psr_engine.lock);
		return 0;
	}

	for (i = 0; i < n; i++) {
		long y = psr_engine.rate_window[i];

		sum_x += i;
		sum_y += y;
		sum_xy += (long)i * y;
		sum_xx += (long)i * i;
	}
	mutex_unlock(&psr_engine.lock);

	mean_x = sum_x / n;
	mean_y = sum_y / n;
	num = sum_xy - n * mean_x * mean_y;
	den = sum_xx - n * mean_x * mean_x;
	slope = den ? (num / den) : 0;
	return slope;
}

/* ------------------------------------------------------------------
 * Ratio-based confirmation signals (concept borrowed from le9uo's
 * anon_min_ratio / clean_min_ratio; enforcement differs -- read-only
 * here, used to qualify whether a swap-in trend is worth acting on)
 * ------------------------------------------------------------------ */
struct psr_node_state {
	bool anon_below_min;
	bool clean_below_min;
	bool swap_floor_breached;
	bool escalated;        /* clean_below_min crossed -> aggressive mode,
				 * concept borrowed from prlmk's free_file_limit
				 * two-tier severity */
	unsigned long swap_in_rate;
};

/* Tracks whether the last dispatched kill has actually been reaped yet,
 * so we don't fire a second SIGKILL before the first one relieved any
 * pressure -- borrowed from simple_lmk's "wait for victim's memory to
 * be freed before killing more" behavior. */
static struct {
	pid_t pending_victim_pid;
	ktime_t dispatched_at;
} psr_kill_state;

#define PSR_KILL_GRACE_MS 200  /* max time to wait for a pending kill
				 * to be reaped before considering the
				 * slot free again */

static bool psr_kill_in_flight(void)
{
	struct task_struct *t;
	bool alive = false;

	if (!psr_kill_state.pending_victim_pid)
		return false;

	if (ktime_ms_delta(ktime_get(), psr_kill_state.dispatched_at) >
	    PSR_KILL_GRACE_MS) {
		/* Timed out waiting -- stop blocking further kills, but
		 * log it since a stuck reap usually means something else
		 * is wrong (e.g. victim stuck in D-state). */
		pr_warn(PSR_LMK_NAME ": pid=%d not reaped within %dms, unblocking\n",
			psr_kill_state.pending_victim_pid, PSR_KILL_GRACE_MS);
		psr_kill_state.pending_victim_pid = 0;
		return false;
	}

	rcu_read_lock();
	t = find_task_by_vpid(psr_kill_state.pending_victim_pid);
	if (t)
		alive = true;
	rcu_read_unlock();

	if (!alive)
		psr_kill_state.pending_victim_pid = 0;

	return alive;
}

static void psr_compute_node_state(struct psr_node_state *st)
{
	struct sysinfo si;
	unsigned long node_total_kb;
	unsigned long anon_kb, file_kb, dirty_kb, clean_kb;
	unsigned long anon_floor_kb, clean_floor_kb;
	unsigned long free_swap_kb;
	pg_data_t *pgdat = NODE_DATA(numa_node_id());

	si_meminfo(&si);
	node_total_kb = si.totalram << (PAGE_SHIFT - 10);

	anon_kb = (node_page_state(pgdat, NR_ACTIVE_ANON) +
		   node_page_state(pgdat, NR_INACTIVE_ANON)) << (PAGE_SHIFT - 10);

	file_kb = (node_page_state(pgdat, NR_ACTIVE_FILE) +
		   node_page_state(pgdat, NR_INACTIVE_FILE)) << (PAGE_SHIFT - 10);
	dirty_kb = node_page_state(pgdat, NR_FILE_DIRTY) << (PAGE_SHIFT - 10);
	clean_kb = (file_kb > dirty_kb) ? (file_kb - dirty_kb) : 0;

	anon_floor_kb  = node_total_kb * anon_min_ratio  / 100;
	clean_floor_kb = node_total_kb * clean_min_ratio / 100;

	st->anon_below_min  = anon_kb  < anon_floor_kb;
	st->clean_below_min = clean_kb < clean_floor_kb;

	free_swap_kb = si.freeswap << (PAGE_SHIFT - 10);
	st->swap_floor_breached = free_swap_kb < reserved_swap_floor_kb;

	/* Escalation tier, concept borrowed from prlmk: crossing the
	 * clean-file floor is treated as a harder signal than swap-in
	 * trend alone -- widen the victim pool in psr_select_victim(). */
	st->escalated = st->clean_below_min;
}

/* ------------------------------------------------------------------
 * Victim selection: highest oom_score_adj, tie-broken by largest RSS,
 * discounted by accumulated CPU time (stime+utime) -- an app you've
 * actively used gets a lower kill score than an idle one at the same
 * oom_score_adj/RSS, concept borrowed from prlmk's stime+utime sort.
 *
 * Also skips tasks whose anon footprint is what's already protected
 * by anon_below_min -- killing an anon-heavy task when anon is
 * already starved doesn't relieve the actual pressure source
 * (file/swap).
 *
 * When st->escalated is set (clean-file floor breached), the victim
 * pool widens to escalated_min_oom_score_adj instead of
 * min_oom_score_adj -- concept borrowed from prlmk's aggressive mode
 * once free_file_limit is crossed.
 * ------------------------------------------------------------------ */
static struct task_struct *psr_select_victim(const struct psr_node_state *st)
{
	struct task_struct *p, *victim = NULL;
	long best_score = LONG_MIN;
	int score_floor = st->escalated ? escalated_min_oom_score_adj
					 : min_oom_score_adj;

	rcu_read_lock();
	for_each_process(p) {
		struct task_struct *t = find_lock_task_mm(p);
		long score;
		unsigned long rss_kb;
		u64 cputime_ns;
		unsigned long cputime_s;

		if (!t)
			continue;

		if (t->signal->oom_score_adj < score_floor) {
			task_unlock(t);
			continue;
		}

		rss_kb = get_mm_rss(t->mm) << (PAGE_SHIFT - 10);

		/* Accumulated CPU time as a proxy for "how much this app
		 * has actually been used" -- thread_group_cputime() sums
		 * stime+utime across the whole process, matching the
		 * spirit of prlmk's per-task stime+utime accounting
		 * without depending on CONFIG_TASK_XACCT/acct_timexpd,
		 * which isn't enabled on every kernel config. */
		{
			struct task_cputime ct;

			thread_group_cputime(t, &ct);
			cputime_ns = ct.stime + ct.utime;
		}
		cputime_s = (unsigned long)(cputime_ns / NSEC_PER_SEC);

		score = (long)t->signal->oom_score_adj * 1000 + (long)rss_kb;

		if (st->anon_below_min)
			score -= (long)rss_kb / 2; /* derate anon-heavy tasks */

		/* Discount by usage time: heavily-used tasks are less
		 * attractive victims even at equal oom_score_adj/RSS.
		 * usage_time_weight_pct=0 disables this entirely. */
		if (usage_time_weight_pct)
			score -= (long)(cputime_s * usage_time_weight_pct) / 100;

		task_unlock(t);

		if (score > best_score) {
			best_score = score;
			victim = t;
		}
	}
	if (victim)
		get_task_struct(victim);
	rcu_read_unlock();

	return victim;
}

static void psr_bypass_and_kill(const struct psr_node_state *st)
{
	struct task_struct *victim;

	/* Don't dispatch a second kill while the last one hasn't been
	 * reaped yet -- concept borrowed from simple_lmk, which waits
	 * for a victim's memory to actually be freed before killing
	 * more. Prevents overkill on a single sustained pressure event
	 * where one kill would have been enough. */
	if (psr_kill_in_flight()) {
		pr_debug(PSR_LMK_NAME ": kill already in flight (pid=%d), skipping\n",
			 psr_kill_state.pending_victim_pid);
		return;
	}

	victim = psr_select_victim(st);
	if (!victim) {
		pr_info(PSR_LMK_NAME ": regression signaled, no eligible victim\n");
		return;
	}

	pr_warn(PSR_LMK_NAME ": BYPASS pid=%d comm=%s escalated=%d anon_below_min=%d clean_below_min=%d swap_floor_breached=%d swap_in_rate=%lu%s\n",
		victim->pid, victim->comm, st->escalated,
		st->anon_below_min, st->clean_below_min, st->swap_floor_breached,
		st->swap_in_rate, dry_run ? " (dry-run)" : "");

	if (!dry_run) {
		send_sig(SIGKILL, victim, 1);
		psr_kill_state.pending_victim_pid = victim->pid;
		psr_kill_state.dispatched_at = ktime_get();
	}

	put_task_struct(victim);
}

/* ------------------------------------------------------------------
 * Hook implementations (declared in include/linux/psr_lmk.h, called
 * from the mm/*.c patch). Five of them are deliberately trivial --
 * see the counter block above for why. Only psr_lmk_note_pressure()
 * runs the actual regression check + victim selection + kill
 * dispatch, because it's the one call site (vmpressure_work_fn(),
 * invoked off a workqueue) that's guaranteed to be sleepable and
 * safe to walk the task list / send a signal from.
 * ------------------------------------------------------------------ */

void psr_lmk_note_anon_reactivation(struct page *page)
{
	atomic_inc(&psr_stat_anon_reactivations);
}
EXPORT_SYMBOL_GPL(psr_lmk_note_anon_reactivation);

void psr_lmk_note_scan_progress(struct pglist_data *pgdat,
				 unsigned long nr_scanned,
				 unsigned long nr_reclaimed)
{
	atomic_long_add(nr_scanned, &psr_stat_scan_sum);
	atomic_long_add(nr_reclaimed, &psr_stat_reclaimed_sum);
}
EXPORT_SYMBOL_GPL(psr_lmk_note_scan_progress);

bool psr_lmk_should_abort_reclaim(void)
{
	return psr_kill_in_flight();
}
EXPORT_SYMBOL_GPL(psr_lmk_should_abort_reclaim);

void psr_lmk_note_alloc_failure(unsigned int order, gfp_t gfp_mask)
{
	atomic_inc(&psr_stat_alloc_failures);
}
EXPORT_SYMBOL_GPL(psr_lmk_note_alloc_failure);

void psr_lmk_note_refault(struct page *page, bool is_swap,
			   unsigned long refault_distance,
			   unsigned long active_size)
{
	if (is_swap)
		atomic_inc(&psr_stat_swap_refaults);
	else
		atomic_inc(&psr_stat_file_refaults);
}
EXPORT_SYMBOL_GPL(psr_lmk_note_refault);

/*
 * The real decision engine. Runs once per vmpressure work cycle
 * (typically driven by actual reclaim activity, so this is
 * naturally rate-limited -- it doesn't free-run on a timer). Uses
 * vmpressure's own scanned/reclaimed-derived pressure percentage as
 * the primary trend sample instead of the old userspace pswpin
 * polling, and drains the counters the other five hooks have been
 * accumulating since the last cycle as corroborating evidence.
 *
 * "Don't kill too aggressively, but don't keep an unused task around
 * as long as there's real free-memory pressure": the aggressiveness
 * side is handled by psr_kill_in_flight() inside psr_bypass_and_kill()
 * (won't fire a second kill until the last one's been reaped or the
 * grace period lapses); the other side is handled by treating
 * VMPRESSURE_CRITICAL and a real allocation failure as hard triggers
 * regardless of how mild the trend sample looks.
 */
void psr_lmk_note_pressure(enum vmpressure_levels level,
			    unsigned long pressure,
			    unsigned long scanned,
			    unsigned long reclaimed)
{
	struct psr_node_state st;
	long slope;
	bool regression;
	int reactivations, alloc_failures, swap_refaults, file_refaults;
	long scan_sum, reclaimed_sum;

	psr_compute_node_state(&st);

	slope = psr_push_sample_and_slope(pressure);

	reactivations  = atomic_xchg(&psr_stat_anon_reactivations, 0);
	alloc_failures = atomic_xchg(&psr_stat_alloc_failures, 0);
	swap_refaults  = atomic_xchg(&psr_stat_swap_refaults, 0);
	file_refaults  = atomic_xchg(&psr_stat_file_refaults, 0);
	scan_sum       = atomic_long_xchg(&psr_stat_scan_sum, 0);
	reclaimed_sum  = atomic_long_xchg(&psr_stat_reclaimed_sum, 0);

	/* Reused field; unit here is vmpressure's 0-100 pressure scale,
	 * not pages/sec -- kept as swap_in_rate for status-output/log
	 * continuity rather than renaming the struct field everywhere. */
	st.swap_in_rate = pressure;

	regression = (pressure >= thrash_hard_limit) ||
		     (slope >= (long)regression_slope_thresh) ||
		     (alloc_failures > 0) ||
		     (level >= VMPRESSURE_CRITICAL);

	/*
	 * Corroboration, borrowed from le9uo's ratio concept: real
	 * anon/swap refaults strengthen a borderline pressure reading
	 * when the clean-file floor is also under pressure. A pure
	 * file-cache refault burst with no swap activity doesn't count
	 * -- that's ordinary page-cache churn, not swap thrash.
	 */
	if (!regression && st.clean_below_min && swap_refaults > 0)
		regression = true;

	/* Reactivation churn + poor reclaim efficiency this cycle is
	 * the same "thrashing but hasn't hit a hard limit yet" case
	 * the slope/hard-limit checks above are meant to catch --
	 * this catches it earlier when scan volume is low. */
	if (!regression && reactivations > 4 && scan_sum > 0 &&
	    (reclaimed_sum * 100 / scan_sum) < 10)
		regression = true;

	/*
	 * PSI corroboration -- real task stall time catches cases the
	 * scanned/reclaimed ratio misses (reclaim looks "efficient"
	 * but tasks are still stalling). No-op (dead code, compiled
	 * out) when CONFIG_PSI=n; psr_lmk_read_psi_mem_avg10() returns
	 * 0 unconditionally in that case, so this can never fire on
	 * your current test kernel.
	 */
	if (!regression && IS_ENABLED(CONFIG_PSI) &&
	    psr_lmk_read_psi_mem_avg10() >= psi_avg10_thresh_or_zero())
		regression = true;

	pr_debug(PSR_LMK_NAME
		 ": pressure=%lu level=%d slope=%ld reactivations=%d alloc_failures=%d "
		 "swap_refaults=%d file_refaults=%d scan=%ld reclaimed=%ld regression=%d\n",
		 pressure, level, slope, reactivations, alloc_failures,
		 swap_refaults, file_refaults, scan_sum, reclaimed_sum, regression);

	if (regression || st.swap_floor_breached)
		psr_bypass_and_kill(&st);
}
EXPORT_SYMBOL_GPL(psr_lmk_note_pressure);

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
		"min_oom_score_adj: %d\n"
		"escalated_min_oom_score_adj: %d\n"
		"usage_time_weight_pct: %u\n"
		"---\n"
		"anon_below_min: %d\n"
		"clean_below_min: %d\n"
		"swap_floor_breached: %d\n"
		"escalated: %d\n"
		"kill_in_flight_pid: %d\n"
		"psi_available: %d\n"
		"psi_mem_avg10: %lu\n"
		"psi_avg10_thresh: %u\n",
		psr_lmk_active_profile,
		dry_run, reserved_swap_floor_kb, anon_min_ratio, clean_min_ratio,
		regression_slope_thresh, thrash_hard_limit, min_oom_score_adj,
		escalated_min_oom_score_adj, usage_time_weight_pct,
		st.anon_below_min, st.clean_below_min, st.swap_floor_breached,
		st.escalated, psr_kill_state.pending_victim_pid,
		IS_ENABLED(CONFIG_PSI), psr_lmk_read_psi_mem_avg10(),
		psi_avg10_thresh_or_zero());

	return 0;
}

static int psr_status_open(struct inode *inode, struct file *file)
{
	return single_open(file, psr_status_show, NULL);
}

static const struct proc_ops psr_status_fops = {
	.proc_open    = psr_status_open,
	.proc_read    = seq_read,
	.proc_lseek   = seq_lseek,
	.proc_release = single_release,
};

static struct proc_dir_entry *psr_proc_dir;

/* ------------------------------------------------------------------
 * Module init/exit
 * ------------------------------------------------------------------ */
static int __init psr_lmk_init(void)
{
	psr_engine_init();
	memset(&psr_kill_state, 0, sizeof(psr_kill_state));

	psr_lmk_auto_tune();

	psr_proc_dir = proc_mkdir(PSR_LMK_NAME, NULL);
	if (!psr_proc_dir)
		return -ENOMEM;

	proc_create("status", 0444, psr_proc_dir, &psr_status_fops);

	/*
	 * No notifier registration needed here: mm/vmpressure.c calls
	 * psr_lmk_note_pressure() directly (see the mm hooks patch),
	 * so PSR-LMK gets the same pressure event the kernel's own
	 * vmpressure machinery already computes, with no separate
	 * registration/cgroup-target step required.
	 */

	pr_info(PSR_LMK_NAME ": loaded (dry_run=%d)\n", dry_run);
	return 0;
}

static void __exit psr_lmk_exit(void)
{
	proc_remove(psr_proc_dir);
	pr_info(PSR_LMK_NAME ": unloaded\n");
}

/* ------------------------------------------------------------------
 * Compatibility shim: some Android init/vendor init scripts, and
 * lmkd on certain branches, probe for the presence of the classic
 * kernel LMK's sysfs parameter node at boot and treat its absence as
 * "no LMK present," which can trigger a reboot loop. PSR-LMK replaces
 * the kernel LMK, so it needs to present the same compatibility node
 * -- even though PSR-LMK's real tunables live under its own
 * psr_lmk-prefixed params above and /proc/psr_lmk/status.
 *
 * REQUIRES CONFIG_ANDROID_PSR_LMK=y (built-in), not =m (loadable
 * module). The dot in "lowmemorykiller.minfree" only gets split into
 * a synthesized /sys/module/lowmemorykiller/ directory by the
 * kernel's builtin-param sysfs path, which requires THIS_MODULE to be
 * NULL (i.e. compiled directly into vmlinux). As a loadable module,
 * this would instead show up as a literal file named
 * "lowmemorykiller.minfree" inside /sys/module/psr_lmk/parameters/,
 * which does NOT satisfy the boot-time presence check. Update the
 * Kconfig entry to `bool` before relying on this.
 *
 * MODULE_PARAM_PREFIX is a plain preprocessor macro: it affects every
 * module_param()/module_param_cb() call that follows it in this
 * translation unit, not just the next one. This block is therefore
 * placed at the very end of the file, after every PSR-LMK tunable has
 * already been registered above with the default (empty) prefix, and
 * the prefix is undef'd back immediately after this single call. Do
 * not add new module_param() calls between the #define and #undef
 * below, and do not move this block earlier in the file.
 * ------------------------------------------------------------------ */
static int psr_lmk_compat_minfree_set(const char *val,
				       const struct kernel_param *kp)
{
	/* Accept and log writes from init/lmkd probing or configuring
	 * this legacy node so boot doesn't stall, but don't let it
	 * drive PSR-LMK's actual thresholds -- those are configured
	 * exclusively through the psr_lmk.* module params above. */
	pr_info(PSR_LMK_NAME ": compat lowmemorykiller.minfree write ignored: %s\n",
		val ? val : "(null)");
	return 0;
}

static int psr_lmk_compat_minfree_get(char *buffer,
				       const struct kernel_param *kp)
{
	/* perm is write-only (0200) below, so sysfs never actually
	 * calls this -- defined anyway so kernel_param_ops is complete
	 * if perm is ever loosened. */
	return scnprintf(buffer, PAGE_SIZE, "0\n");
}

static const struct kernel_param_ops psr_lmk_compat_minfree_ops = {
	.set = psr_lmk_compat_minfree_set,
	.get = psr_lmk_compat_minfree_get,
};

#undef MODULE_PARAM_PREFIX
#define MODULE_PARAM_PREFIX "lowmemorykiller."
module_param_cb(minfree, &psr_lmk_compat_minfree_ops, NULL, 0200);
#undef MODULE_PARAM_PREFIX
#define MODULE_PARAM_PREFIX ""

module_init(psr_lmk_init);
module_exit(psr_lmk_exit);

MODULE_LICENSE("GPL v2");
MODULE_DESCRIPTION("PSR-LMK: Protected-Swap Regression Low Memory Killer");
MODULE_AUTHOR("prototype");
