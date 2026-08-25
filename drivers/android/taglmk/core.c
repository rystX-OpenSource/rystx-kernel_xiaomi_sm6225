// SPDX-License-Identifier: GPL-2.0
/*
 * TAGLMK - Task-aware Android Guided Low Memory Killer
 *
 * The guided core.  Everything that decides *when* to act lives here; what to
 * act on is task.c, how much to act is predict.c and zram.c, and the actual
 * page work is the reclaim driver in fs/proc/task_mmu.c.
 *
 * The shape of a pass follows the contract the ladder in taglmk.h describes:
 *
 *   above both limits   reclaim only, pushing cold anon into ZRAM so that the
 *                       file cache stays large and nothing has to die
 *   swap exhausted      collect, sort by accumulated cputime, and kill from the
 *                       bottom until swap recovers
 *   cache collapsed     the same, in bigger steps and with a lower bar
 *
 * Killing is the last resort and is metered accordingly: the situation is
 * re-examined after every single victim, so a pass gives up the moment it has
 * done enough rather than working through a batch it no longer needs.
 *
 * Copyright (C) 2026 iDeadXS <datarafi43@gmail.com>
 */

#define pr_fmt(fmt) "taglmk: " fmt

#include <linux/atomic.h>
#include <linux/delay.h>
#include <linux/freezer.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/mm.h>
#include <linux/moduleparam.h>
#include <linux/mutex.h>
#include <linux/notifier.h>
#include <linux/oom.h>
#include <linux/sched/signal.h>
#include <linux/signal.h>
#include <linux/slab.h>
#include <linux/swap.h>
#include <linux/vmpressure.h>
#include <linux/vmstat.h>
#include <linux/workqueue.h>
#include <linux/zram_ir.h>

#include "taglmk.h"

/* Pages to kilobytes, for log lines only. */
#define TAGLMK_K(x)	((unsigned long)(x) << (PAGE_SHIFT - 10))

/*
 * How long to let a kill settle before looking at the counters again.  A victim
 * does not give its pages back inside its own SIGKILL; either it runs its exit
 * path or the oom reaper unmaps it, and both take a moment to show up in the
 * statistics.  Re-reading them immediately would always say "still bad" and
 * would spend the whole batch every time.
 */
#define TAGLMK_KILL_SETTLE_MS	20

/*
 * Tuning presets
 * ==============
 *
 * The first entry the machine fits under wins, so the table has to stay sorted
 * smallest first and the last entry has to be unbounded.
 *
 * The target is a device with four gigabytes or less, which is why the small
 * entries are the careful ones.  Below that line the file cache is small enough
 * that losing it is felt immediately, so the limits sit proportionally higher,
 * the batches are larger because there is less time to converge, and burst_gain
 * is turned up so the predictor commits earlier - that gain is the per RAM class
 * compensation applied to the same accelerated calculation on every device.
 */
static const struct taglmk_profile taglmk_profiles[] = {
	{
		.name			= "2G",
		.ram_pages		= TAGLMK_MB_PAGES(2048),
		.free_swap_limit	= TAGLMK_MB_PAGES(24),
		.free_file_limit	= TAGLMK_MB_PAGES(40),
		.reclaim_budget		= 512,
		.scan_limit		= 64,
		.kill_batch		= 2,
		.kill_batch_crit	= 4,
		.burst_gain		= TAGLMK_Q44_ONE * 3 / 2,
		.swap_target_pct	= 92,
	},
	{
		.name			= "3G",
		.ram_pages		= TAGLMK_MB_PAGES(3072),
		.free_swap_limit	= TAGLMK_MB_PAGES(32),
		.free_file_limit	= TAGLMK_MB_PAGES(56),
		.reclaim_budget		= 768,
		.scan_limit		= 96,
		.kill_batch		= 2,
		.kill_batch_crit	= 3,
		.burst_gain		= TAGLMK_Q44_ONE * 5 / 4,
		.swap_target_pct	= 90,
	},
	{
		.name			= "4G",
		.ram_pages		= TAGLMK_MB_PAGES(4096),
		.free_swap_limit	= TAGLMK_MB_PAGES(48),
		.free_file_limit	= TAGLMK_MB_PAGES(72),
		.reclaim_budget		= 1024,
		.scan_limit		= 128,
		.kill_batch		= 1,
		.kill_batch_crit	= 3,
		.burst_gain		= TAGLMK_Q44_ONE,
		.swap_target_pct	= 88,
	},
	{
		/* Anything larger.  Must stay last and stay unbounded. */
		.name			= "large",
		.ram_pages		= ULONG_MAX,
		.free_swap_limit	= TAGLMK_MB_PAGES(64),
		.free_file_limit	= TAGLMK_MB_PAGES(96),
		.reclaim_budget		= 1024,
		.scan_limit		= 128,
		.kill_batch		= 1,
		.kill_batch_crit	= 2,
		.burst_gain		= TAGLMK_Q44_ONE * 3 / 4,
		.swap_target_pct	= 85,
	},
};

struct taglmk_state taglmk = {
	.pressure_min	= 90,
	.min_adj	= 200,
	.min_adj_crit	= 50,
	.enabled	= true,

	/*
	 * The recompression rung, on by default but deliberately timid: a
	 * megabyte of slots at most, and no oftener than every two seconds even
	 * under sustained pressure.  Both are visible in sysfs precisely because
	 * whether this rung pays for itself is a question about a device, not
	 * one this driver can answer in a header.
	 */
	.ir_interval_ms	= 2000,
	.ir_max_pages	= 256,
};

/*
 * Reading the memory situation
 * ============================
 *
 * Every input comes from the same node counters vmscan itself works from, so
 * the driver can never form an opinion the LRU would disagree with.
 */
unsigned long taglmk_free_swap_pages(void)
{
	long free = get_nr_swap_pages();

	return free > 0 ? free : 0;
}

unsigned long taglmk_active_file_pages(void)
{
	return global_node_page_state(NR_ACTIVE_FILE);
}

unsigned long taglmk_inactive_file_pages(void)
{
	return global_node_page_state(NR_INACTIVE_FILE);
}

enum taglmk_level taglmk_mem_level(void)
{
	/*
	 * The file arm is tested first because it describes the worse of the
	 * two situations.  Running out of swap still leaves a cache to fall
	 * back on; a cache this small means every fault goes to storage, which
	 * is what actually makes a device feel broken.
	 */
	if (taglmk_active_file_pages() < taglmk.free_file_limit)
		return TAGLMK_LEVEL_CRITICAL;

	/*
	 * With no swap area configured get_nr_swap_pages() is always zero, so
	 * this arm would read as permanently out of swap.  Skip it instead of
	 * killing continuously on a device that simply has no ZRAM.
	 */
	if (total_swap_pages &&
	    taglmk_free_swap_pages() <= taglmk.free_swap_limit)
		return TAGLMK_LEVEL_LOW;

	return TAGLMK_LEVEL_NONE;
}

/*
 * The oom_score_adj bar for a given level.  Android hands out roughly zero for
 * whatever is on screen, fifty for what the user can still perceive, two
 * hundred for services and seven hundred and up for cached applications, so the
 * default pair of bars means services and cached applications first, and only
 * once the cache has collapsed does anything the user might notice come into
 * range.
 */
short taglmk_min_adj(enum taglmk_level level)
{
	return level == TAGLMK_LEVEL_CRITICAL ? taglmk.min_adj_crit
					      : taglmk.min_adj;
}

/*
 * The classification bar for a given level.  A task is a candidate only while
 * its type sorts strictly below the value returned here, and since
 * TAGLMK_TYPE_CRITICAL is the largest type it is never itself a candidate: no
 * level, however bad, will offer up one of Android's own cores.
 *
 * At TAGLMK_LEVEL_LOW the bar sits at TAGLMK_TYPE_PINNED, so only plain
 * applications are on the table and a pinned package survives.  Raising it at
 * TAGLMK_LEVEL_CRITICAL is what gives pinning its meaning: longer
 * survivability, not immortality.
 */
enum taglmk_task_type taglmk_type_cutoff(enum taglmk_level level)
{
	return level == TAGLMK_LEVEL_LOW ? TAGLMK_TYPE_PINNED
					 : TAGLMK_TYPE_CRITICAL;
}

/*
 * Reclaim pass
 * ============
 */
static void taglmk_share_budget(unsigned int nr, unsigned int budget,
				enum taglmk_reclaim_type type)
{
	unsigned int i;

	if (type == TAGLMK_RECLAIM_ANON) {
		/*
		 * Anon is what ends up in ZRAM, so the balancer owns the split.
		 * It weights each task by what it actually has resident, which
		 * is the only thing that can be handed to the compressor.
		 */
		taglmk_zram_share(taglmk.victims, nr, budget);
		return;
	}

	/* A file pass has no such asymmetry.  Spread it evenly. */
	for (i = 0; i < nr; i++)
		taglmk.victims[i].budget = DIV_ROUND_UP(budget, nr);
}

static void taglmk_reclaim_pass(void)
{
	enum taglmk_reclaim_type type = TAGLMK_RECLAIM_ANON;
	unsigned long asked = 0, got = 0;
	unsigned int budget, i, nr;
	u64 cputime_avg = 0;

	/*
	 * Two corrections, in order.  The predictor says how urgent the coming
	 * window looks, and the balancer says how much more ZRAM is worth
	 * filling; a zero from the balancer means swap already holds as much as
	 * the profile wants it to.
	 */
	budget = taglmk_predict_budget(taglmk.reclaim_budget);
	budget = total_swap_pages ? taglmk_zram_budget(budget) : 0;

	if (!budget) {
		/*
		 * Swap is as full as the profile wants it, so there is no trade
		 * left to make in anonymous memory.  Ask zram to compress what
		 * it is already holding harder instead: that returns memory
		 * without a page leaving anybody's working set, which nothing
		 * else this driver does can claim.
		 *
		 * Purely additive, and deliberately so.  The file pass below
		 * still runs, because what it moves is the active file count
		 * that vmscan and lmkd read, and no amount of recompression
		 * moves that.  The two rungs answer different questions and
		 * neither stands in for the other.
		 */
		taglmk_ir_sweep();

		/*
		 * Nothing to gain from pushing more anon out, so spend the pass
		 * ageing file pages instead.  The driver only deactivates them,
		 * never drops them, so the active file count that both vmscan
		 * and lmkd read moves in the one direction that is useful
		 * without throwing away anything that is still wanted.
		 */
		type = TAGLMK_RECLAIM_FILE;
		budget = taglmk_predict_budget(taglmk.reclaim_budget);
	}

	nr = taglmk_scan_tasks(TAGLMK_LEVEL_NONE);
	if (!nr)
		goto out;

	/* Largest resident set first: the fewest walks for the most pages. */
	taglmk_sort_by_anon();
	taglmk_share_budget(nr, budget, type);

	/*
	 * Reference point for "has this task been busy", used only to pick a
	 * compression depth below.  A mean over the scanned set is enough: the
	 * question is which of these tasks look idle relative to each other,
	 * not what their absolute runtimes are.
	 */
	if (type == TAGLMK_RECLAIM_ANON) {
		u64 sum = 0;

		for (i = 0; i < nr; i++)
			sum += taglmk.victims[i].cputime;

		cputime_avg = div_u64(sum, nr);
	}

	for (i = 0; i < nr && got < budget; i++) {
		struct taglmk_victim *v = &taglmk.victims[i];
		struct taglmk_reclaim_stat stat;
		int ret;

		if (v->skip || !v->budget)
			continue;

		/*
		 * zram is bio based, so the store path runs synchronously in
		 * this worker's context: a hint set on current here reaches
		 * zram_write_page() for every page this walk swaps out.  Only
		 * an anon pass produces stores, and the hint is dropped again
		 * immediately so nothing else this worker does inherits it.
		 */
		if (type == TAGLMK_RECLAIM_ANON)
			zram_ir_set_depth(taglmk_ir_depth(v, cputime_avg));

		/*
		 * A failure here is ordinary: -EBUSY means the address space was
		 * locked by someone else and -ESRCH that the task exited under
		 * us.  Both mean "try the next one", never "give up".
		 */
		ret = taglmk_reclaim_mm(v->tsk, type, v->budget, &stat);

		zram_ir_reset_depth();

		if (ret)
			continue;

		asked += v->budget;
		got += stat.nr_reclaimed + stat.nr_deactivated;
	}

	/*
	 * Only an anon pass teaches the balancer anything.  Deactivating a file
	 * page produces no swap, so feeding it to a regression that models swap
	 * production would only add noise.
	 */
	if (type == TAGLMK_RECLAIM_ANON)
		taglmk_zram_observe(asked, got);

	atomic_long_add(got, &taglmk.nr_reclaimed);
out:
	taglmk_release_victims();
}

/*
 * Kill pass
 * =========
 */
static bool taglmk_kill(const struct taglmk_victim *v)
{
	struct task_struct *tsk = v->tsk;
	struct task_struct *t;

	pr_info("%s %s (pid %d) adj %d type %u anon %luK swap %luK cputime %llums\n",
		taglmk.dry_run ? "would kill" : "killing", tsk->comm,
		task_pid_nr(tsk), v->adj, (unsigned int)v->type,
		TAGLMK_K(v->anon_pages), TAGLMK_K(v->swap_pages),
		div_u64(v->cputime, NSEC_PER_MSEC));

	/*
	 * A dry run has freed nothing, so it must not be counted as a kill
	 * either; the caller uses the return value to decide whether the
	 * situation is worth re-examining.
	 */
	if (taglmk.dry_run)
		return false;

	/*
	 * Signal first.  add_to_oom_reaper() below only takes a task over once
	 * task_will_free_mem() agrees the address space is going away, and that
	 * is not true until SIGKILL is already pending.
	 */
	do_send_sig_info(SIGKILL, SEND_SIG_FORCED, tsk, PIDTYPE_TGID);

	/*
	 * A cached application is usually frozen, and a frozen task never looks
	 * at its pending signals, so nudge every thread awake.
	 *
	 * Deliberately without TIF_MEMDIE.  Setting that flag by hand looks
	 * like the obvious way to stop the freezer taking the task straight
	 * back - freezing_slow_path() does test it - but exit_mm() calls
	 * exit_oom_victim() for any thread carrying it, which would decrement an
	 * oom_victims count that nothing here ever incremented.  Once that
	 * count has gone negative it can never reach zero again and
	 * oom_killer_disable() waits on it forever, taking suspend with it.
	 * Only mark_oom_victim() may set the flag, and it is private to
	 * mm/oom_kill.c.
	 *
	 * So a task held down by the cgroup freezer does go back to sleep, and
	 * that is acceptable: add_to_oom_reaper() hands the address space to the
	 * reaper thread, which unmaps it whether or not the victim ever runs
	 * again.  The memory comes back either way.
	 */
	rcu_read_lock();
	for_each_thread(tsk, t)
		__thaw_task(t);
	rcu_read_unlock();

	add_to_oom_reaper(tsk);

	atomic_long_inc(&taglmk.nr_killed);

	return true;
}

static void taglmk_kill_pass(enum taglmk_level level)
{
	unsigned int batch, killed = 0, i, nr;

	atomic_long_inc(&taglmk.nr_kill_passes);

	nr = taglmk_scan_tasks(level);
	if (!nr) {
		atomic_long_inc(&taglmk.nr_no_candidate);
		pr_warn_ratelimited("no candidate left at level %u, %luK free swap, %luK active file\n",
				    (unsigned int)level,
				    TAGLMK_K(taglmk_free_swap_pages()),
				    TAGLMK_K(taglmk_active_file_pages()));
		goto out;
	}

	/*
	 * Lowest accumulated cputime first.  This is the whole point of the
	 * driver.  The time a user spends in an application accrues here as
	 * utime plus stime summed across its thread group, so the one they live
	 * in sits a long way behind the one they opened once and forgot about,
	 * and the cheap end of the list is where killing starts.
	 */
	taglmk_sort_by_cputime();

	batch = level == TAGLMK_LEVEL_CRITICAL ? taglmk.kill_batch_crit
					       : taglmk.kill_batch;

	for (i = 0; i < nr && killed < batch; i++) {
		if (taglmk.victims[i].skip)
			continue;

		if (!taglmk_kill(&taglmk.victims[i]))
			continue;

		killed++;

		/*
		 * Let the teardown reach the counters, then ask again.  Giving
		 * up part way through a batch, the instant the situation is no
		 * longer bad, is what keeps the number of applications lost to
		 * a single pass as small as it can be.
		 */
		msleep_interruptible(TAGLMK_KILL_SETTLE_MS);
		if (taglmk_mem_level() == TAGLMK_LEVEL_NONE)
			break;
	}
out:
	taglmk_release_victims();
}

/*
 * Pass dispatch
 * =============
 */
static void taglmk_do_pass(struct work_struct *work)
{
	enum taglmk_level level;

	/*
	 * A work item is never re-entered, so this can only be contended by a
	 * second entry point.  There is none today; the lock is what makes the
	 * single owner rule for the victim array true rather than incidental,
	 * and trylock keeps a future caller from ever waiting on a pass that is
	 * already doing the work it wanted.
	 */
	if (!mutex_trylock(&taglmk.lock))
		return;

	if (!taglmk.enabled)
		goto unlock;

	atomic_long_inc(&taglmk.nr_passes);

	/*
	 * Sample before deciding.  The predictor wants a reading from every
	 * pass, including the ones that go on to kill, or its window would only
	 * ever describe the calm.
	 */
	taglmk_predict_sample();

	level = taglmk_mem_level();
	if (level == TAGLMK_LEVEL_NONE)
		taglmk_reclaim_pass();
	else
		taglmk_kill_pass(level);

unlock:
	mutex_unlock(&taglmk.lock);
}

static int taglmk_vmpressure_cb(struct notifier_block *nb,
				unsigned long pressure, void *data)
{
	if (!taglmk.enabled || pressure < taglmk.pressure_min)
		return NOTIFY_DONE;

	/*
	 * This runs on the reclaim path of whoever is currently stalling, and
	 * the global chain is only rung at all while allocations are genuinely
	 * stuck in the slow path, so the filtering that matters has already
	 * happened.  Do as little as possible: hand the pass to a high priority
	 * worker and get out.  A second event while one is already queued adds
	 * nothing, because the pass reads the situation when it runs and not
	 * when it was asked for.
	 */
	if (!work_pending(&taglmk.work))
		queue_work(system_highpri_wq, &taglmk.work);

	return NOTIFY_OK;
}

static struct notifier_block taglmk_vmpressure_nb = {
	.notifier_call	= taglmk_vmpressure_cb,
	.priority	= INT_MAX,
};

/*
 * Bring-up
 * ========
 */
static const struct taglmk_profile *taglmk_pick_profile(void)
{
	unsigned int i;

	for (i = 0; i < ARRAY_SIZE(taglmk_profiles); i++)
		if (totalram_pages <= taglmk_profiles[i].ram_pages)
			return &taglmk_profiles[i];

	/* Unreachable: the last entry is unbounded.  Be explicit anyway. */
	return &taglmk_profiles[ARRAY_SIZE(taglmk_profiles) - 1];
}

static void taglmk_apply_profile(const struct taglmk_profile *p)
{
	taglmk.profile		= p;
	taglmk.free_swap_limit	= p->free_swap_limit;
	taglmk.free_file_limit	= p->free_file_limit;
	taglmk.reclaim_budget	= p->reclaim_budget;
	taglmk.kill_batch	= p->kill_batch;
	taglmk.kill_batch_crit	= p->kill_batch_crit;
	taglmk.scan_limit	= min_t(unsigned int, p->scan_limit,
					TAGLMK_MAX_VICTIMS);
}

static int __init taglmk_init(void)
{
	int ret;

	mutex_init(&taglmk.lock);
	INIT_WORK(&taglmk.work, taglmk_do_pass);

	taglmk_apply_profile(taglmk_pick_profile());

	/*
	 * One allocation for the lifetime of the machine.  A pass must never
	 * allocate: it runs because memory is short, and the array is the only
	 * thing it would have needed memory for.
	 *
	 * Sized at the maximum rather than at the profile's scan_limit, even
	 * though a small device will never fill it.  It costs a few kilobytes
	 * once and it means the array cannot be outgrown by a later write to
	 * scan_limit, so sysfs only has to bound that tunable by a constant
	 * instead of by the history of what happened to be allocated.
	 */
	taglmk.victims = kcalloc(TAGLMK_MAX_VICTIMS, sizeof(*taglmk.victims),
				 GFP_KERNEL);
	if (!taglmk.victims)
		return -ENOMEM;

	/*
	 * Before sysfs and before the notifier, so that the first pass anyone
	 * can trigger already has every resource a pass is allowed to use.
	 */
	ret = taglmk_zram_init();
	if (ret)
		goto free_victims;

	ret = taglmk_task_init();
	if (ret)
		goto zram_exit;

	ret = taglmk_sysfs_init();
	if (ret)
		goto task_exit;

	/* Last, so nothing can queue a pass into a half built state. */
	ret = vmpressure_notifier_register(&taglmk_vmpressure_nb);
	if (ret)
		goto sysfs_exit;

	pr_info("%s profile on %luMB: swap limit %luK, file limit %luK, %s\n",
		taglmk.profile->name, totalram_pages >> (20 - PAGE_SHIFT),
		TAGLMK_K(taglmk.free_swap_limit),
		TAGLMK_K(taglmk.free_file_limit),
		IS_ENABLED(CONFIG_ANDROID_TAGLMK_ARM64_NEON) ?
			"NEON accelerated" : "scalar");

	return 0;

sysfs_exit:
	taglmk_sysfs_exit();
task_exit:
	taglmk_task_exit();
zram_exit:
	taglmk_zram_exit();
free_victims:
	kfree(taglmk.victims);
	taglmk.victims = NULL;
	pr_err("failed to initialise: %d\n", ret);

	return ret;
}

/*
 * Late, because a pass needs slab, workqueues and the oom reaper, and because
 * the RAM size the profile is chosen from is only final once the whole of
 * memory has been handed over.  There is nothing to kill this early in any
 * case, and the ladder would not trip if there were.
 */
late_initcall(taglmk_init);

/*
 * The lmkd contract
 * =================
 *
 * Android's lmkd writes lowmemorykiller.minfree during startup and treats a
 * failed write as "this kernel has no low memory killer", which ends in a
 * reboot.  So the parameter has to exist and the write has to succeed.
 *
 * The value is deliberately ignored.  lmkd's minfree table is a list of free
 * page watermarks paired with oom_score_adj bands, which is a different model
 * from the one here: TAGLMK watches free swap and the active file LRU, and gets
 * its numbers from the RAM profile.  Honouring both would mean two thresholds
 * disagreeing about the same device.  The write is still worth a line in the
 * log the first time, because it is the moment userspace confirms it expects
 * the kernel to be doing this work.
 */
static int taglmk_minfree_set(const char *val, const struct kernel_param *kp)
{
	static atomic_t handshake = ATOMIC_INIT(0);

	if (!atomic_cmpxchg(&handshake, 0, 1))
		pr_info("lmkd is up; in-kernel killing is ours\n");

	return 0;
}

static const struct kernel_param_ops taglmk_minfree_ops = {
	.set = taglmk_minfree_set,
};

#undef MODULE_PARAM_PREFIX
#define MODULE_PARAM_PREFIX "lowmemorykiller."
module_param_cb(minfree, &taglmk_minfree_ops, NULL, 0200);
