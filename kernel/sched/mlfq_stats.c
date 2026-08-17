// SPDX-License-Identifier: GPL-2.0
/*
 * The multi-level feedback queue dashboard, /proc/sched_eevdf_mlfq_stats.
 *
 * Ported from scx_mlfq, a sched_ext scheduler by galpt:
 *   https://github.com/galpt/scx_mlfq
 *
 * scx_mlfq's userspace daemon serves a dashboard over loopback, port 50005:
 * a per-CPU card grid, a plain-language Summary built from the live gauges,
 * and the system counters. Those three sections are what this file
 * reproduces, in that order, and it takes its labels and its wording from
 * scx/ui/index.html so that the same number is called the same thing in both
 * places.
 *
 * Only the transport differs, because there is nothing here for the daemon to
 * be: it existed to poll BPF maps over a file descriptor and republish them,
 * and the state it was polling is this kernel's own per-CPU variables. What is
 * left of it is one seq_file.
 *
 * A single seq_file is also what makes a read of this coherent. Everything
 * here is derived from state that keeps moving while it is read, so the file
 * is generated in one pass into one buffer and handed over whole; a reader
 * cannot see the first half of one sample and the second half of the next.
 * That is why this is not a sysctl under /proc/sys: a sysctl handler is handed
 * a position within a buffer it is expected to have laid out already, so a
 * read large enough to be split would stitch together two samples. It is also
 * not under CONFIG_SCHED_DEBUG, because none of this is debugging output: it
 * is the scheduler reporting what it is doing, which is as useful on a shipped
 * kernel as on a development one. The one thing all of it does depend on is
 * procfs, so that is what the Makefile gates the object on.
 *
 * Two things are printed differently from the way the browser printed them.
 * Counters are exact rather than abbreviated -- the dashboard wrote 1.2M
 * because a card is narrow, and nothing here is narrow. Durations are printed
 * as nanoseconds with the dashboard's own rounded form beside them, because
 * for those the rounded form is the information and the exact figure is the
 * one that is awkward to read.
 *
 * Nothing here takes a lock. The counters are per-CPU and are summed across
 * CPUs while the CPUs that own them carry on counting, so a total can include
 * a CPU's increment from after another CPU's was read: a total is therefore
 * consistent with some interleaving of the events, and not with a single
 * instant. The same holds for the runnable occupancies, which are read from
 * runqueues that are being enqueued and dequeued, and for a CPU's running
 * task, which may have been switched out before the row reaches the buffer.
 * This is a dashboard, and its numbers are the numbers a dashboard can give;
 * anything that needs a coherent instant needs a tracepoint, not a file.
 */
#include <linux/cpufreq.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/timekeeping.h>
#include <linux/topology.h>

#include "sched.h"
#include "mlfq.h"

/* Label column width, so every value in every section lines up. */
#define MLFQ_LABEL_WIDTH		"-24s"

/* The dashboard's Q chips, indexed by level, so slot 0 is an idle CPU. */
static const char * const mlfq_queue_name[MLFQ_NR_QUEUES + 1] = {
	"idle", "Q1", "Q2", "Q3",
};

/**
 * struct mlfq_totals - the per-CPU state summed for the System section
 * @stat:	the event counters, summed over every possible CPU.
 * @q_runnable:	tasks queued at each level, summed over every runqueue.
 *		Indexed by level, so slot 0 is unused.
 * @wake_total:	wakeup enqueues since boot, summed over every possible CPU.
 */
struct mlfq_totals {
	u64	stat[MLFQ_NR_STATS];
	u32	q_runnable[MLFQ_NR_QUEUES + 1];
	u64	wake_total;
};

/*
 * Summed over the possible CPUs rather than the online ones, so that a CPU
 * that has been offlined does not take its share of the counters with it.
 * mlfq_adapt_step() folds its windows over the same set for the same reason.
 */
static void mlfq_read_totals(struct mlfq_totals *t)
{
	int cpu, i;

	memset(t, 0, sizeof(*t));

	for_each_possible_cpu(cpu) {
		struct mlfq_pcpu *pc = &per_cpu(mlfq_pcpu, cpu);

		for (i = 0; i < MLFQ_NR_STATS; i++)
			t->stat[i] += READ_ONCE(pc->stat[i]);

		for (i = 1; i <= MLFQ_NR_QUEUES; i++)
			t->q_runnable[i] += READ_ONCE(pc->q_runnable[i]);

		t->wake_total += READ_ONCE(pc->wake_total);
	}
}

/* Tasks queued on one runqueue, at any level. */
static u32 mlfq_cpu_runnable(int cpu)
{
	struct mlfq_pcpu *pc = &per_cpu(mlfq_pcpu, cpu);
	u32 total = 0;
	int q;

	for (q = 1; q <= MLFQ_NR_QUEUES; q++)
		total += READ_ONCE(pc->q_runnable[q]);

	return total;
}

/*
 * The cache domain a CPU belongs to, named by the first CPU in the domain,
 * which is how the scheduler itself names it. scx_mlfq numbers its domains
 * from zero instead, so the same domain can carry a different label there.
 */
static int mlfq_cpu_llc(int cpu)
{
#ifdef CONFIG_SMP
	return per_cpu(sd_llc_id, cpu);
#else
	return 0;
#endif
}

/*
 * True for the sibling threads of an SMT core and false for the first thread
 * of each, matching the card grid's SMT badge: it marks the threads that share
 * a core with one that came before them, so a glance down the column says how
 * many of the CPUs listed are whole cores.
 */
static bool mlfq_cpu_smt(int cpu)
{
	return (int)cpumask_first(topology_sibling_cpumask(cpu)) != cpu;
}

/*
 * The level of the fair task a CPU is running, 0 for an idle CPU, and -1 for
 * one running a task of a higher class, which has no level to report.
 *
 * A remote runqueue's curr is RCU-protected, which is what makes it safe to
 * look at without that runqueue's lock. It can be switched out immediately
 * afterwards; see the file comment on what this file's numbers are worth.
 *
 * A fair task always carries a level, since __sched_fork() puts every task in
 * the default queue before it can be picked, so the range check below is only
 * there to keep a bad level out of the name table. If one ever did get through,
 * reporting the CPU as idle is the least it could be made to claim.
 */
static int mlfq_cpu_level(int cpu)
{
	struct task_struct *curr;
	int level = 0;

	rcu_read_lock();
	curr = cpu_curr(cpu);
	if (curr && !is_idle_task(curr)) {
		if (curr->sched_class == &fair_sched_class)
			level = curr->mlfq.queue <= MLFQ_NR_QUEUES ?
				curr->mlfq.queue : 0;
		else
			level = -1;
	}
	rcu_read_unlock();

	return level;
}

/*
 * CPUs currently running a task the classifier has a level for, which is what
 * the dashboard's On CPU is. Walked separately from the per-CPU rows and so
 * from a slightly later instant than they were: the two can disagree by a
 * switch, and neither is more nearly true than the other.
 */
static u32 mlfq_on_cpu_now(void)
{
	u32 count = 0;
	int cpu;

	for_each_online_cpu(cpu) {
		if (mlfq_cpu_level(cpu) > 0)
			count++;
	}

	return count;
}

/*
 * A duration in the dashboard's own units. fmtRuntime() in ui/index.html
 * picks the largest unit that leaves a number below that unit's own base and
 * prints one decimal, so the tenths are computed directly in the unit chosen
 * rather than by dividing down and losing them.
 */
static void mlfq_seq_duration(struct seq_file *m, u64 ns)
{
	const char *unit;
	u64 tenths;
	u32 tenth;

	if (ns < (u64)NSEC_PER_SEC * 60) {
		tenths = div_u64(ns, NSEC_PER_SEC / 10);
		unit = "sec";
	} else if (ns < (u64)NSEC_PER_SEC * 3600) {
		tenths = div64_u64(ns, (u64)NSEC_PER_SEC * 6);
		unit = "min";
	} else if (ns < (u64)NSEC_PER_SEC * 86400) {
		tenths = div64_u64(ns, (u64)NSEC_PER_SEC * 360);
		unit = "hr";
	} else {
		tenths = div64_u64(ns, (u64)NSEC_PER_SEC * 8640);
		unit = "days";
	}

	tenths = div_u64_rem(tenths, 10, &tenth);
	seq_printf(m, "%llu ns (%llu.%u %s)\n", ns, tenths, tenth, unit);
}

/* Nanoseconds as whole microseconds, the unit every gauge is reported in. */
static u64 mlfq_us(u64 ns)
{
	return div_u64(ns, NSEC_PER_USEC);
}

/*
 * The adaptation shift as a percentage with one decimal, which is what the
 * dashboard's Adapt shift row is. The shift is one-sided and bounded to a half
 * by MLFQ_ADAPT_MAX_SHIFT, so a tenth of a percent is the whole resolution of
 * it rather than a rounding of something finer, and the only reason the sign is
 * handled at all is to keep a value that should not exist from printing as an
 * enormous positive one. Magnitude first and sign separately, because a shift
 * right of a negative value is not a division by a power of two;
 * mlfq_adapt_band() splits it the same way for the same reason.
 */
static u64 mlfq_shift_pct(s64 shift, u32 *tenth)
{
	u64 mag = shift < 0 ? (u64)-shift : (u64)shift;

	return div_u64_rem((mag * 1000) >> MLFQ_FP_SHIFT, 10, tenth);
}

static void mlfq_seq_count(struct seq_file *m, const char *label, u64 val)
{
	seq_printf(m, " %" MLFQ_LABEL_WIDTH "%llu\n", label, val);
}

static void mlfq_seq_us(struct seq_file *m, const char *label, u64 ns)
{
	seq_printf(m, " %" MLFQ_LABEL_WIDTH "%llu us\n", label, mlfq_us(ns));
}

/*
 * The per-CPU card grid, one row per card. The columns are the card's own four
 * lines: its identity with the SMT and RT badges, the current and maximum
 * frequency, the cache domain, and the level of the task running on it.
 *
 * The last column is where this differs from the card, which shows "idle"
 * whenever no fair task is running, including when a task of a higher class is
 * holding the CPU, because the level it reads is zero either way. There is no
 * reason to repeat that here: a CPU running something from a higher class is
 * not idle, so it gets a dash, and the rt column says what it is doing.
 */
static void mlfq_show_per_cpu(struct seq_file *m)
{
	int cpu;

	seq_puts(m, "\nPer-CPU\n");
	seq_puts(m, " cpu   cur/max MHz   llc   smt    rt   running\n");

	for_each_online_cpu(cpu) {
		unsigned int cur_khz = cpufreq_quick_get(cpu);
		unsigned int max_khz = cpufreq_quick_get_max(cpu);
		int level = mlfq_cpu_level(cpu);

		seq_printf(m, " %3d   ", cpu);
		if (cur_khz)
			seq_printf(m, "%5u", (cur_khz + 500) / 1000);
		else
			seq_puts(m, "    -");
		seq_printf(m, "/%-5u   %3d   %3s   %3s   %s\n",
			   (max_khz + 500) / 1000, mlfq_cpu_llc(cpu),
			   mlfq_cpu_smt(cpu) ? "SMT" : "-",
			   level < 0 ? "RT" : "-",
			   level < 0 ? "-" : mlfq_queue_name[level]);
	}
}

/*
 * The Summary, in the shape buildSummary() gives it: a TL;DR that says whether
 * the classifier is doing its job, and an Explanation that walks through the
 * live gauges and the rule each one is being judged against.
 *
 * Upstream's version is written around its learned burst-prediction tree,
 * which is not part of this port, so the sentences that quote the model's
 * error and its correlation have nothing to quote. The band controller is what
 * adapts here, so they are replaced by the gauges it reads and the shift it
 * has arrived at, and the threshold the TL;DR turns on becomes the
 * controller's own target latency -- upstream's threshold is likewise the
 * scheduler's own published rule for when a fit is good enough, rather than a
 * number picked to make the sentence work.
 */
static void mlfq_show_summary(struct seq_file *m)
{
	u64 target_us = MLFQ_ADAPT_TARGET_LAT_NS / NSEC_PER_USEC;
	u64 base_l_us = MLFQ_THRESH_LOW_NS / NSEC_PER_USEC;
	u64 base_h_us = MLFQ_THRESH_HIGH_NS / NSEC_PER_USEC;
	u64 lat = mlfq_us(READ_ONCE(mlfq_sys_gauge.lat_ema));
	u64 rate = READ_ONCE(mlfq_sys_gauge.rate_ema) >> MLFQ_FP_SHIFT;
	bool folded = READ_ONCE(mlfq_sys_gauge.step_at) != 0;
	s64 shift = READ_ONCE(mlfq_adapt_state.shift_fp);
	struct mlfq_bands bands = mlfq_read_bands();
	u64 pct;
	u32 tenth;

	seq_puts(m, "\nSummary\n TL;DR\n  ");

	if (!sched_feat(MLFQ)) {
		seq_puts(m,
			 "MLFQ is turned off and the fair class is unmodified.\n");
		seq_puts(m, " Explanation\n  ");
		seq_printf(m,
			   "sched_feat(MLFQ) is off, so no task is being classified and every request is %llu us again, straight from sysctl_sched_base_slice. Based on the internal rules of this scheduler, that request is the only thing MLFQ ever changes, so with it off this is EEVDF exactly as it shipped, and the counters below are whatever had accumulated by the time it was turned off.\n",
			   mlfq_us(sysctl_sched_base_slice));
		return;
	}

	/*
	 * Asked before the still-learning state below, because with the
	 * controller off there is nothing for the bands to learn: they are at
	 * their base values and will stay there, whether or not a window has
	 * been folded into the gauge that would otherwise have moved them.
	 */
	if (!sched_feat(MLFQ_ADAPT)) {
		seq_puts(m,
			 "MLFQ is working as expected, with adaptive band tuning turned off.\n");
		seq_puts(m, " Explanation\n  ");
		seq_printf(m,
			   "As you can see from the wakeup-latency gauge, it shows %llu us, against the %llu us target the bands would have been tuned to deliver. sched_feat(MLFQ_ADAPT) is off, so the band edges stay at %llu us and %llu us for good and the wakeup-rate gauge stays frozen, since the only thing that reads it is the storm gate the controller would have applied. Based on the internal rules of this scheduler, the latency gauge is folded either way, so that turning the controller on is an informed choice, and this shows what it would have been reacting to.\n",
			   lat, target_us, mlfq_us(bands.t_l),
			   mlfq_us(bands.t_h));
		return;
	}

	if (!folded) {
		seq_puts(m,
			 "MLFQ is working as expected and is still learning your workload.\n");
		seq_puts(m, " Explanation\n  ");
		seq_printf(m,
			   "No measurement window has been folded yet, so the gauges are still collecting this machine's own wakeup behaviour, and the classifier is comparing against the base band edges of %llu us and %llu us. Based on the internal rules of this scheduler, the bands only ever move away from those edges while the wakeup-latency gauge runs above its %llu us target, and this shows that MLFQ is starting from the thresholds it was designed around.\n",
			   base_l_us, base_h_us, target_us);
		return;
	}

	seq_printf(m, "MLFQ is working as expected and is %s to your workload.\n",
		   lat <= target_us ? "adapting correctly" : "still adapting");
	seq_puts(m, " Explanation\n  ");
	seq_printf(m,
		   "As you can see from the wakeup-latency gauge, it shows %llu us, against a target of %llu us. ",
		   lat, target_us);
	if (rate)
		seq_printf(m, "The wakeup-rate gauge shows %llu per second, ",
			   rate);
	else
		seq_puts(m, "The wakeup-rate gauge is still warming up, ");

	pct = mlfq_shift_pct(shift, &tenth);
	seq_printf(m,
		   "and the classifier is comparing against band edges of %llu us and %llu us, from base edges of %llu us and %llu us. Based on the internal rules of this scheduler, the bands only ever widen, only while the gauge runs above target, and by at most ten percentage points per step, so the final result is a shift of %s%llu.%u %%, and this shows that MLFQ is adapting to what your machine is actually doing.\n",
		   mlfq_us(bands.t_l), mlfq_us(bands.t_h), base_l_us, base_h_us,
		   shift < 0 ? "-" : "", pct, tenth);
}

/*
 * Per-domain runnable occupancy, the dashboard's LLC loads row. Only domains
 * with tracked tasks in them are listed, as the chips were, and a machine with
 * nothing queued anywhere gets a single zero rather than a row of them.
 *
 * The scan is quadratic in the CPU count because a domain is named by one of
 * its own CPUs and there is no array to index by that name. A row is emitted
 * by the lowest-numbered CPU of each domain, which is also the CPU the domain
 * is named after.
 */
static void mlfq_show_llc_loads(struct seq_file *m)
{
	bool any = false;
	int cpu;

	seq_printf(m, " %" MLFQ_LABEL_WIDTH, "llc loads");

	for_each_online_cpu(cpu) {
		int id = mlfq_cpu_llc(cpu);
		int first = -1, other;
		u32 load = 0;

		for_each_online_cpu(other) {
			if (mlfq_cpu_llc(other) != id)
				continue;
			if (first < 0)
				first = other;
			load += mlfq_cpu_runnable(other);
		}

		if (first != cpu || !load)
			continue;

		seq_printf(m, "%sLLC %d: %u", any ? "  " : "", id, load);
		any = true;
	}

	seq_puts(m, any ? "\n" : "0\n");
}

/*
 * The System counter grid: the dashboard's own fields, in its own order, with
 * the adaptation gauges after the counters as they are there. The fields it
 * fills from state this port does not keep are left out rather than printed as
 * zero, and mlfq_show_absent() lists them.
 */
static void mlfq_show_system(struct seq_file *m)
{
	struct mlfq_totals t;
	s64 shift;
	u64 pct;
	u32 tenth;

	mlfq_read_totals(&t);

	seq_puts(m, "\nSystem\n");

	mlfq_seq_count(m, "on cpu", mlfq_on_cpu_now());

	/*
	 * Upstream's On CPU is the gauge above, so the counter kept here goes
	 * under its own name: scx_mlfq raised and lowered one number in
	 * ops.running() and ops.stopping(), where this port counts arrivals
	 * only and reads the gauge from the runqueues directly, which cannot
	 * drift the way a hand-balanced pair can.
	 */
	mlfq_seq_count(m, "switch-ins", t.stat[MLFQ_STAT_ON_CPU]);

	seq_printf(m, " %" MLFQ_LABEL_WIDTH, "uptime");
	mlfq_seq_duration(m, ktime_get_ns());

	seq_printf(m, " %" MLFQ_LABEL_WIDTH, "service");
	mlfq_seq_duration(m, t.stat[MLFQ_STAT_TOTAL_RUNTIME]);

	mlfq_seq_count(m, "q1 placements", t.stat[MLFQ_STAT_Q1_PLACEMENTS]);
	mlfq_seq_count(m, "q2 placements", t.stat[MLFQ_STAT_Q2_PLACEMENTS]);
	mlfq_seq_count(m, "q3 placements", t.stat[MLFQ_STAT_Q3_PLACEMENTS]);
	mlfq_seq_count(m, "q1 runnable", t.q_runnable[MLFQ_Q_INTERACTIVE]);
	mlfq_seq_count(m, "q2 runnable", t.q_runnable[MLFQ_Q_DEFAULT]);
	mlfq_seq_count(m, "q3 runnable", t.q_runnable[MLFQ_Q_BATCH]);
	mlfq_seq_count(m, "promotions", t.stat[MLFQ_STAT_PROMOTIONS]);
	mlfq_seq_count(m, "demotions", t.stat[MLFQ_STAT_DEMOTIONS]);
	mlfq_seq_count(m, "aging boosts", t.stat[MLFQ_STAT_AGING_BOOSTS]);
	mlfq_seq_count(m, "short-sleep boosts",
		       t.stat[MLFQ_STAT_SHORT_SLEEP_BOOSTS]);
	mlfq_seq_count(m, "preemption kicks",
		       t.stat[MLFQ_STAT_PREEMPTION_KICKS]);

	mlfq_seq_us(m, "wakeup lat", READ_ONCE(mlfq_sys_gauge.lat_ema));

	seq_printf(m, " %" MLFQ_LABEL_WIDTH "%llu /s\n", "wakeup rate",
		   READ_ONCE(mlfq_sys_gauge.rate_ema) >> MLFQ_FP_SHIFT);

	seq_printf(m, " %" MLFQ_LABEL_WIDTH "%llu / %llu us\n",
		   "t_l / t_h eff",
		   mlfq_us(READ_ONCE(mlfq_adapt_state.t_l_eff_ns)),
		   mlfq_us(READ_ONCE(mlfq_adapt_state.t_h_eff_ns)));

	shift = READ_ONCE(mlfq_adapt_state.shift_fp);
	pct = mlfq_shift_pct(shift, &tenth);
	seq_printf(m, " %" MLFQ_LABEL_WIDTH "%s%llu.%u %%\n", "adapt shift",
		   shift < 0 ? "-" : "", pct, tenth);

	mlfq_seq_count(m, "wakeups", t.wake_total);
	mlfq_seq_count(m, "adapt steps", READ_ONCE(mlfq_sys_gauge.adapt_steps));

	mlfq_show_llc_loads(m);
}

/*
 * What a reader who knows the dashboard will look for here and not find.
 *
 * Every one of these is a number about a mechanism the fair class does not
 * have, and for each of them the reason is more useful than a zero would be: a
 * zero says the mechanism is idle, where the truth is that the work it counted
 * is either being done by something else under a different name, or is not
 * work this scheduler has to do at all.
 */
static void mlfq_show_absent(struct seq_file *m)
{
	seq_puts(m, "\nNot reported\n");

	seq_puts(m,
		 " cpuperf boosts\n"
		 "   The interactive level does not raise the CPU's performance\n"
		 "   target. schedutil drives frequency from the PELT utilisation\n"
		 "   signal, which already rises for a task that is being given the\n"
		 "   CPU more often, so the level has nothing left to ask for.\n");

	seq_puts(m,
		 " steals, steals same-LLC, steals cross-LLC, keep running\n"
		 "   Dispatch queue mechanics. Tasks move between runqueues here by\n"
		 "   load balancing, which decides on load and locality rather than\n"
		 "   on a queue being empty, and which accounts for itself through\n"
		 "   the schedstats domain counters.\n");

	seq_puts(m,
		 " rt takeovers, rt evacuations, rt redirects, rt reenqs\n"
		 "   The fair class runs below rt and dl by sched_class order, not\n"
		 "   by noticing them and yielding a CPU, so none of this is an\n"
		 "   event that could be counted. The rt column of the per-CPU rows\n"
		 "   above is what is left of it: the pressure is visible, the\n"
		 "   reaction to it does not exist because it is not needed.\n");

	seq_puts(m,
		 " tree gen, tree nodes, tree MAE, tree corr, t_int / t_bnd eff\n"
		 "   The learned burst-prediction model, and the second band pair it\n"
		 "   feeds. Fitting it is userspace work in scx_mlfq, published back\n"
		 "   through a BPF map; there is no daemon here, and the exponential\n"
		 "   gauge in mlfq.h is the whole of the classifier's model.\n");

	seq_puts(m,
		 " guard eff\n"
		 "   The minimum run a task must have had before a same-level wakeup\n"
		 "   may preempt it. Upstream fixes it at zero and derives nothing\n"
		 "   else from it, so the port has no such guard to report; see\n"
		 "   MLFQ_PREEMPT_SLICE_NS in mlfq.h.\n");

	seq_puts(m,
		 " op lat\n"
		 "   Time spent inside a BPF scheduler's own callbacks, measured\n"
		 "   because sched_ext charges them to the scheduler it is hosting.\n"
		 "   These hooks are compiled into the fair class.\n");
}

static int mlfq_stats_show(struct seq_file *m, void *v)
{
	seq_puts(m,
		 "scx_mlfq on EEVDF: multilevel feedback queues, virtual time, placement\n");
	seq_printf(m, "mlfq: %s   adaptive bands: %s\n",
		   sched_feat(MLFQ) ? "on" : "off",
		   sched_feat(MLFQ_ADAPT) ? "on" : "off");

	mlfq_show_per_cpu(m);
	mlfq_show_summary(m);
	mlfq_show_system(m);
	mlfq_show_absent(m);

	return 0;
}

/*
 * subsys_initcall, matching proc_schedstat_init() in stats.c: procfs is up by
 * then, and the scheduler has been running since before any of this, so there
 * is no state to wait for and nothing that reads the file before it exists. A
 * failed registration is not worth failing the boot over, and there is nothing
 * to undo, so the return value is deliberately ignored.
 */
static int __init mlfq_proc_init(void)
{
	proc_create_single("sched_eevdf_mlfq_stats", 0, NULL, mlfq_stats_show);

	return 0;
}
subsys_initcall(mlfq_proc_init);
