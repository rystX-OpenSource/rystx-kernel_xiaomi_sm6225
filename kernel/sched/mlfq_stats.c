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

static void mlfq_seq_count(struct seq_file *m, const char *label, u64 val)
{
	seq_printf(m, " %" MLFQ_LABEL_WIDTH "%llu\n", label, val);
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
 * counters and the rule each one is being judged against.
 *
 * Upstream reads five of the counters below and turns them into two shares of
 * the wakeup total, one for the wakeups that ended in a demotion and one for the
 * wakeups that ended in a promotion of any kind, then picks its sentence from
 * where those two fall against a hundredth and a twentieth. The comparisons here
 * are the same ones, multiplied out into whole counts so there is no division,
 * and they are made against the same totals the System grid prints rather than a
 * second reading of the per-CPU state, so the prose and the numbers below it can
 * never disagree.
 */
static void mlfq_show_summary(struct seq_file *m, const struct mlfq_totals *t)
{
	u64 wakes = t->wake_total;
	u64 demotes = t->stat[MLFQ_STAT_DEMOTIONS];
	u64 ups = t->stat[MLFQ_STAT_PROMOTIONS] +
		  t->stat[MLFQ_STAT_AGING_BOOSTS] +
		  t->stat[MLFQ_STAT_SHORT_SLEEP_BOOSTS];
	u32 on_cpu = mlfq_on_cpu_now();

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

	if (!wakes) {
		seq_puts(m,
			 "MLFQ is working as expected but has not seen a wakeup yet.\n");
		seq_puts(m, " Explanation\n  ");
		seq_printf(m,
			   "Nothing has been classified so far. Based on the internal rules of this scheduler, a task's level is only ever revisited when it wakes from a sleep or when it uses a whole request without sleeping, so every task is still in Q2 on its %llu us request, and this shows a machine that has had nothing to do since MLFQ was turned on.\n",
			   mlfq_us(MLFQ_SLICE_Q2_NS));
		return;
	}

	if (demotes * 100 < wakes && ups) {
		seq_puts(m,
			 "MLFQ is working as expected and your workload is predominantly interactive.\n");
		seq_puts(m, " Explanation\n  ");
		seq_printf(m,
			   "As you can see from the counters, %llu wakeups have produced %llu promotions and %llu demotions. Based on the internal rules of this scheduler, a task is only demoted once it has used %d whole requests in a row without sleeping, so fewer than one demotion per hundred wakeups says that almost everything here sleeps between short pieces of work, and this shows the interactive level holding the tasks that belong in it.",
			   wakes, ups, demotes, MLFQ_DEMOTE_REENQS);
	} else if (demotes * 20 > wakes) {
		seq_puts(m,
			 "MLFQ is working as expected and is sorting a mixed workload.\n");
		seq_puts(m, " Explanation\n  ");
		seq_printf(m,
			   "As you can see from the counters, %llu of the %llu wakeups seen so far ended in a demotion, against %llu promotions the other way. Based on the internal rules of this scheduler, those two are driven by opposite evidence -- %d consecutive exhausted requests demote, a short sleep or a stay of %llu ms promote -- so both of them running at once is what a machine with real work and real interaction looks like, and this shows the classifier telling the two apart.",
			   demotes, wakes, ups, MLFQ_DEMOTE_REENQS,
			   div_u64(MLFQ_AGING_PERIOD_NS, NSEC_PER_MSEC));
	} else {
		seq_puts(m,
			 "MLFQ is working as expected.\n");
		seq_puts(m, " Explanation\n  ");
		seq_printf(m,
			   "As you can see from the counters, %llu wakeups have produced %llu promotions and %llu demotions, and the three levels below hold what that sorting arrived at. Based on the internal rules of this scheduler, a level only changes on evidence that has repeated, so a machine whose tasks are already where they belong reclassifies little, and this shows the classification holding steady rather than idling.",
			   wakes, ups, demotes);
	}

	if (on_cpu)
		seq_printf(m, " %u tasks are on a CPU right now.", on_cpu);

	seq_putc(m, '\n');
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
 * The System counter grid: the dashboard's own fields, in its own order. The
 * fields it fills from state this port does not keep are left out rather than
 * printed as zero, and mlfq_show_absent() lists them.
 */
static void mlfq_show_system(struct seq_file *m, const struct mlfq_totals *t)
{
	seq_puts(m, "\nSystem\n");

	mlfq_seq_count(m, "on cpu", mlfq_on_cpu_now());

	/*
	 * Upstream's On CPU is the gauge above, so the counter kept here goes
	 * under its own name: scx_mlfq raised and lowered one number in
	 * ops.running() and ops.stopping(), where this port counts arrivals
	 * only and reads the gauge from the runqueues directly, which cannot
	 * drift the way a hand-balanced pair can.
	 */
	mlfq_seq_count(m, "switch-ins", t->stat[MLFQ_STAT_ON_CPU]);

	seq_printf(m, " %" MLFQ_LABEL_WIDTH, "uptime");
	mlfq_seq_duration(m, ktime_get_ns());

	seq_printf(m, " %" MLFQ_LABEL_WIDTH, "service");
	mlfq_seq_duration(m, t->stat[MLFQ_STAT_TOTAL_RUNTIME]);

	mlfq_seq_count(m, "q1 placements", t->stat[MLFQ_STAT_Q1_PLACEMENTS]);
	mlfq_seq_count(m, "q2 placements", t->stat[MLFQ_STAT_Q2_PLACEMENTS]);
	mlfq_seq_count(m, "q3 placements", t->stat[MLFQ_STAT_Q3_PLACEMENTS]);
	mlfq_seq_count(m, "q1 runnable", t->q_runnable[MLFQ_Q_INTERACTIVE]);
	mlfq_seq_count(m, "q2 runnable", t->q_runnable[MLFQ_Q_DEFAULT]);
	mlfq_seq_count(m, "q3 runnable", t->q_runnable[MLFQ_Q_BATCH]);
	mlfq_seq_count(m, "promotions", t->stat[MLFQ_STAT_PROMOTIONS]);
	mlfq_seq_count(m, "demotions", t->stat[MLFQ_STAT_DEMOTIONS]);
	mlfq_seq_count(m, "aging boosts", t->stat[MLFQ_STAT_AGING_BOOSTS]);
	mlfq_seq_count(m, "short-sleep boosts",
		       t->stat[MLFQ_STAT_SHORT_SLEEP_BOOSTS]);
	mlfq_seq_count(m, "preemption kicks",
		       t->stat[MLFQ_STAT_PREEMPTION_KICKS]);
	mlfq_seq_count(m, "wakeups", t->wake_total);

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
	struct mlfq_totals t;

	mlfq_read_totals(&t);

	seq_puts(m,
		 "scx_mlfq on EEVDF: multilevel feedback queues, virtual time, placement\n");
	seq_printf(m, "mlfq: %s\n", sched_feat(MLFQ) ? "on" : "off");

	mlfq_show_per_cpu(m);
	mlfq_show_summary(m, &t);
	mlfq_show_system(m, &t);
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
