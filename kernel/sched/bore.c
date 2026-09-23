/*
 *  Burst-Oriented Response Enhancer (BORE-Lite) CPU Scheduler
 *  Copyright (C) 2021-2026 Masahito Suzuki <firelzrd@gmail.com>
 */

#include <linux/cpuset.h>
#include <linux/sched/task.h>
#include <linux/sched/bore.h>
#include "sched.h"

#ifdef CONFIG_SCHED_BORE_LITE
uint __read_mostly sched_credit_cap_us	= 20000;

static int zero;
static int __maybe_unused maxval_1_million = 1000000;

DEFINE_STATIC_KEY_TRUE(sched_credit_key);

void bore_note_sleep(struct task_struct *p, u64 now)
{
	p->bore.credit_sleep = now;
}

u64 bore_credit_ns(struct task_struct *p)
{
	u64 cap = (u64)sched_credit_cap_us * 1000ULL;
	u64 slept = 0, credit = 0;

	if (p->bore.credit_sleep) {
		slept = rq_clock(task_rq(p)) - p->bore.credit_sleep;
		if ((s64)slept > 0) {
			p->bore.credit_sleep = 0;
			credit = slept < cap ? slept : cap;
		} else {
			slept = 0;
		}
	}

	return credit;
}

void __init sched_init_bore(void) {
	printk(KERN_INFO "%s %s by %s\n",
		SCHED_BORE_PROGNAME, SCHED_BORE_VERSION, SCHED_BORE_AUTHOR);

	if (sched_credit_cap_us)
		static_branch_enable(&sched_credit_key);
}

int sched_credit_cap_us_update_handler(struct ctl_table *table,
		int write, void __user *buffer, size_t *lenp, loff_t *ppos) {
	int ret = proc_douintvec_minmax(table, write, buffer, lenp, ppos);
	if (ret || !write)
		return ret;

	if (sched_credit_cap_us)
		static_branch_enable(&sched_credit_key);
	else
		static_branch_disable(&sched_credit_key);

	return 0;
}

#ifdef CONFIG_SYSCTL
static struct ctl_table sched_bore_sysctls[] = {
	{
		.procname	= "sched_credit_cap_us",
		.data		= &sched_credit_cap_us,
		.maxlen		= sizeof(uint),
		.mode		= 0644,
		.proc_handler	= sched_credit_cap_us_update_handler,
		.extra1		= &zero,
		.extra2		= &maxval_1_million,
	},
    { }
};

static int __init sched_bore_sysctl_init(void) {
	register_sysctl_init("kernel", sched_bore_sysctls);
	return 0;
}
late_initcall(sched_bore_sysctl_init);

#endif /* CONFIG_SYSCTL */
#endif /* CONFIG_SCHED_BORE_LITE */