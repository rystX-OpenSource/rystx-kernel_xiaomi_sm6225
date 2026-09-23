#ifndef _KERNEL_SCHED_BORE_H
#define _KERNEL_SCHED_BORE_H

#include <linux/sched.h>
#include <linux/sched/cputime.h>

#define SCHED_BORE_AUTHOR   "Masahito Suzuki"
#define SCHED_BORE_PROGNAME "BORE-Lite CPU Scheduler modification"

#define SCHED_BORE_VERSION  "7.0.0"

extern uint __read_mostly sched_credit_cap_us;
DECLARE_STATIC_KEY_TRUE(sched_credit_key);

extern void bore_note_sleep(struct task_struct *p, u64 now);
extern u64  bore_credit_ns(struct task_struct *p);

extern void sched_init_bore(void);

extern int  sched_credit_cap_us_update_handler(struct ctl_table *table,
	int write, void __user *buffer, size_t *lenp, loff_t *ppos);

#endif /* _KERNEL_SCHED_BORE_H */