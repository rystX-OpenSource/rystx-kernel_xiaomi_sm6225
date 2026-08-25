// SPDX-License-Identifier: GPL-2.0
/*
 * TAGLMK - the /sys/kernel/mm/taglmk interface.
 *
 * Everything the driver decides from is visible here, and everything that is a
 * policy rather than a fact is writable.  That is deliberate: the profiles are
 * chosen from installed RAM and are a starting point, not an answer, and the
 * only way to find the answer for a particular device is to watch it run and
 * move the numbers.
 *
 * Writes are validated and rejected, never clamped.  A tool that asks for a
 * scan_limit of a thousand has misunderstood something, and silently giving it
 * a hundred and twenty eight would hide that; -EINVAL does not.  The one bound
 * that matters for safety rather than sense is scan_limit, which can never
 * exceed the victim array it indexes.
 *
 * Locking: every tunable here is one naturally aligned word, written with a
 * single store and read by a pass with a single load, which is why none of
 * these handlers takes taglmk.lock.  Taking it would mean a sysfs write could
 * block behind a pass that is in the middle of killing something, which is
 * exactly the moment an operator most wants to be able to set dry_run or clear
 * enabled.  See the note on struct taglmk_state for the whole argument.
 *
 * Copyright (C) 2026 iDeadXS <datarafi43@gmail.com>
 */

#include <linux/kernel.h>
#include <linux/kobject.h>
#include <linux/mm.h>
#include <linux/oom.h>
#include <linux/string.h>
#include <linux/sysfs.h>

#include "taglmk.h"

/*
 * Upper bound on reclaim_budget.  Thirty two megabytes of pages in a single
 * pass is already far more than any profile asks for, and bounding it keeps the
 * arithmetic in the predictor and the balancer inside the ranges their comments
 * claim it stays inside.
 */
#define TAGLMK_BUDGET_MAX	8192

static struct kobject *taglmk_kobj;

static const char * const taglmk_level_names[] = {
	[TAGLMK_LEVEL_NONE]	= "none",
	[TAGLMK_LEVEL_LOW]	= "low",
	[TAGLMK_LEVEL_CRITICAL]	= "critical",
};

/*
 * A fixed point value as a decimal with four places, for either format: pass
 * the format's fractional shift and the fraction is scaled from that.
 */
static ssize_t taglmk_show_fixed(char *buf, u32 value, unsigned int shift)
{
	u32 one = 1U << shift;
	u32 frac = (u32)(((u64)(value & (one - 1)) * 10000) >> shift);

	return sprintf(buf, "%u.%04u\n", value >> shift, frac);
}

/*
 * Boilerplate for the tunables.  Each use generates a show that prints the live
 * value, a store that parses and range checks before committing, and the
 * attribute that binds them together.
 *
 * @check is an expression in the parsed value, which the macro calls @val, and
 * every bound is spelled out at the point of use rather than inferred from the
 * type.  That is worth the small loss of macro hygiene: the bounds are the
 * interesting part of each of these, and a reader should not have to work out
 * whether a limit came from the field's width or from a decision somebody made.
 */
#define TAGLMK_ATTR_RW(field, type, fmt, parse, check)			\
static ssize_t field##_show(struct kobject *kobj,			\
			    struct kobj_attribute *attr, char *buf)	\
{									\
	return sprintf(buf, fmt "\n", taglmk.field);			\
}									\
static ssize_t field##_store(struct kobject *kobj,			\
			     struct kobj_attribute *attr,		\
			     const char *buf, size_t len)		\
{									\
	type val;							\
	int ret;							\
									\
	ret = parse(buf, 0, &val);					\
	if (ret)							\
		return ret;						\
									\
	if (!(check))							\
		return -EINVAL;						\
									\
	taglmk.field = val;						\
									\
	return len;							\
}									\
static struct kobj_attribute field##_attr = __ATTR_RW(field)

#define TAGLMK_ATTR_BOOL(field)						\
static ssize_t field##_show(struct kobject *kobj,			\
			    struct kobj_attribute *attr, char *buf)	\
{									\
	return sprintf(buf, "%u\n", taglmk.field ? 1 : 0);		\
}									\
static ssize_t field##_store(struct kobject *kobj,			\
			     struct kobj_attribute *attr,		\
			     const char *buf, size_t len)		\
{									\
	bool val;							\
	int ret;							\
									\
	ret = strtobool(buf, &val);					\
	if (ret)							\
		return ret;						\
									\
	taglmk.field = val;						\
									\
	return len;							\
}									\
static struct kobj_attribute field##_attr = __ATTR_RW(field)

TAGLMK_ATTR_BOOL(enabled);
TAGLMK_ATTR_BOOL(dry_run);

/*
 * A watermark above the amount of memory installed could never be satisfied,
 * so the driver would kill without ever stopping.  That is the bound worth
 * enforcing.  Zero is left legal at the bottom: it means "never act on this
 * signal", which is a coherent thing to ask for and the natural way to isolate
 * one arm of the ladder during an evaluation.  A negative is rejected by
 * kstrtoul() before the check is reached.
 */
TAGLMK_ATTR_RW(free_swap_limit, unsigned long, "%lu", kstrtoul,
	       val <= totalram_pages);
TAGLMK_ATTR_RW(free_file_limit, unsigned long, "%lu", kstrtoul,
	       val <= totalram_pages);

/* Zero here means "kill only, never reclaim", which is also worth being able
 * to ask for.
 */
TAGLMK_ATTR_RW(reclaim_budget, unsigned int, "%u", kstrtouint,
	       val <= TAGLMK_BUDGET_MAX);

/*
 * This one is a safety bound rather than a matter of taste: scan_limit is how
 * far a pass indexes into the victim array, and the array is exactly
 * TAGLMK_MAX_VICTIMS long.
 */
TAGLMK_ATTR_RW(scan_limit, unsigned int, "%u", kstrtouint,
	       val >= 1 && val <= TAGLMK_MAX_VICTIMS);

/* A batch of zero is what the enabled switch is for; the array is the ceiling. */
TAGLMK_ATTR_RW(kill_batch, unsigned int, "%u", kstrtouint,
	       val >= 1 && val <= TAGLMK_MAX_VICTIMS);
TAGLMK_ATTR_RW(kill_batch_crit, unsigned int, "%u", kstrtouint,
	       val >= 1 && val <= TAGLMK_MAX_VICTIMS);

/* vmpressure reports a percentage, so anything above a hundred is a typo. */
TAGLMK_ATTR_RW(pressure_min, unsigned int, "%u", kstrtouint, val <= 100);

/* The oom_score_adj bars, in the range userspace is allowed to set. */
TAGLMK_ATTR_RW(min_adj, short, "%d", kstrtos16,
	       val >= OOM_SCORE_ADJ_MIN && val <= OOM_SCORE_ADJ_MAX);
TAGLMK_ATTR_RW(min_adj_crit, short, "%d", kstrtos16,
	       val >= OOM_SCORE_ADJ_MIN && val <= OOM_SCORE_ADJ_MAX);

/*
 * The pinned package list.  Reading gives one name per line; writing replaces
 * the whole list, and an empty write clears it.  task.c owns both halves
 * because it owns the list and the generation counter that goes with it.
 */
static ssize_t pin_show(struct kobject *kobj, struct kobj_attribute *attr,
			char *buf)
{
	return taglmk_pin_show(buf);
}

static ssize_t pin_store(struct kobject *kobj, struct kobj_attribute *attr,
			 const char *buf, size_t len)
{
	int ret = taglmk_pin_store(buf, len);

	return ret ? ret : len;
}

static struct kobj_attribute pin_attr = __ATTR_RW(pin);

/* Which preset the installed RAM size selected at boot. */
static ssize_t profile_show(struct kobject *kobj, struct kobj_attribute *attr,
			    char *buf)
{
	return sprintf(buf, "%s\n", taglmk.profile->name);
}

static struct kobj_attribute profile_attr = __ATTR_RO(profile);

/* How noisy the predictor thinks the cache load is, Q4.4. */
static ssize_t burstiness_show(struct kobject *kobj,
			       struct kobj_attribute *attr, char *buf)
{
	return taglmk_show_fixed(buf, taglmk_predict_burstiness(),
				 TAGLMK_Q44_SHIFT);
}

static struct kobj_attribute burstiness_attr = __ATTR_RO(burstiness);

/* The multiplier the predictor is currently applying to the budget. */
static ssize_t predicted_factor_show(struct kobject *kobj,
				     struct kobj_attribute *attr, char *buf)
{
	return taglmk_show_fixed(buf, taglmk_predict_factor(), TAGLMK_FP_SHIFT);
}

static struct kobj_attribute predicted_factor_attr =
	__ATTR_RO(predicted_factor);

/* Pages the balancer has learnt to expect back per page asked for. */
static ssize_t swap_efficiency_show(struct kobject *kobj,
				    struct kobj_attribute *attr, char *buf)
{
	return taglmk_show_fixed(buf, taglmk_zram_efficiency(),
				 TAGLMK_FP_SHIFT);
}

static struct kobj_attribute swap_efficiency_attr =
	__ATTR_RO(swap_efficiency);

/*
 * One place to read the whole picture from, in the order a person diagnosing a
 * device would want it: what the driver has done, then what it is looking at.
 * Every figure is sampled independently, so this is a summary and not an
 * instant; that is what makes it cheap enough to poll.
 */
static ssize_t stats_show(struct kobject *kobj, struct kobj_attribute *attr,
			  char *buf)
{
	enum taglmk_level level = taglmk_mem_level();
	size_t len = 0;

	len += scnprintf(buf + len, PAGE_SIZE - len,
			 "passes:        %ld\n",
			 atomic_long_read(&taglmk.nr_passes));
	len += scnprintf(buf + len, PAGE_SIZE - len,
			 "kill_passes:   %ld\n",
			 atomic_long_read(&taglmk.nr_kill_passes));
	len += scnprintf(buf + len, PAGE_SIZE - len,
			 "no_candidate:  %ld\n",
			 atomic_long_read(&taglmk.nr_no_candidate));
	len += scnprintf(buf + len, PAGE_SIZE - len,
			 "killed:        %ld\n",
			 atomic_long_read(&taglmk.nr_killed));
	len += scnprintf(buf + len, PAGE_SIZE - len,
			 "reclaimed:     %ld\n",
			 atomic_long_read(&taglmk.nr_reclaimed));
	len += scnprintf(buf + len, PAGE_SIZE - len,
			 "level:         %s\n", taglmk_level_names[level]);
	len += scnprintf(buf + len, PAGE_SIZE - len,
			 "free_swap:     %lu\n", taglmk_free_swap_pages());
	len += scnprintf(buf + len, PAGE_SIZE - len,
			 "active_file:   %lu\n", taglmk_active_file_pages());
	len += scnprintf(buf + len, PAGE_SIZE - len,
			 "inactive_file: %lu\n", taglmk_inactive_file_pages());
	len += scnprintf(buf + len, PAGE_SIZE - len,
			 "swap_util:     %u\n", taglmk_zram_utilisation());
	len += scnprintf(buf + len, PAGE_SIZE - len,
			 "neon:          %u\n", taglmk_neon_ok() ? 1 : 0);

	return len;
}

static struct kobj_attribute stats_attr = __ATTR_RO(stats);

static struct attribute *taglmk_attrs[] = {
	&enabled_attr.attr,
	&dry_run_attr.attr,
	&free_swap_limit_attr.attr,
	&free_file_limit_attr.attr,
	&reclaim_budget_attr.attr,
	&scan_limit_attr.attr,
	&kill_batch_attr.attr,
	&kill_batch_crit_attr.attr,
	&pressure_min_attr.attr,
	&min_adj_attr.attr,
	&min_adj_crit_attr.attr,
	&pin_attr.attr,
	&profile_attr.attr,
	&burstiness_attr.attr,
	&predicted_factor_attr.attr,
	&swap_efficiency_attr.attr,
	&stats_attr.attr,
	NULL,
};

static const struct attribute_group taglmk_attr_group = {
	.attrs = taglmk_attrs,
};

/**
 * taglmk_sysfs_init - publish /sys/kernel/mm/taglmk
 *
 * Under mm_kobj rather than somewhere under the driver, because what is exposed
 * here is memory policy and that is where Android's tooling already looks for
 * it.  mm_kobj exists by postcore_initcall, well before this runs.
 */
int taglmk_sysfs_init(void)
{
	int ret;

	taglmk_kobj = kobject_create_and_add("taglmk", mm_kobj);
	if (!taglmk_kobj)
		return -ENOMEM;

	ret = sysfs_create_group(taglmk_kobj, &taglmk_attr_group);
	if (ret) {
		kobject_put(taglmk_kobj);
		taglmk_kobj = NULL;
	}

	return ret;
}

/**
 * taglmk_sysfs_exit - withdraw the interface again
 *
 * Only ever reached from the init error path: once the driver is up it stays
 * up, since there is no way to unregister the killer the system relies on.
 */
void taglmk_sysfs_exit(void)
{
	if (!taglmk_kobj)
		return;

	sysfs_remove_group(taglmk_kobj, &taglmk_attr_group);
	kobject_put(taglmk_kobj);
	taglmk_kobj = NULL;
}
