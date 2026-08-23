// SPDX-License-Identifier: GPL-2.0
/*
 * TAGLMK - Task-aware Android Guided Low Memory Killer
 *
 * Task classification, package pinning, and the scan that turns the process
 * list into an ordered set of candidates.
 *
 * A scan runs in two phases, and the split is not cosmetic.  Deciding what a
 * process is needs its full command line, and reading that means taking another
 * address space's mmap lock, which may sleep - so it cannot happen while the
 * process list is held under RCU.  Phase one therefore walks the list under RCU
 * doing only cheap tests, and takes a reference to everything it keeps; phase
 * two does the expensive work on those pinned tasks with nothing held, then
 * drops the ones that turned out not to qualify.  Every reference taken in
 * phase one is released either there or by taglmk_release_victims(), and
 * nothing else in the driver calls put_task_struct().
 *
 * Copyright (C) 2026 iDeadXS <datarafi43@gmail.com>
 */

#define pr_fmt(fmt) "taglmk: " fmt

#include <linux/cred.h>
#include <linux/ctype.h>
#include <linux/kernel.h>
#include <linux/mm.h>
#include <linux/oom.h>
#include <linux/rcupdate.h>
#include <linux/sched.h>
#include <linux/sched/cputime.h>
#include <linux/sched/mm.h>
#include <linux/sched/signal.h>
#include <linux/sched/task.h>
#include <linux/slab.h>
#include <linux/sort.h>
#include <linux/string.h>
#include <linux/uidgid.h>
#include <linux/user_namespace.h>

#include "taglmk.h"

/*
 * Android's oom_score_adj bands, as the framework hands them out.  Only the
 * bottom of the range is named here, because that is the only part this file
 * has to reason about: at or below the persistent service band is Android
 * running itself, and none of it is ours to kill.
 */
#define TAGLMK_ADJ_PERSISTENT		(-700)

/*
 * Android's uid layout.  A uid is the application id plus a hundred thousand
 * per secondary user, so the application id is what identifies the kind of
 * process and the multiple of a hundred thousand only says which user it
 * belongs to.
 */
#define TAGLMK_AID_USER_OFFSET		100000
#define TAGLMK_AID_APP_START		10000
#define TAGLMK_AID_APP_END		19999
#define TAGLMK_AID_ISOLATED_START	99000
#define TAGLMK_AID_ISOLATED_END		99999

/* Longest package name accepted, including the terminator. */
#define TAGLMK_PKG_MAX			128

/* Most packages that may be pinned at once. */
#define TAGLMK_PIN_MAX			32

/**
 * struct taglmk_pin_list - the pinned package names, published by RCU
 * @rcu: For kfree_rcu().
 * @gen: Generation of this list.  It lives inside the object rather than beside
 *	it so that one rcu_dereference() yields a name set and the number that
 *	identifies it as a matched pair, with no ordering to reason about.
 * @nr: Names in @name.
 * @name: The names themselves, fixed width so the whole list is one allocation.
 *
 * The list is immutable once published.  A change builds a new one and swaps
 * the pointer, so a reader either sees the whole of the old list or the whole of
 * the new one.
 */
struct taglmk_pin_list {
	struct rcu_head	rcu;
	u32		gen;
	unsigned int	nr;
	char		name[][TAGLMK_PKG_MAX];
};

static struct taglmk_pin_list __rcu *taglmk_pins;
static DEFINE_MUTEX(taglmk_pin_mutex);		/* one writer at a time */

/*
 * Scratch for one command line at a time.  Allocated once at init because a
 * pass runs precisely when memory is short and must not allocate, and safe to
 * share because the pass mutex means only one classification is ever in flight.
 */
static char *taglmk_name;

/*
 * Package pinning
 * ===============
 */

/*
 * The current generation.  A cached class has to agree with this to be usable,
 * and that is the only thing it has to agree with.
 */
static u32 taglmk_pin_gen(void)
{
	u32 gen;

	rcu_read_lock();
	gen = rcu_dereference(taglmk_pins)->gen;
	rcu_read_unlock();

	return gen;
}

static bool taglmk_pin_match(const char *name)
{
	struct taglmk_pin_list *list;
	unsigned int i;
	bool hit = false;

	rcu_read_lock();
	list = rcu_dereference(taglmk_pins);

	for (i = 0; i < list->nr; i++) {
		size_t len = strlen(list->name[i]);

		if (strncmp(name, list->name[i], len))
			continue;

		/*
		 * A package may run in more than one process, and the extra
		 * ones are named after the package with a colon and a suffix.
		 * They are the same application to the user, so one pin covers
		 * all of them.
		 */
		if (name[len] == '\0' || name[len] == ':') {
			hit = true;
			break;
		}
	}

	rcu_read_unlock();

	return hit;
}

/*
 * Task classification
 * ===================
 */

/* Read a task's command line into the shared scratch buffer. */
static bool taglmk_read_name(struct task_struct *tsk)
{
	int len = get_cmdline(tsk, taglmk_name, TAGLMK_PKG_MAX - 1);

	/*
	 * get_cmdline() does not promise a terminator, and returns nothing at
	 * all for a task whose address space has already gone.  Terminate what
	 * did arrive; the first argument ends at the first NUL either way, and
	 * for an Android application that argument is the package name zygote
	 * gave it.
	 */
	if (len <= 0)
		return false;

	taglmk_name[len] = '\0';

	return taglmk_name[0] != '\0';
}

static bool taglmk_is_app_uid(uid_t app_id)
{
	/* An ordinary installed application. */
	if (app_id >= TAGLMK_AID_APP_START && app_id <= TAGLMK_AID_APP_END)
		return true;

	/*
	 * An isolated process - a web view renderer and the like.  It holds no
	 * state of its own and the framework already treats it as the first
	 * thing to lose, so it counts as an ordinary application here too.
	 */
	if (app_id >= TAGLMK_AID_ISOLATED_START &&
	    app_id <= TAGLMK_AID_ISOLATED_END)
		return true;

	return false;
}

/*
 * Work out what a task is.  The tests are ordered so that the answer can only
 * ever move towards being harder to kill: nothing below can turn a system app
 * into a plain one, and pinning is reached last, where it has nothing left to
 * downgrade.  That monotonicity is what makes the ladder safe to widen at
 * TAGLMK_LEVEL_CRITICAL.
 */
static enum taglmk_task_type taglmk_classify_slow(struct task_struct *tsk)
{
	bool named;
	uid_t app_id;
	short adj;

	/*
	 * Everything Android runs to be Android sits at or below the persistent
	 * service band, so this single test covers init, zygote, system_server,
	 * surfaceflinger and the persistent processes without naming any of
	 * them, and keeps working when a vendor adds one more.
	 */
	adj = READ_ONCE(tsk->signal->oom_score_adj);
	if (adj <= TAGLMK_ADJ_PERSISTENT)
		return TAGLMK_TYPE_CRITICAL;

	named = taglmk_read_name(tsk);
	if (named) {
		/*
		 * An application's first argument is the package name, so it
		 * always contains a dot and never a slash.  Anything else is
		 * either a native process still carrying the path it was
		 * executed from or one of zygote's own, and is part of the
		 * platform by definition.  This is deliberately redundant with
		 * the band test above: a daemon whose oom_score_adj was never
		 * set would otherwise be indistinguishable from an application.
		 */
		if (taglmk_name[0] == '/' || !strchr(taglmk_name, '.'))
			return TAGLMK_TYPE_CRITICAL;
	}

	rcu_read_lock();
	app_id = from_kuid(&init_user_ns, task_uid(tsk)) %
		 TAGLMK_AID_USER_OFFSET;
	rcu_read_unlock();

	/*
	 * Below the application range are the platform uids - system, radio,
	 * bluetooth, nfc and the rest.  Those processes are ordinary in every
	 * other way, but losing one takes a piece of the platform with it, so
	 * they stay out of range until the situation is critical.
	 */
	if (!taglmk_is_app_uid(app_id))
		return TAGLMK_TYPE_SYSTEM_APP;

	if (named && taglmk_pin_match(taglmk_name))
		return TAGLMK_TYPE_PINNED;

	return TAGLMK_TYPE_APP;
}

static enum taglmk_task_type taglmk_classify(struct task_struct *tsk)
{
	enum taglmk_task_type type;
	u32 gen, cached;

	/*
	 * The generation is read once, up front, and both the lookup and the
	 * stamp below use that one value.  A list published while this is
	 * running therefore produces an entry stamped with the older
	 * generation, which the next pass reads as stale and recomputes.  The
	 * error can only ever be in that direction, so a cached class never
	 * claims to describe a list it was not formed against.
	 */
	gen = taglmk_pin_gen();

	cached = READ_ONCE(tsk->signal->taglmk_class);
	if (taglmk_class_gen(cached) == gen)
		return taglmk_class_type(cached);

	type = taglmk_classify_slow(tsk);

	/*
	 * Two passes can arrive here at once for the same task and both write.
	 * They computed the same answer from the same inputs, and the word is
	 * written whole, so the race has no observable effect.
	 */
	WRITE_ONCE(tsk->signal->taglmk_class, taglmk_class_pack(gen, type));

	return type;
}

/*
 * Scanning
 * ========
 */

/*
 * Phase one test.  Runs under RCU with the process list held, so it may not
 * sleep and must stay cheap: everything here is a field read.
 */
static bool taglmk_maybe_candidate(struct task_struct *tsk, u32 gen,
				   short min_adj, enum taglmk_task_type cutoff)
{
	u32 cached;

	/* No address space to take, and nothing that can be killed. */
	if (tsk->flags & PF_KTHREAD)
		return false;

	/* The init of any namespace is not a candidate for anything. */
	if (is_global_init(tsk))
		return false;

	/*
	 * Already on the way out.  Killing it again frees nothing, and walking
	 * an address space that is being torn down only competes with
	 * exit_mmap() for the same pages.
	 */
	if (tsk->flags & PF_EXITING || fatal_signal_pending(tsk) ||
	    tsk_is_oom_victim(tsk))
		return false;

	if (READ_ONCE(tsk->signal->oom_score_adj) < min_adj)
		return false;

	/*
	 * If the cached class was decided against the list that is still
	 * current, the cut-off can be applied here and a task that is out of
	 * range never has to be picked up at all.  Otherwise take it and let
	 * phase two classify it, where sleeping is allowed.
	 */
	cached = READ_ONCE(tsk->signal->taglmk_class);
	if (taglmk_class_gen(cached) == gen &&
	    taglmk_class_type(cached) >= cutoff)
		return false;

	return true;
}

/*
 * Phase two.  A reference is held on @v->tsk and nothing else is, so this may
 * sleep.  Returns false when the task turned out not to be a candidate after
 * all, in which case the caller releases it.
 */
static bool taglmk_fill_victim(struct taglmk_victim *v,
			       enum taglmk_task_type cutoff)
{
	struct task_struct *tsk = v->tsk;
	struct task_cputime cputime;
	struct task_struct *owner;

	/*
	 * The mm belongs to whichever thread of the group still holds it, and
	 * it has to be read under task_lock or exit_mm() can take it away
	 * mid-read.  No reference is needed on the thread itself: nothing here
	 * uses it once the lock is dropped.
	 */
	owner = find_lock_task_mm(tsk);
	if (!owner)
		return false;

	v->anon_pages = get_mm_counter(owner->mm, MM_ANONPAGES);
	v->swap_pages = get_mm_counter(owner->mm, MM_SWAPENTS);
	task_unlock(owner);

	/*
	 * utime plus stime across the whole thread group.  This is the
	 * survivability key: the time a user spends in an application accrues
	 * here, so the application they live in ends up a long way behind the
	 * one they opened once and forgot about, and killing starts at the far
	 * end of that order.
	 */
	thread_group_cputime(tsk, &cputime);
	v->cputime = cputime.utime + cputime.stime;

	v->adj = READ_ONCE(tsk->signal->oom_score_adj);
	v->type = taglmk_classify(tsk);
	v->budget = 0;
	v->skip = false;

	return v->type < cutoff;
}

unsigned int taglmk_scan_tasks(enum taglmk_level level)
{
	enum taglmk_task_type cutoff = taglmk_type_cutoff(level);
	short min_adj = taglmk_min_adj(level);
	struct task_struct *tsk;
	unsigned int i, keep, nr = 0;
	u32 gen;

	/*
	 * Phase one: choose cheaply under RCU and take a reference to each
	 * task kept, so that phase two can let go of the process list.
	 */
	rcu_read_lock();
	gen = rcu_dereference(taglmk_pins)->gen;

	for_each_process(tsk) {
		if (nr == taglmk.scan_limit)
			break;

		if (!taglmk_maybe_candidate(tsk, gen, min_adj, cutoff))
			continue;

		get_task_struct(tsk);
		taglmk.victims[nr].tsk = tsk;
		nr++;
	}
	rcu_read_unlock();

	/*
	 * Published before phase two so that a release always covers every
	 * reference taken above, however phase two ends.
	 */
	taglmk.nr_victims = nr;

	/*
	 * Phase two: fill in the expensive fields and compact the array down to
	 * the tasks that really are candidates, releasing the rest as they are
	 * dropped.
	 */
	for (i = 0, keep = 0; i < nr; i++) {
		struct taglmk_victim *v = &taglmk.victims[i];
		struct task_struct *dropped = v->tsk;

		if (taglmk_fill_victim(v, cutoff)) {
			if (keep != i)
				taglmk.victims[keep] = *v;
			keep++;
			continue;
		}

		put_task_struct(dropped);
	}

	/*
	 * Compacting leaves copies of kept entries above the new end.  Clear
	 * them so that the only task pointers anywhere in the array are the
	 * live ones, and a stray read can never turn into a double release.
	 */
	memset(&taglmk.victims[keep], 0,
	       (nr - keep) * sizeof(*taglmk.victims));
	taglmk.nr_victims = keep;

	return keep;
}

void taglmk_release_victims(void)
{
	unsigned int i;

	for (i = 0; i < taglmk.nr_victims; i++) {
		put_task_struct(taglmk.victims[i].tsk);
		taglmk.victims[i].tsk = NULL;
	}

	taglmk.nr_victims = 0;
}

/*
 * Ordering
 * ========
 */
static int taglmk_cmp_cputime(const void *a, const void *b)
{
	const struct taglmk_victim *x = a, *y = b;

	/*
	 * Ascending, so the least used application comes first.  Compared
	 * rather than subtracted: these are nanosecond counters and their
	 * difference does not fit in the int a comparator returns.
	 */
	if (x->cputime != y->cputime)
		return x->cputime < y->cputime ? -1 : 1;

	/*
	 * Equal time is common shortly after boot, when several cached
	 * applications have barely run.  Break the tie on the larger resident
	 * set, so that the one kill this buys gives back as much as it can.
	 */
	if (x->anon_pages != y->anon_pages)
		return x->anon_pages > y->anon_pages ? -1 : 1;

	return 0;
}

static int taglmk_cmp_anon(const void *a, const void *b)
{
	const struct taglmk_victim *x = a, *y = b;

	/* Descending: the most to reclaim from, first. */
	if (x->anon_pages != y->anon_pages)
		return x->anon_pages > y->anon_pages ? -1 : 1;

	return 0;
}

void taglmk_sort_by_cputime(void)
{
	sort(taglmk.victims, taglmk.nr_victims, sizeof(*taglmk.victims),
	     taglmk_cmp_cputime, NULL);
}

void taglmk_sort_by_anon(void)
{
	sort(taglmk.victims, taglmk.nr_victims, sizeof(*taglmk.victims),
	     taglmk_cmp_anon, NULL);
}

/*
 * The pin list as userspace sees it
 * =================================
 */
ssize_t taglmk_pin_show(char *buf)
{
	struct taglmk_pin_list *list;
	unsigned int i;
	ssize_t n = 0;

	rcu_read_lock();
	list = rcu_dereference(taglmk_pins);

	/*
	 * scnprintf() clamps to the space left and reports what it wrote, so
	 * the running total can never walk past the page even if the list is
	 * somehow longer than one.
	 */
	for (i = 0; i < list->nr; i++)
		n += scnprintf(buf + n, PAGE_SIZE - n, "%s\n", list->name[i]);

	rcu_read_unlock();

	return n;
}

static bool taglmk_pkg_char(char c)
{
	return isalnum(c) || c == '.' || c == '_' || c == ':';
}

/* Advance past separators, then measure the token that follows. */
static const char *taglmk_next_token(const char *p, const char *end,
				     size_t *len)
{
	const char *start;

	while (p < end && (*p == '\0' || *p == ',' || isspace(*p)))
		p++;

	for (start = p; p < end && *p != ',' && !isspace(*p) && *p; p++)
		;

	*len = p - start;

	return start;
}

static int taglmk_pin_validate(const char *tok, size_t len)
{
	size_t i;

	if (len >= TAGLMK_PKG_MAX)
		return -ENAMETOOLONG;

	for (i = 0; i < len; i++)
		if (!taglmk_pkg_char(tok[i]))
			return -EINVAL;

	/*
	 * Every Android package name has at least two segments, so a name with
	 * no dot in it can never match a process.  Refuse it rather than accept
	 * a pin that would silently never fire.
	 */
	if (!memchr(tok, '.', len))
		return -EINVAL;

	return 0;
}

int taglmk_pin_store(const char *buf, size_t len)
{
	struct taglmk_pin_list *new, *old;
	const char *end = buf + len;
	const char *p, *tok;
	unsigned int nr = 0;
	size_t toklen;
	u32 gen;
	int ret;

	/* First pass: count and validate, so nothing is allocated in vain. */
	for (p = buf; p < end; p = tok + toklen) {
		tok = taglmk_next_token(p, end, &toklen);
		if (!toklen)
			break;

		ret = taglmk_pin_validate(tok, toklen);
		if (ret)
			return ret;

		if (++nr > TAGLMK_PIN_MAX)
			return -E2BIG;
	}

	new = kzalloc(sizeof(*new) + nr * TAGLMK_PKG_MAX, GFP_KERNEL);
	if (!new)
		return -ENOMEM;

	/* Second pass: copy.  Validation already proved every token fits. */
	for (p = buf; p < end && new->nr < nr; p = tok + toklen) {
		tok = taglmk_next_token(p, end, &toklen);
		if (!toklen)
			break;

		memcpy(new->name[new->nr], tok, toklen);
		new->name[new->nr][toklen] = '\0';
		new->nr++;
	}

	mutex_lock(&taglmk_pin_mutex);
	old = rcu_dereference_protected(taglmk_pins,
					lockdep_is_held(&taglmk_pin_mutex));

	/*
	 * Numbering from one keeps the zero a signal_struct is allocated with
	 * meaning "never classified".  Twenty four bits is what the packed word
	 * has room for; a wrap would let a class stamped with the same number
	 * sixteen million writes ago read as current, which needs a process
	 * that old and a userspace rewriting this file in a loop.
	 */
	gen = old->gen + 1;
	if (gen > TAGLMK_CLASS_GEN_MAX)
		gen = 1;
	new->gen = gen;

	rcu_assign_pointer(taglmk_pins, new);
	mutex_unlock(&taglmk_pin_mutex);

	kfree_rcu(old, rcu);

	pr_info("%u package(s) pinned, generation %u\n", nr, gen);

	return 0;
}

/*
 * Bring-up
 * ========
 */
int taglmk_task_init(void)
{
	struct taglmk_pin_list *list;

	taglmk_name = kzalloc(TAGLMK_PKG_MAX, GFP_KERNEL);
	if (!taglmk_name)
		return -ENOMEM;

	/*
	 * An empty list rather than a null pointer, so that every reader is a
	 * plain dereference with nothing to check.
	 */
	list = kzalloc(sizeof(*list), GFP_KERNEL);
	if (!list) {
		kfree(taglmk_name);
		taglmk_name = NULL;
		return -ENOMEM;
	}

	list->gen = 1;
	rcu_assign_pointer(taglmk_pins, list);

	return 0;
}

void taglmk_task_exit(void)
{
	struct taglmk_pin_list *list;

	/*
	 * Only reachable from the init error path, where the sysfs files are
	 * already gone and the notifier was never registered, so there is no
	 * reader to wait for.  The mutex is taken regardless, to keep the one
	 * rule about this pointer - never written without it - true everywhere.
	 */
	mutex_lock(&taglmk_pin_mutex);
	list = rcu_dereference_protected(taglmk_pins,
					 lockdep_is_held(&taglmk_pin_mutex));
	RCU_INIT_POINTER(taglmk_pins, NULL);
	mutex_unlock(&taglmk_pin_mutex);

	kfree(list);
	kfree(taglmk_name);
	taglmk_name = NULL;
}
