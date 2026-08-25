/* SPDX-License-Identifier: GPL-2.0 */
/*
 * TAGLMK - Task-aware Android Guided Low Memory Killer
 *
 * Contract shared between the parts of the driver.  Nothing outside
 * drivers/android/taglmk/ includes this; the interface the rest of the kernel
 * sees is include/linux/taglmk.h.
 *
 * Copyright (C) 2026 iDeadXS <datarafi43@gmail.com>
 */
#ifndef _DRIVERS_ANDROID_TAGLMK_H
#define _DRIVERS_ANDROID_TAGLMK_H

#include <linux/atomic.h>
#include <linux/math64.h>
#include <linux/mmzone.h>
#include <linux/sched.h>
#include <linux/sched/signal.h>
#include <linux/taglmk.h>
#include <linux/types.h>
#include <linux/workqueue.h>

/* A size in MiB as a page count, whatever PAGE_SIZE happens to be. */
#define TAGLMK_MB_PAGES(mb)	((unsigned long)(mb) << (20 - PAGE_SHIFT))

/*
 * Fixed point arithmetic
 * ======================
 *
 * Two unsigned formats are used.
 *
 * Q4.4    One byte: four integer and four fractional bits, so [0, 15.9375] in
 *         steps of 1/16.  This is the burstiness factor.  It is small, read
 *         often, and staying in a single byte means a whole window of them
 *         fits in one NEON register.
 *
 * intfp32 Thirty-two bits: sixteen integer and sixteen fractional bits.  Used
 *         wherever range matters - the scalar factor, the predictor output and
 *         the regression state - and it is also what the scalar fallback works
 *         in.  Every operation below is integer only, so a NEON build and a
 *         fallback build agree bit for bit on the result.
 */
#define TAGLMK_Q44_SHIFT	4
#define TAGLMK_Q44_ONE		(1U << TAGLMK_Q44_SHIFT)
#define TAGLMK_Q44_MAX		0xffU

#define TAGLMK_FP_SHIFT		16
#define TAGLMK_FP_ONE		(1U << TAGLMK_FP_SHIFT)

/* Plain integer to intfp32, saturating. */
static inline u32 taglmk_fp(unsigned long v)
{
	if (v >= (1UL << (32 - TAGLMK_FP_SHIFT)))
		return U32_MAX;

	return (u32)v << TAGLMK_FP_SHIFT;
}

/* intfp32 back to a plain integer, truncating. */
static inline unsigned long taglmk_fp_int(u32 fp)
{
	return fp >> TAGLMK_FP_SHIFT;
}

/* Q4.4 to intfp32.  Exact: both are binary fractions. */
static inline u32 taglmk_q44_to_fp(u8 q44)
{
	return (u32)q44 << (TAGLMK_FP_SHIFT - TAGLMK_Q44_SHIFT);
}

/* intfp32 to Q4.4, truncating and saturating at 15.9375. */
static inline u8 taglmk_fp_to_q44(u32 fp)
{
	fp >>= TAGLMK_FP_SHIFT - TAGLMK_Q44_SHIFT;

	return fp > TAGLMK_Q44_MAX ? TAGLMK_Q44_MAX : (u8)fp;
}

/* a * b in intfp32, saturating. */
static inline u32 taglmk_fp_mul(u32 a, u32 b)
{
	u64 prod = ((u64)a * b) >> TAGLMK_FP_SHIFT;

	return prod > U32_MAX ? U32_MAX : (u32)prod;
}

/* a / b in intfp32, saturating.  A zero divisor yields zero, never a trap. */
static inline u32 taglmk_fp_div(u32 a, u32 b)
{
	u64 quot;

	if (!b)
		return 0;

	quot = div_u64((u64)a << TAGLMK_FP_SHIFT, b);

	return quot > U32_MAX ? U32_MAX : (u32)quot;
}

/*
 * a / b in intfp32 straight from two plain integers.  Page counts routinely
 * exceed the 16 bits taglmk_fp() has left for the integer part, so a ratio
 * must never be formed by converting the operands first: taglmk_fp() would
 * saturate both of them and the quotient would come out as one.  Widening
 * first and dividing once keeps every ratio below exact.
 */
static inline u32 taglmk_ratio_fp(unsigned long a, unsigned long b)
{
	u64 quot;

	if (!b)
		return 0;

	quot = div_u64((u64)a << TAGLMK_FP_SHIFT, b);

	return quot > U32_MAX ? U32_MAX : (u32)quot;
}

/*
 * Memory pressure ladder
 * ======================
 *
 * TAGLMK_LEVEL_NONE      Swap is above free_swap_limit and the active file LRU
 *                        is above free_file_limit.  Reclaim only: pages are
 *                        taken from tasks, nothing is killed.
 *
 * TAGLMK_LEVEL_LOW       Free swap has fallen to free_swap_limit.  Tasks are
 *                        collected, sorted, and killed lowest cputime first
 *                        until swap recovers.
 *
 * TAGLMK_LEVEL_CRITICAL  The active file LRU has fallen below
 *                        free_file_limit, so there is no cache left to fall
 *                        back on.  Killing becomes more aggressive: bigger
 *                        batches, and system apps stop being off limits.
 */
enum taglmk_level {
	TAGLMK_LEVEL_NONE,
	TAGLMK_LEVEL_LOW,
	TAGLMK_LEVEL_CRITICAL,
};

/**
 * struct taglmk_profile - tuning preset selected by installed RAM size
 * @name: Shown in the boot message and through sysfs.
 * @ram_pages: The preset applies to devices with at most this much RAM.
 * @free_swap_limit: Free swap, in pages, at or below which killing starts.
 * @free_file_limit: Active file pages below which the situation is critical.
 * @reclaim_budget: Pages one reclaim pass aims to take, before scaling.
 * @scan_limit: How many tasks a single scan will look at.
 * @kill_batch: Victims per pass at %TAGLMK_LEVEL_LOW.
 * @kill_batch_crit: Victims per pass at %TAGLMK_LEVEL_CRITICAL.
 * @burst_gain: Q4.4 compensation applied to the predictor on this class of
 *	device.  Less RAM means less room to be wrong, so it reacts sooner.
 * @swap_target_pct: Swap utilisation the ZRAM balancer aims to hold.
 */
struct taglmk_profile {
	const char	*name;
	unsigned long	ram_pages;
	unsigned long	free_swap_limit;
	unsigned long	free_file_limit;
	unsigned int	reclaim_budget;
	unsigned int	scan_limit;
	unsigned int	kill_batch;
	unsigned int	kill_batch_crit;
	u8		burst_gain;
	u8		swap_target_pct;
};

/* Hard ceiling on tasks held in the victim array, whatever a profile asks. */
#define TAGLMK_MAX_VICTIMS	128

/**
 * struct taglmk_victim - one candidate gathered by a scan
 * @tsk: Thread group member that owns the mm.  A reference is held for as long
 *	as this entry is live, and dropped by taglmk_release_victims().
 * @type: Class decided for @tsk.
 * @adj: oom_score_adj as it was when the scan ran.
 * @anon_pages: MM_ANONPAGES at scan time, which is the reclaim potential.
 * @swap_pages: MM_SWAPENTS at scan time, which is what already went to ZRAM.
 * @cputime: utime + stime summed across the thread group.  This is the
 *	survivability key: the longer the user has spent in an application the
 *	more this has accumulated, and the later the application is killed.
 * @budget: Share of the reclaim budget assigned to this task.
 * @skip: Set when a later stage decided to leave this task alone.
 */
struct taglmk_victim {
	struct task_struct	*tsk;
	enum taglmk_task_type	type;
	short			adj;
	unsigned long		anon_pages;
	unsigned long		swap_pages;
	u64			cputime;
	unsigned int		budget;
	bool			skip;
};

/**
 * struct taglmk_state - everything the driver keeps between passes
 * @profile: Preset chosen at init from the installed RAM size.
 * @free_swap_limit: Live copy of the profile field, writable through sysfs.
 * @free_file_limit: Live copy of the profile field, writable through sysfs.
 * @reclaim_budget: Live copy of the profile field, writable through sysfs.
 * @scan_limit: Live copy of the profile field, writable through sysfs.  Never
 *	above %TAGLMK_MAX_VICTIMS, which is what @victims is sized for.
 * @kill_batch: Live copy of the profile field, writable through sysfs.
 * @kill_batch_crit: Live copy of the profile field, writable through sysfs.
 * @pressure_min: vmpressure level at or above which a pass is queued.
 * @min_adj: Tasks below this oom_score_adj are spared.
 * @min_adj_crit: The same bar once the situation is critical, which is lower:
 *	with no cache left there is nothing to be gained by being polite.
 * @enabled: Master switch.  Clearing it stops new passes being queued.
 * @dry_run: Log what would be killed without sending any signal.
 * @work: The pass itself, run on a high priority workqueue.
 * @lock: Serialises passes and guards @victims and @nr_victims.
 * @victims: Scan output, %TAGLMK_MAX_VICTIMS entries, allocated once at init
 *	and never resized.  A pass fills at most @scan_limit of them.
 * @nr_victims: Live entries in @victims.
 * @nr_killed: Tasks killed since boot.
 * @nr_reclaimed: Pages reclaimed since boot.
 * @nr_passes: Passes run since boot.
 * @nr_kill_passes: Passes that found the situation bad enough to enter the kill
 *	band, whether or not anything was actually killed.  Read together with
 *	@nr_killed this separates "never needed to kill" from "wanted to kill
 *	and could not", which are very different states to be in and look
 *	identical from a kill count alone.
 * @nr_no_candidate: Kill passes that came back with an empty candidate list.
 *
 * The tunables above are plain naturally aligned words.  Passes read them
 * without holding anything and sysfs writes them without holding anything
 * either, which is deliberate: no single read can tear, every one of them is
 * independent of the others, and a pass that acts on the value from a
 * microsecond ago behaves exactly as it would have done had it been queued a
 * microsecond earlier.  The counters are atomics because they are accumulated
 * rather than replaced, and @victims is the one thing here that a reader could
 * be hurt by, which is why it is under @lock.
 */
struct taglmk_state {
	const struct taglmk_profile *profile;

	unsigned long		free_swap_limit;
	unsigned long		free_file_limit;
	unsigned int		reclaim_budget;
	unsigned int		scan_limit;
	unsigned int		kill_batch;
	unsigned int		kill_batch_crit;
	unsigned int		pressure_min;
	short			min_adj;
	short			min_adj_crit;
	bool			enabled;
	bool			dry_run;

	struct work_struct	work;
	struct mutex		lock;
	struct taglmk_victim	*victims;
	unsigned int		nr_victims;

	atomic_long_t		nr_killed;
	atomic_long_t		nr_reclaimed;
	atomic_long_t		nr_passes;
	atomic_long_t		nr_kill_passes;
	atomic_long_t		nr_no_candidate;
};

extern struct taglmk_state taglmk;

/*
 * Task classification cache, kept in signal_struct so it lives and dies with
 * the process.  The class sits in the low byte and the pin list generation it
 * was decided against in the rest, packed into one word so a reader can never
 * see half of an update.  Generations start at one, so the zero a signal_struct
 * is allocated with always reads as "never classified".
 */
#define TAGLMK_CLASS_TYPE_MASK	0xffU
#define TAGLMK_CLASS_GEN_SHIFT	8
#define TAGLMK_CLASS_GEN_MAX	(U32_MAX >> TAGLMK_CLASS_GEN_SHIFT)

static inline u32 taglmk_class_pack(u32 gen, enum taglmk_task_type type)
{
	return (gen << TAGLMK_CLASS_GEN_SHIFT) |
	       ((u32)type & TAGLMK_CLASS_TYPE_MASK);
}

static inline u32 taglmk_class_gen(u32 cached)
{
	return cached >> TAGLMK_CLASS_GEN_SHIFT;
}

static inline enum taglmk_task_type taglmk_class_type(u32 cached)
{
	return (enum taglmk_task_type)(cached & TAGLMK_CLASS_TYPE_MASK);
}

/* core.c */
enum taglmk_level taglmk_mem_level(void);
short taglmk_min_adj(enum taglmk_level level);
enum taglmk_task_type taglmk_type_cutoff(enum taglmk_level level);
unsigned long taglmk_free_swap_pages(void);
unsigned long taglmk_active_file_pages(void);
unsigned long taglmk_inactive_file_pages(void);

/* task.c */
int taglmk_task_init(void);
void taglmk_task_exit(void);
unsigned int taglmk_scan_tasks(enum taglmk_level level);
void taglmk_release_victims(void);
void taglmk_sort_by_cputime(void);
void taglmk_sort_by_anon(void);
ssize_t taglmk_pin_show(char *buf);
int taglmk_pin_store(const char *buf, size_t len);

/* predict.c */
void taglmk_predict_sample(void);
u8 taglmk_predict_burstiness(void);
u32 taglmk_predict_factor(void);
unsigned int taglmk_predict_budget(unsigned int base);

/* zram.c */
void taglmk_zram_observe(unsigned long asked, unsigned long got);
unsigned int taglmk_zram_utilisation(void);
u32 taglmk_zram_efficiency(void);
unsigned int taglmk_zram_budget(unsigned int base);
void taglmk_zram_share(struct taglmk_victim *v, unsigned int nr,
		       unsigned int budget);
u8 taglmk_ir_depth(const struct taglmk_victim *v, u64 cputime_avg);

/* sysfs.c */
int taglmk_sysfs_init(void);
void taglmk_sysfs_exit(void);

/*
 * neon.c - the accelerated kernels, built only when the NEON option is on.
 *
 * taglmk_neon_ok() has to be asked again at every call site rather than cached,
 * because whether the FPU may be touched depends on the context the caller is
 * running in, not on the CPU.  With the option off it folds to a compile time
 * false and the calls below are dropped before they ever reach the linker.
 */
#ifdef CONFIG_ANDROID_TAGLMK_ARM64_NEON
bool taglmk_neon_ok(void);
void taglmk_neon_window_stats(const u32 *x, unsigned int n, u32 *out_mean,
			      u32 *out_absdiff);
void taglmk_neon_regress(const u32 *x, const u32 *y, unsigned int n,
			 u64 *out_sx, u64 *out_sy, u64 *out_sxx, u64 *out_sxy);
void taglmk_neon_share(const u32 *anon, u32 *out, unsigned int n, u32 scale);
#else
static inline bool taglmk_neon_ok(void)
{
	return false;
}

static inline void taglmk_neon_window_stats(const u32 *x, unsigned int n,
					    u32 *out_mean, u32 *out_absdiff)
{
}

static inline void taglmk_neon_regress(const u32 *x, const u32 *y,
				       unsigned int n, u64 *out_sx, u64 *out_sy,
				       u64 *out_sxx, u64 *out_sxy)
{
}

static inline void taglmk_neon_share(const u32 *anon, u32 *out, unsigned int n,
				     u32 scale)
{
}
#endif /* CONFIG_ANDROID_TAGLMK_ARM64_NEON */

#endif /* _DRIVERS_ANDROID_TAGLMK_H */
