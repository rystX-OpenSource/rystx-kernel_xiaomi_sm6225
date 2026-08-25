// SPDX-License-Identifier: GPL-2.0
/*
 * TAGLMK - ZRAM balancer.
 *
 * Reclaiming anonymous memory on an Android device does not free it, it
 * compresses it: the page leaves the LRU and reappears, smaller, inside
 * zsmalloc.  So a reclaim pass is a trade, and this file is what decides
 * whether the trade is still worth making and how large it should be.
 *
 * Three measurements go into that, all of them of the machine as it actually
 * is rather than of what it was configured to be.
 *
 * Utilisation.  How full the swap device is, against the per-profile target.
 * Below the target there is room and the pass runs at full size.  Between the
 * target and full it tapers to nothing, so the last of swap is left for the
 * allocations that genuinely cannot proceed without it.  At full it is zero,
 * and the core spends the pass ageing file pages instead.
 *
 * Density.  How many pages of anonymous memory each physical page handed to
 * zsmalloc is currently holding.  Compression that is going badly - encrypted
 * data, media buffers, anything already compressed - means each page evicted
 * buys back only a fraction of a page, and past a point the trade stops paying
 * for the page table walk it took.  So poor density tapers the budget too.
 *
 * Yield.  What a pass actually gets for what it asks, fitted by least squares
 * across a sliding window of the passes that came before.  This is the adaptive
 * part: the slope of got against asked is the marginal return on asking for one
 * more page, and its reciprocal is how much to inflate a request so the amount
 * that comes back is the amount that was wanted.  A driver that ignored this
 * would systematically under-reclaim on exactly the devices that can least
 * afford it, because they are the ones where the walks come back short.
 *
 * There is a fourth thing here, which is not a measurement but an action: when
 * the three above agree that swap is as full as it should get, this file can
 * ask zram to recompress what it is already holding.  That gives memory back
 * without evicting a page or killing a task, so it is tried as soon as the anon
 * pass has nothing left to gain.  See taglmk_ir_sweep().
 *
 * Locking: the window and the scratch arrays are private to the pass work item,
 * which is serialised by taglmk.lock, so they need nothing of their own.  The
 * fitted yield is published with WRITE_ONCE() for sysfs and for the budget
 * path, on the same reasoning as the predictor's factor.  The sweep's scratch
 * page and interval are private to the pass for the same reason.
 *
 * Copyright (C) 2026 iDeadXS <datarafi43@gmail.com>
 */

#include <linux/compiler.h>
#include <linux/gfp.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/math64.h>
#include <linux/string.h>
#include <linux/swap.h>
#include <linux/types.h>
#include <linux/vmstat.h>
#include <linux/zram_ir.h>

#include "taglmk.h"

/* Passes remembered by the fit.  A multiple of four, as the NEON kernel wants. */
#define TAGLMK_ZRAM_WINDOW	16

/*
 * Samples are clamped to this before they are stored.  Nothing sane reaches it
 * - a pass asks for a few thousand pages at most - but reclaim_budget is
 * writable through sysfs, and bounding the input here is what lets every sum
 * below be reasoned about as fitting comfortably in 64 bits rather than merely
 * fitting in practice.
 */
#define TAGLMK_ZRAM_SAMPLE_MAX	U16_MAX

/* The reciprocal of the yield is never allowed to inflate a request further. */
#define TAGLMK_ZRAM_GAIN_MAX	(2U * TAGLMK_FP_ONE)

/*
 * Density at or above which compression is considered to be working: 1.25
 * pages of anonymous memory per physical page held by zsmalloc.  Below it the
 * budget tapers in proportion, reaching zero only if zsmalloc is somehow using
 * more memory than it stores.
 */
#define TAGLMK_ZRAM_DENSITY_FLOOR	(5U * TAGLMK_FP_ONE / 4)

/*
 * Anonymous pages below which a task is not worth walking.  A reclaim pass
 * costs a traversal of the whole address space; harvesting a quarter of a
 * megabyte from it is not a trade worth making when the machine is short of
 * memory and the pass is holding a reference to the task while it works.
 */
#define TAGLMK_ZRAM_MIN_ANON	64

/*
 * Candidate slots one recompression sweep may track.  zram allocates a small
 * descriptor for every slot it decides is worth revisiting, so an unbounded
 * sweep of a full device would allocate megabytes on its way to freeing some -
 * exactly the wrong shape for something that runs because memory is short.
 * Two thousand descriptors is a few tens of kilobytes and more candidates than
 * one sweep's page budget can spend anyway.
 */
#define TAGLMK_IR_MAX_SLOTS	2048

/*
 * Huge pages below which there is nothing for a targeted sweep to aim at.  A
 * huge page is one the primary algorithm gave up on and zram stored whole, so a
 * scattering of them is normal and says nothing; a megabyte of them says a lot
 * of anonymous memory went out shallow and a second algorithm has room to work.
 */
#define TAGLMK_IR_HUGE_FLOOR	256

/* Oldest pass at [0], newest at [taglmk_nr_obs - 1]. */
static u32 taglmk_asked[TAGLMK_ZRAM_WINDOW];
static u32 taglmk_got[TAGLMK_ZRAM_WINDOW];
static unsigned int taglmk_nr_obs;

/* The fitted marginal yield, intfp32.  One page back per page asked, until a
 * pass says otherwise.
 */
static u32 taglmk_yield = TAGLMK_FP_ONE;

/*
 * Scratch for the proportional split.  Allocated statically for the same reason
 * the victim array is allocated once at init: a pass runs because memory is
 * short and must not ask for any.  Both are only ever touched from
 * taglmk_zram_share(), which the pass mutex serialises.
 */
static u32 taglmk_share_anon[TAGLMK_MAX_VICTIMS];
static u32 taglmk_share_out[TAGLMK_MAX_VICTIMS];

static unsigned long taglmk_swap_used(void)
{
	unsigned long total = total_swap_pages;
	unsigned long free = taglmk_free_swap_pages();

	return total > free ? total - free : 0;
}

/**
 * taglmk_zram_utilisation - how full swap is, as a whole percentage
 *
 * Zero when there is no swap configured at all, which reads the same as an
 * empty device and is the answer that keeps every caller below simple.
 */
unsigned int taglmk_zram_utilisation(void)
{
	unsigned long total = total_swap_pages;

	if (!total)
		return 0;

	return (unsigned int)div64_u64((u64)taglmk_swap_used() * 100, total);
}

/*
 * Pages stored per physical page of zsmalloc, as an intfp32 taper in [0, 1].
 *
 * The stored figure is swap usage rather than zram's own accounting, and the
 * footprint is every zsmalloc page in the system rather than one device's.  On
 * an Android device those are the same thing - zram over zsmalloc is the only
 * swap and the only zsmalloc user - and reading two vmstat counters costs
 * nothing, where reaching into a block device from here would mean this file
 * knowing what zram is.
 *
 * The two ways that assumption can break do not break the same way.  Another
 * zsmalloc user inflates the footprint, which deflates the ratio and tapers
 * harder than the truth warrants: conservative, and harmless.  Another swap
 * device that is not backed by zsmalloc inflates the stored count instead,
 * which flatters the ratio and can hide a taper that was deserved.  That is the
 * direction worth knowing about, and it is bounded: the term is capped at one,
 * so the worst it can do is leave the budget where it would have been with no
 * density term at all.  It can never raise it above that.
 */
static u32 taglmk_zram_density(void)
{
#if IS_ENABLED(CONFIG_ZSMALLOC)
	unsigned long stored = taglmk_swap_used();
	unsigned long zspages = global_zone_page_state(NR_ZSPAGES);
	u32 density;

	/* Nothing stored yet, or nothing to measure it against. */
	if (!stored || !zspages)
		return TAGLMK_FP_ONE;

	density = taglmk_ratio_fp(stored, zspages);
	if (density >= TAGLMK_ZRAM_DENSITY_FLOOR)
		return TAGLMK_FP_ONE;

	return taglmk_fp_div(density, TAGLMK_ZRAM_DENSITY_FLOOR);
#else
	return TAGLMK_FP_ONE;
#endif
}

/* The scalar twin of taglmk_neon_regress(); see neon.c on why they must agree. */
static void taglmk_regress_scalar(const u32 *x, const u32 *y, unsigned int n,
				  u64 *out_sx, u64 *out_sy, u64 *out_sxx,
				  u64 *out_sxy)
{
	u64 sx = 0, sy = 0, sxx = 0, sxy = 0;
	unsigned int i;

	for (i = 0; i < n; i++) {
		sx += x[i];
		sy += y[i];
		sxx += (u64)x[i] * x[i];
		sxy += (u64)x[i] * y[i];
	}

	*out_sx = sx;
	*out_sy = sy;
	*out_sxx = sxx;
	*out_sxy = sxy;
}

static void taglmk_regress(const u32 *x, const u32 *y, unsigned int n,
			   u64 *out_sx, u64 *out_sy, u64 *out_sxx, u64 *out_sxy)
{
	if (taglmk_neon_ok()) {
		taglmk_neon_regress(x, y, n, out_sx, out_sy, out_sxx, out_sxy);
		return;
	}

	taglmk_regress_scalar(x, y, n, out_sx, out_sy, out_sxx, out_sxy);
}

/*
 * The least squares slope of got against asked, in intfp32:
 *
 *	b = (n*Sxy - Sx*Sy) / (n*Sxx - Sx*Sx)
 *
 * Every sample is at most TAGLMK_ZRAM_SAMPLE_MAX and there are at most sixteen
 * of them, so Sxy and Sxx are below 2^36, both products below 2^40, and the
 * numerator still below 2^57 after being shifted up for the fixed point divide.
 *
 * The denominator is zero exactly when every pass asked for the same amount,
 * which is the steady state and not an error; a negative numerator means got
 * fell as asked rose, which a two variable fit cannot express usefully.  Both
 * fall back to the plain ratio of the totals, which is the honest answer to
 * "what fraction of what was asked came back".
 */
static u32 taglmk_fit_yield(void)
{
	u64 sx, sy, sxx, sxy;
	s64 num, den;
	s64 slope;

	if (!taglmk_nr_obs)
		return TAGLMK_FP_ONE;

	taglmk_regress(taglmk_asked, taglmk_got, taglmk_nr_obs,
		       &sx, &sy, &sxx, &sxy);

	if (!sx)
		return TAGLMK_FP_ONE;

	num = (s64)taglmk_nr_obs * (s64)sxy - (s64)sx * (s64)sy;
	den = (s64)taglmk_nr_obs * (s64)sxx - (s64)sx * (s64)sx;

	if (den <= 0 || num <= 0)
		return taglmk_ratio_fp(sy, sx);

	slope = div_s64(num << TAGLMK_FP_SHIFT, den);

	return slope > U32_MAX ? U32_MAX : (u32)slope;
}

/**
 * taglmk_zram_observe - record what a reclaim pass asked for and what it got
 * @asked: Pages the pass handed out as per task budgets.
 * @got: Pages the reclaim driver actually took.
 *
 * Called at the end of every anonymous reclaim pass, and only those: a file
 * pass produces no swap, so feeding it to a fit that models swap production
 * would be noise.  Must only be called from the pass work item.
 */
void taglmk_zram_observe(unsigned long asked, unsigned long got)
{
	/*
	 * A pass that asked for nothing - no candidates, or every candidate
	 * below the walk floor - is not an observation about the compressor.
	 * Recording it as one would drag the fit towards a slope of zero and
	 * make the next pass inflate its request for no reason.
	 */
	if (!asked)
		return;

	if (taglmk_nr_obs == TAGLMK_ZRAM_WINDOW) {
		memmove(taglmk_asked, taglmk_asked + 1,
			(TAGLMK_ZRAM_WINDOW - 1) * sizeof(*taglmk_asked));
		memmove(taglmk_got, taglmk_got + 1,
			(TAGLMK_ZRAM_WINDOW - 1) * sizeof(*taglmk_got));
		taglmk_nr_obs--;
	}

	taglmk_asked[taglmk_nr_obs] =
		(u32)min_t(unsigned long, asked, TAGLMK_ZRAM_SAMPLE_MAX);
	taglmk_got[taglmk_nr_obs] =
		(u32)min_t(unsigned long, got, TAGLMK_ZRAM_SAMPLE_MAX);
	taglmk_nr_obs++;

	WRITE_ONCE(taglmk_yield, taglmk_fit_yield());
}

/**
 * taglmk_zram_efficiency - the fitted marginal yield, intfp32
 *
 * %TAGLMK_FP_ONE means a page comes back for every page asked for.  Reported
 * through sysfs so a device evaluation can watch the balancer learn.
 */
u32 taglmk_zram_efficiency(void)
{
	return READ_ONCE(taglmk_yield);
}

/**
 * taglmk_zram_budget - how many pages this pass should ask for
 * @base: Pages the profile and the predictor between them arrived at.
 *
 * Returns zero when there is nothing to be gained from compressing more, which
 * the core reads as "spend this pass on the file LRU instead".  Otherwise the
 * three terms are applied and the result bounded at twice @base, so the
 * balancer can compensate for a poor yield without ever turning one pass into
 * an unbounded one.
 */
unsigned int taglmk_zram_budget(unsigned int base)
{
	unsigned int target = taglmk.profile->swap_target_pct;
	unsigned int util = taglmk_zram_utilisation();
	unsigned long scaled;
	u32 headroom;
	u32 gain;

	if (!base || !total_swap_pages)
		return 0;

	/* Swap is gone.  Compressing another page has nowhere to put it. */
	if (util >= 100)
		return 0;

	/*
	 * Taper from the target to full.  target < 100 is guaranteed here:
	 * util >= 100 has already returned, so reaching the divide at all means
	 * target <= util < 100.
	 */
	if (util >= target)
		headroom = taglmk_ratio_fp(100 - util, 100 - target);
	else
		headroom = TAGLMK_FP_ONE;

	headroom = taglmk_fp_mul(headroom, taglmk_zram_density());
	if (!headroom)
		return 0;

	gain = taglmk_fp_div(TAGLMK_FP_ONE, READ_ONCE(taglmk_yield));
	gain = clamp_t(u32, gain, TAGLMK_FP_ONE, TAGLMK_ZRAM_GAIN_MAX);

	scaled = taglmk_fp_int(taglmk_fp_mul(taglmk_fp(base),
					     taglmk_fp_mul(headroom, gain)));

	return (unsigned int)min(scaled, (unsigned long)base * 2);
}

/* The scalar twin of taglmk_neon_share(). */
static void taglmk_share_scalar(const u32 *anon, u32 *out, unsigned int n,
				u32 scale)
{
	unsigned int i;

	for (i = 0; i < n; i++)
		out[i] = (u32)(((u64)anon[i] * scale) >> TAGLMK_FP_SHIFT);
}

/**
 * taglmk_zram_share - split a reclaim budget across candidate tasks
 * @v: Victim array, as filled by a scan.
 * @nr: Live entries in @v.
 * @budget: Pages the pass as a whole should try to reclaim.
 *
 * Each task is asked for the same fraction of what it has resident, so the
 * pass takes proportionally from everyone rather than emptying the largest
 * task and leaving the rest untouched.  Tasks too small to be worth a walk are
 * marked to be skipped, and the budget they would have had is left to the
 * fraction rather than redistributed: over-asking here would only mean walking
 * an address space that has nothing more to give.
 *
 * Must only be called from the pass work item, which owns the scratch arrays.
 */
void taglmk_zram_share(struct taglmk_victim *v, unsigned int nr,
		       unsigned int budget)
{
	unsigned long total = 0;
	unsigned int i;
	u32 scale;

	if (!nr)
		return;

	if (WARN_ON_ONCE(nr > TAGLMK_MAX_VICTIMS))
		nr = TAGLMK_MAX_VICTIMS;

	for (i = 0; i < nr; i++) {
		if (v[i].anon_pages < TAGLMK_ZRAM_MIN_ANON) {
			v[i].skip = true;
			taglmk_share_anon[i] = 0;
			continue;
		}

		taglmk_share_anon[i] = (u32)min_t(unsigned long,
						  v[i].anon_pages, U32_MAX);
		total += taglmk_share_anon[i];
	}

	if (!total) {
		for (i = 0; i < nr; i++)
			v[i].budget = 0;
		return;
	}

	/*
	 * The fraction of every task's resident set that adds up to the budget,
	 * capped at all of it: asking for more pages than a task has cannot
	 * produce them and would only inflate what the fit is told was asked.
	 */
	scale = min(taglmk_ratio_fp(budget, total), (u32)TAGLMK_FP_ONE);

	if (taglmk_neon_ok())
		taglmk_neon_share(taglmk_share_anon, taglmk_share_out, nr,
				  scale);
	else
		taglmk_share_scalar(taglmk_share_anon, taglmk_share_out, nr,
				    scale);

	for (i = 0; i < nr; i++)
		v[i].budget = taglmk_share_out[i];
}

/**
 * taglmk_ir_depth - how hard zram should try to compress a victim's pages
 * @v: Victim about to be walked.
 * @cputime_avg: Mean cputime across the tasks in this pass, 0 if unknown.
 *
 * Compression depth is a coldness question, not an importance one.  Pages that
 * will sit in swap for minutes repay a slower algorithm; pages that are about
 * to be faulted straight back in do not, because the depth a page went out at
 * is also the depth it has to be read back through.
 *
 * Return: a depth for zram_ir_set_depth(), %ZRAM_IR_DEPTH_DEFAULT to leave the
 * decision to the zram_recomp_immediate sysctl.
 */
u8 taglmk_ir_depth(const struct taglmk_victim *v, u64 cputime_avg)
{
	switch (v->type) {
	case TAGLMK_TASK_APP:
		/*
		 * A background app is the best candidate for spending CPU on a
		 * better ratio - but a task that has been burning CPU is not
		 * idle whatever its class says, so it keeps the cheap path.
		 */
		if (cputime_avg && v->cputime > cputime_avg)
			return ZRAM_IR_DEPTH_DEFAULT;
		return ZRAM_IR_DEPTH_FULL;
	case TAGLMK_TASK_PINNED:
		/*
		 * Pinned deliberately, so the intent is that it survives and
		 * gets resumed.  Optimise for the fault back in.
		 */
		return ZRAM_IR_DEPTH_MIN;
	default:
		/* System and critical tasks: no opinion worth forcing. */
		return ZRAM_IR_DEPTH_DEFAULT;
	}
}

/*
 * The recompression rung
 * ======================
 *
 * Everything above decides how much anonymous memory to push into swap.  This
 * decides what else can be done when the answer is none: swap is as full as the
 * profile wants it, so there is no trade left to make there.
 *
 * There is still something to be had from swap itself.  zram's store path stops
 * at the first algorithm that gets an object under a page, so nearly everything
 * in swap was compressed by the cheapest one available even where a better one
 * was permitted - and on a full device the objects a second algorithm would
 * shrink further are sitting there in the tens of thousands.  Asking zram to go
 * back and try hands memory back without a page leaving anybody's working set
 * and without a process dying, which makes it the one rung on this ladder that
 * costs nobody anything.
 *
 * It does not replace the rung below it.  Ageing file pages moves the active
 * file count that vmscan and lmkd read, and recompression moves none of that;
 * what recompression buys is physical memory, which is what keeps a pass from
 * being needed again so soon and the situation from reaching the kill band at
 * all.
 *
 * It is not free.  A sweep walks slots under the per-slot lock and spends CPU
 * compressing, both of which are the wrong things to be doing if there is
 * nothing to find, so it is gated three ways: an interval, a page budget, and a
 * precondition read from zram's own statistics.  Zeroing either tunable turns
 * the rung off, which is how to find out whether it earns its keep on a
 * particular device.
 *
 * Locking: the scratch page is written only by zram, under that device's slot
 * lock, and only one sweep can be in flight because sweeps run from the pass
 * work item and the pass is serialised by taglmk.lock.  The interval is private
 * to the same work item.  Neither needs anything of its own.
 */

/*
 * Somewhere for zram to decompress into before it can try a better algorithm.
 * Allocated once at init, because a pass must never allocate: it runs because
 * memory is short, and asking the allocator for a page in order to give pages
 * back is the one shape of that request that can fail for the reason it exists.
 */
static struct page *taglmk_ir_page;

/* Earliest jiffies at which another sweep may run; zero until the first. */
static unsigned long taglmk_ir_next;

/**
 * taglmk_zram_init - claim what the recompression rung needs, once
 *
 * Return: 0, or -ENOMEM if the scratch page could not be had.
 */
int taglmk_zram_init(void)
{
	/*
	 * With no in-kernel recompression to drive there is nothing to
	 * decompress into either, so do not hold a page for it.  The whole rung
	 * folds away with the same test.
	 */
	if (!ZRAM_IR_REACHABLE)
		return 0;

	taglmk_ir_page = alloc_page(GFP_KERNEL);
	if (!taglmk_ir_page)
		return -ENOMEM;

	return 0;
}

/**
 * taglmk_zram_exit - give the scratch page back
 *
 * Safe to call whether or not taglmk_zram_init() got as far as allocating, so
 * that it can sit in the init unwind chain like the other exits.
 */
void taglmk_zram_exit(void)
{
	if (!taglmk_ir_page)
		return;

	__free_page(taglmk_ir_page);
	taglmk_ir_page = NULL;
}

/**
 * taglmk_ir_sweep - recompress part of what swap is already holding
 *
 * Runs only from the pass work item.  Cheap and silent when there is nothing to
 * do: every reason to decline is a load of a counter zram maintains anyway.
 * What it achieved goes to the counters rather than to the caller, because no
 * other rung's decision depends on it - see the call site.
 */
void taglmk_ir_sweep(void)
{
	struct zram_ir_stats before, after;
	struct zram_ir_req req = {};
	unsigned long saved;
	long ret;

	if (!taglmk_ir_page || !taglmk.ir_interval_ms || !taglmk.ir_max_pages)
		return;

	/*
	 * The interval is checked before anything else is read, so that a rung
	 * which is not due costs one comparison.  Zero means no sweep has run
	 * yet and must not be treated as a deadline, because jiffies starts
	 * below zero on some architectures and would compare as still to come.
	 */
	if (taglmk_ir_next && time_before(jiffies, taglmk_ir_next))
		return;

	/*
	 * -EAGAIN here means the device registry is momentarily contended,
	 * which is not worth waiting for: the next pass will ask again.
	 */
	if (zram_ir_get_stats(&before))
		return;

	/* Nothing stored, or nowhere deeper for it to go. */
	if (!before.stored_pages || before.nr_comps < 2)
		return;

	req.scratch = taglmk_ir_page;
	req.max_pages = taglmk.ir_max_pages;
	req.max_slots = TAGLMK_IR_MAX_SLOTS;

	/*
	 * No threshold of our own.  zram replaces an object only when the retry
	 * lands in a strictly smaller size class, which is already exactly the
	 * question "did this save memory", and it hands out its largest objects
	 * first - so the page budget spends itself where there is most to win
	 * without this file having to invent a watermark to describe that.
	 */
	req.threshold = 0;

	/*
	 * Huge pages are the largest win available, a whole page each, and the
	 * clearest evidence of a store that gave up early: the depth hint this
	 * file sets makes a store escalate rather than give up, so the pages
	 * TAGLMK swapped out are the least likely to be among them and a large
	 * count is mostly kswapd's and direct reclaim's, which carry no hint.
	 * Aim at them alone while there are enough to be worth the narrower scan.
	 */
	if (before.huge_pages >= TAGLMK_IR_HUGE_FLOOR)
		req.mode = ZRAM_IR_HUGE;

	ret = zram_ir_recompress(&req);

	/*
	 * Re-arm on every outcome, including the failures.  A sweep that found
	 * nothing, or that backed off because something else was already
	 * post-processing the device, has still spent a walk; retrying it on the
	 * next pass would spend another for the same reasons.
	 */
	taglmk_ir_next = jiffies + msecs_to_jiffies(taglmk.ir_interval_ms);

	if (ret <= 0)
		return;

	atomic_long_add(ret, &taglmk.nr_ir_slots);

	if (zram_ir_get_stats(&after) || after.compr_bytes >= before.compr_bytes)
		return;

	/*
	 * Compressed size is the only honest measure of what a sweep achieved,
	 * because zram counts a slot it looked at and decided to leave alone as
	 * a success.  Swap traffic moves the same counter, though, so credit no
	 * more than the sweep could possibly have caused: one page per slot it
	 * was allowed to touch.  Under-reporting the rung is much less
	 * misleading than letting a burst of swap-in make it look like a
	 * miracle, and the cap is also what keeps the figure inside a long.
	 */
	saved = min_t(u64, before.compr_bytes - after.compr_bytes,
		      (u64)req.max_pages << PAGE_SHIFT);

	atomic_long_add(saved, &taglmk.nr_ir_saved);
}
