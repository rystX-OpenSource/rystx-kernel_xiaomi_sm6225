/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * FLD - Frame Load Decision GPU scheduler
 *
 * fld_math.h - Pure decision math.  This header must stay free of kernel
 * headers so the same code builds in the kernel and in the userspace unit
 * tests (drivers/gpu/msm/fld/tests/).  All loads and estimates are fixed
 * point with FLD_LOAD_SCALE == 1.0 (GPU utilization in [0, 1] and beyond,
 * values above 1.0 mean the feeder set is saturating more than one CPU
 * core worth of work).
 */

#ifndef _FLD_MATH_H
#define _FLD_MATH_H

#define FLD_LOAD_SCALE		1024u

/* Clamp util contributions so one misbehaving feeder cannot dominate. */
#define FLD_LOAD_MAX		(4u * FLD_LOAD_SCALE)

static inline unsigned int fld_clamp_u32(unsigned int v, unsigned int lo,
					unsigned int hi)
{
	if (v < lo)
		return lo;
	if (v > hi)
		return hi;
	return v;
}

/*
 * Asymmetric EWMA from the README: rise fast (up_pct, e.g. 75), decay slow
 * (down_pct, e.g. 12).  Weights are integer percentages.
 */
static inline unsigned int fld_ewma(unsigned int est, unsigned int load,
				unsigned int up_pct, unsigned int down_pct)
{
	unsigned long long delta;

	if (load > est) {
		delta = (unsigned long long)(load - est) * up_pct;
		return est + (unsigned int)(delta / 100u);
	}

	delta = (unsigned long long)(est - load) * down_pct;
	return est - (unsigned int)(delta / 100u);
}

/*
 * Feed-forward blend from the README:
 *   gpu_est = max(ewma, feeder_k * feeder_util)
 * feeder arrives already capacity-normalized (1024 == one full core).
 * feeder_k_milli is the gain in milli-units (1000 == 1.0).
 */
static inline unsigned int fld_blend(unsigned int ewma, unsigned int feeder,
				unsigned int feeder_k_milli)
{
	unsigned long long f = feeder;

	f = f * feeder_k_milli / 1000u;
	if (f > FLD_LOAD_MAX)
		f = FLD_LOAD_MAX;

	return (ewma > f) ? ewma : (unsigned int)f;
}

/*
 * Schedutil-style request from the README:
 *   target = cur_freq * gpu_est / TARGET_LOAD
 * target_load_pct is a percentage in [1, 100].
 */
static inline unsigned long long fld_target_freq(unsigned long long cur_freq,
					unsigned int gpu_est,
					unsigned int target_load_pct)
{
	if (target_load_pct == 0)
		target_load_pct = 1;

	/*
	 * target = cur_freq * (gpu_est / 1024) / (pct / 100)
	 * cur_freq (<= ~2^32) * gpu_est (<= ~2^14) * 100 stays within u64 and
	 * the single division keeps one truncation point.
	 */
	return cur_freq * gpu_est * 100ULL /
		((unsigned long long)target_load_pct * FLD_LOAD_SCALE);
}

/*
 * Snap a target frequency to the nearest entry of a descending frequency
 * table (index 0 = fastest, matching KGSL's pwrlevels[] layout), then clamp
 * the result to the allowed index range [hi_idx .. lo_idx] where lo_idx is
 * the fastest allowed level (KGSL "max_pwrlevel") and hi_idx the slowest
 * (KGSL "min_pwrlevel").  Returns the level index.
 */
static inline int fld_snap_level(const unsigned int *freqs, int n_levels,
				unsigned long long target_freq,
				int lo_idx, int hi_idx)
{
	int i, best = lo_idx;
	unsigned long long best_dist = ~0ULL;

	if (n_levels <= 0)
		return lo_idx;

	for (i = 0; i < n_levels; i++) {
		unsigned long long f = freqs[i];
		unsigned long long dist = (f > target_freq) ?
					f - target_freq : target_freq - f;

		if (dist < best_dist) {
			best_dist = dist;
			best = i;
		}
	}

	if (best < lo_idx)
		best = lo_idx;
	if (best > hi_idx)
		best = hi_idx;

	return best;
}

/*
 * Hysteresis state for the ramp machine.
 */
struct fld_ramp {
	unsigned int est;	/* EWMA state, FLD_LOAD_SCALE fixed point */
	int calm;		/* consecutive calm (target <= active) frames */
};

/*
 * One scheduling decision.  Returns the desired level index, or -1 to keep
 * the current level (ramp-down hysteresis not satisfied yet).
 *
 * load		measured GPU busy fraction this window (FLD_LOAD_SCALE)
 * feeder	capacity-normalized feeder utilization (FLD_LOAD_SCALE)
 * active_level	current level index
 */
static inline int fld_decide(struct fld_ramp *r, unsigned int load,
				unsigned int feeder,
				unsigned int up_pct, unsigned int down_pct,
				unsigned int feeder_k_milli,
				unsigned int target_load_pct,
				int down_frames,
				int active_level,
				const unsigned int *freqs, int n_levels,
				int lo_idx, int hi_idx)
{
	unsigned int gpu_est;
	unsigned long long cur_freq, target;
	int tgt;

	r->est = fld_ewma(r->est, fld_clamp_u32(load, 0, FLD_LOAD_MAX),
			up_pct, down_pct);
	gpu_est = fld_blend(r->est, feeder, feeder_k_milli);

	if (active_level < 0 || active_level >= n_levels)
		active_level = fld_clamp_u32(active_level < 0 ? lo_idx :
					hi_idx, lo_idx, hi_idx);

	cur_freq = freqs[active_level];
	target = fld_target_freq(cur_freq, gpu_est, target_load_pct);

	tgt = fld_snap_level(freqs, n_levels, target, lo_idx, hi_idx);

	/* Ramp up immediately, ramp down only after down_frames calm frames. */
	if (tgt < active_level) {
		r->calm = 0;
		return tgt;
	}

	r->calm++;
	if (r->calm >= down_frames) {
		r->calm = 0;
		return tgt > active_level ? tgt : -1;
	}

	return -1;
}

#endif /* _FLD_MATH_H */
