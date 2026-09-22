/* SPDX-License-Identifier: GPL-2.0 */
#ifndef NAP_H
#define NAP_H

#include <linux/cpuidle.h>
#include <linux/jump_label.h>
#include <linux/ktime.h>

/*
 * 4.19 backport / ARM64 rework notes
 * ==================================
 *
 * Units: this tree's cpuidle core is microsecond-based.  struct cpuidle_state
 * carries exit_latency / target_residency in US, cpuidle_device.last_residency
 * is US, and cpuidle_governor_latency_req() returns US.  The upstream patch was
 * written for a nanosecond core (target_residency_ns, PM_QOS_LATENCY_ANY_NS,
 * ...).  Everything here therefore works in US; the log2 feature space is
 * log2(microseconds), and the ordinal thresholds are seeded/clamped in that
 * same space.  Do not reintroduce an ns/us mismatch: a stray factor of 1000
 * shifts every log2 threshold by ~9.97 and silently mispredicts depth.
 *
 * Arch split (see the design note in the commit message):
 *   x86_64  ->select() may run the float NN under kernel_fpu_begin(), because
 *           x86 may_use_simd() == irq_fpu_usable() is true in the idle task
 *           even with IRQs disabled.  Forward + decision live in nap_fpu.c.
 *   arm64   ->select() runs with hard IRQs disabled, where kernel_neon_begin()
 *           BUG_ON(!may_use_simd())s and the kernel is built -mgeneral-regs-only
 *           (no scalar float either).  So inference is integer fixed-point
 *           (nap_fixp.c, GP registers only) reading a quantized shadow of the
 *           weights, and the float backprop is deferred to ->reflect() (IRQs
 *           enabled) using NEON (nap_nn_neon.c), which re-quantizes the shadow.
 */

/* ================================================================
 * Neural network dimensions
 * ================================================================
 */

#define NAP_INPUT_SIZE    8
#define NAP_HIDDEN_SIZE   8
#define NAP_NUM_CUTS      (CPUIDLE_STATE_MAX - 1)

/*
 * Master (float) weights for an 8-input MLP with an ordinal survival head.
 *
 * The trunk maps input[8] -> hidden[8] (ReLU), feeding a shared linear score
 *   s = w_out . hidden + b_out
 * which is the input to a proportional-odds ordinal head.  For each idle-state
 * boundary k the predicted survival probability that the upcoming idle reaches
 * that state's target_residency is q_k = sigmoid(s - thr_ord[k-1]).
 *
 * Column-major storage: w_h1[j][i] = weight from input j to hidden neuron i.
 *
 * __aligned(32): required by the x86 AVX2 path (vmovaps, 8 floats = one ymm).
 * NEON only needs 16-byte alignment, but 32 is a harmless superset on arm64,
 * and keeping one struct layout for both arches avoids divergence.  These
 * float weights are the learning master on both arches; on arm64 they are
 * only ever touched from ->reflect() under kernel_neon_begin().
 */
struct nap_weights {
	float w_h1[NAP_INPUT_SIZE][NAP_HIDDEN_SIZE];	/* 64 params */
	float b_h1[NAP_HIDDEN_SIZE];			/* 8 params  */
	float w_out[NAP_HIDDEN_SIZE];			/* 8 params  */
	float b_out;					/* 1 param   */
	float thr_ord[NAP_NUM_CUTS];
} __aligned(32);

#ifdef CONFIG_ARM64
/* ================================================================
 * Fixed-point (Q16.16) mirror of the weights + math, for the arm64
 * ->select() inference path.  No float, no NEON: this runs with IRQs
 * disabled at the WFI boundary.  Populated by quantizing the float
 * master after each learning step (nap_nn_neon.c).
 * ================================================================
 */

#define NAP_FX_SHIFT	16
#define NAP_FX_ONE	(1 << NAP_FX_SHIFT)

typedef s32 fx_t;

static inline fx_t fx_mul(fx_t a, fx_t b)
{
	return (fx_t)(((s64)a * (s64)b) >> NAP_FX_SHIFT);
}

static inline fx_t fx_from_int(int v)
{
	return (fx_t)v << NAP_FX_SHIFT;
}

struct nap_weights_fx {
	fx_t w_h1[NAP_INPUT_SIZE][NAP_HIDDEN_SIZE];
	fx_t b_h1[NAP_HIDDEN_SIZE];
	fx_t w_out[NAP_HIDDEN_SIZE];
	fx_t b_out;
	fx_t thr_ord[NAP_NUM_CUTS];
};
#endif /* CONFIG_ARM64 */

#ifdef CONFIG_X86_64
/* ISA-specific forward pass implementations (float, x86 only) */
void nap_nn_forward_sse2(const float *input, float *output,
			 float *hidden_save, const struct nap_weights *w);
void nap_nn_forward_avx2(const float *input, float *output,
			 float *hidden_save, const struct nap_weights *w);

/* ISA-specific online learning (backpropagation), x86 only */
struct nap_cpu_data;
void nap_nn_learn_sse2(struct nap_cpu_data *d);
void nap_nn_learn_avx2(struct nap_cpu_data *d);

/* Static key for x86 ISA dispatch (defined in nap.c) */
DECLARE_STATIC_KEY_FALSE(nap_use_avx2);
#endif /* CONFIG_X86_64 */

/* ================================================================
 * x86 SIMD type definitions and helpers (GCC vector extensions).
 *
 * Only visible when compiled with FPU/SSE flags (nap_fpu.c, nap_nn_*.c).
 * nap.c is compiled without FPU flags and must not see these definitions;
 * arm64 never sees them (the x86 files are not built there).
 * ================================================================
 */

#ifdef __SSE2__

typedef float v4sf  __attribute__((__vector_size__(16)));   /* xmm: 4xfloat  */
typedef int   v4si  __attribute__((__vector_size__(16)));   /* xmm: 4xint32  */
typedef float v8sf  __attribute__((__vector_size__(32)));   /* ymm: 8xfloat  */

#define V4SF_SET1(x)  ((v4sf){ (x), (x), (x), (x) })
#define V4SI_SET1(x)  ((v4si){ (x), (x), (x), (x) })
#define V8SF_SET1(x)  ((v8sf){ (x),(x),(x),(x),(x),(x),(x),(x) })
#define V8SF_ZERO     V8SF_SET1(0.0f)

static inline v4sf v4sf_loadu(const float *p)
{
	v4sf result;
	__builtin_memcpy(&result, p, sizeof(result));
	return result;
}

static inline void v4sf_storeu(float *p, v4sf v)
{
	__builtin_memcpy(p, &v, sizeof(v));
}

#ifdef __AVX__
static inline v8sf v8sf_loadu(const float *p)
{
	v8sf result;
	__builtin_memcpy(&result, p, sizeof(result));
	return result;
}

static inline void v8sf_storeu(float *p, v8sf v)
{
	__builtin_memcpy(p, &v, sizeof(v));
}
#endif /* __AVX__ */

static inline float fclampf(float v, float lo, float hi)
{
	if (v < lo) return lo;
	if (v > hi) return hi;
	return v;
}

static inline v4sf v4sf_clamp(v4sf v, v4sf lo, v4sf hi)
{
	return __builtin_ia32_maxps(__builtin_ia32_minps(v, hi), lo);
}

static inline v4si v4sf_as_v4si(v4sf v)
{
	union { v4sf f; v4si i; } u = { .f = v };
	return u.i;
}

static inline v4sf v4si_as_v4sf(v4si v)
{
	union { v4si i; v4sf f; } u = { .i = v };
	return u.f;
}

static inline v4sf fast_log2f_sse(v4sf x)
{
	const v4si mask_exp  = V4SI_SET1(0xFF);
	const v4si bias      = V4SI_SET1(127);
	const v4si mask_mant = V4SI_SET1(0x7FFFFF);
	const v4si exp_bias  = V4SI_SET1(127 << 23);

	v4si xi    = v4sf_as_v4si(x);
	v4si exp_i = (xi >> 23) & mask_exp;
	exp_i      = exp_i - bias;
	v4sf e     = __builtin_convertvector(exp_i, v4sf);

	v4si mant_i = (xi & mask_mant) | exp_bias;
	v4sf m      = v4si_as_v4sf(mant_i) - V4SF_SET1(1.0f);

	v4sf p;
	p = m * V4SF_SET1(0.4808f);
	p = V4SF_SET1(0.7213f) - p;
	p = m * p;
	p = V4SF_SET1(1.4425f) - p;
	p = m * p;

	return e + p;
}

#endif /* __SSE2__ */

/* ================================================================
 * Feature extraction / history sizing
 * ================================================================
 */

#define NAP_HISTORY_SIZE     8

/*
 * stop_tick threshold.  The upstream patch compared target_residency_ns
 * against a gov.h RESIDENCY_THRESHOLD_NS that does not exist in 4.19 (there is
 * no gov.h here at all).  We define it directly, in US to match the core: for
 * a chosen state deeper than this, retaining the periodic tick is pointless.
 * TICK_NSEC/US is the natural boundary (one tick period).
 */
#define NAP_RESIDENCY_THRESHOLD_US	(TICK_NSEC / NSEC_PER_USEC)

/*
 * Refresh interval for the cached minimum-valid-state lookup.  HZ jiffies (1 s)
 * bounds staleness from sysfs/runtime state-disable events; PM QoS latency
 * changes are detected immediately via the cached latency_req comparison.
 */
#define NAP_MIN_STATE_REFRESH_JIFFIES	HZ

struct nap_stats {
	u64 total_selects;
	u64 total_residency_us;	/* 4.19 backport: core residency is US */
	u64 overshoot_count;
	u64 learn_count;
};

struct nap_cpu_data {
	/* Ring buffer (measured residency, US) */
	u64   history[NAP_HISTORY_SIZE];
	float log_history[NAP_HISTORY_SIZE];
	int   hist_idx;
	int   hist_count;

	/* External signal tracking (predictions/errors in US) */
	u64     prev_idle_exit;		/* local_clock() ns timestamp */
	s64     last_predicted_us;
	s64     last_prediction_error;	/* US */

	/*
	 * Fast path: shallowest enabled C-state satisfying latency_req.  Not
	 * the upstream POLL-limit machinery (this tree has no dev->poll_limit_ns
	 * to steer, so that subsystem is dropped -- see commit message); this
	 * cache just avoids a per-idle state-table scan.
	 */
	bool short_circuited;			/* set in select, read in reflect */
	int  cached_min_state;			/* cached shallowest valid state */
	s64  cached_min_state_latency;		/* latency_req (US) when populated */
	unsigned long cached_min_state_jiffies;	/* jiffies when populated */

	/* Jiffies-based learning rate floor */
	unsigned long last_learn_jiffies;
	unsigned int  learn_jiffies_min;	/* 0 = disabled */

	/* select/reflect handoff */
	int   last_selected_idx;

	/* Shared ordinal score s (~ log2 of predicted idle duration in US). */
	float nn_output;

	/*
	 * hidden_out[], features_f32[] are written with aligned SIMD stores on
	 * x86 (movaps/vmovaps).  On arm64 they are float scratch used only by
	 * the NEON ->reflect() path.  __aligned(32) keeps the x86 aligned loads
	 * legal; harmless on arm64.
	 */
	float hidden_out[NAP_HIDDEN_SIZE] __aligned(32);
	float features_f32[NAP_INPUT_SIZE] __aligned(32);

	/* Backprop scratch */
	float learn_d_out;	/* score gradient g = sum_k (q_k - y_k) */
	float learn_lr;		/* effective learning rate */
	float learn_d_hid[NAP_HIDDEN_SIZE] __aligned(32);

	/* Precomputed per-state log2(target_residency_us). */
	float log2_tres[CPUIDLE_STATE_MAX];

	/* Decayed per-bin idle histogram: robustness-floor survival estimate */
	float bin_count[CPUIDLE_STATE_MAX];

	/* Deferred learning data */
	bool  needs_learn;
	bool  have_sample;	/* a fresh residency awaits processing */
	u64   learn_actual_us;

	/* Single network: float master weights (learning target) */
	struct nap_weights weights;
	struct nap_weights *active_w;	/* always &weights */

#ifdef CONFIG_ARM64
	/*
	 * Fixed-point shadow read by the IRQs-disabled ->select() path, plus
	 * the fixed-point inputs it consumes.  Produced by quantizing the float
	 * master in ->reflect() after init/learning.  fx_ready gates use of the
	 * NN in select(): until the first ->reflect() has initialized and
	 * quantized the weights, select() uses the integer heuristic.
	 */
	struct nap_weights_fx weights_fx;
	fx_t  features_fx[NAP_INPUT_SIZE];
	fx_t  log2_tres_fx[CPUIDLE_STATE_MAX];
	fx_t  bin_count_fx[CPUIDLE_STATE_MAX];
	bool  fx_ready;
#endif

	/* Online learning */
	unsigned int learning_rate_millths;
	unsigned int max_grad_norm_millths;
	unsigned int conf_millths;	/* decision confidence level (500 = 0.5) */
	int   learn_interval;
	int   learn_counter;
	bool  reset_pending;		/* set by sysfs/enable, consumed on init */

	/* sysfs statistics */
	struct nap_stats stats;
};

DECLARE_PER_CPU(struct nap_cpu_data, nap_data);

#ifdef CONFIG_X86_64
/* FPU entry point (nap_fpu.c) -- call only within kernel_fpu_begin/end */
int nap_fpu_select(struct cpuidle_driver *drv,
		   struct cpuidle_device *dev,
		   struct nap_cpu_data *d);
#endif

#ifdef CONFIG_ARM64
/*
 * arm64 fixed-point inference (nap_fixp.c) -- safe with IRQs disabled, no FPU.
 * Returns selected idle-state index (>= 0).  Caller guarantees d->fx_ready.
 */
int nap_fixp_select(struct cpuidle_driver *drv,
		    struct cpuidle_device *dev,
		    struct nap_cpu_data *d, s64 latency_req_us);

/*
 * arm64 NEON deferred work (nap_nn_neon.c) -- call only from ->reflect()
 * (IRQs enabled) and only within kernel_neon_begin/end.
 *   nap_neon_init: (re)initialize float weights + thresholds, quantize shadow.
 *   nap_neon_learn: run the float forward+backprop for the last idle, update
 *                   the floor histogram, and re-quantize the shadow.
 */
void nap_neon_init(struct cpuidle_driver *drv, struct nap_cpu_data *d);
void nap_neon_learn(struct cpuidle_driver *drv, struct nap_cpu_data *d);
#endif

/* Shared integer fallback heuristic (nap.c) -- no FPU, any arch. */
int nap_fallback_heuristic(struct cpuidle_driver *drv,
			   struct cpuidle_device *dev, s64 latency_req_us);

/* sysfs interface */
int  nap_sysfs_init(void);
void nap_sysfs_exit(void);

#endif /* NAP_H */
