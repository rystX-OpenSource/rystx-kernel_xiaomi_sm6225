/* SPDX-License-Identifier: GPL-2.0 */
/*
 * TAGLMK - internal shared definitions.
 *
 * Private to drivers/android/taglmk/.  The cross-subsystem interface lives in
 * <linux/taglmk.h>; this header is only shared between the driver's own
 * translation units.
 */
#ifndef _DRIVERS_ANDROID_TAGLMK_H
#define _DRIVERS_ANDROID_TAGLMK_H

#include <linux/kobject.h>
#include <linux/spinlock.h>
#include <linux/types.h>

struct task_struct;

/* ---------------------------------------------------------------------------
 * Fixed-point arithmetic.
 *
 * Samples are held in Q4.4 (s16, 4 fractional bits) which is what the NEON
 * accelerator crunches eight-wide.  Aggregate results are carried in an
 * integer/float (intfp) scalar type, Q16.16 (s32), so the reductions can be
 * combined with fractional gains without losing resolution.
 * ------------------------------------------------------------------------- */
typedef s16 q4_4_t;
typedef s32 intfp_t;

#define TAGLMK_Q44_FBITS	4
#define TAGLMK_Q44_ONE		(1 << TAGLMK_Q44_FBITS)
#define TAGLMK_Q44_MAX		((q4_4_t)0x7fff)

#define TAGLMK_FP_FBITS		16
#define TAGLMK_FP_ONE		(1 << TAGLMK_FP_FBITS)

static inline intfp_t taglmk_q44_to_fp(q4_4_t v)
{
	return (intfp_t)v << (TAGLMK_FP_FBITS - TAGLMK_Q44_FBITS);
}

static inline intfp_t taglmk_fp_mul(intfp_t a, intfp_t b)
{
	return (intfp_t)(((s64)a * b) >> TAGLMK_FP_FBITS);
}

static inline intfp_t taglmk_fp_div(intfp_t a, intfp_t b)
{
	if (!b)
		return 0;
	return (intfp_t)(((s64)a << TAGLMK_FP_FBITS) / b);
}

/* ---------------------------------------------------------------------------
 * Task classification.
 * ------------------------------------------------------------------------- */
enum taglmk_class {
	TAGLMK_CRITICAL = 0,	/* kthreads / adj < 0 core: never killed */
	TAGLMK_SYSTEM_APP,	/* persistent system apps: escalation only */
	TAGLMK_PINNED,		/* user-pinned package: strong survivability */
	TAGLMK_APP,		/* ordinary app: primary reclaim/kill target */
};

/* ---------------------------------------------------------------------------
 * Prediction (taglmk_predict.c + taglmk_neon.c).
 * ------------------------------------------------------------------------- */
#define TAGLMK_PRED_WINDOW	32

struct taglmk_predictor {
	q4_4_t		samples[TAGLMK_PRED_WINDOW];
	unsigned int	head;		/* next slot to write */
	unsigned int	count;		/* number of valid samples */
	intfp_t		ewma;		/* smoothed mean load (Q16.16) */
	intfp_t		burst;		/* burstiness / mean-abs-dev (Q16.16) */
	intfp_t		forecast;	/* short-horizon forecast (Q16.16) */
	spinlock_t	lock;
};

void taglmk_predict_init(struct taglmk_predictor *p);
void taglmk_predict_push(struct taglmk_predictor *p, q4_4_t sample);
/* Recompute ewma/burst/forecast; returns forecast (Q16.16). Kthread only. */
intfp_t taglmk_predict_eval(struct taglmk_predictor *p);

/*
 * Windowed reductions over a Q4.4 sample array.  Two implementations exist
 * that MUST return identical integer results: a NEON path and a scalar
 * intfp-32 path.  taglmk_predict.c dispatches between them.
 */
s32 taglmk_scalar_sum_q44(const q4_4_t *s, int n);
s32 taglmk_scalar_absdev_q44(const q4_4_t *s, int n, q4_4_t mean);

#ifdef CONFIG_ANDROID_TAGLMK_ARM64_NEON
bool taglmk_neon_usable(void);
s32 taglmk_neon_sum_q44(const q4_4_t *s, int n);
s32 taglmk_neon_absdev_q44(const q4_4_t *s, int n, q4_4_t mean);
#else
static inline bool taglmk_neon_usable(void) { return false; }
static inline s32 taglmk_neon_sum_q44(const q4_4_t *s, int n) { return 0; }
static inline s32 taglmk_neon_absdev_q44(const q4_4_t *s, int n, q4_4_t mean)
{
	return 0;
}
#endif

/* ---------------------------------------------------------------------------
 * Package pinning (taglmk_pin.c).
 * ------------------------------------------------------------------------- */
int taglmk_pin_init(struct kobject *parent);
void taglmk_pin_exit(void);
bool taglmk_is_pinned(struct task_struct *tsk);

/* ---------------------------------------------------------------------------
 * LRU / aging signal (taglmk_lru.c).  This is the file that diverges between
 * the classic-LRU baseline and the MGLRU adaptation.
 * ------------------------------------------------------------------------- */
/* Normalised system cache-load / aging-pressure sample, Q4.4 (0..~16). */
q4_4_t taglmk_lru_load_sample(void);
/* True when the file cache is critically small (aging pressure is extreme). */
bool taglmk_lru_file_critical(unsigned long free_file_limit);

/* ---------------------------------------------------------------------------
 * Low-RAM tuning.
 * ------------------------------------------------------------------------- */
bool taglmk_is_low_ram(void);

#endif /* _DRIVERS_ANDROID_TAGLMK_H */
