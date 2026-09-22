// SPDX-License-Identifier: GPL-2.0
/*
 * nap.c -- Neural Adaptive Predictor cpuidle governor
 *
 * A machine-learning-based cpuidle governor that uses a small MLP trunk and an
 * ordinal survival head to predict, per idle-state boundary, the probability
 * that the upcoming idle reaches that state's target_residency.  The decision
 * layer picks the deepest feasible state whose calibrated survival meets a
 * confidence level.  Weights are initialized from hardware idle-state
 * parameters at boot, then refined via online learning (SGD).
 *
 * IMPORTANT: This file is compiled WITHOUT FPU/SIMD flags (normal kernel
 * compilation).  Governor callbacks (nap_select, nap_reflect) run at the WFI
 * boundary; emitting FPU/SIMD here would corrupt userspace register state
 * (x86) or fault under -mgeneral-regs-only (arm64).  All floating-point code
 * lives in the arch backends:
 *   x86_64  nap_fpu.c + nap_nn_{sse2,avx2}.c (run under kernel_fpu_begin())
 *   arm64   nap_fixp.c (integer inference in ->select()) and nap_nn_neon.c
 *           (float backprop in ->reflect(), under kernel_neon_begin()).
 *
 * 4.19 backport: cpuidle residency/latency units are microseconds here (not
 * the nanoseconds the upstream 6.x patch assumed); there is no drivers/cpuidle
 * gov.h; and struct cpuidle_device has no poll_limit_ns.  See nap.h.
 */

#include <linux/cpuidle.h>
#include <linux/cpu.h>
#include <linux/device.h>
#include <linux/jiffies.h>
#include <linux/jump_label.h>
#include <linux/kobject.h>
#include <linux/math64.h>
#include <linux/percpu.h>
#include <linux/sched/clock.h>
#include <linux/sysfs.h>
#include <linux/string.h>
#include <linux/tick.h>

#ifdef CONFIG_X86_64
#include <asm/simd.h>
#include <asm/fpu/api.h>
#include <asm/processor.h>
#endif

#ifdef CONFIG_ARM64
#include <asm/simd.h>
#include <asm/neon.h>
#endif

#include "nap.h"

/**************************************************************
 * Version Information
 */

#define CPUIDLE_NAP_PROGNAME "Nap CPUIdle Governor"
#define CPUIDLE_NAP_AUTHOR   "Masahito Suzuki"
#define CPUIDLE_NAP_VERSION  "0.5.0-4.19"

/* Governor defaults */
#define NAP_DEFAULT_LR_MILLTHS    1     /* 0.001 = 1 millths */
#define NAP_DEFAULT_INTERVAL      4     /* learn every 4 reflects */
#define NAP_DEFAULT_CLAMP_MILLTHS 1000  /* 1.0 = 1000 millths */
#define NAP_DEFAULT_CONF_MILLTHS  500   /* 0.5 = balanced survival confidence */

/* ================================================================
 * ISA dispatch via static keys (x86 only; NEON is a mandatory arm64
 * baseline, so there is no arm64 detection key).
 * ================================================================
 */

#ifdef CONFIG_X86_64
DEFINE_STATIC_KEY_FALSE(nap_use_avx2);

static void __init nap_detect_simd(void)
{
	if (boot_cpu_has(X86_FEATURE_FMA) &&
	    boot_cpu_has(X86_FEATURE_AVX2)) {
		static_branch_enable(&nap_use_avx2);
		pr_info("nap: using AVX2+FMA\n");
	} else {
		pr_info("nap: using SSE2\n");
	}
}
#else
static inline void nap_detect_simd(void) { }
#endif

/* ================================================================
 * Per-CPU data
 * ================================================================
 */

DEFINE_PER_CPU(struct nap_cpu_data, nap_data);
static struct cpuidle_driver *nap_cached_drv;

/* ================================================================
 * Reflect-time updates (integer-only, no FPU needed)
 * ================================================================
 */

static void nap_history_update(struct nap_cpu_data *d, u64 measured_us)
{
	d->history[d->hist_idx] = measured_us;
	d->hist_idx = (d->hist_idx + 1) % NAP_HISTORY_SIZE;
	if (d->hist_count < NAP_HISTORY_SIZE)
		d->hist_count++;
}

static void nap_update_external_signals(struct nap_cpu_data *d)
{
	d->prev_idle_exit = local_clock();
}

/* ================================================================
 * Governor callbacks
 * ================================================================
 */

/*
 * Shared integer fallback heuristic (deepest state that fits the predicted
 * sleep length and the latency constraint).  Used on x86 when SIMD is briefly
 * unavailable, and on arm64 before the fixed-point weights are ready.
 * 4.19 backport: all comparisons in US.
 */
int nap_fallback_heuristic(struct cpuidle_driver *drv,
			   struct cpuidle_device *dev, s64 latency_req_us)
{
	ktime_t delta_tick;
	u64 sleep_length_us;
	int i;

	sleep_length_us = ktime_to_us(tick_nohz_get_sleep_length(&delta_tick));

	for (i = drv->state_count - 1; i > 0; i--) {
		if (dev->states_usage[i].disable)
			continue;
		if (drv->states[i].exit_latency > latency_req_us)
			continue;
		if (drv->states[i].target_residency > sleep_length_us)
			continue;
		return i;
	}
	return 0;
}

/*
 * Return the shallowest enabled C-state that satisfies the current latency
 * request, or 0 if none exists (POLL is the only option).  Does not consult
 * the NN.  4.19 backport: latency/exit_latency in US.
 */
static int nap_find_min_valid_state(struct cpuidle_driver *drv,
				    struct cpuidle_device *dev,
				    s64 latency_req_us)
{
	int i;

	for (i = 1; i < drv->state_count; i++) {
		if (dev->states_usage[i].disable)
			continue;
		if (drv->states[i].exit_latency > latency_req_us)
			continue;
		return i;
	}
	return 0;
}

/*
 * Cached wrapper around nap_find_min_valid_state().  Invalidated when
 * latency_req changes (immediate PM QoS propagation) or every
 * NAP_MIN_STATE_REFRESH_JIFFIES (bounded staleness for rare sysfs / runtime
 * state-disable events).
 */
static inline int nap_get_min_valid_state(struct nap_cpu_data *d,
					  struct cpuidle_driver *drv,
					  struct cpuidle_device *dev,
					  s64 latency_req_us)
{
	if (unlikely(latency_req_us != d->cached_min_state_latency ||
		     time_after(jiffies,
				d->cached_min_state_jiffies +
				NAP_MIN_STATE_REFRESH_JIFFIES))) {
		d->cached_min_state = nap_find_min_valid_state(drv, dev,
							       latency_req_us);
		d->cached_min_state_latency = latency_req_us;
		d->cached_min_state_jiffies = jiffies;
	}
	return d->cached_min_state;
}

static int nap_select(struct cpuidle_driver *drv,
		      struct cpuidle_device *dev,
		      bool *stop_tick)
{
	struct nap_cpu_data *d = this_cpu_ptr(&nap_data);
	s64 latency_req_us;
	ktime_t delta_tick;
	u64 sleep_length_us;
	int idx, min_state;

	if (unlikely(drv->state_count <= 1))
		return 0;

	latency_req_us = cpuidle_governor_latency_req(dev->cpu);
	sleep_length_us = ktime_to_us(tick_nohz_get_sleep_length(&delta_tick));
	min_state = nap_get_min_valid_state(d, drv, dev, latency_req_us);

	/*
	 * Fast path: when no C-state can amortize its target residency within
	 * the predicted sleep length, the answer is deterministically POLL.
	 * Skip NN inference entirely; nap_reflect skips the feedback path for
	 * short-circuited events.
	 *
	 * 4.19 backport: the upstream code also programmed dev->poll_limit_ns
	 * here to steer the adaptive POLL spin-timeout.  This tree's
	 * cpuidle_device has no poll_limit_ns (only a poll_time_limit bit that
	 * poll_state.c manages itself against a fixed TICK_NSEC/16), so that
	 * write is dropped -- see the commit message's scope-reduction note.
	 */
	if (min_state == 0 ||
	    sleep_length_us < drv->states[min_state].target_residency) {
		*stop_tick = false;
		d->last_selected_idx = 0;
		d->short_circuited = true;
		d->stats.total_selects++;
		return 0;
	}

	d->short_circuited = false;

#if defined(CONFIG_X86_64)
	if (likely(may_use_simd())) {
		kernel_fpu_begin();
		idx = nap_fpu_select(drv, dev, d);
		kernel_fpu_end();

		if (idx < 0)
			idx = nap_fallback_heuristic(drv, dev, latency_req_us);
	} else {
		idx = nap_fallback_heuristic(drv, dev, latency_req_us);
	}
#elif defined(CONFIG_ARM64)
	/*
	 * ->select() runs with hard IRQs disabled: kernel_neon_begin() would
	 * BUG_ON(!may_use_simd()) here.  Inference is integer fixed-point,
	 * reading the shadow weights quantized by the last ->reflect().  Until
	 * that first quantization (fx_ready), use the integer heuristic.
	 */
	if (likely(d->fx_ready))
		idx = nap_fixp_select(drv, dev, d, latency_req_us);
	else
		idx = nap_fallback_heuristic(drv, dev, latency_req_us);
#else
	idx = nap_fallback_heuristic(drv, dev, latency_req_us);
#endif

	*stop_tick = (drv->states[idx].target_residency >
		      NAP_RESIDENCY_THRESHOLD_US);

	d->last_selected_idx = idx;
	d->stats.total_selects++;

	return idx;
}

#ifdef CONFIG_ARM64
/*
 * arm64 deferred FPU work, run from ->reflect() where IRQs are enabled and
 * kernel_neon_begin() is legal.  Handles first-time / sysfs weight (re)init
 * and the online learning step, and re-quantizes the fixed-point shadow that
 * ->select() reads.
 */
static void nap_reflect_neon_work(struct cpuidle_driver *drv,
				  struct nap_cpu_data *d)
{
	if (!d->reset_pending && !d->needs_learn && !d->have_sample)
		return;

	if (unlikely(!may_use_simd()))
		return;		/* try again next reflect */

	kernel_neon_begin();
	if (unlikely(d->reset_pending))
		nap_neon_init(drv, d);
	else
		nap_neon_learn(drv, d);
	kernel_neon_end();
}
#endif

static void nap_reflect(struct cpuidle_device *dev, int index)
{
	struct nap_cpu_data *d = this_cpu_ptr(&nap_data);
	struct cpuidle_driver *drv = cpuidle_get_cpu_driver(dev);
	u64 measured_us;	/* 4.19 backport: last_residency is US */

	if (unlikely(!drv))
		return;

	measured_us = (dev->last_residency > 0) ? (u64)dev->last_residency : 0;

	/*
	 * Short-circuited POLL: the NN was not invoked for this idle, so the
	 * residency is not part of its training distribution and must not feed
	 * the floor histogram or the weight update.  Account aggregate only.
	 */
	if (d->short_circuited) {
		d->stats.total_residency_us += measured_us;
		return;
	}

	nap_history_update(d, measured_us);

	d->last_prediction_error = d->last_predicted_us - (s64)measured_us;
	nap_update_external_signals(d);

	d->learn_actual_us = measured_us;
	d->have_sample = true;

	/*
	 * Throttle the expensive weight update with a dual gate: the per-N
	 * counter AND a jiffies floor (caps learning on rapid idle bursts;
	 * learn_jiffies_min == 0 disables the time gate).
	 */
	if (++d->learn_counter >= d->learn_interval &&
	    time_after_eq(jiffies,
			  d->last_learn_jiffies + d->learn_jiffies_min)) {
		d->learn_counter = 0;
		d->last_learn_jiffies = jiffies;
		d->needs_learn = true;
	}

	d->stats.total_residency_us += measured_us;
	if (index > 0 && measured_us < drv->states[index].target_residency)
		d->stats.overshoot_count++;

#ifdef CONFIG_ARM64
	nap_reflect_neon_work(drv, d);
#endif
}

static int nap_enable(struct cpuidle_driver *drv,
		      struct cpuidle_device *dev)
{
	struct nap_cpu_data *d = per_cpu_ptr(&nap_data, dev->cpu);

	memset(d, 0, sizeof(*d));

	/*
	 * Defer weight initialization: x86 does it in the first nap_select()
	 * FPU path, arm64 in the first nap_reflect() NEON path.  Either way
	 * reset_pending drives it, so init runs on the correct CPU in a legal
	 * FPU/NEON context.
	 */
	WRITE_ONCE(nap_cached_drv, drv);
	d->learning_rate_millths  = NAP_DEFAULT_LR_MILLTHS;
	d->learn_interval = NAP_DEFAULT_INTERVAL;
	d->max_grad_norm_millths  = NAP_DEFAULT_CLAMP_MILLTHS;
	d->conf_millths = NAP_DEFAULT_CONF_MILLTHS;

	/* Force a first-call refresh of the min-valid-state cache. */
	d->cached_min_state_latency = S64_MIN;
	d->cached_min_state_jiffies = jiffies - NAP_MIN_STATE_REFRESH_JIFFIES;
	d->learn_jiffies_min = 1;

	d->reset_pending = true;

	return 0;
}

static void nap_disable(struct cpuidle_driver *drv,
			struct cpuidle_device *dev)
{
	WRITE_ONCE(nap_cached_drv, NULL);
}

/* ================================================================
 * sysfs interface  (/sys/devices/system/cpu/nap/)
 * ================================================================
 */

static ssize_t stats_show(struct kobject *kobj,
			  struct kobj_attribute *attr, char *buf)
{
	int cpu, len = 0;
	u64 total_sel = 0, total_res = 0, total_under = 0, total_learn = 0;

	for_each_online_cpu(cpu) {
		struct nap_cpu_data *d = &per_cpu(nap_data, cpu);

		total_sel   += d->stats.total_selects;
		total_res   += d->stats.total_residency_us;
		total_under += d->stats.overshoot_count;
		total_learn += d->stats.learn_count;
	}

	len += sysfs_emit_at(buf, len, "total_selects: %llu\n", total_sel);
	len += sysfs_emit_at(buf, len, "total_residency_ms: %llu\n",
			     div_u64(total_res, USEC_PER_MSEC));
	len += sysfs_emit_at(buf, len, "overshoot_count: %llu\n", total_under);
	len += sysfs_emit_at(buf, len, "overshoot_rate_permil: %llu\n",
			     total_sel ? div_u64(total_under * 1000, total_sel) : 0);
	len += sysfs_emit_at(buf, len, "learn_count: %llu\n", total_learn);
	return len;
}

static ssize_t learning_rate_show(struct kobject *kobj,
				  struct kobj_attribute *attr, char *buf)
{
	int cpu;

	cpu = cpumask_first(cpu_online_mask);
	if (cpu >= nr_cpu_ids)
		return sysfs_emit(buf, "0\n");
	return sysfs_emit(buf, "%u\n",
			  per_cpu(nap_data, cpu).learning_rate_millths);
}

static ssize_t learning_rate_store(struct kobject *kobj,
				   struct kobj_attribute *attr,
				   const char *buf, size_t count)
{
	unsigned int val;
	int cpu;

	if (kstrtouint(buf, 10, &val) || val == 0 || val > 100)
		return -EINVAL;

	for_each_online_cpu(cpu)
		per_cpu(nap_data, cpu).learning_rate_millths = val;

	return count;
}

static ssize_t learn_interval_show(struct kobject *kobj,
				   struct kobj_attribute *attr, char *buf)
{
	int cpu;

	cpu = cpumask_first(cpu_online_mask);
	if (cpu >= nr_cpu_ids)
		return sysfs_emit(buf, "0\n");
	return sysfs_emit(buf, "%d\n",
			  per_cpu(nap_data, cpu).learn_interval);
}

static ssize_t learn_interval_store(struct kobject *kobj,
				    struct kobj_attribute *attr,
				    const char *buf, size_t count)
{
	unsigned int val;
	int cpu;

	if (kstrtouint(buf, 10, &val) || val == 0 || val > 10000)
		return -EINVAL;

	for_each_online_cpu(cpu)
		per_cpu(nap_data, cpu).learn_interval = val;

	return count;
}

static ssize_t reset_weights_store(struct kobject *kobj,
				   struct kobj_attribute *attr,
				   const char *buf, size_t count)
{
	cpumask_var_t mask;
	int cpu;

	if (!READ_ONCE(nap_cached_drv))
		return -ENODEV;

	/*
	 * Set a per-CPU flag; each CPU reinitializes its own weights in its
	 * own FPU/NEON context (x86 nap_select, arm64 nap_reflect), avoiding
	 * cross-CPU races on the weight arrays.  Accepts "all" or a cpulist.
	 */
	if (sysfs_streq(buf, "all")) {
		for_each_online_cpu(cpu)
			per_cpu(nap_data, cpu).reset_pending = true;
		pr_info("nap: weight reset scheduled for all CPUs\n");
		return count;
	}

	if (!alloc_cpumask_var(&mask, GFP_KERNEL))
		return -ENOMEM;

	if (cpulist_parse(buf, mask)) {
		free_cpumask_var(mask);
		return -EINVAL;
	}

	for_each_cpu_and(cpu, mask, cpu_online_mask)
		per_cpu(nap_data, cpu).reset_pending = true;

	pr_info("nap: weight reset scheduled for CPUs %*pbl\n",
		cpumask_pr_args(mask));
	free_cpumask_var(mask);
	return count;
}

static ssize_t reset_stats_store(struct kobject *kobj,
				 struct kobj_attribute *attr,
				 const char *buf, size_t count)
{
	int cpu;

	for_each_online_cpu(cpu)
		memset(&per_cpu(nap_data, cpu).stats, 0,
		       sizeof(struct nap_stats));

	return count;
}

/*
 * confidence: decision confidence level in millths (1..999, default 500).
 * Higher demands more certainty before entering a deeper state (biases toward
 * responsiveness); lower biases toward energy (deeper).
 */
static ssize_t confidence_show(struct kobject *kobj,
			       struct kobj_attribute *attr, char *buf)
{
	int cpu;

	cpu = cpumask_first(cpu_online_mask);
	if (cpu >= nr_cpu_ids)
		return sysfs_emit(buf, "0\n");
	return sysfs_emit(buf, "%u\n",
			  per_cpu(nap_data, cpu).conf_millths);
}

static ssize_t confidence_store(struct kobject *kobj,
				struct kobj_attribute *attr,
				const char *buf, size_t count)
{
	unsigned int val;
	int cpu;

	if (kstrtouint(buf, 10, &val) || val == 0 || val >= 1000)
		return -EINVAL;

	for_each_online_cpu(cpu)
		per_cpu(nap_data, cpu).conf_millths = val;

	return count;
}

static ssize_t version_show(struct kobject *kobj,
			    struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%s\n", CPUIDLE_NAP_VERSION);
}

static ssize_t simd_show(struct kobject *kobj,
			 struct kobj_attribute *attr, char *buf)
{
#if defined(CONFIG_X86_64)
	if (static_branch_unlikely(&nap_use_avx2))
		return sysfs_emit(buf, "avx2\n");
	else
		return sysfs_emit(buf, "sse2\n");
#elif defined(CONFIG_ARM64)
	/* Fixed-point inference in select(); NEON float learning in reflect(). */
	return sysfs_emit(buf, "fixp+neon\n");
#else
	return sysfs_emit(buf, "scalar\n");
#endif
}

static struct kobj_attribute version_attr        = __ATTR_RO(version);
static struct kobj_attribute simd_attr           = __ATTR_RO(simd);
static struct kobj_attribute stats_attr          = __ATTR_RO(stats);
static struct kobj_attribute learning_rate_attr  = __ATTR_RW(learning_rate);
static struct kobj_attribute learn_interval_attr = __ATTR_RW(learn_interval);
static struct kobj_attribute confidence_attr     = __ATTR_RW(confidence);
static struct kobj_attribute reset_weights_attr  = __ATTR_WO(reset_weights);
static struct kobj_attribute reset_stats_attr    = __ATTR_WO(reset_stats);

static struct attribute *nap_attrs[] = {
	&version_attr.attr,
	&simd_attr.attr,
	&stats_attr.attr,
	&learning_rate_attr.attr,
	&learn_interval_attr.attr,
	&confidence_attr.attr,
	&reset_weights_attr.attr,
	&reset_stats_attr.attr,
	NULL,
};

static const struct attribute_group nap_attr_group = {
	.attrs = nap_attrs,
};

static struct kobject *cpuidle_kobj;

int nap_sysfs_init(void)
{
	int ret;

	/*
	 * 4.19 backport: the upstream patch used bus_get_dev_root(&cpu_subsys),
	 * a 6.x accessor absent here.  In 4.19 struct bus_type exposes dev_root
	 * directly, so use it (no get/put reference dance).
	 */
	if (!cpu_subsys.dev_root)
		return -ENODEV;

	cpuidle_kobj = kobject_create_and_add("nap", &cpu_subsys.dev_root->kobj);
	if (!cpuidle_kobj)
		return -ENOMEM;

	ret = sysfs_create_group(cpuidle_kobj, &nap_attr_group);
	if (ret) {
		kobject_put(cpuidle_kobj);
		cpuidle_kobj = NULL;
	}
	return ret;
}

void nap_sysfs_exit(void)
{
	if (cpuidle_kobj) {
		sysfs_remove_group(cpuidle_kobj, &nap_attr_group);
		kobject_put(cpuidle_kobj);
		cpuidle_kobj = NULL;
	}
}

/* ================================================================
 * Governor registration
 * ================================================================
 */

static struct cpuidle_governor nap_governor = {
	.name    = "nap",
	.rating  = 26,
	.enable  = nap_enable,
	.disable = nap_disable,
	.select  = nap_select,
	.reflect = nap_reflect,
};

static int __init nap_init(void)
{
	int ret;

	nap_detect_simd();

	ret = nap_sysfs_init();
	if (ret)
		pr_warn("nap: sysfs init failed: %d (continuing without sysfs)\n",
			ret);

	ret = cpuidle_register_governor(&nap_governor);
	if (ret) {
		pr_err("nap: register_governor failed: %d\n", ret);
		nap_sysfs_exit();
		return ret;
	}

	pr_info("%s v%s by %s registered (rating=%u)\n",
		CPUIDLE_NAP_PROGNAME, CPUIDLE_NAP_VERSION,
		CPUIDLE_NAP_AUTHOR, nap_governor.rating);
	return 0;
}
postcore_initcall(nap_init);
