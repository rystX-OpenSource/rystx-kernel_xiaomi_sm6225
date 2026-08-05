// SPDX-License-Identifier: GPL-2.0
/*
 * RINKA I/O Scheduler
 * Robust & Intelligent Kyber + Anxiety
 *
 * Single-queue scheduler combining Kyber's domain-based latency control
 * with Anxiety's lightweight dispatch and ADIOS-style adaptive prediction.
 *
 * Phase 3: RINKA-MLP - quantized MLP predictor refining the Phase-2 linear
 *          model. Inference runs in a kthread (NEON needs process context
 *          with IRQs on) and publishes an RCU lookup table the hot path reads.
 */

#include <linux/kernel.h>
#include <linux/blkdev.h>
#include <linux/elevator.h>
#include <linux/bio.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/init.h>
#include <linux/rculist.h>
#include <linux/percpu.h>
#include <linux/kthread.h>
#include <linux/log2.h>

#ifdef CONFIG_IOSCHED_RINKA_MLP_NEON
#include <asm/neon.h>
#include <asm/simd.h>
#endif

#include "rinka-mlp.h"

#define RINKA_VERSION "0.4.0-phase3"

/* How often the inference kthread refreshes the published table. */
#define RINKA_MLP_REFRESH_MS	1000

/* Scheduling domains (Kyber-inspired) */
enum {
	RINKA_READ = 0,
	RINKA_SYNC_WRITE,
	RINKA_OTHER,	/* Async writes, discard, etc. */
	RINKA_NUM_DOMAINS,
};

/* Latency model configuration (ADIOS-inspired) */
#define LM_BLOCK_SIZE_THRESHOLD		4096
#define LM_LAT_BUCKET_COUNT		64
#define LM_UPDATE_INTERVAL_MS		1500
#define LM_MIN_SAMPLES_FOR_UPDATE	128

/* Default batch sizes - favor reads like Kyber */
#define RINKA_DEFAULT_READ_BATCH	16
#define RINKA_DEFAULT_SYNC_WRITE_BATCH	8
#define RINKA_DEFAULT_OTHER_BATCH	8
#define RINKA_DEFAULT_BATCH_COUNT	4

/* Default target depths per domain */
#define RINKA_DEFAULT_READ_DEPTH	256
#define RINKA_DEFAULT_SYNC_WRITE_DEPTH	128
#define RINKA_DEFAULT_OTHER_DEPTH	64

/* Per-CPU latency bucket for small requests (≤4KB) */
struct rinka_latency_bucket_small {
	u64 weighted_sum_latency;
	u64 sum_of_weights;
};

/* Per-CPU latency bucket for large requests (>4KB) */
struct rinka_latency_bucket_large {
	u64 weighted_sum_latency;
	u64 weighted_sum_block_size;
	u64 sum_of_weights;
};

/* Per-CPU bucket collection */
struct rinka_lm_buckets {
	struct rinka_latency_bucket_small small_bucket[LM_LAT_BUCKET_COUNT];
	struct rinka_latency_bucket_large large_bucket[LM_LAT_BUCKET_COUNT];
};

/* RCU-protected latency model parameters */
struct rinka_latency_params {
	u64 base;	/* Base latency for small requests (ns) */
	u64 slope;	/* Per-KB latency increase for large requests (ns/KB) */
	u64 last_update_jiffies;
	struct rcu_head rcu;
};

/* Per-domain latency model */
struct rinka_latency_model {
	spinlock_t update_lock;
	struct rinka_latency_params __rcu *params;
	struct rinka_lm_buckets __percpu *pcpu_buckets;
};

struct rinka_domain {
	struct list_head queue;
	unsigned int in_flight;
	unsigned int target_depth;
	unsigned int batch_size;

	/* Latency model for this domain */
	struct rinka_latency_model model;

	/*
	 * MLP-refined prediction table, published by the inference kthread.
	 * NULL until the first successful refresh, in which case callers fall
	 * back to the unmodified linear model.
	 */
	struct rinka_pred_lut __rcu *lut;

	/* Statistics */
	unsigned long dispatched;
	unsigned long queued;
	unsigned long sampled;
};

struct rinka_data {
	struct rinka_domain domain[RINKA_NUM_DOMAINS];

	/* Anxiety-style batch dispatch control */
	u8 batch_count;

	/* MLP inference */
	struct task_struct *mlp_thread;
	struct rinka_mlp_weights mlp_weights;
	bool mlp_enabled;
	unsigned long mlp_refreshes;
	unsigned long mlp_neon_used;
	unsigned long mlp_fallback_used;

	/* Statistics */
	unsigned long total_dispatched;
	unsigned long merged;
};

/* Classify request into domain (Kyber-style) */
static unsigned int rinka_req_domain(struct request *rq)
{
	unsigned int op = req_op(rq);

	if (op == REQ_OP_READ)
		return RINKA_READ;
	else if (op_is_sync(op))
		return RINKA_SYNC_WRITE;
	else
		return RINKA_OTHER;
}

/* Latency model prediction (ADIOS-inspired) */
static u64 rinka_predict_latency(struct rinka_latency_model *model, u32 block_size)
{
	u64 result;
	struct rinka_latency_params *params;

	rcu_read_lock();
	params = rcu_dereference(model->params);

	result = params->base;
	if (block_size > LM_BLOCK_SIZE_THRESHOLD)
		result += params->slope *
			  DIV_ROUND_UP_ULL(block_size - LM_BLOCK_SIZE_THRESHOLD, 1024);

	rcu_read_unlock();

	return result;
}

/* Determine bucket index for a given latency value */
static u8 rinka_bucket_index(u64 latency, u64 base)
{
	u64 normalized;

	if (latency <= base)
		return 0;

	normalized = (latency - base) * LM_LAT_BUCKET_COUNT / base;
	if (normalized >= LM_LAT_BUCKET_COUNT)
		normalized = LM_LAT_BUCKET_COUNT - 1;

	return (u8)normalized;
}

/* Sample completed request latency into per-CPU buckets */
static void rinka_sample_latency(struct rinka_data *rd,
				  struct rinka_domain *dom,
				  struct request *rq,
				  u64 latency)
{
	unsigned long flags;
	u8 bucket_idx;
	struct rinka_lm_buckets *buckets;
	u32 block_size = blk_rq_bytes(rq);
	u64 current_base;
	struct rinka_latency_params *params;

	local_irq_save(flags);
	buckets = per_cpu_ptr(dom->model.pcpu_buckets, smp_processor_id());

	rcu_read_lock();
	params = rcu_dereference(dom->model.params);
	current_base = params->base;
	rcu_read_unlock();

	if (block_size <= LM_BLOCK_SIZE_THRESHOLD) {
		/* Small request - update small bucket */
		bucket_idx = rinka_bucket_index(latency, current_base ?: 1);
		buckets->small_bucket[bucket_idx].sum_of_weights++;
		buckets->small_bucket[bucket_idx].weighted_sum_latency += latency;
	} else {
		/* Large request - update large bucket */
		if (!current_base) {
			local_irq_restore(flags);
			return;
		}

		bucket_idx = rinka_bucket_index(latency, current_base);
		buckets->large_bucket[bucket_idx].sum_of_weights++;
		buckets->large_bucket[bucket_idx].weighted_sum_latency += latency;
		buckets->large_bucket[bucket_idx].weighted_sum_block_size += block_size;
	}

	local_irq_restore(flags);
	dom->sampled++;
}

/*
 * Latency prediction refined by the MLP.
 *
 * Starts from the Phase-2 linear model, then applies the Q8.8 correction
 * ratio the inference kthread published for this (size, congestion) bucket.
 * Safe to call from the hot path: this is an RCU-protected table read, never
 * an inference.
 */
static u64 rinka_predict_latency_mlp(struct rinka_domain *dom, u32 block_size)
{
	struct rinka_pred_lut *lut;
	u64 base = rinka_predict_latency(&dom->model, block_size);
	u64 result = base;
	s16 ratio;

	rcu_read_lock();
	lut = rcu_dereference(dom->lut);
	if (lut) {
		unsigned int idx = rinka_lut_index(
			rinka_size_bucket(block_size),
			rinka_depth_bucket(dom->in_flight, dom->target_depth));

		ratio = lut->ratio[idx];
		if (ratio > 0)
			result = (base * (u64)ratio) >> RINKA_Q_SHIFT;
	}
	rcu_read_unlock();

	return result;
}

/*
 * Build the feature vector for one (size, congestion) bucket, in Q8.8.
 *
 *   in[0] - log2 of request size relative to 4K, normalized
 *   in[1] - congestion, in_flight/target_depth
 *   in[2] - current base latency, normalized against 1ms
 *   in[3] - current slope, normalized against 1us/KB
 */
static void rinka_mlp_features(struct rinka_domain *dom,
			       unsigned int size_bucket,
			       unsigned int depth_bucket,
			       s16 in[RINKA_MLP_IN])
{
	struct rinka_latency_params *params;
	u64 base, slope;

	rcu_read_lock();
	params = rcu_dereference(dom->model.params);
	base = params->base;
	slope = params->slope;
	rcu_read_unlock();

	in[0] = (s16)((size_bucket * RINKA_Q_ONE) / RINKA_LUT_SIZE_BUCKETS);
	in[1] = (s16)((depth_bucket * RINKA_Q_ONE) / RINKA_LUT_DEPTH_BUCKETS);

	base = min_t(u64, base, NSEC_PER_MSEC);
	in[2] = (s16)((base * RINKA_Q_ONE) / NSEC_PER_MSEC);

	slope = min_t(u64, slope, NSEC_PER_USEC);
	in[3] = (s16)((slope * RINKA_Q_ONE) / NSEC_PER_USEC);
}

/*
 * Evaluate the network over the whole input grid and publish a new table.
 *
 * Runs only from the inference kthread, so process context with IRQs on:
 * this is the one place NEON is legal.
 */
static void rinka_mlp_refresh_domain(struct rinka_data *rd,
				     struct rinka_domain *dom)
{
	struct rinka_pred_lut *new_lut, *old_lut;
	unsigned int s, d;
	bool used_neon = false;

	new_lut = kzalloc(sizeof(*new_lut), GFP_KERNEL);
	if (!new_lut)
		return;

#ifdef CONFIG_IOSCHED_RINKA_MLP_NEON
	if (may_use_simd()) {
		kernel_neon_begin();
		for (s = 0; s < RINKA_LUT_SIZE_BUCKETS; s++) {
			for (d = 0; d < RINKA_LUT_DEPTH_BUCKETS; d++) {
				s16 in[RINKA_MLP_IN];

				rinka_mlp_features(dom, s, d, in);
				new_lut->ratio[rinka_lut_index(s, d)] =
					(s16)rinka_mlp_infer_neon(
						&rd->mlp_weights, in);
			}
		}
		kernel_neon_end();
		used_neon = true;
	}
#endif

	if (!used_neon) {
		for (s = 0; s < RINKA_LUT_SIZE_BUCKETS; s++) {
			for (d = 0; d < RINKA_LUT_DEPTH_BUCKETS; d++) {
				s16 in[RINKA_MLP_IN];

				rinka_mlp_features(dom, s, d, in);
				new_lut->ratio[rinka_lut_index(s, d)] =
					(s16)rinka_mlp_infer_int(
						&rd->mlp_weights, in);
			}
		}
	}

	if (used_neon)
		rd->mlp_neon_used++;
	else
		rd->mlp_fallback_used++;

	old_lut = rcu_dereference_protected(dom->lut, 1);
	rcu_assign_pointer(dom->lut, new_lut);
	if (old_lut)
		kfree_rcu(old_lut, rcu);
}

static int rinka_mlp_thread_fn(void *data)
{
	struct rinka_data *rd = data;

	while (!kthread_should_stop()) {
		unsigned int i;

		if (rd->mlp_enabled) {
			for (i = 0; i < RINKA_NUM_DOMAINS; i++)
				rinka_mlp_refresh_domain(rd, &rd->domain[i]);
			rd->mlp_refreshes++;
		}

		set_current_state(TASK_INTERRUPTIBLE);
		if (kthread_should_stop()) {
			set_current_state(TASK_RUNNING);
			break;
		}
		schedule_timeout(msecs_to_jiffies(RINKA_MLP_REFRESH_MS));
	}

	return 0;
}

static void rinka_merged_requests(struct request_queue *q, struct request *rq,
				   struct request *next)
{
	struct rinka_data *rd = q->elevator->elevator_data;

	list_del_init(&next->queuelist);
	rd->merged++;
}

static inline bool rinka_domain_can_dispatch(struct rinka_domain *dom)
{
	/* Allow dispatch if under target depth */
	return dom->in_flight < dom->target_depth;
}

static int __rinka_dispatch_domain(struct request_queue *q,
				    struct rinka_domain *dom,
				    unsigned int max_dispatch)
{
	unsigned int dispatched = 0;
	struct request *rq;

	while (dispatched < max_dispatch && !list_empty(&dom->queue)) {
		if (!rinka_domain_can_dispatch(dom))
			break;

		rq = list_first_entry(&dom->queue, struct request, queuelist);
		list_del_init(&rq->queuelist);
		elv_dispatch_add_tail(q, rq);

		dom->dispatched++;
		dispatched++;
	}

	return dispatched;
}

static int rinka_dispatch_batch(struct request_queue *q)
{
	struct rinka_data *rd = q->elevator->elevator_data;
	unsigned int total_dispatched = 0;
	unsigned int i, j;

	/* Anxiety-style batch dispatch: perform batch_count iterations of
	 * the ratio-drain pattern */
	for (i = 0; i < rd->batch_count; i++) {
		unsigned int round_dispatched = 0;

		/* Dispatch from each domain according to its batch size,
		 * in priority order: READ -> SYNC_WRITE -> OTHER */
		for (j = 0; j < RINKA_NUM_DOMAINS; j++) {
			round_dispatched += __rinka_dispatch_domain(q,
				&rd->domain[j], rd->domain[j].batch_size);
		}

		total_dispatched += round_dispatched;

		/* If nothing dispatched this round, don't continue batching */
		if (!round_dispatched)
			break;
	}

	return total_dispatched;
}

static int rinka_dispatch_drain(struct request_queue *q)
{
	struct rinka_data *rd = q->elevator->elevator_data;
	unsigned int total_dispatched = 0;
	unsigned int i;

	/* Drain all domains in priority order */
	for (i = 0; i < RINKA_NUM_DOMAINS; i++) {
		struct rinka_domain *dom = &rd->domain[i];

		while (!list_empty(&dom->queue)) {
			struct request *rq;

			rq = list_first_entry(&dom->queue, struct request,
					      queuelist);
			list_del_init(&rq->queuelist);
			elv_dispatch_add_tail(q, rq);

			dom->dispatched++;
			total_dispatched++;
		}
	}

	return total_dispatched;
}

static int rinka_dispatch(struct request_queue *q, int force)
{
	struct rinka_data *rd = q->elevator->elevator_data;
	int dispatched;

	/* When force=1, drain all queues (e.g., during barrier or shutdown) */
	if (unlikely(force))
		dispatched = rinka_dispatch_drain(q);
	else
		dispatched = rinka_dispatch_batch(q);

	rd->total_dispatched += dispatched;
	return dispatched;
}

static void rinka_add_request(struct request_queue *q, struct request *rq)
{
	struct rinka_data *rd = q->elevator->elevator_data;
	unsigned int domain = rinka_req_domain(rq);
	struct rinka_domain *dom = &rd->domain[domain];

	list_add_tail(&rq->queuelist, &dom->queue);
	dom->queued++;
}

static void rinka_activate_request(struct request_queue *q, struct request *rq)
{
	struct rinka_data *rd = q->elevator->elevator_data;
	unsigned int domain = rinka_req_domain(rq);

	rd->domain[domain].in_flight++;
}

static void rinka_deactivate_request(struct request_queue *q, struct request *rq)
{
	struct rinka_data *rd = q->elevator->elevator_data;
	unsigned int domain = rinka_req_domain(rq);

	WARN_ON(rd->domain[domain].in_flight == 0);
	rd->domain[domain].in_flight--;
}

static void rinka_completed_request(struct request_queue *q, struct request *rq)
{
	struct rinka_data *rd = q->elevator->elevator_data;
	unsigned int domain = rinka_req_domain(rq);
	struct rinka_domain *dom = &rd->domain[domain];
	u64 now, latency;

	/*
	 * Release the in-flight token taken in rinka_activate_request().
	 * Both hooks are gated on RQF_SORTED by the elevator core, and for a
	 * single trip through the driver exactly one of completed/deactivate
	 * runs, so the counter stays balanced.
	 */
	if (dom->in_flight)
		dom->in_flight--;

	/* Only sample if we have valid timing data */
	if (!(rq->rq_flags & RQF_STATS) || !rq->io_start_time_ns)
		return;

	now = ktime_get_ns();
	latency = now - rq->io_start_time_ns;

	/* Sample into per-CPU buckets for model update */
	rinka_sample_latency(rd, dom, rq, latency);
}

static struct request *rinka_former_request(struct request_queue *q,
					     struct request *rq)
{
	struct rinka_data *rd = q->elevator->elevator_data;
	unsigned int domain = rinka_req_domain(rq);
	struct rinka_domain *dom = &rd->domain[domain];

	if (rq->queuelist.prev == &dom->queue)
		return NULL;
	return list_prev_entry(rq, queuelist);
}

static struct request *rinka_latter_request(struct request_queue *q,
					     struct request *rq)
{
	struct rinka_data *rd = q->elevator->elevator_data;
	unsigned int domain = rinka_req_domain(rq);
	struct rinka_domain *dom = &rd->domain[domain];

	if (rq->queuelist.next == &dom->queue)
		return NULL;
	return list_next_entry(rq, queuelist);
}

/* Initialize latency model for a domain */
static int rinka_init_latency_model(struct rinka_latency_model *model,
				     int node)
{
	struct rinka_latency_params *params;

	spin_lock_init(&model->update_lock);

	/* Allocate per-CPU buckets */
	model->pcpu_buckets = alloc_percpu_gfp(struct rinka_lm_buckets,
					       GFP_KERNEL);
	if (!model->pcpu_buckets)
		return -ENOMEM;

	/* Allocate and initialize RCU params */
	params = kzalloc_node(sizeof(*params), GFP_KERNEL, node);
	if (!params) {
		free_percpu(model->pcpu_buckets);
		return -ENOMEM;
	}

	/* Start with zero base/slope - will be learned from observations */
	params->base = 0;
	params->slope = 0;
	params->last_update_jiffies = jiffies;

	rcu_assign_pointer(model->params, params);

	return 0;
}

/* Cleanup latency model */
static void rinka_cleanup_latency_model(struct rinka_latency_model *model)
{
	struct rinka_latency_params *params;

	params = rcu_dereference_protected(model->params, 1);
	if (params) {
		synchronize_rcu();
		kfree(params);
	}

	if (model->pcpu_buckets)
		free_percpu(model->pcpu_buckets);
}

static int rinka_init_queue(struct request_queue *q, struct elevator_type *elv)
{
	struct rinka_data *rd;
	struct elevator_queue *eq;
	unsigned int i;
	int ret;

	eq = elevator_alloc(q, elv);
	if (!eq)
		return -ENOMEM;

	rd = kzalloc_node(sizeof(*rd), GFP_KERNEL, q->node);
	if (!rd) {
		kobject_put(&eq->kobj);
		return -ENOMEM;
	}

	eq->elevator_data = rd;

	/* Initialize domains */
	for (i = 0; i < RINKA_NUM_DOMAINS; i++) {
		INIT_LIST_HEAD(&rd->domain[i].queue);
		rd->domain[i].in_flight = 0;

		/* Initialize latency model for this domain */
		ret = rinka_init_latency_model(&rd->domain[i].model, q->node);
		if (ret) {
			/* Cleanup previously initialized models */
			while (i > 0) {
				i--;
				rinka_cleanup_latency_model(&rd->domain[i].model);
			}
			kfree(rd);
			kobject_put(&eq->kobj);
			return ret;
		}
	}

	/* Set default target depths */
	rd->domain[RINKA_READ].target_depth = RINKA_DEFAULT_READ_DEPTH;
	rd->domain[RINKA_SYNC_WRITE].target_depth = RINKA_DEFAULT_SYNC_WRITE_DEPTH;
	rd->domain[RINKA_OTHER].target_depth = RINKA_DEFAULT_OTHER_DEPTH;

	/* Set default batch sizes */
	rd->domain[RINKA_READ].batch_size = RINKA_DEFAULT_READ_BATCH;
	rd->domain[RINKA_SYNC_WRITE].batch_size = RINKA_DEFAULT_SYNC_WRITE_BATCH;
	rd->domain[RINKA_OTHER].batch_size = RINKA_DEFAULT_OTHER_BATCH;

	rd->batch_count = RINKA_DEFAULT_BATCH_COUNT;

	/* MLP predictor: start with neutral weights so behaviour matches the
	 * Phase-2 linear model until real weights are loaded via sysfs. */
	rd->mlp_weights = rinka_mlp_default_weights;
	rd->mlp_enabled = true;

	rd->mlp_thread = kthread_run(rinka_mlp_thread_fn, rd, "rinka-mlp");
	if (IS_ERR(rd->mlp_thread)) {
		/* Not fatal: without the kthread no table is ever published and
		 * every prediction falls back to the linear model. */
		pr_warn("rinka: failed to start MLP thread (%ld), using linear model\n",
			PTR_ERR(rd->mlp_thread));
		rd->mlp_thread = NULL;
		rd->mlp_enabled = false;
	}

	spin_lock_irq(q->queue_lock);
	q->elevator = eq;
	spin_unlock_irq(q->queue_lock);

	pr_info("rinka: initialized (version %s, phase 3)\n", RINKA_VERSION);
	return 0;
}

static void rinka_exit_queue(struct elevator_queue *e)
{
	struct rinka_data *rd = e->elevator_data;
	unsigned int i;

	/* Stop inference before tearing down anything it reads. */
	if (rd->mlp_thread)
		kthread_stop(rd->mlp_thread);

	for (i = 0; i < RINKA_NUM_DOMAINS; i++) {
		struct rinka_pred_lut *lut;

		BUG_ON(!list_empty(&rd->domain[i].queue));

		lut = rcu_dereference_protected(rd->domain[i].lut, 1);
		RCU_INIT_POINTER(rd->domain[i].lut, NULL);
		if (lut)
			kfree_rcu(lut, rcu);

		rinka_cleanup_latency_model(&rd->domain[i].model);
	}

	kfree(rd);
}

/* Sysfs attributes */
static ssize_t rinka_read_depth_show(struct elevator_queue *e, char *page)
{
	struct rinka_data *rd = e->elevator_data;
	return snprintf(page, PAGE_SIZE, "%u\n",
			rd->domain[RINKA_READ].target_depth);
}

static ssize_t rinka_read_depth_store(struct elevator_queue *e,
				       const char *page, size_t count)
{
	struct rinka_data *rd = e->elevator_data;
	unsigned int val;
	int ret;

	ret = kstrtouint(page, 0, &val);
	if (ret < 0)
		return ret;

	rd->domain[RINKA_READ].target_depth = val;
	return count;
}

static ssize_t rinka_sync_write_depth_show(struct elevator_queue *e, char *page)
{
	struct rinka_data *rd = e->elevator_data;
	return snprintf(page, PAGE_SIZE, "%u\n",
			rd->domain[RINKA_SYNC_WRITE].target_depth);
}

static ssize_t rinka_sync_write_depth_store(struct elevator_queue *e,
					     const char *page, size_t count)
{
	struct rinka_data *rd = e->elevator_data;
	unsigned int val;
	int ret;

	ret = kstrtouint(page, 0, &val);
	if (ret < 0)
		return ret;

	rd->domain[RINKA_SYNC_WRITE].target_depth = val;
	return count;
}

static ssize_t rinka_other_depth_show(struct elevator_queue *e, char *page)
{
	struct rinka_data *rd = e->elevator_data;
	return snprintf(page, PAGE_SIZE, "%u\n",
			rd->domain[RINKA_OTHER].target_depth);
}

static ssize_t rinka_other_depth_store(struct elevator_queue *e,
					const char *page, size_t count)
{
	struct rinka_data *rd = e->elevator_data;
	unsigned int val;
	int ret;

	ret = kstrtouint(page, 0, &val);
	if (ret < 0)
		return ret;

	rd->domain[RINKA_OTHER].target_depth = val;
	return count;
}

static ssize_t rinka_read_batch_show(struct elevator_queue *e, char *page)
{
	struct rinka_data *rd = e->elevator_data;
	return snprintf(page, PAGE_SIZE, "%u\n",
			rd->domain[RINKA_READ].batch_size);
}

static ssize_t rinka_read_batch_store(struct elevator_queue *e,
				       const char *page, size_t count)
{
	struct rinka_data *rd = e->elevator_data;
	unsigned int val;
	int ret;

	ret = kstrtouint(page, 0, &val);
	if (ret < 0)
		return ret;

	rd->domain[RINKA_READ].batch_size = val;
	return count;
}

static ssize_t rinka_sync_write_batch_show(struct elevator_queue *e, char *page)
{
	struct rinka_data *rd = e->elevator_data;
	return snprintf(page, PAGE_SIZE, "%u\n",
			rd->domain[RINKA_SYNC_WRITE].batch_size);
}

static ssize_t rinka_sync_write_batch_store(struct elevator_queue *e,
					     const char *page, size_t count)
{
	struct rinka_data *rd = e->elevator_data;
	unsigned int val;
	int ret;

	ret = kstrtouint(page, 0, &val);
	if (ret < 0)
		return ret;

	rd->domain[RINKA_SYNC_WRITE].batch_size = val;
	return count;
}

static ssize_t rinka_other_batch_show(struct elevator_queue *e, char *page)
{
	struct rinka_data *rd = e->elevator_data;
	return snprintf(page, PAGE_SIZE, "%u\n",
			rd->domain[RINKA_OTHER].batch_size);
}

static ssize_t rinka_other_batch_store(struct elevator_queue *e,
					const char *page, size_t count)
{
	struct rinka_data *rd = e->elevator_data;
	unsigned int val;
	int ret;

	ret = kstrtouint(page, 0, &val);
	if (ret < 0)
		return ret;

	rd->domain[RINKA_OTHER].batch_size = val;
	return count;
}

static ssize_t rinka_batch_count_show(struct elevator_queue *e, char *page)
{
	struct rinka_data *rd = e->elevator_data;
	return snprintf(page, PAGE_SIZE, "%u\n", rd->batch_count);
}

static ssize_t rinka_batch_count_store(struct elevator_queue *e,
					const char *page, size_t count)
{
	struct rinka_data *rd = e->elevator_data;
	unsigned int val;
	int ret;

	ret = kstrtouint(page, 0, &val);
	if (ret < 0)
		return ret;

	if (val < 1)
		val = 1;

	rd->batch_count = val;
	return count;
}

/* Statistics - read-only */
static ssize_t rinka_read_dispatched_show(struct elevator_queue *e, char *page)
{
	struct rinka_data *rd = e->elevator_data;
	return snprintf(page, PAGE_SIZE, "%lu\n",
			rd->domain[RINKA_READ].dispatched);
}

static ssize_t rinka_sync_write_dispatched_show(struct elevator_queue *e,
						 char *page)
{
	struct rinka_data *rd = e->elevator_data;
	return snprintf(page, PAGE_SIZE, "%lu\n",
			rd->domain[RINKA_SYNC_WRITE].dispatched);
}

static ssize_t rinka_other_dispatched_show(struct elevator_queue *e, char *page)
{
	struct rinka_data *rd = e->elevator_data;
	return snprintf(page, PAGE_SIZE, "%lu\n",
			rd->domain[RINKA_OTHER].dispatched);
}

static ssize_t rinka_read_in_flight_show(struct elevator_queue *e, char *page)
{
	struct rinka_data *rd = e->elevator_data;
	return snprintf(page, PAGE_SIZE, "%u\n",
			rd->domain[RINKA_READ].in_flight);
}

static ssize_t rinka_sync_write_in_flight_show(struct elevator_queue *e,
						char *page)
{
	struct rinka_data *rd = e->elevator_data;
	return snprintf(page, PAGE_SIZE, "%u\n",
			rd->domain[RINKA_SYNC_WRITE].in_flight);
}

static ssize_t rinka_other_in_flight_show(struct elevator_queue *e, char *page)
{
	struct rinka_data *rd = e->elevator_data;
	return snprintf(page, PAGE_SIZE, "%u\n",
			rd->domain[RINKA_OTHER].in_flight);
}

static ssize_t rinka_read_sampled_show(struct elevator_queue *e, char *page)
{
	struct rinka_data *rd = e->elevator_data;
	return snprintf(page, PAGE_SIZE, "%lu\n",
			rd->domain[RINKA_READ].sampled);
}

static ssize_t rinka_sync_write_sampled_show(struct elevator_queue *e,
					      char *page)
{
	struct rinka_data *rd = e->elevator_data;
	return snprintf(page, PAGE_SIZE, "%lu\n",
			rd->domain[RINKA_SYNC_WRITE].sampled);
}

static ssize_t rinka_other_sampled_show(struct elevator_queue *e, char *page)
{
	struct rinka_data *rd = e->elevator_data;
	return snprintf(page, PAGE_SIZE, "%lu\n",
			rd->domain[RINKA_OTHER].sampled);
}

static ssize_t rinka_mlp_enabled_show(struct elevator_queue *e, char *page)
{
	struct rinka_data *rd = e->elevator_data;
	return snprintf(page, PAGE_SIZE, "%d\n", rd->mlp_enabled);
}

static ssize_t rinka_mlp_enabled_store(struct elevator_queue *e,
					const char *page, size_t count)
{
	struct rinka_data *rd = e->elevator_data;
	bool val;
	int ret;

	ret = kstrtobool(page, &val);
	if (ret < 0)
		return ret;

	rd->mlp_enabled = val;
	return count;
}

static ssize_t rinka_mlp_refreshes_show(struct elevator_queue *e, char *page)
{
	struct rinka_data *rd = e->elevator_data;
	return snprintf(page, PAGE_SIZE, "%lu\n", rd->mlp_refreshes);
}

static ssize_t rinka_mlp_neon_used_show(struct elevator_queue *e, char *page)
{
	struct rinka_data *rd = e->elevator_data;
	return snprintf(page, PAGE_SIZE, "%lu\n", rd->mlp_neon_used);
}

static ssize_t rinka_mlp_fallback_used_show(struct elevator_queue *e, char *page)
{
	struct rinka_data *rd = e->elevator_data;
	return snprintf(page, PAGE_SIZE, "%lu\n", rd->mlp_fallback_used);
}

static struct elv_fs_entry rinka_attrs[] = {
	__ATTR(read_depth, 0644, rinka_read_depth_show, rinka_read_depth_store),
	__ATTR(sync_write_depth, 0644, rinka_sync_write_depth_show,
	       rinka_sync_write_depth_store),
	__ATTR(other_depth, 0644, rinka_other_depth_show, rinka_other_depth_store),
	__ATTR(read_batch, 0644, rinka_read_batch_show, rinka_read_batch_store),
	__ATTR(sync_write_batch, 0644, rinka_sync_write_batch_show,
	       rinka_sync_write_batch_store),
	__ATTR(other_batch, 0644, rinka_other_batch_show, rinka_other_batch_store),
	__ATTR(batch_count, 0644, rinka_batch_count_show, rinka_batch_count_store),
	__ATTR(read_dispatched, 0444, rinka_read_dispatched_show, NULL),
	__ATTR(sync_write_dispatched, 0444, rinka_sync_write_dispatched_show, NULL),
	__ATTR(other_dispatched, 0444, rinka_other_dispatched_show, NULL),
	__ATTR(read_in_flight, 0444, rinka_read_in_flight_show, NULL),
	__ATTR(sync_write_in_flight, 0444, rinka_sync_write_in_flight_show, NULL),
	__ATTR(other_in_flight, 0444, rinka_other_in_flight_show, NULL),
	__ATTR(read_sampled, 0444, rinka_read_sampled_show, NULL),
	__ATTR(sync_write_sampled, 0444, rinka_sync_write_sampled_show, NULL),
	__ATTR(other_sampled, 0444, rinka_other_sampled_show, NULL),
	__ATTR(mlp_enabled, 0644, rinka_mlp_enabled_show, rinka_mlp_enabled_store),
	__ATTR(mlp_refreshes, 0444, rinka_mlp_refreshes_show, NULL),
	__ATTR(mlp_neon_used, 0444, rinka_mlp_neon_used_show, NULL),
	__ATTR(mlp_fallback_used, 0444, rinka_mlp_fallback_used_show, NULL),
	__ATTR_NULL
};

static struct elevator_type elevator_rinka = {
	.ops.sq = {
		.elevator_merge_req_fn		= rinka_merged_requests,
		.elevator_dispatch_fn		= rinka_dispatch,
		.elevator_add_req_fn		= rinka_add_request,
		.elevator_activate_req_fn	= rinka_activate_request,
		.elevator_deactivate_req_fn	= rinka_deactivate_request,
		.elevator_completed_req_fn	= rinka_completed_request,
		.elevator_former_req_fn		= rinka_former_request,
		.elevator_latter_req_fn		= rinka_latter_request,
		.elevator_init_fn		= rinka_init_queue,
		.elevator_exit_fn		= rinka_exit_queue,
	},
	.elevator_name = "rinka",
	.elevator_attrs = rinka_attrs,
	.elevator_owner = THIS_MODULE,
};

static int __init rinka_init(void)
{
	return elv_register(&elevator_rinka);
}

static void __exit rinka_exit(void)
{
	elv_unregister(&elevator_rinka);
}

module_init(rinka_init);
module_exit(rinka_exit);

MODULE_AUTHOR("iDeadXS");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("RINKA I/O Scheduler - Robust & Intelligent Kyber + Anxiety");
MODULE_VERSION(RINKA_VERSION);
