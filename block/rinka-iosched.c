// SPDX-License-Identifier: GPL-2.0
/*
 * RINKA I/O Scheduler
 * Robust & Intelligent Kyber + Anxiety
 *
 * Single-queue scheduler combining Kyber's domain-based latency control
 * with Anxiety's lightweight dispatch and ADIOS-style adaptive prediction.
 *
 * Phase 2: RINKA-Adaptive - ADIOS-style per-CPU-bucketed, RCU-published
 *          linear latency model (base + slope·size) driving depth control
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

#define RINKA_VERSION "0.3.0-phase2"

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

	/* Statistics */
	unsigned long dispatched;
	unsigned long queued;
	unsigned long sampled;
};

struct rinka_data {
	struct rinka_domain domain[RINKA_NUM_DOMAINS];

	/* Anxiety-style batch dispatch control */
	u8 batch_count;

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

static void rinka_merged_requests(struct request_queue *q, struct request *rq,
				   struct request *next)
{
	struct rinka_data *rd = q->elevator->elevator_data;

	list_del_init(&next->queuelist);
	rd->merged++;
}

static inline bool rinka_domain_can_dispatch(struct rinka_domain *dom)
{
	/* Allow dispatch if under target depth or if queue is non-empty
	 * (to avoid starvation even when over limit) */
	return dom->in_flight < dom->target_depth || !list_empty(&dom->queue);
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

	spin_lock_irq(q->queue_lock);
	q->elevator = eq;
	spin_unlock_irq(q->queue_lock);

	pr_info("rinka: initialized (version %s, phase 2)\n", RINKA_VERSION);
	return 0;
}

static void rinka_exit_queue(struct elevator_queue *e)
{
	struct rinka_data *rd = e->elevator_data;
	unsigned int i;

	for (i = 0; i < RINKA_NUM_DOMAINS; i++) {
		BUG_ON(!list_empty(&rd->domain[i].queue));
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
