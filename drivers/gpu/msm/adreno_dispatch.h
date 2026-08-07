/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * Copyright (c) 2008-2019, The Linux Foundation. All rights reserved.
 */

#ifndef ____ADRENO_DISPATCHER_H
#define ____ADRENO_DISPATCHER_H

#include <linux/kobject.h>
#include <linux/kthread.h>

extern unsigned int adreno_drawobj_timeout;

/* ------------------------------------------------------------------ */
/* Infinity GPU scheduling constants                                   */
/* ------------------------------------------------------------------ */
#define INFINITY_GPU_EMA_CLIMB_NS	10000000ULL
#define INFINITY_GPU_EMA_ALPHA		4
#define INFINITY_GPU_EMA_HALFLIFE_NS	32000000ULL
#define INFINITY_GPU_FAST_SUBMIT_NS	8000000ULL
#define INFINITY_GPU_CATCHUP_BONUS_NS	5000000ULL
#define INFINITY_GPU_PASSOVER_MAX_ENTITIES 32

/*
 * plist priority encoding for the Infinity virtual time.
 *
 * The DRM scheduler keys its runqueue on an rbtree ordered by the exact
 * u64 cached_gpu_vtime.  KGSL orders the dispatcher pending queue with a
 * plist, whose key is a plain int, so the vtime has to be projected onto
 * a bounded integer range.  The projection is a quantized delta relative
 * to the runqueue's min_gpu_vtime:
 *
 *	prio = min(( vtime - min_gpu_vtime ) / GRAIN_NS, PRIO_MAX)
 *
 * min_gpu_vtime only advances, so the delta is the part of the vtime that
 * actually discriminates between contexts; the absolute value grows
 * without bound and would saturate any fixed range.  A 250us grain over
 * 1024 buckets covers ~256ms of vtime spread, well past the point where
 * idle compensation has already pulled a returning context to the front.
 * Contexts landing in the same bucket are dispatched FIFO by the plist,
 * which matches the DRM tiebreak on oldest_job_waiting.
 */
#define INFINITY_GPU_VTIME_GRAIN_NS	250000ULL
#define INFINITY_GPU_VTIME_PRIO_MAX	1023

/* ------------------------------------------------------------------ */
/* Infinity sched-class gate (implemented in kernel/sched)             */
/* ------------------------------------------------------------------ */
bool infinity_is_interactive_candidate(struct task_struct *p);

/* ------------------------------------------------------------------ */
/* Infinity stats counters (accessible from infinity_sched.c)          */
/* ------------------------------------------------------------------ */
#include <linux/atomic.h>
extern atomic_t infinity_gpu_completion_callbacks;
extern atomic_t infinity_gpu_accounting_applied;
extern atomic_t infinity_gpu_accounting_skipped;

/*
 * Maximum size of the dispatcher ringbuffer - the actual inflight size will be
 * smaller then this but this size will allow for a larger range of inflight
 * sizes that can be chosen at runtime
 */

#define ADRENO_DISPATCH_DRAWQUEUE_SIZE 128

#define DRAWQUEUE_NEXT(_i, _s) (((_i) + 1) % (_s))

/**
 * struct adreno_dispatcher_drawqueue - List of commands for a RB level
 * @cmd_q: List of command obj's submitted to dispatcher
 * @inflight: Number of commands inflight in this q
 * @head: Head pointer to the q
 * @tail: Queues tail pointer
 * @active_context_count: Number of active contexts seen in this rb drawqueue
 * @expires: The jiffies value at which this drawqueue has run too long
 */
struct adreno_dispatcher_drawqueue {
	struct kgsl_drawobj_cmd *cmd_q[ADRENO_DISPATCH_DRAWQUEUE_SIZE];
	unsigned int inflight;
	unsigned int head;
	unsigned int tail;
	int active_context_count;
	unsigned long expires;
};

/**
 * struct adreno_dispatcher - container for the adreno GPU dispatcher
 * @mutex: Mutex to protect the structure
 * @state: Current state of the dispatcher (active or paused)
 * @timer: Timer to monitor the progress of the drawobjs
 * @inflight: Number of drawobj operations pending in the ringbuffer
 * @fault: Non-zero if a fault was detected.
 * @pending: Priority list of contexts waiting to submit drawobjs
 * @plist_lock: Spin lock to protect the pending queue
 * @work: work_struct to put the dispatcher in a work queue
 * @kobj: kobject for the dispatcher directory in the device sysfs node
 * @idle_gate: Gate to wait on for dispatcher to idle
 * @min_gpu_vtime: Minimum Infinity virtual GPU time across all contexts.
 *		   Monotonic, only advances forward on context selection.
 *		   Protected by @plist_lock.
 */
struct adreno_dispatcher {
	struct mutex mutex;
	unsigned long priv;
	struct timer_list timer;
	struct timer_list fault_timer;
	unsigned int inflight;
	atomic_t fault;
	struct plist_head pending;
	spinlock_t plist_lock;
	struct kthread_work work;
	struct kobject kobj;
	struct completion idle_gate;
	u64 min_gpu_vtime;
};

enum adreno_dispatcher_flags {
	ADRENO_DISPATCHER_POWER = 0,
	ADRENO_DISPATCHER_ACTIVE = 1,
};

struct adreno_device;
struct adreno_context;
struct kgsl_context;
struct kgsl_device;
struct kgsl_device_private;

void adreno_dispatcher_start(struct kgsl_device *device);
void adreno_dispatcher_halt(struct kgsl_device *device);
void adreno_dispatcher_unhalt(struct kgsl_device *device);
int adreno_dispatcher_init(struct adreno_device *adreno_dev);
void adreno_dispatcher_close(struct adreno_device *adreno_dev);
int adreno_dispatcher_idle(struct adreno_device *adreno_dev);
void adreno_dispatcher_irq_fault(struct adreno_device *adreno_dev);
void adreno_dispatcher_stop(struct adreno_device *adreno_dev);
void adreno_dispatcher_stop_fault_timer(struct kgsl_device *device);

struct kgsl_drawobj;

int adreno_dispatcher_queue_cmds(struct kgsl_device_private *dev_priv,
		struct kgsl_context *context, struct kgsl_drawobj *drawobj[],
		uint32_t count, uint32_t *timestamp);

void adreno_dispatcher_schedule(struct kgsl_device *device);
void adreno_dispatcher_pause(struct adreno_device *adreno_dev);
void adreno_dispatcher_queue_context(struct kgsl_device *device,
		struct adreno_context *drawctxt);
void adreno_dispatcher_preempt_callback(struct adreno_device *adreno_dev,
					int bit);
void adreno_preempt_process_dispatch_queue(struct adreno_device *adreno_dev,
	struct adreno_dispatcher_drawqueue *dispatch_q);

static inline bool adreno_drawqueue_is_empty(
		struct adreno_dispatcher_drawqueue *drawqueue)
{
	return (drawqueue != NULL && drawqueue->head == drawqueue->tail);
}
#endif /* __ADRENO_DISPATCHER_H */
