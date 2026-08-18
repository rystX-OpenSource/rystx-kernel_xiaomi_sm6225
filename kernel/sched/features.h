/* SPDX-License-Identifier: GPL-2.0 */

#define SCHED_FEAT_ENFORCE_ELIGIBILITY 0
/*
 * Using the avg_vruntime, do the right thing and preserve lag across
 * sleep+wake cycles. EEVDF placement strategy #1, #2 if disabled.
 */
#define SCHED_FEAT_PLACE_LAG 0
/*
 * Give new tasks half a slice to ease into the competition.
 */
#define SCHED_FEAT_PLACE_DEADLINE_INITIAL 0
/*
 * Preserve relative virtual deadline on 'migration'.
 */
#define SCHED_FEAT_PLACE_REL_DEADLINE 0
/*
 * Inhibit (wakeup) preemption until the current task has either matched the
 * 0-lag point or until is has exhausted it's slice.
 */
#define SCHED_FEAT_RUN_TO_PARITY 0
/*
 * Allow wakeup of tasks with a shorter slice to cancel RESPECT_SLICE for
 * current.
 */
#define SCHED_FEAT_PREEMPT_SHORT 1
/*
 * Prefer to schedule the task we woke last (assuming it failed
 * wakeup-preemption), since its likely going to consume data we
 * touched, increases cache locality.
 */
#define SCHED_FEAT_NEXT_BUDDY 0

/*
 * Allow completely ignoring cfs_rq->next; which can be set from various
 * places:
 *   - NEXT_BUDDY (wakeup preemption)
 *   - yield_to_task()
 *   - cgroup dequeue / pick
 */
#define SCHED_FEAT_PICK_BUDDY 1

/*
 * Consider buddies to be cache hot, decreases the likeliness of a
 * cache buddy being migrated away, increases cache locality.
 */
#define SCHED_FEAT_CACHE_HOT_BUDDY 1

/*
 * Delay dequeueing tasks until they get selected or woken.
 *
 * By delaying the dequeue for non-eligible tasks, they remain in the
 * competition and can burn off their negative lag. When they get selected
 * they'll have positive lag by definition.
 *
 * DELAY_ZERO clips the lag on dequeue (or wakeup) to 0.
 */
#define SCHED_FEAT_DELAY_DEQUEUE 1
#define SCHED_FEAT_DELAY_ZERO 1

#define SCHED_FEAT_PARANOID_AVG 0

/*
 * Allow wakeup-time preemption of the current task:
 */
#define SCHED_FEAT_WAKEUP_PREEMPTION 1

#define SCHED_FEAT_HRTICK 0

/*
 * Decrement CPU capacity based on time not spent running tasks
 */
#define SCHED_FEAT_NONTASK_CAPACITY 1

/*
 * Queue remote wakeups on the target CPU and process them
 * using the scheduler IPI. Reduces rq->lock contention/bounces.
 */
#define SCHED_FEAT_TTWU_QUEUE 0

/*
 * Issue a WARN when we do multiple update_rq_clock() calls
 * in a single rq->lock section. Default disabled because the
 * annotations are not complete.
 */
#define SCHED_FEAT_WARN_DOUBLE_CLOCK 0

#if defined(CONFIG_IRQ_WORK) && defined(CONFIG_SMP)
/*
 * In order to avoid a thundering herd attack of CPUs that are
 * lowering their priorities at the same time, and there being
 * a single CPU that has an RT task that can migrate and is waiting
 * to run, where the other CPUs will try to take that CPUs
 * rq lock and possibly create a large contention, sending an
 * IPI to that CPU and let that CPU push the RT task to where
 * it should go may be a better scenario.
 */
#define SCHED_FEAT_RT_PUSH_IPI 1
#else
#define SCHED_FEAT_RT_PUSH_IPI 0
#endif

#define SCHED_FEAT_RT_RUNTIME_SHARE 0
#define SCHED_FEAT_LB_MIN 0
#define SCHED_FEAT_ATTACH_AGE_LOAD 1

/*
 * Advance detach_tasks()' scan window by rotating the examined block to the
 * head of cfs_tasks, instead of moving each rejected task there one by one.
 *
 * Both schemes give the same coverage -- a later balance pass resumes where
 * the previous one stopped -- but the per-reject move costs one list
 * operation per rejected task under rq_lock, where a block rotation costs
 * one per scan.  It also keeps the list order independent of migration
 * outcome, which the per-reject move does not.
 *
 * Turn off to get the upstream per-reject list_move() behaviour.
 */
#define SCHED_FEAT_LB_ROTATE_BLOCK 1

/*
 * Compare a migration candidate's cost against the remaining imbalance budget
 * directly, instead of relaxing the comparison by sd->nr_balance_failed.
 *
 * The upstream relaxation halves the candidate's apparent cost once per
 * recorded balance failure, so a task larger than the budget is eventually
 * let through.  nr_balance_failed is reset on *any* successful migration at
 * that domain, so a steady supply of cheap-to-move tasks keeps resetting it
 * and the relaxation never reaches the expensive ones: they are considered
 * only once nothing cheaper is left.
 *
 * That relaxation is also the only way an over-budget task ever moves under
 * migrate_load/migrate_util, since imbalanced_active_balance() escalates for
 * migrate_task only.  So dropping it alone would leave a task larger than the
 * imbalance permanently unmovable; detach_tasks() replaces the guarantee with
 * an explicit one -- a scan that admits nothing concedes to the smallest
 * overshoot it saw.  That asks the candidates actually present instead of a
 * domain counter, so no other task's success can defer it, and it acts on the
 * first scan that admits nothing rather than once enough failures have been
 * recorded.
 *
 * Turn off to get the upstream shr_bound() relaxation.
 */
#define SCHED_FEAT_LB_STRICT_BUDGET 1

#define SCHED_FEAT_WA_IDLE 1
#define SCHED_FEAT_WA_WEIGHT 1
#define SCHED_FEAT_WA_BIAS 1

/*
 * UtilEstimation. Use estimated CPU utilization.
 */
#define SCHED_FEAT_UTIL_EST 1
#define SCHED_FEAT_UTIL_EST_FASTUP 1

/*
 * Fast pre-selection of CPU candidates for EAS.
 */
#define SCHED_FEAT_FIND_BEST_TARGET 0

/*
 * Energy aware scheduling algorithm choices:
 * EAS_PREFER_IDLE
 *   Direct tasks in a schedtune.prefer_idle=1 group through
 *   the EAS path for wakeup task placement. Otherwise, put
 *   those tasks through the mainline slow path.
 */
#define SCHED_FEAT_EAS_PREFER_IDLE 1

/*
 * Request max frequency from schedutil whenever a RT task is running.
 */
#define SCHED_FEAT_SUGOV_RT_MAX_FREQ 0

/*
 * Apply schedtune boost hold to tasks of all sched classes.
 * If enabled, schedtune will hold the boost applied to a CPU
 * for 50ms regardless of task activation - if the task is
 * still running 50ms later, the boost hold expires and schedtune
 * boost will expire immediately the task stops.
 * If disabled, this behaviour will only apply to tasks of the
 * RT class.
 */
#define SCHED_FEAT_SCHEDTUNE_BOOST_HOLD_ALL 0

/*
 * Do newidle balancing proportional to its success rate using randomization.
 */
#define SCHED_FEAT_NI_RANDOM 1
#define SCHED_FEAT_NI_RATE 1

/*
 * Classify tasks into three multi-level feedback queues and take the EEVDF
 * request size from the queue instead of sysctl_sched_base_slice. A task the
 * classifier finds interactive gets a shorter request, hence an earlier
 * virtual deadline, and is picked sooner and more often for shorter turns; a
 * CPU-bound one gets a longer request and runs in fewer, longer turns. The
 * share each task receives still follows its weight, and the virtual deadline
 * order still bounds how long any of them waits.
 *
 * See kernel/sched/mlfq.h.
 */
#define SCHED_FEAT_MLFQ 1
