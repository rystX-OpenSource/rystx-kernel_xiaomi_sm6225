
#define schedtune_cpu_boost_with(cpu, p) 0
#define schedtune_task_boost(tsk) 0

#define schedtune_prefer_idle(tsk) 0
#define schedtune_prefer_high_cap(tsk) 0

#define schedtune_enqueue_task(task, cpu) do { } while (0)
#define schedtune_dequeue_task(task, cpu) do { } while (0)
