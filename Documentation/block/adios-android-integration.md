# ADIOS on Android in this tree: selection and integration notes

Companion to `Documentation/block/adios-4.19-port.md`.

## How it gets selected

Building it in is necessary but not sufficient, and the usual knob for a
"default scheduler" does not apply:

* `CONFIG_MQ_IOSCHED_ADIOS=y` (added to `vendor/xiaomi/fog.config` by the
  wiring commit) makes the elevator `elv_register()` itself at boot, so
  `adios` appears in the candidate list for every queue that can host an
  mq elevator.
* `CONFIG_DEFAULT_IOSCHED` is **not** how an mq device gets its scheduler in
  this tree.  That option is only consulted by `elevator_init()`
  (block/elevator.c:229), which only runs from the legacy
  `blk_init_allocated_queue()` path (block/blk-core.c:1193).  For an mq
  queue the init-time default comes from `elevator_init_mq()`
  (block/elevator.c:955), which picks `bfq` when `CONFIG_IOSCHED_BFQ` is on
  (it is, in fog.config) and `none` otherwise -- and only when
  `q->nr_hw_queues == 1`.  A multi-hw-queue device such as UFS is left with
  `q->elevator == NULL` until userspace chooses one.
* So the selection point is userspace, by writing the name to
  `/sys/block/<dev>/queue/scheduler`, which lands in `elv_iosched_store()`.
  `adios` is a legal value there.  To check what is available and in use:

      cat /sys/block/sda/queue/scheduler     # e.g. "[none] adios bfq ssg"
      echo adios > /sys/block/sda/queue/scheduler

  A vendor `init.rc` / `on property:` block, or an app with the right
  permissions, is the normal place to do that.  Nothing in this tree's
  drivers pins a scheduler for UFS or mmc (verified: no
  `queue/scheduler` writes under `drivers/scsi/ufs/` or `drivers/mmc/`).

## Which devices it can attach to in this tree

This matters, because the answer is not "all of them", and the ones it
cannot reach include several that Android leans on:

| Device | Queue type here | Can ADIOS attach? |
|---|---|---|
| UFS (`/dev/block/sd*`, scsi-mq) | blk-mq (`blk_mq_alloc_tag_set()` in drivers/scsi/scsi_lib.c:2397), no `BLK_MQ_F_NO_SCHED` | **Yes** -- this is the target |
| eMMC / SD (`mmcblk*`) | blk-mq with `nr_hw_queues = 1` (drivers/mmc/core/queue.c:417) | **Yes**; and it is the one case where an mq default is applied at init (`elevator_init_mq()`), which will be `bfq`, not `adios` |
| loop (`loop*`: APEX, GSI, Magisk images) | blk-mq but `tag_set.flags` includes `BLK_MQ_F_NO_SCHED` (drivers/block/loop.c:2081) | **No** -- `elv_support_iosched()` returns false, no elevator at all is offered |
| zram (`zram0`) | **legacy non-mq** (`blk_alloc_queue()` + `blk_queue_make_request()`, drivers/block/zram/zram_drv.c:2636) | **No** -- ADIOS sets `.uses_mq = true`, and `elevator_find()` matches on that, so it is not even listed |
| device-mapper (`dm-*`: dm-crypt, dm-verity, metadata encryption) | **legacy non-mq** in this tree (`blk_alloc_queue_node()` + `blk_queue_make_request()`, drivers/md/dm.c:1920) despite the `use_blk_mq` field | **No** |

Two consequences worth knowing before tuning:

* **Swap/zram traffic is untouched by ADIOS.**  On this tree zram is not an
  mq device, so ADIOS's latency model never sees compressed-swap I/O and its
  request batching does not sit in the swap path.  Under memory pressure the
  zram device is scheduled by the legacy path, not by ADIOS.
* **dm and loop I/O is untouched**, so the encrypted/verified layers and
  APEX/Magisk image I/O keep whatever behavior they had.  Only the physical
  UFS queue (and eMMC, if selected there) changes.

## Interaction with lmkd / memory-pressure behavior

The lazy claim would be "lmkd reads I/O pressure, so a batching scheduler
changes when lmkd kills".  In this tree that specific feedback path does not
exist:

* `CONFIG_PSI` is **not set** in fog.config, so there is no
  `/proc/pressure/{cpu,memory,io}` for lmkd to consult.
* `CONFIG_MEMCG` and `CONFIG_MEMCG_SWAP` are **not set** either, so the
  memcg-backed `vmpressure` notifications lmkd's older mode uses are
  unavailable too (`mm/vmpressure.o` still builds, but its event sources are
  memcg-side).

So lmkd here is running its legacy polled path against `/proc/meminfo`
(and its socket to the kernel's shrinker), and its trigger is *memory*
state, not I/O latency.  ADIOS cannot change what lmkd sees directly.

The coupling is indirect, through the latency of page-fault-driven reads
and reclaim writes, and it is real:

* ADIOS's own dispatch latency for a request that has just arrived is
  bounded by `global_latency_window`, because a batch is only grown while
  `tpl + added_lat + pred_lat <= global_latency_window` (`fill_batch_queues()`).
  Defaults are **16 ms** for non-rotational and **22 ms** rotational
  (`default_global_latency_window*`).  UFS is non-rotational, so ~16 ms is
  the figure that applies.
* One comfort: an *empty* batch queue always accepts at least the first
  request regardless of that window, because the window test in
  `fill_batch_queues()` is guarded by `count &&`.  And a refill is
  attempted whenever the active batch page is empty, without consulting
  in-flight latency at all -- the `tpl < 20% of window` test in
  `dispatch_from_bq()` only gates refills while a batch is still draining.
  So a synchronous read arriving into an idle queue is not delayed by the
  window.
* What does apply: a read arriving while a batch is still draining waits for
  the remainder of that batch.  That is the design, and it is the knob to
  reduce if fault latency under pressure turns out to matter.  Both are
  writable at runtime:

      echo 8000000 > /sys/block/sda/queue/iosched/global_latency_window
      echo 50      > /sys/block/sda/queue/iosched/bq_refill_below_ratio

## Storage-health / throughput caveats

* **The async depth throttle is inert at its default value.**  ADIOS calls
  `blk_stat_enable_accounting()` and `sbitmap_queue_min_shallow_depth()`, but
  it sets `async_depth = q->nr_requests` and then converts that back through
  `to_word_depth()`, which yields a full sbitmap word -- i.e. no per-word
  reduction.  Practically: **ADIOS does not cap async/write queue depth** as
  configured.  Kyber does (`async_depth = (1U << shift) * KYBER_ASYNC_PERCENT / 100U`),
  so do not expect kyber's async-flood protection from this port.  If that
  protection is wanted, `ad->async_depth` needs a different value; the port
  deliberately preserves upstream's arithmetic rather than guessing one.
* **Batching of writes is bounded, not eliminated.**  `batch_limit` defaults
  are 36 (read), 72 (write), 1 (discard), 1 (other), and the discard and
  other classes are effectively unbatched.  Discards also carry an 8 s
  latency target (`default_latency_target[ADIOS_DISCARD]`), so they are
  reordered aggressively relative to reads; on UFS, `BLKDISCARD` from
  `fstrim`/`vold` will be held much longer than reads.  That is the intended
  trade (discard latency traded for read latency) but is worth knowing before
  blaming a slow `fstrim` on the storage stack.
* **`RQF_STATS` accounting is on for the whole queue** while ADIOS is
  attached, since `blk_stat_enable_accounting()` is called from
  `adios_init_sched()`.  That is what makes `rq->io_start_time_ns` exist for
  the latency model at all -- without it the model would never receive a
  single valid sample -- but it does mean a `ktime_get_ns()` per issued
  request for as long as the scheduler is selected.
* **Rotational-only knobs.**  `batch_order` can only be raised to
  `ADIOS_BO_ELEVATOR` on a rotational device (the sysfs bound is
  `!!ad->is_rotational`), so the single-queue elevator sweep is
  unreachable on UFS class hardware.  The default `ADIOS_BO_OPTYPE` FIFO
  ordering is what will run.

## Recommended bring-up order

1. Build with the scheduler in (`CONFIG_MQ_IOSCHED_ADIOS=y`), which the
   wiring commit already does for fog.
2. Leave the default scheduler alone; select ADIOS per device at runtime via
   `/sys/block/<dev>/queue/scheduler` so a mis-tuned scheduler is one
   `echo` away from being undone and does not need a reflash.
3. Watch `/sys/block/sda/queue/iosched/batch_actual_max` to see whether
   batches are actually forming, and the per-optype
   `lat_model_*`/`lat_target_*` attributes to see whether the model
   converged.  `lat_model_read` prints `base: <ns>` and `slope: ns/KiB`; a
   `base` of 0 means the read model has not stabilised and the scheduler is
   still in its pre-stability FIFO mode (`prio_queue[1]`).
4. Only then consider making it the default, and if page-fault latency under
   memory pressure becomes a problem, lower `global_latency_window` before
   reaching for anything more drastic.
