# ADIOS 4.19 port: compatibility audit

ADIOS (Adaptive Deadline I/O Scheduler, v3.3.0) was written against a much
newer blk-mq than this tree provides.  This is the record of what was
checked, against which header, and what was found.  Every signature quoted
below was read out of this tree; nothing is inferred from the upstream
file's own version banner.

The port is the commit series immediately preceding this file's addition:

    3ca88630cc5c  block: add ADIOS 3.3.0 as dropped in (verbatim baseline)
    6a1a4704b20e  replace cleanup.h scope guards with explicit spin_lock_irqsave
    96819f71d7e6  move to .ops.mq and set uses_mq
    99ac10c88e0f  correct the barrier/flush notes for this tree's 4.19 blk-mq
    f31297d6ee5e  fix shallow-depth tag throttling for this tree's sbitmap API
    d3d6a2b87ef2  fix remaining 4.19 primitives and C89 conformance warnings
    da6fe830497a  wire up MQ_IOSCHED_ADIOS and export blk_stat_enable_accounting

Line numbers below refer to the file as dropped in (commit 3ca88630cc5c),
so they stay meaningful when reading the baseline diff.

## Audit table

| # | Construct in block/adios.c | Confirmed 4.19 equivalent | Verdict / risk if unverified |
|---|---|---|---|
| 1 | `scoped_guard(spinlock_irqsave, &ad->pq_lock)`, `guard(spinlock_irqsave)(...)` at 912, 983, 1076, 1227, 1269, 1782 | `include/linux/cleanup.h` does not exist in this tree; neither `guard()` nor `scoped_guard()` is defined anywhere. | Absent. Expanded by hand to `spin_lock_irqsave()`/`spin_unlock_irqrestore()`. **Leaked-spinlock risk** at the three multi-exit sites (`dispatch_from_bq`, `dispatch_from_pq`, `adios_read_priority_store`) and the two `do { guard(...) for { ... } } while` loops. Unlocking by hand would have left a deadlock on the first error path, so the control flow was re-derived per function. |
| 2 | `struct elevator_type mq_adios = { .ops = { ... } }`; callbacks `limit_depth`, `depth_updated`, `prepare_request`, `finish_request`, `next_request`, `former_request`, `insert_requests`, `completed_request` | `include/linux/elevator.h`: `ops` is still `union { struct elevator_ops sq; struct elevator_mq_ops mq; }`, plus a `bool uses_mq` on `struct elevator_type`. Every callback name used exists on `struct elevator_mq_ops`, but `.next_request`/`.former_request` live only in the `mq` arm, so the bare `.ops = {}` form does not compile. | Present but restructured. Moved under `.ops.mq` and set `.uses_mq = true`. **Silent misdispatch risk if `uses_mq` were left false**: `elv_merge()`, `elv_merged_request()`, `elv_requests_merged()`, `elv_latter_request()`, former_request, `blk_attempt_req_merge()` and `elevator_alloc()` (`eq->uses_mq`) all branch on it, and would have read the mq callbacks through the legacy single-queue layout. |
| 3 | `adios_completed_request(struct request *rq, u64 now)` at 1317 | `include/linux/elevator.h:116`: `void (*completed_request)(struct request *);` — no timestamp. `include/linux/blk-mq-sched.h:53-58` calls it with `rq` only. | `now` is absent. Sampled with `ktime_get_ns()` at callback entry. **Behavioral compromise, stated:** the sample is a few instructions later than a caller-supplied one would have been. It is taken where the block layer would have taken it, since `blk_mq_sched_completed_request()` is the first thing `__blk_mq_complete_request()` does (block/blk-mq.c:568); the drift is far below the resolution of the 64-bucket latency model. |
| 4 | `to_word_depth()` / `adios_limit_depth()` / `adios_depth_updated()` using `hctx->sched_tags->bitmap_tags`, `data->q`, `data->hctx`, `data->shallow_depth` | `struct blk_mq_alloc_data` (block/blk-mq.h) has exactly `.q`, `.flags`, `.shallow_depth`, `.ctx`, `.hctx` — all four fields used are present. `struct blk_mq_tags` (block/blk-mq-tag.h) has `struct sbitmap_queue bitmap_tags;` **embedded, not a pointer**. | Fields all present; the bitmap is by value. Needed `&...->bitmap_tags` at two sites. **Risk if missed:** compile error only, but passing the struct by value to `sbitmap_queue_min_shallow_depth()` and dereferencing `bitmap_tags->sb.shift` would have silently read the wrong memory had the layout been pointer-shaped. |
| 5 | `blk_mq_sched_try_insert_merge(q, rq)` at 886 | `block/blk-mq-sched.h:17`: `bool blk_mq_sched_try_insert_merge(struct request_queue *q, struct request *rq);` | Already correct. Expands to `rq_mergeable(rq) && elv_attempt_insert_merge(q, rq)`; the absorbed request is freed inside via `blk_attempt_req_merge()` → `__blk_put_request()` → `blk_mq_free_request()`, which calls `.finish_request()` because `RQF_ELVPRIV` is set, releasing the ADIOS `rq_data`. Same shape as `dd_insert_request()` in this tree. |
| 5b | `blk_mq_sched_try_merge(q, bio, nr_segs, &free)` at 875 | `block/blk-mq-sched.h:14`: `bool blk_mq_sched_try_merge(struct request_queue *q, struct bio *bio, struct request **merged_request);` — **3 arguments, and the out-param means "merged away, free it"** | **Mismatched.** 4-arg call would not compile, but the out-parameter's *meaning* was the real hazard: it is the request to `blk_mq_free_request()`, not a segment count or an optional. Rewritten as the 3-arg form; the `blk_mq_free_request(free)` on the merged-away request is retained. |
| 5c | `adios_bio_merge(struct request_queue *q, struct bio *bio, unsigned int nr_segs)` at 865 | `include/linux/elevator.h`: `bool (*bio_merge)(struct blk_mq_hw_ctx *, struct bio *);` — hardware queue, no seg count. `blk_mq_sched.h:33` dispatches it with `hctx`. | **Mismatched.** Retaken as `(hctx, bio)`; request queue recovered from `hctx->queue`. |
| 5d | `adios_prepare_request(struct request *rq)` at 999 | `include/linux/elevator.h`: `void (*prepare_request)(struct request *, struct bio *bio);` | **Mismatched** (extra bio param). Retaken as `(rq, bio)`; bio unused. Still the right hook for the `adios_rq_data` allocation because `blk_mq_get_request()` sets `RQF_ELVPRIV` around it (block/blk-mq.c:393) and `blk_mq_free_request()` only calls `.finish_request()` for requests carrying that flag. |
| 6 | `cmp_rq_pos(void *priv, struct list_head *a, struct list_head *b)` at 1018, used with `list_sort()` | `include/linux/list_sort.h`: `int (*cmp)(void *priv, struct list_head *a, struct list_head *b)` — no `const`. | Already correct. Deliberately **not** cast to the later const-qualified form: that form postdates this tree, and a cast would hide a real prototype divergence if one ever appeared. |
| 7 | Header comment claims about `blk_insert_flush()`, `RQF_FLUSH_SEQ`, `BLK_MQ_INSERT_AT_HEAD` | See "Flush machinery" below. | **Three of four claims wrong for this tree.** Comment rewritten (commit 99ac10c88e0f); no code change was implied by them. |
| 8 | `sbitmap_queue_min_shallow_depth(tags->bitmap_tags, 1)` at 839 | `include/linux/sbitmap.h:469` declares it; `lib/sbitmap.c` defines and `EXPORT_SYMBOL_GPL`s it. Called the same way by this tree's kyber (`&hctx->sched_tags->bitmap_tags`) and bfq. | Present. Only the `&` was needed. `min_shallow_depth = 1` cannot trip `WARN_ON_ONCE(shallow_depth < sbq->min_shallow_depth)` in `lib/sbitmap.c`, since every depth passed is ≥ 1. |
| 9 | `REQ_OP_MASK`, `op_is_sync()`, `op_is_write()`, `op_is_flush()` at 678, 691, 825, 1322 | `include/linux/blk_types.h`: `REQ_OP_BITS 8`, `REQ_OP_MASK ((1 << REQ_OP_BITS) - 1)`; `op_is_write(op) { return (op & 1); }`; `op_is_flush(op) { return op & (REQ_FUA \| REQ_PREFLUSH); }`; `op_is_sync(op)` is READ *or* `op & (REQ_SYNC \| REQ_FUA \| REQ_PREFLUSH)`. | All present with matching semantics; op number and modifier bits are already separate ranges. Recorded in the file. **Semantic trap worth knowing:** `op_is_sync()` is true for FUA/PREFLUSH writes, which is exactly why `adios_limit_depth()` must also test `op_is_write()` before exempting a request from depth limiting. |
| 10 | `timer_reduce(&ad->update_timer, ...)` at 1381 | `include/linux/timer.h:168` declares it; `kernel/time/timer.c:1140` defines and exports it. | Present with the same `(timer, expires) -> int` contract. No `mod_timer()` workaround needed, so the "only ever pull the next update earlier" intent is preserved exactly. |
| 11 | `SYSFS_INT_DECL(batch_order, ADIOS_BO_OPTYPE, !!ad->is_rotational)` at 1762 | The macro splices `max_val` into the generated `*_store()` body, where that body's own `struct adios_data *ad` local is in scope. | **Does compile, but only by the coincidence that `ad` is the local name at every expansion site** — the parameters are named like compile-time bounds but are run-time expressions. Not reproduced as-is: parameters renamed `min_expr`/`max_expr` and the behaviour documented. The `!!ad->is_rotational` bound is kept deliberately (writable upper bound 1 on rotational devices, 0 otherwise). |
| 12 | `__smp_processor_id()` at 618 | Not defined anywhere in this tree. | **Absent.** → `raw_smp_processor_id()`, which is correct here because interrupts are already disabled by the preceding `local_irq_save()`, and is the variant the rest of this tree's block layer uses. Not in the original task list; found by compiling. |
| 13 | `ad->async_depth = q->nr_requests` then `to_word_depth()` at 837/813 | `q->nr_requests` is set by `blk_mq_init_sched()` to `2 * min(tag_set->queue_depth, BLKDEV_MAX_RQ)`. | Compiles and is safe, but the throttle is **inert at its default value**: mapping `q->nr_requests` back through `to_word_depth()` yields a full sbitmap word, i.e. no per-word reduction, so async requests are not actually capped below the sched_tag depth. This is pre-existing upstream behaviour and was deliberately not changed — see "Known behaviour" below. |
| 14 | `q->node` (in `adios_init_sched`), `blk_stat_enable_accounting(q)`, `elevator_alloc(q, e)` | All present. `blk_stat_enable_accounting()` is **not exported**; `blk_stat.c` is in the `obj-y` list. | `elevator_alloc()` is `EXPORT_SYMBOL`. The unexported accounting call is a **real link blocker for `=m`** (it was already a latent one for `ssg-iosched.c`) and is fixed in commit da6fe830497a by adding the export. |

## Flush machinery (audit item 7) in detail

The dropped-in header asserted four things.  Verified line by line against
`block/blk-flush.c`, `block/blk-mq.c` and `block/blk-mq-sched.c`:

| Claim as written | Verified against this tree | Outcome |
|---|---|---|
| "`blk_insert_flush()` always strips `REQ_PREFLUSH` ... before any request ever reaches `->insert_requests()`, and always sets `REQ_SYNC`, so such a request lands in Tier 1" | `blk_insert_flush()` does strip `REQ_PREFLUSH`, conditionally `REQ_FUA`, and always sets `REQ_SYNC` (blk-flush.c:465-474). But a barrier-bearing request never reaches `->insert_requests()` at all: `op_is_flush()` is `op & (REQ_FUA\|REQ_PREFLUSH)`, `blk_mq_get_request()` skips `.prepare_request()` for it, and `blk_mq_sched_insert_request()` diverts it to `blk_insert_flush()`. | **Wrong for this tree.** There is no "flush-derived Tier 1 request". Corrected, and the real reason is now stated because it makes the `op_is_flush() \|\| !rd` guard in `adios_completed_request()` load-bearing: such requests *do* reach the completion hook (every request on an elevator queue is allocated with `BLK_MQ_REQ_INTERNAL`, so `internal_tag` is valid) but carry no `adios_rq_data`. |
| "The data portion ... is resubmitted via `BLK_MQ_INSERT_AT_HEAD` once the flush completes, which lands in Tier 0 here" | No `BLK_MQ_INSERT_AT_HEAD` symbol exists in this tree. The requeue is `blk_flush_queue_rq(rq, true)` → `blk_mq_add_to_requeue_list()` → `blk_mq_requeue_work()` → `blk_mq_insert_request()`, and the front-of-queue placement comes from `blk_mq_sched_bypass_insert()` seeing `RQF_FLUSH_SEQ` and doing `list_add()` onto `hctx->dispatch`. | **Wrong.** The data does jump ahead of everything ADIOS dispatches, but in front of the elevator, not inside `prio_queue[0]`. Corrected. |
| "The synthesized pure `REQ_OP_FLUSH` command always bypasses this elevator entirely (`blk_mq_insert_request()` routes it straight to `hctx->dispatch`)" | It does bypass, via `blk_mq_make_request()`/`blk_mq_sched_insert_request()` and `blk_mq_sched_bypass_insert()`, not via `blk_mq_insert_request()`. | **Right conclusion, wrong mechanism.** Corrected. |
| "`REQ_PREFLUSH`/`REQ_FUA` and `RQF_FLUSH_SEQ` requests are already excluded from merging" | `rq_mergeable()` rejects `REQ_OP_FLUSH`; `REQ_NOMERGE_FLAGS` here is `(REQ_NOMERGE \| REQ_PREFLUSH \| REQ_FUA)`; `blk_rq_merge_ok()` builds on `rq_mergeable()`. | **Verified correct.** No barrier-bearing request can reach the rqhash path in `merge_or_insert_to_dl_tree()` or the merge callbacks. |

Net effect for the scheduler's tiering: tier assignment only ever applies
to ordinary requests.  `at_head` requests go to `prio_queue[0]` (Tier 0),
and everything else enters the deadline tree, where a `REQ_SYNC` request
gets `rq->start_time_ns` as its deadline (Tier 1) and an asynchronous one
gets `start_time_ns` plus latency target and predicted latency (Tier 2).

## Verification performed

Build environment: `O=out`, `ARCH=arm64`, `LLVM=1 LLVM_IAS=1`,
`CROSS_COMPILE=aarch64-linux-gnu-`, config built from `stock_defconfig`
merged with `vendor/xiaomi/fog.config` + `vendor/ksu.config` and run
through `olddefconfig`, giving `CONFIG_MQ_IOSCHED_ADIOS=y` and
`CONFIG_HZ=250`.

* `block/adios.o`: 0 errors, 0 warnings.  Baseline was 20 errors / 25
  warnings.
* `block/blk-stat.o`: 0 errors, 0 warnings.
* `block/adios.ko` also links for the `=m` case, but see the caveat below.
* Every symbol `block/adios.o` references was checked to be exported, the
  only exception being `blk_stat_enable_accounting()`, which is exported by
  commit da6fe830497a.
* checkpatch (`--no-tree --terse -f`), counting findings by text, baseline
  vs. final: **none added**; 11 "Missing a blank line after declarations",
  2 "Block comments use a trailing \*/ on a separate line" and 1 "trailing
  statements should be on next line" removed, 79→78 errors and 40→27
  warnings.  The 67 remaining "open brace '{' following function
  definitions go on the next line" errors are the original file's
  consistent K&R style and were deliberately left alone, since restyling
  every function would defeat the point of landing the port as reviewable
  diffs.

### "Independently buildable" and bisectability

The two properties pull against each other for this file, and the honest
statement is:

* The **tree** builds at every commit in the series.  The Kconfig symbol
  `MQ_IOSCHED_ADIOS` does not exist until the final commit, so the
  `obj-$(CONFIG_MQ_IOSCHED_ADIOS) += adios.o` line does not exist either
  (it is added in that same final commit) and `block/adios.o` is simply not
  part of the build for commits 1-5.
* `block/adios.o` **itself** compiles cleanly only at the final commit of
  the series.  Measured per commit: 20 errors, 3, 3, 1, 0, 0.  This is
  unavoidable: the dropped-in file has eight mutually independent
  compile-blocking defects (audit items 1, 2, 3, 5b, 5c, 5d, 4/8 and 12),
  and no logical subset of them makes the file compile.  Any split that
  kept the requested per-concern reviewability therefore cannot also make
  each intermediate state of the file compile.

If file-level bisectability is preferred over per-concern reviewability,
the series squashes cleanly:
`git reset --soft 3ca88630cc5c~1 && git commit -sS`.

### Verification level for the `=m` case, stated precisely

The export of `blk_stat_enable_accounting()` is verified at source and
object level: `block/blk-stat.o` is in the `obj-y` list, the baseline source
had no export of the symbol, and after the change the object emits
`__ksymtab_blk_stat_enable_accounting` and
`__kstrtab_blk_stat_enable_accounting`.

It was **not** verified end to end.  `scripts/mod/modpost.c` only enforces
undefined-symbol resolution when a vmlinux exists (`if (have_vmlinux &&
!s->weak)`), and this machine cannot link a vmlinux (2 cores, ~2 GB RAM).
`Module.symvers` in the build directory is therefore 0 bytes, and the fact
that `make block/adios.ko` succeeds with no modpost error proves nothing
about the unresolved reference.  The evidence that the fix is correct is
the `__ksymtab` entry, not that build.

## Known behaviour, not changed by this port

* **The async depth throttle is inert at its default.**  `adios_depth_updated()`
  sets `ad->async_depth = q->nr_requests`, and `to_word_depth()` maps that
  back to a full sbitmap word, so no per-word reduction is applied and
  async requests are not capped below the sched_tag depth.  This is
  upstream's own arithmetic; the port preserves it rather than "fixing" it,
  because a different intended value is not knowable from the file.  Kyber
  by contrast derives its `async_depth` as
  `(1U << shift) * KYBER_ASYNC_PERCENT / 100U`.
* **`ad->models_stable` is never cleared**, so after the read and write
  latency models first both report a non-zero base, every subsequent
  request takes `insert_request_post_stability()` and the Tier-0
  `prio_queue[1]` path is never used again.  Left as upstream wrote it.
* The port adds no locking that upstream did not have.  `ad->lock`,
  `ad->pq_lock` and `ad->bq_lock` cover exactly the regions the scope
  guards covered; no lock was added or widened, since lock ordering between
  `bq_lock`/`pq_lock` and `lock` is observable behaviour and was not
  changed.
