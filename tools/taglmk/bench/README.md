# taglmk_bench

A single static binary that measures the two things the taglmk changelog
claims, saves what it measured, and compares a later run against an earlier
one.

It is deliberately narrow. It measures the NEON kernels the driver actually
contains, and it measures a zram device configured the way the release notes
say the bench was configured. It does not measure the driver end to end,
because a reclaim decision cannot be replayed on a phone that is not under the
memory pressure that produced it.

## Build

```sh
ANDROID_NDK=~/Android/Sdk/ndk/26.1.10909125 ./build.sh
```

`API` (default 29), `ARCH` (default `aarch64`) and `OUT` can be overridden.
The result is static, so it runs from `/data/local/tmp` on any image.

Two build rules matter to whether the numbers mean anything, and `build.sh`
enforces both:

- **One translation unit per NEON generation, no LTO.** `kernels_v0.c` and
  `kernels_v1.c` hold the two generations being compared. If the compiler
  could see both at once it might unify what they share and hoist the common
  setup out of the timing loop, and the measured difference would then be a
  property of the optimiser rather than of the code the kernel runs.
- **Plain `-O2`, no vectoriser flags.** The kernel is built at `-O2` with
  neither `-ftree-vectorize` nor `-fno-vectorize`, so the scalar twins here
  are compiled exactly the way the driver's scalar C is.

## Run

```sh
adb push taglmk_bench /data/local/tmp/
adb shell su -c 'cd /data/local/tmp && ./taglmk_bench -v -o before.txt'
# flash the new kernel, reboot
adb shell su -c 'cd /data/local/tmp && \
    ./taglmk_bench -v -o after.txt -b before.txt'
```

Every run saves. The saving is not conditional on the comparing, so a run whose
comparison turns out to be impossible is not a wasted run.

`-h` prints every option. The ones that decide what the numbers mean:

| option | default | effect |
| --- | --- | --- |
| `-o, --output FILE` | `saved.txt` | where this run is saved |
| `-b, --baseline FILE` | none | compare against a saved run |
| `-s, --suite LIST` | `all` | `cpu`, `zram`, `all`, `none` |
| `--rounds N` | 9 | measured batches per case (3..64) |
| `--duration MS` | 30 | grow each batch to at least this long |
| `--cpu N` / `--no-pin` | fastest core | which core everything runs on |
| `--seed N` | 1 | the corpus; binding on a comparison |
| `--zram-size N` | `4G` | disksize |
| `--zram-comp LIST` | `lz4kdr,zstd:3` | the ladder, in priority order |
| `--zram-pages N` | 32768 | pages written (128 MiB at 4 KiB) |
| `--zram-io-pages N` | 32 | pages per request (128 KiB) |
| `--ir-levels LIST` | `1` | `vm.zram_recomp_immediate` levels |
| `--zram-dev N` | hot-add | adopt an existing device instead |
| `--force` | off | allow adopting an initialised device |

Exit status is 0 when everything asked for ran, 1 when a suite did not finish
or the file could not be saved, and 2 for a bad command line. It reflects
whether the run completed, not what the comparison said.

## Where the changelog numbers come from

The vectorisation number is a **within-run** comparison, not a before-and-after
one. One run measures the scalar twin, the old NEON kernel (`v0`) and the new
one (`v1`) in the same process, on the same core, against the same inputs, and
prints the ladder:

```
== variants of one case, same run (positive is better) ==
  suite  case       metric           variant       value    vs ref   vs prev
  cpu    regress    cycles.min       scalar       214.50       ref         -
  cpu    regress    cycles.min       v0           118.25   +44.87%   +44.87%
  cpu    regress    cycles.min       v1            96.75   +54.90%   +18.18%
```

`vs prev` on the `v1` row is the v0 → v1 change with nothing else moved. That
is the number for:

> Reduce CPU cycle by ...%, thanks to Q4.2.1 128bit vectorization with 4-way
> pmull optimization

Read it from `cycles.min`, not `ns.min`. Wall time on a phone is a frequency
measurement as much as a work measurement.

The zram number is the same ladder over `write.mib_s` and `read.mib_s` when the
IR levels are being compared, or the `-b` comparison when two kernels are:

> Increasing ZRAM performance up to ...%

## What each case actually is

### cpu

`regress` — **v0 vs v1 is apples to apples.** The least-squares accumulation
in `zram.c` over its 16-sample window: identical work, identical loop shape,
only the multiply strategy differs. This is the case the "4-way multiply-long"
claim rests on.

`share` — **apples to apples.** The even-share budget over 128 victims: issue
width with no carried dependency.

`window2` — **apples to apples, with a caveat.** v1's reduction technique
producing v0's two outputs. This shape is not in the tree; it exists so the
reduction can be scored apart from the extra work v1 does.

`window3` — **not comparable to v0.** The real v1 window kernel, which also
computes a third sum (`weighted`, the index-weighted total) that v0 never did.
Only its scalar twin is a fair reference.

`q42`, `q44` — not timed; see below.

`window_sums` is split into `window2` and `window3` for exactly one reason: v1
computes something v0 does not, so a single "window" number would be comparing
two different amounts of work and calling the difference an optimisation.

**There is deliberately no Q4.2 cycle number.** Narrowing the advisory format
from Q4.4 to Q4.2 does not change how the factor is computed — the same two
multiplications either way — so there is no cycle count to quote for it. What
the narrowing buys is a bound, and the bound is what the `q42` case checks:
exhaustively over all 64 x 64 pairs the format can represent, the widest
product a margin and a gain can form is 4221, which fits sixteen bits. The
`q44` case records the counterexample, 69105, which does not.

Note what is *not* claimed: Q4.2 is not the widest format that fits. Q4.3 would
too, at 135 x 127 = 17145. Q4.2 was chosen because a quarter is as fine as an
advisory term read by eye needs to be, not because anything wider would
overflow.

### zram

The suite drives a real block device, not swap, so the traffic is the suite's
own and nothing else on the phone is competing for the slots.

`fill` — writing the corpus through the compressor. Publishes `write.mib_s`,
`ratio.pct`, `huge.pct`, `same.pct`, `mem.mib`, `verify`.

`read` — reading it back out. Publishes `read.mib_s`, `verify`.

`sweep.huge` — a `type=huge` recompression pass: the narrowed sweep. Publishes
`recomp.ms`, `recomp.saved_kib`, `verify`.

`sweep.all` — an unnarrowed pass on a freshly identical device. Same metrics.

`sweep.huge` runs on the state the fill left. The device is then reset and
refilled *untimed* before `sweep.all`, so the first sweep cannot change what the
second would have found. The two together are the userspace-visible form of the
recompression-sweep change: a narrowed sweep that finds the same savings in less
time is the whole point.

`verify` is a full read-and-compare against the generated corpus, run after the
fill and again after each sweep. It is not decoration: it is what stops a fast
but wrong compressor, or a lossy recompression path, from being reported as an
improvement. A case whose `verify` is 0 is named and skipped by every
comparison, never scored.

The corpus is a fixed mixture — 5% zero, 30% prose-like, 35% heap-like, 20%
structured with incompressible islands, 10% random — generated once into an
8 MiB ring and written cyclically. It is not tunable, because a compressor's
throughput depends far more on what it is fed than on how fast it is called,
and a knob there would be a knob for producing whichever number you wanted.

## Reading a comparison honestly

**The reported percentage is a lower bound on the kernel's improvement.** Each
variant is called through the same function pointer with the same per-call
overhead, and that overhead is inside both measurements. Dividing two numbers
that both include a constant understates the ratio of the parts that differ.

**A comparison can refuse to score.** Anything binding — the architecture, the
page size, the seed, the input sizes, the zram device, the traffic — that
disagrees between the two runs prints `BLOCKING` and no comparison at all. The
kernel release is deliberately *not* binding: comparing two kernels is the
point.

**A binding fact the baseline has and this run does not is not blocking** if
this run simply did not measure that suite. Running `--suite=cpu` against a
baseline that also holds zram numbers is a narrower run, not an incomparable
one.

**On a two-algorithm ladder, all three IR levels are the same depth.**
`vm.zram_recomp_immediate` sets a depth of `level + 1`, capped at the number of
active compressors, so with `lz4kdr,zstd:3` every level collapses to 2 and the
suite measures it once, saying so. `--ir-levels` only distinguishes anything
with three or more algorithms configured.

**Only the cpu suite runs at real time priority.** It asks for the bottom of
the `SCHED_FIFO` band, which keeps ordinary background work from landing in the
middle of a batch. The kernel's RT throttle answers that by stopping the task
for tens of milliseconds once a second — harmless here, because the score is
the minimum of many short batches and the best one is the one that missed the
stall. A zram pass is a single long timed region with no such escape, so the
tool drops back to normal priority before it and says so.

**Cycle counts need the PMU.** Without `perf_event_open` — not root, or
`kernel.perf_event_paranoid` too high — the run has wall time only and says so
in `cpu.counters`. The counters are all-or-nothing per batch: a batch whose
group read did not validate contributes wall time and nothing else, and
`have_pmu` is only true when every round was counted.

## Safety

The zram suite writes to sysfs and resets a block device. What it will and will
not do:

- **By default it owns its device.** It reads `zram-control/hot_add` for a fresh
  one and removes it in teardown.
- **It waits for the device node.** `hot_add` returns as soon as the disk is
  registered; on Android it is `ueventd` that creates `/dev/block/zramN`, some
  milliseconds later. The suite polls for up to two seconds, and only if the
  node never arrives — no `ueventd` at all, as in a recovery ramdisk — does it
  `mknod` one itself from `/sys/block/zramN/dev`, unlinking it again in
  teardown. It never unlinks a node it did not create.
- **It refuses a device in use as swap**, and `--force` does not override that.
  No amount of forcing makes resetting live swap safe. An unreadable
  `/proc/swaps` is treated as "in use".
- **It refuses an already-initialised adopted device** unless `--force`.
- **It clamps the fill against `MemAvailable`**, assuming nothing compresses at
  all, and warns when the clamp changes a binding fact.
- **It sets `mem_limit`** to an eighth over the fill, so an incompressible
  corpus fails the write rather than walking into the OOM killer.
- **It saves and restores `vm.zram_recomp_immediate`.**
- **`SIGINT` and `SIGTERM` set a flag** checked between requests; teardown is
  always on the main path. `SIGKILL` is the one signal that leaks a scratch
  device, which is why the device name is printed as soon as it exists.

## Saved file format

Plain text, one record per line, stable across versions by refusing to load a
version it does not understand.

```
taglmk-bench 1
M kernel 4.19.325-rystx
M zram.comp lz4kdr,zstd:3
R cpu regress v1 cycles.min - cyc 96.75
```

`M` is metadata, `R` is a result: suite, case, variant, metric, direction
(`+` higher is better, `-` lower is better), unit, value. The parser is strict —
a malformed line rejects the whole file rather than loading half of it — because
a partly loaded baseline would compare against rows that are not there.

A run that did not finish everything it was asked to is still saved, marked
`status partial`, and a comparison against it says so.

## Not yet compiled

These sources have been verified by inspection and by mechanical checks
(column width, bracket balance, printf argument counts). They have not been
built: this tree is not compiled here, and the binary is built with the NDK
separately.
