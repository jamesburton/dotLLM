---
date: 2026-04-24
benchmark: ThreadPoolDispatchBenchmarks
model-context: SmolLM-135M decode matmul size (~576-element row per partition)
host: Strix Halo (32 logical CPUs)
bdn: v0.14.0, .NET 10.0.3, AVX-512F+CD+BW+DQ+VL+VBMI
---

## Headline

At the thread count we actually use for SmolLM-135M decode (`ThreadingConfig.Auto` = 32), `ComputeThreadPool.Dispatch` costs **more than the single-threaded work it replaces** at every problem size relevant to decode. SpinWait mode collapses at 32 threads — one dispatch takes **333 µs** (1300× the 260 ns single-threaded baseline) and a 30-dispatch decode-layer burst takes **10.6 ms** versus 7.8 µs sequential.

This is why the PerfView trace showed 76% of CPU in the worker spin loop: at 32 threads, SpinWait is not helping — the workers are saturating the cache line on `_dispatchGeneration` and spinning through their 10,000-iteration budget before the caller can even increment the generation counter for the next dispatch.

## Per-dispatch cost (ns) — Dispatch(no-op) vs single-threaded baseline

| Threads | EventBased | SpinWait | Single-thread work |
|--------:|-----------:|---------:|-------------------:|
|       2 |        289 |      216 |                260 |
|       4 |        723 |      548 |                262 |
|       8 |      1,522 |    1,042 |                260 |
|      16 |      9,126 |    2,113 |                254 |
|      32 |     23,947 |  **333,405** |             264 |

Any row where Dispatch > Single-thread means parallelism is a net loss at
this work size. That's every row from 4 threads up.

## 30-dispatch burst (one decode layer loop)

Sequential single-threaded = 30 × 260 ns = **7,800 ns (7.8 µs)** ideal.

| Threads | EventBased      | SpinWait          | Ratio vs sequential |
|--------:|----------------:|------------------:|--------------------:|
|       2 |       14,849 ns |       11,553 ns   | 1.5× slower         |
|       4 |       27,550 ns |       17,681 ns   | 2.3× slower         |
|       8 |       43,443 ns |       32,705 ns   | 4.2× slower         |
|      16 |      589,037 ns |       52,266 ns   | 6.7× slower         |
|      32 |    1,415,062 ns | **10,627,799 ns** | **1,362× slower**   |

SmolLM-135M decodes 30 layers per token. At 32 threads SpinWait, each
token spends **~10 ms** just in worker-pool coordination. Across 150
tokens that's 1.6 s of the 22 s total — but the profile showed 76% of
CPU in the spin loop, i.e. ~17 s of summed thread-time across 32
threads. The extra cost shows up as workers spinning *between* bursts:
once a burst ends, workers spin their 10K-iteration budget before
falling through to event wait; if the caller issues the next burst in
under 1 ms, they never reach event wait and keep spinning.

## Why SpinWait degrades so badly at 32T

1. **Cache-line contention.** All 31 workers spin on `_dispatchGeneration`
   (`Volatile.Read` into a single volatile int). When the caller
   increments it, the cache line has to invalidate on 31 cores; the first
   reader wins, others re-fetch. At 32 threads this becomes the bottleneck.
2. **`_completion.Signal()` is contended.** `CountdownEvent.Signal()` is
   an interlocked decrement on a shared counter. 31 threads finishing
   roughly together all take that hit.
3. **OS oversubscription.** 32 logical threads on a 32-core host with
   any other load (explorer.exe, antivirus, the benchmark harness
   itself) means the scheduler de-schedules workers mid-spin, adding
   context-switch cost back on top of the spin cost.
4. **False sharing around `_activeWorkerCount` / `_dispatchThreadCount`.**
   These are read by every worker each iteration; they share a cache
   line with `_dispatchGeneration`.

EventBased scales better at 32T (24 µs vs SpinWait's 333 µs) because the
kernel arbitrates the wake — only one worker runs the signal path at a
time and the rest are parked, not spinning.

## Implications for dotLLM CPU perf

1. **`_decodeThreadCount` default is wrong for this host.** When no
   `NumaTopology` is provided (the common path — `ThreadingConfig.Auto`
   does not build one), `_decodeThreadCount` falls back to `threadCount`
   (32), so SpinWait runs on all 31 workers. It should instead default
   to `min(8, threadCount)` or similar — decode is memory-bandwidth
   bound and two channels per Zen 5 CCD don't reward more than ~8
   concurrent loads.
2. **Single-thread fast path below a size threshold.** Any decode
   matmul where per-thread work < ~500 ns should skip dispatch. For a
   576×576 Q8_0 matmul split across 32 threads, per-worker work is
   roughly 18 rows × 576 elements × ~2 cycles/elem = ~40 µs — above the
   threshold, but at 1 token × 1536-column Gate-Up-Down split across 32
   the per-worker chunk drops well below. A size-gated dispatch would
   pay off here.
3. **Reduce cross-thread sharing on the hot path.** Cache-line-pad
   `_dispatchGeneration`, `_activeWorkerCount`, and `_dispatchThreadCount`.
   Use a tree-reduction completion (two-level `CountdownEvent`) rather
   than 31 threads signalling a single atomic counter.
4. **Prefill still benefits from all 32 threads** — EventBased scales
   linearly through 16T and only starts hurting at 32T. The fix here is
   less about thread count, more about (1)–(3) above.

## Concrete next-step experiments

A. Cap `_decodeThreadCount` to 8 when no topology is provided, rerun
   the full InferenceBenchmarks + cross-lib harness. Cheap — one-line
   change, re-test.

B. Add a `Dispatch` fast-path that skips the pool when `workItems <
   SingleThreadThreshold` (tunable, e.g. 1024 F32 ops). Requires MatMul
   call sites to know their work size, which they already do (M×K).

C. Cache-pad hot fields in `ComputeThreadPool`. Low risk, easy to
   verify with this microbench.

Do A first — it's one line and the microbench already shows 4-16T
EventBased is sane at this dispatch cost, so clamping to 8 should
recover most of the 76% spin CPU. If A doesn't close the gap to
llama.cpp's 34 tok/s, try B and C.

## Artefacts

- `bench/results/ThreadPoolDispatchBenchmarks-report-github.md` — full BDN table
- `bench/results/ThreadPoolDispatchBenchmarks-report.csv`
- `bench/results/ThreadPoolDispatchBenchmarks-report-full-compressed.json`
