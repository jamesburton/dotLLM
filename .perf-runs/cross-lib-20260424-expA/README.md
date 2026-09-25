---
date: 2026-04-24
experiment: A — clamp _decodeThreadCount to min(8, threadCount) when no NumaTopology
baseline: .perf-runs/cross-lib-20260424/ (dotLLM at 8.66 tok/s, 22.17 s wall)
---

## Result

Same harness, same model, same prompt, same host — only change is the
one-line decode-thread-count clamp in `ComputeThreadPool`:

| Backend | Run | tok/s | wall (s) |
|---|---|---:|---:|
| llama.cpp CPU (`-ngl 0`, from baseline) | 1 | 34.73 | 5.53 |
| dotLLM CPU (pre-experiment, from baseline) | 1 | **8.66** | 22.17 |
| dotLLM CPU + experiment A (cold) | 1 | **33.76** | 5.69 |
| dotLLM CPU + experiment A (warm prefix cache) | 2 | 39.03 | 4.92 |
| dotLLM CPU + experiment A (warm prefix cache) | 3 | 39.64 | 4.84 |

First-run / cold comparison: **dotLLM 33.76 vs llama.cpp 34.73 — within 3%**.
The 4× gap reported in the previous cross-lib comparison collapses to noise.

## What changed

One default, in `src/DotLLM.Cpu/Threading/ComputeThreadPool.cs`:

```diff
- : threadCount;
+ : Math.Clamp(DefaultDecodeThreadCountCap /* = 8 */, 2, threadCount);
```

When the pool is built without a `NumaTopology`, `_decodeThreadCount` used
to fall back to `threadCount`, so SpinWait mode ran all 31 workers on a
32-thread host. The `ThreadPoolDispatchBenchmarks` microbench (see
`.perf-runs/cross-lib-20260424/dispatch-microbench.md`) showed that path
collapses: Dispatch(no-op) costs 333 µs at 32T SpinWait vs 1 µs at 8T,
driven by 31 threads hammering one cache line on `_dispatchGeneration`
and a contended `_completion.Signal()` decrement.

Capping to 8 keeps SpinWait in the regime the microbench shows scales
well (2–16 threads, sub-µs dispatch cost). Prefill still uses all 32
threads via EventBased mode — unaffected.

## Why 8

Matches the heuristic already applied when a `NumaTopology` *is*
provided: `topology.MemoryChannelEstimate` (typically 2 channels × 4 on
Zen 5 desktop/workstation = 8). Decode is memory-bandwidth bound, not
compute bound, so extra concurrent loads past a small multiple of memory
channels add cache-line contention without adding useful bandwidth.

## Next

The 3% residual gap vs llama.cpp is within benchmark noise. If we want
to push past llama.cpp on this workload, the remaining levers in
priority order are the ones already listed in
`.perf-runs/cross-lib-20260424/dispatch-microbench.md` §Concrete next-step
experiments B and C (size-gated single-threaded fast path, cache-line
padding in `ComputeThreadPool`). Neither is currently justified by the
numbers — come back to them if we regress or if we see a similar issue
on a host with different topology.
