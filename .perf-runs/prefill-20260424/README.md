---
date: 2026-04-24
model: SmolLM-135M Q8_0 (~145 MB)
host: Strix Halo (Ryzen AI Max+ 395 Zen5 32T), AMD Radeon 8060S iGPU
test: pp512 (512-token prefill-only, llama-bench style)
dotllm-branch: feature/prefill-tiled-gemm (from feature/mamba-3 @ 94b9a24)
---

## Phase 1 re-measurement summary

The stale "7x prefill gap" predates the decode fix (`173fd29`) and the
cross-lib harness. Re-running with the same 512-token prefill workload
shows **dotLLM 334 tok/s vs llama.cpp 2198 tok/s = 6.6x gap**. Decode fix
did not help prefill (as expected — prefill uses `EventBased` dispatch mode,
never on the SpinWait collapse curve). Gate 1 **triggered** — proceed to
Phase 2 profiling.

## llama.cpp pp512 on CPU (Vulkan DLL hidden, true CPU-only)

`C:/Development/llama.cpp/llama-bench.exe -m SmolLM-135M.Q8_0.gguf -p 512 -n 0 -t <N>`

| Threads | pp512 tok/s | Notes |
|--------:|------------:|-------|
|       8 |        1557 | |
|      16 |        2198 | Best |
|      32 |        2150 | Saturated / slight regression |

llama-bench `-ngl 0 -dev none` still loads `ggml-vulkan.dll` and labels the
row Vulkan even though weights are on CPU. To measure true CPU-only we
temporarily renamed `ggml-vulkan.dll`; the numbers above are the result.

## dotLLM prefill via `PrefillBenchmarks` (added this session)

`benchmarks/DotLLM.Benchmarks --filter *PrefillBenchmarks*`

The benchmark calls `model.Forward(tokenIds[512], positions[0..512], kvCache)`
with a fresh `SimpleKvCache`, isolating prefill from tokenizer, sampling,
and HTTP overhead. Uses `ThreadingConfig.Auto` = 32 threads on this box.

| Mean wall | Tokens | Prefill tok/s |
|----------:|-------:|--------------:|
|   1.534 s |    512 |         **334** |

BDN notes: warmupCount=2, iterationCount=5, 1 mild outlier. StdDev 163 ms
(10%). Mean converted to tok/s: 512 / 1.534 = 334.

## Gap

**dotLLM 334 vs llama.cpp 2198 = 6.6×.** Meaningful — proceed to Phase 2.

## Phase 3 fix applied: 2D work partition for `GemmTiledQ8Worker`

Change in `src/DotLLM.Cpu/Kernels/MatMul.cs`:

* When `totalTiles >= threadCount`: pure row-tile partition (old path preserved
  — threads are saturated on M alone, as in `lm_head`-shaped matmuls).
* When `totalTiles < threadCount` (QKV/O/Gate/Up/Down on SmolLM-sized models):
  split M into `totalTiles` uniform row-groups (aligned to 4 for 4-row VecDot)
  and distribute the remaining threads across N tokens within each row-group.
  Weight tile per thread stays L2-resident; all threads now have work.
* `GemmQ8_0(pool)` no longer short-circuits `totalTiles < 2` to a
  single-threaded path when N is big — a single-tile matmul with 512 tokens
  now fans out across 32 threads via the 2D partition.

## Before vs after — full `PrefillBenchmarks` pp512 (SmolLM-135M Q8_0, 32T)

| Variant           | Mean wall | Prefill tok/s | vs llama.cpp |
|-------------------|----------:|--------------:|-------------:|
| Before fix        |  1534 ms  |       **334** |        15%   |
| After 2D partition|   570 ms  |       **898** |      **41%** |
| llama.cpp (16T)   |  233 ms   |          2198 |       100%   |

**2.7× speedup end-to-end.** Closed ~60% of the gap to llama.cpp.

### Per-projection GFLOPS, before vs after (warmed-up steady state)

| Kernel (m×k×n)         | Before | After   | Ratio |
|------------------------|-------:|--------:|------:|
| Q       576×576×512    |   32   |    178  |  5.6× |
| K       192×576×512    |   24   |    113  |  4.7× |
| V       192×576×512    |   24   |    111  |  4.6× |
| O       576×576×512    |   48   |    188  |  3.9× |
| Gate   1536×576×512    |  117   |    235  |  2.0× |
| Up     1536×576×512    |  120   |    274  |  2.3× |
| Down    576×1536×512   |   76   |    246  |  3.2× |
| lm_head 49152×576×512  |  211   |    298  |  1.4× |

Wider-M kernels (Gate, Up, lm_head) improve less because they already had
6+ tiles on 32 threads under the old scheme. The biggest wins are on
projections that had 1-3 tiles — exactly what the diagnosis predicted.

## Phase 2 — kernel-level profile (`.perf-runs/prefill-20260424/profile/`)

The `PrefillProfile` tool calls `MatMul.GemmQ8_0` at the shapes SmolLM-135M uses for
N=512 prefill (Q, K, V, O, Gate, Up, Down, lm_head) with a warmed-up
`ComputeThreadPool` at 32 threads, and reports ms/call and GFLOPS.

```
  name              m      k    n   ms/call   GFLOPS         x30 layers
  ─────────────────────────────────────────────────────────────────────
  Q               576    576  512     10.64     31.9       319.3 ms
  K               192    576  512      4.70     24.1       140.9 ms
  V               192    576  512      4.79     23.7       143.6 ms
  O               576    576  512      7.16     47.5       214.7 ms
  Gate           1536    576  512      7.72    117.3       231.6 ms
  Up             1536    576  512      7.55    119.9       226.6 ms
  Down            576   1536  512     11.91     76.1       357.3 ms

  Sum of MatMul time per layer:   54.46 ms  (× 30 = 1634 ms)
  lm_head       49152    576  512    137.38    211.0       137.4 ms

  Predicted pp512 from MatMul alone: 1634 + 137 = 1771 ms → ~290 tok/s
  Measured pp512 end-to-end:                           1534 ms → 334 tok/s
```

Predicted ≈ measured within noise. **`MatMul.GemmQ8_0` is the leaf — Gate 2 passes.**

### Why the hypothesis in `.continue-here.md` §2 was wrong

The section predicted tiled GEMM with 8×8 or 16×16 cache blocking would close
the gap. Inspection of `MatMul.cs:ComputeGemmTiled` and `GemmTiledQ8Worker`
plus the GFLOPS table above tells a different story:

- `ComputeTileM` already picks tileM = min(256, L2_budget / rowBytes). For
  SmolLM k=576, rowBytes = 612, L2 = 256KB → `tileM = 256`, aligned to 4.
- For the narrow projections (M in {192, 576}), `totalTiles = ceil(M/tileM)`
  is **1-3**. With `threadCount = 32` and
  `tilesPerThread = ceil(totalTiles / threadCount) = 1`, only 1-3 threads
  get any work — 29 of 32 sit idle on the barrier.
- GFLOPS matches this exactly: 192-row projections hit 24 GFLOPS,
  576-row hit 32-48, 1536-row hit 117 (still under-utilised — 6 tiles on 32
  threads), 49152-row `lm_head` hits 211 (saturating, closest to peak).

The weight tile at 150KB fits in Zen 5's 48KB L1D only partially but sits
fully in the 1MB per-core L2 — so each worker is fine on cache, it just has
nothing to do. Cache blocking buys nothing when the problem is empty threads.

### Real diagnosis

**Work partition is coarse-grained on M only, never on N.** `GemmTiledQ8Worker`
splits the M rows into tiles and assigns whole tiles to threads. For the
QKVO projections on SmolLM (M∈{192, 576}), this leaves the majority of
workers idle for the entire 30 × prefill calls. The MLP projections fare
better (1536 rows, 6 tiles) but still under-utilise.

The Nov 2026 `ThreadPoolDispatchBenchmarks` run in `.perf-runs/cross-lib-20260424/`
confirmed EventBased dispatch at 32 threads costs ~24 µs per Dispatch — but
our prefill kernels take 5-12 ms apiece, so dispatch overhead is only ~0.5%
per call. The waste is idle workers, not dispatch cost.

### Fix direction (Phase 3)

Two candidate changes, smallest first:

1. **Split work across both M and N** (row-and-token partition). Each thread
   gets a rectangular subset of the output C[t, m] = sum_k B[t,k] × A[m,k]^T.
   For Q/K/V where M is small, threads get token slices rather than row
   slices, restoring parallelism. No kernel rewrite needed.

2. **If (1) is not enough**, interleaved R4 weight layout currently only
   runs on the N=1 decode path. Enabling it for prefill is a bigger change.

Expected win from (1): ~3-4× on Q, ~6-8× on K/V, ~2× on O, ~2× on Gate/Up,
negligible on lm_head. Weighted by their x30 contribution, total prefill
should drop from ~1634 ms to ~500-700 ms → ~700-1000 tok/s. Still short of
llama.cpp's 2200, but ~2-3× faster than today.

## Repro

```bash
# llama.cpp (rename away Vulkan for CPU-only number)
mv C:/Development/llama.cpp/ggml-vulkan.dll C:/Development/llama.cpp/ggml-vulkan.dll.bak
C:/Development/llama.cpp/llama-bench.exe \
    -m C:/Users/james/.dotllm/test-cache/QuantFactory/SmolLM-135M-GGUF/SmolLM-135M.Q8_0.gguf \
    -p 512 -n 0 -t 16
mv C:/Development/llama.cpp/ggml-vulkan.dll.bak C:/Development/llama.cpp/ggml-vulkan.dll

# dotLLM end-to-end
cd benchmarks/DotLLM.Benchmarks
dotnet run -c Release -- --filter '*PrefillBenchmarks*'

# dotLLM per-projection breakdown
dotnet run --project .perf-runs/prefill-20260424/profile/PrefillProfile.csproj -c Release 512 32
```
