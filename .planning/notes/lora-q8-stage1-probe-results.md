---
topic: LoRA Q8_0 Stage 1 — probe results on Strix Halo
date: 2026-05-18
companion_to: lora-q8-stage1-investigation.md
status: negative_result_B3
---

# LoRA Q8_0 Stage 1 — probe results on Strix Halo

Probe run: `dotnet run --project benchmarks/LoraQ8Stage1Probe -c Release -- --quick`
Hardware: AMD Ryzen AI Max+ 395 (Strix Halo, Zen 5), 16C/32T, AVX-512 incl. VNNI, .NET 10.0.103 RyuJIT
Single-threaded (probe doesn't pass a `ComputeThreadPool`; `OuterProductGemmQ8_0`'s
thread-pool variant gates on `ParallelMinRows=32`, so M=Rank=16 falls through
to the single-threaded path — comparison is apples-to-apples).

## Stage 1 only at canonical LoRA shape (M=16, K=2048, N=512)

| Kernel | Median us/call | vs F32 |
|---|---:|---:|
| F32_DequantOnce (baseline) | ~458 | 1.00× |
| F32_OnlyGemm (excl. dequant) | ~412 | 0.90× |
| F32_OnlyDequant | ~2 | — |
| Q8_0_GemmPreQuantX (currently shipped) | ~770 | 1.68× slower |
| **Q8_0_R4_Reuse_OuterProduct (Path B3)** | **~840** | **1.84× slower** |
| PathC_R16_Single (B1 row-major, 16-acc) | ~2650 | 5.78× slower |
| PathC2_R16_Dual (B1' row-major, dual-block) | ~1500 | 3.27× slower |
| PathBC_R16Interleaved (B2 R16 interleaved) | ~2500 | 5.46× slower |

(Each row is the median of 3 OutputDim sweeps × 200 iter; OutputDim does
not affect Stage 1 work but reseeds caches, giving a useful noise floor
estimate. Single-run variance is ~10%, run-to-run variance for
`Q8_0_R4_Reuse_OuterProduct` was 821 / 879 / 1026 us — taking median of
the warm two: ~850 us.)

## Stage 1 + Stage 2 end-to-end

| OutputDim | E2E_Production_F32_DequantOnce | E2E_New_F32_OuterProduct | E2E_New_F32_FullyFused |
|---:|---:|---:|---:|
| 512 | 1406 | 507 | 859 |
| 2048 | 4493 | 1291 | 1659 |
| 5632 | 11056 | 2815 | 2962 |

The Phase 4d.6 outer-product Stage 2 win shows up cleanly here — 2.8× to
3.9× speedup end-to-end on the production F32 path. **This is the
already-shipped win; nothing new here.**

## Verdict — B3 (R4 reuse via `OuterProductGemmQ8_0`) is a dead end

The investigation's acceptance bar was ≤0.47 ms (470 us) at the
canonical shape. Path B3 measured ~840 us — same order as the currently
shipped `Q8_0_GemmPreQuantX` (~770 us), no closer to the F32 baseline
than the path it was meant to replace.

**The structural argument was wrong.** The investigation predicted that
`OuterProductGemmQ8_0` would beat `GemmQ8_0` because the R4 layout
trades 16-stream row-major access for sequential 4-stream access that
Zen 5's prefetcher can track. The data says either:

1. The 16-stream access pattern in `GemmQ8_0` is NOT prefetcher-limited
   on Zen 5 — perhaps the prefetcher tracks more than the 14 streams
   the AMD SOG suggests, or perhaps the per-block constants
   (Half→F32 scale conversion, ConvertToVector512Single, FMA accumulator
   setup) dominate so much that bandwidth differences are noise.
2. `OuterProductGemmQ8_0` has its own per-block constant cost (the
   4×6 microkernel keeps 24 ZMM accumulators live, has its own scale-
   multiply chain) that cancels out the access-pattern win.

Either way, **at M=16 the two paths are roughly equivalent**, and
neither matches F32_DequantOnce.

## What's left

Of the original investigation's paths:

- **Path A (upstream `Avx512Vnni.V512`)** — months/years out; item 8 PR
  filed an unstick-the-spec comment on dotnet/runtime#86849 last
  session (issuecomment-4472274791). Still the cleanest fix when it
  lands.
- **Path B1 (row-major tiny-M GEMV-per-token)** — 1500 us, 3.3× slower
  than baseline. Disconfirmed.
- **Path B2 (R16 dedicated kernel)** — 2500 us, 5.5× slower. Disconfirmed.
- **Path B3 (R4 reuse)** — 840 us, 1.84× slower. Disconfirmed at
  acceptance bar.

**Nothing in the current investigation's exploration space closes the
gap.** The dispatch refactor (`ApplyQ8_0BWithPreQuantX`) and the Phase
4d.6 Stage 2 outer-product are the wins; Stage 1 stays on the F32
dequant-once path until upstream VNNI lands or a kernel insight not
captured in the agent's analysis emerges.

## Recommended follow-ups

1. **Keep the `DOTLLM_LORA_FORCE_Q8_PREQUANT=1` gate off by default** —
   the path it gates is ~1.7× slower than F32 dequant-once at the
   canonical shape; only ship as default when a measured Stage 1 win
   exists.
2. **Update `docs/LORA.md` Phase 4d.5 with the empirical Stage 1
   verdict** — currently the doc cites the historical kernel-probe
   number (0.97 ms) and frames the −16% gap as gated on the V512 PR.
   The new number (770 us) and the B3 negative result narrow the gap
   slightly but don't close it.
3. **Park `LoraStage1R16ParityTests` + adapter-load `RepackR4_Q8_0`
   helper** from the parallel-task list — neither is needed since
   no Q8_0 Stage 1 kernel is being productionised. The probe stays
   in the tree as a research artifact + acceptance gate for future
   kernel work.
4. **Re-test when .NET 11 + V512 VNNI lands.** Re-run the same probe
   on a build where `Avx512Vnni.V512.MultiplyWideningAndAdd` is
   available. Expected: B1 / B2 / B3 all collapse to ~half the
   current numbers since the int8-dot inner loop drops from 3 ops
   (`vpmaddubsw` + `vpmaddwd` + `vpaddd`) to 1 (`vpdpbusd`).

## What the probe also revealed (silver lining)

The end-to-end measurement caught something useful: at OutputDim=5632
(FFN gate/up_proj shape), the Phase 4d.6 fully-fused E2E path is **3.7×
faster** than the production F32 dequant-once path (2962 vs 11056 us).
This validates the outer-product Stage 2 ROI on Strix Halo, and confirms
the Phase 4d.6 numbers in `docs/LORA.md` aren't dev-laptop artifacts.
