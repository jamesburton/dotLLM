---
topic: LoRA Q8_0 stage-1 — closing the −21% prefill regression (sequential-exploration item 3)
owner: cpu-perf
status: disconfirmed
date: 2026-05-17
closed: 2026-05-18
target: Strix Halo (AMD Ryzen AI Max+ 395, Zen 5, AVX-512F+BW+DQ+VL+VBMI+VNNI), .NET 10.0.103
conclusion: See `.planning/notes/lora-q8-stage1-probe-results.md` — Path B3 (R4 reuse) was disconfirmed empirically at the canonical shape. No in-tree fix path remained; productionisation cancelled.
---

## TL;DR

Prototype **Path B2 (R16-interleaved B repack) first**, single kernel, no
dispatcher changes. The probe at `benchmarks/LoraQ8Stage1Probe/` already
contains a candidate (`Stage1_PathBC_R16Interleaved_Avx512`) and the
adapter-load repack helper (`Kernels.RepackR16`). The mechanism case for B2
is structurally stronger than for B1 on this geometry: at M=rank=16, K=2048,
N=512 the dominant cost is weight-bandwidth × N (17 MB of B re-reads per
LoRA call), and only a layout that lets one weight-load serve `rank`
accumulators reduces that traffic to ~136 KB streaming. B1 (tiny-M
GEMV-per-token) only removes the 4-row batching constant — same bandwidth
profile, so the achievable speedup is bounded by per-block overhead, not
the bottleneck.

Path B1 should still be cheap to prototype as a comparison point (≤80 LOC,
no new dtype) — it tells us whether 16 ZMM accumulators are
register-pressure-bound on Zen 5 before we touch storage layout. The probe
file already has it: `Stage1_PathC_R16_Avx512` (single-block) and
`Stage1_PathC2_R16_Avx512Dual` (dual-block). Run both alongside B2 on the
target.

The previously-deleted-then-restored probe file should be **kept** —
contrary to `.continue-here-lora-final-mile.md` line 84-85, it is currently
present at `benchmarks/LoraQ8Stage1Probe/Program.cs` and `Kernels.cs` (verified
2026-05-17), already contains all candidate kernels including the R16 repack
helper, and adds stage-2 + fully-fused E2E variants beyond what the spec
proposes. Treat it as the spike artefact; do not delete it.

---

## 1. Mechanism analysis on Zen 5 at M=16, K=2048, N=512

### 1.1 Why today's `GemmQ8_0` is 2× slower than `F32_DequantOnce`

`GemmQ8_0(preQuantizedInput=xQ8)` walks `ComputeGemmTiled` →
`ComputeRows` → `VecDotQ8_0Avx512_4Rows`. At M=rank=16, `tileM` clamps to
16 so the M-tile loop is degenerate; per token we do 4 calls to
`VecDotQ8_0Avx512_4Rows` and re-read **34.8 KB of B weights every token**.
At N=512 the cumulative B traffic is **17.8 MB**, far exceeding the
1 MB Zen 5 private L2. The kernel is **memory-bandwidth-bound on B
re-reads, not FLOP-bound** — the SIMD inner loop (sign-trick + dual
`vpmaddubsw`/`vpmaddwd` + `vcvtdq2ps zmm` + `vfmadd231ps zmm`) has a
~5 µs ceiling for N=512; measured 970 µs is 200× off the SIMD ceiling.

The B traffic also takes a *non-sequential* form: at the row-major
stride of 2 176 bytes per row, every K-block iteration's 4-row weight
load is 4 cache-line accesses 2 KB apart. With 16 rows total that
becomes 16 interleaved streams per dual-block — beyond what Zen 5's L2
streaming prefetcher (~12-14 stream tracker) handles cleanly.

`F32_DequantOnce` sidesteps both problems: dequants B into **128 KB F32
scratch (fits in L2)** then runs `TensorPrimitives.Dot` per (token,row).
`TensorPrimitives.Dot` on .NET 10/AVX-512 issues 4-way FMA dep chains
across Vector512 accumulators (Stephen Toub, *Perf Improvements in
.NET 10* §Tensor Primitives) — at K=2048 it's ~25 ns/dot, ~200 µs of
FMA + ~270 µs of L2 latency. Measured ~470 µs matches the model.

Net: F32 wins because B is **read once at full bandwidth into L2** then
**streamed from L2 at L1 latency for 512 tokens**. Q8_0 keeps re-reading
B from L3+RAM with a 16-stream access pattern.

### 1.3 Path B1 — Tiny-M GEMV-per-token (`Stage1_PathC_R16_Avx512`)

Re-orient: outer = tokens, inner = K-block, 16 explicit Vector256<float>
accumulators per token. Register pressure ~16 + ~9 transients ≪ 32 ZMM
— no spills. But **B traffic still 17.8 MB at the same 16-stream
row-major access pattern**. B1 only removes the 4-row M-tile outer loop
overhead (negligible at 4 iterations); the 4-row batching it replaces
was already amortising x-loads across 4 rows.

**Predicted B1: 5-15% gain over `Q8_0_GemmPreQuantX`, still slower than
F32 dequant-once.** Value as a sanity check that 16-acc register
pressure isn't the limiter.

### 1.4 Path B2 — R16-interleaved B (`Stage1_PathBC_R16Interleaved_Avx512`)

Repack B from `[rank=16, K_blocks×34]` row-major to `[K_blocks, rank×34]`
block-interleaved. Per K-block, all 16 weight blocks are 16 × 34 = 544
contiguous bytes (8.5 cache lines; one 4 KB OS page covers ~7.5 K-blocks).

Bandwidth: still **17.8 MB per call** at the byte level. So B2's gain is
*not* in bytes-from-RAM. The gain is in:

1. **Single sequential stream** for B. HW prefetcher locks onto one stream
   and pulls the whole 34 KB B-block-chunk into L1 with maximal efficiency.
   At 64 KB/µs L1 fill from L2 (Zen 5 ~12 cycles latency, full bandwidth
   on hit), B is fed at saturated L2→L1.
2. **No TLB pressure**. Row-major B at rowBytes=2 176 means 16 rows touch
   16 different 2 KB-stride pages per K-block — wait, 2 176 bytes per row
   means 16 rows span 32 KB, all within ~8 4-KB pages. TLB pressure
   present but not catastrophic. R16 has all 16 weight blocks in
   contiguous memory → single page touch.
3. **L3 streaming bandwidth**. 17.8 MB exceeds L2; sequential L3 reads
   on Zen 5 hit ~30-40 GB/s sustained, much better than a 16-stream
   row-major access pattern through the same L3.

This is what llama.cpp's `mul_mat_vec_q` family exploits — the 34 B
Q8_0 block lays exactly on 64-byte cache-line granularity; R16 walks 16
cache lines in a sequential burst per K-position. Register pressure and
inner-loop instruction mix identical to B1 — **B2 is B1 with a different
B-pointer arithmetic**.

**Predicted B2 result: 30-50% faster than B1, competitive with F32
dequant-once at this shape; for larger N (longer prefill) B2 should
overtake F32 because B traffic stays Q8_0 (1.06 B/elem) vs F32 (4 B/elem)
when both spill out of L2.**

### 1.5 Cache footprint summary at M=16, K=2048, N=512

| Path | B per-call traffic | x per-call | Scratch | L2 fits? |
|---|---|---|---|---|
| F32 DequantOnce | 128 KB read + dequant write | 1.1 MB F32 (in-place) | 128 KB | Yes (B scratch) |
| Q8_0 GemmPreQuantX (today) | 17.8 MB row-major (16 streams) | 1.1 MB Q8_0 | none | No |
| Path B1 (PathC) | 17.8 MB row-major (16 streams) | 1.1 MB Q8_0 | none | No |
| Path B2 (PathBC R16) | 17.8 MB sequential (1 stream) | 1.1 MB Q8_0 | 34 KB R16 (one-shot) | No, but streaming |

The "L2 fits?" column undersells B2: even though B doesn't fit, the
sequential access pattern lets Zen 5's L2 streaming prefetcher (designed
to feed 12-14 simultaneous streams) overlap RAM→L2 with L2→L1 ALU
consumption. Row-major can't.

---

## 2. Risk + complexity

### 2.1 Path B1 (`GemvQ8_0PerToken` row-major, 16-acc per token)

| Aspect | Value |
|---|---|
| New code | ~80 LOC kernel + ~30 LOC dispatcher hook |
| New file | none (drop into `MatMul.cs` as `GemvQ8_0PerTokenR16Avx512` next to `VecDotQ8_0Avx512_4Rows`) |
| New dtype | no |
| Adapter load impact | none |
| Test scope | bit-parity vs `MatMul.GemmQ8_0` (existing `LoraDeltaQuantizedQ8_0Tests` already covers this) |
| Regression risk for other shapes | low — gated behind `M == 16` (or specifically `LoraStage1` call site). Does not touch `GemmQ8_0` general path. |
| Interaction with R4 path | none |

### 2.2 Path B2 (R16-interleaved B layout)

| Aspect | Value |
|---|---|
| New code | ~120 LOC kernel + ~40 LOC `Quantize_F32_To_Q8_0_R16` repack helper + ~20 LOC dispatcher hook + ~10 LOC `LoraWeightDType.Q8_0_R16` enum value |
| New file | optional — `Kernels.RepackR16` already exists in the probe and is 18 LOC; production-quality version is the same code. |
| New dtype | **yes** — `LoraWeightDType.Q8_0_R16 = 4` (or store as a flag on the existing Q8_0 enum value — discuss with project owner) |
| Adapter load impact | one-shot repack at `LoraAdapter.LoadAsync` — negligible cost (34 KB × num_layers × num_projections); strict copy-with-permute |
| Test scope | new bit-parity test for the R16 layout (probe already does this) + round-trip via `DequantizeRowToF32` |
| Regression risk for other shapes | **none for non-R16 adapters** — the path is opt-in per `LoraLayerWeights.WeightDType`. Existing Q8_0 adapters keep using `ApplyQ8_0BWithPreQuantX`. |
| Interaction with R4 path | none structurally; could share the `WeightRepacking` helper file if/when that exists |

### 2.3 Cross-shape risk (M ∈ {8, 16, 32, 64})

- **M=8**: half the accumulators (8 ZMM). Same per-K-block work
  count → R16-equivalent (R8) layout. New kernel needed *or* an
  M-templated kernel via generic/source-gen. **Risk**: shipping only R16
  leaves M=8 on the row-major path. Mitigation: prototype R16 first, then
  emit R8 from the same template if the speedup is large.
- **M=32**: register pressure becomes a real concern (32 acc = ZMM file
  full, no headroom for transients). Likely needs to fall back to two
  M=16 tiles via the existing M-tile loop. **Risk**: low if we gate on M.
- **M=64**: definitely falls back to two M-tile passes; behaves like the
  current row-major path on each tile. No new code needed.
- **Different K** (K=4096 — Llama-3.2-3B): `blockCount` = 128. B per row
  doubles to 4.4 KB; total B for M=16 is 68 KB, still 2× L2. Same
  bandwidth story. R16 still wins.

### 2.4 Interaction with the existing R4 path

The base-model Q8_0 GEMM uses `OuterProductGemmQ8_0` with R4 weight
layout for the **base** model (e.g. q_proj's W is repacked R4 at model
load via `WeightRepacking`). LoRA B is *separate* from base W. They
share the GGUF Q8_0 element format but not the layout.

There's a clean reuse opportunity: the AVX-512 4×6 microkernel
(`OuterProductQ8_0Avx512_4x6`) is exactly the right shape for LoRA stage
1 with M=16, treating it as four R4 groups × 6-token tile. If the LoRA-B
buffer is repacked R4 at adapter-load instead of R16, we can call
`OuterProductGemmQ8_0` directly and skip writing a new kernel.

That's actually a **Path B3** worth considering:

**Path B3 — re-pack B as R4 and reuse `OuterProductGemmQ8_0`.** Zero
new SIMD code, only a `RepackR4(bWeight, ...)` helper and a dispatcher
hook in `ApplyQ8_0BWithPreQuantX`. Risk: lowest of the three.

The reason `.continue-here-lora-final-mile.md` doesn't mention this is
historical — `OuterProductGemmQ8_0` landed *after* the Phase 4d.5 notes
were written (the spec date is 2026-05-14, the outer-product GEMM at
M=16 was probably the LoRA stage-2 work of Phase 4d.6 closing on
2026-05-15+).

**Recommendation revision**: prototype B3 *first* — it should literally
be ~30 LOC of dispatcher + repack code, and it bypasses the entire
kernel-design exercise. If B3 hits parity with F32 (which it should, per
the cost model — 4×6 microkernel keeps both x and B in L1, only B
streams from L3), the LoRA work is done. If B3 still falls short, then
B2 is the next step (kernel-tuned-for-rank-16 with all 16 acc live, vs
B3's 4-acc-at-a-time pattern).

---

## 3. Recommended order

1. **B3 (re-use `OuterProductGemmQ8_0`) — try first.** Repack B at
   adapter load into R4 layout; call the existing AVX-512 4×6
   microkernel. Zero new SIMD code. If this hits parity with F32, ship
   it.
   - Implementation: ~30 LOC. Reuse `WeightRepacking.RepackR4` (or its
     equivalent) at adapter load; dispatch in `ApplyQ8_0BWithPreQuantX`.
2. **B2 (R16 layout, dedicated 16-acc kernel) — fallback if B3 doesn't
   close the gap.** This is what the probe (`Stage1_PathBC_R16Interleaved_Avx512`)
   tests directly. ~120 LOC.
3. **B1 (tiny-M GEMV-per-token, row-major) — only as a comparison
   datapoint.** If B3+B2 both fail to beat F32 at this shape, B1's
   results tell us whether the bottleneck is register pressure (Zen 5
   ZMM file) or pure memory bandwidth. The probe already has both
   single-block and dual-block variants.

The three should be benchmarked together on the same probe run on Strix
Halo — the probe already has B1 and B2; **adding B3 is a 30-line
addition** to the probe.

---

## 4. Microbench plan

The probe at `benchmarks/LoraQ8Stage1Probe/` is the right vehicle. Need
to extend it with **shape sweeps and a B3 kernel**.

### 4.1 Shape table

| M (rank) | K | N (seqLen) | Notes |
|---|---|---|---|
| 8 | 2048 | 64, 256, 512 | small-rank adapter |
| 16 | 2048 | 64, 256, 512 | canonical (Llama-3.2-1B Q8_0) |
| 32 | 2048 | 64, 256, 512 | rank-32 adapter |
| 64 | 2048 | 64, 256, 512 | high-rank adapter (rare but supported) |
| 16 | 4096 | 64, 256, 512 | Llama-3.2-3B / Qwen2.5-1.5B hidden |
| 16 | 2048 | 1, 8 | decode shapes (sanity — N=1 path goes via GemvQ8_0 today) |

Kernels under test per shape:
- `F32_DequantOnce` (production baseline, the bar)
- `F32_OnlyGemm` + `F32_OnlyDequant` (decomposed baseline)
- `Q8_0_GemmPreQuantX` (current gated path)
- `Q8_0_R4_Reuse_OuterProduct` (**new — B3**)
- `Q8_0_R16_PerToken_Single` (B1 = `Stage1_PathC_R16_Avx512`)
- `Q8_0_R16_PerToken_Dual` (B1' = `Stage1_PathC2_R16_Avx512Dual`)
- `Q8_0_R16_Interleaved` (B2 = `Stage1_PathBC_R16Interleaved_Avx512`)

### 4.2 Run config

```
RunStrategy: Throughput
Warmup: 5 iterations
Iterations: 80 per shape (matches the Phase 4d.5 probe precedent)
MemoryDiagnoser: on (verify no per-call ArrayPool churn)
DisassemblyDiagnoser: on for B1 inner loop (validate 16 ZMM accumulators are kept live)
GC: Server + ConcurrentGarbageCollection
TieredPGO: true (matches production)
```

### 4.3 Tolerance and acceptance bars

Same as Phase 4d.5 probe:
- **Bit-parity gate**: each candidate's `tmp[t, r]` must agree with the
  `Q8_0_GemmPreQuantX` reference output within `1e-2` absolute (Q8_0
  round-trip tolerance — current `LoraDeltaQuantizedQ8_0Tests` uses
  `5e-2`).
- **Speed gate** (per shape, median of 80 iter):
  - **Win**: at M=16, K=2048, N=512, **≤ 0.47 ms/call** (parity with F32
    dequant-once at the canonical shape)
  - **Strong win**: ≤ 0.40 ms/call (15% faster than F32)
  - **Theoretical floor** (~bandwidth-only): ~230 µs ⇒ a 2× margin remains
    for a future VNNI-enabled kernel

### 4.4 Targets on Strix Halo

Macro-bench acceptance after the kernel lands (rerun
`LoraMacroBenchmarks` on `Llama-3.2-1B-Q8_0`):
- LoRA Q8_0 prefill regression **≤ −10%** vs NoLora (current: −21%)
- LoRA Q8_0 decode tokens/s **at parity** with current NoLora (decode is
  not stage-1-bound — N=1 path)

### 4.5 Probe extension — exact diff

Add to `benchmarks/LoraQ8Stage1Probe/Kernels.cs`:

```csharp
public static byte* RepackR4(byte* bRowMajor, int rank, int blockCount, int rowBytes)
{
    int fullGroups = rank / 4;
    long totalBytes = (long)fullGroups * 4 * blockCount * Kernels.Q8_0Block;
    byte* dst = (byte*)NativeMemory.AlignedAlloc((nuint)totalBytes, 64);
    int groupBytes = 4 * blockCount * Kernels.Q8_0Block;
    for (int g = 0; g < fullGroups; g++)
    {
        byte* groupDst = dst + (long)g * groupBytes;
        for (int b = 0; b < blockCount; b++)
            for (int r = 0; r < 4; r++)
            {
                byte* src = bRowMajor + (long)(g * 4 + r) * rowBytes + b * Kernels.Q8_0Block;
                byte* d = groupDst + b * 4 * Kernels.Q8_0Block + r * Kernels.Q8_0Block;
                Buffer.MemoryCopy(src, d, Kernels.Q8_0Block, Kernels.Q8_0Block);
            }
    }
    return dst;
}
```

Add to `Program.cs` (call site is `MatMul.OuterProductGemmQ8_0` — make it
`internal`-visible to the probe project via `InternalsVisibleTo` or by
exposing a thin public wrapper on `MatMul`):

```csharp
[Benchmark(Description = "Q8_0_R4_Reuse_OuterProduct")]
public void Q8_0_R4_Reuse_OuterProduct()
    => MatMul.OuterProductGemmQ8_0(
        _bR4, _xQ8, _tmp,
        fullGroups: Rank / 4, tailRows: Rank % 4,
        blockCount: _blockCount, m: Rank, n: N);
```

Plus an `_bR4` field and a Setup-time repack call.

---

## 5. Parallel-task suggestions

Independent of the kernel prototype:

1. **Correctness oracle test** (`tests/DotLLM.Tests.Unit/Kernels/LoraStage1R16ParityTests.cs`)
   — scalar Q8_0 vec-dot reference, bit-parity at M=16, K=2048,
   N∈{1, 16, 512}.
2. **Adapter-load repack helper** in `src/DotLLM.Models/Lora/WeightRepacking.cs`
   (or extension of an existing helper) — `RepackQ8_0_RowMajorToR16` /
   `_RowMajorToR4`. Testable in isolation. Independent of which kernel
   layout we ship.
3. **Macro-bench kernel-selector env var** — extend `LoraMacroBenchmarks`
   to honour `DOTLLM_LORA_Q8_KERNEL=row|r4|r16` so the macro-bench
   re-run can pick a candidate without dispatcher rewrites.
4. **Strix Halo dev-loop script** — `scp` the probe binary, run `--quick`,
   print table. Cuts per-iteration cycle from ~5 min to ~30 s.
5. **Macro-bench median tooling** — wrap 5× runs, report min/median/max
   (current ±10 tok/s noise floor masks small kernel wins).

---

## 6. Implementer notes

- `TensorPrimitives.Dot` on .NET 10 + AVX-512 dispatches through
  Vector512 with 4-way FMA dep chains. At K=2048 this is near the
  codegen ceiling. **Beating the F32 baseline requires fewer bytes from
  RAM, not more FLOPs.**
- `Avx512Vnni.V512` is **not** in .NET 10 (see [`dotnet-avx512-vnni-status.md`](dotnet-avx512-vnni-status.md));
  any new kernel must use the sign-trick + `vpmaddubsw`/`vpmaddwd` chain.
  When/if V512 VNNI lands, the int-dot inner loop collapses from 3 ops
  to 1 — recheck this whole analysis at that point.
- Zen 5: no AVX-512 frequency penalty; L1d 48 KB; L2 1 MB private;
  L3 32 MB shared. Streaming prefetcher tracks ~14 streams →
  **row-major M=16 with 16 streams runs into prefetcher contention
  whereas R16 / R4-interleaved sees a single stream.** That's the
  structural reason B2 and B3 should both beat B1 / row-major.

---

## 7. Path B3 (R4 reuse) — production sketch

Production diff in `LoraDelta.ApplyQ8_0BWithPreQuantX`:

```csharp
if (bDType == LoraWeightDType.Q8_0_R4)
{
    int fullGroups = rank / 4;
    int tailRows = rank % 4;
    MatMul.OuterProductGemmQ8_0(
        bWeight, xQ8, tmp,
        fullGroups, tailRows, inputDim / 32,
        m: rank, n: seqLen, pool: pool);
    // Stage 2 unchanged
}
```

Plus a one-shot `WeightRepacking.RepackR4_Q8_0(bRowMajor, bR4, rank, blockCount)`
called at adapter load when the loader opts into `Q8_0_R4` storage.
`WeightRepacking` already exists for the base-model R4 repack — LoRA
adopts the same helper.

**Why this should work**: `OuterProductGemmQ8_0` at M=16 runs the
existing 4×6 AVX-512 microkernel with **24 ZMM accumulators (4 rows × 6
tokens kept live)** — same instruction mix as a hand-written B2, but
already validated, tested, and parallelism-aware. The "geometry
mismatch" the original spec claims against `GemmQ8_0` at M=16 is
specifically about the row-major-tiled path
(`ComputeGemmTiled` → `VecDotQ8_0Avx512_4Rows`), which re-reads B per
token. The outer-product path doesn't.

## 8. Open questions

1. **Does `MatMul.GemmQ8_0(..., pool)` dispatch to `OuterProductGemmQ8_0`
   for the base Q8_0 model at M=16?** Need to follow the dispatch chain.
   `ParallelMinRows` gating (see `MatMulQ5_0.cs`) may select the
   row-major path at M=16. If so, B3 might *implicitly* improve the base
   Q8_0 GEMM too — worth a separate probe.
2. **Decode-path (N=1) sanity**: at N=1 `ApplyQ8_0BWithPreQuantX` takes
   the `GemmQ8_0(... n=1)` → `ComputeRows(preQuantizedInput)` branch.
   The R4 path must handle N=1 cleanly — the existing
   `OuterProductGemmQ8_0` falls back to `VecDotQ8_0Avx2_4RowsR4` for
   small N, so this should work, but verify in the bit-parity test.

---

## Sources

- `C:\Development\dotLLM\.continue-here-lora-final-mile.md` — task spec
- `C:\Development\dotLLM\.planning\notes\dotnet-avx512-vnni-status.md` — confirms VNNI-512 is months/years out
- `C:\Development\dotLLM\src\DotLLM.Cpu\Kernels\MatMul.cs` — `OuterProductQ8_0Avx512_4x6` at line 1451, `ComputeGemmTiled` at line 1082, `VecDotQ8_0Avx512` at line 546, `VecDotQ8_0Avx512_4Rows` at line 627
- `C:\Development\dotLLM\src\DotLLM.Cpu\Kernels\LoraDelta.cs` — call site `ApplyQ8_0BWithPreQuantX` at line 352
- `C:\Development\dotLLM\benchmarks\LoraQ8Stage1Probe\Program.cs` + `Kernels.cs` — existing probe with B1/B1'/B2 candidates and stage-2 fast path
- llama.cpp `ggml/src/ggml-cpu/ggml-cpu-quants.c` — `mul_mat_vec_q` family, R-interleaved Q8 layout (reference implementation for the bandwidth-bound design pattern)
- Stephen Toub, *Performance Improvements in .NET 10*, §Tensor Primitives (Vector512 dispatch on AVX-512)
- AMD Software Optimization Guide for AMD EPYC 9005 Series Processors (Zen 5) — L1/L2/L3 cache geometry, streaming prefetcher characteristics
