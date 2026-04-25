# Tensor-core (mma) batched MMQ — design note

**Branch:** `cuda-mma-batched-research` (forked from `feature/mamba-3-cuda` @ `69ef507`)
**Author:** dotnet-perf-expert agent, 2026-04-25
**Status:** Research only — no kernel code in this commit.
**Predecessor:** `docs/perf/MLPUP_GEMV_GAP.md` (closed the batch=1 GEMV gap).

## TL;DR

llama.cpp's tensor-core MMQ kernel (mma.m16n8k16 INT8 on sm_75+) activates at
**batch ≥ 64** on Ampere (`MMQ_DP4A_MAX_BATCH_SIZE = 64`, with `MMVQ_MAX_BATCH_SIZE = 8`
gating the GEMV path below that). dotLLM today runs at **batch = 1** for every
decode step — the multi-request scheduler that would produce non-trivial batch
sizes is `IScheduler` with no implementation (Phase 9 / step 35 on the
roadmap). Without that scheduler, an mma kernel **has nowhere to apply** in the
production hot path.

The recommended single next change is **option B — SKIP the mma kernel for now;
land continuous batching first**. Option A (write the kernel speculatively
behind a gate, validate on synthetic data) is a defensible Plan B if the kernel
author wants to keep CUDA momentum, but the work cannot translate into
end-user perf until the engine produces real batches.

The honest upper bound on mma's contribution today: **+0%** (no batched
workload exists). With continuous batching at a typical server batch of
B = 16 the upper bound is roughly **+30–60% throughput** on the GEMV-heavy
categories (MlpUp / MlpDown / QkvProj / LmHead) versus dp4a, but *only when
B ≥ 8-ish per decode step*. Speculative decoding alone (B = K = 4–8) lands at
the bottom of that band.

---

## 1. Threshold analysis

### 1a. Per-warp arithmetic peak — dp4a vs mma INT8

Both use INT8 inputs and FP32 accumulators. Per-cycle, per-warp throughput on
sm_86:

| Path | Op | Per thread / cycle | Per warp / cycle | Notes |
|---|---|---|---|---|
| dp4a | `__dp4a(int, int, int)` | 4 INT8 MACs | **128 INT8 MACs** | Issued on the INT pipe, sm_61+ |
| mma  | `mma.m16n8k16.s8.s8.s32` | 4×8×16 / 32 = **64 MACs / lane** | **2048 INT8 MACs** | Tensor core, sm_75+ |

So per-warp peak is **16× higher** for mma than dp4a. The catch is the M
dimension: mma.m16n8k16 produces a 16×8 output tile per warp from a 16×16
fragment of A and an 8×16 fragment of B. **The M=16 lanes are physically
allocated whether the work is there or not.**

### 1b. Effective utilisation as a function of batch B

In a transformer GEMV-as-GEMM, M is the active batch (number of concurrent
sequences in this decode step) and N is the output dimension (output rows of
the projection). For batch=1 decode:

| B | mma M lanes used | Effective utilisation | dp4a path |
|---|---|---|---|
| 1   | 1 / 16  | **6.25%** | 100% (no M-fragment waste) |
| 4   | 4 / 16  | 25%       | 100% |
| 8   | 8 / 16  | 50%       | 100% |
| 16  | 16 / 16 | **100%**  | 100% |
| 32  | 32 / 16 = 2 mma issues × 100% | 100% | 100% |

So **B=16 is the natural breakeven on raw arithmetic alone**: at B=16, mma
delivers 16× the per-warp throughput of dp4a *and* uses every M lane.

But arithmetic peak is not the whole story.

### 1c. Bandwidth-bound regime — when does compute matter?

The MMQ_GEMV_GAP note already established that at batch=1, MlpUp on Qwen3-8B
runs at ~50–200 GB/s depending on the kernel — fundamentally **bandwidth-bound
at this scale on a 360 GB/s RTX 3060**. Switching from dp4a to mma at B=1 buys
nothing because the bottleneck is memory, not compute.

Crossover heuristic — at what B does compute start to dominate?

- **Weight bytes per layer** (Qwen3-8B Q4_K_M MlpUp): ~57 MiB
- **Compute per layer per output column at B=1**: 2·k = 8192 INT8 MACs per row,
  × 24576 rows = ~200M MACs.
- **Arithmetic intensity** at B=1: ~200M MACs / 57 MiB ≈ 3.5 op/byte. Way
  below the RTX 3060 ridge point (~30 op/byte for INT8).
- At **B = 16**, weight reads are amortised across 16 outputs: arithmetic
  intensity scales linearly to ~56 op/byte → **compute-bound**, mma should
  shine.
- **Crossover ≈ B = 8–10** — below that, weight bandwidth dominates and the
  kernel choice barely matters; above, the per-warp throughput multiplier
  starts paying off.

This matches llama.cpp's empirically chosen `MMVQ_MAX_BATCH_SIZE = 8`: below
B=8 they use the MMVQ (GEMV) path; above, they switch to the batched MMQ. And
within MMQ, `MMQ_DP4A_MAX_BATCH_SIZE = 64` keeps dp4a until B is very clearly
in the compute-bound regime.

### 1d. The actual llama.cpp dispatch (cited verbatim)

From `ggml/src/ggml-cuda/mmvq.cuh`:

```c
#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.
```

From `ggml/src/ggml-cuda/mmq.cuh`:

```c
#define MMQ_DP4A_MAX_BATCH_SIZE 64 // Max. batch size to use for dp4a MMQ kernels when FP16 tensor cores are available.
```

From `ggml/src/ggml-cuda/ggml-cuda.cu`, the per-op dispatch:

```c
bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear
    && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
    && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
```

And from `ggml/src/ggml-cuda/mmq.cu` (`ggml_cuda_should_use_mmq`):

```c
if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
    return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
```

(That return value is "use the dp4a MMQ path"; when `false`, MMA is selected.)

### 1e. Putting the thresholds together — Ampere routing

| Active batch B | llama.cpp path | dotLLM today |
|---|---|---|
| 1     | MMVQ (GEMV, dp4a) | MMVQ-large + pre-Q8_1 (matches structurally) |
| 2..8  | MMVQ              | Same kernel, no special path |
| 9..63 | MMQ dp4a (batched, no tensor cores) | **No path** |
| 64+   | MMQ mma (tensor cores) | **No path** |

The interesting takeaway: **dp4a-based batched MMQ at B = 9..63 is a
distinct optimisation tier from mma**. If we ever build out a batched path,
the dp4a-batched-MMQ may be more impactful than mma — most realistic server
batches sit in the 8..32 range, not ≥64.

### 1f. The mma instruction itself

Used by llama.cpp on sm_75+ (Turing) and reused on sm_80/86/89/90:

```
mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32
```

This consumes:
- A: 16×16 INT8 row-major (held across the warp's 32 threads, 4 INT8s per
  thread per A fragment)
- B: 8×16 INT8 column-major (2 INT8s per thread)
- C: 16×8 INT32 (4 ints per thread)

Each warp produces a 16×8 INT32 output tile per `mma.sync` call. Per dp4a:
warp produces 32 INT32 outputs (1 per thread). So **per-warp throughput
ratio = 128 / 32 = 4× more outputs per warp from mma**, on top of the
arithmetic-density advantage from working on a 16×K tile rather than a
single dot product.

---

## 2. dotLLM applicability

### 2a. Today: B = 1, single-stream decode

`src/DotLLM.Engine/TextGenerator.cs` is the one and only generation path. It
calls `_model.Forward(...)` with a single token at every decode step. There
is no batched-forward API on `IModel`. There is no scheduler. `IScheduler`
exists in `src/DotLLM.Engine/IScheduler.cs` as an interface contract with a
`SchedulerMetrics` struct, but **no implementation exists in the repo**
(grep for `: IScheduler` returns zero matches; `ContinuousBatching` matches
only docs and roadmap, no code).

**MMA value to dotLLM today: zero.** No batched workload is produced
anywhere in the engine.

### 2b. Speculative decoding (already on the branch)

`SpeculativeDecoder.DraftAndVerify` already runs target verification with a
batch of K candidate tokens. K defaults to 5 (`_speculativeCandidates = 5`
in `TextGenerator`); the implementation accepts up to ~8 in practice.

This **is** a real batched-forward call, but it has three constraints:

1. **B = K = 4–8** sits in MMVQ territory (B ≤ 8). llama.cpp would not
   activate even the dp4a-batched MMQ at this size, let alone mma.
2. Greedy-only today (`IsEffectivelyGreedy(options)` gate in
   `TextGenerator`). Wave 8 would lift this, but the batch size doesn't
   change.
3. Speculative + graph capture is not yet on (Open §4 in `.continue-here.md`).

For mma specifically, speculative decoding is a **non-trigger**. It does
not get into the M ≥ 16 regime where mma starts winning meaningfully.

### 2c. Continuous batching (Phase 9, step 35 — not implemented)

This is the realistic mma trigger. From `docs/SCHEDULING.md`:

> Each scheduler iteration:
> 1. Check completions: Sequences hitting EOS/max tokens/stop conditions → evict, free KV blocks.
> 2. Admit new requests: Fill freed capacity from the priority queue.
> 3. Prefill: For newly admitted sequences, process full prompt tokens (batch prefill).
> 4. Decode: For all active sequences, generate one token each (batched decode).

In step 4, the scheduler aggregates **B active sequences** into a single
forward pass. Production server batches typically run B = 8–32 depending on
KV-cache headroom and request mix. At B = 16, the MlpUp GEMV becomes a
24576 × 4096 × 16 GEMM — **mma's natural sweet spot**.

But this requires:

- An `IScheduler` implementation (continuous batching loop, sequence state
  machine, KV-block admission, completion eviction, sampler-per-sequence
  fan-out).
- A batched-forward signature on `IModel` (`Forward(span<spans-of-tokens>,
  span<spans-of-positions>, span<kvCaches>)` or equivalent fan-out).
- Per-sequence sampling pipelines run after a single forward (already exist
  per-request, just need to be invoked B times per step from disjoint logit
  rows).
- Paged KV-cache (step 36) so concurrent sequences don't fragment memory.

That is a **multi-week engine project**, and mma is one of many kernels
needed under it (also: the KV-cache concat path, the attention kernel needs
batched seq_kv, etc.).

### 2d. Dynamic batching across multi-request server

Same as 2c, just a slightly different admission policy (window-based vs
iteration-based). Same kernel requirements. Production batch sizes B = 4–32.

### 2e. Per-case feasibility / cost / kernel speedup summary

| Trigger | Realistic B | Feasible today? | Kernel-only speedup vs current dp4a | E2E throughput uplift |
|---|---|---|---|---|
| Single-stream decode (today) | 1 | n/a | 0% (mma loses at B=1) | 0% |
| Speculative + graph (post Open §4 fix) | 4–8 | ~1 week eng | Marginal: mma maybe break-even at B=8, often slower due to M=8/16 waste | <5% |
| Continuous batching (step 35) | 8–32 | Multi-week eng | dp4a-batched-MMQ: ~2× over per-request dp4a (amortised weight reads). mma on top: another 1.3–1.8× | **Realistic +30–60%** in server scenarios; 0% in single-user CLI |
| Dynamic batching (server) | 8–32 | Same as continuous batching | Same | Same |

**Critical observation**: even when continuous batching lands, the **first**
big win is the dp4a-batched MMQ kernel (re-using llama.cpp's MMQ-dp4a
structure, which keeps weight reads coalesced across multiple sequences). mma
is the *second* lever, only kicking in meaningfully at the high end of
production batch sizes.

---

## 3. Recommended path

### 3a. Step ordering

1. **Continuous batching scheduler** (`IScheduler` impl, batched-forward on
   `IModel`, paged KV-cache integration). Effort: **3–6 weeks**. Unlocks any
   batched kernel work. Risk: medium-high — touches engine architecture,
   sampler pipeline fan-out, KV memory model, prefix cache integration.
2. **Batched-MMQ-dp4a kernels** (single weight read amortised across B
   active sequences; `mul_mat_q` analog from llama.cpp, dp4a only). Effort:
   **1–2 weeks** per quant type. Risk: low — same dp4a primitive we already
   use, just multi-row M instead of M=1. Expected speedup at B=16:
   **~2×** versus running B independent GEMVs.
3. **mma-MMQ kernels** (tensor-core variant of #2 for B ≥ 32-ish). Effort:
   **2–3 weeks** per quant type, plus the load-tile path for K-quants which
   is fiddly. Risk: medium — fragment layout debugging, ldmatrix vs manual
   loads, INT8 saturation behaviour. Expected speedup at B=64:
   **~1.5–1.8×** over batched dp4a; at B=16 break-even or slightly worse.
4. **Dispatcher gate** that picks the right kernel per launch based on B
   (mirrors llama.cpp's MMVQ_MAX_BATCH_SIZE / MMQ_DP4A_MAX_BATCH_SIZE
   logic). Effort: **1 day** — trivial once the kernels exist.

### 3b. End-to-end perf model

Take Qwen3-8B Q4_K_M as the anchor. Single-user decode is at 30.9 tok/s and
**already inside llama.cpp's range**.

- B = 1 (current): 30.9 tok/s per sequence, **30.9 tok/s total**.
- B = 4 server, dp4a-batched-MMQ: ~22 tok/s per seq (weight reads share
  modestly), **~88 tok/s aggregate**.
- B = 16 server, dp4a-batched-MMQ: ~14 tok/s per seq, **~225 tok/s aggregate**.
- B = 16 server, mma-MMQ: ~18 tok/s per seq (faster compute), **~290 tok/s
  aggregate**. **Δ vs dp4a-batched ~30%.**
- B = 32 server, mma-MMQ: ~12 tok/s per seq (KV / attention bottleneck
  takes over), **~380 tok/s aggregate**. Δ vs dp4a-batched ~40%.

The **mma-specific lift** is +30–40% on top of a properly batched dp4a
kernel. The total step from "no batching" to "batching + mma" is much
larger (~10× aggregate throughput on 16-way concurrent server load), but
**most of that comes from the scheduler, not the mma kernel.**

### 3c. Honest sizing of the mma upper bound

To be blunt: **mma is a refinement on top of the engine work, not a
standalone win**. The work is meaningful when:

1. dotLLM is being deployed as a server, not embedded as a CLI.
2. The server consistently runs B ≥ 16 active sequences.
3. The user's bottleneck is aggregate throughput, not single-request latency.

For a single-user developer workflow (which is dotLLM's most common
demonstrated use case today, per the bench scripts and `samples/`),
**mma does not apply**.

---

## 4. Recommended single next change

**B — SKIP the mma kernel work entirely until continuous batching lands.**

### Per-option rationale

#### A) Write the mma kernel as a microbenchmark, gate it off, validate on synthetic batched inputs

- **Effort**: 1 week (write kernel, validate vs dp4a reference, microbench).
- **End-user perf today**: 0% — no caller exists.
- **Future leverage**: kernel exists when the scheduler arrives; a contributor
  to the scheduler doesn't have to also write the kernel.
- **Risk of bitrot**: high — without integration, the kernel may not even
  match the eventual batched-forward calling convention. mma kernel APIs
  depend on how the batched activation buffer is laid out (interleaved vs
  stacked), and that's a scheduler-level decision.
- **Recommendation: SKIP.** The leverage isn't worth the risk of building
  the wrong kernel. Speculative kernel work on a non-existent engine layer
  is a classic premature-optimisation trap.

#### B) Wait — implement nothing until continuous batching lands

- **Effort**: 0 (today).
- **Future cost**: pay the kernel cost when it's actually needed.
- **Recommendation: DO.** This is the right answer. The engine work is the
  bottleneck and should be scoped, designed, and started without a kernel
  dependency.

#### C) Implement speculative-decoding extension first (small batch B = K = 4–8)

- **Speculative decode is already greedy-on this branch**. The remaining
  work is the Wave 8 / issue #121 pipeline-aware acceptance — that's an
  engine change, not a kernel change.
- B = K = 5 (default) does **not** activate mma in any reasonable
  threshold model (M = 5 / 16 = 31% lane utilisation, well under the
  dp4a crossover).
- Even if speculative-decode ends up using a batched verification kernel,
  that kernel should be a small-batch dp4a-MMQ variant, not mma.
- **Recommendation: SKIP for the mma question.** Speculative is worth
  finishing for its own (greedy-lift) reasons, but it does not unlock mma.

#### D) Other — implement dp4a-batched MMQ first (without mma), gate by B ≥ 4

- The dp4a-batched-MMQ has a meaningful payoff at B = 4..32, well below
  mma's threshold.
- Same kernel structure as the existing single-row dp4a; the change is
  expanding the per-block tile to (M_block, N_block) outputs and amortising
  weight reads across M.
- Effort: ~1 week per quant type (Q4_K, Q5_K, Q6_K).
- **Still requires** the scheduler to produce B ≥ 4 — same blocker as A
  and the mma path.
- **Recommendation: SKIP for now, but rank it ABOVE mma when batching
  arrives.** It's the correct kernel to write **first** when there's a
  caller for it.

### Final ranking

1. **DO**: Land continuous batching in the engine (option B = wait + redirect
   effort to the actual blocker).
2. **DO**: When batching is producing real B ≥ 4 batches, add the
   dp4a-batched MMQ kernel (option D).
3. **DO**: When server deployments demonstrably run B ≥ 16 consistently, add
   the mma MMQ kernel (the topic of this note).
4. **SKIP**: Writing the mma kernel speculatively today (option A).

---

## 5. Optional tiny prototype

**Not implemented in this commit.** Per option B above, a prototype today
would be premature work without a caller. The kernel author's time is more
valuably spent on:

- The CUDA MLA Phase 1 work currently in flight on the parallel agent (real
  end-user perf for DeepSeek-V2/V3 on CUDA), or
- Engine-layer scheduling design (the actual blocker for any batched kernel
  work).

If a future session decides to override this recommendation and write the
prototype anyway (option A), the entry point should be:

- File: `native/kernels/quantized_gemm_mma_q4_k.cu` (new — sibling to
  `quantized_gemv_mmq.cu`, NOT a replacement).
- Signature: `void quantized_gemm_q4_k_mma_b8(const uint8_t* weight,
  const int8_t* x_q8_1, const half* x_d, half* y, int n, int k, int batch)`.
- Tile: M = 16 (one mma fragment), N_per_block = 16 (so 16 output rows per
  block), K iterated in chunks of 16 INT8s (one mma.m16n8k16 step).
- One warp per mma; 4 warps per block (128 threads) for occupancy parity
  with the existing MMVQ-large kernel.
- Validation: synthetic random batched inputs, compare against
  `quantized_gemv_q4_k_mmq_preq` called B times in a loop. INT8 noise
  budget: same ≤1% peak-relative drift as the existing MMQ tests.
- Microbench harness: extend `Q4KGemvVariants` (proposed in MLPUP_GEMV_GAP)
  to a `Q4KGemmBatched` variant that sweeps B = 1, 4, 8, 16, 32, 64.

The prototype is documented as a follow-up; **not** done here.

---

## 6. Constraints and engineering risk for option B

If the recommendation (option B = wait) is accepted, the engine work
sequencing is:

1. **`IScheduler` skeleton** — port the loop in `docs/SCHEDULING.md` to a
   real implementation behind the existing interface. Single-sequence
   smoke test first. Effort: 1 week.
2. **Batched `Forward`** on `IModel` — new method
   `Forward(ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions,
   ReadOnlySpan<int> sequenceIds, IReadOnlyList<IKvCache> kvCaches)`,
   where `tokens.Length` is the total batch size and `sequenceIds[i]`
   selects the per-token KV cache. CUDA backend will need batched RoPE,
   batched KV-write, and a per-sequence attention dispatch. Effort:
   2–3 weeks.
3. **Paged KV-cache integration** — use the existing `PagedKvCache`
   (already in repo at `src/DotLLM.Engine/KvCache/PagedKvCache.cs`)
   as the multi-sequence backing store. Block-table indirection lets
   us pack multiple sequences contiguously. Effort: 1 week.
4. **Sampler fan-out** — run `SamplerPipeline.Sample` once per active
   sequence on disjoint logit rows. Existing per-request pipeline
   already supports this; just needs to be driven from the scheduler.
   Effort: <1 week.
5. **Then** kernel work (option D, then this note's mma kernel).

Total: **~6–8 weeks** of engine work before mma becomes addressable.
That work is high-value on its own — every server-class deployment
needs continuous batching regardless of mma — so it is **not** wasted
even if mma never lands.

---

## 7. References

### llama.cpp source (MIT — cited, not redistributed)

- [`ggml/src/ggml-cuda/mmvq.cuh`](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cuh)
  — `MMVQ_MAX_BATCH_SIZE = 8` define.
- [`ggml/src/ggml-cuda/mmq.cuh`](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cuh)
  — `MMQ_DP4A_MAX_BATCH_SIZE = 64` define; `MMQ_NWARPS = 8`,
  `MMQ_TILE_NE_K = 32`, MMQ tile constants for the mma kernel.
- [`ggml/src/ggml-cuda/mmq.cu`](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cu)
  — `ggml_cuda_should_use_mmq` dispatch logic.
- [`ggml/src/ggml-cuda/ggml-cuda.cu`](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu)
  — `ggml_cuda_mul_mat` MMVQ-vs-MMQ entry-point dispatch.
- [`ggml/src/ggml-cuda/mma.cuh`](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-cuda/mma.cuh)
  — `mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32` PTX, Turing/Ampere
  fragment layout helpers.

### NVIDIA documentation

- [PTX ISA — Warp-level matrix instructions (mma)](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions)
- [PTX ISA — `mma.sync` for INT8](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-fragments-for-mma-m16n8k16-mma-m16n8k32)

### dotLLM source

- `docs/perf/MLPUP_GEMV_GAP.md` — predecessor note (closed the batch=1 gap).
- `src/DotLLM.Engine/IScheduler.cs` — interface contract; **no impl exists**.
- `src/DotLLM.Engine/TextGenerator.cs` — the only generation path today
  (single-request, B=1).
- `src/DotLLM.Engine/SpeculativeDecoder.cs` — produces small batches
  (K=4–8) via target-model verification; greedy-only today.
- `src/DotLLM.Engine/KvCache/PagedKvCache.cs` — paged KV available, ready
  for multi-sequence integration.
- `native/kernels/quantized_gemv_mmq.cu` — current MMQ kernels; mma kernel
  would sit alongside as a sibling, not replace.
- `src/DotLLM.Cuda/CudaKernels.cs` `LaunchQuantizedGemvMmq` /
  `HasMmqQ4K` — the dispatcher that would gate by B once both kernels
  exist.
- `docs/SCHEDULING.md`, `docs/ROADMAP.md` step 35 — engine layer that
  must land before mma is addressable.
