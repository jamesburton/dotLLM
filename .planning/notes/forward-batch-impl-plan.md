---
topic: IModel.ForwardBatch backend overrides — implementation plan
status: drafting
date: 2026-05-17
sequence_item: 5
predecessor: .continue-here-scheduler-batched-forward.md
---

# Forward batch override — implementation plan

## Real scope vs. handoff estimate

`.continue-here-scheduler-batched-forward.md` estimated "~1d" for CPU + Vulkan. A
close reading of `TransformerModel.Forward` (1509 LoC, multiple precision paths,
multiple architectural branches) puts true scope at 2-3 days for a complete,
parity-tested implementation across every supported precision and architecture.

The throughput win is real and worth pursuing, but it benefits from a phased
approach: each phase ships a contained, parity-tested chunk and can be merged
independently.

## Architectural simplification — fall-back path

The cleanest way to land this without rewriting `Forward`: implement
`ForwardBatch` so that for **simple** sequences it fuses matmuls, and for
**complex** sequences it falls back to the existing per-seq loop. Definition of
"simple":

| Property | Simple (fuse) | Complex (per-seq loop) |
|---|---|---|
| Architecture | GQA / MHA / MQA dense | MLA, MoE |
| LoRA | none, OR same adapter across all simple seqs | per-seq adapter ≠ |
| KV-cache type | standard | quantized? (TBD — likely simple too) |
| Sliding window | uniform across seqs | (no current code path differs) |

Sequences are partitioned into a "simple subgroup" and a "complex subgroup".
The simple subgroup runs through the new batched path; the complex subgroup
falls through to the per-seq loop. This bounds the scope of new code to the
simple path while preserving correctness on edge cases.

For the very first cut, treat even LoRA-active sequences as complex (per-seq
fallback) and only fuse the LoRA-free GQA case. LoRA fusion is a follow-up.

## CPU phases

### Phase 5a — Embedding + lm_head fusion *(small, low-risk, ~80 LoC + ~150 test LoC)*

Override `ForwardBatch` so that:

1. **Embedding lookup** is one big call: concat per-seq `tokenIds` into a single
   buffer of length `Σ N_i`, look up all rows at once, write to the batched
   hidden state.
2. **Each transformer layer is still a per-seq inner loop** — for each
   sequence, slice the batched hidden buffer, run the existing layer code on
   the slice with the seq's own positions / KV cache / adapter, write the
   slice back.
3. **Final RMSNorm + lm_head** is one big GEMM at `[Σ N_i, hidden] × [hidden,
   vocab]`. Split logits back per-seq.

This skips the hard part (batched matmuls inside the layer loop) but lands
the embedding-batch and lm_head-batch wins. The lm_head batched GEMM has
real impact for prefill bursts.

Parity contract: byte-identical logits vs. the default per-seq loop. Tests
in `tests/DotLLM.Tests.Unit/Models/Architectures/TransformerModelForwardBatchTests.cs`
covering 1/2/4 sequences, decode and prefill, with and without KV cache.

**Commit cadence**: one commit. Ships with the parity test.

### Phase 5b — Batched Q/K/V + O + gate/up + down projections (LoRA-free GQA only) *(medium, ~250 LoC + ~150 test LoC)*

For the GQA non-LoRA non-MoE non-MLA case:

1. Concat per-seq hidden states into a single batched buffer of size
   `[Σ N_i, hidden]` at the **start of each layer**.
2. Run **one batched RMSNorm** over the full buffer.
3. Run **one batched QuantizeInput** if quantized weights.
4. Run **one batched Q/K/V GEMM** at `[Σ N_i, hidden] × [hidden, dim]`.
5. **Split** Q/K/V back per-seq for RoPE + attention (each seq has its own
   positions and KV cache — attention stays per-seq).
6. **Concat** attention output back, run batched O projection + residual.
7. Same pattern for the FFN block (RMSNorm + gate/up GEMM + SwiGLU + down GEMM
   + residual).

LoRA-active sequences fall back to per-seq loop for that sequence. So does any
MLA/MoE layer.

Parity contract extension: byte-identical logits vs. per-seq loop for GQA
non-LoRA models. Existing decode + prefill paths must remain bit-equal.

The matmul fusion win shows up here — at `Σ N_i = 4 × decode = 4 tokens`, the
single `[4, hidden] × [hidden, dim]` is ~4× faster than four `[1, hidden]`
GEMVs (much better cache utilisation + thread-pool amortisation).

**Commit cadence**: one commit. Ships with the parity test extension.

### Phase 5c — LoRA fusion in batched path *(later, deferred)*

Only fuse LoRA when all active sequences share the same adapter (cheap check).
Otherwise per-seq. Probably worth ~80 LoC; defer until a real use-case surfaces.

### Phase 5d — MLA / MoE in batched path *(much later)*

Both have substantial per-token complexity (cache shape for MLA; routing for
MoE). Fold these in if/when a measured workload demands them.

## Vulkan phases (path-1, per-seq attention loop)

### Phase 5e — Vulkan embedding + lm_head fusion *(small, ~80 LoC + ~50 test LoC)*

Mirror Phase 5a but on `VulkanTransformerModel`. Existing kernels already
accept `[N, hidden]` for both `EmbeddingF32Kernel` and the matmul kernels — so
the batched lm_head call is mechanical.

**Commit cadence**: one commit per host (dense `VulkanTransformerModel`
first; the other 3 hosts can be follow-ups since most production load goes
through dense).

### Phase 5f — Vulkan batched Q/K/V + O + gate/up + down *(medium, ~300 LoC + ~150 test LoC)*

Mirror Phase 5b on Vulkan. Per-seq RoPE + attention dispatch. Position-offset
uniform passed per attention call.

Major risk surface: **descriptor-set lifetime per sub-dispatch**. The existing
kernel's `InvalidateDescriptorCache` is called when scratch buffers re-allocate
— per-sub-dispatch slicing within a single ForwardBatch must not invalidate.
Confirm by reading the `MatMulF32Kernel.Record` implementation before coding.

**Commit cadence**: one commit per host.

### Phase 5g — Vulkan path-2 (block-table attention) *(much later)*

vLLM-style one-attention-kernel-reads-block-tables. Significant new shader.
Defer until path-1's perf data justifies it.

## Test surface

New test class: `tests/DotLLM.Tests.Unit/Models/Architectures/TransformerModelForwardBatchTests.cs`

Fixtures:
- 1-seq batch → must equal Forward
- 2-seq batch, decode (1 token each)
- 2-seq batch, prefill (≠ token counts)
- 4-seq batch, mixed decode+prefill
- with and without KV cache
- with and without (Phase 5c) LoRA
- F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K weight precisions

Tolerance: **byte-identical** (the batched path must reproduce the per-seq loop
exactly when sequences are independent at attention; the only "fusion" is in
matmul tile boundaries, which produce the same FP result for the same inputs
when the inner accumulation order is preserved).

If matmul tile boundaries drift the result by FP-noise, relax to abs 1e-5 /
rel 1e-4 with documented reason.

## Out of scope for items 5/5b

- Streaming yield per scheduled sequence (separate Gap 3).
- Per-sequence `IncrementalDetokenizer` (Gap 4).
- Prefix-cache trie (already shipped — Phase 9 Step 37).
- Preemption / VRAM-pressure swap (item 7 / Phase 9 Step 59).

## Recommendation for this session

**Land Phase 5a + 5e** (embedding + lm_head fusion CPU + Vulkan dense host),
parity-tested. Two commits. ~300 LoC total + ~250 LoC tests.

Phase 5b + 5f (the bigger matmul-fusion win) is a follow-up session; the plan
above is enough to pick it back up.

Rationale: the embedding + lm_head fusion is real, mechanically simple,
parity-easy, and ships measurable benefit (the lm_head GEMM is one of the
biggest single layer costs at prefill). The matmul-fusion inside the transformer
block is where the complexity is — better to land it deliberately in its own
session with proper attention to the per-precision paths.
