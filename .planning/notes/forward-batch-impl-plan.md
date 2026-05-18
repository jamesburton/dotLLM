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

---

## Phase 5a — code-level refactor (CPU)

Scope re-cut after a closer reading: skip batched embedding (it's a
row-gather, not a GEMM — minimal savings) and target only the lm_head
fusion. That keeps the refactor surface to two private helpers + a
ForwardBatch override.

### Step 1 — extract two private helpers from `TransformerModel.Forward(...)`

The current public `Forward(tokenIds, positions, deviceId, kvCache)` body
(line 337-1005 of `src/DotLLM.Models/Architectures/TransformerModel.cs`)
splits naturally at line 988 (after the final RMSNorm) into:

- **lines 340-988**: embedding + transformer layer loop + final RMSNorm
  → leaves the final hidden state at `_state.HiddenState[0..seqLen*hiddenSize]`.
- **lines 990-1004**: lm_head GEMM + tensor allocation + copy + return.

Extract those two into private helpers:

```csharp
// Existing body lines 340-988 verbatim, minus the deviceId param (only
// used in the tensor allocation, which has moved to RunLmHead). All
// adapter / KV-cache / MLA / MoE / LoRA branches stay exactly as-is.
private unsafe void RunLayersAndFinalNormCore(
    ReadOnlySpan<int> tokenIds,
    ReadOnlySpan<int> positions,
    IKvCache? kvCache)
{
    int maxSeq = Config.MaxSequenceLength;
    for (int i = 0; i < positions.Length; i++)
    {
        if ((uint)positions[i] >= (uint)maxSeq)
            throw new ArgumentOutOfRangeException(nameof(positions), ...);
    }

    int seqLen = tokenIds.Length;
    int hiddenSize = Config.HiddenSize;
    /* ... rest of original Forward body up to and including step 3 (final RMSNorm) ... */
}

// Existing lines 990-1004 verbatim.
private unsafe ITensor RunLmHead(int seqLen, int deviceId)
{
    int vocabSize = Config.VocabSize;
    float* hidden = (float*)_state.HiddenState;
    float* logits = (float*)_state.Logits;

    var rwOutput = _weights.RepackedOutput ?? default;
    GemmInterleaved(_weights.OutputWeight, _weights.OutputQuantType,
        hidden, logits,
        _weights.OutputOutputDim, _weights.OutputInputDim, seqLen,
        null, in rwOutput);

    var shape = new TensorShape(seqLen, vocabSize);
    var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId);
    new Span<float>(logits, seqLen * vocabSize).CopyTo(
        new Span<float>((void*)result.DataPointer, seqLen * vocabSize));
    return result;
}

// Refactored public method — three lines.
public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                       int deviceId, IKvCache? kvCache)
{
    RunLayersAndFinalNormCore(tokenIds, positions, kvCache);
    return RunLmHead(tokenIds.Length, deviceId);
}
```

The LoRA-aware `Forward(..., adapter)` overload (lines 298-325) is
unchanged — it sets `_currentAdapter` then delegates to the no-adapter
Forward as before.

### Step 2 — `ForwardBatch` override

```csharp
public IReadOnlyList<ITensor> ForwardBatch(
    IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
{
    ArgumentNullException.ThrowIfNull(requests);
    if (requests.Count == 0) return Array.Empty<ITensor>();
    if (requests.Count == 1)
    {
        var r0 = requests[0];
        return new[] { Forward(r0.TokenIds.Span, r0.Positions.Span,
                               deviceId, r0.KvCache, r0.Adapter) };
    }

    int hiddenSize = Config.HiddenSize;
    int vocabSize = Config.VocabSize;
    int totalTokens = 0;
    foreach (var r in requests) totalTokens += r.TokenIds.Length;

    _state.EnsureCapacity(totalTokens);

    // Rent a managed snapshot buffer — the per-seq RunLayersAndFinalNormCore
    // calls all write to _state.HiddenState[0..N_i*hidden], which the next
    // seq's call would clobber. We snapshot each seq's hidden out, then
    // stack them at the end for a single batched lm_head GEMM.
    var pool = ArrayPool<float>.Shared;
    float[] batched = pool.Rent(totalTokens * hiddenSize);
    try
    {
        int tokOffset = 0;
        foreach (var r in requests)
        {
            int n = r.TokenIds.Length;
            if (r.Adapter is not null)
            {
                ValidateAdapterForModel(r.Adapter);
                LoraStage2.PrewarmAdapter(r.Adapter as LoraAdapter);
                _currentAdapter = r.Adapter;
            }
            try
            {
                RunLayersAndFinalNormCore(r.TokenIds.Span, r.Positions.Span, r.KvCache);
                new Span<float>((float*)_state.HiddenState, n * hiddenSize)
                    .CopyTo(batched.AsSpan(tokOffset * hiddenSize, n * hiddenSize));
            }
            finally
            {
                if (r.Adapter is not null) _currentAdapter = null;
            }
            tokOffset += n;
        }

        // Copy the stacked snapshot back into _state.HiddenState so the existing
        // GemmInterleaved API consumes it from the expected location.
        batched.AsSpan(0, totalTokens * hiddenSize)
            .CopyTo(new Span<float>((float*)_state.HiddenState, totalTokens * hiddenSize));

        // One batched lm_head GEMM at seqLen = ΣN_i. _state.Logits is sized for
        // this by the EnsureCapacity(totalTokens) above.
        var rwOutput = _weights.RepackedOutput ?? default;
        GemmInterleaved(_weights.OutputWeight, _weights.OutputQuantType,
            (float*)_state.HiddenState, (float*)_state.Logits,
            _weights.OutputOutputDim, _weights.OutputInputDim, totalTokens,
            null, in rwOutput);

        // Split logits per-seq.
        var results = new ITensor[requests.Count];
        int srcOff = 0;
        float* logitsPtr = (float*)_state.Logits;
        for (int i = 0; i < requests.Count; i++)
        {
            int n = requests[i].TokenIds.Length;
            var shape = new TensorShape(n, vocabSize);
            var tensor = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId);
            new Span<float>(logitsPtr + (long)srcOff * vocabSize, n * vocabSize).CopyTo(
                new Span<float>((void*)tensor.DataPointer, n * vocabSize));
            results[i] = tensor;
            srcOff += n;
        }
        return results;
    }
    finally
    {
        pool.Return(batched);
    }
}
```

### Risks + edge cases

- **`EnsureCapacity` shrinkage**: confirm `_state.EnsureCapacity(N)` is
  grow-only (does not shrink). If not, the per-seq RunLayersAndFinalNormCore
  loop will shrink the buffer mid-batch. Mitigation: call EnsureCapacity at
  the start (with totalTokens) and the per-seq calls re-check but won't
  shrink. Quick read of `Mamba3ForwardScratch.EnsureCapacity` shows it IS
  grow-only — same pattern should hold in the dense-transformer state. **Verify
  before merging.**
- **Adapter handling**: each seq sets/clears `_currentAdapter` independently
  inside the loop. The lm_head step has no adapter (lm_head is never a
  LoRA target). No conflict.
- **MLA layers**: per-seq `_mlaKvState.Reset()` triggers when `positions[0] == 0`.
  That semantics carries through unchanged via the per-seq call.
- **Quantized models**: the lm_head uses the same `GemmInterleaved` path. Q8_0
  / Q4_K / Q5_K / Q6_K weights work identically at seqLen = totalTokens — the
  GEMM tile loop iterates over the row dimension which is `totalTokens` here.
- **Empty seq (N_i == 0)**: should never happen (scheduler invariant) but
  the loop handles it correctly (zero-size copy + zero-size lm_head row).
- **Memory pressure**: the ArrayPool rental is `totalTokens × hiddenSize × 4`
  bytes. At hidden=2048, totalTokens=64 that's 512 KB — well within ArrayPool's
  large-pool capacity. For very large batches it falls back to a fresh allocation,
  acceptable.

### Parity test surface

New file: `tests/DotLLM.Tests.Unit/Models/Architectures/TransformerModelForwardBatchTests.cs`

Use the existing synthetic safetensors fixture pattern from
`TransformerModelMlaForwardTests.cs` lines 39-50 (per-test scratch dir,
clean-up Dispose) but with a tiny GQA model (HiddenSize=16, NumLayers=2,
NumHeads=2, NumKvHeads=2, VocabSize=8, IntermediateSize=24, no MLA, no
MoE).

Required tests:
1. `ForwardBatch_SingleSeq_EqualsForward` — 1-seq batch matches Forward bit-exact.
2. `ForwardBatch_TwoSeqs_MatchesPerSeqLoop` — 2-seq batch: each seq's logits
   equal a separate Forward call's logits, bit-exact.
3. `ForwardBatch_MixedPrefillDecode_MatchesPerSeqLoop` — 1 seq with 4 tokens
   + 1 seq with 1 token, both with KV caches at non-zero positions.
4. `ForwardBatch_F32Weights` `_F16Weights` `_Q8_0Weights` — same parity check
   across the lm_head's main precision paths (only one parametrised test
   needed, take `Theory(MemberData)` on the weight DType).
5. `ForwardBatch_LoRAActiveOnOneSeq` — only one of the two sequences passes
   a LoRA adapter; both produce correct logits.

Tolerance: byte-identical for non-quantized; abs 1e-5 rel 1e-4 for quantized
(the GEMM tile boundary at the batched lm_head shouldn't drift this, but
guard with a small envelope just in case).

## Phase 5e — Vulkan dense host (code-level refactor)

**Finding from the Phase 5a implementation pass (2026-05-18):** Vulkan's
existing `Forward` is structured *differently* from CPU's Forward —
the Vulkan path only runs the lm_head GEMM on the LAST token
(`seqLen=1`) and returns `[1, vocab]`, while CPU runs it for ALL
positions and returns `[seqLen, vocab]`. See
`VulkanTransformerModel.cs` lines 1153-1183: the final block copies
the last hidden row into `_state.NormOutput`, runs RMSNorm on that
1-row buffer, dispatches `RecordMatmul` with `seqLen: 1`, downloads
`[vocab]` floats to host.

That changes the Phase 5e win shape: instead of fusing a
`[Σ N_i, hidden] × [hidden, vocab]` GEMM (which on CPU saves a lot
because Σ N_i can be hundreds-of-tokens for prefill batches), Vulkan
would only fuse a `[N, hidden] × [hidden, vocab]` GEMM (one row per
sequence, since each seq contributes only its last-token hidden).

For typical N ≤ 8 active sequences and vocab ≈ 32k, that's saving
N-1 small `[1, hidden] × [hidden, vocab]` dispatches. Each costs
~20-50 µs on Strix Halo (matmul kernel call overhead + per-dispatch
barriers); total save ~150-350 µs per batch step. Compared to the
~5-10 ms per-step prefill cost or 12-15 ms per-step decode cost,
that's noise-floor savings.

**Decision: defer Phase 5e until Phase 5f (intra-block matmul fusion)
lands.** The intra-block matmuls (Q/K/V/O/gate/up/down) are the real
Vulkan win for batched scheduling, not the trailing lm_head. The
default `ForwardBatch` implementation on `IModel` (per-seq Forward
loop) is correct on Vulkan — just not optimal.

If Phase 5e were to ship in isolation:

Mirror Phase 5a on `src/DotLLM.Vulkan/VulkanTransformerModel.cs`. The
Vulkan side has a similar structure — a single `Forward` method that
ends with the lm_head GEMM dispatched via the appropriate matmul kernel.

Key difference: Vulkan kernels already accept arbitrary `[N, hidden]`
inputs, so the batched lm_head dispatch is mechanical:

```csharp
public override IReadOnlyList<ITensor> ForwardBatch(...)
{
    // 1. Per-seq RunLayersAndFinalNormCore — each writes to its own
    //    output device buffer (allocate one per seq up front, or stage
    //    through a shared scratch and copy out per seq).
    // 2. Stage N device buffers into a single packed [ΣN_i, hidden]
    //    device buffer (vkCmdCopyBuffer × N).
    // 3. Single matmul dispatch on the packed buffer.
    // 4. Read back per-seq slices into per-seq ITensors.
}
```

Risk: descriptor-set invalidation across the per-seq RunLayersAndFinalNormCore
calls. The existing kernels' `InvalidateDescriptorCache` should not fire
mid-batch (the bound buffers don't change), but verify by reading
`MatMulF32Kernel.Record` invalidation conditions before coding.

Parity test: mirror Phase 5a tests but for `VulkanTransformerModel`.

## Verification checklist before merging Phase 5a + 5e

- [ ] All existing TransformerModel tests still pass (1426+ unit tests).
- [ ] All existing Vulkan model tests still pass (559 with Strix Halo
      kernels green).
- [ ] New parity tests cover the 5 cases listed above for each backend.
- [ ] No allocations added to the per-seq hot path (the ArrayPool rental
      is per-batch, not per-token).
- [ ] Run `LoraMacroBenchmarks` once with `MaxActiveSequences=1` vs `=4`
      to quantify the win on this host (it won't be Strix Halo numbers,
      but a directional check that ForwardBatch isn't slower than the
      loop on a tiny test model).
