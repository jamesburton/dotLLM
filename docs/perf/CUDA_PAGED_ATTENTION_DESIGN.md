# CUDA native (non-staging-buffer) paged decode attention — design note

**Issue:** #200
**Author:** dotnet-perf-expert agent, 2026-08-01
**Status:** Design only — no kernel code in this commit. Confirms and extends the two prior
comments on #200 (2026-07-27); nothing material has changed since.
**Predecessor pattern:** `docs/perf/MMA_BATCHED_MMQ.md` — same shape of problem (a real kernel
idea with a missing engine-side producer), same conclusion structure.

## TL;DR

Re-verified as of 2026-08-01: **the prerequisite chain #200's own comments identified on
2026-07-27 is unchanged.** `IModel.ForwardBatch` continuous batching is real and sophisticated at
the Engine layer (`ContinuousBatchScheduler` — chunked prefill, priority+preemption,
disaggregation, fairness, all shipped per Step 59), but CUDA has **no fused `ForwardBatch`
override** (`CudaTransformerModel.ForwardBatch` does not exist — confirmed by grep, zero matches)
and **no paged KV-cache** (`CudaPagedKvCache`/`CudaKvBlockPool` do not exist — confirmed by grep,
zero matches). `docs/SCHEDULING.md`'s own `ForwardBatch` status table still says, verbatim,
*"CUDA: per-seq fallback. Same mirror needed when a CUDA host is available."* `docs/KV_CACHE.md`
still says *"CPU `PagedKvCache` only. CUDA and hybrid models fall back to their native
KV-cache."* No dedicated tracking issue for "CUDA multi-sequence batched decode" or "CUDA paged
KV-cache" exists in the repo (checked `gh issue list`); both are only Future-Considerations rows
in `docs/ROADMAP.md`.

**Recommendation: still HOLD on implementation.** This note also directly answers the extra
question this session was asked to investigate — *"could a kernel-only, testable-in-isolation
version be built today against a synthetic block table, ahead of the real CUDA paged KV-cache
infra?"* — with a reasoned **no** (§4). The honest reason is sharper than "wait for the
scheduler": the bigger prerequisite isn't the scheduler, it's a **CUDA-side paged KV-cache**
(block pool + block table with a GPU-resident representation), which doesn't exist and which no
one has designed the ABI for yet. Writing the attention kernel first, against a block-table shape
invented for a throwaway test, risks building the wrong ABI and immediately requires informed
consent, revision, unlike a straightforward wait.

---

## 1. Re-verification of the prerequisite (2026-08-01)

Repeating the checks from #200's second comment (2026-07-27) with current `dev`:

| Check | 2026-07-27 finding | 2026-08-01 re-check | Changed? |
|---|---|---|---|
| `CudaTransformerModel.ForwardBatch` override | Doesn't exist, per-seq fallback | `grep -n "ForwardBatch" src/DotLLM.Cuda/CudaTransformerModel.cs` → zero matches | **No** |
| `docs/SCHEDULING.md` `ForwardBatch` status table | "CUDA: per-seq fallback" | Same line present verbatim today (the table was touched by Step 59 commits for other rows, this one untouched) | **No** |
| `CudaPagedKvCache` / `CudaKvBlockPool` | Doesn't exist | `grep -rln "CudaPagedKvCache\|PagedKvCache" src/DotLLM.Cuda/ src/DotLLM.Vulkan/` → zero matches | **No** |
| `docs/KV_CACHE.md` Limitations (v1) | "CPU `PagedKvCache` only. CUDA and hybrid models fall back to their native KV-cache." | Same text, file last touched only by unrelated commits | **No** |
| Dedicated tracking issue for "CUDA batched decode" or "CUDA paged KV-cache" | Not found | `gh issue list --search "CUDA batched decode OR CUDA paged"` → only #200 itself and unrelated results (#250 BitNet-MoE GEMM, Vulkan issues) | **No** |
| Continuous batching itself (Engine layer) | Real, `ContinuousBatchScheduler` MVP just landed | Substantially *more* built out now — Step 59 shipped chunked prefill, priority+preemption, disaggregated prefill/decode scheduler, fairness (SFQ + per-key weights), recurrent-state threaded batching for Mamba-3/Qwen3-MoE-Hybrid GDN. See `docs/SCHEDULING.md`. | **Yes, but irrelevant to CUDA** — all of this batching activity runs through `IModel.ForwardBatch`, and CUDA is the one backend still on the per-sequence-loop default implementation. More scheduler sophistication does not change CUDA's status. |

The one thing that *did* change is that continuous batching got considerably richer — which
initially looks like it might have moved the goalposts. It didn't: all of that richness lives at
the `IModel.ForwardBatch` seam, and CUDA's implementation of that seam is still the interface's
default per-sequence loop (`IModel.cs`'s doc comment: *"The default interface implementation loops
over `Forward` per request — backends pay the per-sequence kernel-dispatch overhead until they
override with a fused implementation."*). Whatever `ContinuousBatchScheduler` does upstream of the
model call, a CUDA-served request still runs through N independent `Forward` calls, each of which
uses `CudaKvCache` (non-paged), not `PagedKvCache`. This issue's kernel has nothing to attach to.

## 2. Cross-backend gap-sharing — reconfirmed, still CPU-only

`docs/KV_CACHE.md`'s staging-buffer-gather description is backend-neutral prose, but the actual
implementation lives in exactly one class: `src/DotLLM.Engine/KvCache/PagedKvCache.cs`
(`GetKeysRef`/`GetValuesRef` → `GatherIntoStaging`, block-by-block `Buffer.MemoryCopy` into a
pre-allocated contiguous staging buffer — see the class for the full gather loop). This is
`DotLLM.Engine`, backend-neutral code, but it is only ever wired up behind `NaiveAttentionStrategy`
on the CPU host. CUDA/Vulkan both fall back to their non-paged native caches when `--paged`/
`UsePaged` is requested (`ServerStartup.cs:151-160`, mirrored in `RunCommand.cs` / `ChatCommand.cs`
per the prior comment). So: **the gap this issue names — "a staging-buffer gather that a direct
block-table read could eliminate" — is real, but only exists on CPU today.** There is no CUDA (or
Vulkan) staging-buffer gather to eliminate, because neither backend has a paged KV-cache in the
first place.

## 3. What the current CPU paged-KV layout looks like (context for the eventual CUDA design)

For the future implementer, the shapes that any CUDA block-table ABI will eventually need to
mirror or diverge from deliberately:

- `KvBlockPool` (`src/DotLLM.Engine/KvCache/KvBlockPool.cs`): one contiguous unmanaged buffer per
  layer, shape `[totalBlocks, blockSize, kvStride]` floats, `blockSize` default **16** tokens,
  free-list allocator with per-block refcounts (CoW / prefix sharing).
- `KvBlockTable` (`src/DotLLM.Engine/KvCache/KvBlockTable.cs`): per-sequence `List<int>` mapping
  logical block index → physical block ID in the pool; `Resolve(position)` returns
  `(blockId, offsetInBlock)`.
- `PagedKvCache.GetKeysRef`/`GetValuesRef` gather every block for a sequence, in logical order,
  into one contiguous staging buffer sized for `maxSeqLen`, which is what the (unmodified)
  attention kernel then reads.

A CUDA-side equivalent needs a **GPU-resident** version of this: block storage that already lives
in device memory (trivial — CUDA tensors already do), and a **block table the kernel can index
inside the device function**, not a C#-side `List<int>` — i.e. an `int[]` (or `int2[]` for
block-id+refcount together) array itself uploaded to the GPU, one per active sequence, or one
flattened `[batch, maxBlocksPerSeq]` array once CUDA has a real multi-sequence batched-decode call
shape. This second point is exactly why item 1 in the prerequisite chain (§5) has to land before
the block-table ABI can be pinned down with confidence.

## 4. Could a kernel-only version be built and tested today, against a synthetic block table?

This is the extra question this session was asked to investigate, since the project's test-first
culture (e.g. `CudaTensorCoreAttentionParityTests.cs`, `CudaAttentionF32SplitKvTests.cs` — small
synthetic shapes, oracle parity against the existing non-paged kernel, tight tolerance) makes this
a reasonable thing to ask before assuming a hold is required.

**Investigated and rejected, for a specific reason — not just "wait for the scheduler."**

The naive framing is: write a new kernel that reads K/V through a device-side block-table array
instead of a flat `k + tkv * kv_stride` index (the change is genuinely small —
`attention_f32.cu`'s inner loop at line 91, `const float* k_vec = k + (size_t)tkv * kv_stride + hkv
* head_dim;`, becomes a two-step lookup: resolve `tkv` to `(blockId, offset)` via the block table,
then index into `block_base_ptrs[blockId] + offset * kv_stride + hkv * head_dim`). Build a
synthetic scattered-block KV layout in a unit test (analogous to how the CPU `KvBlockPool` scatters
blocks), run the new kernel, compare against `attention_f32` run on the same logical content laid
out contiguously. This is real, buildable, testable in isolation, and mechanically not much harder
than the #197/#198 GQA-split kernel work already shipped. Correctness risk is genuinely LOW (same
math, per the issue's own framing) — this part of the prior comment's assessment holds up.

The reason to still not do it is what the block-table **ABI** commits you to, not the kernel body:

1. **Single- vs multi-sequence-per-launch is an engine decision the kernel would have to guess at.**
   Item 1 of the prerequisite chain (§5) — CUDA multi-sequence batched decode call shape — doesn't
   exist yet, so no one has decided whether a future `CudaTransformerModel.ForwardBatch` dispatches
   **one kernel launch per sequence** (grid indexed by `(numKvHeads, kvSplit)` same as today, block
   table is one array, trivial extension of the existing per-seq launch) or **one kernel launch
   across the whole batch** (grid needs a `batchIdx` dimension, block table becomes
   `[batch, maxBlocksPerSeq]`, Q/output pointers become per-sequence-indexed, and the KV-split
   co-residency ceiling math in `CudaKernels.ComputeAttentionKvSplit`/`ComputeSafeKvSplit` has to be
   redone for a batch dimension). These are materially different kernels, not the same kernel with
   an extra loop — vLLM's own evolution shows this: `paged_attention_v1` is single-sequence-grid,
   `paged_attention_v2` adds the batch dimension and a different KV-split partition/reduce strategy.
   Building a synthetic single-sequence-only test today can't tell you which of these to build,
   and picking wrong is exactly the kind of throwaway rework the issue's own risk section wants to
   avoid.
2. **Three live decode-attention kernel variants already exist and a paged version has to decide
   which one(s) it replaces**, not just "the" attention kernel the issue was scoped against on
   2026-07-27: plain `attention_f32` (grid=`numHeads`), `attention_f32_gqa_split_kv` (#197/#198,
   grid=`(numKvHeads, kvSplit)`, register-blocked across the GQA group, cooperative-launch combine)
   — now the default per `docs/CUDA.md` — and `attention_flash_mma_decode_gqa_split_f16` (#199 v2,
   tensor-core mma, narrowly gated to `headDim==256`, `seqQ==1`, `group<=8`, flipped to default-ON
   2026-07-31). A paged-KV variant of the *plain* kernel alone would not be representative of the
   actual decode hot path anymore, since that path already went through the GQA-split kernel by
   default; but composing block-table indirection with the GQA-split kernel's shared-memory tiling
   and cooperative cross-split combine is real, non-trivial additional design work beyond the
   simple index-substitution sketched above — and composing it with the even narrower tensor-core
   kernel is a third, separate design surface. A synthetic-test kernel built today would necessarily
   pick one of these three to prototype against, and whichever is picked is disconnected from
   "the" decode kernel by the time any real CUDA paged KV-cache exists to drive it (this kernel
   landscape has already changed twice — #197/#198 then #199 v2 — in the five days since #200's own
   comments were written).
3. **The bigger, currently-undesigned lift is the block pool/table itself, not the kernel.** The
   prior comment already flagged this (finding 2, chain item 2) but it is worth restating plainly:
   a `CudaKvBlockPool` needs its own design pass — block size choice (mirror CPU's 16? tune for GPU
   memory-transaction alignment instead?), allocation/refcount data structure (CPU uses a managed
   `int[]` free-list under a C# lock; a GPU-resident pool wants allocation host-side but block
   *storage* device-side, and needs to decide whether the block table lives in host-pinned memory
   uploaded once per step vs. rebuilt every launch), and CoW/prefix-sharing semantics translated to
   a device-visible refcount. None of that exists, is speculative today, and — critically — a
   kernel written against a guessed-at synthetic ABI does not reduce the size of that undesigned
   lift by writing the kernel first; if anything it adds a constraint the block-pool designer either
   has to honor or invalidate.

Put simply: the kernel-only piece **is** low-risk and buildable in isolation exactly as the task
asked to check — but "buildable" isn't "worth building now," because the artifact it produces
(a kernel bound to an invented block-table shape) has a meaningful chance of needing to be
rewritten once the real ABI is decided by whoever designs `CudaKvBlockPool` and the CUDA batched
call shape together, and there is no user-visible or even engine-visible way to exercise it in the
interim. This is the same "premature speculative kernel work on a non-existent engine layer" trap
`docs/perf/MMA_BATCHED_MMQ.md` named for the mma-batched-MMQ kernel, for the same reason: the risk
isn't the kernel's correctness, it's that the kernel's *interface* is downstream of engine
decisions that haven't been made.

## 5. Prerequisite chain (updated)

Same three-step chain the prior comment identified, re-confirmed unchanged, with the concrete
"pick single-launch or batched-launch" decision now called out explicitly as part of step 1:

1. **CUDA multi-sequence batched decode call shape.** `CudaTransformerModel.ForwardBatch` needs a
   real fused implementation (mirroring the CPU/Vulkan dense-host pattern in
   `docs/SCHEDULING.md`'s `ForwardBatch` section — Phase 5a/5b-style intra-block matmul fusion),
   and as part of that, a decision on whether decode attention itself batches across sequences in
   one launch or stays one-launch-per-sequence with the scheduler just calling it N times faster
   than today (no scheduler changes needed for the latter — the win would be that non-attention
   layers fuse while attention stays per-seq, same tradeoff Nemotron-H/Qwen3-MoE-Hybrid already
   made on Vulkan, see `docs/SCHEDULING.md`'s `ForwardBatch` section). **Not yet tracked under a
   dedicated issue.** Recommend filing one if/when a CUDA host is prioritized for continuous batching
   — `docs/ROADMAP.md`'s Future Considerations table already has empty slots for this ("CUDA paged
   KV-cache" row) it could sit next to.
2. **CUDA paged KV-cache infrastructure** (`CudaKvBlockPool` + block-table plumbing through
   `SequenceForwardRequest`/the CUDA forward path). Does not exist. Bigger lift than this issue
   currently scopes for (§4.3). Also not yet tracked under a dedicated issue —
   `docs/ROADMAP.md`'s Future Considerations already lists it as its own row, separate from "Paged
   attention kernels" (this issue's row), which itself confirms the project's own roadmap already
   expects these to be sequenced as two separate steps.
3. **This issue** — the direct-block-read kernel — becomes buildable-for-real once 1 and 2 exist,
   at which point the block-table ABI is a known quantity (decided by whichever of 1/2 lands first)
   rather than a guess, and there's a live producer/consumer to validate against end-to-end, not
   just a synthetic fixture.

## 6. Kernel design (kept as forward-looking collateral, not for immediate execution)

This section preserves and updates the design from #200's first comment (2026-07-27) for whoever
picks up step 3 above, adjusted for the current kernel landscape (§4.2).

**Proposed signature (single-sequence-launch variant, i.e. if step 1 above resolves to
"per-sequence launch, just N of them instead of a staging-copy-then-launch"):**

```c
extern "C" __global__ void __launch_bounds__(256) attention_f32_paged_gqa_split_kv(
    const float* __restrict__ q,               // [seq_q, num_heads * head_dim] -- unchanged
    const float* const* __restrict__ block_ptrs, // [num_blocks_this_seq] device pointers, one per K/V block
    const int*   __restrict__ block_table,      // [ceil(seq_kv / block_size)] logical->physical block index
    float* __restrict__ output,
    const int seq_q, const int seq_kv, const int block_size,
    const int num_heads, const int num_kv_heads, const int head_dim,
    const int position_offset, const int sliding_window,
    // ...kv_split/partial-combine params, unchanged from attention_f32_gqa_split_kv
);
```

Key design choices, and why:

- **`block_ptrs` as device pointers, not host-relative block IDs into one flat buffer.** Mirrors
  how CPU's `KvBlockPool.GetKeyPtr(blockId, layerIndex)` resolves a pointer, but on GPU each block
  can be a genuinely separate allocation (or a slice of one big pool buffer — either works;
  pointer-indirection keeps the kernel agnostic to which the pool implementation picks). One
  `float* const*` array per sequence per layer, uploaded once per step (it only changes when a new
  block is allocated, i.e. roughly every `block_size` decode tokens) — not rebuilt every launch.
- **`block_size` as a kernel parameter, not a compile-time constant**, so the same kernel binary
  serves whatever block size the eventual `CudaKvBlockPool` picks (CPU's default is 16; GPU may
  want a different size for coalescing — e.g. 32 or 64 to better match a warp's worth of
  transactions per block boundary).
- **Inner-loop change is genuinely small**: replace the flat index
  `k + (size_t)tkv * kv_stride + hkv * head_dim` with
  `block_ptrs[block_table[tkv / block_size]] + (tkv % block_size) * kv_stride + hkv * head_dim`.
  Everything else in the tiled online-softmax loop (`attention_f32.cu` lines ~78-160) is unchanged.
  This confirms the issue's own "risk-equivalent to a pure refactor" framing (§4 in this note) —
  the *math* is untouched, only the memory-access pattern's indirection.
- **Compose with the GQA-split grid, not the plain kernel**, since the GQA-split kernel
  (`attention_f32_gqa_split_kv`) is the actual default decode path today (§4.2) — building the paged
  variant against the plain kernel would optimize a code path real traffic no longer takes.
  Whether to also build a paged variant of the narrower tensor-core kernel
  (`attention_flash_mma_decode_gqa_split_f16`) is a separate, later decision — start with the
  GQA-split F32 kernel since it covers the general case (no `headDim==256` gate).

**Validation plan (unchanged from the first comment, still the right shape):**

- New `CudaAttentionF32PagedTests.cs` (mirrors `CudaAttentionF32SplitKvTests.cs`): synthetic
  scattered-block KV layout (allocate blocks out of order, build a `block_table`/`block_ptrs` pair
  by hand) vs. the same logical content laid out contiguously through the existing
  `attention_f32_gqa_split_kv`. Bit-exact per the "risk-equivalent to pure refactor" framing.
  Cases: partial tail block, exact block-size-boundary token counts, single-block sequences.
- Performance validation needs a genuinely multi-sequence decode benchmark (N=4/8/16 concurrent
  sequences), not the single-sequence Bonsai-27B bench this investigation otherwise uses — see the
  first comment's §5 for the full reasoning, unchanged here.

## 7. Recommendation

**Hold, unchanged from both prior comments.** Concretely:

1. Do not implement the CUDA paged-attention kernel now. Nothing has changed since 2026-07-27 that
   would make it viable in production, and (§4) a synthetic-test-only version isn't worth the ABI
   risk either.
2. Consider filing two new, narrowly-scoped issues to make the prerequisite chain trackable
   (currently only implicit in `docs/ROADMAP.md`'s Future Considerations table and
   `docs/CUDA.md`'s Future Work prose):
   - "CUDA multi-sequence batched decode (`CudaTransformerModel.ForwardBatch`)" — chain item 1.
   - "CUDA paged KV-cache (`CudaKvBlockPool` + block-table plumbing)" — chain item 2, already has a
     `docs/ROADMAP.md` Future-Considerations row but no issue.
3. Keep #200 open, scoped as "design ready, blocked on the above," with this note (§6) as the
   ready-to-execute kernel design once either prerequisite lands. No re-derivation needed by a
   future implementer.

---

## References

### dotLLM source

- `src/DotLLM.Engine/KvCache/PagedKvCache.cs`, `KvBlockPool.cs`, `KvBlockTable.cs` — CPU paged
  KV-cache, the only backend with a real staging-buffer gather today.
- `src/DotLLM.Cuda/CudaKvCache.cs` — CUDA's current non-paged KV-cache.
- `native/kernels/attention_f32.cu` — plain, GQA-split-kv, and split-kv-hp kernel variants.
- `native/kernels/attention_flash_mma_decode_gqa_split.cu` — #199 v2 tensor-core decode kernel.
- `src/DotLLM.Cuda/CudaAttentionMmaDecodeGqaSplit.cs` — its C# wrapper and scope doc-comment
  (`CanUse` gating).
- `src/DotLLM.Core/Models/IModel.cs`, `SequenceForwardRequest.cs` — the `ForwardBatch` seam.
- `docs/SCHEDULING.md` — `ContinuousBatchScheduler`, `ForwardBatch` per-backend status table.
- `docs/KV_CACHE.md` — paged KV-cache design, "Limitations (v1)" section.
- `docs/CUDA.md` — Future Work, decode-attention ncu findings, #199 v2 status.
- `docs/ROADMAP.md` § Future Considerations — "Paged attention kernels" and "CUDA paged KV-cache"
  rows (separate, unimplemented).
- `docs/perf/MMA_BATCHED_MMQ.md` — analogous prior investigation (kernel idea blocked on an
  engine-layer prerequisite); same recommendation shape.

### Prior art (cited, not redistributed)

- vLLM `paged_attention_v1`/`v2` — block-table-indexed KV read, `v2`'s added batch/partition
  dimension for the KV-split reduce.
- FlashInfer — block-sparse attention with page-table indirection.
