# gpt-oss CUDA Per-Head Attention Sinks (issue #365) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add gpt-oss's per-head attention sinks (a learned scalar logit per head joining the softmax denominator) to the CUDA attention path, matching the CPU reference `Attention.SoftmaxRowWithSink` numerically, then (as the second of the two gpt-oss attention plans to land) replace the `Architecture.GptOss` guard in `CudaModelLoader.CreateFromGguf` with real dispatch.

**Architecture:** The sink semantics (CPU `src/DotLLM.Cpu/Kernels/Attention.cs:211-218`, matching llama.cpp `ggml_soft_max_add_sinks`): `m = max(max(scores), sink)`, `p_j = exp(x_j − m) / (Σ exp(x_i − m) + exp(sink − m))` — the sink absorbs probability mass but contributes no value vector. In `attention_f32`'s ONLINE softmax (`native/kernels/attention_f32.cu`), this is a post-loop epilogue: treat the sink as one final "virtual tile" with no V contribution — rescale `running_sum`/`out_accum` by `exp(running_max − new_max)` where `new_max = max(running_max, sink)`, add `exp(sink − new_max)` to `running_sum`, then normalize as before. Mathematically identical to the CPU's two-pass form. The sinks tensor (`TransformerLayerWeights.AttnSinks`, loaded from `attn_sinks.weight`, one float per head) uploads to a per-layer device buffer and passes as a nullable pointer (nullptr = no sinks = today's behavior, bit-identical).

**Sequencing:** This plan executes AFTER `2026-08-15-gptoss-cuda-swa-alternating.md` (#366). This plan's final task removes the loader guard — the point where gpt-oss becomes loadable on CUDA — so both features must already be in.

**Tech Stack:** CUDA C (`native/kernels/attention_f32.cu` → PTX via pinned `E:\CUDA_v12.8.1` toolkit), C# (`CudaKernels.cs` launcher, `CudaWeights`/`CudaTransformerModel` wiring), xUnit real-GPU tests.

## Global Constraints

- Branch `issue/365-cuda-attention-sinks` from `dev` (with #366's merge already in).
- PTX: pinned toolkit `E:\CUDA_v12.8.1\bin\nvcc.exe` → `.version 8.7` (ambient nvcc 13.1 emits 9.1 — WRONG); committed `.ptx` mtime must be NEWER than its `.cu` (local MSBuild CompileCudaPtx is broken); `attention_f32.cu` is NOT in the NO_FMA or FAST_MATH lists today — keep it that way (sinks math is one exp/max, no new precision constraint beyond the kernel's existing `fast_exp_neg` usage; the parity tolerance absorbs it, as it already does for the kernel's existing exp).
- Kernel-signature change ⇒ BOTH `attention_f32` AND `attention_f32_split_kv` (same file, ~line 298) get the sink parameter — the dispatcher can choose either at runtime; diverging them silently corrupts the split-KV path. The flash/MMA variants (`attention_flash_mma*.cu`, `_g3Attention`) do NOT get sinks in this plan: instead the dispatch gate (Task 4) routes sink-bearing layers away from flash paths. (Flash+sinks is a follow-up perf issue, filed in Task 6.)
- Signature-extension style: append `const float* __restrict__ sinks` (nullable) as the LAST pointer param, `nullptr` ⇒ exact current behavior. Every existing call site passes `0`/`nullptr` — behavior-preserving for all non-gpt-oss models.
- New device allocations in weight loading go through the #383 allocation-ledger pattern. MakeCurrent conventions per #368. Documented-tolerance tests, never `SequenceEqual`.
- CPU reference is authoritative; llama.cpp `ggml_soft_max_add_sinks` is the cross-check.
- The `Architecture.GptOss` guard in `CudaModelLoader.CreateFromGguf` is removed ONLY in Task 5, and only after Tasks 1-4 are green AND #366 is confirmed merged.

---

### Task 1: Sink epilogue in `attention_f32` + `attention_f32_split_kv`

**Files:**
- Modify: `native/kernels/attention_f32.cu`
- Rebuild: `native/ptx/attention_f32.ptx` (pinned toolkit)

**Interfaces:**
- Produces: both kernels gain trailing param `const float* __restrict__ sinks` (per-head, indexed by `hq`; `nullptr` = disabled).

- [ ] **Step 1:** Extend `attention_f32`'s signature (line 38-43): append `, const float* __restrict__ sinks` after `const int sliding_window`. Insert the epilogue between the tile loop's closing brace and the `// Normalize and write` block (i.e. immediately before the `float sum_inv = ...` line at ~167):

```c
    // gpt-oss attention sinks (#365): the per-head sink logit joins the softmax
    // denominator as a final "virtual tile" with no V contribution. Matches CPU
    // Attention.SoftmaxRowWithSink / llama.cpp ggml_soft_max_add_sinks:
    //   m' = max(running_max, sink)
    //   running_sum = running_sum * exp(running_max - m') + exp(sink - m')
    //   out_accum  *= exp(running_max - m')
    // nullptr => bit-identical to the pre-#365 kernel (no rescale, no added term).
    if (sinks != nullptr)
    {
        float sink = sinks[hq];
        float new_max = fmaxf(running_max, sink);
        float correction = (running_max > -FLT_MAX + 1.0f)
                           ? fast_exp_neg(running_max - new_max) : 0.0f;
        running_sum = running_sum * correction + fast_exp_neg(sink - new_max);
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            out_accum[d] *= correction;
        running_max = new_max;
        // No __syncthreads() needed: each thread rescales and later reads only its
        // own strided d-elements of out_accum; running_sum/running_max are
        // per-thread registers already uniform across the block at this point.
    }
```

VERIFY the "uniform across the block" claim before committing: read the preceding tile-loop code — `running_max`/`running_sum` are per-thread locals updated from `warp_scratch[0]` broadcasts after `__syncthreads()` (lines ~114-116, ~146-148), so they ARE block-uniform here. If your read disagrees, stop and report.

- [ ] **Step 2:** Same change to `attention_f32_split_kv` (~line 298): CAUTION — in the split-KV kernel each block owns a KV SUB-RANGE and partials merge afterwards. The sink must join EXACTLY ONCE, not once per split. Read the kernel's merge step: if partials merge via the standard (m, l, acc) rule in a separate reduction pass/kernel, the sink belongs in the MERGE, not the per-split body. Determine which structure holds and place the epilogue at the single point where the final (m, l) is known — if the merge happens on the host or in a second kernel, extend THAT instead and document. If the structure makes single-injection awkward, an acceptable alternative (document it): route sink-bearing layers away from split-KV in the Task 4 dispatch gate, exactly as for flash, and leave `attention_f32_split_kv` unchanged — then this step's deliverable becomes that documented routing decision instead of kernel code.
- [ ] **Step 3:** Rebuild PTX: `E:\CUDA_v12.8.1\bin\nvcc.exe -ptx -arch=compute_75 -allow-unsupported-compiler -o native/ptx/attention_f32.ptx native/kernels/attention_f32.cu` (match `build_ptx.bat`'s exact flags for this kernel — read the script first; it is NOT in NO_FMA/FAST_MATH so default fmad applies). Verify `.version 8.7`, both entry symbols, and `.ptx` mtime > `.cu`.
- [ ] **Step 4: Commit**: `git commit -am "feat(cuda): attention sinks epilogue in attention_f32 kernels (#365)"`

### Task 2: Launcher + sink-buffer plumbing in `CudaKernels.cs`

**Files:**
- Modify: `src/DotLLM.Cuda/CudaKernels.cs` (`LaunchAttentionF32` at ~line 2255, and the split-KV launcher if Task 1 Step 2 extended that kernel)

- [ ] **Step 1:** Add `nint sinks = 0` as trailing optional parameter to `LaunchAttentionF32`; append it to the `stackalloc void*[]` args array (kernel param order must match EXACTLY — count the params). All existing call sites compile unchanged (default 0 = nullptr). Update the XML doc: sink semantics, per-head length = numHeads floats, 0 = disabled.
- [ ] **Step 2:** Build 0/0. Run the existing attention kernel unit tests (grep for the test class exercising `LaunchAttentionF32`) — must pass unchanged (nullptr path is bit-identical; the PTX changed, so this run also validates the rebuilt PTX loads).
- [ ] **Step 3: Commit**: `git commit -am "feat(cuda): LaunchAttentionF32 sink parameter (#365)"`

### Task 3: Kernel-level parity test vs `SoftmaxRowWithSink`

**Files:**
- Test: `tests/DotLLM.Tests.Unit/Cuda/CudaAttentionSinksKernelTests.cs` (create)

- [ ] **Step 1:** Test 1 (`NullSinks_BitIdenticalToBaseline`): run `LaunchAttentionF32` twice on identical inputs, sinks=0 vs a device buffer of `-FLT_MAX`-equivalent... NO — sinks=0 (nullptr) vs the PREVIOUS behavior can't be compared cross-build; instead assert sinks=0 output equals a CPU `Attention.Execute` reference WITHOUT sinks at the established F32 tolerance (this is the no-regression gate). Test 2 (`WithSinks_MatchesCpuSoftmaxRowWithSink`): CPU side computes attention with `lw.AttnSinks`-style per-head sinks via the public CPU path (`Attention.Execute(..., sinks)` — read its signature at the call site `TransformerModel.cs:1387-1398` for the exact overload); CUDA side passes an uploaded sink buffer with DISTINCT per-head values (e.g. `sinks[h] = -1.0f + 0.7f*h` — distinct so a wrong-head-index bug discriminates; include one sink LARGER than any score so the `m = sink` branch is exercised, and one much smaller so the sink is negligible — both regimes in one fixture). Shapes: heads=4, kvHeads=2 (GQA repeat 2 — sinks index by `hq` not `hkv`; a collision here is exactly the bug class to catch), headDim=16, seqQ=5, seqKv=9, non-degenerate. Documented tolerance calibrated from observed (~1e-6 expected; state observed + margin in comment).
- [ ] **Step 2:** Mutation check (required): temporarily index `sinks[hkv]` instead of `sinks[hq]` in a scratch rebuild — the GQA-repeat-2 fixture must FAIL; revert + verify clean tree; record numbers in remarks. (This validates the discriminating claim, per the #384/#385 precedent. NOTE this requires one extra PTX rebuild each way — follow the pinned-toolkit + mtime rules both times.)
- [ ] **Step 3:** Run on real GPU, all green. Commit: `git commit -am "test(cuda): attention-sinks kernel parity vs CPU SoftmaxRowWithSink (#365)"`

### Task 4: Model-level wiring — upload `AttnSinks`, pass per layer, gate flash paths

**Files:**
- Modify: `src/DotLLM.Cuda/CudaWeights.cs` (per-layer sink device buffer), `src/DotLLM.Cuda/CudaTransformerModel.cs` (attention dispatch)

- [ ] **Step 1:** In `CudaWeights`' per-layer attention weight loading (find where `attn_output.weight` etc. upload — the loader path #383 hardened), add: if the GGUF has `attn_sinks.weight` for the layer (same tensor name the CPU loader binds to `TransformerLayerWeights.AttnSinks` — `TransformerWeights.cs:648`), upload the numHeads floats to a new `nint AttnSinksDevice` per-layer field (0 when absent). Ledger the allocation (#383 pattern — append to the factory's alloc list immediately after `cuMemAlloc_v2`). Free in the same place sibling per-layer buffers free.
- [ ] **Step 2:** In `CudaTransformerModel`'s attention dispatch (the call sites carrying `slidingWindow` from #366): pass `lw.AttnSinksDevice` to `LaunchAttentionF32`. **Gate the flash/G3/split-KV fast paths**: where `_flashAttention.CanUse(...)` / `_g3Attention.CanUse(...)` (and split-KV, if Task 1 Step 2 chose routing) are consulted, add `&& lw.AttnSinksDevice == 0` so sink-bearing layers take the `attention_f32` path that implements sinks. Comment: "flash+sinks is follow-up #<Task 6's issue>; correctness first."
- [ ] **Step 3:** Build 0/0; full CUDA unit filter regression (non-gpt-oss models all have AttnSinksDevice==0 ⇒ bit-identical dispatch).
- [ ] **Step 4: Commit**: `git commit -am "feat(cuda): upload attn_sinks + sink-aware attention dispatch (#365)"`

### Task 5: Remove the GptOss loader guard + end-to-end parity test

**Files:**
- Modify: `src/DotLLM.Cuda/CudaModelLoader.cs` (the `case Architecture.GptOss:` guard from #348)
- Modify: `tests/DotLLM.Tests.Integration/Cuda/CudaUnsupportedArchitectureGuardTests.cs` (drop/replace the GptOss `[InlineData]`)
- Test: `tests/DotLLM.Tests.Integration/Cuda/CudaGptOssParitySyntheticTests.cs` (create)

- [ ] **Step 1:** PRECONDITION: verify #366 is merged into the branch's base (`git log --oneline dev | grep 366` from the worktree). If not, STOP.
- [ ] **Step 2:** Replace the guard case with the real dispatch (route to the same `CudaTransformerModel` path Llama-family uses — GptOss is `CudaTransformerModel`-shaped per the CPU implementation; read how CPU `ModelLoader` dispatches GptOss and mirror). Remove the GptOss `[InlineData]` from the guard test.
- [ ] **Step 3:** Synthetic end-to-end CPU-vs-CUDA gpt-oss parity test: small fixture with ALL FOUR gpt-oss features active — MoE biases + clamped SwiGLU (#348), alternating SWA (#366, seqLen > window), sinks (#365, distinct per-head values), pattern=2 across 4 layers. Fixture mechanics: mirror whatever synthetic-GGUF/checkpoint builder the #348 tests used (grep the #348 test files). Documented tolerance from observed. This is the gate that proves the three PRs compose.
- [ ] **Step 4:** Run on real GPU. Update `docs/SUPPORTED_MODELS.md` (gpt-oss CUDA: supported, synthetic-verified; note no real-weight run yet unless a checkpoint is cached — check `~/.dotllm/test-cache` and say which). Commit: `git commit -am "feat(cuda): enable GptOss in CudaModelLoader — sinks+SWA complete (#365)"`

### Task 6: File the flash+sinks follow-up

- [ ] **Step 1:** `gh issue create` — title "flash/MMA attention kernels lack sink support — gpt-oss routed to attention_f32 (perf follow-up)": body cites the Task 4 gate location, the sink epilogue design (this plan), and llama.cpp's flash-attn sink handling as the reference. (GitHub connectivity has been flapping — use the `until gh ...; do sleep 120; done` retry pattern with a pre-written body file if it times out.)

## Self-review checklist (author ran per the writing-plans skill)
- Spec coverage: kernel port ✓ (Task 1, exact epilogue math derived from the CPU two-pass form), weight wiring ✓ (Task 4), discriminating with-and-without test ✓ (Task 3: null-sink no-regression + distinct-per-head + GQA-collision mutation check), guard removal sequenced last ✓ (Task 5 with precondition).
- No placeholders: epilogue code complete; the one deliberately-open design point (split-KV single-injection, Task 1 Step 2) specifies BOTH resolutions concretely and how to choose.
- Type consistency: `sinks` is `nint` (device ptr) in C#, `const float*` in CUDA, indexed by `hq`, length numHeads — consistent across Tasks 1/2/3/4.
- Known risk carried from repo history: brief errata cluster at numerical constants and persistence lines — the epilogue math here was derived against the CPU source read directly (`m'=max`, rescale, `+exp(sink−m')`), and Task 3's mutation check guards the head-indexing line.
