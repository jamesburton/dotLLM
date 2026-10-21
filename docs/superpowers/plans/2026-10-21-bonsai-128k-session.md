# Bonsai-2 128k Continuing Sessions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One continuing Bonsai-2 session on a 12 GB RTX 3060 grows to 128k tokens, with each turn prefilling only what it adds, and `model[128k]` / `model[Nk-mtp]` ids that select the settings to get there.

**Architecture:** Four PRs, each with its own issue, in this order:
- **(A) engine:** a stable snapshot boundary taken from the chat template's generation-prompt suffix, plus exact-size KV growth before giving up. Covers the legacy, tiered and scheduler paths.
- **(B) CUDA:** a flash-prefill kernel that dequantizes q8_0/q4_0 K/V tiles in its load stage, so no full-history F16 scratch is needed.
- **(C) CUDA:** the MTP head's KV grows in place via VMM.
- **(D) server:** window-hint ids with a persisted fit record and in-place reconfiguration of a loaded model.

**Tech Stack:** C# / .NET 10, xUnit, CUDA PTX via nvcc (`native/build_ptx.bat`), ASP.NET minimal API.

**Spec:** `docs/superpowers/specs/2026-10-21-bonsai-128k-session-design.md` (read it first; this plan argues from it).

## Global Constraints

- Build: Release with analyzers ON: `dotnet build -c Release -m:1 -nr:false -p:UseSharedCompilation=false -p:EnableSourceControlManagerQueries=false`.
- No new full-tensor widen or copy of quantized KV or weights, including in any fallback the Bonsai shape reaches (user's critical rule). State per-token byte costs in each PR.
- No numerical tolerance may be widened. Never spill into WDDM shared memory; refuse instead.
- Zero allocations on the hot path. Use `[LibraryImport]`. `*.ptx` files are tracked: rebuild and commit them with kernel changes (`native/build_ptx.bat`; PTX `.version` 8.7; the four flash-mma kernels plus any new one are compute_86 + C++17).
- XML docs on public APIs. StyleCop comment rules. File-scoped namespaces.
- System.Text.Json source-gen DTOs: no initializers on `init` properties (use nullable + resolve in code). Add a new response-only DTO to `JsonContextDefaultsTests.ResponseOnly`.
- Workflow: create a GitHub issue per PR. Branch `issue/{n}-{slug}` from `dev` (PR D from the merged result of B + C). Put `(#n)` in every commit. PR body says `Closes #n`. After verification, merge into `dev` and push (the user authorized this).
- GPU tests and benches are serialized on this box. Wait for `nvidia-smi` < 300 MiB before each run. Real Bonsai tests need `DOTLLM_BONSAI_PQ2_0_GGUF=C:/Users/james/.dotllm/models/BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-v2-PQ2_0-MTP-GGUF/Ternary-Bonsai-2-27B-Abliterated-v2-PQ2_0-MTP.gguf`.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Pass multi-line messages with `git commit -F <file>`, not a PowerShell pipe.

## Review Focus

1. **A chat whose template ends in a thinking opener** (`<think>\n`), when the client drops reasoning from the history. The snapshot must still advance every turn. Pinned in Task 2.
2. **A generation-prompt suffix that BPE merges with the preceding text**, so the token tail doesn't equal `Encode(suffix)`. Expect a fallback to today's boundary, never a wrong boundary. Pinned in Task 1.
3. **Prompts arriving through the scheduler path** (`--no-mtp`, the default for that setting). They must get the same boundary and growth behaviour as the generator paths. Pinned in Tasks 1 and 3.
4. **A hinted id requested while another variant of the same weights is mid-request.** It waits, then reconfigures. It must never reconfigure under a running request. Pinned in Task 11.
5. **Reconfigure leaks.** Switching variants several times must return free VRAM to within 64 MiB of the post-load figure. Pinned in Task 10.

---

## PR A — stable snapshot boundary and exact growth (spec C1, C2)

### Task 0: Issues and branch

- [ ] Create four GitHub issues (A, B, C, D) on `jamesburton/dotLLM`. Each issue gets its spec section, acceptance criteria and the measured baseline table. Rename the worktree branch to `issue/{A}-stable-snapshot-boundary`.

### Task 1: Generation-suffix boundary in the engine (all three resolve paths)

**Files:**
- Modify: `src/DotLLM.Core/Configuration/InferenceOptions.cs` (add the property at the end of the record)
- Create: `src/DotLLM.Engine/RecurrentSnapshotBoundary.cs`
- Modify: `src/DotLLM.Engine/TextGenerator.cs` (`ResolveRecurrentPrefix`, ~1658-1720; pass options through `ResolveKvCache` ~1438)
- Modify: `src/DotLLM.Engine/TextGenerator.RecurrentSnapshots.cs` (`ResolveTieredRecurrentPrefix` ~48-61)
- Modify: `src/DotLLM.Engine/ContinuousBatchScheduler.cs` (`ResolveRecurrentPrefix(SchedulerRequest, int[])` ~895-935)
- Test: `tests/DotLLM.Tests.Unit/Engine/RecurrentSnapshotBoundaryTests.cs`, plus new cases in `tests/DotLLM.Tests.Unit/Engine/TextGeneratorRecurrentPrefixCacheTests.cs`

**Interfaces:**
- Produces:
  - `InferenceOptions.GenerationPromptSuffix { get; init; }` (`string?`). This is the text the template appended for the generation prompt. Null means unknown, and the code keeps today's behaviour.
  - `static class RecurrentSnapshotBoundary`, method `static int? FromGenerationSuffix(ReadOnlySpan<int> promptIds, ITokenizer tokenizer, string? suffix)`.
    - Returns `promptIds.Length - suffixIds.Length` when `suffixIds = tokenizer.Encode(suffix)` is non-empty and equals the tail of `promptIds`.
    - Returns null otherwise (empty or null suffix, tail mismatch, or a result < 1).
  - Policy for all three paths: `int Choose(int legacyBoundary, int? suffixBoundary, int promptLen)`. It returns `max(legacyBoundary, suffixBoundary ?? -1)`, capped at `promptLen - 1`. Put it in `RecurrentSnapshotBoundary` as `static int Choose(int legacyBoundary, int? suffixBoundary, int promptLen)`.

- [ ] **Step 1: Write failing tests** in `RecurrentSnapshotBoundaryTests`:
  - `FromGenerationSuffix_ReturnsLengthMinusSuffix_WhenTailMatches`. With `CsvTokenizer`-style ids `[1..20]` and suffix ids `[19,20]`, it returns `18`.
  - `FromGenerationSuffix_ReturnsNull_WhenTailDiffers` (BPE-merge case: tail `[19,99]`).
  - `FromGenerationSuffix_ReturnsNull_ForNullOrEmptySuffix`.
  - `Choose_TakesTheLaterBoundary` (legacy 30, suffix 25, len 40 → 30; legacy 10, suffix 25 → 25).
  - `Choose_CapsAtPromptLenMinusOne`.
- [ ] **Step 2: Write failing generator tests** in `TextGeneratorRecurrentPrefixCacheTests`, reusing `Build`, `HistoryModel` and `CsvTokenizer`.
  - `DivergingHistory_WithSuffixHint_HitsTheWholePreviousTurn`. The setup models the Bonsai template with reasoning dropped:
    - Turn 1 is `P1 + G`, where G is the 2-token generation suffix.
    - Turn 2 is `P1 + R + U2 + G`, where R is the rendered reply and differs from G.
    - Each turn passes `GenerationPromptSuffix` = G's text.
  - Assert that turn 2's `CachedTokenCount == len(P1)` and turn 3's `== len(P1 + R + U2)`. That shows the snapshot advances every turn.
  - Run it on both the legacy path and the tiered path (`RecurrentPrefixCacheOptions` with 2 slots).
  - `ExactExtension_StillSnapshotsAtPromptEnd_WithSuffixHint` is the no-regression case: an exact-extension prompt still caches `promptLen_prev - 1`.
- [ ] **Step 3: Write a failing scheduler test** in `tests/DotLLM.Tests.Unit/Engine/ContinuousBatchSchedulerRecurrentPrefixTests.cs` (create it; model the fakes on the generator test's `HistoryModel`). It is the same diverging three-turn script through `EnqueueAsync` with `InferenceRequest.Options.GenerationPromptSuffix`, and asserts the same cached counts.
- [ ] **Step 4: Run them to confirm they fail.** Run `dotnet test tests/DotLLM.Tests.Unit -c Release --filter "FullyQualifiedName~RecurrentSnapshotBoundary|FullyQualifiedName~DivergingHistory|FullyQualifiedName~ContinuousBatchSchedulerRecurrentPrefix"`. Expect compile errors or failures.
- [ ] **Step 5: Implement.**
  - Add `RecurrentSnapshotBoundary`.
  - Thread `options.GenerationPromptSuffix` into the three resolve methods.
  - In each method, compute today's boundary, then apply `Choose`.
  - Keep the #1091 first-turn rule. It sets `promptLen - 1` when there is no entry; with a hint, `Choose` now takes `max`, so pass the suffix boundary instead of `promptLen - 1` when a hint exists. Otherwise turn 1's snapshot includes G and never matches.
  - Keep the ≥ `RecurrentPrefixMinTokens` move-forward rule.
- [ ] **Step 6: Run the tests.** Expect all new tests to pass, plus the existing `TextGeneratorRecurrentPrefix*`, `RecurrentPrefixSnapshotCacheTests` and `RecurrentPrefixDiskCacheTests`.
- [ ] **Step 7: Commit:** `engine: snapshot recurrent prefixes before the generation prompt (#A)`.

### Task 2: Server passes the generation-prompt suffix

**Files:**
- Modify: `src/DotLLM.Server/ReasoningSupport.cs`. Add the method next to `TryApply` (~147).
- Modify: `src/DotLLM.Server/Endpoints/ChatCompletionEndpoint.cs` (~105-112 render, ~202-226 and ~398 dispatch).
- Modify: `src/DotLLM.Server/Endpoints/MessagesEndpoint.cs` (~694 render, ~180/189/290 dispatch).
- Test: `tests/DotLLM.Tests.Unit/Server/GenerationPromptSuffixTests.cs`

**Interfaces:**
- Consumes: `InferenceOptions.GenerationPromptSuffix` (Task 1).
- Produces: `static string? ReasoningSupport.GenerationPromptSuffix(IChatTemplate template, IReadOnlyList<ChatMessage> messages, ChatTemplateOptions options, string prompt)`. It renders with `options with { AddGenerationPrompt = false }`. When `prompt` starts with that rendering and is longer, it returns `prompt[rendering.Length..]`. Otherwise, or if rendering throws `JinjaException`, it returns null.

- [ ] **Step 1: Write failing tests:**
  - `Suffix_ForQwen35Template_IsTheAssistantHeaderAndThinkOpener`. It uses fixture `Tokenizers/ChatTemplates/Fixtures/qwen35-ornith-chat-template.jinja`, two messages, and `EnableThinking = true`. Assert the suffix starts with `<|im_start|>assistant\n` and contains `<think>`.
  - `Suffix_ForQwen38Template_WithPreserveThinking_MatchesItsGenerationPrompt`. It uses fixture `qwen3.8-27b-chat-template.jinja`.
  - `Suffix_IsNull_WhenTheNoGenerationRenderIsNotAPrefix`. Use an inline template that renders differently.
  - `ChatEndpoint_PassesSuffixInOptions`. Add a small seam: the endpoint builds `InferenceOptions` through a testable static, `ChatCompletionEndpoint.BuildChoiceOptions(...)`. If that refactor is too invasive, assert via `ReasoningSupport` only and cover the endpoint in Task 4's real run.
- [ ] **Step 2: Run them to confirm they fail.**
- [ ] **Step 3: Implement** `GenerationPromptSuffix`. Set it on the options handed to `Generate`, `GenerateStreamingTokensAsync` and `scheduler.EnqueueAsync` in both endpoints.
- [ ] **Step 4: Run them to confirm they pass.** Also run `JinjaQwen3_8_27BReferenceRenderTests` and `ReasoningOpenAiTests`.
- [ ] **Step 5: Commit:** `server: pass the chat template's generation-prompt suffix to the snapshot boundary (#A)`.

### Task 3: Grow exactly before missing; surface refusal as a context error (all three paths)

**Files:**
- Modify: `src/DotLLM.Engine/TextGenerator.cs` (`EnsureRecurrentPrefixKvCapacity` ~1723 and its caller ~1675).
- Modify: `src/DotLLM.Engine/ContinuousBatchScheduler.cs` (~930, `boundary > seq.KvCache!.MaxLength` silently returns 0).
- Modify: `src/DotLLM.Engine/TextGenerator.RecurrentSnapshots.cs` (fresh-KV admission failure path).
- Test: `tests/DotLLM.Tests.Unit/Engine/TextGeneratorRecurrentPrefixCacheTests.cs`

**Interfaces:**
- Produces: the private method `bool EnsureRecurrentPrefixKvCapacity(IKvCache kv, int needed)` keeps its signature but now tries `target` and then `needed`. Add the constant `internal const int RecurrentPrefixRefillMissMaxTokens = 4096`. Below it, a refused growth may still fall back to a miss (re-prefill is cheap). At or above it, throw `DeviceMemoryAdmissionException` with a message naming the needed positions and the KV's current `MaxLength`. The server already maps that to HTTP 400 `context_length_exceeded`.

- [ ] **Step 1: Write failing tests:**
  - Add `LimitedGrowthKv(SimpleKvCache inner, int limit) : IKvCache`. It wraps like `SharedStorageKv`, and `TryGrowTo(n) => n <= limit && inner.TryGrowTo(n)`.
  - `RefusedHeadroomGrowth_RetriesExactSize_AndKeepsTheHit`. Use a limit of `needed`, which refuses `needed + headroom`. Assert the `CachedTokenCount` stays > 0, and that `model.Calls` shows only the suffix was forwarded.
  - `RefusedExactGrowth_OnALongPrompt_ThrowsDeviceMemoryAdmission`. Use a prompt of at least 4096 tokens and a limit below `needed`. Assert it throws `DeviceMemoryAdmissionException`, and that no forward of the whole prompt happened.
  - `RefusedExactGrowth_OnAShortPrompt_FallsBackToAMiss` (the existing behaviour).
  - Add a scheduler equivalent of the first two in `ContinuousBatchSchedulerRecurrentPrefixTests`.
- [ ] **Step 2: Run them to confirm they fail.**
- [ ] **Step 3: Implement** in all three paths. Add an XML/inline comment on the constant explaining why 4096.
- [ ] **Step 4: Run the tests.** Expect the new and existing recurrent-prefix tests to pass, plus `Server/RecurrentPrefixSnapshotLifecycleTests` and `RecurrentPrefixObservabilityTests`.
- [ ] **Step 5: Commit:** `engine: grow a hit's KV to the exact size before missing; refuse long refills (#A)`.

### Task 4: Verify PR A on hardware, docs, PR

**Files:**
- Create: `scripts/session-growth.py`. Copy the session scratchpad `growth.py`, then parameterise it with argparse: `--url`, `--model`, `--turn-chars`, `--turns`, `--out`, `--max-tokens`. Also fix TTFT measurement so it counts reasoning deltas (`reasoning_content`) as the first token.
- Create: `scripts/context-fit-probe.ps1`. Copy `fitprobe.ps1`, then parameterise it: `-Cli`, `-Model`, `-Configs`.
- Modify: `docs/SERVER.md`, section "Prefix reuse across agent turns (#1080)". Add the boundary rule and the growth policy.

- [ ] **Step 1: Full CPU suite.** Run `dotnet test tests/DotLLM.Tests.Unit -c Release --no-build --filter "FullyQualifiedName!~DotLLM.Tests.Unit.Cuda&FullyQualifiedName!~DotLLM.Tests.Unit.Hip&FullyQualifiedName!~DotLLM.Tests.Unit.Vulkan&Category!=GPU&Category!=Benchmark&Category!=RealModel"`. Expect 0 failures. A known order-dependent flake, `OllamaApiEndpointTests.Tags_Lists…`, counts only if it also fails when run alone.
- [ ] **Step 2: Real growth run.** Use q4_0, `--no-mtp`, `DOTLLM_CUDA_PREFILL_CHUNK=512` and `--ctx-size 110592`, with 8k-token turns up to the fit.
  - Expect every turn N≥2 to have `cached_tok` ≥ the previous turn's prompt minus 64. That is the generation prompt plus slack for the rendered reply.
  - Expect TTFT per 8k turn to be about half of the Step 0 baseline.
  - Record the table in the PR.
  - Repeat one turn pair with MTP on (q4_0) to cover the generator path.
- [ ] **Step 3: Docs.** Update `docs/SERVER.md`, then commit `docs: snapshot boundary and growth policy (#A)` with the scripts.
- [ ] **Step 4: Open the PR** (Closes #A). Merge into `dev`, push, and update `.docs/HANDOFF.md`.

## PR B — flash prefill over quantized KV (spec C3)

### Task 5: The kernel

**Files:**
- Create: `native/kernels/attention_flash_mma_prefill_hd256_kvq.cu`. It includes `attention_mma_decode_v2.cuh` for `V2LoadQ8` / `V2LoadQ4`. Factor the F16 kernel's body into a shared `.cuh` (`attention_flash_mma_prefill_hd256.cuh`), templated on the K and V loaders, so the F16 and kvq entries share one body.
- Modify: `native/kernels/attention_flash_mma_prefill_hd256.cu` (use the shared body).
- Modify: `native/build_ptx.bat` (add the new kernel to `ARCH_86` and `CXX17`), `native/build.sh`, `native/build.ps1`, and `src/DotLLM.Cuda/DotLLM.Cuda.csproj` (the arch override list).
- Create: `native/ptx/attention_flash_mma_prefill_hd256_kvq.ptx`. Regenerate `native/ptx/attention_flash_mma_prefill_hd256.ptx`.
- Modify: `src/DotLLM.Cuda/CudaKernels.cs` (load next to ~1423, launch next to ~1500).
- Test: `tests/DotLLM.Tests.Unit/Cuda/CudaAttentionFlashPrefillHd256KvqTests.cs`

**Interfaces:**
- Produces:
  - Four kernel entries, `attention_flash_mma_prefill_hd256_kvq_q8q8`, `_q4q4`, `_q8q4` and `_q4q8`. Signature: `(const half* q, const uint8_t* k, const uint8_t* v, float* output, int seq_q, int seq_kv, int num_heads, int num_kv_heads, int position_offset, float scale)`. Same grid, block and shared-memory size as the F16 kernel. The tile dequant writes F16 into the same padded shared memory, so the MMA body is unchanged.
  - `public bool HasAttentionFlashPrefillHd256Kvq(KvCacheDType k, KvCacheDType v)`.
  - `public void LaunchAttentionFlashPrefillHd256Kvq(nint q, nint k, nint v, KvCacheDType kDtype, KvCacheDType vDtype, nint output, int seqQ, int seqKv, int numHeads, int numKvHeads, int positionOffset, nint stream, float scale = 0f)`.

- [ ] **Step 1: Write the failing test.** `KvqPrefill_BitIdenticalToScratchDequant(seqQ, posOffset, kDtype, vDtype)`:
  - Inputs: for all four dtype pairs, the cases (1,0), (16,0), (17,5), (512,0), (513,3600) and (1000,3600). Use NumHeads 24, NumKvHeads 4, HeadDim 256.
  - Quantize random K and V into Q8_0/Q4_0 blocks on the device with the existing KV-write kernel, or a host quantizer matching the `load_kv_elem` layouts.
  - Reference: `LaunchKvDequantToF16` into scratch, then `LaunchAttentionFlashPrefillHd256`.
  - Assert the outputs are **bit-identical** (`diff == 0`), as `Kvq_FusedBitIdenticalToScratchDequant_V2` does. The dequantized F16 values are the same, so the MMA inputs match. NaN-poison the output first.
  - Add `Bench_KvqPrefill_vs_ScratchDequant(seqKv)` (`Category=Benchmark`) at 4096 and 65536.
- [ ] **Step 2: Build the PTX and run the test to confirm it fails.** Run `native/build_ptx.bat`, then `dotnet test ... --filter FullyQualifiedName~CudaAttentionFlashPrefillHd256KvqTests`. Expect a failure because the function is not loaded.
- [ ] **Step 3: Implement** the kernel, the shared body, the loader and the launch.
- [ ] **Step 4: Run the tests.** Expect the new tests to pass, plus the unchanged `CudaAttentionFlashPrefillHd256Tests` (16 cases, tolerances 1e-3 / 2e-3).
- [ ] **Step 5: Commit** the kernel and PTX: `cuda: flash prefill hd256 over q8_0/q4_0 KV, dequantized in the tile load (#B)`.

### Task 6: Route the model to it; no full-history scratch on the Bonsai path

**Files:**
- Modify: `src/DotLLM.Cuda/Architectures/CudaQwen3HybridDenseTransformerModel.cs`:
  - Dispatch, ~3612-3715: a new `useFlashPrefillHd256Kvq` tier, and `ResolveKvF16` not called on it.
  - `PerRowKvqMaxRows`, ~222: the 9-row verify. Either route `seqLen > PerRowKvqMaxRows` multi-row forwards to the kvq prefill kernel, or raise the limit to 9. Choose the kvq prefill route, which needs no per-row loop.
  - `ContextBytesPerToken`, ~401: no scratch term when the kvq prefill applies.
- Test: `tests/DotLLM.Tests.Unit/Cuda/CudaQwen3HybridDenseKvqPrefillTests.cs`

**Interfaces:**
- Consumes: `HasAttentionFlashPrefillHd256Kvq`, `LaunchAttentionFlashPrefillHd256Kvq` (Task 5).
- Produces: `internal bool FlashPrefillHd256KvqForTest { get; set; }`, default true. `ContextBytesPerToken(bool mtp)` drops `quantSides * kvElems * 2` when the kvq prefill is eligible.

- [ ] **Step 1: Write failing tests** on real Bonsai weights (skip without `DOTLLM_BONSAI_PQ2_0_GGUF`):
  - `Prefill_OverQuantizedKv_AllocatesNoDequantScratch`. For q4/q4, q8/q8 and q8/q4, a 2-tile prefill at positions 0 and 3000 gives `DebugKvDequantCallsForTest == 0`.
  - `Prefill_OverQuantizedKv_MatchesTheScratchPath`. Toggle `FlashPrefillHd256KvqForTest`. Assert the logits are bit-identical, or cosine > 0.9999 with equal argmax if bit-identity fails because of reduction-order differences. Use the same thresholds as `CudaQwen3HybridDensePerRowKvqVerifyTests`.
  - `NineRowVerify_OverQuantizedKv_AllocatesNoDequantScratch`.
  - `ContextBytesPerToken_DropsTheScratchTerm`: q4/q4 without MTP gives `18 * 1024` bytes.
- [ ] **Step 2: Run them to confirm they fail.**
- [ ] **Step 3: Implement the routing.**
- [ ] **Step 4: Run the new tests**, plus `CudaQwen3HybridDensePerRowKvqVerifyTests`, `CudaQwen3HybridDenseFlashPrefillHd256Test`, `CudaQwen3HybridDenseContextFitTests`, `CudaQwen3HybridDensePrefillChunkingTest`, `CudaQwen3HybridDenseMtpPrefillTilingTests` and `CudaQwen3HybridDenseMtpLongContextTests`. Expect all to pass.
- [ ] **Step 5: Hardware fit check.** Run `scripts/context-fit-probe.ps1` with q4_0, no MTP and tile 512. Expect a fitted window of at least 131,072 (about 135k).
- [ ] **Step 6: Commit** `cuda: route quantized-KV prefill and 9-row verifies to the fused kernel; drop the scratch from the fit (#B)`. Update `docs/ATTENTION.md` and `docs/KV_CACHE.md` with the removed 4 KiB/token. List any remaining `ResolveKvF16` callers and their costs in the PR body. Open the PR (Closes #B), merge into `dev`, and push.

## PR C — MTP head KV grows in place (spec C6)

### Task 7: VMM-backed MTP KV

**Files:**
- Modify: `src/DotLLM.Cuda/Architectures/CudaMtpState.cs` (ctor ~84, `EnsureCapacity` ~138, `AllocatedBytes` ~406).
- Modify: `src/DotLLM.Cuda/Architectures/CudaQwen3HybridDenseTransformerModel.cs` (state creation ~802: pass `_kvVirtualMemory`; `ContextBytesPerToken` MTP term).
- Test: `tests/DotLLM.Tests.Unit/Cuda/CudaMtpStateVmmGrowthTests.cs`

**Interfaces:**
- Consumes: `CudaGrowableDeviceBuffer(ICudaVirtualMemory vm, long reserveBytes)`, `EnsureCapacity(long)`, `Pointer`, and `CudaDriverVirtualMemory.TryCreate`.
- Produces:
  - A new optional constructor parameter, `ICudaVirtualMemory? virtualMemory = null`. When it is non-null, K and V are `CudaGrowableDeviceBuffer`s reserved for `maxSteps` rows. `EnsureCapacity` maps only the added bytes, with no copy and no second allocation. When it is null, the code keeps the copying growth.
  - The admission is charged only the added bytes.
  - `ContextBytesPerToken` MTP term: `2L * head.NumKvHeads * HeadDim * sizeof(float) + sizeof(int)` when in-place growth is active, and the existing doubled term otherwise.
  - The iota table may keep copying (4 bytes per position), but note its cost in a comment.

- [ ] **Step 1: Write failing tests:**
  - `Growth_InPlace_KeepsRowsAndPointer`. Use the real CUDA VMM, skipped without a device. Write rows at positions 0 to 1023, grow to 3000, and assert the K pointer is unchanged and rows 0 to 1023 read back bit-identical.
  - `Growth_InPlace_AdmitsOnlyAddedBytes`. Record `admitGrowth` calls and assert the charged bytes are ≤ the added rows × row bytes + iota.
  - `ContextBytesPerToken_WithMtp_CountsTheHeadOnce`. On real Bonsai, q4/q4 with MTP gives `(18 + 8) * 1024 + 4` bytes after PR B.
  - Also run the existing MTP greedy parity tests (`CudaQwen3HybridDenseMtpLongContextTests`, `TextGeneratorRecurrentPrefixMtpTests`).
- [ ] **Step 2: Run them to confirm they fail. Step 3: Implement. Step 4: Run them to confirm they pass.**
- [ ] **Step 5: Hardware fit.** Run q4_0 + MTP with tile 512. Expect a fit of about 93k (the spec's arithmetic). Record the actual value.
- [ ] **Step 6: Commit** `cuda: grow the MTP head's KV in place (#C)`. Update `docs/SERVER.md` (fit table) and `docs/SPECULATIVE.md`, then open the PR (Closes #C), merge into `dev` and push.

## PR D — window-hint ids (spec C4)

### Task 8: Parse the hint

**Files:**
- Create: `src/DotLLM.Server/ContextWindowHint.cs`
- Test: `tests/DotLLM.Tests.Unit/Server/ContextWindowHintTests.cs`

**Interfaces:**
- Produces:
  - `readonly record struct ContextWindowHint(int Tokens, bool Mtp)`.
  - `static bool TryParse(string modelId, out string baseId, out ContextWindowHint? hint, out string? error)`.
    - The grammar is a trailing `[<size>]` or `[<size>-mtp]`, case-insensitive. `<size>` is `<n>k` (×1024), `<n>m` (×1,048,576) or a plain integer ≥ 1024.
    - With no brackets, it returns true with a null hint.
    - It returns false with an error for malformed brackets (`[`, `[0k]`, `[12q]`, `[128k-foo]`, or an overflow above `int.MaxValue`).
  - `static string Format(string baseId, ContextWindowHint hint)` returns the canonical id, lowercase `k`, e.g. `bonsai[128k-mtp]`.

- [ ] **Step 1: Write failing tests:**
  - `[Theory]` valid forms: `m[128k]` gives (131072, false), `M[128K-MTP]` gives (131072, true), `m[1m]` gives 1048576, `m[65536]` gives 65536, and `owner/repo:Q4_K_M[64k]` gives baseId `owner/repo:Q4_K_M`.
  - Invalid forms, as above.
  - The round-trip `Format(TryParse(x))` is canonical.
- [ ] **Step 2: Fail. Step 3: Implement. Step 4: Pass. Step 5: Commit** `server: parse model[Nk] / model[Nk-mtp] window hints (#D)`.

### Task 9: Fit record store and the settings ladder

**Files:**
- Create: `src/DotLLM.Core/Configuration/ContextFitRecord.cs` (record + store)
- Create: `src/DotLLM.Server/ContextWindowPlanner.cs` (ladder)
- Modify: `src/DotLLM.Cuda/Architectures/CudaQwen3HybridDenseTransformerModel.cs`. Make the per-token arithmetic static: `public static long ContextBytesPerToken(ModelConfig config, KvCacheDType key, KvCacheDType value, bool mtp, bool kvqPrefill, bool mtpInPlace, int mtpHeadKvHeads)`. The instance method delegates to it.
- Test: `tests/DotLLM.Tests.Unit/Server/ContextWindowPlannerTests.cs`, `tests/DotLLM.Tests.Unit/Core/ContextFitStoreTests.cs`

**Interfaces:**
- Produces:
  - `sealed record ContextFitRecord(long FreeAfterLoadDefaultTileBytes, long FreeAfterLoadSmallTileBytes, int DefaultTileTokens, int SmallTileTokens, long ReserveBytes, long FixedBytes, IReadOnlyDictionary<string, long> BytesPerTokenBySetting, int NativeContext, DateTime MeasuredUtc)`. The keys are `"{k}/{v}/{mtp|nomtp}"` with k and v in `f16|q8_0|q4_0`. Free-after-load is measured at each tile because the tile's forward scratch is what differs between them. On this box that is 2,426 MiB at the default tile and 2,782 MiB at the 512 tile. `FixedBytes` is the fit's existing fixed term (149 MiB).
  - `readonly record struct ContextFitKey(string Gpu, string Driver, string ModelPath, long ModelBytes, long ModelMtimeUtcTicks)`. `ToString()` joins the fields with `|`. Build it with `ForFile(...)`, reading size and mtime as `MoeCalibrationKey.ForFile` does.
  - `static class ContextFitStore` with `int Version = 1`, `string DefaultPath` (`%LOCALAPPDATA%\dotllm\context-fit.json`), `ContextFitRecord? TryLoad(string path, ContextFitKey key)` and `bool TrySave(string path, ContextFitKey key, ContextFitRecord record)`.
    - Copy the `MoeCalibrationStore` pattern: `Utf8JsonWriter`/`JsonDocument`, an atomic temp-file move, preserving other entries, and reads that never throw.
  - `sealed record ContextWindowPlan(KvCacheDType Key, KvCacheDType Value, int PrefillTileTokens, bool Mtp, int Window)`.
  - `static class ContextWindowPlanner`:
    - `ContextWindowPlan? Choose(ContextFitRecord fit, ContextWindowHint hint, KvCacheDType? fidelityCeiling, out string refusal)`.
    - The ladder is F16, then q8_0, then q4_0 (both sides), each at the default tile and then at the small tile, with MTP set by the hint.
    - `fidelityCeiling` (from an explicit `--kv-quant-k/-v`) skips rungs above it.
    - The window per rung is `(FreeAfterLoad(tile) − Reserve − Fixed) / BytesPerToken`, rounded down to 1024 and capped at native. This is the same formula as `DeviceMemoryBudget.MaxFittingContext`, so reuse it.
    - The first rung reaching `hint.Tokens` wins.
    - On refusal, `refusal` lists the best window per MTP mode in the spec's form: `"{id} does not fit: q4_0 + MTP reaches 93k; [128k] (MTP off) fits"`. When the non-MTP form fits too, say so. Otherwise give its best.

- [ ] **Step 1: Write failing tests:**
  - The planner, using synthetic fits shaped like the measured box: free 2,426 MiB (default tile) and 2,782 MiB (512 tile), reserve 256 MiB, fixed 149 MiB, and bytes/token f16 = 64 KiB, q8 = 34 KiB, q4 = 18 KiB, plus 8 KiB with MTP (after PRs B and C). Expected windows: q4/no-MTP at the 512 tile is (2782 − 405) MiB / 18 KiB = 135,225 → 134,144 rounded down to 1024; q4 + MTP at the 512 tile is ≈ 93,184.
    - `Choose_PicksTheMostFaithfulRungThatFits`: `[32k]` gives f16 at the default tile.
    - `Choose_FallsToQ4AndSmallTile_For128k`.
    - `Choose_Refuses128kMtp_WithTheSpecMessage`.
    - `Choose_HonoursFidelityCeiling`: a q8_0 ceiling means no f16 rung.
  - The store: `TrySave_ThenTryLoad_RoundTrips`, `TryLoad_ReturnsNull_ForCorruptOrWrongVersion` and `TrySave_PreservesOtherEntries`, using a scratch path.
- [ ] **Step 2: Fail. Step 3: Implement. Step 4: Pass. Step 5: Commit** `server: context-fit record store and window-hint settings ladder (#D)`.

### Task 10: Reconfigure a loaded CUDA hybrid in place

**Files:**
- Modify: `src/DotLLM.Cuda/Architectures/CudaQwen3HybridDenseTransformerModel.cs`. Make the prefill tile an instance setting that falls back to `PrefillChunkTokens`. Add the method below. `ConfigureKvStorage` (~278) stays as is.
- Modify: `MeasureServingContextFit` (~427). Add `HybridContextFit[] MeasureFitRecordInputs()`, or have `ContextFitRecord.FromModel(...)` measure the fixed bytes at both tiles in one call.
- Test: `tests/DotLLM.Tests.Unit/Cuda/CudaQwen3HybridDenseReconfigureTests.cs`

**Interfaces:**
- Produces:
  - `public void Reconfigure(KvCacheDType key, KvCacheDType value, int prefillTileTokens, int contextLimit)`.
    - It requires no live sequence. It throws `InvalidOperationException` if a KV handle is outstanding. Track live `CudaHybridKvCacheHandle`s with a count.
    - It frees the KV storage, the dequant/F32 staging, the checkpoints and the row snapshots, then sets the KV dtypes, the tile and `Config.MaxSequenceLength = min(contextLimit, NativeMaxSequenceLength)`.
    - Raising the context limit back is allowed.
  - `public int PrefillTileTokens { get; }`.
  - `public ContextFitRecord MeasureContextFitRecord()`. It runs the existing fit measurement once at the default tile and once at 512 (restoring the tile afterwards), records free-after-load for each plus the fixed term, and computes bytes/token for f16, q8_0 and q4_0 (same type on both sides) × MTP on/off using the Task 9 static.

- [ ] **Step 1: Write failing tests** on real Bonsai (skip without the env var):
  - `Reconfigure_SwitchesKvTypeAndTile_WithoutReloadingWeights`. Load, run a short prefill+decode at f16, `Reconfigure(q4_0, q4_0, 512, 120_000)`, run again, and assert the logits argmax matches the same prompt run on a freshly loaded q4_0 model.
  - `Reconfigure_RepeatedSwitches_ReturnFreeVramWithin64MiB`. Run 5 alternations and compare `QueryFreeDeviceBytes()` before and after.
  - `Reconfigure_WithALiveKvHandle_Throws`.
  - `MeasureContextFitRecord_PredictsTheServedFit`. For q4/q4 without MTP at tile 512, the predicted window is within 2048 tokens of `MeasureServingContextFit` after a `Reconfigure` to that setting.
- [ ] **Step 2: Fail. Step 3: Implement. Step 4: Pass**, plus the existing `CudaQwen3HybridDenseContextFitTests` and `CudaQwen3HybridDensePrefillChunkingTest`.
- [ ] **Step 5: Commit** `cuda: reconfigure a loaded hybrid's KV type, prefill tile and window in place (#D)`.

### Task 11: Server integration

**Files:**
- Modify: `src/DotLLM.Server/ServerStartup.cs`:
  - Strip the hint in `ResolveModelPath` (~39), `ModelIdFor` (~52) and `ResolveOrPullAsync` (~63).
  - Apply the plan in `LoadModelCore` (~321), around `ConfigureKvStorage` (~484) and `ApplyHybridContextFit` (~257).
  - Save the record after the first fit.
- Modify: `src/DotLLM.Server/ServerState.cs`:
  - `ActivateLockedAsync` (~729): resolve the hint before `Catalog.IsEnabled` / `ResolveOrPullAsync`, predict before `FreeForIncomingLocked` (~819), and reconfigure in place when the base weights are already active (~791 options).
  - `MatchesActiveAndReady` (~994).
  - Residency key: weights path, with the active variant id as state.
- Modify: `src/DotLLM.Server/ServerOptions.cs`: add `ContextWindowHint? WindowHint` (nullable, no initializer) and `string? CacheTypeExplicitK/V`. Alternatively derive "explicit" from `CacheTypeK != "f32"` and document that `f32` means not given.
- Modify: `src/DotLLM.Server/Models/ModelListResponse.cs` (`ModelInfoDto`: add `[JsonPropertyName("max_sequence_length")] int? MaxSequenceLength`), `ModelEndpoint.BuildList` (~52), `ResidentModelInfo`/`ResidentModelSnapshot` (carry the window).
- Modify: `src/DotLLM.Server/Models/PropsResponse.cs` + `Endpoints/PropsEndpoint.cs`: add `kv_type_k`, `kv_type_v`, `prefill_tile`, `window_hint` (nullable).
- Modify: `src/DotLLM.Cli/Commands/ServeCommand.cs` (~478-505): parse the hint before `GgufFileResolver.Resolve`. A plan refusal throws `ContextWindowUnavailableException` (new, in `src/DotLLM.Server/`). Catch it next to `DeviceUnavailableException` (~500), print it in red, and return 1.
- Modify: `src/DotLLM.Server/ServerJsonContext.cs` only if a new DTO is added. Also update `JsonContextDefaultsTests.ResponseOnly` if needed.
- Test: `tests/DotLLM.Tests.Unit/Server/WindowHintServerTests.cs`

**Interfaces:**
- Consumes: Tasks 8, 9 and 10.
- Produces:
  - The rules below.
  - `ContextWindowUnavailableException : Exception` with a `string Refusal` property. The server maps it to HTTP 400 with `code: "context_length_exceeded"` and the refusal text as the message.

**Rules (from the spec, verbatim decisions):**
- For a hinted id, the hint's window and MTP override `--ctx-size`, `--no-mtp` and `--mtp`.
- An explicit `--kv-quant-k/-v` is a fidelity ceiling.
- A request whose stored record predicts a refusal is refused before anything is unloaded.
- With no record, load once and measure. If the measured plan still refuses, revert the model to its previous variant (or unload it if there was none), then refuse.
- Variants sharing a weights file share one loaded model. Switching runs `Reconfigure` plus a new `TextGenerator` (MTP is fixed per generator, see the `ServerState.cs:1154` rebuild precedent), and drops the prefix snapshots.
- A switch waits on `_requestGate` like a model switch.
- A `serve` hint that doesn't fit exits with code 1 and the refusal text.

- [ ] **Step 1: Write failing tests**, without a GPU and using the `state.ModelLoader` seam plus a fake reconfigurable model behind an `internal interface IWindowReconfigurable { ContextFitRecord MeasureContextFitRecord(); void Reconfigure(...); int MaxContext {get;} }`. The CUDA model implements it.
  - `HintedRequest_WithStoredRefusingRecord_RefusesWithoutEvicting`. Assert the active model is unchanged and the loader is not called.
  - `HintedRequest_ForActiveWeights_ReconfiguresInPlace_WithoutReload`. Loader call count stays 1, and one `Reconfigure` call is made.
  - `HintedRequest_OverridesServeNoMtpAndCtxSize`.
  - `ExplicitKvQuant_IsAFidelityCeiling`.
  - `HintedRequest_WhileAnotherIsInFlight_WaitsThenReconfigures`. Use the `ModelSwapRobustnessTests` `StreamAsync` helpers and assert there is no `Reconfigure` call before the first request completes.
  - `ModelsList_ReportsMaxSequenceLength_ForTheLoadedVariant`, using `ModelEndpoint.BuildList`.
  - `Serve_WithAnUnfittableHint_ReturnsExitCode1`, using `ServeCommand` with a stubbed loader if the seam exists. Otherwise test the catch with `ContextWindowUnavailableException` thrown from `LoadModel`.
  - Add `JsonContextDefaultsTests` coverage, which runs automatically for the changed DTOs.
- [ ] **Step 2: Fail. Step 3: Implement. Step 4: Pass**, plus the full `tests/DotLLM.Tests.Unit/Server` namespace.
- [ ] **Step 5: Commit** `server: model[Nk] / model[Nk-mtp] ids choose, advertise and refuse context windows (#D)`.

### Task 12: Measure end to end, docs, PR

- [ ] **Step 1: Full CPU suite** (the filter from Task 4) and the GPU set:
  - `CudaAttentionFlashPrefillHd256*`
  - `CudaQwen3HybridDense*`
  - `CudaMtpState*`
  - `TextGeneratorRecurrentPrefix*`
  - `CudaGrowableDeviceBufferTests`
  
  Expect 0 failures.
- [ ] **Step 2: Real runs** with `scripts/session-growth.py`. Restart the server between runs.
  - `dotllm serve "<bonsai>[128k]"` with 8k turns to 128k.
  - `dotllm serve "<bonsai>[64k-mtp]"` with 8k turns to 64k.
  - `dotllm serve "<bonsai>[128k-mtp]"`, which must exit 1 with the refusal text. Record the text.
  - One in-place switch, `[64k-mtp]` → `[128k]`, through requests. Measure the switch time and free VRAM before and after.
- [ ] **Step 3: Docs.** In `docs/SERVER.md`:
  - Add a "Context window hints" section: grammar, ladder, precedence, predict-before-evict, in-place switching, the startup exit, the fit cache file, and the tiered-path note (long contexts need the host tier).
  - Replace the fit and growth tables with the new measurements, keeping the old ones as "before".
  - Update `README.md` News if this is a significant milestone.
- [ ] **Step 4: PR** (Closes #D). Merge into `dev`, push, and update `.docs/HANDOFF.md` with results and the follow-ups: quantize the MTP head KV, the Nemotron-H ports, and prefill throughput.
