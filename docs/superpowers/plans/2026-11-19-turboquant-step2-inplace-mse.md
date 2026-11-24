# TurboQuant step 2: in-place MSE attention on CPU, CUDA and Vulkan — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `--cache-type-k tqN --cache-type-v tqN` (MSE stage, N in 1..8) runs TurboQuant KV for real, read in place with no decoded
history, on:
- the CPU `TransformerModel`;
- the CUDA dense non-paged `CudaTransformerModel`;
- the Vulkan dense chain and Flint;

for decode and prefill, from serve, run, chat and bench.

**Architecture:**
- The query is rotated in place with the K signs: `q~ = H(S_k ⊙ q)/√d`.
- Each K and V element is then the per-element value `norm[pos,h] · centroid[code_i]`. That makes TurboQuant one more **per-side KV
  format** in the existing in-place twin machinery:
  - CUDA `KvqFmt` / `load8` / `PkLoadQuant`;
  - Vulkan `kv_load.glsl`;
  - CPU `LoadHeadRow`.
- The output is rotated back in place with the V signs, after any split-KV combine: `o = S_v ⊙ H(o)/√d`.
- Codes and norms live in per-layer buffers sized to the cache; nothing wider is stored.

**Tech Stack:** C#/.NET 10; CUDA C (PTX via `native/msbuild/CudaPtx.targets`); GLSL compute (glslc via `CompileVulkanShaders`); xUnit.

**Spec:** `docs/superpowers/specs/2026-11-18-turboquant-in-place-design.md` (approach A, step 2). Issue #1892 (its checklist is Task 9).
Epic #1887.

## Global Constraints

- **Branching:**
  - One integration branch `issue/1892-tq-inplace` and **one PR to `dev`**, so nothing user-visible lands until all three backends pass.
  - Commits carry `(#1892)`.
  - The PR says `Closes #1892`; close it by hand after the merge.
- **CLAUDE.md rule 1:** no FP32 storage of the history. Accumulation is F32 in registers. The norm format is decided by Task 1.
  - An FP32 norm needs James's sign-off, with per-candidate data and the options still open.
- **CLAUDE.md rule 2:** kernels read codes in place. A per-tile decode into registers or shared memory is allowed; any full-history or
  persistent decoded copy is a defect.
- **Hosts in scope:**
  - CPU `TransformerModel`;
  - CUDA `CudaTransformerModel` (dense, non-paged);
  - Vulkan `VulkanTransformerModel` dense chain (`IsDenseF16KvEligible`) and Flint.
  
  Every other host keeps refusing via #1900.
- **Refused in step 2:**
  - `tq*q` on GPU (step 5, #1894); CPU `tq*q` stays on the scratch path;
  - paged (step 3);
  - mixed and V-only (step 6);
  - head dim not a power of two or above 256 (#1897);
  - `--cache-window > 0` with tq (new; there is no original-space ring with TurboQuant).
- **Formats are a per-side pair:** a K format and a V format. TurboQuant on a side means that side reads codes. The pre-rotation is
  gated on K being tq, the post-rotation on V being tq. Mixed pairs stay refused until step 6, but the code is pair-shaped.
- **Merge gates:**
  - in-place output matches the scratch-path oracle with the **same** norm format;
  - encode is byte-exact across backends;
  - in-place is no slower than the scratch path at 7.6K and 32K context;
  - KL, PPL and greedy identity against f16 are reported per bit width (2, 3, 4, 8), not gated.
- Every correctness gate is shown able to fail, with mutants named in the PR.
- **GPU coordination:**
  - Strix runs hold ResourceLock `gpu`, and the job starts only after the lock is confirmed.
  - CUDA runs on framework, in a window agreed with James up front; that machine is functional-only.
  - Give the time estimate and cap before any run longer than 30 min.
- **Build traps:**
  - Rebuild after touching `.spv` (they are copied into `bin`).
  - The `.ptx` files are committed.
  - Rebuild after mutant runs before any `--no-build` run.

## Review Focus

1. **Vulkan split-KV is on by default from about 17 rows of context.** The output rotation must run after the combine. A per-split
   rotation is still linear and would appear to work, but the combine of `max`/`sum` partials is not, so it gives wrong output on
   long contexts. Tested in Task 7.
2. **A prefill chunk whose current rows are attended from fresh K/V rather than from the cache.** If any arm does that, it mixes
   original-space rows into code-space attention. Tested in Tasks 4, 6 and 7 with fixtures where only the current-chunk rows differ.
3. **CUDA decode under graph capture** (`attention_kvq_dyn`). The TurboQuant twin and both rotations must be capture-safe. Tested in
   Task 6.
4. **A `--device auto` that lands on a host without the full twin set** (e.g. Vulkan Gemma-4) must refuse, not fall back. Tested in
   Task 8 through `SupportsInPlaceTurboQuant`.
5. **Bit width 3, where codes straddle uint boundaries, and bit width 8**, where an hd256 centroid table is 1 KB. These must work on
   every backend. Every parity Theory includes bits 3 and 8.

---

### Task 0: Branch, issue checklist, GPU window

- [ ] Create the worktree `issue/1892-tq-inplace` from `dev`. Commit this plan.
- [ ] Ask James (AskUserQuestion) for a framework CUDA window for Tasks 6 and 10, with an estimate: about 2 h of builds and tests plus
  about 1 h of benches.
- [ ] Fix the PTX build-path divergence: add `turboquant.cu` to the csproj `-fmad=false` list (`DotLLM.Cuda.csproj:82`) to match
  `build_ptx.bat:86`. Rebuild `turboquant.ptx` and commit it.

### Task 1: Shared layout and the norm-format decision

**Files:**
- Create: `src/DotLLM.Core/Attention/TurboQuantLayout.cs`
- Create: `src/DotLLM.Engine/KvCache/Codecs/TurboQuantNormCodec.cs`
- Modify: `TurboQuantCodec.cs`: expose `FastWalshHadamard` as `internal static`, and add the argmin tie policy.
- Create: `tools/tq-norm-eval/` (a bounded harness)
- Test: `tests/DotLLM.Tests.Unit/Engine/KvCache/TurboQuantNormCodecTests.cs`

**Interfaces (produces):**
```csharp
public enum TqNormFormat : byte { F32Reference, F16, Bf16, Fp8E4M3, Fp8E5M2, Log8 }   // F32Reference: oracle/A-B only
public readonly record struct TurboQuantLayout(int HeadDim, int NumKvHeads, int MseBits, TqNormFormat Norm) {
  public int CodeUintsPerVec { get; }   // ceil(ceil(HeadDim*MseBits/8)/4)
  public int NormBytes { get; }         // 4/2/2/1/1/1
  public long CodeBytesPerLayer(int rows);   // rows*NumKvHeads*CodeUintsPerVec*4 + 4 (guard uint)
  public long NormBytesPerLayer(int rows); }
public static class TurboQuantNormCodec {
  public static void Encode(TqNormFormat f, ReadOnlySpan<float> norms, Span<byte> dst);
  public static float Decode(TqNormFormat f, ReadOnlySpan<byte> src, int index);
  public const float Log8Min = 1e-4f, Log8Max = 1e4f; }   // Log8: round(255*(log2 n - log2 Min)/(log2 Max - log2 Min)); 0 maps to code 0 = exact 0
```
- **Tie policy (all backends):** argmin picks the **lower** centroid index on an exact tie. Encode computes in F32 with no FMA
  contraction. On Vulkan, add `precise` / NoContraction to the encode arithmetic.

- [ ] **Step 1: Write the failing tests:**
  - `[Theory] Roundtrip_RelativeError_WithinFormatBound(format)`, with bounds:
    - F16 2^-11;
    - BF16 2^-8;
    - E4M3 2^-4;
    - E5M2 2^-3;
    - Log8: relative step `2^((log2 1e4 - log2 1e-4)/255) - 1`, about 3.7%.
    
    Inputs are log-uniform over [1e-3, 1e3], plus exact 0.
  - `Saturation_IsCountedNotInfinite`: values above the range clamp to the max, and are never Inf or NaN.
  - `Layout_ByteCounts`: hd128, bits 4, nKv 8 → `CodeUintsPerVec == 16`; F16 → `NormBytes == 2`.
  - `ArgminTie_PicksLowerIndex`.
- [ ] **Step 2:** Run `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~TurboQuantNormCodec"`. Expected: FAIL (no
  types).
- [ ] **Step 3:** Implement. FP8 encodes round-to-nearest-even with saturation.
- [ ] **Step 4:** Re-run. Expected: PASS. Mutant: Log8 using `floor` instead of `round`. The bound test must die.
- [ ] **Step 5: The measurement harness** `tools/tq-norm-eval` (a console project, CPU):
  - It loads a GGUF, runs the existing scratch-path `TurboQuantKvCache` with norms quantised and dequantised through
    `TurboQuantNormCodec`, and an f16 `SimpleKvCache` reference.
  - Teacher-forced over the wikitext fixture slice, it reports mean KL per (bits ∈ {2, 3, 4, 8}) × (format) at contexts 2K and 8K.
  - **Decode check:** greedy-identity tokens over 256 generated tokens per arm, each against the F32-norm arm.
  - Models: Qwen2.5-0.5B q8_0 (the #1884 fixture) and Qwen3-4B (resolved via `TestFixtureResolver`).
  - Estimate about 40 min on CPU. Run it in the background and give the estimate first.
- [ ] **Step 6: Decide.** Pick the smallest format whose KL cost against F32 norms is under 10% of the tq-vs-f16 KL at every bit
  width, on both models.
  - Record the table in `docs/PRECISION.md` (TurboQuant norms) and in the PR.
  - **If nothing below FP32 passes:** stop and ask James, with the table and the options (per-head scale, learned range, a split).
  - The decision becomes `TurboQuantLayout.DefaultNorm`.
- [ ] **Step 7:** Commit `feat(tq): shared layout, norm codecs and the norm-format decision (#1892)`.

### Task 2: One oracle with the chosen norm format

**Files:**
- Modify: `TurboQuantKvCache.cs`, `CudaTurboQuantKvCache.cs`, `VulkanTurboQuantKvCache.cs`: take a `TqNormFormat`, store norms in it,
  and decode through the same codec.
- Modify: the CUDA and Vulkan dequant kernels: decode the norm per the format.
- Tests: extend `TurboQuantKvCacheTests`, `CudaTurboQuantKvEndToEndTests`, `VulkanTurboQuantKvCacheTests`.

- [ ] **Step 1:** Write the failing test `Oracle_NormsRoundTripThroughTheSharedCodec`, on each backend: the stored norm bytes equal
  `TurboQuantNormCodec.Encode` of the encode-time norm.
- [ ] **Step 2:** Run it. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Re-run. Expected: PASS.
- [ ] **Step 5: Cross-backend encode byte-exactness.** Test `Encode_IsByteIdentical_CpuCudaVulkan(bits ∈ {1,2,3,4,8}, hd ∈ {64,128,256})`
  on random K/V with ties injected. It's GPU-gated; CUDA runs on framework.
  - Mutant: swap the tie policy on one backend. The test must die.
- [ ] **Step 6:** Commit `feat(tq): oracles store the decided norm format; encode byte-exact across backends (#1892)`.

### Task 3: Rotation kernels and the per-side format pair

**Files:**
- **CPU:** `src/DotLLM.Cpu/Kernels/TurboQuantRotation.cs` wraps `Hadamard.ForwardRow`/`InverseRow` with `blockSize = headDim` and
  sbyte signs converted once.
- **CUDA:**
  - `native/kernels/turboquant_rotate.cu` with `tq_rotate_q_f16`, `tq_rotate_o_f32` and `_dyn` capture-safe twins (row count read from
    a device pointer);
  - add it to the csproj `compute_86` list if needed.
- **Vulkan:** `native/vulkan/shaders/tq_rotate_q.comp` and `tq_rotate_o.comp`, one workgroup per (row, head), with the butterfly in
  shared memory (the `turboquant_dequant_f32.comp` step-2 pattern).
- **Format pair:** `src/DotLLM.Core/Attention/KvSideFormat.cs` defines `enum KvSideFormat { F16, F32, Q8_0, Q4_0, Tq }` and
  `readonly record struct KvFormatPair(KvSideFormat K, KvSideFormat V)`.
  - Vulkan maps `VulkanKvStorage` onto it.
  - New storages: `VulkanKvStorage.Tq` (both sides).
  - Keep the enum for existing values; new combinations come through the pair.

**Interfaces (produces):**
- **CPU:**
  - `TurboQuantRotation.RotateQueryInPlace(float* q, int rows, int numHeads, int headDim, ReadOnlySpan<sbyte> signsK)`;
  - `TurboQuantRotation.RotateOutputInPlace(float* o, int rows, int numHeads, int headDim, ReadOnlySpan<sbyte> signsV)`.
- **CUDA:** `CudaKernels.LaunchTqRotateQ(nint qF16, int rows, int numHeads, int headDim, nint signsK, nint stream)` and
  `LaunchTqRotateO(nint oF32, …, nint signsV, …)`, plus `Dyn` variants.
- **Vulkan:** `TqRotateKernel.RecordQuery(cmd, qBuf, rows, numHeads, headDim, signsK)` / `RecordOutput(...)`.

- [ ] **Step 1:** Write the failing tests:
  - `RotateQuery_MatchesCodecRotate` and `RotateOutput_InvertsCodec`, per backend, against `TurboQuantCodec`'s
    `Rotate`/`InverseRotate`;
  - `QueryDotCodes_EqualsQueryDotDecodedK` on CPU: `q~ · (norm·centroid[code])` equals `q · Decode(code)` to 1e-5 relative.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Re-run. Expected: PASS.
- [ ] **Step 5: Mutants:**
  - V signs used in the query rotation;
  - the `1/√d` factor dropped.
  
  Both must die.
- [ ] **Step 6:** Commit `feat(tq): in-place Q/O rotation kernels on CPU, CUDA, Vulkan; per-side KV format pair (#1892)`.

### Task 4: CPU in-place attention and cache

**Files:**
- Create: `src/DotLLM.Engine/KvCache/TurboQuantInPlaceKvCache.cs`. It implements `IQuantizedKvCache` and a new
  `ITurboQuantKvCache : IQuantizedKvCache`, with `KeyDType`/`ValueDType = TurboQuant`, codes plus norms per layer, no window, and no
  scratch.
- Create: `src/DotLLM.Core/Attention/ITurboQuantKvCache.cs`: `TurboQuantLayout Layout`; `nint GetKeyNormsPtr(int layer)`,
  `GetValueNormsPtr(int layer)`; `ReadOnlySpan<float> Centroids`; `ReadOnlySpan<sbyte> SignsK`, `SignsV`.
- Modify: `src/DotLLM.Cpu/Kernels/Attention.cs`: `LoadHeadRow` gets a `TurboQuant` arm that decodes `norm·centroid[code]` per element
  for the row (in registers or the existing row temp), with the same signature and paths.
- Modify: `TransformerModel.cs` (attention site, about `:1813`): when the cache is `ITurboQuantKvCache`:
  - after Q prescale, `RotateQueryInPlace`;
  - then `Attention.Execute(IQuantizedKvCache…)`;
  - then `RotateOutputInPlace`.
- Test: `tests/DotLLM.Tests.Unit/Cpu/TurboQuantInPlaceAttentionTests.cs`

- [ ] **Step 1: Write the failing tests:**
  - `[Theory] InPlace_MatchesScratchOracle(bits ∈ {2,3,4,8}, hd ∈ {64,128,256}, groups ∈ {1,4}, seqQ ∈ {1,7,64}, window ∈ {null,32}, sinks ∈ {off,on})`:
    - same seed and norm format;
    - tolerance `1e-4·√(seqKv·hd)` relative to the output L2.
  - `CurrentChunkRows_ComeFromTheCache`: two prefills that differ only in the current chunk's K/V. Both the in-place and oracle outputs
    change, and they agree.
  - `HistoryIsNeverDecodedWhole`: an allocation probe. No allocation at or above `rows·hd·4` bytes during attention.
- [ ] **Step 2:** Run with `--filter "FullyQualifiedName~TurboQuantInPlaceAttention"`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Re-run. Expected: PASS.
- [ ] **Step 5: Mutants:**
  - skip the output rotation;
  - drop the norm in the V load;
  - bit offset off by one;
  - tight packing instead of `CodeUintsPerVec`;
  - pre-rotate with the V signs.
  
  Each must die.
- [ ] **Step 6:** Commit `feat(cpu): TurboQuant KV read in place (#1892)`.

### Task 5: CUDA in-place cache

**Files:**
- Create: `src/DotLLM.Cuda/CudaTurboQuantInPlaceKvCache.cs`: per-layer codes and norms, the shared centroids, `signsK`/`signsV`, no
  scratch. Encode reuses `turboquant.cu`'s F16 encode, writing norms in the decided format.
- Modify: `CudaTransformerModel.CreateKvCache(int, KvCacheConfig)` returns it for `kv.IsTurboQuantAny` (both-side tq), admitted through
  `AdmitReleasingG3`.
- Test: `tests/DotLLM.Tests.Unit/Cuda/CudaTurboQuantInPlaceKvCacheTests.cs` (GPU).

- [ ] **Step 1:** Write the failing tests `Codes_MatchCpuEncode_ByteForByte` and `Norms_InDecidedFormat`.
- [ ] **Step 2:** Run them on framework. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Re-run. Expected: PASS.
- [ ] **Step 5:** Commit `feat(cuda): TurboQuant in-place KV cache (#1892)`.

### Task 6: CUDA attention twins

**Files:**
- Modify:
  - `native/kernels/attention_kvq.cu`: a `KvqFmt<TQ>` with `elem(row, h, i)` = `norm·centroid[code]`; extra params
    `k_norms, v_norms, centroids, mse_bits, code_uints_per_vec, norm_format`; the `k_fmt`/`v_fmt` runtime branch gains TQ. Also the
    `_dyn` twin.
  - `attention_mma_decode_v2.cuh`: `V2LoadTq::load8`.
  - `attention_flash_mma_prefill_kvq.cu`: a `PkLoadQuant<Tq>` chunk; new `PK_ENTRY` exports
    `attention_flash_mma_prefill_kvq_hd{64,128}_tqtq`.
  - `attention_flash_mma_prefill_hd256_kvq.cu`: a staged TQ loader; export `_tqtq`.
- Modify: `CudaKernels.cs`: `KvqIndex` gains `tqtq`; add the Has*/Launch* overloads with the norm and centroid pointers.
- Modify: `CudaTransformerModel.DispatchQuantizedKvAttention`: for `CudaTurboQuantInPlaceKvCache`, `LaunchTqRotateQ`, then the kvq
  dispatch as today (flash prefill when eligible, else scalar), then `LaunchTqRotateO`. Under graph capture use the `Dyn` variants.
- Test: `tests/DotLLM.Tests.Unit/Cuda/CudaTurboQuantInPlaceAttentionTests.cs` (GPU, framework).

- [ ] **Step 1: Write the failing tests:**
  - `[Theory] Scalar_MatchesOracle(bits ∈ {2,3,4,8}, hd ∈ {64,128,256}, groups, seqQ ∈ {1,7}, sinks)`;
  - `FlashPrefill_MatchesOracle(hd ∈ {64,128,256}, seqQ ∈ {16,64})`;
  - `CurrentChunkRows_ComeFromTheCache`;
  - `GraphCapturedDecode_MatchesEager`.
  
  The oracle is `CudaTurboQuantKvCache` with the same norm format. Tolerance `2e-3·√(seqKv·hd)` (Q is half).
- [ ] **Step 2:** Run them on framework. Expected: FAIL.
- [ ] **Step 3:** Implement. **Keep the centroid table out of shared memory**: read it from global (`__ldg`), or from constant memory
  when bits ≤ 4.
- [ ] **Step 4:** Re-run. Expected: PASS.
- [ ] **Step 5: Mutants:** the same five as Task 4, plus "`_dyn` rotation reads a stale row count". Each must die.
- [ ] **Step 6:** Commit `feat(cuda): TurboQuant in-place attention twins (#1892)`.

### Task 7: Vulkan cache, writer and twins

**Files:**
- **Shared loader:** `native/vulkan/shaders/kv_load.glsl` gains `KV_K_TQ` / `KV_V_TQ`.
  - Element `i` in the `[pos, nKv·HD]` grid: `hv = i / HD`, `c = i % HD`; read `mseBits` bits at `hv·codeUintsPerVec·32 + c·mseBits`
    (straddle-safe); then `centroid[code] · norm(hv)`.
  - Norms and centroids use **fixed high bindings 12 (K norms), 13 (V norms), 14 (centroids)**. Assert that no body uses 12–14.
  - Push or specialization constants: `mseBits`, `codeUintsPerVec`, `normFormat`.
- **Wrappers** `*_kvtq.comp` for every body the dense chain and Flint dispatch: per-token `attention_f32`, `qpos`, the qpos tiled,
  `splitkv_gqa`, `splitkv_gqa_v2`, flash GQA prefill, coopmat, `sinks` split and flash, and the reference sink kernel's twin
  (`GptOssKernels`), with `_hd128`/`_hd256` where the F32 body has them.
- **Writer:** reuse `turboquant_encode_f32.comp` writing codes and norms (norm format via a specialization constant). The scatter
  writer comes in step 4.
- **Cache:** `VulkanKvCache` with `Storage == Tq` holds per-layer code buffers plus norm buffers, the shared centroids and signs, and
  `DeviceBytes` counting all of them.
- **Kernels:** `BlockKvTwins` for `Tq` (`_kvTqSet`); `InitBlockKvKernels` / `InitDenseBlockKvKernels` create the TQ twins.
- **Dispatch:** `RecordAttentionDispatch` records `tq_rotate_q` before the arm and `tq_rotate_o` after it. For split-KV, the output
  rotation goes **after the combine** pass.
- **Test:** `tests/DotLLM.Tests.Unit/Vulkan/VulkanTurboQuantInPlaceKernelTests.cs`, mirroring `VulkanDenseQ8KvKernelTests` with one test
  per arm against the scratch oracle (same norm format).

- [ ] **Step 1: Write the failing tests:**
  - per arm: `PerToken_`, `QPos_`, `QPosTiled_`, `SplitKvDecode_`, `SplitKvV2Decode_`, `ScalarGqaFlash_`, `CoopmatGqaFlash_`,
    `SinkSplitKv_`, `SinkFlash_`, `ReferenceSink_` + `ReadsTqInPlace_MatchesOracle(bits ∈ {2,3,4,8}, hd ∈ {64,128,256})`;
  - `SplitKv_LongContext_RotatesAfterCombine` (ctx 4K);
  - `CurrentChunkRows_ComeFromTheCache`;
  - `Writer_MatchesCpuEncode_ByteForByte`.
- [ ] **Step 2:** Run them under the lock: `--filter "FullyQualifiedName~VulkanTurboQuantInPlace"`. Expected: FAIL.
- [ ] **Step 3:** Implement. Rebuild the project so `bin/spv` picks up the new `.spv`.
- [ ] **Step 4:** Re-run. Expected: PASS. Measure the hd64 LDS budget: no TQ twin may add LDS to the hd64 body (341 B headroom).
- [ ] **Step 5: Mutants:**
  - the Task 4 five;
  - output rotation per split instead of after the combine (`SplitKv_LongContext` must die);
  - a binding-14 collision.
- [ ] **Step 6:** Commit `feat(vulkan): TurboQuant in-place KV for the dense chain and Flint (#1892)`.

### Task 8: Capability query, resolver and wiring

**Files:**
- Modify: `src/DotLLM.Core/Models/IModel.cs`: add `bool SupportsInPlaceTurboQuant(KvCacheConfig kv) => false;`. It is true only when
  **every attention arm the host can dispatch** has a TQ twin loaded, the head dim is a power of two ≤ 256, there's no `tq*q` on GPU,
  and the pair is both-side tq.
  - Implement it in `TransformerModel`, `CudaTransformerModel` and `VulkanTransformerModel`, the latter checking the full `_kvTqSet` the
    way `TryUseInPlaceQuantizedKv` checks its set.
- Modify: `TurboQuantKvSupport`:
  - The backend rule and the run/chat/bench rule are removed.
  - The host rule becomes "the loaded model's `SupportsInPlaceTurboQuant(kv)` is false". It still cites #1900, or #1894 for `tq*q` on
    GPU.
  - New rule: `--cache-window > 0` with tq.
- Modify: `ServerStartup` (the tq-first branch builds the backend's in-place cache: CPU `TurboQuantInPlaceKvCache`, CUDA via
  `CreateKvCache`, Vulkan via the model factory with `Storage = Tq`); `RunCommand`, `ChatCommand` and `BenchCommand` (parse bits and
  QJL per side; build the same caches); and `KvStorageCli` help.
- Test:
  - flip the tests listed in #1892's checklist;
  - `Auto_OnAHostWithoutTheFullTwinSet_Refuses` (a Vulkan fake model whose `SupportsInPlaceTurboQuant` is false);
  - `CacheWindow_WithTq_Refuses`;
  - `Serve_Run_Chat_Bench_BuildTheInPlaceCache` (CPU, TinyQwen);
  - `ServerTurboQuantAutoDeviceTests` inverted (GPU, under the lock).

- [ ] **Step 1:** Write the new and flipped tests.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Re-run. Expected: PASS. Regenerate the CLI docs (`DOTLLM_REGEN_CLI_DOCS=1`, `CliHelpTests`).
- [ ] **Step 5: Mutants:**
  - resolver ignores `SupportsInPlaceTurboQuant`;
  - Vulkan's capability skips one arm (e.g. Coopmat).
  
  Both must die.
- [ ] **Step 6:** Commit `feat: TurboQuant KV wired on CPU, CUDA, Vulkan; capability-gated (#1892)`.

### Task 9: #1892 checklist and docs

- [ ] Work through every box in #1892's "remove the regressive guidance" list:
  - hints and help text;
  - the availability tables in `docs/CLI.md` and `docs/KV_CACHE.md`;
  - the flipped tests.
- [ ] Update `docs/PRECISION.md` (norm format and evidence) and the `docs/KV_CACHE.md` TurboQuant section (layout, rotations, twins
  per backend).
- [ ] Commit `docs: TurboQuant in place; availability updated (#1892)`.

### Task 10: Gates and benches

- [ ] **Model-level oracle parity**, per backend: in-place vs scratch oracle on Qwen2.5-0.5B q8_0 and Qwen3-4B, bits {2,3,4,8}:
  - teacher-forced logits within tolerance;
  - greedy identity over 256 tokens.
  
  Strix runs under the lock; CUDA runs on framework.
- [ ] **Speed gate:** in-place vs scratch at 7.6K and 32K context, decode and prefill, 3 or more reps with ranges, same-session A/B,
  warm-up first. Any arm that is slower blocks the merge; fix it, or bring the numbers to James.
- [ ] **Report** KL, PPL and greedy identity against f16 per bit width (the harness from Task 1 extended to the in-place caches), and the
  bytes per token against f16, q8_0 and q4_0.
- [ ] Run the full CPU-only unit suite in the background (about 1 h). Run the Vulkan GPU suite filter for TurboQuant and #1884 under
  the lock. Run the CUDA TurboQuant suite on framework.
- [ ] Run a whole-branch fresh review, fix Critical and Important findings with RED→GREEN tests, open the PR (mutant list, tables,
  rulings), and merge under full-auto after CI. Close #1892 and tick its checklist. Clean up.
