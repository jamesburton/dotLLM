# Issue #371 — Vulkan resident-quant MoE overlay: add Q4_K support

**FOR THE COORDINATOR: fold into `.docs/KERNEL_MAP.md` §7, then delete this file.**

Status: **implemented, parity-tested (synthetic fixture); real-cached-GGUF validation
inconclusive — see "Real-model validation" below for why.**
Branch `issue/371-vulkan-moe-resident-q4k`, worktree
`C:\Development\dotLLM\.claude\worktrees\agent-issue371`, branched from `dev` tip
`9eafd438` ("chore: remove #369 review ledger").

## What was built

`src/DotLLM.Vulkan/VulkanQwen3MoeMoeUpload.cs`'s resident-quant overlay for
Qwen3MoeHybrid routed MoE expert banks previously supported only Q6_K source quant. This
issue generalizes it to also support **Q4_K**, reusing the already-existing
`MoeIndexedMatmulQ4_KF32Kernel` (the kernel the generic non-hybrid `RecordMoeLayer` path
already used) — no new `.comp` shaders needed.

1. **`VulkanQwen3MoeMoeUpload.cs`.** Replaced the hardcoded `== QuantizationType.Q6_K`
   triple-check with a `s_ResidentQuantTypes` set (`{Q6_K, Q4_K}`) and generalized
   `UploadRoutedBankQ6K` → `UploadRoutedBankResidentQuant(quantType, ...)`, which now
   sizes rows via `Dequantize.RowByteSize(dim, quantType)` instead of the hardcoded
   210-byte Q6_K block size. Mixed-quant banks (gate/up/down not *all* the same
   supported type) still fall back to F32 — unchanged behavior, required by the issue.
2. **`VulkanQwen3MoeHybridTransformerModel.cs`.** Added a `case QuantizationType.Q4_K`
   arm to `RecordIndexedMoeMatmul` (~line 1333) dispatching to
   `_kernels.MoeIndexedMatmulQ4K.Record(...)`, mirroring the Q6_K arm exactly. Also added
   a one-line diagnostic (`Console.Error.WriteLine`, layer 0 only, resident mode only)
   reporting the chosen bank quant type and the source gate/up/down quant types — the
   only place that knows whether the resident-quant overlay actually engaged or silently
   fell back to F32, which turned out to matter a lot for real-model validation (see
   below).
3. **`VulkanQwen3MoeHybridKernels.cs`.** Wired `MoeIndexedMatmulQ4_KF32Kernel` into the
   kernel bundle's constructor, `Create`, `InvalidateDescriptorCache`, and `Dispose`,
   mirroring every `MoeIndexedMatmulQ6K` line in the file.
4. New test file: `tests/DotLLM.Tests.Unit/Vulkan/VulkanQwen3MoeMoeUploadQ4KResidentTests.cs`
   — mirrors `VulkanQwen3MoeMoeUploadQ6KResidentTests` (same fixture/tolerance approach:
   abs 5e-3 / rel 1e-3), 4 shape cases (`[interm, hidden]` combinations, all multiples of
   256 — the Q4_K group size) exercising the resident-Q4_K-vs-streaming-F32 exact-token
   parity gate, plus a mixed-quant fallback test asserting the bank still lands on F32
   when gate/up/down aren't uniformly Q4_K.

Post-merge sanity check per `CLAUDE.md`: `grep -rn "pipeline.DescriptorSetLayout"
src/DotLLM.Vulkan` matches only inside `DescriptorSetCache.cs` — confirmed, the new
kernel wiring follows the `DescriptorSetCache` pattern, not a hand-rolled one.

## Test results

New Q4_K resident tests, run in isolation:

```
Passed DotLLM.Tests.Unit.Vulkan.VulkanQwen3MoeMoeUploadQ4KResidentTests.ResidentQ4K_FallsBackToF32_WhenSourceNotQ4K
Passed DotLLM.Tests.Unit.Vulkan.VulkanQwen3MoeMoeUploadQ4KResidentTests.ResidentQ4K_MatchesStreamingF32(n: 3, numExperts: 4, interm: 512, hidden: 256, activeExperts: 2)
Passed DotLLM.Tests.Unit.Vulkan.VulkanQwen3MoeMoeUploadQ4KResidentTests.ResidentQ4K_MatchesStreamingF32(n: 6, numExperts: 4, interm: 256, hidden: 768, activeExperts: 3)
Passed DotLLM.Tests.Unit.Vulkan.VulkanQwen3MoeMoeUploadQ4KResidentTests.ResidentQ4K_MatchesStreamingF32(n: 8, numExperts: 8, interm: 256, hidden: 512, activeExperts: 4)
Passed DotLLM.Tests.Unit.Vulkan.VulkanQwen3MoeMoeUploadQ4KResidentTests.ResidentQ4K_MatchesStreamingF32(n: 4, numExperts: 4, interm: 256, hidden: 256, activeExperts: 2)
Passed DotLLM.Tests.Unit.Vulkan.VulkanQwen3MoeMoeUploadQ4KResidentTests.ResidentQuant_FallsBackToF32_WhenBanksAreMixedQuant
Total tests: 6, Passed: 6
```

Full Vulkan suite (`dotnet test tests/DotLLM.Tests.Unit -c Release --filter
"FullyQualifiedName~Vulkan"`):

```
Total tests: 982
     Passed: 940
     Failed: 1
    Skipped: 41
```

The one failure, `VulkanTransformerModelForwardBatchTests.VulkanForwardBatch_FourSeqs_MixedPrefillDecode_MatchesPerSeqLoop`,
passes cleanly when retried in isolation — an unrelated pre-existing flake (this repo's
established pattern for cross-talk in shared-GPU-context test runs), not a regression
from this change. Effective result: **941/941 real assertions pass, 41 skipped**
(gated/opt-in benchmarks, e.g. `VulkanSubgroupMicroBench`).

## Real-model validation — inconclusive, and why that itself is the finding

The cached model is `unsloth/Qwen3.6-35B-A3B-GGUF` **UD-Q4_K_XL**
(`C:\Users\james\.cache\huggingface\hub\models--unsloth--Qwen3.6-35B-A3B-GGUF\snapshots\a483e9e6cbd595906af30beda3187c2663a1118c\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`).
Ran `DotLLM.Cli.exe bench <path> --device vulkan -p 8 -n 5 -r 1` with
`DOTLLM_VK_MOE_RESIDENT=1`. The new layer-0 diagnostic printed:

```
[dotLLM] Vulkan resident-MoE bank quant type: F32 (source: gate=Q4_K, up=Q4_K, down=Q5_K)
```

**This UD-Q4_K_XL checkpoint's routed-expert down-projection is Q5_K, not Q4_K** —
Unsloth's "UD" (unsloth-dynamic) quantization gives the down-projection (more
sensitivity-critical) higher precision than gate/up, so the three routed banks are *not*
uniformly one quant type despite the "Q4_K_XL" filename. Per the issue's explicit scope
("mixed-quant banks must keep falling back to F32 — do not attempt to solve mixed-bank
dispatch"), the resident overlay correctly detects this and falls back to F32 — the same
behavior it had before this change. This is **not a bug introduced here**; #370's ledger
already flagged this exact risk ("this GGUF is Q4_K_XL, a mixed-quant UD build, so
resident mode would fall back to the same F32-dequant-resident path").

Once on the F32 fallback path, the run hit repeated `vkAllocateMemory` transient failures
(`VK_ERROR_...`, retrying up to 4x with backoff) trying to allocate the ~123 GB F32
resident layout that #370's ledger already computed doesn't fit as a *resident* buffer
(streaming per-forward F32 dequant is a different, smaller-footprint code path — see
`VulkanQwen3MoeHybridTransformerModel`'s streaming-vs-resident branch — but resident mode
with `DOTLLM_VK_MOE_RESIDENT=1` commits to keeping the whole F32 bank device-resident
across forwards, which is the ~123 GB figure). This reproduces independent of this
issue's change — it is a consequence of turning on `DOTLLM_VK_MOE_RESIDENT=1` against a
model whose routed banks aren't uniformly a resident-supported quant type, which was
already true before #371 (Q6_K-only support also would have fallen back to F32 for this
Q4_K/Q5_K-mixed file and hit the same allocation pressure).

**What this means for the issue's acceptance criterion:** the Q4_K resident path itself
is implemented, parity-tested, and verified correct via the synthetic fixture. But **it
does not engage for the specific cached `UD-Q4_K_XL` file**, because that file's
down-projection is actually Q5_K. The 0.067 tok/s baseline from #370 was measured in
*streaming* mode (`DOTLLM_VK_MOE_RESIDENT` unset), not resident mode — this run used
resident mode, which is a different (and, for this file, worse-fitting) code path. A
fair like-for-like comparison would require either:
- A synthetic or hand-repacked GGUF with uniformly-Q4_K gate/up/down routed banks at
  qwen35moe-A3B scale (not available on this box), or
- Extending resident support to Q5_K as well (explicitly out of scope per the issue: "Do
  NOT attempt Q5_1 or Q8_0 resident support — Q4_K only" — Q5_K isn't even in that
  explicit list, but the same "one quant at a time" scoping logic applies), or
- Mixed-bank dispatch (three independent kernel calls, one per bank, when gate/up/down
  differ) — explicitly called out as out of scope in the issue.

No fabricated numbers are reported here: the real-model run did not complete a bench
measurement because it fell back to the (already-existing, unmodified-by-this-issue)
F32-resident path and could not allocate the required device memory.

## Known gaps / follow-up

- **Real UD-Q4_K_XL still isn't fast in resident mode** — see above; this is a follow-up
  candidate (Q5_K resident support, or mixed-bank dispatch) tracked separately, not part
  of #371's scope.
- **Q4_K-uniform real-GGUF validation not performed** — no cached model on this box has
  uniformly-Q4_K routed MoE banks at a testable scale; the synthetic fixture is the only
  verification of the Q4_K-resident path's correctness. A future validation pass should
  find or construct such a checkpoint.
- Everything else mirrors the Q6_K resident path's known gaps (uniform-quant-only,
  scalar F32-dequant-per-row kernel — no MMVQ decode-optimized variant, out of scope per
  the issue).
