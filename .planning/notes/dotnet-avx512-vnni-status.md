# .NET 512-bit VPDPBUSD (`Avx512Vnni`) status

**Date**: 2026-05-15
**Subject**: Is 512-bit VNNI (`VPDPBUSD zmm, zmm, zmm`) callable from C# yet,
and if not, what does shipping it look like?
**Trigger**: Closing the residual −16% LoRA Q8_0 prefill regression on Strix Halo
(Zen 5 + AVX-512-VNNI). See `.continue-here-lora-final-mile.md` and
`docs/LORA.md` "Spike notes" (line 220).

---

## TL;DR

**Status**: `STATUS_API_APPROVED_NOT_IMPLEMENTED` — with a wrinkle.

| Layer | State | Evidence |
|---|---|---|
| API proposal | **api-approved** July 2023 | [#86849][86849] label `api-approved`, video review [link](https://www.youtube.com/watch?v=XKrNmVjiEpY&t=0h27m20s) |
| API spec | **stale; needs re-review** under AVX10 pattern | [@saucecontrol comment 2024-11-06][newshape], [@saucecontrol nudge 2026-02-24][nudge] (no response from area owner yet) |
| JIT/VM plumbing (CPUID, ISA enum, Ready2Run) | **landed in main** June 2025 | PR [#116230][116230] "Add JIT/VM support for the approved but NYI xarch isas" — explicitly states "does not actually add the intrinsic IDs or managed support required for their use" |
| Managed C# class (`Avx512Vnni`, methods, intrinsic IDs) | **not present in main** | `gh search code` returns zero `*.cs` matches under `System.Runtime.Intrinsics.X86`. `Avx512Vnni.cs` does not exist in the directory listing. `AvxVnni.cs` has no `V512` nested class. |
| .NET 10.0.103 (current dotLLM target) | **not present** | Confirmed by previous agent investigation. |
| .NET 11 Preview 4 (latest as of 2026-05-12) | **not present** | Source of truth = main; nothing has merged. |
| .NET 11 RTM (target Nov 2026) | **unlikely without action** | Milestone on tracking issue is `Future`, not `11.0.0`. |

The negative twist: the JIT-side enum (`Avx512Vnni` instruction set ID) and VM
plumbing already exist in main as of PR #116230. The work that's *actually*
missing is the public managed surface plus intrinsic-ID wiring — which is
small, well-bounded, and modeled directly on the already-shipped
`AvxVnniInt8` / `AvxVnniInt16` classes.

---

## Phase 1 — Definitive status check

### 1.1  Tracking issue [dotnet/runtime#86849][86849]

- **State**: OPEN
- **Milestone**: `Future` (explicitly: "may happen, or may not; it is not committed")
- **Labels**: `api-approved`, `area-System.Runtime.Intrinsics`, `avx512`
- **Linked PRs**: none (zero branches, zero PRs)
- **Last activity**: 2026-02-24, [@saucecontrol asking @tannergooding][nudge]
  whether the spec needs to go back through review under the new AVX10
  pattern. **No response yet.**
- **Assignees**: none

The 2023 API review approved the original "`Avx512Vnni : Avx512F` with `.VL`
nested class" shape. In November 2024 [@saucecontrol proposed a refactor][newshape]
to the modern AVX10 pattern (a `V512` nested class on the existing `AvxVnni`
class), which is what `AvxVnniInt8` / `AvxVnniInt16` actually shipped with.
**That refactor was never re-reviewed**, which is presumably why nobody has
opened the implementation PR — the spec is in limbo.

### 1.2  Source code in `dotnet/runtime` `main`

Confirmed by direct fetch on 2026-05-15:

| File | Contains 512-bit VNNI? |
|---|---|
| `src/coreclr/tools/Common/JitInterface/ThunkGenerator/InstructionSetDesc.txt` | YES — registers `Avx512Vnni` as ISA #79 (group AVX512v3); also `Avx10v1_V512`, `Avx10v2_V512`, `AVXVNNIINT_V512`. Plumbing landed via [#116230][116230]. |
| `src/libraries/System.Private.CoreLib/src/System/Runtime/Intrinsics/X86/` | NO `Avx512Vnni.cs` file exists. Directory jumps from `Avx512F.cs` straight to `Avx512Vbmi.cs`. |
| `…/AvxVnni.cs` | 256-bit and 128-bit overloads only. No `V512` nested class. Only nested class is `X64`. |
| `…/AvxVnniInt8.cs` | **Has `V512` nested class with all 6 overloads** (3× `MultiplyWideningAndAdd`, 3× `MultiplyWideningAndAddSaturate`). Reference template. |
| `…/AvxVnniInt16.cs` | Same pattern as `AvxVnniInt8.cs`. |
| `…/Avx10v1.cs`, `…/Avx10v2.cs` | Have `V512` nested classes, but VNNI methods are intentionally NOT here — they live on the dedicated VNNI ISA classes per design. |

So: the .NET-runtime team **established the modern shape** via `AvxVnniInt8.V512`
and `AvxVnniInt16.V512` (PR #113956 in 2025), but **never went back** to apply
the same shape to plain `AvxVnni` for the AVX-512 forms.

### 1.3  AVX10 unification — is it hidden under `Avx10v1.V512`?

**No.** Verified by direct fetch:
- `Avx10v1.cs` and `Avx10v2.cs` do not contain `MultiplyWideningAndAdd`,
  `MultiplyWideningAndAddSaturate`, or any reference to `VPDPBUSD`/`VPDPWSSD`.
- The pattern @tannergooding clarified in [issue #86849 comment 2024-11-07][tannermsg]:
  "rather than `class Avx512Isa { class VL { } }` we can have
  `class AvxIsa { class V512 { } }`".
- VNNI under that schema lives on `AvxVnni.V512`, not `Avx10v1.V512`. The
  `AvxVnni.V512` class **does not exist yet in main**.

### 1.4  Nightly / preview SDK availability

Since the C# surface is not in main, no preview, nightly, or daily build
exposes it. Installing `--channel 11.0 --quality daily` would not unlock it.

---

## Phase 2 verdict — should we PR it?

**Yes, this is a tractable contribution** — but a clarifying comment on
[#86849][86849] should land first to unstick the spec. Concretely:

1. **Step 0 (blocking, ~1 day)**: Comment on [#86849][86849] proposing to
   adopt @saucecontrol's 2024-11-06 shape ([`AvxVnni.V512`][newshape]) plus a
   `MultiplyWideningAndAdd` overload set for the `short`/`int` (`VPDPWSSD`)
   forms that `AvxVnniInt8` does not cover (since `Int8` is byte/sbyte only).
   Tag @tannergooding and @saucecontrol; ask for either a quick re-review or a
   green light to PR under the existing approval. Mention the dotLLM
   motivating workload (Q8_0 LoRA prefill on Zen 5).

2. **Step 1 (PR, ~2-4 weeks for first-time contributor)**: Implement the
   class + intrinsic IDs + tests, modeled byte-for-byte on `AvxVnniInt8.V512`.
   Detailed scope in the companion document
   [`dotnet-runtime-avx512-vnni-pr-scope.md`](dotnet-runtime-avx512-vnni-pr-scope.md).

The "scope much smaller than expected" insight: PR #116230 already did the
hard ISA-registration work. We are filling in the trailing 5 files that PR
deliberately deferred.

---

## Phase 3 — Build path back to dotLLM

### 3.1  If the PR lands in .NET 11 RTM (Nov 2026)

Optimistic timeline. Requires:
- spec re-confirmation in May 2026,
- PR opened June 2026,
- merged before .NET 11 code-freeze (~Sep 2026).

**dotLLM consumption recipe**:

```xml
<!-- src/Directory.Build.props or per-project -->
<PropertyGroup>
  <TargetFramework>net11.0</TargetFramework>
</PropertyGroup>
```

Feature-detect at startup (place near existing `Avx512F.IsSupported` checks):

```csharp
// src/DotLLM.Cpu/Kernels/Q8_0Kernels.cs (illustrative)
internal static class CpuFeatures
{
    public static readonly bool HasAvx512Vnni =
        // .NET 11+: use the new V512 nested class
        AvxVnni.V512.IsSupported;
}
```

The runtime IsSupported check costs nothing — JIT folds the constant and DCEs
the dead branch.

### 3.2  If the PR lands only in .NET 12 (May 2027)

The likely outcome if Step 0 stalls. dotLLM's options:

(a) **Wait** — accept the −16% LoRA Q8_0 prefill gap on Strix Halo until
    .NET 12 ships. Justified given that decode (the user-perceived hot path)
    is unaffected; the gap is a prefill regression in a relatively niche
    LoRA-on-quantized configuration.
(b) **Multi-target** — `<TargetFrameworks>net10.0;net11.0</TargetFrameworks>`
    on the kernel assembly and use `#if NET11_0_OR_GREATER` guards.
(c) **Preview consumption** — pin to a `.NET 11 preview` SDK in CI for the
    Strix Halo benchmark suite, ship the GA build with the .NET 10
    fallback. Not recommended for a library.

### 3.3  Workarounds available *today* (.NET 10.0.103)

The previous-agent investigation already enumerated and rejected the obvious
options. For completeness:

| Workaround | Throughput vs ideal `VPDPBUSD-zmm` | Verdict |
|---|---|---|
| 2× `AvxVnni.MultiplyWideningAndAdd` (256-bit) per 512-bit chunk | ~50% of zmm path on Zen 5 (one extra issue slot, but Zen 5 cracks 512-bit ops into two µops anyway) | **What dotLLM does today**; this is the negative-result baseline. |
| `Avx512BW.MultiplyAddAdjacent` × `Vector512<short>.One` chain (the AVX-512 fallback shown in MadProbe's Adler32 example, [issue body][86849]) | ~40-60% of `VPDPBUSD-zmm` depending on µarch — two dependent EVEX ops vs one | Slower than the 2×ymm path on Zen 5 per Microsoft's own example fallback. |
| `[DllImport]` to a tiny C/asm shim that issues a single `vpdpbusd zmm` | Theoretical full speed, but adds a P/Invoke transition (~20-50 ns even with `[SuppressGCTransition]`) per call | **Anti-pattern for an inner loop.** P/Invoke overhead would dominate at the per-block granularity needed (block_q8_0 = 32 elements). Only viable if amortised across an entire row, in which case a full native GEMM kernel is the right answer instead. |
| LLVM-IR via `Mono.Cecil` / runtime emit | 0% — JIT does not lower arbitrary IL to `VPDPBUSD-zmm`; only the `[Intrinsic]`-marked methods are recognised. | Not viable. |

**Bottom line**: there is no in-process workaround that beats the 2×ymm path
on .NET 10. The PR is the only way forward.

---

## Sources

- [dotnet/runtime#86849 — `[API Proposal]: Add support for AVX-512 VNNI hardware instructions`][86849]
- [PR #116230 — `Add JIT/VM support for the approved but NYI xarch isas`][116230] (merged 2025-06-07)
- [PR #113956 — `Add CPUID for AvxVnniInt8 and AvxVnniInt16`][cpuidpr] (merged 2025-07-07)
- [`AvxVnniInt8.cs` in `main`](https://github.com/dotnet/runtime/blob/main/src/libraries/System.Private.CoreLib/src/System/Runtime/Intrinsics/X86/AvxVnniInt8.cs) — reference template for `V512` nested class shape
- [`InstructionSetDesc.txt` in `main`](https://github.com/dotnet/runtime/blob/main/src/coreclr/tools/Common/JitInterface/ThunkGenerator/InstructionSetDesc.txt) — confirms `Avx512Vnni` is registered as ISA #79
- [.NET 11 Preview 1 announcement (2026-02)](https://devblogs.microsoft.com/dotnet/dotnet-11-preview-1/)
- [dotnet/install-scripts](https://github.com/dotnet/install-scripts) — for nightly SDK install recipes
- [area-owners.md — area-System.Runtime.Intrinsics owners](https://github.com/dotnet/runtime/blob/main/docs/area-owners.md): lead @jeffhandley, owners @echesakovMSFT, @kunalspathak

[86849]: https://github.com/dotnet/runtime/issues/86849
[116230]: https://github.com/dotnet/runtime/pull/116230
[cpuidpr]: https://github.com/dotnet/runtime/pull/113956
[newshape]: https://github.com/dotnet/runtime/issues/86849#issuecomment-2458603484
[nudge]: https://github.com/dotnet/runtime/issues/86849#issuecomment-3948357993
[tannermsg]: https://github.com/dotnet/runtime/issues/86849#issuecomment-2461973932
