# Upstream PR scope — `AvxVnni.V512` (`VPDPBUSD-zmm`) for `dotnet/runtime`

**Date**: 2026-05-15
**Companion to**: `dotnet-avx512-vnni-status.md`
**Tracking issue**: [dotnet/runtime#86849][86849]
**Verdict from Phase 1**: api-approved (2023) but spec needs a re-confirmation
nudge under the AVX10-era `AvxVnni.V512` shape proposed by @saucecontrol on
2024-11-06 ([comment][newshape]).

This document scopes the PR so the user can decide go/no-go before opening it.
Scope is intentionally minimal — model byte-for-byte on the already-shipped
`AvxVnniInt8.V512` to maximise reviewer confidence.

---

## 0  Pre-PR — unstick the spec

Before any code, leave a comment on [#86849][86849]:

> **dotLLM** (an open-source .NET LLM inference engine) needs `VPDPBUSD-zmm`
> to close a measured −16% prefill regression on Strix Halo (Zen 5,
> AVX-512-VNNI). Confirming we'd like to PR this. Per
> [@saucecontrol's 2024-11-06 comment][newshape], we propose the
> `AvxVnni.V512` shape (matching `AvxVnniInt8.V512` / `AvxVnniInt16.V512`
> already in main) plus the `MultiplyWideningAndAdd` overloads for
> `Vector512<short>` (`VPDPWSSD`) that the byte-only `AvxVnniInt*` classes
> don't cover. JIT/VM ISA plumbing already exists from #116230. Happy to do
> the implementation if a re-review isn't required, or to wait for one if it
> is. cc @tannergooding @jeffhandley @kunalspathak

Wait for an "lgtm proceed" or a re-review video link before sinking PR effort.

---

## 1  Files to touch

All paths relative to `dotnet/runtime` repo root. File counts and shapes match
what merged in PR [#113956][113956] (CPUID for `AvxVnniInt8` / `Int16`) and
[#116230][116230] (JIT/VM plumbing). Search the diffs of those PRs as templates.

### 1.1  Public managed surface (2 files, NEW)

| File | Shape |
|---|---|
| `src/libraries/System.Private.CoreLib/src/System/Runtime/Intrinsics/X86/AvxVnni.cs` | **EDIT** — add nested `V512` class with 4 overloads (`MultiplyWideningAndAdd` × 2 for byte/sbyte and short/short, `MultiplyWideningAndAddSaturate` × 2 for the `S` variants). |
| `src/libraries/System.Private.CoreLib/src/System/Runtime/Intrinsics/X86/AvxVnni.PlatformNotSupported.cs` | **EDIT** — mirror the new `V512` nested class with `[DoesNotReturn] throw new PlatformNotSupportedException()` bodies. |

The `V512` class declaration template (from `AvxVnniInt8.cs` in main):

```csharp
namespace System.Runtime.Intrinsics.X86
{
    public abstract class AvxVnni : Avx2
    {
        // ... existing methods unchanged ...

        [Intrinsic]
        public abstract class V512
        {
            internal V512() { }

            public static bool IsSupported { get => IsSupported; }

            /// <summary>
            ///   __m512i _mm512_dpbusd_epi32 (__m512i src, __m512i a, __m512i b)
            ///   VPDPBUSD zmm1, zmm2, zmm3/m512
            /// </summary>
            public static Vector512<int> MultiplyWideningAndAdd(
                Vector512<int> addend, Vector512<byte> left, Vector512<sbyte> right)
                    => MultiplyWideningAndAdd(addend, left, right);

            /// <summary>
            ///   __m512i _mm512_dpwssd_epi32 (__m512i src, __m512i a, __m512i b)
            ///   VPDPWSSD zmm1, zmm2, zmm3/m512
            /// </summary>
            public static Vector512<int> MultiplyWideningAndAdd(
                Vector512<int> addend, Vector512<short> left, Vector512<short> right)
                    => MultiplyWideningAndAdd(addend, left, right);

            /// <summary>
            ///   __m512i _mm512_dpbusds_epi32 (__m512i src, __m512i a, __m512i b)
            ///   VPDPBUSDS zmm1, zmm2, zmm3/m512
            /// </summary>
            public static Vector512<int> MultiplyWideningAndAddSaturate(
                Vector512<int> addend, Vector512<byte> left, Vector512<sbyte> right)
                    => MultiplyWideningAndAddSaturate(addend, left, right);

            /// <summary>
            ///   __m512i _mm512_dpwssds_epi32 (__m512i src, __m512i a, __m512i b)
            ///   VPDPWSSDS zmm1, zmm2, zmm3/m512
            /// </summary>
            public static Vector512<int> MultiplyWideningAndAddSaturate(
                Vector512<int> addend, Vector512<short> left, Vector512<short> right)
                    => MultiplyWideningAndAddSaturate(addend, left, right);
        }
    }
}
```

### 1.2  JIT intrinsic ID + opcode wiring (5 files, EDIT)

The ISA enum entry for `Avx512Vnni` already exists from #116230 — we only
register *intrinsic IDs* against it.

| File | Change shape |
|---|---|
| `src/coreclr/jit/hwintrinsiclistxarch.h` | Add 4 `HARDWARE_INTRINSIC` macro entries for `NI_AVX512VNNI_V512_MultiplyWideningAndAdd` (×2 overloads) and `…_MultiplyWideningAndAddSaturate` (×2 overloads). Each entry specifies operand types, instruction (`INS_vpdpbusd`, `INS_vpdpwssd`, `INS_vpdpbusds`, `INS_vpdpwssds`), simdSize=64, isa=`InstructionSet_AVX512VNNI`. Look at the existing `NI_AVXVNNI_MultiplyWideningAndAdd` entries (256-bit) as the template — same flags, only `simdSize` and the `Vector512` operand type differ. |
| `src/coreclr/jit/hwintrinsicxarch.cpp` | Add the `case NI_AVX512VNNI_V512_*:` branches in `lookupNamedIntrinsic` / `impSpecialIntrinsic` — typically just delegates to the same handler as the 256-bit forms. Search for `NI_AVXVNNI_MultiplyWideningAndAdd` to find the existing dispatch site to extend. |
| `src/coreclr/jit/hwintrinsiccodegenxarch.cpp` | Likely **no change** — the codegen path is shared with the 256-bit form via `genHWIntrinsic_R_R_R_RM` selecting on `simdSize`. Verify by grep'ing for `INS_vpdpbusd`. |
| `src/coreclr/jit/instrsxarch.h` | Verify `INS_vpdpbusd` etc. already have EVEX 512-bit encoding entries (they should, since the 256-bit forms use EVEX-256). If only VEX-256 forms exist, add the EVEX-512 row. |
| `src/coreclr/jit/lowerxarch.cpp` | Likely **no change** — the existing `LowerHWIntrinsic` cases for `NI_AVXVNNI_MultiplyWideningAndAdd` handle the constant-folding/containment paths simdSize-agnostically. Verify. |

### 1.3  Mono (1 file, EDIT — likely)

| File | Change shape |
|---|---|
| `src/mono/mono/mini/simd-intrinsics.c` | Add `Avx512Vnni.V512` to the `supported_x86_intrinsics[]` table and wire the LLVM intrinsic IDs (`llvm.x86.avx512.vpdpbusd.512` etc.). Follow the pattern of the `AvxVnniInt8` entry that landed in the AvxVnniInt8 implementation PR. |

(Mono support is required for PR acceptance even if dotLLM doesn't run on
Mono — area owners will block on it.)

### 1.4  Cross-platform metadata (3 files, EDIT)

These were touched by [#116230][116230] for the ISA enum, but the per-intrinsic
ID may need additions:

| File | Change |
|---|---|
| `src/coreclr/inc/corinfoinstructionset.h` | Verify `InstructionSet_AVX512VNNI` and `InstructionSet_AVX512VNNI_X64` enum values exist. If a new sub-ISA is needed for `V512`, add it. |
| `src/coreclr/tools/Common/Internal/Runtime/ReadyToRunInstructionSet.cs` | Same verification. |
| `src/coreclr/inc/readytoruninstructionset.h` | Same. |

If #116230 already added everything the JIT needs, these files are no-ops.
Confirm by searching for `Avx512Vnni` in each.

### 1.5  Tests (1 directory + 1 generator invocation, NEW)

| Path | Shape |
|---|---|
| `src/tests/JIT/HardwareIntrinsics/X86/AvxVnni.V512/` | **NEW directory**. Contains 4 `.cs` test classes (one per overload) generated by re-running `src/tests/JIT/HardwareIntrinsics/X86/Shared/GenerateHWIntrinsicTests_X86.csx` with new template entries. The template entries describe the operation and the script generates the boilerplate (correctness check vs scalar reference, register allocation stress, register/stack/memory operand variants, broadcast variants). |
| `src/tests/JIT/HardwareIntrinsics/X86/Shared/GenerateHWIntrinsicTests_X86.csx` | **EDIT** — add 4 entries to the `Avx512VnniInputs` (or equivalent) array. Each entry: `(opName, baseType, vectorSize, instructionName)`. |
| `src/tests/JIT/HardwareIntrinsics/X86/AvxVnni.V512/Program.AvxVnni.V512.cs` | **NEW** — the dispatch entry point listing all 4 generated test functions. Auto-generated by the script. |

**Estimated test count**: 4 overloads × 2 element types each × ~6 operand
permutations (reg/mem/broadcast/etc.) ≈ **~24-32 generated test methods**,
plus a handful of hand-written sanity tests.

### 1.6  Documentation (1 file, optional)

| File | Change |
|---|---|
| `docs/coding-guidelines/api-guidelines/hwintrinsics.md` | No change expected; new ISA follows established pattern. |

### 1.7  Reference assemblies (1 file, EDIT)

| File | Change |
|---|---|
| `src/libraries/System.Runtime.Intrinsics/ref/System.Runtime.Intrinsics.cs` | Add the public surface declarations for `AvxVnni.V512` (4 method signatures + `IsSupported`). Mirror `AvxVnniInt8.V512` exactly. |

---

## 2  Total file count summary

| Category | Files | Lines (rough) |
|---|---|---|
| Public managed surface | 2 EDIT | +120 / +60 (PNS) |
| JIT/VM wiring | 5 EDIT | +40 (mostly enum table rows) |
| Mono | 1 EDIT | +20 |
| R2R / cross-platform metadata | 3 EDIT (or 0 no-op) | +15 |
| Tests | 2 EDIT + 1 dir NEW (~6 generated files) | +800 generated |
| Reference assembly | 1 EDIT | +20 |
| **Total** | **~14 files** | **~250 hand-written, ~800 generated** |

This is a small-to-medium HW intrinsic PR, well within the size of the
already-merged [`AvxVnniInt8`/`AvxVnniInt16` body of work][prtemplate].

---

## 3  PR description draft

**Title** (≤72 chars, follows `dotnet/runtime` convention):
> `Add Avx512Vnni V512 (VPDPBUSD/VPDPWSSD-zmm) intrinsics`

Or, matching the AvxVnniInt8 PR convention more closely:
> `Implement AvxVnni.V512 (Avx512Vnni) hardware intrinsics`

**Body**:

```markdown
## Summary

Adds the public managed surface and JIT intrinsic IDs for the 512-bit forms
of AVX-512 VNNI (`VPDPBUSD`, `VPDPWSSD`, `VPDPBUSDS`, `VPDPWSSDS` on `zmm`
registers), exposed via a new `AvxVnni.V512` nested class following the
pattern established by `AvxVnniInt8.V512` and `AvxVnniInt16.V512`.

JIT/VM ISA plumbing already landed in #116230. This PR fills in the trailing
managed surface and intrinsic-ID wiring that #116230 explicitly deferred.

Closes #86849.

## Approved API surface

Per #86849 (api-approved 2023-07-20) and refactored to the AVX10-era pattern
in #86849 (comment 2024-11-06):

```csharp
namespace System.Runtime.Intrinsics.X86;

public abstract class AvxVnni : Avx2
{
    [Intrinsic]
    public abstract class V512
    {
        public static bool IsSupported { get; }

        // VPDPBUSD zmm
        public static Vector512<int> MultiplyWideningAndAdd(
            Vector512<int> addend, Vector512<byte> left, Vector512<sbyte> right);

        // VPDPWSSD zmm
        public static Vector512<int> MultiplyWideningAndAdd(
            Vector512<int> addend, Vector512<short> left, Vector512<short> right);

        // VPDPBUSDS zmm (saturating)
        public static Vector512<int> MultiplyWideningAndAddSaturate(
            Vector512<int> addend, Vector512<byte> left, Vector512<sbyte> right);

        // VPDPWSSDS zmm (saturating)
        public static Vector512<int> MultiplyWideningAndAddSaturate(
            Vector512<int> addend, Vector512<short> left, Vector512<short> right);
    }
}
```

`IsSupported` reports `true` when CPUID reports `AVX-512 VNNI` and `AVX-512F`
(or, equivalently for newer hardware, `AVX10v1` with V512 support).

## Motivation

The 256-bit form (`AvxVnni.MultiplyWideningAndAdd` on `Vector256`) currently
forces .NET callers to issue two `VPDPBUSD-ymm` instructions to consume a
512-bit chunk of activations. On Zen 5 / Granite Rapids / Sapphire Rapids
hardware where `VPDPBUSD-zmm` retires at the same throughput as one
`VPDPBUSD-ymm`, this leaves ~50% of available INT8 dot-product throughput on
the floor.

Concrete consumer: dotLLM (open-source .NET LLM inference engine,
https://github.com/kkokosa/dotLLM) measured a −16% prefill regression on
Strix Halo for LoRA Q8_0 inference that traces directly to this gap. See
the [dotLLM workload analysis](link-to-LORA.md-Phase-4d.6) for the
benchmark.

## Performance numbers

Throughput micro-benchmark on AMD Ryzen AI 9 HX 370 (Zen 5, Strix Halo):

| Variant | GFLOPS (INT8 ops/sec) | Notes |
|---|---|---|
| 2× `AvxVnni.MultiplyWideningAndAdd` ymm | XX | current .NET 10 path |
| 1× `AvxVnni.V512.MultiplyWideningAndAdd` zmm | YY | this PR |
| Speedup | ZZ× | |

(Will be filled in after PR is hooked up; benchmark code in
`src/tests/JIT/HardwareIntrinsics/X86/Shared/MicroBench/Avx512Vnni.cs`.)

## Validation

- 32 generated correctness tests under `src/tests/JIT/HardwareIntrinsics/X86/AvxVnni.V512/`,
  validated against scalar reference per element.
- Tested on:
  - Intel Sapphire Rapids (CI: `Linux x64 AVX-512`)
  - AMD Zen 5 / Strix Halo (developer machine)
  - Intel Alder Lake (P-core only, AVX-512 fused off — verifies
    `IsSupported == false` path)
- Mono path tested via `runtime-mono-x64-tests` CI leg.

## Out of scope

- AVX-VNNI-INT8 / AVX-VNNI-INT16 V512 forms (already shipped).
- Avx10v1 unification details — `AvxVnni.V512.IsSupported` returns `true` for
  any of `AVX512VNNI+AVX512VL`, `AVX10v1.V512`, or future converged ISAs that
  imply VNNI.

cc @tannergooding @jeffhandley @kunalspathak @echesakovMSFT
```

---

## 4  Reviewers / CODEOWNERS

From `docs/area-owners.md` (verified 2026-05-15):
- **Lead**: @jeffhandley (area-System.Runtime.Intrinsics)
- **Owners / consultants**: @echesakovMSFT, @kunalspathak
- **JIT consultant for HW intrinsics historically**: @tannergooding (member,
  most active commenter on #86849, owns the AVX10 design discussion)
- **JIT codegen owner**: @JulieLeeMSFT + @dotnet/jit-contrib

Tag all four on the PR. @tannergooding's sign-off is effectively required
given his ownership of the AVX10 unification design.

---

## 5  Estimated timeline

For a first-time HW-intrinsic contributor to `dotnet/runtime`:

| Phase | Time |
|---|---|
| Spec re-confirmation on #86849 | 1-3 weeks (depends on @tannergooding's response time) |
| Implementation + local tests pass | 3-5 days |
| First PR review round | 1 week |
| Address review feedback (typical: tighten test coverage, fix Mono path, adjust naming) | 3-7 days |
| CI flakes + final sign-off + merge | 1 week |
| **Total** | **5-9 weeks from first comment to merge** |

If we get lucky and the spec confirmation lands in the same week we ask, and
the AvxVnniInt8 PR conversion is clean, this could close in **3-4 weeks**.

---

## 6  References

- [#86849][86849] — tracking issue
- [#116230][116230] — JIT/VM plumbing PR (already merged, the foundation we build on)
- [#113956][cpuidpr] — `AvxVnniInt8`/`Int16` CPUID PR (closest template)
- [`AvxVnniInt8.cs` in main][int8src] — the file structure we mirror
- [`InstructionSetDesc.txt` in main][isasrc] — confirms `Avx512Vnni` is registered as ISA #79

[86849]: https://github.com/dotnet/runtime/issues/86849
[newshape]: https://github.com/dotnet/runtime/issues/86849#issuecomment-2458603484
[116230]: https://github.com/dotnet/runtime/pull/116230
[113956]: https://github.com/dotnet/runtime/pull/113956
[cpuidpr]: https://github.com/dotnet/runtime/pull/113956
[prtemplate]: https://github.com/dotnet/runtime/pull/113956
[int8src]: https://github.com/dotnet/runtime/blob/main/src/libraries/System.Private.CoreLib/src/System/Runtime/Intrinsics/X86/AvxVnniInt8.cs
[isasrc]: https://github.com/dotnet/runtime/blob/main/src/coreclr/tools/Common/JitInterface/ThunkGenerator/InstructionSetDesc.txt
