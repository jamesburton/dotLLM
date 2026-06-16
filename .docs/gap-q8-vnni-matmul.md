# Gap memo — Q8_0 outer-product GEMM blocked-on / coupled-to VNNI (paused)

**Status:** PAUSED pending user decision. Raised per the explicit "pause and highlight the gap" instruction when dotnet/runtime#128365 turned out to be directly relevant.
**Date:** 2026-06-09

## TL;DR

The deferred Q8_0 outer-product GEMM follow-up to **PR #315** is exactly the kernel that wants
**VNNI int8 dot-product** (`AvxVnni.MultiplyWideningAndAdd` → `VPDPBUSD`). The single reason the
Q8_0 tile was deferred — **AVX2 register pressure** — is the precise thing VNNI fixes. But the
just-merged 512-bit form (dotnet/runtime#128365) is **not in a consumable SDK** (it's .NET 11), and
it is the first of a planned **3-4 related CPU-operator PRs** to the runtime. So building the dotLLM
Q8_0 kernel against V512 *now* would couple it to unreleased BCL. **Recommend: don't hand-roll the
old 23-YMM Q8_0 tile; pick a VNNI strategy first (decision below).**

## The two facts that collided

1. **PR #315 (merged-pending upstream)** shipped only the **F32** half of ROADMAP Phase 3 Step 26.
   It explicitly deferred Q8_0 with this rationale (from the PR body):
   > PR #61 landed the Q8_0 / Q5_0 / K-quant outer-product kernels but **reverted GEMM dispatch** —
   > the 4×3 Q8_0 microkernel was ~14% slower than 3× `VecDot4RowsR4` on AVX2 because RyuJIT spilled
   > 7+ YMM registers (**23 needed: 12 acc + 6 token + 1 `ones` + 4 data**). That register pressure is
   > quantization-specific.

   The `ones` register is the AVX2 `_mm256_maddubs_epi16` + `_mm256_madd_epi16(ones)` two-instruction
   widening-accumulate idiom. The handoff "Next Steps #3" flagged exploring a different tile shape
   (4×2, or AVX-512) to relieve this.

2. **dotnet/runtime#128365 (MERGED 2026-06-09, by James Burton, approved Tanner Gooding)** adds:
   - `AvxVnni.V512.MultiplyWideningAndAdd(Vector512<int> acc, …, …)` → **VPDPBUSD** (u8×s8 → +i32)
   - `AvxVnni.V512.MultiplyWideningAndAddSaturate` → **VPDPBUSDS**
   - (also the s16×s16 → i32 **VPDPWSSD** form)

   VNNI fuses *multiply + widen + horizontal-add-into-i32-accumulator* into **one instruction**. That
   collapses the `maddubs`+`madd` pair into a single op and **deletes the `ones` register entirely**,
   plus removes the separate product temporaries. The Q8_0 tile's 23-YMM budget very plausibly drops
   to ≤16 — i.e. VNNI directly dissolves the blocker that defeated #61's Q8_0 dispatch.

## Why this is a "pause", not a "go"

- **SDK availability.** dotLLM targets `net10.0`, builds on SDK **10.0.300** (`global.json`
  rollForward latestMinor from 10.0.100). `AvxVnni.V512` merged into runtime `main`, which is
  **.NET 11** development. New `System.Runtime.Intrinsics.X86` public surface is **never** added in
  .NET 10 servicing bands (10.0.1xx/2xx/3xx) — so V512 VNNI is **not callable from 10.0.300**. A V512
  Q8_0 path today would need a .NET 11 bump or a reflection/`OperatingSystem`-gated shim — neither
  desirable in a focused PR.
- **Series, not a one-off.** #128365 is the **first of ~3-4 planned related CPU-operator PRs** to
  dotnet/runtime. Designing the dotLLM Q8_0 kernel around V512 before the series settles risks
  rework if later PRs change/extend the surface (e.g. additional widening ops, BF16, AVX10.2 forms).
- **Therefore:** writing the "old" 23-YMM AVX2 Q8_0 tile now would be throwaway work (VNNI obsoletes
  the shape), and writing the V512 VNNI tile now is premature (unreleased BCL + unsettled series).

## What IS actionable today (the V256 path)

`AvxVnni` **256-bit** (`MultiplyWideningAndAdd` on `Vector256<int>`, VPDPBUSD) has shipped since
**.NET 5** and is fully available in net10.0 — **and the repo uses it nowhere**. This is the
immediately-buildable way to attack the #61/#315 Q8_0 register-pressure block **without** waiting on
.NET 11:

- Rewrite the Q8_0 4×N outer-product microkernel using `Avx512.IsSupported ? … : AvxVnni.IsSupported
  ? V256-VNNI : maddubs-fallback`. The VNNI branch removes the `ones` + product-temp registers, so a
  4×3 (or wider) tile should fit the 16-YMM file and finally beat `3× VecDot4RowsR4`.
- Implementation caveat: **VPDPBUSD is u8×s8**, Q8_0 is **s8×s8**. Use the standard zero-point trick
  (offset activations by +128 to unsigned, subtract a `128·Σweights` compensation term per output) —
  this is exactly how llama.cpp / oneDNN do VNNI int8 GEMM. Reference `ggml-cpu` VNNI path.
- This gives a real perf win on every VNNI-capable CPU (Ice Lake+/Zen4+) **now**, and the V512 form
  drops in as a one-branch upgrade once dotLLM moves to .NET 11.

## Decision needed (why this is paused for the user)

The user authored #128365 and plans 3-4 follow-up runtime CPU-op PRs. The right sequencing is the
user's call:

- **(A) V256-VNNI Q8_0 kernel now** (net10.0-clean, ships immediately, V512 upgrade later). My
  recommendation if we want a landed Q8_0 win this campaign.
- **(B) Hold all Q8_0 GEMM work** until the runtime VNNI series + a .NET 11 bump land, then do V256+V512
  together in one PR. Cleaner single PR, but no Q8_0 win until .NET 11.
- **(C) Proceed with a non-VNNI tile reshape** (4×2 AVX2, no VNNI) as #315's body floated — but this is
  likely throwaway once VNNI lands; not recommended.

I did **not** write any kernel code — paused here as instructed.

---

## Addendum (2026-06-09) — verified ISA facts for the 512-bit path

Empirically confirmed (clean net10.0 compile + reflection on `System.Private.CoreLib`, SDK 10.0.300):

- **There is NO `System.Runtime.Intrinsics.X86.Avx512Vnni` class in net10.0.** The only `*Vnni*` types are `AvxVnni`, `AvxVnniInt8`, `AvxVnniInt16`. The BCL deliberately unified the 512-bit VNNI under `AvxVnni.V512` rather than a standalone `Avx512Vnni`.
- `AvxVnniInt8.V512.MultiplyWideningAndAdd` exists in net10.0 but is gated on **AVX_VNNI_INT8** (AVX10.2 / Granite Rapids), NOT AVX512-VNNI — so it is `IsSupported == false` on Zen4/Zen5/Ice Lake-SP/Sapphire Rapids. **Not usable for these targets.**
- `AvxVnni.V512` (dotnet/runtime#128365, **.NET 11**) maps to the **`AVX512v3`** group (`BITALG+VBMI2+VPOPCNTDQ+VNNI`) = standard **AVX512-VNNI** (CPUID leaf 7 ECX bit 11) → `IsSupported == true` on **Zen5 / Strix Halo**. This is the correct route, and it requires the .NET 11 bump (tracked in #322).

**Consequence:** the AVX-512 VNNI optimisation for Zen5 **cannot be written on net10.0** at all — it is hard-gated on .NET 11. The net10.0 deliverable is the V256 path only (#321). Do not attempt an `Avx512Vnni`-based AVX-512 path until dotLLM moves to .NET 11.
