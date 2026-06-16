# TICKET — Bump to .NET 11 and upgrade Q8_0 outer-product GEMM to AVX-512 VNNI (V512)

> **FILED upstream as [kkokosa/dotLLM#322](https://github.com/kkokosa/dotLLM/issues/322)** (2026-06-09),
> cross-linked to the V256 work in #321 and the merged dotnet/runtime#128365. This file is the local
> working draft; #322 is the live tracker. Tracks the V256→V512 upgrade of the Q8_0 outer-product kernel.

---

**Title:** `perf(cpu/matmul): upgrade Q8_0 outer-product GEMM to AVX-512 VNNI (V512) after .NET 11 bump`

**Type:** enhancement · perf · follow-up
**Depends on:** the V256-VNNI Q8_0 outer-product kernel PR (the immediate work — follow-up to #315 / #312); a project-wide bump to **.NET 11**.
**Blocked until:** .NET 11 ships the `AvxVnni.V512` surface in a consumable SDK.

## Background

The Q8_0 outer-product GEMM (deferred from #315 / blocked in #61 by AVX2 register pressure — 23 YMM
needed) is being implemented now on **`AvxVnni` V256** (`MultiplyWideningAndAdd` → VPDPBUSD-256),
which is available in net10.0 and removes the `ones` mask + product temporaries that caused the
spills. See `.docs/gap-q8-vnni-matmul.md` for the full analysis.

The 512-bit form of the same instruction was added to the BCL in
**[dotnet/runtime#128365](https://github.com/dotnet/runtime/pull/128365)** (merged 2026-06-09):

- `AvxVnni.V512.MultiplyWideningAndAdd(Vector512<int>, Vector512<byte>, Vector512<sbyte>)` → **VPDPBUSD** (u8×s8 → +i32)
- `AvxVnni.V512.MultiplyWideningAndAddSaturate` → **VPDPBUSDS**
- (s16×s16 → i32 **VPDPWSSD** form also added)

Folded under the **AVX512v3** ISA grouping; hardware-gated via CPUID leaf 7 ECX bit 11.

This API landed in runtime `main` (= .NET 11). It is **not** in the .NET 10 servicing bands
(dotLLM currently builds on SDK 10.0.300 / `net10.0`), so it cannot be consumed today. It is also the
first of a planned **~3-4 related runtime CPU-operator PRs**.

## Scope of this ticket

Once dotLLM moves to **.NET 11** (and the runtime VNNI series has settled):

1. Bump `Directory.Build.props` `TargetFramework` → `net11.0` and `global.json` SDK pin → 11.0.x.
2. Add a `Avx512v3.IsSupported` (VNNI-512) branch to the Q8_0 outer-product microkernel that processes
   **64 int8 lanes/step** via `AvxVnni.V512.MultiplyWideningAndAdd` into `Vector512<int>` accumulators
   — a one-branch addition above the V256 path (same abs/sign zero-point preprocessing, same tile
   geometry, wider lane).
3. Re-validate the register budget: a 4×N tile on ZMM (32 registers available in AVX-512) has far more
   headroom than the AVX2 16-YMM file, so the tile can likely widen (e.g. 4×4 or 6×4) — benchmark.
4. Keep the dispatch ladder: `Avx512v3 (VNNI-512)` → `AvxVnni (VNNI-256)` → `maddubs+madd (AVX2)` →
   scalar. Numerical parity vs scalar reference required at each tier (discriminating tests, not
   degenerate shapes — per CLAUDE.md cross-backend rule).
5. Benchmark V512 vs V256 vs the reverted #61 baseline on a VNNI-512-capable CPU (Sapphire Rapids /
   Zen4+) and record before/after in the PR.

## Acceptance criteria

- [ ] Project targets .NET 11; CI green on the bumped TFM.
- [ ] Q8_0 outer-product kernel has a VNNI-512 path, parity-tested against scalar (bit-exact scalar /
      tolerance-bounded vector) with discriminating shapes.
- [ ] Dispatch prefers VNNI-512 when `Avx512v3.IsSupported`, falls back cleanly.
- [ ] Benchmark numbers (V512 vs V256 vs baseline) in the PR body.
- [ ] ROADMAP Phase 3 Step 26 Q8_0 portion noted complete.

## References

- dotnet/runtime#128365 (the V512 VNNI intrinsics — merged 2026-06-09)
- `.docs/gap-q8-vnni-matmul.md` (gap analysis)
- PR #315 / issue #312 (F32 outer-product GEMM — the parent kernel)
- PR #61 (original Q8_0 outer-product kernels, dispatch reverted)
- llama.cpp `ggml-cpu` VNNI int8 path (authoritative reference for the zero-point compensation)
