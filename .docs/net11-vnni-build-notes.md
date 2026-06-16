# Branch notes — `issue/322-q8-avx512-vnni-net11` (AVX-512 VNNI Q8_0)

> ⚠️ **LONG-RUNNING .NET 11 BRANCH.** The AVX-512 VNNI path here uses `AvxVnni.V512`
> (`VPDPBUSD-512`), which ships in **.NET 11** (dotnet/runtime#128365). It is **not** in
> net10.0. The kernel code is therefore guarded `#if NET11_0_OR_GREATER` so the branch still
> builds on net10.0 (with the V512 path excluded). Do **not** bump dotLLM's TargetFramework to
> net11.0 on `main` / merge this until dotLLM officially moves to .NET 11.

## What's on this branch
- `OuterProductQ8_0Avx512Vnni_4x6` + helper `Avx512DualBlockVnniFma` in `src/DotLLM.Cpu/Kernels/MatMul.cs`
  — the existing AVX-512 maddubs reduction (`Avx2.Sign` + `maddubs` + `madd(ones256)` on two 256-bit
  halves) replaced by a single `AvxVnni.V512.MultiplyWideningAndAdd` over `Vector512`. Drops `ones256`.
- A/B benchmark `benchmarks/DotLLM.Benchmarks/OuterProductAvx512VnniBenchmarks.cs`
  (`Avx512_4x6_Maddubs` baseline vs `Avx512_4x6_Vnni`).
- Gated parity test (`AvxVnni.V512.IsSupported`) in `OuterProductGemmTests.cs`.

## How to build / benchmark it (requires an AVX-512-VNNI CPU, e.g. Zen5 / Strix Halo)

`AvxVnni.V512` is not in any shipping SDK yet, so you need a **local dotnet/runtime build**:

1. Build the runtime (branch with `AvxVnni.V512`, e.g. `feature/avxvnni.v512`):
   ```
   build.cmd -s clr+libs+packs+host -c Release /p:FeatureInterpreter=true
   ```
   (`/p:FeatureInterpreter=true` works around a platform-manifest drift where `clrinterpreter.dll`
   is built in Release but not listed unless the interpreter feature is on.)
   Produces `Microsoft.NETCore.App.{Ref,Runtime.win-x64}.11.0.0-dev.nupkg` in
   `artifacts/packages/Release/Shipping`, and `Core_Root` under
   `artifacts/tests/coreclr/<os>.<arch>.Release/Tests/Core_Root` (corerun + dev CoreLib).

2. Consume it from dotLLM. The net11 consume config is **temporary scaffolding** (remove before merge):
   - TFM → `net11.0`
   - `Directory.Build.targets`: `<FrameworkReference Update="Microsoft.NETCore.App"
     TargetingPackVersion="11.0.0-dev" RuntimeFrameworkVersion="11.0.0-dev" />`
     (the **TargetingPackVersion** override is essential — `RuntimeFrameworkVersion` alone does NOT
     pull the dev *ref* pack, so the new intrinsic won't resolve at compile time)
   - restore feed: `-p:RestoreAdditionalProjectSources=<runtime>/artifacts/packages/Release/Shipping`
   - neutralise `global.json` (it pins the 10.0.x SDK) and build with the runtime's preview-11 SDK
     at `<runtime>/.dotnet/dotnet.exe`
   - suppress `NU1603` (ILLink.Tasks 11.0.0-dev isn't in the feed; resolves an approximate version)

3. Run the benchmark on the local runtime via BenchmarkDotNet's CoreRun toolchain:
   `--coreRun <runtime>/artifacts/tests/coreclr/<os>.<arch>.Release/Tests/Core_Root/corerun.exe`

A ready-to-run wrapper that applies the config, runs, and auto-reverts is at
`.claude/strix/run-vnni-bench.cmd` (Strix-local; not part of the build).

## Status — COMPLETE (negative result)

Fully characterised on AMD Zen5 / Strix Halo against the local `11.0.0-dev` runtime:

- **Compiles:** `DotLLM.Cpu` + the benchmark project build clean net11.0 against the local ref pack.
- **Correct:** the gated V512 parity tests pass **9/9** (guards forced open so they can't skip-pass;
  run on `Core_Root`-overlaid runtime) — `OuterProductQ8_0Avx512Vnni_4x6` matches the scalar reference
  across block counts 1/2/3/8/17/18/48/128 + the discrimination self-check.
- **Performance (3-way, CoreRun, 30 iter / 10 warmup):**

  | Q8_0 4×6 outer-product, K=4096 | Mean | vs maddubs |
  |---|---|---|
  | `Avx512_4x6_Maddubs` (baseline) | 3.508 µs | 1.00 |
  | `Avx512_4x6_Vnni` (sign-trick, per-cell packs) | 4.708 µs | 1.34× slower |
  | `Avx512_4x6_VnniZp` (zero-point, hoisted) | **3.370 µs** | **~0.96× (parity)** |

  The sign-trick kernel's 1.34–1.43× slowdown was an **implementation artifact** — per-cell
  `Vector256→Vector512` packing + the 256-bit-only `vpsignb` sign trick. The **zero-point (+128)**
  rewrite (`u = x XOR 0x80` hoisted per-token, `VPDPBUSD(u, raw_w)` with `Σw` hoisted per-row,
  deferred `−128·Σw` correction) removes that overhead and reaches **statistical parity** with maddubs
  (the ~4% edge is within noise; maddubs baseline itself drifts ~9% run-to-run).

**Conclusion:** VNNI is **not** fundamentally behind on Zen5 for Q8_0 — properly implemented it **matches**
maddubs. It doesn't clearly win (Q8_0's per-block scales cap VNNI's int-accumulation advantage; Zen5's
256-bit maddubs throughput is strong). No reason to ship over maddubs today, but it's a viable,
competitive path — a clean baseline for when more 512-bit ops / a scale-separated data layout open
further potential. Tracked on kkokosa/dotLLM#322 (kept open as the baseline).

### Gotcha — shipping runtime pack reports the new ISA unsupported
`AvxVnni.V512.IsSupported == false` from the `Microsoft.NETCore.App.Runtime.win-x64.11.0.0-dev` **pack**
even on Zen5 (self-contained app throws `PlatformNotSupportedException`). Only the test `Core_Root`
(corerun) reports it correctly. `DOTNET_ReadyToRun=0` did NOT fix it — overlaying `Core_Root`'s
`coreclr.dll`/`clrjit.dll`/`System.Private.CoreLib.dll` over the published output did. Runners that
encode the working flows: `.claude/strix/run-vnni-bench.cmd` (benchmark, via `--coreRun`) and
`.claude/strix/run-vnni-parity.cmd` (parity, via the binary overlay).

---

## BF16 result (2026-06-13, Zen5/Strix Halo, CoreRun on dotnet-runtime-dev)

4×6 Q8_0 outer-product tile, K=4096, 30 iter:

| Kernel | Mean | StdDev | Ratio vs maddubs |
|---|---:|---:|---:|
| Avx512_4x6_Maddubs | 4.67 µs | 0.53 (bimodal/thermal) | 1.00 |
| Avx512_4x6_Bf16 | 1.62 µs | 0.17 | 0.35 (≈2.9× faster) |
| Avx512_4x6_Vnni / _VnniZp | NA (AvxVnni.V512 PNSE on dev) | — | — |

- BF16 ≥2.4× faster even fastest-to-fastest (maddubs min 3.55 µs vs BF16 max 1.93 µs). Win is robust to the noisy maddubs baseline.
- Correctness: BF16 parity 8/8; full outer-product suite 100/0/0 on dev. Tolerance |bf16−scalar| ≤ 3e-2·(tile-max|scalar|+1e-3).
- Why: scale folded into bf16 value → inner loop is a pure VDPBF16PS chain into one fp32 acc; no per-block integer reduction/dual-scale fold.
- Runtime path to get here: Avx512Bf16 ushort-shape API (runtime bc979bc) + JIT codegen fixes (dispatch case + FullOpts EVEX encoding for zmm16–31). dotLLM kernel commit 3d79d73.
- AvxVnni.V512.IsSupported == false on the dev build despite the feature/avxvnni.v512 merge — same-runtime 4-way pending runtime-side ISA re-enablement. VNNI figures stand from the feature/avxvnni.v512 runtime (Vnni ~1.36–1.43× slower, VnniZp ~parity).
- OPEN: BF16 is an 8-bit-mantissa approximation; end-to-end accuracy (perplexity) vs the ~2.9× speed is the product decision.

### Consolidated single-runtime 4-way (2026-06-13, after runtime 67051208b96 re-enabled AvxVnni.V512 on dev)

4×6 Q8_0 tile, K=4096, CoreRun, 30 iter, Zen5. Full parity 100/0/0 with all V512 guards forced.

| Kernel | Mean | Ratio vs maddubs |
|---|---:|---:|
| Avx512_4x6_Maddubs | 4.06 µs | 1.00 |
| Avx512_4x6_Vnni (sign-trick) | 5.20 µs | 1.33 (slower) |
| Avx512_4x6_VnniZp (zero-point) | 3.84 µs | 0.98 (~parity) |
| Avx512_4x6_Bf16 (VDPBF16PS) | 1.67 µs | 0.43 (~2.3× faster than best int kernel) |

Reproduces prior VNNI findings (sign-trick slower, zero-point ~parity) and confirms BF16 as the clear winner on one runtime.
