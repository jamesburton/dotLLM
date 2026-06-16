# Handoff — Q8_0 AVX-512 kernel ISA investigation (VNNI / BF16) on Strix Halo

> Prior thread (now steady-state): the **upstream PR campaign** (47 open PRs decomposing `feature/qwen3.6`
> into focused PRs on `kkokosa/dotLLM`, held/draft, waiting on maintainer review). That work is captured in
> git history + the open issues/PRs; nothing actionable there but fortnightly polite re-pings. This handoff
> covers the **live** thread below.

## Goal
Determine whether AVX-512 quantized-GEMM instructions can beat the existing **maddubs** Q8_0 outer-product
kernel on **AMD Zen5 / Strix Halo**, and drive any needed `dotnet/runtime` intrinsic additions. We benchmark
each candidate on Strix via a local runtime build, and only ship a kernel if it actually wins.

## Current State
Four Q8_0 4×6 outer-product variants live on branch **`issue/322-q8-avx512-vnni-net11`** (fork
`jamesburton/dotLLM`), all `#if NET11_0_OR_GREATER`, all parity-verified correct:

| Variant | Result on Zen5 (CoreRun, 30 iter) | Verdict |
|---|---|---|
| `Avx512_4x6_Maddubs` (baseline) | 3.5 µs | — |
| `Avx512_4x6_Vnni` (sign-trick, `AvxVnni.V512`) | **1.36–1.43× slower** | impl artifact (per-cell packing + 256-bit-only `vpsignb`) |
| `Avx512_4x6_VnniZp` (zero-point, hoisted packs) | **~0.96× (parity)** | the regression was implementation; VNNI ≈ maddubs |
| `Avx512_4x6_Bf16` (dequant→bf16 + `Avx512Bf16` VDPBF16PS) | **1.620 µs — Ratio 0.35 vs maddubs (≈2.9× faster)** | **MEASURED 2026-06-13: BF16 WINS** |

### BF16 result (2026-06-13, Zen5/Strix Halo, CoreRun on `dotnet-runtime-dev`, 30 iter)
| Method | Mean | StdDev | Ratio |
|---|---|---|---|
| `Avx512_4x6_Maddubs` (baseline) | 4.670 µs | 0.527 (bimodal, thermal) | 1.00 |
| `Avx512_4x6_Bf16` | **1.620 µs** | 0.174 | **0.35** |
| `Avx512_4x6_Vnni` / `_VnniZp` | NA — `PlatformNotSupportedException` | — | — |

- **BF16 is ~2.9× faster than maddubs** (and ≥2.4× even comparing fastest-to-fastest: maddubs min 3.55 µs vs
  BF16 min 1.45 µs). The maddubs baseline is noisy here (bimodal, ~11% StdDev — laptop thermal/freq variance),
  but BF16's whole distribution sits below maddubs's minimum, so the win is robust. Correctness: BF16 parity
  **8/8 PASS** within the bf16 tolerance `|bf16−scalar| <= 3e-2·(tile-max|scalar| + 1e-3)`.
- **Why BF16 wins:** scale folded into the bf16 value → the inner loop is a pure `VDPBF16PS` chain into one fp32
  accumulator with NO per-block integer reduction / dual-scale fold / `vpmaddwd`-with-ones step that maddubs
  pays every block. Fewer inner-loop instructions outweighs bf16's 2× element width.
- **CAVEAT — VNNI not measurable on dev:** `AvxVnni.V512` throws PNSE on `dotnet-runtime-dev` (env shows only
  `AvxVnni VectorSize=256`; the AVX512-VNNI feature isn't enabled in this build — the handoff's earlier "dev has
  BOTH" was inaccurate). VNNI/VnniZp numbers stand from the `C:/dotnet-runtime` `feature/avxvnni.v512` build
  (Vnni ~1.36–1.43× slower, VnniZp ~parity). A true 4-way on one runtime needs a build with both ISAs enabled.
- **OPEN — accuracy for inference:** parity passes a ~3%-relative tolerance, but BF16 is an approximation
  (8-bit mantissa). Whether that end-to-end quality is acceptable vs the ~2.9× speed is the product decision.

**BLOCKER (updated 2026-06-12, evening):** two prior blockers cleared; one NEW runtime-JIT blocker found.

1. **API pivot — DONE.** The `Avx512Bf16` intrinsics moved `Vector512<BFloat16>` → **`Vector512<ushort>`**
   (runtime-dev `bc979bc536f`), per maintainer (Tanning) feedback on PR #129326 (ushort lands cheaper; `BFloat16`
   overloads later). dotLLM kernel retargeted to `ushort` and pushed (**`3d79d73`**).
2. **Ref-pack type-generation blocker — RESOLVED (confirmed).** Rebuilt the `11.0.0-dev` packs from
   `bc979bc536f` (Ref nupkg now `18:27`, was stale at `15:42`), regenerated Core_Root, synced Strix dotLLM to
   `3d79d73`. **`DotLLM.Cpu` now compiles clean net11 against the dev pack** — the public `Avx512Bf16` type +
   ushort overloads resolve (no `CS0234`, no cref `CS1574`). The primitive-`ushort` pivot fixed it.
3. **Runtime JIT — NotImplementedException gap FIXED.** Runtime agent added the missing
   `case InstructionSet_AVX512_BF16:` in `genHWIntrinsic` (hwintrinsiccodegenxarch.cpp) so SpecialCodeGen
   intrinsics no longer fall through to `unreached()`. New `clrjit.dll` in Core_Root @ 22:39. The 3-register
   `bf16probe` now prints P1/P2/P3 OK.

4. **Runtime JIT — optimizing codegen FIXED (2026-06-13).** Runtime agent landed 3 commits: `Wire JIT codegen
   for Avx512Bf16 ushort-shape intrinsics`, `Add AVX512_BF16 to per-ISA codegen dispatch`, `Mark INS_vdpbf16ps
   as 3-operand RMW in Is3OpRmwInstruction`. New `clrjit.dll` in Core_Root @ 10:26. **`bf16stress` all-green**
   (FullOpts 4-acc + 24-acc both clean) and **BF16 parity 8/8 PASS** against the scalar oracle (all blockCounts
   2/3/8/17/18/48/128 + trivial 1). The kernel is correct under FullOpts. *(Side note for runtime agent, not a
   dotLLM blocker: `bf16stress` under `DOTNET_JITMinOpts=1` computes `c=0` while FullOpts computes the correct
   ~49.5 — a possible MinOpts value bug; dotLLM is unaffected since the kernel is AggressiveOptimization/FullOpts.)*

   ~~PRIOR BLOCKER (now fixed):~~ the dispatch fix was only validated under **MinOpts** (tier-0 probes). The
   **optimizing JIT** emitted **broken codegen** for `vcvtne2ps2bf16`:
   - `DOTNET_JITMinOpts=1` → **clean**;  FullOpts 4-acc (no spill) → **#UD `Illegal instruction` (0xC000001D)**;
     FullOpts 24-acc → **AccessViolation (0xC0000005)**.
   - Opt-level sensitivity ⇒ JIT codegen bug, not dotLLM (MinOpts runs the same algorithm+buffers fault-free).
   - **Hypothesis:** EVEX encoding of the AVX512-BF16 ops is malformed for the **high register bank
     (zmm16–31)** — the optimizing allocator uses zmm16+ (disasm confirms `vcvtne2ps2bf16 zmm31, zmm31, zmm15`),
     MinOpts confines to zmm0–15 and masks it. 4-acc still uses zmm16+ → #UD; 24-acc spills → AV downstream.
   - **Repro (self-contained, no dotLLM):** `cmd /c C:\bf16stress\run-bf16stress.cmd` (control 4-acc vs faithful
     24-acc vs MinOpts). Disasm saved: `C:\temp\bf16-run4-fullopts.asm`, `C:\temp\bf16-run24-fullopts.asm`.
   - **Relay doc:** `C:\temp\avx512bf16-fullopts-codegen-bug.md` (supersedes `avx512bf16-v512-codegen-gap.md`).
   - **BF16 parity 7/8 FAIL** (AccessViolation in the AggressiveOptimization kernel); only `blockCount=1` passes
     (exact AVX2 tail, never enters the bf16 loop). **The 4-way benchmark cannot run** until the optimizing
     codegen is fixed. No dotLLM/pack rebuild needed for the retry — only `clrjit.dll`.

Branch HEAD: `3d79d73` (… → bf16 tolerance fix `7ecb470` → **ushort retarget `3d79d73`**).

## Key Decisions
- **VNNI is not a clear win on Zen5** — properly implemented (zero-point, no sign trick, hoisted packs) it
  reaches *parity* with maddubs, not a win. Q8_0's per-block fp16 scales cap VNNI's int32-accumulation
  advantage; Zen5's maddubs throughput is already strong.
- **BF16 is the remaining candidate**: dequant int8→bf16 **folding the per-block scale into the value**, then
  `VDPBF16PS` accumulates already-scaled products across all blocks into one fp32 acc — removing the per-block
  reduction that is likely the real bottleneck. Trade-off: bf16 precision (approximation) + 2× dot ops (16-bit
  vs int8's 8-bit). Might win or lose — that's why we measure.
- **`Avx512Bf16` / `Avx512Ifma` are NOT in .NET 10/11** (only `Avx512F/BW/CD/DQ/Vbmi/Vbmi2` + `AvxVnni` 256
  are exposed; `AvxVnniInt8/16`, `Avx10v*` exist but are unsupported on Zen5). So BF16 needed a new BCL
  intrinsic → **raised `dotnet/runtime#129323` (API proposal) + PR #129326**. Proposal at
  `.docs/proposal-avx512-bf16.md`. The PR's API matches: `Avx512Bf16.MultiplyWideningAndAdd(Vector512<float>,
  Vector512<BFloat16>, Vector512<BFloat16>)` + `ConvertToBFloat16(...)`; `BFloat16` is `System.Numerics.BFloat16`.
- **`AvxVnni.V512`** (the earlier intrinsic, dotnet/runtime#128365, in `C:/dotnet-runtime` `feature/avxvnni.v512`)
  is gated on AVX512-VNNI (AVX512v3) → true on Zen5. Tracked in `kkokosa/dotLLM#322`.

## What Worked — the Strix build/consume pipeline (hard-won; reuse exactly)
- **SSH:** `ssh strix` (Tailscale, key auth). Strix = Windows, Zen5 / Ryzen AI Max.
- **Build runtime:** `cd <worktree> && build.cmd -s clr+libs+packs+host -c Release /p:FeatureInterpreter=true`
  — the `FeatureInterpreter=true` flag works around a manifest break (`clrinterpreter.dll` built in Release but
  not in the platform manifest). Produces `Microsoft.NETCore.App.{Ref,Runtime.win-x64}.11.0.0-dev.nupkg` in
  `<worktree>/artifacts/packages/Release/Shipping`.
- **Generate Core_Root** (separate step, fresh worktrees lack it): `src\tests\build.cmd x64 Release generatelayoutonly`
  → `<worktree>/artifacts/tests/coreclr/windows.x64.Release/Tests/Core_Root/corerun.exe`.
- **Consume from dotLLM** (TEMP scaffolding, applied + reverted by the runner scripts):
  - TFM → `net11.0` (edit `Directory.Build.props`).
  - `Directory.Build.targets`: `<FrameworkReference Update="Microsoft.NETCore.App" TargetingPackVersion="11.0.0-dev"
    RuntimeFrameworkVersion="11.0.0-dev" />`. **The `TargetingPackVersion` override is ESSENTIAL** —
    `RuntimeFrameworkVersion` alone does NOT pull the ref pack, so the new intrinsic won't resolve at compile.
  - `-p:RestoreAdditionalProjectSources=<worktree>\artifacts\packages\Release\Shipping`, `-p:NoWarn=NU1603`
    `-p:TreatWarningsAsErrors=false`.
  - Neutralize `global.json` (`ren global.json gj.bak`; it pins the 10.0.x SDK) and build with the worktree's
    preview-11 SDK at `<worktree>\.dotnet\dotnet.exe`.
- **Benchmark (BDN):** `--coreRun <Core_Root>\corerun.exe --cli <worktree>\.dotnet\dotnet.exe` — the **`--cli`
  is required** or BDN's bootstrap build uses the system 10.x SDK and fails (can't target net11).
- **Parity (xunit):** self-contained build, then **overlay** `Core_Root`'s `coreclr.dll`/`clrjit.dll`/
  `System.Private.CoreLib.dll` over the publish output, run `--no-build`. **Force the gated guards open**
  (`if (!Xxx.IsSupported)` → `if (false)`) so a skip can't masquerade as a pass.
- **Runner scripts on Strix:** `.claude/strix/run-vnni-bench.cmd`, `run-vnni-parity.cmd` (currently point at
  `C:/dotnet-runtime`; **repoint to `C:/dotnet-runtime-dev` for BF16**). `.claude/strix/isaprobe/` reflects a
  runtime's `System.Runtime.Intrinsics.X86` IsSupported set.

## Strix runner-script gotchas (2026-06-14 — cost a confusing debug cycle)
- **NEVER name a script variable `OUTDIR` (or anything case-insensitively equal to `OutDir`).** Windows env
  vars are case-insensitive, so `set OUTDIR=c:\temp\...` sets MSBuild's **`OutDir`** property → the whole build
  redirects its output to `c:\temp\...` instead of `bin\`, and the Core_Root overlay (copied to `bin\`) is never
  actually used (tests silently run on the stock 11.0.0-dev pack runtime). Harmless for decode / AVX-512-quant
  probes (those don't need bf16/VNNI.V512, which are the only things the pack gates off on Zen5), but it breaks
  Phase-B disasm (checked clrjit overlay won't apply) and left a zombie `testhost` locking the `c:\temp` DLL,
  failing the next build. **Fixed: renamed to `RESULTDIR` in `run-avx512-disasm.cmd` / `run-decode-closeloop.cmd`.**
  (`run-decode-sweep.cmd` / `run-decode-edge.cmd` still have the old name but already produced valid results.)
- **One `dotnet test` invocation whose filter matches multiple probe classes runs them in PARALLEL** (distinct
  `[Collection]`s) → multiple 30-thread spin-wait decode pools collide → garbage throughput (4–11 tok/s). Run
  **one model per invocation** (sequential), as `run-decode-sweep.cmd` does.
- **Kill zombies before a Strix build:** `taskkill /F /IM testhost.exe /T & taskkill /F /IM dotnet.exe /T` —
  a hung/parallel test leaves a `testhost` holding the output DLLs.

## What Didn't Work (don't retry)
- **Self-contained / shipping runtime PACK reports new-ISA `IsSupported == FALSE`** even on Zen5 →
  `PlatformNotSupportedException`. Only `Core_Root`'s JIT-evaluated CoreLib reports true. `DOTNET_ReadyToRun=0`
  did NOT fix it. Use `corerun`/`Core_Root` or the binary overlay.
- **Cache collisions:** both runtime worktrees produce version `11.0.0-dev`, so stale packs poison the NuGet
  cache. Must `rmdir /s /q %USERPROFILE%\.nuget\packages\microsoft.netcore.app.{ref,runtime.win-x64,host}\11.0.0-dev`
  before restoring from a different worktree's feed. (NB: the Strix cache is currently DIRTY from overlay
  attempts — clear it before the BF16 retry.)
- **`findstr "Avx512Bf16"` on the ref DLL is NOT a valid consumability check** — it gives false positives. Use a
  1-line compile (`Avx512Bf16.IsSupported`) as the real test.
- Earlier BF16 dev build failed with 12× `CS0234 'BFloat16' does not exist in namespace 'System'` — the
  `BFloat16` ref-layering (it's `System.Numerics.BFloat16`, ref in `System.Runtime` but intrinsics ref compiles
  in CoreLib ref). Fixed on the runtime side (commit `eaf19c5` "HARDWARE_INTRINSIC sort-order fix"), but the
  Avx512Bf16 *public ref* still isn't generated.

## Recent Changes
- **`issue/322-q8-avx512-vnni-net11`** (fork): `600c2d7` VNNI-sign, `d6cd1c5` VNNI-zeropoint, `794d42b` BF16
  kernel+bench+test, `13848a1` `using System.Numerics`, `7ecb470` bf16 tolerance fix. Kernels in
  `src/DotLLM.Cpu/Kernels/MatMul.cs`; bench `benchmarks/DotLLM.Benchmarks/OuterProductAvx512VnniBenchmarks.cs`;
  tests `tests/DotLLM.Tests.Unit/Cpu/Kernels/OuterProductGemmTests.cs`.
- **`kkokosa/dotLLM`**: #321 / PR #323 (V256 VNNI Q8_0, ~1.3–1.6× faster on Meteor Lake); #322 (issue tracking
  the AVX-512 VNNI investigation — has the full benchmark + parity comment history).
- **`dotnet/runtime`**: #129323 (Avx512Bf16 API proposal) + #129326 (impl PR). Also #128365 (`AvxVnni.V512`).
- **`.docs/`** (untracked working notes): `net11-vnni-build-notes.md` (full VNNI record), `gap-q8-vnni-matmul.md`,
  `ticket-net11-vnni-v512-swap.md`, `proposal-avx512-bf16.md`.
- **`.claude/strix/`** (Strix-local tooling, untracked): VNNI runner cmds + `isaprobe/`; **BF16 additions:**
  `run-bf16-parity.cmd`, `run-bf16-bench.cmd` (repointed to `C:/dotnet-runtime-dev`, clear stale `11.0.0-dev`
  cache, force `Avx512Bf16.IsSupported` guard open, filter `OuterProductAvx512Bf16` / `*Avx512*`),
  `bf16probe/` (isolated 3-intrinsic repro — tier-0/MinOpts only, deployed to Strix `C:\bf16probe`),
  `bf16stress/` (self-contained FullOpts codegen discriminator: 4-acc control vs 24-acc faithful vs MinOpts,
  deployed to `C:\bf16stress`), and the rebuild wrappers `bf16-rebuild.cmd` / `bf16-gen-coreroot.cmd` (deployed
  to `C:\dotnet-runtime-dev`). Relay docs + disasm in shared `C:\temp` (`avx512bf16-fullopts-codegen-bug.md`,
  `bf16-run{4,24}-fullopts.asm`).

## Important Context
- **Two runtime worktrees on Strix**: `C:/dotnet-runtime` (`feature/avxvnni.v512` → `AvxVnni.V512.IsSupported
  == true`) and `C:/dotnet-runtime-dev` (`dev`). dev source MERGED both features, but **on the dev build
  `AvxVnni.V512.IsSupported == false`** (Avx512Bf16=true, AvxVnni 256=true) — the V512 ISA enablement didn't
  survive the merge (verified 2026-06-13 via `C:\isaprobe2`). So a same-runtime 4-way needs the runtime agent to
  re-enable `AvxVnni.V512` on dev. Relay: `C:\temp\avxvnni-v512-disabled-on-dev.md`. Each worktree has its own
  `.dotnet` (preview-11 SDK) + `artifacts`.
- **Strix git CANNOT push** (no non-interactive creds) — fetch/read only on Strix; commit+push from the local
  dev machine (`C:\Development\dotLLM`, creds OK as `jamesburton`). To edit a branch file, use a worktree:
  `git worktree add --detach <path> <sha>`, edit, `git push origin HEAD:issue/322-q8-avx512-vnni-net11`.
- **The user (James / `jamesburton`) is a `dotnet/runtime` contributor.** The runtime-side PR authoring/CI/fixes
  are handled by a **separate agent on Strix** — DO NOT fix the runtime branch yourself; hand precise, testable
  repros back to the user to pass on.
- dotLLM `issue/322` stays **net10.0 + `#if NET11` guarded**; the net11 consume config is script-applied and
  reverted, so nothing committed needs removing before merge. The kernels can't be benchmarked on the local dev
  box (net10 SDK, no AVX-512) — Strix is the only validator.
- The kernel agents authored against the API spec; net10 compiles only that `#if NET11` regions are balanced, so
  net11-only bugs (missing `using`, API mismatch) surface ONLY on the Strix net11 compile.

## Done (2026-06-13)
- ✅ BF16 kernel retargeted to `ushort`, pushed `3d79d73`; compiles clean net11 against the dev pack.
- ✅ Runtime JIT: NotImplementedException gap + FullOpts #UD/AV codegen bug both FIXED by the runtime agent
  (`clrjit.dll` @ 10:26). `bf16stress` all-green.
- ✅ **BF16 parity 8/8 PASS**; **full outer-product parity 100/0/0** on dev (maddubs + BF16 real-pass; VNNI-V512
  tests vacuous — early-return while `AvxVnni.V512`=false; VNNI validated on the feature/avxvnni.v512 runtime).
- ✅ **Consolidated single-runtime 4-way** (K=4096 tile, after runtime `67051208b96` re-enabled `AvxVnni.V512`
  on dev; full parity 100/0/0 with all V512 guards forced):
  | Kernel | Mean | Ratio |
  |---|---|---|
  | maddubs | 4.06 µs | 1.00 |
  | Vnni (sign-trick) | 5.20 µs | 1.33 (slower) |
  | VnniZp (zero-point) | 3.84 µs | 0.98 (~parity) |
  | **Bf16** | **1.67 µs** | **0.43 (~2.3× faster than best int kernel)** |
  Reproduces the prior VNNI findings and lands BF16 as the clear winner. (Earlier dev-only run showed
  BF16 1.62 µs vs a noisier maddubs 4.67 µs = 2.9×; the maddubs baseline is multimodal/thermal, BF16 is stable
  at ~1.6–1.7 µs.) Draft for #322: `.docs/issue-322-bf16-result-DRAFT.md`.

## E2E BF16-accuracy workstream (2026-06-13, two new branches off `dev`)
Goal: measure the BF16 Q8_0 kernel's *accuracy* end-to-end (the open product question). Two worktrees:
- **`issue/perplexity-harness`** (worktree `C:/Development/dotLLM-ppl`): `PerplexityEvaluator` + `dotllm eval
  perplexity` CLI. Anchored against a committed PyTorch reference (reproduces meanNLL 3.3018) + shift-discriminating
  + closed-form math tests. SmolLM-135M Q8_0 → ppl ~7.3–9.1. Unit 6/6 pass (verified). Touches neither MatMul.cs
  nor InferenceOptions. NOT pushed.
- **`issue/q8-outerproduct-prefill`** (worktree `C:/Development/dotLLM-q8outer`): wires `OuterProductGemmQ8_0`
  into the prefill (n>1) Q8_0 path behind flag **`TransformerModel.UseOuterProductQ8Prefill`** (off by default).
  Key finding: **no new RepackR4 needed** — the forward path already R4-repacks Q8_0 weights at load time
  (`TransformerWeights.RepackWeights()`), exactly the layout the dispatcher wants. A **BF16 dispatch seam**
  (`#if NET11`) is marked in `GemmOuterProductQ8_0` for the issue/322 BF16 kernel. Commits `0417a85` (wiring) +
  `dfe7e79` (Llama parity test). NOT pushed.
  - **Discriminating gate (verified): full-logits A/B, same process, outer-product vs inner-product, both models
    bit-identical** (maxRelDiff = 0.0): SmolLM-135M (211 outer-product calls/forward) + Llama-3.2-1B (113). Proves
    the wiring/layout/shape handling is correct on AVX2/scalar (local box has no AVX-512).
- **Local validation only** (AVX2): the AVX-512 4×6 and BF16 paths are NOT exercised locally — that's the Strix step.

### Bug hunt RESOLVED (2026-06-13): NOT a kernel bug — over-strict full-forward parity tolerance
Combined branch **`issue/q8-bf16-prefill-e2e`** (pushed) builds clean net11 on Strix. The same-process logits
parity initially **FAILED** on AVX-512 for n=4/n=10 (maxAbsDiff ~0.55–0.62, worst cell sign-flipped), while
bit-identical on the local AVX2 box. Full systematic-debugging investigation (all on Strix AVX-512):
1. **Kernel exonerated** — `OuterProductGemmTests` 28/28 pass; new `OuterProductGemm_RealisticParallel_MatchesReference`
   (realistic m=576/1536 × n=4/5/10 × parallel, ProcessorCount threads, looped 8×) passes; per-matmul outer-vs-reference
   diff is **0.0 bit-identical at n=4/5 (k=576)**, ≤5.5e-4 at n=10.
2. **Opt-level invariant** — same failure under `DOTNET_JITMinOpts=1` (identical values) ⇒ not a JIT FullOpts
   codegen bug, not a race; fully deterministic.
3. **Arbiter (decisive)** — `GroundTruth_GemmAndOuter_AtRealisticShapes` compares BOTH real paths against a SCALAR
   ground truth from the ORIGINAL row-major weights (shares neither path's R4 repack): **GEMM_vs_truth ≈ 3e-5 abs /
   1e-4 rel AND OUTER_vs_truth ≈ 3e-5 abs / 1e-4 rel** at n=4/10. **Both matmul paths are correct to 5 sig figs.**

**Root cause:** the two paths round *correctly but differently* (~7e-5/matmul); on AVX2 the roundings coincide
(bit-identical), on AVX-512 they differ slightly, and over ~30 layers / ~210 matmuls (residual + softmax) this
compounds to ~2% of the max-logit magnitude — with near-zero (cancellation) logits showing as spurious "sign
flips" despite both values being correct. The parity test's **per-cell 1e-3 tolerance was wrong** for a
full-forward comparison (it assumed AVX2's coincidental bit-identity). **Fix:** scale-normalized tolerance
(`scaleNormDiff = maxAbsDiff/maxBaselineAbs ≤ 5e-2`), matching the BF16 test; tight correctness lives in the
per-matmul scalar-truth arbiter. Commits: `797d555` (RealisticParallel), `490bd26` (arbiter), `30462e7` (tolerance fix).

**Verified PASS (2026-06-13):** SmolLM parity 3/3 (scaleNormDiff 3.4e-7 / 1.98e-2 / 2.28e-2 for n=2/10/4) +
arbiter ~1e-4 + RealisticParallel green. **The outer-product prefill (and BF16) path is CORRECT.**

### BF16 accuracy answer (the original goal)
With a verified-correct integer baseline, the BF16 A/B is valid. **`OuterProductBf16Accuracy`** on SmolLM-135M Q8_0
now measures **three** paths on the same 32-token passage so the cost is anchored against the TRUE production
default (inner-product), not integer-outer (advisor catch — commit `58947cc`, verified Strix 2026-06-13):

| Prefill path | Perplexity | vs inner baseline |
|---|---:|---:|
| inner-product (cache-tiled, production default) | 26.2386 | — |
| integer outer-product (R4 maddubs) | 26.1375 | **−0.39%** |
| bf16 outer-product (VDPBF16PS) | 26.6587 | **+1.60%** |

Two findings: (1) **the outer-product restructuring is quality-neutral** — integer-outer vs inner is −0.39%
(within FP rounding; new ≤1% regression gate in the test passes). (2) **bf16's honest end-to-end cost is +1.60%
perplexity** vs what users run today, for ~2.3–2.9× microkernel speed. (Measured vs integer-outer alone it reads
+1.99%, but integer-outer sits slightly below the true baseline, so +1.60% is the apples-to-apples figure.)

### Llama-3.2-1B: output alignment + end-to-end operator benchmark (2026-06-13, Strix)
**Alignment (gate 1) — PASS.** Extended the scalar-ground-truth arbiter to Llama K dims (k=2048 → 64 blocks,
k=8192 → 256 blocks, down_proj): integer outer-product matches exact scalar Q8_0 truth to abs ~1e-4…6e-4 even at
4.5× SmolLM's depth — kernel correct at these shapes. E2E parity (integer-outer vs inner full logits) **3/3 PASS**
at scaleNormDiff ≤1.42e-2 (tighter than SmolLM; Llama-3.2-1B has 16 layers vs SmolLM's 30, so less compounding).
No tolerance change. (Both integer-only → validated on stock net10; commits `70bea61`.)

**Benchmark (gate 2) — end-to-end prefill, net11 + Core_Root (bf16 codegen live), 32 threads, median of 9:**

| Prefill | inner-product (default) | integer outer-product | bf16 outer-product |
|---|---:|---:|---:|
| pp256 | 978 ms / 262 tok/s | 1152 ms / **0.85×** | 531 ms / **1.84×** |
| pp512 | 2167 ms / 236 tok/s | 2561 ms / **0.85×** | 1115 ms / **1.94×** |

Headline: **bf16 outer-product is ~1.9× faster end-to-end** than the production inner-product default — close to
the 2.3× microkernel figure, i.e. matmul dominates prefill more than expected (little dilution from
attention/RoPE/RMSNorm). **Caveat — integer-outer alone is ~15% SLOWER than inner on net11**: the outer-product
restructuring's value is *entirely* in enabling bf16, not the integer path. (On stock net10 integer-outer was
parity, 0.99–1.04×; net11's inner path is ~1.6× faster in absolute terms — root cause unconfirmed and irrelevant
to the headline, which is a same-runtime comparison — that raises the bar the integer maddubs outer-kernel can't
clear, but bf16 arithmetic clears easily.) Note the clean arithmetic-only comparison integer-outer→bf16 =
1152→531 ms = **2.17×**, matching the 2.3× microkernel figure: bf16 pays the same outer-product structure
penalty integer-outer does and still wins. Vacuousness guards confirmed distinct paths (gemmCalls 1017 for
outer/0 for inner; bf16Tiles >0 only for bf16). Commits `04662ce` (bench), script
`.claude/strix/run-llama-prefill-bench.cmd` (+ Strix `c:\temp\`).

**Decode is unaffected (commit `6b10d35`):** operators are gated `n>1`; decode (n=1) routes to inner GEMV. Measured
256-token-context decode: inner 37.0 / integer 37.5 / bf16 37.1 tok/s — **identical, gemmCalls & bf16Tiles = 0 for
all configs** (proof the operators never engage). So real generation = blend of 1.9× (prefill) and 1.0× (decode),
prompt:completion-weighted. **Decode-headroom aside:** ~27 ms/token ≈ 40 GB/s effective for the 1.1GB model — well
under Strix's LPDDR5X peak, so decode looks kernel-bound (256-bit maddubs GEMV, no VNNI/bf16/full-512) rather than
bandwidth-saturated → a future full-width AVX-512/VNNI decode GEMV may have headroom (separate work; bf16 wouldn't
cut bytes-moved so its decode value is unclear — measure before investing).

**bf16 accuracy on Llama-3.2-1B (longer corpus, robust — commits `f21d02f`/`6b10d35`):** 373-token passage —
inner 14.8216 / integer 14.8216 (**bit-identical, +0%**) / bf16 14.8653 (**+0.30%**). The integer outer-product is
exactly quality-neutral; bf16 costs **+0.30% perplexity** (the 32-token sample's −0.45% was noise; the longer
corpus averages it to the true small positive). Logits scaleNormDiff 1.77e-2 even at K=8192. Net trade:
**~1.9× prefill for ~0.3% perplexity** on Llama-3.2-1B (decode unchanged).

## Decode threading workstream (2026-06-14, branch `issue/q8-bf16-prefill-e2e`)
Separate from BF16: the AVX-512 microbench (commit `062efcf`) showed the wave-1/2 AVX-512 quant kernels are
net-neutral-to-NEGATIVE on Zen5 (Q8_0ToF32 0.17×). A read-only threading investigation
(`.docs/decode-threading-investigation.md`) resolved the decode thread-scaling anomaly and surfaced a pinning
footgun. Acting on both findings:

**Landed locally (committed, build-clean, no Strix needed):**
- ✅ **Pinning cap-2 footgun FIXED** (commit `21b06a4`). `--numa-pin`/`--pcore-only` built a `NumaTopology`
  whose single-node `MemoryChannelEstimate=2` drove the decode cap from 8 → 2 (~3× slower decode on Strix).
  Floored the topology branch at `DefaultDecodeThreadCountCap` (pinning can only raise the cap now). Added
  public `ComputeThreadPool.DecodeThreadCount` + host-independent regression test (10/10 pool tests pass).
- ✅ **§5 measurement infra** (commit `abf5d09`). `DecodeThreadScalingSweep` + SmolLM2-135M / Bielik-1.5B /
  Llama-3.2-1B probe wrappers; finer grid {2,4,8,16,24,30,32} (24/30/32 discriminate oversubscription vs
  cache contention), short(128)/long(2048) context, and a production-config-path run (default vs pinning) that
  validates the footgun fix end-to-end. All opt-in `DOTLLM_RUN_PREFILL_BENCH`.

**Strix-gated (need ONE "Go" — these need measurement to tune/justify, per the investigation's own §5
decision rule "re-measure before changing the default"):**
- ✅ **AVX-512 Phase A (net11 timing) — RUN 2026-06-14.** Script `.claude/strix/run-avx512-disasm.cmd`.
  **net11 does NOT rescue the kernels on Zen5 — so it is NOT a stock-net10-JIT-only issue:**

  | kernel | net10 AVX-512 speedup | net11 AVX-512 speedup |
  |---|---:|---:|
  | KvQuantize.Q8_0ToF32 | 0.17× | **0.12× (8× SLOWER)** |
  | KvQuantize.Q4_0ToF32 | 1.08× | 1.17× |
  | KvQuantize.F32ToQ4_0 | 0.63× | 0.91× |
  | FusedOps.RmsNormQ8_0  | 1.01× | 1.20× |
  | FusedOps.RmsNormQ8_1  | 0.88× | 0.90× |
  | FusedOps.RmsNormQ8_K  | 1.02× | 0.96× |

  Net11 nudges several up (Q4_0ToF32, RmsNormQ8_0 now ~1.2×; F32ToQ4_0 0.63→0.91) but **Q8_0ToF32 stays
  catastrophic on both**. **Dispatch disposition (Zen5) is therefore settled by timing alone:** the AVX-512
  quant paths are net-neutral-to-negative on Zen5; at minimum Q8_0ToF32 (and the <1.0× kernels) must NOT
  dispatch to AVX-512 on Zen5. The two remaining sub-causes for Q8_0ToF32 (fixable-codegen vs fundamental-Zen5
  512-bit-convert throughput) collapse to the SAME Zen5 action, so the disposition does not depend on Phase B.
- ✅ **AVX-512 Phase B (disasm) — RUN 2026-06-14 (checked clrjit built `build-checked-jit.cmd`).** Disasm at
  `.docs/avx512-phaseB-disasm.asm`. **Verdict: fundamental Zen5, NOT fixable codegen.** `Q8_0ToF32Avx512` and
  `Q8_0ToF32Avx2` are algorithmically identical; the ONLY difference is the widening width:
  - AVX2 (fast): `vpmovsxbd ymm, qword [mem]` — 8 int8→int32, ×4 per 32-block.
  - AVX-512 (8× slower): `vpmovsxbd zmm, xmmword [mem]` — 16 int8→int32, ×2 per 32-block.

  Codegen is textbook-optimal both ways (no spills/scalar/gather). The JIT's static **PerfScore rates the
  AVX-512 version CHEAPER (184 vs 272)** — it does not model that Zen5 runs the 512-bit `vpmovsxbd` byte→dword
  widening at drastically reduced throughput. So this is a hardware penalty, not a codegen bug; the kernels are
  correct (byte-exact 73/73) and would likely win on Intel AVX-512 (fast 512-bit widening). **Note for the
  runtime agent (cost-model nit, NOT a bug): JIT PerfScore underweights 512-bit `vpmovsxbd`/widening converts on
  Zen5 — would help auto-vectorization heuristics, but won't change this kernel's codegen since it explicitly
  requests Vector512.** This confirms the broader Zen5 lesson: wide-ISA integer/convert kernels don't win;
  only bf16 won (via instruction-collapse).
- ✅ **Dispatch disposition — DONE (user chose surgical, commit `9c9ae31`).** Gated by measured Zen5 result:
  reverted to AVX2 the net-negative kernels (Q8_0ToF32 0.12×, F32ToQ4_0 0.91×, RmsNormQ8_1 0.90×, RmsNormQ8_K
  ~0.96×); kept AVX-512 for the real Zen5 winners (Q4_0ToF32 1.17×, RmsNormQ8_0 1.20×). All Avx512 methods
  retained + still unit-tested (byte-exact) for a future Intel-gated revisit; intent documented at each
  dispatch site. Local kernel tests 51/0 (+6 AVX-512 byte-exact facts skip locally, run on Strix). No Strix
  re-validation needed — pure dispatch-branch removal; the methods are unchanged and were validated 73/73 on
  Strix in waves 1/2.
- ✅ **Decode knee multi-model sweep — RUN 2026-06-14 (net11+Core_Root, 32-core Strix). Overturns the
  investigation's central hypothesis.** All three models scale strongly PAST cap-8, peaking at 24–30T:

  | model | cap-8 (ctx128) | peak | peak@ | 32T (ctx128) |
  |---|---:|---:|---:|---:|
  | SmolLM2-135M | 240 | **466.7** | 30T | **16.5 (28× cliff)** |
  | Bielik-1.5B  | 29.3 | 44.6 | 24T | 37.7 |
  | Llama-3.2-1B | 35.9 | **67.2** | 30T | 59.5 |

  Findings: (1) **cap-8 leaves 1.5–1.95× decode on the table for EVERY model size** — small included. (2) The
  knee is NOT model-size-dependent the way hypothesized; ALL models want ~24–30T. (3) **The 32T point degrades
  for all (catastrophic for SmolLM ctx128: 30T healthy 466 → 32T 16.5)** — looks like OS oversubscription
  (no spare core at full-core-count), severity inversely tracking per-dispatch work. (4) **Footgun fix
  validated e2e** — pinning ≥ default everywhere (SmolLM 1.07×, Bielik 1.24×, Llama 1.01×), never the pre-fix
  ~0.4×. Raw output: Strix `c:\temp\decode-sweep\{smollm2,llama32,bielik}.txt`.
- ❌ **Option A (work-size-adaptive gate) — KILLED by the sweep.** Its premise (small models are dispatch-bound
  and want single-threading) is false e2e: SmolLM scales to 30T. Did NOT build it.
- ❌ **Option C (cache-line padding) — KILLED by the edge probe.** Mechanism is OS oversubscription, not
  contention (see below), so padding the dispatch counter has no target. Did NOT build it.
- ✅ **Edge discriminator RUN 2026-06-14** (`run-decode-edge.cmd`, SmolLM ctx128 {24,28,30,31,32} ×4 reps):
  the "28× cliff" was a SINGLE unlucky run — 32T usually does ~460 tok/s but *intermittently* stalls to 15.3
  (one of 4 reps). Medians are flat ~435–474 across 28–32T; the catastrophic tail is isolated to the full-core
  count = **intermittent OS oversubscription** (spin-wait workers on all cores starve OS/GC). Raw: Strix
  `c:\temp\decode-sweep\edge.txt`.
- ✅ **IMPLEMENTED & committed (`5f79bb7`): auto decode cap raised from 8 → `threadCount-2`** (leave OS
  headroom; floored at 2; topology-independent → also retires the footgun by construction). 14/14 pool tests
  pass (incl. a new auto-cap Theory). Clears the §5 decision rule (SmolLM 466@30T ≫ cap-8's 240). Caveat
  documented in the commit/code: validated on one 32-core single-node Zen5; ≥64-core / multi-socket untested.
- ✅ **Close-the-loop RUN 2026-06-14 (sequential per model; the new cap is live in the default path):**

  | model | old cap-8 default | new cap-30 default | pinned (numa / pcore) |
  |---|---:|---:|---:|
  | Llama-3.2-1B | 36.8 | **60.1 (+63%, ≈ peak 67)** | 60.3 / 64.4 |
  | Bielik-1.5B  | 22.6 | **42.7 (+89%, ≈ peak 44)** | 41.9 / 49.4 |
  | SmolLM-135M  | 253.7 | 127.7 ⚠ (unreliable single draw) | 346 / 288 |

  **Loop cleanly closed for the 1B-class models that matter** — default now lands at peak, corroborated by tight
  pinned/unpinned clusters (Llama 60/60/64 within 7%). The SmolLM 127.7 is the "16.5 cliff in reverse": one draw
  of a config already proven high-variance (4-rep edge median was 474 @30T vs 240 @8T), and its own three configs
  span 127/346/288 (2.7×) — self-evidently noise + pinning genuinely helping tiny models. Per advisor: do NOT
  back off the value (24T is not more stable — its worst draw 145.9 < 30T's worst 324.1; only full-core 32T has
  the catastrophic tail, which threadCount-2 already excludes). **Cap `5f79bb7` stands; comment softened so the
  headline doesn't overstate for the tiny-model non-pinned case.**
- ⏳ **Option C (cache-line padding + tree-reduction completion)** — NOT YET WRITTEN. Layout/primitive change to
  `ComputeThreadPool` hot fields; benefit only measurable via the dispatch microbench before/after on Strix, and
  it touches the delicate spin coordination — deliberately deferred to land alongside its proof rather than blind.

## Next Steps
1. ✅ **DONE:** `AvxVnni.V512` re-enabled on dev by runtime `67051208b96` (`V512VersionOfIsa` wiring); full
   parity 100/0/0 (all V512 guards forced) + consolidated single-runtime 4-way benchmark captured (see Done).
2. **Post the BF16 result to `kkokosa/dotLLM#322`** — draft ready at `.docs/issue-322-bf16-result-DRAFT.md`
   (clean single-runtime 4-way; BF16 ~2.3× faster than the best int kernel; correctness 100/0/0 + 8/8 BF16;
   accuracy caveat). User posts (outward-facing). Already appended to `.docs/net11-vnni-build-notes.md`.
3. ✅ **DONE — end-to-end accuracy + Llama benchmark:** SmolLM-135M perplexity A/B/C measured (bf16 +1.60% vs
   inner); Llama-3.2-1B output aligned (parity 3/3) and benchmarked (bf16 ~1.9× prefill). **Correction to earlier
   guidance:** the Llama benchmark *overturns the speed half* of "integer-outer is the safe default to ship
   first" — integer-outer is ~15% SLOWER than inner on net11 with no quality benefit, so shipping it alone is
   net-negative. **The value is the bf16 path, full stop; integer-outer is only the vehicle that enables it.**
   Remaining: Llama bf16 perplexity A/B/C (cost on the same model the speed was measured on); longer corpus
   before defaulting bf16 on.
4. Decide #322 disposition (keep open / pursue a realistic Q8_0-weights × bf16-activations mixed kernel).
5. **Runtime-side (tracked by the runtime agent, not dotLLM):** `67051208b96` is dev-local; the `AvxVnni.V512`
   ISA fix still needs landing on `feature/avxvnni.v512` (#128365, APPROVED+DIRTY — awaiting Tanner's
   fold-in-vs-follow-up call) and BF16 on PR #129326.
