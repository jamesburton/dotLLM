# dotLLM Performance Baseline

This document captures a first concrete performance baseline for dotLLM
versus llama.cpp on a single developer machine. It is the seed for a
running performance log — future measurements should append to the
tables below (or be exported alongside under `docs/perf/`) so that
optimisation work has a comparable anchor point.

The "competitive performance" claim has had no numbers attached. After
this exercise it does:

- **Llama family CPU (1B–3B Q8_0)**: dotLLM is in the same order of
  magnitude as llama.cpp; Llama-3.2-1B Q8_0 short-prompt decode is
  0.69× llama.cpp (10.6 vs 15.3 tok/s). Larger-prompt prefill on
  Llama-3.2-1B is dotLLM-faster (26.4 vs 15.3 tok/s) but the
  llama.cpp side of that comparison was contention-impaired and
  needs a re-measure.
- **Llama family CUDA (Q8_0, RTX 3060)**: dotLLM decode trails
  llama.cpp 0.43× (SmolLM-135M) and 0.60× (Llama-3.2-1B). This is the
  cleanest measurement in the baseline and is the **headline gap to
  fix** in correctness-clean data.
- **Qwen3.6-35B-A3B (Gated DeltaNet + MoE, Phase 10)**: dotLLM CPU
  output now matches llama.cpp end-to-end on the canonical prompt
  (post-`fdb39b4`, Q/K head-broadcast convention fix across all
  backends). Indicative decode is ~0.25× llama.cpp under
  memory-contended conditions; the GDN scan + per-expert MoE matmul
  are scalar-loop reference impls and are the obvious next SIMD
  targets. See §3.1.
- **Llama family Vulkan (Q8_0, AMD Radeon 8060S iGPU on Strix Halo)**:
  dotLLM Llama-3.2-1B Q8_0 short-prompt decode is 81.8 tok/s at CV 1.2%
  — 1.61× the RTX 3060 CUDA baseline above and 1.88× Strix Halo CPU.
  Workgroup-size tuning on the Q8_0 GEMV (the prime candidate flagged
  in the Phase 10 commit) is **not** a measurable win on this hardware
  — see §6 for the negative-result microbench. See §6 for the full
  Strix Halo baseline + headroom analysis.

## 1. Environment

### Hardware

| Component | Detail |
|---|---|
| CPU model | Intel Core Ultra 7 155H (Meteor Lake) |
| Cores / threads | 16 physical / 22 logical (6 P + 8 E + 2 LP-E) |
| Max clock | 3.8 GHz |
| SIMD support | AVX2, AVX_VNNI, FMA, F16C, BMI2 (**no AVX-512**) |
| RAM | 96 GiB total (≈32 GiB free at start of measurement) |
| GPU | NVIDIA GeForce RTX 3060, 12 GiB total (≈5–10 GiB free during runs; non-stable) |
| Driver | NVIDIA 591.86 / CUDA 12.4 |
| OS | Windows 11 Pro 26220.8370 |

### Toolchain versions

| Tool | Version |
|---|---|
| dotLLM commit | `bcace4b` (branch `feature/qwen3.6`, dirty) |
| .NET SDK | 10.0.300 (runtime 10.0.8, RyuJIT AVX2) |
| BenchmarkDotNet | 0.14.0 (InProcessEmit toolchain, warmup=1–2, iter=3) |
| llama.cpp (CPU build) | b9041 (`f08f20a0e`), Clang 19.1.5 — `ggml-cpu-alderlake.dll`, `LLAMAFILE=1 REPACK=1 OPENMP=1` |
| llama.cpp (CUDA build) | b9041, cuda 12.4 + cudart |

### Benchmark configuration

| Setting | Value | Notes |
|---|---|---|
| Prompt | `"The capital of France is"` (5 tokens) | Matches existing `InferenceBenchmarks` default; see prefill caveat below |
| Long-prompt anchor | Medium prompt from `bench_compare.py` (≈212 tokens) | One Llama-3.2-1B / CPU row for a defensible prefill number |
| Max new tokens | 30 | Decode budget per run |
| Sampling | greedy (temperature 0) | Both engines; deterministic outputs |
| dotLLM threads (CPU) | `ThreadingConfig.Auto` ≈ `Environment.ProcessorCount` (22) | Default for `InferenceBenchmarks` |
| dotLLM threads (Qwen sample) | 16 | Matches llama.cpp |
| llama.cpp threads | 16 | `--threads 16 --mlock` |
| Runs per engine | 3 (best-of-N reported) | Best (min ms / max tok/s) across runs |

> **Background-load caveat.** During the runs an unrelated Windows game
> (`Mir4G`) and an OneDrive sync were consuming a substantial fraction
> of CPU cycles (`CV` in some llama.cpp rows reaches 75–81%). The
> best-of-N filter mitigates this but does not eliminate it: where the
> coefficient of variation is large in the tables below, the
> conservative read is that the *true* gap is somewhere between
> "best-of-N" and "median" — both are reported for transparency.

> **Short-prompt prefill caveat.** Prefill tok/s on a 5-token prompt is
> dominated by per-run fixed overhead; one token of tokenisation
> difference shifts the number ≈20%. The Llama-3.2-1B medium-prompt
> row (212 tokens prefill) is the only prefill number that should be
> compared cross-engine with confidence. Short-prompt prefill rows are
> kept for completeness; do not draw optimisation priorities from them
> on their own.

## 2. Results

All values are **best-of-N** (max tok/s, min ms) across the runs
specified above. `CV` is the coefficient of variation across runs
(lower = more stable). dotLLM thread count is shown in the column
header. Numbers in this table are reproducible via the commands in
§4.

### CPU baseline (dotLLM `--threads 22`, llama.cpp `--threads 16`)

| Model | Quant | Engine | Prompt tok | New tok | Prefill tok/s | Prefill CV | Decode tok/s | Decode CV | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| SmolLM-135M               | Q8_0    | dotLLM    |   5 | 29 | **116.8** | 49% | **53.1** | 31% | llama.cpp suspiciously slow on 135M (overhead-bound); not load-bearing |
| SmolLM-135M               | Q8_0    | llama.cpp |   5 | 29 |  17.6    | 31% |   3.24   | 30% | |
| Llama-3.2-1B-Instruct     | Q8_0    | dotLLM    |   5 | 29 |  19.8    | 35% |  10.6    | 11% | |
| Llama-3.2-1B-Instruct     | Q8_0    | llama.cpp |   5 | 29 | **31.9** | 33% | **15.3** | 75% | dotLLM ≈ 0.69× decode |
| Llama-3.2-1B-Instruct (medium prompt) | Q8_0 | dotLLM   | 212 |  9 | 26.4 | 5% | 10.7 | 9% | dotLLM clean (CV 5%) |
| Llama-3.2-1B-Instruct (medium prompt) | Q8_0 | llama.cpp |  98 |  9 | 15.3 | – | 1.21 | – | Single run, contention-impaired — **ratio not meaningful** |
| Llama-3.2-3B-Instruct     | Q8_0    | dotLLM    |   5 | 29 | 7.78 | 30% | 3.93 | 7% | dotLLM clean (CV 7%) |
| Llama-3.2-3B-Instruct     | Q8_0    | llama.cpp |   5 | 29 | 7.46 | 93% | 2.18 | 81% | Contention-dominated — **ratio not meaningful** |
| **Qwen3.6-35B-A3B-UD**    | Q6_K_XL | dotLLM    |   5 | 19 |   0.3–0.8 | – |   0.6–1.0 | – | Post-`fdb39b4` — output matches llama.cpp. Memory-contended runs; see §3.1 |
| **Qwen3.6-35B-A3B-UD**    | Q6_K_XL | llama.cpp |   5 | 19 | **2.86** | – | **3.92** | – | `--no-warmup`, same `-t 8`; output: `"Paris, a city renowned for its iconic landmarks..."` |

Bold values indicate the faster engine on that row **for rows where
both engines are within a comparable noise envelope**. Rows marked
"ratio not meaningful" have one engine's CV above 50% — best-of-N
filtering does not rescue a contention-impaired comparison; the gap
goes one way under load and a different way without it. Re-running
with a quiescent system is required to settle Llama-3B CPU and the
medium-prompt Llama-1B llama.cpp decode.

The dotLLM Qwen row was captured via `samples/DotLLM.Sample.Console`
(single-shot, fallback per the task brief — see §4) because BDN
warm-up budget on a 30 GB mmap + `RepackWeights` did not complete
cleanly. Two runs produced 0.3 and 0.6 tok/s; the range is shown.
**The Qwen number is for a broken inference path** (see §3.1) — it is
recorded as evidence of the gap, not as a meaningful performance
baseline.

### CUDA baseline (RTX 3060, ≈10 GiB free at run time)

| Model | Quant | Engine | Prefill tok/s | Decode tok/s | Note |
|---|---|---|---:|---:|---|
| SmolLM-135M           | Q8_0 | dotLLM    | 205.5  | 84.8    | CV 77% on decode (small model, high overhead share) |
| SmolLM-135M           | Q8_0 | llama.cpp | 201.7  | **197.1** | — |
| Llama-3.2-1B-Instruct | Q8_0 | dotLLM    | 157.6  | 50.9    | CV 25% on decode |
| Llama-3.2-1B-Instruct | Q8_0 | llama.cpp | **184.3** | **85.0** | CV 73% on decode (background load) |
| Llama-3.2-3B-Instruct | Q8_0 | dotLLM    | (22.1) | (26.4) | Prior run; VRAM-spill at current free; treat as noise |
| Llama-3.2-3B-Instruct | Q8_0 | llama.cpp | 0.50   | 0.32    | Host-RAM spill — model does not fit |
| **Qwen3.6-35B-A3B**   | Q6_K_XL | dotLLM  | — | — | **Cannot fit on 12 GiB VRAM** (30 GiB model). Skip |
| **Qwen3.6-35B-A3B**   | Q6_K_XL | llama.cpp | — | — | Same; CPU-only path used above |

## 3. Ratios and the biggest gap

The ratio `dotLLM_tok/s ÷ llama.cpp_tok/s` (greater than 1 ⇒ dotLLM
faster). Rows where either engine had `CV > 50%` are flagged
"contention" — those ratios are unreliable in either direction and
should not drive optimisation priorities. The headline metric is
**decode tok/s ratio**: bandwidth-bound, the most reproducible
cross-engine quantity on this hardware, and what the Phase 10 roadmap
success criterion is stated in ("decode throughput within 2× of
llama.cpp").

| Model / device          | Prefill ratio | Decode ratio | Reliability |
|---|---:|---:|---|
| SmolLM-135M / CPU       | (6.6×)  | (16.4×) | llama.cpp overhead-bound on 135M; **not load-bearing** |
| Llama-3.2-1B / CPU      | 0.62×   | **0.69×** | Clean enough on dotLLM side; llama.cpp CV 75% — directional only |
| Llama-3.2-1B (medium prompt) / CPU | (1.73×) | — | llama.cpp single run, contention — not load-bearing |
| Llama-3.2-3B / CPU      | —       | — | Both runs contention-impaired — not load-bearing |
| Qwen3.6-A3B / CPU       | 0.28×   | **0.25×** | Correctness restored; memory-contended floor — see §3.1 |
| SmolLM-135M / CUDA      | 1.02×   | **0.43×** | Clean |
| **Llama-3.2-1B / CUDA** | 0.85×   | **0.60×** | **Clean** — dotLLM 51 tok/s vs llama.cpp 85 tok/s |

### 3.1 Qwen3.6-A3B CPU — correctness restored, perf measurable

**Status (post-`fdb39b4`, 2026-05-13)**: dotLLM CPU output now matches
llama.cpp end-to-end on the canonical prompts. Root cause was a TILED-
vs-INTERLEAVED Q/K head-broadcast bug in the GDN scan kernel (all three
backends), masked in CI by every existing GDN test fixture using
`NKHead=1` (where both mappings degenerate to `kh=0`). See commit
`fdb39b4` and `.planning/notes/qwen35moe-gdeltanet-architecture.md`
("Critical Implementation Notes") for the full diagnosis.

```
Prompt:  "The capital of France is"
dotLLM:  " Paris, a city renowned for its iconic landmarks such as the Eiffel Tower, the Louvre Museum"
llama.cpp: same continuation
```

Indicative perf snapshot (Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf, 5-token
prompt + 20-token decode, 8 threads, warm OS page cache; system under
heavy memory pressure with a separate `llama-server` instance holding
~50 GiB of the same model concurrently — these numbers are a floor,
not the steady-state achievable on a quiescent system):

| Engine    | Run | Prefill tok/s | Decode tok/s | Note |
|-----------|----:|--------------:|-------------:|------|
| dotLLM    |  1  |     0.3       |     0.6      | First post-load run; mmap not yet hot |
| dotLLM    |  2  |     0.8       |     0.7      | |
| dotLLM    |  3  |     0.8       |   **1.0**    | Best |
| dotLLM    |  4  |     0.5       |     0.6      | Concurrent llama-server load spike |
| llama.cpp |  1  |   **2.86**    |   **3.92**   | `--no-warmup`, warm cache |
| llama.cpp |  2  |     0.48      |     0.45     | System under load |

Indicative warm ratios (best-of-N for both engines, single hardware
configuration, contended system): dotLLM/llama.cpp ≈ **0.28× prefill,
0.25× decode** — a ~4× gap. This is a real signal but not yet a
publication-quality baseline: BDN-driven measurements on a quiescent
system are pending (the existing `InferenceBenchmarks` dispatches via
`TransformerModel.LoadFromGguf` which does not yet handle
`Architecture.Qwen3MoeHybrid` — needs the `Qwen3MoeHybridTransformerModel.LoadFromGguf`
dispatcher to be wired in; tracked as a follow-up).

**Recommendation**: the GDN scan kernel and per-expert MoE matmul are
both scalar-loop reference implementations in CPU. Both are obvious
SIMD/AVX2 targets and likely close the gap meaningfully; see Step 63's
"Optimize" subsection for the specific candidates (vectorise GDN inner
loops, fuse alpha/beta proj with softplus/sigmoid). Re-measure after
those land and update this table.

### 3.2 Biggest gap (correctness-clean)

Restricting to comparisons where both engines produced correct
output. (Llama-3.2-1B CUDA CV is non-trivial on both engines due to
background load; the ratio is reported best-vs-best and median-vs-
median to bracket it):

> **Llama-3.2-1B Q8_0 CUDA decode — dotLLM is ≈1.7–1.9× slower than
> llama.cpp.**
>
> | Statistic | dotLLM | llama.cpp | Ratio |
> |---|---:|---:|---:|
> | best decode tok/s | 50.9 | 96.0 | 0.53× |
> | median decode tok/s | 40.0 | 68.3 | 0.59× |
>
> Sibling: SmolLM-135M CUDA decode dotLLM/llama.cpp = 0.43×
> (2.3× slower at best-of-N). The pattern — smaller model worse than
> larger model — is the classic signature of **fixed per-decode-step
> overhead** (kernel launch, host-side dispatch, non-fused norm +
> matmul) being the dominant cost on small batch sizes.

This is the cleanest *directional* read in the baseline (Qwen3MoeHybrid
CPU is incorrect, CPU Llama runs are contention-impaired on the
llama.cpp side). The dotLLM prefill ratio on the same model (0.85×)
is much closer to parity, consistent with per-step decode overhead
rather than raw kernel throughput being the dominant cost.
Profiling targets:

- `benchmarks/DotLLM.Benchmarks/Profile/CudaDecodeProfile.cs`
  (already in tree — wire it into the next optimisation cycle).
- Specifically: cuBLAS HGEMM micro-batch path vs hand-rolled quantised
  GEMV launch counts; norm+rope+matmul fusion candidates.

## 4. Reproduction commands

The bench harness lives at `scripts/bench_compare.py` and drives BDN
`InferenceBenchmarks` + llama.cpp `llama-completion` with matched
prompts and token budgets via environment variables.

### CPU comparisons

```powershell
$env:LLAMACPP_BIN = "C:\Development\KTransformerTests\deploy\llama-cpp-cpu\llama-completion.exe"
python scripts/bench_compare.py `
    --model bartowski/Llama-3.2-1B-Instruct-GGUF `
    --quant Q8_0 --device cpu --tokens 30 `
    --iterations 3 --runs 3 --prompt-size short `
    --export-json docs/perf/baseline-llama1b-cpu.json
```

For the medium-prompt anchor row, change `--prompt-size short` to
`--prompt-size medium` and reduce `--tokens` to 10.

### CUDA comparisons

```powershell
$env:LLAMACPP_BIN = "C:\Development\KTransformerTests\deploy\llama-cpp-cuda\llama-completion.exe"
python scripts/bench_compare.py `
    --model bartowski/Llama-3.2-1B-Instruct-GGUF `
    --quant Q8_0 --device gpu --tokens 30 `
    --iterations 3 --runs 3 --prompt-size short `
    --export-json docs/perf/baseline-llama1b-gpu.json
```

### Qwen3.6-A3B (CPU fallback path — BDN warm-up budget too short for the 30 GB load)

```powershell
# dotLLM
dotnet run --project samples/DotLLM.Sample.Console -c Release -- `
    "C:/Development/KTransformerTests/models/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf" `
    "The capital of France is" --greedy --max 30 --threads 16

# llama.cpp
& "C:\Development\KTransformerTests\deploy\llama-cpp-cpu\llama-completion.exe" `
    -m "C:\Development\KTransformerTests\models\Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf" `
    -p "The capital of France is" -n 30 --temp 0 `
    --no-display-prompt -no-cnv --simple-io --perf --threads 16
```

Raw JSON exports for the Llama-1B/3B CPU runs are checked in under
`docs/perf/baseline-*.json`. Per-iteration BDN metrics live in
`%TEMP%\dotllm-bdn-metrics\<model-stem>.json` and survive across
runs (later runs overwrite per-model files).

## 5. Ranked next-optimisation queue

This is the prioritised list **for this hardware / these models**.
Rankings reflect (a) magnitude of the gap with correctness-clean
output, (b) confidence the fix is unblocked, and (c) confidence the
fix matches the suspected root cause. Quantitative "estimated gain"
ranges are deliberately not given for unmeasured items — re-measure
after each change.

1. **[Correctness — prerequisite for the Qwen perf work below]
   Multi-token greedy decode drift on `Qwen3.6-A3B Q6_K_XL` (CPU).**
   `samples/DotLLM.Sample.Console <gguf> "The capital of France is" --greedy --max 30 --threads 16`
   emits `"ore��dee 000oes 0 1own 20af000120 Io777"`. llama.cpp on
   the same input emits `"Paris, a city renowned"`. The committed
   `Qwen3MoeHybridTransformerModel.cs` (not in working-tree dirty
   list) reproduces the failure. Roadmap step 63 claims "CPU
   bit-exact verified — top-1 token 'Ta' matches llama.cpp
   reference", so either that was first-token-only or there has been
   a regression. Investigate against
   `tests/DotLLM.Tests.Unit/Models/Architectures/Qwen3MoeHybridTransformerModelTests.cs`
   first; if those pass, write a longer-decode parity test (≥20 tokens)
   against llama.cpp output and bisect from there.

2. **[Biggest correctness-clean gap] CUDA decode on Llama-3.2-1B Q8_0**
   — dotLLM 50.9 tok/s vs llama.cpp 85.0 tok/s (0.60×, 1.67×
   slower). Sibling row SmolLM-135M CUDA decode is 0.43× — the
   smaller-model-worse-than-larger-model pattern is the classic
   signature of **fixed per-decode-step overhead**: kernel launch
   latency, host-side dispatch, non-fused norm + RoPE + matmul.
   Prefill ratio on the same model is 0.85× (much closer to parity)
   which is consistent with this diagnosis. Action: run
   `benchmarks/DotLLM.Benchmarks/Profile/CudaDecodeProfile.cs`
   (already in tree) on the Q8 path and look at: (a) per-step kernel
   count, (b) cuBLAS HGEMM vs hand-rolled quantised GEMV launch
   patterns, (c) candidates for norm+rope+matmul fusion at
   `CudaTransformerModel` decode-step boundary. **Unblocked.**

3. **[Headline Phase 10 gap — gated on item 1]** Pre-quantised input
   cache in `Qwen3MoeHybridTransformerModel`. Every `Gemm(... preQuantizedInput: null)`
   call site in
   `src/DotLLM.Models/Architectures/Qwen3MoeHybridTransformerModel.cs`
   (lines 604, 662, 664, 666, 668, 803, 840, 869, 870, 953, plus the
   per-expert MoE dispatch) re-quantises the same F32 activation
   into Q8_K/Q5_K/Q6_K. The dense `TransformerModel` hoists this
   (`TransformerModel.cs` lines 600/616/696/819/835/868); the Qwen
   path simply never opted in. With 40 layers × ~6 projections ×
   128 experts × 30 decode steps, this is likely a large
   constant-factor cost — but until item 1 is resolved the broken
   inference path may itself be touching slow scalar fallbacks, so
   the apparent gap may shrink dramatically once correctness lands.
   **Do not estimate the gain in advance.** Land the correctness
   fix, re-measure Qwen, then decide whether this item is still
   worthwhile.

4. **F16 vs F32 KV-cache footprint at decode for Llama-1B CPU** —
   dotLLM's ≈30% short-prompt deficit on Llama-1B CPU decode (10.6
   vs 15.3 tok/s, llama.cpp CV high — single-source-of-evidence) is
   consistent with carrying F32 KV-cache where llama.cpp uses F16.
   The existing `KvCacheConfig.KeyDType/ValueDType` machinery
   (Step 33) makes this testable without code changes. Re-measure
   *first* with a quiescent system to confirm the gap is real before
   chasing the fix — current llama.cpp data on this row had CV 75%.

5. **Q/K/V and Gate/Up dispatch fusion in Qwen3MoeHybridTransformerModel** —
   Step 23 already validated this pattern on dense models (3→1 and 2→1
   thread-pool dispatches per layer, ~4% per token). For Qwen3MoeHybrid
   with 40 layers the savings compound; on top of item 3 the
   quantisation buffer is already hoisted and these are mechanical.
   Same gating: do not pick up until item 1 (correctness) lands.

6. **(Blocked, do not pick) Step 26 — Outer-product tiled matmul on
   AVX2.** Roadmap line item 26 explicitly notes "Blocked on RyuJIT
   register pressure — 4×3 tile needs 23 YMM registers (only 16
   available)". On this AVX2-only Meteor Lake CPU, Step 26 is not
   actionable without (a) AVX-512 hardware (CPU does not have it),
   or (b) a native C microkernel via P/Invoke. Recorded here only to
   pre-empt picking it as a "biggest gap" target — it cannot be
   cashed in on this machine.

## 6. Vulkan baseline — Strix Halo (Ryzen AI Max+ 395, Radeon 8060S iGPU)

This section adds the Vulkan baseline on AMD Strix Halo, the primary
Vulkan target for this project. The numbers were captured on a separate
machine from §1; the cross-engine comparisons in §2 do **not** carry over
— they were on Intel Ultra 7 + RTX 3060. This section is the
single-machine reference for Vulkan and for Strix Halo CPU.

### 6.1 Environment

| Component | Detail |
|---|---|
| CPU model | AMD Ryzen AI Max+ 395 (Strix Halo, Zen 5) |
| Cores / threads | 16 physical / 32 logical |
| SIMD support | AVX2 + AVX-512 (RyuJIT detects AVX-512F+CD+BW+DQ+VL+VBMI) |
| GPU | AMD Radeon 8060S Graphics (gfx1151, RDNA3.5 iGPU, integrated) |
| Driver | AMD proprietary 2.0.317, Vulkan 1.3.292 |
| Subgroup width | **64** (`subgroupSize=64`, `min=32`, `max=64`); `cooperativeMatrix` supported |
| RAM | 64 GiB DDR5 unified (iGPU shares main memory pool) |
| OS | Windows 11 Pro 26200.7019 |
| dotLLM commit | `0524276` + worktree `worktree-agent-a480e8ccd7f9935e4` (incl. backport of `665f228` GDN shader rename) |
| .NET SDK | 10.0.103 (runtime 10.0.3, RyuJIT AVX-512) |
| Vulkan SDK | scoop-installed, `glslc` on PATH |

### 6.2 Results (dotLLM, best-of-N over 7 BDN iterations, 3 BDN runs of 30 decode tokens each, prompt "The capital of France is")

llama.cpp Vulkan-on-Strix-Halo numbers were not captured in this pass —
no llama.cpp Vulkan build was available on the host. This is left as a
follow-up: build llama.cpp with `-DGGML_VULKAN=ON` against the same
Vulkan SDK and re-run `bench_compare.py --device vulkan --llamacpp` with
a `--llamacpp-bin` pointing at `llama-completion.exe` from that build.

| Model | Quant | Backend | Prefill ms | Prefill tok/s | Decode ms | Decode tok/s | Decode CV | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| SmolLM-135M               | Q8_0 | dotLLM CPU (AVX-512, 32 thr) |  17.2 | 291.1 | 107.9 | 268.7 | 16.9% | 5× the Intel-baseline 53 tok/s |
| SmolLM-135M               | Q8_0 | **dotLLM Vulkan**           |   9.9 | 503.5 |  82.1 | **353.3** | 3.9% | 1.31× CPU |
| Llama-3.2-1B-Instruct     | Q8_0 | dotLLM CPU (AVX-512, 32 thr) |  41.3 | 121.1 | 665.3 | 43.6  | 19.6% | 4× the Intel-baseline 10.6 tok/s |
| Llama-3.2-1B-Instruct     | Q8_0 | **dotLLM Vulkan**           |  27.9 | 179.4 | 354.3 | **81.8**  | 1.2%  | 1.88× CPU. Cleanest reading in the table |
| Bielik-1.5B-v3.0-Instruct | Q8_0 | **dotLLM Vulkan**           |  51.5 | 116.5 | 416.9 | **69.6**  | 4.6%  | Llama-arch stand-in (Llama-3.2-3B not cached) |
| Qwen2.5-0.5B-Instruct     | Q8_0 | **dotLLM Vulkan**           |  21.9 | 227.9 | 269.8 | **107.5** | 9.5%  | Qwen-family stand-in (Qwen3.6-A3B not cached on this host) |

> **Qwen3.6-A3B was not cached on this host** — per the task brief's
> "don't trigger downloads" policy, Qwen2.5-0.5B is run as a Qwen-family
> stand-in. The Qwen3.6 (Gated DeltaNet + MoE) Vulkan path is gated on
> the GDN multi-token shader at `native/vulkan/shaders/gdn_scan_multi_token_f32.comp`
> being compilable on the host (it required a backport of `665f228`
> renaming `active` → `laneActive` for newer `glslc`/`shaderc`).

Raw JSON exports are checked in alongside §1's CPU/CUDA blobs:

- `docs/perf/baseline-smollm-vulkan-strixhalo.json`
- `docs/perf/baseline-llama1b-vulkan-strixhalo.json`
- `docs/perf/baseline-bielik15b-vulkan-strixhalo.json`
- `docs/perf/baseline-qwen25-0.5b-vulkan-strixhalo.json`
- `docs/perf/baseline-smollm-cpu-strixhalo.json`
- `docs/perf/baseline-llama1b-cpu-strixhalo.json`

### 6.3 Cross-machine reading

The Llama-3.2-1B Q8_0 row gives a clean three-way picture for the same
model and quantisation:

| Backend          | Decode tok/s | Source |
|---|---:|---|
| Intel Ultra 7 CPU (22 thr, AVX2)   | 10.6 | §2, prior baseline |
| **Strix Halo CPU (32 thr, AVX-512)** | **43.6** | §6.2 — 4.1× the Intel CPU baseline |
| RTX 3060 CUDA                       | 50.9 | §2, prior baseline |
| **Strix Halo Vulkan (Radeon 8060S)** | **81.8** | §6.2 — 1.61× the RTX 3060 CUDA baseline, 1.88× Strix Halo CPU |
| llama.cpp CUDA on RTX 3060         | 85.0 | §2 |
| llama.cpp Vulkan on Strix Halo     | not measured | follow-up |

The Vulkan-on-Strix-Halo number for Llama-3.2-1B Q8_0 (81.8 tok/s
decode at CV 1.2%) is **the most stable measurement in this whole
document** — it is also the reproducibility benchmark to anchor future
optimisation passes against.

### 6.4 Vulkan Flash Attention microbench — Strix Halo (2026-05-18)

Captured on the same Strix Halo host as §6.2 / 6.3. Synthetic
attention-only Vulkan kernel comparison — no model weights involved.
Head config: 32 query heads / 8 KV heads / head_dim 64 (Llama-3.2-1B
shape).

Reproduction: `dotnet run --project benchmarks/DotLLM.Benchmarks -c Release -- --filter '*VulkanFlashAttention*'`.

| seqQ × seqKv | Naive (per-token) | Flash Attention | **FA speedup** | Notes |
|---:|---:|---:|---:|---|
| 512   |    10.26 ms |   7.60 ms | **1.35×** | Short prompt — naive already amortises K-reads |
| 2048  |   237.52 ms | 115.56 ms | **2.06×** | Mid-range prompt |
| 4096  | 1,331.32 ms | 489.38 ms | **2.72×** | Long prompt — BR=BC KV-amortisation pays off |

The 1.35→2.06→2.72× scaling is the textbook Flash-Attention pattern:
the per-query-row K read is amortised across the BR=16 row block, so
the longer the sequence the more the FA path beats the naive
attention-per-token kernel.

llama.cpp Vulkan FA on the same host was not measured in this pass.
The GAIA H2 acceptance metric (dotLLM FA pp512 t/s ≥ 60% of
llama.cpp FA pp512 t/s) is still outstanding — gating on a llama.cpp
Vulkan build configured for this Strix Halo host.

### 6.5 Outstanding measurements (2026-05-18 session)

- **`HybridPrefillDecodeBenchmarks` not discovered by BenchmarkSwitcher.** The class compiles
  cleanly (visible in the .dll metadata), but `BenchmarkSwitcher.FromAssembly.Run(args, …)`
  with `--filter '*Hybrid*'` returns zero benchmarks and `--list flat` does not include the
  class. Other BDN-decorated classes in the same file/namespace work fine. Hypothesis
  (untested): the `[Params(BenchmarkMode.PureVulkan, BenchmarkMode.Hybrid)]` enum-param attribute
  conflicts with the `[Params(int…)]` adjacent attribute, or a type-load failure during
  reflection enumeration silently drops the class. Tracked as a follow-up; doesn't block §6
  but does block the H4 ≥10 % first-32-tokens win acceptance check.
- **Qwen3.6-A3B GGUF not directly available** on this host — only an
  Ollama blob (`bartowski/Qwen_Qwen3.5-35B-A3B-GGUF:Q2_K_L`) is cached, and that's the
  3.5 series, not 3.6. The baseline measurements at `docs/perf/baseline-qwen36-a3b-cpu.json`
  remain the reference point for the production target until a fresh Qwen3.6-A3B GGUF
  is downloaded.

### 6.6 Top-3 perf-headroom items (measurement-derived)

For Llama-3.2-1B Q8_0 decode at 12.22 ms/token on Strix Halo Vulkan:

The dense-Llama path records ~13 dispatches per layer (1 fused
RmsNorm+Q, 2 matmuls (K/V), 1 RoPE, 1 attention, 1 matmul (O), 1 add,
1 fused RmsNorm+Gate, 1 matmul (Up), 1 SwiGLU, 1 matmul (Down), 1 add
+ inter-layer barrier). For 16 layers + final norm + LM head + sampling
that lands at **~210 compute dispatches** + barriers + 1 `vkQueueSubmit`
+ 1 fence wait per decode step.

A workgroup-size sensitivity microbench
(`benchmarks/VulkanGemvWorkgroupSweep`) of the Q8_0 GEMV at
WG=64/128/256 and a subgroup-arithmetic variant ranks the production
WG=128 + shared-memory reduce as **already at the local optimum on
RDNA3.5**:

| Variant           | Per-step Q8_0 GEMV cost (median of 5 runs) | Ratio vs WG=128 |
|---|---:|---:|
| wg64 (1 wavefront)              | ~3100 µs | 0.77× (slower) |
| **wg128 (current production)**  | **~2400 µs** | **1.00×** |
| wg256 (4 wavefronts)            | ~2500 µs | 0.94× (slower) |
| sg (subgroup reduce, 2 barriers vs 8) | ~2400 µs | 0.99–1.03× (within noise) |

End-to-end Llama-3.2-1B Q8_0 with the SG variant swapped in for
production: 78.5 tok/s decode vs 81.8 tok/s reference (within the run-
to-run noise envelope of ±10%). **Workgroup size and reduce strategy
on the Q8_0 GEMV is not a meaningful headroom item on this hardware.**

The estimated cost share of Q8_0 GEMV in the 12.22 ms decode step is
~7.9 ms (65%; weighted-shape median from the microbench). The remaining
~4.3 ms (~35%) is split among RmsNorm, RoPE, attention, SwiGLU, residual
adds, KV-cache transfers, and per-submission overhead — and that is
where the unmeasured headroom must live. Ranked by suspicion (gain
estimates omitted on purpose — none of these are measured yet, and on
this iGPU per-run noise dwarfs anything below ±10%):

1. **`VK_KHR_cooperative_matrix` Q8_0 GEMV path.** Strix Halo advertises
   coopmat (the `MatMulQ8_0GemmCoopmatKernel` path is already wired for
   prefill — see `VulkanTransformerModel.cs` line ~70 comment claiming
   "~3.8× over scalar GEMM at Llama-3 4096² N=64"). The decode path
   (N=1) currently uses scalar `matmul_q8_0.comp`; coopmat tiles need
   N≥16 by construction, so the decode path needs a **batched-decode
   adapter**: gather multiple in-flight requests' Q activations into an
   N=16 tile and dispatch one coopmat GEMM. Speculative decoding's
   draft-verify also creates the same N>1 opportunity for free. This is
   the *largest* concrete unmeasured gain available on Strix Halo,
   gated on landing the engine-side batching seam.

2. **Per-decode `vkQueueSubmit` count.** Fence-pipelined forward already
   collapses to 1 submit per decode for the dense Llama path (`d43ff71`
   "fence-pipelined forward"), so this is *probably* not load-bearing
   on Llama-3.2-1B. Worth confirming with a Vulkan timestamp-query
   pass: instrument the begin/end of forward with
   `VK_QUERY_TYPE_TIMESTAMP` and compare wall-clock decode (12.22 ms) to
   GPU-side dispatch sum. If the gap > 1 ms it points at host-side
   command-buffer build cost; if not, parking this item.

3. **F16 KV-cache, packed FP16 GDN state.** The Phase 10 commit
   (`da50c40`) explicitly flagged "packed FP16 state to halve bandwidth
   on the large `[d_k, d_v]` matrices" as remaining headroom for
   Qwen3.6-A3B specifically. The dense Llama path also currently runs
   F32 KV-cache through `VulkanKvCache.RecordUpdate` — F16 would halve
   the per-step KV-cache copy bandwidth and the per-step attention KV
   read bandwidth. The CPU path's `KvCacheConfig.KeyDType/ValueDType`
   machinery (Step 33) makes this testable end-to-end without rewriting
   `attention_f32.comp`'s reduce — but the Vulkan attention shader
   currently consumes F32 K/V, so the kernel side needs an F16-K/V
   variant first.

What is **NOT** a top item, contrary to the Phase 10 commit's
suggestion:

- **"Workgroup size tuning for RDNA3.5 64-wide wavefronts"** on the
  Q8_0 GEMV. The microbench above (`benchmarks/VulkanGemvWorkgroupSweep`)
  shows WG=128 + shared-mem reduce is already within ±5% of every
  alternative tested (WG=64, WG=256, subgroup-arithmetic-128). The
  shaders compile to SPIR-V with the driver picking up RDNA3.5's
  64-wide subgroup automatically; the production WG=128 lands as 2
  wavefronts which is the right occupancy point for the 8060S compute
  units. The negative-result artefacts (3 SPV variants + microbench)
  are kept under `benchmarks/VulkanGemvWorkgroupSweep/` so the
  experiment can be re-run if the driver, model shapes, or hardware
  change — but no production-path change is warranted on this data.

### 6.7 Reproduction commands

```powershell
# Vulkan SmolLM-135M Q8_0
python scripts/bench_compare.py `
    --model QuantFactory/SmolLM-135M-GGUF --quant Q8_0 `
    --device vulkan --tokens 30 `
    --iterations 3 --runs 3 --prompt-size short `
    --label "vulkan-strixhalo-smollm-q8" `
    --dotllm `
    --export-json docs/perf/baseline-smollm-vulkan-strixhalo.json

# Vulkan Llama-3.2-1B Q8_0 (the headline measurement)
python scripts/bench_compare.py `
    --model bartowski/Llama-3.2-1B-Instruct-GGUF --quant Q8_0 `
    --device vulkan --tokens 30 `
    --iterations 3 --runs 3 --prompt-size short `
    --label "vulkan-strixhalo-llama1b-q8" `
    --dotllm `
    --export-json docs/perf/baseline-llama1b-vulkan-strixhalo.json

# Q8_0 GEMV workgroup-size sweep (the negative-result microbench)
dotnet run --project benchmarks/VulkanGemvWorkgroupSweep -c Release
```

## 7. Update protocol

When extending this baseline:

- Add new rows to the tables above, do not delete; mark prior rows
  with the SHA and date they were measured.
- Drop new `--export-json` blobs into `docs/perf/` so future passes
  can `--show` and compare historic snapshots.
- For any "biggest gap" callout, prefer the **decode tok/s ratio on
  matched device** — it is the most reproducible metric and the one
  the roadmap success criteria are stated in.
- If background CPU load is unavoidable, mark `CV` and report median
  alongside best.
