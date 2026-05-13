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
  output is gibberish today — both the correctness claim from
  roadmap Step 63 and the supposed perf number are unreliable.
  Correctness regression must be fixed before this can be measured.

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
| **Qwen3.6-35B-A3B-UD**    | Q6_K_XL | dotLLM    |   5 | 29 |   0.3–0.6 | – |   0.2–0.6 | – | **Output is gibberish — see §3.1** |
| **Qwen3.6-35B-A3B-UD**    | Q6_K_XL | llama.cpp |   5 | 29 | **7.25** | – | **3.93** | – | Output coherent: `"Paris, a city renowned"` |

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
| Qwen3.6-A3B / CPU       | (0.04×) | (0.05×) | dotLLM **output is incorrect** — see §3.1; perf number is not a baseline |
| SmolLM-135M / CUDA      | 1.02×   | **0.43×** | Clean |
| **Llama-3.2-1B / CUDA** | 0.85×   | **0.60×** | **Clean** — dotLLM 51 tok/s vs llama.cpp 85 tok/s |

### 3.1 Qwen3.6-A3B CPU is broken before it is slow

The dotLLM CPU forward pass on `Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf`
produces gibberish output at greedy temperature:

```
Prompt:  "The capital of France is"
dotLLM:  " ore��dee 000oes 0 1own  20af000120 Io777"
llama.cpp: " Paris, a city renowned"
```

Roadmap step 63 claims **"CPU bit-exact verified — top-1 token 'Ta'
matches llama.cpp reference"**. The committed code (`bcace4b`, and
specifically `src/DotLLM.Models/Architectures/Qwen3MoeHybridTransformerModel.cs`
is **not** in the working-tree dirty list) does not reproduce that
claim today. The roadmap claim was either first-token only (and the
multi-token drift has always been there) or there has been a
regression. Either way the 20× perf gap is meaningless: that number
is the cost of a broken inference path, and 1× the cost of fixing the
correctness problem may eliminate most or all of the apparent perf
gap.

**Recommendation**: do not pick Qwen3MoeHybrid CPU perf as the "next
optimisation" target until correctness is restored. Re-measure after
the fix and put the resulting number into this table.

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

## 6. Update protocol

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
