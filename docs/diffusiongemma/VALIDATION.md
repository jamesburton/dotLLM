# DiffusionGemma — Validation & Benchmarking Methodology

How we validate the dotLLM DiffusionGemma implementation: which small models, how to obtain weights,
how to verify numerical correctness, how to compare capability, and how to measure throughput — with
a hard GPU-free guard because the dev box (Strix Halo iGPU) shares the GPU with concurrent work.

---

## 1. Chosen validation models

### Primary — `diffusionfamily/diffugpt-s`
<https://huggingface.co/diffusionfamily/diffugpt-s>
- **124M params**, **GPT-2 backbone** adapted to **absorbing-state masked diffusion**
  (DiffuLLaMA / "Scaling Diffusion Language Models via Adaptation from AR Models", arXiv 2410.17891).
- **safetensors (F16)**, **Apache-2.0**, GPT-2 BPE tokenizer.
- **Why:** smallest model that exercises the *core* DiffusionGemma ops — **bidirectional attention** +
  **iterative masked denoise** + **confidence/entropy unmasking** — and runs comfortably on CPU.
  The GPT-2 backbone reuses dotLLM's existing dense `TransformerModel`, isolating the diffusion seam.

### Fallback / architecturally-closest — `inclusionAI/LLaDA2.1-mini`
<https://huggingface.co/inclusionAI/LLaDA2.1-mini>
- `model_type=llada2_moe` — **MoE + masked diffusion**, safetensors, Apache-2.0.
- **Why:** the closest single open model to DiffusionGemma's **MoE + diffusion** combination. Use once
  the MoE diffusion path (issue 02 + 06) is in, to validate the MoE+diffusion interaction before the
  full 26B model. Heavier than DiffuGPT-S; not the first-line CPU harness.

### Ops-coverage matrix
| Op | DiffuGPT-S | LLaDA2.1-mini | DiffusionGemma-26B |
|---|---|---|---|
| Bidirectional attention | ✅ | ✅ | ✅ |
| Iterative masked denoise | ✅ | ✅ | ✅ |
| Confidence/entropy unmask | ✅ | ✅ | ✅ |
| MoE (top-k experts) | ❌ | ✅ | ✅ (128/top-8) |
| Gemma norms / GeGLU / softcap / QPAS | ❌ | ❌ | ✅ |
| Partial-rotary RoPE per attn-type | ❌ | ❌ | ✅ |
| Encoder-prefill + multi-canvas | partial | ✅ | ✅ |

Net: **DiffuGPT-S validates the diffusion seam**; **LLaDA2.1-mini adds MoE+diffusion**; the
**Gemma-specific numerics** are validated separately by the M1 synthetic Gemma forward tests + the
real DiffusionGemma config parse (issue 07). No single small model covers all ops — that is expected.

### Obtaining weights
```
huggingface-cli download diffusionfamily/diffugpt-s --local-dir ./models/diffugpt-s
# fallback:
huggingface-cli download inclusionAI/LLaDA2.1-mini --local-dir ./models/llada2.1-mini
```
The mask token id is **not** in `config.json` — read it from the downloaded `tokenizer_config.json` /
`special_tokens_map.json` (issue 05/07).

---

## 2. Numerical correctness

Prefer **CPU** (no GPU contention). Two levels:

1. **Single-forward canvas-logit parity.** Capture an HF-reference logit dump for a fixed
   `(prompt, canvas-mask-pattern)` from the reference modelling code, committed as a small fixture.
   Run the dotLLM bidirectional forward over the same input; compare:
   - cosine similarity per masked position ≥ **0.999** (tolerance documented per model),
   - max-abs-diff bounded (state the bound), F32-vs-F16 accumulation differences accounted for.
2. **Full-denoise decoded-text comparison.** Run the whole denoise loop (scheduler + unmask sampler)
   on a fixed prompt set with a fixed seed/temperature; compare decoded text to the HF reference
   within a stated edit-distance / token-overlap bound. Greedy (temp→0) for determinism.

Gemma numerics are validated independently by the M1 synthetic forward tests (GeGLU vs SwiGLU,
4-norm vs 2-norm, embed-scale on/off all discriminative) — these need no external weights.

---

## 3. Capability comparison

On a fixed prompt set drawn from the public DiffusionGemma demo tasks:
- **code generation** (cf. `diffusiongemma-codegen` space),
- **OCR / text correction** (cf. `diffusiongemma-ocr-correction` space),
- **short QA / instruction following**.

Measure coherence/quality of the dotLLM output vs the HF reference output of the **same** model
(exact-match where deterministic; otherwise a rubric/LLM-judge score). The goal is *parity with the
reference implementation of the same weights*, not absolute task SOTA.

---

## 4. Throughput metrics

Measured on the diffusion decode path and compared to an **AR baseline of comparable size**
(e.g. a small Gemma/Llama AR model) — the diffusion value proposition is parallel-canvas speedup at
fixed quality:
- **tokens/sec** (decoded canvas tokens / wall-clock),
- **denoise-steps/sec**,
- **canvas latency** (time to fully denoise one 256-token canvas),
- **effective tokens/sec vs AR** at matched model size **and matched output quality**,
- adaptive-stop step count distribution (expect 12–16 vs the 48 cap).

Context for sanity-checking (not a target on this box): Google reports >1000 tok/s on one H100, ~4× AR.
We report Strix-Halo-relative numbers.

---

## 5. GPU-free guard (mandatory before any GPU-heavy run)

The dev box **is** the Strix Halo target and the AMD iGPU is shared with concurrent work. **CPU
correctness jobs always run. Any GPU-heavy validation/benchmark job MUST first confirm the iGPU is
idle and abort with a clear skip message if not.**

### Concrete check (Windows / Strix Halo, AMD iGPU)
Preferred — AMD SMI (ships with the ROCm/AMD stack), gate on GPU utilization:
```powershell
# Returns nonzero busy% if the iGPU is in use; treat >5% as "busy".
$busy = (& amd-smi metric --usage --csv 2>$null | Select-Object -Skip 1 |
         ForEach-Object { ($_ -split ',')[1] } | Measure-Object -Maximum).Maximum
if ([int]$busy -gt 5) { Write-Host "GPU busy ($busy%); skipping GPU benchmark."; exit 0 }
```
Fallbacks if `amd-smi` is unavailable on this image:
```powershell
# (a) Windows perf counter for GPU engine utilization (works for the AMD iGPU):
$gpu = (Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples |
        Measure-Object -Property CookedValue -Maximum
if ($gpu.Maximum -gt 5) { Write-Host "GPU busy; skipping."; exit 0 }

# (b) Coarse VRAM-pressure check via the AMD control stack, or simply assert no other
#     dotLLM/bench process holds the device (Get-Process) before proceeding.
```
The benchmark harness (issue 11) embeds this guard: the GPU jobs are no-ops when the check fails;
the CPU correctness + capability jobs run unconditionally. **Do not run any heavy GPU validation by
hand without running the guard first.**

### Policy
- Correctness (logit parity, decoded-text, Gemma numerics): **CPU, always**.
- Throughput / GPU kernel benchmarks: **only after the guard passes**.
- This research+planning task itself ran **no** GPU operations.
