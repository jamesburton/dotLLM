# Capability-focused ternary distillation — design

**Date:** 2026-07-12. **Goal:** distill dotLLM's *target* capabilities — C# code+CoT, tool-use, instruction-following — into a ternary (1.58-bit) student that retains them, and test whether **model size** is the blocker. Supersedes the general-text GSM8K probe (which gave PPL 67→48 but **GSM8K 0%**).

## 1. Why the first probe failed (research)

The general-text logit-KD probe (fineweb-edu, τ=1, 5M tok) improved PPL but retained **zero** reasoning. The literature says this was over-determined:

- **Quantization damages reasoning ≫ perplexity.** Low-bit quant elevates method/execution errors that cascade from the first flipped step; PPL barely moves. ([2504.04823](https://hf.co/papers/2504.04823), [2505.11574](https://arxiv.org/html/2505.11574v4)) — our PPL≠capability, exactly.
- **Reasoning/capability transfers only from capability data.** "Reasoning-domain data is key… general text produces substantially lower accuracy"; vanilla logit-KD without rationales does not transfer reasoning. ([2601.14888](https://arxiv.org/html/2601.14888v1), [2511.05184](https://arxiv.org/pdf/2511.05184), [2212.08410](https://hf.co/papers/2212.08410)) — **we used general web text → the primary fixable cause.**
- **Recipe that works:** GPTQ-style init > RTN; **KD > SFT** (limits drop ~9% vs ~21%); RL (GRPO) only after a KD cold-start; recovery is strong at 2–3 bit but **1.58-bit ternary is more aggressive than anything these papers recover.** ([2601.14888](https://arxiv.org/html/2601.14888v1), BitDistiller [2402.10631](https://hf.co/papers/2402.10631))

**Implication:** fix the *data/recipe* first (biggest lever, fits the 3060), test *size* second — a bigger model on the wrong recipe would also likely score ~0% and confound the experiment. User chose **both**.

## 2. Reuse map (what already exists in `scripts/lora/`)

| Capability | Data (reuse) | Eval (reuse) | Gap to build |
|---|---|---|---|
| **Tool-use** | `hermes-function-calling-v1/glaive_func_calling`, `tooluse_render.py` (chat-template parity), split `train[:3000]`/`[3000:]` | `eval_tooluse.py` name/args scoring (shells to dotLLM) | **PyTorch-side** in-process eval for frequent checkpoints |
| **Instruction** | `HuggingFaceH4/no_robots` (train/test) + formatter | `eval_instruction.py` 12 constraint probes (pairwise judge) | Auto-scorable subset + PyTorch-side eval |
| **C# code+CoT** | 2,852 `.cs` files + 8 API refs (`csharp_data.py`) — **raw source only** | **PPL headroom only** (`EVAL_mote.md` C#=4.22) | **No CoT/instruction pairs; no execution/pass@1 eval.** Biggest gap |
| **Reasoning (bonus)** | `openai/gsm8k` via `bitdistill_data.py` | `eval_gsm8k()` (4-shot CoT, in-process) ✅ | — |

Core harness (`bitdistill.py`) with the **offline top-K logit-cache** (τ=1, 93–97% mass, unit-validated) already exists and is the training substrate. `capability_data.py` already yields fixed-length token seqs for tooluse/instruction (used by MoTE probes) → feeds cache-gen directly. Prior warm-start-QAT survey: `.planning/…ternary-conversion-research.md` (LSQ/PACT/Reasoning-QAT PTQ-init).

## 3. Data design

**Corpus = capability data rendered via chat template** (train==serve parity), teacher FP logits cached offline, student trained on CE(completion-masked) + sparse top-K KD.

**Core-vs-blend (research-guided default):** heavily weight capability, keep a general anchor to survive ternarization without catastrophic forgetting.
- **~75% capability / ~25% general (fineweb-edu).**
- Within capability, default **equal thirds** across tool-use / instruction / C# — reweight by priority (user decision).

**Curriculum / cycles:** start **interleaved** (round-robin batches across the 3 capabilities + general) — simplest, avoids forgetting. If a target plateaus, add a **2-phase curriculum**: Phase A (~15% tokens) general+broad-instruction to stabilize post-ternarization, Phase B capability-heavy. (Curriculum is a secondary lever — [CoT Curriculum Distillation]; don't over-engineer up front.)

**Holdouts (strict test never trained/validated on):**
- Tool-use: existing `train[:3000]` / val+test from `[3000:]` (split into val/test).
- Instruction: `no_robots` train / test (carve val from train tail).
- C#: family-stratified 80/10/10 over the 2,852 files (hold whole files out, not lines).
- Report val each checkpoint (selection), test once at the end.

## 4. Eval design — frequent, multi-target, in-process

**Cadence:** every **~0.5M tokens** (≈10 checkpoints over 5M) — for visibility + interruptibility (user ask). Milestones list → `0.5e6,1e6,…,5e6`.

**All evals PyTorch-side (in-process `model.generate`)** so they're fast (batched, use_cache toggled) and need no dotLLM export mid-run:
- **Tool-use:** N held-out glaive prompts → parse `<tool_call>`, score name-match + args-JSON-equal (reuse `eval_tooluse.py` scoring logic, generate in-process).
- **Instruction:** auto-checkable constraint subset (length/format/keyword rules) — deterministic, no judge; full judge eval only at the end.
- **C#:** short-term **PPL on held-out C#** (have it) as a proxy; real signal needs a **compile+xUnit pass-rate** eval (build item B2).
- **GSM8K:** existing 4-shot CoT (bonus reasoning sanity).
- Baselines to plot against: **FP teacher** per capability (e.g. FP-1.7B GSM8K 4-shot = 70%) and the **base ternary** (no capability distill).

## 5. Recipe knobs

- **τ=1 offline top-K KD** (validated, 93–97% mass). Keep.
- **Completion-masked CE** on capability turns (reuse `masking.py` idea) so the student learns to *produce* tool-calls/CoT, not just model tokens.
- **Warm-start init:** research says GPTQ-init > STE/RTN. Optional upgrade (more work) — start with current STE, add if capability stalls.
- **Optimizer:** adafactor+grad-ckpt (fits 1.7B on 12 GB). `--save-ckpt` the winning config (so it can export → dotLLM and run the full serving evals + constrained-tool rescue).
- **RL (GRPO):** out of scope for v1; possible follow-on after KD cold-start.

## 6. Execution phases

- **P0 — plumbing:** B1 PyTorch-side tool-use + instruction evals; B4 multi-corpus cache + mixing/curriculum in `bitdistill_data`; frequent-eval milestones; `--save-ckpt`. (No GPU beyond quick smokes.)
- **P1 — 1.7B recipe-fixed (3060):** offline-KD on the capability mix, frequent multi-target evals. Isolates *"does the recipe fix it at 1.7B?"* vs the current 0%.
- **P2 — larger model (venue):** 4B same recipe on Strix (ROCm fix) or cloud (4B ternary *student* won't fit 12 GB).
- **C# track (parallel):** B3 synthesize C# instruction+CoT (teacher/stronger-model CoT over `OllamaBenchmarks` coding_tasks specs + API refs; or filter a multilingual code-instruct set — [MSCoT 2504.10178](https://arxiv.org/pdf/2504.10178)); B2 dotnet-build+xUnit execution eval. Start P1 with tool-use+instruction ready + C# as CPT/PPL, fold in C# capability once built.

## 7. Open decisions (for the user)

1. **Capability priority weighting** — equal thirds, or weight toward one (tool-use? C#?).
2. **C# now vs later** — synthesize C# CoT data + build execution eval up front (slower, real C# capability), or run P1 with tool-use+instruction first and C# as PPL-only, add C# capability next.
3. **Warm-start** — add GPTQ-style init now, or STE-first and add if it stalls.
4. **Eval cadence** — every 0.5M (10 evals) acceptable given per-eval cost, or coarser (every 1M)?
5. **General-anchor ratio** — 25% default OK, or leaner (more capability) / richer (more retention)?

## DECISIONS LOCKED (2026-07-12)
- Weighting: toward tool-use + C# (proposed tool-use 35 / C# 35 / instruction 20 / general 10).
- C# FIRST: build C# instruction(+CoT) data + dotnet-build/xUnit execution eval before P1.
- Warm-start: GPTQ-style init now (research: > STE/RTN).
- Eval cadence ~0.5M (10 checkpoints); general-anchor 10-25% (tune on forgetting).
- Build order: (1) C# data, (2) C# exec eval, (3) GPTQ warm-start init, (4) PyTorch tool-use+instruction evals, (5) multi-corpus weighted mixing + --save-ckpt, (6) P1 1.7B, (7) P2 larger.

## KEY REUSE FINDINGS (2026-07-12)
- **C# EXECUTION EVAL = REUSE `OllamaBenchmarks/scripts/coding_tasks/task_runner.py`**: 20 task specs (tasks/*.yaml), each with xUnit `test_code` + `references` + .NET project `template`. task_runner does `dotnet restore`/build/test -> (build_success, tests_passed, tests_total). Wire: student generates C# for a task prompt -> task_runner -> pass@1. Templates: test_project, blazor_project, layer2_project. These 20 are HELD-OUT eval only.
- **C# TRAINING DATA (B3) sources**: OpenCodeReasoning (NVIDIA, 735K, w/ reasoning) or OpenCodeInstruct (5M) FILTERED to C#; C#-specific recipe = Safurai Evol-Instruct synthetic Q&A (arXiv 2311.03243, 56% pass@1 CodeLlama-34B). Optionally synthesize more via a strong local model (ollama) verified by task_runner compile.
- Tool-use TRAIN = glaive (ready); Instruction TRAIN = no_robots (ready); C# TRAIN = OpenCodeReasoning-C# (source) + optional synth.
