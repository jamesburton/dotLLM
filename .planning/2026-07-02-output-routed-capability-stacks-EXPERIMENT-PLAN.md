# Output-Routed Capability Stacks — Scaled Experiment Plan

**Date:** 2026-07-02
**Companion:** `2026-07-02-output-routed-capability-stacks-design.md`
**Status:** Planning — Step 0 confirmed (JA 74.4→61.7, output-layer bottleneck confirmed); 3060
  stalled at step ~1180 due to lm_head gradient ~12 GB; needs ≥16 GB to continue.
**Author:** James Burton (with Claude)

---

## Hardware

### Why the 3060 cannot finish these experiments

A 128k-vocab `lm_head` in bf16 (256 MB weights) with grad-checkpoint and AdamW optimizer state
consumes ~12 GB on the RTX 3060 12 GB at the batch sizes needed for stable training. All three
experiments below require training the `lm_head` continuously. **Minimum VRAM: ≥16 GB.**

### Recommended platforms (in priority order)

| Platform | VRAM | Notes |
|----------|------|-------|
| Kaggle P100 16 GB | 16 GB single | Just fits lm_head training at bs=2; no NVLink overhead; free |
| Kaggle T4×2 | 2×15 GB, tensor-split | The harness already supports teacher-split (`kaggle/` scripts); teacher on GPU 0, student on GPU 1 needed for Exp C |
| Cloud 24 GB (A10G, RTX 3090, RTX 4090) | 24 GB | Most headroom; needed for Exp B (K stacks simultaneously) and Exp C (KD with strong JA teacher) |
| Strix Halo | 128 GB unified | 128 GB unified memory fits everything in CPU pinned memory; ROCm/Vulkan only — PyTorch CUDA training **unproven** — do not assume |

**Default recommendation for Exp A:** Kaggle P100 (free, single-GPU, no split complexity).
**Default recommendation for Exp B and C:** Kaggle T4×2 (teacher-split harness already exists) or
  a 24 GB cloud GPU.
**Strix:** Confirm ROCm PyTorch training is viable *before* committing a multi-day run. A 30-minute
  smoke test (10 steps, log GPU utilisation) is the gate.

---

## Experiment A — Complete the JA confirmation cleanly

**Goal:** Run `lm_head` + final-block training on EN+JA to full convergence (not stalled at step
~1180). Establish how far JA PPL drops and whether EN holds.

### Setup

```bash
# Existing script — just needs bigger GPU and more steps
python mote_train.py \
  --train-lm-head \
  --grad-checkpoint \
  --train-final-block \
  --data mix_en_ja \
  --steps 5000 \
  --eval-every 500 \
  --save-every 500
```

Add `--eval-en` to `mote_eval_ja.py` so EN and JA PPL are logged at every checkpoint.

### Data

- JA: `wikimedia/wikipedia 20231101.ja` (existing dataset path in `capability_data.py`)
- EN: `no_robots` + existing chat data
- Mix ratio: 50/50 by token, interleaved per batch (not alternating epochs)

### Success criteria

| Metric | Target | Hard stop |
|--------|--------|-----------|
| JA PPL at convergence | < 30 | < 50 (partial win) |
| EN PPL at convergence | ≤ 15.5 (~+15% over base 13.44) | > 18 (regression) |
| JA PPL trend at step 5000 | Flattened or falling | Still falling (needs more steps) |

JA < 30 would mean parity with German (18.42) is plausible with more data/steps. JA < 50 at
minimum shows the lm_head unlock is real and scalable. EN ≤ 15.5 confirms the output stack
specialization plan is safe for EN capability.

### What this proves

That the 3060 step-0 result (74.4→61.7 at step 1000, still falling) extrapolates to a genuine
new-language capability, not a training artefact. Sets the "unconstrained baseline" for Exp B
to beat.

### Estimated cost / time

- Kaggle P100 16 GB: ~4–6 hours for 5000 steps at bs=2, seq=512.
- Token budget: ~5M EN + 5M JA = 10M total.

---

## Experiment B — Per-type output stacks with format-prefix commit routing

**Goal:** Validate the core architectural idea: K dedicated output stacks (final FFN + RMSNorm +
`lm_head`), each trained on type-pure data, selected by a format-prefix commit router. Does each
type improve vs. the shared baseline? Does EN/general hold? Does the router pick the right stack?

### Architecture

```
Frozen BitNet-2B backbone (layers 0–28)
  │  hidden state h
  ├── Stack 0: FFN_0 + Norm_0 + lm_head_0  ← prose / EN general
  ├── Stack 1: FFN_1 + Norm_1 + lm_head_1  ← tool-call / JSON
  └── Stack 2: FFN_2 + Norm_2 + lm_head_2  ← Japanese
```

Router: lightweight MLP on h at the format-prefix commit token (`<tool_call>`, `{`, first JA
Unicode char, or the default prose path at sequence start).

### Phase B1 — Train stacks independently (no router yet)

Train each stack on type-pure data, backbone frozen. Evaluate per-type PPL without routing (always
route to the correct stack during training eval). This isolates stack quality from router quality.

```bash
# Stack 1: tool-call
python mote_train.py \
  --output-stack tool \
  --train-lm-head --train-final-block \
  --data hermes_function_calling \
  --steps 3000 --eval-every 500

# Stack 2: Japanese
python mote_train.py \
  --output-stack ja \
  --train-lm-head --train-final-block \
  --data wikimedia_ja \
  --steps 5000 --eval-every 500

# Stack 0: prose (EN general; also the default stack)
python mote_train.py \
  --output-stack prose \
  --train-lm-head --train-final-block \
  --data no_robots \
  --steps 3000 --eval-every 500
```

Data sources (all exist in `capability_data.py`):
- Prose: `no_robots`
- Tool-call: `teknium/OpenHermes-2.5` (hermes-function-calling subset)
- JA: `wikimedia/wikipedia 20231101.ja`

### Phase B2 — Add router, joint fine-tune

Train router MLP on labeled-mixture data. The format-prefix token label (prose / tool / ja) is
derived from the sequence itself (first `{`, `<tool_call>`, or Unicode CJK Unified Ideograph range
start). Fine-tune router for 1000 steps on the labeled mixture; keep stacks frozen.

### Metrics

| Metric | Measured by |
|--------|-------------|
| Per-type PPL (prose, tool-call, JA) | `mote_eval_ja.py` extended with `--type` flag |
| Routing accuracy | Fraction of sequences where committed stack matches gold label |
| EN PPL regression | Prose stack vs. dense baseline |
| Latency added by router | Wall-clock at format-prefix token position; target < 2 ms |

### Success criteria

- Each type-specific stack PPL ≤ Exp A unconstrained baseline for that type.
- EN prose PPL ≤ 15.5 (prose stack protects EN; no shared-head forgetting).
- Router accuracy ≥ 95% on held-out labeled sequences.
- JA PPL from JA stack < 50 at stack convergence.

### What this proves

That per-type lm_heads prevent the EN regression that a single shared lm_head causes (Exp A:
EN rises to ~14.6). The hypothesis is that dedicated stacks bound the forgetting because each
stack is only updated on its own type — proving this is the primary scientific contribution of
this experiment.

### Hardware

- Phase B1 stacks: P100/T4 per stack (three parallel Kaggle runs, or sequential on one).
- Phase B2 router: T4×2 (can hold all three stacks + router in 30 GB combined; or a 24 GB single GPU).
- JA stack training: same as Exp A (P100 or T4; ≥16 GB).

### Estimated cost / time

- B1: 3 Kaggle runs × 3–6 hours = 9–18 GPU-hours.
- B2: 1–2 hours (router only, stacks frozen).
- Token budget: ~5M per type (15M total B1) + ~2M labeled mixture (B2).

---

## Experiment C — bf16 stacks with KD, then ternary compression

**Goal:** Validate the full training recipe from §3 of the design spec:
(1) train stacks in bf16 with KD from the frozen base teacher,
(2) compress bf16 → ternary (1.58-bit) using STE + lambda warmup + SubLN,
(3) check whether added capability survives ternary compression.

### Phase C1 — bf16 stacks with KD from resident teacher

The BitNet-2B base in bf16 is the teacher. It is always resident for inference anyway, so no
extra model load is needed. KD objective: KL-divergence on output logit distributions.

```bash
python mote_train.py \
  --output-stack ja \
  --train-lm-head --train-final-block \
  --kd-weight 0.5 \              # KL(student || teacher) weight
  --kd-teacher base_bitnet_bf16 \
  --data wikimedia_ja \
  --steps 5000 --eval-every 500
```

For the JA stack, the base BitNet teacher is a weak JA teacher (it doesn't know Japanese well).
To get a strong JA teacher signal, use a JA-capable 7B (e.g. `Qwen2.5-7B-Instruct` or
`llama-3-8B-swallow-instruct`) on the second T4 (teacher on GPU 0 in bf16, student + base on
GPU 1). The T4×2 teacher-split is already implemented in the `kaggle/` harness; adapt it for
the 7B teacher.

For prose and tool stacks, the BitNet base is a strong teacher — single T4 (16 GB) is sufficient.

### Phase C2 — ternary compression

Apply after C1 bf16 converges. This is the most uncertain step.

```bash
python mote_train.py \
  --output-stack ja \
  --load-checkpoint ja_stack_bf16_5000.pt \
  --ternary-qat \               # STE + weight clipping
  --lambda-warmup 0.0 0.1 500 \ # linear ramp from 0 to 0.1 over 500 steps
  --sublayer-rms-norm \         # extra RMSNorm before each linear (arXiv:2505.08823)
  --kd-teacher ja_stack_bf16 \  # teacher is now the phase-C1 bf16 stack
  --steps 2000 --eval-every 200
```

Monitor: if JA PPL rises by > 10 PPL points vs. the bf16 stack during phase C2, stop and
report — the ternary compression is not preserving the JA capability and is not worth pursuing
further at this training scale.

### Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| C1 (bf16 + KD) | JA PPL vs. no-KD stack (Exp B) | ≤ Exp B stack PPL (KD should not hurt) |
| C1 (bf16 + strong JA teacher) | JA PPL | < 40 |
| C2 (ternary) | JA PPL vs. C1 bf16 | < +10 PPL degradation |
| C2 (ternary) | Stack weight size | ~8× smaller than bf16 (1.58-bit vs. 16-bit) |

### What this proves

Whether the capability-stack training recipe (bf16 warm-up → ternary compression with KD) can
produce a compressed output stack that retains the new capability. This is the §6 Q2 open
question: "does ternary compression preserve added capability?" If yes, the full architecture
(frozen ternary backbone + ternary output stacks) fits in ~600 MB total, within the 3060
inference envelope.

### Hardware

- C1 prose/tool: single T4 16 GB.
- C1 JA with strong 7B teacher: T4×2 (teacher 7B on GPU 0 in bf16 ~14 GB, student + BitNet base on GPU 1 ~12 GB).
- C2 ternary QAT: same hardware as C1 (stacks are same size as C1 student).

### Estimated cost / time

- C1: 5000 steps per stack × 2–4 hours = 6–12 GPU-hours for all 3 stacks.
- C1 JA (strong teacher, T4×2): 6–8 hours.
- C2: 2000 steps × 1–2 hours per stack.
- Total: ~20–30 GPU-hours on Kaggle free tier (multiple sequential Kaggle notebooks).

---

## Novel Arm (optional) — Diffusion-trained JA stack

**Goal:** If Exp B/C JA PPL does not reach < 30, probe whether a masked-diffusion training
objective for the JA stack helps. Motivation: `arXiv:2507.15857` shows diffusion beats AR in
data-constrained regimes because masked-token randomization acts as data augmentation.
JA is exactly such a regime (limited high-quality JA data relative to compute budget).

### Approach

Replace the AR cross-entropy objective on the JA stack with a masked-diffusion (MDLM-style)
objective:

1. Randomly mask 0–90% of JA tokens in the training sequence.
2. The JA stack learns to denoise (reconstruct) masked tokens from the frozen backbone's hidden
   state on the unmasked context.
3. Evaluate on the same JA held-out set using AR PPL for comparability (run a separate AR-mode
   eval pass; the stack can do both since the backbone is AR).

This is novel and speculative (no precedent for diffusion-objective training on top of a frozen
AR backbone). Treat as a research spike, not a production path. Run only if Exp B JA PPL > 50
and time/compute budget allows.

Script: new `mote_train_diffusion.py` modelled on `mote_train.py` with a `--diffusion-mask-rate`
flag and MDLM-style loss.

---

## Experiment Sequence and Dependencies

```
Exp A  (P100, ~5k steps)
  │  confirms convergence trajectory + EN headroom
  ▼
Exp B1 (T4 or P100, 3 parallel stacks)
  │  stacks trained independently
  ▼
Exp B2 (T4×2 or 24 GB, router fine-tune)
  │  router + stacks validated together
  ▼
Exp C1 (T4 or T4×2 for JA, bf16 + KD)
  ▼
Exp C2 (T4, ternary QAT)
  ▼ (optional, only if JA PPL > 50 after C2)
Diffusion arm (new script, 1–2 Kaggle runs)
```

Exp A is independent and should run first — it validates the scale of the effect and gives a
clean reference number for Exp B to beat. Exp B1 stacks can run in parallel across Kaggle
notebooks. Exp C depends on B having converged.

---

## Quick-Reference: Scripts and Flags

| Script | Key flags | Used in |
|--------|-----------|---------|
| `mote_train.py` | `--train-lm-head`, `--grad-checkpoint`, `--train-final-block` | All experiments |
| `mote_train.py` | `--output-stack {prose,tool,ja}`, `--kd-weight`, `--kd-teacher` | B, C |
| `mote_train.py` | `--ternary-qat`, `--lambda-warmup`, `--sublayer-rms-norm` | C2 |
| `mote_eval_ja.py` | `--eval-en`, `--type` | A, B, C eval |
| `capability_data.py` | data source definitions | All |
| `kaggle/` harness | teacher-split T4×2 setup | C1 JA |
| `mote_train_diffusion.py` | `--diffusion-mask-rate` (new) | Diffusion arm |

---

## What Each Experiment Proves (Summary)

| Exp | Core question answered |
|-----|----------------------|
| A | How far does JA drop with unlimited lm_head training? Does EN hold at scale? Sets convergence target. |
| B | Do per-type dedicated stacks prevent EN forgetting while improving per-type PPL? Does the format-prefix router route correctly? |
| C | Does the bf16 → ternary compression recipe preserve the capability gain? Is the final architecture viable in < 1 GB? |
| Diffusion arm | Does a masked-diffusion objective help for the hardest (JA) data-constrained stack? |
