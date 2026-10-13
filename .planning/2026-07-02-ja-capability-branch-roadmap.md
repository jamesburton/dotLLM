# JA / Output-Routed Capability — Branch Roadmap

**Date:** 2026-07-02
**Framing:** One coherent branch tying together the Japanese capability work, the output-routed
capability-stacks pivot, and the KD/Strix enablers. Unifies the existing docs:
- `.planning/2026-07-02-output-routed-capability-stacks-design.md` (the pivot design)
- `.planning/2026-07-02-output-routed-capability-stacks-EXPERIMENT-PLAN.md` (Experiments A/B/C)
- `strix-kd-teacher` memory (KD strategy)

## The thesis (validated)
The base BitNet-2B-4T is broadly strong; the ONE real capability gap found across domains/code/
math/tool-use/instructions was **cross-lingual (Japanese, dense ppl 74.4)**. We proved the gap lives
in the **output layers**: training `lm_head`+`norm` moves JA ppl down hard while FFN-only training is
flat. This is the empirical basis for **output-routed capability stacks** — route by a format/type
prefix to a tailored top-stack (final block(s) + norm + lm_head) per output type, plugging into
dotLLM's constrained-decoding FSM.

## Current state (2026-07-02)
- **JA heal run (Experiment A, partial):** `mote_train.py --train-lm-head --self-kd off --kd-weight 0`,
  training `lm_head`+`model.norm` only (NOT the final block), 2 MoTE experts at layer 29, seq 256,
  LM-only, adafactor, grad-checkpoint, on the 3060 (`.docs/mote/ja_heal_lmonly`).
  Curve: JA 74.4 → 69.8(500) → 61.7(1000) → 56.8(1500) → 53.9(2000) → 51.9(2500) → 50.2(3000),
  EN held ~14.4. **Descending but decelerating; likely to plateau ~45-50, short of <30.**
- **KD infra built** (`--self-kd`, `--teacher-cache`; Strix dump scripts) — self-KD too heavy for the
  3060 at this config; EN already held without it. Strix teacher cache reserved for full-capacity KD.

## Key open insight
The current run trains `lm_head`+`norm` only. **Experiment A proper is `lm_head` + final decoder
block(s) + norm.** The plateau ~50 suggests the final block IS needed to reach JA<30 — the norm+head
can reshape the output distribution but the last block's representations also carry the gap. Next JA
step: add the final block(s) to the trainable set (VRAM permitting on the 3060; else fewer
tokens/shorter seq, or defer the deeper-stack variant to a ≥16 GB card).

## Roadmap (this branch)
1. **A0 (now):** let the lm_head+norm heal run to its plateau; record the floor it reaches (the
   demonstration that output layers carry the JA gap — already clear).
2. **A1 (next):** re-run with **final block + norm + lm_head** trainable (Experiment A proper),
   target JA<30. Watch VRAM (already tight); reduce seq/tokens or checkpoint-heavy if needed.
   Gate: JA<30 with EN within ~10% of dense.
3. **B:** **K=3 output stacks + format-prefix commit router** (per EXPERIMENT-PLAN Exp B) — one stack
   per output type (e.g. JA / structured / default), router reads the first structural token and
   commits a stack for the sequence. Gate: per-type stack beats shared stack on its type; router
   commits correctly ≥99%.
4. **C:** **bf16 → 1.58-bit ternary QAT** of the trained stack(s) (Exp C), using our STE ternary
   quant. Gate: ternary stack retains most of the bf16 JA gain.
5. **Integration:** wire the committed-stack selection into dotLLM's constrained-decoding FSM
   (the format prefix that selects a grammar also selects the output stack).

## Enablers
- **KD:** EN-preservation via `--self-kd` (EN-gated) if forgetting appears at deeper-stack training;
  Strix teacher cache for any full-capacity KD (coordinate Strix use with the user).
- **Bigger card:** A1 (final-block) and B/C may exceed the 3060's 12 GB; Kaggle T4×2 (when hours
  refresh) or a ≥16 GB card is the home for the deeper-stack + multi-stack runs.

## Non-goals (settled)
Tool-use and instruction-following are NOT capability gaps for this base (flat PPL, no routing
specialization) — they do not need dedicated stacks; constrained decoding already handles their
formats.
