# Ternary Conversion of Diffusion-Generated Adapters — Findings & Ranked Paths (2026-07-06)

## What we proved works
Weight-space diffusion generates a **bf16 LoRA adapter** that recovers **93% of a real trained adapter's
task gain** (base 13.72 → gen 12.41 vs real 12.32 on no_robots), from noise, with zero per-adapter
gradient training. This is solid and reproducible.

## What we proved does NOT work: post-hoc ternary conversion
Goal was a **ternary-only endpoint** matching the bf16 capability, trained faster than ternary-from-scratch
by warm-starting from the generation. **Every conversion+training technique lands at PPL ~17.9 — worse than
the un-adapted base (13.72).**

| method | ternary PPL | notes |
|---|---|---|
| post-hoc merge + ternarize | 31–204 | catastrophic |
| QAT one-jump (LM-loss) | 17.90 | plateau, all scopes/steps |
| QAT λ-ramp (gradual soft→hard) | 17.93 | gradual conversion did NOT help |
| KD-QAT (bf16 teacher, cached logits) | 17.93 | student matched teacher top-k (kd 0.4) yet eval capped |
| base + QAT (control, no adapter) | 13.72 | harness is fine; LM-loss also didn't improve *below* base |

### The decisive insight — it's REPRESENTATIONAL, not optimization
KD trained the student to match the bf16 teacher's top-k logits on the training corpus (kd loss → 0.4),
yet held-out PPL stayed 17.9. So the ternary grid **cannot hold this adaptation for generalization**, no
matter the training signal. The ~10% task adaptation is below the ~50% ternary quant-noise floor, and
merging it disrupts the base's ternary structure → *worse than base*.

### The warm-start premise was false here
The converted-ternary adapter (17.9) is **worse than the base ternary (13.72)** — so as a ternary warm
start it is worse than doing nothing. "Closer than random/identity" is true but irrelevant; the bar is
"better than base ternary," which conversion fails.

## Ranked paths forward

1. **USE THE bf16 HYBRID (recommended, working now).** Base ternary + bf16 LoRA delta = the standard
   MoTE-with-LoRA form, 93% of real. If a bf16 delta at inference is acceptable, we are done — the
   generation is the contribution.

2. **Test on a LARGER-gain task before concluding "ternary is dead."** Our negative is strongest for a
   SMALL adaptation (no_robots ≈ 10% bf16 gain = the worst case for ternary). A task where the bf16
   adapter gains 2–5× more might survive ternarization. Cheap to test with the existing pipeline on a
   harder task (e.g. code/math). *Highest information-per-effort next step.*

3. **Higher-rank / deeper LoRA generation (the user's idea, untested).** Generate rank-32/64 so more
   capacity might survive quantization. Requires re-bootstrapping the adapter population + AE/diffusion
   at the new rank (~half a day). Uncertain: the limit looks like base/task representational capacity,
   not adapter rank — but it is genuinely untested.

4. **Native ternary training from scratch (for a pure-ternary endpoint).** Train the adaptation *inside*
   the ternary grid from the start (BitNet regime), not convert. The only route the evidence supports for
   pure ternary, but large effort and the base-QAT control (13.72, no improvement) warns that even native
   ternary task-adaptation is weak for small-signal tasks.

5. **LoftQ / staged block-wise conversion (low expected payoff).** The robust 17.9 across methods —
   including the gradual λ-ramp — indicates conversion-side tricks won't escape the representational floor.
   Deprioritize.

## Infra notes (reusable)
- Intermittent 2B-model-load **segfaults** (GPTQModel/BitNet) — wrap GPU runs in retry loops; kill via the
  PowerShell tool (git-bash `pkill` unreliable on Windows).
- Full-scope QAT (1.6B trainable) OOMs/thrashes the 3060; use `--last-n-layers` (added) + adafactor.
- Both drives run full easily; save models to E:, TMP→E:; the C: `~/.cache/huggingface` is a **junction**
  to E: (deleting it deletes the real cache).
- λ-ramp mechanism validated: λ=0 == pure bf16, λ=1 == native ternary (exact), STE grads flow.
