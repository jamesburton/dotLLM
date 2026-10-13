# Track R — recursion-only retrofit (looped ternary depth) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`). **Most "tests" are falsifiable metric-gates (ML experiment). Confirm the 3060 is idle + coordinate the GPU before every run. NEVER launch a detached, self-relaunching GPU job (a spike runaway did exactly that this session — one buggy CPU-fallback run held the GPU for an hour and respawned on kill).**

**Goal:** Retrofit BitNet b1.58-2B-4T into an adaptive looped-depth model — a middle slab re-run N times — on a single 3060, and test the novel contribution: an **inference-time live-perplexity/entropy exit gate** that loops more on hard tokens up to N_max.

**Architecture:** Python-first (HF transformers `BitNetForCausalLM` bf16 master = the ternary path via BitLinear). Prelude/recurrent-slab/coda `(P,R,C)` with a fusion adapter + a **per-loop renorm** (form chosen by Task 1). Curriculum-N heal-train (LM + KD). The exit gate is inference-time, gating on a measured signal (entropy and/or logit-margin) — no learned gate.

**Tech Stack:** Python 3.11, transformers ≥5.4, peft, datasets; `PYTHONUTF8=1`, `HF_HOME=E:/.cache/huggingface`. Base `microsoft/bitnet-b1.58-2B-4T-bf16`. Spec: `.planning/2026-06-29-looped-ternary-moe-tracks-design.md`. Prior spikes: Probe 0 (`spike/recursion-stability` `9b25c64`) showed looping is stable, N=2 improves PPL 20.6→17.0, only failure = residual inflation at N≥4.

## Global Constraints
- N_max = 8. Default slab `(P,R,C)` ≈ `[0:7) / [7:22) / [22:30)` (Probe 0's slab), sweepable.
- Loss = LM + KD (KL to frozen dense BitNet logits), KD weight ≈ 0.5.
- Curriculum-N: recurrence count sampled Poisson-Lognormal, mean μ ramped up to N_max over training.
- Gate signal (DEFAULT, approved): **entropy AND logit-margin** (top1−top2) — evaluate both; gate loops while either flags "hard," up to N_max.
- GPU: confirm idle + coordinate; correct device placement (`.to('cuda')` — the failed 0b ran on CPU by accident); use a KV cache for any generation to avoid O(seq²) recompute.
- Issue-driven: branch `issue/{N}-trackR-recursion`; commits ref `(#N)`; footer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Artifacts under `.docs/recursion/` (gitignored); checked-in `scripts/lora/EVAL_recursion.md`.

## File Structure
| File | Responsibility |
|---|---|
| `scripts/lora/recur_probe_renorm.py` (create) | FAST renorm re-probe (PPL/norm only, KV-cached, GPU) — picks the renorm form. Replaces the failed Probe 0b. |
| `scripts/lora/recur_model.py` (create) | `(P,R,C)` looped forward with fusion adapter + per-loop renorm; configurable recurrence N; reusable for train, eval, gate. |
| `scripts/lora/recur_train.py` (create) | Curriculum-N retrofit heal-train (LM + KD); QLoRA on slab + adapter + renorm/gate params. |
| `scripts/lora/recur_gate.py` (create) | Inference-time live exit gate (entropy / logit-margin), loops to N_max. |
| `scripts/lora/recur_eval.py` (create) | PPL-vs-N, hard-token-benefit (Iₜ vs difficulty), gate-vs-fixed-N quality/compute. |
| `scripts/lora/EVAL_recursion.md` (create) | Methodology + results + verdict. |

---

## Task 1: Renorm form — RESOLVED by Probe 0b (commit f6c08ee); confirm + lock the learned residual gate
Probe 0b already answered this (no re-probe needed): fixed renorms (rescale-to-input-norm, weightless RMSNorm) are **harmful** (collapse quality from N=2); the **damped residual update** `h←h_in+γ(slab−h_in)`, γ=0.5, holds PPL flat-or-improving + decode clean through **N=8** (N=2/4 beat N=1), breaking at N=16. **Decision (locked): the looped block uses a LEARNED per-loop residual gate** `h ← h_in + g·(slab(h_in)−h_in)`, `g` init ≈ 0.5 (scalar or per-channel; optionally indexed by loop-step). It is trained in Task 3.
- [ ] **Step 1 (optional, fast confirm — only if a fresh check is wanted):** on **CUDA** (`assert model.is_cuda`), PPL-only (no greedy decode), reproduce 0b's damped-γ=0.5 row (N∈{1,2,4,8}) to confirm the damped baseline before training the gate. Skip if trusting f6c08ee. **Do NOT re-run the harmful fixed variants or the N=16 no-cache decode that crashed the CUDA context.**
- [ ] **Step 2: Commit** (only if Step 1 run) `chore(lora): confirm damped residual baseline for looped ternary depth (#N)`. The learned gate itself lands in Task 2/3.

## Task 2: `(P,R,C)` looped model module + init-sanity gate
**Files:** create `scripts/lora/recur_model.py`.
**Interfaces produced:** `build_recur(base, P, Q, renorm: str) -> RecurModel` with `forward(input_ids, recurrence: int)` running `embed → layers[0:P) → LOOP( renorm ∘ adapter(layers[P:Q)) )×recurrence → layers[Q:L) → norm → lm_head`. Fusion adapter = `nn.Linear(2*hidden, hidden)` fusing prelude-out + recurrent state (per arXiv:2511.07384); renorm = the **learned per-loop residual gate** `h ← h_in + g·(slab(h_in)−h_in)`, `g` init ≈ 0.5 (Probe 0b, f6c08ee), trained in Task 3.
- [ ] **Step 1: sanity test:** with `recurrence=1` and the adapter init to pass-through (or pre-train-neutral), `build_recur(...).forward(ids, 1)` logits ≈ stock model within tolerance (the retrofit starts near the base). Document the adapter init (random Gaussian per the paper — so N=1 won't be identical; instead assert it's finite + same shape, and that the UNADAPTED base path (recurrence applied to raw slab without adapter) reproduces Task-1's N=1==stock).
- [ ] **Step 2:** Implement; run sanity. **Step 3: Commit** `feat(lora): (P,R,C) looped ternary model module (#N)`.

## Task 3: Curriculum-N retrofit heal-train (LM + KD)
**Files:** create `scripts/lora/recur_train.py`.
**Interfaces produced:** CLI `recur_train.py --P 7 --Q 22 --tokens <2-5e9-scaled-down> --mu-ramp 2..8 --kd-weight 0.5 --out .docs/recursion/r1` → trains the slab + adapter + renorm/gate params (QLoRA/bf16-shadow ternary), freezing prelude/coda/attn/embeds; per step sample recurrence ~Poisson-Lognormal(μ), μ linearly ramped 2→8; loss = LM(at sampled N) + 0.5·KD. Writes adapter + `metrics.json`.
- [ ] **Step 1:** Reuse the harness loader; teacher = frozen dense BitNet. **Step 2:** smoke (`--tokens 1e5`): loss finite, decreases; recurrence sampling works. **Step 3: Commit** `feat(lora): curriculum-N recursion retrofit train + KD (#N)`.

## Task 4: Live exit gate (the novel contribution)
**Files:** create `scripts/lora/recur_gate.py`.
**Interfaces produced:** `generate_adaptive(model, ids, n_max=8, signal="both", ent_thresh, margin_thresh) -> (tokens, per_token_loops)` — at each decode step, loop the slab while (entropy > ent_thresh) OR (top1−top2 margin < margin_thresh), capped at n_max; **KV-cached** (cache per-loop-step KV at prefill, last-step reuse at decode — Ouro finding). Records loops-per-token.
- [ ] **Step 1: test:** on a trained Task-3 adapter, `generate_adaptive` returns coherent text, `per_token_loops ∈ [1, n_max]`, and **varies across tokens** (not constant). Threshold defaults calibrated from the held-out entropy/margin distribution.
- [ ] **Step 2:** Implement; run. **Step 3: Commit** `feat(lora): live perplexity/entropy exit gate for looped decode (#N)`.

## Task 5: Eval — the three success criteria
**Files:** create `scripts/lora/recur_eval.py`.
- [ ] **Step 1 (criterion 1):** val-PPL vs fixed recurrence N∈{1..8} on held-out — **gate:** PPL decreases with N up to train-max and ideally at N=train-max+1 (extrapolation).
- [ ] **Step 2 (criterion 2 — the hypothesis):** per-token loss-improvement `Iₜ(N)=L_{t}(N−1)−L_t(N)`; correlate with token difficulty (entropy/margin). **Gate:** high-difficulty tokens get larger `Iₜ` from extra loops than easy tokens (positive correlation) — the direct test of "hard tokens benefit from more passes."
- [ ] **Step 3 (criterion 3):** the live gate (Task 4) vs best fixed-N: **gate:** matches fixed-N quality (PPL/downstream) at **lower mean loops/token**. Also run the reused `eval_*.py` (tool/coding/instruction) base-vs-recur.
- [ ] **Step 4: Commit** `feat(lora): recursion eval (PPL-vs-N, hard-token benefit, gate-vs-fixedN) (#N)`.

## Task 6: Results writeup + verdict
- [ ] Write `scripts/lora/EVAL_recursion.md` (checked in): renorm form, PPL-vs-N curve, the hard-token-benefit correlation, gate-vs-fixed-N compute/quality, verdict (is adaptive looped ternary depth a real, difficulty-scaled lever?). Note the dotLLM serving follow-up (looped forward + renorm + gate in TransformerModel) gated on a win. Commit. **Track-R acceptance artifact.**

## Self-Review notes
- Spec coverage: renorm (was 0b) → Task 1; (P,R,C)+adapter+renorm → Task 2; curriculum-N + KD → Task 3; live perplexity/entropy+margin gate → Task 4; the 3 success criteria → Task 5. ✓
- The renorm-form open item from the spec is resolved by Task 1 (with a learned-renorm fallback if no fixed form holds). ✓
- Convergence with Track M (looped MoTE) is explicitly a later spec — not here. ✓
- Open risk: KV-cache correctness under variable per-token depth (Ouro: per-step KV at prefill, last-step reuse at decode) — Task 4 must implement this, not the O(seq²) recompute that broke 0b.
