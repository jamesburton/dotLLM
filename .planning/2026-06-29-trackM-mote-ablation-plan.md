# Track M — MoTE ternary-MoE upcycle ablation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute this plan. Steps use checkbox (`- [ ]`) syntax. **Each "test" is a falsifiable metric-gate (this is an ML experiment, not unit code). Confirm the RTX 3060 is idle and coordinate the GPU before every heal run — the user runs a parallel GPU workload.**

**Goal:** Determine whether a dense BitNet b1.58-2B-4T can be upcycled into a ternary Mixture-of-Experts (MoTE) that recovers/beats the dense baseline on a single 3060, via a small ablation over expert count / top-k / shared-expert precision.

**Architecture:** Python-first (HF transformers + the existing `scripts/lora/` QLoRA heal-train harness). Clone the dense FFN into N ternary routed experts + a BF16 router (+ optional frozen FP shared expert); heal-train ~0.3–0.5B tokens with LM + KD loss; measure expert-utilization entropy, val-PPL vs dense, router input-dependence. dotLLM serving uses kernel #116 (`moe_indexed_matmul_i2_s`, CPU, already validated) once a cell wins.

**Tech Stack:** Python 3.11 (`C:/Python311/python.exe`), transformers ≥5.4 (`BitNetForCausalLM`), peft, datasets, bitsandbytes; `PYTHONUTF8=1`, `HF_HOME=E:/.cache/huggingface`. Base: `microsoft/bitnet-b1.58-2B-4T-bf16` (cached). Spec: `.planning/2026-06-29-looped-ternary-moe-tracks-design.md`.

## Global Constraints
- LoRA/MoE config rules from the spec: routed experts ternary {−1,0,1} (absmean per-tensor scale, the BitNet b1.58 quant); **BF16 router**, init σ=0.02; **Switch load-balance aux loss γ=0.01**; do NOT normalize router combine weights (hurts upcycled LMs). lr 1e-4 on the MoE path.
- Loss = LM + **KD** (KL to frozen dense BitNet logits), KD weight ≈ 0.5.
- Heal budget per cell ≈ 0.3–0.5B tokens; convert the **upper ~half** of the 30 blocks to MoE first.
- Experts initialized as **identical clones** of the dense FFN (routing breaks symmetry; no router noise).
- GPU: confirm 3060 idle (`nvidia-smi`) + coordinate before each run; High-Perf power plan; NEVER run a detached self-relaunching GPU job (a spike runaway burned an hour this session).
- Issue-driven: branch `issue/{N}-trackM-mote`; commits ref `(#N)`; footer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- All experiment artifacts (adapters, logs, results JSON) under `.docs/mote/` (gitignored); a checked-in results table in `scripts/lora/EVAL_mote.md`.

## File Structure
| File | Responsibility |
|---|---|
| `scripts/lora/mote_upcycle.py` (create) | Build a MoTE model from the dense base: clone FFN→N ternary experts, add BF16 router, optional FP shared expert, per config. |
| `scripts/lora/mote_train.py` (create) | Heal-train loop (LM + KD + Switch aux), config-driven; writes adapter + metrics JSON. |
| `scripts/lora/mote_eval.py` (create) | Per-cell metrics: expert-utilization entropy, val-PPL vs dense, router input-dependence. |
| `scripts/lora/tests/test_mote_upcycle.py` (create) | pytest: upcycled model is function-identical to dense at init (uniform router + identical clones), and config wiring is correct. |
| `scripts/lora/EVAL_mote.md` (create) | Checked-in methodology + results grid + verdict. |

---

## Task 1: MoTE upcycle module + init-identity gate
**Files:** create `scripts/lora/mote_upcycle.py`, `scripts/lora/tests/test_mote_upcycle.py`.
**Interfaces produced:** `build_mote(base_model, layers: list[int], n_experts: int, top_k: int, shared: str) -> nn.Module` where `shared ∈ {"fp","ternary","none"}`; replaces each FFN in `layers` with a MoTE block = (optional shared FFN) + N routed-expert FFNs (clones) + a BF16 `nn.Linear(hidden, n_experts)` router. `mote_forward` returns `(hidden, aux_loss, expert_counts)`.

- [ ] **Step 1: Write the failing init-identity test.** With `shared="fp"` (the dense FFN as frozen shared) **or** `shared="none"` + identical clones + a router forced uniform (or top-1 with all experts identical), the MoTE block output must equal the original dense FFN output within 1e-4 at init (clones are identical, so any routing gives the same value; with an FP shared expert present, account for the documented `x + MoE + FFN` sum form — test the routed-MoE-only sub-block equals dense, and the shared-expert path equals dense, separately).
```python
def test_mote_init_matches_dense():
    base = load_bitnet_bf16()
    dense_ffn = base.model.layers[20].mlp
    mote = build_mote(base, layers=[20], n_experts=4, top_k=1, shared="none")
    x = torch.randn(2, 8, base.config.hidden_size, dtype=torch.bfloat16)
    y_dense = dense_ffn(x)
    y_mote, aux, counts = mote.model.layers[20].mlp(x)   # identical clones → any expert == dense
    assert torch.allclose(y_dense, y_mote, atol=1e-4)
```
- [ ] **Step 2: Run → FAIL** (`No module named 'mote_upcycle'`). `dotnet`-free; `python -m pytest scripts/lora/tests/test_mote_upcycle.py -k init_matches_dense`.
- [ ] **Step 3: Implement `build_mote`.** Clone `mlp` weights into N expert MLPs (deep copy); wrap routed experts so their `forward` applies BitNet absmean ternary weight-quant + int8 activation quant via the base's BitLinear (reuse the base's quant, do not reinvent); BF16 router `nn.Linear` init `normal_(0, 0.02)`; `shared` per arg (fp=frozen copy of dense FFN bf16; ternary=quantized copy; none=omit). `mote_forward`: `g = softmax(router(x)); idx = topk(g, k); out = Σ_k g·expert_idx(x)` (+ shared(x) if present); aux = Switch loss `E·mean(frac_tokens_i · mean_prob_i)`.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Add config-wiring test** (n_experts/top_k/shared produce the right module shapes + param counts; routed-expert weights are ternary-quantized in forward). Run → PASS.
- [ ] **Step 6: Commit** `feat(lora): MoTE upcycle module (clone FFN→ternary experts + BF16 router) (#N)`.

## Task 2: Heal-train loop (LM + KD + aux)
**Files:** create `scripts/lora/mote_train.py`.
**Interfaces produced:** CLI `mote_train.py --config <name> --n-experts --top-k --shared --layers upper-half --tokens 4e8 --kd-weight 0.5 --out .docs/mote/<config>` → writes adapter + `metrics.json` (final LM loss, KD loss, aux loss, expert-count histogram over the last K steps).
**Interfaces consumed:** `mote_upcycle.build_mote`; the existing heal-train data/loader + masking from `train_task_lora.py`.

- [ ] **Step 1:** Reuse the dataset/loader from `train_task_lora.py` (a general text slice, e.g. a `no_robots`/`fineweb`-style corpus already cached — pick what the harness already loads). Freeze base + (FP) shared expert + attention + embeddings; train router + routed experts (QAT: bf16 shadow, ternary forward) + (LoRA optional). Loss = `LM + 0.5·KL(student||teacher_frozen_dense) + 0.01·aux`. **Teacher** = a second frozen copy of the dense BitNet (no grad).
- [ ] **Step 2:** `py_compile` + a 5-step smoke run (`--tokens 1e5`) → loss finite, aux finite, expert histogram non-degenerate (not all tokens to one expert). Gate: no NaN; ≥2 experts used.
- [ ] **Step 3: Commit** `feat(lora): MoTE heal-train loop with KD + Switch aux (#N)`.

## Task 3: Eval module (the metric-gates)
**Files:** create `scripts/lora/mote_eval.py`.
**Interfaces produced:** `mote_eval.py --adapter <dir> --base <bf16>` → prints + writes `eval.json`: (a) **expert-utilization entropy** H(expert-usage) over a held-out slice (collapse if H ≪ log N), (b) **val-PPL** of the MoTE vs the dense baseline on the same held-out tokens, (c) **router input-dependence** = fraction of distinct argmax experts across tokens (constant-argmax = dead router).
- [ ] **Step 1:** Implement; run on the dense baseline first to record the reference PPL.
- [ ] **Step 2:** Smoke on the Task-2 5-step adapter → all three metrics computed, no crash.
- [ ] **Step 3: Commit** `feat(lora): MoTE eval (expert entropy / val-PPL / router dependence) (#N)`.

## Task 4: Cell #0 — fast-fail (4 ternary experts, top-1, TERNARY shared)
- [ ] Confirm 3060 idle + coordinate. Run `mote_train.py --config c0_ternshared --n-experts 4 --top-k 1 --shared ternary --tokens 4e8 --out .docs/mote/c0`. Then `mote_eval.py`.
- [ ] **Metric-gate (decision):** if val-PPL **recovers to within ~5% of dense** AND expert-entropy stays high → a cheap (all-ternary) anchor works; record and still run the FP-shared cells for comparison. If PPL is far worse / experts collapse → **confirms MoTE's thesis that the FP anchor is needed**; record the fast-fail and proceed with FP shared. Either outcome is informative. Append to `EVAL_mote.md`.

## Task 5: Core 4-expert grid (FP shared vs none × top-1 vs top-2)
- [ ] Run 4 cells (coordinate GPU each): `c1` (4,top1,fp), `c2` (4,top2,fp), `c3` (4,top1,none), `c4` (4,top2,none), each `--tokens 4e8`, eval each.
- [ ] **Metric-gate:** tabulate val-PPL vs dense, expert-entropy, router-dependence, active-params/FLOPs per cell. **Winner = lowest val-PPL with non-collapsed experts at acceptable active cost.** Record the FP-shared-vs-none and top1-vs-top2 deltas (the experiment's core question). Append table to `EVAL_mote.md`.

## Task 6: Capacity scaling — 8 experts (best routing/shared from Task 5)
- [ ] Run `c5` = 8 ternary experts with the winning top-k + shared from Task 5, `--tokens 4e8`, eval.
- [ ] **Metric-gate:** does doubling experts improve val-PPL (capacity headroom) without collapse / disproportionate active cost? Append to `EVAL_mote.md`.

## Task 7: Results writeup + verdict
- [ ] Write `scripts/lora/EVAL_mote.md` (checked in): methodology + the full grid table + verdict — does ternary-MoE upcycling recover/beat dense on a 3060, and which config? Note the dotLLM serving follow-ups (CUDA/Vulkan `moe_indexed_matmul_i2_s`, loader, MoTE block) gated on a win. Commit. **This is the Track-M acceptance artifact.**
- [ ] If a cell clearly wins: open the kernel #116 PR (CPU) and file follow-up issues for the GPU serving path + MoTE block wiring.

## Self-Review notes
- Spec coverage: upcycle recipe → Task 1–2; KD → Task 2; ablation grid (fast-fail #0, 4-expert ×4, 8-expert) → Tasks 4–6; metrics (entropy/PPL/router) → Task 3; serving via #116 → Task 7. ✓
- Metric-gates (not unit tests) are intentional for the heal runs; the only true pass/fail unit test is Task 1's init-identity (functional correctness of the upcycle). ✓
- Open risk: corpus choice for heal-train — use whatever the existing harness already loads to avoid a new download; document it.
