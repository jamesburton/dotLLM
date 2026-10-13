# No-Merge Ternary LoRA — the working ternary conversion (2026-07-06)

## TL;DR
The prior campaign concluded "ternary conversion is representationally dead" — but that was for
**merge-based** conversion only (merge the LoRA delta into the base weight, then ternarize → PPL
17.9–55, worse than base). A **no-merge** ternary LoRA **works**: keep A,B as a *separate* low-rank
term, ternarize BOTH A and B via STE, add alongside the **frozen, untouched** ternary base, and QAT
on the task. It is the first ternary conversion in this whole campaign to **beat base**, it scales
strongly with rank, and at matched rank/scope it recovers ~71% of the full bf16 capability.

## Method
`scripts/diffusion/pdiff_ternary_lora.py` (LoTA-QAF style):
- `y = base(x) + (alpha/r) * WeightQuant_ste(B) @ WeightQuant_ste(A) @ x`
- base `AutoBitLinear` FROZEN throughout (its ternary grid is never disturbed).
- A,B are bf16 masters, ternarized every forward via STE; only A,B receive gradients.
- λ-ramp 0→1 over the first N steps phases in ternarization (warm QAT start), then full ternary.
- Optional warm-start of A,B from a trained/generated adapter; else kaiming-A / zeros-B cold start.

Why it works where merge failed: merging perturbs every base weight by the delta and re-ternarizes
the *sum*, corrupting the base grid in proportion to the delta magnitude (→ 17.9–55). Adding a
separate ternary term leaves the base grid intact, so the worst case is ≈ base, and the ternary
low-rank term contributes task capacity on top.

## Results — all on the same harness
no_robots, eval `train[:500]` n=100, train `train[500:2500]`, last-12 layers unless noted, **base PPL 11.675**.

### Rank is the dominant, still-unsaturated lever (scratch, 1500 steps)
| rank | ternary PPL | Δ vs base |
|---|---|---|
| 16  | 11.499 | −0.176 |
| 64  | 11.283 | −0.392 |
| 128 | 11.199 | −0.476 |
| 256 | 11.092 | −0.583 |

Roughly log-linear in rank; no plateau by r256. Rank buys ternary-survivable capacity — this is the
main knob for closing the gap to non-ternary.

### Ternary vs the true bf16 ceiling (r128, last-12)
| variant | PPL | gain | note |
|---|---|---|---|
| base (no LoRA) | 11.675 | — | |
| bf16-trained LoRA (ceiling) | 11.007 | −0.668 | honest non-ternary reference |
| **ternary QAT LoRA** | **11.199** | **−0.476** | **captures ~71% of the ceiling** |
| post-hoc ternary of a bf16-trained LoRA | 11.356 | −0.319 | QAT-in-ternary beats post-hoc |

The ternary QAT term captures ~71% of the achievable bf16 gain at matched rank/scope, and QAT
(training inside the ternary grid) clearly beats post-hoc ternarizing a bf16 LoRA.

### Warm-start (diffusion-generated adapter) vs scratch — last-12
| budget | warm | scratch | warm advantage |
|---|---|---|---|
| 300 steps | −0.015 | 0.000 | +0.015 |
| 1500 steps | −0.231 | −0.176 | +0.055 |

Warm-start is consistently but only modestly better. Reason (below): almost none of the generated
adapter's task gain lives in the last-12 layers.

### The diffusion warm-start reframe
`ceiling_gen_notrain` (gen adapter, last-12, NO training): bf16 gain within last-12 = **0.024**.
The generated adapter's real gain (~1.3 on its own full-30-layer harness) is spread across all 30
layers; restricted to the 12 layers the ternary experiments train, it is nearly absent. So the
ternary gains above are driven by **QAT task-training**, not the diffusion warm-start.

### Full-scope (all 30 layers) warm vs scratch — the fair test of the generation
Full-scope is cheap here because the base is FROZEN (only tiny LoRA A/B train; the old merge-QAT OOM
was from training the 1.6B base). This tests the warm-start where the gen adapter's gain lives.

| config | ternary PPL | Δ vs base | note |
|---|---|---|---|
| warm_r16_full (all 30) | 11.103 | −0.572 | diffusion warm-start |
| scratch_r16_full (all 30) | 11.208 | −0.467 | baseline |
| **warm advantage** | | **+0.105** | vs +0.055 at last-12 |

Two findings: (a) **scope matters as much as rank** — full-scope r16 (−0.572) ≈ last-12 r256
(−0.583); (b) **the diffusion warm-start earns its keep at full scope** — its advantage over scratch
grows from +0.055 (last-12) to +0.105 (full-30), consistent with the gen adapter's gain being spread
across all layers. Modest but real and scope-consistent.

## Bottom line
1. **No-merge ternary LoRA is a working path to a ternary-only capability boost** — beats base,
   scales with rank AND scope, recovers ~71% of the bf16 capability at matched rank. First
   non-catastrophic ternary conversion in the campaign. This is the durable contribution.
2. **The diffusion generation is rehabilitated as a warm-start** — at full scope it gives a durable
   +0.105 PPL edge over scratch (~22% larger gain). Not the headline lever, but it earns its keep
   where its gain lives, at zero per-adapter training cost.
3. **Two levers stack**: rank (r16→r256: −0.18→−0.58, unsaturated) and scope (last-12→full-30 at r16:
   −0.18→−0.57). Best single result: warm_r16_full −0.572 and scratch_r256 −0.583. Combining
   high-rank + full-scope + warm-start is the apex configuration, but **full-scope × r64 is
   memory-throttled on the 3060** (11.9/12.3 GB, 0.12 steps/s → ~3.3 h/run) — not cheaply measurable
   here. Each lever independently reaches ~−0.58; stacking is expected to exceed that. Confirming the
   apex number is a T2 (Strix) / T3 (Kaggle) follow-up, not worth 3+ h of 3060 time.
4. **Recommendation.** For a **pure-ternary endpoint**: no-merge ternary LoRA + QAT at the highest
   rank that fits, full scope, warm-started from a generated (or trained) adapter. For a **hybrid**:
   the bf16 LoRA ceiling is ~29% better (r128: −0.668 vs −0.476), so ship the bf16 hybrid only if a
   bf16 delta at inference is acceptable; otherwise the ternary path now exists and closes ~71% of
   that gap while staying fully ternary.

## Infra notes (reusable)
- Device gotcha: base loads via `device_map={"":cuda}` but freshly-created wrapper `lora_A/lora_B`
  default to CPU → move them (`p.data=p.data.to(device)`) after wrapping. (Fixed in-script.)
- Do NOT queue a second GPU orchestrator while the first is still running with a short gpu-wait
  budget — it will give up waiting and collide (OOM). Wait for explicit completion, then relaunch.
- Full-scope r16 fits the 3060 at ~12.0GB (near cap) because base is frozen; larger rank at full
  scope may need last-n-layers.
