# dg-pr11 → focused upstream PRs — decomposition plan (PREPARE-ONLY)

> Status: **proposal for human review.** Nothing here has been pushed, no GitHub issue/PR has been
> opened. This document decomposes the local stack `dg-pr11-gemma4-arch` (8 commits,
> `bd7a60c..9686e24`) into small, individually-reviewable PRs per the CLAUDE.md issue-driven workflow.

## 0. Lineage finding (read this first — it changes the issue mapping)

The existing planning artifacts (`PR-PLAN.md`, `proposed-issues/01..12`) were written for, and have
**already been implemented by**, an *earlier* body of work — the SafeTensors-based chain that sits
**below** `dg-pr11`'s base:

```
e8580ec  dev-diffusiongemma  (holds PR-PLAN.md + proposed-issues/)
  └─ 9ef076d  Gemma backbone — embed scale, GeGLU, 4-norm, (1+w), QK-norm   (#23  == proposed 01)
  └─ b6b284f  Gemma-4 MoE arch + SafeTensors loader                         (#24/#25 == proposed 02/03)
  └─ eb16573  bidirectional / hybrid attention-mask seam                    (#26  == proposed 04)
  └─ cd84c57  DiffusionConfig + mask-token plumbing                         (#27  == proposed 05)
  └─ ea5ce57  denoise scheduler + entropy-bound unmask sampler              (#30/#31 == proposed 08/09)
  └─ 6230300  DiffusionTextGenerator — iterative denoise loop               (#28  == proposed 06)
  └─ 720d684  DiffusionGemma model load + config extractor + dispatch       (#29  == proposed 07)
  └─ 8229b1c  server: route diffusion models through generator              (#34  == proposed 12)
  └─ 543d1ea/821a055  load GGUF (Llama) as diffusion + LLaDA validation     (#32)
  └─ cc1021c  per-layer attention head_dim for Gemma-4 (cacheless)          (#36)
  └─ 70e859a  docs
  └─ bd7a60c  WIP: partial GGUF gemma4/diffusion-gemma arch mapping         (#40, base of dg-pr11)
```

So **proposed-issues 01–12 / fork issues #23–#34 are spent.** The `dg-pr11` stack is the *next*
body of work: it takes the SafeTensors/synthetic foundation and brings the **real GGUF 26B**
(`gemma-4-26B-A4B-it`, `diffusiongemma-26B-A4B-it`) to correct, validated CPU inference — a GGUF
config-extractor path, the source-confirmed gemma4 MoE forward validated on real weights,
self-conditioning, the PKV throughput cache, a portable synthetic-GGUF regression fixture + GGUF
quantizers, and a cross-backend timing/gap harness.

**These eight commits therefore need NEW issues.** This plan proposes six (`#G1..#G6`) and gives
ready-to-paste issue text in §3. The PR-PLAN.md milestone/issue numbers are NOT reused.

### Important caveats for the reviewer
- **Neither the dg-pr11 stack nor its base chain (`e8580ec..bd7a60c`) is on the fork's `origin/dev`
  yet** (`origin/dev` is 570 commits behind `upstream/main`'s lineage and does not contain
  `dev-diffusiongemma`). The merge-base of everything is `upstream/main @ 2c7ea8e`. So "stack onto
  `origin/dev`" per CLAUDE.md is not literally possible today: the realistic base for these PRs is
  **`dev-diffusiongemma`** (the integration mirror that already carries #23–#34). Each PR below is
  expressed as a branch off `dev-diffusiongemma` (or off the prior PR in the stack). Whether the
  whole `dev-diffusiongemma` line is itself first fast-forwarded into `origin/dev` is a separate
  human decision (out of scope for this prepare-only task).
- **All eight commits reference `(#40)`** — a single umbrella issue. Decomposing means giving each
  PR its own focused issue and re-tagging commits, OR keeping `#40` as the epic and adding the
  per-PR issues as children. Recommended: keep `#40` as the **epic**, open `#G1..#G6` as children,
  and have each PR `Closes #Gn` + `Refs #40`.

## 1. Proposed PR list (titles + issue mapping + stacking order)

Stacking order is linear and follows the commit DAG (each PR is reviewable on top of the previous).
Base for PR-A is `dev-diffusiongemma`; each later PR stacks on the branch above it.

| PR | Title | Closes | Commit(s) | Stacks on |
|----|-------|--------|-----------|-----------|
| **A** | `feat(gemma4): Gemma-4 MoE autoregressive GGUF forward (real 26B validated)` | **#G1** (Refs #40) | `f6b3e68` | `dev-diffusiongemma` |
| **B** | `feat(diffusiongemma): region-aware GGUF diffusion forward + mask-token suppression` | **#G2** (Refs #40) | `5906e1b` (+ optionally fold `478a47c`) | A |
| **C** | `feat(diffusiongemma): self-conditioning for coherent masked-diffusion generation` | **#G3** (Refs #40) | `6076ee7` | B |
| **D** | `test(gemma4): synthetic GGUF fixture + F32/Q8_0/Q5/Q4_K quantizers + regression harness` | **#G4** (Refs #40) | `8fc653c` (+ `478a47c`(b) if not folded into B) | A (logically); ordered after C in the stack |
| **E** | `perf(diffusiongemma): opt-in prompt-KV (PKV) prefill/decode cache — byte-exact` | **#G5** (Refs #40) | `4e5a44e` | C (+ D for its equivalence test fixture) |
| **F** | `test(backends): cross-backend timing harness + gemma4 GPU gap report + Vulkan fail-fast` | **#G6** (Refs #40) | `ee946ae` + `9686e24` | D (needs synthetic fixture) |

**Linear stacked-branch order (recommended, each `Closes` its issue):**

```
dev-diffusiongemma
  └─ A  issue/G1-gemma4-moe-gguf-ar-forward          (f6b3e68)
       └─ B  issue/G2-diffusiongemma-gguf-forward      (5906e1b [+478a47c])
            └─ C  issue/G3-diffusiongemma-self-cond     (6076ee7)
                 └─ D  issue/G4-synthetic-gguf-fixture   (8fc653c [+478a47c(b)])
                      └─ E  issue/G5-diffusion-pkv-cache  (4e5a44e)
                           └─ F  issue/G6-cross-backend-harness (ee946ae + 9686e24)
```

### Why this order (dependency rationale)
- **A first** — `f6b3e68` introduces `ModelConfig.Gemma4DualFfn`, the gemma4 RoPE/RmsNorm kernel
  additions, the `TransformerModel.RunGemma4Layer`/`LoadGemma4Layer`/`BuildGemma4Config` path, and
  the GGUF config-extractor changes. Everything else references these.
- **B on A** — region-aware diffusion forward edits the *same* `TransformerModel.cs` /
  `TransformerWeights.cs` / `GgufModelConfigExtractor.cs` regions A created, and removes A's
  `diffusion_gemma → NotSupported` throw.
- **C on B** — self-conditioning adds to the canvas-embed block B introduced (it layers `sc_sig`
  onto B's weight-less rms_norm) and extends the same diffusion forward tests.
- **D after A (content), placed after C in the stack** — `8fc653c` is *purely additive* (new files
  only; no edits to existing src), so it has no hard code conflict with B/C and depends only on A's
  forward existing for its regression assertions. Placing it after C keeps the branch linear and
  lets its tests exercise the self-conditioned path too. (If you prefer, D can branch directly off A
  and be reviewed in parallel with B/C — see §2 "parallelizable" note.)
- **E on C (+D)** — PKV rides the diffusion forward (needs B+C) and its equivalence test uses the
  synthetic diffusion-gemma fixture (D).
- **F last** — the cross-backend harness + GPU-gap probe load the synthetic fixture (D); the Vulkan
  fail-fast guard (`9686e24`) keys on `config.Gemma4DualFfn` (A) and is the clean message the gap
  probe (`ee946ae`) asserts, so the two belong in one PR.

## 2. Commits that resist a clean split (and the recommendation)

There are **no commits that resist splitting at the file level** — the eight commits already have
clean, mostly self-contained seams (see the per-commit file table in §4). The only judgement calls:

1. **`478a47c` bundles two things**: (a) a real correctness fix in `DiffusionTextGenerator.
   FillRemainingMasked` (never commit the mask token in the step-limit fallback) and (b) a new
   `Hidden512` multi-super-block Q4_K fixture regression test. They are *separable* by file
   (`DiffusionTextGenerator.cs` vs `SyntheticGemma4RegressionTests.cs`) but conceptually paired:
   (b) is part of validating the Q4_K path that the fixture (D) exercises, and (a) is a
   diffusion-forward correctness fix that belongs with B. **Recommendation:** do NOT hard-split it.
   Fold hunk (a) into **PR-B** (it is a diffusion-forward correctness fix and the natural home is the
   region-aware diffusion PR; it can be cherry-picked as a follow-on commit on the B branch) and
   hunk (b) into **PR-D** (it is a fixture test). If a single clean commit is preferred over a
   split, keep `478a47c` whole inside **PR-D** — it still builds and the fix is small. The split is
   cosmetic, not load-bearing.

2. **`f6b3e68` is large (1070 LoC, the biggest commit)** but is a *single cohesive feature* (the
   gemma4 MoE AR forward) — RoPE partial-NeoX, weight-less RmsNorm, the dual-FFN forward, the fused
   gate_up split, and the GGUF config extractor are all interdependent pieces of one graph and were
   validated together against the real 26B. Splitting it further (e.g. "kernels" vs "forward" vs
   "config extractor") would produce intermediate states that **don't load or validate any model**
   and have no independent acceptance criterion. **Recommendation:** keep `f6b3e68` as ONE PR (A).
   It is large but coherent and individually reviewable against `docs/diffusiongemma/
   GEMMA4-GRAPH-SPEC.md` (the implementation contract committed alongside it).

3. **`ee946ae` + `9686e24` are two commits, one PR.** The harness/probe commit *documents* the
   Vulkan NRE; the fail-fast commit *fixes* it to a clean `NotSupportedException` that the probe then
   asserts. Shipping the probe without the guard would land a test asserting an NRE. **Recommendation:**
   combine into PR-F.

**Parallelizable (not a split problem, an ordering option):** PR-D (`8fc653c`, purely additive) and
the B→C diffusion chain touch disjoint files, so D could be reviewed in parallel off A rather than
stacked after C. The table in §1 stacks it linearly for a simple single-line review; either is fine.

## 3. Proposed new issues (ready-to-paste text)

> Keep `#40` as the **epic**. Open these as children; each PR does `Closes #Gn` and `Refs #40`.
> Effort tags follow the proposed-issues convention (S/M/L/XL).

### #G1 — Gemma-4 MoE autoregressive GGUF forward (real 26B) — **L**
**Motivation.** The SafeTensors Gemma-4 MoE path (#24/#25) and the GGUF arch mapping WIP (`bd7a60c`,
#40) exist, but the real `unsloth/gemma-4-26B-A4B-it` **GGUF** does not yet load+predict. Gemma-4's
layer differs from Gemma-3 in ways the shared forward did not cover.
**Scope.** Behind a gated `ModelConfig.Gemma4DualFfn`: V-from-K on V-less global layers
[5,11,17,23,29] + weight-less V-RMSNorm (V never roped); attention softmax scale 1.0
(`QueryPreAttnScalar=1.0`); dual *parallel* FFN (dense GeGLU summed with 128-expert top-8 MoE) with
the five FFN norms in exact positions; custom router (`rms(attn_out)·1/√hidden·ffn_gate_inp_s` →
softmax/top-8/renorm with 6.1e-5 clamp; per-expert `ffn_down_exps.scale` folded into routing);
fused `ffn_gate_up_exps` split (gate=first Ie rows, up=last Ie rows, stride 2·Ie); per-layer
`layer_output_scale`; partial-NeoX RoPE on global layers (rotate first 64 pairs, pairing offset
headDim/2, freq denom over full head dim); plain GGUF norm load (no +1, `norm_shift=0`). Extend
`BuildGemma4Config` (PartialRotaryFactor=0.25, QPAS=1.0, Gemma4DualFfn=true, parse Bool-array
`sliding_window_pattern`). `diffusion_gemma` GGUF still throws NotSupported (handled by #G2).
**Acceptance.** Real 26B Q4_K_M cacheless causal forward: "The Eiffel Tower is located in" → " Paris"
top-1 (gated on `DOTLLM_GEMMA4_GGUF`). New `Gemma4GgufForwardTests`. No regression to
Llama/Qwen/Gemma3/Mixtral/QwenMoe/DeepSeek. Commit `f6b3e68`.

### #G2 — DiffusionGemma region-aware GGUF forward + mask-token suppression — **M**
**Motivation.** Put the masked-diffusion decode on the #G1 gemma4 backbone so the real
`diffusiongemma-26B-A4B-it` GGUF loads and runs the denoise loop end-to-end.
**Scope.** GGUF config builds the full gemma4-identical config with a non-null `DiffusionConfig`
(canvas_length, mask_token_id, CanvasAttentionMode=Hybrid); remove the #G1 NotSupported throw. Load
per-layer `enc_layer_output_scale` (diffusion-gemma only). Gated region-aware forward
(`DiffusionConfig != null`, so AR is byte-identical): canvas rows get an extra weight-less rms_norm;
prompt rows use `enc_layer_output_scale`, canvas rows use `layer_output_scale`; P =
`_currentMaskSpec.PrefixLength`. Suppress the mask token from each masked position before sampling
in `DiffusionTextGenerator` (DiffusionGemma ranks mask id 4 argmax at masked positions; without
suppression every unmask is a no-op). **Optionally fold `478a47c` hunk (a)** — the matching
step-limit-fallback mask suppression in `FillRemainingMasked`.
**Acceptance.** Real 26B: prompt-only causal forward → " Paris" (logit 22.6); denoise loop runs end
to end, canvas fully materialised (no surviving mask tokens). `DiffusionGemmaGgufForwardTests`;
extractor tests. No regression to gemma4 AR / LLaDA. Commit `5906e1b` (+ `478a47c`(a)).

### #G3 — DiffusionGemma self-conditioning (coherent generation) — **M**
**Motivation.** Without self-conditioning the canvas degenerates to a repeated token; SC is the
final piece for coherent text and is how the model was trained to denoise.
**Scope.** Load the four model-level `self_cond_*` tensors (pre_norm F32, gate/up Q4_K, down Q5_0)
into a `Gemma4SelfCondWeights` bundle (null when absent → AR/LLaDA unaffected). Add
`IModel.SetDiffusionSelfCond(prevCanvasLogits, canvasLen, scUse)` (default no-op). Implement
`ApplySelfConditioning` (soft-embed over the tied table once per step → gated GeGLU MLP → added to
canvas rows before the weight-less rms_norm); `sc_temp_inv=1.0`, `sc_use=(step==0)?0:1` so step 0 is
the byte-identical zero-SC path. Generator feeds the prior step's post-softcap canvas-region logits.
**Acceptance.** Real 26B canvas 16/16 steps: "The Eiffel Tower is located in" denoises to text
containing Paris/French/Eiffel; SC gated so AR + LLaDA byte-identical. Commit `6076ee7`.

### #G4 — Synthetic GGUF fixture + GGUF quantizers + regression/timing harness — **M**
**Motivation.** A deterministic, architecturally-complete TINY gemma4 + diffusion-gemma GGUF lets
the full feature set be regression-tested and kernels optimised across backends WITHOUT the 26B
resident, and is the portable cross-backend artifact (CI-safe, no checkpoint).
**Scope.** GGUF quantizers (the inverse of the dequant kernels, bit-compatible): `Quantize.cs`
(Q8_0/Q5_0/Q5_1) + `QuantizeKQuants.cs` (Q4_K). Minimal GGUF v3 `GgufWriter`.
`SyntheticGemma4Config`/`SyntheticGemma4Gguf` (seeded xorshift, Tiny+Bench, per-class quant mirroring
the real model; emits gemma4 + diffusion-gemma exercising every feature: V-less global, dual head
dim, partial rope, QK-norm, dual FFN, fused experts, softcap, layer scales). `SyntheticGemma4Harness`
(per-phase CSV timing). Tests: `QuantizeRoundTripTests` (16), `SyntheticGemma4RegressionTests`
(all-features + deterministic golden argmax=144 + AVX2-gated FNV checksum), `SyntheticGemma4HarnessTests`.
**Optionally fold `478a47c` hunk (b)** — the Hidden512 multi-super-block Q4_K fixture test.
**Acceptance.** Tests generate their own fixture (no checkpoint); quantizer round-trips within each
format's quant error; deterministic golden byte-stable. Purely additive (no existing src modified).
Commit `8fc653c` (+ `478a47c`(b)).

### #G5 — Opt-in prompt-KV (PKV) prefill/decode cache — byte-exact — **M**
**Motivation.** Compute the prompt prefix ONCE per canvas block and reuse it across denoise steps
instead of recomputing [prompt|canvas] each step. Pure throughput optimisation, byte-exact.
**Scope.** `DiffusionPromptKvStore` (per-layer F32 K/V row buffers, per-layer sized for sliding vs
global). `IModel` PKV seam (`DiffusionPrefillPromptKv`/`DiffusionDecodeWithPromptKv` +
`SupportsDiffusionPromptKv`, default-throwing; only `TransformerModel` implements). Prefill
(prompt-only causal forward captures post-norm/post-rope K + weight-less-normed V incl. V-from-K,
no LM head); Decode (canvas-only, K/V = concat(cached prompt | fresh canvas), Bidirectional mask
with positionOffset=P). Generator: opt-in `enablePromptKv` flag, gated on
`SupportsDiffusionPromptKv && CanvasAttentionMode==Hybrid` (off for LLaDA). Does NOT use `IKvCache`
(respects the #26/#G-AR-cache seam). Default off ⇒ unified path verbatim.
**Acceptance.** Synthetic diffusion-gemma fixture: single-step canvas logits maxDelta=0 (bit-exact);
full-gen + 3-block ids identical PKV-on vs PKV-off; PKV-off byte-identical to pre-PKV golden. Commit
`4e5a44e`. Depends on #G2/#G3 (forward) + #G4 (fixture for the equivalence test).

### #G6 — Cross-backend timing harness + gemma4 GPU gap report + Vulkan fail-fast — **S/M**
**Motivation.** A reusable CPU/Vulkan/CUDA timing harness over the synthetic fixture, a probe that
documents where gemma4 currently fails on GPU, and a clean fail-fast guard (so the failure is a
clear `NotSupportedException`, not an NRE deep in weight upload).
**Scope.** `CrossBackendTimingHarness` (Run/TryRun/IsAvailable over IModel+IKvCache; warmup + N
prefill + M decode; CSV `phase,name,ms,tokens_per_sec`; graceful skip when a backend/device absent).
`VulkanCrossBackendTimingDemoTests` (SkippableFact, loads a SUPPORTED Llama arch on Vulkan).
`Gemma4GpuGapProbeTests` (loads the synthetic gemma4 fixture, probes Vulkan/CUDA — documents the
upload-time rejection). `GEMMA4-GPU-GAPS.md` roadmap. Vulkan fail-fast: guard `config.Gemma4DualFfn`
in `RejectUnsupportedArchitecture` → clear NotSupported pointing at GEMMA4-GPU-GAPS.md.
**Acceptance.** Vulkan demo + gemma4 probe pass on the real GPU (CUDA probe skips when no device);
the probe asserts the clean message (not an NRE); no regression. Commits `ee946ae` + `9686e24`.

## 4. Per-commit → PR map (the seam evidence)

| Commit | Subject (short) | LoC | PR | Files (kind) |
|--------|------------------|-----|----|--------------|
| `f6b3e68` | gemma4 MoE AR GGUF forward | +1070 | **A / #G1** | Core ModelConfig; Cpu RmsNorm+RoPE; Models arch (TransformerModel/Weights/Arch/ForwardState) + GgufModelConfigExtractor; tests Gemma4Gguf + extractor |
| `5906e1b` | region-aware diffusion forward | +485 | **B / #G2** | Engine DiffusionTextGenerator; Models TransformerModel/Weights + GgufModelConfigExtractor; tests DiffusionGemmaGguf + extractor; spec doc |
| `6076ee7` | self-conditioning | +391 | **C / #G3** | Core IModel; Engine DiffusionTextGenerator; Models TransformerModel/Weights; tests DiffusionGemmaGguf; spec doc |
| `8fc653c` | synthetic GGUF fixture + quantizers | +1853 | **D / #G4** | Cpu Quantize+QuantizeKQuants; Models Gguf (Writer/Config/Gguf/Harness); tests Synthetic + QuantizeRoundTrip; 2 fixture docs — *all additive* |
| `478a47c` | mask-token fallback fix + Q4_K test | +57 | **B(a) + D(b)** (or whole→D) | Engine DiffusionTextGenerator (fix); tests SyntheticGemma4Regression (test) |
| `4e5a44e` | opt-in PKV cache | +606 | **E / #G5** | Core DiffusionPromptKvStore + IModel; Engine DiffusionTextGenerator; Models TransformerModel; tests PromptKvEquivalence |
| `ee946ae` | cross-backend harness + gap report | +710 | **F / #G6** | tests Backends (Harness/Probe/Demo); GEMMA4-GPU-GAPS.md — *all additive* |
| `9686e24` | Vulkan fail-fast | +11 | **F / #G6** | Vulkan VulkanTransformerModel (guard) |

## 5. Staged local branches

See `git branch --list 'issue/G*'` in this worktree. Staging status is recorded below by the agent
that produced this plan. **Nothing was pushed; no PR/issue was opened.**
