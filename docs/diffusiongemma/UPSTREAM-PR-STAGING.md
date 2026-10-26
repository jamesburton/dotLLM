# Upstream PR staging — Gemma-4 / DiffusionGemma (#G1–#G6 under epic #40)

> **Status: STAGED, HELD.** The six PR branches are pushed to the **fork** (`origin` =
> `jamesburton/dotLLM`). **Nothing is opened on `upstream` (`kkokosa/dotLLM`) yet** — by design.
> This doc is the ready-to-open package; open the issues + PRs upstream only once the
> **precondition** below is satisfied.

## Why held (the precondition)

The G1–G6 stack builds on a base lineage (the SafeTensors foundation #23–#34 plus the GGUF
arch-mapping/head-dim/diffusion WIP `bd7a60c..70e859a`) that is **not on `upstream/main`**.
Measured against the current tree:

- `upstream/main` (`2c7ea8e`) is the **merge-base** of the whole stack — upstream has *none* of it.
- A focused PR of **G1** shows **567 commits** vs `upstream/main`, and **14 commits** even vs
  `dev-diffusiongemma` (it still carries the unmerged base chain).

So a clean, individually-reviewable PR is impossible until the base lineage lands upstream first.
**This is the "pending others merging" gate.** Until then the branches live on the fork only.

## Pushed fork branches (origin)

Linear stack, each on the prior. Tip `970525f` is byte-identical to `dg-pr11-gemma4-arch @ 9686e24`
(lossless decomposition — verified).

| PR | Branch (on `origin`) | Closes | Commit(s) |
|----|----------------------|--------|-----------|
| A | `issue/G1-gemma4-moe-gguf-ar-forward` | #G1 (Refs #40) | `f6b3e68` |
| B | `issue/G2-diffusiongemma-gguf-forward` | #G2 (Refs #40) | `5906e1b` (+`478a47c`(a)) |
| C | `issue/G3-diffusiongemma-self-cond` | #G3 (Refs #40) | `6076ee7` |
| D | `issue/G4-synthetic-gguf-fixture` | #G4 (Refs #40) | `8fc653c` (+`478a47c`(b)) |
| E | `issue/G5-diffusion-pkv-cache` | #G5 (Refs #40) | `4e5a44e` |
| F | `issue/G6-cross-backend-harness` | #G6 (Refs #40) | `ee946ae` + `9686e24` |

## Issue + PR text

The **ready-to-paste issue bodies** for the #40 epic and #G1–#G6 children live in
[`PR-DECOMPOSITION.md` §3](PR-DECOMPOSITION.md). Each PR `Closes #Gn` + `Refs #40`. PR titles:

- **A** — `feat(gemma4): Gemma-4 MoE autoregressive GGUF forward (real 26B validated)`
- **B** — `feat(diffusiongemma): region-aware GGUF diffusion forward + mask-token suppression`
- **C** — `feat(diffusiongemma): self-conditioning for coherent masked-diffusion generation`
- **D** — `test(gemma4): synthetic GGUF fixture + F32/Q8_0/Q5/Q4_K quantizers + regression harness`
- **E** — `perf(diffusiongemma): opt-in prompt-KV (PKV) prefill/decode cache — byte-exact`
- **F** — `test(backends): cross-backend timing harness + gemma4 GPU gap report + Vulkan fail-fast`

Note: the **Vulkan/CUDA backend work** (dg-pr12/13/14/15/16: non-causal attention, AR gemma4 on
Vulkan, quantized-expert MoE shaders, CUDA forward) is a **separate upstream track** from G1–G6
(which are CPU + diffusion-engine only). It needs its own decomposition before upstreaming, and the
diagnostic harness methods on `VulkanTransformerModel` (`RunGemma4LayerOnHost` /
`RunGemma4EmbeddingOnHost` / `RunGemma4FinalHeadOnHost`, currently `public`) should be trimmed or
made `internal` (with `InternalsVisibleTo` for the test project) before that track is opened.

## Open-when-ready checklist (upstream)

1. Land the base lineage (#23–#34 + the `bd7a60c..70e859a` WIP) on `upstream/main`.
2. Rebase the G1–G6 stack onto the now-upstream base (should be clean — see §5 of PR-DECOMPOSITION).
3. Open epic **#40** (if not already) + children **#G1–#G6** using the §3 text.
4. Open the 6 PRs in stack order (A→F), each as **draft** until its parent merges, `Closes #Gn`
   `Refs #40`.
5. Decompose + open the Vulkan/CUDA backend track separately (after the diagnostic-API tidy).
