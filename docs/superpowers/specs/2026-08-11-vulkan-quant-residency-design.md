# Vulkan quantized residency: close the F32 expansion gaps

**Date:** 2026-08-11
**Status:** design approved, not yet implemented

## Problem

`VulkanWeights.DeviceQuantTypeFor` resolves a matrix's on-device storage type through a chain of
`Keep*OnDevice` predicates and ends with an unconditional `return QuantizationType.F32`. Any
quantisation type without a Vulkan kernel therefore **silently expands to F32 on upload** rather than
failing or warning.

Five types hit this path. All five are implemented on CPU, so this is a Vulkan-only gap:

| type | block bytes / elems | bits per weight | expansion to F32 | Vulkan shaders today |
|---|---|---|---|---|
| MXFP4 | 17 / 32 | 4.25 | **7.5x** | none |
| Q4_0 | 18 / 32 | 4.5 | **7.1x** | none |
| Q4_1 | 20 / 32 | 5.0 | **6.4x** | none |
| Q5_0 | 22 / 32 | 5.5 | **5.8x** | none |
| Q5_1 | 24 / 32 | 6.0 | **5.3x** | `moe_indexed` only, no dense path |

### Why this is not a niche concern

These are not merely "legacy" types. **K-quants require the contraction axis to be a multiple of
`QK_K = 256`.** When a model's FFN or expert dimension is not, llama.cpp is *forced* onto exactly
these types for those tensors. DeepSeek-V2-Lite is the worked example: `moe_intermediate_size = 1408`,
so 14 of 26 `ffn_down_exps` banks are emitted as Q5_0, and dotLLM expands all of them — 53.6 GiB of
host F32 against ~31.6 GiB of OS-visible RAM on the Strix Halo box, i.e. an OOM.

So the trigger is not "old model" but "awkward dimension", which is common in MoE architectures.

A corollary worth stating: **a `*_K_M` filename does not mean K-quant tensors throughout.**

## Goals

1. Every quantisation type dotLLM can load is held on the Vulkan device in its **packed** form.
2. Decode and prefill run through real quantised kernels, not a dequantise-to-F32 fallback.
3. An F32 expansion, where one still occurs, is **reported** rather than silent.

## Non-goals

- CPU or CUDA kernel work. CUDA's residency model differs (see "Cross-backend note").
- New quantisation *formats*. This is coverage of formats we already parse.
- Changing the `Keep*OnDevice` 256-alignment gates for K-quants. Those are defensive and, by GGUF
  construction, unreachable for a genuine K-quant tensor.

## Decomposition

Six units. Unit 0 first; units 1-5 are mutually independent and may proceed in parallel.

### Unit 0 — routing and expansion visibility (prerequisite)

The silent `return F32` is the reason this went unnoticed for so long. Unit 0 makes expansion
observable, and is **the measurement instrument every later unit depends on** — it is how a unit
proves its kernel is genuinely routed to, rather than inferring it from a capability flag.

- Report at load: each tensor expanded to F32, its source type, packed bytes vs uploaded bytes, and a
  total. Zero cost when nothing expands.
- Fix the routed-MoE skip from **model-global to per-bank** (issue #327). Today one unsupported
  sibling forces every bank in the model to F32; in the DeepSeek Q4_K_M case 64 of 78 banks are
  already resident-capable, so per-bank resolution alone cuts the fallback ~5x **with no new kernels**.
- Test: for a model whose types are all supported, assert **zero** tensors expand.

**Acceptance:** the expansion report is accurate on a known-mixed model, and the per-bank change is
demonstrated to reduce DeepSeek-V2-Lite's fallback without altering numerics.

### Units 1-5 — one quantisation type each

Ordered by real-world impact:

| # | type | rationale | MoE-indexed needed? |
|---|---|---|---|
| 1 | **Q5_0** | unblocks DeepSeek-V2-Lite / Coder-V2-Lite; real model on box | **yes** — appears as expert banks |
| 2 | **MXFP4** | GPT-OSS family; real model on box; CUDA has a dequant kernel to cross-check | no |
| 3 | **Q4_0** | most common legacy dense type | if observed |
| 4 | **Q4_1** | less common | if observed |
| 5 | **Q5_1** | least common; `moe_indexed` already exists, dense path missing | already present |

Per type, the kernel family is:

```
{q}_dequant_f32           fallback + test oracle
matmul_{q}_f32_gemv       decode, S == 1
matmul_{q}_f32_gemm       prefill
matmul_{q}_mmvq           decode, dp4a
matmul_{q}_mmq            prefill, dp4a          [measurement-gated, see below]
moe_indexed_matmul_{q}_*  routed expert banks    [only where the type appears as one]
```

plus the `Keep{Q}OnDevice` predicate and its `DeviceQuantTypeFor` wiring, without which the kernels
are unreachable.

## Kernel templates

Two families, so these are adaptations rather than 25 from-scratch kernels:

- **Symmetric** — Q4_0 `d*(q-8)`, Q5_0 `d*(q-16)`, MXFP4 `table[q] * 2^e`.
  Template: **Q8_0 / Q6_K**.
- **Asymmetric** — Q4_1 and Q5_1 `d*q + m`.
  Template: **Q4_K / Q5_K**, which already solve the min-offset problem in a dp4a accumulation
  (the `sum(activations) * m` term must be carried alongside the integer dot product).

MXFP4 specifics: an E8M0 power-of-two scale byte plus E2M1 nibbles resolved through a 16-entry value
table. llama.cpp stores that table doubled (`kvalues_mxfp4`); the CPU port in
`Dequantize.cs` already follows this and the shader must match it exactly.

### Highest-risk surface

**Q5_0 and Q5_1 carry the 5th bit of each weight in a separate `qh` field**, so the mapping from
element index to (nibble position, `qh` bit) is exactly the class of indexing that Q3_K got wrong —
transposed, self-consistently, across every backend, for months. These two get the most review
attention and the most explicit real-bytes coverage.

## Verification standard

This section is deliberately prescriptive; it encodes failures this project has already paid for.

1. **Real GGUF bytes are the primary oracle.** Follow `RealGgufQ3KDequantParityTests`
   (PRs #321 / #340): read tensors of the type under test straight out of a real llama.cpp-quantised
   GGUF and compare Vulkan against the CPU path.
2. **A self-authored fixture may never be the sole oracle.** Q3_K shipped broken because `Q3KFixture`
   encoded with the same transposed layout the kernels decoded with — a closed loop that never touched
   real bytes. Fixtures remain useful for shapes, tails and edge cases; they do not establish
   correctness.
3. **Every type ships a negative control.** The test must be shown to *fail* against a deliberately
   broken implementation. A test that cannot fail proves nothing, whatever colour it reports.
4. **Assert the fixture actually contains the type.** mradermacher's SmolLM-135M `i1-Q3_K_M` contains
   no Q3_K tensor at all. A filename is not evidence of a quantisation type; list the types found when
   the assertion fires.
5. **Anchor the CPU oracle per type.** `Q3_K` is currently the *only* dequant type with a literal
   llama.cpp transcription test. Each unit adds the equivalent for its type over dense random blocks.
   Anchors are written in parallel with kernels, but **a type's anchor must merge before that type's
   kernel PR merges** — otherwise an unverified oracle certifies merged kernels.
6. **Full builds only.** Never `--no-build`. `.spv` shaders load from the repo tree at runtime while
   CPU kernels are compiled into `DotLLM.Cpu.dll`, so a stale build yields a *half-updated* system and
   a large, convincing, fictitious divergence (issue #341).
7. **Prove routing by observation, not capability flags.** Use Unit 0's expansion report, or output
   perturbation. `IsSupported` says a path exists, not that it ran.

### Buffer lifetime hazard for test authors

The kernels' `DescriptorSetCache` is keyed on raw Vulkan buffer handles, and Vulkan recycles handles.
Allocating and freeing a buffer per loop iteration can return a descriptor set still bound to a dead,
smaller buffer; writes past its extent drop to **zero**, which reads convincingly like kernel
truncation. **Allocate once at maximum size and reuse.**

## Fixtures

| type | source |
|---|---|
| MXFP4, Q5_0 | already present on the Strix Halo box |
| Q4_0, Q4_1, Q5_1 | generate with `llama-quantize` into `~/.dotllm/quant-ladder/` |

Model weights never enter the repository; fixtures resolve through `TestFixtureResolver`.

## Performance policy

Land `dequant / GEMV / GEMM / MMVQ` per type. **The MMQ tier is measurement-gated**: add it only
where a same-session, order-reversed A/B shows a real gain on gfx1151; otherwise GEMM remains the
default and MMQ stays opt-in.

The justification is empirical, not conservatism: issues #384-#391 tested four MMQ tiling variants,
including the exact llama.cpp configuration, and every one regressed on this hardware — the most
faithful attempts regressing hardest. Assuming MMQ is a win here contradicts the measurements.

GEMV should follow the coalesced-decode pattern established by the #338/#339 campaign, which is the
largest decode win recorded in this backend.

Benchmarking rules that apply: acquire `scripts/gpu-lock.sh`; use same-session order-reversed A/B
(cold-vs-warm launches on small shapes can show 2-3x phantom deltas from GPU clock ramp); on this UMA
box absolute decode tok/s swings ~40% with CPU memory-bandwidth contention.

## Acceptance criteria (whole effort)

- [ ] No supported quantisation type expands to F32 on Vulkan for a well-formed model.
- [ ] Any remaining expansion is reported at load with source type and byte cost.
- [ ] DeepSeek-V2-Lite Q4_K_M loads on Vulkan without the host-F32 blow-up.
- [ ] Each type has: real-GGUF-bytes parity, a proven negative control, and a llama.cpp-anchored CPU
      oracle merged no later than its kernels.
- [ ] Full Vulkan GPU suite green on each PR.

## Risks

| risk | mitigation |
|---|---|
| 25+ shaders is a large correctness surface | per-type PRs; two shared templates; one shared real-bytes test harness |
| Q5_0/Q5_1 `qh` bit indexing repeats the Q3_K error | real-bytes oracle + negative control; explicit review focus |
| An unanchored CPU oracle certifies wrong kernels | anchor must merge before that type's kernels (item 5) |
| MMQ absorbs effort for no gain | measurement gate before writing it |
| Parallel agents contend on GPU / collide in git | `scripts/gpu-lock.sh`; never `git stash` (shared across worktrees) |

## Cross-backend note

CUDA does **not** need equivalent work, and cannot be used as an oracle here. It has no packed Q3_K
matmul at all — one entry point, `dequant_q3_k_f16`, decoding once at load into a persistent FP16 copy
consumed by type-agnostic cuBLAS (PR #330). Its residency model is fundamentally different, so
"Vulkan has kernel X" implies nothing about CUDA and vice versa. The only cross-backend reference for
a packed layout is the CPU scalar path.
