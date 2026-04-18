# DESIGN — Mamba-3 support in dotLLM

Branch: `feature/mamba-3` (this is the *real* one, for the Mamba-3 algorithm
from Lahoti et al., [arXiv 2603.15569](https://arxiv.org/abs/2603.15569),
published 16 Mar 2026, accepted to ICLR 2026).

> **Scope difference from `feature/nemotron-and-mamba-2`.** That branch
> implemented the Mamba-2 hybrid architecture used by NVIDIA's Nemotron-3
> family. Mamba-3 is a distinct, newer algorithm that **cannot be validated
> the same way**: there are no GGUF checkpoints on HuggingFace, no
> llama.cpp support, and the only known reference implementation is
> [VikramKarLex/mamba3-minimal](https://github.com/VikramKarLex/mamba3-minimal)
> (minimal PyTorch, ~800 lines, endorsed by Albert Gu) plus the
> unreleased authors' integration into `state-spaces/mamba`. The realistic
> first deliverable here is **kernel-level parity with the minimal PyTorch
> reference**, not end-to-end inference.

## 1. What's new in Mamba-3

Three core methodological improvements over Mamba-2, expressed concretely:

| # | Change | Effect |
|---|---|---|
| 1 | Trapezoidal discretization (Prop. 1 of the paper) | Adds a second input term to the recurrence; removes the need for conv1d |
| 2 | Complex-valued state update (applied as data-dependent RoPE on B, C) | Richer state tracking; no explicit complex arithmetic needed |
| 3 | MIMO (multi-input multi-output) formulation | Half the state size for equivalent perplexity; 2× faster decode |

Plus several architectural refinements (QK-normalization on B/C, learnable
BC bias, removal of input-side gating).

### 1.1 Trapezoidal recurrence (Eq. 9)

```
h_t = α_t · h_{t-1}  +  β_t · B̄_{t-1} · x_{t-1}  +  γ_t · B̄_t · x_t
```

where, per-token-per-head:

```
α_t = exp(dt · A)             # decay, same as Mamba-2
β_t = (1 - λ) · dt · α_t      # previous-step coefficient (NEW)
γ_t = λ · dt                  # current-step coefficient
λ   = sigmoid(λ_raw)          # trapezoidal interpolation param, per-head
```

Vs Mamba-2's `h_t = α · h_{t-1} + B̄_t · x_t` — Mamba-3 weights the
*previous* input as well as the current, giving a second-order
integration rule. This also **eliminates the depthwise conv1d** that
Mamba-2 uses — the β-term provides the cross-token memory that the conv
was emulating.

### 1.2 Complex state via RoPE on B, C

The complex-valued update rule is implemented as a data-dependent
rotation applied to B and C, rather than explicit complex arithmetic.
Per-head rotation angles are cumulative:

```
raw_angles[t, h, :] = dt[t, h] · θ[h, :]     # θ is learned, shape (n_heads, d_state/2)
cum_angles          = -cumsum(raw_angles, dim=time)
B, C                = rope_2d(B, C, cum_angles)   # 2D rotation per adjacent d_state pair
```

The 2D rotation on pairs of state dimensions absorbs what would
otherwise be a complex-valued state update — each pair `(x[2k], x[2k+1])`
rotates as `(cos·x[2k] - sin·x[2k+1], sin·x[2k] + cos·x[2k+1])`.

### 1.3 MIMO variant

When enabled, B and C become rank-R matrices (shape `(d_state, R)`) and
the SSM input is rank-expanded before the scan:

```
# SISO (Mamba-2 style):                       BX = outer(B, x)     → (P, d_state)
# MIMO (rank-R factored):  x_exp = x ⊙ mimo_x_proj[R,P]
                                             BX = B @ x_exp.T      → (P, d_state)
```

Same scan kernel; different B/C/x shapes at the input side.

### 1.4 Two-SSD decomposition

Implementation trick from the minimal reference: the trapezoidal update
is run as **two separate SSD calls** (one for the γ-term using current
`B_t, x_t`, one for the β-term using previous `B_{t-1}, x_{t-1}`), then
summed. This lets Mamba-3 reuse an unchanged Mamba-2 SSD kernel instead
of needing a custom two-input recurrence kernel. **We can do the same.**

## 2. Ecosystem status (as of 2026-04-18)

| Item | Status |
|---|---|
| Paper | ICLR 2026, arXiv 2603.15569, Mar 2026 |
| Official code | `state-spaces/mamba` repo (PyTorch) — production impl with Triton kernels |
| Minimal reference | `VikramKarLex/mamba3-minimal` — ~800 lines pure PyTorch, Albert Gu-endorsed |
| Alternative minimal | `yang3121099/mamba3` — another port (need to verify authenticity) |
| HuggingFace checkpoints | **None.** `state-spaces/*` org hosts Mamba-1/Mamba-2 but no Mamba-3 yet |
| GGUF converter | **None.** No `tensor_mapping.py` entries for Mamba-3 tensors in llama.cpp |
| llama.cpp arch support | **None** in current `master`. No `LLM_ARCH_MAMBA3` or equivalent |

This means **we cannot do the llama.cpp cross-reference validation that
the Nemotron-H work relied on**. Our options for correctness validation:

1. **Port tests against the minimal PyTorch reference.** Fix a random
   seed, run Mamba-3 forward in PyTorch on synthetic inputs, save the
   intermediate tensors to a `.npz` / `.bin` file, then have our C#
   unit tests load and compare against those fixtures. Feasible today.
2. **Wait for an official checkpoint + GGUF converter.** No known ETA.
3. **Port the state-spaces/mamba Triton kernels semantically and trust
   the reference.** Risky — complex math, easy to drift.

We use **option 1** for initial kernel work.

## 3. What dotLLM needs

### 3.1 Reuseable from `feature/nemotron-and-mamba-2`

All already merged on the base branch:
* `Mamba2SelectiveScan.Execute` — the inner-loop scan. **Mamba-3 calls
  this twice per layer** (γ and β decomposition), so it ships as-is.
* `SsmStateCache` — generalizes. Mamba-3 additionally needs a
  `prev_Bx` buffer per head, which is just another same-shape slot.
* `RmsNorm` — used for QK-normalization on B, C.
* `NemotronHForwardState` scratch pool pattern — reuse conceptually, not
  literally. A fresh `Mamba3ForwardState` is cleaner.
* `NemotronHDiagnostics` — the env-var-gated trace pattern applies as-is.

### 3.2 New kernels

| Kernel | Purpose | Complexity |
|---|---|---|
| `Mamba3Discretize` | Compute α, β, γ per token-head from dt, A, λ | Simple element-wise |
| `Mamba3DataRoPE` | 2D rotation on B and C using cumulative angles derived from dt and θ | Like standard RoPE but data-dependent (not position-based); per-pair rotation |
| `Mamba3QkNorm` | RMSNorm on B and C vectors | Trivial wrapper |
| `Mamba3MimoProject` *(MIMO variant)* | Rank-R expansion of x into SSM input; rank-R contraction of y | Matmul-shaped |
| `Mamba3SelectiveScan` | Orchestrator: two-SSD decomposition calling existing `Mamba2SelectiveScan` twice + sum | Thin wrapper |

### 3.3 No new architecture wiring *yet*

Until a checkpoint + GGUF format materialize, there is nothing to load
end-to-end. The work here is:
1. Kernels (stage A, below)
2. PyTorch reference-fixture test harness (stage B)
3. Mamba-3 block class that composes kernels the way the minimal
   reference does — not wired into `ModelLoader` since there are no
   models to load (stage C)

When checkpoints + GGUF support land, a follow-up branch adds the
`Architecture.Mamba3` enum, `Mamba3TransformerModel`, and the
`ModelLoader` dispatch — mirroring what Stage 3 of the Nemotron-H plan did.

## 4. Staged plan

Commit-granular; each stage builds green and adds tests.

### Stage A — Kernels, each validated against a scalar Python reference

1. **`Mamba3Discretize`** — compute α, β, γ from dt, A, λ per (t, h).
   Unit tests: scalar reference; a λ=1 sanity check (recovers
   Mamba-2's first-order discretization when β→0, γ→dt).

2. **`Mamba3DataRoPE`** — data-dependent 2D rotation on a
   `[T, n_group, d_state]` tensor using angles derived from
   `cumsum(-dt[:, h] * θ[h, :])`. Unit tests: identity when θ=0;
   recoverable rotation when θ is piecewise constant.

3. **`Mamba3QkNorm`** — trivial, existing `RmsNorm.Execute` in a loop;
   wrap for clarity only.

4. **`Mamba3SelectiveScan`** — orchestrator that:
   - computes α, β, γ (kernel 1)
   - data-RoPE's B, C (kernel 2)
   - calls `Mamba2SelectiveScan` twice (once for γ-path with `B_t · x_t`,
     once for β-path with `B_{t-1} · x_{t-1}`)
   - sums the two outputs
   Unit tests: tiny hand-computed example (n_head=1, d_state=2,
   head_dim=2, T=3); matches a pure-scalar Python reference within 1e-4.

5. *(MIMO variant)* — `Mamba3MimoProject` + `Mamba3SelectiveScan`
   `useMimo=true` path. Unit tests: rank-1 MIMO degenerates to SISO.

### Stage B — PyTorch reference fixtures

**Schema decisions (derived from reading `.mamba3-reference/mamba3-minimal/mamba3.py`):**

The Mamba-3 block input projection produces seven splits, not five:
`z, x, B, C, dt_raw, lam_raw, theta` — where `theta` (shape `[T, dState/2]`)
is **per-token data-dependent**, not a learned per-head table. `dt` is then
`softplus(dt_raw + dt_bias)`, `lam` is `sigmoid(lam_raw)`. `A` comes from
`-exp(A_log)` (per-head, always negative). BC-bias is learned per-head,
per-channel (init 1.0), added *after* QK-Norm, *before* RoPE. For SISO the
scan input is `B` shape `[T, nHead, dState]` (post-bias-broadcast) and our
`Mamba3SelectiveScan` accepts `nGroup=nHead` in that layout.

**Fixture capture points for the Stage B golden file:**
1. Inputs: `u` (post-pre-norm hidden), all weights (`in_proj`, `A_log`,
   `dt_bias`, `B_bias`, `C_bias`, `D`, `out_proj`, `B_norm.weight`,
   `C_norm.weight`).
2. Post-split intermediates: `z, x, B_raw, C_raw, dt_raw, lam_raw, theta`.
3. Post-activation: `dt` (softplus), `lam` (sigmoid).
4. Computed discretization: `dA, alpha, beta, gamma`.
5. Post-QK-Norm: `B_qkn, C_qkn`.
6. Post-BC-bias add (per-head broadcast): `B_biased, C_biased`.
7. Cum angles: `cum_angles` shape `[T, nHead, dState/2]`.
8. Post-RoPE: `B_roped, C_roped` shape `[T, nHead, dState]`.
9. Post-scan: `y_scan` (pre-D), `ssm_state`, `last_Bx`.
10. Post-D + gate: `y_gated` (after `y + x*D` and `y * silu(z)`).

**Format:** single `fixture.json` with tensor name → flat F32 array +
shape tuple. Simple for C# to parse (no NPZ/zip machinery). Config:
tiny model (d_model=8, nheads=2, headdim=4, dState=4, seqLen=4, SISO
first then MIMO R=2).



Script (Python) that runs `VikramKarLex/mamba3-minimal` on a fixed seed
and dumps:
- A single-layer forward's input (hidden state, weights) as `.bin` files
- Intermediate tensors after each step (discretize output, post-RoPE
  B/C, scan output, final y) as `.bin` files

Corresponding C# fixture loader + integration test that reads the
.bin files, feeds the input into our kernels, and asserts byte-equal
outputs to 4 decimal places. This is the direct analogue of what we did
for Mamba-2 via llama-eval-callback, just with a PyTorch reference
instead of a C reference.

### Stage C — `Mamba3Block` orchestrator (no model loading)

A `Mamba3Block` class that composes the kernels in the order the
minimal reference does:

```
    RmsNorm → input-projection (GEMM) → split (z, x, B, C, dt, λ, θ)
           → QK-Norm(B, C) → + BC-bias → DataRoPE(B, C)
           → SelectiveScan → y·SiLU(z) → RmsNorm → output-projection
```

Followed by a separate `SwiGLU-MLP` block that's just the existing
FFN wiring. No `Mamba3TransformerModel`, no GGUF loading — this is a
composable primitive, validated via a third PyTorch fixture test that
runs a full layer.

### Stage D (deferred) — when a checkpoint exists

- `Architecture.Mamba3` enum value
- GGUF tensor-name mapping (needs to be reverse-engineered from whichever
  converter the first checkpoint uses)
- `Mamba3TransformerModel : IModel` analogous to `NemotronHTransformerModel`
- `ModelLoader.LoadFromGguf` dispatch

## 5. Open questions to confirm before starting

1. **Use `VikramKarLex/mamba3-minimal` or `yang3121099/mamba3` as the
   reference?** The former is explicitly Albert Gu-endorsed; defaulting
   to it unless we find a reason otherwise.
2. **Do we care about the MIMO variant for the initial deliverable,
   or SISO-only?** Paper says MIMO gives the extra 1.2pp accuracy;
   implementation complexity is low (rank-R expansion is one extra
   matmul). Default: implement both in stage A, test both in stage B.
3. **How strict should the fixture-comparison tolerance be?** PyTorch
   vs our scalar C# on the same F32 weights should match to ~1e-5.
   Proposed: 1e-4 element-wise absolute tolerance + 1e-3 relative.

## 6. Risk assessment

| Risk | Mitigation |
|---|---|
| Minimal PyTorch reference has subtle bugs or diverges from the paper | Cross-check key equations against the published paper; spot-check outputs against `state-spaces/mamba` full impl when it becomes easy to install |
| RoPE-on-BC convention varies between references | Lock to the VikramKarLex convention; document the angle-sign and pair-ordering choices |
| Complex state is actually complex arithmetic in some impls | Paper and minimal reference both express it as real-valued 2D rotation — we follow that; re-verify once the official code is reproduced |
| Porting without a checkpoint means latent bugs won't be caught end-to-end | Stage B's PyTorch-fixture tests catch numerical drift at the layer level; stage D will catch any architecture-level bugs once a checkpoint lands |
| Scope creep: MIMO + SISO + optimization kernels | Stage A first (SISO correctness), MIMO as optional stage A5, SIMD later under a separate branch |

## 7. Done criteria

**For this branch to be considered complete:**

* All four Stage-A kernels land with passing unit tests.
* Stage-B PyTorch fixture comparison test is green: our C# Mamba-3 scan
  matches the minimal reference's layer output element-wise within
  `1e-4` abs / `1e-3` rel on a 64-token synthetic input.
* Stage C `Mamba3Block` composes the kernels correctly and passes a
  full-layer reference comparison.
* `DESIGN_MAMBA_3.md` (this file) is updated with any convention choices
  that turn out to differ from the paper abstract.

**Stage D is a separate branch** once checkpoints exist. Do not block
this branch on upstream checkpoint availability.
