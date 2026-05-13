---
title: "Qwen3.6-35B-A3B: Gated DeltaNet + MoE Hybrid Architecture"
date: 2026-05-12
context: "Architecture research for Qwen3.6-35B-A3B GGUF benchmark — qwen35moe arch string"
---

## Summary

Qwen3.6-35B-A3B uses the **Gated DeltaNet (GDN)** SSM variant (NVlabs, ICLR 2025, arXiv:2412.06464)
in a hybrid architecture combined with MoE FFN. It is **not** Mamba-2 and **not** Mamba-3.
The Qwen3 technical report is arXiv:2505.09388.

GGUF arch string: `qwen35moe`
dotLLM enum: `Architecture.Qwen3MoeHybrid` (new)

---

## Layer Alternation Rule

Layers alternate between two types, controlled by `full_attn_interval` (default 4):

- **GDN layers** (3 of every 4): linear attention via Gated DeltaNet recurrence
- **Full-attention layers** (every 4th): standard GQA with sigmoid output gate

Both layer types share the **same MoE FFN sublayer**. The SSM and attention paths are **never
parallel within the same token** — they are strictly layer-level alternates.

For the 35B-A3B model: 40 total layers → 30 GDN + 10 full-attention.

---

## Per-Block Structure (for both GDN and full-attention layers)

```
input
  │
  ├── pre_norm (attn_norm.weight — RMSNorm)
  │     │
  │     ├─ [GDN layers]    → GDN recurrence (ssm_* tensors)
  │     └─ [Attn layers]   → GQA + sigmoid gate (attn_qkv.weight + attn_gate.weight)
  │
  ├── residual add
  │
  ├── post_attention_norm (post_attention_norm.weight — RMSNorm before FFN)
  │
  ├── MoE FFN (ffn_gate_inp + experts + shared expert + shared expert gate)
  │
  └── residual add
```

The dual-norm pattern (`attn_norm` before attention/GDN, `post_attention_norm` before FFN) is
similar to Gemma2's post-attention norm design.

---

## Gated DeltaNet Recurrence (per GDN layer)

State shape per head: `[d_k, d_v]` — a full matrix (not diagonal/vector as in Mamba-2).

```
// Projections (all from hidden state h):
g      = exp(softplus(alpha_proj(h) + dt_bias) * A)   // decay factor, shape [n_head]
k      = conv1d(key_proj(h))                            // key, shape [n_head, d_k], causal conv
v      = value_proj(h)                                  // value, shape [n_head, d_v]
q      = query_proj(h)                                  // query, shape [n_head, d_k]
beta   = sigmoid(beta_proj(h))                          // write gate, shape [n_head]

// Recurrence:
state  = g * state                                      // decay: state[n_head, d_k, d_v]
retrieved = einsum('hkv,hk->hv', state, k)             // read: shape [n_head, d_v]
delta  = beta * (v - retrieved)                         // error signal: shape [n_head, d_v]
state  = state + einsum('hk,hv->hkv', k, delta)        // delta-rule write
output = einsum('hkv,hk->hv', state, q) / sqrt(d_k)   // read output
```

Key differences from Mamba-2:
- State is a full `[d_k × d_v]` matrix per head (associative memory)
- Delta-rule outer-product write (not SSD scan / DPLR)
- No discretization / continuous-time structure
- Conv1d applied only to K (not the full state projection)
- No Z gating term

---

## Tensor → Component Mapping (GGUF names for blk.N)

### GDN layers

| GGUF key | Component | Notes |
|---|---|---|
| `blk.N.attn_norm.weight` | Pre-norm RMSNorm | Before GDN |
| `blk.N.ssm_a` | Log-space decay magnitude A | Per-head scalar vector, no `.weight` suffix |
| `blk.N.ssm_alpha.weight` | Decay projection `alpha_proj` | `[hidden, n_head]` → softplus + dt_bias + A → exp → g |
| `blk.N.ssm_beta.weight` | Write gate projection `beta_proj` | `[hidden, n_head]` → sigmoid → β |
| `blk.N.ssm_conv1d.weight` | Causal 1D conv on K | Applied after key projection, before GDN recurrence |
| `blk.N.ssm_dt.bias` | Delta-time bias | Per-head, added inside alpha computation |
| `blk.N.ssm_norm.weight` | SSM output RMSNorm | Applied to GDN output before residual |
| `blk.N.ssm_out.weight` | Output projection | `[d_inner, hidden]` |

Key/Query/Value projections are part of `ssm_out` or absorbed into alpha/beta — exact Q/K/V
projection tensor names need to be confirmed from llama.cpp source (`llama.cpp` `model.cpp`
Qwen35MoE block loader).

### Full-attention layers

| GGUF key | Component | Notes |
|---|---|---|
| `blk.N.attn_norm.weight` | Pre-norm RMSNorm | Same as GDN layers |
| `blk.N.attn_qkv.weight` | Fused QKV projection | GQA — split by head counts |
| `blk.N.attn_gate.weight` | Sigmoid output gate | Applied to attention output BEFORE W_o: `attn_out * sigmoid(gate)` |

Note: there is **no `attn_output.weight`** (W_o) in the GGUF. The output is either absorbed
into `attn_gate` or this is effectively a no-op projection. **Verify against llama.cpp source.**

### Shared by all layers (FFN)

| GGUF key | Component | Notes |
|---|---|---|
| `blk.N.post_attention_norm.weight` | Post-attn RMSNorm | Before MoE FFN |
| `blk.N.ffn_gate_inp.weight` | MoE router logits | `[hidden, n_experts]` |
| `blk.N.ffn_gate_exps.weight` | Expert gate projections | Stacked `[n_experts, intermediate, hidden]` |
| `blk.N.ffn_up_exps.weight` | Expert up projections | Stacked |
| `blk.N.ffn_down_exps.weight` | Expert down projections | Stacked |
| `blk.N.ffn_gate_shexp.weight` | Shared expert gate proj | Always-active expert, SwiGLU |
| `blk.N.ffn_up_shexp.weight` | Shared expert up proj | |
| `blk.N.ffn_down_shexp.weight` | Shared expert down proj | |
| `blk.N.ffn_gate_inp_shexp.weight` | Shared expert scalar gate | `[hidden, 1]` → sigmoid → scales entire shared expert output |

---

## GDN State

- Shape per GDN layer per sequence: `[n_head, d_k, d_v]`
- This is a **matrix state** — much larger than Mamba-2's diagonal/vector state
- For 30 GDN layers, n_head=32, d_k=d_v=64: 30 × 32 × 64 × 64 × 4 bytes ≈ **15 MB per sequence**

A new `GdnStateCache` class is needed (different from `SsmStateCache` which holds vector states).

---

## What Exists in dotLLM

| Component | Status |
|---|---|
| `Conv1dCausal` (CPU + Vulkan) | ✅ Reuse for K preprocessing |
| `MoeSwiGluMlp` with shared expert | ✅ Extend with scalar gate |
| `HybridLayerLayout` | ✅ Extend with `GatedDeltaNet` kind |
| `MambaSsmConfig` | ➕ New `GatedDeltaNetConfig` needed |
| GDN recurrence kernel | ❌ New |
| CUDA SSM kernels | ❌ None — CUDA is transformer-only |
| Vulkan Mamba-2/3 kernels | ✅ Patterns exist, adapt for GDN |
| `post_attention_norm` | ❌ New (existing code has pre-norm only) |
| Shared expert scalar gate | ❌ `ffn_gate_inp_shexp` not yet wired |
| Gated attention (sigmoid on output) | ❌ New |

---

## Open Questions

1. **Exact Q/K/V projection tensor names for GDN layers** — the GGUF only shows `ssm_out.weight`
   for the output. Where are Q/K/V defined? Check llama.cpp `llama_model_load_tensors` for
   `qwen35moe` and the GDN block's `ssm_in` projection.

2. **`attn_output.weight` (W_o) for full-attention layers** — absent from GGUF. Is it absorbed
   into `attn_gate`, or is W_o present under a different name?

3. **Exact GGUF metadata keys for GDN hyperparameters** — `full_attn_interval`, `ssm.d_inner`,
   `ssm.n_head`, `ssm.d_k`, `ssm.d_v`. Check `qwen35moe` case in `llama.cpp:llama_model_hparams`.

---

## References

- **Gated Delta Networks: Improving Mamba2 with Delta Rule** — NVlabs, ICLR 2025. arXiv:2412.06464
- **Qwen3 Technical Report** — Alibaba, 2025. arXiv:2505.09388
- **llama.cpp source** — `src/llama.cpp`, search `qwen35moe` for hparams and tensor loading
