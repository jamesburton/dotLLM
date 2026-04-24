# Supported Models

Authoritative matrix of every architecture variant currently declared by the
`DotLLM.Core.Configuration.Architecture` enum, cross-referenced with the
loader dispatch in `ModelLoader`, the field consumption in
`HfConfigExtractor` / `GgufModelConfigExtractor`, and the verification
evidence in the test tree.

**How to read.** Each row is one `Architecture` enum variant. The `Verified
on` column reports the strongest evidence that exists today, tagged per the
legend at the bottom of this page. Rows tagged `planned` have an enum value
but no working load-and-forward path. For deeper detail on a given row see
its subsection below the matrix and the linked loader source.

**Contributing a new architecture.** Add the enum variant in
`src/DotLLM.Core/Configuration/Architecture.cs`, wire `ResolveArchitecture`
in `src/DotLLM.Models/SafeTensors/HfConfigExtractor.cs` (or
`GgufModelConfigExtractor` for GGUF-first), add a loader method under
`src/DotLLM.Models/Architectures/`, extend the dispatch in
`ModelLoader.LoadFromSafetensors` / `ModelLoader.LoadFromGguf`, then add a
tiny-random integration test under
`tests/DotLLM.Tests.Integration/Models/Loaders/` and a row here. See
`docs/MODEL_CONFIG.md` for the config plumbing and `docs/ARCHITECTURE.md`
for system context.

## Matrix

| Architecture | Enum | Tokenizer | RoPE | KV-cache | MoE | Required config fields | Verified on | Notes |
|---|---|---|---|---|---|---|---|---|
| Meta Llama | `Architecture.Llama` | HF tokenizer.json (BPE + ByteLevel) | RoPE Norm (interleaved pairs) | GQA | no | `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `vocab_size`, `max_position_embeddings`, `rope_theta`, `rms_norm_eps`, `tie_word_embeddings` | `verified: real weights` — TinyLlama-1.1B-Chat-v1.0 (2.1 GB, `TinyLlama_11B_LoadsAndForwardsEndToEnd`) | Dense SwiGLU, standard `q/k/v/o_proj`, `mlp.{gate,up,down}_proj`. GGUF `general.architecture = llama`. |
| Mistral AI | `Architecture.Mistral` | HF tokenizer.json (BPE + ByteLevel or SPM) | RoPE Norm | GQA, optional `sliding_window` | no | Same as Llama plus optional `sliding_window` | `verified: tiny-random` (config only) — `HfConfigExtractorTests.Mistral_UsesNormRoPE` | Routes through the same `LoadLayer` path as Llama; GGUF `general.architecture = mistral` (or `mistral3`). End-to-end forward on real Mistral weights is untested in CI. |
| Microsoft Phi | `Architecture.Phi` | HF tokenizer.json (BPE + ByteLevel) | RoPE NeoX (non-interleaved) | GQA | no | Llama set plus `architectures[0]` starting with `phi` or `model_type` in `{phi, phi2, phi3}`. `tie_word_embeddings` defaults to true. | `verified: real weights` — Phi-3.5-mini-instruct (7.6 GB, `Phi35Mini_LoadsAndForwardsEndToEnd`) | Fused tensors: `self_attn.qkv_proj.weight [Q+K+V, H]` and `mlp.gate_up_proj.weight [2*I, H]` are split at load time into independent F32 slabs (see `SplitFusedProjection`). |
| Alibaba Qwen | `Architecture.Qwen` | HF tokenizer.json (BPE + ByteLevel) | RoPE NeoX | GQA, optional `sliding_window` | no | Llama set plus optional `head_dim` (Qwen3). Qwen2/3 commonly ship `q/k/v` biases; Qwen3 ships per-head `q_norm`/`k_norm` RMSNorms. | `verified: real weights` — Qwen2.5-0.5B (999 MB, `Qwen25_0_5B_LoadsAndForwardsEndToEnd`) | `ResolveOptionalBias` picks up Qwen2 biases; `ResolveOptionalNorm` picks up Qwen3 QK-norms. `tied_embeddings=true` typical for small SKUs. |
| DeepSeek (legacy) | `Architecture.DeepSeek` | GGUF tokenizer only (no HF safetensors dispatch) | RoPE | GQA (no MLA) | no | GGUF `general.architecture` in `{deepseek, deepseek2}` | `planned` — no working forward path. `TransformerArchitecture.CreateModel` throws `NotSupportedException("DeepSeek models require Multi-Latent Attention (MLA) which is not yet implemented.")` | Kept for GGUF metadata parsing back-compat. New checkpoints land on `DeepSeekV2`/`DeepSeekV3`; this variant is a placeholder. |
| DeepSeek-V2 | `Architecture.DeepSeekV2` | HF tokenizer.json (BPE + ByteLevel) | RoPE Norm + YaRN softmax mscale² (frequency rescaling deferred) | MLA (low-rank Q/KV + decoupled RoPE) + Phase A expanded KV-cache | yes: routed + multi-shared expert; `first_k_dense_replace` dense prefix | Llama set plus MLA block (`kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`, optional `q_lora_rank`), MoE block (`n_routed_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `n_shared_experts`, `first_k_dense_replace`), optional `rope_scaling` | `verified: real weights` — DeepSeek-V2-Lite (30 GB, `DeepSeekV2Lite_LoadsAndForwardsEndToEnd`, commit `5ff5312`) plus `yujiepan/deepseek-v2-tiny-random` config-level test | Attention routes through `LoadDeepSeekMlaLayer` → `MlaLayerWeights`; scalar `MlaAttention` kernel with optional cache pointers. `MlaExpandedKvState` caches expanded K_nope / V / K_pe across calls. Latent compression + `W_UK` absorption = Phase B (open). |
| DeepSeek-V3 | `Architecture.DeepSeekV3` | HF tokenizer.json (BPE + ByteLevel) | RoPE Norm with optional YaRN | MLA | yes: sigmoid-router top-k, multi-shared expert | Same as DeepSeek-V2. Router discriminator is `architectures[0] = DeepseekV3ForCausalLM` or `model_type = deepseek_v3`. | `verified: tiny-random` — `yujiepan/deepseek-v3-tiny-random`, `katuni4ka/tiny-random-deepseek-v3` (same test) | Same attention path as V2; V3's node-level aux-loss-free routing + sigmoid scoring are V3-specific MoE refinements. No real-weight forward run in CI. |
| NVIDIA Nemotron-H | `Architecture.NemotronH` | GGUF BPE (via `GgufBpeTokenizerFactory`) | RoPE on attention layers, none on SSM layers | GQA per attention layer; per-layer SSM state cache | no | GGUF `general.architecture = nemotron_h` with per-layer `head_count_kv` + `feed_forward_length` arrays (hybrid layout), plus SSM config keys (`ssm.*`) | `planned`/`verified: real weights (gated)` — `NemotronHTextGeneratorTests` run end-to-end against a local Nemotron-3-Nano-4B Q4_K_M GGUF when `DOTLLM_NEMOTRON_H_GGUF` is set; CI does not pull the checkpoint. Config-level detection is covered by `GgufModelConfigExtractorTests`. | **GGUF-only** — safetensors dispatch in `ModelLoader.LoadFromSafetensors` does NOT enumerate `NemotronH`. Activation function is `ReluSquared`, not SiLU. Dedicated `NemotronHTransformerModel` with a hybrid Mamba-2 / attention forward. |
| Mamba-3 (pure SSM) | `Architecture.Mamba3` | HF tokenizer.json (SPM + Metaspace + ByteFallback for `ib-ssm`) | Data-dependent RoPE on B/C inside the SSM block | SSM state cache (no KV-cache) | no | `model_type = mamba3`, `hidden_size`, `num_hidden_layers`, `num_heads`, `head_dim`, `state_size`, `vocab_size`, `expand`, `n_groups`, `chunk_size`, `mimo_rank`, `is_mimo`, `is_outproj_norm`, `use_l2warp`, `tie_word_embeddings`, `rescale_prenorm_residual`, `rope_fraction` | `verified: real weights` — `ib-ssm/mamba3-370M-10BT` (1.55 GB, `IbSsmMamba3RealWeightsLoadTests`, gated on `DOTLLM_IBSSM_CHECKPOINT_PATH`); `verified: tiny-random (synthetic)` — `TinyMamba3SafetensorsLoadTests` synthesises a deterministic miniature on disk; MIMO forward exercised by `Mamba3TransformerModelMimoTests` synthetic fixtures | Non-MIMO real-weight + MIMO synthetic (commit `0499465`). No public MIMO checkpoint — real-weight MIMO verification deferred P4.3. No upstream GGUF mapping. |
| Mistral Mixtral | `Architecture.Mixtral` | HF tokenizer.json | RoPE Norm | GQA | yes: `block_sparse_moe.gate` + per-expert `experts.{j}.w{1,2,3}` | Llama set plus `num_local_experts`, `num_experts_per_tok`. `architectures[0] = MixtralForCausalLM` or `model_type = mixtral`. | `verified: tiny-random` — `yujiepan/mixtral-tiny-random` (520 KB, `TinyMixtralSafetensorsLoadTests`) | Mixtral-convention MoE loader (`LoadMixtralMoeLayer`); no shared experts by design. Real-weight Mixtral / OLMoE 14 GB run tracked as PLANS.md P2.5. |
| Qwen-MoE (1.5 / 2 / 3) | `Architecture.QwenMoe` | HF tokenizer.json (BPE + ByteLevel) | RoPE NeoX | GQA, optional `sliding_window` | yes: `mlp.gate` + per-expert `experts.{j}.{gate,up,down}_proj`; optional shared-expert branch (Qwen1.5-MoE) with sigmoid gate; layer-level sparsity (Qwen3-MoE) via `decoder_sparse_step` + `mlp_only_layers` | Llama set plus `num_experts` or `num_local_experts`, `num_experts_per_tok`, `moe_intermediate_size`, optional `shared_expert_intermediate_size`, optional `norm_topk_prob`, optional `decoder_sparse_step`, optional `mlp_only_layers` | `verified: tiny-random` — `yujiepan/qwen3-moe-tiny-random` (20 MB, `TinyQwenMoeSafetensorsLoadTests`) and synthetic unit fixtures in `TransformerSafetensorsLoadTests` covering shared-expert + sigmoid-gate paths | `LoadQwenMoeLayer` resolves both singular `shared_expert.*` (Qwen1.5-MoE-A2.7B) and plural `shared_experts.{k}.*` (DeepSeek, reused). Real-weight Qwen1.5-MoE-A2.7B (14 GB) is gated but not exercised in CI. |
| IBM Granite-3.x MoE | `Architecture.GraniteMoe` | HF tokenizer.json (BPE + ByteLevel) | RoPE Norm | GQA | yes: fused per-layer `block_sparse_moe.{router.layer, input_linear, output_linear}` | Llama set plus `num_local_experts`, `num_experts_per_tok`, `moe_intermediate_size`. `architectures[0] = GraniteMoeForCausalLM` or `model_type = granitemoe`. | `verified: real weights` — `ibm-granite/granite-3.0-3b-a800m-instruct` (6.3 GB, `Granite3Moe_LoadsAndForwardsEndToEnd`) | Fused per-expert layout: `input_linear [E, 2*I, H]` packs w1 (rows `[0..I)`) + w3 (rows `[I..2*I)`), `output_linear [E, H, I]` packs w2. Each expert is upcast into its own F32 slab via `AllocPartAsF32`. No shared expert; typical top-k is unusually high (8 of 40). |

**Row count: 12 / 12 `Architecture` enum variants covered.**

## Per-architecture notes

### Llama (`Architecture.Llama`)
Dense SwiGLU transformer; `q/k/v/o_proj` split, optional biases off by
default, no QK-norm. GGUF path: `TransformerArchitecture` →
`TransformerModel.LoadFromGguf`. Safetensors path:
`TransformerWeightsSafetensorsLoader.LoadLayer` in
[`TransformerWeightsSafetensors.cs`](../src/DotLLM.Models/Architectures/TransformerWeightsSafetensors.cs).
See [docs/MODEL_CONFIG.md](MODEL_CONFIG.md) and [docs/ATTENTION.md](ATTENTION.md).

### Mistral (`Architecture.Mistral`)
Same tensor shape as Llama; the discriminator is `architectures[0]` /
`model_type`. Sliding-window attention is declared via `sliding_window`; the
kernel correctness for contexts longer than the window is covered by
PLANS.md P2.4. No end-to-end test in CI — config-only coverage.

### Phi (`Architecture.Phi`)
Phi-3 convention fuses `qkv_proj` and `gate_up_proj`; the safetensors loader
splits them into Q/K/V and gate/up via `SplitFusedProjection` so the forward
path is uniform with Llama/Mistral. Uses NeoX-style (non-interleaved) RoPE
pairs. See [docs/POSITION_ENCODING.md](POSITION_ENCODING.md).

### Qwen (`Architecture.Qwen`)
Dense Qwen2 / Qwen3 models. Qwen2 commonly ships `q/k/v_proj.bias`; Qwen3
additionally ships per-head `q_norm` / `k_norm` RMSNorm tensors. Both are
resolved optionally so Qwen2 weights load without Qwen3 tensors and vice
versa. Small SKUs tie embeddings (`tied_embeddings=true`).

### DeepSeek legacy (`Architecture.DeepSeek`)
Placeholder for older `general.architecture = deepseek` / `deepseek2` GGUF
labels. `TransformerArchitecture.CreateModel` explicitly throws
`NotSupportedException` because MLA was not wired at the time this enum
value was introduced. New work targets `DeepSeekV2` / `DeepSeekV3` instead.
This row exists as a capability claim audit: the enum member is visible
from the public API but there is no forward path.

### DeepSeek-V2 (`Architecture.DeepSeekV2`) and DeepSeek-V3 (`Architecture.DeepSeekV3`)
MLA attention (see [docs/ATTENTION.md](ATTENTION.md) and
`MlaConfig` in [`src/DotLLM.Core/Models/`](../src/DotLLM.Core/Models/)):
low-rank factorised Q (`q_a_proj` / `q_b_proj` — or monolithic `q_proj` on
the Lite variant when `q_lora_rank = 0`) and KV (`kv_a_proj_with_mqa`,
`kv_a_layernorm`, `kv_b_proj`) with decoupled RoPE on the `qk_rope_head_dim`
slice only. MoE side: multi-shared-expert (DeepSeek uses plural
`mlp.shared_experts.{k}.*` with no sigmoid gate), plus a dense-MLP prefix
for the first `first_k_dense_replace` layers that is folded into
`MlpOnlyLayers`. YaRN softmax mscale² correction is applied when
`rope_scaling.factor > 1` and `mscale_all_dim != 0` (commit `3091944`,
`MlaConfig.ComputeYarnSoftmaxScaleMultiplier()`); YaRN RoPE frequency
rescaling is deferred. **Phase A KV-cache** is active: `MlaExpandedKvState`
stores expanded per-head K_nope / V / shared K_pe across calls (commit
`6d2dc0b`), so autoregressive decode is no longer the O(N²) re-recompute
path the PoC was. Phase B (latent `[kv_lora_rank + qk_rope_head_dim]`
cache + `W_UK` absorption — the ~7× KV-memory win) is the current open
optimisation; Phase A is its numerical oracle (split-call matches
single-call within 1e-4).

### NemotronH (`Architecture.NemotronH`)
Hybrid Mamba-2 SSM + attention per-layer. Loaded **only** from GGUF —
`ModelLoader.LoadFromGguf` dispatches to `NemotronHTransformerModel`, and
`ModelLoader.LoadFromSafetensors` does not enumerate this arch. Config
parsing uses per-layer `head_count_kv` + `feed_forward_length` arrays
(hybrid layout) rather than the scalar keys; zero entries mark layers of
the "other" kind. Activation is ReLU-squared, not SiLU.

### Mamba3 (`Architecture.Mamba3`)
Pure SSM — no attention, no convolution. Safetensors-first (no upstream
GGUF mapping as of 2026-04). Loaded via `Mamba3ConfigExtractor` (HF config
parsing), `Mamba3WeightLoader`, and `Mamba3TransformerModel`. The
**non-MIMO** path is real-weight verified against `ib-ssm/mamba3-370M-10BT`
(1.55 GB) and synthetic-fixture verified via `TinyMamba3SafetensorsLoadTests`.
The **MIMO** path was blocked until P0.3 landed (commit `0499465`): the
weight loader now resolves `mimo_x` / `mimo_z` / `mimo_o` tensors and
`[H, R, N]`-shaped B_bias / C_bias, `Mamba3TransformerModel.Forward`
dispatches to `ForwardMimo`, and `Mamba3TransformerModelMimoTests`
exercises end-to-end MIMO forward on a synthetic fixture. Real-weight
MIMO verification is indefinitely deferred (P4.3) because no public MIMO
checkpoint exists.

### Mixtral (`Architecture.Mixtral`)
Dense transformer with top-k MoE FFN in every layer. Discriminator is
`architectures[0] = MixtralForCausalLM` / `model_type = mixtral` — checked
before the generic "mistral" substring match so that `mistralai/Mixtral-*`
repo names don't shadow it. Tensor-name convention:
`block_sparse_moe.gate` + `experts.{j}.(w1|w2|w3)`. No shared expert.
Routed by `LoadMixtralMoeLayer`.

### Qwen-MoE (`Architecture.QwenMoe`)
Covers Qwen1.5-MoE-A2.7B, Qwen2-MoE, and Qwen3-MoE. Tensor naming follows
HF Llama convention (`mlp.experts.{j}.{gate,up,down}_proj`), not Mixtral's
`w1/w2/w3`. Qwen1.5 ships a singular `mlp.shared_expert.*` branch
optionally gated by a sigmoid over `shared_expert_gate.weight`; Qwen3
drops the shared branch entirely but adds layer-level sparsity via
`decoder_sparse_step` + `mlp_only_layers` — dense Qwen-MoE layers fall
through to the standard Llama SwiGLU loader. `LoadQwenMoeLayer` is also
reused by the DeepSeek FFN path (with plural `mlp.shared_experts.{k}.*`).

### GraniteMoe (`Architecture.GraniteMoe`)
Fused-per-expert layout: all experts of one layer live in three rank-3
tensors (`router.layer`, `input_linear`, `output_linear`). The loader
(`LoadGraniteMoeLayer`) slices expert slabs out of `input_linear`
(`[E, 2*I, H]` — w1 top half, w3 bottom half) and `output_linear`
(`[E, H, I]` — w2), allocating per-expert F32 buffers via `AllocPartAsF32`.
Unusually high top-k (8 of 40 on the 3B-A800M SKU). No shared expert.

## Legend

| Tag | Meaning |
|---|---|
| `verified: real weights` | A real HuggingFace checkpoint is downloaded (off-CI, gated on env var or conventional path) and `Forward` returns finite logits with non-zero stddev. Exact test name + checkpoint cited. |
| `verified: tiny-random` | A tiny-random HF checkpoint (KB–MB, weights are random but shapes match the architecture) is fetched by the test runner into `~/.dotllm/test-cache/`, config is parsed and asserted, and — where the tiny-random ships usable weights — a forward pass runs. |
| `verified: tiny-random (synthetic)` | The test builds a deterministic miniature checkpoint on disk at run time because no public tiny-random exists (Mamba-3 is the sole case). |
| `planned` | Enum variant exists but no working load-and-forward path. Tagged when the loader throws `NotSupportedException` or no dispatch arm covers the variant. |

Evidence citations point to test types under
[`tests/DotLLM.Tests.Integration/Models/Loaders/`](../tests/DotLLM.Tests.Integration/Models/Loaders/)
and
[`tests/DotLLM.Tests.Integration/Engine/`](../tests/DotLLM.Tests.Integration/Engine/).

## See also

- [docs/MODEL_CONFIG.md](MODEL_CONFIG.md) — `ModelConfig` schema and the parameterised architecture pattern
- [docs/ARCHITECTURE.md](ARCHITECTURE.md) — system data flow
- [docs/GGUF_FORMAT.md](GGUF_FORMAT.md) — GGUF key conventions consumed by `GgufModelConfigExtractor`
- [docs/ATTENTION.md](ATTENTION.md) — GQA / MLA kernel selection
- [docs/POSITION_ENCODING.md](POSITION_ENCODING.md) — RoPE Norm vs NeoX pair conventions
- [docs/ROADMAP.md](ROADMAP.md) — step-by-step plan
- [PLANS.md](../PLANS.md) — outstanding gaps on `feature/mamba-3`
