# Gemma-4 E2B (text tower): MatFormer + Per-Layer Embeddings — gap analysis & design

Date: 2026-07-08. Branch: `issue/gemma4-matformer-ple` (off `origin/dev`).

## 1. What E2B actually is (verified against the real config + HF transformers)

Source of truth: `google/gemma-4-E2B` `config.json` (fetched) + `transformers` `modeling_gemma4.py`
/ `configuration_gemma4.py` / `modeling_gemma3n.py` (PLE originated in Gemma 3n).

Wrapper `model_type: "gemma4"`, `architectures: ["Gemma4ForConditionalGeneration"]` (multimodal:
text + vision + audio). We target **only** the text tower, `text_config.model_type = "gemma4_text"`:

| field | value | dotLLM status |
|---|---|---|
| num_hidden_layers | 35 | ok |
| hidden_size | 1536 | ok |
| intermediate_size | 6144 | ok |
| num_attention_heads | 8 | ok |
| num_key_value_heads | 1 (**MQA**) | ok (`NumKvHeads=1`) |
| head_dim | 256 | ok |
| vocab_size | 262144 | ok |
| hidden_activation | gelu_pytorch_tanh (**GeGLU-tanh**) | ok (`ActivationFunction.GELUTanh` + `FusedOps.GeGLUTanh`) |
| sliding_window | 512 | ok (`SlidingWindowSize` / `PerLayerSlidingWindow`) |
| layer_types | 4:1 sliding:full (full at idx 4,9,…,34) | ok (`PerLayerSlidingWindow` + `IsFullAttentionLayer`) |
| rms_norm_eps | 1e-6 | ok |
| rope (sliding) | theta 1e4, default | ok (`RoPEConfig`) |
| rope (full) | theta 1e6, **partial_rotary_factor 0.25**, "proportional" | partial: `GlobalRoPEConfig`+`PartialRotaryFactor` (rope_type "proportional" scaling = TODO, see §5) |
| final_logit_softcapping | 30.0 | ok (`FinalLogitSoftcap`) |
| tie_word_embeddings | true | ok |
| four-norm layout + (1+w) RMSNorm | yes | ok (Gemma path) |
| **vocab_size_per_layer_input** | 262144 | **NEW (PLE)** |
| **hidden_size_per_layer_input** | 256 | **NEW (PLE)** |
| num_kv_shared_layers | 20 | **NOT DONE (KV sharing) — §5** |
| use_double_wide_mlp | true | **NOT DONE — §5** |
| enable_moe_block | false | dense path (existing Gemma4 support is the 26B-A4B **MoE** tower; E2B is DENSE) |

**Key finding:** dotLLM's existing Gemma3/Gemma4 dense infrastructure already covers almost the
entire E2B block (four norms, (1+w) absorption, MQA/GQA, GeGLU-tanh, sliding/full interleave, dual
RoPE, partial-rotary, softcapping, sqrt(hidden) embed scale, tied LM head). The **only genuinely new
piece is PLE**. Unlike Gemma 3n, the Gemma-4 text layer has **no AltUp and no Laurel** — PLE is a
clean gated residual added after the MLP block.

## 2. PLE — exact math (from `modeling_gemma4.py`)

Once per forward, from `input_ids` and the scaled main embedding `inputs_embeds` (= the sqrt(hidden)-
scaled token embedding dotLLM already computes in `EmbeddingLookup`):

```
# token-identity component — embed_tokens_per_layer is a ScaledWordEmbedding
per_layer_inputs = embed_tokens_per_layer(input_ids)          # gather, scale by sqrt(hidden_size_per_layer_input)=16
                     .reshape(seq, num_layers, ple_dim)        # ple_dim = 256
# context component
plp = per_layer_model_projection(inputs_embeds) * hidden_size**-0.5   # Linear hidden->num_layers*ple_dim
plp = plp.reshape(seq, num_layers, ple_dim)
plp = per_layer_projection_norm(plp)                           # RMSNorm over ple_dim (Gemma (1+w))
per_layer_inputs = (plp + per_layer_inputs) * rsqrt(2)         # [seq, num_layers, ple_dim]
```

Per decoder layer `i`, AFTER the MLP sublayer's post-FFN-norm + residual add (i.e. on the layer output):

```
residual = h
h = per_layer_input_gate(h)              # Linear hidden->ple_dim
h = act_fn(h)                            # gelu_pytorch_tanh
h = h * per_layer_inputs[:, i, :]        # elementwise, ple_dim
h = per_layer_projection(h)              # Linear ple_dim->hidden
h = post_per_layer_input_norm(h)         # RMSNorm over hidden (Gemma (1+w))
h = residual + h
```

Tensor names (HF module attrs; synthetic fixture uses `model.` prefix like the existing loader):
- `model.embed_tokens_per_layer.weight` [vocab_ple, num_layers*ple_dim]
- `model.per_layer_model_projection.weight` [num_layers*ple_dim, hidden]
- `model.per_layer_projection_norm.weight` [ple_dim]  (Gemma (1+w))
- `model.layers.{i}.per_layer_input_gate.weight` [ple_dim, hidden]
- `model.layers.{i}.per_layer_projection.weight` [hidden, ple_dim]
- `model.layers.{i}.post_per_layer_input_norm.weight` [hidden]  (Gemma (1+w))

## 3. MatFormer

E2B is a nested sub-network of E4B (MatFormer: selective layers/width). The published
`google/gemma-4-E2B` checkpoint ships the **already-extracted** 35-layer standalone sub-network —
its safetensors are a complete dense model, so E2B loads as a normal 35-layer model with no
selection logic. **Decision:** support the standalone-extracted E2B checkpoint (no MatFormer slicing
needed to load E2B). MatFormer elastic width/layer selection from an E4B checkpoint is documented as
follow-up (§5) — not required to serve E2B.

## 4. Reuse-vs-build map (file/class anchors)

| concern | reuse / build | anchor |
|---|---|---|
| four-norm + (1+w) + GeGLU-tanh + MQA + sliding/full + dual RoPE | **reuse** | `TransformerModel.RunLayersAndFinalNormCore` dense path; `LoadLayer`; `ModelConfig` Gemma fields |
| config: PLE fields | **build** | `PerLayerEmbeddingConfig` (new) + `ModelConfig.PerLayerEmbedding` |
| config: detect E2B dense-PLE tower | **build** | `HfConfigExtractor` (gemma4_text + hidden_size_per_layer_input, enable_moe_block false) |
| PLE weight slots | **build** | `PerLayerEmbeddingWeights` (new, model-level) + 3 fields on `TransformerLayerWeights` |
| PLE loader | **build** | `TransformerWeightsSafetensorsLoader.Load` / `LoadLayer` |
| PLE math (compute inputs + per-layer inject) | **build** | `DotLLM.Cpu.Kernels.PerLayerEmbeddings` (new); F32; reuses `MatMul.GemmF32`, `RmsNorm`, `FusedOps.GeGLUTanh` |
| forward wiring | **build** | `RunLayersAndFinalNormCore`: compute inputs post-embed; inject after dense residual add (line ~1522) |

## 5. Proven vs Designed vs TODO

- **Implemented + tested this pass:** PLE config record + extraction; PLE weight slots + safetensors
  loader; `PerLayerEmbeddings` CPU kernel (unit-tested numerically vs a hand-computed reference);
  forward wiring; a tiny synthetic E2B-like fixture proving load + finite forward + PLE-changes-output.
- **Designed, not wired this pass:** CUDA/Vulkan PLE (cross-backend, §6).
- **TODO for real-weight parity (documented, not blocking synthetic):**
  1. **num_kv_shared_layers=20** — the last 20 layers reuse earlier layers' K/V projections (no own
     k_proj/v_proj tensors). Loader + forward must resolve shared KV. Not exercised by the synthetic
     fixture (which gives every layer its own KV).
  2. **use_double_wide_mlp=true** — E2B MLP width handling; confirm gate/up layout vs `intermediate_size`.
  3. **RoPE "proportional" scaling** on full-attention layers (rope_type="proportional",
     partial_rotary_factor 0.25) — partial-rotary is wired; the "proportional" theta scaling is not.
  4. **Multimodal safetensors prefix** — the real `Gemma4ForConditionalGeneration` checkpoint prefixes
     text-tower tensors (e.g. `model.language_model.` / `language_model.model.`). The loader currently
     assumes `model.`. Pre-existing for the whole text tower, not PLE-specific.
  5. **MatFormer** elastic slicing from an E4B checkpoint.
  6. Numerical parity gate vs HF transformers on real gated weights (we have none here).

## 6. Cross-backend note (per CLAUDE.md)

CPU implemented first. For CUDA (`DotLLM.Cuda`) and Vulkan (`DotLLM.Vulkan`) PLE needs, per layer:
one small GEMM hidden->ple_dim, a GeGLU-tanh elementwise, an elementwise multiply by the precomputed
per-layer input slice, a GEMM ple_dim->hidden, an RMSNorm, and a residual add — all ops these
backends already have; only the precomputed `per_layer_inputs` buffer (gather + projection + norm,
computed once) must be uploaded to device or computed device-side. No new kernel primitives required.
The injection point is identical (after the MLP residual add). No change to the native C API.
