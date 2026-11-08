# Model Configuration — dotLLM

## ModelConfig Record

Comprehensive record describing any transformer variant. Populated from GGUF metadata at model load.

```
ModelConfig:
  Architecture          Llama | Mistral | Phi | Qwen | DeepSeekV2 | DeepSeekV3 | NemotronH | Lfm2Moe | Mamba3 | Mixtral | QwenMoe | GraniteMoe
  VocabSize             int
  HiddenSize            int
  IntermediateSize      int       (FFN intermediate dim)
  NumLayers             int
  NumAttentionHeads     int
  NumKvHeads            int       (== NumAttentionHeads for MHA, 1 for MQA, between for GQA)
  HeadDim               int       (typically HiddenSize / NumAttentionHeads)
  MaxSequenceLength     int
  AttentionType         GQA | MLA
  PositionEncodingType  RoPE | ALiBi | Absolute | None
  PositionEncodingConfig (type-specific: RoPE theta, scaling, etc.)
  ActivationFunction    SiLU | GELU | GELUTanh
  NormType              RMSNorm | LayerNorm
  NormEpsilon           float
  TiedEmbeddings        bool
  SlidingWindowSize     int?      (null = no sliding window)
  MlaConfig             LatentDim, RopeDim (only for MLA)
  ChatTemplate          string?   (Jinja2 template from metadata)
```

## Architecture Pattern

All supported architectures follow this pattern — parameterize, do not duplicate:

```
Token Embedding
  → (optional) Absolute Position Encoding
→ N × [
    Norm → Attention(Q, K, V, pos_enc, kv_cache, mask) → Residual
    → Norm → FFN (gate × up, activation, down) → Residual
  ]
→ Final Norm → LM Head
```

Differences between architectures are captured entirely in ModelConfig.

## Architecture-Specific Details

### Llama (2, 3, 3.1, 3.2, 3.3)
- Norm: RMSNorm
- Attention: GQA (Llama 2 70B: 64Q/8KV, Llama 3 8B: 32Q/8KV)
- Position: RoPE (theta=10000 for Llama 2, 500000 for Llama 3)
- Activation: SiLU
- FFN: SwiGLU (gate + up, SiLU, down)

### Mistral
- Same as Llama but with `SlidingWindowSize` (typically 4096)
- Some Mistral models disable sliding window for longer context

### Phi-3
- Norm: RMSNorm
- Attention: GQA
- Position: RoPE (often with su/longrope scaling)
- Activation: SiLU
- May have different tensor naming in GGUF

### Qwen2
- Norm: RMSNorm
- Attention: GQA
- Position: RoPE
- Activation: SiLU
- Tied embeddings common in smaller variants

### DeepSeek-V2/V3
- Attention: **MLA** (Multi-head Latent Attention) — structurally distinct
- Position: Partial RoPE (only on rope dimensions, rest is non-positional)
- MlaConfig: latent_dim (e.g., 512), rope_dim (e.g., 64)
- See [ATTENTION.md](ATTENTION.md) for MLA details

### GLM-4.5 MoE (`glm4moe`: GLM-4.5-Air, Cerebras REAP-82B, GLM-4.5) - CPU forward (#1599), Vulkan forward (#1600); CUDA refuses

Facts below were read from llama.cpp (`src/models/glm4-moe.cpp`, `src/llama-arch.cpp`, `conversion/glm.py` `Glm4MoeModel`, `llama_model_rope_type`) and
checked against a header range-read (first 16 MB) of `bartowski/cerebras_GLM-4.5-Air-REAP-82B-A12B-GGUF` IQ4_XS (803 tensors, 47 blocks, 52 KV) and the HF
`config.json` of `zai-org/GLM-4.5-Air` and `cerebras/GLM-4.5-Air-REAP-82B-A12B` (identical except `n_routed_experts` 128 vs 96).

- **CPU forward (#1599)** runs on the shared `TransformerModel` (no new model class): `ModelLoader.CreateCpuModelFromGguf` -> `TransformerModel.LoadFromGguf`, which passes
  `allowGlm4MoeCpu: true` to `TransformerWeights.LoadFromGguf`. Layers `>= leading_dense_block_count` take the MiMo-V2/Flint MoE arm (`LoadDeepSeekMoeLayer` with `skipF32Dequant`:
  routed banks, shared expert and the router stay in the raw quant view, never widened) and run `ForwardMoeGrouped`; layer 0 is the plain dense SwiGLU. The trunk is `block_count -
  nextn_predict_layers` layers; the NextN block (`blk.46.*`) is neither loaded nor run (no MTP). RoPE is `RoPE.Execute` with `ropeDim` = `RoPEConfig.DimensionCount` (64) and NeoX pairing,
  applied once (`PartialRotaryFactor` is only consulted by the Gemma-4 global-layer path). Q/K norm is loaded iff `attn_{q,k}_norm.weight` exists. `packedLegacyExperts` is on for
  glm4moe (IQ4_NL / Q5_1 banks use the packed x Q8 dot, as llama.cpp does). CUDA and the Vulkan pipeline / hybrid hosts still refuse (`Glm4MoeUnsupportedMessage`).
- **Vulkan forward (#1600)** runs on the generic `VulkanModelLoader` default arm (`VulkanTransformerModel`, single device; no new model class): `LoadFromGguf` passes `allowGlm4MoeCpu: true` to the shared
  weight loader, so the routed banks, the shared expert and the dense layer 0 stay raw GGUF views and are uploaded packed. Nothing is widened: REAP-82B IQ4_XS (attn_q IQ4_XS, attn_k/shared Q8_0, attn_v/o and shared gate/up Q5_K,
  routed gate/up IQ4_XS, routed down IQ4_NL, dense L0 down IQ4_NL, shared down Q5_1 on 24 layers / Q8_0 on 21, output Q6_K) reports `ExpandedTensorCount == 0` and 42.9 GiB on the device. Three gaps were filled for it:
  (1) the generic IQ-bank gate demanded K % 256 for every IQ format, which pushed the IQ4_NL down bank (K = 1408 = 22 x 64) to F32 (7x the bytes, device OOM at layer ~19);
  the legacy 32-element-block formats (IQ4_NL / Q4_0 / Q4_1) now need K % 64 (`VulkanWeights.MoeIqKGranule`; the grouped coopmat stages two blocks per round, the indexed MMVQ needs K % 32);
  (2) dense Q5_1 (`matmul_q5_1_f32_gemv` / `_gemm`, plain F32-in kernels) for the shared-expert down projection; (3) the IQ4_XS token table (151552 x 4096, 314 MiB packed) is gathered on the host
  (`TokenEmbeddingOnHost`: rows are dequantised from the mmap into the host-visible staging buffer, a few KB per token) instead of being widened to a 2.3 GiB F32 device copy. KV is F16 (`IsDenseF16KvEligible`).
  The fused single-token MoE decode layer (`TryRecordMoeLayerDecodeFused`: rmsnorm+quantize -> router / shared gate+up -> top-k -> routed gate/up ->
  fused SwiGLU+quantize -> routed down / shared down -> one combine+residual dispatch; 13 dispatches / 7 barriers per MoE layer instead of ~20 / 16) runs on
  glm4moe since #1614 (`DOTLLM_VK_MOE_DECODE_FUSED=0` disables). Two arms were added for it: Q8_0 gate/up banks (no xDiv shader) get the row broadcast +
  quantised into the routed scratch inside the router / top-k epochs, and a shared down projection without an MMVQ kernel (REAP's Q5_1 `ffn_down_shexp`
  on 24 of 46 layers) runs `RecordMatmul` (the plain F32-in Q5_1 GEMV) over the F32 SwiGLU output the fused activation kernel writes anyway.
  MTP/NextN is not run.
- **Validation recipe**: `tests/scripts/glm4moe_synthetic_hf_reference.py` loads the `SyntheticGlm4MoeGguf` tensors into HF `Glm4MoeForCausalLM` (F32, eager; use a STOCK torch wheel - a locally built
  torch 2.14 gave run-to-run different MoE logits with multithreaded CPU kernels) and prints the golden pasted into `Glm4MoeForwardTests`. On real files: `llama-perplexity -dev none --kl-divergence-base`
  + `dotllm perplexity --tokens-file` **without `--bos`** (GLM adds no BOS: the kld ids start at `
`, and `--bos` replaces each window's first token with `[gMASK]`, giving PPL 2264 instead of 11.8)
  + `scripts/kld_compare.py` with `DOTLLM_PPL_LOGITS_DUMP`.
- **GGUF metadata** (prefix `glm4moe.`): `block_count` (= trunk + `nextn_predict_layers`, 47 for Air), `context_length`, `embedding_length`, `feed_forward_length` (dense layer, 10944),
  `attention.head_count` 96, `attention.head_count_kv` 8 (scalar), `attention.key_length`/`value_length` 128, `attention.layer_norm_rms_epsilon`, `rope.freq_base` 1e6,
  `rope.dimension_count` **64** (= `int(head_dim * partial_rotary_factor)`, converter default 0.5), `expert_count` 128 (REAP 96), `expert_used_count` 8, `expert_shared_count` 1,
  `expert_feed_forward_length` 1408, `leading_dense_block_count` 1, `expert_gating_func` 2 (sigmoid; absent/NONE defaults to sigmoid in llama.cpp), `expert_weights_scale` 1.0 (= `routed_scaling_factor`),
  `expert_weights_norm` true (= `norm_topk_prob`), `nextn_predict_layers` 1. No `rope.scaling.*`, no `expert_group_count` (n_group = 1). `tokenizer.ggml.model = gpt2`, `pre = glm4`.
  There is NO QK-norm key: Q/K norm is discovered from `blk.N.attn_{q,k}_norm.weight` (llama.cpp: optional tensors); `ModelConfig.UseQkNorm` carries the HF `use_qk_norm` flag (null from GGUF).
- **Tensors** (GGUF ne-order shapes; Air numbers): `token_embd.weight [4096,151552]`, `output_norm.weight`, `output.weight` (optional; tied to token_embd if absent). Per block: `attn_norm`, `attn_q.weight [4096,12288]` + `attn_q.bias`,
  `attn_k.weight [4096,1024]` + bias, `attn_v.weight [4096,1024]` + bias, `attn_output.weight [12288,4096]` (no bias), `post_attention_norm.weight` (this is the **pre-FFN** norm = HF `post_attention_layernorm`,
  not a sandwich post-norm). Dense layer 0: `ffn_gate/up [4096,10944]`, `ffn_down [10944,4096]`. MoE layers: `ffn_gate_inp.weight [4096,E]`, `exp_probs_b.bias [E]`, fused `ffn_{gate,up}_exps.weight [4096,1408,E]`,
  `ffn_down_exps.weight [1408,4096,E]`, `ffn_{gate,up}_shexp.weight [4096,1408*n_shared]`, `ffn_down_shexp.weight [1408*n_shared,4096]` (no `ffn_gate_inp_shexp`: shared expert is ungated).
- **NextN lives in the trunk file** (bartowski/unsloth conversions): block `blk.46` (= `block_count`-1) is a full attention+MoE block plus `nextn.eh_proj [8192,4096]`, `nextn.enorm`, `nextn.hnorm`, and (optional) `nextn.embed_tokens`,
  `nextn.shared_head_head`, `nextn.shared_head_norm`. `ModelConfig.NumLayers` stays trunk-only (46) and `NextnPredictLayers` = 1. ggml-org ships the head as a separate `mtp-*.gguf` instead (trunk file then lacks block 46: not a census error).
- **RoPE**: `LLAMA_ROPE_TYPE_NEOX` for `LLM_ARCH_GLM4_MOE` (the converter does not permute Q/K: "GLM4V_MOE uses Neox ordering already"), rotated width `n_rot = rope.dimension_count` = 64 of 128. `ModelConfig.RoPEConfig.DimensionCount` = 64 and
  `ModelConfig.PartialRotaryFactor` = 0.5 are both set; the S1 forward pass must apply the partial rotation once (do not apply both).
- **MoE semantics** (`build_moe_ffn`): scores = sigmoid(router logits); top-k selection on `scores + exp_probs_b`, gating weights gathered from the UNBIASED scores, renormalised by their sum when `expert_weights_norm`, times `expert_weights_scale`;
  plus the shared expert added ungated; block output `routed + shared`. Config mapping: `SigmoidGating`, `HasSelectionBias`, `NormalizeExpertWeights`, `ExpertWeightsScale`, `NumSharedExperts`, `SharedExpertIntermediateSize` (= `moe * n_shared`, fused), `MlpOnlyLayers` = `[0..leading_dense)`.
- GLM-4.7-Flash is a different architecture (`deepseek2`, MLA); dense `glm4` / `Glm4ForCausalLM` is not mapped.

### Ripple (`ricursive`: ricursiveAI Ripple-65B-Preview) - config + census + tokenizer + template only (#1643), STUB

Facts below come from the vendor `modeling_ricursive.py` / `configuration_ricursive.py` / `config.json`, both community llama.cpp ports
(AtomicChat `ricursive-port.patch`, prometheusAIR `llama.cpp-ricursive.patch`), llama.cpp PR #31692 (head `8a5f7b0`), and header-only range reads
(first 10 MiB, no weights) of the AtomicChat Q4_K_M and prometheusAIR Q6_K GGUFs (1203 tensors = 3 + 80 x 15 each). Tracker #1642.

- **No forward pass.** `GgufModelConfigExtractor` (`BuildRicursiveConfig`), `HfConfigExtractor` (`RicursiveForCausalLM` / `model_type=ricursive`),
  `RicursiveTensorCensus` and `SyntheticRicursiveGguf` (tiny GGUF + tiny HF directory) exist. `ModelLoader.CreateCpuModelFromGguf` runs the census
  (invalid data on an unknown / mixed tensor table) and then refuses; `TransformerWeights` (central guard), `ModelLoader.LoadFromSafetensors`,
  `CudaModelLoader` and `VulkanModelLoader` refuse with `RicursiveConfig.UnsupportedMessage`. CPU forward: #1644; Vulkan: #1645.
  `debug gguf-config --header-only` prints the config and census of a header-only range read (`GgufHeaderOnly`).
- **Block** (per layer `l`): `a = RMSNorm_in(h)`; `q = Wq a`, `k = Wk a`, `v = Wv lerp(x0, a, sigmoid(lambda_l))` with `x0` the **scaled** token
  embedding (not normed, the same for every layer); per-head Q/K RMSNorm, then NeoX RoPE on sliding layers only (full layers are NoPE);
  SDPA scale `1/sqrt(head_dim)`; `o = attn * sigmoid(F32(Wg a))` elementwise **before** `Wo`; `h += RMSNorm_post_attn(Wo o)`;
  `f = RMSNorm_pre_ffn(h)`; `h += RMSNorm_post_ffn(Wdown(silu(min(Wgate f, c)) * clamp(Wup f, -c, c)))`. Plain RMSNorm `w * x / rms` (no `1 + w`).
  Untied LM head, no soft-caps, no biases, no MoE.
- **ModelConfig mapping**: `PerLayerSlidingWindow` (window on sliding layers, null on full) + `SlidingWindowSize` (1024 keys **including** the current
  token: HF masks `key >= query - (window - 1)`), `NoRopeLayers` = the full layers (derived from the pattern; no separate key), `RoPEConfig`
  (NeoX, `DimensionCount` = head_dim), `EmbeddingScale`, `UseQkNorm = true`. `ModelConfig.Ricursive` (`RicursiveConfig`) carries
  `UseGatedAttention` / `UseValueResidual` (HF flags; **null from GGUF**, where the census decides from tensor presence, all layers or none),
  `GateFullPrecision`, per-layer `FfnClamp` (0 = none) + `FfnClampDeclared`, `EmbeddingScaleFolded`. Compact SWA KV geometry
  (`KvGeometry.FromConfig(config, compactSlidingWindows: true)`) windows exactly the sliding slots (64 x 1024 rows fixed + 16 growing full layers);
  the S1 model must report `SupportsCompactSlidingKv`.
- **GGUF metadata** (prefix `ricursive.`): `block_count` 80, `context_length` 2097152, `embedding_length` 5120, `feed_forward_length` 40960,
  `attention.head_count` 80, `attention.head_count_kv` 8, `attention.key_length` = `value_length` 128, `attention.layer_norm_rms_epsilon` 1e-6,
  `rope.freq_base` 10000 (no `freq_base_swa` in either file; honoured if present), `rope.dimension_count` 128 (must equal head_dim),
  `attention.sliding_window` 1024 (**required**, as in PR #31692), `attention.sliding_window_pattern` **bool[80]** (true = sliding + RoPE; absent =
  every 5th layer full; length must equal `block_count`; at least one sliding layer), optional `attention.rope_pattern` (PR #31692; must equal the
  sliding pattern), `embedding_scale` 71.554176 (absent = folded; a declared 0 is rejected), `swiglu_clamp` 20.0 (F32 scalar in both published files;
  PR #31692's current converter writes a per-layer array - both accepted; absent or 0 = no clamp; negative / non-finite rejected).
  AtomicChat adds `ricursive.vocab_size`; prometheusAIR does not (vocab falls back to the token array length).
- **Tensors** (GGUF ne-order): `token_embd [5120,100352]`, `output_norm`, `output [5120,100352]` (optional, tied fallback). Per block: `attn_norm`,
  `attn_q [5120,10240]`, `attn_k [5120,1024]`, `attn_v [5120,1024]`, `attn_output [10240,5120]`, `attn_q_norm [128]`, `attn_k_norm [128]`,
  `attn_gate [5120,10240]` (separate full-width gate, not packed into Q; arch-scoped - unrelated to the qwen35 GDN `attn_gate`), the value-residual
  logit `[1]` F32 named **`attn_vres_lambda`** (AtomicChat, PR #31692) **or `attn_value_res`** (prometheusAIR) - one spelling per file, chosen by
  layer 0, the other spelling anywhere is reported as unexpected - then `post_attention_norm`, `ffn_norm` (= pre-FFN norm), `ffn_gate`, `ffn_up`,
  `ffn_down`, `post_ffw_norm`. HF names: `model.layers.N.{input_layernorm, self_attn.{q,k,v,o,g}_proj, self_attn.{q,k}_norm, self_attn.value_res_lambda
  (no .weight), post_attention_layernorm, pre_feedforward_layernorm, post_feedforward_layernorm, mlp.{gate,up,down}_proj}`.
- **HF config** (`configuration_ricursive.py` defaults): `layer_types` (`sliding_attention` / `full_attention`; default full iff `i % 5 == 4`;
  a stored `hybrid_layer_pattern` that disagrees is rejected), `sliding_window` 1024, `use_gated_attention` true, `gate_full_precision` true,
  `use_valres` **false** (the instruct config sets true), `ffn_clamp_threshold` null (reference: `if clamp_threshold:`), `embed_scale` null (folded).
- **Tokenizer**: `gpt2` + `pre = dbrx` (the llama3 / cl100k regex; HF uses the equivalent `(?i:'s|...)`), 100352 tokens (100278..100351 are
  `[PADn]` rows), 100000 merges. bos = unk = `<|endoftext|>` 100257, eos = eot = `<|im_end|>` 100265, pad 100277. **No BOS**: HF `post_processor` is null;
  `GgufAddBosResolver` forces false for `general.architecture = ricursive` unless an explicit key says otherwise. `GgufBpeTokenizerFactory` adds
  `<|endoftext|>` (and `<|im_end|>`) to the EOG set by text, matching HF `eos_token_id = [100265, 100257]`.
- **Chat template**: the embedded `tokenizer.chat_template` is byte-identical to the vendor `chat_template.jinja` in both GGUFs (`tokenizer_config.json`
  carries a bare ChatML stub - do not use it). ChatML; tools in the system turn as one JSON object per line inside `<tools>`; assistant calls as
  `\n<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>`; tool results merged into one **user** turn of `<tool_response>` blocks; prior
  `reasoning_content` re-inserted as `<think>\n...\n</think>\n` unless `clear_thinking`; the generation prompt does not open `<think>` (the model
  opens it itself). Parsers: `XmlToolCallParser` (template-detected, wraps Hermes) / `HermesToolCallParser` (arch fallback), `ReasoningSplitter`.

### gpt-oss (OpenAI gpt-oss-20b / 120b)
- GGUF arch string: `gpt-oss` (llama.cpp `LLM_ARCH_OPENAI_MOE`)
- Norm: RMSNorm; pre-FFN norm tensor is named `post_attention_norm`
- Attention: GQA (20b: 64Q/8KV, head_dim 64) with **per-head attention sinks**
  (`attn_sinks.weight` [numHeads] — a learned scalar logit that joins each
  head's softmax denominator) and Q/K/V/O biases
- **Alternating sliding window**: window 128 on even layers, dense on odd
  (`SlidingWindowPattern` = 2, llama.cpp `set_swa_pattern(2, dense_first=false)`)
- Position: NeoX RoPE, theta 150000, YaRN (factor 32, original context 4096);
  cos/sin tables carry the ggml mscale `attn_factor * (1 + 0.1*ln(factor))`
- FFN: routed MoE in **every** layer — 32 experts, top-4, router bias,
  **top-k on raw logits then softmax over the selected k** (softmax-after-top-k),
  per-expert gate/up/down biases, clamped `swiglu_oai` activation
  (`x = min(gate,7); y = clamp(up,-7,7); out = x*sigmoid(1.702x)*(y+1)`)
- Expert weights: MXFP4 (consumed straight from the mmap on CPU —
  `MoeQuantSwiGluMlp`); attention/embeddings/LM head: Q8_0
- Tokenizer: gpt2-model BPE with `gpt-4o` (o200k) pre-tokenizer

### MiMo-V2 / V2.5 / V2.6-Flash (`mimo2`) - CPU reference (#936), tracker #935
- GGUF arch string: `mimo2` (llama.cpp `LLM_ARCH_MIMO2`, `src/models/mimo2.cpp`); HF `MiMoV2ForCausalLM`
  (`modeling_mimo_v2.py`, remote code). 310.8B total / 15B active, 48 trunk layers (+ 3 NextN/MTP blocks that the GGUF appends:
  `block_count` = 51, `nextn_predict_layers` = 3; `ModelConfig.NumLayers` is the trunk, `NextnPredictLayers` the rest).
- **Attention** (plain GQA, no QK-norm, no bias, fused `attn_qkv`): 64 Q heads, **K head_dim 192 but V head_dim 128**
  (`attention.key_length` / `attention.value_length`; `ModelConfig.HeadDim` = K, `ValueHeadDim` = V; `o_proj` input is
  `64 * 128`). Score scale `1/sqrt(192)` (K dim). Fused rows = `[Q 64*192 | K nkv*192 | V nkv*128]`.
- **Per-layer global / sliding-window mix** (`sliding_window_pattern`, 1 = sliding): 9 global layers (indices 0,5,11,...,47, 4 KV
  heads) and 39 sliding-window layers (window 128, 8 KV heads). The GGUF stores `head_count_kv` as a per-layer array; sliding values map
  to `NumKvHeads` / `RoPEConfig` and global ones to `NumGlobalKvHeads` / `GlobalRoPEConfig` (the Gemma-4 mechanism).
- **Attention sinks on sliding layers only** (`blk.N.attn_sinks.weight` [64] F32; absent on global layers): a learned per-head
  logit joins the softmax denominator and is excluded from the V sum (same math as gpt-oss).
- **RoPE**: NEOX pairing on the first `rope.dimension_count` = 64 of 192 dims of EVERY layer (rotated span baked into the key, so
  `PartialRotaryFactor` stays null); theta `rope.freq_base` 1e7 (global) / `rope.freq_base_swa` 1e4 (sliding); frequency
  denominator is the rotated dim 64 (HF `compute_default_rope_parameters`, ggml `n_rot`), NOT the 192 head dim. No scaling.
- **V scale**: `attention.value_scale` = 0.707 (`AttentionValueScale`); HF scales V, llama.cpp scales the layer output after
  `o_proj`; V is linear in the output so the CPU applies it to the pre-`o_proj` attention output. Not folded into quantised weights.
- **FFN**: layer 0 dense SwiGLU (16384); layers 1..47 MoE: 256 routed experts, top-8, expert ff 2048, **no shared expert**,
  **sigmoid** scores, `blk.N.exp_probs_b.bias` added for top-k *selection only* (weights are the unbiased sigmoids), selected weights
  renormalised, `routed_scaling_factor` null (= 1). The GGUF does not record which layers are dense (`moe_layer_freq`); the loader
  keys off the presence of `ffn_gate_exps` and the extractor defaults `MlpOnlyLayers` to `[0]`
  (`leading_dense_block_count` overrides). `expert_weights_norm` is not written by the converter and defaults to true.
- **MTP head** (3 layers, `blk.48..50`): SWA attention (own sinks) + dense FFN 16384 + `nextn.eh_proj [2H,H]`, `nextn.enorm`,
  `nextn.hnorm`, `layer_output_norm`. Present in the file but NOT used: the trunk ignores them (llama.cpp marks them "preserved but disabled").
- Tokenizer: Qwen2 BPE (`pre = qwen2`), vocab 152576, no BOS. Chat template: MiMo family (same as the Distill-Qwen-9B).
- **CPU realisation of V != K head dim (reference, exact)**: each V head is zero-padded from 128 to 192 columns right after the
  projection so the shared attention kernels and every KV cache (single K/V row stride) are used unchanged; the pad columns of the
  attention output are exact zeros and are compacted away before `o_proj` (`TransformerModel.ExpandValueHeads` /
  `CompactAttentionHeads`). Cost: 1.5x V memory and PV work on the CPU. A native V-dim kernel is a GPU-backend task.
- **Vulkan (single device, #955)**: runs on the generic `VulkanTransformerModel`. V projection rows are zero-padded to the K head dim at
  upload (exact for every block format), the attention output is compacted back to `nH*128` (and `attention_value_scale` applied) by
  `attn_compact_scale_f32` before `o_proj`; per-layer KV heads (`GetLayerKvHeads`), dual rope theta on the inverse-frequency kernel, sink
  attention on the sliding layers (gpt-oss kernels), the GLM sigmoid+bias router, and IQ1_M / IQ2_XXS / IQ2_XS / IQ3_S routed banks kept
  packed (indexed IQ MMVQ at <16 rows, grouped IQ coopmat above). KV cache is the dense per-layer-strided `VulkanKvCache` (no sliding ring yet,
  F32 K and padded V, ~535 KiB/token). Pipeline-parallel / hybrid Vulkan+CUDA and **CUDA refuse** `mimo2` (`ModelConfig.MiMoV2UnsupportedMessage`).
  MTP is not implemented anywhere.
- Verification: `tools/mimo2-tiny-oracle` builds a tiny random MiMoV2 (K 48 / V 32, rotary 16/48, GQA 4 global / 2 sliding,
  window 4, sinks, 8 experts top-2, dense layer 0) in HF torch, converts it with llama.cpp's own `convert_hf_to_gguf.py`, and
  compares HF torch / llama.cpp CPU / dotLLM CPU (max |dlogit| 1.9e-5, KL 8.6e-12, top-1 100%; decode through the per-layer-strided KV cache
  matches too). `MiMoV2SyntheticGoldenTests` is the hermetic variant (C#-written GGUF vs llama.cpp golden logits, plus mutants).

### Recursive Flint (`flint`) - CPU (#1158), tracker #1157
- GGUF arch string: `flint` (llama.cpp `LLM_ARCH_FLINT`, `src/models/flint.cpp`, ggml-org/llama.cpp#30928); HF `FlintForCausalLM`
  (no transformers modeling code; references are llama.cpp, SGLang, vLLM, MLX `rec`). 39.8B total / ~2.5B active, 49 layers, hidden 2048.
- **Attention**: 32 Q heads / 4 KV heads, head_dim 128 (Q projection 4096 wide, wider than hidden), separate `attn_q/k/v`, per-head
  QK-RMSNorm (`attn_q_norm` / `attn_k_norm` [128]) before RoPE, learned per-head **attention sinks on every layer**
  (`attn_sinks.weight` [32]), score scale `1/sqrt(128)`.
- **SWA / full mix**: `attention.sliding_window` = 513 (current + 512 previous, llama.cpp `LLAMA_SWA_TYPE_STANDARD`) on the layers
  flagged by `attention.sliding_window_pattern` (bool array, true = sliding; 35 of 49), full attention on the other 14.
- **RoPE**: NEOX, theta 1e4, full head dim, on the layers flagged by `attention.rope_pattern` (bool array, true = RoPE). The full layers are
  **NoPE**; the extractor maps them onto `ModelConfig.NoRopeLayers` and leaves `GlobalRoPEConfig` null when every full layer is NoPE.
- **FFN**: layers 0-1 dense SwiGLU 6144 (`leading_dense_block_count` = 2); layers 2-48 MoE: 256 experts, top-6, expert ff 512, **sigmoid**
  scores, renormalised (`expert_weights_norm`), scale 1.0, no selection bias in any shipped file (`exp_probs_b` is optional and loads as
  "no bias"), plus **one ungated shared expert** of width `expert_shared_feed_forward_length` = 1024 added to the routed output.
  `expert_shared_count` = 2 is that width over the expert width, not two experts (`NumSharedExperts` = 1).
- **Built by** `BuildMiMoV2Config` (shared with `mimo2`); the CPU loader reuses the MiMo-V2 MoE branch, which keeps the routed AND the
  shared expert banks in the raw GGUF quant view (no F32 copy, rule 2). `ForwardMoeGrouped` runs the shared expert from the raw view
  through the same `DispatchGemm` as the routed experts, and uses the packed x Q8_1 dot for Q4_0/Q4_1/Q5_1/IQ4_NL banks (the official file is
  Q4_1 throughout; batch-invariant).
- Tokenizer: GPT-2 BPE, `tokenizer.ggml.pre = flint` (3-stage pipeline: `\p{N}{1,3}`, CJK/kana runs, GPT-4o-style main regex with
  ` ?[\p{P}\p{S}]+[\r\n]*`), vocab 150144, no BOS for chat. EOG: `tokenizer.ggml.eog_token_ids` plus `<|user|>` / `<|observation|>`
  (llama.cpp forces both for this pre-type). Chat template: `<|system|>` / `<|developer|>` / `<|user|>` / `<|assistant|>` /
  `<|observation|>`, `<think>` reasoning, GLM-4.5-style tool calls (`GlmToolCallParser`); it strips `"\x00"`, which needs the Jinja lexer's
  hex escapes.
- **F16-safe down input**: Flint's routed SwiGLU output reaches ~13000; the Q8_1 block sum `s = d * sum(q)` is stored as F16 and
  overflowed to inf (then NaN, then an out-of-range expert index in the next layer's routing). The routed down projection now divides
  each input row by `e = 2^ceil(log2(max|row| / 1024))` and multiplies the projected row back (llama.cpp `build_amax_src1_f16`). A
  power of two is exact, so rows at or below 1024 are untouched; it applies to every architecture.
- **KV**: sliding layers keep only their recent rows on CPU (compact sliding-window KV, #1169, `docs/KV_CACHE.md`): at 262K the 14 full layers dominate (~14 GiB F32, ~3.7 GiB with `--cache-type-k q8_0 --cache-type-v q8_0 --cache-window 0`).
- **Activation range** (#1191, `docs/PRECISION.md` E13, `DOTLLM_AMAX_PROBE`): on 16K tokens of agent trace + wikitext the routed
  down input reaches 2.4e5 (guarded on every backend), the residual stream 3.5e4 and the MoE output 3.3e4 (layers 46-48),
  the shared-expert down input 6.3e3, everything else < 3.4e3. CUDA keeps the residual in F16, so the Flint FP16 loop carries it
  scaled by 1/8 (exact: every residual read is an RMSNorm with eps scaled to match; `DOTLLM_CUDA_RESIDUAL_SCALE=1` disables it).
  Vulkan keeps the residual and MoE outputs in F32 and its Q8_1 scales in F32, so no further guard is needed there.
- **GPU**: CUDA runs `flint` on the generic FP16 MoE loop (#1172, F6: per-layer NoPE, sigmoid routing, in-place Q4_1 GEMV /
  GEMM / grouped MoE kernels; see `docs/CUDA.md`). On a smaller GPU the routed experts of the first layers stay on the CPU, read from the mmap
  (#1187, F7: `--cpu-moe` / `--n-cpu-moe` or automatic; `docs/MOE_OFFLOAD.md`): all 49 layers run on a 12 GB RTX 3060.
- **Vulkan** (#1171, single-device `VulkanTransformerModel` only; pipeline / hybrid refuse): sinks required on every layer (the gpt-oss
  sink kernels), NoPE through the generic `IsNoRopeLayer` path, the ungated shared expert through the existing shared-expert branch, sigmoid
  top-6 on `moe_topk_sigmoid_bias_f32` with a zero bias, and Q4_1 read in place everywhere (dense MMVQ / GEMV / GEMM / coopmat, MoE indexed
  MMVQ + grouped coopmat). KV is **F16** (`VulkanKvCache.Storage`, `DOTLLM_VK_KV=f32` for A/B) at full length for every layer: the compact
  sliding-window ring is not on Vulkan yet (follow-up), so 98 KiB/token (28 KiB of it the 14 full layers).
  Sink prefill (#1200) runs the `attention_flash_gqa{,_cm}_sinks{,_kvf16}` twins of the GQA flash kernel (K / V tiles shared by the 8 query
  heads of a group, the sink seeding the online softmax, tiles before the window start skipped); the reference `attention_sinks_f32` kernel
  (one workgroup per query and head over the whole KV range) tripped the TDR at 32K. `DOTLLM_VK_SINK_FLASH=0` restores it for A/B.
  Decode takes the GQA-grouped split-KV pass with the sink in the merge (`DOTLLM_VULKAN_SPLITKV_GQA=0` for A/B). Strix, chunk 512, compact
  ring: pp512 1,004-1,239 / pp8K 981 / pp32K 620 tok/s (reference kernel: 500-546 / 40 / device lost); decode 70 / 54 / 28.2 tok/s at
  0.5K / 8K / 32K (per-head split pass at 32K: 17.7). llama.cpp master Vulkan: pp 1,632 / 1,389 / 950, tg 73 / 65 / 51. The pp512 gap is
  the MoE (grouped gate-up + down + scatter/group/broadcast/router, ~68 % of the forward), not attention (~9 %).
  The grouped coopmat MoE kernels fold the down input into an F16 operand; real Flint tokens exceed 65504 there (inf, then a NaN token that
  an F16 KV cache spreads to every later row). `row_pow2_downscale_f32` / `row_scale_mul_f32` apply the CPU's power-of-two rescale around
  the grouped down projection (all architectures; `DOTLLM_VK_MOE_DOWN_RESCALE=0` disables it for A/B).
- Verification: `FlintSyntheticGoldenTests` writes a random-weight `flint` GGUF (GQA 8:2, Q wider than hidden, window 4 over 16 tokens,
  NoPE on a MoE layer, sinks everywhere, shared expert) and compares against llama.cpp CPU logits on the same bytes: F32 max |dlogit|
  2.1e-6, Q4_1 routed + Q8_0 shared 8.0e-6; NoPE, window, shared expert, renormalisation and sinks each move the logits by > 4.
  `JinjaFlintReferenceRenderTests` renders the real template byte-identically to transformers.
- **Real weights** (`RecursiveSI/flint-int4-gguf`, Q4_1, vs llama.cpp ggml-org/llama.cpp#30928 CPU, 155H):
  - token ids identical to `llama-tokenize` on prose, C# source and a mixed CJK/emoji/digits/code text;
  - teacher-forced top-1 on llama.cpp's greedy 128-token answers (5 chat prompts): 98.2-100 %, every miss a near-tie (gap 0.06-0.46);
  - peak private memory 423 MiB for the 24.8 GB file (no expert expansion, no copy).
- **KL against llama.cpp is not a parity metric for Flint.** Its Q4_1 path quantises activations to Q8_1, and Flint's large
  activations make that coarse: llama.cpp's own Q4_1 file vs the same weights dequantised to F32 is KL 6e-3 (synthetic, head_dim
  128, 512 tokens), while dotLLM's batched prefill (exact dequantised weights x F32 activations) matches the dequantised reference
  to 2e-5. On the real model, dotLLM vs llama.cpp (both F32 KV) is mean KL 0.016 / 0.018 / 0.073 (prose / code / mixed), and dotLLM's
  own prefill vs its Q8_1-activation decode is 0.011; llama.cpp vs itself at a different thread count is 0.000. dotLLM's prefill PPL
  is 1-6 % above llama.cpp's on those corpora (exact activations vs Q8_1; which is closer to the BF16 reference is open).
  Per-row KL between any two Flint backends (or one backend and a 1e-6-perturbed copy of itself) has a heavy tail from
  Q8_1 expert-activation rounding and router near-ties, at every depth: gate on NLL / PPL, not mean KL (`docs/PRECISION.md` E14, #1218).

### Qwen4-Exp / Qwen3.8-Flash-Next (`qwen4exp`) — config (#815) + CPU reference forward (#816), epic #814
- GGUF arch string: `qwen4exp` (llama.cpp `LLM_ARCH_QWEN4EXP`) -> `Architecture.Qwen4Exp`. The **CPU** loader builds `Qwen4ExpTransformerModel` (the
  numerical oracle, validated against HF `qwen4_exp` on a tiny random-weight model); **Vulkan** builds `VulkanQwen4ExpTransformerModel` (#818, below);
  CUDA still refuses it with `Qwen4ExpConfig.UnsupportedMessage(...)` (issue C1 of the epic).
- 48 layers = `(GDN, GDN, GDN, QSA) x 12` (`full_attention_interval` 4, reuses `GdnConfig` + `HybridLayout`), every block a
  512-expert top-10 **softmax** MoE (`NormTopKProb`) with one sigmoid-gated shared expert (reuses `MoeConfig`). Optional trailing MTP
  block: `block_count` 49 + `nextn_predict_layers` 1 (the Unsloth trunk files have neither; MTP ships as a separate GGUF).
- Model-specific parameters live in `ModelConfig.Qwen4Exp` (`Qwen4ExpConfig` / `Qwen4ExpPleConfig`), read under the exact keys llama.cpp's
  `load_arch_hparams` reads (verified against the real UD-Q4_K_XL header):

| GGUF key | field | notes |
|---|---|---|
| `hyper_connection.count` / `.low_rank` | `HyperConnectionCount` / `HyperConnectionLowRank` | 4 / 320; count must be > 1 |
| `attention.indexer.head_count` / `.key_length` / `.top_k` | `IndexerHeadCount` / `IndexerKeyLength` / `IndexerTopK` | 4 / 128 / 2048 (a TOKEN budget) |
| `attention.compress_ratios` | `CompressRatios`, `IndexerBlockSize` | per-block i32 (0 = GDN, 4 = QSA); one shared ratio > 1 dividing `top_k` |
| `rope.dimension_sections` | `RopeSections` | `[11, 11, 10, 0]`, required |
| `ple.layers` | `Ple.Layers` | **zero-based** (`[1]`), strictly ascending; every entry must be a GDN layer; several supported (#844) |
| `ple.ngram_size`, `.heads_per_ngram`, `.conv_kernel`, `.eos_token_id`, `.image_token_id` | `Ple.*` | image id optional (null -> EOS) |
| `embedding_length_per_layer_input` | `Ple.RowDim` | 160 (not under `ple.*`) |
| `ple.layer_multipliers`, `.head_offsets`, `.head_vocab_sizes` | `Ple.*` as `ulong` | exact uint64 (never via float/double/signed); ONE SET PER PLE LAYER, layer-major (module j = position in `ple.layers` uses `[j*ngram_size, ..)` / `[j*heads, ..)`); a single set with several layers is refused |

- Tensor contract: `Qwen4ExpTensors` (names + shapes, `FindProblems` diff) — asserted equal to the real 1224-tensor trunk table and the 34-tensor
  MTP table. Differences from the early design notes: indexer tensors are **dotted** (`blk.N.indexer.q_proj`), the head mixer is
  `output_hc_{norm,down,up}`, experts ship **split** (`ffn_gate_exps` + `ffn_up_exps`), `ssm_a` has no `.weight`, the 51.2 B-param
  n-gram table is the single tensor `per_layer_token_embd.weight` `[160, 320001536]` (IQ4_NL) and sits in its own shard.
- Test fixture: `SyntheticQwen4ExpGguf` (4 blocks, 8 experts, optional MTP block, optional `-0000N-of-0000M` split output).
- **CPU reference forward** (`Qwen4ExpTransformerModel`, #816). Per block: optional PLE add (layer 1), gated-residual (GR) read -> token mixer
  -> GR write, GR read -> MoE -> GR write; final head mixer replaces the output norm. Residual layout `[T, hc, H]` (flat `hc*H` row per token).
  - GR (`Qwen4ExpGatedResidual`): `xn = groupRMS(R)*gamma` (per-stream RMS and gamma), `h = mean_s(sigmoid(up(silu(down(xn)/S))) * xn)`,
    `inj = 2*sigmoid(inject(xn)/S)`, write `R[s] += inj[s]*y`. Gammas are GGUF-convention (HF `1+w` folded by the converter; the same fold
    applies to the PLE and indexer norms; `ssm_norm` is NOT folded).
  - GDN = Qwen3.5 GDN with a **sigmoid** output gate (`norm(core)*sigmoid(z)`), unlike Qwen3.5's silu (llama.cpp `build_norm_gated`).
  - QSA (`Qwen4ExpQsaLayer`): pool-then-rope indexer (`rope(rmsnorm(mean of 4 raw keys), pos 4b)`), block-causal `sum_h relu(q.k)/sqrt(D)`,
    top `top_k/4` blocks (ties -> lower index) + the incomplete tail; exactly dense for <= `top_k + 3` (2051) tokens. `ForceDense` is a diagnostic.
  - PLE (`Qwen4ExpPleBranch`): exact int64 hash (`(t0*m0) ^ (t1*m1) [^ (t2*m2)]`, signed floor-mod, EOS cuts the window), row gather straight
    from the lazy table (IQ4_NL/BF16/...; never copied), signed-sqrt sigmoid gate, dilated (3) depthwise conv with a 9-row history.
  - **Several PLE modules and image tokens (#844).** HF gives every PLE module (`ple_layer_ids`, 1-based, sorted) its own n-gram table, key/value
    projections, norms and dilated conv, and derives its hash multipliers (`_build_layer_multipliers(..., ple_layer_index, seed)`) and its slice of the head
    primes (`global_head_idx = ple_layer_index * ngram_heads + h`) from the module's position `ple_layer_index`. llama.cpp's hparams hold ONE set of constants
    and one `per_layer_token_embd.weight`, so it cannot describe a second module; the dotLLM convention (verified only against HF, no llama.cpp-written
    multi-module file exists) concatenates the tables into that one tensor and stores the constants module-major, offsets already shifted by the rows of the
    tables before (`Reference/gen_model_ple2.py`, `Fixtures/tiny_model_ple2.json`: modules on layers 1 and 2). Each module keeps its own `Qwen4ExpPleState`
    (hash window + conv history), checkpointed, row-snapshotted and accounted like the single module. Image tokens: HF passes the original `input_ids`
    (the image placeholder) as `ple_input_ids` while `inputs_embeds` carries the vision-tower rows; llama.cpp substitutes `ple.image_token_id` (EOS if the key is
    absent). `Qwen4ExpTransformerModel.Forward(..., externalEmbeddings)` marks such positions with `ExternalEmbeddingToken` (-1), takes their embeddings from the
    supplied rows and hashes them as the stand-in id, reproducing HF when `ple.image_token_id` equals the placeholder id.
  - The model keeps its own sequence state (`Qwen4ExpSequenceState`: GDN + PLE window/conv + QSA K/V and pooled indexer keys); positions must
    continue it.
  - **Quantised KV (#841).** The QSA K/V rows can live in a `QuantizedKvCache` (`--cache-type-k/-v q8_0|q4_0`, both sides quantised; the KV row
    `nKv*headDim` must be a multiple of 32). The sparse attention gathers the selected rows, so `Qwen4ExpKvRows` dequantises exactly those rows (quantised
    region) or copies them (fp32 window) into a scratch, and the rows of the chunk being processed are read from the fresh fp32 projections: a chunk is
    attended at full precision, only rows from earlier chunks carry rounding (single-shot prefill is bit-identical to fp32). An update longer than the
    fp32 window is split into window-sized pieces (a longer update would quantise unwritten ring slots). Pooled indexer keys stay **fp32**: they decide
    which blocks are attended, and quantising them saves 94 of the 1,216 bytes/token/layer (measured Q8_0 round-trip keeps 99.7% of the top-512 block set
    on random keys, the worst case). Measured, standalone QSA layer at the real 2048-token budget, T=2400 (rows past 2051 are sparse): Q8_0 relative L2
    error 2.2e-3 (proxy-logit KL 9e-7, top-1 99.7%), Q4_0 4.1e-2 (KL 3e-4, top-1 94.3%); KV bytes 3.76x smaller at Q8_0 (the real 512-wide row:
    2,048 B in bf16 -> 1,088 B at Q8_0 per K+V per token per layer; 12 QSA layers x 262K ctx: ~6 GiB -> ~3.2 GiB).
    `Qwen4ExpStateBytes.Estimate(config, ctx, keyDType, valueDType, window)` accounts for it. Snapshots taken from a quantised cache hold the dequantised
    rows. **Vulkan (V2, #819):** the QSA gather kernel would dequantise Q8_0 blocks (34 B / 32 elements) on load, read the newest `window` rows from the fp32
    ring and the in-flight chunk from the fresh projections, and keep the pooled keys + top-k in fp32.
  - **Engine state (#817).** `Qwen4ExpSequenceState` is an `IGdnState` (so the scheduler threads it with no changes): GDN, PLE window/conv and each QSA
    layer's pooled indexer keys + raw tail are native memory; the QSA K/V rows live in the engine `IKvCache` (slot = QSA ordinal, stride
    `nKv*headDim`; `KvGeometry.FromConfig` over-allocates the 36 unused GDN slots, as for the other hybrids) or, with no cache, in a lazily allocated
    native store inside the state. Implements `CreateSequenceState`/`SupportsThreadedSequenceState`, `ForwardBatch` (per-sequence loop, last-row
    logits), `CheckpointRecurrentState`/`RestoreRecurrentState` (logically a full copy, so valid after the live state moved to another history, but
    physically incremental, #840: the GDN buffers are exchanged with the pooled shell and the live state lazily reads them in the next forward's first
    scan step - `GatedDeltaNetScan.Execute(stateSource:)`, bit-identical, no separate copy pass - and pooled keys / own K/V rows are copied only
    where their `Qwen4ExpRowStamps` content stamps differ; at the released size 4-12 ms -> ~0.1 ms per checkpoint and per restore, see PR #840), per-row snapshots (`ForwardWithRecurrentSnapshots`/`RestoreRecurrentStateToRow`: GDN by REPLAY (#842) - one pre-chunk state kept without a copy plus the keys/decays/deltas of each row, `GatedDeltaNetScan.Replay` rebuilds the state after any row bit-identically to the scan; scratch 112 MiB + ~1.8 MiB/row instead of 112 MiB/row, restore to row r costs r+1 rank-1 passes (~10-15 ms at the released size) instead of one copy (~2.4 ms) but nothing is copied per row during the verify forward; PLE history and the indexer
    tail rebuilt from the recorded chunk) and `SnapshotSequencePrefix`/`RestoreSequencePrefix`. Accounting: `Qwen4ExpStateBytes.Estimate(config, ctx)` /
    `model.EstimateSequenceStateBytes(ctx)` split into Gdn (~113 MiB at the released size, constant), Ple, IndexerTail (constant), IndexerPooled
    (128 B per token per QSA layer) and Kv; `state.ResidentBytes` is the allocated counterpart. Exact (bit-identical) rollback holds between runs that
    use the same forward shapes; snapshot-vs-differently-shaped-fresh comparisons agree to ULP-level drift. Reference fixtures: `tests/DotLLM.Tests.Unit/Models/Qwen4Exp/Reference/gen_*.py` (HF transformers >= 5.19; recipe in
    `qwen4exp_ref_common.py`).
- **Vulkan forward** (`VulkanQwen4ExpTransformerModel`, #818 V1). Composes the Qwen3MoeHybrid Vulkan blocks (GDN layer, full-GQA layer,
  routed+shared MoE, quant-aware matmul) behind its internal `Q4*` surface and adds the GR shader (`qwen4exp_gated_residual.comp`: broadcast,
  `silu(v/S)`, mix-and-mean, `2*sigmoid(g/S)`, write), the sigmoid GDN gate (`gdn_post_scan_gate_sigmoid_f32.comp`) and the three routers at
  `MAX_EXPERTS` 512. Validated against the CPU oracle on random-weight checkpoints (tiny, 512-expert top-10, 256-wide K-quant, Q8_0/Q5_1/BF16
  mixes): per-layer residual and last-row logits, 24-step greedy decode, chunked prefill. Attention is dense up to `top_k + block - 1` = 2051 tokens (QSA == dense there) and **sparse QSA beyond it** (#819: device indexer
  pooling / scoring / exact top-k / gather attention, see `docs/ATTENTION.md`; context capacity default 8192 tokens, `DOTLLM_VK_QWEN4EXP_CONTEXT`); token embeddings
  are gathered on the host (no 2.4 GiB device table); a call longer than `MaxRowsPerForward` (`DOTLLM_VK_PLANNED_ROWS`, default 1024 rows: the scratch planned at load)
  is split into chunks internally instead of growing the scratch past the resident-memory wall. V1 limits: the n-gram branch runs on the **host** at its layer (one residual
  round-trip; table registered host-only so no upload/import/staging path can take it); model-owned single sequence state (#817); no MTP;
  Q5_1 / Q8_0 / IQ expert banks are widened to F32 (no resident kernel yet), which the pre-load gate (`Qwen4ExpResidencyPlan`, refuses over
  `ResidentCapacityBytes - headroom`, `DOTLLM_VK_ALLOW_OVERCOMMIT=1` overrides) accounts for.

### LFM2.5-24B-A2B (`lfm2moe`, Liquid AI, #1015)

- **Layer algebra** (`Architecture.Lfm2Moe`, CPU `Lfm2MoeTransformerModel`, Vulkan `VulkanLfm2MoeTransformerModel`): every layer is two residual sub-layers,
  `x += mixer(operator_norm(x)); x += ffn(ffn_norm(x))`. The mixer is a **gated short convolution** on 30 layers
  (`in_proj -> [B | C | X]; y = out_proj(C * causal_depthwise_conv_k3(B * X))`, no bias, no activation; state = the last `L_cache-1 = 2` rows of `B*X` per
  channel, `ModelConfig.ShortConvCacheLength`) or **GQA with per-head QK RMS-norm and NeoX RoPE** on 10 layers (indices 2, 6, ..., 38; the layout comes
  from the per-layer `head_count_kv` array, 0 = conv: `HybridLayerKind.ShortConv` / `Attention`). The FFN is a dense SwiGLU (`feed_forward_length` 11776)
  for the first `leading_dense_block_count` = 2 layers, then 64-expert top-4 SwiGLU MoE with **sigmoid gating + `exp_probs_b` selection bias**
  (top-k on `sigmoid + bias`, weights = the unbiased sigmoids, renormalised: llama.cpp `lfm2moe.cpp` hard-codes `norm_w = true`), no shared expert.
  The final norm is `token_embd_norm.weight` and the head is tied to `token_embd`.
- **State**: `Lfm2ConvStateCache` / `VulkanLfm2ConvStateCache` are `ISsmState`s (per sequence through `CreateSequenceState` / `ForwardBatch`); attention layers
  use the sparse hybrid KV cache (slots only for the 10 attention layers). No recurrent checkpoint/rollback yet (so no speculative draft/verify).
- **Quantisation**: the QAD-Q4_0 file is Q4_0 + a Q6_K tied head. CPU routes Q4_0 expert banks through the packed x Q8_1 dot (opt-in
  `packedLegacyExperts`, batch-invariant). Vulkan keeps Q4_0 dense and routed weights packed when integer-dot-product kernels are available.
  Dense prefill uses cooperative matrices where supported, otherwise the portable packed F32 GEMM; wave64 is not required for packed residency.
  `DOTLLM_VK_Q4_0_NATIVE=0` restores the exact Q4_0 -> Q8_0 weight transcode (same f16 scale, `q8 = nibble - 8`).
  Dense portable prefill uses F32 activations while decode MMVQ and CPU packed execution quantize activations to Q8_1; identical weight values
  do not imply identical arithmetic across these paths. A synthetic all-Q4 comparison against the Q8 transcode measured 0.58% normalized maximum
  logit error; this is not a full-checkpoint quality result. Portable dense kernels are checked against dequantized F32 weights separately from
  packed-expert parity. Short-convolution projection scratch holds `3 * HiddenSize` values per token independently of routing top-K.
  Causal chunk/full parity is tested with `DOTLLM_VULKAN_DISABLE_COOPMAT_ATTENTION=1`, keeping attention in F32. Default fast attention retains
  F16 cooperative-matrix prefill and F32 decode: the synthetic top-K-one fixture measured 0.277% normalized maximum logit difference between
  a three-token prefill plus decode and a four-token prefill, including with the legacy Q8 transcode. Default-path packed/transcoded parity is
  checked on identical schedules; strict cross-schedule parity for mixed attention precision remains unestablished. The default fast path is unchanged.
  Router dump: `DOTLLM_TENSOR_DUMP_DIR`.
- **Chat / tools**: always-thinking ChatML (`...assistant
<think>`), Pythonic tool calls `<|tool_call_start|>[fn(a='x')]<|tool_call_end|>` parsed by
  `PythonicToolCallParser`. Raw `dotllm run` adds no BOS (prefix `<|startoftext|>` or use the chat/serve path).

### LFM3 (`lfm3`, Liquid AI, #1558 / #1559)

- **What it is**: a "tri-hybrid", **not** a state-space model. LFM3-1.7B has 24 layers: 13 gated short-convolution layers (`L_cache = 7`, state = the last 6 rows of
  `B*X` per channel), 5 **sliding-window** GQA layers (window 256, RoPE theta 1e4) and 6 **global** GQA layers (indices 2, 6, ..., 22; RoPE theta 1e6). Only
  the 6 global layers' KV grows with context (the sliding layers need at most 256 live rows and the conv state is constant), so long-context decode touches about a
  quarter of the KV a plain transformer of the same depth would. hidden 2048, 16 heads / 4 KV heads, head_dim 128, dense SwiGLU FFN 7680 on every layer, tied head, 128k vocab.
- **Layer algebra** (`Architecture.Lfm3`, CPU `Lfm3TransformerModel`; layout from the per-layer `head_count_kv` array (0 = conv) and the per-layer bool array
  `attention.sliding_window_pattern`, surfaced as `ModelConfig.PerLayerSlidingWindow`): `x += post_operator_norm(mixer(operator_norm(x)))`, then
  `x += post_ffn_norm(ffn(ffn_norm(x)))` (**sandwich norms**, GGUF `post_attention_norm` / `post_ffw_norm`; both optional, `use_sandwich_norm`). The attention mixer adds a **per-head
  sigmoid gate**: `attn_out[h] *= sigmoid(attn_gate . x)` (`attn_gate`, hidden to n_heads) before `o_proj`, on top of per-head QK RMS-norm and NeoX RoPE.
- **RoPE**: sliding layers rotate all 64 pairs with theta `rope.freq_base_swa` (1e4). Global layers use theta `rope.freq_base` (1e6) but rotate only the first 32 pairs: the
  GGUF stores `rope_freqs.weight` (64 divisors, 1.0 for the first 32, 1e30 after), so the frequency progression of the full 128-wide head is kept and the tail is left unrotated
  (HF writes the same thing as a zero-tail `inv_freq`). The CPU model builds cos/sin per forward from these and applies a pair-limited NeoX rotation.
- **State**: reuses `Lfm2ConvStateCache` (per sequence via `CreateSequenceState` / `ForwardBatch`); the KV cache is the sparse hybrid cache (slots only for the 11 attention layers).
  Sliding layers currently keep the full history in the cache and mask by window (compute is O(window) per token; memory is not yet bounded). No recurrent checkpoint/rollback yet.
- **Final norm** is `output_norm.weight` (LFM2 files name it `token_embd_norm.weight`); the head is tied to `token_embd`.
- **Chat / tools**: ChatML with `<|startoftext|>`, thinking via `<think>` (`enable_thinking` default true, so the generation prompt ends `<think>`), XML tool calls
  `<tool_call>
<function=name>
<parameter=k>
v
</parameter>
</function>
</tool_call>` parsed by `QwenXmlToolCallParser`; image/video placeholders
  `<|vision_start|><|image|><|vision_end|>`. GGUF `tokenizer.ggml.pre = lfm2`, `add_bos_token = false`.
- **Parity** (vs llama.cpp `dde879bc`, `-dev none`, wikitext-2, 512-token windows, full-vocab logits): BF16 4 windows top-1 99.5%, mean KL 6.0e-4; Q8_0 16 windows top-1 97.4%,
  mean KL 2.8e-3 (activation-quantisation floor), PPL 25.9443 vs 25.9400 (+0.017%). See `scripts/kld_compare.py` and `docs/PERPLEXITY.md`.

## GGUF → ModelConfig Mapping

```csharp
var arch = metadata["general.architecture"]; // e.g., "llama"
var config = new ModelConfig
{
    Architecture = ParseArchitecture(arch),
    VocabSize = metadata.GetOrDefault($"{arch}.vocab_size",
                    metadata["tokenizer.ggml.tokens"].Length),
    HiddenSize = metadata[$"{arch}.embedding_length"],
    IntermediateSize = metadata[$"{arch}.feed_forward_length"],
    NumLayers = metadata[$"{arch}.block_count"],
    NumAttentionHeads = metadata[$"{arch}.attention.head_count"],
    NumKvHeads = metadata.GetOrDefault($"{arch}.attention.head_count_kv",
                    config.NumAttentionHeads),
    NormEpsilon = metadata[$"{arch}.attention.layer_norm_rms_epsilon"],
    MaxSequenceLength = metadata[$"{arch}.context_length"],
    ChatTemplate = metadata.GetOrDefault("tokenizer.chat_template", null),
    // RoPE config
    PositionEncodingConfig = new RoPEConfig
    {
        Theta = metadata.GetOrDefault($"{arch}.rope.freq_base", 10000f),
        ScalingType = ParseScalingType(metadata.GetOrDefault(
            $"{arch}.rope.scaling.type", "none")),
    }
};
```

## Adding New Architectures

1. Check if the architecture fits the standard pattern (Norm → Attention → Residual → Norm → FFN → Residual).
2. If yes: add a new `Architecture` enum value, map GGUF metadata keys to ModelConfig, done.
3. If the attention mechanism is different (like MLA): implement a dedicated attention path in the forward pass.
4. If the FFN structure is different: parameterize or add a new FFN variant.
5. Verify numerical output against HuggingFace transformers reference for the new architecture.
