# Model Configuration — dotLLM

## ModelConfig Record

Comprehensive record describing any transformer variant. Populated from GGUF metadata at model load.

```
ModelConfig:
  Architecture          Llama | Mistral | Phi | Qwen | DeepSeekV2 | DeepSeekV3 | NemotronH | Mamba3 | Mixtral | QwenMoe | GraniteMoe
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
