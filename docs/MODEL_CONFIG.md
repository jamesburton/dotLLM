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
  are gathered on the host (no 2.4 GiB device table). V1 limits: the n-gram branch runs on the **host** at its layer (one residual
  round-trip; table registered host-only so no upload/import/staging path can take it); model-owned single sequence state (#817); no MTP;
  Q5_1 / Q8_0 / IQ expert banks are widened to F32 (no resident kernel yet), which the pre-load gate (`Qwen4ExpResidencyPlan`, refuses over
  `ResidentCapacityBytes - headroom`, `DOTLLM_VK_ALLOW_OVERCOMMIT=1` overrides) accounts for.

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
