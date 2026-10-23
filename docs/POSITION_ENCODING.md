# Position Encoding — dotLLM

## IPositionEncoding Interface

Optional and pluggable. `ModelConfig.PositionEncodingType` selects variant or `None`.

```
IPositionEncoding:
  Apply(Q, K, positions) → (Q_encoded, K_encoded)
  PrecomputeTables(maxSeqLen, config) → void
  InvalidateCache() → void   // When parameters change at runtime
```

## RoPE — Rotary Position Embeddings

Default for Llama, Mistral, Qwen, Phi-3, DeepSeek.

### Mechanism

Rotates pairs of dimensions in Q and K by position-dependent angles:
```
θ_i = θ_base^(-2i/d)
angle = pos × θ_i
q'[2i]   = q[2i] × cos(angle) - q[2i+1] × sin(angle)
q'[2i+1] = q[2i] × sin(angle) + q[2i+1] × cos(angle)
```

### Implementation

1. Pre-compute cos/sin tables `[max_seq_len × head_dim/2]` at model load.
2. At inference: lookup `cos_table[pos]`, `sin_table[pos]`, apply rotation.
3. SIMD: process dimension pairs with `Vector256<float>` — two muls + add/sub per pair.

### RoPEConfig (from GGUF)

```
Theta: float (10000.0 default, Llama 3 = 500000.0)
DimensionCount: int
ScalingType: None | Linear | YaRN | NTK | DynamicNTK
ScalingFactor: float
OrigMaxSeqLen: int
AttnFactor, BetaFast, BetaSlow: float (YaRN-specific)
```

## Scaling Variants

### Linear (Position Interpolation)
`angle = (pos / factor) × θ_i`. Simplest. 4K trained + factor 4 → 16K.

### NTK-Aware
Modifies base: `θ_new = θ × factor^(d/(d-2))`. Preserves local patterns better.

### YaRN
Combines NTK with attention scaling. Per-dimension: high-freq unchanged, low-freq fully scaled, medium interpolated.

The ramp math lives in one place, `RoPE.ComputeYarnInverseFrequencies` (a port of HF `_compute_yarn_parameters`), and
`RoPE.ResolveDenseYarn(rope, architecture, ropeDim)` hands every host the same inverse frequencies and cos/sin
multiplier: the CPU `RoPERotation`, the CUDA upload (`LaunchRoPEF32(ropeInvFreq, ropeMscale)`) and
the Vulkan `rope_yarn_f32` kernel. Two details decide whether a host matches HF:

- **The ramp runs over the rotary dim, not the head.** HF uses `dim = head_dim * partial_rotary_factor`. For Qwen3.5/3.8
  and Bonsai-2 (head 256, rotary 64) the correction range is (14, 22) over the 64 rotary dims; over 256 it would be a
  different curve entirely.
- **mscale (`attention_factor`) multiplies cos and sin.** `RoPEConfig.ComputeYarnMscaleMultiplier` gives gpt-oss and the
  Qwen3.5/3.8 family (`Qwen3HybridDense`, `Qwen3MoeHybrid`, `Qwen4Exp`) HF's default `0.1·ln(factor) + 1` (1.1386 at
  factor 4), the llama.cpp `ggml_rope_ext` convention. SmolLM3 and Llama keep the plain `AttnFactor`. Because it scales
  every position, YaRN changes short-context numerics too; that is why Qwen recommends enabling it only when needed.

#### Bonsai-2 / Qwen3.8 past 262,144 tokens (#1101)

Qwen3.8 (and Bonsai-2, built from it) trains to 262,144 tokens and supports about 1M with opt-in YaRN, factor 4. The
GGUF carries no rope-scaling keys, so turn it on at load:

```
dotllm serve <bonsai> --device vulkan --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144 --ctx-size 327680
```

- `ApplyRoPEOverride` raises `MaxSequenceLength` to `factor × original` (1,048,576) whenever the result has dense YaRN
  active. Everything that caps the window reads that value: the CUDA VRAM fit (`NativeMaxSequenceLength`), `--ctx-size`
  (now honoured on the Vulkan hybrid as well, via `IContextLimitable`), the KV caches' growth limits and the prompt-length
  checks. It never lowers a window.
- All three hybrid hosts (CPU `Qwen3HybridDenseTransformerModel`, `CudaQwen3HybridDenseTransformerModel`,
  `VulkanQwen3HybridDenseTransformerModel`) rotate the trunk and the MTP head through the YaRN path. Before #1101 they read
  only `Theta` and silently ignored the override.
- Verified against HF's own `_compute_yarn_parameters` (torch float32; script and output in
  `.docs/ctx128k/yarn/`), checked in as constants in `RoPEYarnQwen35ReferenceTests`; cross-backend parity at positions
  past the native window in `*Qwen3HybridDenseYarnParityTests`.
- **Cost.** RoPE itself costs nothing extra: every host keeps one 128-byte inverse-frequency vector. The CPU host
  (`RoPERotation`) computes cos/sin rows only for the positions in flight, so nothing scales with the window (its old
  position-indexed F32 tables would have been 256 MiB at 1M; #1107). The real cost is KV: Bonsai-2 keeps
  16 attention layers × 4 KV heads × 256 dims, so F32 KV is 128 KiB/token (300k ≈ 37 GiB, 1M = 128 GiB), F16 64 KiB/token,
  q8_0 about 34 KiB/token. And prefill: attention is quadratic, so a fresh prompt at that depth takes hours; the
  realistic use is a session grown turn by turn with prefix reuse.
- **Leave it off below 262,144.** The mscale changes every position, so short prompts get (slightly) different output.
- The `Qwen3MoeHybrid` (qwen35moe) and `Qwen4Exp` hosts get the same mscale convention but are not wired yet.

### Dynamic NTK
Scales only when `seq_len > orig_max`. Requires recomputing tables dynamically.

### Runtime Context Extension

RoPE params overridable at inference time. When changed:
1. `InvalidateCache()` — clear cos/sin tables
2. Recompute tables with new params
3. **Invalidate KV-cache** — cached K vectors have stale rotation angles

## ALiBi — Attention with Linear Biases

BLOOM, MPT. Adds bias to attention scores, no Q/K modification:
```
scores[h][i][j] += m_h × (j - i)
m_h = 2^(-8h/H)   (per-head slope)
```
No position limit, no tables. Less common in modern models.

## Absolute Learned Embeddings

GPT-2 style. Lookup table `position_emb[pos]` added to token embedding before first layer. Limited to training context length.

## None

No position encoding applied. Set `PositionEncodingType = None`.
