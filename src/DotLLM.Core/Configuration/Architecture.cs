namespace DotLLM.Core.Configuration;

/// <summary>
/// Supported model architectures.
/// </summary>
public enum Architecture
{
    /// <summary>Meta Llama family.</summary>
    Llama,

    /// <summary>Mistral AI family.</summary>
    Mistral,

    /// <summary>Microsoft Phi family.</summary>
    Phi,

    /// <summary>Alibaba Qwen family.</summary>
    Qwen,

    /// <summary>
    /// DeepSeek family (pre-V2; legacy GGUF metadata placeholder).
    /// Prefer <see cref="DeepSeekV2"/> or <see cref="DeepSeekV3"/> for supported
    /// MLA-based checkpoints.
    /// </summary>
    [System.Obsolete("Architecture.DeepSeek is a legacy pre-V2 placeholder kept for GGUF metadata compatibility. Use DeepSeekV2 or DeepSeekV3.")]
    DeepSeek,

    /// <summary>
    /// DeepSeek-V2 family (<c>model_type=deepseek_v2</c>,
    /// <c>architectures[0]=DeepseekV2ForCausalLM</c>). Multi-head Latent
    /// Attention (MLA) with low-rank Q/KV factorisation + decoupled RoPE,
    /// combined with dense MoE in later layers (governed by
    /// <c>first_k_dense_replace</c>). Lite variant: 16 heads, qk_nope=128,
    /// qk_rope=64, v_head=128, kv_lora_rank=512, q_lora_rank=1536. Carries
    /// optional YaRN rope scaling. See <see cref="DotLLM.Core.Models.MlaConfig"/>.
    /// </summary>
    DeepSeekV2,

    /// <summary>
    /// DeepSeek-V3 family (<c>model_type=deepseek_v3</c>,
    /// <c>architectures[0]=DeepseekV3ForCausalLM</c>). Same MLA attention
    /// mechanism as V2 plus V3-specific MoE refinements (sigmoid router
    /// scoring, node-level aux-loss-free routing) — wired into the same
    /// <see cref="DotLLM.Core.Models.MlaConfig"/> for the attention side.
    /// </summary>
    DeepSeekV3,

    /// <summary>
    /// NVIDIA Nemotron-H — hybrid Mamba2 SSM + Transformer attention/MLP per layer.
    /// Used by the Nemotron-3 family. GGUF architecture string: <c>nemotron_h</c>.
    /// </summary>
    NemotronH,

    /// <summary>
    /// Mamba-3 (Lahoti et al., ICLR 2026, arXiv 2603.15569) — pure SSM with
    /// trapezoidal discretization, data-dependent RoPE on B/C, and optional
    /// MIMO. No convolution, no attention. HF <c>model_type</c>: <c>mamba3</c>.
    /// GGUF has no upstream mapping as of 2026-04-19, so loading is
    /// safetensors-first (see <see cref="DotLLM.Core.Models.Mamba3Config"/>).
    /// </summary>
    Mamba3,

    /// <summary>
    /// Mistral Mixtral family — dense transformer with top-k MoE FFN in every
    /// layer. HF <c>model_type</c>: <c>mixtral</c>. Same attention path as
    /// <see cref="Mistral"/> (GQA, RoPE, no sliding window by default); the
    /// MLP is replaced by <c>num_local_experts</c> parallel SwiGLU experts
    /// with <c>num_experts_per_tok</c> active per token. Shared experts are
    /// <b>not</b> a Mixtral thing (DeepSeek-V3 / old Qwen1.5-MoE territory,
    /// tracked separately). See <see cref="DotLLM.Core.Models.MoeConfig"/>.
    /// </summary>
    Mixtral,

    /// <summary>
    /// Alibaba Qwen-MoE family — Qwen1.5-MoE-A2.7B (<c>model_type=qwen2_moe</c>),
    /// Qwen2-MoE, Qwen3-MoE (<c>model_type=qwen3_moe</c>). Shares the Qwen
    /// attention path (GQA, NeoX-pair RoPE, optional sliding window, Qwen3
    /// QK-norm) with the dense <see cref="Qwen"/> variant but replaces the
    /// FFN with a top-k MoE block using HF tensor names
    /// <c>mlp.gate</c> + <c>mlp.experts.{j}.{gate_proj,up_proj,down_proj}</c>
    /// (NOT Mixtral's <c>block_sparse_moe.gate</c> / <c>experts.{j}.w1/w2/w3</c>).
    /// Optional shared-expert branch — a dense SwiGLU MLP running in parallel
    /// on EVERY token, optionally gated by a <c>sigmoid(hidden @ shared_expert_gate)</c>
    /// scalar — is present on Qwen1.5-MoE-A2.7B but absent on Qwen3-MoE.
    /// Qwen3-MoE further interleaves dense-MLP and MoE layers via
    /// <c>decoder_sparse_step</c> and <c>mlp_only_layers</c>. See
    /// <see cref="DotLLM.Core.Models.MoeConfig"/> for the per-layer flags.
    /// </summary>
    QwenMoe,

    /// <summary>
    /// IBM Granite-3.x MoE family — Granite-3.0-3B-A800M-instruct and larger
    /// variants (<c>model_type=granitemoe</c>,
    /// <c>architectures[0]=GraniteMoeForCausalLM</c>). Standard GQA attention
    /// (separate <c>q/k/v/o_proj</c>) paired with a Mixtral-shaped top-k MoE
    /// FFN whose per-expert weights are packed as three fused tensors:
    /// <c>block_sparse_moe.router.layer.weight [E, H]</c>,
    /// <c>block_sparse_moe.input_linear.weight [E, 2*I, H]</c> (per-expert
    /// w1 rows [0:I) + w3 rows [I:2*I)), and
    /// <c>block_sparse_moe.output_linear.weight [E, H, I]</c> (per-expert w2).
    /// No shared expert; top-k is unusually large (8 of 40 for the 3B-A800M
    /// SKU). See <see cref="DotLLM.Core.Models.MoeConfig"/>.
    /// </summary>
    GraniteMoe,

    /// <summary>
    /// Alibaba Qwen3.6 — Gated DeltaNet (GDN) SSM + sparse MoE hybrid.
    /// GGUF architecture string: <c>qwen35moe</c>.
    /// Layers alternate between Gated DeltaNet recurrence (3 of every 4) and full
    /// GQA attention (every 4th layer, controlled by
    /// <c>qwen35moe.full_attention_interval</c>). Both layer types share a sparse
    /// MoE FFN sublayer. The GDN carries a full <c>[n_head, d_k, d_v]</c> matrix
    /// state updated via the delta rule. See
    /// <see cref="DotLLM.Core.Models.GatedDeltaNetConfig"/> for GDN-specific parameters.
    /// References: <see href="https://arxiv.org/abs/2412.06464">Gated Delta Networks
    /// (NVlabs, ICLR 2025)</see>; <see href="https://arxiv.org/abs/2505.09388">Qwen3
    /// Technical Report</see>.
    /// </summary>
    Qwen3MoeHybrid,

    /// <summary>
    /// HuggingFace SmolLM3 — Llama-shaped GQA transformer (3B SKU: 36 layers,
    /// 16 Q-heads, 4 KV-heads, head_dim=128, hidden=2048, intermediate=11008,
    /// vocab=128256, max_pos=65536). Distinguishing features vs Llama:
    /// <list type="bullet">
    ///   <item><b>NoPE layers</b> — every 4th layer (indices 3, 7, 11, ...
    ///     in the 3B config, supplied via the HF <c>no_rope_layers</c> 0/1
    ///     mask) skips RoPE entirely, attending on position-free Q/K.
    ///     Threaded as a per-layer boolean inside the forward dispatch;
    ///     see <see cref="DotLLM.Core.Models.ModelConfig.NoRopeLayers"/>.</item>
    ///   <item><b>YaRN context extension</b> — long-context SKUs ship
    ///     <c>rope_scaling</c> (<c>type=yarn</c>, factor &gt; 1) that lifts
    ///     the effective context to 128k. The base 3B checkpoint ships
    ///     <c>rope_scaling=null</c>; YaRN fields are surfaced via
    ///     <see cref="DotLLM.Core.PositionEncoding.RoPEConfig.ScalingType"/>.</item>
    ///   <item>Tool calling — Hermes-compatible
    ///     <c>&lt;tool_call&gt;{...}&lt;/tool_call&gt;</c> XML wrapper or
    ///     Pythonic <c>function_name(arg=value, ...)</c> syntax.</item>
    /// </list>
    /// HF discriminators: <c>architectures[0]=SmolLM3ForCausalLM</c>,
    /// <c>model_type=smollm3</c>. Tensor naming is identical to Llama, so
    /// loading routes through the standard <see cref="Llama"/> safetensors
    /// path with a conditional RoPE per layer.
    /// </summary>
    SmolLM3,

    /// <summary>
    /// Google Gemma 3 family. Text-only checkpoints carry
    /// <c>model_type=gemma3_text</c> + <c>architectures[0]=Gemma3TextForCausalLM</c>;
    /// multimodal checkpoints carry <c>model_type=gemma3</c> +
    /// <c>architectures[0]=Gemma3ForConditionalGeneration</c> and house the text-tower
    /// config under a <c>text_config</c> sub-object (we read only the text tower).
    /// <para>
    /// Distinguishing features vs Llama/Mistral/Qwen:
    /// <list type="bullet">
    ///   <item><b>GeGLU MLP</b> — gate path uses GELU (tanh approximation, HF
    ///     <c>hidden_activation=gelu_pytorch_tanh</c>) instead of SiLU; otherwise
    ///     shape-identical to SwiGLU (<c>down(act(gate(x)) * up(x))</c>).</item>
    ///   <item><b>RMSNorm <c>(1 + weight)</c> convention</b> — every RMSNorm scales by
    ///     <c>(1 + w)</c> rather than <c>w</c>. Absorbed at load time by pre-adding
    ///     <c>1.0</c> to every dequantised norm weight, so the existing kernel runs
    ///     unchanged.</item>
    ///   <item><b>Four RMSNorms per layer</b> — <c>input_layernorm</c>,
    ///     <c>post_attention_layernorm</c>, <c>pre_feedforward_layernorm</c>,
    ///     <c>post_feedforward_layernorm</c>. The post-attn and post-FFN norms run
    ///     <i>between</i> the sublayer output and the residual add (Gemma 2 design,
    ///     retained in Gemma 3).</item>
    ///   <item><b>Per-head Q/K RMSNorms</b> applied before RoPE (dim = <c>head_dim</c>,
    ///     identical plumbing to Qwen3 QK-norm).</item>
    ///   <item><b>Interleaved local/global attention</b> — most layers use
    ///     sliding-window attention of size <c>sliding_window</c>; every
    ///     <c>sliding_window_pattern</c>-th layer (1-indexed) uses full attention.
    ///     Encoded as a per-layer attention-type list on
    ///     <see cref="DotLLM.Core.Models.ModelConfig.PerLayerSlidingWindow"/>.</item>
    ///   <item><b>Query-pre-attn scalar</b> — attention score scale is
    ///     <c>1 / sqrt(query_pre_attn_scalar)</c> instead of the default
    ///     <c>1 / sqrt(head_dim)</c>.</item>
    ///   <item><b>Logit soft-capping</b> — optional <c>tanh(z / cap) * cap</c> applied
    ///     to attention scores (<c>attn_logit_softcapping</c>) and the final LM-head
    ///     logits (<c>final_logit_softcapping</c>). Gemma 2 sets both (50.0 and 30.0);
    ///     Gemma 3 leaves them null but the plumbing is wired regardless via
    ///     <see cref="DotLLM.Core.Models.ModelConfig.AttnLogitSoftcap"/> and
    ///     <see cref="DotLLM.Core.Models.ModelConfig.FinalLogitSoftcap"/>.</item>
    /// </list>
    /// </para>
    /// <para>
    /// Gemma 4 was not yet public when this enum variant was added (2026-05). The
    /// implementation is built for Gemma 3 — the latest publicly-shipped Gemma — and
    /// will forward-port to Gemma 4 cleanly if Google retains the Gemma 2/3 shape. See
    /// <c>.continue-here-step57.md</c> at the repo root for the assumption trail and
    /// the upgrade path when Gemma 4 lands.
    /// </para>
    /// </summary>
    Gemma3,

    /// <summary>
    /// Google Gemma 4 family — the text tower powering DiffusionGemma. Retains the
    /// Gemma 2/3 backbone shape (four RMSNorms per layer, <c>(1 + weight)</c> RMSNorm
    /// convention, GeGLU activation, <c>sqrt(hidden)</c> embedding scaling, per-head
    /// Q/K RMSNorms, interleaved local/global attention, final-logit soft-capping)
    /// and is therefore handled by the same <c>DotLLM.Models</c> transformer
    /// path as <see cref="Gemma3"/> (see
    /// <see cref="DotLLM.Core.Models.ModelConfig.IsGemmaArchitecture"/>).
    /// <para>
    /// <b>Deltas vs <see cref="Gemma3"/>:</b>
    /// <list type="bullet">
    ///   <item><b>Sparse MoE FFN</b> — the dense GeGLU MLP is replaced by a top-k
    ///     dense-routing Mixture-of-Experts block (<c>num_experts</c> experts,
    ///     <c>top_k_experts</c> active per token, per-expert width
    ///     <c>moe_intermediate_size</c>). The MoE variant is identified by
    ///     <see cref="DotLLM.Core.Models.ModelConfig.Moe"/> being non-null. The
    ///     experts are GeGLU (not SwiGLU), honouring
    ///     <see cref="DotLLM.Core.Models.ModelConfig.ActivationFunction"/> exactly as
    ///     the dense Gemma FFN does. Tensor naming follows the Qwen-MoE convention
    ///     (<c>mlp.gate</c> router + <c>mlp.experts.{j}.{gate,up,down}_proj</c>).</item>
    ///   <item><b>Per-attention-type RoPE</b> — full-attention layers and
    ///     sliding-window layers use different RoPE configurations (different
    ///     <c>rope_theta</c>, and the full-attention layers additionally apply a
    ///     <c>partial_rotary_factor</c> so only the leading fraction of each head
    ///     is rotated). Surfaced via
    ///     <see cref="DotLLM.Core.Models.ModelConfig.GlobalRoPEConfig"/> +
    ///     <see cref="DotLLM.Core.Models.ModelConfig.PartialRotaryFactor"/>; the
    ///     sliding layers keep the standard
    ///     <see cref="DotLLM.Core.Models.ModelConfig.RoPEConfig"/>.</item>
    ///   <item><b>Dual KV-head counts / head dims</b> — sliding-window layers use
    ///     <c>num_key_value_heads</c> KV heads at <c>head_dim</c>; full-attention
    ///     layers use <c>num_global_key_value_heads</c> KV heads at
    ///     <c>global_head_dim</c>. Surfaced via
    ///     <see cref="DotLLM.Core.Models.ModelConfig.NumGlobalKvHeads"/> +
    ///     <see cref="DotLLM.Core.Models.ModelConfig.GlobalHeadDim"/>. The current
    ///     CPU forward path supports a uniform head_dim across layer types; a
    ///     checkpoint whose full-attention head_dim differs from its sliding
    ///     head_dim is rejected with a clear deferral message until the shared
    ///     forward-state / KV-cache plumbing is generalised.</item>
    /// </list>
    /// </para>
    /// <para>
    /// HF discriminators: <c>architectures[0]=Gemma4ForCausalLM</c> /
    /// <c>Gemma4TextForCausalLM</c>, <c>model_type=gemma4</c> / <c>gemma4_text</c>.
    /// The DiffusionGemma wrapper (<c>model_type=diffusion_gemma</c> /
    /// <c>diffusion_gemma_text</c>) embeds a Gemma-4 text tower; loading the
    /// diffusion wrapper itself is <see cref="DiffusionGemma"/> (issue #29).
    /// </para>
    /// </summary>
    Gemma4,

    /// <summary>
    /// Google DiffusionGemma — a <b>block masked-diffusion</b> text model built on the
    /// Gemma-4 MoE backbone. HF discriminators: <c>model_type=diffusion_gemma</c>
    /// (text-tower-only checkpoints carry <c>model_type=diffusion_gemma_text</c>),
    /// <c>architectures=["DiffusionGemmaForBlockDiffusion"]</c>.
    /// <para>
    /// <b>Backbone.</b> The transformer tower is byte-identical in shape to
    /// <see cref="Gemma4"/> — four RMSNorms per layer, the <c>(1 + weight)</c>
    /// RMSNorm convention, GeGLU sparse MoE experts, <c>sqrt(hidden)</c> embedding
    /// scaling, per-head Q/K RMSNorms, interleaved local/global attention with
    /// per-attention-type RoPE (full layers <c>rope_theta = 1e6</c> +
    /// <c>partial_rotary_factor = 0.25</c>; sliding layers <c>rope_theta = 1e4</c>),
    /// dual KV-head counts, and final-logit soft-capping. It is therefore handled by
    /// the same <c>DotLLM.Models</c> transformer forward path as <see cref="Gemma4"/>
    /// (<see cref="DotLLM.Core.Models.ModelConfig.IsGemmaArchitecture"/> includes both).
    /// The real 26B-A4B SKU: hidden 2816, 30 layers, 16 heads, 8 sliding-KV / 2
    /// global-KV heads, head_dim 256 (global_head_dim 512), 128 experts top-8,
    /// vocab 262144.
    /// </para>
    /// <para>
    /// <b>Diffusion delta.</b> Instead of autoregressive next-token decoding, the
    /// model refines a fixed-length <c>canvas_length</c> (default 256) canvas of
    /// masked positions over a denoising schedule, attending bidirectionally over the
    /// canvas while cross-attending the causal prompt prefix (the hybrid attention
    /// mask). The diffusion decode parameters (canvas length, denoise schedule,
    /// tokenizer-resolved mask token id) are carried on
    /// <see cref="DotLLM.Core.Models.ModelConfig.DiffusionConfig"/> and consumed by
    /// <c>DiffusionTextGenerator</c>; the backbone weights load through the standard
    /// Gemma-4 MoE safetensors path.
    /// </para>
    /// </summary>
    DiffusionGemma,

    /// <summary>Microsoft BitNet b1.58 family (ternary-weight, I2_S quantization).</summary>
    BitNet
}
