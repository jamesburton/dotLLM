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

    /// <summary>DeepSeek family.</summary>
    DeepSeek,

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
    QwenMoe
}
