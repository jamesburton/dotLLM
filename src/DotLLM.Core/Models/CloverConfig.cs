namespace DotLLM.Core.Models;

/// <summary>
/// DiffuSpace Clover (<c>clover_moe</c>, a block-diffusion fine-tune of Qwen3.6-35B-A3B; issue #1794) metadata the
/// <see cref="DotLLM.Core.Configuration.Architecture.Qwen3MoeHybrid"/> host cannot infer from the generic fields. Non-null iff the
/// GGUF declares <c>general.architecture = clover_moe</c>.
/// </summary>
/// <remarks>
/// <para>
/// The causal (autoregressive) forward of Clover is the Qwen3MoeHybrid forward unchanged: llama.cpp PR #31306's
/// <c>src/models/clover-moe.cpp</c> builds exactly the qwen35moe graph. So Clover maps onto <c>Architecture.Qwen3MoeHybrid</c> and keeps
/// every arch-keyed fast path (the #1787 grouped split-KV decode included). Everything here is either inert on that causal path
/// (the block-diffusion parameters, kept for stages 3-5 of #1794) or a load-time tensor-convention fix-up.
/// </para>
/// <para>
/// The official GGUF carries 4 bidirectional-preview tensors per GDN layer that the causal forward never reads:
/// <c>ssm_conv1d.weight_right</c> (HF <c>conv_right_weight</c>), <c>ssm_bd_logit_scale.weight</c> (<c>bd_softmax_logit_scale</c>),
/// <c>ssm_bd_history.bias</c> (<c>bd_softmax_history_bias</c>) and <c>ssm_bd_history.weight</c> (<c>bd_softmax_history_c</c>).
/// They stay in the mmap, unconsumed; the loader names them once at load.
/// </para>
/// </remarks>
public sealed record CloverConfig
{
    /// <summary>Diffusion block length (<c>clover_moe.block_size</c>; HF <c>block_size</c>, 32).</summary>
    public required int BlockSize { get; init; }

    /// <summary><c>&lt;|mask|&gt;</c> token id (<c>tokenizer.ggml.mask_token_id</c>; HF <c>mask_token_id</c>, 151669); -1 if absent.</summary>
    public required int MaskTokenId { get; init; }

    /// <summary>Per-block expert coreset during denoising (<c>clover_moe.moe_block_coreset_enabled</c>). Unused on the causal path.</summary>
    public required bool MoeBlockCoresetEnabled { get; init; }

    /// <summary>Routing-mass fraction the coreset keeps (<c>clover_moe.moe_block_coreset_top_p</c>, 0.9). Unused on the causal path.</summary>
    public required float MoeBlockCoresetTopP { get; init; }
}
