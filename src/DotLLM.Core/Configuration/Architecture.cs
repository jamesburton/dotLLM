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
    SmolLM3
}
