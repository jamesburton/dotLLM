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
    Mamba3
}
