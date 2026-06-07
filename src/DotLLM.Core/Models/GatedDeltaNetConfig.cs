namespace DotLLM.Core.Models;

/// <summary>
/// Gated DeltaNet (GDN) SSM configuration for
/// <see cref="DotLLM.Core.Configuration.Architecture.Qwen3MoeHybrid"/> models.
/// </summary>
/// <remarks>
/// Per-head state is a full <c>[DState × DState]</c> associative-memory matrix
/// updated via the delta rule:
/// <code>state = g * state + outer(k, β * (v − state @ k))</code>
/// where <c>g = exp(softplus(alpha_proj(h) + dt_bias) * A)</c> is a per-head
/// scalar decay, <c>k</c> is post-conv1d key (shape <c>[n_head, DState]</c>),
/// and <c>β = sigmoid(beta_proj(h))</c> is a per-head write gate.
/// </remarks>
/// <param name="FullAttnInterval">
/// Full-attention layer stride (GGUF: <c>{arch}.full_attention_interval</c>).
/// Layer <c>i</c> (1-indexed) is full GQA when <c>i % FullAttnInterval == 0</c>;
/// all other layers use GDN recurrence. Typically 4, giving 30 GDN + 10 attn
/// layers in the 40-layer Qwen3.6-35B-A3B model.
/// </param>
/// <param name="NVHead">
/// Number of GDN value heads (GGUF: <c>{arch}.ssm.time_step_rank</c>).
/// Equals the first dimension of the per-sequence state matrix.
/// </param>
/// <param name="NKHead">
/// Number of GDN key heads (GGUF: <c>{arch}.ssm.group_count</c>).
/// Key heads are shared: <c>NVHead / NKHead</c> value heads broadcast per key head.
/// </param>
/// <param name="DState">
/// Per-head key and value projection dimension (GGUF: <c>{arch}.ssm.state_size</c>).
/// The full per-sequence GDN state tensor has shape <c>[NVHead, DState, DState]</c>.
/// </param>
/// <param name="DInner">
/// SSM inner dimension (GGUF: <c>{arch}.ssm.inner_size</c>). This is the
/// width of the combined input projection driving Q, K, V, α, β.
/// </param>
/// <param name="DConv">
/// Causal 1-D convolution kernel size applied to the full Q/K/V concatenated
/// projection before the recurrence (GGUF: <c>{arch}.ssm.conv_kernel</c>).
/// </param>
public readonly record struct GatedDeltaNetConfig(
    int FullAttnInterval,
    int NVHead,
    int NKHead,
    int DState,
    int DInner,
    int DConv)
{
    /// <summary>Value heads per key head (<c>NVHead / NKHead</c>).</summary>
    public int VHeadsPerKHead => NVHead / NKHead;

    /// <summary>
    /// Total per-sequence GDN state elements: <c>NVHead × DState × DState</c>.
    /// Each element is F32 — multiply by 4 for bytes.
    /// </summary>
    public int StateElements => NVHead * DState * DState;

    /// <summary>
    /// Per-sequence conv1d rolling-buffer elements for the full Q/K/V concat:
    /// <c>(DConv − 1) × (2 × NKHead + NVHead) × DState</c>.
    /// </summary>
    public int ConvStateElements => (DConv - 1) * (2 * NKHead + NVHead) * DState;
}
