namespace DotLLM.Cpu;

/// <summary>
/// Zero-cost-when-off tap on the CPU MoE router (<c>dotllm perplexity --dump-routing</c>, #1706).
/// </summary>
/// <remarks>
/// <para>Off state is <see cref="Sink"/> == <see langword="null"/>: <c>MoeSwiGluMlp.Route</c> pays one static
/// read and a null check per call (per layer, not per token), and the layer loop pays the same to publish
/// <see cref="Layer"/>. No allocation, no event, no delegate.</para>
/// <para>The sink sees the router's FINAL decision per (token, slot): the selected expert id and the gate weight the
/// expert output is multiplied by (after softmax / renormalisation / scale, i.e. what the model actually uses).
/// The probe is process-global and the forward runs on one caller thread, so arm it for one model at a time.</para>
/// </remarks>
public static class MoeRoutingProbe
{
    /// <summary>Receives one router decision batch.</summary>
    public interface ISink
    {
        /// <summary>Called once per MoE layer invocation, in token order.</summary>
        /// <param name="layer">Layer index published by the forward loop (-1 if the caller does not publish it).</param>
        /// <param name="seqLen">Tokens in this invocation.</param>
        /// <param name="topK">Experts per token.</param>
        /// <param name="experts">Selected expert ids, token-major <c>[seqLen * topK]</c>, slot order = descending router score. Valid only for the call.</param>
        /// <param name="weights">Gate weights, same layout as <paramref name="experts"/>. Valid only for the call.</param>
        void Record(int layer, int seqLen, int topK, ReadOnlySpan<int> experts, ReadOnlySpan<float> weights);
    }

    /// <summary>The active sink, or <see langword="null"/> (off).</summary>
    public static volatile ISink? Sink;

    /// <summary>Layer index of the MoE layer currently executing; written by the forward loop only while <see cref="Sink"/> is set.</summary>
    public static int Layer = -1;
}
