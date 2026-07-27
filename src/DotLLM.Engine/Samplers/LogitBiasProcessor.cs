using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// Applies a per-token additive logit bias, OpenAI API compatible (<c>logit_bias</c>):
/// <c>logits[token_id] += bias</c>. Out-of-range token ids are ignored.
/// </summary>
public sealed class LogitBiasProcessor : ILogitProcessor
{
    /// <inheritdoc/>
    public void Process(Span<float> logits, IReadOnlyList<int> previousTokens, ProcessorContext context)
    {
        var bias = context.LogitBias;
        if (bias is null || bias.Count == 0)
            return;

        foreach (var (tokenId, value) in bias)
        {
            if ((uint)tokenId >= (uint)logits.Length)
                continue;
            logits[tokenId] += value;
        }
    }
}
