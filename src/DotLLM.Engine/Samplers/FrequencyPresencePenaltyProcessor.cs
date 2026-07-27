using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// Applies OpenAI-style frequency and presence penalties over the full token history:
/// <c>logit -= frequencyPenalty * count(token) + presencePenalty * (count(token) &gt; 0 ? 1 : 0)</c>.
/// Distinct from <see cref="RepetitionPenaltyProcessor"/> (multiplicative, no windowing by default).
/// </summary>
public sealed class FrequencyPresencePenaltyProcessor : ILogitProcessor
{
    /// <inheritdoc/>
    public void Process(Span<float> logits, IReadOnlyList<int> previousTokens, ProcessorContext context)
    {
        float frequencyPenalty = context.FrequencyPenalty;
        float presencePenalty = context.PresencePenalty;
        if (frequencyPenalty == 0f && presencePenalty == 0f)
            return;
        if (previousTokens.Count == 0)
            return;

        var counts = new Dictionary<int, int>();
        for (int i = 0; i < previousTokens.Count; i++)
        {
            int tokenId = previousTokens[i];
            counts[tokenId] = counts.TryGetValue(tokenId, out int c) ? c + 1 : 1;
        }

        foreach (var (tokenId, count) in counts)
        {
            if ((uint)tokenId >= (uint)logits.Length)
                continue;

            logits[tokenId] -= frequencyPenalty * count + presencePenalty;
        }
    }
}
