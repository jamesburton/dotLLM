using System.Buffers;
using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// Applies repetition penalty to logits for tokens that appeared in recent history.
/// Positive logits are divided by the penalty factor, negative logits are multiplied,
/// effectively reducing the probability of repeated tokens in both cases.
/// </summary>
public sealed class RepetitionPenaltyProcessor : ILogitProcessor
{
    private const int StackWindowThreshold = 256;

    /// <inheritdoc/>
    public void Process(Span<float> logits, IReadOnlyList<int> previousTokens, ProcessorContext context)
    {
        float penalty = context.RepetitionPenalty;
        if (penalty == 1.0f || previousTokens.Count == 0)
            return;

        int window = context.RepetitionPenaltyWindow;
        int startIndex = window > 0 ? Math.Max(0, previousTokens.Count - window) : 0;
        int windowLength = previousTokens.Count - startIndex;

        if (windowLength <= StackWindowThreshold)
        {
            Span<int> scratch = stackalloc int[windowLength];
            CopyWindow(previousTokens, startIndex, scratch);
            ApplySortedPenalty(logits, scratch, penalty);
        }
        else
        {
            int[] rented = ArrayPool<int>.Shared.Rent(windowLength);
            try
            {
                Span<int> scratch = rented.AsSpan(0, windowLength);
                CopyWindow(previousTokens, startIndex, scratch);
                ApplySortedPenalty(logits, scratch, penalty);
            }
            finally
            {
                ArrayPool<int>.Shared.Return(rented);
            }
        }
    }

    private static void CopyWindow(IReadOnlyList<int> previousTokens, int startIndex, Span<int> scratch)
    {
        for (int i = 0; i < scratch.Length; i++)
            scratch[i] = previousTokens[startIndex + i];
    }

    private static void ApplySortedPenalty(Span<float> logits, Span<int> tokenIds, float penalty)
    {
        tokenIds.Sort();

        int prev = -1;
        for (int i = 0; i < tokenIds.Length; i++)
        {
            int tokenId = tokenIds[i];
            if (tokenId == prev)
                continue;
            prev = tokenId;

            if ((uint)tokenId >= (uint)logits.Length)
                continue;

            if (logits[tokenId] > 0f)
                logits[tokenId] /= penalty;
            else
                logits[tokenId] *= penalty;
        }
    }
}
