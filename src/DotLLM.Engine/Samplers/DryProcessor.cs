using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// DRY (Don't Repeat Yourself) repetition penalty (llama.cpp <c>--dry-multiplier</c> family).
/// Detects, for the token immediately following the current position, whether generating it would
/// continue a previously-seen repeated n-gram ending at the most recent token. Candidates that would
/// extend a match of length ≥ <see cref="ProcessorContext.DryAllowedLength"/> receive an exponentially
/// growing penalty as the matched length increases — meaningfully better than plain repetition
/// penalty at killing verbatim loops without flattening the rest of the distribution.
/// </summary>
/// <remarks>
/// Sequence-breaker tokens (resolved from <see cref="ProcessorContext.DrySequenceBreakers"/> by the
/// caller, typically <c>SamplerPipeline</c>, since resolving text to token ids needs a tokenizer) stop
/// n-gram extension — a match never crosses a breaker token, and a breaker token can never itself
/// anchor a match.
/// </remarks>
public sealed class DryProcessor : ILogitProcessor
{
    private readonly IReadOnlySet<int>? _breakerTokenIds;

    /// <summary>Creates a DRY processor that reads all parameters from <see cref="ProcessorContext"/>.</summary>
    /// <param name="breakerTokenIds">
    /// Token ids that act as sequence breakers (n-gram matches cannot cross or start on them).
    /// Null/empty = no breakers (matches can span the entire window).
    /// </param>
    public DryProcessor(IReadOnlySet<int>? breakerTokenIds = null) => _breakerTokenIds = breakerTokenIds;

    /// <inheritdoc/>
    public void Process(Span<float> logits, IReadOnlyList<int> previousTokens, ProcessorContext context)
    {
        float multiplier = context.DryMultiplier;
        if (multiplier <= 0f)
            return;

        int n = previousTokens.Count;
        if (n < 2)
            return;

        int windowSize = context.DryPenaltyLastN > 0 ? Math.Min(n, context.DryPenaltyLastN) : n;
        if (windowSize < 2)
            return;

        int start = n - windowSize;
        var history = new int[windowSize];
        for (int i = 0; i < windowSize; i++)
            history[i] = previousTokens[start + i];

        int allowedLength = Math.Max(1, context.DryAllowedLength);
        float baseValue = context.DryBase;

        int last = windowSize - 1;
        int tailToken = history[last];

        // best[candidateToken] = longest matched n-gram length found for that candidate.
        Dictionary<int, int>? best = null;

        for (int i = 0; i < last; i++)
        {
            if (history[i] != tailToken || IsBreaker(history[i]))
                continue;

            // Extend the match backward from the anchor at `i` (paired with `last`).
            int maxOffset = Math.Min(i, last - i - 1);
            int offset = 1;
            while (offset <= maxOffset)
            {
                int a = history[i - offset];
                int b = history[last - offset];
                if (a != b || IsBreaker(a) || IsBreaker(b))
                    break;
                offset++;
            }

            int matchLength = offset; // includes the anchor token itself (offset counts matched pairs, min 1)
            if (matchLength < allowedLength)
                continue;

            int candidate = history[i + 1];
            best ??= new Dictionary<int, int>();
            if (!best.TryGetValue(candidate, out int current) || matchLength > current)
                best[candidate] = matchLength;
        }

        if (best is null)
            return;

        foreach (var (token, matchLength) in best)
        {
            if ((uint)token >= (uint)logits.Length)
                continue;

            float penalty = multiplier * MathF.Pow(baseValue, matchLength - allowedLength);
            logits[token] -= penalty;
        }
    }

    private bool IsBreaker(int token) => _breakerTokenIds is not null && _breakerTokenIds.Contains(token);
}
