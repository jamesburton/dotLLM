using DotLLM.Core.Configuration;
using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers.StopConditions;

/// <summary>
/// Stops generation when the decoded text ends with a specified stop string.
/// The stop string is excluded from the output.
/// </summary>
public sealed class StopStringCondition : IStopCondition
{
    private readonly string _stopString;
    private readonly StopGate? _gate;
    private bool _inside;
    private bool _startDecided;

    /// <summary>
    /// Creates a new stop string condition.
    /// </summary>
    /// <param name="stopString">The string that triggers generation stop.</param>
    public StopStringCondition(string stopString)
    {
        _stopString = stopString;
    }

    /// <summary>
    /// Creates a stop string condition that is suspended while the model is inside a reasoning block
    /// (#767). The "inside" state is per-instance, so build one condition set per generated sequence.
    /// </summary>
    /// <param name="stopString">The string that triggers generation stop.</param>
    /// <param name="gate">The reasoning gate; null = always active.</param>
    public StopStringCondition(string stopString, StopGate? gate)
    {
        _stopString = stopString;
        _gate = gate;
        _inside = gate?.StartsInside ?? false;
    }

    /// <summary>The stop string this condition matches against. Caller ensures the decoded tail
    /// view it passes is at least this long.</summary>
    public string StopString => _stopString;

    /// <inheritdoc/>
    public StopResult ShouldStop(int tokenId, IReadOnlyList<int> generatedTokens, string decodedText)
        => ShouldStop(tokenId, generatedTokens, decodedText.AsSpan());

    /// <inheritdoc/>
    public StopResult ShouldStop(int tokenId, IReadOnlyList<int> generatedTokens, ReadOnlySpan<char> decodedTail)
    {
        int contentStart = 0;
        if (_gate is { } g)
        {
            // Track whether the model is inside a reasoning block from the decoded tail. The tail holds at
            // least the last 64 characters, so a tag completed by the newest token is always visible; the
            // flag persists once the tag has scrolled out of the window.
            int open = decodedTail.LastIndexOf(g.OpenTag.AsSpan(), StringComparison.Ordinal);
            int close = decodedTail.LastIndexOf(g.CloseTag.AsSpan(), StringComparison.Ordinal);

            if (!_startDecided)
            {
                // Output start: did the model open a block itself? Wait while the text could still become
                // the open tag ("<thi"), decide at the first text that cannot.
                var first = decodedTail.TrimStart();
                if (first.Length > 0
                    && !(first.Length < g.OpenTag.Length && g.OpenTag.AsSpan().StartsWith(first, StringComparison.Ordinal)))
                {
                    _startDecided = true;
                    if (first.StartsWith(g.OpenTag.AsSpan(), StringComparison.Ordinal))
                        _inside = true;
                }
            }
            if (!g.OpenOnlyAtStart && open >= 0 && open > close)
                _inside = true;
            if (close >= 0 && close > open)
            {
                _inside = false;
                contentStart = close + g.CloseTag.Length;
                while (contentStart < decodedTail.Length && char.IsWhiteSpace(decodedTail[contentStart]))
                    contentStart++;
            }

            if (_inside)
                return StopResult.Continue;
        }

        if (!decodedTail.EndsWith(_stopString.AsSpan(), StringComparison.Ordinal))
            return StopResult.Continue;
        return decodedTail.Length - _stopString.Length >= contentStart ? StopResult.Stop : StopResult.Continue;
    }

    /// <summary>
    /// Builds the per-sequence stop-string conditions for <paramref name="options"/>, honouring
    /// <see cref="DotLLM.Core.Configuration.InferenceOptions.ReasoningStopGate"/>. Call once per
    /// generated sequence: the armed state lives in the returned instances.
    /// </summary>
    public static List<StopStringCondition> CreateAll(DotLLM.Core.Configuration.InferenceOptions options)
    {
        var list = new List<StopStringCondition>(options.StopSequences.Count);
        var gate = options.ReasoningStopGate;
        foreach (string seq in options.StopSequences)
        {
            bool gated = gate is not null
                && (options.StopSequencesUngated is null || !options.StopSequencesUngated.Contains(seq));
            list.Add(new StopStringCondition(seq, gated ? gate : null));
        }
        return list;
    }
}
