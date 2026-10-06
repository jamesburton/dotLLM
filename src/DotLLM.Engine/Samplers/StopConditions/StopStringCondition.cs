using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers.StopConditions;

/// <summary>
/// Stops generation when the decoded text ends with a specified stop string.
/// The stop string is excluded from the output.
/// </summary>
public sealed class StopStringCondition : IStopCondition
{
    private readonly string _stopString;
    private readonly string? _armedAfter;
    private bool _armed;

    /// <summary>
    /// Creates a new stop string condition.
    /// </summary>
    /// <param name="stopString">The string that triggers generation stop.</param>
    public StopStringCondition(string stopString)
    {
        _stopString = stopString;
    }

    /// <summary>
    /// Creates a stop string condition that only fires on text produced after
    /// <paramref name="armedAfter"/> has been generated (#767: a reasoning block's closing tag).
    /// The armed flag is per-instance state, so build one condition set per generated sequence.
    /// </summary>
    /// <param name="stopString">The string that triggers generation stop.</param>
    /// <param name="armedAfter">Marker that arms the condition; null = always armed.</param>
    public StopStringCondition(string stopString, string? armedAfter)
    {
        _stopString = stopString;
        _armedAfter = string.IsNullOrEmpty(armedAfter) ? null : armedAfter;
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
        if (!decodedTail.EndsWith(_stopString.AsSpan(), StringComparison.Ordinal))
        {
            // Still watch for the arming marker so a later tail that no longer contains it is armed.
            if (_armedAfter is not null && !_armed && decodedTail.IndexOf(_armedAfter.AsSpan(), StringComparison.Ordinal) >= 0)
                _armed = true;
            return StopResult.Continue;
        }

        if (_armedAfter is null)
            return StopResult.Stop;

        // The match must lie entirely after the marker and the whitespace that follows it (the template's
        // "\n\n" separator is not answer text a stop string should be able to hit).
        int idx = decodedTail.LastIndexOf(_armedAfter.AsSpan(), StringComparison.Ordinal);
        int contentStart = 0;
        if (idx >= 0)
        {
            _armed = true;
            contentStart = idx + _armedAfter.Length;
            while (contentStart < decodedTail.Length && char.IsWhiteSpace(decodedTail[contentStart]))
                contentStart++;
        }
        if (!_armed)
            return StopResult.Continue;
        return decodedTail.Length - _stopString.Length >= contentStart ? StopResult.Stop : StopResult.Continue;
    }

    /// <summary>
    /// Builds the per-sequence stop-string conditions for <paramref name="options"/>, honouring
    /// <see cref="DotLLM.Core.Configuration.InferenceOptions.StopSequencesArmedAfter"/>. Call once per
    /// generated sequence: the armed state lives in the returned instances.
    /// </summary>
    public static List<StopStringCondition> CreateAll(DotLLM.Core.Configuration.InferenceOptions options)
    {
        var list = new List<StopStringCondition>(options.StopSequences.Count);
        string? marker = options.StopSequencesArmedAfter;
        foreach (string seq in options.StopSequences)
        {
            bool gated = !string.IsNullOrEmpty(marker)
                && (options.StopSequencesUngated is null || !options.StopSequencesUngated.Contains(seq));
            list.Add(new StopStringCondition(seq, gated ? marker : null));
        }
        return list;
    }
}
