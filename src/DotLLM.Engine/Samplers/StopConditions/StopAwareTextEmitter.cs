using System;
using System.Collections.Generic;
using System.Text;
using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers.StopConditions;

/// <summary>
/// Drives a text-level streaming callback (issue #424) such that the concatenation of every
/// emitted fragment equals the text the non-streaming path returns — stop-string suffix
/// trimmed at the character boundary, and the stop string itself never emitted.
/// </summary>
/// <remarks>
/// <para>
/// A token-id callback cannot express this. Once a stop string overlaps only the
/// <em>suffix</em> of the final token (<c>"ld&lt;|im_end|&gt;"</c> against stop string
/// <c>"&lt;|im_end|&gt;"</c>), there is no token id whose text is <c>"ld"</c>; see
/// <see cref="StopSuffixTrimmer"/>, which performs the same trim on the returned text.
/// </para>
/// <para>
/// Emission cannot simply follow the per-token decode delta, because a stop string may span
/// several tokens: with stop string <c>"XY"</c> and tokens <c>"ab"</c>, <c>"X"</c>, <c>"Y"</c>
/// the returned text is <c>"ab"</c>, yet a delta-driven callback would already have emitted
/// <c>"abX"</c> and cannot un-emit it. So this type <b>withholds</b> the longest suffix of the
/// pending text that is a proper prefix of some registered stop string; those characters are
/// released only once a later token proves they were not the start of a match. That makes the
/// held-back region always contain any eventual match in full, so
/// <see cref="FlushTrimmingStopSuffix"/> can drop exactly the matched characters.
/// </para>
/// <para>
/// Entirely inert when the callback is <see langword="null"/>: <see cref="IsActive"/> is false
/// and every method returns immediately, so the id-only hot path is unchanged.
/// </para>
/// </remarks>
internal sealed class StopAwareTextEmitter
{
    /// <summary>Shared placeholder for the inert case; never written to.</summary>
    private static readonly StringBuilder EmptyBuffer = new(0);

    private readonly Action<ReadOnlySpan<char>>? _onText;
    private readonly IReadOnlyList<IStopCondition> _conditions;
    private readonly int _maxStopLength;
    private readonly StringBuilder _pending;

    /// <summary>
    /// Creates an emitter for <paramref name="onText"/>. When <paramref name="onText"/> is
    /// <see langword="null"/> the emitter is inert.
    /// </summary>
    /// <param name="onText">The text-level callback, or null.</param>
    /// <param name="conditions">All registered stop conditions. Only
    /// <see cref="StopStringCondition"/> entries influence withholding.</param>
    public StopAwareTextEmitter(Action<ReadOnlySpan<char>>? onText, IReadOnlyList<IStopCondition> conditions)
    {
        _onText = onText;
        _conditions = conditions;

        if (onText is null)
        {
            // Inert: skip the buffer allocation and the scan entirely.
            _pending = EmptyBuffer;
            _maxStopLength = 0;
            return;
        }

        _pending = new StringBuilder();

        int max = 0;
        for (int i = 0; i < conditions.Count; i++)
        {
            if (conditions[i] is StopStringCondition ssc && ssc.StopString.Length > max)
                max = ssc.StopString.Length;
        }
        _maxStopLength = max;
    }

    /// <summary>True when a callback is attached; false makes every operation a no-op.</summary>
    public bool IsActive => _onText is not null;

    /// <summary>
    /// Adds a decode delta and emits everything that cannot be the beginning of a stop string.
    /// </summary>
    /// <param name="delta">Newly decoded text, as produced by the incremental detokenizer.</param>
    public void Append(string delta)
    {
        if (_onText is null || delta.Length == 0)
            return;

        _pending.Append(delta);
        EmitAllBut(HeldBackLength());
    }

    /// <summary>
    /// Terminal flush for a stop-string match: drops the matched stop-string suffix at the
    /// character boundary and emits the characters that were kept — the partial-token text the
    /// id-level callback cannot represent.
    /// </summary>
    /// <param name="finalDelta">
    /// Decoded text of the token that triggered the stop. It is buffered rather than passed
    /// through <see cref="Append"/>, because the completed stop string is by definition not a
    /// <em>proper</em> prefix of itself and so would not be withheld — <see cref="Append"/>
    /// would emit it before there was any chance to trim it.
    /// </param>
    public void FlushTrimmingStopSuffix(string finalDelta)
    {
        if (_onText is null)
            return;

        _pending.Append(finalDelta);

        // The pending buffer is guaranteed to hold the whole match: every proper prefix of a
        // stop string is withheld, so the match never straddles the already-emitted boundary.
        string pending = _pending.ToString();
        string trimmed = StopSuffixTrimmer.TrimMatchedSuffix(pending, _conditions);

        _pending.Clear();
        _pending.Append(trimmed);
        EmitAllBut(0);
    }

    /// <summary>
    /// Terminal flush for every non-stop-string ending (length limit, EOS, cache exhaustion):
    /// releases all withheld text, since nothing further can match.
    /// </summary>
    public void Flush()
    {
        if (_onText is null)
            return;
        EmitAllBut(0);
    }

    /// <summary>
    /// Length of the longest suffix of the pending buffer that is a proper prefix of some
    /// registered stop string, and therefore must not be emitted yet.
    /// </summary>
    private int HeldBackLength()
    {
        if (_maxStopLength <= 1)
            return 0;

        int pendingLen = _pending.Length;
        int maxHold = Math.Min(_maxStopLength - 1, pendingLen);

        for (int hold = maxHold; hold > 0; hold--)
        {
            int start = pendingLen - hold;
            for (int i = 0; i < _conditions.Count; i++)
            {
                if (_conditions[i] is not StopStringCondition ssc)
                    continue;

                string stop = ssc.StopString;
                if (stop.Length <= hold)
                    continue;

                bool match = true;
                for (int j = 0; j < hold; j++)
                {
                    if (_pending[start + j] != stop[j]) { match = false; break; }
                }
                if (match)
                    return hold;
            }
        }
        return 0;
    }

    /// <summary>Emits the pending text except its last <paramref name="hold"/> characters.</summary>
    private void EmitAllBut(int hold)
    {
        int emitLen = _pending.Length - hold;
        if (emitLen <= 0)
            return;

        // StringBuilder has no span accessor; the fragment is short-lived and only
        // materialized when a text callback is actually attached.
        string fragment = _pending.ToString(0, emitLen);
        _pending.Remove(0, emitLen);
        _onText!(fragment.AsSpan());
    }
}
