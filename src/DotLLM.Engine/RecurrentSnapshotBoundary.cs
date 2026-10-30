using DotLLM.Tokenizers;

namespace DotLLM.Engine;

/// <summary>
/// Where a recurrent model's prefix snapshot goes for a chat request: just before the chat template's generation prompt
/// (issue #1093). The next turn re-renders this turn's generation prompt as the start of the assistant reply, and that
/// rendering can differ (Bonsai's template, with the client dropping reasoning, re-tokenizes an empty think block), so a
/// snapshot that includes the generation prompt never matches again. Shared by <see cref="TextGenerator"/>'s single-slot
/// and tiered paths and by the continuous-batch scheduler.
/// </summary>
public static class RecurrentSnapshotBoundary
{
    /// <summary>
    /// The prompt length minus the generation prompt's tokens, when the generation prompt encodes to exactly the prompt's tail.
    /// </summary>
    /// <param name="promptIds">The request's prompt tokens.</param>
    /// <param name="tokenizer">The tokenizer that produced <paramref name="promptIds"/>.</param>
    /// <param name="suffix">The generation prompt's text (<see cref="Core.Configuration.InferenceOptions.GenerationPromptSuffix"/>).</param>
    /// <returns>
    /// The boundary, or <see langword="null"/> when <paramref name="suffix"/> is null or empty, its tokens are not the prompt's
    /// tail (it merged with the preceding text under BPE), or nothing would precede it.
    /// </returns>
    public static int? FromGenerationSuffix(ReadOnlySpan<int> promptIds, ITokenizer tokenizer, string? suffix)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        if (string.IsNullOrEmpty(suffix)) return null;

        // EncodeRaw: the suffix is mid-stream text, so a tokenizer that prepends BOS on Encode must not do it here.
        int[] suffixIds = tokenizer.EncodeRaw(suffix);
        int boundary = promptIds.Length - suffixIds.Length;
        if (suffixIds.Length == 0 || boundary < 1 || !promptIds[boundary..].SequenceEqual(suffixIds))
            return null;
        return boundary;
    }

    /// <summary>
    /// Combines the boundary a resolve path computed without the hint with the generation-prompt boundary: the later one
    /// wins (an exact extension of the previous prompt snapshots one token short of the end, which is longer), capped so at
    /// least one prompt token is left to run and produce the logits.
    /// </summary>
    /// <param name="legacyBoundary">The boundary computed without the hint.</param>
    /// <param name="suffixBoundary">The generation-prompt boundary, or <see langword="null"/> when there is none.</param>
    /// <param name="promptLen">The prompt length.</param>
    /// <returns><c>max(legacyBoundary, suffixBoundary ?? -1)</c>, at most <c>promptLen - 1</c>.</returns>
    public static int Choose(int legacyBoundary, int? suffixBoundary, int promptLen)
        => Math.Min(Math.Max(legacyBoundary, suffixBoundary ?? -1), promptLen - 1);

    /// <summary>
    /// The generation-prompt boundary, kept only when this prompt continues the previous one: it shares everything the
    /// previous prompt had before its own generation prompt. That is a chat turn, whose next turn will continue this one.
    /// A prompt that shares less (a classifier: the same system prompt, a different question) keeps its shared-prefix
    /// boundary, because its generation-prompt boundary would never be hit.
    /// </summary>
    /// <param name="suffixBoundary">This prompt's generation-prompt boundary, or <see langword="null"/>.</param>
    /// <param name="shared">Tokens this prompt shares with the previous one.</param>
    /// <param name="previousSuffixBoundary">The previous prompt's generation-prompt boundary, or <see langword="null"/>.</param>
    /// <returns><paramref name="suffixBoundary"/> when the prompt continues the previous one, else <see langword="null"/>.</returns>
    public static int? ForContinuation(int? suffixBoundary, int shared, int? previousSuffixBoundary)
        => previousSuffixBoundary is int previous && shared >= previous ? suffixBoundary : null;
}
