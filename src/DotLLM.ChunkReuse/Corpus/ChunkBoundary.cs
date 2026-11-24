using DotLLM.Tokenizers;

namespace DotLLM.ChunkReuse.CodeCorpus;

/// <summary>
/// Token stability of a chunk edge (Phase I Task 19, e2e findings F-A1 / F-A2). A harness that shows two chunks back to back
/// sends their concatenation; the server then tokenises the whole prompt, and a chunk counts as reusable only where its own
/// ids occur at its position (<c>chunk_template_altered</c> otherwise). A BPE merge across the edge (a trailing newline and a
/// leading newline become one <c>"\n\n"</c> token) breaks that. An edge is <b>stable</b> when
/// <c>Encode(left + right) == Encode(left) ++ Encode(right)</c>, with <see cref="ChunkTokenization.Encode"/>.
/// </summary>
public static class ChunkBoundary
{
    /// <summary>Characters of context taken on each side of an edge by <see cref="IsStableAt"/>.</summary>
    public const int Window = 128;

    /// <summary>Exact check of two whole texts: tokenisation of the concatenation equals the concatenation of tokenisations.</summary>
    /// <param name="tokenizer">The model's tokenizer.</param>
    /// <param name="left">Text before the edge.</param>
    /// <param name="right">Text after the edge.</param>
    /// <param name="joiner">Text a harness inserts between them (<c>""</c> for back to back, <c>"\n"</c> for the Anthropic inline
    /// join); its own ids are expected between the two sides.</param>
    /// <returns>Whether the edge is token-stable.</returns>
    public static bool IsStable(ITokenizer tokenizer, string left, string right, string joiner = "")
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        ArgumentNullException.ThrowIfNull(joiner);
        int[] a = ChunkTokenization.Encode(tokenizer, left);
        int[] j = joiner.Length == 0 ? [] : ChunkTokenization.Encode(tokenizer, joiner);
        int[] b = ChunkTokenization.Encode(tokenizer, right);
        int[] whole = ChunkTokenization.Encode(tokenizer, left + joiner + right);
        return whole.AsSpan().SequenceEqual([.. a, .. j, .. b]);
    }

    /// <summary>
    /// Windowed check of the edge at <paramref name="pos"/> inside <paramref name="text"/>: the same test on the
    /// <see cref="Window"/> characters on each side (windows never split a surrogate pair). Tokenisers merge only inside a short
    /// pre-token, so the window decides the edge without encoding whole chunks.
    /// </summary>
    /// <param name="tokenizer">The model's tokenizer.</param>
    /// <param name="text">The file text.</param>
    /// <param name="pos">The edge: characters before it end one chunk, characters from it start the next (0 and the end are stable).</param>
    /// <returns>Whether the edge is token-stable.</returns>
    public static bool IsStableAt(ITokenizer tokenizer, string text, int pos)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (pos <= 0 || pos >= text.Length) return true;
        int from = Math.Max(0, pos - Window);
        if (from > 0 && char.IsLowSurrogate(text[from])) from++;
        int to = Math.Min(text.Length, pos + Window);
        if (to < text.Length && char.IsLowSurrogate(text[to])) to--;
        return IsStable(tokenizer, text[from..pos], text[pos..to]);
    }
}
