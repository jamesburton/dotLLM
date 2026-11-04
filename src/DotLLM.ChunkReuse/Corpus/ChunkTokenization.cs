using DotLLM.Tokenizers;

namespace DotLLM.ChunkReuse.CodeCorpus;

/// <summary>
/// The one way library code turns chunk text into token ids (chunk-cache contract ruling 21): exactly as the chat path
/// tokenises caller content. The chat template marks every special-token spelling in user / tool content with U+FDD0
/// (<see cref="ITokenizer.MarkUntrustedSpecials"/>, #1339) and the encoder strips the mark, so a spelling such as
/// <c>&lt;think&gt;</c> or <c>&lt;|im_start|&gt;</c> is encoded as plain text. Bare <see cref="ITokenizer.EncodeRaw"/> parses
/// such a spelling into its control id, so a chunk holding one would get ids (and a store key) the server never computes.
/// </summary>
/// <remarks>
/// <para>
/// This is the same <i>function</i> the chat path applies to user / tool message text, not a promise that a span's ids inside
/// a rendered prompt always equal its standalone ids. They differ for assistant-message spans containing
/// <c>&lt;think&gt;</c> / <c>&lt;/think&gt;</c> (the template exempts the reasoning tags there, so they stay control tokens),
/// and at span boundaries (a SentencePiece leading space, BPE merges with the template text before or after the span).
/// Span location and hit checks on the server (Phase I Tasks 11 and 13) must compare ids, not assume equality.
/// </para>
/// Text without special-token spellings encodes exactly as <see cref="ITokenizer.EncodeRaw"/> (existing entries stay valid).
/// No BOS is added. The chunk text is used as given: a caller's own U+FDD0 is dropped by the encoder, as on the chat path.
/// </remarks>
public static class ChunkTokenization
{
    /// <summary>Token ids of a chunk's text: <c>EncodeRaw(MarkUntrustedSpecials(chunkText))</c>.</summary>
    /// <param name="tokenizer">The model's tokenizer.</param>
    /// <param name="chunkText">The chunk text (as sent on the wire).</param>
    /// <returns>The ids.</returns>
    public static int[] Encode(ITokenizer tokenizer, string chunkText)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(chunkText);
        return tokenizer.EncodeRaw(tokenizer.MarkUntrustedSpecials(chunkText));
    }
}
