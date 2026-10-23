using System.Buffers;
using System.Text;

namespace DotLLM.Tokenizers.Hf;

/// <summary>
/// GPT-2 / HuggingFace <c>ByteLevel</c> decoder. The inverse of
/// <see cref="ByteLevelPreTokenizer"/>: each character in a token string
/// represents one byte via the GPT-2 <c>unicode_to_bytes</c> mapping, and
/// the concatenation of all chars across a token sequence is the original
/// UTF-8 byte stream.
/// </summary>
/// <remarks>
/// <para>
/// Unlike the SentencePiece / <c>Metaspace</c> decoder, <see cref="ByteLevelDecoder"/>
/// does <b>not</b> insert a leading space. Any leading space in the original
/// input is already represented as the byte-encoded form of <c>0x20</c>
/// inside the first token (the pre-tokenizer's <c>add_prefix_space</c>
/// option controls whether a leading space is forced before encoding, not
/// inserted during decoding).
/// </para>
/// <para>
/// This type is primarily a documentation / testing façade. In production,
/// the actual decode pipeline for ByteLevel tokenizers lives inside
/// <see cref="DotLLM.Tokenizers.Bpe.BpeTokenizer"/> via the
/// <c>CreateTiktoken</c> factory — <see cref="Decode(ReadOnlySpan{int}, string[])"/>
/// is a reference implementation for tests and external callers who hold a
/// vocabulary-indexed <c>idToToken</c> table but no full
/// <see cref="DotLLM.Tokenizers.Bpe.BpeTokenizer"/>.
/// </para>
/// </remarks>
public static class ByteLevelDecoder
{
    /// <summary>
    /// Decodes a sequence of token IDs back to the original text. Each
    /// character in each token's string form is mapped to a byte via the
    /// reverse GPT-2 table, the byte stream concatenated across all tokens,
    /// then UTF-8 decoded.
    /// </summary>
    /// <param name="tokenIds">Token IDs to decode.</param>
    /// <param name="idToToken">
    /// Vocabulary indexed by ID. IDs outside <c>[0, idToToken.Length)</c>
    /// and empty entries are skipped.
    /// </param>
    /// <returns>Decoded UTF-8 text.</returns>
    public static string Decode(ReadOnlySpan<int> tokenIds, string[] idToToken)
    {
        ArgumentNullException.ThrowIfNull(idToToken);
        if (tokenIds.IsEmpty) return string.Empty;

        // Heuristic: average 4 bytes per token is a safe upper bound for
        // most languages; grow on overflow.
        int capacity = Math.Max(16, tokenIds.Length * 4);
        byte[] buf = ArrayPool<byte>.Shared.Rent(capacity);
        int count = 0;

        try
        {
            foreach (int id in tokenIds)
            {
                if ((uint)id >= (uint)idToToken.Length) continue;
                string token = idToToken[id];
                if (string.IsNullOrEmpty(token)) continue;

                foreach (char c in token)
                {
                    int idx = (int)c;
                    short b = (uint)idx < (uint)ByteLevelPreTokenizer.UnicodeToByteTable.Length
                        ? ByteLevelPreTokenizer.UnicodeToByteTable[idx]
                        : (short)-1;
                    if (b < 0) continue;

                    if (count >= buf.Length)
                    {
                        byte[] larger = ArrayPool<byte>.Shared.Rent(buf.Length * 2);
                        buf.AsSpan(0, count).CopyTo(larger);
                        ArrayPool<byte>.Shared.Return(buf);
                        buf = larger;
                    }
                    buf[count++] = (byte)b;
                }
            }

            return Encoding.UTF8.GetString(buf, 0, count);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buf);
        }
    }

    /// <summary>
    /// Decodes a single token's byte-level-encoded chars to the UTF-8 text
    /// fragment it represents. Note: a single token may not be a valid
    /// UTF-8 sequence on its own (the byte stream is only guaranteed
    /// self-consistent after concatenating all tokens of a segment).
    /// Callers decoding token-by-token should expect replacement chars
    /// (U+FFFD) at multi-byte-character boundaries.
    /// </summary>
    public static string DecodeToken(int tokenId, string[] idToToken)
    {
        ArgumentNullException.ThrowIfNull(idToToken);
        if ((uint)tokenId >= (uint)idToToken.Length) return string.Empty;
        string token = idToToken[tokenId];
        if (string.IsNullOrEmpty(token)) return string.Empty;

        byte[]? rented = null;
        try
        {
            Span<byte> bytes = token.Length <= 256
                ? stackalloc byte[256]
                : (rented = ArrayPool<byte>.Shared.Rent(token.Length));
            bytes = bytes[..token.Length];
            int count = 0;
            for (int i = 0; i < token.Length; i++)
            {
                int idx = (int)token[i];
                short b = (uint)idx < (uint)ByteLevelPreTokenizer.UnicodeToByteTable.Length
                    ? ByteLevelPreTokenizer.UnicodeToByteTable[idx]
                    : (short)-1;
                if (b >= 0) bytes[count++] = (byte)b;
            }
            return Encoding.UTF8.GetString(bytes[..count]);
        }
        finally
        {
            if (rented is not null) ArrayPool<byte>.Shared.Return(rented);
        }
    }
}
