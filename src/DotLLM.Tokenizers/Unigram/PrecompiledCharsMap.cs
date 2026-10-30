namespace DotLLM.Tokenizers.Unigram;

/// <summary>
/// SentencePiece <c>precompiled_charsmap</c> normaliser (the NFKC-plus-NMT rule table that XLM-R, T5 and
/// every other SentencePiece model embeds in its <c>.model</c> file).
/// </summary>
/// <remarks>
/// <para>Blob layout (GGUF <c>tokenizer.ggml.precompiled_charsmap</c>, byte-identical to SentencePiece's
/// <c>NormalizerSpec.precompiled_charsmap</c>): <c>u32 LE trieBytes</c>, then <c>trieBytes</c> of
/// 32-bit double-array units (darts-clone, "XOR-compressed compact double array"), then the replacement pool
/// of NUL-terminated UTF-8 strings. A trie value is a byte offset into that pool.</para>
/// <para>This is a port of llama.cpp's <c>llm_tokenizer_ugm_session::normalize_prefix</c> (the authority for
/// GGUF models), which matches SentencePiece's <c>Normalizer::NormalizePrefix</c>: the <b>longest</b> trie
/// match at the current position wins; with no match the single UTF-8 character passes through unchanged.
/// It deliberately does not use <see cref="System.String.Normalize(System.Text.NormalizationForm)"/>:
/// the table is frozen at SentencePiece training time, so .NET's Unicode data would drift from it.</para>
/// </remarks>
internal sealed class PrecompiledCharsMap
{
    private readonly uint[] _units;
    private readonly byte[] _pool;

    private PrecompiledCharsMap(uint[] units, byte[] pool)
    {
        _units = units;
        _pool = pool;
    }

    /// <summary>Parses a charsmap blob; returns <c>null</c> for an empty blob (no normalisation rules).</summary>
    public static PrecompiledCharsMap? Parse(ReadOnlySpan<byte> blob)
    {
        if (blob.IsEmpty)
            return null;
        if (blob.Length < 4)
            throw new InvalidDataException("precompiled_charsmap is shorter than its 4-byte header.");

        uint trieBytes = BitConverter.ToUInt32(blob[..4]);
        if (trieBytes % 4 != 0 || 4L + trieBytes > blob.Length)
            throw new InvalidDataException($"precompiled_charsmap declares a {trieBytes}-byte trie in a {blob.Length}-byte blob.");

        var units = new uint[trieBytes / 4];
        for (int i = 0; i < units.Length; i++)
            units[i] = BitConverter.ToUInt32(blob.Slice(4 + i * 4, 4));
        byte[] pool = blob[(4 + (int)trieBytes)..].ToArray();
        return new PrecompiledCharsMap(units, pool);
    }

    /// <summary>
    /// Finds the longest rule matching a prefix of <paramref name="input"/>.
    /// </summary>
    /// <param name="input">UTF-8 bytes, positioned at the start of the prefix to normalise.</param>
    /// <param name="consumed">Bytes of <paramref name="input"/> the rule consumes (0 when no rule matches).</param>
    /// <returns>The replacement bytes (may be empty: the rule deletes the input), or empty when nothing matched.</returns>
    public ReadOnlySpan<byte> MatchLongest(ReadOnlySpan<byte> input, out int consumed)
    {
        consumed = 0;
        int valueOffset = 0;

        uint nodeIndex = Base(Unit(0));
        for (int i = 0; i < input.Length; i++)
        {
            byte c = input[i];
            if (c == 0)
                break;
            nodeIndex ^= c;
            uint unit = Unit(nodeIndex);
            if ((unit & ((1u << 31) | 0xFFu)) != c)
                break;
            nodeIndex ^= Base(unit);
            if (((unit >> 8) & 1) != 0)
            {
                // The child's BASE points at a leaf unit that stores the pool offset.
                consumed = i + 1;
                valueOffset = (int)(Unit(nodeIndex) & ((1u << 31) - 1));
            }
        }

        if (consumed == 0)
            return default;

        if ((uint)valueOffset >= (uint)_pool.Length)
            throw new InvalidDataException("precompiled_charsmap replacement offset is outside the pool.");
        int end = Array.IndexOf(_pool, (byte)0, valueOffset);
        if (end < 0)
            throw new InvalidDataException("precompiled_charsmap contains an unterminated replacement string.");
        return _pool.AsSpan(valueOffset, end - valueOffset);
    }

    private uint Unit(uint index)
    {
        if (index >= (uint)_units.Length)
            throw new InvalidDataException("precompiled_charsmap trie index out of bounds.");
        return _units[index];
    }

    // darts-clone unit: bits 10..30 = offset (shifted left by 8 when the extension bit 9 is set).
    private static uint Base(uint unit) => (unit >> 10) << (int)((unit & (1u << 9)) >> 6);
}
