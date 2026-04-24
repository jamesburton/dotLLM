using System.Buffers;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.RegularExpressions;

namespace DotLLM.Tokenizers.Bpe;

/// <summary>
/// GPT-2 / tiktoken BPE encoding (Llama 3, GPT-4).
/// Each character in a token string represents one byte via the GPT-2 byte-to-Unicode mapping.
/// Merge priority is determined by rank (lower rank = applied first).
/// </summary>
/// <remarks>
/// The merge rank dictionary uses <c>(leftTokenId, rightTokenId)</c> tuple keys to avoid
/// the string allocation that a <c>"leftText rightText"</c> key lookup would incur on every
/// bigram check during the hot encode path.
/// </remarks>
internal sealed class Gpt2TiktokenEncoding : IBpeEncoding
{
    // -------------------------------------------------------------------------
    // GPT-2 byte-to-unicode tables (static — shared across all instances)
    // -------------------------------------------------------------------------

    /// <summary>
    /// Maps a raw byte value (0–255) to its GPT-2 Unicode character representation.
    /// GPT-2's byte_encoder maps printable ASCII (33–126) and Latin-1 (161–255, minus 173)
    /// to the same code-point; remaining bytes map to U+0100+n to avoid control characters.
    /// </summary>
    private static readonly char[] Gpt2ByteToUnicode = BuildGpt2ByteToUnicode();

    /// <summary>
    /// Reverse of <see cref="Gpt2ByteToUnicode"/>. Index = Unicode char (up to 0x0144).
    /// Value = byte value (0–255), or -1 if the char is not a GPT-2-encoded byte.
    /// </summary>
    private static readonly short[] Gpt2UnicodeToByteTable = BuildGpt2UnicodeToByteTable();

    private static char[] BuildGpt2ByteToUnicode()
    {
        char[] table = new char[256];
        // Printable ASCII 33..126 → same code point.
        for (int b = 33; b <= 126; b++) table[b] = (char)b;
        // Latin-1 supplement 161..172 → same code point.
        for (int b = 161; b <= 172; b++) table[b] = (char)b;
        // Latin-1 supplement 174..255 → same code point.
        for (int b = 174; b <= 255; b++) table[b] = (char)b;
        // Remaining bytes (0..32, 127..160, 173) → U+0100+n.
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (table[b] == 0) // not yet assigned
                table[b] = (char)(0x100 + n++);
        }
        return table;
    }

    private static short[] BuildGpt2UnicodeToByteTable()
    {
        char[] byteToChar = BuildGpt2ByteToUnicode();
        int maxChar = 0;
        foreach (char c in byteToChar) if (c > maxChar) maxChar = c;
        short[] table = new short[maxChar + 1];
        for (int i = 0; i < table.Length; i++) table[i] = -1;
        for (int b = 0; b < 256; b++) table[(int)byteToChar[b]] = (short)b;
        return table;
    }

    // -------------------------------------------------------------------------
    // Instance state
    // -------------------------------------------------------------------------

    private readonly string[] _idToToken;
    private readonly int[] _byteToTokenId;
    private readonly Trie _vocabTrie;

    /// <summary>
    /// Merge rank table keyed by (leftTokenId, rightTokenId) value-type tuple.
    /// Zero allocation per bigram lookup — eliminates the <c>string mergeKey</c> hot-path alloc.
    /// </summary>
    private readonly Dictionary<(int, int), int> _mergeRanks;

    private readonly int _unkId;

    /// <summary>
    /// Compiled pre-tokenization regex that splits input at word/punctuation boundaries
    /// before BPE merges. Null means no pre-tokenization (whole text = one segment).
    /// </summary>
    private readonly Regex? _preRegex;

    internal Gpt2TiktokenEncoding(string[] tokens, string[] merges, int[]? tokenTypes, Regex? preRegex = null)
    {
        _idToToken = tokens;
        _byteToTokenId = BpeCore.BuildByteToTokenId(tokens);

        _unkId = Array.FindIndex(tokens, t => t is "<unk>" or "<UNK>");
        if (_unkId < 0) _unkId = 0;

        _vocabTrie = new Trie();
        for (int i = 0; i < tokens.Length; i++)
        {
            if (!string.IsNullOrEmpty(tokens[i]))
                _vocabTrie.Add(tokens[i].AsSpan(), i, 0f);
        }

        // Build token string → ID reverse lookup (one pass at init time).
        var tokenToId = new Dictionary<string, int>(tokens.Length, StringComparer.Ordinal);
        for (int i = 0; i < tokens.Length; i++)
            tokenToId[tokens[i]] = i;

        // Parse "A B" merge entries → (idA, idB) tuple keys.
        var mergeRanks = new Dictionary<(int, int), int>(merges.Length);
        for (int rank = 0; rank < merges.Length; rank++)
        {
            int sep = merges[rank].IndexOf(' ');
            if (sep < 0) continue;
            string a = merges[rank][..sep], b = merges[rank][(sep + 1)..];
            if (tokenToId.TryGetValue(a, out int idA) && tokenToId.TryGetValue(b, out int idB))
                mergeRanks[(idA, idB)] = rank;
        }
        _mergeRanks = mergeRanks;
        _preRegex = preRegex;
    }

    public int[] Encode(string text)
    {
        // HF-compatible order: regex split the raw text first, then byte-encode
        // each match, then BPE-encode each byte-encoded chunk. The regex is
        // written against the raw Unicode categories (\p{L} is 'é', not the
        // byte-mapped 'Ã©'); applying it after byte-encoding would
        // misclassify multi-byte chars and split them into per-byte segments,
        // preventing the BPE merge table (which encodes byte pairs as a
        // single token) from firing on non-ASCII input.
        //
        // The byte-mapped segment buffer is rented once up-front — UTF-8
        // byte count of the whole text is an upper bound for any chunk — to
        // eliminate the per-regex-match string allocation that the previous
        // ByteMap-returns-string implementation incurred.
        if (_preRegex is null)
        {
            // No pre-tokenization: byte-map the whole string and feed it as one
            // segment. Only hit by ByteLevel(use_regex:false) with no
            // upstream Split — an uncommon configuration.
            int wholeMaxChars = Encoding.UTF8.GetMaxByteCount(text.Length);
            char[] wholeBuf = ArrayPool<char>.Shared.Rent(wholeMaxChars);
            try
            {
                ByteMapIntoSpan(text, wholeBuf, out int wholeWritten);
                return EncodeSegment(wholeBuf.AsSpan(0, wholeWritten));
            }
            finally
            {
                ArrayPool<char>.Shared.Return(wholeBuf);
            }
        }

        var result = new List<int>(Math.Max(16, text.Length));
        int maxChars = Encoding.UTF8.GetMaxByteCount(text.Length);
        char[] byteMapBuf = ArrayPool<char>.Shared.Rent(maxChars);
        try
        {
            foreach (var match in _preRegex.EnumerateMatches(text))
            {
                ReadOnlySpan<char> rawChunk = text.AsSpan(match.Index, match.Length);
                if (rawChunk.IsEmpty) continue;
                ByteMapIntoSpan(rawChunk, byteMapBuf, out int written);
                EncodeSegmentInto(byteMapBuf.AsSpan(0, written), result);
            }
        }
        finally
        {
            ArrayPool<char>.Shared.Return(byteMapBuf);
        }
        return result.ToArray();
    }

    /// <summary>
    /// UTF-8 encodes <paramref name="text"/> and applies the GPT-2
    /// bytes_to_unicode mapping into a caller-provided destination. Returns
    /// the number of chars written via <paramref name="written"/>. The
    /// destination must be large enough — <c>Encoding.UTF8.GetMaxByteCount(length)</c>
    /// of the input char length is a safe upper bound. Allocation-free on
    /// the hot path; replaces the former <c>string</c>-returning variant
    /// that allocated one string per regex match in <see cref="Encode"/>.
    /// </summary>
    private static void ByteMapIntoSpan(ReadOnlySpan<char> text, Span<char> dest, out int written)
    {
        if (text.IsEmpty)
        {
            written = 0;
            return;
        }
        int utf8Len = Encoding.UTF8.GetByteCount(text);
        byte[] rentedUtf8 = ArrayPool<byte>.Shared.Rent(utf8Len);
        try
        {
            int actual = Encoding.UTF8.GetBytes(text, rentedUtf8);
            for (int i = 0; i < actual; i++)
                dest[i] = Gpt2ByteToUnicode[rentedUtf8[i]];
            written = actual;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rentedUtf8);
        }
    }

    /// <summary>
    /// Encodes a single pre-tokenized segment using BPE merges.
    /// </summary>
    private int[] EncodeSegment(ReadOnlySpan<char> segment)
    {
        Symbol[] symbols = ArrayPool<Symbol>.Shared.Rent(segment.Length * 2);
        int symbolCount;
        try
        {
            symbolCount = BuildInitialSymbols(segment, symbols);

            var queue = new PriorityQueue<BgramEntry, (int, int)>(symbolCount);
            for (int i = 0; i < symbolCount - 1; i++)
                TryEnqueueBigram(symbols, i, i + 1, queue);

            RunMergeLoop(symbols, queue);
            return BpeCore.CollectTokenIds(symbols, symbolCount);
        }
        finally
        {
            ArrayPool<Symbol>.Shared.Return(symbols, clearArray: false);
        }
    }

    /// <summary>
    /// Encodes a segment and appends token IDs directly to <paramref name="dest"/>,
    /// avoiding intermediate <c>int[]</c> allocation per segment.
    /// </summary>
    private void EncodeSegmentInto(ReadOnlySpan<char> segment, List<int> dest)
    {
        Symbol[] symbols = ArrayPool<Symbol>.Shared.Rent(segment.Length * 2);
        try
        {
            int symbolCount = BuildInitialSymbols(segment, symbols);

            var queue = new PriorityQueue<BgramEntry, (int, int)>(symbolCount);
            for (int i = 0; i < symbolCount - 1; i++)
                TryEnqueueBigram(symbols, i, i + 1, queue);

            RunMergeLoop(symbols, queue);
            BpeCore.CollectTokenIds(symbols, symbolCount, dest);
        }
        finally
        {
            ArrayPool<Symbol>.Shared.Return(symbols, clearArray: false);
        }
    }

    public string Decode(ReadOnlySpan<int> tokenIds)
    {
        // GPT-2 decode: every char in a token string is a GPT-2-encoded byte.
        // Map each char back to its byte, then UTF-8 decode the combined byte stream.
        int maxBytes = tokenIds.Length * 8;
        byte[] buf = ArrayPool<byte>.Shared.Rent(maxBytes);
        int count = 0;

        foreach (int id in tokenIds)
        {
            if ((uint)id >= (uint)_idToToken.Length) continue;
            string token = _idToToken[id];
            foreach (char c in token)
            {
                if (count >= buf.Length)
                {
                    byte[] larger = ArrayPool<byte>.Shared.Rent(buf.Length * 2);
                    buf.AsSpan(0, count).CopyTo(larger);
                    ArrayPool<byte>.Shared.Return(buf);
                    buf = larger;
                }
                // Look up the byte value for this GPT-2 Unicode char.
                int idx = (int)c;
                if ((uint)idx < (uint)Gpt2UnicodeToByteTable.Length)
                {
                    short b = Gpt2UnicodeToByteTable[idx];
                    if (b >= 0) buf[count++] = (byte)b;
                }
            }
        }

        string result = Encoding.UTF8.GetString(buf, 0, count);
        ArrayPool<byte>.Shared.Return(buf);
        return result;
    }

    public string DecodeToken(int tokenId)
    {
        if ((uint)tokenId >= (uint)_idToToken.Length) return string.Empty;
        string token = _idToToken[tokenId];
        // GPT-2: each token char encodes one byte.
        // stackalloc for typical tokens (≤256 chars), ArrayPool fallback for safety.
        byte[]? rented = null;
        try
        {
            Span<byte> bytes = token.Length <= 256
                ? stackalloc byte[256]
                : (rented = ArrayPool<byte>.Shared.Rent(token.Length));
            bytes = bytes[..token.Length];
            for (int i = 0; i < token.Length; i++)
            {
                int idx = (int)token[i];
                short bval = (uint)idx < (uint)Gpt2UnicodeToByteTable.Length
                    ? Gpt2UnicodeToByteTable[idx] : (short)-1;
                bytes[i] = bval >= 0 ? (byte)bval : (byte)0;
            }
            return Encoding.UTF8.GetString(bytes);
        }
        finally
        {
            if (rented is not null) ArrayPool<byte>.Shared.Return(rented);
        }
    }

    private int BuildInitialSymbols(ReadOnlySpan<char> text, Symbol[] symbols)
    {
        int count = 0;
        int i = 0;
        Span<byte> utf8 = stackalloc byte[4]; // pre-allocate outside loop (CA2014)
        while (i < text.Length)
        {
            int charLen = char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1])
                ? 2 : 1;
            ReadOnlySpan<char> cpSpan = text.Slice(i, charLen);
            i += charLen;

            if (_vocabTrie.TryMatchLongest(cpSpan, out int tokenId, out _, out int ml) && ml == charLen)
            {
                symbols[count] = new Symbol { Prev = count - 1, Next = count + 1, TokenId = tokenId };
                count++;
            }
            else
            {
                // Byte fallback: emit one symbol per UTF-8 byte.
                // If the byte has no <0xNN> token, emit <unk> rather than silently dropping it.
                int byteLen = Encoding.UTF8.GetBytes(cpSpan, utf8);
                for (int b = 0; b < byteLen; b++)
                {
                    int byteId = _byteToTokenId[utf8[b]];
                    int effectiveId = byteId >= 0 ? byteId : _unkId;
                    symbols[count] = new Symbol { Prev = count - 1, Next = count + 1, TokenId = effectiveId };
                    count++;
                }
            }
        }
        if (count > 0) symbols[count - 1].Next = -1;
        return count;
    }

    private void TryEnqueueBigram(
        Symbol[] symbols, int leftIdx, int rightIdx,
        PriorityQueue<BgramEntry, (int, int)> queue)
    {
        if (leftIdx < 0 || rightIdx < 0) return;

        // Zero allocation: tuple key is a value type — no string concat needed.
        if (!_mergeRanks.TryGetValue((symbols[leftIdx].TokenId, symbols[rightIdx].TokenId), out int rank)) return;

        // Resolve merged token ID via trie (stack-allocated concat — no heap alloc).
        // ArrayPool fallback for the rare case where combined token length exceeds 256.
        string leftText = _idToToken[symbols[leftIdx].TokenId];
        string rightText = _idToToken[symbols[rightIdx].TokenId];
        int totalLen = leftText.Length + rightText.Length;
        char[]? rented = null;
        try
        {
            Span<char> buf = totalLen <= 256
                ? stackalloc char[256]
                : (rented = ArrayPool<char>.Shared.Rent(totalLen));
            Span<char> concat = buf[..totalLen];
            leftText.AsSpan().CopyTo(concat);
            rightText.AsSpan().CopyTo(concat[leftText.Length..]);

            if (_vocabTrie.TryMatchLongest(concat, out int mergedId, out _, out int ml) && ml == totalLen)
            {
                int leftToken = symbols[leftIdx].TokenId;
                int rightToken = symbols[rightIdx].TokenId;
                queue.Enqueue(new BgramEntry(leftIdx, rightIdx, mergedId, leftToken, rightToken),
                    (rank, leftIdx));
            }
        }
        finally
        {
            if (rented is not null) ArrayPool<char>.Shared.Return(rented);
        }
    }

    private void RunMergeLoop(Symbol[] symbols, PriorityQueue<BgramEntry, (int, int)> queue)
    {
        while (queue.Count > 0)
        {
            BgramEntry entry = queue.Dequeue();
            ref Symbol left = ref symbols[entry.Left];
            ref Symbol right = ref symbols[entry.Right];

            if (left.Deleted || right.Deleted
                || left.Next != entry.Right
                || left.TokenId != entry.ExpectedLeft
                || right.TokenId != entry.ExpectedRight)
                continue;

            left.TokenId = entry.MergedId;
            right.Deleted = true;
            int nextIdx = right.Next;
            left.Next = nextIdx;
            if (nextIdx >= 0) symbols[nextIdx].Prev = entry.Left;

            TryEnqueueBigram(symbols, left.Prev, entry.Left, queue);
            TryEnqueueBigram(symbols, entry.Left, nextIdx, queue);
        }
    }
}
