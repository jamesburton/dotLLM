using System.Buffers;
using System.Text;

namespace DotLLM.Tokenizers.Bpe;

/// <summary>
/// Gemma-4 SPM-style merge-ranked BPE encoding (GGUF <c>tokenizer.ggml.model = "gemma4"</c>).
/// Matches llama.cpp's <c>LLAMA_VOCAB_PRE_TYPE_GEMMA4</c> semantics:
/// spaces are escaped to <c>▁</c> (U+2581) by the normalizer, then merge-ranked BPE runs
/// over the raw UTF-8 text (no GPT-2 byte-to-unicode encoding) with only a newline split
/// as pre-tokenization (<c>[^\n]+|[\n]+</c>). Whole newline runs that exist in the
/// vocabulary bypass BPE (llama.cpp PR #21343). Code points absent from the vocabulary
/// fall back to <c>&lt;0xNN&gt;</c> byte tokens.
/// </summary>
/// <remarks>
/// Merge priority is rank-based (lower rank = applied first, ties broken by leftmost
/// position), keyed by <c>(leftTokenId, rightTokenId)</c>. This is faithful to llama.cpp's
/// string-pair rank table because every merge part and merge result in the gemma-4 merge
/// table is itself a vocabulary token (verified against the released GGUFs).
/// </remarks>
internal sealed class Gemma4SpmBpeEncoding : IBpeEncoding
{
    private const char SpaceMarker = '▁'; // ▁ (SentencePiece word-boundary marker)

    private readonly string[] _idToToken;
    private readonly int[] _byteToTokenId;
    private readonly Trie _vocabTrie;

    /// <summary>Merge rank table keyed by (leftTokenId, rightTokenId); lower rank wins.</summary>
    private readonly Dictionary<(int, int), int> _mergeRanks;

    private readonly int _unkId;

    internal Gemma4SpmBpeEncoding(string[] tokens, string[] merges, int[]? tokenTypes)
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

        var tokenToId = new Dictionary<string, int>(tokens.Length, StringComparer.Ordinal);
        for (int i = 0; i < tokens.Length; i++)
            tokenToId[tokens[i]] = i;

        // Parse "A B" merge entries → (idA, idB) tuple keys. llama.cpp splits gemma4
        // merges at the first space AT OR AFTER index 1 (a merge part is never empty).
        var mergeRanks = new Dictionary<(int, int), int>(merges.Length);
        for (int rank = 0; rank < merges.Length; rank++)
        {
            string entry = merges[rank];
            if (entry.Length < 3) continue;
            int sep = entry.IndexOf(' ', 1);
            if (sep < 0) continue;
            string a = entry[..sep], b = entry[(sep + 1)..];
            if (tokenToId.TryGetValue(a, out int idA) && tokenToId.TryGetValue(b, out int idB))
                mergeRanks.TryAdd((idA, idB), rank);
        }
        _mergeRanks = mergeRanks;
    }

    public int[] Encode(string text) => EncodeCore(text);

    /// <summary>
    /// Continuation segments encode identically — gemma-4 has no BOS-space prepend
    /// (<c>tokenizer.ggml.add_space_prefix = false</c>); whitespace escaping is applied
    /// per fragment, exactly as llama.cpp escapes each raw-text fragment between
    /// special tokens.
    /// </summary>
    public int[] EncodeSegment(string text) => EncodeCore(text);

    private int[] EncodeCore(string text)
    {
        if (text.Length == 0)
            return [];

        // 1. Normalize: replace ' ' with ▁ throughout (llama_escape_whitespace).
        //    No ▁ prepend — gemma-4 sets add_space_prefix = false.
        char[] rentedNorm = ArrayPool<char>.Shared.Rent(text.Length);
        try
        {
            text.AsSpan().CopyTo(rentedNorm);
            MemoryExtensions.Replace(rentedNorm.AsSpan(0, text.Length), ' ', SpaceMarker);
            ReadOnlySpan<char> normalized = rentedNorm.AsSpan(0, text.Length);

            // 2. Pre-tokenize: [^\n]+|[\n]+ — split into maximal runs of
            //    non-newline / newline characters, BPE each run independently.
            var result = new List<int>(Math.Max(16, text.Length / 3));
            int pos = 0;
            while (pos < normalized.Length)
            {
                bool isNewlineRun = normalized[pos] == '\n';
                int end = pos + 1;
                while (end < normalized.Length && (normalized[end] == '\n') == isNewlineRun)
                    end++;
                EncodeRun(normalized[pos..end], isNewlineRun, result);
                pos = end;
            }
            return result.ToArray();
        }
        finally
        {
            ArrayPool<char>.Shared.Return(rentedNorm);
        }
    }

    /// <summary>Encodes a single pre-tokenized run using merge-ranked BPE.</summary>
    private void EncodeRun(ReadOnlySpan<char> run, bool isNewlineRun, List<int> dest)
    {
        // Newline-run fast path (llama.cpp PR #21343): if the whole run is a vocab
        // token, emit it directly without BPE. Longer runs than the vocab covers
        // fall through to the normal per-character merge loop.
        if (isNewlineRun
            && _vocabTrie.TryMatchLongest(run, out int runId, out _, out int runLen)
            && runLen == run.Length)
        {
            dest.Add(runId);
            return;
        }

        Symbol[] symbols = ArrayPool<Symbol>.Shared.Rent(run.Length * 2);
        try
        {
            int symbolCount = BuildInitialSymbols(run, symbols);

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

    private int BuildInitialSymbols(ReadOnlySpan<char> text, Symbol[] symbols)
    {
        int count = 0;
        int i = 0;
        Span<byte> utf8 = stackalloc byte[4]; // pre-allocate outside loop (CA2014)
        while (i < text.Length)
        {
            // Consume one Unicode code point (1 or 2 chars for a surrogate pair).
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
                // Byte fallback: emit one <0xNN> symbol per UTF-8 byte. Safe to expand
                // here (rather than at collection time like llama.cpp) because the
                // gemma-4 merge table contains no <0xNN> merges — byte symbols never
                // participate in merges either way. Bytes without a <0xNN> token emit
                // <unk> rather than being silently dropped.
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

        if (!_mergeRanks.TryGetValue((symbols[leftIdx].TokenId, symbols[rightIdx].TokenId), out int rank)) return;

        // Resolve merged token ID. Every gemma-4 merge result is a vocab token;
        // guard anyway so a malformed merge table degrades to "merge not applied".
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
                // Lower rank = higher priority; leftIdx breaks ties (leftmost first),
                // matching llama.cpp's llm_bigram_bpe comparator.
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

            // Discard stale entries: symbol deleted, no longer adjacent, or token
            // changed since enqueue (merged into something else).
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

    // -------------------------------------------------------------------------
    // Decode — SPM-style: ▁ → space, <0xNN> byte tokens accumulated as UTF-8.
    // -------------------------------------------------------------------------

    public string Decode(ReadOnlySpan<int> tokenIds) => Decode(tokenIds, stripBosSpace: false);

    public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace)
    {
        // stripBosSpace is a no-op: gemma-4 never prepends ▁ at encode time.
        var sb = new StringBuilder(tokenIds.Length * 4);
        byte[]? byteBuffer = null;
        int byteCount = 0;

        foreach (int id in tokenIds)
        {
            if ((uint)id >= (uint)_idToToken.Length) continue;
            string token = _idToToken[id];
            if (BpeCore.IsByteToken(token, out byte b))
            {
                byteBuffer ??= ArrayPool<byte>.Shared.Rent(16);
                if (byteCount >= byteBuffer.Length)
                {
                    byte[] larger = ArrayPool<byte>.Shared.Rent(byteBuffer.Length * 2);
                    byteBuffer.AsSpan(0, byteCount).CopyTo(larger);
                    ArrayPool<byte>.Shared.Return(byteBuffer);
                    byteBuffer = larger;
                }
                byteBuffer[byteCount++] = b;
            }
            else
            {
                BpeCore.FlushByteBuffer(sb, byteBuffer, ref byteCount);
                int startLen = sb.Length;
                sb.Append(token);
                sb.Replace(SpaceMarker, ' ', startLen, token.Length);
            }
        }
        BpeCore.FlushByteBuffer(sb, byteBuffer, ref byteCount);

        if (byteBuffer != null)
            ArrayPool<byte>.Shared.Return(byteBuffer);

        return sb.ToString();
    }

    public string DecodeToken(int tokenId)
    {
        if ((uint)tokenId >= (uint)_idToToken.Length) return string.Empty;
        string token = _idToToken[tokenId];
        if (BpeCore.IsByteToken(token, out byte b))
        {
            Span<byte> single = stackalloc byte[] { b };
            return Encoding.Latin1.GetString(single);
        }
        return token.Contains(SpaceMarker) ? token.Replace(SpaceMarker, ' ') : token;
    }
}
