using System.Text;

namespace DotLLM.Tokenizers.Unigram;

/// <summary>
/// SentencePiece Unigram tokenizer (GGUF <c>tokenizer.ggml.model == "t5"</c>, llama.cpp's "UGM"): XLM-R
/// (multilingual-e5, bge-m3), T5, mT5, ALBERT-style vocabularies.
/// </summary>
/// <remarks>
/// <para>Pipeline, a port of llama.cpp's <c>llm_tokenizer_ugm</c> which itself follows SentencePiece's
/// optimized Viterbi: split literal control tokens (<c>&lt;s&gt;</c>, <c>&lt;/s&gt;</c>, …) out of the input →
/// per segment: <see cref="PrecompiledCharsMap"/> normalisation (longest rule wins), whitespace collapse /
/// trim (<c>remove_extra_whitespaces</c>), every space run becomes one <c>▁</c> (U+2581) prefixed to the next
/// word (<c>add_space_prefix</c>) → Viterbi over the UTF-8 bytes maximising the sum of piece scores →
/// consecutive unknown characters fuse into a single <c>&lt;unk&gt;</c>.</para>
/// <para>Unknown character cost = (smallest score among NORMAL pieces) − 10, exactly as SentencePiece.
/// Only NORMAL and USER_DEFINED pieces are matchable; CONTROL, UNKNOWN, UNUSED and BYTE are not.
/// <see cref="Encode(string)"/> wraps the ids in BOS/EOS per the GGUF <c>add_bos_token</c>/<c>add_eos_token</c>
/// flags (<c>&lt;s&gt; … &lt;/s&gt;</c> for XLM-R, which is what the BERT-class encoder expects as CLS/SEP).</para>
/// </remarks>
public sealed class UnigramTokenizer : ITokenizer
{
    private const string EscapedSpace = "▁";
    private static readonly byte[] EscapedSpaceUtf8 = [0xE2, 0x96, 0x81];
    private const double UnknownPenalty = 10.0;

    // GGUF token types (llama.cpp llama_token_type).
    private const int TypeNormal = 1, TypeUnknown = 2, TypeControl = 3, TypeUserDefined = 4;

    private readonly string[] _tokens;
    private readonly double[] _scores;
    private readonly bool _addSpacePrefix;
    private readonly bool _removeExtraWhitespaces;
    private readonly PrecompiledCharsMap? _charsMap;
    private readonly double _unknownScore;

    // Byte trie over matchable pieces; children of a node are contiguous, sorted by byte.
    private readonly int[] _childStart;
    private readonly ushort[] _childCount;
    private readonly byte[] _edgeByte;   // indexed by child node id
    private readonly int[] _nodeToken;   // token id or -1

    private readonly byte[][] _userDefined;
    private readonly (string Text, int Id)[] _literalSpecials;

    /// <summary>Id of <c>&lt;unk&gt;</c>.</summary>
    public int UnkTokenId { get; }

    /// <summary>Id of <c>&lt;pad&gt;</c>, or -1.</summary>
    public int PadTokenId { get; }

    /// <summary>True when <see cref="Encode(string)"/> prepends BOS.</summary>
    public bool AddBos { get; }

    /// <summary>True when <see cref="Encode(string)"/> appends EOS.</summary>
    public bool AddEos { get; }

    /// <inheritdoc/>
    public int VocabSize => _tokens.Length;

    /// <inheritdoc/>
    public int BosTokenId { get; }

    /// <inheritdoc/>
    public int EosTokenId { get; }

    /// <summary>Creates a Unigram tokenizer.</summary>
    /// <param name="tokens">Piece texts by id (the word-boundary marker is the literal <c>▁</c>).</param>
    /// <param name="scores">Log-probability per piece.</param>
    /// <param name="tokenTypes">GGUF token type per piece (1 normal, 2 unknown, 3 control, 4 user-defined, 5 unused, 6 byte); <c>null</c> = all normal.</param>
    /// <param name="unkId">Id of the unknown piece.</param>
    /// <param name="bosId">Id of BOS (<c>&lt;s&gt;</c>).</param>
    /// <param name="eosId">Id of EOS (<c>&lt;/s&gt;</c>).</param>
    /// <param name="padId">Id of PAD or -1.</param>
    /// <param name="addBos">Prepend BOS in <see cref="Encode(string)"/>.</param>
    /// <param name="addEos">Append EOS in <see cref="Encode(string)"/>.</param>
    /// <param name="addSpacePrefix">SentencePiece <c>add_dummy_prefix</c>.</param>
    /// <param name="removeExtraWhitespaces">SentencePiece <c>remove_extra_whitespaces</c>.</param>
    /// <param name="precompiledCharsMap">The <c>precompiled_charsmap</c> blob, empty for none.</param>
    public UnigramTokenizer(
        string[] tokens, float[] scores, int[]? tokenTypes, int unkId, int bosId, int eosId, int padId = -1,
        bool addBos = true, bool addEos = true, bool addSpacePrefix = true, bool removeExtraWhitespaces = true,
        ReadOnlySpan<byte> precompiledCharsMap = default)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        ArgumentNullException.ThrowIfNull(scores);
        if (scores.Length != tokens.Length)
            throw new ArgumentException("scores and tokens must have the same length.", nameof(scores));
        if (tokenTypes is not null && tokenTypes.Length != tokens.Length)
            throw new ArgumentException("tokenTypes and tokens must have the same length.", nameof(tokenTypes));
        if ((uint)unkId >= (uint)tokens.Length)
            throw new ArgumentOutOfRangeException(nameof(unkId), "unk id is outside the vocabulary.");

        _tokens = tokens;
        _scores = new double[tokens.Length];
        for (int i = 0; i < scores.Length; i++) _scores[i] = scores[i];
        UnkTokenId = unkId;
        BosTokenId = bosId;
        EosTokenId = eosId;
        PadTokenId = padId;
        AddBos = addBos;
        AddEos = addEos;
        _addSpacePrefix = addSpacePrefix;
        _removeExtraWhitespaces = removeExtraWhitespaces;
        _charsMap = PrecompiledCharsMap.Parse(precompiledCharsMap);

        int TypeOf(int id) => tokenTypes is null ? TypeNormal : tokenTypes[id];

        var matchable = new List<(byte[] Bytes, int Id)>(tokens.Length);
        var userDefined = new List<byte[]>();
        var specials = new List<(string, int)>();
        double minScore = double.MaxValue;
        for (int id = 0; id < tokens.Length; id++)
        {
            int type = TypeOf(id);
            if (type == TypeNormal)
                minScore = Math.Min(minScore, _scores[id]);
            if (type is TypeNormal or TypeUserDefined && tokens[id].Length > 0)
                matchable.Add((Encoding.UTF8.GetBytes(tokens[id]), id));
            if (type == TypeUserDefined && tokens[id].Length > 0)
                userDefined.Add(Encoding.UTF8.GetBytes(tokens[id]));
            if (type is TypeControl or TypeUnknown && tokens[id].Length > 0)
                specials.Add((tokens[id], id));
        }
        _unknownScore = (minScore == double.MaxValue ? 0.0 : minScore) - UnknownPenalty;
        _userDefined = userDefined.ToArray();
        _literalSpecials = specials.ToArray();

        (_childStart, _childCount, _edgeByte, _nodeToken) = BuildTrie(matchable);
    }

    /// <summary>Encodes <paramref name="text"/>, adding BOS/EOS per <see cref="AddBos"/>/<see cref="AddEos"/>.</summary>
    public int[] Encode(string text) => Encode(text, addSpecialTokens: true);

    /// <summary>Encodes without BOS/EOS framing.</summary>
    public int[] EncodeRaw(string text) => Encode(text, addSpecialTokens: false);

    /// <summary>Encodes <paramref name="text"/>, optionally framed with BOS/EOS.</summary>
    public int[] Encode(string text, bool addSpecialTokens)
    {
        ArgumentNullException.ThrowIfNull(text);
        var ids = new List<int>(text.Length / 2 + 4);
        if (addSpecialTokens && AddBos) ids.Add(BosTokenId);

        int pos = 0;
        while (pos < text.Length)
        {
            int next = -1, hitId = -1, hitLen = 0;
            foreach (var (s, id) in _literalSpecials)
            {
                int idx = text.IndexOf(s, pos, StringComparison.Ordinal);
                if (idx >= 0 && (next < 0 || idx < next || (idx == next && s.Length > hitLen)))
                {
                    next = idx; hitId = id; hitLen = s.Length;
                }
            }

            if (next < 0)
            {
                EncodeSegment(text.AsSpan(pos), ids);
                break;
            }
            if (next > pos)
                EncodeSegment(text.AsSpan(pos, next - pos), ids);
            ids.Add(hitId);
            pos = next + hitLen;
        }

        if (addSpecialTokens && AddEos) ids.Add(EosTokenId);
        return ids.ToArray();
    }

    /// <inheritdoc/>
    public int CountTokens(string text) => Encode(text).Length;

    /// <inheritdoc/>
    public string DecodeToken(int tokenId)
        => (uint)tokenId < (uint)_tokens.Length ? _tokens[tokenId] : string.Empty;

    /// <inheritdoc/>
    public string Decode(ReadOnlySpan<int> tokenIds)
    {
        var sb = new StringBuilder();
        foreach (int id in tokenIds)
        {
            if ((uint)id >= (uint)_tokens.Length || id == BosTokenId || id == EosTokenId || id == PadTokenId)
                continue;
            sb.Append(_tokens[id]);
        }
        string s = sb.ToString().Replace(EscapedSpace, " ");
        return s.Length > 0 && s[0] == ' ' ? s[1..] : s;
    }

    // ─────────────────────────── pipeline ───────────────────────────

    private void EncodeSegment(ReadOnlySpan<char> segment, List<int> ids)
    {
        byte[] normalized = Normalize(Encoding.UTF8.GetBytes(segment.ToString()));
        if (normalized.Length == 0)
            return;
        Viterbi(normalized, ids);
    }

    /// <summary>Port of <c>llm_tokenizer_ugm_session::normalize</c> (escape whitespace, prepend marker, merge spaces).</summary>
    private byte[] Normalize(byte[] input)
    {
        var output = new List<byte>(input.Length * 2 + 3);
        bool prepended = false, inWord = false;

        for (int offset = 0; offset < input.Length;)
        {
            ReadOnlySpan<byte> replacement = NormalizePrefix(input, offset, out int consumed);
            foreach (byte c in replacement)
            {
                if (c != (byte)' ')
                {
                    if (!inWord)
                    {
                        inWord = true;
                        if ((_addSpacePrefix && !prepended) || _removeExtraWhitespaces)
                        {
                            output.AddRange(EscapedSpaceUtf8);
                            prepended = true;
                        }
                    }
                    output.Add(c);
                }
                else
                {
                    inWord = false;
                    if (!_removeExtraWhitespaces)
                        output.AddRange(EscapedSpaceUtf8);
                }
            }
            offset += consumed;
        }
        return output.ToArray();
    }

    private ReadOnlySpan<byte> NormalizePrefix(byte[] input, int offset, out int consumed)
    {
        var rest = input.AsSpan(offset);

        // A user-defined piece is never rewritten.
        int udLen = 0;
        foreach (byte[] ud in _userDefined)
            if (ud.Length > udLen && rest.StartsWith(ud)) udLen = ud.Length;
        if (udLen > 0)
        {
            consumed = udLen;
            return rest[..udLen];
        }

        if (_charsMap is not null)
        {
            ReadOnlySpan<byte> rep = _charsMap.MatchLongest(rest, out int matched);
            if (matched > 0)
            {
                consumed = matched;
                return rep;
            }
        }

        consumed = Math.Min(Utf8Length(rest[0]), rest.Length);
        return rest[..consumed];
    }

    private void Viterbi(byte[] s, List<int> output)
    {
        int n = s.Length;
        var bestToken = new int[n + 1];
        var bestFrom = new int[n + 1];
        var bestScore = new double[n + 1];
        Array.Fill(bestScore, double.NegativeInfinity);
        bestScore[0] = 0;
        Array.Fill(bestToken, UnkTokenId);

        for (int offset = 0; offset < n;)
        {
            int cpLen = Math.Min(Utf8Length(s[offset]), n - offset);
            double cur = bestScore[offset];
            bool singleCodePoint = false;

            int node = 0;
            for (int p = offset; p < n; p++)
            {
                node = Child(node, s[p]);
                if (node < 0) break;
                int tid = _nodeToken[node];
                if (tid < 0) continue;
                int end = p + 1;
                if (end - offset == cpLen) singleCodePoint = true;
                double challenger = cur + _scores[tid];
                if (challenger > bestScore[end])
                {
                    bestScore[end] = challenger;
                    bestToken[end] = tid;
                    bestFrom[end] = offset;
                }
            }

            if (!singleCodePoint)
            {
                int end = offset + cpLen;
                double challenger = cur + _unknownScore;
                if (challenger > bestScore[end])
                {
                    bestScore[end] = challenger;
                    bestToken[end] = UnkTokenId;
                    bestFrom[end] = offset;
                }
            }
            offset += cpLen;
        }

        // Backtrack, fusing consecutive unknowns.
        int mark = output.Count;
        bool prevUnknown = false;
        for (int pos = n; pos > 0; pos = bestFrom[pos])
        {
            bool unknown = bestToken[pos] == UnkTokenId;
            if (!(prevUnknown && unknown))
                output.Add(bestToken[pos]);
            prevUnknown = unknown;
        }
        output.Reverse(mark, output.Count - mark);
    }

    private int Child(int node, byte b)
    {
        int count = _childCount[node];
        int lo = _childStart[node], hi = lo + count - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1;
            byte e = _edgeByte[mid];
            if (e == b) return mid;
            if (e < b) lo = mid + 1; else hi = mid - 1;
        }
        return -1;
    }

    private static int Utf8Length(byte lead)
        => lead < 0x80 ? 1 : lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;

    // ─────────────────────────── trie ───────────────────────────

    private static (int[] ChildStart, ushort[] ChildCount, byte[] EdgeByte, int[] NodeToken) BuildTrie(
        List<(byte[] Bytes, int Id)> pieces)
    {
        // Stable sort keeps duplicate texts in id order; the last duplicate wins (llama.cpp's trie insert overwrites).
        var order = new int[pieces.Count];
        for (int i = 0; i < order.Length; i++) order[i] = i;
        Array.Sort(order, (a, b) =>
        {
            int c = pieces[a].Bytes.AsSpan().SequenceCompareTo(pieces[b].Bytes);
            return c != 0 ? c : a.CompareTo(b);
        });

        var childStart = new List<int> { 0 };
        var childCount = new List<ushort> { 0 };
        var edgeByte = new List<byte> { 0 };
        var nodeToken = new List<int> { -1 };

        // BFS over (node, range [lo,hi) of sorted pieces sharing the node's prefix, depth).
        var queue = new Queue<(int Node, int Lo, int Hi, int Depth)>();
        queue.Enqueue((0, 0, order.Length, 0));
        while (queue.Count > 0)
        {
            var (node, lo, hi, depth) = queue.Dequeue();

            // Pieces equal to the prefix sort first; the last of them owns the node.
            while (lo < hi && pieces[order[lo]].Bytes.Length == depth)
            {
                nodeToken[node] = pieces[order[lo]].Id;
                lo++;
            }

            int first = childStart.Count;
            childStart[node] = first;
            int groups = 0;
            for (int i = lo; i < hi;)
            {
                byte b = pieces[order[i]].Bytes[depth];
                int j = i + 1;
                while (j < hi && pieces[order[j]].Bytes[depth] == b) j++;
                childStart.Add(0);
                childCount.Add(0);
                edgeByte.Add(b);
                nodeToken.Add(-1);
                queue.Enqueue((first + groups, i, j, depth + 1));
                groups++;
                i = j;
            }
            childCount[node] = (ushort)groups;
        }

        return (childStart.ToArray(), childCount.ToArray(), edgeByte.ToArray(), nodeToken.ToArray());
    }
}
