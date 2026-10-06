using System.Globalization;
using System.Text;

namespace DotLLM.Tokenizers.WordPiece;

/// <summary>
/// BERT-style WordPiece tokenizer (HuggingFace <c>BertTokenizer</c> semantics).
/// </summary>
/// <remarks>
/// <para>Pipeline: NFC → clean (drop NUL/U+FFFD/control, map whitespace to space) → pad CJK ideographs
/// with spaces → whitespace split → per word: optional lower-casing + accent stripping (NFD, drop
/// <c>Mn</c>) → split on punctuation → greedy longest-match-first WordPiece with the <c>##</c>
/// continuation prefix (a word that cannot be fully segmented, or is longer than
/// <see cref="MaxCharsPerWord"/> code points, becomes a single <c>[UNK]</c>).</para>
/// <para><see cref="Encode(string)"/> wraps the result as <c>[CLS] … [SEP]</c>, which is what every BERT
/// embedding model expects (llama.cpp's <c>llama_tokenize(add_special=true)</c> does the same).
/// Literal special-token strings in the input (<c>[CLS]</c>, <c>[SEP]</c>, <c>[MASK]</c>, …) are
/// recognised before normalisation, like HF.</para>
/// <para>Canonical vocabulary form here: word-initial pieces are stored bare, continuations as
/// <c>##piece</c>. GGUF stores the llama.cpp "WPM" form instead (word-initial pieces prefixed with
/// U+2581 <c>▁</c>, <c>##</c> stripped, bracketed specials untouched); <see cref="FromGgufTokens"/>
/// converts.</para>
/// </remarks>
public sealed class WordPieceTokenizer : ITokenizer
{
    private const string ContinuationPrefix = "##";

    private readonly Dictionary<string, int> _vocab;
    private readonly string[] _tokens;
    private readonly HashSet<string> _specialStrings;
    private readonly bool _lowerCase;
    private readonly bool _stripAccents;
    private readonly bool _tokenizeChinese;

    /// <summary>Maximum code points per word before it is replaced by <c>[UNK]</c> (BERT default 100).</summary>
    public int MaxCharsPerWord { get; }

    /// <summary>Id of <c>[CLS]</c>.</summary>
    public int ClsTokenId { get; }

    /// <summary>Id of <c>[SEP]</c>.</summary>
    public int SepTokenId { get; }

    /// <summary>Id of <c>[UNK]</c>.</summary>
    public int UnkTokenId { get; }

    /// <summary>Id of <c>[PAD]</c>, or -1 when absent.</summary>
    public int PadTokenId { get; }

    /// <summary>
    /// Creates a tokenizer over a canonical (HF-style) vocabulary.
    /// </summary>
    /// <param name="tokens">Vocabulary indexed by id; continuation pieces carry the <c>##</c> prefix.</param>
    /// <param name="clsId">Id of <c>[CLS]</c>.</param>
    /// <param name="sepId">Id of <c>[SEP]</c>.</param>
    /// <param name="unkId">Id of <c>[UNK]</c>.</param>
    /// <param name="padId">Id of <c>[PAD]</c> or -1.</param>
    /// <param name="lowerCase">Lower-case input (uncased models).</param>
    /// <param name="stripAccents">Strip combining marks; <c>null</c> follows <paramref name="lowerCase"/> (HF default).</param>
    /// <param name="tokenizeChinese">Split CJK ideographs into single-character words.</param>
    /// <param name="maxCharsPerWord">Per-word code point limit.</param>
    /// <param name="specialStrings">Literal strings recognised verbatim in the input; defaults to the five BERT specials present in the vocab.</param>
    public WordPieceTokenizer(
        string[] tokens, int clsId, int sepId, int unkId, int padId = -1,
        bool lowerCase = true, bool? stripAccents = null, bool tokenizeChinese = true,
        int maxCharsPerWord = 100, IEnumerable<string>? specialStrings = null)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        _tokens = tokens;
        _vocab = new Dictionary<string, int>(tokens.Length, StringComparer.Ordinal);
        for (int i = 0; i < tokens.Length; i++)
            _vocab.TryAdd(tokens[i], i);

        foreach (var (name, id) in new[] { ("CLS", clsId), ("SEP", sepId), ("UNK", unkId) })
        {
            if ((uint)id >= (uint)tokens.Length)
                throw new ArgumentOutOfRangeException(nameof(tokens), $"[{name}] id {id} is outside the {tokens.Length}-entry vocabulary.");
        }

        ClsTokenId = clsId;
        SepTokenId = sepId;
        UnkTokenId = unkId;
        PadTokenId = padId;
        _lowerCase = lowerCase;
        _stripAccents = stripAccents ?? lowerCase;
        _tokenizeChinese = tokenizeChinese;
        MaxCharsPerWord = maxCharsPerWord;

        _specialStrings = new HashSet<string>(StringComparer.Ordinal);
        foreach (string s in specialStrings ?? ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"])
        {
            if (_vocab.ContainsKey(s))
                _specialStrings.Add(s);
        }
    }

    /// <summary>
    /// Builds a tokenizer from llama.cpp-style GGUF tokens (<c>tokenizer.ggml.model == "bert"</c>).
    /// </summary>
    /// <param name="ggufTokens">
    /// <c>tokenizer.ggml.tokens</c>: word-initial pieces prefixed with U+2581, continuation pieces bare,
    /// bracketed specials (<c>[CLS]</c>) bare.
    /// </param>
    /// <param name="clsId">CLS id (GGUF <c>cls_token_id</c> or, for older files, <c>bos_token_id</c>); -1 resolves <c>[CLS]</c> by string.</param>
    /// <param name="sepId">SEP id (<c>seperator_token_id</c> or <c>eos_token_id</c>); -1 resolves <c>[SEP]</c> by string.</param>
    /// <param name="unkId">UNK id (<c>unknown_token_id</c>); -1 resolves <c>[UNK]</c> by string.</param>
    /// <param name="padId">PAD id or -1 (resolved by string when -1).</param>
    /// <param name="lowerCase">GGUF carries no casing flag; llama.cpp always lower-cases, so the default is true.</param>
    public static WordPieceTokenizer FromGgufTokens(
        string[] ggufTokens, int clsId = -1, int sepId = -1, int unkId = -1, int padId = -1, bool lowerCase = true)
    {
        ArgumentNullException.ThrowIfNull(ggufTokens);
        var canonical = new string[ggufTokens.Length];
        for (int i = 0; i < ggufTokens.Length; i++)
        {
            string t = ggufTokens[i];
            if (t.Length > 1 && t[0] == '▁')
                canonical[i] = t[1..];
            else if (t.Length > 2 && t[0] == '[' && t[^1] == ']')
                canonical[i] = t;
            else
                canonical[i] = ContinuationPrefix + t;
        }

        int Find(string s)
        {
            for (int i = 0; i < canonical.Length; i++)
                if (canonical[i] == s) return i;
            return -1;
        }

        if (clsId < 0) clsId = Find("[CLS]");
        if (sepId < 0) sepId = Find("[SEP]");
        if (unkId < 0) unkId = Find("[UNK]");
        if (padId < 0) padId = Find("[PAD]");
        if (clsId < 0 || sepId < 0 || unkId < 0)
            throw new InvalidDataException("BERT GGUF vocabulary is missing [CLS]/[SEP]/[UNK].");

        return new WordPieceTokenizer(canonical, clsId, sepId, unkId, padId, lowerCase);
    }

    /// <inheritdoc/>
    public int VocabSize => _tokens.Length;

    /// <inheritdoc/>
    public int BosTokenId => ClsTokenId;

    /// <inheritdoc/>
    public int EosTokenId => SepTokenId;

    /// <summary>Encodes <paramref name="text"/> as <c>[CLS] pieces… [SEP]</c>.</summary>
    public int[] Encode(string text) => Encode(text, addSpecialTokens: true);

    /// <summary>Encodes <paramref name="text"/>, optionally wrapped in <c>[CLS]</c>/<c>[SEP]</c>.</summary>
    public int[] Encode(string text, bool addSpecialTokens)
    {
        ArgumentNullException.ThrowIfNull(text);
        var ids = new List<int>(text.Length / 3 + 4);
        if (addSpecialTokens) ids.Add(ClsTokenId);

        // Split out literal special tokens first; everything between goes through the pipeline.
        int pos = 0;
        while (pos < text.Length)
        {
            int next = -1;
            string? hit = null;
            foreach (string s in _specialStrings)
            {
                int idx = text.IndexOf(s, pos, StringComparison.Ordinal);
                if (idx >= 0 && (next < 0 || idx < next || (idx == next && s.Length > hit!.Length)))
                {
                    next = idx;
                    hit = s;
                }
            }

            if (next < 0)
            {
                EncodeSegment(text.AsSpan(pos), ids);
                break;
            }

            if (next > pos)
                EncodeSegment(text.AsSpan(pos, next - pos), ids);
            ids.Add(_vocab[hit!]);
            pos = next + hit!.Length;
        }

        if (addSpecialTokens) ids.Add(SepTokenId);
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
            if ((uint)id >= (uint)_tokens.Length) continue;
            string t = _tokens[id];
            if (t.StartsWith(ContinuationPrefix, StringComparison.Ordinal))
                sb.Append(t, ContinuationPrefix.Length, t.Length - ContinuationPrefix.Length);
            else
            {
                if (sb.Length > 0) sb.Append(' ');
                sb.Append(t);
            }
        }
        return sb.ToString();
    }

    // ─────────────────────────── pipeline ───────────────────────────

    private void EncodeSegment(ReadOnlySpan<char> segment, List<int> ids)
    {
        string normalized = BasicNormalize(segment);
        foreach (string word in BasicSplit(normalized))
            WordPiece(word, ids);
    }

    /// <summary>NFC, clean and CJK padding (HF <c>BasicTokenizer</c> steps 1-2).</summary>
    private string BasicNormalize(ReadOnlySpan<char> segment)
    {
        string nfc = segment.IsEmpty ? string.Empty : segment.ToString().Normalize(NormalizationForm.FormC);
        var sb = new StringBuilder(nfc.Length + 8);
        foreach (Rune r in nfc.EnumerateRunes())
        {
            int cp = r.Value;
            if (cp == 0 || cp == 0xFFFD || IsControl(r)) continue;
            if (IsWhitespace(r))
            {
                sb.Append(' ');
                continue;
            }

            if (_tokenizeChinese && IsCjk(cp))
            {
                sb.Append(' ').Append(r.ToString()).Append(' ');
                continue;
            }
            sb.Append(r.ToString());
        }
        return sb.ToString();
    }

    /// <summary>Whitespace split, lower/strip-accents, punctuation split.</summary>
    private IEnumerable<string> BasicSplit(string normalized)
    {
        foreach (string raw in normalized.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
            string w = raw;
            if (_lowerCase) w = w.ToLowerInvariant();
            if (_stripAccents) w = StripAccents(w);

            var cur = new StringBuilder();
            foreach (Rune r in w.EnumerateRunes())
            {
                if (IsPunctuation(r))
                {
                    if (cur.Length > 0) { yield return cur.ToString(); cur.Clear(); }
                    yield return r.ToString();
                }
                else
                {
                    cur.Append(r.ToString());
                }
            }
            if (cur.Length > 0) yield return cur.ToString();
        }
    }

    private void WordPiece(string word, List<int> ids)
    {
        // Code point boundaries so a candidate never splits a surrogate pair.
        var bounds = new List<int>(word.Length + 1);
        for (int i = 0; i < word.Length; i += char.IsHighSurrogate(word[i]) && i + 1 < word.Length ? 2 : 1)
            bounds.Add(i);
        bounds.Add(word.Length);
        int cpCount = bounds.Count - 1;

        if (cpCount > MaxCharsPerWord)
        {
            ids.Add(UnkTokenId);
            return;
        }

        int mark = ids.Count;
        int start = 0;
        while (start < cpCount)
        {
            int end = cpCount;
            int found = -1;
            while (end > start)
            {
                string piece = word.Substring(bounds[start], bounds[end] - bounds[start]);
                if (start > 0) piece = ContinuationPrefix + piece;
                if (_vocab.TryGetValue(piece, out int id))
                {
                    found = id;
                    break;
                }
                end--;
            }

            if (found < 0)
            {
                ids.RemoveRange(mark, ids.Count - mark);
                ids.Add(UnkTokenId);
                return;
            }

            ids.Add(found);
            start = end;
        }
    }

    // ─────────────────────────── Unicode predicates (HF _is_* equivalents) ───────────────────────────

    private static bool IsWhitespace(Rune r)
        => r.Value is ' ' or '\t' or '\n' or '\r' || Rune.GetUnicodeCategory(r) == UnicodeCategory.SpaceSeparator;

    private static bool IsControl(Rune r)
    {
        if (r.Value is '\t' or '\n' or '\r') return false;
        var cat = Rune.GetUnicodeCategory(r);
        return cat is UnicodeCategory.Control or UnicodeCategory.Format
            or UnicodeCategory.Surrogate or UnicodeCategory.PrivateUse or UnicodeCategory.OtherNotAssigned;
    }

    private static bool IsPunctuation(Rune r)
    {
        int cp = r.Value;
        if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126))
            return true;
        var cat = Rune.GetUnicodeCategory(r);
        return cat is UnicodeCategory.ConnectorPunctuation or UnicodeCategory.DashPunctuation
            or UnicodeCategory.OpenPunctuation or UnicodeCategory.ClosePunctuation
            or UnicodeCategory.InitialQuotePunctuation or UnicodeCategory.FinalQuotePunctuation
            or UnicodeCategory.OtherPunctuation;
    }

    private static bool IsCjk(int cp)
        => (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF)
        || (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F)
        || (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF)
        || (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);

    private static string StripAccents(string s)
    {
        string d = s.Normalize(NormalizationForm.FormD);
        var sb = new StringBuilder(d.Length);
        foreach (Rune r in d.EnumerateRunes())
        {
            if (Rune.GetUnicodeCategory(r) != UnicodeCategory.NonSpacingMark)
                sb.Append(r.ToString());
        }
        return sb.ToString();
    }
}
