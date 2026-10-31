namespace DotLLM.Tokenizers.Bpe;

/// <summary>
/// Byte-pair encoding tokenizer supporting SentencePiece and tiktoken variants.
/// Vocabulary data is loaded at construction; all encoding and decoding is delegated
/// to the variant-specific <see cref="IBpeEncoding"/> implementation.
/// Special tokens (control/user-defined) are pre-split from the input and emitted
/// as single token IDs, bypassing BPE encoding.
/// </summary>
public sealed class BpeTokenizer : ITokenizer
{
    private readonly IBpeEncoding _encoding;

    /// <summary>
    /// Special tokens sorted by descending length so longer tokens match first.
    /// Kept for <see cref="BuildSpecialTokenTable"/>'s output contract; the hot-path
    /// encode loop uses <see cref="_specialTokenTrie"/> instead.
    /// </summary>
    private readonly (string Text, int Id)[] _specialTokens;

    /// <summary>
    /// Trie built from <see cref="_specialTokens"/> for O(L) longest-prefix matching
    /// at each text position. <see langword="null"/> when there are no special tokens
    /// (the tokenizer takes the fast path in <see cref="Encode"/>).
    /// </summary>
    private readonly Trie? _specialTokenTrie;

    /// <inheritdoc/>
    public int BosTokenId { get; }

    /// <inheritdoc/>
    public int EosTokenId { get; }

    /// <summary>Extra end-of-generation ids from the model file (eot / eom); set by the GGUF loader.</summary>
    public int[] ExtraEndOfGenerationTokenIds { get; set; } = [];

    IReadOnlyList<int> ITokenizer.ExtraEndOfGenerationTokenIds => ExtraEndOfGenerationTokenIds;

    /// <inheritdoc/>
    public int VocabSize { get; }

    private BpeTokenizer(IBpeEncoding encoding, (string Text, int Id)[] specialTokens,
        int bosId, int eosId, int vocabSize)
    {
        _encoding = encoding;
        _specialTokens = specialTokens;
        _specialTokenTrie = BuildSpecialTokenTrie(specialTokens);
        BosTokenId = bosId;
        EosTokenId = eosId;
        VocabSize = vocabSize;
    }

    private static Trie? BuildSpecialTokenTrie((string Text, int Id)[] specialTokens)
    {
        if (specialTokens.Length == 0)
            return null;
        var trie = new Trie();
        foreach (var (text, id) in specialTokens)
            trie.Add(text, id, score: 0f);
        return trie;
    }

    /// <summary>
    /// Creates a SentencePiece BPE tokenizer (Llama 1/2, Mistral, TinyLlama, SmolLM).
    /// </summary>
    /// <param name="tokens">Vocabulary strings indexed by token ID.</param>
    /// <param name="scores">Unigram log-probability scores (higher = preferred merge).</param>
    /// <param name="tokenTypes">Per-token type flags (1=normal, 2=unknown, 3=control, 4=user-defined, 5=unused, 6=byte). Null = all normal.</param>
    /// <param name="bosId">Beginning-of-sequence token ID.</param>
    /// <param name="eosId">End-of-sequence token ID.</param>
    /// <param name="addBosSpace">Prepend ▁ to text that doesn't start with a space (matches SentencePiece default).</param>
    public static BpeTokenizer CreateSentencePiece(
        string[] tokens, float[] scores, int[]? tokenTypes,
        int bosId, int eosId, bool addBosSpace = true)
    {
        float[] safeScores = scores.Length == tokens.Length ? scores : new float[tokens.Length];
        var specialTokens = BuildSpecialTokenTable(tokens, tokenTypes);
        return new BpeTokenizer(
            new SentencePieceEncoding(tokens, safeScores, tokenTypes, addBosSpace),
            specialTokens, bosId, eosId, tokens.Length);
    }

    /// <summary>
    /// Creates a tiktoken BPE tokenizer (Llama 3, GPT-4).
    /// </summary>
    /// <param name="tokens">Vocabulary strings indexed by token ID.</param>
    /// <param name="merges">Merge table entries in "A B" format; index = rank (lower = applied first).</param>
    /// <param name="tokenTypes">Per-token type flags. Null = all normal.</param>
    /// <param name="bosId">Beginning-of-sequence token ID.</param>
    /// <param name="eosId">End-of-sequence token ID.</param>
    /// <param name="preTokenizerType">GGUF <c>tokenizer.ggml.pre</c> value (e.g., "llama3", "gpt2").
    /// Null/absent = the GPT-2 default pipeline (llama.cpp's "missing pre-tokenizer type" fallback);
    /// an unknown value throws <see cref="System.IO.InvalidDataException"/> unless
    /// <c>DOTLLM_ALLOW_UNKNOWN_PRETOKENIZER=1</c> (issue #373).</param>
    public static BpeTokenizer CreateTiktoken(
        string[] tokens, string[] merges, int[]? tokenTypes,
        int bosId, int eosId, string? preTokenizerType = null)
    {
        var specialTokens = BuildSpecialTokenTable(tokens, tokenTypes);
        var preRegexes = TiktokenPreTokenizer.GetRegexes(preTokenizerType);
        return new BpeTokenizer(
            new Gpt2TiktokenEncoding(tokens, merges, tokenTypes, preRegexes),
            specialTokens, bosId, eosId, tokens.Length);
    }

    /// <summary>
    /// Creates a Gemma-4 SPM-style merge-ranked BPE tokenizer
    /// (GGUF <c>tokenizer.ggml.model = "gemma4"</c>).
    /// Spaces are escaped to <c>▁</c> before merge-ranked BPE runs over raw UTF-8 text;
    /// pre-tokenization only splits newline runs (llama.cpp <c>LLAMA_VOCAB_PRE_TYPE_GEMMA4</c>).
    /// </summary>
    /// <param name="tokens">Vocabulary strings indexed by token ID.</param>
    /// <param name="merges">Merge table entries in "A B" format; index = rank (lower = applied first).</param>
    /// <param name="tokenTypes">Per-token type flags. Null = all normal.</param>
    /// <param name="bosId">Beginning-of-sequence token ID.</param>
    /// <param name="eosId">End-of-sequence token ID.</param>
    public static BpeTokenizer CreateGemma4(
        string[] tokens, string[] merges, int[]? tokenTypes, int bosId, int eosId)
    {
        var specialTokens = BuildSpecialTokenTable(tokens, tokenTypes);
        return new BpeTokenizer(
            new Gemma4SpmBpeEncoding(tokens, merges, tokenTypes),
            specialTokens, bosId, eosId, tokens.Length);
    }

    /// <summary>
    /// Creates a tiktoken BPE tokenizer with an explicit, pre-compiled
    /// pre-tokenization regex. Used by the HuggingFace <c>tokenizer.json</c>
    /// adapter when the model declares a <c>Split</c> regex that is not
    /// one of the built-in GGUF presets (e.g., Qwen2's custom
    /// <c>(?i:'s|'t|...)|\p{N}|...</c> pattern).
    /// </summary>
    /// <param name="tokens">Vocabulary strings indexed by token ID.</param>
    /// <param name="merges">Merge table entries in "A B" format; index = rank (lower = applied first).</param>
    /// <param name="tokenTypes">Per-token type flags. Null = all normal.</param>
    /// <param name="bosId">Beginning-of-sequence token ID.</param>
    /// <param name="eosId">End-of-sequence token ID.</param>
    /// <param name="preRegex">
    /// Compiled pre-tokenization regex. <see langword="null"/> means no
    /// pre-splitting — the whole input is fed to BPE as one segment.
    /// </param>
    public static BpeTokenizer CreateTiktokenWithRegex(
        string[] tokens, string[] merges, int[]? tokenTypes,
        int bosId, int eosId, System.Text.RegularExpressions.Regex? preRegex)
    {
        var specialTokens = BuildSpecialTokenTable(tokens, tokenTypes);
        // A single expression is a one-stage pipeline. Pre-types whose reference definition has
        // several stages (the StarCoder/SmolLM family) must go through CreateTiktoken instead.
        return new BpeTokenizer(
            new Gpt2TiktokenEncoding(tokens, merges, tokenTypes, preRegex is null ? null : [preRegex]),
            specialTokens, bosId, eosId, tokens.Length);
    }

    /// <inheritdoc/>
    public int[] EncodeRaw(string text)
    {
        if (text.Length == 0)
            return [];

        // #1339: markers (ControlTokenGuard) are removed here; text they protected is encoded as plain text.
        bool marked = ControlTokenGuard.IsMarked(text);

        // Fast path: no special tokens to split on
        if (_specialTokens.Length == 0)
        {
            string plain = marked ? ControlTokenGuard.Strip(text) : text;
            return plain.Length == 0 ? [] : _encoding.Encode(plain);
        }
        return EncodeWithSpecialTokens(text, marked);
    }

    /// <inheritdoc/>
    public string MarkUntrustedSpecials(string text, IReadOnlyList<string>? exemptPrefixes = null)
    {
        var trie = _specialTokenTrie;
        if (trie is null || text.Length < 2)
            return text;

        System.Text.StringBuilder? sb = null;
        int copied = 0;
        for (int p = 0; p < text.Length - 1; p++)
        {
            if (!trie.TryMatchLongest(text.AsSpan(p), out _, out _, out _))
                continue;
            if (exemptPrefixes is { Count: > 0 } && StartsWithAny(text, p, exemptPrefixes))
                continue;
            // The marker goes after the match's first character: every special spelling has length > 1, so the
            // spelling can no longer match, and overlapping spellings (each with its own start) are all broken.
            sb ??= new System.Text.StringBuilder(text.Length + 8);
            sb.Append(text, copied, p + 1 - copied).Append(ControlTokenGuard.Marker);
            copied = p + 1;
        }
        if (sb is null)
            return text;
        return sb.Append(text, copied, text.Length - copied).ToString();
    }

    private static bool StartsWithAny(string text, int pos, IReadOnlyList<string> prefixes)
    {
        foreach (string prefix in prefixes)
        {
            if (prefix.Length > 0 && string.CompareOrdinal(text, pos, prefix, 0, prefix.Length) == 0)
                return true;
        }
        return false;
    }

    /// <inheritdoc/>
    public int[] Encode(string text)
    {
        if (text.Length == 0)
            return [];

        int[] ids = EncodeRaw(text);

        if (AddBosToken && (ids.Length == 0 || ids[0] != BosTokenId))
        {
            var withBos = new int[ids.Length + 1];
            withBos[0] = BosTokenId;
            ids.CopyTo(withBos, 1);
            ids = withBos;
        }
        if (AddEosToken && (ids.Length == 0 || ids[^1] != EosTokenId))
        {
            var withEos = new int[ids.Length + 1];
            ids.CopyTo(withEos, 0);
            withEos[^1] = EosTokenId;
            ids = withEos;
        }
        return ids;
    }

    /// <summary>
    /// When true, <see cref="Encode"/> appends <see cref="EosTokenId"/> unless the encoded text already ends with it.
    /// Off by default; the GGUF factory switches it on only for embedding towers whose tokenizer template frames the
    /// text as <c>&lt;bos&gt; … &lt;eos&gt;</c> (EmbeddingGemma, <c>gemma-embedding</c>). Never set for generative models:
    /// an EOS after the prompt would end the turn.
    /// </summary>
    public bool AddEosToken { get; set; }

    /// <summary>
    /// When true, <see cref="Encode"/> prepends <see cref="BosTokenId"/> unless the encoded text
    /// already starts with it (e.g. a chat-template-rendered prompt that begins with the literal
    /// <c>&lt;bos&gt;</c> special token). llama.cpp's <c>add_bos</c> behaviour. Off by default —
    /// the GGUF factory switches it on only for Gemma 1/2, which produce garbage without BOS.
    /// </summary>
    public bool AddBosToken { get; set; }

    /// <inheritdoc/>
    public string Decode(ReadOnlySpan<int> tokenIds) =>
        tokenIds.IsEmpty ? string.Empty : _encoding.Decode(tokenIds);

    /// <inheritdoc/>
    public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) =>
        tokenIds.IsEmpty ? string.Empty : _encoding.Decode(tokenIds, stripBosSpace);

    /// <inheritdoc/>
    public string DecodeToken(int tokenId) => _encoding.DecodeToken(tokenId);

    /// <inheritdoc/>
    public int CountTokens(string text) => Encode(text).Length;

    /// <summary>
    /// Builds the special token table from vocabulary and token types.
    /// Control tokens (type 3) and user-defined tokens (type 4) that are non-empty
    /// and not single-byte are treated as special tokens for pre-splitting.
    /// Sorted by descending length for longest-match-first semantics.
    /// </summary>
    private static (string Text, int Id)[] BuildSpecialTokenTable(string[] tokens, int[]? tokenTypes)
    {
        if (tokenTypes is null)
            return [];

        var special = new List<(string Text, int Id)>();
        for (int i = 0; i < tokens.Length && i < tokenTypes.Length; i++)
        {
            // Type 3 = control, Type 4 = user-defined (added tokens)
            // Skip single chars and empty strings — they're not useful for pre-splitting
            if ((tokenTypes[i] == 3 || tokenTypes[i] == 4) &&
                tokens[i].Length > 1 &&
                !string.IsNullOrEmpty(tokens[i]))
            {
                special.Add((tokens[i], i));
            }
        }

        // Sort by descending length so longer tokens match first
        special.Sort((a, b) => b.Text.Length.CompareTo(a.Text.Length));
        return special.ToArray();
    }

    /// <summary>
    /// Encodes text with special token pre-splitting. Scans for special tokens via the
    /// pre-built trie (O(L) per position where L is the match length), emits their IDs
    /// directly, and BPE-encodes the text segments between them.
    /// </summary>
    private int[] EncodeWithSpecialTokens(string text, bool marked)
    {
        var trie = _specialTokenTrie!; // Encode() fast-paths when _specialTokens.Length == 0
        var result = new List<int>();
        int pos = 0;
        bool isFirstSegment = true;

        while (pos < text.Length)
        {
            if (trie.TryMatchLongest(text.AsSpan(pos), out int matchedId, out _, out int matchedLen))
            {
                // Emit the special token ID directly
                result.Add(matchedId);
                pos += matchedLen;
                isFirstSegment = false;
            }
            else
            {
                // Find the next special token (or end of string)
                int nextSpecialPos = FindNextSpecialToken(text, pos + 1);
                string segment = text[pos..nextSpecialPos];
                if (marked)
                    segment = ControlTokenGuard.Strip(segment);

                // BPE-encode the segment.
                // First segment gets normal encoding (with BOS space prepend for SentencePiece).
                // Subsequent segments use EncodeSegment (no BOS space) to avoid spurious ▁ markers.
                if (segment.Length > 0)
                {
                    int[] segmentIds = isFirstSegment
                        ? _encoding.Encode(segment)
                        : _encoding.EncodeSegment(segment);
                    result.AddRange(segmentIds);
                }

                pos = nextSpecialPos;
                isFirstSegment = false;
            }
        }

        return result.ToArray();
    }

    /// <summary>
    /// Finds the position of the next special token starting from <paramref name="startPos"/>.
    /// Returns <c>text.Length</c> if no special token is found.
    /// </summary>
    private int FindNextSpecialToken(string text, int startPos)
    {
        var trie = _specialTokenTrie!;
        for (int pos = startPos; pos < text.Length; pos++)
        {
            if (trie.TryMatchLongest(text.AsSpan(pos), out _, out _, out _))
                return pos;
        }

        return text.Length;
    }
}
