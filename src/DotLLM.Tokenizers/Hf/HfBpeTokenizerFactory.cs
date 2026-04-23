using System.Text;
using System.Text.RegularExpressions;
using DotLLM.Tokenizers.Bpe;

namespace DotLLM.Tokenizers.Hf;

/// <summary>
/// Bridges a parsed HuggingFace <see cref="HfTokenizerSpec"/> into a
/// <see cref="ITokenizer"/>. HF <c>tokenizer.json</c> files declare the
/// tokenization pipeline as (normalizer, pre-tokenizer, BPE model, decoder);
/// this factory inspects the spec and selects the matching dotLLM
/// encoder — either <see cref="BpeTokenizer.CreateSentencePiece"/> for the
/// Llama-family <c>Metaspace + ByteFallback</c> layout or
/// <see cref="BpeTokenizer.CreateTiktoken"/> for the GPT-2-family
/// <c>ByteLevel + ByteLevel</c> layout (Qwen / Llama-3 / Granite dense / GPT-2).
/// </summary>
/// <remarks>
/// <para>
/// <b>Pipelines supported today:</b>
/// </para>
/// <list type="bullet">
///   <item><description><c>Metaspace</c> pre-tokenizer + <c>Sequence[Replace, ByteFallback, Fuse, Strip]</c>
///     decoder (Llama 1/2, Mistral, TinyLlama, Phi-3.5-mini, ib-ssm).</description></item>
///   <item><description><c>ByteLevel</c> pre-tokenizer (<c>use_regex: true</c>) +
///     <c>ByteLevel</c> decoder — Granite-3 dense, GPT-2 proper.</description></item>
///   <item><description><c>Sequence[Split, ByteLevel(use_regex:false)]</c> pre-tokenizer +
///     <c>ByteLevel</c> decoder — Qwen-2 / Qwen-2.5.</description></item>
/// </list>
/// <para>
/// <b>Not supported yet:</b> multi-step pre-tokenizer sequences (DeepSeek-V2
/// uses five <c>Split</c>s + <c>Digits</c> + <c>ByteLevel</c>). That requires
/// sequentially applying each pre-tokenizer step and splitting the text at
/// matches — tracked as P0.1 follow-up.
/// </para>
/// <para>
/// <b>Merge scoring.</b> For the Metaspace path, the merge rank is converted
/// to a synthetic score (<c>score = -rank</c>) that matches HF's
/// earliest-rank-wins semantics. For the ByteLevel path the merge list is
/// passed through directly as <c>"left right"</c> strings to
/// <see cref="BpeTokenizer.CreateTiktoken"/>, which keys the merge table by
/// rank.
/// </para>
/// <para>
/// <b>Special tokens.</b> Tokens declared in <c>added_tokens</c> with
/// <c>special: true</c> map to <see cref="Bpe.BpeTokenizer"/> type-3 control
/// tokens and pre-split the input during encoding. The factory also infers
/// BOS / EOS / UNK IDs when explicit values are not supplied.
/// </para>
/// </remarks>
public static class HfBpeTokenizerFactory
{
    /// <summary>Default synthetic score assigned to vocab entries that are not produced by any merge.</summary>
    private const float NonMergeBaseScore = -1e9f;

    /// <summary>
    /// Builds a tokenizer from a parsed HuggingFace spec. Pass -1 for
    /// <paramref name="bosId"/> / <paramref name="eosId"/> to let the
    /// factory auto-detect <c>&lt;s&gt;</c> / <c>&lt;/s&gt;</c> / <c>&lt;|endoftext|&gt;</c>
    /// from the added-tokens list (falling back to <c>1</c> / <c>2</c> if
    /// not found, which matches Llama-2 convention).
    /// </summary>
    /// <param name="spec">Parsed tokenizer spec.</param>
    /// <param name="bosId">Override BOS token ID, or -1 to auto-detect.</param>
    /// <param name="eosId">Override EOS token ID, or -1 to auto-detect.</param>
    /// <returns>An <see cref="ITokenizer"/> ready for encode/decode calls.</returns>
    /// <exception cref="InvalidDataException">
    /// The spec declares a pipeline this factory does not yet support
    /// (e.g. an un-wired pre-tokenizer kind or a multi-step Sequence that
    /// isn't <c>[Split, ByteLevel]</c>).
    /// </exception>
    public static ITokenizer Create(HfTokenizerSpec spec, int bosId = -1, int eosId = -1)
    {
        ArgumentNullException.ThrowIfNull(spec);

        return spec.PreTokenizerKind switch
        {
            HfPreTokenizerKind.Metaspace or HfPreTokenizerKind.None
                => CreateSentencePiece(spec, bosId, eosId),
            HfPreTokenizerKind.ByteLevel
                => CreateByteLevel(spec, bosId, eosId),
            _ => throw new InvalidDataException(
                $"HF tokenizer.json pre-tokenizer '{spec.PreTokenizerKind}' is not supported yet. "
                + "Supported pipelines: Metaspace+ByteFallback (SPM) and ByteLevel (GPT-2/Qwen/Granite)."),
        };
    }

    /// <summary>
    /// Convenience entry point — parses the JSON and creates the tokenizer
    /// in one call.
    /// </summary>
    public static ITokenizer Create(string jsonContent, int bosId = -1, int eosId = -1)
        => Create(HfTokenizerJsonParser.Parse(jsonContent), bosId, eosId);

    /// <summary>
    /// Loads a HuggingFace tokenizer from a checkpoint directory. Tries the
    /// consolidated <c>tokenizer.json</c> first (the modern layout used by
    /// most post-2023 checkpoints). When that is absent, falls back to the
    /// legacy trio — <c>vocab.json</c> + <c>merges.txt</c> + optionally
    /// <c>tokenizer_config.json</c> — used by GPT-2 proper and older Granite /
    /// Llama-family repos. Returns <see langword="null"/> when neither layout
    /// is present.
    /// </summary>
    /// <param name="directory">Checkpoint directory.</param>
    /// <param name="bosId">Override BOS token ID, or -1 to auto-detect.</param>
    /// <param name="eosId">Override EOS token ID, or -1 to auto-detect.</param>
    /// <returns>An <see cref="ITokenizer"/>, or <see langword="null"/> if no recognised tokenizer files were found.</returns>
    public static ITokenizer? TryLoadFromDirectory(string directory, int bosId = -1, int eosId = -1)
    {
        ArgumentNullException.ThrowIfNull(directory);

        // Preferred path: the consolidated tokenizer.json.
        string jsonPath = Path.Combine(directory, "tokenizer.json");
        if (File.Exists(jsonPath))
        {
            string content = File.ReadAllText(jsonPath);
            return Create(content, bosId, eosId);
        }

        // Legacy fallback: vocab.json + merges.txt + tokenizer_config.json
        // (GPT-2, IBM Granite-3.0 MoE, older Llama / Mistral checkpoints).
        HfTokenizerSpec? legacySpec = HfLegacyBpeLoader.TryLoad(directory);
        return legacySpec is null ? null : Create(legacySpec, bosId, eosId);
    }

    // -------------------------------------------------------------------------
    // SentencePiece / Metaspace path (Llama 1/2, Mistral, TinyLlama, Phi-3.5, ib-ssm)
    // -------------------------------------------------------------------------

    private static ITokenizer CreateSentencePiece(HfTokenizerSpec spec, int bosId, int eosId)
    {
        string[] tokens = BuildIdToTokenTable(spec.Vocab);
        float[] scores = BuildScoreTable(tokens, spec.Vocab, spec.Merges);
        int[] tokenTypes = BuildTokenTypes(tokens, spec.AddedTokens);

        int resolvedBos = bosId >= 0 ? bosId : FindAddedTokenId(spec, "<s>", fallback: 1);
        int resolvedEos = eosId >= 0 ? eosId : FindAddedTokenId(spec, "</s>", fallback: 2);

        // addBosSpace matches Metaspace's prepend_scheme = "first" (Llama default).
        // "never" disables it, but no SPM checkpoint we support ships with that.
        bool addBosSpace = !string.Equals(spec.MetaspacePrependScheme, "never", StringComparison.Ordinal);

        return BpeTokenizer.CreateSentencePiece(
            tokens, scores, tokenTypes,
            resolvedBos, resolvedEos, addBosSpace);
    }

    // -------------------------------------------------------------------------
    // ByteLevel / GPT-2 path (Qwen, Llama-3, Granite dense, GPT-2)
    // -------------------------------------------------------------------------

    private static ITokenizer CreateByteLevel(HfTokenizerSpec spec, int bosId, int eosId)
    {
        string[] tokens = BuildIdToTokenTable(spec.Vocab);
        int[] tokenTypes = BuildTokenTypes(tokens, spec.AddedTokens);
        string[] mergeStrings = BuildByteLevelMerges(spec.Merges);

        int resolvedBos = bosId >= 0 ? bosId : FindByteLevelBos(spec);
        int resolvedEos = eosId >= 0 ? eosId : FindByteLevelEos(spec);

        // Regex selection:
        //   • Sequence[Split, ByteLevel] → use the Split's regex (Qwen2).
        //   • Standalone ByteLevel with use_regex:true → default GPT-2 regex (Granite-3, GPT-2).
        //   • Standalone ByteLevel with use_regex:false → no regex (whole string as one segment).
        Regex? preRegex;
        if (!string.IsNullOrEmpty(spec.ByteLevelSplitRegex))
            preRegex = new Regex(spec.ByteLevelSplitRegex, RegexOptions.Compiled);
        else if (spec.ByteLevelUseRegex ?? true)
            preRegex = ByteLevelPreTokenizer.DefaultGpt2Regex;
        else
            preRegex = null;

        ITokenizer inner = BpeTokenizer.CreateTiktokenWithRegex(
            tokens, mergeStrings, tokenTypes,
            resolvedBos, resolvedEos, preRegex);

        // Apply a normalizer decorator if the spec declares one (Qwen2 → NFC).
        return spec.NormalizerKind switch
        {
            HfNormalizerKind.Nfc => new NormalizingTokenizer(inner, NormalizationForm.FormC),
            HfNormalizerKind.Nfd => new NormalizingTokenizer(inner, NormalizationForm.FormD),
            HfNormalizerKind.Nfkc => new NormalizingTokenizer(inner, NormalizationForm.FormKC),
            HfNormalizerKind.Nfkd => new NormalizingTokenizer(inner, NormalizationForm.FormKD),
            _ => inner,
        };
    }

    /// <summary>
    /// Formats HF merges as the <c>"left right"</c> strings that
    /// <see cref="BpeTokenizer.CreateTiktoken"/> expects.
    /// </summary>
    private static string[] BuildByteLevelMerges(IReadOnlyList<(string Left, string Right)> merges)
    {
        string[] result = new string[merges.Count];
        for (int i = 0; i < merges.Count; i++)
        {
            (string left, string right) = merges[i];
            result[i] = string.Concat(left, " ", right);
        }
        return result;
    }

    private static int FindByteLevelBos(HfTokenizerSpec spec)
    {
        // Prefer explicit BOS-like tokens in the added list; fall back to
        // "<|endoftext|>" (GPT-2) or <s> (Granite Granite-3 encodes BOS as <|end_of_text|>).
        foreach (HfAddedToken t in spec.AddedTokens)
        {
            if (t.Content is "<|endoftext|>" or "<|begin_of_text|>" or "<s>")
                return t.Id;
        }
        // Fall back: vocab lookup.
        if (spec.Vocab.TryGetValue("<|endoftext|>", out int id1)) return id1;
        if (spec.Vocab.TryGetValue("<|begin_of_text|>", out int id2)) return id2;
        return -1;
    }

    private static int FindByteLevelEos(HfTokenizerSpec spec)
    {
        foreach (HfAddedToken t in spec.AddedTokens)
        {
            if (t.Content is "<|endoftext|>" or "<|end_of_text|>" or "<|im_end|>" or "</s>")
                return t.Id;
        }
        if (spec.Vocab.TryGetValue("<|endoftext|>", out int id1)) return id1;
        if (spec.Vocab.TryGetValue("<|im_end|>", out int id2)) return id2;
        if (spec.Vocab.TryGetValue("<|end_of_text|>", out int id3)) return id3;
        return -1;
    }

    // -------------------------------------------------------------------------
    // Shared helpers
    // -------------------------------------------------------------------------

    private static string[] BuildIdToTokenTable(IReadOnlyDictionary<string, int> vocab)
    {
        int maxId = -1;
        foreach (KeyValuePair<string, int> kv in vocab)
            if (kv.Value > maxId) maxId = kv.Value;
        if (maxId < 0) return [];

        string[] tokens = new string[maxId + 1];
        // Empty-string slots are treated as "unused" by the trie builder, which
        // skips them via `string.IsNullOrEmpty` — safe default for gap IDs.
        for (int i = 0; i <= maxId; i++) tokens[i] = string.Empty;
        foreach (KeyValuePair<string, int> kv in vocab)
            tokens[kv.Value] = kv.Key;
        return tokens;
    }

    /// <summary>
    /// Assigns each vocabulary entry a score for the merge priority queue.
    /// Tokens produced by a declared merge get <c>-rank</c> (earlier =
    /// higher priority). Tokens that are never the merge result (single
    /// characters, byte-fallback entries, special tokens) get
    /// <see cref="NonMergeBaseScore"/> so they are still present in the
    /// trie but never re-win a bigram contest.
    /// </summary>
    private static float[] BuildScoreTable(
        string[] tokens,
        IReadOnlyDictionary<string, int> vocab,
        IReadOnlyList<(string Left, string Right)> merges)
    {
        float[] scores = new float[tokens.Length];
        for (int i = 0; i < scores.Length; i++) scores[i] = NonMergeBaseScore;

        for (int rank = 0; rank < merges.Count; rank++)
        {
            (string left, string right) = merges[rank];
            string merged = left + right;
            if (vocab.TryGetValue(merged, out int id) && (uint)id < (uint)scores.Length)
            {
                // Using -rank keeps the merge queue behaviour identical to HF:
                // earliest-declared merge (smallest rank) wins ties.
                scores[id] = -rank;
            }
        }
        return scores;
    }

    /// <summary>
    /// Produces the token-type array consumed by <see cref="BpeTokenizer"/>:
    /// <list type="bullet">
    ///   <item><description>Special added tokens → type 3 (control) — enables special-token pre-splitting.</description></item>
    ///   <item><description>Byte-fallback entries (<c>&lt;0xNN&gt;</c>) → type 6 (byte).</description></item>
    ///   <item><description>Everything else → type 1 (normal).</description></item>
    /// </list>
    /// </summary>
    private static int[] BuildTokenTypes(string[] tokens, IReadOnlyList<HfAddedToken> added)
    {
        int[] types = new int[tokens.Length];
        for (int i = 0; i < tokens.Length; i++)
        {
            string tok = tokens[i];
            if (BpeCore.TryParseByteLiteral(tok, out _))
                types[i] = 6; // byte
            else if (!string.IsNullOrEmpty(tok))
                types[i] = 1; // normal
            else
                types[i] = 5; // unused / gap
        }

        foreach (HfAddedToken at in added)
        {
            if ((uint)at.Id >= (uint)types.Length) continue;
            if (!at.Special) continue;
            // Control tokens: pre-split inputs; the BpeTokenizer only treats
            // tokens with length > 1 as split candidates, which is exactly the
            // invariant we want (single-char tokens are BPE-encodable).
            types[at.Id] = 3;
        }
        return types;
    }

    private static int FindAddedTokenId(HfTokenizerSpec spec, string content, int fallback)
    {
        foreach (HfAddedToken at in spec.AddedTokens)
        {
            if (at.Content == content) return at.Id;
        }
        if (spec.Vocab.TryGetValue(content, out int id)) return id;
        return fallback;
    }
}

/// <summary>
/// <see cref="ITokenizer"/> decorator that applies a
/// <see cref="NormalizationForm"/> to input text before encoding. Used by
/// the ByteLevel factory path when the spec declares a Unicode normalizer
/// (Qwen2 uses NFC). Decode is untouched — the inner tokenizer already
/// produces a concatenated UTF-8 byte stream.
/// </summary>
internal sealed class NormalizingTokenizer(ITokenizer inner, NormalizationForm form) : ITokenizer
{
    private readonly ITokenizer _inner = inner ?? throw new ArgumentNullException(nameof(inner));

    public int VocabSize => _inner.VocabSize;
    public int BosTokenId => _inner.BosTokenId;
    public int EosTokenId => _inner.EosTokenId;

    public int[] Encode(string text) =>
        text.Length == 0 || text.IsNormalized(form)
            ? _inner.Encode(text)
            : _inner.Encode(text.Normalize(form));

    public string Decode(ReadOnlySpan<int> tokenIds) => _inner.Decode(tokenIds);
    public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => _inner.Decode(tokenIds, stripBosSpace);
    public string DecodeToken(int tokenId) => _inner.DecodeToken(tokenId);
    public int CountTokens(string text) => Encode(text).Length;
}
