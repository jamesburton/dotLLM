using DotLLM.Tokenizers.Bpe;

namespace DotLLM.Tokenizers.Hf;

/// <summary>
/// Bridges a parsed HuggingFace <see cref="HfTokenizerSpec"/> into a
/// <see cref="BpeTokenizer"/>. HF <c>tokenizer.json</c> expresses merge order
/// as a ranked list, whereas dotLLM's <see cref="BpeTokenizer"/> drives the
/// merge loop off per-token scores — higher score = higher priority. The
/// factory converts the rank into a synthetic score per merged token:
/// <c>score(merged) = -rank</c>, which reproduces the HF library's
/// earliest-rank-wins merge order deterministically.
/// </summary>
/// <remarks>
/// <para>
/// <b>Scope.</b> Only the SentencePiece-style layout used by Llama 1/2, Mistral,
/// TinyLlama and Mamba-3 ib-ssm is wired today: Metaspace pre-tokenizer,
/// ByteFallback decoder, <c>model.type = "BPE"</c>. Byte-level (GPT-2 style)
/// tokenizers will need a separate bridge to <see cref="BpeTokenizer.CreateTiktoken"/>.
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
    /// Builds a <see cref="BpeTokenizer"/> from a parsed HuggingFace spec.
    /// Pass -1 for <paramref name="bosId"/> / <paramref name="eosId"/> to
    /// let the factory auto-detect <c>&lt;s&gt;</c> / <c>&lt;/s&gt;</c> from
    /// the added-tokens list (falling back to 1 / 2 if not found, which
    /// matches Llama 2 convention).
    /// </summary>
    /// <param name="spec">Parsed tokenizer spec.</param>
    /// <param name="bosId">Override BOS token ID, or -1 to auto-detect.</param>
    /// <param name="eosId">Override EOS token ID, or -1 to auto-detect.</param>
    /// <returns>A <see cref="BpeTokenizer"/> ready for encode/decode calls.</returns>
    /// <exception cref="InvalidDataException">
    /// The spec declares a pre-tokenizer this factory does not yet support
    /// (anything other than <see cref="HfPreTokenizerKind.Metaspace"/> or
    /// <see cref="HfPreTokenizerKind.None"/>).
    /// </exception>
    public static BpeTokenizer Create(HfTokenizerSpec spec, int bosId = -1, int eosId = -1)
    {
        ArgumentNullException.ThrowIfNull(spec);

        if (spec.PreTokenizerKind is not HfPreTokenizerKind.Metaspace and not HfPreTokenizerKind.None)
        {
            throw new InvalidDataException(
                $"HF tokenizer.json pre-tokenizer '{spec.PreTokenizerKind}' is not supported yet. "
                + "Only 'Metaspace' (SentencePiece BPE) is wired in this adapter.");
        }

        string[] tokens = BuildIdToTokenTable(spec.Vocab);
        float[] scores = BuildScoreTable(tokens, spec.Vocab, spec.Merges);
        int[] tokenTypes = BuildTokenTypes(tokens, spec.AddedTokens);

        int resolvedBos = bosId >= 0 ? bosId : FindAddedTokenId(spec, "<s>", fallback: 1);
        int resolvedEos = eosId >= 0 ? eosId : FindAddedTokenId(spec, "</s>", fallback: 2);

        // addBosSpace matches Metaspace's prepend_scheme = "first" (Llama default).
        // "never" would disable it, but no checkpoint we support ships with that.
        bool addBosSpace = !string.Equals(spec.MetaspacePrependScheme, "never", StringComparison.Ordinal);

        return BpeTokenizer.CreateSentencePiece(
            tokens, scores, tokenTypes,
            resolvedBos, resolvedEos, addBosSpace);
    }

    /// <summary>
    /// Convenience entry point — parses the JSON and creates the tokenizer
    /// in one call.
    /// </summary>
    public static BpeTokenizer Create(string jsonContent, int bosId = -1, int eosId = -1)
        => Create(HfTokenizerJsonParser.Parse(jsonContent), bosId, eosId);

    /// <summary>
    /// Loads <c>tokenizer.json</c> from a HuggingFace checkpoint directory
    /// and builds a tokenizer. Returns <see langword="null"/> when
    /// <paramref name="directory"/> has no <c>tokenizer.json</c>.
    /// </summary>
    /// <param name="directory">Directory containing <c>tokenizer.json</c>.</param>
    /// <param name="bosId">Override BOS token ID, or -1 to auto-detect.</param>
    /// <param name="eosId">Override EOS token ID, or -1 to auto-detect.</param>
    /// <returns>A <see cref="BpeTokenizer"/>, or <see langword="null"/> if no file was found.</returns>
    public static BpeTokenizer? TryLoadFromDirectory(string directory, int bosId = -1, int eosId = -1)
    {
        ArgumentNullException.ThrowIfNull(directory);
        string path = Path.Combine(directory, "tokenizer.json");
        if (!File.Exists(path)) return null;
        string content = File.ReadAllText(path);
        return Create(content, bosId, eosId);
    }

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
