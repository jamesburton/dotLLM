namespace DotLLM.Models.Gguf;

/// <summary>
/// Resolves whether a model's vocab prepends a BOS token to a tokenized stream, reproducing
/// llama.cpp's rule: a default taken from the <b>vocab type / pre-tokenizer</b>, which an
/// explicit <c>tokenizer.ggml.add_bos_token</c> key then <b>overrides</b>.
/// </summary>
/// <remarks>
/// <para>
/// Ported from <c>llama.cpp</c>'s <c>llama-vocab.cpp</c>, <c>llama_vocab::impl::load</c> — the
/// per-vocab-type block that seeds <c>add_bos</c>, the <c>tokenizer_pre</c> chain that overrides
/// it for certain BPE pre-types, and the "Handle add_bos, add_eos and add_sep" block that applies
/// the KV last.
/// </para>
/// <para>
/// <b>Reading the KV alone is not sufficient, and failing that way is silent.</b> Llama-3.2 GGUFs
/// carry <b>no</b> <c>add_bos_token</c> key at all (verified with gguf-py: only
/// <c>bos_token_id</c> and <c>eos_token_id</c> are present) and llama.cpp prepends BOS regardless,
/// because <c>pre = "llama-bpe"</c> routes to <c>LLAMA_VOCAB_PRE_TYPE_LLAMA3</c>, which sets
/// <c>add_bos = true</c>. A <c>GetBoolOrDefault(..., false)</c> implementation therefore gets the
/// most common comparison model wrong while looking entirely reasonable. That is issue #515: the
/// perplexity harness scored Llama-3.2 with no BOS while llama.cpp prepended one, so every chunk
/// boundary was displaced by a token and the engines compared different text for two sessions.
/// See <c>docs/PERPLEXITY.md</c>, *The BOS trap*.
/// </para>
/// </remarks>
public static class GgufAddBosResolver
{
    /// <summary>
    /// BPE pre-tokenizer types that llama.cpp routes through
    /// <c>LLAMA_VOCAB_PRE_TYPE_LLAMA3</c>, plus the two other pre-types whose branches set
    /// <c>add_bos = true</c> (<c>tekken</c>, <c>chameleon</c>).
    /// </summary>
    /// <remarks>
    /// The llama3 group duplicates the list in
    /// <c>DotLLM.Tokenizers.Bpe.TiktokenPreTokenizer.GetRegexes</c>, which is internal to that
    /// assembly and keyed to regex pipelines rather than to an enum. Both lists come from the same
    /// <c>llama-vocab.cpp</c> case block; if one gains a pre-type, the other should too.
    /// </remarks>
    private static readonly HashSet<string> BpePreTypesThatAddBos = new(StringComparer.Ordinal)
    {
        "llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1",
        "pixtral", "midm-2.0", "lfm2", "jina-v5-nano",
        "tekken", "chameleon",
    };

    /// <summary>
    /// Returns true when a BOS token should be prepended to a tokenized stream for this vocab.
    /// </summary>
    /// <param name="metadata">Metadata parsed from a GGUF file.</param>
    public static bool Resolve(GgufMetadata metadata)
    {
        string model = metadata.GetStringOrDefault("tokenizer.ggml.model", "llama");
        string pre = metadata.GetStringOrDefault("tokenizer.ggml.pre");

        // Vocab-type seed. llama.cpp's default is false; SPM and WPM set it true, UGM and RWKV
        // leave it false. dotLLM's tokenizer.ggml.model values map: "llama"/"mistral" -> SPM,
        // "bert" -> WPM, "t5" -> UGM, "rwkv" -> RWKV, "gpt2"/"llama3"/"gemma4" -> BPE.
        bool addBos = model switch
        {
            "gpt2" or "llama3" or "gemma4" => BpePreTypesThatAddBos.Contains(pre),
            "bert" => true,
            "t5" or "rwkv" => false,
            // SentencePiece ("llama", "mistral", and the historical default).
            _ => true,
        };

        // An explicit key wins over the pre-type default — this is what makes SmolLM
        // (pre = "smollm", add_bos_token = false) differ from Llama-3.2 (pre = "llama-bpe",
        // no key). Both are tokenizer.ggml.model = "gpt2".
        if (metadata.TryGetValue("tokenizer.ggml.add_bos_token", out _))
            addBos = metadata.GetBool("tokenizer.ggml.add_bos_token");

        // Gemma 4 forces BOS on after the KV (llama.cpp works around GGUFs that say false).
        // ref: https://github.com/ggml-org/llama.cpp/pull/21500
        if (model == "gemma4" || pre == "gemma4")
            addBos = true;

        return addBos;
    }
}
