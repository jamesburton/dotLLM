using DotLLM.Tokenizers.Bpe;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Creates a <see cref="BpeTokenizer"/> from the tokenizer metadata embedded in a GGUF file.
/// Reads the <c>tokenizer.ggml.*</c> metadata keys and dispatches to the appropriate
/// tokenizer variant (SentencePiece for <c>"llama"</c>/<c>"mistral"</c>,
/// tiktoken for <c>"gpt2"</c>/<c>"llama3"</c>, SPM-style merge-ranked BPE for
/// <c>"gemma4"</c>).
/// </summary>
public static class GgufBpeTokenizerFactory
{
    /// <summary>
    /// Loads a <see cref="BpeTokenizer"/> from the given GGUF metadata.
    /// </summary>
    /// <param name="metadata">Metadata parsed from a GGUF file.</param>
    /// <returns>A fully configured <see cref="BpeTokenizer"/>.</returns>
    /// <exception cref="KeyNotFoundException">Required metadata key is absent.</exception>
    public static BpeTokenizer Load(GgufMetadata metadata)
    {
        string model = metadata.GetStringOrDefault("tokenizer.ggml.model", "llama");
        string[] tokens = metadata.GetStringArray("tokenizer.ggml.tokens");

        int[]? tokenTypes = metadata.ContainsKey("tokenizer.ggml.token_type")
            ? metadata.GetInt32Array("tokenizer.ggml.token_type")
            : null;

        int bosId = (int)metadata.GetUInt32OrDefault("tokenizer.ggml.bos_token_id", 1u);
        int eosId = (int)metadata.GetUInt32OrDefault("tokenizer.ggml.eos_token_id", 2u);

        return model switch
        {
            "gpt2" or "llama3" => LoadTiktoken(metadata, tokens, tokenTypes, bosId, eosId),
            // gemma-4 GGUFs carry merge-ranked BPE (llama.cpp LLAMA_VOCAB_PRE_TYPE_GEMMA4:
            // spaces escaped to ▁, newline-only pre-split, raw UTF-8). Files without a
            // merge table keep the historical SentencePiece longest-match fallback.
            "gemma4" when metadata.ContainsKey("tokenizer.ggml.merges") =>
                LoadGemma4(metadata, tokens, tokenTypes, bosId, eosId),
            _ => LoadSentencePiece(metadata, tokens, tokenTypes, bosId, eosId),
        };
    }

    private static BpeTokenizer LoadGemma4(
        GgufMetadata metadata, string[] tokens, int[]? tokenTypes, int bosId, int eosId)
    {
        string[] merges = metadata.GetStringArray("tokenizer.ggml.merges");
        return BpeTokenizer.CreateGemma4(tokens, merges, tokenTypes, bosId, eosId);
    }

    private static BpeTokenizer LoadSentencePiece(
        GgufMetadata metadata, string[] tokens, int[]? tokenTypes, int bosId, int eosId)
    {
        float[] scores = metadata.ContainsKey("tokenizer.ggml.scores")
            ? metadata.GetFloat32Array("tokenizer.ggml.scores")
            : new float[tokens.Length];

        // tokenizer.ggml.add_space_prefix: SentencePiece defaults to prepending ▁
        // to text that doesn't start with a space; gemma4 (and some Llama
        // derivatives) explicitly disable it — "The" must encode to the bare 'The'
        // token, not '▁The'. Absent key keeps the historical default (true).
        bool addSpacePrefix = metadata.GetBoolOrDefault("tokenizer.ggml.add_space_prefix", true);

        return BpeTokenizer.CreateSentencePiece(tokens, scores, tokenTypes, bosId, eosId, addSpacePrefix);
    }

    private static BpeTokenizer LoadTiktoken(
        GgufMetadata metadata, string[] tokens, int[]? tokenTypes, int bosId, int eosId)
    {
        string[] merges = metadata.ContainsKey("tokenizer.ggml.merges")
            ? metadata.GetStringArray("tokenizer.ggml.merges")
            : [];

        string preType = metadata.GetStringOrDefault("tokenizer.ggml.pre");
        string? pre = preType.Length > 0 ? preType : null;

        return BpeTokenizer.CreateTiktoken(tokens, merges, tokenTypes, bosId, eosId, pre);
    }
}
