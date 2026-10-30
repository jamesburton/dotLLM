using DotLLM.Tokenizers;
using DotLLM.Tokenizers.Unigram;
using DotLLM.Tokenizers.WordPiece;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Creates the tokenizer for a GGUF file, dispatching on <c>tokenizer.ggml.model</c>:
/// <c>"bert"</c> yields a <see cref="WordPieceTokenizer"/>, everything else the existing
/// <see cref="GgufBpeTokenizerFactory"/> result.
/// </summary>
public static class GgufTokenizerFactory
{
    /// <summary>True when the GGUF declares a BERT WordPiece vocabulary.</summary>
    public static bool IsWordPiece(GgufMetadata metadata)
        => metadata.GetStringOrDefault("tokenizer.ggml.model", "llama") == "bert";

    /// <summary>True when the GGUF declares a SentencePiece Unigram vocabulary (<c>"t5"</c>, llama.cpp's UGM: XLM-R, T5).</summary>
    public static bool IsUnigram(GgufMetadata metadata)
        => metadata.GetStringOrDefault("tokenizer.ggml.model", "llama") == "t5";

    /// <summary>Loads the tokenizer for <paramref name="metadata"/>.</summary>
    public static ITokenizer Load(GgufMetadata metadata)
        => IsWordPiece(metadata) ? LoadWordPiece(metadata)
         : IsUnigram(metadata) ? LoadUnigram(metadata)
         : GgufBpeTokenizerFactory.Load(metadata);

    /// <summary>Loads a SentencePiece Unigram tokenizer (<c>tokenizer.ggml.model == "t5"</c>).</summary>
    /// <remarks>
    /// Reads <c>scores</c>, <c>token_type</c>, <c>precompiled_charsmap</c>, <c>add_space_prefix</c> and
    /// <c>remove_extra_whitespaces</c> (SentencePiece defaults true). BOS/EOS framing follows
    /// <c>add_bos_token</c>/<c>add_eos_token</c>; for encoder GGUFs these are CLS/SEP, the separator id
    /// (<c>seperator_token_id</c>) taking precedence over <c>eos_token_id</c> as in <see cref="LoadWordPiece"/>.
    /// </remarks>
    public static UnigramTokenizer LoadUnigram(GgufMetadata metadata)
    {
        string[] tokens = metadata.GetStringArray("tokenizer.ggml.tokens");
        if (!metadata.ContainsKey("tokenizer.ggml.scores"))
            throw new InvalidDataException(
                "Unigram GGUF has no tokenizer.ggml.scores; the vocabulary cannot be segmented. "
                + "Re-convert the model with a current llama.cpp convert_hf_to_gguf.py.");
        float[] scores = metadata.GetFloat32Array("tokenizer.ggml.scores");
        int[]? types = metadata.ContainsKey("tokenizer.ggml.token_type") ? metadata.GetInt32Array("tokenizer.ggml.token_type") : null;

        int Id(int fallback, params string[] keys)
        {
            foreach (string k in keys)
                if (metadata.ContainsKey(k)) return (int)metadata.GetUInt32(k);
            return fallback;
        }

        int unk = Id(-1, "tokenizer.ggml.unknown_token_id");
        if (unk < 0) unk = Array.IndexOf(tokens, "<unk>");
        if (unk < 0) throw new InvalidDataException("Unigram GGUF declares no unknown token.");
        int bos = Id(-1, "tokenizer.ggml.bos_token_id", "tokenizer.ggml.cls_token_id");
        int eos = Id(-1, "tokenizer.ggml.seperator_token_id", "tokenizer.ggml.separator_token_id", "tokenizer.ggml.eos_token_id");
        int pad = Id(-1, "tokenizer.ggml.padding_token_id");

        return new UnigramTokenizer(
            tokens, scores, types, unk, bos, eos, pad,
            addBos: metadata.GetBoolOrDefault("tokenizer.ggml.add_bos_token", bos >= 0),
            addEos: metadata.GetBoolOrDefault("tokenizer.ggml.add_eos_token", eos >= 0),
            addSpacePrefix: metadata.GetBoolOrDefault("tokenizer.ggml.add_space_prefix", true),
            removeExtraWhitespaces: metadata.GetBoolOrDefault("tokenizer.ggml.remove_extra_whitespaces", true),
            precompiledCharsMap: metadata.GetByteBlobOrEmpty("tokenizer.ggml.precompiled_charsmap"));
    }

    /// <summary>Loads a WordPiece tokenizer (<c>tokenizer.ggml.model == "bert"</c>).</summary>
    /// <remarks>
    /// llama.cpp's converter stores CLS/SEP as <c>bos</c>/<c>eos</c> (older files) or as
    /// <c>cls_token_id</c>/<c>seperator_token_id</c> (sic); any missing id is resolved by its
    /// <c>[CLS]</c>/<c>[SEP]</c>/<c>[UNK]</c>/<c>[PAD]</c> string.
    /// </remarks>
    public static WordPieceTokenizer LoadWordPiece(GgufMetadata metadata)
    {
        string[] tokens = metadata.GetStringArray("tokenizer.ggml.tokens");

        int Id(params string[] keys)
        {
            foreach (string k in keys)
                if (metadata.ContainsKey(k)) return (int)metadata.GetUInt32(k);
            return -1;
        }

        int cls = Id("tokenizer.ggml.cls_token_id", "tokenizer.ggml.bos_token_id");
        int sep = Id("tokenizer.ggml.seperator_token_id", "tokenizer.ggml.separator_token_id", "tokenizer.ggml.eos_token_id");
        int unk = Id("tokenizer.ggml.unknown_token_id");
        int pad = Id("tokenizer.ggml.padding_token_id");
        return WordPieceTokenizer.FromGgufTokens(tokens, cls, sep, unk, pad);
    }
}
