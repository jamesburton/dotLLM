using DotLLM.Tokenizers;
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

    /// <summary>Loads the tokenizer for <paramref name="metadata"/>.</summary>
    public static ITokenizer Load(GgufMetadata metadata)
        => IsWordPiece(metadata) ? LoadWordPiece(metadata) : GgufBpeTokenizerFactory.Load(metadata);

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
