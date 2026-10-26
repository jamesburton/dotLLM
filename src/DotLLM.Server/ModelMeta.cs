using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Gguf;
using DotLLM.Server.Models;

namespace DotLLM.Server;

/// <summary>
/// Builds the llama.cpp-shaped <c>meta</c> object for <c>GET /v1/models</c> and the context facts for
/// <c>GET /health</c> (#1289). The idea (publish the post-fit <c>n_ctx</c> next to llama.cpp's fields so
/// clients can read the served window) comes from xhinker/azllm's <c>tools/serve/model_info.hpp</c>
/// (Apache-2.0); no code was copied.
/// </summary>
internal static class ModelMeta
{
    /// <summary>
    /// Builds the <c>meta</c> object for one loaded model, or <c>null</c> when nothing is loaded.
    /// </summary>
    /// <param name="config">The served config: <see cref="ModelConfig.MaxSequenceLength"/> is the window after any fit or <c>--ctx-size</c>.</param>
    /// <param name="gguf">The model's GGUF, for the trained window, vocabulary type and tensor directory. May be null (non-GGUF loads).</param>
    /// <returns>The meta object, or <c>null</c> when <paramref name="config"/> is null.</returns>
    public static ModelMetaDto? Build(ModelConfig? config, GgufFile? gguf)
    {
        if (config is null)
            return null;

        long nParams = 0, size = 0;
        var bytesByType = new Dictionary<QuantizationType, long>();
        if (gguf is not null)
        {
            foreach (var t in gguf.Tensors)
            {
                long elems = t.Shape.ElementCount;
                nParams += elems;
                long bytes;
                try
                {
                    bytes = t.QuantizationType.ComputeByteCount(elems);
                }
                catch (ArgumentOutOfRangeException)
                {
                    continue; // a type the size table does not know costs the byte count, never the endpoint
                }

                size += bytes;
                bytesByType[t.QuantizationType] = bytesByType.GetValueOrDefault(t.QuantizationType) + bytes;
            }
        }

        return new ModelMetaDto
        {
            VocabType = VocabType(gguf),
            NVocab = config.VocabSize,
            NCtx = config.MaxSequenceLength,
            NCtxTrain = TrainedContext(config, gguf),
            NEmbd = config.HiddenSize,
            NParams = nParams,
            Size = size,
            Ftype = Ftype(bytesByType, size, nParams),
        };
    }

    /// <summary>
    /// The checkpoint's trained window (<c>{arch}.context_length</c>), falling back to the served window
    /// when the GGUF does not declare one.
    /// </summary>
    /// <param name="config">The served config.</param>
    /// <param name="gguf">The model's GGUF, or null.</param>
    /// <returns>The trained context length in tokens.</returns>
    public static int TrainedContext(ModelConfig config, GgufFile? gguf)
    {
        string arch = gguf?.Metadata.GetStringOrDefault("general.architecture") ?? "";
        uint train = arch.Length == 0 ? 0 : gguf!.Metadata.GetUInt32OrDefault($"{arch}.context_length");
        return train > 0 ? (int)train : config.MaxSequenceLength;
    }

    /// <summary>
    /// llama.cpp's <c>llama_vocab_type</c> numbering for <c>tokenizer.ggml.model</c>:
    /// none 0, spm 1, bpe 2, wpm 3, ugm 4, rwkv 5, plamo2 6.
    /// </summary>
    /// <param name="gguf">The model's GGUF, or null.</param>
    /// <returns>The vocabulary type id.</returns>
    internal static int VocabType(GgufFile? gguf) =>
        gguf?.Metadata.GetStringOrDefault("tokenizer.ggml.model") switch
        {
            "llama" => 1,
            "gpt2" => 2,
            "bert" => 3,
            "t5" => 4,
            "rwkv" => 5,
            "plamo2" => 6,
            _ => 0,
        };

    /// <summary>
    /// Names the storage type holding the most bytes plus the measured bits per weight over the whole file,
    /// e.g. <c>"Q8_0 - 8.50 bpw"</c>. Measured from the tensor directory rather than <c>general.file_type</c>,
    /// which converters do not always stamp truthfully.
    /// </summary>
    /// <param name="bytesByType">Tensor bytes per storage type.</param>
    /// <param name="size">Total tensor bytes.</param>
    /// <param name="nParams">Total element count.</param>
    /// <returns>The label, or <c>null</c> when there are no tensors.</returns>
    internal static string? Ftype(Dictionary<QuantizationType, long> bytesByType, long size, long nParams)
    {
        if (bytesByType.Count == 0 || nParams == 0)
            return null;

        var dominant = bytesByType.MaxBy(kv => kv.Value).Key;
        return FormattableString.Invariant($"{dominant} - {size * 8.0 / nParams:F2} bpw");
    }
}
