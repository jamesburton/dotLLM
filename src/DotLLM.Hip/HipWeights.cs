using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Hip.DirectStorage;
using DotLLM.Hip.Kernels;
using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Hip;

/// <summary>Resident dense weights, read from their original file ranges without host staging.</summary>
internal sealed class HipWeights : IDisposable
{
    internal readonly record struct Matrix(nint Pointer, QuantizationType Type);
    internal readonly record struct Layer(nint AttnNorm, Matrix Q, Matrix K, Matrix V, Matrix O,
        nint FfnNorm, Matrix Gate, Matrix Up, Matrix Down);

    private readonly List<IDisposable> _owned = [];
    internal Matrix Embedding { get; private set; }
    internal nint OutputNorm { get; private set; }
    internal Matrix Output { get; private set; }
    internal Layer[] Layers { get; private set; } = [];
    internal long AllocatedBytes { get; private set; }
    internal long StoredBytes { get; private set; }

    private static Dictionary<string, (int[] Shape, bool Norm, string HfName)> Specifications(ModelConfig config, bool hasOutput)
    {
        int h = config.HiddenSize, q = checked(config.NumAttentionHeads * config.HeadDim);
        int kv = checked(config.NumKvHeads * config.HeadDim), f = config.IntermediateSize;
        var expected = new Dictionary<string, (int[] Shape, bool Norm, string HfName)>(StringComparer.Ordinal)
        {
            ["token_embd.weight"] = ([h, config.VocabSize], false, "model.embed_tokens.weight"),
            ["output_norm.weight"] = ([h], true, "model.norm.weight"),
        };
        if (hasOutput) expected.Add("output.weight", ([h, config.VocabSize], false, "lm_head.weight"));
        for (int i = 0; i < config.NumLayers; i++)
        {
            string prefix = $"blk.{i}.";
            string hf = $"model.layers.{i}.";
            expected.Add(prefix + "attn_norm.weight", ([h], true, hf + "input_layernorm.weight"));
            expected.Add(prefix + "attn_q.weight", ([h, q], false, hf + "self_attn.q_proj.weight"));
            expected.Add(prefix + "attn_k.weight", ([h, kv], false, hf + "self_attn.k_proj.weight"));
            expected.Add(prefix + "attn_v.weight", ([h, kv], false, hf + "self_attn.v_proj.weight"));
            expected.Add(prefix + "attn_output.weight", ([q, h], false, hf + "self_attn.o_proj.weight"));
            expected.Add(prefix + "ffn_norm.weight", ([h], true, hf + "post_attention_layernorm.weight"));
            expected.Add(prefix + "ffn_gate.weight", ([h, f], false, hf + "mlp.gate_proj.weight"));
            expected.Add(prefix + "ffn_up.weight", ([h, f], false, hf + "mlp.up_proj.weight"));
            expected.Add(prefix + "ffn_down.weight", ([f, h], false, hf + "mlp.down_proj.weight"));
        }
        return expected;
    }

    internal static void Validate(GgufFile file, ModelConfig config)
    {
        var expected = Specifications(config, file.TensorsByName.ContainsKey("output.weight"));
        foreach (var (name, spec) in expected)
        {
            if (!file.TensorsByName.TryGetValue(name, out var tensor))
                throw new InvalidDataException($"HIP model is missing tensor '{name}'.");
            if (!tensor.Shape.Dimensions.AsSpan().SequenceEqual(spec.Shape))
                throw new InvalidDataException($"HIP tensor '{name}' has shape {tensor.Shape}; expected [{string.Join(", ", spec.Shape)}].");
            bool supported = spec.Norm
                ? tensor.QuantizationType is QuantizationType.F16 or QuantizationType.F32
                : tensor.QuantizationType is QuantizationType.F16 or QuantizationType.Q4_0 or QuantizationType.Q8_0;
            if (!supported)
                throw new NotSupportedException($"HIP dense inference requires F16/Q4_0/Q8_0 matrices and F16/F32 norms; '{name}' is {tensor.QuantizationType}.");
            if (!spec.Norm && tensor.QuantizationType != QuantizationType.F16 && spec.Shape[0] % 32 != 0)
                throw new NotSupportedException($"HIP packed matrix '{name}' requires an input dimension divisible by 32.");
        }
        foreach (string name in file.TensorsByName.Keys)
            if (!expected.ContainsKey(name))
                throw new NotSupportedException($"HIP dense inference does not implement tensor '{name}'. Biases, extra norms and architecture-specific tensors require a qualified kernel path.");
    }

    internal static void Validate(ISafetensorsTensorSource file, ModelConfig config)
    {
        if (file is PackedQuantizedSafetensorsSource
            || file.TensorsByName.Keys.Any(name => name.EndsWith(".qweight", StringComparison.Ordinal)))
            throw new NotSupportedException("HIP does not support packed AWQ/GPTQ SafeTensors. Use the qualified CPU path for these checkpoints.");
        bool hasOutput = file.TensorsByName.ContainsKey("lm_head.weight");
        if (!hasOutput && !config.TiedEmbeddings)
            throw new InvalidDataException("HIP SafeTensors model requires lm_head.weight unless tie_word_embeddings is true.");
        var expected = Specifications(config, hasOutput).Values.ToDictionary(s => s.HfName, StringComparer.Ordinal);
        foreach (var (name, spec) in expected)
        {
            if (!file.TensorsByName.TryGetValue(name, out var tensor))
                throw new InvalidDataException($"HIP model is missing tensor '{name}'.");
            int[] shape = spec.Norm ? spec.Shape : [spec.Shape[1], spec.Shape[0]];
            if (!tensor.Shape.AsSpan().SequenceEqual(shape))
                throw new InvalidDataException($"HIP tensor '{name}' must have shape [{string.Join(", ", shape)}].");
            if (tensor.DType != SafetensorsDType.F16 && !(spec.Norm && tensor.DType == SafetensorsDType.F32))
                throw new NotSupportedException($"HIP SafeTensors currently requires F16 matrices and F16/F32 norms; '{name}' is {tensor.DType}. BF16 and packed AWQ/GPTQ need dedicated HIP kernels.");
            _ = file.GetTensorFileLocation(name);
        }
        foreach (string name in file.TensorsByName.Keys)
            if (!expected.ContainsKey(name))
                throw new NotSupportedException($"HIP dense SafeTensors does not implement tensor '{name}'.");
    }

    private readonly record struct FileTensor(string Name, string Path, long Offset, long Count, QuantizationType Type);

    private static long PayloadBytes(long elements, QuantizationType type) => type switch
    {
        QuantizationType.F32 => checked(elements * 4),
        QuantizationType.F16 => checked(elements * 2),
        QuantizationType.Q4_0 => checked(elements / 32 * 18),
        QuantizationType.Q8_0 => checked(elements / 32 * 34),
        _ => throw new NotSupportedException($"HIP weight format {type} is unsupported."),
    };

    internal static HipWeights Load(GgufFile file, ModelConfig config, HipContext context,
        HipStream stream, DenseF16Kernels kernels)
    {
        Validate(file, config);
        var tensors = Specifications(config, file.TensorsByName.ContainsKey("output.weight")).Select(pair =>
        {
            var location = file.GetTensorFileLocation(pair.Key);
            long count = pair.Value.Shape.Aggregate(1L, (product, dimension) => checked(product * dimension));
            return new FileTensor(pair.Key, location.Path, location.FileOffset, count,
                file.TensorsByName[pair.Key].QuantizationType);
        });
        return Load(tensors, config, context, stream, kernels);
    }

    internal static HipWeights Load(ISafetensorsTensorSource file, ModelConfig config, HipContext context,
        HipStream stream, DenseF16Kernels kernels)
    {
        Validate(file, config);
        var tensors = Specifications(config, file.TensorsByName.ContainsKey("lm_head.weight")).Select(pair =>
        {
            var location = file.GetTensorFileLocation(pair.Value.HfName);
            var tensor = file.TensorsByName[pair.Value.HfName];
            return new FileTensor(pair.Key, location.Path, location.FileOffset, tensor.ElementCount,
                tensor.DType == SafetensorsDType.F32 ? QuantizationType.F32 : QuantizationType.F16);
        });
        return Load(tensors, config, context, stream, kernels);
    }

    private static HipWeights Load(IEnumerable<FileTensor> tensors, ModelConfig config, HipContext context,
        HipStream stream, DenseF16Kernels kernels)
    {
        var weights = new HipWeights();
        using var store = new HipDirectWeightStore(context);
        try
        {
            var pointers = new Dictionary<string, Matrix>(StringComparer.Ordinal);
            foreach (var tensor in tensors)
            {
                string name = tensor.Name;
                long count = tensor.Count;
                bool narrow = tensor.Type == QuantizationType.F32;
                long bytes = PayloadBytes(count, tensor.Type);
                var imported = store.Load(tensor.Path, tensor.Offset, bytes);
                if (narrow)
                {
                    using (imported)
                    {
                        var converted = new HipDeviceBuffer(context, checked(count * sizeof(ushort)));
                        try
                        {
                            kernels.ConvertF32ToF16(converted.Pointer, imported.Pointer, checked((int)count));
                            stream.Synchronize();
                            pointers.Add(name, new(converted.Pointer, QuantizationType.F16));
                            weights._owned.Add(converted);
                            weights.AllocatedBytes += converted.ByteLength;
                        }
                        catch { converted.Dispose(); throw; }
                    }
                }
                else
                {
                    weights._owned.Add(imported);
                    pointers.Add(name, new(imported.Pointer, tensor.Type));
                    weights.AllocatedBytes += imported.AllocatedBytes;
                }
                weights.StoredBytes += narrow ? checked(count * sizeof(ushort)) : bytes;
            }
            weights.Embedding = pointers["token_embd.weight"];
            weights.OutputNorm = pointers["output_norm.weight"].Pointer;
            weights.Output = pointers.GetValueOrDefault("output.weight", weights.Embedding);
            weights.Layers = new Layer[config.NumLayers];
            for (int i = 0; i < weights.Layers.Length; i++)
            {
                string p = $"blk.{i}.";
                weights.Layers[i] = new(pointers[p + "attn_norm.weight"].Pointer, pointers[p + "attn_q.weight"],
                    pointers[p + "attn_k.weight"], pointers[p + "attn_v.weight"], pointers[p + "attn_output.weight"],
                    pointers[p + "ffn_norm.weight"].Pointer, pointers[p + "ffn_gate.weight"], pointers[p + "ffn_up.weight"], pointers[p + "ffn_down.weight"]);
            }
            return weights;
        }
        catch { weights.Dispose(); throw; }
    }

    public void Dispose()
    {
        foreach (var allocation in _owned) allocation.Dispose();
        _owned.Clear();
        AllocatedBytes = StoredBytes = 0;
    }
}
