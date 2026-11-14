using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;

namespace DotLLM.Models.SafeTensors;

/// <summary>
/// Validated AutoAWQ GEMM4 / legacy AutoGPTQ4 tensor groups. Owns the
/// underlying source, including all shards; weights and F16 scales stay mapped.
/// Dispose the model before disposing its returned source, as for plain SafeTensors.
/// </summary>
public sealed class PackedQuantizedSafetensorsSource : ISafetensorsTensorSource
{
    private readonly ISafetensorsTensorSource _source;
    private readonly Dictionary<string, Packed4Weight> _weights;
    private readonly HashSet<string> _zeroBiases;
    private bool _disposed;

    private PackedQuantizedSafetensorsSource(ISafetensorsTensorSource source,
        Dictionary<string, Packed4Weight> weights, QuantizationType format, HashSet<string> zeroBiases)
    {
        _source = source;
        _weights = weights;
        _zeroBiases = zeroBiases;
        Format = format;
        PackedBytes = weights.Values.Sum(w => w.MappedBytes);
        PackedParameterCount = weights.Values.Sum(w => (long)w.InputFeatures * w.OutputFeatures);
        LogicalParameterCount = checked(PackedParameterCount + source.Tensors
            .Where(t => !IsPackedMember(t.Name) && !t.Name.EndsWith(".scales", StringComparison.Ordinal)
                && !zeroBiases.Contains(t.Name))
            .Sum(t => t.ElementCount));
    }

    /// <summary>Private CPU descriptor format, not a GGUF quantization block.</summary>
    public QuantizationType Format { get; }
    /// <summary>Original mapped qweight, qzeros, scales and g_idx bytes (no weight copy).</summary>
    public long PackedBytes { get; }
    /// <summary>Logical weight elements in packed projections, excluding scales/zero points/group indices.</summary>
    public long PackedParameterCount { get; }
    /// <summary>Logical parameters in all projections and ordinary tensors; packed auxiliary data and ignored zero-bias padding are excluded.</summary>
    public long LogicalParameterCount { get; }
    /// <summary>Number of validated linear projections.</summary>
    public int ProjectionCount => _weights.Count;
    /// <inheritdoc/>
    public IReadOnlyList<SafetensorsTensorDescriptor> Tensors => _source.Tensors;
    /// <inheritdoc/>
    public IReadOnlyDictionary<string, SafetensorsTensorDescriptor> TensorsByName => _source.TensorsByName;
    /// <inheritdoc/>
    public nint GetTensorPointer(string name)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _source.GetTensorPointer(name);
    }
    /// <inheritdoc/>
    public ReadOnlySpan<byte> GetTensorSpan(string name)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _source.GetTensorSpan(name);
    }

    /// <inheritdoc/>
    public (string Path, long FileOffset) GetTensorFileLocation(string name)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _source.GetTensorFileLocation(name);
    }

    internal bool TryGetWeight(string name, out Packed4Weight weight)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _weights.TryGetValue(name, out weight);
    }

    internal bool IsZeroBias(string name) => _zeroBiases.Contains(name);

    internal static bool HasPackedTensors(ISafetensorsTensorSource source) =>
        source.Tensors.Any(t => IsPackedMember(t.Name));

    private static bool IsPackedMember(string name) =>
        name.EndsWith(".qweight", StringComparison.Ordinal)
        || name.EndsWith(".qzeros", StringComparison.Ordinal)
        || name.EndsWith(".g_idx", StringComparison.Ordinal);

    internal static unsafe ISafetensorsTensorSource WrapIfQuantized(
        ISafetensorsTensorSource source, JsonElement model)
    {
        bool hasPacked = HasPackedTensors(source);
        if (!model.TryGetProperty("quantization_config", out var quant)
            || quant.ValueKind == JsonValueKind.Null)
        {
            if (hasPacked)
                throw new InvalidDataException("Packed qweight/qzeros/g_idx tensors require an explicit quantization_config in config.json; cannot infer AWQ versus GPTQ.");
            return source;
        }
        if (quant.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException("quantization_config must be an object.");
        string? method = OptionalString(quant, "quant_method");
        if (method is not ("awq" or "gptq"))
        {
            if (hasPacked)
                throw new NotSupportedException($"Packed quant_method '{method ?? "missing"}' is unsupported; use AWQ GEMM4 or legacy AutoGPTQ4.");
            // Other existing loaders (e.g. BitNet) own their quantization metadata.
            return source;
        }

        if (RequiredInt(quant, "bits") != 4)
            throw new NotSupportedException("AWQ/GPTQ loading supports exactly 4 bits.");
        int groupSize = RequiredInt(quant, "group_size");
        if (groupSize is not (32 or 64 or 128 or -1))
            throw new NotSupportedException("AWQ/GPTQ group_size must be 32, 64, 128, or -1 (one group per input axis).");
        bool awq = method == "awq";
        bool descAct = false, symmetric = false;
        if (awq)
        {
            if (!string.Equals(OptionalString(quant, "version"), "gemm", StringComparison.OrdinalIgnoreCase)
                || !RequiredBool(quant, "zero_point"))
                throw new NotSupportedException("AWQ loading requires version='gemm', bits=4 and zero_point=true; GEMV, Marlin and symmetric AWQ layouts are unsupported.");
        }
        else
        {
            descAct = RequiredBool(quant, "desc_act");
            symmetric = RequiredBool(quant, "sym");
            string? format = OptionalString(quant, "checkpoint_format");
            if (format is not (null or "gptq")
                || (quant.TryGetProperty("is_marlin_format", out var marlin) && marlin.ValueKind != JsonValueKind.False)
                || (OptionalString(quant, "pack_dtype") is { } dtype && dtype is not ("int32" or "torch.int32")))
                throw new NotSupportedException("GPTQ loading supports the legacy AutoGPTQ int32 layout (checkpoint_format='gptq', stored zero + 1); GPTQ v2, Marlin and other packing are unsupported.");
        }

        ValidateModel(model);
        int hidden = PositiveInt(model, "hidden_size"), intermediate = PositiveInt(model, "intermediate_size");
        int heads = PositiveInt(model, "num_attention_heads");
        int kvHeads = model.TryGetProperty("num_key_value_heads", out _) ? PositiveInt(model, "num_key_value_heads") : heads;
        int layers = PositiveInt(model, "num_hidden_layers");
        if (hidden % heads != 0 || heads % kvHeads != 0)
            throw new InvalidDataException("Packed Llama hidden/head dimensions must divide evenly.");
        int headDim = hidden / heads;
        if (model.TryGetProperty("head_dim", out _) && PositiveInt(model, "head_dim") != headDim)
            throw new NotSupportedException("Packed Llama requires head_dim = hidden_size / num_attention_heads.");

        var expected = new Dictionary<string, (int M, int K)>(StringComparer.Ordinal);
        for (int layer = 0; layer < layers; layer++)
        {
            string prefix = $"model.layers.{layer}.";
            expected.Add(prefix + "self_attn.q_proj", (hidden, hidden));
            expected.Add(prefix + "self_attn.k_proj", (checked(kvHeads * headDim), hidden));
            expected.Add(prefix + "self_attn.v_proj", (checked(kvHeads * headDim), hidden));
            expected.Add(prefix + "self_attn.o_proj", (hidden, hidden));
            expected.Add(prefix + "mlp.gate_proj", (intermediate, hidden));
            expected.Add(prefix + "mlp.up_proj", (intermediate, hidden));
            expected.Add(prefix + "mlp.down_proj", (hidden, intermediate));
        }

        var weights = new Dictionary<string, Packed4Weight>(StringComparer.Ordinal);
        var members = new HashSet<string>(StringComparer.Ordinal);
        var zeroBiases = new HashSet<string>(StringComparer.Ordinal);
        QuantizationType layout = awq ? QuantizationType.Awq4 : QuantizationType.Gptq4;
        foreach (var tensor in source.Tensors)
        {
            if (tensor.Name.StartsWith("model.layers.", StringComparison.Ordinal)
                && ((awq && tensor.Name.EndsWith(".bias", StringComparison.Ordinal))
                    || tensor.Name.Contains(".q_norm.", StringComparison.Ordinal)
                    || tensor.Name.Contains(".k_norm.", StringComparison.Ordinal)
                    || tensor.Name.Contains(".qkv_proj.", StringComparison.Ordinal)
                    || tensor.Name.Contains(".gate_up_proj.", StringComparison.Ordinal)))
                throw new NotSupportedException($"Packed Llama does not support biased, QK-normalized or fused projection '{tensor.Name}'.");
            if (!awq && tensor.Name.EndsWith(".bias", StringComparison.Ordinal))
            {
                string biasPrefix = tensor.Name[..^".bias".Length];
                if (!expected.TryGetValue(biasPrefix, out var biasShape)
                    || !source.TensorsByName.ContainsKey(biasPrefix + ".qweight"))
                    throw new NotSupportedException($"Unsupported GPTQ bias '{tensor.Name}'.");
                Require(source, tensor.Name, SafetensorsDType.F16, biasShape.M);
                var bias = (Half*)source.GetTensorPointer(tensor.Name);
                for (int i = 0; i < biasShape.M; i++)
                    if (bias[i] != (Half)0)
                        throw new NotSupportedException($"GPTQ bias '{tensor.Name}' must be exactly zero; nonzero biases are unsupported.");
                // Some AutoGPTQ exports serialize zero bias buffers for bias-free Llama.
                zeroBiases.Add(tensor.Name);
            }
            if (!tensor.Name.EndsWith(".qweight", StringComparison.Ordinal)) continue;
            string prefix = tensor.Name[..^".qweight".Length];
            if (!expected.TryGetValue(prefix, out var shape))
                throw new NotSupportedException($"Packed projection '{prefix}' is unsupported; only separate dense Llama q/k/v/o and gate/up/down linears are supported. Embeddings and lm_head must remain floating point.");
            if (source.TensorsByName.ContainsKey(prefix + ".weight") || (awq && source.TensorsByName.ContainsKey(prefix + ".bias")))
                throw new InvalidDataException($"Packed projection '{prefix}' cannot also contain a dense weight or bias.");
            int k = shape.K, m = shape.M, group = groupSize == -1 ? k : groupSize;
            if (k % group != 0 || m % 8 != 0 || (!awq && k % 32 != 0))
                throw new NotSupportedException($"Packed projection '{prefix}' shape [{m},{k}] requires whole groups, out_features divisible by 8, and GPTQ in_features divisible by 32.");
            int groups = k / group;
            var qw = Require(source, prefix + ".qweight", SafetensorsDType.I32, awq ? k : k / 8, awq ? m / 8 : m);
            var qz = Require(source, prefix + ".qzeros", SafetensorsDType.I32, groups, m / 8);
            var sc = Require(source, prefix + ".scales", SafetensorsDType.F16, groups, m);
            var scales = (Half*)source.GetTensorPointer(sc.Name);
            for (long i = 0; i < sc.ElementCount; i++)
                if (!Half.IsFinite(scales[i]) || scales[i] <= (Half)0)
                    throw new InvalidDataException($"Packed projection '{prefix}' scales must be finite and positive.");
            int* indices = null;
            long bytes = checked(qw.ByteCount + qz.ByteCount + sc.ByteCount);
            if (!awq)
            {
                var gi = Require(source, prefix + ".g_idx", SafetensorsDType.I32, k);
                indices = (int*)source.GetTensorPointer(gi.Name);
                var counts = new int[groups];
                for (int i = 0; i < k; i++)
                {
                    int g = indices[i];
                    if ((uint)g >= (uint)groups || (!descAct && g != i / group))
                        throw new InvalidDataException($"Packed projection '{prefix}' g_idx is out of range or contradicts desc_act=false.");
                    counts[g]++;
                }
                if (counts.Any(count => count != group))
                    throw new InvalidDataException($"Packed projection '{prefix}' g_idx must assign exactly group_size inputs to each group.");
                if (symmetric)
                {
                    var zeros = (uint*)source.GetTensorPointer(qz.Name);
                    for (long i = 0; i < qz.ElementCount; i++)
                        if (zeros[i] != 0x77777777u)
                            throw new InvalidDataException($"Packed projection '{prefix}' symmetric legacy GPTQ requires zero point 8 (stored nibble 7).");
                }
                bytes = checked(bytes + gi.ByteCount);
                members.Add(gi.Name);
            }
            members.Add(qw.Name); members.Add(qz.Name); members.Add(sc.Name);
            weights.Add(prefix + ".weight", new Packed4Weight
            {
                QWeight = (uint*)source.GetTensorPointer(qw.Name), QZeros = (uint*)source.GetTensorPointer(qz.Name),
                Scales = scales, GroupIndices = indices, InputFeatures = k, OutputFeatures = m,
                GroupSize = group, Layout = layout, MappedBytes = bytes
            });
        }
        if (weights.Count == 0)
            throw new InvalidDataException("AWQ/GPTQ quantization_config declares packed weights but no supported qweight tensors exist.");
        foreach (var tensor in source.Tensors)
            if ((IsPackedMember(tensor.Name) || tensor.Name.EndsWith(".scales", StringComparison.Ordinal)) && !members.Contains(tensor.Name))
                throw new InvalidDataException($"Orphan or unsupported packed component '{tensor.Name}'.");
        return new PackedQuantizedSafetensorsSource(source, weights, layout, zeroBiases);
    }

    private static void ValidateModel(JsonElement model)
    {
        if (OptionalString(model, "model_type") != "llama"
            || (OptionalString(model, "hidden_act") is { } act && act != "silu"))
            throw new NotSupportedException("Packed AWQ/GPTQ inference currently supports dense Llama with SiLU only.");
        foreach (string flag in new[] { "attention_bias", "mlp_bias", "qk_norm", "use_qk_norm" })
            if (model.TryGetProperty(flag, out var value) && value.ValueKind != JsonValueKind.False)
                throw new NotSupportedException($"Packed Llama does not support '{flag}'.");
        foreach (string property in new[] { "num_local_experts", "num_experts", "sliding_window", "partial_rotary_factor" })
            if (model.TryGetProperty(property, out var value) && value.ValueKind != JsonValueKind.Null)
                throw new NotSupportedException($"Packed Llama does not support '{property}'.");
        if (model.TryGetProperty("pretraining_tp", out _) && RequiredInt(model, "pretraining_tp") != 1)
            throw new NotSupportedException("Packed Llama requires pretraining_tp=1.");
    }

    private static SafetensorsTensorDescriptor Require(ISafetensorsTensorSource source, string name,
        SafetensorsDType dtype, params int[] shape)
    {
        if (!source.TensorsByName.TryGetValue(name, out var tensor))
            throw new InvalidDataException($"Packed projection is missing '{name}'.");
        long count = 1;
        foreach (int dim in shape) count = checked(count * dim);
        int size = dtype == SafetensorsDType.F16 ? 2 : 4;
        if (tensor.DType != dtype || !tensor.Shape.SequenceEqual(shape) || tensor.ByteCount != checked(count * size))
            throw new InvalidDataException($"Packed component '{name}' requires {dtype} shape [{string.Join(',', shape)}] and exactly {count * size} bytes.");
        return tensor;
    }
    private static int PositiveInt(JsonElement json, string name)
    {
        int value = RequiredInt(json, name);
        if (value <= 0) throw new InvalidDataException($"'{name}' must be positive.");
        return value;
    }
    private static int RequiredInt(JsonElement json, string name) =>
        json.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out int result)
            ? result : throw new InvalidDataException($"'{name}' must be an explicit integer.");
    private static bool RequiredBool(JsonElement json, string name) =>
        json.TryGetProperty(name, out var value) && value.ValueKind is JsonValueKind.True or JsonValueKind.False
            ? value.GetBoolean() : throw new InvalidDataException($"'{name}' must be an explicit boolean.");
    private static string? OptionalString(JsonElement json, string name)
    {
        if (!json.TryGetProperty(name, out var value) || value.ValueKind == JsonValueKind.Null) return null;
        return value.ValueKind == JsonValueKind.String ? value.GetString()
            : throw new InvalidDataException($"'{name}' must be a string.");
    }
    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _source.Dispose();
    }
}
