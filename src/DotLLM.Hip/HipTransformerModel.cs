using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Hip.Interop;
using DotLLM.Hip.Kernels;
using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Hip;

/// <summary>Dense Llama inference with F16 activations, directly loaded packed weights and exclusive paged KV caches.</summary>
/// <remarks>Forward calls share scratch and are serialized. Prefill processes tokens sequentially.</remarks>
public sealed unsafe class HipTransformerModel : IModel
{
    private readonly object _gate = new();
    private readonly HipContext _context;
    private readonly HipStream _stream;
    private readonly DenseF16Kernels _kernels;
    private readonly HipWeights _weights;
    private readonly HipDeviceBuffer _scratch;
    private readonly UnmanagedTensor _hostLogits;
    private readonly nint _hidden, _norm, _q, _k, _v, _attention, _projection, _ffnGate, _ffnUp, _logits;
    private bool _disposed;

    /// <inheritdoc/>
    public ModelConfig Config { get; }
    /// <summary>Selected HIP device ordinal.</summary>
    public int DeviceId => _context.DeviceId;
    /// <inheritdoc/>
    public long ComputeMemoryBytes => _disposed ? 0 : _scratch.ByteLength + _hostLogits.ByteCount;
    /// <summary>Resident weight allocation bytes, including DirectStorage heap alignment.</summary>
    public long WeightMemoryBytes => _weights.AllocatedBytes;
    /// <summary>Resident weight payload bytes, excluding allocation padding.</summary>
    public long StoredWeightBytes => _weights.StoredBytes;
    /// <inheritdoc/>
    public int MaxAllRowLogitsLength => 1;
    /// <inheritdoc/>
    public int PreferredPrefillChunkTokens => 16;

    private HipTransformerModel(ModelConfig config, HipContext context, HipStream stream,
        DenseF16Kernels kernels, HipWeights weights, HipDeviceBuffer scratch, UnmanagedTensor hostLogits)
    {
        Config = config;
        _context = context; _stream = stream; _kernels = kernels; _weights = weights;
        _scratch = scratch; _hostLogits = hostLogits;
        nint cursor = scratch.Pointer;
        nint Take(int count) { nint pointer = cursor; cursor += checked(count * sizeof(ushort)); return pointer; }
        int h = config.HiddenSize, q = checked(config.NumAttentionHeads * config.HeadDim), kv = checked(config.NumKvHeads * config.HeadDim);
        _hidden = Take(h); _norm = Take(h); _q = Take(q); _k = Take(kv); _v = Take(kv);
        _attention = Take(q); _projection = Take(h);
        _ffnGate = Take(config.IntermediateSize); _ffnUp = Take(config.IntermediateSize); _logits = Take(config.VocabSize);
    }

    /// <summary>Loads a supported GGUF checkpoint. The caller retains ownership of the file.</summary>
    public static HipTransformerModel LoadFromGguf(GgufFile file, ModelConfig? config = null,
        int deviceId = 0, string? coDirectory = null)
    {
        ArgumentNullException.ThrowIfNull(file);
        config ??= GgufModelConfigExtractor.Extract(file.Metadata);
        ValidateConfig(config);
        HipWeights.Validate(file, config);
        return Load(config, deviceId, coDirectory, (context, stream, kernels) => HipWeights.Load(file, config, context, stream, kernels));
    }

    /// <summary>Loads supported F16 HuggingFace weights directly from their original shards. The caller owns the source.</summary>
    public static HipTransformerModel LoadFromSafetensors(ISafetensorsTensorSource source, ModelConfig config,
        int deviceId = 0, string? coDirectory = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        ValidateConfig(config);
        HipWeights.Validate(source, config);
        return Load(config, deviceId, coDirectory, (context, stream, kernels) => HipWeights.Load(source, config, context, stream, kernels));
    }

    private static HipTransformerModel Load(ModelConfig config, int deviceId, string? coDirectory,
        Func<HipContext, HipStream, DenseF16Kernels, HipWeights> loadWeights)
    {
        HipContext? context = null;
        HipStream? stream = null;
        DenseF16Kernels? kernels = null;
        HipWeights? weights = null;
        HipDeviceBuffer? scratch = null;
        UnmanagedTensor? hostLogits = null;
        try
        {
            context = HipContext.Create(deviceId);
            stream = HipStream.Create();
            kernels = new DenseF16Kernels(context, stream, coDirectory);
            weights = loadWeights(context, stream, kernels);
            long elements = checked(3L * config.HiddenSize + 2L * config.NumAttentionHeads * config.HeadDim
                + 2L * config.NumKvHeads * config.HeadDim + 2L * config.IntermediateSize + config.VocabSize);
            scratch = new HipDeviceBuffer(context, checked(elements * sizeof(ushort)));
            hostLogits = UnmanagedTensor.Allocate(new TensorShape(1, config.VocabSize), DType.Float16);
            if (hostLogits.DataPointer == 0) throw new OutOfMemoryException();
            return new HipTransformerModel(config, context, stream, kernels, weights, scratch, hostLogits);
        }
        catch
        {
            hostLogits?.Dispose(); scratch?.Dispose(); weights?.Dispose(); kernels?.Dispose(); stream?.Dispose(); context?.Dispose();
            throw;
        }
    }

    /// <summary>Checks the implemented architecture boundary before any native allocation.</summary>
    internal static void ValidateConfig(ModelConfig c)
    {
        ArgumentNullException.ThrowIfNull(c);
        if (c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.VocabSize <= 0 || c.NumLayers <= 0
            || c.NumAttentionHeads <= 0 || c.NumKvHeads <= 0 || c.HeadDim <= 0 || c.MaxSequenceLength <= 0)
            throw new InvalidDataException("HIP model dimensions and context length must be positive.");
        if (c.HeadDim > 256 || c.NumAttentionHeads % c.NumKvHeads != 0
            || (long)c.NumAttentionHeads * c.HeadDim != c.HiddenSize)
            throw new NotSupportedException("HIP dense attention requires headDim <= 256, hiddenSize = heads * headDim, and heads divisible by KV heads.");
        if (c.Architecture != Architecture.Llama || c.NormType != NormType.RMSNorm
            || c.ActivationFunction != ActivationFunction.SiLU || c.AttentionType != AttentionType.GQA
            || c.PositionEncodingType != PositionEncodingType.RoPE || c.Moe is not null || c.MlaConfig is not null
            || c.HybridLayout is not null || c.SsmConfig is not null || c.Mamba3Config is not null || c.GdnConfig is not null
            || c.DiffusionConfig is not null || c.PerLayerEmbedding is not null || c.HadamardFold is not null
            || c.NextnPredictLayers != 0 || c.BidirectionalAttention || c.SlidingWindowSize is > 0
            || c.PerLayerSlidingWindow is not null || c.NoRopeLayers is { Count: > 0 }
            || c.AttnLogitSoftcap.HasValue || c.FinalLogitSoftcap.HasValue || c.EmbeddingScale.HasValue
            || c.HasGraniteScalars || c.QueryPreAttnScalar.HasValue || c.QkNormWholeProjection
            || c.AttentionValueScale.HasValue || c.HasDistinctValueHeadDim || c.GlobalHeadDim.HasValue
            || c.NumGlobalKvHeads.HasValue || c.GlobalRoPEConfig.HasValue || c.PartialRotaryFactor.HasValue
            || c.AttnTemperatureScale != 0 || c.AttnTemperatureFloorScale != 0)
            throw new NotSupportedException("HIP currently supports dense Llama with RMSNorm, SiLU, causal GQA and standard/linear RoPE. This checkpoint requires an unimplemented architecture feature.");
        if (!float.IsFinite(c.NormEpsilon) || c.NormEpsilon <= 0) throw new InvalidDataException("Invalid normalization epsilon.");
        RoPEConfig r = c.RoPEConfig ?? new RoPEConfig(10000f);
        int dim = r.DimensionCount == 0 ? c.HeadDim : r.DimensionCount;
        if (dim <= 0 || dim > c.HeadDim || dim % 2 != 0 || r.Type is not (RoPEType.Norm or RoPEType.NeoX)
            || r.ScalingType is not (RoPEScalingType.None or RoPEScalingType.Linear) || r.AttnFactor != 1
            || !float.IsFinite(r.Theta) || r.Theta <= 0 || !float.IsFinite(r.ScalingFactor) || r.ScalingFactor <= 0)
            throw new NotSupportedException("HIP requires even rotary dimensions within the head and standard or linear RoPE scaling.");
    }

    /// <summary>Reserves a logical context without allocating its KV pages.</summary>
    public HipPagedKvCache CreateKvCache(int maxSequenceLength, int pageTokens = 16,
        KvCacheDType keyType = KvCacheDType.F32, KvCacheDType valueType = KvCacheDType.F32)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if ((keyType != KvCacheDType.F32 || valueType != KvCacheDType.F32) && Config.HeadDim % 32 != 0)
                throw new NotSupportedException("HIP quantized KV requires head dimensions divisible by 32.");
            return new HipPagedKvCache(_context, Config.NumLayers, checked(Config.NumKvHeads * Config.HeadDim),
                maxSequenceLength, Config.MaxSequenceLength, pageTokens, keyType, valueType);
        }
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, null, false);
    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
        => Forward(tokenIds, positions, deviceId, kvCache, false);
    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache, ILoraAdapter? adapter)
        => Forward(tokenIds, positions, deviceId, kvCache, adapter, null, false);
    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache,
        ILoraAdapter? adapter, IMtpState? mtpState)
        => Forward(tokenIds, positions, deviceId, kvCache, adapter, mtpState, false);
    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache,
        ILoraAdapter? adapter, IMtpState? mtpState, bool lastTokenLogitsOnly)
    {
        if (adapter is not null || mtpState is not null)
            throw new NotSupportedException("HIP dense inference does not support LoRA adapters or MTP state.");
        return Forward(tokenIds, positions, deviceId, kvCache, lastTokenLogitsOnly);
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache, bool lastTokenLogitsOnly)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (deviceId != DeviceId) throw new ArgumentException($"This HIP model belongs to device {DeviceId}.", nameof(deviceId));
            if (tokenIds.IsEmpty || tokenIds.Length != positions.Length) throw new ArgumentException("HIP forward requires equally sized, nonempty tokens and positions.");
            if (!lastTokenLogitsOnly && tokenIds.Length > 1)
                throw new NotSupportedException("HIP returns one logit row. Request lastTokenLogitsOnly for prefill, or score one token at a time.");
            if (kvCache is not null && (kvCache is not HipPagedKvCache supplied || !ReferenceEquals(supplied.Owner, _context)))
                throw new ArgumentException("Use a KV cache created by this HIP model; host and foreign-model caches are unsupported.", nameof(kvCache));
            using HipPagedKvCache? temporary = kvCache is null ? CreateKvCache(Config.MaxSequenceLength) : null;
            var cache = (HipPagedKvCache?)kvCache ?? temporary!;
            lock (cache.SyncRoot)
            {
                int start = cache.CurrentLength;
                for (int i = 0; i < tokenIds.Length; i++)
                    if ((uint)tokenIds[i] >= (uint)Config.VocabSize || positions[i] != (long)start + i
                        || positions[i] >= cache.MaxLength)
                        throw new ArgumentException("Tokens must be in vocabulary and positions must append contiguously within the cache's context.");
                _context.MakeCurrent();
                try
                {
                    for (int i = 0; i < tokenIds.Length; i++)
                    {
                        cache.EnsureNextPosition(positions[i], _stream);
                        ForwardToken(tokenIds[i], positions[i], cache);
                        cache.CommitPosition();
                    }
                    _kernels.RmsNorm(_norm, _hidden, _weights.OutputNorm, Config.HiddenSize, Config.NormEpsilon);
                    Linear(_logits, _norm, _weights.Output, Config.VocabSize, Config.HiddenSize);
                    _stream.Synchronize();
                    HipDriverApi.hipMemcpyDtoH(_hostLogits.DataPointer, _logits, checked((nuint)Config.VocabSize * sizeof(ushort))).ThrowOnError();
                    var result = UnmanagedTensor.Allocate(new TensorShape(1, Config.VocabSize), DType.Float32);
                    if (result.DataPointer == 0) { result.Dispose(); throw new OutOfMemoryException(); }
                    // The IModel sampling boundary owns one F32 logit row; resident GPU tensors remain F16.
                    var source = new ReadOnlySpan<Half>((void*)_hostLogits.DataPointer, Config.VocabSize);
                    var destination = new Span<float>((void*)result.DataPointer, Config.VocabSize);
                    for (int i = 0; i < source.Length; i++) destination[i] = (float)source[i];
                    return result;
                }
                catch
                {
                    _stream.Synchronize();
                    cache.Rollback(start);
                    throw;
                }
            }
        }
    }

    private void ForwardToken(int token, int position, HipPagedKvCache cache)
    {
        int h = Config.HiddenSize, heads = Config.NumAttentionHeads, kvHeads = Config.NumKvHeads, d = Config.HeadDim;
        int q = checked(heads * d), kv = checked(kvHeads * d), f = Config.IntermediateSize;
        RoPEConfig rope = Config.RoPEConfig ?? new RoPEConfig(10000f);
        int rotary = rope.DimensionCount == 0 ? d : rope.DimensionCount;
        _kernels.Embed(_hidden, _weights.Embedding.Pointer, token, h, _weights.Embedding.Type);
        for (int layer = 0; layer < _weights.Layers.Length; layer++)
        {
            var w = _weights.Layers[layer];
            _kernels.RmsNorm(_norm, _hidden, w.AttnNorm, h, Config.NormEpsilon);
            Linear(_q, _norm, w.Q, q, h);
            Linear(_k, _norm, w.K, kv, h);
            Linear(_v, _norm, w.V, kv, h);
            _kernels.Rope(_q, _k, heads, kvHeads, d, rotary, position, rope.Theta,
                rope.ScalingType == RoPEScalingType.Linear ? 1f / rope.ScalingFactor : 1f, rope.Type == RoPEType.NeoX);
            _kernels.WriteKv(cache.KeyTable(layer), cache.ValueTable(layer), _k, _v, cache.PageTokens,
                kvHeads, d, position, cache.KeyType, cache.ValueType);
            _kernels.Attention(_attention, _q, cache.KeyTable(layer), cache.ValueTable(layer),
                cache.PageTokens, heads, kvHeads, d, position + 1, 1f / MathF.Sqrt(d), cache.KeyType, cache.ValueType);
            Linear(_projection, _attention, w.O, h, q);
            _kernels.Add(_hidden, _hidden, _projection, h);
            _kernels.RmsNorm(_norm, _hidden, w.FfnNorm, h, Config.NormEpsilon);
            Linear(_ffnGate, _norm, w.Gate, f, h);
            Linear(_ffnUp, _norm, w.Up, f, h);
            _kernels.SwiGlu(_ffnGate, _ffnGate, _ffnUp, f);
            Linear(_projection, _ffnGate, w.Down, h, f);
            _kernels.Add(_hidden, _hidden, _projection, h);
        }
    }

    private void Linear(nint output, nint input, HipWeights.Matrix weight, int rows, int columns)
        => _kernels.Linear(output, input, weight.Pointer, rows, columns, weight.Type);

    /// <inheritdoc/>
    public IReadOnlyList<ITensor> ForwardBatch(IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
    {
        ArgumentNullException.ThrowIfNull(requests);
        var results = new ITensor[requests.Count];
        try
        {
            for (int i = 0; i < requests.Count; i++)
            {
                var r = requests[i];
                if (r.GdnState is not null || r.MambaState is not null || r.SsmState is not null)
                    throw new NotSupportedException("HIP dense inference does not support recurrent request state.");
                results[i] = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache, r.Adapter, null, r.LastTokenLogitsOnly);
            }
            return results;
        }
        catch { foreach (var result in results) result?.Dispose(); throw; }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            _stream.Synchronize();
            _hostLogits.Dispose(); _scratch.Dispose(); _weights.Dispose(); _kernels.Dispose(); _stream.Dispose(); _context.Dispose();
            _disposed = true;
        }
    }
}
