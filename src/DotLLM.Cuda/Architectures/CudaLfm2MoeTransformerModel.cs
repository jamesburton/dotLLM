using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// CUDA LFM2 / LFM2.5 MoE host: gated short-convolution or QK-normalized GQA mixers,
/// leading dense SwiGLU FFNs, and sigmoid-with-selection-bias routed experts.
/// Routed weights stay packed on device; existing CUDA MoE kernels dequantize selected experts only.
/// </summary>
public sealed unsafe class CudaLfm2MoeTransformerModel : IModel
{
    private readonly CudaContext _context;
    private readonly CudaStream _stream;
    private readonly CudaCublasHandle _cublas;
    private readonly CudaKernels _kernels;
    private readonly CudaLfm2Kernels _lfm;
    private readonly Lfm2MoeTransformerModel _cpu;
    private readonly List<nint> _weights = [];
    private readonly Dictionary<string, (nint Ptr, long Bytes)> _scratch = [];
    private readonly CudaMoeScratch _moeScratch = new();
    private readonly Action<nint, nint, nint, int, int, int, nint> _projectQ4;
    private readonly Layer?[] _layers;
    private Proj _embed, _output;
    private nint _outputNorm;
    private nint _embedLookup;
    private QuantizationType _embedLookupQt;
    private bool _disposed;
    private IKvCache? _cacheOwner;
    private int _allRowLogitsLimit;
    private long _weightBytes;

    private readonly record struct Proj(nint Ptr, QuantizationType Qt, int In, int Out);
    private sealed class Layer
    {
        public nint OperatorNorm, FfnNorm, ConvWeight, ConvState, QNorm, KNorm, SelectionBias;
        public Proj In, Out, Q, K, V, O, Gate, Up, Down;
        public CudaMoeLayerWeights? Moe;
        public int KvHeads;
    }

    private CudaLfm2MoeTransformerModel(ModelConfig config, CudaContext context, CudaStream stream,
        CudaCublasHandle cublas, CudaKernels kernels, CudaLfm2Kernels lfm, Lfm2MoeTransformerModel cpu)
    {
        Config = config; _context = context; _stream = stream; _cublas = cublas;
        _kernels = kernels; _lfm = lfm; _cpu = cpu; _layers = new Layer[config.NumLayers];
        _projectQ4 = ProjectQ4;
    }
    /// <inheritdoc/>
    public ModelConfig Config { get; }
    /// <inheritdoc/>
    public long ComputeMemoryBytes => _scratch.Values.Sum(b => b.Bytes) + _moeScratch.AllocatedBytes;
    /// <summary>Bytes allocated for resident weights and recurrent convolution state.</summary>
    public long DeviceWeightBytes => _weightBytes;
    /// <inheritdoc/>
    public bool RequiresPerSequenceState => true;
    /// <summary>Number of attention layers in the sparse cache.</summary>
    public int AttentionLayerCount => _cpu.AttentionLayerCount;
    /// <summary>Creates a capacity handle for this model's private GPU KV buffers.</summary>
    public CudaHybridKvCacheHandle CreateKvCache(int maxSeqLen) => new(maxSeqLen);
    /// <summary>Rejects KV storage settings the model-owned F32 cache cannot honor.</summary>
    public static void ValidateKvStorage(KvCacheDType key, KvCacheDType value, int cacheWindow = 0)
    {
        if (key != KvCacheDType.F32 || value != KvCacheDType.F32 || cacheWindow != 0)
            throw new NotSupportedException("CUDA LFM2 stores full-precision F32 KV only; quantized KV and mixed-precision cache windows are not supported.");
    }
    /// <summary>Loads an LFM2 MoE GGUF on a CUDA device. The GGUF must outlive the returned model.</summary>
    public static CudaLfm2MoeTransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config, int deviceId = 0, string? ptxDir = null)
    {
        ArgumentNullException.ThrowIfNull(gguf); ArgumentNullException.ThrowIfNull(config);
        if (config.Architecture != Architecture.Lfm2Moe || config.HybridLayout is null || config.Moe is null || config.ShortConvCacheLength < 2)
            throw new ArgumentException("CUDA LFM2 requires an Lfm2Moe hybrid configuration.", nameof(config));
        if (!config.Moe.SigmoidGating || !config.Moe.NormalizeExpertWeights || config.Moe.ExpertWeightsScale != 1f || config.Moe.SharedExpertIntermediateSize is > 0)
            throw new NotSupportedException("CUDA LFM2 requires normalized sigmoid routing without shared experts.");
        ptxDir = CudaKernels.ResolveAndValidatePtxDirectory(ptxDir);
        if (!File.Exists(Path.Combine(ptxDir, "lfm2.ptx"))) throw new FileNotFoundException("CUDA LFM2 requires native/ptx/lfm2.ptx.");
        CudaContext? context = null; CudaStream? stream = null; CudaCublasHandle? cublas = null;
        CudaKernels? kernels = null; CudaLfm2Kernels? lfm = null; Lfm2MoeTransformerModel? cpu = null;
        CudaLfm2MoeTransformerModel? model = null;
        try
        {
            context = CudaContext.Create(deviceId); stream = CudaStream.Create();
            cublas = CudaCublasHandle.Create(); cublas.SetStream(stream);
            kernels = new CudaKernels(ptxDir); lfm = new CudaLfm2Kernels(ptxDir);
            cpu = Lfm2MoeTransformerModel.LoadFromGguf(gguf, config);
            model = new(config, context, stream, cublas, kernels, lfm, cpu);
            model.Upload(); return model;
        }
        catch
        {
            if (model is not null) model.Dispose();
            else { cpu?.Dispose(); lfm?.Dispose(); kernels?.Dispose(); cublas?.Dispose(); stream?.Dispose(); context?.Dispose(); }
            throw;
        }
    }
    private nint UploadBytes(nint source, long bytes)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint ptr, (nuint)bytes).ThrowOnError();
        _weights.Add(ptr); _weightBytes += bytes;
        CudaDriverApi.cuMemcpyHtoD_v2(ptr, source, (nuint)bytes).ThrowOnError(); return ptr;
    }
    private nint UploadFloats(ReadOnlySpan<float> values)
    {
        fixed (float* p = values) return UploadBytes((nint)p, (long)values.Length * 4);
    }
    private Proj UploadProj(nint ptr, QuantizationType qt, int input, int output)
    {
        if (qt == QuantizationType.Q4_0 && input % 32 != 0) throw new InvalidDataException("CUDA Q4_0 rows require a multiple-of-32 input dimension.");
        return new(UploadBytes(ptr, Dequantize.RowByteSize(input, qt) * output), qt, input, output);
    }
    private Proj UploadProj(Lfm2MoeTransformerModel.Lfm2Proj p) => UploadProj(p.Ptr, p.Qt, p.InDim, p.OutDim);
    private void Upload()
    {
        var emb = _cpu.TokenEmbed; var head = _cpu.Output;
        _embed = UploadProj(emb.Ptr, emb.Qt, Config.HiddenSize, Config.VocabSize);
        _output = head.Ptr == emb.Ptr ? _embed : UploadProj(head.Ptr, head.Qt, head.InDim, head.OutDim);
        _outputNorm = UploadFloats(_cpu.OutputNormWeight);
        _embedLookup = _embed.Ptr; _embedLookupQt = _embed.Qt;
        if (_embed.Qt is not (QuantizationType.F32 or QuantizationType.F16 or QuantizationType.Q8_0 or QuantizationType.Q4_0))
        {
            long count = (long)Config.VocabSize * Config.HiddenSize;
            CudaDriverApi.cuMemAlloc_v2(out _embedLookup, (nuint)(count * 2)).ThrowOnError();
            _weights.Add(_embedLookup); _weightBytes += count * 2; _embedLookupQt = QuantizationType.F16;
            _kernels.LaunchDequantToF16(_embed.Ptr, _embed.Qt, _embedLookup, checked((int)count), _stream.Handle);
        }
        for (int i = 0; i < _layers.Length; i++)
        {
            var src = _cpu.LayerWeights[i]; var dst = new Layer(); _layers[i] = dst;
            dst.OperatorNorm = UploadFloats(src.OperatorNorm); dst.FfnNorm = UploadFloats(src.FfnNorm);
            if (src.ConvWeightT is not null)
            {
                dst.In = UploadProj(src.InProj); dst.Out = UploadProj(src.OutProj); dst.ConvWeight = UploadFloats(src.ConvWeightT);
                long bytes = (long)(Config.ShortConvCacheLength - 1) * Config.HiddenSize * 4;
                CudaDriverApi.cuMemAlloc_v2(out dst.ConvState, (nuint)bytes).ThrowOnError(); _weights.Add(dst.ConvState); _weightBytes += bytes;
                CudaDriverApi.cuMemsetD8_v2(dst.ConvState, 0, (nuint)bytes).ThrowOnError();
            }
            else
            {
                var a = src.Attn!; dst.KvHeads = a.NumKvHeads;
                dst.Q = UploadProj(a.QWeight, a.QQuantType, a.QInputDim, a.QOutputDim);
                dst.K = UploadProj(a.KWeight, a.KQuantType, a.KInputDim, a.KOutputDim);
                dst.V = UploadProj(a.VWeight, a.VQuantType, a.VInputDim, a.VOutputDim);
                dst.O = UploadProj(a.OWeight, a.OQuantType, a.OInputDim, a.OOutputDim);
                dst.QNorm = UploadFloats(a.QNormWeight!); dst.KNorm = UploadFloats(a.KNormWeight!);
            }
            if (src.Moe is { } moe)
            {
                int e = moe.NumExperts, h = Config.HiddenSize, f = moe.IntermediateSize;
                nint[] Bank(nint p, QuantizationType qt, int input, int output)
                {
                    var bank = UploadProj(p, qt, input, output * e); long stride = Dequantize.RowByteSize(input, qt) * output;
                    return Enumerable.Range(0, e).Select(j => bank.Ptr + (nint)(j * stride)).ToArray();
                }
                bool allF32 = moe.GateExpsRawQt == QuantizationType.F32 && moe.UpExpsRawQt == QuantizationType.F32 && moe.DownExpsRawQt == QuantizationType.F32;
                dst.Moe = new(e, moe.NumExpertsPerTok, h, f, true, UploadFloats(moe.Gate),
                    Bank(moe.GateExpsRaw, moe.GateExpsRawQt, h, f), Bank(moe.UpExpsRaw, moe.UpExpsRawQt, h, f), Bank(moe.DownExpsRaw, moe.DownExpsRawQt, f, h),
                    0, 0, null, null, null, 0, allF32 ? MoePrecision.F32 : MoePrecision.Quantized,
                    moe.GateExpsRawQt, moe.UpExpsRawQt, moe.DownExpsRawQt, QuantizationType.F32, QuantizationType.F32, QuantizationType.F32);
                if (moe.SelectionBias is not null) dst.SelectionBias = UploadFloats(moe.SelectionBias);
            }
            else { dst.Gate = UploadProj(src.DenseGate); dst.Up = UploadProj(src.DenseUp); dst.Down = UploadProj(src.DenseDown); }
        }
        _stream.Synchronize();
    }
    private nint Scratch(string key, long bytes)
    {
        bytes = Math.Max(bytes, 4);
        if (_scratch.TryGetValue(key, out var old))
        { if (old.Bytes >= bytes) return old.Ptr; _scratch.Remove(key); CudaDriverApi.cuMemFree_v2(old.Ptr).ThrowOnError(); }
        CudaDriverApi.cuMemAlloc_v2(out nint ptr, (nuint)bytes).ThrowOnError(); _scratch[key] = (ptr, bytes); return ptr;
    }
    private void Copy(nint dst, nint src, long bytes) => CudaDriverApi.cuMemcpyDtoDAsync_v2(dst, src, (nuint)bytes, _stream.Handle).ThrowOnError();
    private void Project(Proj w, nint input, nint output, int tokens)
    {
        if (w.Qt == QuantizationType.F32) { CudaGemm.LinearF32(_cublas.Handle, input, w.Ptr, output, tokens, w.In, w.Out, _stream.Handle); return; }
        if (w.Qt == QuantizationType.Q4_0)
        { ProjectQ4(w.Ptr, input, output, w.Out, w.In, tokens, _stream.Handle); return; }
        if (tokens == 1 && w.Qt == QuantizationType.Q8_0)
        { _kernels.LaunchQuantizedGemvF32In(w.Ptr, input, output, w.Out, w.In, _stream.Handle); return; }
        nint weight16 = Scratch("weight16", (long)w.In * w.Out * 2);
        nint input16 = Scratch("input16", (long)tokens * w.In * 2), output16 = Scratch("output16", (long)tokens * w.Out * 2);
        _kernels.LaunchDequantToF16(w.Ptr, w.Qt, weight16, checked(w.In * w.Out), _stream.Handle);
        _kernels.LaunchConvertF32ToF16(input, input16, checked(tokens * w.In), _stream.Handle);
        CudaGemm.LinearF16(_cublas.Handle, input16, weight16, output16, tokens, w.In, w.Out, _stream.Handle);
        _kernels.LaunchConvertF16ToF32(output16, output, checked(tokens * w.Out), _stream.Handle);
    }
    private void ProjectQ4(nint weight, nint input, nint output, int rows, int cols, int tokens, nint stream)
    {
        int blocks = checked(tokens * (cols / 32));
        nint quant = Scratch("q4_activation_q8_1", (long)blocks * 36);
        _lfm.QuantizeQ8(input, quant, blocks, stream);
        _lfm.GemmQ4Q8(weight, quant, output, rows, cols, tokens, stream);
    }
    /// <inheritdoc/>
    public void ResetSequenceState()
    {
        ObjectDisposedException.ThrowIf(_disposed, this); _context.MakeCurrent(); _stream.Synchronize();
        foreach (var l in _layers) if (l!.ConvState != 0)
            CudaDriverApi.cuMemsetD8_v2(l.ConvState, 0, (nuint)((long)(Config.ShortConvCacheLength - 1) * Config.HiddenSize * 4)).ThrowOnError();
    }
    /// <inheritdoc/>
    public bool TrySetAllRowLogitsLimit(int maxSeqLen) { _allRowLogitsLimit = Math.Max(_allRowLogitsLimit, maxSeqLen); return true; }
    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId) => Forward(tokenIds, positions, deviceId, null);
    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
        => ForwardCore(tokenIds, positions, kvCache, tokenIds.Length > _allRowLogitsLimit);
    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache, bool lastTokenLogitsOnly)
        => ForwardCore(tokenIds, positions, kvCache, lastTokenLogitsOnly);
    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache, ILoraAdapter? adapter)
    {
        if (adapter is not null) throw new NotSupportedException("CUDA LFM2 LoRA is not supported.");
        return Forward(tokenIds, positions, deviceId, kvCache);
    }
    /// <inheritdoc/>
    public IReadOnlyList<ITensor> ForwardBatch(IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
    {
        if (requests.Count > 1 || requests.Any(r => r.SsmState is not null))
            throw new NotSupportedException("CUDA LFM2 currently supports one model-owned recurrent sequence at a time.");
        return requests.Select(r => Forward(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache, r.Adapter)).ToArray();
    }
    private ITensor ForwardCore(ReadOnlySpan<int> ids, ReadOnlySpan<int> positions, IKvCache? cache, bool lastOnly)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int t = ids.Length, h = Config.HiddenSize;
        if (t == 0 || t != positions.Length) throw new ArgumentException("Token IDs and positions must have equal, nonzero lengths.");
        for (int i = 0; i < t; i++)
        {
            if ((uint)ids[i] >= (uint)Config.VocabSize) throw new ArgumentOutOfRangeException(nameof(ids));
            if ((uint)positions[i] >= (uint)Config.MaxSequenceLength || (i > 0 && positions[i] != positions[i - 1] + 1)) throw new ArgumentOutOfRangeException(nameof(positions));
        }
        if (cache is not null && cache is not CudaHybridKvCacheHandle) throw new ArgumentException("CUDA LFM2 requires its own cache handle.", nameof(cache));
        if (cache is not null && (positions[0] != cache.CurrentLength || positions[^1] >= cache.MaxLength))
            throw new ArgumentException("CUDA LFM2 cache forwards must append contiguous positions within capacity.", nameof(positions));
        if (!ReferenceEquals(cache, _cacheOwner) && cache is { CurrentLength: > 0 }) throw new NotSupportedException("CUDA LFM2 cannot resume another handle's overwritten KV storage.");
        _context.MakeCurrent(); _cacheOwner = cache;
        nint hidden = Scratch("hidden", (long)t * h * 4), residual = Scratch("residual", (long)t * h * 4), norm = Scratch("norm", (long)t * h * 4);
        nint proj = Scratch("proj", (long)t * 3 * h * 4), mix = Scratch("mix", (long)t * Math.Max(h, Config.NumAttentionHeads * Config.HeadDim) * 4);
        nint gate = Scratch("gate", (long)t * Config.IntermediateSize * 4), up = Scratch("up", (long)t * Config.IntermediateSize * 4);
        nint idDev = Scratch("ids", (long)t * 4), posDev = Scratch("positions", (long)t * 4);
        fixed (int* p = ids) CudaDriverApi.cuMemcpyHtoD_v2(idDev, (nint)p, (nuint)(t * 4)).ThrowOnError();
        fixed (int* p = positions) CudaDriverApi.cuMemcpyHtoD_v2(posDev, (nint)p, (nuint)(t * 4)).ThrowOnError();
        if (_embedLookupQt == QuantizationType.Q4_0) _lfm.EmbeddingQ4(_embedLookup, idDev, hidden, h, t, _stream.Handle);
        else _kernels.LaunchEmbeddingLookupF32(_embedLookup, _embedLookupQt, idDev, hidden, t, h, _stream.Handle);
        long hb = (long)t * h * 4; float eps = Config.NormEpsilon;
        for (int i = 0; i < _layers.Length; i++)
        {
            var l = _layers[i]!; Copy(residual, hidden, hb);
            _kernels.LaunchRmsNormF32(hidden, l.OperatorNorm, norm, h, eps, t, _stream.Handle);
            if (l.ConvState != 0)
            {
                Project(l.In, norm, proj, t); _lfm.Conv(l.ConvState, proj, l.ConvWeight, mix, Config.ShortConvCacheLength, h, t, _stream.Handle); Project(l.Out, mix, norm, t);
            }
            else Attention(l, i, t, positions, posDev, norm, mix, cache);
            _kernels.LaunchAddF32(residual, norm, hidden, t * h, _stream.Handle);
            Copy(residual, hidden, hb); _kernels.LaunchRmsNormF32(hidden, l.FfnNorm, norm, h, eps, t, _stream.Handle);
            if (l.Moe is { } moe)
            {
                _moeScratch.EnsureCapacity(t, moe);
                CudaGemm.LinearF32(_cublas.Handle, norm, moe.Router, _moeScratch.Logits, t, h, moe.NumExperts, _stream.Handle);
                _lfm.Route(_moeScratch.Logits, l.SelectionBias, _moeScratch.TopkIdx, _moeScratch.TopkWeight, moe.NumExperts, moe.NumExpertsPerTok, t, _stream.Handle);
                CudaMoeFfn.ForwardWithPrecomputedRouting(norm, mix, t, moe, _moeScratch, _cublas.Handle, _kernels, _stream.Handle, _projectQ4);
                Copy(norm, mix, hb);
            }
            else
            {
                Project(l.Gate, norm, gate, t); Project(l.Up, norm, up, t); _kernels.LaunchSwiGLUF32(gate, up, gate, l.Gate.Out, t, _stream.Handle); Project(l.Down, gate, norm, t);
            }
            _kernels.LaunchAddF32(residual, norm, hidden, t * h, _stream.Handle);
        }
        int rows = lastOnly ? 1 : t; nint headIn = hidden + (nint)((long)(t - rows) * h * 4);
        _kernels.LaunchRmsNormF32(headIn, _outputNorm, headIn, h, eps, rows, _stream.Handle);
        nint logits = Scratch("logits", (long)rows * Config.VocabSize * 4); Project(_output, headIn, logits, rows); _stream.Synchronize();
        var result = UnmanagedTensor.Allocate(new TensorShape(rows, Config.VocabSize), DType.Float32, -1);
        try { CudaDriverApi.cuMemcpyDtoH_v2(result.DataPointer, logits, (nuint)((long)rows * Config.VocabSize * 4)).ThrowOnError(); }
        catch { result.Dispose(); throw; }
        if (cache is CudaHybridKvCacheHandle handle) handle.Advance(positions[^1] + 1);
        return result;
    }
    private void Attention(Layer l, int index, int tokens, ReadOnlySpan<int> positions, nint posDev, nint norm, nint output, IKvCache? cache)
    {
        int hd = Config.HeadDim, nh = Config.NumAttentionHeads, nk = l.KvHeads, qElems = nh * hd, kvElems = nk * hd;
        nint q = Scratch("q", (long)tokens * qElems * 4), k = Scratch("k", (long)tokens * kvElems * 4), v = Scratch("v", (long)tokens * kvElems * 4);
        Project(l.Q, norm, q, tokens); Project(l.K, norm, k, tokens); Project(l.V, norm, v, tokens);
        _kernels.LaunchRmsNormF32(q, l.QNorm, q, hd, Config.NormEpsilon, tokens * nh, _stream.Handle);
        _kernels.LaunchRmsNormF32(k, l.KNorm, k, hd, Config.NormEpsilon, tokens * nk, _stream.Handle);
        _kernels.LaunchRoPEF32(q, k, posDev, tokens, nh, nk, hd, Config.RoPEConfig?.DimensionCount ?? hd, Config.RoPEConfig?.Theta ?? 10000f, 1, _stream.Handle);
        int seqKv = tokens, offset = 0;
        if (cache is not null)
        {
            nint ck = Scratch($"cache_k_{index}", (long)cache.MaxLength * kvElems * 4), cv = Scratch($"cache_v_{index}", (long)cache.MaxLength * kvElems * 4);
            long byteOffset = (long)positions[0] * kvElems * 4;
            Copy(ck + (nint)byteOffset, k, (long)tokens * kvElems * 4); Copy(cv + (nint)byteOffset, v, (long)tokens * kvElems * 4);
            k = ck; v = cv; seqKv = positions[^1] + 1; offset = positions[0];
        }
        _kernels.LaunchAttentionF32(q, k, v, output, tokens, seqKv, nh, nk, hd, offset, 0, _stream.Handle);
        Project(l.O, output, norm, tokens);
    }
    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return; _disposed = true;
        // Cleanup must still release the context after an asynchronous kernel error.
        CudaDriverApi.cuCtxSetCurrent(_context.Handle);
        CudaDriverApi.cuStreamSynchronize(_stream.Handle);
        _moeScratch.Dispose();
        foreach (var buffer in _scratch.Values) CudaDriverApi.cuMemFree_v2(buffer.Ptr);
        foreach (nint ptr in _weights) CudaDriverApi.cuMemFree_v2(ptr);
        _scratch.Clear(); _weights.Clear(); _cpu.Dispose(); _lfm.Dispose(); _kernels.Dispose(); _cublas.Dispose(); _stream.Dispose(); _context.Dispose();
    }
}
