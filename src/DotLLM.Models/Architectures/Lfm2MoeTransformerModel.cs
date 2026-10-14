using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using NativeMemory = System.Runtime.InteropServices.NativeMemory;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Liquid AI LFM2 / LFM2.5 MoE (<c>lfm2moe</c>) CPU model. Reference: llama.cpp <c>src/models/lfm2moe.cpp</c> /
/// <c>lfm2.cpp</c> and HF <c>modeling_lfm2_moe.py</c>.
/// </summary>
/// <remarks>
/// Every layer is two residual sub-layers: <c>x += mixer(operator_norm(x))</c>, <c>x += ffn(ffn_norm(x))</c>.
/// The mixer is either a gated short convolution (<c>in_proj -> [B, C, X]; y = out_proj(C * causal_depthwise_conv(B * X))</c>,
/// kernel <c>L_cache</c>, no bias, no activation) or GQA attention with per-head QK RMS-norm and NeoX RoPE. The FFN is
/// a dense SwiGLU for the leading dense layers and a sigmoid + selection-bias, renormalised top-k SwiGLU MoE afterwards.
/// The LM head is tied to the token embedding and the final norm is stored as <c>token_embd_norm.weight</c>.
/// </remarks>
public sealed unsafe class Lfm2MoeTransformerModel : IModel
{
    private readonly GgufFile? _gguf;
    private readonly Lfm2Layer[] _layers;
    private readonly float[] _outputNormWeight;
    private readonly nint _tokenEmbedWeight;
    private readonly QuantizationType _tokenEmbedQuantType;
    private readonly nint _outputWeight;
    private readonly QuantizationType _outputQuantType;
    private readonly int _outputOutputDim;
    private readonly int _outputInputDim;
    private readonly HybridLayerLayout _layout;
    private readonly int[] _kvSlotForLayer;
    private readonly int[] _convOrdinalForLayer;
    private readonly int _attentionLayerCount;
    private readonly int _convLayerCount;
    private readonly int _lCache;
    private readonly float[] _ropeCosTable;
    private readonly float[] _ropeSinTable;
    private readonly int _ropeDim;
    private readonly Lfm2ConvStateCache _defaultState;
    private readonly ComputeThreadPool? _threadPool;
    private readonly bool _ownsThreadPool;
    private readonly Lfm2Scratch _s = new();
    private Lfm2ConvStateCache? _activeState;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _s.AllocatedBytes + _defaultState.AllocatedBytes;

    /// <summary>Number of full-attention layers (the sparse KV-cache slot count).</summary>
    public int AttentionLayerCount => _attentionLayerCount;

    private Lfm2MoeTransformerModel(
        ModelConfig config, GgufFile? gguf, Lfm2Layer[] layers, float[] outputNorm,
        nint embed, QuantizationType embedQt, nint output, QuantizationType outputQt, int outM, int outK,
        int[] kvSlot, int[] convOrdinal, int attnCount, int convCount,
        float[] cos, float[] sin, int ropeDim, ComputeThreadPool? pool, bool ownsPool)
    {
        Config = config;
        _gguf = gguf;
        _layers = layers;
        _outputNormWeight = outputNorm;
        _tokenEmbedWeight = embed;
        _tokenEmbedQuantType = embedQt;
        _outputWeight = output;
        _outputQuantType = outputQt;
        _outputOutputDim = outM;
        _outputInputDim = outK;
        _layout = config.HybridLayout!;
        _kvSlotForLayer = kvSlot;
        _convOrdinalForLayer = convOrdinal;
        _attentionLayerCount = attnCount;
        _convLayerCount = convCount;
        _lCache = config.ShortConvCacheLength;
        _ropeCosTable = cos;
        _ropeSinTable = sin;
        _ropeDim = ropeDim;
        _threadPool = pool;
        _ownsThreadPool = ownsPool;
        _defaultState = new Lfm2ConvStateCache(convCount, _lCache - 1, config.HiddenSize);
    }

    /// <summary>Loads an LFM2-MoE model from an opened GGUF (which must outlive the model).</summary>
    public static Lfm2MoeTransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config)
        => LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);

    /// <summary>Loads an LFM2-MoE model from an opened GGUF with threading configuration.</summary>
    public static Lfm2MoeTransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config, ThreadingConfig threading)
    {
        if (config.Architecture != Architecture.Lfm2Moe)
            throw new ArgumentException($"Lfm2MoeTransformerModel requires Architecture.Lfm2Moe, got {config.Architecture}.", nameof(config));
        if (config.HybridLayout is null || config.Moe is null || config.ShortConvCacheLength < 2)
            throw new ArgumentException("Lfm2Moe config must have HybridLayout, Moe and ShortConvCacheLength populated.", nameof(config));

        var dataBase = new GgufDataBase(gguf);
        var tensors = gguf.TensorsByName;
        var layout = config.HybridLayout;
        int hidden = config.HiddenSize;

        var embDesc = tensors["token_embd.weight"];
        nint embPtr = dataBase.Of(embDesc);
        float[] outNorm = DequantizeF32(dataBase, tensors["token_embd_norm.weight"], hidden);

        nint outPtr; QuantizationType outQt; int outM, outK;
        if (tensors.TryGetValue("output.weight", out var outDesc))
        {
            outPtr = dataBase.Of(outDesc); outQt = outDesc.QuantizationType; outK = outDesc.Shape[0]; outM = outDesc.Shape[1];
        }
        else
        {
            outPtr = embPtr; outQt = embDesc.QuantizationType; outK = embDesc.Shape[0]; outM = embDesc.Shape[1];
        }

        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        var owned = new List<nint>();
        var layers = new Lfm2Layer[config.NumLayers];
        var kvSlot = new int[config.NumLayers];
        var convOrd = new int[config.NumLayers];
        int attnCount = 0, convCount = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            layers[i] = LoadLayer(i, dataBase, tensors, config, owned);
            bool isAttn = layout.LayerKind[i] == HybridLayerKind.Attention;
            kvSlot[i] = isAttn ? attnCount++ : -1;
            convOrd[i] = isAttn ? -1 : convCount++;
        }

        float theta = config.RoPEConfig?.Theta ?? 10000.0f;
        int half = ropeDim / 2;
        var cos = new float[config.MaxSequenceLength * half];
        var sin = new float[config.MaxSequenceLength * half];
        RoPE.PrecomputeFrequencyTable(config.MaxSequenceLength, ropeDim, theta, cos, sin);

        ComputeThreadPool? pool = CreatePool(threading);
        return new Lfm2MoeTransformerModel(config, gguf, layers, outNorm, embPtr, embDesc.QuantizationType,
            outPtr, outQt, outM, outK, kvSlot, convOrd, attnCount, convCount, cos, sin, ropeDim, pool, pool is not null);
    }

    private static ComputeThreadPool? CreatePool(ThreadingConfig threading)
    {
        if (!threading.IsParallel) return null;
        int n = threading.EffectiveThreadCount;
        if (threading.EnableNumaPinning || threading.EnablePCorePinning)
        {
            var topology = NumaTopology.Detect();
            if (threading.EnablePCorePinning && topology.IsHybrid)
                n = Math.Min(n, topology.PerformanceCoreIds.Count);
            return new ComputeThreadPool(n, topology, threading);
        }
        return new ComputeThreadPool(n, topology: null, threading);
    }

    private static Lfm2Layer LoadLayer(int layerIdx, GgufDataBase db, IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
                                       ModelConfig config, List<nint> owned)
    {
        string p = $"blk.{layerIdx}";
        int hidden = config.HiddenSize;
        var layer = new Lfm2Layer
        {
            OperatorNorm = DequantizeF32(db, tensors[$"{p}.attn_norm.weight"], hidden),
            FfnNorm = DequantizeF32(db, tensors[$"{p}.ffn_norm.weight"], hidden),
        };

        if (config.HybridLayout!.LayerKind[layerIdx] == HybridLayerKind.ShortConv)
        {
            var inProj = tensors[$"{p}.shortconv.in_proj.weight"];
            var outProj = tensors[$"{p}.shortconv.out_proj.weight"];
            var conv = tensors[$"{p}.shortconv.conv.weight"];
            int lc = config.ShortConvCacheLength;
            if (inProj.Shape[0] != hidden || inProj.Shape[1] != 3 * hidden)
                throw new InvalidDataException($"{p}.shortconv.in_proj.weight shape [{inProj.Shape[0]},{inProj.Shape[1]}] != [{hidden},{3 * hidden}].");
            if (conv.Shape[0] != lc || conv.Shape[1] != hidden)
                throw new InvalidDataException($"{p}.shortconv.conv.weight shape [{conv.Shape[0]},{conv.Shape[1]}] != [{lc},{hidden}].");
            float[] w = DequantizeF32(db, conv, lc * hidden); // GGUF: ne0 = taps (fastest), ne1 = channels -> w[c * lc + j]
            var wT = new float[lc * hidden];
            for (int c = 0; c < hidden; c++)
                for (int j = 0; j < lc; j++)
                    wT[j * hidden + c] = w[c * lc + j];
            layer.ConvWeightT = wT;
            layer.InProj = Proj(db, inProj);
            layer.OutProj = Proj(db, outProj);
        }
        else
        {
            var q = tensors[$"{p}.attn_q.weight"]; var k = tensors[$"{p}.attn_k.weight"];
            var v = tensors[$"{p}.attn_v.weight"]; var o = tensors[$"{p}.attn_output.weight"];
            layer.Attn = new Qwen3FullAttnWeights
            {
                QWeight = db.Of(q), QQuantType = q.QuantizationType, QInputDim = q.Shape[0], QOutputDim = q.Shape[1],
                KWeight = db.Of(k), KQuantType = k.QuantizationType, KInputDim = k.Shape[0], KOutputDim = k.Shape[1],
                VWeight = db.Of(v), VQuantType = v.QuantizationType, VInputDim = v.Shape[0], VOutputDim = v.Shape[1],
                OWeight = db.Of(o), OQuantType = o.QuantizationType, OInputDim = o.Shape[0], OOutputDim = o.Shape[1],
                NumKvHeads = config.HybridLayout.HeadCountKv[layerIdx],
                QNormWeight = DequantizeF32(db, tensors[$"{p}.attn_q_norm.weight"], config.HeadDim),
                KNormWeight = DequantizeF32(db, tensors[$"{p}.attn_k_norm.weight"], config.HeadDim),
            };
        }

        if (config.Moe!.IsMoeLayer(layerIdx))
        {
            layer.Moe = TransformerWeights.LoadDeepSeekMoeLayer(layerIdx, db, tensors, config, owned, skipRoutedF32Only: true);
        }
        else
        {
            layer.DenseGate = Proj(db, tensors[$"{p}.ffn_gate.weight"]);
            layer.DenseUp = Proj(db, tensors[$"{p}.ffn_up.weight"]);
            layer.DenseDown = Proj(db, tensors[$"{p}.ffn_down.weight"]);
        }
        return layer;
    }

    private static Lfm2Proj Proj(GgufDataBase db, GgufTensorDescriptor d)
        => new(db.Of(d), d.QuantizationType, d.Shape[0], d.Shape[1]);

    private static float[] DequantizeF32(GgufDataBase db, GgufTensorDescriptor desc, int expected)
    {
        nint ptr = db.Of(desc);
        float[] r = new float[expected];
        Dequantize.ToFloat32(ptr, expected, desc.QuantizationType, r);
        return r;
    }

    // ── IModel ───────────────────────────────────────────────────────────────

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => ForwardCore(tokenIds, positions, deviceId, null, null, false);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
        => ForwardCore(tokenIds, positions, deviceId, kvCache, null, false);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                           IKvCache? kvCache, bool lastTokenLogitsOnly)
        => ForwardCore(tokenIds, positions, deviceId, kvCache, null, lastTokenLogitsOnly && tokenIds.Length > 1);

    /// <summary>Forward with a caller-supplied per-sequence conv state (null = model-owned default state).</summary>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                           IKvCache? kvCache, ISsmState? ssmState)
        => ForwardCore(tokenIds, positions, deviceId, kvCache, ResolveState(ssmState), false);

    private Lfm2ConvStateCache? ResolveState(ISsmState? s)
    {
        if (s is null) return null;
        if (s is Lfm2ConvStateCache c && c.NumSsmLayers == _convLayerCount) return c;
        throw new ArgumentException($"Lfm2MoeTransformerModel requires an {nameof(Lfm2ConvStateCache)} with {_convLayerCount} layers; got {s.GetType().Name}.", nameof(s));
    }

    /// <inheritdoc/>
    public void ResetSequenceState() => _defaultState.Reset();

    /// <inheritdoc/>
    public bool RequiresPerSequenceState => true;

    /// <inheritdoc/>
    public bool SupportsThreadedSequenceState => true;

    /// <inheritdoc/>
    public IRecurrentSequenceState? CreateSequenceState() => new Lfm2ConvStateCache(_convLayerCount, _lCache - 1, Config.HiddenSize);

    /// <inheritdoc/>
    public IReadOnlyList<ITensor> ForwardBatch(IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
    {
        ArgumentNullException.ThrowIfNull(requests);
        var results = new ITensor[requests.Count];
        for (int i = 0; i < requests.Count; i++)
        {
            var r = requests[i];
            if (r.Adapter is not null)
                throw new NotSupportedException("Lfm2MoeTransformerModel does not support LoRA adapters.");
            if (requests.Count > 1 && r.SsmState is null)
                throw new ArgumentException($"ForwardBatch with {requests.Count} requests requires a per-seq SsmState; request[{i}] has none.");
            results[i] = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache, r.SsmState);
        }
        return results;
    }

    [SkipLocalsInit]
    private ITensor ForwardCore(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                                IKvCache? kvCache, Lfm2ConvStateCache? state, bool lastOnly)
    {
        int T = tokenIds.Length;
        if (T == 0 || T != positions.Length)
            throw new ArgumentException("tokenIds and positions must have equal, non-zero length.");
        int H = Config.HiddenSize, V = Config.VocabSize;
        float eps = Config.NormEpsilon;
        for (int i = 0; i < positions.Length; i++)
            if ((uint)positions[i] >= (uint)Config.MaxSequenceLength)
                throw new ArgumentOutOfRangeException(nameof(positions), $"Position {positions[i]} at index {i} exceeds max sequence length {Config.MaxSequenceLength}.");

        var prevState = _activeState;
        _activeState = state ?? _defaultState;
        try
        {
            _s.Ensure(T, Config, _lCache, lastOnly ? 1 : T);
            _threadPool?.SetDispatchMode(T == 1 ? DispatchMode.SpinWait : DispatchMode.EventBased);

            float* hidden = _s.Hidden, residual = _s.Residual, normOut = _s.NormOut;
            EmbedTokens(tokenIds, hidden, H);
            if (TensorDump.Enabled) TensorDump.Dump2D("token_embd", hidden, T, H);

            var kinds = _layout.LayerKind;
            for (int layer = 0; layer < _layers.Length; layer++)
            {
                var lw = _layers[layer];
                new Span<float>(hidden, T * H).CopyTo(new Span<float>(residual, T * H));
                RmsRows(hidden, lw.OperatorNorm, eps, normOut, T, H);
                if (kinds[layer] == HybridLayerKind.ShortConv)
                    ForwardConv(lw, _convOrdinalForLayer[layer], layer, T, normOut);
                else
                    ForwardAttention(lw.Attn!, layer, T, positions, normOut, kvCache);
                if (TensorDump.Enabled) TensorDump.Dump2D($"blk.{layer}.mixer_out", normOut, T, H);
                AddRows(residual, normOut, hidden, T * H);

                new Span<float>(hidden, T * H).CopyTo(new Span<float>(residual, T * H));
                RmsRows(hidden, lw.FfnNorm, eps, normOut, T, H);
                if (lw.Moe is not null) ForwardMoe(lw.Moe, layer, T, H, normOut);
                else ForwardDense(lw, T, H, normOut);
                if (TensorDump.Enabled) TensorDump.Dump2D($"blk.{layer}.ffn_out", normOut, T, H);
                AddRows(residual, normOut, hidden, T * H);
                if (TensorDump.Enabled) TensorDump.Dump2D($"blk.{layer}.l_out", hidden, T, H);
            }

            int rows = lastOnly ? 1 : T;
            float* headIn = hidden + (long)(T - rows) * H;
            RmsRows(headIn, _outputNormWeight, eps, headIn, rows, H);
            if (TensorDump.Enabled) TensorDump.Dump2D("result_norm", headIn, rows, H);
            Gemm(_outputWeight, _outputQuantType, headIn, _s.Logits, _outputOutputDim, _outputInputDim, rows);

            var result = UnmanagedTensor.Allocate(new TensorShape(rows, V), DType.Float32, deviceId);
            new Span<float>(_s.Logits, rows * V).CopyTo(new Span<float>((void*)result.DataPointer, rows * V));
            return result;
        }
        finally
        {
            _activeState = prevState;
        }
    }

    // ── Sub-layers ───────────────────────────────────────────────────────────

    private void ForwardConv(Lfm2Layer lw, int convOrdinal, int layer, int T, float* normOut)
    {
        int H = Config.HiddenSize, L = _lCache;
        float* proj = _s.ConvProj;       // [T, 3H] = B | C | X per token
        float* buf = _s.ConvBuf;         // [(L-1+T), H]
        float* y = _s.ConvOut;           // [T, H]
        Gemm(lw.InProj.Ptr, lw.InProj.Qt, normOut, proj, lw.InProj.OutDim, lw.InProj.InDim, T);

        float* st = _activeState!.Layer(convOrdinal);
        new Span<float>(st, (L - 1) * H).CopyTo(new Span<float>(buf, (L - 1) * H));
        for (int t = 0; t < T; t++)
        {
            float* row = proj + (long)t * 3 * H;
            TensorPrimitives.Multiply(new ReadOnlySpan<float>(row, H), new ReadOnlySpan<float>(row + 2 * H, H),
                                      new Span<float>(buf + (long)(L - 1 + t) * H, H));
        }
        float[] wT = lw.ConvWeightT!;
        for (int t = 0; t < T; t++)
        {
            var o = new Span<float>(y + (long)t * H, H);
            TensorPrimitives.Multiply(new ReadOnlySpan<float>(buf + (long)t * H, H), wT.AsSpan(0, H), o);
            for (int j = 1; j < L; j++)
                TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(buf + (long)(t + j) * H, H), wT.AsSpan(j * H, H), o, o);
            TensorPrimitives.Multiply(o, new ReadOnlySpan<float>(proj + (long)t * 3 * H + H, H), o); // * C
        }
        // New state = last L-1 rows of the extended sequence.
        new Span<float>(buf + (long)T * H, (L - 1) * H).CopyTo(new Span<float>(st, (L - 1) * H));
        if (TensorDump.Enabled) TensorDump.Dump2D($"blk.{layer}.conv_y", y, T, H);
        Gemm(lw.OutProj.Ptr, lw.OutProj.Qt, y, normOut, lw.OutProj.OutDim, lw.OutProj.InDim, T);
    }

    private void ForwardAttention(Qwen3FullAttnWeights a, int layer, int T, ReadOnlySpan<int> positions, float* normOut, IKvCache? kvCache)
    {
        int nh = Config.NumAttentionHeads, nkv = a.NumKvHeads, hd = Config.HeadDim;
        int qE = nh * hd, kvE = nkv * hd;
        float* q = _s.Q, k = _s.K, v = _s.V, attnOut = _s.AttnOut;
        Gemm(a.QWeight, a.QQuantType, normOut, q, a.QOutputDim, a.QInputDim, T);
        Gemm(a.KWeight, a.KQuantType, normOut, k, a.KOutputDim, a.KInputDim, T);
        Gemm(a.VWeight, a.VQuantType, normOut, v, a.VOutputDim, a.VInputDim, T);
        Mamba3QkNorm.Execute(new Span<float>(q, T * qE), a.QNormWeight, Config.NormEpsilon, T, nh, hd);
        Mamba3QkNorm.Execute(new Span<float>(k, T * kvE), a.KNormWeight, Config.NormEpsilon, T, nkv, hd);
        RoPE.Execute(new Span<float>(q, T * qE), new Span<float>(k, T * kvE), positions, nh, nkv, hd, _ropeDim,
                     _ropeCosTable, _ropeSinTable, RoPEType.NeoX);
        if (kvCache is not null)
        {
            int slot = _kvSlotForLayer[layer];
            if (slot < 0) throw new InvalidOperationException($"Layer {layer} has no KV-cache slot.");
            var kRef = new TensorRef(T, kvE, DType.Float32, -1, (nint)k);
            var vRef = new TensorRef(T, kvE, DType.Float32, -1, (nint)v);
            kvCache.Update(kRef, vRef, positions, slot);
            int seqKv = kvCache.CurrentLength;
            var ck = kvCache.GetKeysRef(slot); var cv = kvCache.GetValuesRef(slot);
            Attention.Execute(q, (float*)ck.DataPointer, (float*)cv.DataPointer, attnOut, T, seqKv, nh, nkv, hd,
                              positions[0], pool: null, slidingWindowSize: null);
        }
        else
        {
            Attention.Execute(q, k, v, attnOut, T, T, nh, nkv, hd, 0, pool: null, slidingWindowSize: null);
        }
        Gemm(a.OWeight, a.OQuantType, attnOut, normOut, a.OOutputDim, a.OInputDim, T);
    }

    private void ForwardDense(Lfm2Layer lw, int T, int H, float* normOut)
    {
        int F = lw.DenseGate.OutDim;
        float* g = _s.FfnGate, u = _s.FfnUp;
        Gemm(lw.DenseGate.Ptr, lw.DenseGate.Qt, normOut, g, F, H, T);
        Gemm(lw.DenseUp.Ptr, lw.DenseUp.Qt, normOut, u, lw.DenseUp.OutDim, H, T);
        var gs = new Span<float>(g, T * F);
        SiLu.Execute(gs, gs);
        TensorPrimitives.Multiply(gs, new ReadOnlySpan<float>(u, T * F), gs);
        Gemm(lw.DenseDown.Ptr, lw.DenseDown.Qt, g, normOut, lw.DenseDown.OutDim, F, T);
    }

    private void ForwardMoe(MoeLayerWeights moe, int layer, int T, int H, float* normOut)
    {
        int E = moe.NumExperts, K = moe.NumExpertsPerTok, I = moe.IntermediateSize;
        int total = T * K;
        Span<int> assignExpert = _s.MoeAssignExpert.AsSpan(0, total);
        Span<float> assignWeight = _s.MoeAssignWeight.AsSpan(0, total);
        Span<int> cursors = _s.MoeBucketCursors.AsSpan(0, E + 1);
        Span<int> bucketTokens = _s.MoeBucketTokens.AsSpan(0, total);
        Span<int> bucketSlots = _s.MoeBucketSlots.AsSpan(0, total);
        Span<int> unique = _s.MoeUniqueExperts.AsSpan(0, E);

        int uniqueCount = MoeSwiGluMlp.Route(
            hidden: new ReadOnlySpan<float>(normOut, T * H), gateWeights: moe.Gate,
            assignExpert: assignExpert, assignWeight: assignWeight,
            bucketCursors: cursors, bucketTokens: bucketTokens, bucketSlots: bucketSlots, uniqueExperts: unique,
            numExperts: E, numExpertsPerTok: K, hiddenSize: H, seqLen: T,
            normTopKProb: moe.NormTopKProb,
            sigmoidGating: moe.SigmoidGating,
            selectionBias: moe.SelectionBias is null ? default : moe.SelectionBias.AsSpan(),
            weightsScale: moe.WeightsScale);

        if (TensorDump.Enabled)
        {
            // Router decisions for cross-engine agreement checks: [T, K] expert ids as floats.
            float* ids = _s.Q; // scratch: free during the FFN sub-layer
            for (int i = 0; i < total; i++) ids[i] = assignExpert[i];
            TensorDump.Dump2D($"blk.{layer}.moe_topk", ids, T, K);
        }
        var gQt = moe.GateExpsRawQt; var uQt = moe.UpExpsRawQt; var dQt = moe.DownExpsRawQt;
        MoeSwiGluMlp.ExecuteRoutedFromAssignments(
            hidden: new ReadOnlySpan<float>(normOut, T * H),
            gateExpsRawBase: moe.GateExpsRaw, gateExpsQt: gQt, gateExpsRowBytes: Dequantize.RowByteSize((long)I * H, gQt), gateExpsF32Ptrs: ReadOnlySpan<nint>.Empty,
            upExpsRawBase: moe.UpExpsRaw, upExpsQt: uQt, upExpsRowBytes: Dequantize.RowByteSize((long)I * H, uQt), upExpsF32Ptrs: ReadOnlySpan<nint>.Empty,
            downExpsRawBase: moe.DownExpsRaw, downExpsQt: dQt, downExpsRowBytes: Dequantize.RowByteSize((long)H * I, dQt), downExpsF32Ptrs: ReadOnlySpan<nint>.Empty,
            assignExpert: assignExpert, assignWeight: assignWeight,
            bucketCursors: cursors, bucketTokens: bucketTokens, bucketSlots: bucketSlots,
            uniqueExperts: unique, uniqueExpertCount: uniqueCount,
            output: new Span<float>(normOut, T * H),
            numExperts: E, numExpertsPerTok: K, hiddenSize: H, intermediateSize: I, seqLen: T,
            sharedGateProj: moe.SharedGateProj, sharedUpProj: moe.SharedUpProj, sharedDownProj: moe.SharedDownProj,
            sharedIntermediateSize: moe.SharedIntermediateSize, sharedExpertGate: ReadOnlySpan<float>.Empty,
            loraAdapter: null, loraLayer: layer, threadPool: _threadPool,
            packedLegacyExperts: LegacyPacked);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private static void RmsRows(float* src, float[] w, float eps, float* dst, int rows, int H)
    {
        for (int t = 0; t < rows; t++)
            RmsNorm.Execute(new ReadOnlySpan<float>(src + (long)t * H, H), w, eps, new Span<float>(dst + (long)t * H, H));
    }

    private static void AddRows(float* a, float* b, float* dst, int n)
        => TensorPrimitives.Add(new ReadOnlySpan<float>(a, n), new ReadOnlySpan<float>(b, n), new Span<float>(dst, n));

    private void EmbedTokens(ReadOnlySpan<int> tokenIds, float* hidden, int H)
    {
        var qt = _tokenEmbedQuantType;
        for (int t = 0; t < tokenIds.Length; t++)
        {
            int id = tokenIds[t];
            if ((uint)id >= (uint)Config.VocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token ID {id} at position {t} is out of range [0, {Config.VocabSize}).");
            var dest = new Span<float>(hidden + (long)t * H, H);
            if (qt == QuantizationType.F32)
                new ReadOnlySpan<float>((float*)_tokenEmbedWeight + (long)id * H, H).CopyTo(dest);
            else if (qt == QuantizationType.F16)
                TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>((Half*)_tokenEmbedWeight + (long)id * H, H), dest);
            else
                Dequantize.ToFloat32(_tokenEmbedWeight + (nint)((long)id * Dequantize.RowByteSize(H, qt)), H, qt, dest);
        }
    }

    private static readonly bool LegacyPacked = Environment.GetEnvironmentVariable("DOTLLM_MOE_LEGACY_PACKED") != "0";

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Gemm(nint w, QuantizationType qt, float* b, float* c, int m, int k, int n)
    {
        switch (qt)
        {
            case QuantizationType.Q8_0: MatMul.GemmQ8_0((byte*)w, b, c, m, k, n, _threadPool, null); return;
            case QuantizationType.Q5_0: MatMul.GemmQ5_0((byte*)w, b, c, m, k, n, _threadPool, null); return;
            case QuantizationType.Q4_K: MatMul.GemmQ4_K((byte*)w, b, c, m, k, n, _threadPool, null); return;
            case QuantizationType.Q5_K: MatMul.GemmQ5_K((byte*)w, b, c, m, k, n, _threadPool, null); return;
            case QuantizationType.Q6_K: MatMul.GemmQ6_K((byte*)w, b, c, m, k, n, _threadPool, null); return;
            case QuantizationType.F32: MatMul.GemmF32((float*)w, b, c, m, k, n, _threadPool); return;
            case QuantizationType.F16: MatMul.GemmF16(w, b, c, m, k, n, _threadPool); return;
            case QuantizationType.Q4_0:
            case QuantizationType.Q4_1:
            case QuantizationType.Q5_1:
            case QuantizationType.IQ4_NL:
                if (LegacyPacked) MatMul.GemmLegacyQuant((byte*)w, qt, b, c, m, k, n, _threadPool, null);
                else MatMul.GemmLegacyQuantOrDequant((byte*)w, qt, b, c, m, k, n, _threadPool, null);
                return;
            default:
                MatMul.GemmDequantRows((byte*)w, qt, b, c, m, k, n, pool: null); return;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_ownsThreadPool) _threadPool?.Dispose();
        _s.Dispose();
        _defaultState.Dispose();
        GC.SuppressFinalize(this);
    }

    // ── Weights / scratch ────────────────────────────────────────────────────

    private readonly record struct Lfm2Proj(nint Ptr, QuantizationType Qt, int InDim, int OutDim);

    private sealed class Lfm2Layer
    {
        public required float[] OperatorNorm;
        public required float[] FfnNorm;
        public Lfm2Proj InProj, OutProj;
        public float[]? ConvWeightT;       // [L, H] transposed (tap-major) so the conv vectorises over channels
        public Qwen3FullAttnWeights? Attn;
        public Lfm2Proj DenseGate, DenseUp, DenseDown;
        public MoeLayerWeights? Moe;
    }

    private sealed class Lfm2Scratch : IDisposable
    {
        private int _cap;
        private int _logitRows;
        private readonly List<nint> _all = new();
        public float* Hidden, Residual, NormOut, Logits, ConvProj, ConvBuf, ConvOut, Q, K, V, AttnOut, FfnGate, FfnUp;
        public int[] MoeAssignExpert = [], MoeBucketTokens = [], MoeBucketSlots = [], MoeBucketCursors = [], MoeUniqueExperts = [];
        public float[] MoeAssignWeight = [];
        public long AllocatedBytes;

        public void Ensure(int T, ModelConfig c, int lCache, int logitRows)
        {
            int H = c.HiddenSize;
            if (T > _cap || logitRows > _logitRows)
            {
                int newCap = _cap == 0 ? 1 : _cap;
                while (newCap < T) newCap *= 2;
                int newRows = Math.Max(_logitRows, logitRows);
                // Logits sized for the requested row count (not the full capacity) so a 4k-token prefill with
                // lastTokenLogitsOnly does not pin 2 GB; full-logit callers (perplexity) grow it on demand.
                FreeAll();
                int nh = c.NumAttentionHeads, hd = c.HeadDim;
                int kvHeads = c.NumKvHeads;
                int dense = c.IntermediateSize;
                _cap = newCap; _logitRows = Math.Max(newRows, 1);
                Hidden = A((long)_cap * H); Residual = A((long)_cap * H); NormOut = A((long)_cap * H);
                Logits = A((long)_logitRows * c.VocabSize);
                ConvProj = A((long)_cap * 3 * H); ConvBuf = A((long)(lCache - 1 + _cap) * H); ConvOut = A((long)_cap * H);
                Q = A((long)_cap * nh * hd); K = A((long)_cap * kvHeads * hd); V = A((long)_cap * kvHeads * hd); AttnOut = A((long)_cap * nh * hd);
                FfnGate = A((long)_cap * dense); FfnUp = A((long)_cap * dense);
                int k = c.Moe!.NumExpertsPerTok, e = c.Moe.NumExperts;
                MoeAssignExpert = new int[_cap * k]; MoeAssignWeight = new float[_cap * k];
                MoeBucketTokens = new int[_cap * k]; MoeBucketSlots = new int[_cap * k];
                MoeBucketCursors = new int[e + 1]; MoeUniqueExperts = new int[e];
            }
        }

        private float* A(long elems)
        {
            long bytes = Math.Max(elems, 1) * sizeof(float);
            nint p = (nint)NativeMemory.AlignedAlloc((nuint)bytes, 64);
            _all.Add(p);
            AllocatedBytes += bytes;
            return (float*)p;
        }

        private void FreeAll()
        {
            foreach (var p in _all) NativeMemory.AlignedFree((void*)p);
            _all.Clear();
            AllocatedBytes = 0;
        }

        public void Dispose() => FreeAll();
    }
}
