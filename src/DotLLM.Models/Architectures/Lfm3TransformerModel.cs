using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using NativeMemory = System.Runtime.InteropServices.NativeMemory;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Liquid AI LFM3 (<c>lfm3</c>) CPU model. Reference: llama.cpp <c>src/models/lfm3.cpp</c> and HF <c>modeling_lfm3.py</c>.
/// </summary>
/// <remarks>
/// Every layer is two residual sub-layers with sandwich norms:
/// <c>x += post_operator_norm(mixer(operator_norm(x)))</c>, <c>x += post_ffn_norm(ffn(ffn_norm(x)))</c>.
/// The mixer is a gated short convolution (identical to LFM2, <c>L_cache = 7</c>), a sliding-window GQA layer (window 256,
/// theta 1e4, full rotary) or a global GQA layer (theta 1e6, only the first <c>rope_freqs.weight == 1</c> frequency pairs
/// rotate). Both attention flavours apply per-head QK RMS-norm, NeoX RoPE and a per-head sigmoid gate
/// (<c>attn_gate</c>, hidden to n_heads) on the attention output before <c>o_proj</c>. The FFN is a dense SwiGLU on every
/// layer. The LM head is tied to the token embedding and the final norm is <c>output_norm.weight</c>.
/// </remarks>
public sealed unsafe class Lfm3TransformerModel : IModel
{
    private readonly GgufFile? _gguf;
    private readonly Lfm3Layer[] _layers;
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
    private readonly int _ropeHalf;
    private readonly float[] _invFreqLocal;
    private readonly float[] _invFreqGlobal;
    private readonly int _globalRotPairs;
    private readonly Lfm2ConvStateCache _defaultState;
    private readonly ComputeThreadPool? _threadPool;
    private readonly bool _ownsThreadPool;
    private readonly Lfm3Scratch _s = new();
    private Lfm2ConvStateCache? _activeState;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _s.AllocatedBytes + _defaultState.AllocatedBytes;

    // Internal accessors for the Vulkan host (weights are read from the same mmap'd GGUF views).
    internal Lfm3Layer[] LayerWeights => _layers;
    internal float[] OutputNormWeight => _outputNormWeight;
    internal (nint Ptr, QuantizationType Qt) TokenEmbed => (_tokenEmbedWeight, _tokenEmbedQuantType);
    internal (nint Ptr, QuantizationType Qt, int OutDim, int InDim) Output => (_outputWeight, _outputQuantType, _outputOutputDim, _outputInputDim);
    internal int[] KvSlotForLayer => _kvSlotForLayer;
    internal int[] ConvOrdinalForLayer => _convOrdinalForLayer;
    internal int ConvLayerCount => _convLayerCount;
    internal float[] InvFreqLocal => _invFreqLocal;
    internal float[] InvFreqGlobal => _invFreqGlobal;
    internal int GlobalRotatedPairs => _globalRotPairs;

    /// <summary>Number of attention layers (global + sliding): the sparse KV-cache slot count.</summary>
    public int AttentionLayerCount => _attentionLayerCount;

    private Lfm3TransformerModel(
        ModelConfig config, GgufFile? gguf, Lfm3Layer[] layers, float[] outputNorm,
        nint embed, QuantizationType embedQt, nint output, QuantizationType outputQt, int outM, int outK,
        int[] kvSlot, int[] convOrdinal, int attnCount, int convCount,
        float[] invFreqLocal, float[] invFreqGlobal, int globalRotPairs, ComputeThreadPool? pool, bool ownsPool)
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
        _ropeHalf = invFreqLocal.Length;
        _invFreqLocal = invFreqLocal;
        _invFreqGlobal = invFreqGlobal;
        _globalRotPairs = globalRotPairs;
        _threadPool = pool;
        _ownsThreadPool = ownsPool;
        _defaultState = new Lfm2ConvStateCache(convCount, _lCache - 1, config.HiddenSize);
    }

    /// <summary>Loads an LFM3 model from an opened GGUF (which must outlive the model).</summary>
    public static Lfm3TransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config)
        => LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);

    /// <summary>Loads an LFM3 model from an opened GGUF with threading configuration.</summary>
    public static Lfm3TransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config, ThreadingConfig threading)
    {
        if (config.Architecture != Architecture.Lfm3)
            throw new ArgumentException($"Lfm3TransformerModel requires Architecture.Lfm3, got {config.Architecture}.", nameof(config));
        if (config.HybridLayout is null || config.ShortConvCacheLength < 2 || config.PerLayerSlidingWindow is null)
            throw new ArgumentException("Lfm3 config must have HybridLayout, PerLayerSlidingWindow and ShortConvCacheLength populated.", nameof(config));

        var dataBase = new GgufDataBase(gguf);
        var tensors = gguf.TensorsByName;
        Nvfp4Support.RejectUnappliedScales(tensors, consumedScales: null, "The LFM3 loader"); // #1038
        var layout = config.HybridLayout;
        int hidden = config.HiddenSize;

        var embDesc = tensors["token_embd.weight"];
        nint embPtr = dataBase.Of(embDesc);
        // llama.cpp lfm3.cpp: output_norm.weight (LFM2 files name it token_embd_norm.weight).
        var normDesc = tensors.TryGetValue("output_norm.weight", out var on) ? on : tensors["token_embd_norm.weight"];
        float[] outNorm = DequantizeF32(dataBase, normDesc, hidden);

        nint outPtr; QuantizationType outQt; int outM, outK;
        if (tensors.TryGetValue("output.weight", out var outDesc))
        {
            outPtr = dataBase.Of(outDesc); outQt = outDesc.QuantizationType; outK = outDesc.Shape[0]; outM = outDesc.Shape[1];
        }
        else
        {
            outPtr = embPtr; outQt = embDesc.QuantizationType; outK = embDesc.Shape[0]; outM = embDesc.Shape[1];
        }

        int hd = config.HeadDim;
        int half = hd / 2;
        int ropeDim = config.RoPEConfig?.DimensionCount ?? hd;
        if (ropeDim != hd)
            throw new NotSupportedException($"LFM3 expects rope.dimension_count == head_dim ({hd}); got {ropeDim}.");
        float localTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        float globalTheta = config.GlobalRoPEConfig?.Theta ?? config.RoPEConfig?.Theta ?? 1000000.0f;

        // llama.cpp: sliding layers pass rope_factors = nullptr; global layers pass rope_freqs.weight (divisors, 1.0 = rotate,
        // 1e30 = frequency ~ 0 = identity). HF writes the same thing as an inv_freq vector with a zero tail.
        float[] factors = new float[half];
        Array.Fill(factors, 1f);
        if (tensors.TryGetValue("rope_freqs.weight", out var rf))
        {
            if (rf.Shape[0] != half) throw new InvalidDataException($"rope_freqs.weight has {rf.Shape[0]} entries, expected {half}.");
            factors = DequantizeF32(dataBase, rf, half);
        }
        var invLocal = new float[half];
        var invGlobal = new float[half];
        int rotPairs = 0;
        for (int i = 0; i < half; i++)
        {
            invLocal[i] = MathF.Pow(localTheta, -2f * i / hd);
            invGlobal[i] = MathF.Pow(globalTheta, -2f * i / hd) / factors[i];
            if (factors[i] < 1e20f) rotPairs = i + 1; // pairs beyond the last finite factor are identity rotations
        }
        // Zero the (divided-by-1e30) tail exactly rather than rely on a ~1e-36 angle.
        for (int i = rotPairs; i < half; i++) invGlobal[i] = 0f;

        var layers = new Lfm3Layer[config.NumLayers];
        var kvSlot = new int[config.NumLayers];
        var convOrd = new int[config.NumLayers];
        int attnCount = 0, convCount = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            layers[i] = LoadLayer(i, dataBase, tensors, config);
            bool isAttn = layout.LayerKind[i] == HybridLayerKind.Attention;
            kvSlot[i] = isAttn ? attnCount++ : -1;
            convOrd[i] = isAttn ? -1 : convCount++;
        }

        ComputeThreadPool? pool = CreatePool(threading);
        return new Lfm3TransformerModel(config, gguf, layers, outNorm, embPtr, embDesc.QuantizationType,
            outPtr, outQt, outM, outK, kvSlot, convOrd, attnCount, convCount, invLocal, invGlobal, rotPairs, pool, pool is not null);
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

    private static Lfm3Layer LoadLayer(int layerIdx, GgufDataBase db, IReadOnlyDictionary<string, GgufTensorDescriptor> tensors, ModelConfig config)
    {
        string p = $"blk.{layerIdx}";
        int hidden = config.HiddenSize;
        // Sandwich norms are optional together (use_sandwich_norm=false checkpoints omit both; llama.cpp requires both or neither).
        bool hasPostOp = tensors.TryGetValue($"{p}.post_attention_norm.weight", out var postOp);
        bool hasPostFfn = tensors.TryGetValue($"{p}.post_ffw_norm.weight", out var postFfn);
        if (hasPostOp != hasPostFfn)
            throw new InvalidDataException($"{p}: LFM3 requires both post_attention_norm and post_ffw_norm, or neither.");
        var layer = new Lfm3Layer
        {
            OperatorNorm = DequantizeF32(db, tensors[$"{p}.attn_norm.weight"], hidden),
            FfnNorm = DequantizeF32(db, tensors[$"{p}.ffn_norm.weight"], hidden),
            PostOperatorNorm = hasPostOp ? DequantizeF32(db, postOp!, hidden) : null,
            PostFfnNorm = hasPostFfn ? DequantizeF32(db, postFfn!, hidden) : null,
            DenseGate = Proj(db, tensors[$"{p}.ffn_gate.weight"]),
            DenseUp = Proj(db, tensors[$"{p}.ffn_up.weight"]),
            DenseDown = Proj(db, tensors[$"{p}.ffn_down.weight"]),
            Window = config.PerLayerSlidingWindow![layerIdx],
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
            layer.AttnGate = Proj(db, tensors[$"{p}.attn_gate.weight"]);
            if (layer.AttnGate.OutDim != config.NumAttentionHeads)
                throw new InvalidDataException($"{p}.attn_gate.weight out dim {layer.AttnGate.OutDim} != n_head {config.NumAttentionHeads}.");
        }
        return layer;
    }

    private static Lfm3Proj Proj(GgufDataBase db, GgufTensorDescriptor d)
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
        if (s is Lfm2ConvStateCache c && c.NumSsmLayers == _convLayerCount && c.StateSlots == _lCache - 1 && c.HiddenSize == Config.HiddenSize) return c;
        throw new ArgumentException($"Lfm3TransformerModel requires an {nameof(Lfm2ConvStateCache)} with {_convLayerCount} layers; got {s.GetType().Name}.", nameof(s));
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
            if (requests.Count > 1 && r.SsmState is null)
                throw new ArgumentException($"ForwardBatch with {requests.Count} requests requires a per-seq SsmState; request[{i}] has none.");
            if (r.Adapter is not null)
                throw new NotSupportedException("LoRA adapters are not supported for LFM3 yet.");
            results[i] = ForwardCore(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache, ResolveState(r.SsmState), r.LastTokenLogitsOnly);
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
            _s.Ensure(T, Config, _lCache, _ropeHalf);
            _threadPool?.SetDispatchMode(T == 1 ? DispatchMode.SpinWait : DispatchMode.EventBased);
            ComputeRopeTables(positions, T);

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
                    ForwardAttention(lw, layer, T, positions, normOut, kvCache);
                if (TensorDump.Enabled) TensorDump.Dump2D($"blk.{layer}.mixer_out", normOut, T, H);
                if (lw.PostOperatorNorm is not null)
                    RmsRows(normOut, lw.PostOperatorNorm, eps, normOut, T, H);
                AddRows(residual, normOut, hidden, T * H);

                new Span<float>(hidden, T * H).CopyTo(new Span<float>(residual, T * H));
                RmsRows(hidden, lw.FfnNorm, eps, normOut, T, H);
                ForwardDense(lw, T, H, normOut);
                if (TensorDump.Enabled) TensorDump.Dump2D($"blk.{layer}.ffn_out", normOut, T, H);
                if (lw.PostFfnNorm is not null)
                    RmsRows(normOut, lw.PostFfnNorm, eps, normOut, T, H);
                AddRows(residual, normOut, hidden, T * H);
                if (TensorDump.Enabled) TensorDump.Dump2D($"blk.{layer}.l_out", hidden, T, H);
            }

            int rows = lastOnly ? 1 : T;
            float* headIn = hidden + (long)(T - rows) * H;
            RmsRows(headIn, _outputNormWeight, eps, headIn, rows, H);
            if (TensorDump.Enabled) TensorDump.Dump2D("result_norm", headIn, rows, H);
            var result = UnmanagedTensor.Allocate(new TensorShape(rows, V), DType.Float32, deviceId);
            Gemm(_outputWeight, _outputQuantType, headIn, (float*)result.DataPointer, _outputOutputDim, _outputInputDim, rows);
            return result;
        }
        finally
        {
            _activeState = prevState;
        }
    }

    // ── Sub-layers ───────────────────────────────────────────────────────────

    private void ForwardConv(Lfm3Layer lw, int convOrdinal, int layer, int T, float* normOut)
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

    /// <summary>
    /// Per-token cos/sin for the two RoPE flavours: <c>[T, half]</c> for the sliding layers (every pair rotates) and the
    /// global layers (pairs past <see cref="_globalRotPairs"/> have inverse frequency 0, i.e. identity).
    /// </summary>
    private void ComputeRopeTables(ReadOnlySpan<int> positions, int T)
    {
        int half = _ropeHalf;
        float* lc = _s.CosLocal, ls = _s.SinLocal, gc = _s.CosGlobal, gs = _s.SinGlobal;
        for (int t = 0; t < T; t++)
        {
            float p = positions[t];
            for (int i = 0; i < half; i++)
            {
                (float s, float c) = MathF.SinCos(p * _invFreqLocal[i]);
                lc[t * half + i] = c; ls[t * half + i] = s;
            }
            for (int i = 0; i < _globalRotPairs; i++)
            {
                (float s, float c) = MathF.SinCos(p * _invFreqGlobal[i]);
                gc[t * half + i] = c; gs[t * half + i] = s;
            }
        }
    }

    /// <summary>NeoX rotation of the first <paramref name="pairs"/> (i, i + half) pairs of every head; later pairs are untouched.</summary>
    private void ApplyRope(float* x, int T, int heads, int hd, float* cosT, float* sinT, int pairs)
    {
        int half = _ropeHalf;
        for (int t = 0; t < T; t++)
        {
            float* ct = cosT + (long)t * half, st = sinT + (long)t * half;
            for (int h = 0; h < heads; h++)
            {
                float* v = x + ((long)t * heads + h) * hd;
                for (int i = 0; i < pairs; i++)
                {
                    float a = v[i], b = v[i + half];
                    v[i] = a * ct[i] - b * st[i];
                    v[i + half] = b * ct[i] + a * st[i];
                }
            }
        }
    }

    private void ForwardAttention(Lfm3Layer lw, int layer, int T, ReadOnlySpan<int> positions, float* normOut, IKvCache? kvCache)
    {
        var a = lw.Attn!;
        int nh = Config.NumAttentionHeads, nkv = a.NumKvHeads, hd = Config.HeadDim;
        int qE = nh * hd, kvE = nkv * hd;
        float* q = _s.Q, k = _s.K, v = _s.V, attnOut = _s.AttnOut, gate = _s.Gate;
        Gemm(a.QWeight, a.QQuantType, normOut, q, a.QOutputDim, a.QInputDim, T);
        Gemm(a.KWeight, a.KQuantType, normOut, k, a.KOutputDim, a.KInputDim, T);
        Gemm(a.VWeight, a.VQuantType, normOut, v, a.VOutputDim, a.VInputDim, T);
        Gemm(lw.AttnGate.Ptr, lw.AttnGate.Qt, normOut, gate, lw.AttnGate.OutDim, lw.AttnGate.InDim, T);
        Mamba3QkNorm.Execute(new Span<float>(q, T * qE), a.QNormWeight, Config.NormEpsilon, T, nh, hd);
        Mamba3QkNorm.Execute(new Span<float>(k, T * kvE), a.KNormWeight, Config.NormEpsilon, T, nkv, hd);
        bool sliding = lw.Window is not null;
        float* cosT = sliding ? _s.CosLocal : _s.CosGlobal, sinT = sliding ? _s.SinLocal : _s.SinGlobal;
        int pairs = sliding ? _ropeHalf : _globalRotPairs;
        ApplyRope(q, T, nh, hd, cosT, sinT, pairs);
        ApplyRope(k, T, nkv, hd, cosT, sinT, pairs);
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
                              positions[0], pool: _threadPool, slidingWindowSize: lw.Window);
        }
        else
        {
            Attention.Execute(q, k, v, attnOut, T, T, nh, nkv, hd, 0, pool: _threadPool, slidingWindowSize: lw.Window);
        }
        // Per-head sigmoid gate (llama.cpp lfm3.cpp: attn_out * sigmoid(attn_gate . cur_in)) before the output projection.
        for (int t = 0; t < T; t++)
        {
            for (int h = 0; h < nh; h++)
            {
                float g = 1f / (1f + MathF.Exp(-gate[t * nh + h]));
                TensorPrimitives.Multiply(new ReadOnlySpan<float>(attnOut + ((long)t * nh + h) * hd, hd), g,
                                          new Span<float>(attnOut + ((long)t * nh + h) * hd, hd));
            }
        }
        Gemm(a.OWeight, a.OQuantType, attnOut, normOut, a.OOutputDim, a.OInputDim, T);
    }

    private void ForwardDense(Lfm3Layer lw, int T, int H, float* normOut)
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
            case QuantizationType.BF16: MatMul.GemmBf16(w, b, c, m, k, n, _threadPool); return;   // shift-widen, in place (#1118)
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

    internal readonly record struct Lfm3Proj(nint Ptr, QuantizationType Qt, int InDim, int OutDim);

    internal sealed class Lfm3Layer
    {
        public required float[] OperatorNorm;
        public required float[] FfnNorm;
        public float[]? PostOperatorNorm;
        public float[]? PostFfnNorm;
        public required Lfm3Proj DenseGate, DenseUp, DenseDown;
        public Lfm3Proj InProj, OutProj;
        public float[]? ConvWeightT;       // [L, H] transposed (tap-major) so the conv vectorises over channels
        public Qwen3FullAttnWeights? Attn;
        public Lfm3Proj AttnGate;
        /// <summary>Sliding-window size of a windowed attention layer; null for global and conv layers.</summary>
        public int? Window;
    }

    private sealed class Lfm3Scratch : IDisposable
    {
        private int _cap;
        private readonly List<nint> _all = new();
        public float* Hidden, Residual, NormOut, ConvProj, ConvBuf, ConvOut, Q, K, V, AttnOut, Gate, FfnGate, FfnUp;
        public float* CosLocal, SinLocal, CosGlobal, SinGlobal;
        public long AllocatedBytes;

        public void Ensure(int T, ModelConfig c, int lCache, int ropeHalf)
        {
            int H = c.HiddenSize;
            if (T <= _cap) return;
            int newCap = _cap == 0 ? 1 : _cap;
            while (newCap < T) newCap *= 2;
            FreeAll();
            int nh = c.NumAttentionHeads, hd = c.HeadDim, kvHeads = c.NumKvHeads, dense = c.IntermediateSize;
            _cap = newCap;
            Hidden = A((long)_cap * H); Residual = A((long)_cap * H); NormOut = A((long)_cap * H);
            ConvProj = A((long)_cap * 3 * H); ConvBuf = A((long)(lCache - 1 + _cap) * H); ConvOut = A((long)_cap * H);
            Q = A((long)_cap * nh * hd); K = A((long)_cap * kvHeads * hd); V = A((long)_cap * kvHeads * hd); AttnOut = A((long)_cap * nh * hd);
            Gate = A((long)_cap * nh);
            FfnGate = A((long)_cap * dense); FfnUp = A((long)_cap * dense);
            CosLocal = A((long)_cap * ropeHalf); SinLocal = A((long)_cap * ropeHalf);
            CosGlobal = A((long)_cap * ropeHalf); SinGlobal = A((long)_cap * ropeHalf);
        }

        private float* A(long elems)
        {
            long bytes = Math.Max(elems, 1) * sizeof(float);
            nint p = (nint)NativeMemory.AlignedAlloc((nuint)bytes, 64);
            // Cos/sin tails (pairs a layer type never writes) must read as identity, so zero the whole block.
            NativeMemory.Clear((void*)p, (nuint)bytes);
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
