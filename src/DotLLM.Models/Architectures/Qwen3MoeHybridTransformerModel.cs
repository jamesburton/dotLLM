using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Qwen3MoeHybrid (<c>qwen35moe</c>) model — Gated DeltaNet recurrence + sparse MoE FFN.
///
/// Each of the 40 layers has:
///   - a token-mixing path: GDN (Gated DeltaNet) or full GQA attention, and
///   - a sparse MoE SwiGLU FFN (always present, all 40 layers).
/// Full-attention layers occur every <see cref="GatedDeltaNetConfig.FullAttnInterval"/> steps
/// (1-indexed), e.g. layers 4, 8, 12, … 40 for interval=4.
/// </summary>
public sealed unsafe class Qwen3MoeHybridTransformerModel : IModel
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;

    private readonly GgufFile? _gguf; // kept alive; null when built from prebuilt weights
    private readonly Qwen3MoeLayerWeights[] _layers;
    private readonly float[] _outputNormWeight;

    private readonly nint _tokenEmbedWeight;
    private readonly QuantizationType _tokenEmbedQuantType;
    private readonly nint _outputWeight;
    private readonly QuantizationType _outputQuantType;
    private readonly int _outputOutputDim;  // vocab size
    private readonly int _outputInputDim;   // hidden size

    private readonly HybridLayerLayout _layout;
    private readonly GatedDeltaNetConfig _gdn;

    // KV-cache slot per layer index (-1 for GDN layers).
    private readonly int[] _kvSlotForLayer;
    private readonly int _attentionLayerCount;

    // GDN layer ordinal per layer index (-1 for attention layers).
    private readonly int[] _gdnLayerOrdinal;

    private readonly float[] _ropeCosTable;
    private readonly float[] _ropeSinTable;
    private readonly int _ropeDim;

    private readonly Qwen3MoeHybridForwardState _state;
    private readonly GdnStateCache _gdnCache;

    private readonly ComputeThreadPool? _threadPool;
    private readonly bool _ownsThreadPool;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _state.AllocatedBytes + _gdnCache.AllocatedBytes;

    /// <summary>Number of full-attention layers — the matching sparse KV-cache slot count.</summary>
    public int AttentionLayerCount => _attentionLayerCount;

    private Qwen3MoeHybridTransformerModel(
        ModelConfig config,
        GgufFile? gguf,
        Qwen3MoeLayerWeights[] layers,
        float[] outputNormWeight,
        nint tokenEmbedWeight, QuantizationType tokenEmbedQuantType,
        nint outputWeight, QuantizationType outputQuantType, int outputOutputDim, int outputInputDim,
        int[] kvSlotForLayer, int attentionLayerCount,
        float[] ropeCosTable, float[] ropeSinTable, int ropeDim,
        ComputeThreadPool? threadPool, bool ownsPool)
    {
        Config = config;
        _gguf = gguf;
        _layers = layers;
        _outputNormWeight = outputNormWeight;
        _tokenEmbedWeight = tokenEmbedWeight;
        _tokenEmbedQuantType = tokenEmbedQuantType;
        _outputWeight = outputWeight;
        _outputQuantType = outputQuantType;
        _outputOutputDim = outputOutputDim;
        _outputInputDim = outputInputDim;
        _layout = config.HybridLayout!;
        _gdn = config.GdnConfig!.Value;
        _kvSlotForLayer = kvSlotForLayer;
        _attentionLayerCount = attentionLayerCount;
        _ropeCosTable = ropeCosTable;
        _ropeSinTable = ropeSinTable;
        _ropeDim = ropeDim;
        _threadPool = threadPool;
        _ownsThreadPool = ownsPool;

        _gdnLayerOrdinal = new int[config.NumLayers];
        int gdnOrdinal = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            _gdnLayerOrdinal[i] = _layout.LayerKind[i] == HybridLayerKind.GatedDeltaNet
                ? gdnOrdinal++
                : -1;
        }

        _gdnCache = new GdnStateCache(_gdn, gdnOrdinal);

        // MoE routing scratch is sized to (numExperts + 1) for the cursor histogram and
        // (seqLen × numExpertsPerTok) per-(token,slot) arrays grown lazily by EnsureCapacity.
        // No per-layer weight dequant scratch — the routed path now matmuls the raw GGUF
        // quant view directly via MoeSwiGluMlp.ExecuteRoutedFromAssignments.
        var moe = config.Moe!;
        _state = new Qwen3MoeHybridForwardState(
            hiddenSize: config.HiddenSize,
            vocabSize: config.VocabSize,
            qElems: config.NumAttentionHeads * config.HeadDim,
            kvElems: config.NumKvHeads * config.HeadDim,
            convDim: (2 * _gdn.NKHead + _gdn.NVHead) * _gdn.DState,
            dConv: _gdn.DConv,
            nVHead: _gdn.NVHead,
            nKHead: _gdn.NKHead,
            dState: _gdn.DState,
            moeNumExperts: moe.NumExperts,
            moeNumExpertsPerTok: moe.NumExpertsPerTok);
    }

    /// <summary>
    /// Loads a Qwen3MoeHybrid model from an opened GGUF file (single-threaded).
    /// The <paramref name="gguf"/> must remain alive for the lifetime of the returned model.
    /// </summary>
    /// <param name="gguf">An opened GGUF file.</param>
    /// <param name="config">Model configuration extracted from the file.</param>
    /// <returns>A loaded model.</returns>
    public static Qwen3MoeHybridTransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config)
        => LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);

    /// <summary>
    /// Loads a Qwen3MoeHybrid model from an opened GGUF file with threading configuration.
    /// When <paramref name="threading"/> is parallel, creates a <see cref="ComputeThreadPool"/>
    /// owned by this model (disposed with the model). The <paramref name="gguf"/> must remain
    /// alive for the lifetime of the returned model.
    /// </summary>
    /// <param name="gguf">An opened GGUF file.</param>
    /// <param name="config">Model configuration extracted from the file.</param>
    /// <param name="threading">Threading configuration: number of threads, NUMA / P-core pinning, dispatch policy.</param>
    /// <returns>A loaded model.</returns>
    public static Qwen3MoeHybridTransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config, ThreadingConfig threading)
    {
        if (config.Architecture != Architecture.Qwen3MoeHybrid)
            throw new ArgumentException(
                $"Qwen3MoeHybridTransformerModel requires Architecture.Qwen3MoeHybrid, got {config.Architecture}.",
                nameof(config));
        if (config.HybridLayout is null)
            throw new ArgumentException("Qwen3MoeHybrid config must have HybridLayout populated.", nameof(config));
        if (config.GdnConfig is null)
            throw new ArgumentException("Qwen3MoeHybrid config must have GdnConfig populated.", nameof(config));
        if (config.Moe is null)
            throw new ArgumentException("Qwen3MoeHybrid config must have Moe populated.", nameof(config));

        nint dataBase = gguf.DataBasePointer;
        var tensors = gguf.TensorsByName;
        var layout = config.HybridLayout;

        var embDesc = tensors["token_embd.weight"];
        nint embPtr = dataBase + (nint)embDesc.DataOffset;

        var outNormDesc = tensors["output_norm.weight"];
        float[] outputNormWeight = DequantizeF32(dataBase, outNormDesc, config.HiddenSize);

        nint outputPtr;
        QuantizationType outputQt;
        int outputM, outputK;
        if (tensors.TryGetValue("output.weight", out var outDesc))
        {
            outputPtr = dataBase + (nint)outDesc.DataOffset;
            outputQt = outDesc.QuantizationType;
            outputK = outDesc.Shape[0];
            outputM = outDesc.Shape[1];
        }
        else
        {
            outputPtr = embPtr;
            outputQt = embDesc.QuantizationType;
            outputK = embDesc.Shape[0];
            outputM = embDesc.Shape[1];
        }

        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        if ((ropeDim & 1) != 0)
            throw new InvalidDataException(
                $"Qwen3MoeHybrid rope_dim={ropeDim} must be even for pair-wise rotation.");
        if (ropeDim > config.HeadDim)
            throw new InvalidDataException(
                $"Qwen3MoeHybrid rope_dim={ropeDim} exceeds head_dim={config.HeadDim}.");

        var owned = new List<nint>();
        var layers = new Qwen3MoeLayerWeights[config.NumLayers];
        var kvSlotForLayer = new int[config.NumLayers];
        int attentionLayerCount = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            layers[i] = LoadLayer(i, dataBase, tensors, config, owned);
            kvSlotForLayer[i] = layout.LayerKind[i] == HybridLayerKind.Attention
                ? attentionLayerCount++
                : -1;
        }

        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        int halfRope = ropeDim / 2;
        var ropeCos = new float[config.MaxSequenceLength * halfRope];
        var ropeSin = new float[config.MaxSequenceLength * halfRope];
        RoPE.PrecomputeFrequencyTable(config.MaxSequenceLength, ropeDim, ropeTheta, ropeCos, ropeSin);

        ComputeThreadPool? pool = CreatePool(threading);

        return new Qwen3MoeHybridTransformerModel(
            config, gguf, layers, outputNormWeight,
            embPtr, embDesc.QuantizationType,
            outputPtr, outputQt, outputM, outputK,
            kvSlotForLayer, attentionLayerCount,
            ropeCos, ropeSin, ropeDim,
            pool, ownsPool: pool is not null);
    }

    private static ComputeThreadPool? CreatePool(ThreadingConfig threading)
    {
        if (!threading.IsParallel)
            return null;

        int effectiveThreads = threading.EffectiveThreadCount;
        if (threading.EnableNumaPinning || threading.EnablePCorePinning)
        {
            var topology = NumaTopology.Detect();
            if (threading.EnablePCorePinning && topology.IsHybrid)
                effectiveThreads = Math.Min(effectiveThreads, topology.PerformanceCoreIds.Count);
            return new ComputeThreadPool(effectiveThreads, topology, threading);
        }

        return new ComputeThreadPool(effectiveThreads, topology: null, threading);
    }

    /// <summary>
    /// Builds a Qwen3MoeHybrid model from caller-owned, pre-dequantised weight pointers — used by
    /// parity tests that construct synthetic weight banks in unmanaged memory directly, bypassing
    /// the GGUF loader. Caller retains ownership of every <see cref="nint"/> pointer (token embed,
    /// output, plus every projection inside <paramref name="layers"/>).
    /// </summary>
    /// <remarks>
    /// Unlike <see cref="LoadFromGguf(GgufFile, ModelConfig)"/> there is no <see cref="GgufFile"/>
    /// to keep alive — the model holds <c>null</c> for the gguf reference. Disposing frees only
    /// the forward scratch and the GDN state cache; weight memory belongs to the caller.
    /// </remarks>
    internal static Qwen3MoeHybridTransformerModel BuildFromPrebuiltWeights(
        ModelConfig config,
        Qwen3MoeLayerWeights[] layers,
        float[] outputNormWeight,
        nint tokenEmbedWeight, QuantizationType tokenEmbedQuantType,
        nint outputWeight, QuantizationType outputQuantType, int outputOutputDim, int outputInputDim)
    {
        if (config.Architecture != Architecture.Qwen3MoeHybrid)
            throw new ArgumentException(
                $"Qwen3MoeHybridTransformerModel requires Architecture.Qwen3MoeHybrid, got {config.Architecture}.",
                nameof(config));
        if (config.HybridLayout is null)
            throw new ArgumentException("Qwen3MoeHybrid config must have HybridLayout populated.", nameof(config));
        if (config.GdnConfig is null)
            throw new ArgumentException("Qwen3MoeHybrid config must have GdnConfig populated.", nameof(config));
        if (config.Moe is null)
            throw new ArgumentException("Qwen3MoeHybrid config must have Moe populated.", nameof(config));
        if (layers.Length != config.NumLayers)
            throw new ArgumentException(
                $"layers length {layers.Length} != config.NumLayers {config.NumLayers}.", nameof(layers));

        var layout = config.HybridLayout!;

        var kvSlotForLayer = new int[config.NumLayers];
        int attentionLayerCount = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            kvSlotForLayer[i] = layout.LayerKind[i] == HybridLayerKind.Attention
                ? attentionLayerCount++
                : -1;
        }

        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        if ((ropeDim & 1) != 0)
            throw new ArgumentException(
                $"rope_dim={ropeDim} must be even for pair-wise rotation.", nameof(config));
        if (ropeDim > config.HeadDim)
            throw new ArgumentException(
                $"rope_dim={ropeDim} exceeds head_dim={config.HeadDim}.", nameof(config));

        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        int halfRope = ropeDim / 2;
        float[] ropeCos, ropeSin;
        if (attentionLayerCount > 0)
        {
            ropeCos = new float[config.MaxSequenceLength * halfRope];
            ropeSin = new float[config.MaxSequenceLength * halfRope];
            RoPE.PrecomputeFrequencyTable(config.MaxSequenceLength, ropeDim, ropeTheta, ropeCos, ropeSin);
        }
        else
        {
            ropeCos = Array.Empty<float>();
            ropeSin = Array.Empty<float>();
        }

        return new Qwen3MoeHybridTransformerModel(
            config, gguf: null, layers, outputNormWeight,
            tokenEmbedWeight, tokenEmbedQuantType,
            outputWeight, outputQuantType, outputOutputDim, outputInputDim,
            kvSlotForLayer, attentionLayerCount,
            ropeCos, ropeSin, ropeDim,
            threadPool: null, ownsPool: false);
    }

    private static Qwen3MoeLayerWeights LoadLayer(
        int layerIdx,
        nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config,
        List<nint> owned)
    {
        string prefix = $"blk.{layerIdx}";
        int hiddenSize = config.HiddenSize;
        var layout = config.HybridLayout!;

        var attnNormDesc = tensors[$"{prefix}.attn_norm.weight"];
        float[] attnNormWeight = DequantizeF32(dataBase, attnNormDesc, hiddenSize);

        // qwen35moe uses LLM_TENSOR_ATTN_POST_NORM → "blk.%d.post_attention_norm" for the
        // pre-MoE norm (verified against llama.cpp src/models/qwen35moe.cpp: tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i)).
        // This is *not* the standard "ffn_norm" name used by Qwen2/Qwen3 dense MoE.
        var postAttnNormDesc = tensors[$"{prefix}.post_attention_norm.weight"];
        float[] postAttnNormWeight = DequantizeF32(dataBase, postAttnNormDesc, hiddenSize);

        var tokenMixing = layout.LayerKind[layerIdx] switch
        {
            HybridLayerKind.GatedDeltaNet =>
                (gdn: LoadGdnLayer(prefix, dataBase, tensors, config), attn: (Qwen3FullAttnWeights?)null),
            HybridLayerKind.Attention =>
                (gdn: (GdnTokenMixingWeights?)null, attn: LoadFullAttnLayer(prefix, dataBase, tensors, config, layout.HeadCountKv[layerIdx])),
            _ => throw new InvalidOperationException(
                $"Unexpected HybridLayerKind {layout.LayerKind[layerIdx]} at layer {layerIdx} in Qwen3MoeHybrid."),
        };

        // skipRoutedF32Only: skip the F32 dequant of routed experts (~120 GB at Qwen3.6-35B-A3B
        // scale) while keeping shared-expert F32 (small, ~10 MB per layer). The routed experts'
        // raw quant views are populated; ForwardMoeBody dequantizes them on-demand per layer.
        MoeLayerWeights moe = TransformerWeights.LoadDeepSeekMoeLayer(
            layerIdx, dataBase, tensors, config, owned, skipRoutedF32Only: true);

        return new Qwen3MoeLayerWeights
        {
            AttnNormWeight = attnNormWeight,
            PostAttnNormWeight = postAttnNormWeight,
            Gdn = tokenMixing.gdn,
            FullAttn = tokenMixing.attn,
            Moe = moe,
        };
    }

    private static GdnTokenMixingWeights LoadGdnLayer(
        string prefix,
        nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config)
    {
        var gdn = config.GdnConfig!.Value;

        var qkvDesc = tensors[$"{prefix}.attn_qkv.weight"];
        var gateDesc = tensors[$"{prefix}.attn_gate.weight"];
        var aDesc = tensors[$"{prefix}.ssm_a"];
        var alphaDesc = tensors[$"{prefix}.ssm_alpha.weight"];
        var betaDesc = tensors[$"{prefix}.ssm_beta.weight"];
        var conv1dWDesc = tensors[$"{prefix}.ssm_conv1d.weight"];
        var dtBDesc = tensors[$"{prefix}.ssm_dt.bias"];
        var normDesc = tensors[$"{prefix}.ssm_norm.weight"];
        var outDesc = tensors[$"{prefix}.ssm_out.weight"];

        int convDim = (2 * gdn.NKHead + gdn.NVHead) * gdn.DState;
        float[] conv1dWeight = DequantizeF32(dataBase, conv1dWDesc, gdn.DConv * convDim);
        float[] conv1dBias = new float[convDim]; // GDN has no conv bias — zeros satisfy Conv1dCausal precondition
        float[] a = DequantizeF32(dataBase, aDesc, gdn.NVHead);
        float[] dtBias = DequantizeF32(dataBase, dtBDesc, gdn.NVHead);
        float[] ssmNormWeight = DequantizeF32(dataBase, normDesc, gdn.DState);

        return new GdnTokenMixingWeights
        {
            QkvWeight = dataBase + (nint)qkvDesc.DataOffset,
            QkvQuantType = qkvDesc.QuantizationType,
            QkvInputDim = qkvDesc.Shape[0],
            QkvOutputDim = qkvDesc.Shape[1],

            GateWeight = dataBase + (nint)gateDesc.DataOffset,
            GateQuantType = gateDesc.QuantizationType,
            GateInputDim = gateDesc.Shape[0],
            GateOutputDim = gateDesc.Shape[1],

            A = a,

            AlphaWeight = dataBase + (nint)alphaDesc.DataOffset,
            AlphaQuantType = alphaDesc.QuantizationType,
            AlphaInputDim = alphaDesc.Shape[0],
            AlphaOutputDim = alphaDesc.Shape[1],

            BetaWeight = dataBase + (nint)betaDesc.DataOffset,
            BetaQuantType = betaDesc.QuantizationType,
            BetaInputDim = betaDesc.Shape[0],
            BetaOutputDim = betaDesc.Shape[1],

            Conv1dWeight = conv1dWeight,
            Conv1dBias = conv1dBias,
            DtBias = dtBias,
            SsmNormWeight = ssmNormWeight,

            OutWeight = dataBase + (nint)outDesc.DataOffset,
            OutQuantType = outDesc.QuantizationType,
            OutInputDim = outDesc.Shape[0],
            OutOutputDim = outDesc.Shape[1],
        };
    }

    private static Qwen3FullAttnWeights LoadFullAttnLayer(
        string prefix,
        nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config,
        int numKvHeads)
    {
        var q = tensors[$"{prefix}.attn_q.weight"];
        var k = tensors[$"{prefix}.attn_k.weight"];
        var v = tensors[$"{prefix}.attn_v.weight"];
        var o = tensors[$"{prefix}.attn_output.weight"];

        // qwen35moe full-attn: attn_q.weight has output dim 2 * nQ * headDim (Q + Gate fused per head).
        // Verify shape so we don't silently regress if the GGUF naming convention shifts.
        int expectedQGateOut = 2 * config.NumAttentionHeads * config.HeadDim;
        if (q.Shape[1] != expectedQGateOut)
        {
            throw new InvalidDataException(
                $"{prefix}.attn_q.weight has output dim {q.Shape[1]} but qwen35moe expects " +
                $"{expectedQGateOut} = 2 * {config.NumAttentionHeads} * {config.HeadDim} (Q+Gate fused).");
        }

        // QK-norm tensors are required by llama.cpp's qwen35moe loader (no TENSOR_NOT_REQUIRED flag).
        float[] qNorm = DequantizeF32(dataBase, tensors[$"{prefix}.attn_q_norm.weight"], config.HeadDim);
        float[] kNorm = DequantizeF32(dataBase, tensors[$"{prefix}.attn_k_norm.weight"], config.HeadDim);

        return new Qwen3FullAttnWeights
        {
            QWeight = dataBase + (nint)q.DataOffset,
            QQuantType = q.QuantizationType,
            QInputDim = q.Shape[0],
            QOutputDim = q.Shape[1],

            KWeight = dataBase + (nint)k.DataOffset,
            KQuantType = k.QuantizationType,
            KInputDim = k.Shape[0],
            KOutputDim = k.Shape[1],

            VWeight = dataBase + (nint)v.DataOffset,
            VQuantType = v.QuantizationType,
            VInputDim = v.Shape[0],
            VOutputDim = v.Shape[1],

            OWeight = dataBase + (nint)o.DataOffset,
            OQuantType = o.QuantizationType,
            OInputDim = o.Shape[0],
            OOutputDim = o.Shape[1],

            NumKvHeads = numKvHeads,
            QNormWeight = qNorm,
            KNormWeight = kNorm,
        };
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null, gdnState: null);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache)
        => Forward(tokenIds, positions, deviceId, kvCache, gdnState: null);

    /// <summary>
    /// Runs a forward pass with optional KV-cache (for the GQA layers) and optional
    /// per-sequence GDN recurrent state (for the GDN layers). When
    /// <paramref name="gdnState"/> is <see langword="null"/>, falls back to the
    /// model-owned default cache — safe only for single-sequence dispatch from a
    /// freshly-constructed model. Multi-seq batched dispatch must supply a fresh
    /// per-seq <see cref="GdnStateCache"/> for each request, otherwise state leaks
    /// across sequences.
    /// </summary>
    /// <param name="tokenIds">Input token IDs.</param>
    /// <param name="positions">Position indices for each token.</param>
    /// <param name="deviceId">Target device for the returned tensor.</param>
    /// <param name="kvCache">Optional per-seq KV-cache for the GQA layers.</param>
    /// <param name="gdnState">
    /// Optional per-seq GDN recurrent state container. Must be a
    /// <see cref="GdnStateCache"/> sized for this model's GDN-layer count.
    /// </param>
    [SkipLocalsInit]
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache, IGdnState? gdnState)
    {
        // Resolve the GDN state: caller-supplied container preferred, model-owned
        // fallback for the single-seq Forward callers that pre-date the per-seq API.
        // The fallback is unsafe across multi-seq batched dispatch — that path is
        // expected to pass a fresh per-seq state via ForwardBatch.
        GdnStateCache gdnCache;
        if (gdnState is null)
        {
            gdnCache = _gdnCache;
        }
        else if (gdnState is GdnStateCache typed)
        {
            if (typed.NumGdnLayers != _gdnCache.NumGdnLayers)
                throw new ArgumentException(
                    $"GdnState NumGdnLayers ({typed.NumGdnLayers}) does not match model GDN-layer count ({_gdnCache.NumGdnLayers}).",
                    nameof(gdnState));
            gdnCache = typed;
        }
        else
        {
            throw new ArgumentException(
                $"Qwen3MoeHybridTransformerModel requires a CPU GdnStateCache; got {gdnState.GetType().Name}.",
                nameof(gdnState));
        }

        int seqLen = tokenIds.Length;
        if (seqLen == 0 || seqLen != positions.Length)
            throw new ArgumentException("tokenIds and positions must have equal, non-zero length.");

        int hiddenSize = Config.HiddenSize;
        int vocabSize = Config.VocabSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        float eps = Config.NormEpsilon;
        int maxSeq = Config.MaxSequenceLength;

        for (int i = 0; i < positions.Length; i++)
        {
            if ((uint)positions[i] >= (uint)maxSeq)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {positions[i]} at index {i} exceeds max sequence length {maxSeq}.");
        }

        _state.EnsureCapacity(seqLen);

        // Adaptive dispatch mode: spin-wait for decode (short, frequent dispatches),
        // event-based for prefill (long dispatches where kernel transition cost is negligible).
        _threadPool?.SetDispatchMode(seqLen == 1 ? DispatchMode.SpinWait : DispatchMode.EventBased);

        float* hidden = (float*)_state.HiddenState;
        float* residual = (float*)_state.Residual;
        float* normOut = (float*)_state.NormOutput;
        float* logits = (float*)_state.Logits;
        byte* inputQ8Scratch = (byte*)_state.InputQ8Scratch;
        float* qAttn = (float*)_state.QScratch;
        float* kAttn = (float*)_state.KScratch;
        float* vAttn = (float*)_state.VScratch;
        float* attnOut = (float*)_state.AttnOutput;

        EmbedTokens(tokenIds, hidden, hiddenSize);

        if (TensorDump.Enabled)
            TensorDump.Dump2D("token_embd", hidden, seqLen, hiddenSize);

        var kinds = _layout.LayerKind;
        for (int layer = 0; layer < _layers.Length; layer++)
        {
            var lw = _layers[layer];
            // ── Token-mixing sub-layer ─────────────────────────────────────────
            // Snapshot hidden as residual, then attn_norm.
            new Span<float>(hidden, seqLen * hiddenSize)
                .CopyTo(new Span<float>(residual, seqLen * hiddenSize));
            for (int t = 0; t < seqLen; t++)
            {
                RmsNorm.Execute(
                    new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                    lw.AttnNormWeight, eps,
                    new Span<float>(normOut + t * hiddenSize, hiddenSize));
            }
            if (TensorDump.Enabled)
                TensorDump.Dump2D($"blk.{layer}.attn_norm", normOut, seqLen, hiddenSize);

            if (kinds[layer] == HybridLayerKind.GatedDeltaNet)
                ForwardGdnBody(lw.Gdn!, layer, seqLen, hiddenSize, normOut, eps, gdnCache);
            else
                ForwardFullAttnBody(lw.FullAttn!, layer, seqLen, positions,
                    normOut, qAttn, kAttn, vAttn, attnOut,
                    numHeads, numKvHeads, headDim, kvCache);
            // First residual add: hidden = residual + normOut (token-mixing output).
            for (int t = 0; t < seqLen; t++)
            {
                Add.Execute(
                    new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize),
                    new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                    new Span<float>(hidden + t * hiddenSize, hiddenSize));
            }
            // ── MoE FFN sub-layer ─────────────────────────────────────────────
            // Snapshot updated hidden as residual again, then post_attn_norm.
            new Span<float>(hidden, seqLen * hiddenSize)
                .CopyTo(new Span<float>(residual, seqLen * hiddenSize));
            for (int t = 0; t < seqLen; t++)
            {
                RmsNorm.Execute(
                    new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                    lw.PostAttnNormWeight, eps,
                    new Span<float>(normOut + t * hiddenSize, hiddenSize));
            }

            if (TensorDump.Enabled)
                TensorDump.Dump2D($"blk.{layer}.attn_post_norm", normOut, seqLen, hiddenSize);

            ForwardMoeBody(lw.Moe, seqLen, hiddenSize, normOut);
            if (TensorDump.Enabled)
                TensorDump.Dump2D($"blk.{layer}.ffn_out", normOut, seqLen, hiddenSize);

            // Second residual add.
            for (int t = 0; t < seqLen; t++)
            {
                Add.Execute(
                    new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize),
                    new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                    new Span<float>(hidden + t * hiddenSize, hiddenSize));
            }
            if (TensorDump.Enabled)
                TensorDump.Dump2D($"blk.{layer}.l_out", hidden, seqLen, hiddenSize);
        }

        // Final output norm + logit projection.
        for (int t = 0; t < seqLen; t++)
        {
            RmsNorm.Execute(
                new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                _outputNormWeight, eps,
                new Span<float>(hidden + t * hiddenSize, hiddenSize));
        }
        if (TensorDump.Enabled)
            TensorDump.Dump2D("result_norm", hidden, seqLen, hiddenSize);

        Gemm(_outputWeight, _outputQuantType, hidden, logits,
             _outputOutputDim, _outputInputDim, seqLen, preQuantizedInput: null);

        if (TensorDump.Enabled)
            TensorDump.Dump2D("result_output", logits, seqLen, vocabSize);

        var shape = new TensorShape(seqLen, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId);
        new Span<float>(logits, seqLen * vocabSize).CopyTo(
            new Span<float>((void*)result.DataPointer, seqLen * vocabSize));

        return result;
    }

    /// <summary>
    /// Per-sequence loop over <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?, IGdnState?)"/>
    /// — threads each request's GDN state through to the GDN scan, so multi-seq batched
    /// dispatch is safe as long as every request supplies a fresh
    /// <see cref="GdnStateCache"/>. The default <see cref="IModel.ForwardBatch"/> would
    /// silently corrupt state across sequences because it ignores
    /// <see cref="SequenceForwardRequest.GdnState"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Qwen3MoeHybrid carries no LoRA path today, so adapter-bearing requests throw
    /// up-front — same shape as the Vulkan host's override. Per-seq fusion across
    /// the GDN scan + MoE routing is not viable (the scan is per-token recurrent and
    /// the MoE routing mask is per-token); this override simply loops Forward,
    /// trading the per-iter dispatch-overhead amortisation for correctness.
    /// </para>
    /// </remarks>
    public IReadOnlyList<ITensor> ForwardBatch(
        IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0) return Array.Empty<ITensor>();

        for (int i = 0; i < requests.Count; i++)
        {
            if (requests[i].Adapter is not null)
                throw new NotSupportedException(
                    "Qwen3MoeHybridTransformerModel.ForwardBatch does not support LoRA adapters " +
                    "(no Qwen3MoeHybrid LoRA path today). Re-issue the request without an adapter.");
        }

        var results = new ITensor[requests.Count];
        for (int i = 0; i < requests.Count; i++)
        {
            var r = requests[i];
            results[i] = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache, r.GdnState);
        }
        return results;
    }

    /// <summary>
    /// GDN (Gated DeltaNet) token-mixing forward pass. Reads pre-normed activations from
    /// <paramref name="normOut"/> and writes the <c>ssm_out</c> projection back to the same buffer.
    /// Advances the per-layer GDN conv and associative-memory state in place.
    /// </summary>
    /// <remarks>
    /// Operation order (confirmed from llama.cpp <c>qwen35moe.cpp</c>):
    /// <list type="number">
    ///   <item>Project <c>attn_qkv</c>, <c>attn_gate</c>, <c>ssm_alpha</c>, <c>ssm_beta</c> from <c>input</c>.</item>
    ///   <item>Sigmoid(<c>beta</c>); compute decay <c>g = exp(softplus(alpha + dt_bias) × A)</c>.</item>
    ///   <item>Conv1d on QKV concat (prepend rolling conv state, causal 1-D, SiLU).</item>
    ///   <item>De-interleave conv output into Q, K, V; L2-normalise both Q and K.</item>
    ///   <item><see cref="GatedDeltaNetScan.Execute"/> → GDN output.</item>
    ///   <item>Per-head <c>RMSNorm(out, ssm_norm_weight) × silu(z)</c> gating.</item>
    ///   <item><c>ssm_out</c> projection back into <paramref name="normOut"/>.</item>
    /// </list>
    /// </remarks>
    [SkipLocalsInit]
    private void ForwardGdnBody(
        GdnTokenMixingWeights gdnW, int absoluteLayerIdx, int seqLen,
        int hiddenSize, float* normOut, float eps, GdnStateCache gdnCache)
    {
        int nVHead = _gdn.NVHead;
        int nKHead = _gdn.NKHead;
        int dState = _gdn.DState;
        int dConv = _gdn.DConv;
        int convDim = (2 * nKHead + nVHead) * dState;
        int vDim = nVHead * dState;   // NVHead*DState per token
        int kDim = nKHead * dState;   // NKHead*DState per token

        float* qkvBuf = (float*)_state.GdnQkvBuf;
        float* zBuf = (float*)_state.GdnZBuf;
        float* alphaBuf = (float*)_state.GdnAlphaBuf;
        float* betaBuf = (float*)_state.GdnBetaBuf;
        float* qBuf = (float*)_state.GdnQBuf;
        float* kBuf = (float*)_state.GdnKBuf;
        float* vBuf = (float*)_state.GdnVBuf;
        float* gdnOut = (float*)_state.GdnOut;
        float* convInput = (float*)_state.GdnConvInput;

        int gdnOrdinal = _gdnLayerOrdinal[absoluteLayerIdx];

        // ── 1. Projections from normed input ──────────────────────────────────
        // All four projections read from normOut (the attn_norm output).
        Gemm(gdnW.QkvWeight, gdnW.QkvQuantType, normOut, qkvBuf,
             gdnW.QkvOutputDim, gdnW.QkvInputDim, seqLen, preQuantizedInput: null);
        Gemm(gdnW.GateWeight, gdnW.GateQuantType, normOut, zBuf,
             gdnW.GateOutputDim, gdnW.GateInputDim, seqLen, preQuantizedInput: null);
        Gemm(gdnW.AlphaWeight, gdnW.AlphaQuantType, normOut, alphaBuf,
             gdnW.AlphaOutputDim, gdnW.AlphaInputDim, seqLen, preQuantizedInput: null);
        Gemm(gdnW.BetaWeight, gdnW.BetaQuantType, normOut, betaBuf,
             gdnW.BetaOutputDim, gdnW.BetaInputDim, seqLen, preQuantizedInput: null);

        if (TensorDump.Enabled)
        {
            TensorDump.Dump2D($"blk.{absoluteLayerIdx}.linear_attn_qkv_mixed", qkvBuf, seqLen, convDim);
            TensorDump.Dump2D($"blk.{absoluteLayerIdx}.z", zBuf, seqLen, vDim);
            TensorDump.Dump2D($"blk.{absoluteLayerIdx}.alpha_proj", alphaBuf, seqLen, nVHead);
            TensorDump.Dump2D($"blk.{absoluteLayerIdx}.beta_proj", betaBuf, seqLen, nVHead);
        }

        // ── 2. Compute decay g and write-gate beta ────────────────────────────
        // g[t,vh] = exp(softplus(alpha[t,vh] + DtBias[vh]) * A[vh])
        for (int t = 0; t < seqLen; t++)
        {
            int gbOff = t * nVHead;
            for (int vh = 0; vh < nVHead; vh++)
            {
                float alpha = alphaBuf[gbOff + vh] + gdnW.DtBias[vh];
                float sp = MathF.Log(1f + MathF.Exp(alpha)); // softplus
                alphaBuf[gbOff + vh] = MathF.Exp(sp * gdnW.A[vh]);
            }
        }
        // beta = sigmoid(beta_proj)
        TensorPrimitives.Sigmoid(
            new ReadOnlySpan<float>(betaBuf, seqLen * nVHead),
            new Span<float>(betaBuf, seqLen * nVHead));

        if (TensorDump.Enabled)
        {
            TensorDump.Dump2D($"blk.{absoluteLayerIdx}.g", alphaBuf, seqLen, nVHead);
            TensorDump.Dump2D($"blk.{absoluteLayerIdx}.beta_sigmoid", betaBuf, seqLen, nVHead);
        }

        // ── 3. Conv1d on QKV concat ────────────────────────────────────────────
        // Fill ConvInput: [conv_state (DConv-1 rows) | qkvBuf (seqLen rows)]
        var convState = gdnCache.GetConvState(gdnOrdinal);
        convState.CopyTo(new Span<float>(convInput, (dConv - 1) * convDim));
        for (int t = 0; t < seqLen; t++)
        {
            new ReadOnlySpan<float>(qkvBuf + t * convDim, convDim)
                .CopyTo(new Span<float>(convInput + (dConv - 1 + t) * convDim, convDim));
        }

        // Conv1d → qkvBuf (reuse as output), then SiLU in place.
        int convInputElems = (dConv - 1 + seqLen) * convDim;
        Conv1dCausal.Execute(
            input: new ReadOnlySpan<float>(convInput, convInputElems),
            weight: gdnW.Conv1dWeight,
            bias: gdnW.Conv1dBias,
            output: new Span<float>(qkvBuf, seqLen * convDim),
            dConv: dConv,
            channels: convDim,
            seqLen: seqLen);
        SiLu.Execute(
            new ReadOnlySpan<float>(qkvBuf, seqLen * convDim),
            new Span<float>(qkvBuf, seqLen * convDim));

        if (TensorDump.Enabled)
            TensorDump.Dump2D($"blk.{absoluteLayerIdx}.conv_output_silu", qkvBuf, seqLen, convDim);

        // Save the trailing (dConv-1) rows of convInput back as rolling state.
        for (int r = 0; r < dConv - 1; r++)
        {
            new ReadOnlySpan<float>(convInput + (seqLen + r) * convDim, convDim)
                .CopyTo(convState.Slice(r * convDim, convDim));
        }

        // ── 4. De-interleave Q/K/V and L2-normalise Q and K ──────────────────
        // Conv output layout per token: [Q (kDim) | K (kDim) | V (vDim)]
        for (int t = 0; t < seqLen; t++)
        {
            float* row = qkvBuf + t * convDim;
            new ReadOnlySpan<float>(row,          kDim).CopyTo(new Span<float>(qBuf + t * kDim, kDim));
            new ReadOnlySpan<float>(row + kDim,   kDim).CopyTo(new Span<float>(kBuf + t * kDim, kDim));
            new ReadOnlySpan<float>(row + 2 * kDim, vDim).CopyTo(new Span<float>(vBuf + t * vDim, vDim));
        }
        if (TensorDump.Enabled)
        {
            TensorDump.Dump3D($"blk.{absoluteLayerIdx}.q_conv", qBuf, seqLen, nKHead, dState);
            TensorDump.Dump3D($"blk.{absoluteLayerIdx}.k_conv", kBuf, seqLen, nKHead, dState);
            TensorDump.Dump3D($"blk.{absoluteLayerIdx}.v_conv", vBuf, seqLen, nVHead, dState);
        }
        GatedDeltaNetScan.L2NormalizeHeads(new Span<float>(qBuf, seqLen * kDim), dState);
        GatedDeltaNetScan.L2NormalizeHeads(new Span<float>(kBuf, seqLen * kDim), dState);
        if (TensorDump.Enabled)
        {
            TensorDump.Dump3D($"blk.{absoluteLayerIdx}.q_conv_predelta", qBuf, seqLen, nKHead, dState);
            TensorDump.Dump3D($"blk.{absoluteLayerIdx}.k_conv_predelta", kBuf, seqLen, nKHead, dState);
        }

        // ── 5. GDN scan ───────────────────────────────────────────────────────
        var gdnState = gdnCache.GetGdnState(gdnOrdinal);
        GatedDeltaNetScan.Execute(
            state: gdnState,
            q: new ReadOnlySpan<float>(qBuf, seqLen * kDim),
            k: new ReadOnlySpan<float>(kBuf, seqLen * kDim),
            v: new ReadOnlySpan<float>(vBuf, seqLen * vDim),
            g: new ReadOnlySpan<float>(alphaBuf, seqLen * nVHead),
            beta: new ReadOnlySpan<float>(betaBuf, seqLen * nVHead),
            output: new Span<float>(gdnOut, seqLen * vDim),
            nVHead: nVHead,
            nKHead: nKHead,
            dState: dState,
            seqLen: seqLen);
        if (TensorDump.Enabled)
            TensorDump.Dump3D($"blk.{absoluteLayerIdx}.attn_output", gdnOut, seqLen, nVHead, dState);

        // ── 6. Per-head RMSNorm(out) * silu(z) gating ─────────────────────────
        // ssm_norm_weight [dState] is broadcast across all heads.
        for (int t = 0; t < seqLen; t++)
        {
            int tBase = t * vDim;
            for (int vh = 0; vh < nVHead; vh++)
            {
                int headOff = tBase + vh * dState;
                // RMSNorm in place with shared norm weight.
                RmsNorm.Execute(
                    new ReadOnlySpan<float>(gdnOut + headOff, dState),
                    gdnW.SsmNormWeight, eps,
                    new Span<float>(gdnOut + headOff, dState));
                // Multiply by silu(z[head]).
                float* zHead = zBuf + headOff;
                float* outHead = gdnOut + headOff;
                for (int i = 0; i < dState; i++)
                {
                    float zi = zHead[i];
                    outHead[i] *= zi * (1f / (1f + MathF.Exp(-zi))); // silu(z) = z * sigmoid(z)
                }
            }
        }
        if (TensorDump.Enabled)
            TensorDump.Dump3D($"blk.{absoluteLayerIdx}.final_output", gdnOut, seqLen, nVHead, dState);

        // ── 7. ssm_out projection into normOut ────────────────────────────────
        Gemm(gdnW.OutWeight, gdnW.OutQuantType, gdnOut, normOut,
             gdnW.OutOutputDim, gdnW.OutInputDim, seqLen, preQuantizedInput: null);

        if (TensorDump.Enabled)
            TensorDump.Dump2D($"blk.{absoluteLayerIdx}.linear_attn_out", normOut, seqLen, hiddenSize);
    }

    /// <summary>
    /// Full GQA attention forward (qwen35moe variant). Reads pre-normed activations from
    /// <paramref name="normOut"/> and writes the gated output projection back to the same buffer.
    /// </summary>
    /// <remarks>
    /// Operation order (verified against llama.cpp <c>build_layer_attn</c> in qwen35moe.cpp):
    /// <list type="number">
    ///   <item>Fused QG projection: <c>QG = attn_q @ norm_in</c>, output dim <c>2 * nQ * headDim</c>
    ///         interleaved per head as <c>[Q_h0, Gate_h0, Q_h1, Gate_h1, ...]</c>.</item>
    ///   <item>De-interleave QG into Q (per-head offset 0) and Gate (per-head offset <c>headDim</c>).</item>
    ///   <item>Q RMSNorm with <c>attn_q_norm</c>.</item>
    ///   <item>K = <c>attn_k @ norm_in</c>, then K RMSNorm with <c>attn_k_norm</c>.</item>
    ///   <item>V = <c>attn_v @ norm_in</c>.</item>
    ///   <item>RoPE on Q and K (text-only mRoPE with all-equal positions collapses to single-axis RoPE
    ///         over the rotary partial-dim slice).</item>
    ///   <item>Standard GQA attention.</item>
    ///   <item>Multiply attention output element-wise by <c>sigmoid(Gate)</c>.</item>
    ///   <item>Output projection <c>attn_output @ gated_attn</c>.</item>
    /// </list>
    /// </remarks>
    private void ForwardFullAttnBody(
        Qwen3FullAttnWeights attn, int layer, int seqLen, ReadOnlySpan<int> positions,
        float* normOut, float* q, float* k, float* v, float* attnOut,
        int numHeads, int numKvHeads, int headDim, IKvCache? kvCache)
    {
        int qElems = numHeads * headDim;
        int qgElems = 2 * qElems;
        float* qgBuf = (float*)_state.QGateScratch;
        float* gate = (float*)_state.GateScratch;

        // 1. Fused Q+Gate projection.
        Gemm(attn.QWeight, attn.QQuantType, normOut, qgBuf, attn.QOutputDim, attn.QInputDim, seqLen, preQuantizedInput: null);
        if (TensorDump.Enabled)
            TensorDump.Dump2D($"blk.{layer}.fa_qg", qgBuf, seqLen, qgElems);

        // 2. De-interleave QG → Q and Gate. Layout per token: [Q_h0(headDim), Gate_h0(headDim), Q_h1, Gate_h1, ...].
        //    Each head occupies 2*headDim contiguous floats in qgBuf with Q first, Gate second.
        for (int t = 0; t < seqLen; t++)
        {
            float* qgRow = qgBuf + (long)t * qgElems;
            float* qRow = q + (long)t * qElems;
            float* gRow = gate + (long)t * qElems;
            for (int h = 0; h < numHeads; h++)
            {
                int qgHeadOff = h * 2 * headDim;
                int hOff = h * headDim;
                new ReadOnlySpan<float>(qgRow + qgHeadOff, headDim)
                    .CopyTo(new Span<float>(qRow + hOff, headDim));
                new ReadOnlySpan<float>(qgRow + qgHeadOff + headDim, headDim)
                    .CopyTo(new Span<float>(gRow + hOff, headDim));
            }
        }

        if (TensorDump.Enabled)
        {
            TensorDump.Dump2D($"blk.{layer}.fa_q_split", q, seqLen, numHeads * headDim);
            TensorDump.Dump2D($"blk.{layer}.fa_gate_split", gate, seqLen, numHeads * headDim);
        }

        // 3. K and V projections.
        Gemm(attn.KWeight, attn.KQuantType, normOut, k, attn.KOutputDim, attn.KInputDim, seqLen, preQuantizedInput: null);
        Gemm(attn.VWeight, attn.VQuantType, normOut, v, attn.VOutputDim, attn.VInputDim, seqLen, preQuantizedInput: null);
        if (TensorDump.Enabled)
        {
            TensorDump.Dump2D($"blk.{layer}.fa_k", k, seqLen, numKvHeads * headDim);
            TensorDump.Dump2D($"blk.{layer}.fa_v", v, seqLen, numKvHeads * headDim);
        }

        // 4. Per-head QK-norm (Qwen3 convention: normalise Q and K before RoPE).
        Mamba3QkNorm.Execute(
            new Span<float>(q, seqLen * qElems),
            attn.QNormWeight, Config.NormEpsilon, seqLen, numHeads, headDim);
        Mamba3QkNorm.Execute(
            new Span<float>(k, seqLen * numKvHeads * headDim),
            attn.KNormWeight, Config.NormEpsilon, seqLen, numKvHeads, headDim);
        if (TensorDump.Enabled)
        {
            TensorDump.Dump2D($"blk.{layer}.fa_q_postnorm", q, seqLen, qElems);
            TensorDump.Dump2D($"blk.{layer}.fa_k_postnorm", k, seqLen, numKvHeads * headDim);
        }

        // 5. RoPE — partial-rotary over the first ropeDim of each head.
        //    NOTE: qwen35moe uses mRoPE with sections [11,11,10,0] and mrope_interleaved=true. For text-only
        //    inference (positions identical across all 4 axes), this collapses to single-axis RoPE.
        //    Pair pattern is NeoX (HuggingFace rotate_half) — verified by the synthetic CPU↔CUDA parity
        //    fixture in tests/DotLLM.Tests.Unit/Cuda/CudaQwen3MoeHybridParityTests.cs.
        int kvStride = numKvHeads * headDim;
        RoPE.Execute(
            new Span<float>(q, seqLen * qElems),
            new Span<float>(k, seqLen * kvStride),
            positions,
            numHeads, numKvHeads, headDim, _ropeDim,
            _ropeCosTable, _ropeSinTable, RoPEType.NeoX);
        if (TensorDump.Enabled)
        {
            TensorDump.Dump2D($"blk.{layer}.fa_q_postrope", q, seqLen, qElems);
            TensorDump.Dump2D($"blk.{layer}.fa_k_postrope", k, seqLen, kvStride);
        }

        // 6. Attention.
        if (kvCache is not null)
        {
            int kvSlot = _kvSlotForLayer[layer];
            if (kvSlot < 0)
                throw new InvalidOperationException(
                    $"Layer {layer} has no KV-cache slot.");

            var kRef = new TensorRef(seqLen, kvStride, DType.Float32, -1, (nint)k);
            var vRef = new TensorRef(seqLen, kvStride, DType.Float32, -1, (nint)v);
            kvCache.Update(kRef, vRef, positions, kvSlot);

            int seqKv = kvCache.CurrentLength;
            var cachedK = kvCache.GetKeysRef(kvSlot);
            var cachedV = kvCache.GetValuesRef(kvSlot);

            Attention.Execute(q, (float*)cachedK.DataPointer, (float*)cachedV.DataPointer, attnOut,
                seqLen, seqKv, numHeads, numKvHeads, headDim, positions[0], pool: null,
                slidingWindowSize: null);
        }
        else
        {
            Attention.Execute(q, k, v, attnOut,
                seqLen, seqLen, numHeads, numKvHeads, headDim, 0, pool: null,
                slidingWindowSize: null);
        }
        if (TensorDump.Enabled)
            TensorDump.Dump2D($"blk.{layer}.fa_attnout_pregate", attnOut, seqLen, qElems);

        // 7. Apply sigmoid(gate) element-wise to attention output BEFORE O-proj.
        //    sigmoid(x) = 1 / (1 + exp(-x)); fused into one pass over qElems-sized rows.
        for (int t = 0; t < seqLen; t++)
        {
            float* aRow = attnOut + (long)t * qElems;
            float* gRow = gate + (long)t * qElems;
            for (int i = 0; i < qElems; i++)
            {
                float gi = gRow[i];
                aRow[i] *= 1f / (1f + MathF.Exp(-gi));
            }
        }
        if (TensorDump.Enabled)
            TensorDump.Dump2D($"blk.{layer}.fa_attnout_postgate", attnOut, seqLen, qElems);

        // 8. Output projection.
        Gemm(attn.OWeight, attn.OQuantType, attnOut, normOut, attn.OOutputDim, attn.OInputDim, seqLen, preQuantizedInput: null);
    }

    /// <summary>
    /// MoE SwiGLU FFN forward. Reads pre-normed activations from <paramref name="normOut"/>
    /// and writes the weighted-sum expert output back to the same buffer.
    /// </summary>
    /// <summary>
    /// MoE SwiGLU FFN forward (qwen35moe variant). Reads pre-normed activations from
    /// <paramref name="normOut"/> and writes the routed + shared expert output back to the same buffer.
    /// </summary>
    /// <remarks>
    /// qwen35moe uses the Qwen1.5-MoE shared-expert convention: a single shared SwiGLU expert whose
    /// output is gated by a per-token sigmoid (loaded from <c>ffn_gate_inp_shexp.weight</c> into
    /// <see cref="MoeLayerWeights.SharedExpertGate"/>). When that gate is absent the call degenerates
    /// to the routed-only Mixtral path.
    /// </remarks>
    private void ForwardMoeBody(MoeLayerWeights moe, int seqLen, int hiddenSize, float* normOut)
    {
        int numExperts = moe.NumExperts;
        int numExpertsPerTok = moe.NumExpertsPerTok;
        int intermediate = moe.IntermediateSize;
        int totalAssignments = seqLen * numExpertsPerTok;

        // Phase 1: routing + bucketing. Reuses persistent state-side arrays grown by
        // EnsureCapacity; the bucket cursor / unique-expert arrays are sized for numExperts
        // (small constant) at construction time.
        Span<int> assignExpert = _state.MoeAssignExpert.AsSpan(0, totalAssignments);
        Span<float> assignWeight = _state.MoeAssignWeight.AsSpan(0, totalAssignments);
        Span<int> bucketCursors = _state.MoeBucketCursors.AsSpan(0, numExperts + 1);
        Span<int> bucketTokens = _state.MoeBucketTokens.AsSpan(0, totalAssignments);
        Span<int> bucketSlots = _state.MoeBucketSlots.AsSpan(0, totalAssignments);
        Span<int> uniqueExperts = _state.MoeUniqueExperts.AsSpan(0, Math.Min(numExperts, _state.MoeUniqueExperts.Length));

        int uniqueCount = MoeSwiGluMlp.Route(
            hidden: new ReadOnlySpan<float>(normOut, seqLen * hiddenSize),
            gateWeights: moe.Gate,
            assignExpert: assignExpert,
            assignWeight: assignWeight,
            bucketCursors: bucketCursors,
            bucketTokens: bucketTokens,
            bucketSlots: bucketSlots,
            uniqueExperts: uniqueExperts,
            numExperts: numExperts,
            numExpertsPerTok: numExpertsPerTok,
            hiddenSize: hiddenSize,
            seqLen: seqLen,
            normTopKProb: moe.NormTopKProb);

        // Phase 2: per-expert SwiGLU directly against the raw GGUF quant view (no per-call
        // dequant). When the load path didn't populate the raw view (synthetic F32 fixtures,
        // safetensors), fall through to the F32 per-expert pointer arrays in moe.W1/W2/W3.
        ReadOnlySpan<float> sharedGateSpan = moe.SharedExpertGate is not null
            ? moe.SharedExpertGate.AsSpan()
            : ReadOnlySpan<float>.Empty;

        bool useRawQuantView = moe.HasRawQuantView;
        nint gateBase = useRawQuantView ? moe.GateExpsRaw : 0;
        nint upBase = useRawQuantView ? moe.UpExpsRaw : 0;
        nint downBase = useRawQuantView ? moe.DownExpsRaw : 0;
        QuantizationType gateQt = useRawQuantView ? moe.GateExpsRawQt : QuantizationType.F32;
        QuantizationType upQt = useRawQuantView ? moe.UpExpsRawQt : QuantizationType.F32;
        QuantizationType downQt = useRawQuantView ? moe.DownExpsRawQt : QuantizationType.F32;

        // Per-expert byte stride into the fused gate/up/down tensors. The slice for expert e
        // is at base + e * RowByteSize(M*K, qt) — i.e. M * RowByteSize(K, qt) for valid quant
        // data (K divisible by the block size).
        long gateRowBytes = useRawQuantView
            ? Dequantize.RowByteSize((long)intermediate * hiddenSize, gateQt)
            : 0;
        long upRowBytes = useRawQuantView
            ? Dequantize.RowByteSize((long)intermediate * hiddenSize, upQt)
            : 0;
        long downRowBytes = useRawQuantView
            ? Dequantize.RowByteSize((long)hiddenSize * intermediate, downQt)
            : 0;

        // F32 per-expert pointer overrides — populated only on the synthetic / safetensors path
        // where weights are discontiguous F32 allocations. Empty when we're going through the
        // strided raw quant view.
        ReadOnlySpan<nint> gateF32Ptrs = useRawQuantView ? ReadOnlySpan<nint>.Empty : moe.W1;
        ReadOnlySpan<nint> upF32Ptrs = useRawQuantView ? ReadOnlySpan<nint>.Empty : moe.W3;
        ReadOnlySpan<nint> downF32Ptrs = useRawQuantView ? ReadOnlySpan<nint>.Empty : moe.W2;

        MoeSwiGluMlp.ExecuteRoutedFromAssignments(
            hidden: new ReadOnlySpan<float>(normOut, seqLen * hiddenSize),
            gateExpsRawBase: gateBase, gateExpsQt: gateQt, gateExpsRowBytes: gateRowBytes, gateExpsF32Ptrs: gateF32Ptrs,
            upExpsRawBase: upBase, upExpsQt: upQt, upExpsRowBytes: upRowBytes, upExpsF32Ptrs: upF32Ptrs,
            downExpsRawBase: downBase, downExpsQt: downQt, downExpsRowBytes: downRowBytes, downExpsF32Ptrs: downF32Ptrs,
            assignExpert: assignExpert,
            assignWeight: assignWeight,
            bucketCursors: bucketCursors,
            bucketTokens: bucketTokens,
            bucketSlots: bucketSlots,
            uniqueExperts: uniqueExperts,
            uniqueExpertCount: uniqueCount,
            output: new Span<float>(normOut, seqLen * hiddenSize),
            numExperts: numExperts,
            numExpertsPerTok: numExpertsPerTok,
            hiddenSize: hiddenSize,
            intermediateSize: intermediate,
            seqLen: seqLen,
            sharedGateProj: moe.SharedGateProj,
            sharedUpProj: moe.SharedUpProj,
            sharedDownProj: moe.SharedDownProj,
            sharedIntermediateSize: moe.SharedIntermediateSize,
            sharedExpertGate: sharedGateSpan,
            loraAdapter: null,
            loraLayer: 0,
            threadPool: _threadPool);
    }

    private void EmbedTokens(ReadOnlySpan<int> tokenIds, float* hidden, int hiddenSize)
    {
        nint embPtr = _tokenEmbedWeight;
        var qt = _tokenEmbedQuantType;

        for (int t = 0; t < tokenIds.Length; t++)
        {
            int tokenId = tokenIds[t];
            if ((uint)tokenId >= (uint)Config.VocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokenIds),
                    $"Token ID {tokenId} at position {t} is out of range [0, {Config.VocabSize}).");

            float* dest = hidden + t * hiddenSize;
            var destSpan = new Span<float>(dest, hiddenSize);

            if (qt == QuantizationType.F32)
            {
                new ReadOnlySpan<float>((float*)embPtr + (long)tokenId * hiddenSize, hiddenSize)
                    .CopyTo(destSpan);
            }
            else if (qt == QuantizationType.F16)
            {
                TensorPrimitives.ConvertToSingle(
                    new ReadOnlySpan<Half>((Half*)embPtr + (long)tokenId * hiddenSize, hiddenSize),
                    destSpan);
            }
            else
            {
                long rowBytes = Dequantize.RowByteSize(hiddenSize, qt);
                Dequantize.ToFloat32(embPtr + (nint)((long)tokenId * rowBytes), hiddenSize, qt, destSpan);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Gemm(nint weights, QuantizationType qt, float* b, float* c,
                      int m, int k, int n, byte* preQuantizedInput)
    {
        // Pool-aware GEMM dispatch — mirrors TransformerModel.Gemm. Passes the model's
        // ComputeThreadPool so projection matmuls (Q/K/V/O, GDN QKV/Gate/Alpha/Beta/Out,
        // output head) parallelise across rows. The MoE expert FFN uses a separate
        // parallel path inside MoeSwiGluMlp.ExecuteRoutedFromAssignments.
        switch (qt)
        {
            case QuantizationType.Q8_0:
                MatMul.GemmQ8_0((byte*)weights, b, c, m, k, n, _threadPool, preQuantizedInput);
                return;
            case QuantizationType.Q5_0:
                MatMul.GemmQ5_0((byte*)weights, b, c, m, k, n, _threadPool, preQuantizedInput);
                return;
            case QuantizationType.Q4_K:
                MatMul.GemmQ4_K((byte*)weights, b, c, m, k, n, _threadPool, preQuantizedInput);
                return;
            case QuantizationType.Q5_K:
                MatMul.GemmQ5_K((byte*)weights, b, c, m, k, n, _threadPool, preQuantizedInput);
                return;
            case QuantizationType.Q6_K:
                MatMul.GemmQ6_K((byte*)weights, b, c, m, k, n, _threadPool, preQuantizedInput);
                return;
            case QuantizationType.F32:
                MatMul.GemmF32((float*)weights, b, c, m, k, n, _threadPool);
                return;
            case QuantizationType.F16:
                MatMul.GemmF16(weights, b, c, m, k, n, _threadPool);
                return;
            default:
                GemmDequantFallback(weights, qt, b, c, m, k, n);
                return;
        }
    }

    private static void GemmDequantFallback(nint weights, QuantizationType qt, float* b, float* c,
                                            int m, int k, int n)
    {
        long rowBytes = Dequantize.RowByteSize(k, qt);
        float[] rowBuf = ArrayPool<float>.Shared.Rent(k);
        try
        {
            var rowSpan = rowBuf.AsSpan(0, k);
            for (int t = 0; t < n; t++)
            {
                var xSpan = new ReadOnlySpan<float>(b + t * k, k);
                for (int i = 0; i < m; i++)
                {
                    Dequantize.ToFloat32(weights + i * (nint)rowBytes, k, qt, rowSpan);
                    c[t * m + i] = TensorPrimitives.Dot(new ReadOnlySpan<float>(rowBuf, 0, k), xSpan);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rowBuf);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_ownsThreadPool)
            _threadPool?.Dispose();
        _state.Dispose();
        _gdnCache.Dispose();
        GC.SuppressFinalize(this);
    }

    private static float[] DequantizeF32(nint dataBase, GgufTensorDescriptor desc, int expectedSize)
    {
        nint ptr = dataBase + (nint)desc.DataOffset;
        float[] result = new float[expectedSize];
        Dequantize.ToFloat32(ptr, expectedSize, desc.QuantizationType, result);
        return result;
    }
}
