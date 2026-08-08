using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
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
/// Qwen3HybridDense (<c>qwen35</c>) model — Gated DeltaNet recurrence + dense SwiGLU FFN.
/// First observed in PrismML's Bonsai-27B (a PQ2_0 ternary model distilled from
/// Qwen/Qwen3.6-27B).
///
/// Each layer has:
///   - a token-mixing path: GDN (Gated DeltaNet) or full GQA attention, and
///   - a dense SwiGLU FFN (always present, every layer — no MoE routing at all).
/// Full-attention layers occur every <see cref="GatedDeltaNetConfig.FullAttnInterval"/> steps
/// (1-indexed), e.g. layers 4, 8, 12, … for interval=4.
///
/// Adapted from <see cref="Qwen3MoeHybridTransformerModel"/>: the GDN/full-attention
/// token-mixing paths, RoPE precompute, and embed/output projections lift verbatim (identical
/// tensor naming and forward semantics, confirmed against real Bonsai-27B GGUF tensor names).
/// The only structural difference is the FFN sublayer — dense gate/up/down instead of sparse
/// MoE routing — and the corresponding per-model quant dispatch gaining a <c>PQ2_0</c> arm
/// (Bonsai's ternary quant type; see <c>MatMul.PQ2S.cs</c>).
/// </summary>
public sealed unsafe class Qwen3HybridDenseTransformerModel : IModel
{
    private readonly GgufFile? _gguf; // kept alive; null when built from prebuilt weights
    private readonly Qwen3HybridDenseLayerWeights[] _layers;
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

    private readonly Qwen3HybridDenseForwardState _state;
    private readonly GdnStateCache _gdnCache;

    private readonly ComputeThreadPool? _threadPool;
    private readonly bool _ownsThreadPool;

    // Multi-Token Prediction (MTP / "NextN") head — issue #253. Null for every GGUF without a
    // nextn.* tensor group, which is the overwhelming majority: every other field and code path
    // in this class is completely unaffected by MTP being absent.
    private readonly MtpHeadWeights? _mtpHead;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _state.AllocatedBytes + _gdnCache.AllocatedBytes;

    /// <summary>Number of full-attention layers — the matching sparse KV-cache slot count.</summary>
    public int AttentionLayerCount => _attentionLayerCount;

    /// <inheritdoc/>
    public bool SupportsMtp => _mtpHead is not null;

    private Qwen3HybridDenseTransformerModel(
        ModelConfig config,
        GgufFile? gguf,
        Qwen3HybridDenseLayerWeights[] layers,
        float[] outputNormWeight,
        nint tokenEmbedWeight, QuantizationType tokenEmbedQuantType,
        nint outputWeight, QuantizationType outputQuantType, int outputOutputDim, int outputInputDim,
        int[] kvSlotForLayer, int attentionLayerCount,
        float[] ropeCosTable, float[] ropeSinTable, int ropeDim,
        ComputeThreadPool? threadPool, bool ownsPool,
        MtpHeadWeights? mtpHead = null)
    {
        Config = config;
        _gguf = gguf;
        _layers = layers;
        _outputNormWeight = outputNormWeight;
        _mtpHead = mtpHead;
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

        _state = new Qwen3HybridDenseForwardState(
            hiddenSize: config.HiddenSize,
            vocabSize: config.VocabSize,
            qElems: config.NumAttentionHeads * config.HeadDim,
            kvElems: config.NumKvHeads * config.HeadDim,
            convDim: (2 * _gdn.NKHead + _gdn.NVHead) * _gdn.DState,
            dConv: _gdn.DConv,
            nVHead: _gdn.NVHead,
            nKHead: _gdn.NKHead,
            dState: _gdn.DState,
            intermediateSize: config.IntermediateSize);
    }

    /// <summary>
    /// Loads a Qwen3HybridDense model from an opened GGUF file (single-threaded).
    /// The <paramref name="gguf"/> must remain alive for the lifetime of the returned model.
    /// </summary>
    /// <param name="gguf">An opened GGUF file.</param>
    /// <param name="config">Model configuration extracted from the file.</param>
    /// <returns>A loaded model.</returns>
    public static Qwen3HybridDenseTransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config)
        => LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);

    /// <summary>
    /// Loads a Qwen3HybridDense model from an opened GGUF file with threading configuration.
    /// When <paramref name="threading"/> is parallel, creates a <see cref="ComputeThreadPool"/>
    /// owned by this model (disposed with the model). The <paramref name="gguf"/> must remain
    /// alive for the lifetime of the returned model.
    /// </summary>
    /// <param name="gguf">An opened GGUF file.</param>
    /// <param name="config">Model configuration extracted from the file.</param>
    /// <param name="threading">Threading configuration: number of threads, NUMA / P-core pinning, dispatch policy.</param>
    /// <returns>A loaded model.</returns>
    public static Qwen3HybridDenseTransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config, ThreadingConfig threading)
    {
        if (config.Architecture != Architecture.Qwen3HybridDense)
            throw new ArgumentException(
                $"Qwen3HybridDenseTransformerModel requires Architecture.Qwen3HybridDense, got {config.Architecture}.",
                nameof(config));
        if (config.HybridLayout is null)
            throw new ArgumentException("Qwen3HybridDense config must have HybridLayout populated.", nameof(config));
        if (config.GdnConfig is null)
            throw new ArgumentException("Qwen3HybridDense config must have GdnConfig populated.", nameof(config));

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
                $"Qwen3HybridDense rope_dim={ropeDim} must be even for pair-wise rotation.");
        if (ropeDim > config.HeadDim)
            throw new InvalidDataException(
                $"Qwen3HybridDense rope_dim={ropeDim} exceeds head_dim={config.HeadDim}.");

        var layers = new Qwen3HybridDenseLayerWeights[config.NumLayers];
        var kvSlotForLayer = new int[config.NumLayers];
        int attentionLayerCount = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            layers[i] = LoadLayer(i, dataBase, tensors, config);
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

        // MTP (issue #253): load the trailing NextN head when the GGUF carries one. Zero behavior
        // change for every other checkpoint — LoadMtpHeadIfPresent returns null unless
        // config.NextnPredictLayers > 0 AND the nextn.* tensors are actually present.
        MtpHeadWeights? mtpHead = LoadMtpHeadIfPresent(dataBase, tensors, config);

        return new Qwen3HybridDenseTransformerModel(
            config, gguf, layers, outputNormWeight,
            embPtr, embDesc.QuantizationType,
            outputPtr, outputQt, outputM, outputK,
            kvSlotForLayer, attentionLayerCount,
            ropeCos, ropeSin, ropeDim,
            pool, ownsPool: pool is not null,
            mtpHead);
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
    /// Loads ONLY the CPU-resident tail <c>[startLayer, fullConfig.NumLayers)</c> of a
    /// Qwen3HybridDense model — the CPU half of a CPU/GPU partial-offload split (issue #291).
    /// Pairs with a GPU-side head instance
    /// (<c>DotLLM.Cuda.Architectures.CudaQwen3HybridDenseTransformerModel.LoadHeadFromGguf</c>)
    /// covering layers <c>[0, startLayer)</c>; the composition model D2H-transfers the GPU head's
    /// boundary hidden state and feeds it into this tail via <see cref="ForwardFromHiddenState"/>.
    /// </summary>
    /// <remarks>
    /// Reuses <see cref="LoadLayer"/> — the SAME per-layer tensor-name resolution the full
    /// CPU-only <see cref="LoadFromGguf(GgufFile, ModelConfig, ThreadingConfig)"/> path already
    /// gets right for this architecture's GDN-vs-full-attention naming split. Before this method
    /// existed, the generic (architecture-unaware) <c>DotLLM.Cuda.HybridTransformerModel</c>
    /// partial-offload splitter was the only CPU/GPU-split path, and it always called the
    /// Llama-style <c>TransformerWeights.LoadFromGguf</c> — which assumes every layer has a
    /// uniform <c>attn_output.weight</c>, throwing <c>KeyNotFoundException</c> the moment it hit a
    /// GDN layer (no attention output projection at all). This is issue #291's actual root cause:
    /// the splitter was never taught this architecture's per-layer-kind naming, not a subtly wrong
    /// boundary calculation.
    /// </remarks>
    /// <param name="gguf">Opened GGUF file (must remain alive for the model's lifetime).</param>
    /// <param name="fullConfig">
    /// The FULL model's configuration (<c>NumLayers</c> = the whole trunk, <c>HybridLayout</c>
    /// covering every layer). <see cref="LoadLayer"/> needs the untouched global layout to resolve
    /// <c>blk.{startLayer + i}</c>'s tensor names/kind correctly — this tail's OWN
    /// <see cref="Config"/> is a re-sliced view scoped to just <c>[startLayer, NumLayers)</c>.
    /// </param>
    /// <param name="startLayer">
    /// First GLOBAL layer index this tail owns (the GPU head owns <c>[0, startLayer)</c>). Must be
    /// in <c>(0, fullConfig.NumLayers)</c>.
    /// </param>
    /// <param name="threading">CPU threading configuration for this tail's layers.</param>
    internal static Qwen3HybridDenseTransformerModel LoadTailFromGguf(
        GgufFile gguf, ModelConfig fullConfig, int startLayer, ThreadingConfig threading)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(fullConfig);
        if (fullConfig.Architecture != Architecture.Qwen3HybridDense)
            throw new ArgumentException(
                $"Qwen3HybridDenseTransformerModel requires Architecture.Qwen3HybridDense, got {fullConfig.Architecture}.",
                nameof(fullConfig));
        if (fullConfig.HybridLayout is null)
            throw new ArgumentException("Qwen3HybridDense config must have HybridLayout populated.", nameof(fullConfig));
        if (fullConfig.GdnConfig is null)
            throw new ArgumentException("Qwen3HybridDense config must have GdnConfig populated.", nameof(fullConfig));
        if (startLayer <= 0 || startLayer >= fullConfig.NumLayers)
            throw new ArgumentOutOfRangeException(nameof(startLayer),
                $"startLayer must be between 1 and {fullConfig.NumLayers - 1} for a GPU/CPU split.");

        nint dataBase = gguf.DataBasePointer;
        var tensors = gguf.TensorsByName;
        int tailCount = fullConfig.NumLayers - startLayer;

        // Sliced layout/config: LOCAL layer index i (0..tailCount-1) here corresponds to GLOBAL
        // layer (startLayer + i). Weight loading below still passes the UNSLICED fullConfig plus
        // the GLOBAL index to LoadLayer (tensor-name/kind resolution needs the real blk.N index
        // against the full layout); this sliced config only drives the tail's own local
        // bookkeeping (KV slots, GDN ordinals, Config.NumLayers, RoPE/GDN-cache sizing).
        var tailLayout = new HybridLayerLayout
        {
            LayerKind = fullConfig.HybridLayout.LayerKind[startLayer..],
            HeadCountKv = fullConfig.HybridLayout.HeadCountKv[startLayer..],
            FeedForwardLength = fullConfig.HybridLayout.FeedForwardLength[startLayer..],
        };
        // MTP is not yet supported through the partial-offload split (out of scope for #291) —
        // force NextnPredictLayers=0 so this slice never tries (and fails) to resolve an MTP
        // block at the wrong raw index.
        var tailConfig = fullConfig with { NumLayers = tailCount, HybridLayout = tailLayout, NextnPredictLayers = 0 };

        var embDesc = tensors["token_embd.weight"];
        nint embPtr = dataBase + (nint)embDesc.DataOffset;

        var outNormDesc = tensors["output_norm.weight"];
        float[] outputNormWeight = DequantizeF32(dataBase, outNormDesc, fullConfig.HiddenSize);

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

        int ropeDim = fullConfig.RoPEConfig?.DimensionCount ?? fullConfig.HeadDim;
        if ((ropeDim & 1) != 0)
            throw new InvalidDataException(
                $"Qwen3HybridDense rope_dim={ropeDim} must be even for pair-wise rotation.");
        if (ropeDim > fullConfig.HeadDim)
            throw new InvalidDataException(
                $"Qwen3HybridDense rope_dim={ropeDim} exceeds head_dim={fullConfig.HeadDim}.");

        var layers = new Qwen3HybridDenseLayerWeights[tailCount];
        var kvSlotForLayer = new int[tailCount];
        int attentionLayerCount = 0;
        for (int i = 0; i < tailCount; i++)
        {
            // Global raw GGUF block index (startLayer + i) against the UNSLICED fullConfig —
            // LoadLayer indexes fullConfig.HybridLayout.LayerKind[layerIdx] directly, which only
            // lines up with the real blk.N tensor names when layerIdx is the global index.
            layers[i] = LoadLayer(startLayer + i, dataBase, tensors, fullConfig);
            kvSlotForLayer[i] = tailLayout.LayerKind[i] == HybridLayerKind.Attention
                ? attentionLayerCount++
                : -1;
        }

        float ropeTheta = fullConfig.RoPEConfig?.Theta ?? 10000.0f;
        int halfRope = ropeDim / 2;
        var ropeCos = new float[fullConfig.MaxSequenceLength * halfRope];
        var ropeSin = new float[fullConfig.MaxSequenceLength * halfRope];
        RoPE.PrecomputeFrequencyTable(fullConfig.MaxSequenceLength, ropeDim, ropeTheta, ropeCos, ropeSin);

        ComputeThreadPool? pool = CreatePool(threading);

        return new Qwen3HybridDenseTransformerModel(
            tailConfig, gguf, layers, outputNormWeight,
            embPtr, embDesc.QuantizationType,
            outputPtr, outputQt, outputM, outputK,
            kvSlotForLayer, attentionLayerCount,
            ropeCos, ropeSin, ropeDim,
            pool, ownsPool: pool is not null,
            mtpHead: null);
    }

    /// <summary>
    /// Runs this model's layer loop + final norm + lm_head starting from a CALLER-SUPPLIED
    /// initial hidden state instead of an embedding lookup — the CPU-tail half of a CPU/GPU
    /// partial-offload split (issue #291): <paramref name="initialHidden"/> is the GPU head's
    /// boundary hidden state (D2H-transferred by the composition model), continuing exactly where
    /// the GPU's layer prefix left off.
    /// </summary>
    /// <remarks>
    /// SYNC WARNING: mirrors <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?, IGdnState?, IMtpState?)"/>
    /// byte-for-byte except for the embedding step (no <c>EmbedTokens</c> call — the caller
    /// supplies the starting hidden state directly) and the omitted MTP-capture hook (MTP is not
    /// yet supported through the partial-offload split). Any future fix to the layer loop /
    /// GDN-vs-attention dispatch / final norm / lm-head in that method must be mirrored here.
    /// </remarks>
    /// <param name="initialHidden">
    /// Row-major <c>[seqLen, hiddenSize]</c> F32 hidden state to resume from, in place of
    /// embedding <paramref name="positions"/>.Length tokens. <c>seqLen</c> is inferred from
    /// <c>initialHidden.Length / Config.HiddenSize</c> and must exactly match
    /// <paramref name="positions"/>.Length.
    /// </param>
    /// <param name="positions">Position indices for each row of <paramref name="initialHidden"/>.</param>
    /// <param name="deviceId">Target device for the returned logits tensor (CPU: -1).</param>
    /// <param name="kvCache">Optional KV-cache for this tail's attention layers. Null runs uncached.</param>
    /// <param name="gdnState">
    /// Optional caller-supplied GDN state container. Null falls back to this model's own
    /// model-owned default state (see <see cref="ResetSequenceState"/>).
    /// </param>
    internal ITensor ForwardFromHiddenState(ReadOnlySpan<float> initialHidden, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache, IGdnState? gdnState)
    {
        int hiddenSize = Config.HiddenSize;
        int seqLen = positions.Length;
        if (seqLen == 0)
            throw new ArgumentException("positions must be non-empty.", nameof(positions));
        if (initialHidden.Length != seqLen * hiddenSize)
            throw new ArgumentException(
                $"initialHidden.Length ({initialHidden.Length}) must equal positions.Length * Config.HiddenSize " +
                $"({seqLen} * {hiddenSize} = {seqLen * hiddenSize}).", nameof(initialHidden));

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
                $"Qwen3HybridDenseTransformerModel requires a CPU GdnStateCache; got {gdnState.GetType().Name}.",
                nameof(gdnState));
        }

        int vocabSize = Config.VocabSize;
        int intermediateSize = Config.IntermediateSize;
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
        _threadPool?.SetDispatchMode(seqLen == 1 ? DispatchMode.SpinWait : DispatchMode.EventBased);

        float* hidden = (float*)_state.HiddenState;
        float* residual = (float*)_state.Residual;
        float* normOut = (float*)_state.NormOutput;
        float* logits = (float*)_state.Logits;
        float* qAttn = (float*)_state.QScratch;
        float* kAttn = (float*)_state.KScratch;
        float* vAttn = (float*)_state.VScratch;
        float* attnOut = (float*)_state.AttnOutput;

        initialHidden.CopyTo(new Span<float>(hidden, seqLen * hiddenSize));

        var kinds = _layout.LayerKind;
        for (int layer = 0; layer < _layers.Length; layer++)
        {
            var lw = _layers[layer];
            // ── Token-mixing sub-layer ─────────────────────────────────────────
            new Span<float>(hidden, seqLen * hiddenSize)
                .CopyTo(new Span<float>(residual, seqLen * hiddenSize));
            for (int t = 0; t < seqLen; t++)
            {
                RmsNorm.Execute(
                    new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                    lw.AttnNormWeight, eps,
                    new Span<float>(normOut + t * hiddenSize, hiddenSize));
            }

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
            // ── Dense SwiGLU FFN sub-layer ──────────────────────────────────────
            new Span<float>(hidden, seqLen * hiddenSize)
                .CopyTo(new Span<float>(residual, seqLen * hiddenSize));
            for (int t = 0; t < seqLen; t++)
            {
                RmsNorm.Execute(
                    new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                    lw.PostAttnNormWeight, eps,
                    new Span<float>(normOut + t * hiddenSize, hiddenSize));
            }

            ForwardDenseFfnBody(lw, seqLen, hiddenSize, intermediateSize, normOut);

            // Second residual add.
            for (int t = 0; t < seqLen; t++)
            {
                Add.Execute(
                    new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize),
                    new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                    new Span<float>(hidden + t * hiddenSize, hiddenSize));
            }
        }

        // Final output norm + logit projection.
        for (int t = 0; t < seqLen; t++)
        {
            RmsNorm.Execute(
                new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                _outputNormWeight, eps,
                new Span<float>(hidden + t * hiddenSize, hiddenSize));
        }

        Gemm(_outputWeight, _outputQuantType, hidden, logits,
             _outputOutputDim, _outputInputDim, seqLen);

        var shape = new TensorShape(seqLen, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId);
        new Span<float>(logits, seqLen * vocabSize).CopyTo(
            new Span<float>((void*)result.DataPointer, seqLen * vocabSize));

        return result;
    }

    /// <summary>
    /// Builds a Qwen3HybridDense model from caller-owned, pre-dequantised weight pointers —
    /// used by parity tests that construct synthetic weight banks in unmanaged memory directly,
    /// bypassing the GGUF loader. Caller retains ownership of every <see cref="nint"/> pointer
    /// (token embed, output, plus every projection inside <paramref name="layers"/>).
    /// </summary>
    /// <remarks>
    /// Unlike <see cref="LoadFromGguf(GgufFile, ModelConfig)"/> there is no <see cref="GgufFile"/>
    /// to keep alive — the model holds <c>null</c> for the gguf reference. Disposing frees only
    /// the forward scratch and the GDN state cache; weight memory belongs to the caller.
    /// </remarks>
    internal static Qwen3HybridDenseTransformerModel BuildFromPrebuiltWeights(
        ModelConfig config,
        Qwen3HybridDenseLayerWeights[] layers,
        float[] outputNormWeight,
        nint tokenEmbedWeight, QuantizationType tokenEmbedQuantType,
        nint outputWeight, QuantizationType outputQuantType, int outputOutputDim, int outputInputDim)
    {
        if (config.Architecture != Architecture.Qwen3HybridDense)
            throw new ArgumentException(
                $"Qwen3HybridDenseTransformerModel requires Architecture.Qwen3HybridDense, got {config.Architecture}.",
                nameof(config));
        if (config.HybridLayout is null)
            throw new ArgumentException("Qwen3HybridDense config must have HybridLayout populated.", nameof(config));
        if (config.GdnConfig is null)
            throw new ArgumentException("Qwen3HybridDense config must have GdnConfig populated.", nameof(config));
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

        return new Qwen3HybridDenseTransformerModel(
            config, gguf: null, layers, outputNormWeight,
            tokenEmbedWeight, tokenEmbedQuantType,
            outputWeight, outputQuantType, outputOutputDim, outputInputDim,
            kvSlotForLayer, attentionLayerCount,
            ropeCos, ropeSin, ropeDim,
            threadPool: null, ownsPool: false);
    }

    private static Qwen3HybridDenseLayerWeights LoadLayer(
        int layerIdx,
        nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config)
    {
        string prefix = $"blk.{layerIdx}";
        int hiddenSize = config.HiddenSize;
        var layout = config.HybridLayout!;

        var attnNormDesc = tensors[$"{prefix}.attn_norm.weight"];
        float[] attnNormWeight = DequantizeF32(dataBase, attnNormDesc, hiddenSize);

        // Same LLM_TENSOR_ATTN_POST_NORM → "blk.%d.post_attention_norm" convention as
        // qwen35moe (confirmed against real Bonsai-27B GGUF tensor names this session) — NOT
        // the standard "ffn_norm" name used by Qwen2/Qwen3 dense MoE.
        var postAttnNormDesc = tensors[$"{prefix}.post_attention_norm.weight"];
        float[] postAttnNormWeight = DequantizeF32(dataBase, postAttnNormDesc, hiddenSize);

        var tokenMixing = layout.LayerKind[layerIdx] switch
        {
            HybridLayerKind.GatedDeltaNet =>
                (gdn: LoadGdnLayer(prefix, dataBase, tensors, config), attn: (Qwen3FullAttnWeights?)null),
            HybridLayerKind.Attention =>
                (gdn: (GdnTokenMixingWeights?)null, attn: LoadFullAttnLayer(prefix, dataBase, tensors, config, layout.HeadCountKv[layerIdx])),
            _ => throw new InvalidOperationException(
                $"Unexpected HybridLayerKind {layout.LayerKind[layerIdx]} at layer {layerIdx} in Qwen3HybridDense."),
        };

        // Dense SwiGLU FFN — standard ffn_gate/up/down naming, confirmed against the real
        // Ternary-Bonsai-27B-Q2_0.gguf (no "_exps" suffix, no expert_count metadata at all).
        var gateDesc = tensors[$"{prefix}.ffn_gate.weight"];
        var upDesc = tensors[$"{prefix}.ffn_up.weight"];
        var downDesc = tensors[$"{prefix}.ffn_down.weight"];

        return new Qwen3HybridDenseLayerWeights
        {
            AttnNormWeight = attnNormWeight,
            PostAttnNormWeight = postAttnNormWeight,
            Gdn = tokenMixing.gdn,
            FullAttn = tokenMixing.attn,

            GateWeight = dataBase + (nint)gateDesc.DataOffset,
            GateQuantType = gateDesc.QuantizationType,
            GateInputDim = gateDesc.Shape[0],
            GateOutputDim = gateDesc.Shape[1],

            UpWeight = dataBase + (nint)upDesc.DataOffset,
            UpQuantType = upDesc.QuantizationType,
            UpInputDim = upDesc.Shape[0],
            UpOutputDim = upDesc.Shape[1],

            DownWeight = dataBase + (nint)downDesc.DataOffset,
            DownQuantType = downDesc.QuantizationType,
            DownInputDim = downDesc.Shape[0],
            DownOutputDim = downDesc.Shape[1],
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

        // qwen35 full-attn: attn_q.weight has output dim 2 * nQ * headDim (Q + Gate fused per
        // head) — same convention as qwen35moe, confirmed against real Bonsai-27B GGUF tensor
        // shapes this session. Verify shape so we don't silently regress if the naming shifts.
        int expectedQGateOut = 2 * config.NumAttentionHeads * config.HeadDim;
        if (q.Shape[1] != expectedQGateOut)
        {
            throw new InvalidDataException(
                $"{prefix}.attn_q.weight has output dim {q.Shape[1]} but qwen35 expects " +
                $"{expectedQGateOut} = 2 * {config.NumAttentionHeads} * {config.HeadDim} (Q+Gate fused).");
        }

        // QK-norm tensors are required by llama.cpp's qwen35(moe) loader (no TENSOR_NOT_REQUIRED flag).
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

    /// <summary>
    /// Loads the trailing Multi-Token Prediction (MTP / "NextN") head when the GGUF has one
    /// (issue #253), or returns <see langword="null"/> for a checkpoint without MTP — the
    /// overwhelming majority of GGUFs, completely unaffected by this method.
    /// </summary>
    /// <remarks>
    /// Tensor naming and layout confirmed against llama.cpp PR ggml-org/llama.cpp#22673
    /// (<c>src/models/qwen35.cpp</c>'s <c>load_block_mtp</c>): the MTP block sits at raw GGUF
    /// block index <c>config.NumLayers</c> (trunk layers occupy <c>[0, NumLayers)</c>, since
    /// <c>GgufModelConfigExtractor</c> already subtracted <c>nextn_predict_layers</c>
    /// back out of the raw <c>block_count</c> — see <see cref="ModelConfig.NextnPredictLayers"/>).
    /// It is structurally a full-attention decoder layer (same tensors <see cref="LoadLayer"/>
    /// would load for an <see cref="HybridLayerKind.Attention"/> block), plus four "nextn.*"
    /// tensors. Only <c>nextn_predict_layers == 1</c> is supported, matching llama.cpp's own
    /// <c>GGML_ASSERT(hparams.nextn_predict_layers == 1)</c> for the QWEN35 family today.
    /// </remarks>
    private static MtpHeadWeights? LoadMtpHeadIfPresent(
        nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config)
    {
        if (config.NextnPredictLayers <= 0)
            return null;

        if (config.NextnPredictLayers != 1)
            throw new NotSupportedException(
                $"Qwen3HybridDense MTP only supports a single trailing MTP block " +
                $"(nextn_predict_layers=1); got {config.NextnPredictLayers}. Matches llama.cpp's own " +
                "current QWEN35 MTP assertion (issue #253 scope: Qwen3.6, not a future multi-head variant).");

        int mtpBlk = config.NumLayers; // trunk occupies [0, NumLayers); MTP is appended right after
        string prefix = $"blk.{mtpBlk}";

        // The hparam can be set without the tensors actually being present (e.g. a trunk-only
        // GGUF someone hand-edited the metadata on) — treat that as "no MTP head", not an error,
        // to keep the zero-behavior-change guarantee unconditional.
        if (!tensors.ContainsKey($"{prefix}.nextn.eh_proj.weight"))
            return null;

        int hiddenSize = config.HiddenSize;

        // The MTP block's own attn+ffn tensors use the exact same naming/shapes as any other
        // full-attention Qwen3HybridDense layer — reuse the trunk loaders directly.
        var attn = LoadFullAttnLayer(prefix, dataBase, tensors, config, config.NumKvHeads);
        var attnNorm = DequantizeF32(dataBase, tensors[$"{prefix}.attn_norm.weight"], hiddenSize);
        var postAttnNorm = DequantizeF32(dataBase, tensors[$"{prefix}.post_attention_norm.weight"], hiddenSize);

        var gateDesc = tensors[$"{prefix}.ffn_gate.weight"];
        var upDesc = tensors[$"{prefix}.ffn_up.weight"];
        var downDesc = tensors[$"{prefix}.ffn_down.weight"];

        var layer = new Qwen3HybridDenseLayerWeights
        {
            AttnNormWeight = attnNorm,
            PostAttnNormWeight = postAttnNorm,
            FullAttn = attn,
            Gdn = null,

            GateWeight = dataBase + (nint)gateDesc.DataOffset,
            GateQuantType = gateDesc.QuantizationType,
            GateInputDim = gateDesc.Shape[0],
            GateOutputDim = gateDesc.Shape[1],

            UpWeight = dataBase + (nint)upDesc.DataOffset,
            UpQuantType = upDesc.QuantizationType,
            UpInputDim = upDesc.Shape[0],
            UpOutputDim = upDesc.Shape[1],

            DownWeight = dataBase + (nint)downDesc.DataOffset,
            DownQuantType = downDesc.QuantizationType,
            DownInputDim = downDesc.Shape[0],
            DownOutputDim = downDesc.Shape[1],
        };

        var ehProjDesc = tensors[$"{prefix}.nextn.eh_proj.weight"];
        float[] enorm = DequantizeF32(dataBase, tensors[$"{prefix}.nextn.enorm.weight"], hiddenSize);
        float[] hnorm = DequantizeF32(dataBase, tensors[$"{prefix}.nextn.hnorm.weight"], hiddenSize);

        nint? embedTokensPtr = null;
        QuantizationType embedTokensQt = default;
        if (tensors.TryGetValue($"{prefix}.nextn.embed_tokens.weight", out var embedDesc))
        {
            embedTokensPtr = dataBase + (nint)embedDesc.DataOffset;
            embedTokensQt = embedDesc.QuantizationType;
        }

        nint? sharedHeadPtr = null;
        QuantizationType sharedHeadQt = default;
        if (tensors.TryGetValue($"{prefix}.nextn.shared_head_head.weight", out var sharedHeadDesc))
        {
            sharedHeadPtr = dataBase + (nint)sharedHeadDesc.DataOffset;
            sharedHeadQt = sharedHeadDesc.QuantizationType;
        }

        float[]? sharedHeadNorm = tensors.TryGetValue($"{prefix}.nextn.shared_head_norm.weight", out var shnDesc)
            ? DequantizeF32(dataBase, shnDesc, hiddenSize)
            : null;

        return new MtpHeadWeights
        {
            Layer = layer,

            EhProjWeight = dataBase + (nint)ehProjDesc.DataOffset,
            EhProjQuantType = ehProjDesc.QuantizationType,
            EhProjInputDim = ehProjDesc.Shape[0],
            EhProjOutputDim = ehProjDesc.Shape[1],

            EnormWeight = enorm,
            HnormWeight = hnorm,

            EmbedTokensWeight = embedTokensPtr,
            EmbedTokensQuantType = embedTokensQt,

            SharedHeadHeadWeight = sharedHeadPtr,
            SharedHeadHeadQuantType = sharedHeadQt,

            SharedHeadNormWeight = sharedHeadNorm,
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
    /// LoRA-aware forward, accepted for <see cref="IModel"/> parity with
    /// <see cref="Qwen3MoeHybridTransformerModel"/>'s equivalent overload. Currently a no-op
    /// pass-through: there is no per-expert MoE routing to hook a LoRA delta into on this
    /// dense-FFN architecture, and GDN/full-attention projection LoRA is out of scope for this
    /// first pass. <paramref name="adapter"/> is accepted but has no effect.
    /// </summary>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache, ILoraAdapter? adapter)
        => Forward(tokenIds, positions, deviceId, kvCache, gdnState: null);

    /// <inheritdoc/>
    /// <remarks>
    /// MTP (issue #253): when <paramref name="mtpState"/> is a <see cref="CpuMtpState"/>, this
    /// call additionally captures the pre-final-norm hidden state for every position in
    /// <paramref name="tokenIds"/> into it — see the capture point inside the core
    /// <c>Forward(..., IGdnState?, IMtpState?)</c> overload. The returned logits are byte-identical
    /// to calling without <paramref name="mtpState"/>.
    /// </remarks>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache, ILoraAdapter? adapter, IMtpState? mtpState)
        => Forward(tokenIds, positions, deviceId, kvCache, gdnState: null, mtpState);

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
    /// <param name="mtpState">
    /// Optional MTP state (issue #253). When a <see cref="CpuMtpState"/>, this call additionally
    /// captures the pre-final-norm hidden state for every position into it as a side effect — see
    /// the capture point right before the final RMSNorm below. Ignored (and safe to pass null) for
    /// callers that don't use MTP self-speculative decoding.
    /// </param>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache, IGdnState? gdnState, IMtpState? mtpState = null)
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
                $"Qwen3HybridDenseTransformerModel requires a CPU GdnStateCache; got {gdnState.GetType().Name}.",
                nameof(gdnState));
        }

        int seqLen = tokenIds.Length;
        if (seqLen == 0 || seqLen != positions.Length)
            throw new ArgumentException("tokenIds and positions must have equal, non-zero length.");

        int hiddenSize = Config.HiddenSize;
        int vocabSize = Config.VocabSize;
        int intermediateSize = Config.IntermediateSize;
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
            // ── Dense SwiGLU FFN sub-layer ──────────────────────────────────────
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

            ForwardDenseFfnBody(lw, seqLen, hiddenSize, intermediateSize, normOut);
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

        // MTP (issue #253): capture the pre-final-norm hidden state for every position, one row
        // per input token, BEFORE the final RMSNorm below overwrites `hidden` in place. This is
        // the exact quantity llama.cpp's MTP head consumes (`h_pre_norm` / `llama_get_embeddings_pre_norm`)
        // — a pure side effect that never changes the logits this call returns.
        if (mtpState is CpuMtpState mtpCapture)
            mtpCapture.SetCapturedRows(new ReadOnlySpan<float>(hidden, seqLen * hiddenSize), seqLen);

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
             _outputOutputDim, _outputInputDim, seqLen);

        if (TensorDump.Enabled)
            TensorDump.Dump2D("result_output", logits, seqLen, vocabSize);

        var shape = new TensorShape(seqLen, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId);
        new Span<float>(logits, seqLen * vocabSize).CopyTo(
            new Span<float>((void*)result.DataPointer, seqLen * vocabSize));

        return result;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Re-zeroes the model-owned Gated-DeltaNet cache used by every forward that does not carry a
    /// caller-supplied <see cref="IGdnState"/>. Callers that treat each forward as an independent
    /// sequence (perplexity windows) must call this between sequences — see issue #261.
    /// </remarks>
    public void ResetSequenceState() => _gdnCache.Reset();

    /// <inheritdoc/>
    public bool RequiresPerSequenceState => true;

    /// <inheritdoc/>
    public bool SupportsThreadedSequenceState => true;

    /// <inheritdoc/>
    public IRecurrentSequenceState? CreateSequenceState() => new GdnStateCache(_gdn, _gdnCache.NumGdnLayers);

    /// <inheritdoc/>
    /// <remarks>
    /// Sized for the MTP head's own attention (<see cref="Config"/>'s standard head count/dim —
    /// the MTP block is a normal full-attention layer, see <see cref="MtpHeadWeights"/>), with a
    /// KV-cache deep enough for <see cref="MtpDefaultMaxDraftSteps"/> autoregressive draft steps.
    /// </remarks>
    public IMtpState? CreateMtpState()
    {
        if (_mtpHead is null)
            return null;

        return new CpuMtpState(
            hiddenSize: Config.HiddenSize,
            numKvHeads: _mtpHead.Layer.FullAttn!.NumKvHeads,
            headDim: Config.HeadDim,
            maxSteps: MtpDefaultMaxDraftSteps);
    }

    /// <summary>
    /// Default MTP KV-cache depth when a caller doesn't need a specific candidate count K up
    /// front. Callers that know K in advance (e.g. an MTP self-speculative decoder, see issue
    /// #253) can size their own <see cref="CpuMtpState"/> directly instead of going through
    /// <see cref="CreateMtpState"/>.
    /// </summary>
    public const int MtpDefaultMaxDraftSteps = 16;

    /// <inheritdoc/>
    public ITensor ForwardMtp(IMtpState state, int tokenId, int position)
    {
        if (_mtpHead is null)
            throw new NotSupportedException(
                $"{nameof(Qwen3HybridDenseTransformerModel)} has no MTP head loaded (SupportsMtp=false).");
        if (state is not CpuMtpState mtp)
            throw new ArgumentException(
                $"Qwen3HybridDenseTransformerModel requires a CPU CpuMtpState; got {state.GetType().Name}.",
                nameof(state));
        if ((uint)tokenId >= (uint)Config.VocabSize)
            throw new ArgumentOutOfRangeException(nameof(tokenId));

        return ForwardMtpCore(_mtpHead, mtp, tokenId, position);
    }

    /// <summary>
    /// Per-sequence loop over <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?, IGdnState?, IMtpState?)"/>
    /// — threads each request's GDN state through to the GDN scan, so multi-seq batched
    /// dispatch is safe as long as every request supplies a fresh <see cref="GdnStateCache"/>.
    /// </summary>
    public IReadOnlyList<ITensor> ForwardBatch(
        IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0) return Array.Empty<ITensor>();

        for (int i = 0; i < requests.Count; i++)
        {
            if (requests[i].Adapter is not null)
                throw new NotSupportedException(
                    "Qwen3HybridDenseTransformerModel.ForwardBatch does not support LoRA adapters yet. " +
                    "Re-issue the request without an adapter.");
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
    /// Operation order (confirmed from llama.cpp <c>qwen35moe.cpp</c> — identical for the dense
    /// <c>qwen35</c> variant, only the FFN sublayer differs):
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
             gdnW.QkvOutputDim, gdnW.QkvInputDim, seqLen);
        Gemm(gdnW.GateWeight, gdnW.GateQuantType, normOut, zBuf,
             gdnW.GateOutputDim, gdnW.GateInputDim, seqLen);
        Gemm(gdnW.AlphaWeight, gdnW.AlphaQuantType, normOut, alphaBuf,
             gdnW.AlphaOutputDim, gdnW.AlphaInputDim, seqLen);
        Gemm(gdnW.BetaWeight, gdnW.BetaQuantType, normOut, betaBuf,
             gdnW.BetaOutputDim, gdnW.BetaInputDim, seqLen);

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
             gdnW.OutOutputDim, gdnW.OutInputDim, seqLen);

        if (TensorDump.Enabled)
            TensorDump.Dump2D($"blk.{absoluteLayerIdx}.linear_attn_out", normOut, seqLen, hiddenSize);
    }

    /// <summary>
    /// Full GQA attention forward (qwen35 dense variant — identical semantics to qwen35moe's
    /// equivalent). Reads pre-normed activations from <paramref name="normOut"/> and writes the
    /// gated output projection back to the same buffer.
    /// </summary>
    /// <remarks>
    /// Operation order (verified against llama.cpp <c>build_layer_attn</c> in qwen35moe.cpp —
    /// the qwen35 dense variant shares the same attention implementation):
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
        Gemm(attn.QWeight, attn.QQuantType, normOut, qgBuf, attn.QOutputDim, attn.QInputDim, seqLen);
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
        Gemm(attn.KWeight, attn.KQuantType, normOut, k, attn.KOutputDim, attn.KInputDim, seqLen);
        Gemm(attn.VWeight, attn.VQuantType, normOut, v, attn.VOutputDim, attn.VInputDim, seqLen);
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
        //    NOTE: qwen35(moe) uses mRoPE with sections [11,11,10,0] and mrope_interleaved=true. For
        //    text-only inference (positions identical across all 4 axes), this collapses to
        //    single-axis RoPE. Pair pattern is NeoX (HuggingFace rotate_half) — verified by the
        //    synthetic CPU↔CUDA parity fixture in CudaQwen3MoeHybridParityTests.cs (shared attention
        //    implementation).
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
        Gemm(attn.OWeight, attn.OQuantType, attnOut, normOut, attn.OOutputDim, attn.OInputDim, seqLen);
    }

    /// <summary>
    /// Dense SwiGLU FFN sub-layer. Reads pre-normed activations from <paramref name="normOut"/>
    /// and overwrites the same buffer with the FFN output (caller adds the residual). Replaces
    /// <c>Qwen3MoeHybridTransformerModel.ForwardMoeBody</c>'s sparse-routing MoE FFN — no
    /// routing, no per-expert bucketing, every layer runs the same dense gate/up/down.
    /// </summary>
    private void ForwardDenseFfnBody(
        Qwen3HybridDenseLayerWeights lw, int seqLen, int hiddenSize, int intermediateSize, float* normOut)
    {
        float* ffnGate = (float*)_state.FfnGate;
        float* ffnUp = (float*)_state.FfnUp;
        float* siluOut = (float*)_state.SiluOutput;

        Gemm(lw.GateWeight, lw.GateQuantType, normOut, ffnGate, lw.GateOutputDim, lw.GateInputDim, seqLen);
        Gemm(lw.UpWeight, lw.UpQuantType, normOut, ffnUp, lw.UpOutputDim, lw.UpInputDim, seqLen);

        for (int t = 0; t < seqLen; t++)
        {
            var gateSpan = new ReadOnlySpan<float>(ffnGate + (long)t * intermediateSize, intermediateSize);
            var upSpan = new ReadOnlySpan<float>(ffnUp + (long)t * intermediateSize, intermediateSize);
            var outSpan = new Span<float>(siluOut + (long)t * intermediateSize, intermediateSize);
            FusedOps.SwiGLU(gateSpan, upSpan, outSpan);
        }

        Gemm(lw.DownWeight, lw.DownQuantType, siluOut, normOut, lw.DownOutputDim, lw.DownInputDim, seqLen);
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

    /// <summary>Single-token embedding lookup against an arbitrary embedding table — used by
    /// <see cref="ForwardMtpCore"/> to embed from either <c>nextn.embed_tokens</c> (when present)
    /// or the trunk's own <c>token_embd.weight</c> (fallback), mirroring <see cref="EmbedTokens"/>'s
    /// per-quant-type dispatch for a single row.</summary>
    private static void EmbedOneToken(int tokenId, nint embPtr, QuantizationType qt, Span<float> dest, int hiddenSize)
    {
        if (qt == QuantizationType.F32)
        {
            new ReadOnlySpan<float>((float*)embPtr + (long)tokenId * hiddenSize, hiddenSize).CopyTo(dest);
        }
        else if (qt == QuantizationType.F16)
        {
            TensorPrimitives.ConvertToSingle(
                new ReadOnlySpan<Half>((Half*)embPtr + (long)tokenId * hiddenSize, hiddenSize), dest);
        }
        else
        {
            long rowBytes = Dequantize.RowByteSize(hiddenSize, qt);
            Dequantize.ToFloat32(embPtr + (nint)((long)tokenId * rowBytes), hiddenSize, qt, dest);
        }
    }

    /// <summary>
    /// Runs one MTP head autoregressive draft step (issue #253) — see <see cref="ForwardMtp"/>.
    /// Off the trunk's hot forward path (single token, called K≤~16 times per speculation round),
    /// so this uses <see cref="ArrayPool{T}"/> scratch rather than the trunk's dedicated
    /// <see cref="Qwen3HybridDenseForwardState"/> scratch arena — simplicity over micro-optimising
    /// a call that runs orders of magnitude less often than the main per-layer forward loop.
    /// </summary>
    /// <remarks>
    /// Operation order confirmed against llama.cpp PR ggml-org/llama.cpp#22673
    /// (<c>src/models/qwen35.cpp</c>'s <c>graph_mtp</c> constructor):
    /// <list type="number">
    ///   <item><c>h_norm = RMSNorm(pendingHidden, nextn.hnorm)</c>; <c>e_norm = RMSNorm(embed(tokenId), nextn.enorm)</c>.</item>
    ///   <item><c>cur = eh_proj @ concat(e_norm, h_norm)</c> — this becomes the attention sub-block's residual (<c>inpSA</c>).</item>
    ///   <item>Gated full attention over the MTP head's own KV-cache (identical math to
    ///         <see cref="ForwardFullAttnBody"/>, but seqQ=1 against the head's private cache rather
    ///         than the trunk's), residual-added back onto <c>inpSA</c>.</item>
    ///   <item>Dense SwiGLU FFN, residual-added — the result is the MTP block's own output hidden
    ///         state ("h_pre_norm"), which seeds <em>this state's next</em> <see cref="ForwardMtp"/> call.</item>
    ///   <item><c>shared_head_norm</c> (or the trunk's <c>output_norm</c> fallback) then
    ///         <c>shared_head_head</c> (or the trunk's own LM head fallback) → logits.</item>
    /// </list>
    /// </remarks>
    private ITensor ForwardMtpCore(MtpHeadWeights mtpHead, CpuMtpState state, int tokenId, int position)
    {
        int hiddenSize = Config.HiddenSize;
        int vocabSize = Config.VocabSize;
        var attn = mtpHead.Layer.FullAttn!;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = attn.NumKvHeads;
        int headDim = Config.HeadDim;
        int qElems = numHeads * headDim;
        int kvStride = numKvHeads * headDim;
        int intermediateSize = mtpHead.Layer.GateOutputDim;
        float eps = Config.NormEpsilon;

        int step = state.CurrentLength;
        if (step >= state.MaxSteps)
            throw new InvalidOperationException(
                $"CpuMtpState KV-cache exhausted ({state.MaxSteps} steps advanced). Size the state for " +
                "at least numCandidates MTP draft steps per speculation round.");

        var pool = ArrayPool<float>.Shared;
        float[] tokEmbedArr = pool.Rent(hiddenSize);
        float[] eNormArr = pool.Rent(hiddenSize);
        float[] hNormArr = pool.Rent(hiddenSize);
        float[] concatArr = pool.Rent(2 * hiddenSize);
        float[] curArr = pool.Rent(hiddenSize);
        float[] residualArr = pool.Rent(hiddenSize);
        float[] normedArr = pool.Rent(hiddenSize);
        float[] qgArr = pool.Rent(2 * qElems);
        float[] qArr = pool.Rent(qElems);
        float[] gateArr = pool.Rent(qElems);
        float[] kArr = pool.Rent(kvStride);
        float[] vArr = pool.Rent(kvStride);
        float[] attnOutArr = pool.Rent(qElems);
        float[] ffnGateArr = pool.Rent(intermediateSize);
        float[] ffnUpArr = pool.Rent(intermediateSize);
        float[] siluArr = pool.Rent(intermediateSize);
        float[] normedHeadArr = pool.Rent(hiddenSize);
        float[] logitsArr = pool.Rent(vocabSize);

        try
        {
            var tokEmbed = tokEmbedArr.AsSpan(0, hiddenSize);
            var eNorm = eNormArr.AsSpan(0, hiddenSize);
            var hNorm = hNormArr.AsSpan(0, hiddenSize);
            var concat = concatArr.AsSpan(0, 2 * hiddenSize);
            var cur = curArr.AsSpan(0, hiddenSize);
            var residual = residualArr.AsSpan(0, hiddenSize);
            var normed = normedArr.AsSpan(0, hiddenSize);
            var qg = qgArr.AsSpan(0, 2 * qElems);
            var q = qArr.AsSpan(0, qElems);
            var gate = gateArr.AsSpan(0, qElems);
            var k = kArr.AsSpan(0, kvStride);
            var v = vArr.AsSpan(0, kvStride);
            var attnOut = attnOutArr.AsSpan(0, qElems);
            var ffnGate = ffnGateArr.AsSpan(0, intermediateSize);
            var ffnUp = ffnUpArr.AsSpan(0, intermediateSize);
            var silu = siluArr.AsSpan(0, intermediateSize);
            var normedHead = normedHeadArr.AsSpan(0, hiddenSize);
            var logits = logitsArr.AsSpan(0, vocabSize);

            // ── Embed predicted-from token, combine with pending trunk/MTP hidden state ──
            EmbedOneToken(tokenId,
                mtpHead.EmbedTokensWeight ?? _tokenEmbedWeight,
                mtpHead.EmbedTokensWeight is not null ? mtpHead.EmbedTokensQuantType : _tokenEmbedQuantType,
                tokEmbed, hiddenSize);

            RmsNorm.Execute(tokEmbed, mtpHead.EnormWeight, eps, eNorm);
            RmsNorm.Execute(state.PendingHidden, mtpHead.HnormWeight, eps, hNorm);

            eNorm.CopyTo(concat);
            hNorm.CopyTo(concat.Slice(hiddenSize));

            fixed (float* concatPtr = concat, curPtr = cur)
                Gemm(mtpHead.EhProjWeight, mtpHead.EhProjQuantType, concatPtr, curPtr,
                     mtpHead.EhProjOutputDim, mtpHead.EhProjInputDim, 1);

            // inpSA: the attention sub-block's residual is the eh_proj output, not the raw input.
            cur.CopyTo(residual);

            // ── Attention sub-block — same gated-QKV math as ForwardFullAttnBody, seqQ=1 ──
            RmsNorm.Execute(cur, mtpHead.Layer.AttnNormWeight, eps, normed);

            fixed (float* normedPtr = normed, qgPtr = qg)
                Gemm(attn.QWeight, attn.QQuantType, normedPtr, qgPtr, attn.QOutputDim, attn.QInputDim, 1);

            for (int h = 0; h < numHeads; h++)
            {
                int qgOff = h * 2 * headDim;
                int hOff = h * headDim;
                qg.Slice(qgOff, headDim).CopyTo(q.Slice(hOff));
                qg.Slice(qgOff + headDim, headDim).CopyTo(gate.Slice(hOff));
            }

            fixed (float* normedPtr = normed, kPtr = k, vPtr = v)
            {
                Gemm(attn.KWeight, attn.KQuantType, normedPtr, kPtr, attn.KOutputDim, attn.KInputDim, 1);
                Gemm(attn.VWeight, attn.VQuantType, normedPtr, vPtr, attn.VOutputDim, attn.VInputDim, 1);
            }

            Mamba3QkNorm.Execute(q, attn.QNormWeight, eps, 1, numHeads, headDim);
            Mamba3QkNorm.Execute(k, attn.KNormWeight, eps, 1, numKvHeads, headDim);

            Span<int> posSpan = stackalloc int[1];
            posSpan[0] = position;
            RoPE.Execute(q, k, posSpan, numHeads, numKvHeads, headDim, _ropeDim,
                _ropeCosTable, _ropeSinTable, RoPEType.NeoX);

            // Append this step's K/V into the MTP head's own tiny cache and attend causally over
            // everything drafted so far in this round (NOT the trunk's KV-cache).
            k.CopyTo(state.GetKeyRow(step));
            v.CopyTo(state.GetValueRow(step));

            int seqKv = step + 1;
            Attention.Execute(
                q,
                new ReadOnlySpan<float>(state.KeyCachePtr, seqKv * kvStride),
                new ReadOnlySpan<float>(state.ValueCachePtr, seqKv * kvStride),
                attnOut,
                /* seqQ */ 1, seqKv, numHeads, numKvHeads, headDim,
                /* positionOffset */ step,
                /* slidingWindowSize */ (int?)null);

            // sigmoid(gate) applied to attention output before O-proj (Qwen3.5/3.6 gated attention).
            for (int i = 0; i < qElems; i++)
                attnOut[i] *= 1f / (1f + MathF.Exp(-gate[i]));

            fixed (float* attnOutPtr = attnOut, curPtr = cur)
                Gemm(attn.OWeight, attn.OQuantType, attnOutPtr, curPtr, attn.OOutputDim, attn.OInputDim, 1);

            Add.Execute(residual, cur, cur); // cur = inpSA + attn_out_projected

            // ── Dense SwiGLU FFN sub-layer ──
            cur.CopyTo(residual); // ffn_residual
            RmsNorm.Execute(cur, mtpHead.Layer.PostAttnNormWeight, eps, normed);

            fixed (float* normedPtr = normed, ffnGatePtr = ffnGate, ffnUpPtr = ffnUp)
            {
                Gemm(mtpHead.Layer.GateWeight, mtpHead.Layer.GateQuantType, normedPtr, ffnGatePtr,
                     mtpHead.Layer.GateOutputDim, mtpHead.Layer.GateInputDim, 1);
                Gemm(mtpHead.Layer.UpWeight, mtpHead.Layer.UpQuantType, normedPtr, ffnUpPtr,
                     mtpHead.Layer.UpOutputDim, mtpHead.Layer.UpInputDim, 1);
            }

            FusedOps.SwiGLU(ffnGate, ffnUp, silu);

            fixed (float* siluPtr = silu, curPtr = cur)
                Gemm(mtpHead.Layer.DownWeight, mtpHead.Layer.DownQuantType, siluPtr, curPtr,
                     mtpHead.Layer.DownOutputDim, mtpHead.Layer.DownInputDim, 1);

            Add.Execute(residual, cur, cur); // cur = ffn_residual + ffn_out

            // `cur` is now the MTP block's own output hidden state ("h_pre_norm" in llama.cpp) —
            // seed the NEXT ForwardMtp call's pending hidden with it before the head-norm below
            // consumes it, then advance the MTP KV-cache length.
            cur.CopyTo(state.PendingHiddenMutable);
            state.Advance();

            // ── Shared LM head (falls back to the trunk's output_norm/output.weight when the
            //    GGUF didn't ship head-local nextn.shared_head_* tensors) ──
            float[] headNormWeight = mtpHead.SharedHeadNormWeight ?? _outputNormWeight;
            RmsNorm.Execute(cur, headNormWeight, eps, normedHead);

            nint headWeight = mtpHead.SharedHeadHeadWeight ?? _outputWeight;
            QuantizationType headQt = mtpHead.SharedHeadHeadWeight is not null
                ? mtpHead.SharedHeadHeadQuantType : _outputQuantType;
            int headOutputDim = mtpHead.SharedHeadHeadWeight is not null ? vocabSize : _outputOutputDim;
            int headInputDim = mtpHead.SharedHeadHeadWeight is not null ? hiddenSize : _outputInputDim;

            fixed (float* normedHeadPtr = normedHead, logitsPtr = logits)
                Gemm(headWeight, headQt, normedHeadPtr, logitsPtr, headOutputDim, headInputDim, 1);

            var shape = new TensorShape(1, vocabSize);
            var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
            logits.CopyTo(new Span<float>((void*)result.DataPointer, vocabSize));
            return result;
        }
        finally
        {
            pool.Return(tokEmbedArr);
            pool.Return(eNormArr);
            pool.Return(hNormArr);
            pool.Return(concatArr);
            pool.Return(curArr);
            pool.Return(residualArr);
            pool.Return(normedArr);
            pool.Return(qgArr);
            pool.Return(qArr);
            pool.Return(gateArr);
            pool.Return(kArr);
            pool.Return(vArr);
            pool.Return(attnOutArr);
            pool.Return(ffnGateArr);
            pool.Return(ffnUpArr);
            pool.Return(siluArr);
            pool.Return(normedHeadArr);
            pool.Return(logitsArr);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Gemm(nint weights, QuantizationType qt, float* b, float* c, int m, int k, int n,
                      byte* preQuantizedInput = null)
    {
        // Pool-aware GEMM dispatch — mirrors TransformerModel.Gemm and
        // Qwen3MoeHybridTransformerModel.Gemm. Adds PQ2_0 (Bonsai's ternary quant type) and
        // I2_S arms that the MoE hybrid's own copy of this switch is still missing (tracked
        // separately) — every dense-FFN and GDN/attention projection on Bonsai is PQ2_0, so
        // this arm is on the hot path for every layer, not an edge case.
        switch (qt)
        {
            case QuantizationType.PQ2_0:
                MatMul.GemmPQ2_0((byte*)weights, b, c, m, k, n, _threadPool);
                return;
            case QuantizationType.I2_S:
                MatMul.GemmI2_S((byte*)weights, b, c, m, k, n, _threadPool);
                return;
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
                // Shared dequantize-and-dot fallback (#263): decodes each weight row once and
                // reuses it across all n columns instead of re-decoding the matrix per token.
                MatMul.GemmDequantRows((byte*)weights, qt, b, c, m, k, n, pool: null);
                return;
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
