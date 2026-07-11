using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Transformer forward pass: embedding lookup → N × transformer blocks → final norm → LM head → logits.
/// Operates entirely on the CPU using pre-allocated scratch buffers for zero-allocation inference.
/// </summary>
public sealed unsafe class TransformerModel : IModel
{
    /// <summary>Q8_0 block: 2 bytes (Half scale) + 32 bytes (sbyte values).</summary>
    private const int Q8_0BlockBytes = 34;

    /// <summary>Elements per Q8_0 block.</summary>
    private const int Q8_0GroupSize = 32;

    /// <summary>Elements per Q8_1 block.</summary>
    private const int Q8_1GroupSize = 32;

    private readonly TransformerWeights _weights;
    private readonly TransformerForwardState _state;
    // Persistent KV cache for MLA layers. Exactly one of these is non-null
    // at any time, selected by Config.MlaConfig.UseLatentCache at first use.
    // Both are lazily constructed on the first MLA forward and reset when
    // the caller signals a fresh sequence via positions[0] == 0. See
    // MlaExpandedKvState / MlaLatentKvState docstrings for the Phase A vs
    // Phase B distinction (correctness oracle vs ~7× memory win).
    private MlaExpandedKvState? _mlaKvState;
    private MlaLatentKvState? _mlaLatentKvState;
    // Active LoRA adapter for the current Forward call (when invoked via the
    // 5-arg adapter-aware overload). Cleared back to null in the try/finally
    // surrounding the call. Not thread-safe — TransformerModel as a whole is
    // single-threaded per instance (forward state buffers, MLA caches are
    // also instance-scoped) so this is consistent with existing semantics.
    private ILoraAdapter? _currentAdapter;
    // Active attention-mask spec for the current Forward call. Set by the mask-aware
    // Forward overload, cleared back to Causal in the try/finally surrounding the call.
    // Same single-threaded-per-instance contract as _currentAdapter. Defaults to Causal
    // so every existing call path is byte-identical to the pre-bidirectional code.
    private AttentionMaskSpec _currentMaskSpec = AttentionMaskSpec.Causal;
    // DiffusionGemma self-conditioning state for the NEXT forward (set by the diffusion
    // generator via SetDiffusionSelfCond each denoise step). _scUse is the SC gate (0 on
    // step 0 ⇒ zero-SC, exactly the AR/LLaDA-identical path; 1 on steps > 0). _scPrevLogits
    // holds the previous step's canvas-region logits [_scCanvasLen × vocab] (post-softcap);
    // null/zero-length on step 0. Single-threaded per generation, like _currentMaskSpec.
    private float[]? _scPrevLogits;
    private int _scCanvasLen;
    private float _scUse;
    // ── DiffusionGemma prompt-KV (PKV) phase state ──────────────────────────
    // Drives the two-phase PKV optimisation inside RunGemma4Layer. None: normal
    // cacheless forward (DEFAULT — every other path unchanged). Prefill: capture
    // each layer's prompt K/V into _pkvStore. Decode: read cached prompt K/V and
    // attend [prompt K/V | fresh canvas K/V] under a rectangular bidirectional mask.
    // _pkvPromptLen is the prompt length P (the cached prefix); the canvas RoPE
    // positions on a decode forward start at P. Single-threaded per generation,
    // like _currentMaskSpec / _scUse.
    private DiffusionKvPhase _pkvPhase = DiffusionKvPhase.None;
    private DiffusionPromptKvStore? _pkvStore;
    private int _pkvPromptLen;

    private enum DiffusionKvPhase { None, Prefill, Decode }
    // Lifetime anchor for the underlying mmap-backed weight file. Holds a
    // strong reference so the GC cannot collect the GgufFile / SafetensorsFile
    // while weight pointers are still in use. Not null for any loaded model.
#pragma warning disable IDE0052, CA1823 // field used only as a GC root
    private readonly object _mmapAnchor;
#pragma warning restore IDE0052, CA1823
    private readonly int _ropeDim;
    private readonly RoPEType _ropeType;
    // Per-attention-type RoPE element-pairing convention for the FULL-attention
    // layers (Gemma 4). Equal to _ropeType when no global RoPE table is present.
    private readonly RoPEType _globalRopeType;
    private readonly int? _slidingWindowSize;
    private readonly ComputeThreadPool? _threadPool;
    private readonly bool _ownsThreadPool;
    // Gemma-family embedding scale: input embeddings are multiplied by
    // sqrt(hidden_size) immediately after the lookup. 1.0f (a no-op) for every
    // architecture that leaves ModelConfig.EmbeddingScale null.
    private readonly float _embeddingScale;
    // True when the dense FFN must use the GeGLU (tanh-approximate GELU) gate
    // activation instead of SwiGLU (SiLU). Gemma sets ActivationFunction =
    // GELUTanh; every other dense architecture keeps SwiGLU.
    private readonly bool _useGeGLU;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <summary>Total bytes allocated for inference scratch buffers.</summary>
    public long ComputeMemoryBytes => _state.AllocatedBytes;

    /// <summary>Debug: limit the number of transformer layers processed. 0 = all layers (default). -1 = skip all layers (embedding + LM head only).</summary>
    internal int DebugMaxLayers { get; set; }

    /// <summary>
    /// Diagnostic hybrid hook (bug-#2 bisection). When set, Gemma-4 layers
    /// selected by <see cref="Gemma4LayerOverrideSelector"/> are computed by this
    /// delegate (e.g. a Vulkan single-layer) instead of the in-process
    /// <c>RunGemma4Layer</c> — letting a working CPU forward swap layers to
    /// another backend one at a time and re-test. The delegate must overwrite the
    /// <c>[seqLen × hiddenSize]</c> residual stream in place. NOT a production path.
    /// </summary>
    internal unsafe delegate void Gemma4LayerOverrideFn(float* hidden, int layer, int seqLen);
    /// <summary>Diagnostic override for selected Gemma-4 layers. See <see cref="Gemma4LayerOverrideFn"/>.</summary>
    internal Gemma4LayerOverrideFn? Gemma4LayerOverride { get; set; }
    /// <summary>Predicate selecting which layers the override applies to. Null ⇒ all gemma4 layers when an override is set.</summary>
    internal Func<int, bool>? Gemma4LayerOverrideSelector { get; set; }
    /// <summary>Diagnostic: invoked with the residual stream right AFTER the embedding lookup (layer = -1), so a harness can capture or replace the embedding (e.g. inject a Vulkan embedding). NOT a production path.</summary>
    internal Gemma4LayerOverrideFn? Gemma4PostEmbeddingHook { get; set; }

    private TransformerModel(ModelConfig config, TransformerWeights weights, TransformerForwardState state,
                       object? mmapAnchor, int ropeDim, RoPEType ropeType,
                       ComputeThreadPool? threadPool, bool ownsPool, RoPEType globalRopeType = RoPEType.Norm)
    {
        Config = config;
        _weights = weights;
        _state = state;
        // mmapAnchor is non-null on production load paths (mmap-backed weights need to be
        // pinned for the model's lifetime); test paths that build from prebuilt weights
        // may pass null when the source memory is owned by the caller.
        _mmapAnchor = mmapAnchor!;
        _ropeDim = ropeDim;
        _ropeType = ropeType;
        _globalRopeType = globalRopeType;
        _slidingWindowSize = config.SlidingWindowSize;
        _threadPool = threadPool;
        _ownsThreadPool = ownsPool;
        _embeddingScale = config.EmbeddingScale ?? 1.0f;
        _useGeGLU = config.ActivationFunction == ActivationFunction.GELUTanh;
    }

    /// <summary>
    /// Loads a transformer model from an opened GGUF file (single-threaded).
    /// The <paramref name="gguf"/> must remain alive for the lifetime of the returned model.
    /// </summary>
    public static TransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config)
        => LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);

    /// <summary>
    /// Loads a transformer model from an opened GGUF file with threading configuration.
    /// When <paramref name="threading"/> is parallel, creates a <see cref="ComputeThreadPool"/>
    /// owned by this model (disposed with the model).
    /// </summary>
    public static TransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config, ThreadingConfig threading)
    {
        var weights = TransformerWeights.LoadFromGguf(gguf, config);
        // Route through the shared state builder so the GGUF path gets the same
        // per-attention-type RoPE tables, partial-rotary handling, and distinct
        // per-layer head-dim scratch sizing as the safetensors path (Gemma 4 needs
        // all three). The GGUF file is the mmap anchor.
        return BuildFromPrebuiltWeightsInternal(weights, config, threading, anchorSource: gguf);
    }

    /// <summary>
    /// Loads a transformer model from an opened HuggingFace-convention
    /// safetensors file (single-threaded). The <paramref name="file"/> must
    /// remain alive for the lifetime of the returned model — internally
    /// anchored to prevent GC, but the caller must still dispose it after
    /// disposing the model.
    /// </summary>
    public static TransformerModel LoadFromSafetensors(ISafetensorsTensorSource file, ModelConfig config)
        => LoadFromSafetensors(file, config, ThreadingConfig.SingleThreaded);

    /// <summary>
    /// Loads a transformer model from an opened HuggingFace-convention
    /// safetensors source (single-file or multi-shard) with threading
    /// configuration.
    /// </summary>
    public static TransformerModel LoadFromSafetensors(
        ISafetensorsTensorSource file, ModelConfig config, ThreadingConfig threading)
        => LoadFromSafetensors(file, config, threading, i2sCache: null);

    /// <summary>
    /// Loads a transformer model from an opened HuggingFace-convention safetensors source with
    /// threading configuration and an optional BitNet I2_S weight cache. When
    /// <paramref name="i2sCache"/> is supplied (BitNet checkpoints only), each linear
    /// projection's ternary-packed bytes are served from / persisted to disk, avoiding the
    /// dominant online bf16→I2_S quantization cost on repeated loads.
    /// </summary>
    internal static TransformerModel LoadFromSafetensors(
        ISafetensorsTensorSource file, ModelConfig config, ThreadingConfig threading,
        BitNetI2SCacheContext? i2sCache)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(config);

        var weights = TransformerWeightsSafetensorsLoader.Load(file, config, i2sCache);
        return BuildFromPrebuiltWeightsInternal(weights, config, threading, anchorSource: file);
    }

    /// <summary>
    /// Test-only factory that wires a model around already-built CPU
    /// <see cref="TransformerWeights"/>. Used by parity tests that mutate
    /// <see cref="MoeLayerWeights"/> (e.g. attaching a Q8_0 overlay) between the
    /// safetensors load and model construction — there is no production loader
    /// today that emits a Q8_0-overlay-bearing <c>MoeLayerWeights</c>, so the
    /// tests need to skip the loader's F32 upcast for those projections.
    /// </summary>
    internal static TransformerModel BuildFromPrebuiltWeights(
        TransformerWeights weights, ModelConfig config, ThreadingConfig? threading = null)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(config);
        return BuildFromPrebuiltWeightsInternal(weights, config, threading ?? ThreadingConfig.SingleThreaded, anchorSource: null);
    }

    private static TransformerModel BuildFromPrebuiltWeightsInternal(
        TransformerWeights weights, ModelConfig config, ThreadingConfig threading, object? anchorSource)
    {
        weights.RepackWeights();

        // For MLA (DeepSeek-V2/V3) RoPE applies only to the decoupled
        // qk_rope_head_dim sub-dimension — NOT the full qk_head_dim carried
        // in ModelConfig.HeadDim. Size the RoPE table accordingly so the MLA
        // kernel's [pos, qk_rope_head_dim / 2] indexing lines up.
        int ropeDim = config.MlaConfig is not null
            ? config.MlaConfig.QkRopeHeadDim
            : (config.RoPEConfig?.DimensionCount ?? config.HeadDim);
        if (ropeDim == 0) ropeDim = config.HeadDim;
        float ropeTheta = config.MlaConfig?.RopeTheta ?? config.RoPEConfig?.Theta ?? 10000.0f;
        RoPEType ropeType = config.RoPEConfig?.Type ?? RoPEType.Norm;

        // Per-attention-type RoPE (Gemma 4 / DiffusionGemma). Full-attention
        // layers may use a different base theta AND a partial-rotary factor than
        // the sliding-window layers. We build a secondary cos/sin table for the
        // full-attention layers and dispatch per layer in the forward pass.
        // The full-attention layers may ALSO use a distinct per-head dimension
        // (GlobalHeadDim != HeadDim): the cacheless forward resolves the layer
        // head dim per layer and sizes its scratch for the larger of the two
        // (handled below). A KV-cache combined with a distinct head dim is
        // rejected at Forward time (the single-stride cache can't hold two
        // per-layer K/V block sizes) — see GuardKvCacheHeadDim.
        int globalRopeDim = 0;
        float globalRopeTheta = 0f;
        RoPEType globalRopeType = ropeType;
        if (config.GlobalRoPEConfig is RoPEConfig gcfg && config.MlaConfig is null)
        {
            // RoPE rotates pairs within each head's leading dims. For a partial
            // rotary factor on the full-attention layers, the rotated span is a
            // fraction of the FULL-attention head dim (GlobalHeadDim when set).
            int baseDim = gcfg.DimensionCount > 0
                ? gcfg.DimensionCount
                : (config.GlobalHeadDim ?? config.HeadDim);
            // Partial-rotary factor (Gemma 4 full-attention layers): only the
            // leading round-down-to-even fraction of each head rotates.
            float prf = config.PartialRotaryFactor ?? 1.0f;
            int rotated = (int)MathF.Floor(prf * baseDim);
            rotated &= ~1; // round down to even (RoPE rotates dim pairs)
            if (rotated < 2) rotated = 2;
            globalRopeDim = Math.Min(rotated, baseDim);
            globalRopeTheta = gcfg.Theta;
            globalRopeType = gcfg.Type;
        }

        // Per-token Q/KV scratch block sizes. With a uniform head dim these are
        // the standard numHeads*headDim and numKvHeads*headDim. With Gemma 4's
        // distinct global_head_dim the full-attention layers project a wider
        // per-head slice and a different KV-head count, so we size each block to
        // the LARGER of the sliding and full layer types — a single allocation
        // then covers both. The per-layer dispatch in the forward uses the
        // layer's own head dim / KV-head count.
        int slidingHeadDim = config.HeadDim;
        int fullHeadDim = config.GlobalHeadDim ?? config.HeadDim;
        int slidingKvHeads = config.NumKvHeads;
        int fullKvHeads = config.NumGlobalKvHeads ?? config.NumKvHeads;
        int qBlockElems = config.NumAttentionHeads * Math.Max(slidingHeadDim, fullHeadDim);
        int kvBlockElems = Math.Max(slidingKvHeads * slidingHeadDim, fullKvHeads * fullHeadDim);

        var state = new TransformerForwardState(
            config.HiddenSize,
            config.NumAttentionHeads,
            // Size the K/V scratch for the LARGER of the two KV-head counts so
            // both sliding (NumKvHeads) and full (NumGlobalKvHeads) layers fit.
            // Per-layer dispatch uses the layer's own count. (kvBlockElems below
            // supersedes this for the distinct-head-dim case.)
            Math.Max(config.NumKvHeads, config.NumGlobalKvHeads ?? config.NumKvHeads),
            config.HeadDim,
            config.IntermediateSize,
            config.VocabSize,
            config.MaxSequenceLength,
            ropeDim,
            ropeTheta,
            globalRopeDim,
            globalRopeTheta,
            qBlockElems,
            kvBlockElems,
            // Full global head dim — the partial-rotary frequency denominator
            // (Gemma 4 global layers: rotate globalRopeDim=128 dims but use freq
            // base over the full head dim 512). Equals globalRopeDim when there is
            // no partial rotary (Gemma 3), collapsing to the standard precompute.
            globalFullHeadDim: config.GlobalHeadDim ?? config.HeadDim);

        // For MLA + YaRN (DeepSeek-V2/V3 long-context), rebuild cos/sin tables
        // using per-dim ramped inverse frequencies. Plain precompute above is a
        // no-op for positions < original_max_position_embeddings, but the two
        // paths diverge beyond that threshold — the ramp mixes the original
        // base (fast rotations — extrapolation) with the scaled base (slow
        // rotations — interpolation). See DeepSeek-V2 paper §3.2.
        if (config.MlaConfig is { RopeScalingFactor: float scalingFactor } mla
            && scalingFactor > 1.0f
            && mla.RopeScalingOriginalMaxPositionEmbeddings is int originalMaxPos
            && originalMaxPos > 0)
        {
            // HF multiplies cos/sin by yarn_get_mscale(factor, mscale) /
            // yarn_get_mscale(factor, mscale_all_dim). For V2-Lite mscale ==
            // mscale_all_dim so the ratio is 1.0 — we still compute it to
            // track configs where they diverge.
            float mscaleNum = (mla.RopeScalingMscale is float m && m != 0.0f)
                ? 0.1f * m * MathF.Log(scalingFactor) + 1.0f : 1.0f;
            float mscaleDen = (mla.RopeScalingMscaleAllDim is float mad && mad != 0.0f)
                ? 0.1f * mad * MathF.Log(scalingFactor) + 1.0f : 1.0f;
            float mscaleMultiplier = mscaleNum / mscaleDen;

            DotLLM.Cpu.Kernels.RoPE.PrecomputeFrequencyTableYarn(
                config.MaxSequenceLength, ropeDim, ropeTheta,
                scalingFactor, originalMaxPos,
                mla.RopeScalingBetaFast, mla.RopeScalingBetaSlow,
                mscaleMultiplier,
                state.CosTable, state.SinTable);
        }
        // Dense-path YaRN (SmolLM3 128k SKU, Llama 3.1+ extended context). Same
        // ramped-inverse-frequency kernel as MLA above, only without the MLA
        // mscale split (RoPEConfig.AttnFactor carries the optional softmax
        // multiplier — when ScalingFactor>1 and OrigMaxSeqLen>0, the YaRN ramp
        // is applied; positions below the threshold still produce identical
        // cos/sin to the plain table, so the base 3B SmolLM3 (scaling=null,
        // factor==1) is byte-identical to the non-YaRN path).
        else if (config.MlaConfig is null
                 && config.RoPEConfig is RoPEConfig rcfg
                 && rcfg.ScalingType == RoPEScalingType.YaRN
                 && rcfg.ScalingFactor > 1.0f
                 && rcfg.OrigMaxSeqLen > 0)
        {
            DotLLM.Cpu.Kernels.RoPE.PrecomputeFrequencyTableYarn(
                config.MaxSequenceLength, ropeDim, ropeTheta,
                rcfg.ScalingFactor, rcfg.OrigMaxSeqLen,
                rcfg.BetaFast, rcfg.BetaSlow,
                mscaleMultiplier: rcfg.AttnFactor,
                state.CosTable, state.SinTable);
        }

        ComputeThreadPool? pool = null;
        if (threading.IsParallel)
        {
            int effectiveThreads = threading.EffectiveThreadCount;
            if (threading.EnableNumaPinning || threading.EnablePCorePinning)
            {
                var topology = NumaTopology.Detect();
                if (threading.EnablePCorePinning && topology.IsHybrid)
                    effectiveThreads = Math.Min(effectiveThreads, topology.PerformanceCoreIds.Count);
                pool = new ComputeThreadPool(effectiveThreads, topology, threading);
            }
            else
            {
                pool = new ComputeThreadPool(effectiveThreads, topology: null, threading);
            }
        }

        return new TransformerModel(config, weights, state, anchorSource, ropeDim, ropeType, pool, ownsPool: pool is not null, globalRopeType);
    }

    /// <summary>
    /// Returns true when this model has a distinct per-layer attention head
    /// dimension (Gemma 4 <c>global_head_dim</c> != <c>head_dim</c>). The cacheless
    /// forward path fully supports this (scratch buffers + the per-layer attention
    /// dispatch are sized/threaded per layer); the KV-cached path does not — the
    /// single-stride KV cache cannot hold two distinct per-layer K/V block sizes.
    /// </summary>
    private bool HasDistinctPerLayerHeadDim =>
        Config.GlobalHeadDim is int ghd && ghd != Config.HeadDim;

    /// <summary>
    /// Validates a supplied KV-cache against a distinct-per-layer-head-dim model
    /// (Gemma 4 <c>global_head_dim</c> != <c>head_dim</c>). Such a model has DIFFERENT
    /// per-layer K/V row widths (full-attention layers <c>NumGlobalKvHeads *
    /// GlobalHeadDim</c>; sliding layers <c>NumKvHeads * HeadDim</c>). As of KV Phase 0
    /// the contiguous F32 <c>SimpleKvCache</c> supports this via per-layer
    /// strides, so the cache is accepted when its geometry MATCHES
    /// <see cref="KvGeometry.FromConfig"/>; a mismatched cache (e.g. a uniform cache
    /// built for a different model) fails fast. Quantized / paged caches do not yet
    /// carry per-layer strides for distinct-head-dim models and are still rejected.
    /// Non-distinct (uniform) models skip this entirely — every cache type is valid.
    /// </summary>
    private void GuardKvCacheHeadDim(IKvCache? kvCache)
    {
        if (kvCache is null || !HasDistinctPerLayerHeadDim)
            return;

        var geom = KvGeometry.FromConfig(Config);

        if (kvCache is IPerLayerKvCache perLayer)
        {
            if (perLayer.LayerCount != geom.LayerCount)
                throw new ArgumentException(
                    $"KV-cache layer count {perLayer.LayerCount} does not match model layer count "
                    + $"{geom.LayerCount}.", nameof(kvCache));
            for (int l = 0; l < geom.LayerCount; l++)
            {
                if (perLayer.KvStrideOf(l) != geom.KvStrideOf(l))
                    throw new ArgumentException(
                        $"KV-cache geometry mismatch at layer {l}: cache stride {perLayer.KvStrideOf(l)} "
                        + $"!= model stride {geom.KvStrideOf(l)}. Build the cache with "
                        + "KvGeometry.FromConfig(Config).", nameof(kvCache));
            }
            return;
        }

        throw new NotSupportedException(
            $"Gemma 4 with distinct global_head_dim ({Config.GlobalHeadDim}) and head_dim "
            + $"({Config.HeadDim}) requires a per-layer-strided KV-cache. Only the contiguous F32 "
            + $"SimpleKvCache carries per-layer strides today; the supplied "
            + $"{kvCache.GetType().Name} does not. Build a SimpleKvCache with "
            + "KvGeometry.FromConfig(Config) (F32) or run cacheless. Per-layer quantized / paged "
            + "KV strides are tracked as future work.");
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <summary>
    /// LoRA-aware forward. When <paramref name="adapter"/> is non-null, each
    /// adapted projection adds <c>scale × (x · B) · A</c> on top of the base
    /// projection. When null, this is byte-equivalent to the 4-arg overload.
    /// </summary>
    /// <remarks>
    /// MoE FFN sites are not adapted in this Phase 4a slice — if the model
    /// has any MoE layer AND <paramref name="adapter"/> targets a gate / up /
    /// down projection, the call throws <see cref="NotSupportedException"/>.
    /// MLA-specific projections (DeepSeek-V2/V3 q_a_proj, kv_a_proj_with_mqa,
    /// …) are also out of scope and silently passed through.
    /// </remarks>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache, ILoraAdapter? adapter)
    {
        if (adapter is null)
            return Forward(tokenIds, positions, deviceId, kvCache);

        ValidateAdapterForModel(adapter);

        // Phase 4d.6 — eager transposed-A materialisation. The outer-product
        // stage-2 fast path needs a [rank, outputDim] view of A; building it
        // is O(outputDim × rank) per (layer, proj) — a few ms total for a
        // typical Llama-3.2-1B / rank=16 adapter. PrewarmAdapter is
        // idempotent so the actual cost is paid only on first activation;
        // hoisting it out of the per-Apply lazy path eliminates first-token
        // latency contamination AND smooths low-iteration BDN measurement
        // variance. No-op for rank != 16 or non-AVX-512 hosts.
        LoraStage2.PrewarmAdapter(adapter as LoraAdapter);

        _currentAdapter = adapter;
        try
        {
            return Forward(tokenIds, positions, deviceId, kvCache);
        }
        finally
        {
            _currentAdapter = null;
        }
    }

    /// <summary>
    /// Mask-mode-aware forward (causal / bidirectional / hybrid). When
    /// <paramref name="maskSpec"/> is the causal default this is byte-identical to the
    /// adapter-aware overload — the mask spec only diverts the core attention masking for
    /// the non-causal modes.
    /// </summary>
    /// <remarks>
    /// Non-causal masking is wired through the cacheless GQA attention path only. A non-null
    /// <paramref name="kvCache"/> combined with a non-causal mask throws
    /// <see cref="NotSupportedException"/> — the canvas/KV-cache interaction is a later PR (#28).
    /// MLA layers are causal-only and likewise reject non-causal masks.
    /// </remarks>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache, ILoraAdapter? adapter, AttentionMaskSpec maskSpec)
    {
        if (maskSpec.IsCausal)
            return Forward(tokenIds, positions, deviceId, kvCache, adapter);

        if (kvCache is not null)
            throw new NotSupportedException(
                "Non-causal attention (bidirectional / hybrid) is only supported on the cacheless "
                + "forward path. KV-cache wiring for the diffusion canvas is tracked separately (#28).");
        if (Config.MlaConfig is not null)
            throw new NotSupportedException(
                "Non-causal attention is not supported for MLA (DeepSeek-V2/V3) layers.");
        if (maskSpec.Mode == AttentionMaskMode.Hybrid && maskSpec.PrefixLength > tokenIds.Length)
            throw new ArgumentOutOfRangeException(nameof(maskSpec),
                $"Hybrid prefix length {maskSpec.PrefixLength} exceeds sequence length {tokenIds.Length}.");

        _currentMaskSpec = maskSpec;
        _currentAdapter = adapter;
        try
        {
            if (adapter is not null)
            {
                ValidateAdapterForModel(adapter);
                LoraStage2.PrewarmAdapter(adapter as LoraAdapter);
            }
            return Forward(tokenIds, positions, deviceId, kvCache: null);
        }
        finally
        {
            _currentAdapter = null;
            _currentMaskSpec = AttentionMaskSpec.Causal;
        }
    }

    /// <inheritdoc/>
    public void SetDiffusionSelfCond(ReadOnlySpan<float> prevCanvasLogits, int canvasLen, float scUse)
    {
        // Stash the previous step's canvas logits for the canvas region-embed in the
        // NEXT forward. scUse == 0 (step 0) ⇒ no SC: clear the buffer so the region
        // embed keeps the byte-identical zero-SC rms_noscale path. We copy into a model-
        // owned buffer (the caller's span is a view into a pooled/disposed tensor).
        if (scUse > 0f && !prevCanvasLogits.IsEmpty && canvasLen > 0)
        {
            int need = canvasLen * Config.VocabSize;
            if (prevCanvasLogits.Length < need)
                throw new ArgumentException(
                    $"prevCanvasLogits length {prevCanvasLogits.Length} < canvasLen*vocab {need}.",
                    nameof(prevCanvasLogits));
            if (_scPrevLogits is null || _scPrevLogits.Length < need)
                _scPrevLogits = new float[need];
            prevCanvasLogits[..need].CopyTo(_scPrevLogits);
            _scCanvasLen = canvasLen;
            _scUse = scUse;
        }
        else
        {
            _scCanvasLen = 0;
            _scUse = 0f;
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Supported only on the diffusion-gemma backbone (the Gemma-4 tower with a non-null
    /// <see cref="ModelConfig.DiffusionConfig"/>). PKV reuses the per-layer prompt K/V across
    /// denoise steps; the cacheless unified forward remains the default and is unaffected.
    /// </remarks>
    public bool SupportsDiffusionPromptKv =>
        Config.DiffusionConfig is not null && _weights.Layers.Length > 0
        && _weights.Layers[0].Gemma4 is not null;

    /// <inheritdoc/>
    public void DiffusionPrefillPromptKv(
        ReadOnlySpan<int> promptTokens, ReadOnlySpan<int> positions, DiffusionPromptKvStore store)
    {
        ArgumentNullException.ThrowIfNull(store);
        if (!SupportsDiffusionPromptKv)
            throw new NotSupportedException(
                "DiffusionPrefillPromptKv requires a diffusion-gemma (Gemma-4) model.");
        if (promptTokens.Length == 0)
            throw new ArgumentException("Prompt must be non-empty for PKV prefill.", nameof(promptTokens));
        if (promptTokens.Length != positions.Length)
            throw new ArgumentException("promptTokens and positions length mismatch.", nameof(positions));

        int p = promptTokens.Length;
        int numLayers = Config.NumLayers;
        // Per-layer KV block width (nKvHead*headDim) — differs for sliding vs global layers.
        Span<int> kvBlockElems = numLayers <= 256 ? stackalloc int[numLayers] : new int[numLayers];
        for (int l = 0; l < numLayers; l++)
            kvBlockElems[l] = GetLayerKvHeads(l) * Config.GetLayerHeadDim(l);
        store.BeginPrefill(p, kvBlockElems);

        // Run a prompt-only causal forward (Hybrid(P) with P == seqLen ⇒ every row is in the
        // causal prefix, so the region-embed and region scalar deltas are inert exactly as the
        // unified path's prompt rows). RunGemma4Layer captures K/V per layer while _pkvPhase is
        // Prefill. No SC on the prompt (the generator clears SC before prefill). No KV-cache.
        _pkvStore = store;
        _pkvPromptLen = p;
        _pkvPhase = DiffusionKvPhase.Prefill;
        _currentMaskSpec = AttentionMaskSpec.Hybrid(p);
        try
        {
            RunLayersAndFinalNormCore(promptTokens, positions, kvCache: null);
            // No LM head needed for prefill — we only want the captured K/V. (Running the head
            // would be wasted work; the store now holds every layer's prompt K/V.)
        }
        finally
        {
            _pkvPhase = DiffusionKvPhase.None;
            _pkvStore = null;
            _pkvPromptLen = 0;
            _currentMaskSpec = AttentionMaskSpec.Causal;
        }
    }

    /// <inheritdoc/>
    public ITensor DiffusionDecodeWithPromptKv(
        ReadOnlySpan<int> canvasTokens, ReadOnlySpan<int> positions, int deviceId,
        DiffusionPromptKvStore store)
    {
        ArgumentNullException.ThrowIfNull(store);
        if (!SupportsDiffusionPromptKv)
            throw new NotSupportedException(
                "DiffusionDecodeWithPromptKv requires a diffusion-gemma (Gemma-4) model.");
        if (store.PromptLen <= 0)
            throw new InvalidOperationException("PKV store is empty — run DiffusionPrefillPromptKv first.");
        if (canvasTokens.Length == 0)
            throw new ArgumentException("Canvas must be non-empty for PKV decode.", nameof(canvasTokens));
        if (canvasTokens.Length != positions.Length)
            throw new ArgumentException("canvasTokens and positions length mismatch.", nameof(positions));

        _pkvStore = store;
        _pkvPromptLen = store.PromptLen;
        _pkvPhase = DiffusionKvPhase.Decode;
        // Bidirectional mask spec so the region-embed (p == 0 ⇒ all C rows are canvas) and the
        // region per-layer scalar (regionP == 0 ⇒ all C rows use layer_output_scale) treat every
        // decode row as a canvas row. The actual canvas↔[prompt|canvas] attention is built inside
        // RunGemma4Layer from the cached prompt K/V (the mask spec's attention mode is overridden
        // there for the Decode phase).
        _currentMaskSpec = AttentionMaskSpec.Bidirectional;
        try
        {
            RunLayersAndFinalNormCore(canvasTokens, positions, kvCache: null);
            return RunLmHead(canvasTokens.Length, deviceId);
        }
        finally
        {
            _pkvPhase = DiffusionKvPhase.None;
            _pkvStore = null;
            _pkvPromptLen = 0;
            _currentMaskSpec = AttentionMaskSpec.Causal;
        }
    }

    /// <summary>
    /// Runs a forward pass with optional KV-cache. When <paramref name="kvCache"/> is provided,
    /// K/V projections are stored in the cache after RoPE, and attention reads from the full
    /// cached context — enabling O(1) per-token decode instead of O(n) recomputation.
    /// </summary>
    /// <param name="tokenIds">Input token IDs for this step (all prompt tokens for prefill, single token for decode).</param>
    /// <param name="positions">Position indices for each token.</param>
    /// <param name="deviceId">Target device for computation.</param>
    /// <param name="kvCache">Optional KV-cache. When null, behaves identically to the uncached forward pass.</param>
    /// <returns>Logits tensor of shape [seqLen, vocab_size] for all input positions.</returns>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache)
    {
        RunLayersAndFinalNormCore(tokenIds, positions, kvCache);
        return RunLmHead(tokenIds.Length, deviceId);
    }

    /// <summary>
    /// Returns the effective sliding-window size for <paramref name="layer"/>.
    /// Honours <see cref="ModelConfig.PerLayerSlidingWindow"/> when set (each entry
    /// may be null for full attention or a positive int for sliding); otherwise
    /// falls back to the model-wide <see cref="ModelConfig.SlidingWindowSize"/>.
    /// Used for Gemma 3's interleaved local/global pattern.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int? GetLayerSlidingWindow(int layer)
    {
        var perLayer = Config.PerLayerSlidingWindow;
        if (perLayer is not null && (uint)layer < (uint)perLayer.Count)
            return perLayer[layer];
        return _slidingWindowSize;
    }

    /// <summary>
    /// Resolves the per-attention-type RoPE table set + rotated-dim + element
    /// pairing for <paramref name="layer"/>. Returns the secondary (global)
    /// table — different base theta and optional partial-rotary — for the
    /// full-attention layers when a global RoPE table is present (Gemma 4);
    /// otherwise returns the primary (sliding) table. When no global table
    /// exists every layer uses the primary table, so non-Gemma-4 architectures
    /// are unaffected.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private (float[] Cos, float[] Sin, int RopeDim, RoPEType Type) GetLayerRope(int layer)
    {
        if (_state.GlobalCosTable is float[] gCos && _state.GlobalSinTable is float[] gSin
            && Config.IsFullAttentionLayer(layer))
        {
            return (gCos, gSin, _state.GlobalRopeDim, _globalRopeType);
        }
        return (_state.CosTable, _state.SinTable, _ropeDim, _ropeType);
    }

    /// <summary>
    /// Resolves the KV-head count for <paramref name="layer"/>. Full-attention
    /// layers use <see cref="ModelConfig.NumGlobalKvHeads"/> when set (Gemma 4);
    /// sliding-window layers and every other architecture use
    /// <see cref="ModelConfig.NumKvHeads"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int GetLayerKvHeads(int layer)
    {
        if (Config.NumGlobalKvHeads is int g && Config.IsFullAttentionLayer(layer))
            return g;
        return Config.NumKvHeads;
    }

    /// <summary>
    /// Embedding lookup + transformer layer loop + final RMSNorm. Leaves the final
    /// hidden state in <c>_state.HiddenState[0..seqLen*hiddenSize]</c>. Used by both
    /// <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?)"/>
    /// (which then runs the lm_head per call) and <see cref="ForwardBatch"/> (which
    /// invokes this once per sequence, snapshots each result, then runs ONE batched
    /// lm_head GEMM on the stacked snapshot).
    /// </summary>
    private unsafe void RunLayersAndFinalNormCore(
        ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, IKvCache? kvCache)
    {
        // A distinct per-layer head dim (Gemma 4 global_head_dim) is supported on
        // the cacheless path only — reject a KV-cache up front with a clear message.
        GuardKvCacheHeadDim(kvCache);

        int maxSeq = Config.MaxSequenceLength;
        for (int i = 0; i < positions.Length; i++)
        {
            if ((uint)positions[i] >= (uint)maxSeq)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {positions[i]} at index {i} exceeds max sequence length {maxSeq}.");
        }

        int seqLen = tokenIds.Length;
        int hiddenSize = Config.HiddenSize;
        int numHeads = Config.NumAttentionHeads;
        // Note: the per-head dimension is resolved PER LAYER inside the loop
        // (Config.GetLayerHeadDim) — Gemma 4 full-attention layers may use a
        // distinct GlobalHeadDim. Every other architecture collapses to the
        // uniform Config.HeadDim.
        int intermediateSize = Config.IntermediateSize;
        int vocabSize = Config.VocabSize;
        float eps = Config.NormEpsilon;

        _state.EnsureCapacity(seqLen);

        // Adaptive dispatch mode: spin-wait for decode (short, frequent dispatches),
        // event-based for prefill (long dispatches where kernel transition cost is negligible).
        _threadPool?.SetDispatchMode(seqLen == 1 ? DispatchMode.SpinWait : DispatchMode.EventBased);

        float* hidden = (float*)_state.HiddenState;
        float* residual = (float*)_state.Residual;
        float* normOut = (float*)_state.NormOutput;
        float* q = (float*)_state.Q;
        float* k = (float*)_state.K;
        float* v = (float*)_state.V;
        float* attnOut = (float*)_state.AttnOutput;
        float* ffnGate = (float*)_state.FfnGate;
        float* ffnUp = (float*)_state.FfnUp;
        float* siluOut = (float*)_state.SiluOutput;
        float* logits = (float*)_state.Logits;

        // 1. EMBEDDING LOOKUP
        EmbeddingLookup(tokenIds, hidden, hiddenSize);

        // Diagnostic embedding hook (bug-#2 bisection): capture or replace the
        // scaled embedding before the layers. No-op on the normal path (null).
        Gemma4PostEmbeddingHook?.Invoke(hidden, -1, seqLen);

        // 1b. DIFFUSION region embedding (diffusion-gemma only). The unified
        // [prompt | canvas] forward splits at P = _currentMaskSpec.PrefixLength
        // (the Hybrid prompt length / region split). Prompt rows [0, P) keep the
        // scaled embedding; canvas rows [P, seqLen) get an EXTRA weight-less
        // rms_norm(row, eps) (no scale) (diffusion-gemma.cpp:363-365,378-386).
        // Gated on DiffusionConfig so the autoregressive gemma4 / every other
        // architecture is byte-identical.
        //
        // SELF-CONDITIONING: on denoise steps > 0 (_scUse > 0 with prev canvas logits
        // set + the SC weights present), the canvas rms_noscale is applied to
        // (scaled_embed + sc_signal), where sc_signal is a gated GeGLU MLP over a soft
        // token-embedding of the previous step's canvas logits (dg_canvas_embed). On
        // step 0 (_scUse == 0) this term is skipped ⇒ exactly the zero-SC path.
        if (Config.DiffusionConfig is not null)
        {
            int p = _currentMaskSpec.Mode == AttentionMaskMode.Hybrid
                ? _currentMaskSpec.PrefixLength
                : 0;
            if (p < 0) p = 0;
            if (p > seqLen) p = seqLen;

            int canvasLen = seqLen - p;
            bool applySc = _scUse > 0f
                && _weights.SelfCond is not null
                && _scPrevLogits is not null
                && _scCanvasLen == canvasLen
                && canvasLen > 0;
            if (applySc)
            {
                // Add sc_signal to the canvas rows of `hidden` (scaled embeddings)
                // BEFORE the weight-less rms_norm. soft-embed sweeps the vocab once.
                ApplySelfConditioning(hidden, p, canvasLen, hiddenSize, eps);
            }

            for (int t = p; t < seqLen; t++)
            {
                var row = new Span<float>(hidden + t * hiddenSize, hiddenSize);
                RmsNorm.ExecuteUnit(row, eps, row);
            }
        }

        // 1c. Per-Layer Embeddings (PLE) — Gemma-4 dense text tower (E2B/E4B).
        // Build the per-layer input tensor [seq, numLayers*pleDim] ONCE from the
        // scaled main embedding (`hidden`) and the token-identity table, then inject
        // a gated residual into every layer's output inside the loop. Null for every
        // other architecture (buffers stay null, no work). Cross-backend: CUDA/Vulkan
        // would compute/upload this same buffer and reuse the identical injection.
        float* pleInputs = null, pleIdentity = null, pleGateScratch = null, pleProjScratch = null;
        var pleWeights = _weights.PerLayerEmbedding;
        int pleDim = pleWeights?.PerLayerDim ?? 0;
        int pleLp = pleWeights is not null ? Config.NumLayers * pleDim : 0;
        if (pleWeights is not null)
        {
            pleInputs = (float*)NativeMemory.AlignedAlloc((nuint)(sizeof(float) * seqLen * pleLp), 64);
            pleIdentity = (float*)NativeMemory.AlignedAlloc((nuint)(sizeof(float) * seqLen * pleLp), 64);
            pleGateScratch = (float*)NativeMemory.AlignedAlloc((nuint)(sizeof(float) * seqLen * pleDim), 64);
            pleProjScratch = (float*)NativeMemory.AlignedAlloc((nuint)(sizeof(float) * seqLen * hiddenSize), 64);

            // Token-identity gather: embed_tokens_per_layer[token] scaled by √pleDim.
            GatherPerLayerIdentity(tokenIds, pleIdentity, pleLp, MathF.Sqrt(pleDim),
                pleWeights.EmbedTokensPerLayer, pleWeights.EmbedTokensPerLayerQt);

            // Combine with the context projection (output aliases projScratch = pleInputs).
            PerLayerEmbeddings.ComputeInputs(
                tokenIdentity: pleIdentity,
                inputsEmbeds: hidden,
                projWeight: (float*)pleWeights.ModelProjection,
                projNormWeight: pleWeights.ProjectionNorm,
                projScratch: pleInputs,
                output: pleInputs,
                seqLen: seqLen, hiddenSize: hiddenSize,
                numLayers: Config.NumLayers, pleDim: pleDim, eps: eps);
        }

        // 2. TRANSFORMER LAYERS
        var repackedLayers = _weights.RepackedLayers;
        int numLayers = DebugMaxLayers switch
        {
            < 0 => 0,
            0 => Config.NumLayers,
            _ => Math.Min(DebugMaxLayers, Config.NumLayers)
        };

        // MLA cache lifecycle: allocated lazily on the first MLA forward
        // pass, reset when positions[0] == 0 so successive unrelated calls
        // (integration tests, multiple prompts, …) don't reuse stale KV.
        // Phase A (default) uses MlaExpandedKvState; Phase B / Phase C use
        // the smaller MlaLatentKvState. Phase C (UseHybridMlaCache) shares
        // the Phase B cache layout verbatim — the only difference is which
        // kernel consumes it (absorbed decode, expand-then-MHA prefill).
        // UseLatentCache and UseHybridMlaCache are mutually exclusive.
        if (Config.MlaConfig is not null)
        {
            var mla = Config.MlaConfig;
            if (mla.UseLatentCache && mla.UseHybridMlaCache)
                throw new InvalidOperationException(
                    "MlaConfig.UseLatentCache and MlaConfig.UseHybridMlaCache are mutually exclusive.");

            if (mla.UseLatentCache || mla.UseHybridMlaCache)
            {
                if (_mlaLatentKvState is null)
                {
                    _mlaLatentKvState = new MlaLatentKvState(
                        numLayers: Config.NumLayers,
                        maxSeqLen: Config.MaxSequenceLength,
                        kvLoraRank: mla.KvLoraRank,
                        qkRopeHeadDim: mla.QkRopeHeadDim);
                }
                if (positions[0] == 0)
                    _mlaLatentKvState.Reset();
            }
            else
            {
                if (_mlaKvState is null)
                {
                    _mlaKvState = new MlaExpandedKvState(
                        numLayers: Config.NumLayers,
                        maxSeqLen: Config.MaxSequenceLength,
                        numHeads: Config.NumAttentionHeads,
                        qkNopeHeadDim: mla.QkNopeHeadDim,
                        vHeadDim: mla.VHeadDim,
                        qkRopeHeadDim: mla.QkRopeHeadDim);
                }
                if (positions[0] == 0)
                    _mlaKvState.Reset();
            }
        }

        for (int layer = 0; layer < numLayers; layer++)
        {
            ref readonly var lw = ref _weights.Layers[layer];
            var rl = repackedLayers?[layer];

            // Per-attention-type KV-head count (Gemma 4: full-attention layers
            // use NumGlobalKvHeads, sliding layers use NumKvHeads). Collapses to
            // the uniform Config.NumKvHeads for every other architecture.
            int numKvHeadsLayer = GetLayerKvHeads(layer);
            // Per-attention-type head dim (Gemma 4: full-attention layers use
            // GlobalHeadDim, sliding layers use HeadDim). Collapses to the uniform
            // Config.HeadDim for every other architecture, so headDimLayer ==
            // headDim and qStrideLayer == numHeads*headDim on the standard path.
            int headDimLayer = Config.GetLayerHeadDim(layer);
            int qStrideLayer = numHeads * headDimLayer;
            int kvStrideLayer = numKvHeadsLayer * headDimLayer;

            // Declared once for the whole layer so both the GQA and MLA
            // paths share the same input-quantisation scratch region.
            byte* inputQ8Scratch = (byte*)_state.InputQ8Scratch;

            // a. Copy hiddenState → residual
            new Span<float>(hidden, seqLen * hiddenSize).CopyTo(new Span<float>(residual, seqLen * hiddenSize));

            // ── Gemma 4 MoE branch ───────────────────────────────────────
            // The gemma4 graph is sufficiently distinct (V-from-K, weight-less
            // V-norm, attn scale 1.0, dual-parallel FFN with a custom router,
            // per-expert down scale, layer_output_scale) that it runs as a
            // self-contained per-layer method rather than threading flags through
            // the shared GQA/MoE/dense blocks. Cacheless single-forward only.
            if (lw.Gemma4 is not null)
            {
                // Diagnostic hybrid bisection: a selected layer is computed by the
                // override backend (e.g. Vulkan) instead of in-process. No-op on
                // the normal path (Gemma4LayerOverride null).
                if (Gemma4LayerOverride is { } over
                    && (Gemma4LayerOverrideSelector is null || Gemma4LayerOverrideSelector(layer)))
                {
                    over(hidden, layer, seqLen);
                }
                else
                {
                    RunGemma4Layer(
                        in lw, layer, seqLen,
                        hidden, residual, normOut, q, k, v, attnOut,
                        numKvHeadsLayer, headDimLayer, qStrideLayer, kvStrideLayer,
                        positions, eps, kvCache);
                }
                continue;
            }

            // ── MLA branch (DeepSeek-V2/V3) ──────────────────────────────
            // Routes through the standalone MlaAttention kernel: RMSNorm → Q
            // path (LoRA or monolithic) → KV path (LoRA + MQA-shared rope-K)
            // → decoupled RoPE on the rope sub-dim only → per-head
            // scaled-dot-product attention with causal mask → o_proj.
            //
            // Cache: the kernel writes new K_nope / V / K_pe into the
            // persistent per-layer _mlaKvState store at offset
            // currentLength[layer] and attends over all (currentLength +
            // seqLen) tokens. This is the "non-absorbed reference" path per
            // the P2.3 plan — it matches the cacheless kernel numerically
            // and unblocks generation-loop tests on DeepSeek. Phase B
            // (latent compression + W_UK absorption) will layer on top,
            // using this as the correctness oracle. The caller-supplied
            // IKvCache is still ignored for MLA layers (shape-incompatible).
            if (lw.Mla is not null)
            {
                // RMSNorm per token into normOut (MLA kernel consumes the
                // normalised hidden state).
                for (int t = 0; t < seqLen; t++)
                {
                    RmsNorm.Execute(
                        new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                        lw.AttnNormWeight, eps,
                        new Span<float>(normOut + t * hiddenSize, hiddenSize));
                }

                MlaLayerWeights mlaW = lw.Mla!;
                int qTotalElems = mlaW.NumHeads * (mlaW.QkNopeHeadDim + mlaW.QkRopeHeadDim);
                int kvAElems = mlaW.KvLoraRank + mlaW.QkRopeHeadDim;
                int kvBElems = mlaW.NumHeads * (mlaW.QkNopeHeadDim + mlaW.VHeadDim);
                int oElems = hiddenSize * (mlaW.NumHeads * mlaW.VHeadDim);
                int qAElems = mlaW.QLoraRank > 0 ? mlaW.QLoraRank * hiddenSize : 0;
                int qBElems = mlaW.QLoraRank > 0 ? qTotalElems * mlaW.QLoraRank : 0;
                int qMonoElems = mlaW.QLoraRank > 0 ? 0 : qTotalElems * hiddenSize;

                int ropeHalf = mlaW.QkRopeHeadDim / 2;
                int ropeTableLen = _state.CosTable.Length;

                float mlaScaleMultiplier = Config.MlaConfig!.ComputeYarnSoftmaxScaleMultiplier();
                if (_mlaLatentKvState is not null)
                {
                    // Phase B (pure absorbed) OR Phase C (hybrid
                    // expand-prefill / absorbed-decode) — both share the
                    // latent cache layout; the config flag picks the kernel.
                    bool hybrid = Config.MlaConfig!.UseHybridMlaCache;
                    if (hybrid)
                    {
                        MlaAttention.ExecuteLatentHybrid(
                            hidden: new ReadOnlySpan<float>(normOut, seqLen * hiddenSize),
                            output: new Span<float>(attnOut, seqLen * hiddenSize),
                            seqLen: seqLen,
                            positionOffset: positions[0],
                            hiddenSize: hiddenSize,
                            numHeads: mlaW.NumHeads,
                            qkNopeHeadDim: mlaW.QkNopeHeadDim,
                            qkRopeHeadDim: mlaW.QkRopeHeadDim,
                            vHeadDim: mlaW.VHeadDim,
                            qLoraRank: mlaW.QLoraRank,
                            kvLoraRank: mlaW.KvLoraRank,
                            rmsNormEps: eps,
                            ropeCosTable: _state.CosTable.AsSpan(0, ropeTableLen),
                            ropeSinTable: _state.SinTable.AsSpan(0, ropeTableLen),
                            qAProj: qAElems > 0 ? new ReadOnlySpan<float>((void*)mlaW.QAProj, qAElems) : ReadOnlySpan<float>.Empty,
                            qALayernormWeight: mlaW.QALayernormWeight ?? (ReadOnlySpan<float>)ReadOnlySpan<float>.Empty,
                            qBProj: qBElems > 0 ? new ReadOnlySpan<float>((void*)mlaW.QBProj, qBElems) : ReadOnlySpan<float>.Empty,
                            qProj: qMonoElems > 0 ? new ReadOnlySpan<float>((void*)mlaW.QProj, qMonoElems) : ReadOnlySpan<float>.Empty,
                            kvAProjWithMqa: new ReadOnlySpan<float>((void*)mlaW.KvAProjWithMqa, kvAElems * hiddenSize),
                            kvALayernormWeight: mlaW.KvALayernormWeight,
                            kvBProj: new ReadOnlySpan<float>((void*)mlaW.KvBProj, kvBElems * mlaW.KvLoraRank),
                            oProj: new ReadOnlySpan<float>((void*)lw.OWeight, oElems),
                            cachedLatent: _mlaLatentKvState.GetLatentPointer(layer),
                            cachedKPe: _mlaLatentKvState.GetKPePointer(layer),
                            cachedLength: _mlaLatentKvState.GetCurrentLength(layer),
                            attnScaleMultiplier: mlaScaleMultiplier);
                    }
                    else
                    {
                        MlaAttention.ExecuteLatent(
                            hidden: new ReadOnlySpan<float>(normOut, seqLen * hiddenSize),
                            output: new Span<float>(attnOut, seqLen * hiddenSize),
                            seqLen: seqLen,
                            positionOffset: positions[0],
                            hiddenSize: hiddenSize,
                            numHeads: mlaW.NumHeads,
                            qkNopeHeadDim: mlaW.QkNopeHeadDim,
                            qkRopeHeadDim: mlaW.QkRopeHeadDim,
                            vHeadDim: mlaW.VHeadDim,
                            qLoraRank: mlaW.QLoraRank,
                            kvLoraRank: mlaW.KvLoraRank,
                            rmsNormEps: eps,
                            ropeCosTable: _state.CosTable.AsSpan(0, ropeTableLen),
                            ropeSinTable: _state.SinTable.AsSpan(0, ropeTableLen),
                            qAProj: qAElems > 0 ? new ReadOnlySpan<float>((void*)mlaW.QAProj, qAElems) : ReadOnlySpan<float>.Empty,
                            qALayernormWeight: mlaW.QALayernormWeight ?? (ReadOnlySpan<float>)ReadOnlySpan<float>.Empty,
                            qBProj: qBElems > 0 ? new ReadOnlySpan<float>((void*)mlaW.QBProj, qBElems) : ReadOnlySpan<float>.Empty,
                            qProj: qMonoElems > 0 ? new ReadOnlySpan<float>((void*)mlaW.QProj, qMonoElems) : ReadOnlySpan<float>.Empty,
                            kvAProjWithMqa: new ReadOnlySpan<float>((void*)mlaW.KvAProjWithMqa, kvAElems * hiddenSize),
                            kvALayernormWeight: mlaW.KvALayernormWeight,
                            kvBProj: new ReadOnlySpan<float>((void*)mlaW.KvBProj, kvBElems * mlaW.KvLoraRank),
                            oProj: new ReadOnlySpan<float>((void*)lw.OWeight, oElems),
                            cachedLatent: _mlaLatentKvState.GetLatentPointer(layer),
                            cachedKPe: _mlaLatentKvState.GetKPePointer(layer),
                            cachedLength: _mlaLatentKvState.GetCurrentLength(layer),
                            attnScaleMultiplier: mlaScaleMultiplier);
                    }
                    _mlaLatentKvState.Advance(layer, seqLen);
                }
                else
                {
                    // Phase A — expanded cache + standard per-head attention.
                    MlaAttention.Execute(
                        hidden: new ReadOnlySpan<float>(normOut, seqLen * hiddenSize),
                        output: new Span<float>(attnOut, seqLen * hiddenSize),
                        seqLen: seqLen,
                        positionOffset: positions[0],
                        hiddenSize: hiddenSize,
                        numHeads: mlaW.NumHeads,
                        qkNopeHeadDim: mlaW.QkNopeHeadDim,
                        qkRopeHeadDim: mlaW.QkRopeHeadDim,
                        vHeadDim: mlaW.VHeadDim,
                        qLoraRank: mlaW.QLoraRank,
                        kvLoraRank: mlaW.KvLoraRank,
                        rmsNormEps: eps,
                        ropeCosTable: _state.CosTable.AsSpan(0, ropeTableLen),
                        ropeSinTable: _state.SinTable.AsSpan(0, ropeTableLen),
                        qAProj: qAElems > 0 ? new ReadOnlySpan<float>((void*)mlaW.QAProj, qAElems) : ReadOnlySpan<float>.Empty,
                        qALayernormWeight: mlaW.QALayernormWeight ?? (ReadOnlySpan<float>)ReadOnlySpan<float>.Empty,
                        qBProj: qBElems > 0 ? new ReadOnlySpan<float>((void*)mlaW.QBProj, qBElems) : ReadOnlySpan<float>.Empty,
                        qProj: qMonoElems > 0 ? new ReadOnlySpan<float>((void*)mlaW.QProj, qMonoElems) : ReadOnlySpan<float>.Empty,
                        kvAProjWithMqa: new ReadOnlySpan<float>((void*)mlaW.KvAProjWithMqa, kvAElems * hiddenSize),
                        kvALayernormWeight: mlaW.KvALayernormWeight,
                        kvBProj: new ReadOnlySpan<float>((void*)mlaW.KvBProj, kvBElems * mlaW.KvLoraRank),
                        oProj: new ReadOnlySpan<float>((void*)lw.OWeight, oElems),
                        attnScaleMultiplier: mlaScaleMultiplier,
                        cachedKNope: _mlaKvState!.GetKNopePointer(layer),
                        cachedV: _mlaKvState.GetVPointer(layer),
                        cachedKPe: _mlaKvState.GetKPePointer(layer),
                        cachedLength: _mlaKvState.GetCurrentLength(layer),
                        loraAdapter: _currentAdapter,
                        loraLayer: layer);
                    _mlaKvState.Advance(layer, seqLen);
                }

                // Bias on o_proj (rare — DeepSeek doesn't ship one by default).
                AddBias(lw.OBias, attnOut, hiddenSize, seqLen);

                // Residual add: attnOut + residual → hidden
                for (int t = 0; t < seqLen; t++)
                {
                    Add.Execute(
                        new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize),
                        new ReadOnlySpan<float>(attnOut + t * hiddenSize, hiddenSize),
                        new Span<float>(hidden + t * hiddenSize, hiddenSize));
                }

                // Prepare residual for FFN.
                new Span<float>(hidden, seqLen * hiddenSize).CopyTo(new Span<float>(residual, seqLen * hiddenSize));
            }
            else
            {
                // ── GQA branch ──────────────────────────────────────────────
                // b. RMSNorm + Pre-quantize + Q/K/V projections
                //
                // When a LoRA adapter is active we need the F32 normalised
                // hidden state (`normOut`) to feed LoraDelta — the fused
                // RmsNormQuantize decode path skips that intermediate. Force
                // the unfused path in that case.
            bool adapterActive = _currentAdapter is not null;
            // Phase 4d.5 / Gap 2: hoist preQuantNorm out of the decode/prefill
            // sub-branches so the LoRA delta call site (Q8_0-B fast path) can
            // re-use the buffer for stage 1. Pre-LoRA-Q8_0 this was scoped
            // inside each sub-branch.
            byte* preQuantNormQkv = null;
            // The fused decode kernels don't support I2_S; route ternary weights through the
            // standard (unfused) projection path, which dispatches to the I2_S GEMV.
            if (seqLen == 1 && _threadPool != null && !adapterActive && lw.QQuantType != QuantizationType.I2_S)
            {
                // Decode path: try fused RmsNorm+Quantize (skips normOut intermediate)
                byte* preQuantNorm = null;
                if (IsCompatiblePreQuant(lw.QQuantType, lw.KQuantType)
                    && IsCompatiblePreQuant(lw.QQuantType, lw.VQuantType))
                {
                    preQuantNorm = FusedOps.RmsNormQuantize(hidden, lw.AttnNormWeight, eps,
                        inputQ8Scratch, hiddenSize, lw.QQuantType);
                }

                if (preQuantNorm == null)
                {
                    // Fallback: unfused (F32/F16 weights or cross-family projections)
                    RmsNorm.Execute(
                        new ReadOnlySpan<float>(hidden, hiddenSize),
                        lw.AttnNormWeight, eps,
                        new Span<float>(normOut, hiddenSize));
                    preQuantNorm = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, 1, lw.QQuantType);
                }

                FusedQkvDecode(in lw, normOut, preQuantNorm, q, k, v);
                preQuantNormQkv = preQuantNorm;
            }
            else
            {
                // Prefill path: unfused RmsNorm + Quantize + individual projections
                for (int t = 0; t < seqLen; t++)
                {
                    RmsNorm.Execute(
                        new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                        lw.AttnNormWeight, eps,
                        new Span<float>(normOut + t * hiddenSize, hiddenSize));
                }

                byte* preQuantNorm = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, seqLen, lw.QQuantType);

                var rwQ = rl?.Q ?? default;
                var rwK = rl?.K ?? default;
                var rwV = rl?.V ?? default;
                GemmInterleaved(lw.QWeight, lw.QQuantType, normOut, q, lw.QOutputDim, lw.QInputDim, seqLen,
                    preQuantNorm, in rwQ);
                GemmInterleaved(lw.KWeight, lw.KQuantType, normOut, k, lw.KOutputDim, lw.KInputDim, seqLen,
                    IsCompatiblePreQuant(lw.QQuantType, lw.KQuantType) ? preQuantNorm : null, in rwK);
                GemmInterleaved(lw.VWeight, lw.VQuantType, normOut, v, lw.VOutputDim, lw.VInputDim, seqLen,
                    IsCompatiblePreQuant(lw.QQuantType, lw.VQuantType) ? preQuantNorm : null, in rwV);
                preQuantNormQkv = preQuantNorm;
            }

            // Optional bias: y = Wx + b (no-op when null)
            AddBias(lw.QBias, q, lw.QOutputDim, seqLen);
            AddBias(lw.KBias, k, lw.KOutputDim, seqLen);
            AddBias(lw.VBias, v, lw.VOutputDim, seqLen);

            // LoRA delta (q/k/v): y += scale * (normOut · B) · A. No-op when
            // no adapter is active. Applied AFTER bias and BEFORE QK-norm /
            // RoPE so the delta contributes to the same downstream pipeline
            // as the base projection. F32 normOut is guaranteed materialised
            // here (we forced the unfused path above when adapter is active).
            //
            // Phase 4d.5 / Gap 2: when the base projection is Q8_0 the
            // `preQuantNormQkv` buffer is the Q8_0-encoded F32 input. We hand
            // that to ApplyLoraDelta so a Q8_0-B adapter's stage 1 can re-use
            // the buffer via `GemmQ8_0(preQuantizedInput=preQuantNormQkv)`,
            // skipping the activation quantise step that Phase 4d.4 had to
            // pay per-projection. Re-quantised path (`QuantizeInput` returning
            // null) drops through to the F32 / dequant-once fallback as before.
            if (_currentAdapter is not null)
            {
                // preQuantNormQkv is only valid for k/v when K/V quant types
                // are compatible with Q (same IsCompatiblePreQuant check the
                // base GEMM uses for the shared-input optimisation).
                byte* preQ_q = preQuantNormQkv;
                byte* preQ_k = (preQ_q is not null && IsCompatiblePreQuant(lw.QQuantType, lw.KQuantType)) ? preQ_q : null;
                byte* preQ_v = (preQ_q is not null && IsCompatiblePreQuant(lw.QQuantType, lw.VQuantType)) ? preQ_q : null;
                ApplyLoraDelta(layer, "q_proj", normOut, q, seqLen, lw.QInputDim, lw.QOutputDim,
                               preQ_q, lw.QQuantType);
                ApplyLoraDelta(layer, "k_proj", normOut, k, seqLen, lw.KInputDim, lw.KOutputDim,
                               preQ_k, lw.KQuantType);
                ApplyLoraDelta(layer, "v_proj", normOut, v, seqLen, lw.VInputDim, lw.VOutputDim,
                               preQ_v, lw.VQuantType);
            }

            // Optional QK-norms (Qwen3-style): per-head RMSNorm on Q/K after projection, before RoPE
            if (lw.QNormWeight is not null)
                ApplyPerHeadNorm(lw.QNormWeight, q, numHeads, headDimLayer, seqLen, eps);
            if (lw.KNormWeight is not null)
                ApplyPerHeadNorm(lw.KNormWeight, k, numKvHeadsLayer, headDimLayer, seqLen, eps);

            // d. RoPE (in-place on Q and K for all tokens). SmolLM3 marks
            // selected layers as NoPE (skip RoPE entirely) via
            // ModelConfig.NoRopeLayers — the attention math runs unmodified on
            // position-free Q/K, which is the whole point of NoPE. Gemma 4 picks
            // a per-attention-type table/dim/pairing via GetLayerRope (full vs
            // sliding); every other architecture resolves to the single table.
            if (!Config.IsNoRopeLayer(layer))
            {
                var (ropeCos, ropeSin, ropeDimLayer, ropeTypeLayer) = GetLayerRope(layer);
                RoPE.Execute(
                    new Span<float>(q, seqLen * qStrideLayer),
                    new Span<float>(k, seqLen * kvStrideLayer),
                    positions,
                    numHeads, numKvHeadsLayer, headDimLayer, ropeDimLayer,
                    ropeCos, ropeSin, ropeTypeLayer);
            }

            // e. Attention — with or without KV-cache
            // Gemma 3 family extras (no-op on every other architecture):
            //  - PerLayerSlidingWindow[layer]: per-layer sliding-window override
            //    (Gemma 3 interleaves local/global attention).
            //  - QueryPreAttnScalar: override the default 1/sqrt(headDim) scale.
            //  - AttnLogitSoftcap: pre-softmax tanh soft-cap (Gemma 2 sets 50.0;
            //    Gemma 3 leaves null but the plumbing is wired).
            int? layerSlidingWindow = GetLayerSlidingWindow(layer);
            float attnScale = Config.QueryPreAttnScalar is float qpas && qpas > 0
                ? 1.0f / MathF.Sqrt(qpas)
                : 1.0f / MathF.Sqrt(headDimLayer);
            float attnSoftCap = Config.AttnLogitSoftcap ?? 0f;

            if (kvCache is not null)
            {
                // KV-cache + distinct per-layer head dim is rejected at entry
                // (GuardKvCacheHeadDim), so headDimLayer == headDim here and the
                // single-stride cache geometry is consistent.
                // Store new K/V in cache, then attend over full cached context (zero allocations)
                var kRef = new TensorRef(seqLen, kvStrideLayer, DType.Float32, -1, (nint)k);
                var vRef = new TensorRef(seqLen, kvStrideLayer, DType.Float32, -1, (nint)v);

                kvCache.Update(kRef, vRef, positions, layer);

                int seqKv = kvCache.CurrentLength;

                if (kvCache is IQuantizedKvCache qkvCache)
                {
                    // Quantized path: dequantize KV tiles on-the-fly during attention
                    Attention.Execute(q, qkvCache, layer, attnOut,
                        seqLen, seqKv, numHeads, numKvHeadsLayer, headDimLayer, positions[0], _threadPool,
                        layerSlidingWindow, attnSoftCap);
                }
                else
                {
                    var cachedK = kvCache.GetKeysRef(layer);
                    var cachedV = kvCache.GetValuesRef(layer);

                    Attention.Execute(q, (float*)cachedK.DataPointer, (float*)cachedV.DataPointer, attnOut,
                        seqLen, seqKv, numHeads, numKvHeadsLayer, headDimLayer, positions[0], attnScale,
                        _threadPool, layerSlidingWindow, attnSoftCap);
                }
            }
            else
            {
                // Cacheless attention. _currentMaskSpec is Causal for every existing caller
                // (the field defaults to Causal and only the mask-aware Forward overload sets
                // it otherwise), so this call is byte-identical to the pre-bidirectional code
                // on the causal path. Bidirectional / hybrid divert masking inside the kernel.
                Attention.Execute(q, k, v, attnOut,
                    seqLen, seqLen, numHeads, numKvHeadsLayer, headDimLayer, 0, attnScale, _threadPool,
                    layerSlidingWindow, attnSoftCap,
                    _currentMaskSpec.Mode, _currentMaskSpec.PrefixLength);
            }

            // Optional attention sub-norm (BitNet Sub-LN): RMSNorm over the attention output
            // before the output projection. In-place per token. No-op for non-BitNet models.
            if (lw.AttnSubNormWeight is not null)
            {
                for (int t = 0; t < seqLen; t++)
                {
                    var s = new Span<float>(attnOut + t * qStrideLayer, qStrideLayer);
                    RmsNorm.Execute(s, lw.AttnSubNormWeight, eps, s);
                }
            }

            // f. Batched O projection. The O projection input width is the
            // attention output stride (numHeads * headDimLayer == lw.OInputDim).
            byte* preQuantAttn = QuantizeInput(attnOut, inputQ8Scratch, qStrideLayer, seqLen, lw.OQuantType);
            var rwO = rl?.O ?? default;
            GemmInterleaved(lw.OWeight, lw.OQuantType, attnOut, normOut, lw.OOutputDim, lw.OInputDim, seqLen,
                preQuantAttn, in rwO);
            AddBias(lw.OBias, normOut, lw.OOutputDim, seqLen);

            // LoRA delta (o_proj): y += scale * (attnOut · B) · A.
            // Phase 4d.5 / Gap 2: pass preQuantAttn so Q8_0-B adapter stage 1
            // re-uses the activation Q8_0 buffer.
            if (_currentAdapter is not null)
            {
                ApplyLoraDelta(layer, "o_proj", attnOut, normOut, seqLen, lw.OInputDim, lw.OOutputDim,
                               preQuantAttn, lw.OQuantType);
            }

            // g0. Gemma post-attention RMSNorm — applied to the attention
            // sublayer output (normOut) BEFORE the residual add (four-norm
            // layout). No-op for non-Gemma (PostAttnNormWeight is null).
            if (lw.PostAttnNormWeight is float[] postAttnNorm)
            {
                for (int t = 0; t < seqLen; t++)
                    RmsNorm.Execute(
                        new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                        postAttnNorm, eps,
                        new Span<float>(normOut + t * hiddenSize, hiddenSize));
            }

            // g. Residual add (per token)
            for (int t = 0; t < seqLen; t++)
            {
                Add.Execute(
                    new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize),
                    new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                    new Span<float>(hidden + t * hiddenSize, hiddenSize));
            }

            // h. Copy hiddenState → residual
            new Span<float>(hidden, seqLen * hiddenSize).CopyTo(new Span<float>(residual, seqLen * hiddenSize));
            }

            // ── MoE branch ──────────────────────────────────────────────
            // Mixtral-convention top-k dense routing replaces the dense FFN
            // block entirely. Takes post-attn hidden + FFN RMSNorm weight,
            // runs router + top-k experts, writes into normOut, then residual
            // adds into hidden and continues to the next layer. No R4 repack
            // (expert GEMMs are tiny), no pre-quantise (experts are F32).
            if (lw.Moe is not null)
            {
                // FFN RMSNorm per token into normOut.
                for (int t = 0; t < seqLen; t++)
                {
                    RmsNorm.Execute(
                        new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                        lw.FfnNormWeight, eps,
                        new Span<float>(normOut + t * hiddenSize, hiddenSize));
                }

                MoeLayerWeights moe = lw.Moe!;
                // BitNet-ternary MoE (identity-MoTE): experts are I2_S with a relu² +
                // per-expert ffn_sub_norm body, dispatched through the indexed I2_S kernel.
                // The router (Gate + optional GateBias) stays F32.
                // CROSS-BACKEND TODO (CPU-first landed here): mirror this in CudaMoeFfn
                // (add an I2_S variant of native moe_grouped_gemv.cu — the quant path already
                // uploads per-expert quant bytes) and VulkanTransformerModel (new
                // moe_indexed_matmul_i2s_f32.comp; the q4_k/q8_0 indexed shaders are templates),
                // plus the GateBias add in each backend's router. See
                // .planning/2026-07-08-mote-dotllm-export-design.md §3.4.
                if (moe.IsBitNetI2S)
                {
                    MoeSwiGluMlp.ExecuteBitNetMoe(
                        hidden: new ReadOnlySpan<float>(normOut, seqLen * hiddenSize),
                        gateWeights: moe.Gate,
                        gateBias: moe.GateBias is not null ? moe.GateBias.AsSpan() : ReadOnlySpan<float>.Empty,
                        gateBank: (byte*)moe.GateExpsI2SBase, gateRowBytes: moe.GateExpsI2SRowBytes, gateScales: moe.GateExpsI2SScales!,
                        upBank: (byte*)moe.UpExpsI2SBase, upRowBytes: moe.UpExpsI2SRowBytes, upScales: moe.UpExpsI2SScales!,
                        downBank: (byte*)moe.DownExpsI2SBase, downRowBytes: moe.DownExpsI2SRowBytes, downScales: moe.DownExpsI2SScales!,
                        expertFfnSubNorm: moe.ExpertFfnSubNorm!,
                        output: new Span<float>(normOut, seqLen * hiddenSize),
                        numExperts: moe.NumExperts,
                        numExpertsPerTok: moe.NumExpertsPerTok,
                        hiddenSize: hiddenSize,
                        intermediateSize: moe.IntermediateSize,
                        seqLen: seqLen,
                        normTopKProb: moe.NormTopKProb,
                        rmsEps: eps,
                        threadPool: _threadPool);
                }
                // Route through the shared-expert-aware overload iff we need
                // shared-expert addition OR the raw-softmax (non-renormalised)
                // Qwen1.5-MoE gating. The simple Mixtral path stays the call
                // target for the common case.
                else if (moe.HasSharedExpert || !moe.NormTopKProb)
                {
                    ReadOnlySpan<float> sharedGateSpan = moe.SharedExpertGate is not null
                        ? moe.SharedExpertGate.AsSpan()
                        : ReadOnlySpan<float>.Empty;
                    MoeSwiGluMlp.ExecuteWithSharedExpert(
                        hidden: new ReadOnlySpan<float>(normOut, seqLen * hiddenSize),
                        gateWeights: moe.Gate,
                        expertsW1: moe.W1,
                        expertsW2: moe.W2,
                        expertsW3: moe.W3,
                        output: new Span<float>(normOut, seqLen * hiddenSize),
                        numExperts: moe.NumExperts,
                        numExpertsPerTok: moe.NumExpertsPerTok,
                        hiddenSize: hiddenSize,
                        intermediateSize: moe.IntermediateSize,
                        seqLen: seqLen,
                        normTopKProb: moe.NormTopKProb,
                        sharedGateProj: moe.SharedGateProj,
                        sharedUpProj: moe.SharedUpProj,
                        sharedDownProj: moe.SharedDownProj,
                        sharedIntermediateSize: moe.SharedIntermediateSize,
                        sharedExpertGate: sharedGateSpan,
                        loraAdapter: _currentAdapter,
                        loraLayer: layer,
                        useGeGLU: _useGeGLU);
                }
                else
                {
                    MoeSwiGluMlp.Execute(
                        hidden: new ReadOnlySpan<float>(normOut, seqLen * hiddenSize),
                        gateWeights: moe.Gate,
                        expertsW1: moe.W1,
                        expertsW2: moe.W2,
                        expertsW3: moe.W3,
                        output: new Span<float>(normOut, seqLen * hiddenSize),
                        numExperts: moe.NumExperts,
                        numExpertsPerTok: moe.NumExpertsPerTok,
                        hiddenSize: hiddenSize,
                        intermediateSize: moe.IntermediateSize,
                        seqLen: seqLen,
                        loraAdapter: _currentAdapter,
                        loraLayer: layer,
                        useGeGLU: _useGeGLU);
                }

                // Residual add (per token) → hidden. Same as dense path.
                for (int t = 0; t < seqLen; t++)
                {
                    Add.Execute(
                        new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize),
                        new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                        new Span<float>(hidden + t * hiddenSize, hiddenSize));
                }
                continue;
            }

            // i. FFN RMSNorm + Pre-quantize + Gate/Up projections
            // When a LoRA adapter is active we need F32 normOut for delta —
            // skip the fused decode path so it materialises (same trick as Q/K/V).
            bool ffnAdapterActive = _currentAdapter is not null;
            // Phase 4d.5 / Gap 2: hoist preQuantFfn out of both sub-branches
            // so the LoRA delta call site can reuse the activation Q8_0
            // buffer for stage 1.
            byte* preQuantFfnHoisted = null;
            // I2_S (BitNet) is unsupported by the fused decode kernels — use the unfused path.
            if (seqLen == 1 && _threadPool != null && !ffnAdapterActive && lw.GateQuantType != QuantizationType.I2_S)
            {
                // Decode path: try fused RmsNorm+Quantize (skips normOut intermediate)
                byte* preQuantFfn = null;
                if (IsCompatiblePreQuant(lw.GateQuantType, lw.UpQuantType))
                {
                    preQuantFfn = FusedOps.RmsNormQuantize(hidden, lw.FfnNormWeight, eps,
                        inputQ8Scratch, hiddenSize, lw.GateQuantType);
                }

                if (preQuantFfn == null)
                {
                    // Fallback: unfused (F32/F16 weights or cross-family projections)
                    RmsNorm.Execute(
                        new ReadOnlySpan<float>(hidden, hiddenSize),
                        lw.FfnNormWeight, eps,
                        new Span<float>(normOut, hiddenSize));
                    preQuantFfn = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, 1, lw.GateQuantType);
                }

                FusedGateUpDecode(in lw, normOut, preQuantFfn, ffnGate, ffnUp);
                preQuantFfnHoisted = preQuantFfn;
            }
            else
            {
                // Prefill path: unfused RmsNorm + Quantize + individual projections
                for (int t = 0; t < seqLen; t++)
                {
                    RmsNorm.Execute(
                        new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                        lw.FfnNormWeight, eps,
                        new Span<float>(normOut + t * hiddenSize, hiddenSize));
                }

                byte* preQuantFfn = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, seqLen, lw.GateQuantType);

                var rwGate = rl?.Gate ?? default;
                var rwUp = rl?.Up ?? default;
                GemmInterleaved(lw.GateWeight, lw.GateQuantType, normOut, ffnGate, lw.GateOutputDim, lw.GateInputDim, seqLen,
                    preQuantFfn, in rwGate);
                GemmInterleaved(lw.UpWeight, lw.UpQuantType, normOut, ffnUp, lw.UpOutputDim, lw.UpInputDim, seqLen,
                    IsCompatiblePreQuant(lw.GateQuantType, lw.UpQuantType) ? preQuantFfn : null, in rwUp);
                preQuantFfnHoisted = preQuantFfn;
            }
            AddBias(lw.GateBias, ffnGate, lw.GateOutputDim, seqLen);
            AddBias(lw.UpBias, ffnUp, lw.UpOutputDim, seqLen);

            // LoRA delta (gate/up): y += scale * (normOut · B) · A.
            // Phase 4d.5 / Gap 2: pass the hoisted preQuantFfn so the Q8_0-B
            // adapter stage 1 re-uses the activation Q8_0 buffer.
            if (_currentAdapter is not null)
            {
                byte* preQ_gate = preQuantFfnHoisted;
                byte* preQ_up = (preQ_gate is not null && IsCompatiblePreQuant(lw.GateQuantType, lw.UpQuantType)) ? preQ_gate : null;
                ApplyLoraDelta(layer, "gate_proj", normOut, ffnGate, seqLen, lw.GateInputDim, lw.GateOutputDim,
                               preQ_gate, lw.GateQuantType);
                ApplyLoraDelta(layer, "up_proj", normOut, ffnUp, seqLen, lw.UpInputDim, lw.UpOutputDim,
                               preQ_up, lw.UpQuantType);
            }

            // Fused gate activation: GeGLU (tanh-approx GELU) for Gemma (GELUTanh),
            // ReLU² for BitNet (ReluSquared), otherwise SwiGLU (SiLU). Single tiled
            // pass per token; all kernels are shape-identical (down(act(gate) * up)).
            bool useReluSquared = Config.ActivationFunction == ActivationFunction.ReluSquared;
            for (int t = 0; t < seqLen; t++)
            {
                float* gateT = ffnGate + t * intermediateSize;
                float* upT = ffnUp + t * intermediateSize;
                float* siluT = siluOut + t * intermediateSize;

                var gateSpan = new ReadOnlySpan<float>(gateT, intermediateSize);
                var upSpan = new ReadOnlySpan<float>(upT, intermediateSize);
                var outSpan = new Span<float>(siluT, intermediateSize);
                if (_useGeGLU)
                    FusedOps.GeGLUTanh(gateSpan, upSpan, outSpan);
                else if (useReluSquared)
                    FusedOps.ReLU2GLU(gateSpan, upSpan, outSpan);
                else
                    FusedOps.SwiGLU(gateSpan, upSpan, outSpan);
            }

            // Optional FFN sub-norm (BitNet Sub-LN): RMSNorm over the gated intermediate
            // before the down projection. In-place per token. No-op for non-BitNet models.
            if (lw.FfnSubNormWeight is not null)
            {
                for (int t = 0; t < seqLen; t++)
                {
                    var s = new Span<float>(siluOut + t * intermediateSize, intermediateSize);
                    RmsNorm.Execute(s, lw.FfnSubNormWeight, eps, s);
                }
            }

            // Pre-quantize siluOutput for Down projection (different input dim = intermediateSize)
            byte* preQuantSilu = QuantizeInput(siluOut, inputQ8Scratch, intermediateSize, seqLen, lw.DownQuantType);

            // Batched Down projection (output into normOut as scratch)
            var rwDown = rl?.Down ?? default;
            GemmInterleaved(lw.DownWeight, lw.DownQuantType, siluOut, normOut, lw.DownOutputDim, lw.DownInputDim, seqLen,
                preQuantSilu, in rwDown);
            AddBias(lw.DownBias, normOut, lw.DownOutputDim, seqLen);

            // LoRA delta (down_proj): y += scale * (siluOut · B) · A.
            // Input is post-SwiGLU (siluOut), not normOut. The base GEMM
            // already wrote into normOut, so we accumulate delta in place.
            // Phase 4d.5 / Gap 2: pass preQuantSilu so Q8_0-B adapter stage 1
            // re-uses the activation Q8_0 buffer.
            if (_currentAdapter is not null)
            {
                ApplyLoraDelta(layer, "down_proj", siluOut, normOut, seqLen, lw.DownInputDim, lw.DownOutputDim,
                               preQuantSilu, lw.DownQuantType);
            }

            // j0. Gemma post-FFN RMSNorm — applied to the FFN sublayer output
            // (normOut) BEFORE the residual add (four-norm layout). No-op for
            // non-Gemma (PostFfnNormWeight is null).
            if (lw.PostFfnNormWeight is float[] postFfnNorm)
            {
                for (int t = 0; t < seqLen; t++)
                    RmsNorm.Execute(
                        new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                        postFfnNorm, eps,
                        new Span<float>(normOut + t * hiddenSize, hiddenSize));
            }

            // k. Residual add (per token)
            for (int t = 0; t < seqLen; t++)
            {
                Add.Execute(
                    new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize),
                    new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                    new Span<float>(hidden + t * hiddenSize, hiddenSize));
            }

            // l. Per-Layer Embeddings (PLE) injection — Gemma-4 dense text tower.
            // Gated residual added to the layer output using this layer's slice of the
            // precomputed per-layer input tensor. No-op for every other architecture
            // (pleInputs null). See docs verified against HF Gemma4TextDecoderLayer.
            if (pleInputs is not null && lw.PleGateWeight != 0)
            {
                PerLayerEmbeddings.InjectLayer(
                    hidden: hidden,
                    perLayerInputs: pleInputs,
                    layerIdx: layer, numLayers: Config.NumLayers,
                    gateWeight: (float*)lw.PleGateWeight,
                    projWeight: (float*)lw.PleProjWeight,
                    postNormWeight: lw.PlePostNormWeight,
                    gateScratch: pleGateScratch,
                    projScratch: pleProjScratch,
                    seqLen: seqLen, hiddenSize: hiddenSize, pleDim: pleDim, eps: eps);
            }
        }

        if (pleInputs is not null)
        {
            NativeMemory.AlignedFree(pleInputs);
            NativeMemory.AlignedFree(pleIdentity);
            NativeMemory.AlignedFree(pleGateScratch);
            NativeMemory.AlignedFree(pleProjScratch);
        }

        // 3. FINAL NORM (in-place: hidden → hidden)
        for (int t = 0; t < seqLen; t++)
        {
            float* hiddenT = hidden + t * hiddenSize;
            // Use normOut as temp so we can copy back
            float* normOutT = normOut + t * hiddenSize;

            RmsNorm.Execute(
                new ReadOnlySpan<float>(hiddenT, hiddenSize),
                _weights.OutputNormWeight,
                eps,
                new Span<float>(normOutT, hiddenSize));

            new Span<float>(normOutT, hiddenSize).CopyTo(new Span<float>(hiddenT, hiddenSize));
        }
    }

    /// <summary>
    /// Runs one Gemma-4 MoE transformer layer (cacheless, causal). Implements the
    /// source-confirmed gemma4 graph (<c>docs/diffusiongemma/GEMMA4-GRAPH-SPEC.md</c>):
    /// QK-normed attention with V-from-K on V-less layers + a weight-less V-norm and
    /// softmax scale 1.0, then a dual <i>parallel</i> FFN (a dense GeGLU MLP summed
    /// with a 128-expert MoE driven by a custom router), wrapped by the five FFN
    /// norms, residual-added, and finally multiplied by <c>layer_output_scale</c>.
    /// </summary>
    /// <remarks>
    /// On entry <paramref name="hidden"/> = <paramref name="residual"/> = the layer
    /// input (the loop copied hidden→residual before this call). On exit
    /// <paramref name="hidden"/> holds the layer output. Scratch buffers
    /// <paramref name="normOut"/>/<paramref name="q"/>/<paramref name="k"/>/
    /// <paramref name="v"/>/<paramref name="attnOut"/> are reused; hidden-sized
    /// branch temporaries are rented from <see cref="ArrayPool{T}"/> (seqLen is
    /// small on the single-forward path).
    /// </remarks>
    private unsafe void RunGemma4Layer(
        in TransformerLayerWeights lw, int layer, int seqLen,
        float* hidden, float* residual, float* normOut,
        float* q, float* k, float* v, float* attnOut,
        int numKvHeadsLayer, int headDimLayer, int qStrideLayer, int kvStrideLayer,
        ReadOnlySpan<int> positions, float eps, IKvCache? kvCache = null)
    {
        int hiddenSize = Config.HiddenSize;
        int numHeads = Config.NumAttentionHeads;
        var g4 = lw.Gemma4!;

        // DIFFUSION region per-layer scalar split. On diffusion-gemma the LAST
        // per-layer op uses enc_layer_output_scale for the PROMPT rows [0, P) and
        // layer_output_scale for the CANVAS rows [P, seqLen) — same backbone
        // weights, the encoder contributes ONLY the scalar (diffusion-gemma.cpp:
        // 475-489). P is the Hybrid region split (prompt length). On autoregressive
        // gemma4 (DiffusionConfig null) regionP == 0 so every row uses
        // LayerOutputScale, byte-identical to the validated AR path.
        bool diffusion = Config.DiffusionConfig is not null;
        int regionP = 0;
        if (diffusion && _currentMaskSpec.Mode == AttentionMaskMode.Hybrid)
        {
            regionP = _currentMaskSpec.PrefixLength;
            if (regionP < 0) regionP = 0;
            if (regionP > seqLen) regionP = seqLen;
        }
        float encScale = diffusion && g4.EncLayerOutputScale is float e
            ? e
            : g4.LayerOutputScale;

        // ── Attention ─────────────────────────────────────────────────────
        // normIn = rms(hidden) * attn_norm  (into normOut)
        for (int t = 0; t < seqLen; t++)
            RmsNorm.Execute(
                new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                lw.AttnNormWeight, eps,
                new Span<float>(normOut + t * hiddenSize, hiddenSize));

        // Q = wq · normIn ; K = wk · normIn (raw projections, GEMM quantizes internally)
        Gemm(lw.QWeight, lw.QQuantType, normOut, q, lw.QOutputDim, lw.QInputDim, seqLen);
        Gemm(lw.KWeight, lw.KQuantType, normOut, k, lw.KOutputDim, lw.KInputDim, seqLen);

        // V branch: off the RAW K projection when wv absent (global layers),
        // else V = wv · normIn. K is captured BEFORE k-norm/rope, so when
        // VFromK we copy k → v now (k still holds the raw projection).
        if (g4.VFromK)
            new Span<float>(k, seqLen * kvStrideLayer).CopyTo(new Span<float>(v, seqLen * kvStrideLayer));
        else
            Gemm(lw.VWeight, lw.VQuantType, normOut, v, lw.VOutputDim, lw.VInputDim, seqLen);

        // LoRA delta (q/k/v): y += scale * (normIn · B) · A. No-op when no adapter is
        // active. Applied AFTER the base projection and BEFORE QK-norm/RoPE, same
        // ordering as the generic Llama-style layer path. No v_proj delta on VFromK
        // layers — the base model has no v_proj weight there for an adapter to target.
        if (_currentAdapter is not null)
        {
            ApplyLoraDelta(layer, "q_proj", normOut, q, seqLen, lw.QInputDim, lw.QOutputDim);
            ApplyLoraDelta(layer, "k_proj", normOut, k, seqLen, lw.KInputDim, lw.KOutputDim);
            if (!g4.VFromK)
                ApplyLoraDelta(layer, "v_proj", normOut, v, seqLen, lw.VInputDim, lw.VOutputDim);
        }

        // Q-norm (rms * attn_q_norm per head), then K-norm (rms * attn_k_norm per
        // kv head). V-norm is WEIGHT-LESS rms per kv head (no scale), all layers.
        if (lw.QNormWeight is not null)
            ApplyPerHeadNorm(lw.QNormWeight, q, numHeads, headDimLayer, seqLen, eps);
        if (lw.KNormWeight is not null)
            ApplyPerHeadNorm(lw.KNormWeight, k, numKvHeadsLayer, headDimLayer, seqLen, eps);
        ApplyPerHeadNormWeightless(v, numKvHeadsLayer, headDimLayer, seqLen, eps);

        // RoPE on Q and K (V is NOT roped). Per-attention-type table/dim/pairing.
        // Global (full-attention) layers use PARTIAL NeoX rope: only the leading
        // GlobalRopeDim dims rotate, but the rotate_half pairing offset is the FULL
        // head's half-dim (dims [0,64) ↔ [256,320)). Sliding layers use full rope.
        var (ropeCos, ropeSin, ropeDimLayer, ropeTypeLayer) = GetLayerRope(layer);
        var qSpan = new Span<float>(q, seqLen * qStrideLayer);
        var kSpan = new Span<float>(k, seqLen * kvStrideLayer);
        bool partialGlobal = Config.IsFullAttentionLayer(layer)
            && Config.PartialRotaryFactor is float prf && prf > 0f && prf < 1f
            && ropeDimLayer < headDimLayer;
        if (partialGlobal)
            RoPE.ExecutePartialNeoX(
                qSpan, kSpan, positions,
                numHeads, numKvHeadsLayer, headDimLayer, ropeDimLayer / 2,
                ropeCos, ropeSin);
        else
            RoPE.Execute(
                qSpan, kSpan, positions,
                numHeads, numKvHeadsLayer, headDimLayer, ropeDimLayer,
                ropeCos, ropeSin, ropeTypeLayer);

        // Attention: softmax(Qᵀ·K * 1.0 + causal mask) · V, GQA broadcast. Scale is
        // 1.0 (q_norm/k_norm make Q,K unit) — QueryPreAttnScalar=1.0 → 1/sqrt(1)=1.
        float attnScale = Config.QueryPreAttnScalar is float qpas && qpas > 0
            ? 1.0f / MathF.Sqrt(qpas)
            : 1.0f / MathF.Sqrt(headDimLayer);
        int? layerSlidingWindow = GetLayerSlidingWindow(layer);

        // ── PKV phase split ─────────────────────────────────────────────────
        // Prefill: capture this layer's post-rope K (and post-vnorm V) for reuse
        //   across denoise steps, then run the normal causal prompt attention.
        // Decode: attend the C canvas queries over [cached prompt K/V | fresh canvas
        //   K/V] (length P+C) under a RECTANGULAR bidirectional mask — a canvas query
        //   (RoPE position P+i) attends every prompt key + every canvas key, clipped by
        //   the per-layer sliding window. positionOffset = P maps the canvas query to
        //   logical position P+i so the sliding-window lower bound matches the unified
        //   Hybrid path exactly.
        // None (default): the verbatim unified cacheless attention — byte-identical.
        if (_pkvPhase == DiffusionKvPhase.Prefill)
        {
            // K/V are row-major [seqLen × kvStrideLayer]; store them whole (seqLen == P).
            var store = _pkvStore!;
            int kvElems = seqLen * kvStrideLayer;
            new Span<float>(k, kvElems).CopyTo(new Span<float>(store.Keys(layer), kvElems));
            new Span<float>(v, kvElems).CopyTo(new Span<float>(store.Values(layer), kvElems));

            Attention.Execute(q, k, v, attnOut,
                seqLen, seqLen, numHeads, numKvHeadsLayer, headDimLayer, 0, attnScale, _threadPool,
                layerSlidingWindow, softCap: 0f,
                _currentMaskSpec.Mode, _currentMaskSpec.PrefixLength);
        }
        else if (_pkvPhase == DiffusionKvPhase.Decode)
        {
            int p = _pkvPromptLen;            // cached prompt length
            int c = seqLen;                   // canvas length (this forward's row count)
            int kvCtx = p + c;                // concat key/value rows
            var store = _pkvStore!;
            long concatElems = (long)kvCtx * kvStrideLayer;
            float[] kCat = ArrayPool<float>.Shared.Rent((int)concatElems);
            float[] vCat = ArrayPool<float>.Shared.Rent((int)concatElems);
            try
            {
                fixed (float* kCatP = kCat)
                fixed (float* vCatP = vCat)
                {
                    // [0, P) = cached prompt K/V (post-norm/post-rope, V-from-K on global
                    // layers exactly as captured); [P, P+C) = fresh canvas K/V.
                    int promptKvElems = p * kvStrideLayer;
                    int canvasKvElems = c * kvStrideLayer;
                    new Span<float>(store.Keys(layer), promptKvElems).CopyTo(new Span<float>(kCatP, promptKvElems));
                    new Span<float>(store.Values(layer), promptKvElems).CopyTo(new Span<float>(vCatP, promptKvElems));
                    new Span<float>(k, canvasKvElems).CopyTo(new Span<float>(kCatP + promptKvElems, canvasKvElems));
                    new Span<float>(v, canvasKvElems).CopyTo(new Span<float>(vCatP + promptKvElems, canvasKvElems));

                    // seqQ = C canvas queries, seqKv = P+C keys, positionOffset = P (canvas
                    // query i has logical position P+i). Bidirectional: every canvas query
                    // attends all P+C keys, with the sliding-window lower bound applied.
                    Attention.Execute(q, kCatP, vCatP, attnOut,
                        c, kvCtx, numHeads, numKvHeadsLayer, headDimLayer, p, attnScale, _threadPool,
                        layerSlidingWindow, softCap: 0f,
                        AttentionMaskMode.Bidirectional, 0);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(kCat);
                ArrayPool<float>.Shared.Return(vCat);
            }
        }
        else if (kvCache is not null)
        {
            // Autoregressive Gemma-4 decode. Store this layer's post-rope K and
            // post-vnorm V (exactly what cacheless attention consumes) into the
            // per-layer-strided cache at `positions`, then attend over the full
            // cached context [0, CurrentLength). The cache row width for this layer
            // is kvStrideLayer (= numKvHeadsLayer * headDimLayer), which matches the
            // KvGeometry.FromConfig stride validated up front by GuardKvCacheHeadDim
            // — so the sliding (e.g. 2×16) and global (2×32) layers each address
            // their own buffer correctly. Causal mask, positionOffset = positions[0].
            var kRef = new TensorRef(seqLen, kvStrideLayer, DType.Float32, -1, (nint)k);
            var vRef = new TensorRef(seqLen, kvStrideLayer, DType.Float32, -1, (nint)v);
            kvCache.Update(kRef, vRef, positions, layer);

            int seqKv = kvCache.CurrentLength;
            if (kvCache is IQuantizedKvCache qkvCache)
            {
                // Quantized KV: dequantize tiles on-the-fly during attention. (A
                // per-layer-strided quantized cache for distinct-head-dim Gemma-4 is
                // future work — GuardKvCacheHeadDim only admits per-layer F32 today,
                // so this branch is reached only for uniform-head-dim Gemma-4.)
                Attention.Execute(q, qkvCache, layer, attnOut,
                    seqLen, seqKv, numHeads, numKvHeadsLayer, headDimLayer, positions[0], _threadPool,
                    layerSlidingWindow, softCap: 0f);
            }
            else
            {
                var cachedK = kvCache.GetKeysRef(layer);
                var cachedV = kvCache.GetValuesRef(layer);
                Attention.Execute(q, (float*)cachedK.DataPointer, (float*)cachedV.DataPointer, attnOut,
                    seqLen, seqKv, numHeads, numKvHeadsLayer, headDimLayer, positions[0], attnScale,
                    _threadPool, layerSlidingWindow, softCap: 0f);
            }
        }
        else
        {
            Attention.Execute(q, k, v, attnOut,
                seqLen, seqLen, numHeads, numKvHeadsLayer, headDimLayer, 0, attnScale, _threadPool,
                layerSlidingWindow, softCap: 0f,
                _currentMaskSpec.Mode, _currentMaskSpec.PrefixLength);
        }

        // O projection: attnOut [seqLen × numHeads*headDim] → normOut [seqLen × hidden]
        Gemm(lw.OWeight, lw.OQuantType, attnOut, normOut, lw.OOutputDim, lw.OInputDim, seqLen);
        if (_currentAdapter is not null)
            ApplyLoraDelta(layer, "o_proj", attnOut, normOut, seqLen, lw.OInputDim, lw.OOutputDim);

        // post_attention_norm, then residual add: attn_out = rms(O)*post_attn + input.
        for (int t = 0; t < seqLen; t++)
            RmsNorm.Execute(
                new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                lw.PostAttnNormWeight!, eps,
                new Span<float>(normOut + t * hiddenSize, hiddenSize));

        long hElems = (long)seqLen * hiddenSize;
        float[] attnOutBuf = ArrayPool<float>.Shared.Rent((int)hElems); // post-attn residual (input to both FFN branches)
        float[] denseBuf = ArrayPool<float>.Shared.Rent((int)hElems);
        float[] moeBuf = ArrayPool<float>.Shared.Rent((int)hElems);
        float[] tmpNormBuf = ArrayPool<float>.Shared.Rent((int)hElems);
        try
        {
            fixed (float* attnOutP = attnOutBuf)
            fixed (float* denseP = denseBuf)
            fixed (float* moeP = moeBuf)
            fixed (float* tmpP = tmpNormBuf)
            {
                // attn_out = post_attn_norm(O) + residual(input)
                for (int t = 0; t < seqLen; t++)
                    Add.Execute(
                        new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize),
                        new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                        new Span<float>(attnOutP + t * hiddenSize, hiddenSize));

                // ── Dense FFN branch (shared expert) ──────────────────────
                // cur_mlp = down( geglu(gate·n) * (up·n) ), n = rms(attn_out)*ffn_norm
                Gemma4DenseFfn(in lw, layer, attnOutP, denseP, tmpP, normOut, seqLen, eps);
                // cur_mlp = rms(cur_mlp) * post_ffw_norm_1
                for (int t = 0; t < seqLen; t++)
                    RmsNorm.Execute(
                        new ReadOnlySpan<float>(denseP + t * hiddenSize, hiddenSize),
                        g4.PostFfwNorm1, eps,
                        new Span<float>(denseP + t * hiddenSize, hiddenSize));

                // ── MoE branch ─────────────────────────────────────────────
                Gemma4Moe(in lw, layer, attnOutP, moeP, tmpP, seqLen, eps);
                // cur_moe = rms(cur_moe) * post_ffw_norm_2
                for (int t = 0; t < seqLen; t++)
                    RmsNorm.Execute(
                        new ReadOnlySpan<float>(moeP + t * hiddenSize, hiddenSize),
                        g4.PostFfwNorm2, eps,
                        new Span<float>(moeP + t * hiddenSize, hiddenSize));

                // ── Combine: cur = rms(cur_mlp + cur_moe)*post_ffw_norm + attn_out ──
                for (int t = 0; t < seqLen; t++)
                {
                    var dSpan = new ReadOnlySpan<float>(denseP + t * hiddenSize, hiddenSize);
                    var mSpan = new ReadOnlySpan<float>(moeP + t * hiddenSize, hiddenSize);
                    var sumSpan = new Span<float>(tmpP + t * hiddenSize, hiddenSize);
                    TensorPrimitives.Add(dSpan, mSpan, sumSpan);
                    RmsNorm.Execute(
                        (ReadOnlySpan<float>)sumSpan, g4.PostFfwNorm, eps, sumSpan);
                    // cur = cur + attn_out, then * layer_output_scale (LAST op).
                    // Diffusion: prompt rows [0,P) use enc_layer_output_scale,
                    // canvas rows [P,seqLen) use layer_output_scale. Non-diffusion
                    // gemma4 keeps layer_output_scale for ALL rows (regionP == 0).
                    float scale = t < regionP ? encScale : g4.LayerOutputScale;
                    float* outT = hidden + t * hiddenSize;
                    float* aoT = attnOutP + t * hiddenSize;
                    float* sT = tmpP + t * hiddenSize;
                    for (int j = 0; j < hiddenSize; j++)
                        outT[j] = (sT[j] + aoT[j]) * scale;
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(attnOutBuf);
            ArrayPool<float>.Shared.Return(denseBuf);
            ArrayPool<float>.Shared.Return(moeBuf);
            ArrayPool<float>.Shared.Return(tmpNormBuf);
        }
    }

    /// <summary>
    /// Gemma-4 dense FFN branch: <c>down( geglu(gate·n) * (up·n) )</c> where
    /// <c>n = rms(attn_out) * ffn_norm</c>. Writes [seqLen × hidden] into
    /// <paramref name="dense"/>. Uses the layer's dense gate/up/down slots and the
    /// model-wide dense intermediate width. LoRA (when active) applies to this
    /// dense/"shared expert" branch here; the routed MoE experts
    /// (<see cref="Gemma4Moe"/>) get their own per-expert LoRA delta via the shared
    /// <c>MoeSwiGluMlp.ExecuteRoutedFromAssignments</c> kernel's
    /// <c>"mlp.experts.{j}.{proj}"</c> lookup.
    /// </summary>
    private unsafe void Gemma4DenseFfn(
        in TransformerLayerWeights lw, int layer, float* attnOut, float* dense, float* normScratch,
        float* hiddenScratch, int seqLen, float eps)
    {
        int hiddenSize = Config.HiddenSize;
        int interm = lw.GateOutputDim; // dense FFN width (2112)
        float* ffnGate = (float*)_state.FfnGate;
        float* ffnUp = (float*)_state.FfnUp;
        float* siluOut = (float*)_state.SiluOutput;

        // n = rms(attn_out) * ffn_norm
        for (int t = 0; t < seqLen; t++)
            RmsNorm.Execute(
                new ReadOnlySpan<float>(attnOut + t * hiddenSize, hiddenSize),
                lw.FfnNormWeight, eps,
                new Span<float>(hiddenScratch + t * hiddenSize, hiddenSize));

        Gemm(lw.GateWeight, lw.GateQuantType, hiddenScratch, ffnGate, lw.GateOutputDim, lw.GateInputDim, seqLen);
        Gemm(lw.UpWeight, lw.UpQuantType, hiddenScratch, ffnUp, lw.UpOutputDim, lw.UpInputDim, seqLen);
        if (_currentAdapter is not null)
        {
            ApplyLoraDelta(layer, "gate_proj", hiddenScratch, ffnGate, seqLen, lw.GateInputDim, lw.GateOutputDim);
            ApplyLoraDelta(layer, "up_proj", hiddenScratch, ffnUp, seqLen, lw.UpInputDim, lw.UpOutputDim);
        }

        for (int t = 0; t < seqLen; t++)
        {
            var gSpan = new ReadOnlySpan<float>(ffnGate + t * interm, interm);
            var uSpan = new ReadOnlySpan<float>(ffnUp + t * interm, interm);
            var oSpan = new Span<float>(siluOut + t * interm, interm);
            FusedOps.GeGLUTanh(gSpan, uSpan, oSpan);
        }

        Gemm(lw.DownWeight, lw.DownQuantType, siluOut, dense, lw.DownOutputDim, lw.DownInputDim, seqLen);
        if (_currentAdapter is not null)
            ApplyLoraDelta(layer, "down_proj", siluOut, dense, seqLen, lw.DownInputDim, lw.DownOutputDim);
    }

    /// <summary>
    /// Gemma-4 MoE branch with the custom router and per-expert down scale.
    /// Expert input <c>cur_moe = rms(attn_out) * pre_ffw_norm_2</c>; router logits
    /// <c>= ffn_gate_inp · (rms(attn_out) * 1/sqrt(hidden) * ffn_gate_inp_s)</c>.
    /// Routes (softmax → top-k → renorm with 6.1e-5 clamp), runs the GeGLU experts
    /// over the fused gate_up bank (raw quant, split by row offset) + the down bank,
    /// folds <c>ffn_down_exps.scale[e]</c> into each routing weight, and writes the
    /// weighted sum [seqLen × hidden] into <paramref name="moe"/>.
    /// </summary>
    private unsafe void Gemma4Moe(
        in TransformerLayerWeights lw, int layer, float* attnOut, float* moe, float* normScratch,
        int seqLen, float eps)
    {
        int hiddenSize = Config.HiddenSize;
        var g4 = lw.Gemma4!;
        var moeW = lw.Moe!;
        int numExperts = moeW.NumExperts;
        int topK = moeW.NumExpertsPerTok;
        int moeInterm = moeW.IntermediateSize;
        float invSqrtH = 1.0f / MathF.Sqrt(hiddenSize);

        long hElems = (long)seqLen * hiddenSize;
        float[] expertInBuf = ArrayPool<float>.Shared.Rent((int)hElems);   // rms(attn_out)*pre_ffw_norm_2
        float[] routerInBuf = ArrayPool<float>.Shared.Rent((int)hElems);   // rms(attn_out)*invSqrtH*router_scale
        float[] rmsBuf = ArrayPool<float>.Shared.Rent(hiddenSize);

        int totalAssign = seqLen * topK;
        int[] assignExpert = ArrayPool<int>.Shared.Rent(totalAssign);
        float[] assignWeight = ArrayPool<float>.Shared.Rent(totalAssign);
        int[] bucketCursors = ArrayPool<int>.Shared.Rent(numExperts + 1);
        int[] bucketTokens = ArrayPool<int>.Shared.Rent(totalAssign);
        int[] bucketSlots = ArrayPool<int>.Shared.Rent(totalAssign);
        int[] uniqueExperts = ArrayPool<int>.Shared.Rent(Math.Min(numExperts, totalAssign));
        float[] logitsBuf = ArrayPool<float>.Shared.Rent(numExperts);
        float[] probsBuf = ArrayPool<float>.Shared.Rent(numExperts);
        try
        {
            // Expert input + router input (both derived from rms(attn_out)).
            for (int t = 0; t < seqLen; t++)
            {
                var rms = rmsBuf.AsSpan(0, hiddenSize);
                RmsNorm.ExecuteUnit(
                    new ReadOnlySpan<float>(attnOut + t * hiddenSize, hiddenSize), eps, rms);
                // expert input = rms * pre_ffw_norm_2
                TensorPrimitives.Multiply(rms, g4.PreFfwNorm2, expertInBuf.AsSpan(t * hiddenSize, hiddenSize));
                // router input = rms * invSqrtH * router_scale
                var rin = routerInBuf.AsSpan(t * hiddenSize, hiddenSize);
                for (int j = 0; j < hiddenSize; j++)
                    rin[j] = rms[j] * invSqrtH * g4.RouterScale[j];
            }

            // Custom routing: logits = ffn_gate_inp · routerInput, softmax over E,
            // top-k, renorm (sum 1) with min-clamp 6.1e-5. Fill buckets.
            bucketCursors.AsSpan(0, numExperts + 1).Clear();
            Span<int> topkIdx = stackalloc int[topK];
            Span<float> topkProb = stackalloc float[topK];
            fixed (float* gatePtr = moeW.Gate)
            fixed (float* routerInP = routerInBuf)
            fixed (float* logitsP = logitsBuf)
            {
                for (int t = 0; t < seqLen; t++)
                {
                    MatMul.GemvF32(gatePtr, routerInP + (long)t * hiddenSize, logitsP, numExperts, hiddenSize);

                    Softmax.Execute(logitsBuf.AsSpan(0, numExperts), probsBuf.AsSpan(0, numExperts));
                    MoeSwiGluMlp.SelectTopK(probsBuf.AsSpan(0, numExperts), topkIdx, topkProb);

                    float sum = 0f;
                    for (int i = 0; i < topK; i++) sum += topkProb[i];
                    // Renorm to sum 1, clamping the denominator at 6.1e-5 to avoid
                    // dividing by a vanishing top-k mass (gemma4.cpp).
                    if (sum < 6.103515625e-05f) sum = 6.103515625e-05f;
                    float invSum = 1.0f / sum;
                    for (int i = 0; i < topK; i++)
                    {
                        float w = topkProb[i] * invSum;
                        int e = topkIdx[i];
                        int idx = t * topK + i;
                        assignExpert[idx] = e;
                        // Fold the per-expert down scale into the routing weight: the
                        // final accumulation does sum_e w[e]*down_e, and the spec
                        // scales down_e by ffn_down_exps.scale[e] — equivalent.
                        assignWeight[idx] = w * g4.DownExpertScale[e];
                        bucketCursors[e]++;
                    }
                }
            }

            // Exclusive prefix sum → bucket offsets.
            int running = 0;
            for (int e = 0; e <= numExperts; e++)
            {
                int c = bucketCursors[e];
                bucketCursors[e] = running;
                running += c;
            }
            // Fill buckets.
            int[] cursor = ArrayPool<int>.Shared.Rent(numExperts);
            try
            {
                for (int e = 0; e < numExperts; e++) cursor[e] = bucketCursors[e];
                for (int t = 0; t < seqLen; t++)
                    for (int slot = 0; slot < topK; slot++)
                    {
                        int e = assignExpert[t * topK + slot];
                        int pos = cursor[e]++;
                        bucketTokens[pos] = t;
                        bucketSlots[pos] = slot;
                    }
            }
            finally { ArrayPool<int>.Shared.Return(cursor); }

            int uniqueCount = 0;
            for (int e = 0; e < numExperts; e++)
                if (bucketCursors[e + 1] - bucketCursors[e] > 0)
                    uniqueExperts[uniqueCount++] = e;

            // Per-expert GeGLU over the raw fused gate_up bank (gate offset 0, up
            // offset Ie rows, both step the 2*Ie-row slab) + the down bank. The
            // kernel applies assignWeight (which carries the per-expert down scale).
            MoeSwiGluMlp.ExecuteRoutedFromAssignments(
                expertInBuf.AsSpan(0, (int)hElems),
                gateExpsRawBase: moeW.GateExpsRaw, moeW.GateExpsRawQt, g4.GateUpExpsRowBytes, ReadOnlySpan<nint>.Empty,
                upExpsRawBase: moeW.UpExpsRaw, moeW.UpExpsRawQt, g4.GateUpExpsRowBytes, ReadOnlySpan<nint>.Empty,
                downExpsRawBase: moeW.DownExpsRaw, moeW.DownExpsRawQt, g4.DownExpsRowBytes, ReadOnlySpan<nint>.Empty,
                assignExpert.AsSpan(0, totalAssign),
                assignWeight.AsSpan(0, totalAssign),
                bucketCursors.AsSpan(0, numExperts + 1),
                bucketTokens.AsSpan(0, totalAssign),
                bucketSlots.AsSpan(0, totalAssign),
                uniqueExperts.AsSpan(0, uniqueCount),
                uniqueCount,
                new Span<float>(moe, (int)hElems),
                numExperts, topK, hiddenSize, moeInterm, seqLen,
                ReadOnlySpan<nint>.Empty, ReadOnlySpan<nint>.Empty, ReadOnlySpan<nint>.Empty,
                0, ReadOnlySpan<float>.Empty,
                loraAdapter: _currentAdapter, loraLayer: layer,
                threadPool: _threadPool,
                useGeGLU: true);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(expertInBuf);
            ArrayPool<float>.Shared.Return(routerInBuf);
            ArrayPool<float>.Shared.Return(rmsBuf);
            ArrayPool<int>.Shared.Return(assignExpert);
            ArrayPool<float>.Shared.Return(assignWeight);
            ArrayPool<int>.Shared.Return(bucketCursors);
            ArrayPool<int>.Shared.Return(bucketTokens);
            ArrayPool<int>.Shared.Return(bucketSlots);
            ArrayPool<int>.Shared.Return(uniqueExperts);
            ArrayPool<float>.Shared.Return(logitsBuf);
            ArrayPool<float>.Shared.Return(probsBuf);
        }
    }

    /// <summary>
    /// Weight-less per-head RMSNorm (rms only, no learned scale) applied in place to
    /// each of <paramref name="numKvHeads"/> head slices of width
    /// <paramref name="headDim"/>. Gemma 4's V-norm: <c>Vcur = rms(Vcur)</c> with no
    /// weight, on every layer, before attention (V is not roped).
    /// </summary>
    private static unsafe void ApplyPerHeadNormWeightless(
        float* v, int numKvHeads, int headDim, int seqLen, float eps)
    {
        int stride = numKvHeads * headDim;
        for (int t = 0; t < seqLen; t++)
            for (int h = 0; h < numKvHeads; h++)
            {
                float* head = v + t * stride + h * headDim;
                RmsNorm.ExecuteUnit(
                    new ReadOnlySpan<float>(head, headDim), eps,
                    new Span<float>(head, headDim));
            }
    }

    /// <summary>
    /// LM head GEMM at <paramref name="seqLen"/> rows. Reads the final hidden state
    /// from <c>_state.HiddenState[0..seqLen*hiddenSize]</c> (left there by
    /// <see cref="RunLayersAndFinalNormCore"/>), writes logits into
    /// <c>_state.Logits</c>, allocates a freshly-owned tensor and copies the logits
    /// into it. Caller disposes the tensor.
    /// </summary>
    private unsafe ITensor RunLmHead(int seqLen, int deviceId)
    {
        int vocabSize = Config.VocabSize;
        float* hidden = (float*)_state.HiddenState;
        float* logits = (float*)_state.Logits;

        var rwOutput = _weights.RepackedOutput ?? default;
        GemmInterleaved(_weights.OutputWeight, _weights.OutputQuantType,
            hidden, logits, _weights.OutputOutputDim, _weights.OutputInputDim, seqLen,
            null, in rwOutput);

        // Optional Gemma 2/3 final-logit soft-cap (z' = cap * tanh(z / cap)).
        // Fires when Config.FinalLogitSoftcap is non-null and positive. Uses
        // TensorPrimitives.Tanh for the SIMD-accelerated kernel.
        ApplyFinalLogitSoftcap(logits, (long)seqLen * vocabSize);

        var shape = new TensorShape(seqLen, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId);
        new Span<float>(logits, seqLen * vocabSize).CopyTo(
            new Span<float>((void*)result.DataPointer, seqLen * vocabSize));
        return result;
    }

    /// <summary>
    /// Applies <c>z' = cap * tanh(z / cap)</c> in-place over <paramref name="count"/> floats
    /// at <paramref name="logits"/> when <see cref="ModelConfig.FinalLogitSoftcap"/> is set
    /// (Gemma 2 / Gemma 3). No-op when the field is null or non-positive. Uses
    /// <see cref="TensorPrimitives"/> for SIMD-accelerated multiply/tanh.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private unsafe void ApplyFinalLogitSoftcap(float* logits, long count)
    {
        if (Config.FinalLogitSoftcap is not float cap || cap <= 0f) return;
        // Process in <= int.MaxValue chunks (the span constructor is int-bounded).
        long offset = 0;
        while (offset < count)
        {
            int chunk = (int)Math.Min(count - offset, int.MaxValue);
            var span = new Span<float>(logits + offset, chunk);
            float inv = 1.0f / cap;
            TensorPrimitives.Multiply(span, inv, span);
            TensorPrimitives.Tanh(span, span);
            TensorPrimitives.Multiply(span, cap, span);
            offset += chunk;
        }
    }

    /// <summary>
    /// Fused forward across multiple in-flight sequences. Sequences are partitioned
    /// into a SIMPLE subgroup (GQA / MHA / MQA, no MLA, no MoE, no adapter) and a
    /// COMPLEX subgroup (any of those features present). The simple subgroup runs
    /// through <see cref="RunLayersAndFinalNormBatched"/>, which fuses the per-layer
    /// Q/K/V/O/gate/up/down GEMMs across sequences (one big <c>[Σ N_i, hidden] × W</c>
    /// dispatch instead of N small ones — the matmul-fusion win this method exists for).
    /// Complex sequences fall back to a per-seq <see cref="RunLayersAndFinalNormCore"/>
    /// loop. The lm_head GEMM is fused across the union of both subgroups.
    /// </summary>
    /// <remarks>
    /// <para>Phase 5a fused the lm_head only. Phase 5b adds intra-block matmul fusion
    /// for the simple subgroup. Attention still runs per-seq (each sequence has its
    /// own KV cache, positions, and position offset) — only the GEMMs at the seam
    /// of the attention block are fused.</para>
    /// <para>Parity contract: byte-identical per-element logits vs the per-seq
    /// <see cref="Forward(System.ReadOnlySpan{int},System.ReadOnlySpan{int},int,IKvCache?)"/>
    /// loop. Each batched-GEMM output element is an independent dot product over a
    /// fixed-length contraction axis, so per-row results don't depend on the batched
    /// row count.</para>
    /// </remarks>
    public IReadOnlyList<ITensor> ForwardBatch(
        IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0) return Array.Empty<ITensor>();
        if (requests.Count == 1)
        {
            var r0 = requests[0];
            return new[] { Forward(r0.TokenIds.Span, r0.Positions.Span,
                                   deviceId, r0.KvCache, r0.Adapter) };
        }

        int hiddenSize = Config.HiddenSize;
        int vocabSize = Config.VocabSize;
        int totalTokens = 0;
        foreach (var r in requests) totalTokens += r.TokenIds.Length;

        _state.EnsureCapacity(totalTokens);

        // Partition into simple (matmul-fused) vs complex (per-seq fallback)
        // subgroups. The model-level "has complex layer" check is one-shot — if
        // ANY layer is MLA or MoE, the batched path can't fuse safely (the per-
        // layer branch executes for every sequence in the batch, so even a
        // simple-looking sequence would hit the MLA/MoE branch). LoRA adapters
        // are per-sequence, so each request is judged individually.
        bool modelHasComplexLayer = ModelHasMlaOrMoeLayer();
        Span<int> simpleIdxs = requests.Count <= 256
            ? stackalloc int[requests.Count]
            : new int[requests.Count];
        Span<int> complexIdxs = requests.Count <= 256
            ? stackalloc int[requests.Count]
            : new int[requests.Count];
        int simpleCount = 0;
        int complexCount = 0;
        int simpleTotalTokens = 0;
        for (int i = 0; i < requests.Count; i++)
        {
            var r = requests[i];
            bool seqComplex = modelHasComplexLayer || r.Adapter is not null;
            if (seqComplex)
            {
                complexIdxs[complexCount++] = i;
            }
            else
            {
                simpleIdxs[simpleCount++] = i;
                simpleTotalTokens += r.TokenIds.Length;
            }
        }

        // Per-batch snapshot buffer: each per-seq RunLayersAndFinalNormCore call
        // and the batched simple-subgroup pass all write final hidden states into
        // _state.HiddenState (overlapping). We copy each seq's slice OUT to its
        // index-ordered offset in `batched` immediately after producing it, then
        // copy the whole thing BACK into _state.HiddenState for the batched
        // lm_head dispatch. The total snapshot footprint is the same as Phase 5a
        // (totalTokens * hidden * 4 bytes).
        var pool = ArrayPool<float>.Shared;
        float[] batched = pool.Rent(totalTokens * hiddenSize);
        try
        {
            // Per-seq token offsets in the original (caller-supplied) request
            // order — drives the lm_head logits-split-back step and the per-seq
            // copy destination in `batched`.
            Span<int> tokOffsets = requests.Count <= 256
                ? stackalloc int[requests.Count]
                : new int[requests.Count];
            int running = 0;
            for (int i = 0; i < requests.Count; i++)
            {
                tokOffsets[i] = running;
                running += requests[i].TokenIds.Length;
            }

            // ── Simple subgroup: batched matmul path ────────────────────────
            // Writes its sequences' final hidden states into _state.HiddenState
            // packed in the order of `simpleIdxs[0..simpleCount]`. We snapshot
            // each one out into its original-index offset in `batched`.
            if (simpleCount > 0)
            {
                RunLayersAndFinalNormBatched(requests, simpleIdxs[..simpleCount], simpleTotalTokens);

                int packedOff = 0;
                float* hidden = (float*)_state.HiddenState;
                for (int s = 0; s < simpleCount; s++)
                {
                    int origIdx = simpleIdxs[s];
                    int n = requests[origIdx].TokenIds.Length;
                    new Span<float>(hidden + packedOff * hiddenSize, n * hiddenSize)
                        .CopyTo(batched.AsSpan(tokOffsets[origIdx] * hiddenSize, n * hiddenSize));
                    packedOff += n;
                }
            }

            // ── Complex subgroup: per-seq fallback (Phase 5a behaviour) ─────
            for (int c = 0; c < complexCount; c++)
            {
                int origIdx = complexIdxs[c];
                var r = requests[origIdx];
                int n = r.TokenIds.Length;
                if (r.Adapter is not null)
                {
                    ValidateAdapterForModel(r.Adapter);
                    LoraStage2.PrewarmAdapter(r.Adapter as LoraAdapter);
                    _currentAdapter = r.Adapter;
                }
                try
                {
                    RunLayersAndFinalNormCore(r.TokenIds.Span, r.Positions.Span, r.KvCache);
                    new Span<float>((float*)_state.HiddenState, n * hiddenSize)
                        .CopyTo(batched.AsSpan(tokOffsets[origIdx] * hiddenSize, n * hiddenSize));
                }
                finally
                {
                    if (r.Adapter is not null) _currentAdapter = null;
                }
            }

            // Stack the per-seq snapshots back into _state.HiddenState in original
            // request order, then run one batched lm_head dispatch at seqLen = Σ N_i.
            batched.AsSpan(0, totalTokens * hiddenSize)
                .CopyTo(new Span<float>((float*)_state.HiddenState, totalTokens * hiddenSize));

            float* logitsPtr = (float*)_state.Logits;
            var rwOutput = _weights.RepackedOutput ?? default;
            GemmInterleaved(_weights.OutputWeight, _weights.OutputQuantType,
                (float*)_state.HiddenState, logitsPtr,
                _weights.OutputOutputDim, _weights.OutputInputDim, totalTokens,
                null, in rwOutput);

            // Optional Gemma 2/3 final-logit soft-cap over the entire batched logits
            // block. Same convention as the per-seq path; no-op when not configured.
            ApplyFinalLogitSoftcap(logitsPtr, (long)totalTokens * vocabSize);

            // Split logits per-seq.
            var results = new ITensor[requests.Count];
            for (int i = 0; i < requests.Count; i++)
            {
                int n = requests[i].TokenIds.Length;
                int srcOff = tokOffsets[i];
                var shape = new TensorShape(n, vocabSize);
                var tensor = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId);
                new Span<float>(logitsPtr + (long)srcOff * vocabSize, n * vocabSize).CopyTo(
                    new Span<float>((void*)tensor.DataPointer, n * vocabSize));
                results[i] = tensor;
            }
            return results;
        }
        finally
        {
            pool.Return(batched);
        }
    }

    /// <summary>
    /// Returns true when any layer of this model uses MLA (DeepSeek-V2/V3) or
    /// MoE (Mixtral / Qwen-MoE / DeepSeek-V2/V3). Such layers carry per-layer
    /// kernels that aren't trivially batchable across sequences in the Phase 5b
    /// matmul-fused path, so the entire batch falls back to per-seq when this
    /// returns true.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private bool ModelHasMlaOrMoeLayer()
    {
        var layers = _weights.Layers;
        for (int i = 0; i < layers.Length; i++)
        {
            if (layers[i].Mla is not null || layers[i].Moe is not null) return true;
        }
        return false;
    }

    /// <summary>
    /// Phase 5b matmul-fused layer loop for the SIMPLE subgroup (GQA / MHA / MQA,
    /// no MLA / no MoE / no LoRA adapter). For each transformer layer:
    /// <list type="number">
    /// <item>Concat per-seq hidden states into a single <c>[Σ N_i, hidden]</c>
    ///   batched buffer (residual copy already does this via the packed layout
    ///   — sequences are stored contiguously in <c>_state.HiddenState</c>).</item>
    /// <item>One batched RMSNorm over <c>Σ N_i</c> rows.</item>
    /// <item>One batched QuantizeInput.</item>
    /// <item>One batched Q/K/V GEMM at <c>[Σ N_i, hidden] × [hidden, dim]</c>.</item>
    /// <item>Q/K/V outputs are sliced per-seq for RoPE + attention (each seq has
    ///   independent positions / position offset / KV cache).</item>
    /// <item>One batched O projection + residual.</item>
    /// <item>Same pattern for the FFN block (RMSNorm + gate/up GEMM + SwiGLU +
    ///   down GEMM + residual).</item>
    /// </list>
    /// At return, each simple sequence's final hidden state is packed contiguously
    /// in <c>_state.HiddenState</c> in <paramref name="simpleIdxs"/> order, having
    /// passed through the final RMSNorm.
    /// </summary>
    /// <remarks>
    /// Parity contract with <see cref="RunLayersAndFinalNormCore"/>: byte-identical
    /// per-element output. Each batched-GEMM output element is an independent dot
    /// product over a fixed-length contraction axis, so the FP accumulation order
    /// (and therefore the per-row result) does NOT depend on whether the GEMM
    /// processes 1 or <c>Σ N_i</c> rows.
    /// </remarks>
    [SkipLocalsInit]
    private unsafe void RunLayersAndFinalNormBatched(
        IReadOnlyList<SequenceForwardRequest> requests,
        ReadOnlySpan<int> simpleIdxs,
        int simpleTotalTokens)
    {
        // The batched path attends through per-request KV caches; a distinct
        // per-layer head dim is cacheless-only (see GuardKvCacheHeadDim).
        if (HasDistinctPerLayerHeadDim)
            throw new NotSupportedException(
                $"Gemma 4 with distinct global_head_dim ({Config.GlobalHeadDim}) and head_dim "
                + $"({Config.HeadDim}) is not supported on the batched (KV-cached) forward path. "
                + "Use the cacheless single-sequence Forward path.");

        int maxSeq = Config.MaxSequenceLength;
        // Validate positions per-seq (mirrors the per-seq core).
        for (int s = 0; s < simpleIdxs.Length; s++)
        {
            var positions = requests[simpleIdxs[s]].Positions.Span;
            for (int i = 0; i < positions.Length; i++)
            {
                if ((uint)positions[i] >= (uint)maxSeq)
                    throw new ArgumentOutOfRangeException(nameof(requests),
                        $"Position {positions[i]} at index {i} of sequence {simpleIdxs[s]} exceeds max sequence length {maxSeq}.");
            }
        }

        int hiddenSize = Config.HiddenSize;
        int numHeads = Config.NumAttentionHeads;
        int headDim = Config.HeadDim;
        int intermediateSize = Config.IntermediateSize;
        int qStride = numHeads * headDim;
        float eps = Config.NormEpsilon;

        // Total tokens across simple seqs. The caller has already called
        // EnsureCapacity on _state for at least this much.
        int total = simpleTotalTokens;

        // EventBased: batched is by definition multi-token (we early-return at
        // requests.Count==1 above, so simpleCount + complexCount ≥ 2; even with
        // 4× decode the batched matmul is "prefill-shaped" relative to a 1-token
        // dispatch). The per-seq fallback path may flip back to SpinWait inside
        // RunLayersAndFinalNormCore — that's fine, both modes are independent.
        _threadPool?.SetDispatchMode(DispatchMode.EventBased);

        float* hidden = (float*)_state.HiddenState;
        float* residual = (float*)_state.Residual;
        float* normOut = (float*)_state.NormOutput;
        float* q = (float*)_state.Q;
        float* k = (float*)_state.K;
        float* v = (float*)_state.V;
        float* attnOut = (float*)_state.AttnOutput;
        float* ffnGate = (float*)_state.FfnGate;
        float* ffnUp = (float*)_state.FfnUp;
        float* siluOut = (float*)_state.SiluOutput;

        // Packed per-seq token offsets (into the batched [total, *] buffers).
        // simpleIdxs[s] gives the caller-supplied request index for sub-seq s,
        // packedOffsets[s] gives where that seq starts in the batched buffers.
        Span<int> packedOffsets = simpleIdxs.Length <= 256
            ? stackalloc int[simpleIdxs.Length]
            : new int[simpleIdxs.Length];
        int run = 0;
        for (int s = 0; s < simpleIdxs.Length; s++)
        {
            packedOffsets[s] = run;
            run += requests[simpleIdxs[s]].TokenIds.Length;
        }

        // 1. EMBEDDING LOOKUP — pack per-seq directly into the batched buffer.
        for (int s = 0; s < simpleIdxs.Length; s++)
        {
            var r = requests[simpleIdxs[s]];
            int n = r.TokenIds.Length;
            EmbeddingLookup(r.TokenIds.Span, hidden + (long)packedOffsets[s] * hiddenSize, hiddenSize);
        }

        // 2. TRANSFORMER LAYERS
        var repackedLayers = _weights.RepackedLayers;
        int numLayers = DebugMaxLayers switch
        {
            < 0 => 0,
            0 => Config.NumLayers,
            _ => Math.Min(DebugMaxLayers, Config.NumLayers)
        };

        for (int layer = 0; layer < numLayers; layer++)
        {
            ref readonly var lw = ref _weights.Layers[layer];
            var rl = repackedLayers?[layer];

            // Per-attention-type KV-head count (Gemma 4 dual-KV-head). Uniform
            // Config.NumKvHeads for every other architecture.
            int numKvHeadsLayer = GetLayerKvHeads(layer);
            int kvStrideLayer = numKvHeadsLayer * headDim;

            byte* inputQ8Scratch = (byte*)_state.InputQ8Scratch;

            // a. Copy hidden → residual (whole packed buffer).
            new Span<float>(hidden, total * hiddenSize).CopyTo(new Span<float>(residual, total * hiddenSize));

            // b. Batched RMSNorm: same per-row math as the prefill path of the
            // unfused loop (each row is an independent normalisation). Loop is
            // identical to RunLayersAndFinalNormCore's prefill RMSNorm.
            for (int t = 0; t < total; t++)
            {
                RmsNorm.Execute(
                    new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                    lw.AttnNormWeight, eps,
                    new Span<float>(normOut + t * hiddenSize, hiddenSize));
            }

            // c. Batched QuantizeInput + Q/K/V projections at n=total.
            byte* preQuantNorm = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, total, lw.QQuantType);

            var rwQ = rl?.Q ?? default;
            var rwK = rl?.K ?? default;
            var rwV = rl?.V ?? default;
            GemmInterleaved(lw.QWeight, lw.QQuantType, normOut, q, lw.QOutputDim, lw.QInputDim, total,
                preQuantNorm, in rwQ);
            GemmInterleaved(lw.KWeight, lw.KQuantType, normOut, k, lw.KOutputDim, lw.KInputDim, total,
                IsCompatiblePreQuant(lw.QQuantType, lw.KQuantType) ? preQuantNorm : null, in rwK);
            GemmInterleaved(lw.VWeight, lw.VQuantType, normOut, v, lw.VOutputDim, lw.VInputDim, total,
                IsCompatiblePreQuant(lw.QQuantType, lw.VQuantType) ? preQuantNorm : null, in rwV);

            // Optional bias (operates over all batched rows uniformly).
            AddBias(lw.QBias, q, lw.QOutputDim, total);
            AddBias(lw.KBias, k, lw.KOutputDim, total);
            AddBias(lw.VBias, v, lw.VOutputDim, total);

            // Optional QK-norms (Qwen3-style) — independently applied per row.
            if (lw.QNormWeight is not null)
                ApplyPerHeadNorm(lw.QNormWeight, q, numHeads, headDim, total, eps);
            if (lw.KNormWeight is not null)
                ApplyPerHeadNorm(lw.KNormWeight, k, numKvHeadsLayer, headDim, total, eps);

            // d/e. Per-sequence RoPE + Attention + KV-cache update. The Q/K/V
            // slices live at packedOffsets[s] in the batched buffers; we hand
            // each slice and the seq's own positions to the per-seq kernels.
            // Attention writes its output back into attnOut at the same offset,
            // re-stacking the post-attention tokens into a single packed buffer
            // ready for the next batched GEMM (O projection).
            bool applyRoPE = !Config.IsNoRopeLayer(layer);
            var (ropeCos, ropeSin, ropeDimLayer, ropeTypeLayer) = GetLayerRope(layer);
            for (int s = 0; s < simpleIdxs.Length; s++)
            {
                int origIdx = simpleIdxs[s];
                var r = requests[origIdx];
                int n = r.TokenIds.Length;
                int off = packedOffsets[s];
                var positions = r.Positions.Span;

                float* qSlice = q + (long)off * qStride;
                float* kSlice = k + (long)off * kvStrideLayer;
                float* vSlice = v + (long)off * kvStrideLayer;
                float* aSlice = attnOut + (long)off * qStride;

                if (applyRoPE)
                {
                    RoPE.Execute(
                        new Span<float>(qSlice, n * qStride),
                        new Span<float>(kSlice, n * kvStrideLayer),
                        positions,
                        numHeads, numKvHeadsLayer, headDim, ropeDimLayer,
                        ropeCos, ropeSin, ropeTypeLayer);
                }

                IKvCache kvCache = r.KvCache;
                // KV cache is required on the request — write new K/V then attend
                // over the cached range.
                var kRef = new TensorRef(n, kvStrideLayer, DType.Float32, -1, (nint)kSlice);
                var vRef = new TensorRef(n, kvStrideLayer, DType.Float32, -1, (nint)vSlice);
                kvCache.Update(kRef, vRef, positions, layer);

                int seqKv = kvCache.CurrentLength;
                int? layerSlidingWindow = GetLayerSlidingWindow(layer);
                float attnScale = Config.QueryPreAttnScalar is float qpas && qpas > 0
                    ? 1.0f / MathF.Sqrt(qpas)
                    : 1.0f / MathF.Sqrt(headDim);
                float attnSoftCap = Config.AttnLogitSoftcap ?? 0f;
                if (kvCache is IQuantizedKvCache qkvCache)
                {
                    Attention.Execute(qSlice, qkvCache, layer, aSlice,
                        n, seqKv, numHeads, numKvHeadsLayer, headDim, positions[0], _threadPool,
                        layerSlidingWindow, attnSoftCap);
                }
                else
                {
                    var cachedK = kvCache.GetKeysRef(layer);
                    var cachedV = kvCache.GetValuesRef(layer);
                    Attention.Execute(qSlice, (float*)cachedK.DataPointer, (float*)cachedV.DataPointer, aSlice,
                        n, seqKv, numHeads, numKvHeadsLayer, headDim, positions[0], attnScale,
                        _threadPool, layerSlidingWindow, attnSoftCap);
                }
            }

            // f. Batched O projection: [total, qStride] × [qStride, hidden] → [total, hidden] into normOut.
            byte* preQuantAttn = QuantizeInput(attnOut, inputQ8Scratch, qStride, total, lw.OQuantType);
            var rwO = rl?.O ?? default;
            GemmInterleaved(lw.OWeight, lw.OQuantType, attnOut, normOut, lw.OOutputDim, lw.OInputDim, total,
                preQuantAttn, in rwO);
            AddBias(lw.OBias, normOut, lw.OOutputDim, total);

            // g0. Gemma post-attention RMSNorm before the residual add (no-op
            // when PostAttnNormWeight is null — non-Gemma two-norm layout).
            if (lw.PostAttnNormWeight is float[] postAttnNorm)
            {
                for (int t = 0; t < total; t++)
                    RmsNorm.Execute(
                        new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                        postAttnNorm, eps,
                        new Span<float>(normOut + t * hiddenSize, hiddenSize));
            }

            // g. Residual add: hidden ← residual + normOut (all batched rows).
            for (int t = 0; t < total; t++)
            {
                Add.Execute(
                    new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize),
                    new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                    new Span<float>(hidden + t * hiddenSize, hiddenSize));
            }

            // h. Copy hidden → residual (snapshot for FFN block).
            new Span<float>(hidden, total * hiddenSize).CopyTo(new Span<float>(residual, total * hiddenSize));

            // i. Batched FFN RMSNorm + Gate/Up + SwiGLU + Down.
            for (int t = 0; t < total; t++)
            {
                RmsNorm.Execute(
                    new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                    lw.FfnNormWeight, eps,
                    new Span<float>(normOut + t * hiddenSize, hiddenSize));
            }

            byte* preQuantFfn = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, total, lw.GateQuantType);

            var rwGate = rl?.Gate ?? default;
            var rwUp = rl?.Up ?? default;
            GemmInterleaved(lw.GateWeight, lw.GateQuantType, normOut, ffnGate, lw.GateOutputDim, lw.GateInputDim, total,
                preQuantFfn, in rwGate);
            GemmInterleaved(lw.UpWeight, lw.UpQuantType, normOut, ffnUp, lw.UpOutputDim, lw.UpInputDim, total,
                IsCompatiblePreQuant(lw.GateQuantType, lw.UpQuantType) ? preQuantFfn : null, in rwUp);

            AddBias(lw.GateBias, ffnGate, lw.GateOutputDim, total);
            AddBias(lw.UpBias, ffnUp, lw.UpOutputDim, total);

            // Fused gate activation per row: GeGLU (Gemma) or SwiGLU.
            for (int t = 0; t < total; t++)
            {
                float* gateT = ffnGate + t * intermediateSize;
                float* upT = ffnUp + t * intermediateSize;
                float* siluT = siluOut + t * intermediateSize;

                var gateSpan = new ReadOnlySpan<float>(gateT, intermediateSize);
                var upSpan = new ReadOnlySpan<float>(upT, intermediateSize);
                var outSpan = new Span<float>(siluT, intermediateSize);
                if (_useGeGLU)
                    FusedOps.GeGLUTanh(gateSpan, upSpan, outSpan);
                else
                    FusedOps.SwiGLU(gateSpan, upSpan, outSpan);
            }

            // Batched Down projection: [total, intermediate] × [intermediate, hidden] → [total, hidden] into normOut.
            byte* preQuantSilu = QuantizeInput(siluOut, inputQ8Scratch, intermediateSize, total, lw.DownQuantType);
            var rwDown = rl?.Down ?? default;
            GemmInterleaved(lw.DownWeight, lw.DownQuantType, siluOut, normOut, lw.DownOutputDim, lw.DownInputDim, total,
                preQuantSilu, in rwDown);
            AddBias(lw.DownBias, normOut, lw.DownOutputDim, total);

            // j0. Gemma post-FFN RMSNorm before the residual add (no-op when null).
            if (lw.PostFfnNormWeight is float[] postFfnNorm)
            {
                for (int t = 0; t < total; t++)
                    RmsNorm.Execute(
                        new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                        postFfnNorm, eps,
                        new Span<float>(normOut + t * hiddenSize, hiddenSize));
            }

            // k. Final residual add.
            for (int t = 0; t < total; t++)
            {
                Add.Execute(
                    new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize),
                    new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                    new Span<float>(hidden + t * hiddenSize, hiddenSize));
            }
        }

        // 3. FINAL NORM (in-place: hidden → hidden) over all batched rows.
        for (int t = 0; t < total; t++)
        {
            float* hiddenT = hidden + t * hiddenSize;
            float* normOutT = normOut + t * hiddenSize;

            RmsNorm.Execute(
                new ReadOnlySpan<float>(hiddenT, hiddenSize),
                _weights.OutputNormWeight,
                eps,
                new Span<float>(normOutT, hiddenSize));

            new Span<float>(normOutT, hiddenSize).CopyTo(new Span<float>(hiddenT, hiddenSize));
        }
    }

    /// <summary>
    /// Validates that <paramref name="adapter"/> is compatible with this
    /// model and that its targeted projections do not collide with
    /// out-of-scope MLA / MoE structures. Called once per LoRA-aware Forward.
    /// </summary>
    private void ValidateAdapterForModel(ILoraAdapter adapter)
    {
        if (!adapter.IsCompatible(Config))
            throw new InvalidOperationException(
                $"LoRA adapter '{adapter.Name}' is not compatible with the loaded model "
                + "(layer count, hidden size, or per-projection dimensions mismatch).");

        // Phase 4d.2: MLA / MoE rejections are lifted. The standard
        // ApplyLoraDelta call sites are only reached on non-MLA / dense FFN
        // layers (the MLA branch in Forward routes through MlaAttention which
        // has its own LoRA hooks; MoE routes through MoeSwiGluMlp, which now
        // also resolves per-expert "mlp.experts.{j}.{proj}" entries — see
        // Gemma4Moe / PeftAdapterLoader's MoeExpertPathRegex). Adapters that
        // target standard q/k/v/o or gate/up/down on MLA layers still pass
        // through silently for the MLA-specific projection names
        // (q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj) that have no
        // ApplyLoraDelta hook yet — a non-applicable target is a no-op rather
        // than an error.
    }

    /// <summary>
    /// Applies the LoRA delta for <paramref name="projName"/> at
    /// <paramref name="layer"/> if the active adapter targets that site.
    /// No-op when there is no active adapter or no entry for this projection.
    /// <para>
    /// Phase 4d.5 / Gap 2 — when the caller has already quantised
    /// <paramref name="x"/> for the base projection's GEMM
    /// (<see cref="QuantizeInput"/>) and passes the resulting buffer as
    /// <paramref name="preQuantX"/>, AND <paramref name="preQuantXType"/> is
    /// <see cref="QuantizationType.Q8_0"/>, AND the adapter's B factor is
    /// <see cref="LoraWeightDType.Q8_0"/>, the LoRA stage-1 GEMM re-uses
    /// the pre-quantised buffer via
    /// <see cref="LoraDelta.ApplyQ8_0BWithPreQuantX"/> instead of dequanting B
    /// to F32 and running an F32 GEMM. This closes the residual −16% prefill
    /// regression the Phase 4d.4 dequant-once path left on the table on a
    /// Q8_0 base (Strix Halo / Llama-3.2-1B). The default arguments give the
    /// legacy F32 / dequant-once behaviour.
    /// </para>
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ApplyLoraDelta(int layer, string projName,
                                float* x, float* y, int seqLen, int inputDim, int outputDim,
                                byte* preQuantX = null,
                                QuantizationType preQuantXType = QuantizationType.F32)
    {
        if (_currentAdapter is null) return;

        // DIFFUSION region split: real DiffusionGemma PEFT adapters train INDEPENDENT
        // LoRA deltas for the prompt (encoder) and canvas (decoder) rows of the SAME
        // unified [prompt|canvas] forward — mirroring the backbone's own region-aware
        // per-layer scalar (see RunGemma4Layer's regionP / enc_layer_output_scale).
        // preQuantX is never non-null on this path (only the generic non-Gemma4 layer
        // path passes it, and that path never runs under DiffusionConfig), so a plain
        // pointer-offset split is safe here.
        bool diffusion = Config.DiffusionConfig is not null;
        if (diffusion && _currentMaskSpec.Mode == AttentionMaskMode.Hybrid && preQuantX is null)
        {
            int p = Math.Clamp(_currentMaskSpec.PrefixLength, 0, seqLen);
            if (p > 0)
                LoraProjection.Apply(_currentAdapter, layer, projName, x, y,
                                     p, inputDim, outputDim, _threadPool,
                                     region: LoraRegion.Encoder);
            if (p < seqLen)
                LoraProjection.Apply(_currentAdapter, layer, projName,
                                     x + (long)p * inputDim, y + (long)p * outputDim,
                                     seqLen - p, inputDim, outputDim, _threadPool,
                                     region: LoraRegion.Decoder);
            return;
        }

        LoraProjection.Apply(_currentAdapter, layer, projName, x, y,
                             seqLen, inputDim, outputDim, _threadPool,
                             preQuantX, preQuantXType);
    }

    /// <summary>
    /// Applies the LoRA delta for DiffusionGemma's model-level self-conditioning module
    /// (<see cref="ApplySelfConditioning"/>'s gate/up/down GeGLU MLP). Unlike
    /// <see cref="ApplyLoraDelta"/>, this does NOT go through the diffusion
    /// encoder/decoder region split: self-conditioning is a single computation that runs
    /// once per forward and only ever touches canvas rows (there is no prompt/canvas row
    /// split to make for it — the caller already restricts <paramref name="x"/> /
    /// <paramref name="y"/> to the <c>canvasLen</c> rows). Looked up under
    /// <see cref="LoraAdapter.SelfConditioningLayerIndex"/> with
    /// <see cref="LoraRegion.Any"/>, using the same <c>gate_proj</c>/<c>up_proj</c>/
    /// <c>down_proj</c> names the dense FFN uses (see
    /// <see cref="PeftAdapterLoader"/>'s <c>SelfConditioningPathRegex</c>).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ApplySelfConditioningLoraDelta(string projName,
                                                float* x, float* y, int canvasLen, int inputDim, int outputDim)
    {
        if (_currentAdapter is null) return;

        LoraProjection.Apply(_currentAdapter, LoraAdapter.SelfConditioningLayerIndex, projName, x, y,
                             canvasLen, inputDim, outputDim, _threadPool,
                             region: LoraRegion.Any);
    }

    /// <summary>
    /// Applies RMSNorm per attention head to a Q or K tensor [seqLen, numHeads * headDim].
    /// Used for QK-norm (Qwen3-style) where each head vector is independently normalized
    /// after projection and before RoPE.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyPerHeadNorm(float[] normWeight, float* qk,
        int numHeads, int headDim, int seqLen, float eps)
    {
        int stride = numHeads * headDim;

        // OLMoE applies a SINGLE RMSNorm over the entire Q/K projection (weight
        // length == num_heads*head_dim) before splitting into heads, whereas
        // Qwen3/Gemma apply a PER-HEAD RMSNorm (weight length == head_dim) after
        // the split. Distinguish by the resolved weight length. When numHeads==1
        // the two are numerically identical, so the guard is harmless.
        if (numHeads > 1 && normWeight.Length == stride)
        {
            for (int t = 0; t < seqLen; t++)
            {
                float* row = qk + t * stride;
                RmsNorm.Execute(
                    new ReadOnlySpan<float>(row, stride), normWeight, eps,
                    new Span<float>(row, stride));
            }
            return;
        }

        for (int t = 0; t < seqLen; t++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                float* head = qk + t * stride + h * headDim;
                var input = new ReadOnlySpan<float>(head, headDim);
                var output = new Span<float>(head, headDim);
                RmsNorm.Execute(input, normWeight, eps, output);
            }
        }
    }

    /// <summary>
    /// Adds a bias vector [outputDim] to each row of a [seqLen, outputDim] output buffer.
    /// No-op when <paramref name="bias"/> is null (zero overhead for bias-less models).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddBias(float[]? bias, float* output, int outputDim, int seqLen)
    {
        if (bias is null) return;
        for (int t = 0; t < seqLen; t++)
        {
            var row = new Span<float>(output + t * outputDim, outputDim);
            TensorPrimitives.Add((ReadOnlySpan<float>)row, bias, row);
        }
    }

    /// <summary>
    /// Dispatches to the appropriate GEMV kernel based on quantization type.
    /// Passes <see cref="_threadPool"/> for parallel execution.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Gemv(nint weights, QuantizationType qt, float* x, float* y, int m, int k)
    {
        if (qt == QuantizationType.Q8_0)
            MatMul.GemvQ8_0((byte*)weights, x, y, m, k, _threadPool);
        else if (qt == QuantizationType.Q5_0)
            MatMul.GemvQ5_0((byte*)weights, x, y, m, k, _threadPool);
        else if (qt == QuantizationType.Q4_K)
            MatMul.GemvQ4_K((byte*)weights, x, y, m, k, _threadPool);
        else if (qt == QuantizationType.Q5_K)
            MatMul.GemvQ5_K((byte*)weights, x, y, m, k, _threadPool);
        else if (qt == QuantizationType.Q6_K)
            MatMul.GemvQ6_K((byte*)weights, x, y, m, k, _threadPool);
        else if (qt == QuantizationType.F32)
            MatMul.GemvF32((float*)weights, x, y, m, k, _threadPool);
        else if (qt == QuantizationType.F16)
            MatMul.GemvF16(weights, x, y, m, k, _threadPool);
        else if (qt == QuantizationType.I2_S)
            MatMul.GemvI2_S((byte*)weights, x, y, m, k, _threadPool);
        else
            GemvDequantFallback(weights, qt, x, y, m, k);
    }

    /// <summary>
    /// Fallback GEMV for quant types without dedicated vec_dot kernels.
    /// Dequantizes one weight row at a time and computes float dot product.
    /// Correct but slower than fused kernels.
    /// </summary>
    private static void GemvDequantFallback(nint weights, QuantizationType qt, float* x, float* y, int m, int k)
    {
        long rowBytes = Dequantize.RowByteSize(k, qt);
        float[] rowBuf = ArrayPool<float>.Shared.Rent(k);
        try
        {
            var rowSpan = rowBuf.AsSpan(0, k);
            var xSpan = new ReadOnlySpan<float>(x, k);
            for (int i = 0; i < m; i++)
            {
                Dequantize.ToFloat32(weights + i * (nint)rowBytes, k, qt, rowSpan);
                y[i] = TensorPrimitives.Dot(new ReadOnlySpan<float>(rowBuf, 0, k), xSpan);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rowBuf);
        }
    }

    /// <summary>
    /// Dispatches to the appropriate GEMM kernel based on quantization type.
    /// Passes <see cref="_threadPool"/> for parallel execution.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Gemm(nint weights, QuantizationType qt, float* b, float* c,
                      int m, int k, int n, byte* preQuantizedInput = null)
    {
        if (qt == QuantizationType.Q8_0)
            MatMul.GemmQ8_0((byte*)weights, b, c, m, k, n, _threadPool, preQuantizedInput);
        else if (qt == QuantizationType.Q5_0)
            MatMul.GemmQ5_0((byte*)weights, b, c, m, k, n, _threadPool, preQuantizedInput);
        else if (qt == QuantizationType.Q4_K)
            MatMul.GemmQ4_K((byte*)weights, b, c, m, k, n, _threadPool, preQuantizedInput);
        else if (qt == QuantizationType.Q5_K)
            MatMul.GemmQ5_K((byte*)weights, b, c, m, k, n, _threadPool, preQuantizedInput);
        else if (qt == QuantizationType.Q6_K)
            MatMul.GemmQ6_K((byte*)weights, b, c, m, k, n, _threadPool, preQuantizedInput);
        else if (qt == QuantizationType.F32)
            MatMul.GemmF32((float*)weights, b, c, m, k, n, _threadPool);
        else if (qt == QuantizationType.F16)
            MatMul.GemmF16(weights, b, c, m, k, n, _threadPool);
        else if (qt == QuantizationType.I2_S)
            MatMul.GemmI2_S((byte*)weights, b, c, m, k, n, _threadPool);
        else
            GemmDequantFallback(weights, qt, b, c, m, k, n);
    }

    /// <summary>
    /// Fallback GEMM for quant types without dedicated vec_dot kernels.
    /// Iterates per input row, calling <see cref="GemvDequantFallback"/> for each.
    /// </summary>
    private static void GemmDequantFallback(nint weights, QuantizationType qt, float* b, float* c,
                                            int m, int k, int n)
    {
        for (int t = 0; t < n; t++)
        {
            GemvDequantFallback(weights, qt, b + t * k, c + t * m, m, k);
        }
    }

    /// <summary>
    /// Minimum row byte stride for R4 interleaving to be beneficial.
    /// Below this, 4 rows span &lt; 4KB (1 page) and the hardware prefetcher
    /// handles the original stride efficiently. Above this, R4 contiguity
    /// avoids cross-page TLB misses and prefetcher stride-limit failures.
    /// </summary>
    private const int InterleavedMinRowBytes = 1024;

    /// <summary>
    /// GEMV using R4-interleaved repacked weights for improved cache locality.
    /// Falls back to original Gemv when repacked weight is default (Ptr == 0)
    /// or when row stride is too small for interleaving to help.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void GemvInterleaved(nint origWeights, QuantizationType qt, float* x, float* y,
                                 int m, int k, in WeightRepacking.RepackedWeight rw)
    {
        if (rw.Ptr == 0 || rw.RowBytes < InterleavedMinRowBytes)
        {
            Gemv(origWeights, qt, x, y, m, k);
            return;
        }

        // Quantize input for the interleaved ComputeRows variant
        byte* inputQ8Scratch = (byte*)_state.InputQ8Scratch;
        if (qt == QuantizationType.Q8_0)
        {
            int blockCount = k / Q8_0GroupSize;
            int xQ8Bytes = blockCount * Q8_0BlockBytes;
            byte* xQ8 = (byte*)(_threadPool?.GetWorkerScratch(0, xQ8Bytes) ?? (nint)inputQ8Scratch);
            MatMul.QuantizeF32ToQ8_0(x, xQ8, k);
            MatMul.ComputeRowsQ8_0Interleaved((byte*)rw.Ptr, xQ8, y, rw.FullGroupCount, rw.TailRows, blockCount, _threadPool);
        }
        else if (qt == QuantizationType.Q5_0)
        {
            int blockCount = k / Q8_0GroupSize;
            int xQ8Bytes = blockCount * MatMul.Q8_1BlockBytes;
            byte* xQ8 = (byte*)(_threadPool?.GetWorkerScratch(0, xQ8Bytes) ?? (nint)inputQ8Scratch);
            MatMul.QuantizeF32ToQ8_1(x, xQ8, k);
            MatMul.ComputeRowsQ5_0Interleaved((byte*)rw.Ptr, xQ8, y, rw.FullGroupCount, rw.TailRows, blockCount, _threadPool);
        }
        else if (qt is QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K)
        {
            int superBlockCount = k / 256;
            int xQ8KBytes = superBlockCount * MatMul.Q8_K_BlockBytes;
            byte* xQ8K = (byte*)(_threadPool?.GetWorkerScratch(0, xQ8KBytes) ?? (nint)inputQ8Scratch);
            MatMul.QuantizeF32ToQ8_K(x, xQ8K, k);
            if (qt == QuantizationType.Q4_K)
                MatMul.ComputeRowsQ4_KInterleaved((byte*)rw.Ptr, xQ8K, y, rw.FullGroupCount, rw.TailRows, superBlockCount, _threadPool);
            else if (qt == QuantizationType.Q5_K)
                MatMul.ComputeRowsQ5_KInterleaved((byte*)rw.Ptr, xQ8K, y, rw.FullGroupCount, rw.TailRows, superBlockCount, _threadPool);
            else
                MatMul.ComputeRowsQ6_KInterleaved((byte*)rw.Ptr, xQ8K, y, rw.FullGroupCount, rw.TailRows, superBlockCount, _threadPool);
        }
        else
        {
            Gemv(origWeights, qt, x, y, m, k);
        }
    }

    /// <summary>
    /// GEMM using R4-interleaved repacked weights. For single-token (n=1) uses interleaved ComputeRows.
    /// Multi-token (n&gt;1) falls back to original Gemm — outer-product microkernels don't win on AVX2
    /// due to RyuJIT register pressure (12 YMM accumulators spill with only 16 registers available).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void GemmInterleaved(nint origWeights, QuantizationType qt, float* b, float* c,
                                 int m, int k, int n, byte* preQuantizedInput,
                                 in WeightRepacking.RepackedWeight rw)
    {
        if (rw.Ptr == 0 || n > 1 || rw.RowBytes < InterleavedMinRowBytes)
        {
            // Multi-token or small row stride: use original tiled GEMM path
            Gemm(origWeights, qt, b, c, m, k, n, preQuantizedInput);
            return;
        }

        // Single-token with pre-quantized input: use interleaved ComputeRows directly
        if (preQuantizedInput != null)
        {
            DispatchInterleavedComputeRows(qt, (byte*)rw.Ptr, preQuantizedInput, c,
                rw.FullGroupCount, rw.TailRows, k);
            return;
        }

        // Single-token without pre-quantized: quantize + interleaved dispatch
        GemvInterleaved(origWeights, qt, b, c, m, k, in rw);
    }

    /// <summary>
    /// Dispatches interleaved ComputeRows for a given quant type with pre-quantized input.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void DispatchInterleavedComputeRows(QuantizationType qt, byte* repackedWeights,
        byte* preQuantInput, float* result, int fullGroups, int tailRows, int k)
    {
        if (qt == QuantizationType.Q8_0)
            MatMul.ComputeRowsQ8_0Interleaved(repackedWeights, preQuantInput, result,
                fullGroups, tailRows, k / 32, _threadPool);
        else if (qt == QuantizationType.Q5_0)
            MatMul.ComputeRowsQ5_0Interleaved(repackedWeights, preQuantInput, result,
                fullGroups, tailRows, k / 32, _threadPool);
        else if (qt == QuantizationType.Q4_K)
            MatMul.ComputeRowsQ4_KInterleaved(repackedWeights, preQuantInput, result,
                fullGroups, tailRows, k / 256, _threadPool);
        else if (qt == QuantizationType.Q5_K)
            MatMul.ComputeRowsQ5_KInterleaved(repackedWeights, preQuantInput, result,
                fullGroups, tailRows, k / 256, _threadPool);
        else if (qt == QuantizationType.Q6_K)
            MatMul.ComputeRowsQ6_KInterleaved(repackedWeights, preQuantInput, result,
                fullGroups, tailRows, k / 256, _threadPool);
    }

    /// <summary>
    /// Returns true when a pre-quantized buffer produced for <paramref name="preQuantSource"/>
    /// can be safely reused for a GEMM targeting <paramref name="target"/>.
    /// K-quant types share Q8_K layout. Q8_0 and Q5_0 each use different input quantization
    /// (Q8_0 and Q8_1 respectively) and cannot share pre-quantized buffers.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsCompatiblePreQuant(QuantizationType preQuantSource, QuantizationType target)
    {
        if (preQuantSource == target) return true;

        bool sourceIsKQuant = preQuantSource is QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K;
        bool targetIsKQuant = target is QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K;
        if (sourceIsKQuant && targetIsKQuant) return true;

        // Q8_0 and Q5_0 no longer share input format — Q8_0 uses Q8_0, Q5_0 uses Q8_1.
        return false;
    }

    /// <summary>
    /// Pre-quantizes [seqLen, dim] f32 input for GEMM reuse across Q/K/V or Gate/Up projections.
    /// K-quant types (Q4_K, Q5_K, Q6_K) use Q8_K (float32 scale, 256 elements/block).
    /// Q8_0 uses Q8_0 (Half scale, 32 elements/block).
    /// Q5_0 uses Q8_1 (Half d + Half s, 32 elements/block) with precomputed block sums.
    /// Returns the scratch pointer if quantized, otherwise null.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte* QuantizeInput(float* input, byte* scratch, int dim, int seqLen,
                                       QuantizationType qt)
    {
        if (qt == QuantizationType.Q4_K || qt == QuantizationType.Q5_K || qt == QuantizationType.Q6_K)
        {
            int blockCount = dim / 256; // Q8_K_GroupSize
            int q8kRowBytes = blockCount * MatMul.Q8_K_BlockBytes;
            for (int t = 0; t < seqLen; t++)
                MatMul.QuantizeF32ToQ8_K(input + t * dim, scratch + t * q8kRowBytes, dim);
            return scratch;
        }

        if (qt == QuantizationType.Q5_0)
        {
            int blockCount = dim / Q8_1GroupSize;
            int q8_1RowBytes = blockCount * MatMul.Q8_1BlockBytes;
            for (int t = 0; t < seqLen; t++)
                MatMul.QuantizeF32ToQ8_1(input + t * dim, scratch + t * q8_1RowBytes, dim);
            return scratch;
        }

        if (qt == QuantizationType.Q8_0)
        {
            int blockCount = dim / Q8_0GroupSize;
            int q8RowBytes = blockCount * Q8_0BlockBytes;
            for (int t = 0; t < seqLen; t++)
                MatMul.QuantizeF32ToQ8_0(input + t * dim, scratch + t * q8RowBytes, dim);
            return scratch;
        }

        return null;
    }

    /// <summary>
    /// Copies or dequantizes one row of the embedding table per token into the hidden state buffer.
    /// </summary>
    private void EmbeddingLookup(ReadOnlySpan<int> tokenIds, float* hidden, int hiddenSize)
    {
        nint embPtr = _weights.TokenEmbedWeight;
        var qt = _weights.TokenEmbedQuantType;

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
                // Direct copy
                float* src = (float*)embPtr + (long)tokenId * hiddenSize;
                new ReadOnlySpan<float>(src, hiddenSize).CopyTo(destSpan);
            }
            else if (qt == QuantizationType.Q8_0)
            {
                // Dequantize one row: each row is hiddenSize elements in Q8_0 blocks
                int blocksPerRow = hiddenSize / Q8_0GroupSize;
                long rowOffset = (long)tokenId * blocksPerRow * Q8_0BlockBytes;
                nint rowPtr = embPtr + (nint)rowOffset;
                Dequantize.ToFloat32(rowPtr, hiddenSize, QuantizationType.Q8_0, destSpan);
            }
            else if (qt == QuantizationType.F16)
            {
                Half* src = (Half*)embPtr + (long)tokenId * hiddenSize;
                System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(
                    new ReadOnlySpan<Half>(src, hiddenSize), destSpan);
            }
            else
            {
                // Generic dequant fallback for any supported quant type (Q4_K, Q5_K, Q6_K, Q5_0, etc.)
                long rowBytes = Dequantize.RowByteSize(hiddenSize, qt);
                long rowOffset = (long)tokenId * rowBytes;
                nint rowPtr = embPtr + (nint)rowOffset;
                Dequantize.ToFloat32(rowPtr, hiddenSize, qt, destSpan);
            }

            // Gemma embedding scaling: multiply by sqrt(hidden_size). No-op for
            // every architecture that leaves ModelConfig.EmbeddingScale null
            // (_embeddingScale == 1.0f), so non-Gemma output is bit-identical.
            if (_embeddingScale != 1.0f)
                TensorPrimitives.Multiply(destSpan, _embeddingScale, destSpan);
        }
    }

    /// <summary>
    /// Gathers the Per-Layer Embeddings (PLE) token-identity rows into
    /// <paramref name="dest"/> <c>[seq, rowWidth]</c> (rowWidth = numLayers*pleDim),
    /// dequantizing per token and applying the ScaledWordEmbedding factor
    /// <paramref name="scale"/> = √pleDim. Mirrors <see cref="EmbeddingLookup"/>'s
    /// per-row dequant switch but for the wide per-layer table.
    /// </summary>
    private void GatherPerLayerIdentity(
        ReadOnlySpan<int> tokenIds, float* dest, int rowWidth, float scale,
        nint tablePtr, QuantizationType qt)
    {
        int pleVocab = _weights.PerLayerEmbedding!.VocabSize;
        for (int t = 0; t < tokenIds.Length; t++)
        {
            int tokenId = tokenIds[t];
            if ((uint)tokenId >= (uint)pleVocab)
                throw new ArgumentOutOfRangeException(nameof(tokenIds),
                    $"PLE token ID {tokenId} at position {t} is out of range [0, {pleVocab}).");

            var destSpan = new Span<float>(dest + t * rowWidth, rowWidth);
            if (qt == QuantizationType.F32)
            {
                float* src = (float*)tablePtr + (long)tokenId * rowWidth;
                new ReadOnlySpan<float>(src, rowWidth).CopyTo(destSpan);
            }
            else if (qt == QuantizationType.F16)
            {
                Half* src = (Half*)tablePtr + (long)tokenId * rowWidth;
                TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>(src, rowWidth), destSpan);
            }
            else
            {
                long rowBytes = Dequantize.RowByteSize(rowWidth, qt);
                nint rowPtr = tablePtr + (nint)((long)tokenId * rowBytes);
                Dequantize.ToFloat32(rowPtr, rowWidth, qt, destSpan);
            }

            TensorPrimitives.Multiply(destSpan, scale, destSpan);
        }
    }

    /// <summary>
    /// Dequantizes one token-embedding row (token id <paramref name="tokenId"/>) into
    /// <paramref name="dest"/> [hiddenSize], WITHOUT the Gemma embedding scale. Mirrors the
    /// per-row dequant in <see cref="EmbeddingLookup"/> but raw (the SC soft-embed folds
    /// the sqrt(n_embd) scale once, per canvas column, after the vocab sweep).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void DequantEmbeddingRowRaw(int tokenId, Span<float> dest, int hiddenSize)
    {
        nint embPtr = _weights.TokenEmbedWeight;
        var qt = _weights.TokenEmbedQuantType;
        if (qt == QuantizationType.F32)
        {
            float* src = (float*)embPtr + (long)tokenId * hiddenSize;
            new ReadOnlySpan<float>(src, hiddenSize).CopyTo(dest);
        }
        else if (qt == QuantizationType.F16)
        {
            Half* src = (Half*)embPtr + (long)tokenId * hiddenSize;
            TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>(src, hiddenSize), dest);
        }
        else
        {
            long rowBytes = Dequantize.RowByteSize(hiddenSize, qt);
            nint rowPtr = embPtr + (nint)((long)tokenId * rowBytes);
            Dequantize.ToFloat32(rowPtr, hiddenSize, qt, dest);
        }
    }

    /// <summary>
    /// DiffusionGemma self-conditioning (dg_canvas_embed). For each of the
    /// <paramref name="canvasLen"/> canvas rows (sequence rows [P, P+canvasLen)), adds a
    /// gated GeGLU MLP signal of the PREVIOUS step's canvas logits to the canvas embedding
    /// IN PLACE on <paramref name="hidden"/>, BEFORE the caller's weight-less rms_norm:
    /// <code>
    /// soft[c]   = sqrt(n_embd) * Σ_v softmax(prev_logits[c])[v] * tok_embd[v]
    /// normed[c] = rms_norm(soft[c]) * self_cond_pre_norm
    /// sc_sig[c] = self_cond_down( gelu_tanh(self_cond_gate·normed[c]) * (self_cond_up·normed[c]) )
    /// hidden[P+c] += sc_sig[c]
    /// </code>
    /// The soft-embed sweeps the (vocab × n_embd) table ONCE per step — each embedding row is
    /// dequantized once and scatter-accumulated into all C canvas soft-vectors weighted by that
    /// token's probability. The gate/up/down are batched as single [C × …] GEMMs. SC consumes the
    /// FULL distribution (no mask suppression). Caller guarantees <c>_weights.SelfCond</c>,
    /// <c>_scPrevLogits</c>, and <c>_scCanvasLen == canvasLen</c> are valid.
    /// </summary>
    private void ApplySelfConditioning(float* hidden, int p, int canvasLen, int hiddenSize, float eps)
    {
        var sc = _weights.SelfCond!;
        int vocab = Config.VocabSize;
        int ff = sc.GateOut;            // dense FFN width (2112)
        float embScale = _embeddingScale; // sqrt(n_embd)
        float[] prev = _scPrevLogits!;

        // soft[c] : weighted token-embedding sum per canvas column [canvasLen × hidden].
        float[] softBuf = ArrayPool<float>.Shared.Rent(canvasLen * hiddenSize);
        // probs[c] : softmax of prev_logits[c] over vocab (sc_temp_inv = 1.0).
        float[] probsBuf = ArrayPool<float>.Shared.Rent(canvasLen * vocab);
        float[] rowBuf = ArrayPool<float>.Shared.Rent(hiddenSize);
        // GeGLU MLP scratch (batched over canvasLen).
        float[] normedBuf = ArrayPool<float>.Shared.Rent(canvasLen * hiddenSize);
        float[] gateBuf = ArrayPool<float>.Shared.Rent(canvasLen * ff);
        float[] upBuf = ArrayPool<float>.Shared.Rent(canvasLen * ff);
        float[] geluBuf = ArrayPool<float>.Shared.Rent(canvasLen * ff);
        float[] sigBuf = ArrayPool<float>.Shared.Rent(canvasLen * hiddenSize);
        try
        {
            var softSpan = softBuf.AsSpan(0, canvasLen * hiddenSize);
            softSpan.Clear();

            // probs[c] = softmax(prev_logits[c]) over the full vocab (post-softcap logits,
            // sc_temp_inv = 1.0 so no rescale).
            for (int c = 0; c < canvasLen; c++)
                Softmax.Execute(
                    prev.AsSpan(c * vocab, vocab),
                    probsBuf.AsSpan(c * vocab, vocab));

            // Single vocab sweep: read each embedding row ONCE, accumulate into every
            // canvas soft-vector weighted by that token's probability. soft += prob * row.
            var rowSpan = rowBuf.AsSpan(0, hiddenSize);
            for (int v = 0; v < vocab; v++)
            {
                DequantEmbeddingRowRaw(v, rowSpan, hiddenSize);
                for (int c = 0; c < canvasLen; c++)
                {
                    float w = probsBuf[c * vocab + v];
                    if (w == 0f) continue;
                    TensorPrimitives.MultiplyAdd(
                        rowSpan, w, softSpan.Slice(c * hiddenSize, hiddenSize),
                        softSpan.Slice(c * hiddenSize, hiddenSize));
                }
            }

            // soft *= sqrt(n_embd); normed = rms_norm(soft) * self_cond_pre_norm.
            for (int c = 0; c < canvasLen; c++)
            {
                var softC = softSpan.Slice(c * hiddenSize, hiddenSize);
                if (embScale != 1.0f)
                    TensorPrimitives.Multiply(softC, embScale, softC);
                RmsNorm.Execute(
                    softC, sc.PreNorm, eps,
                    normedBuf.AsSpan(c * hiddenSize, hiddenSize));
            }

            fixed (float* normedP = normedBuf)
            fixed (float* gateP = gateBuf)
            fixed (float* upP = upBuf)
            fixed (float* sigP = sigBuf)
            {
                // g = gate·normed ; u = up·normed  (batched [canvasLen × ff]).
                Gemm(sc.GatePtr, sc.GateQt, normedP, gateP, sc.GateOut, sc.GateIn, canvasLen);
                Gemm(sc.UpPtr, sc.UpQt, normedP, upP, sc.UpOut, sc.UpIn, canvasLen);
                if (_currentAdapter is not null)
                {
                    ApplySelfConditioningLoraDelta("gate_proj", normedP, gateP, canvasLen, sc.GateIn, sc.GateOut);
                    ApplySelfConditioningLoraDelta("up_proj", normedP, upP, canvasLen, sc.UpIn, sc.UpOut);
                }

                // gelu_tanh(g) * u   (SAME GeGLU-tanh as the dense FFN).
                for (int c = 0; c < canvasLen; c++)
                    FusedOps.GeGLUTanh(
                        gateBuf.AsSpan(c * ff, ff),
                        upBuf.AsSpan(c * ff, ff),
                        geluBuf.AsSpan(c * ff, ff));

                // sc_sig = down·(gelu*u)   [canvasLen × hidden].
                fixed (float* geluP = geluBuf)
                {
                    Gemm(sc.DownPtr, sc.DownQt, geluP, sigP, sc.DownOut, sc.DownIn, canvasLen);
                    if (_currentAdapter is not null)
                        ApplySelfConditioningLoraDelta("down_proj", geluP, sigP, canvasLen, sc.DownIn, sc.DownOut);
                }

                // canvas += sc_sig (added to the scaled embeddings; caller rms_noscales after).
                for (int c = 0; c < canvasLen; c++)
                {
                    var dst = new Span<float>(hidden + (p + c) * hiddenSize, hiddenSize);
                    TensorPrimitives.Add(
                        dst, sigBuf.AsSpan(c * hiddenSize, hiddenSize), dst);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(softBuf);
            ArrayPool<float>.Shared.Return(probsBuf);
            ArrayPool<float>.Shared.Return(rowBuf);
            ArrayPool<float>.Shared.Return(normedBuf);
            ArrayPool<float>.Shared.Return(gateBuf);
            ArrayPool<float>.Shared.Return(upBuf);
            ArrayPool<float>.Shared.Return(geluBuf);
            ArrayPool<float>.Shared.Return(sigBuf);
        }
    }

    /// <summary>
    /// Fused Q/K/V decode: dispatches all three projections in a single pool.Dispatch() call
    /// when they share the same quant family, saving 2 dispatch overheads per layer.
    /// Cross-family projections dispatch individually with self-quantizing GEMV.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void FusedQkvDecode(ref readonly TransformerLayerWeights lw,
        float* normOut, byte* preQuantNorm, float* q, float* k, float* v)
    {
        // preQuantNorm was quantized for lw.QQuantType's family.
        // FusedDecodeGemv3 handles cross-family by dispatching those projections
        // individually with self-quantizing GEMV (null preQuant).
        MatMul.FusedDecodeGemv3(
            (byte*)lw.QWeight, lw.QQuantType, q, lw.QOutputDim,
            (byte*)lw.KWeight, lw.KQuantType, k, lw.KOutputDim,
            (byte*)lw.VWeight, lw.VQuantType, v, lw.VOutputDim,
            normOut, preQuantNorm, lw.QInputDim,
            _threadPool!);
    }

    /// <summary>
    /// Fused Gate/Up decode: dispatches both projections in a single pool.Dispatch() call
    /// when they share the same quant family, saving 1 dispatch overhead per layer.
    /// Cross-family projections dispatch individually with self-quantizing GEMV.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void FusedGateUpDecode(ref readonly TransformerLayerWeights lw,
        float* normOut, byte* preQuantFfn, float* ffnGate, float* ffnUp)
    {
        // preQuantFfn was quantized for lw.GateQuantType's family.
        // FusedDecodeGemv2 handles cross-family by dispatching separately.
        MatMul.FusedDecodeGemv2(
            (byte*)lw.GateWeight, lw.GateQuantType, ffnGate, lw.GateOutputDim,
            (byte*)lw.UpWeight, lw.UpQuantType, ffnUp, lw.UpOutputDim,
            normOut, preQuantFfn, lw.GateInputDim,
            _threadPool!);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_ownsThreadPool)
            _threadPool?.Dispose();
        _state.Dispose();
        _mlaKvState?.Dispose();
        _mlaLatentKvState?.Dispose();
        _weights.Dispose(); // free R4-interleaved weight buffers and any owned bf16→F32 scratch
        // _mmapAnchor is not owned by us — caller disposes the GgufFile / SafetensorsFile.
    }
}
