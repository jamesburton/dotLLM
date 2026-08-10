using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Threading;
using DotLLM.Cuda.Interop;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// CPU/GPU partial-offload split for the Qwen3HybridDense (<c>qwen35</c>) architecture — issue
/// #291. Composes a GPU-resident layer prefix (<see cref="CudaQwen3HybridDenseTransformerModel.LoadHeadFromGguf"/>,
/// layers <c>[0, numGpuLayers)</c>) with a CPU-resident tail
/// (<see cref="Qwen3HybridDenseTransformerModel.LoadTailFromGguf"/>, layers
/// <c>[numGpuLayers, NumLayers)</c>), D2H-transferring the boundary hidden state between them —
/// the architecture-aware counterpart to <see cref="HybridTransformerModel"/> (which only knows
/// the uniform Llama-style layer shape every layer in THAT model's supported architectures shares).
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this class exists (issue #291).</b> <see cref="HybridTransformerModel"/>'s partial-offload
/// splitter always loads weights via the generic, Llama-style <c>TransformerWeights.LoadFromGguf</c>,
/// which assumes every layer has a uniform <c>attn_output.weight</c> tensor. Qwen3HybridDense
/// interleaves Gated-DeltaNet (GDN) layers — which have NO attention output projection at all —
/// with full-attention layers, gated by <see cref="GatedDeltaNetConfig.FullAttnInterval"/>. Loading
/// such a checkpoint through the generic splitter throws <c>KeyNotFoundException: 'blk.0.attn_output.weight'</c>
/// the moment it reaches a GDN layer. This class instead reuses the SAME per-layer GDN-vs-attention
/// tensor-name resolution the already-correct CPU-only and full-GPU-offload paths for this
/// architecture use (<c>Qwen3HybridDenseTransformerModel.LoadLayer</c> /
/// <c>CudaQwen3HybridDenseTransformerModel.LoadLayerDevice</c>), sliced by layer range instead of
/// duplicated.
/// </para>
/// <para>
/// <b>VRAM.</b> The GPU head deliberately skips uploading the embedding table, lm_head, and
/// output-norm to device (see <see cref="CudaQwen3HybridDenseTransformerModel.LoadHeadFromGguf"/>'s
/// remarks) — those live only on the CPU tail, which already needs them for its own final norm +
/// lm_head. VRAM scales with <c>numGpuLayers</c> alone (each layer's own GDN/attention/FFN
/// weights), not with a fixed "always upload the output stage" tax — the actual saving a smaller
/// <c>--gpu-layers</c> count is supposed to deliver.
/// </para>
/// <para>
/// <b>Not yet supported through this split:</b> LoRA adapters and Multi-Token Prediction (MTP) —
/// both out of scope for #291. A model with an MTP head loads fine (the tail simply never resolves
/// the trailing MTP block), but <see cref="IModel.SupportsMtp"/> reports <see langword="false"/>
/// for the split composition regardless of the checkpoint.
/// </para>
/// </remarks>
public sealed class HybridQwen3HybridDenseTransformerModel : IModel
{
    private readonly CudaQwen3HybridDenseTransformerModel _headModel;
    private readonly Qwen3HybridDenseTransformerModel _tailModel;
    private readonly int _numGpuLayers;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _headModel.ComputeMemoryBytes + _tailModel.ComputeMemoryBytes;

    /// <summary>Number of transformer layers running on GPU.</summary>
    public int NumGpuLayers => _numGpuLayers;

    /// <summary>Non-null when GPU-side weights exceed available VRAM.</summary>
    public string? VramWarning { get; }

    /// <inheritdoc/>
    public bool RequiresPerSequenceState => true;

    private HybridQwen3HybridDenseTransformerModel(
        ModelConfig config, CudaQwen3HybridDenseTransformerModel headModel,
        Qwen3HybridDenseTransformerModel tailModel, int numGpuLayers, string? vramWarning)
    {
        Config = config;
        _headModel = headModel;
        _tailModel = tailModel;
        _numGpuLayers = numGpuLayers;
        VramWarning = vramWarning;
    }

    /// <summary>
    /// Loads a Qwen3HybridDense model split between GPU (first <paramref name="numGpuLayers"/>
    /// layers) and CPU (the remainder).
    /// </summary>
    /// <param name="gguf">Opened GGUF file (must remain alive for model lifetime).</param>
    /// <param name="config">Model configuration extracted from GGUF metadata.</param>
    /// <param name="numGpuLayers">Number of layers to run on GPU (must be &gt; 0 and &lt; config.NumLayers).</param>
    /// <param name="deviceId">GPU device ordinal (0-based).</param>
    /// <param name="threading">CPU threading configuration for the CPU-side tail.</param>
    public static HybridQwen3HybridDenseTransformerModel LoadFromGguf(
        GgufFile gguf, ModelConfig config, int numGpuLayers, int deviceId, ThreadingConfig threading)
    {
        if (config.Architecture != Architecture.Qwen3HybridDense)
            throw new ArgumentException(
                $"HybridQwen3HybridDenseTransformerModel requires Architecture.Qwen3HybridDense, got {config.Architecture}.",
                nameof(config));
        if (numGpuLayers <= 0 || numGpuLayers >= config.NumLayers)
            throw new ArgumentOutOfRangeException(nameof(numGpuLayers),
                $"numGpuLayers must be between 1 and {config.NumLayers - 1} for hybrid mode. " +
                $"Use Qwen3HybridDenseTransformerModel for pure CPU or CudaQwen3HybridDenseTransformerModel for pure GPU.");

        var headModel = CudaQwen3HybridDenseTransformerModel.LoadHeadFromGguf(gguf, config, numGpuLayers, deviceId);
        var tailModel = Qwen3HybridDenseTransformerModel.LoadTailFromGguf(gguf, config, numGpuLayers, threading);

        // VRAM estimation and warning — mirrors HybridTransformerModel.LoadFromGguf's identical check.
        string? vramWarning = null;
        if (CudaDriverApi.cuMemGetInfo_v2(out nuint freeAfter, out nuint totalVram) == 0
            && totalVram > 0)
        {
            double freePercent = (double)freeAfter / totalVram;
            if (freePercent < 0.10)
            {
                long freeMb = (long)freeAfter / (1024 * 1024);
                long totalMb = (long)totalVram / (1024 * 1024);
                vramWarning = $"VRAM nearly full after loading {numGpuLayers}/{config.NumLayers} layers " +
                              $"({freeMb}/{totalMb} MB free). Consider reducing --gpu-layers.";
            }
        }

        return new HybridQwen3HybridDenseTransformerModel(config, headModel, tailModel, numGpuLayers, vramWarning);
    }

    /// <summary>Creates a split KV-cache: GPU-internal storage for the head, host storage for the tail.</summary>
    public Qwen3HybridDenseSplitKvCache CreateKvCache(int maxSeqLen)
    {
        var gpuHandle = _headModel.CreateKvCache(maxSeqLen);
        // CPU tail's own (sliced) Config.NumLayers/NumKvHeads/HeadDim already reflect just the
        // tail's local layer range — local layer index i's KV slot (kvSlotForLayer[i], computed by
        // LoadTailFromGguf) indexes into a cache sized for the tail alone, restarting at 0. Some
        // slots go unused for the tail's own GDN layers (kvSlotForLayer[i] == -1, never touched) —
        // harmless, matches how the plain CPU-only Qwen3HybridDenseTransformerModel path already
        // sizes its default KV-cache (TextGenerator.AllocateKvCache) the same way.
        var cpuCache = new SimpleKvCache(_tailModel.Config.NumLayers, _tailModel.Config.NumKvHeads,
            _tailModel.Config.HeadDim, maxSeqLen);
        return new Qwen3HybridDenseSplitKvCache(gpuHandle, cpuCache);
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <inheritdoc/>
    /// <remarks>
    /// Runs the GPU head (embedding + GPU-resident layers, <see cref="CudaQwen3HybridDenseTransformerModel.ForwardHead"/>),
    /// D2H-transfers the resulting <c>[seqLen, hiddenSize]</c> F32 boundary hidden state, then feeds
    /// it into the CPU tail (<see cref="Qwen3HybridDenseTransformerModel.ForwardFromHiddenState"/>),
    /// which runs its own layers plus the final norm + lm_head. Both halves use their own
    /// model-owned default GDN state when <paramref name="kvCache"/> carries no explicit per-sequence
    /// GDN container — same single-sequence convention <see cref="Qwen3HybridDenseTransformerModel"/>'s
    /// own uncached <c>Forward</c> overloads use; call <see cref="ResetSequenceState"/> between
    /// independent sequences (e.g. perplexity windows) to avoid state leaking across them.
    /// </remarks>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache)
    {
        var split = kvCache as Qwen3HybridDenseSplitKvCache;
        if (kvCache is not null && split is null)
            throw new ArgumentException(
                $"HybridQwen3HybridDenseTransformerModel requires a Qwen3HybridDenseSplitKvCache (from CreateKvCache); got {kvCache.GetType().Name}.",
                nameof(kvCache));

        ITensor boundaryHidden = _headModel.ForwardHead(tokenIds, positions, split?.GpuHandle);
        try
        {
            unsafe
            {
                var hiddenSpan = new ReadOnlySpan<float>((void*)boundaryHidden.DataPointer,
                    (int)boundaryHidden.ElementCount);
                return _tailModel.ForwardFromHiddenState(hiddenSpan, positions, deviceId, split?.CpuCache, gdnState: null);
            }
        }
        finally
        {
            boundaryHidden.Dispose();
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Re-zeroes both halves' model-owned GDN state (see <see cref="Qwen3HybridDenseTransformerModel.ResetSequenceState"/>
    /// / <see cref="CudaQwen3HybridDenseTransformerModel.ResetSequenceState"/>) — required for any
    /// caller (e.g. perplexity scoring) that treats each uncached <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int)"/>
    /// call as an independent sequence.
    /// </remarks>
    public void ResetSequenceState()
    {
        _headModel.ResetSequenceState();
        _tailModel.ResetSequenceState();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _headModel.Dispose();
        _tailModel.Dispose();
    }
}

/// <summary>
/// Split KV-cache for <see cref="HybridQwen3HybridDenseTransformerModel"/>: GPU-internal storage
/// (length-only handle — actual K/V lives inside <see cref="CudaQwen3HybridDenseTransformerModel"/>)
/// for the GPU head's attention layers, host <see cref="SimpleKvCache"/> for the CPU tail's.
/// Neither half is reachable through the top-level <see cref="IKvCache"/> surface — the composition
/// model's <c>Forward</c> routes each half directly to its own sub-model, which is the only code
/// that ever calls <see cref="GpuHandle"/>/<see cref="CpuCache"/> — so every storage-bearing member
/// here throws, mirroring <see cref="CudaHybridKvCacheHandle"/>'s own convention.
/// </summary>
public sealed class Qwen3HybridDenseSplitKvCache : IKvCache
{
    /// <summary>GPU head's length-only KV-cache handle (actual F16 storage owned by the head model).</summary>
    internal CudaHybridKvCacheHandle GpuHandle { get; }

    /// <summary>CPU tail's host-resident KV-cache.</summary>
    internal SimpleKvCache CpuCache { get; }

    /// <summary>Creates a split KV-cache from an already-sized GPU handle and CPU cache.</summary>
    public Qwen3HybridDenseSplitKvCache(CudaHybridKvCacheHandle gpuHandle, SimpleKvCache cpuCache)
    {
        GpuHandle = gpuHandle;
        CpuCache = cpuCache;
    }

    /// <inheritdoc/>
    public int CurrentLength => CpuCache.CurrentLength;

    /// <inheritdoc/>
    public int MaxLength => CpuCache.MaxLength;

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException(
            "Qwen3HybridDenseSplitKvCache routes storage through the composition model's own head/tail calls, not IKvCache.Update().");

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException(
            "Qwen3HybridDenseSplitKvCache routes storage through the composition model's own head/tail calls, not IKvCache.Update().");

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
        => throw new NotSupportedException(
            "Qwen3HybridDenseSplitKvCache routes storage through the composition model's own head/tail calls, not IKvCache.GetKeys().");

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
        => throw new NotSupportedException(
            "Qwen3HybridDenseSplitKvCache routes storage through the composition model's own head/tail calls, not IKvCache.GetValues().");

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex)
        => throw new NotSupportedException(
            "Qwen3HybridDenseSplitKvCache routes storage through the composition model's own head/tail calls, not IKvCache.GetKeysRef().");

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex)
        => throw new NotSupportedException(
            "Qwen3HybridDenseSplitKvCache routes storage through the composition model's own head/tail calls, not IKvCache.GetValuesRef().");

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        GpuHandle.Rollback(length);
        CpuCache.Rollback(length);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        GpuHandle.Dispose();
        CpuCache.Dispose();
    }
}
