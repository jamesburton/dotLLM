using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

namespace DotLLM.Vulkan;

/// <summary>
/// Two-device Vulkan pipeline-parallel transformer (layer-spanning): layers <c>[0..SplitLayer)</c> run on
/// the first Vulkan device and layers <c>[SplitLayer..L)</c> plus the final norm + LM head on the second.
/// The hidden state crosses the boundary device0 → host (FP32) → device1, reusing
/// <see cref="VulkanTransformerModel.DownloadHiddenState"/> and the resume-from-hidden
/// <c>ForwardFromHidden</c> entry — so a model can span more VRAM than any single device holds.
/// </summary>
/// <remarks>
/// <para>
/// The two stages are independent <see cref="VulkanTransformerModel"/> instances; stage 1 is built with a
/// <c>firstLayer</c> layer-window so it holds only its slice of device weights. Both stages share one
/// host-side <see cref="TransformerWeights"/> (its <c>Dispose</c> is idempotent, so disposing both stages
/// is safe). This is the Vulkan→Vulkan analogue of <c>HybridVulkanCudaTransformerModel</c> (Vulkan→CUDA).
/// </para>
/// <para>
/// M-scope: standard causal forward (no diffusion). <c>ForwardBatch</c> uses the default per-request
/// loop — micro-batched pipeline overlap (hiding the handoff latency) is follow-up work.
/// </para>
/// </remarks>
public sealed unsafe class VulkanPipelineTransformerModel : IModel
{
    private readonly VulkanTransformerModel _stage0; // layers [0..SplitLayer)
    private readonly VulkanTransformerModel _stage1; // layers [SplitLayer..L) + final norm + LM head
    private readonly int _splitLayer;
    // The shared host TransformerWeights is kept rooted (and disposed) by the two stage models, so this
    // class holds no reference to it. _gguf is owned only by the LoadFromGguf path.
    private readonly GgufFile? _gguf;
    private readonly VulkanDevice? _ownedDevice0;
    private readonly VulkanDevice? _ownedDevice1;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _stage0.ComputeMemoryBytes + _stage1.ComputeMemoryBytes;

    /// <summary>Global layer index at which the second pipeline stage begins.</summary>
    public int SplitLayer => _splitLayer;

    private VulkanPipelineTransformerModel(
        ModelConfig config, VulkanTransformerModel stage0, VulkanTransformerModel stage1, int splitLayer,
        GgufFile? gguf, VulkanDevice? ownedDevice0, VulkanDevice? ownedDevice1)
    {
        Config = config;
        _stage0 = stage0;
        _stage1 = stage1;
        _splitLayer = splitLayer;
        _gguf = gguf;
        _ownedDevice0 = ownedDevice0;
        _ownedDevice1 = ownedDevice1;
    }

    /// <summary>
    /// Builds a pipeline model over two caller-owned devices from pre-built CPU weights (the device weights
    /// are uploaded per-stage as the windows <c>[0..splitLayer)</c> and <c>[splitLayer..L)</c>). Neither the
    /// devices nor <paramref name="cpuWeights"/> are disposed by this model — the caller owns them.
    /// </summary>
    internal static VulkanPipelineTransformerModel BuildFromPrebuiltWeights(
        VulkanDevice device0, VulkanDevice device1, ModelConfig config,
        TransformerWeights cpuWeights, int splitLayer, string spvDir)
    {
        ArgumentNullException.ThrowIfNull(device0);
        ArgumentNullException.ThrowIfNull(device1);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(cpuWeights);
        ArgumentNullException.ThrowIfNull(spvDir);
        ValidateSplit(splitLayer, config.NumLayers);

        var (stage0, stage1) = BuildStages(device0, device1, config, cpuWeights, splitLayer, spvDir);
        return new VulkanPipelineTransformerModel(
            config, stage0, stage1, splitLayer, gguf: null, ownedDevice0: null, ownedDevice1: null);
    }

    /// <summary>
    /// Loads a two-device pipeline model from an opened GGUF, binding stage 0 to physical Vulkan device
    /// <paramref name="device0Index"/> and stage 1 to <paramref name="device1Index"/>. This model owns the
    /// two devices (disposed with it); the caller owns <paramref name="gguf"/> only until this returns —
    /// thereafter the model holds it alive for the weights' mmap and disposes it.
    /// </summary>
    public static VulkanPipelineTransformerModel LoadFromGguf(
        GgufFile gguf, ModelConfig config, int splitLayer,
        int device0Index, int device1Index, string? spvDir = null)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);
        ValidateSplit(splitLayer, config.NumLayers);
        spvDir ??= Path.Combine(AppContext.BaseDirectory, "spv");

        VulkanDevice? device0 = null, device1 = null;
        TransformerWeights? cpuWeights = null;
        try
        {
            device0 = VulkanDevice.Create(device0Index);
            device1 = VulkanDevice.Create(device1Index);
            cpuWeights = TransformerWeights.LoadFromGguf(gguf, config); // shared host weights for both stages
            var (stage0, stage1) = BuildStages(device0, device1, config, cpuWeights, splitLayer, spvDir);
            return new VulkanPipelineTransformerModel(
                config, stage0, stage1, splitLayer, gguf, device0, device1);
        }
        catch
        {
            cpuWeights?.Dispose();
            device1?.Dispose();
            device0?.Dispose();
            throw;
        }
    }

    private static (VulkanTransformerModel Stage0, VulkanTransformerModel Stage1) BuildStages(
        VulkanDevice device0, VulkanDevice device1, ModelConfig config,
        TransformerWeights cpuWeights, int splitLayer, string spvDir)
    {
        // Each stage is a self-consistent model whose NumLayers is its window size; device weights are
        // uploaded for [0..splitLayer) and [splitLayer..L) respectively. Stage 1 carries firstLayer so its
        // CPU-side per-layer lookups index the right global layer (see VulkanTransformerModel._firstLayer).
        var cfg0 = config with { NumLayers = splitLayer };
        var cfg1 = config with { NumLayers = config.NumLayers - splitLayer };
        VulkanTransformerModel? stage0 = null;
        try
        {
            stage0 = VulkanTransformerModel.BuildFromPrebuiltWeights(device0, cfg0, cpuWeights, spvDir, firstLayer: 0);
            var stage1 = VulkanTransformerModel.BuildFromPrebuiltWeights(device1, cfg1, cpuWeights, spvDir, firstLayer: splitLayer);
            return (stage0, stage1);
        }
        catch
        {
            stage0?.Dispose();
            throw;
        }
    }

    private static void ValidateSplit(int splitLayer, int numLayers)
    {
        if (splitLayer <= 0 || splitLayer >= numLayers)
            throw new ArgumentOutOfRangeException(nameof(splitLayer),
                $"splitLayer must be between 1 and {numLayers - 1}; use VulkanTransformerModel for a single device.");
    }

    /// <summary>Creates a composite KV-cache: stage-0 layers on device 0, stage-1 layers on device 1.</summary>
    public IKvCache CreateKvCache(int maxSeqLen)
    {
        VulkanKvCache? kv0 = null;
        try
        {
            kv0 = _stage0.CreateKvCache(maxSeqLen);
            var kv1 = _stage1.CreateKvCache(maxSeqLen);
            return new VulkanPipelineKvCache(kv0, kv1, _splitLayer);
        }
        catch
        {
            kv0?.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <inheritdoc/>
    /// <remarks>
    /// Stage 0 (embedding + layers <c>[0..SplitLayer)</c>) runs on device 0; its hidden state is downloaded
    /// to host FP32 and fed to stage 1 (layers <c>[SplitLayer..L)</c> + final norm + LM head) on device 1
    /// via <c>ForwardFromHidden</c>. A <see cref="VulkanPipelineKvCache"/> supplies each stage's device-local
    /// KV-cache; pass null for a cacheless prefill.
    /// </remarks>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
    {
        if (tokenIds.Length != positions.Length)
            throw new ArgumentException("tokenIds and positions must have the same length.");
        var pipeline = kvCache as VulkanPipelineKvCache;
        if (kvCache is not null && pipeline is null)
            throw new ArgumentException(
                $"Expected a {nameof(VulkanPipelineKvCache)} from {nameof(CreateKvCache)}, got {kvCache.GetType().Name}.",
                nameof(kvCache));

        int seqLen = tokenIds.Length;

        // Stage 0 on device 0: embedding + layers [0..SplitLayer). Discard its (head-less here) logits; the
        // hidden state is what crosses the boundary.
        using (var _ = _stage0.Forward(tokenIds, positions, deviceId: 0, pipeline?.Stage0)) { }
        using ITensor hidden = _stage0.DownloadHiddenState(seqLen); // device0 → host FP32

        // Stage 1 on device 1: resume from the host hidden rows, run layers [SplitLayer..L) + norm + LM head.
        var hiddenRows = new ReadOnlySpan<float>((void*)hidden.DataPointer, seqLen * Config.HiddenSize);
        return _stage1.ForwardFromHidden(hiddenRows, positions, pipeline?.Stage1); // host → device1
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        // Each stage's Dispose disposes the shared _cpuWeights; TransformerWeights.Dispose is idempotent,
        // so the second call is a no-op (and any caller that owns the weights in the prebuilt path can
        // still dispose them safely).
        _stage1.Dispose();
        _stage0.Dispose();
        _ownedDevice1?.Dispose();
        _ownedDevice0?.Dispose();
        _gguf?.Dispose();
    }
}
