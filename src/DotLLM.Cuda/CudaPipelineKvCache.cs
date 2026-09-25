using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;

namespace DotLLM.Cuda;

/// <summary>
/// Composite KV-cache for a two-device CUDA pipeline (<see cref="CudaPipelineTransformerModel"/>):
/// layers <c>[0..SplitLayer)</c> live in <see cref="Stage0"/> (on the first CUDA device) and layers
/// <c>[SplitLayer..L)</c> in <see cref="Stage1"/> (on the second CUDA device). The pipeline forward drives
/// the two stage caches directly (each stage attends/updates its own layers with window-local indices via
/// <see cref="CudaKvCache.UpdateDevice"/>); the <see cref="IKvCache"/> surface here routes any external
/// (scheduler / prefix-cache) access to the right stage, remapping the global layer index to the stage's
/// 0-based local index. This is the CUDA→CUDA analogue of <c>VulkanPipelineKvCache</c> and the
/// two-context sibling of <see cref="HybridVulkanCudaKvCache"/> (Vulkan→CUDA).
/// </summary>
public sealed class CudaPipelineKvCache : IKvCache
{
    private readonly int _splitLayer;

    /// <summary>KV-cache for layers <c>[0..SplitLayer)</c> on the first pipeline device (context 0).</summary>
    internal CudaKvCache Stage0 { get; }

    /// <summary>KV-cache for layers <c>[SplitLayer..L)</c> on the second pipeline device (local-indexed, context 1).</summary>
    internal CudaKvCache Stage1 { get; }

    /// <summary>Global layer index at which the second pipeline stage begins.</summary>
    public int SplitLayer => _splitLayer;

    /// <summary>Creates the composite cache from the two per-stage device caches.</summary>
    public CudaPipelineKvCache(CudaKvCache stage0, CudaKvCache stage1, int splitLayer)
    {
        ArgumentNullException.ThrowIfNull(stage0);
        ArgumentNullException.ThrowIfNull(stage1);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(splitLayer);
        Stage0 = stage0;
        Stage1 = stage1;
        _splitLayer = splitLayer;
    }

    /// <inheritdoc/>
    public int CurrentLength => Stage1.CurrentLength; // both stages advance in lockstep per forward

    /// <inheritdoc/>
    public int MaxLength => Stage0.MaxLength;

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
        => Route(layerIndex).Update(keys, values, positions, Local(layerIndex));

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
        => Route(layerIndex).Update(keys, values, positions, Local(layerIndex));

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex) => Route(layerIndex).GetKeys(Local(layerIndex));

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex) => Route(layerIndex).GetValues(Local(layerIndex));

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex) => Route(layerIndex).GetKeysRef(Local(layerIndex));

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex) => Route(layerIndex).GetValuesRef(Local(layerIndex));

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        Stage0.Rollback(length);
        Stage1.Rollback(length);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        Stage0.Dispose();
        Stage1.Dispose();
    }

    private CudaKvCache Route(int globalLayer) => globalLayer < _splitLayer ? Stage0 : Stage1;
    private int Local(int globalLayer) => globalLayer < _splitLayer ? globalLayer : globalLayer - _splitLayer;
}
