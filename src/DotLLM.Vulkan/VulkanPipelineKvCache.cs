using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;

namespace DotLLM.Vulkan;

/// <summary>
/// Composite KV-cache for a two-device Vulkan pipeline (<see cref="VulkanPipelineTransformerModel"/>):
/// layers <c>[0..SplitLayer)</c> live in <see cref="Stage0"/> (on the first device) and layers
/// <c>[SplitLayer..L)</c> in <see cref="Stage1"/> (on the second device). The pipeline forward drives the
/// two stage caches directly (each stage attends/updates its own layers with window-local indices); the
/// <see cref="IKvCache"/> surface here routes any external (scheduler / prefix-cache) access to the right
/// stage, remapping the global layer index to the stage's 0-based local index.
/// </summary>
public sealed class VulkanPipelineKvCache : IKvCache
{
    private readonly int _splitLayer;

    /// <summary>KV-cache for layers <c>[0..SplitLayer)</c> on the first pipeline device.</summary>
    internal VulkanKvCache Stage0 { get; }

    /// <summary>KV-cache for layers <c>[SplitLayer..L)</c> on the second pipeline device (local-indexed).</summary>
    internal VulkanKvCache Stage1 { get; }

    /// <summary>Global layer index at which the second pipeline stage begins.</summary>
    public int SplitLayer => _splitLayer;

    /// <summary>Creates the composite cache from the two per-stage device caches.</summary>
    public VulkanPipelineKvCache(VulkanKvCache stage0, VulkanKvCache stage1, int splitLayer)
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

    private VulkanKvCache Route(int globalLayer) => globalLayer < _splitLayer ? Stage0 : Stage1;
    private int Local(int globalLayer) => globalLayer < _splitLayer ? globalLayer : globalLayer - _splitLayer;
}
