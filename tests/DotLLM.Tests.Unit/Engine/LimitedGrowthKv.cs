using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// A KV whose growth an admission refuses above <paramref name="limit"/> positions, like the CUDA hybrid's handle when the
/// added pages would not fit in free VRAM (issue #1093). Below the limit it grows like <see cref="SimpleKvCache"/>, rows kept.
/// </summary>
/// <param name="inner">The storage.</param>
/// <param name="limit">The largest <see cref="IKvCache.MaxLength"/> a growth may reach.</param>
internal sealed class LimitedGrowthKv(SimpleKvCache inner, int limit) : IKvCache
{
    /// <summary>Every length a growth was asked for, in order, refused or not.</summary>
    public List<int> GrowthRequests { get; } = [];

    public int CurrentLength => inner.CurrentLength;
    public int MaxLength => inner.MaxLength;
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex) => inner.Update(keys, values, positions, layerIndex);
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex) => inner.Update(keys, values, positions, layerIndex);
    public ITensor GetKeys(int layerIndex) => inner.GetKeys(layerIndex);
    public ITensor GetValues(int layerIndex) => inner.GetValues(layerIndex);
    public TensorRef GetKeysRef(int layerIndex) => inner.GetKeysRef(layerIndex);
    public TensorRef GetValuesRef(int layerIndex) => inner.GetValuesRef(layerIndex);
    public void Rollback(int length) => inner.Rollback(length);
    public void Dispose() => inner.Dispose();

    public bool TryGrowTo(int maxLength)
    {
        if (maxLength <= inner.MaxLength) return true;
        GrowthRequests.Add(maxLength);
        return maxLength <= limit && inner.TryGrowTo(maxLength);
    }
}
