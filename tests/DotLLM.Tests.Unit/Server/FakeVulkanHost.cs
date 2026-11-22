namespace DotLLM.Vulkan.TestDoubles;

/// <summary>A model that reports itself as a Vulkan host (by namespace) with a chosen in-place TurboQuant capability, no device needed.</summary>
internal sealed class FakeVulkanHost(DotLLM.Core.Models.ModelConfig config, bool supportsInPlace) : DotLLM.Core.Models.IModel
{
    public DotLLM.Core.Models.ModelConfig Config { get; } = config;
    public long ComputeMemoryBytes => 0;
    public bool SupportsInPlaceTurboQuant(DotLLM.Core.Configuration.KvCacheConfig kv) => supportsInPlace;
    public DotLLM.Core.Tensors.ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId) => throw new NotSupportedException();
    public DotLLM.Core.Tensors.ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, DotLLM.Core.Attention.IKvCache? kvCache) => throw new NotSupportedException();
    public void Dispose() { }
}
