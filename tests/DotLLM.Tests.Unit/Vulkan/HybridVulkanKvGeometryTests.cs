using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #839: the Vulkan hybrid KV cache (<see cref="VulkanNemotronHKvCache"/>) already allocates attention layers only; the shared
/// <see cref="KvGeometry"/> slot map must agree with the slot map the Vulkan models build, so CPU and Vulkan hybrids size their
/// caches identically.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class HybridVulkanKvGeometryTests
{
    [SkippableFact]
    public void VulkanHybridCache_SlotCount_MatchesSharedGeometry()
    {
        Skip.If(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN"), "1", StringComparison.Ordinal), "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/device.");

        var A = HybridLayerKind.Attention; var G = HybridLayerKind.GatedDeltaNet;
        HybridLayerKind[] kinds = [G, G, A, G, G, G, A, A, G];   // irregular: slots 0,1,2 at layers 2,6,7
        var cfg = new ModelConfig
        {
            Architecture = Architecture.NemotronH, VocabSize = 8, HiddenSize = 16, IntermediateSize = 16, NumLayers = kinds.Length,
            NumAttentionHeads = 4, NumKvHeads = 2, HeadDim = 8, MaxSequenceLength = 32,
            AttentionType = AttentionType.GQA, PositionEncodingType = PositionEncodingType.None,
            ActivationFunction = ActivationFunction.SiLU, NormType = NormType.RMSNorm, NormEpsilon = 1e-5f,
            HybridLayout = new HybridLayerLayout { LayerKind = kinds, HeadCountKv = new int[kinds.Length], FeedForwardLength = new int[kinds.Length] },
        };
        var geom = KvGeometry.FromConfig(cfg);
        // The convention every hybrid model (CPU/Vulkan/CUDA) builds: slot = running attention count, -1 elsewhere.
        var map = new int[kinds.Length];
        int n = 0;
        for (int i = 0; i < kinds.Length; i++) map[i] = kinds[i] == A ? n++ : -1;
        for (int i = 0; i < kinds.Length; i++) Assert.Equal(map[i], geom.SlotOfLayer(i));

        using var device = VulkanDevice.Create();
        using var cache = new VulkanNemotronHKvCache(device, map, geom.LayerCount, cfg.NumKvHeads, cfg.HeadDim, 32);
        Assert.Equal(3, cache.AttentionLayerCount);
        Assert.Equal(geom.LayerCount, cache.AttentionLayerCount);
    }
}
