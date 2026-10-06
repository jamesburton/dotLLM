using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Server;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// #760: default-flag CPU serve of a hybrid model (Bonsai-2-27B: 64 layers, 16 KV layers, 262144 ctx) died at startup because
/// the generic paged pool is sized NumLayers x MaxSequenceLength. Hybrid/recurrent configs must skip it; dense ones keep it.
/// </summary>
public class CpuHybridPagedKvDecisionTests
{
    private static ModelConfig Bonsai(GatedDeltaNetConfig? gdn) => new()
    {
        Architecture = Architecture.Qwen3HybridDense,
        VocabSize = 248320, HiddenSize = 5120, IntermediateSize = 17408,
        NumLayers = 64, NumAttentionHeads = 24, NumKvHeads = 4, HeadDim = 256,
        MaxSequenceLength = 262144,
        GdnConfig = gdn,
    };

    [Fact]
    public void Bonsai27B_GdnConfig_SkipsPagedPool()
        => Assert.True(ServerStartup.IsHybridOrRecurrentCpuModel(Bonsai(new GatedDeltaNetConfig(4, 48, 16, 128, 6144, 4)), null));

    [Fact]
    public void DenseConfig_KeepsPagedPool()
        => Assert.False(ServerStartup.IsHybridOrRecurrentCpuModel(Bonsai(null), null));
}
