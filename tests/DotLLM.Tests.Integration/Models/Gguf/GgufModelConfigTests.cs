using DotLLM.Core.Configuration;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Gguf;

/// <summary>
/// Tests <see cref="GgufModelConfigExtractor"/> against a real GGUF model (SmolLM-135M, llama architecture).
/// Auto-downloads on first run, cached in <c>~/.dotllm/models/</c>.
/// </summary>
[Collection("SmallModel")]
public class GgufModelConfigTests
{
    private readonly SmallModelFixture _fixture;

    public GgufModelConfigTests(SmallModelFixture fixture)
    {
        _fixture = fixture;
    }

    [Fact]
    public void Extract_ProducesValidConfig()
    {
        using var gguf = GgufFile.Open(_fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.True(config.VocabSize > 0, "VocabSize should be positive.");
        Assert.True(config.HiddenSize > 0, "HiddenSize should be positive.");
        Assert.True(config.IntermediateSize > 0, "IntermediateSize should be positive.");
        Assert.True(config.NumLayers > 0, "NumLayers should be positive.");
        Assert.True(config.NumAttentionHeads > 0, "NumAttentionHeads should be positive.");
        Assert.True(config.NumKvHeads > 0, "NumKvHeads should be positive.");
        Assert.True(config.HeadDim > 0, "HeadDim should be positive.");
        Assert.True(config.MaxSequenceLength > 0, "MaxSequenceLength should be positive.");
    }

    [Fact]
    public void Extract_HeadDimMatchesHiddenSizeDividedByHeads()
    {
        using var gguf = GgufFile.Open(_fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.Equal(config.HiddenSize / config.NumAttentionHeads, config.HeadDim);
    }

    [Fact]
    public void Extract_NumKvHeadsDoesNotExceedAttentionHeads()
    {
        using var gguf = GgufFile.Open(_fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.True(config.NumKvHeads <= config.NumAttentionHeads,
            $"NumKvHeads ({config.NumKvHeads}) should not exceed NumAttentionHeads ({config.NumAttentionHeads}).");
    }

    [Fact]
    public void Extract_ArchitectureIsKnown()
    {
        using var gguf = GgufFile.Open(_fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.True(Enum.IsDefined(config.Architecture),
            $"Architecture '{config.Architecture}' is not a defined enum value.");
    }

    [Fact]
    public void Extract_TensorCountIsReasonableForLayerCount()
    {
        using var gguf = GgufFile.Open(_fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.True(gguf.Tensors.Count >= config.NumLayers,
            $"Expected at least {config.NumLayers} tensors for {config.NumLayers} layers, got {gguf.Tensors.Count}.");
    }

    [Fact]
    public void Extract_RoPEConfigConsistentWithPositionEncoding()
    {
        using var gguf = GgufFile.Open(_fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        if (config.RoPEConfig.HasValue)
        {
            Assert.Equal(PositionEncodingType.RoPE, config.PositionEncodingType);
            Assert.True(config.RoPEConfig.Value.Theta > 0, "RoPE theta should be positive.");
            Assert.True(config.RoPEConfig.Value.DimensionCount > 0, "RoPE dimension count should be positive.");
        }
        else
        {
            Assert.NotEqual(PositionEncodingType.RoPE, config.PositionEncodingType);
        }
    }
}
