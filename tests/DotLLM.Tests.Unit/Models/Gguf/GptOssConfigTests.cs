using DotLLM.Core.Configuration;
using DotLLM.Core.PositionEncoding;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Gguf;

/// <summary>
/// gpt-oss ModelConfig extraction: architecture mapping, MoE config
/// (softmax-after-top-k + swiglu_oai + biases), alternating sliding-window
/// pattern, NeoX RoPE with YaRN scaling — mirrors the real
/// gpt-oss-20b-mxfp4.gguf metadata.
/// </summary>
public class GptOssConfigTests
{
    private static GgufMetadata BuildGptOssMetadata(Action<GgufTestData>? extra = null)
    {
        var data = new GgufTestData(version: 3);
        data.AddString("general.architecture", "gpt-oss");
        data.AddUInt32("gpt-oss.block_count", 24);
        data.AddUInt32("gpt-oss.context_length", 131072);
        data.AddUInt32("gpt-oss.embedding_length", 2880);
        data.AddUInt32("gpt-oss.feed_forward_length", 2880);
        data.AddUInt32("gpt-oss.attention.head_count", 64);
        data.AddUInt32("gpt-oss.attention.head_count_kv", 8);
        data.AddFloat32("gpt-oss.rope.freq_base", 150000.0f);
        data.AddFloat32("gpt-oss.attention.layer_norm_rms_epsilon", 1e-5f);
        data.AddUInt32("gpt-oss.expert_count", 32);
        data.AddUInt32("gpt-oss.expert_used_count", 4);
        data.AddUInt32("gpt-oss.attention.key_length", 64);
        data.AddUInt32("gpt-oss.attention.value_length", 64);
        data.AddUInt32("gpt-oss.attention.sliding_window", 128);
        data.AddUInt32("gpt-oss.expert_feed_forward_length", 2880);
        data.AddString("gpt-oss.rope.scaling.type", "yarn");
        data.AddFloat32("gpt-oss.rope.scaling.factor", 32.0f);
        data.AddUInt32("gpt-oss.rope.scaling.original_context_length", 4096);
        data.AddUInt32("gpt-oss.vocab_size", 201088);
        extra?.Invoke(data);

        byte[] bytes = data.Build();
        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream);
        var header = GgufReader.ReadHeader(reader);
        var raw = GgufReader.ReadMetadata(reader, header);
        return new GgufMetadata(raw);
    }

    [Fact]
    public void Extract_GptOss_CoreFields()
    {
        var config = GgufModelConfigExtractor.Extract(BuildGptOssMetadata());

        Assert.Equal(Architecture.GptOss, config.Architecture);
        Assert.Equal(2880, config.HiddenSize);
        Assert.Equal(24, config.NumLayers);
        Assert.Equal(64, config.NumAttentionHeads);
        Assert.Equal(8, config.NumKvHeads);
        Assert.Equal(64, config.HeadDim);       // key_length, NOT hidden/heads (=45)
        Assert.Equal(201088, config.VocabSize);
        Assert.Equal(131072, config.MaxSequenceLength);
    }

    [Fact]
    public void Extract_GptOss_MoeConfig_SoftmaxAfterTopKAndSwiGluOai()
    {
        var config = GgufModelConfigExtractor.Extract(BuildGptOssMetadata());

        Assert.NotNull(config.Moe);
        Assert.Equal(32, config.Moe.NumExperts);
        Assert.Equal(4, config.Moe.NumExpertsPerTok);
        Assert.Equal(2880, config.Moe.MoeIntermediateSize);
        Assert.True(config.Moe.SoftmaxAfterTopK);
        Assert.True(config.Moe.UseSwiGluOai);
        Assert.True(config.Moe.HasExpertBiases);
        // Every layer is a routed-MoE layer.
        for (int i = 0; i < 24; i++)
            Assert.True(config.Moe.IsMoeLayer(i));
    }

    [Fact]
    public void Extract_GptOss_AlternatingSlidingWindow_DefaultPattern2()
    {
        var config = GgufModelConfigExtractor.Extract(BuildGptOssMetadata());

        Assert.Equal(128, config.SlidingWindowSize);
        // No sliding_window_pattern key in the file → llama.cpp default 2:
        // swa on layers where il % 2 < 1 (even layers), dense on odd.
        Assert.Equal(2, config.SlidingWindowPattern);
    }

    [Fact]
    public void Extract_GptOss_ExplicitSlidingWindowPattern()
    {
        var config = GgufModelConfigExtractor.Extract(BuildGptOssMetadata(d =>
            d.AddUInt32("gpt-oss.attention.sliding_window_pattern", 4)));

        Assert.Equal(4, config.SlidingWindowPattern);
    }

    [Fact]
    public void Extract_GptOss_RoPE_NeoXWithYarn()
    {
        var config = GgufModelConfigExtractor.Extract(BuildGptOssMetadata());

        Assert.NotNull(config.RoPEConfig);
        var rope = config.RoPEConfig.Value;
        Assert.Equal(RoPEType.NeoX, rope.Type);
        Assert.Equal(150000.0f, rope.Theta);
        Assert.Equal(RoPEScalingType.YaRN, rope.ScalingType);
        Assert.Equal(32.0f, rope.ScalingFactor);
        Assert.Equal(4096, rope.OrigMaxSeqLen);
        // No rope.dimension_count key → rotate the full head dim (64).
        Assert.Equal(64, rope.DimensionCount);
    }
}
