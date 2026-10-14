using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Lfm2Moe;

/// <summary>
/// #1015: config / layout extraction for <c>lfm2moe</c> from the real LFM2.5-24B-A2B GGUF header (returns without asserting when no such file
/// is available; resolve via <c>DOTLLM_LFM2_GGUF</c> or the HF hub cache).
/// </summary>
public sealed class Lfm2MoeConfigTests
{
    private static string? FindGguf()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_LFM2_GGUF");
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;
        string hub = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "huggingface", "hub", "models--LiquidAI--LFM2.5-24B-A2B-GGUF", "snapshots");
        return Directory.Exists(hub) ? Directory.EnumerateFiles(hub, "*.gguf", SearchOption.AllDirectories).FirstOrDefault() : null;
    }

    [Fact]
    public void RealHeader_ExtractsHybridLayoutAndSigmoidBiasMoe()
    {
        string? path = FindGguf();
        if (path is null) return;
        using var gguf = GgufFile.Open(path);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.Equal(Architecture.Lfm2Moe, cfg.Architecture);
        Assert.Equal(40, cfg.NumLayers);
        Assert.Equal(2048, cfg.HiddenSize);
        Assert.Equal(32, cfg.NumAttentionHeads);
        Assert.Equal(8, cfg.NumKvHeads);
        Assert.Equal(64, cfg.HeadDim);
        Assert.Equal(11776, cfg.IntermediateSize);
        Assert.Equal(3, cfg.ShortConvCacheLength);
        Assert.Equal(RoPEType.NeoX, cfg.RoPEConfig!.Value.Type);

        var kinds = cfg.HybridLayout!.LayerKind;
        for (int i = 0; i < 40; i++)
            Assert.Equal(i % 4 == 2 ? HybridLayerKind.Attention : HybridLayerKind.ShortConv, kinds[i]);   // 10 attention at 2, 6, ... 38

        var moe = cfg.Moe!;
        Assert.Equal(64, moe.NumExperts);
        Assert.Equal(4, moe.NumExpertsPerTok);
        Assert.Equal(1536, moe.MoeIntermediateSize);
        Assert.True(moe.SigmoidGating);
        Assert.True(moe.HasSelectionBias);
        Assert.True(moe.NormalizeExpertWeights);   // llama.cpp lfm2moe.cpp hard-codes norm_w = true
        Assert.False(moe.IsMoeLayer(0));
        Assert.False(moe.IsMoeLayer(1));
        Assert.True(moe.IsMoeLayer(2));
        Assert.True(moe.IsMoeLayer(39));
    }
}
