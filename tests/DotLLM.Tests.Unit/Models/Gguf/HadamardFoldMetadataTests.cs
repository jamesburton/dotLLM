using DotLLM.Core.Models;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Gguf;

/// <summary>
/// <c>prism.hadamard.*</c> metadata parsing across format versions (issue #1276). Version 1 is the
/// Bonsai 2 27B layout (own folded <c>output.weight</c>); version 2 is the re-published 2B/4B/9B
/// layout (<c>tied_output</c>, lm_head = rotated <c>token_embd</c>). The rules mirror PrismML's
/// <c>llama-model.cpp</c> loader.
/// </summary>
public sealed class HadamardFoldMetadataTests
{
    private static readonly string[] BlockNames = ["blk.0.ffn_up.weight", "blk.0.ffn_down.weight"];

    private static GgufMetadata Build(uint version, bool? tiedOutput, string[] weightNames, string[]? inverse,
        uint blockSize = 512)
    {
        var e = new Dictionary<string, GgufMetadataValue>
        {
            ["prism.hadamard.version"] = new(GgufValueType.UInt32, version),
            ["prism.hadamard.block_size"] = new(GgufValueType.UInt32, blockSize),
            ["prism.hadamard.transform"] = new(GgufValueType.String, HadamardFoldConfig.SupportedTransform),
            ["prism.hadamard.axis"] = new(GgufValueType.String, HadamardFoldConfig.SupportedAxis),
            ["prism.hadamard.sign_mode"] = new(GgufValueType.String, "identity"),
            ["prism.hadamard.weight_names"] = new(GgufValueType.Array, weightNames),
        };
        if (tiedOutput is bool t)
            e["prism.hadamard.tied_output"] = new(GgufValueType.Bool, t);
        if (inverse is not null)
            e["prism.hadamard.inverse_weight_names"] = new(GgufValueType.Array, inverse);
        return new GgufMetadata(e);
    }

    [Fact]
    public void V1_IsUnchanged()
    {
        string[] names = ["output.weight", .. BlockNames];
        var fold = GgufHadamardFoldExtractor.TryExtract(
            Build(1, tiedOutput: null, names, ["token_embd.weight"], blockSize: 1024))!;

        Assert.False(fold.TiedOutput);
        Assert.Equal(1024, fold.BlockSize);
        Assert.Equal(names.Order(), fold.FoldedWeights.Order());
        Assert.True(fold.IsInverse("token_embd.weight"));
    }

    [Fact]
    public void V2_TiedOutput_RegistersTheLmHeadSite()
    {
        var fold = GgufHadamardFoldExtractor.TryExtract(
            Build(2, tiedOutput: true, BlockNames, ["token_embd.weight"]))!;

        Assert.True(fold.TiedOutput);
        Assert.Equal(512, fold.BlockSize);
        // The tied head is forward-rotated like a folded output.weight; the table stays inverse.
        Assert.True(fold.IsFolded("output.weight"));
        Assert.True(fold.IsInverse("token_embd.weight"));
        Assert.False(fold.IsFolded("token_embd.weight"));
        Assert.Equal(BlockNames.Length + 1, fold.FoldedWeights.Count);
    }

    [Theory]
    [InlineData(0u)]
    [InlineData(3u)]
    [InlineData(99u)]
    public void UnknownVersion_IsRefused(uint version)
    {
        var ex = Assert.Throws<NotSupportedException>(() => GgufHadamardFoldExtractor.TryExtract(
            Build(version, tiedOutput: true, BlockNames, ["token_embd.weight"])));
        Assert.Contains("prism.hadamard.version", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void V2_WithoutTiedOutput_IsRefused() =>
        Assert.Throws<InvalidDataException>(() => GgufHadamardFoldExtractor.TryExtract(
            Build(2, tiedOutput: null, BlockNames, ["token_embd.weight"])));

    [Fact]
    public void V1_WithTiedOutput_IsRefused() =>
        Assert.Throws<InvalidDataException>(() => GgufHadamardFoldExtractor.TryExtract(
            Build(1, tiedOutput: true, ["output.weight", .. BlockNames], ["token_embd.weight"])));

    [Fact]
    public void V2_WithoutLatentEmbedding_IsRefused() =>
        Assert.Throws<InvalidDataException>(() => GgufHadamardFoldExtractor.TryExtract(
            Build(2, tiedOutput: true, BlockNames, inverse: null)));

    [Fact]
    public void V2_WithSeparateFoldedOutputWeight_IsRefused() =>
        Assert.Throws<InvalidDataException>(() => GgufHadamardFoldExtractor.TryExtract(
            Build(2, tiedOutput: true, ["output.weight", .. BlockNames], ["token_embd.weight"])));

    [Fact]
    public void ValidateOutputHead_EnforcesTiedness()
    {
        var v2 = GgufHadamardFoldExtractor.TryExtract(Build(2, true, BlockNames, ["token_embd.weight"]))!;
        v2.ValidateOutputHead(hasOutputWeight: false);
        Assert.Throws<NotSupportedException>(() => v2.ValidateOutputHead(hasOutputWeight: true));

        var v1 = GgufHadamardFoldExtractor.TryExtract(
            Build(1, null, ["output.weight", .. BlockNames], ["token_embd.weight"]))!;
        v1.ValidateOutputHead(hasOutputWeight: true);
        Assert.Throws<NotSupportedException>(() => v1.ValidateOutputHead(hasOutputWeight: false));
    }
}
