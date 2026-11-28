using DotLLM.Core.Configuration;
using DotLLM.Core.Vision;
using DotLLM.Models.Vision;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vision;

/// <summary>
/// CPU Qwen3-VL vision tower of Bonsai 2.5 against llama.cpp b12780 CPU (<c>mtmd_encode_chunk</c>, F32 embeddings cached by
/// <c>tools/bonsai25-vision-oracle</c>), issue #2073. The reference runs Q8_0 activation quantisation through 27 blocks, so the embedding gap is a noise
/// floor (see docs/INKLING2.md section 12 for the same finding on Inkling 2): this gate catches gross errors, the checkpoint-free sharp gate is the
/// end-to-end KL in BonsaiVisionEndToEndTests.
/// </summary>
[Trait("Category", "Heavy")]
public sealed class BonsaiVisionEncoderTests
{
    private readonly ITestOutputHelper _out;
    public BonsaiVisionEncoderTests(ITestOutputHelper output) => _out = output;

    [SkippableTheory]
    [InlineData("rainbow80", 9)]
    [InlineData("news224x168", 35)]
    [InlineData("news640x488", 300)]
    public void Embeddings_MatchLlamaCpp(string name, int tokens)
    {
        var o = BonsaiVisionFixtures.LoadOracle(name);
        using var enc = QwenVlVisionEncoder.Load(BonsaiVisionFixtures.ResolveMmproj(), ThreadingConfig.Auto);
        Assert.Equal(tokens, enc.CountTokens(o.Image.Width, o.Image.Height));
        Assert.Equal(tokens, o.ImageTokens);
        using var e = enc.Encode(o.Image);
        Assert.Equal(tokens, e.Tokens);
        Assert.Equal(5120, e.Dimension);
        var (rel, cos) = BonsaiVisionFixtures.Compare(e, o.Embd);
        _out.WriteLine($"{name}: tokens {e.Tokens} grid {e.GridWidth}x{e.GridHeight} relRMS {rel:E3} minRowCos {cos:F5}");
        Assert.True(rel < 0.06, $"relRMS {rel}");
        Assert.True(cos > 0.97, $"min row cosine {cos}");
    }

    [SkippableFact]
    public void Report_Arms()
    {
        foreach (string name in BonsaiVisionFixtures.Images)
        {
            var o = BonsaiVisionFixtures.LoadOracle(name);
            foreach (var prec in new[] { VisionEmbedPrecision.F32, VisionEmbedPrecision.BF16 })
                foreach (bool exactGelu in new[] { false, true })
                    foreach (bool skipF16 in new[] { false, true })
                    {
                        using var enc = QwenVlVisionEncoder.Load(BonsaiVisionFixtures.ResolveMmproj(), ThreadingConfig.Auto, embedPrecision: prec);
                        enc.ExactGelu = exactGelu; enc.SkipIm2colF16Rounding = skipF16;
                        using var e = enc.Encode(o.Image);
                        var (rel, cos) = BonsaiVisionFixtures.Compare(e, o.Embd);
                        _out.WriteLine($"{name} embd={prec} exactGelu={exactGelu} skipIm2colF16={skipF16}: relRMS {rel:E3} minCos {cos:F5}");
                    }
        }
    }
}
