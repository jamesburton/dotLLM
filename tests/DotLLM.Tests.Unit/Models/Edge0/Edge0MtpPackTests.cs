using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.Edge0;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Edge0;

/// <summary>
/// The Edge0 MTP head files (G6, #1990). The planar Q2_0 expert layout comes from edge0-strata <c>tools/mtp_rt.py</c> (<c>blob_of</c>); here an INDEPENDENT planar decoder written
/// from that description is compared with the production dequant of the relaid-out ggml banks, so a plane offset, the gate/up interleave or the block order cannot be wrong in both.
/// </summary>
public sealed class Edge0MtpPackTests(ITestOutputHelper output)
{
    [Fact]
    public void BlobGeometry_MatchesTheFileSize()
    {
        Assert.Equal(1_382_400L, Edge0MtpPack.ExpertBlobBytes);
        Assert.Equal(707_788_800L, Edge0MtpPack.Experts * Edge0MtpPack.ExpertBlobBytes);   // the shipped experts.bin
        Assert.Equal(3L * Edge0MtpPack.ExpertFf * Edge0MtpPack.Hidden / 64 * 18, Edge0MtpPack.ExpertBlobBytes);
    }

    /// <summary>Reference planar Q2_0 row decode: value j of block b = ((code >> 2(j%4)) & 3) - 1) * d (codes 0..3 = -d, 0, +d, +2d), straight from the planes.</summary>
    private static unsafe float[] PlanarRow(byte* blob, int row, int blocksPerRow, long codeOff, long scaleOff)
    {
        var y = new float[blocksPerRow * 64];
        for (int b = 0; b < blocksPerRow; b++)
        {
            long bi = (long)row * blocksPerRow + b;
            float d = (float)BitConverter.UInt16BitsToHalf((ushort)(blob[scaleOff + bi * 2] | (blob[scaleOff + bi * 2 + 1] << 8)));
            for (int j = 0; j < 64; j++)
            {
                int code = (blob[codeOff + bi * 16 + j / 4] >> (2 * (j % 4))) & 3;
                y[b * 64 + j] = (code - 1) * d;
            }
        }
        return y;
    }

    [SkippableFact]
    public unsafe void Relayout_DecodesLikeTheIndependentPlanarReader_OnTheRealExperts()
    {
        var fx = Locate();
        Skip.IfNot(fx.Found, fx.SkipMessage("Edge0-145B-q3k (mtp/)"));
        string mtp = Edge0MtpPack.Find(Path.GetDirectoryName(fx.Path!)!) ?? throw new Xunit.Sdk.XunitException("no mtp folder beside the skeleton");
        using var pack = Edge0MtpPack.Open(mtp, 248320);
        var (gate, up, down) = pack.RelayoutToGgmlBanks();
        var w = new float[2560];
        foreach (int e in new[] { 0, 7, 511 })
        {
            byte* blob = (byte*)pack.ExpertBlob(e);
            foreach (int r in new[] { 0, 1, 333, 639 })
            {
                Dequantize.ToFloat32(gate + (nint)(((long)e * 640 + r) * 40 * 18), 2560, QuantizationType.Q2_0, w);
                AssertEqual(PlanarRow(blob, 2 * r, 40, Edge0MtpPack.GuCodeOffset, Edge0MtpPack.GuScaleOffset), w, $"gate e{e} r{r}");
                Dequantize.ToFloat32(up + (nint)(((long)e * 640 + r) * 40 * 18), 2560, QuantizationType.Q2_0, w);
                AssertEqual(PlanarRow(blob, 2 * r + 1, 40, Edge0MtpPack.GuCodeOffset, Edge0MtpPack.GuScaleOffset), w, $"up e{e} r{r}");
            }
            var wd = new float[640];
            foreach (int r in new[] { 0, 1, 1700, 2559 })
            {
                Dequantize.ToFloat32(down + (nint)(((long)e * 2560 + r) * 10 * 18), 640, QuantizationType.Q2_0, wd);
                AssertEqual(PlanarRow(blob, r, 10, Edge0MtpPack.DownCodeOffset, Edge0MtpPack.DownScaleOffset), wd, $"down e{e} r{r}");
            }
        }
        // the decoded real weights are sane: finite, nonzero, and gate/up are different matrices
        Dequantize.ToFloat32(gate, 2560, QuantizationType.Q2_0, w);
        double rms = Math.Sqrt(w.Sum(v => (double)v * v) / w.Length);
        output.WriteLine($"expert 0 gate row 0 rms {rms:G3}");
        Assert.True(rms > 1e-4 && rms < 1, "gate row decodes to implausible magnitudes");
        var u0 = new float[2560]; Dequantize.ToFloat32(up, 2560, QuantizationType.Q2_0, u0);
        Assert.NotEqual(w, u0);
    }

    private static void AssertEqual(float[] want, float[] got, string what)
    {
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++) Assert.True(want[i] == got[i], $"{what} element {i}: planar {want[i]:G9} vs relaid-out {got[i]:G9}");
    }

    [SkippableFact]
    public void RealHead_FitsTheQwen4ExpHeadTensorContract_AndTheDraftVocabIsPlausible()
    {
        var fx = Locate();
        Skip.IfNot(fx.Found, fx.SkipMessage("Edge0-145B-q3k (mtp/)"));
        using var skel = GgufFile.Open(fx.Path!);
        var config = GgufModelConfigExtractor.Extract(skel.Metadata);
        string mtp = Edge0MtpPack.Find(Path.GetDirectoryName(fx.Path!)!) ?? throw new Xunit.Sdk.XunitException("no mtp folder beside the skeleton");
        using var pack = Edge0MtpPack.Open(mtp, config.VocabSize);
        using var view = pack.BuildHeadGguf(config, ggmlExpertBanks: true, trunk: skel);
        var cfg = Qwen4ExpTensors.HeadConfig(config, view.TensorsByName);
        var problems = Qwen4ExpTensors.FindProblems(view.TensorsByName, cfg, includeTrunk: false, includeMtp: true);
        Assert.True(problems.Count == 0, string.Join("; ", problems.Take(10)));
        output.WriteLine($"draft vocab: {pack.DraftVocab.Length} tokens, first {pack.DraftVocab[0]}, last {pack.DraftVocab[^1]}");
        Assert.InRange(pack.DraftVocab.Length, 10_000, config.VocabSize);
        Assert.Equal(pack.DraftVocab.Length, pack.DraftVocab.Distinct().Count());   // a set of token ids (not sorted: the file orders them by the generator's rank)
    }

    private static FixtureLocation Locate() => TestFixtureResolver.ResolveFile("DOTLLM_EDGE0_SKELETON", "Edge0", "Edge0-145B-q3k", "model-skeleton.gguf",
        "native_experts.txt", "experts.bin", "ple-fp8.gguf", "mtp/experts.bin");
}
