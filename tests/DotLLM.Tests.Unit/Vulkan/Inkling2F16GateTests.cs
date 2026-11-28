using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Core.Vision;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Models.Vision;
using DotLLM.Tests.Unit.Models.Architectures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Gates 2 and 3 of the F16 activation default (#2075, docs/INKLING2.md 12c): the runtime guard (fail loudly on a non-finite F16 logit; the ViT recomputes in BF16) and the GPU
/// overflow sweeps over long-context and multi-image prompts (non-finite count and logit range of every row of every chunk on the F16 arm). The sweeps are long and opt-in
/// (<c>DOTLLM_INK_F16_SWEEP=1</c>); hold the Strix GPU lock (and cpu-bench: the CPU encoder runs in this process).
/// </summary>
public sealed class Inkling2F16GateTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    public Inkling2F16GateTests(ITestOutputHelper o) => _out = o;
    public void Dispose() { }

    private static void SkipUnlessGpu()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available on this host.");
    }

    private static void SweepOnly() => Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_INK_F16_SWEEP") == "1", "set DOTLLM_INK_F16_SWEEP=1 (long; Strix GPU lock + cpu-bench)");

    private void Log(string line)
    {
        _out.WriteLine(line);
        if (Environment.GetEnvironmentVariable("DOTLLM_INK_PREC_OUT") is { Length: > 0 } f) File.AppendAllText(f, line + Environment.NewLine);
    }

    [Fact]
    public void AllFinite_DetectsNaNAndInfinityInHalfRows()
    {
        var rows = new Half[1000]; for (int i = 0; i < rows.Length; i++) rows[i] = (Half)(i * 0.5f - 100f);
        Assert.True(VulkanInkling2VisionEncoder.AllFinite(rows));
        foreach (Half bad in new[] { Half.NaN, Half.PositiveInfinity, Half.NegativeInfinity })
        {
            var r = (Half[])rows.Clone(); r[777] = bad;
            Assert.False(VulkanInkling2VisionEncoder.AllFinite(r));
        }
    }

    /// <summary>Gate 3, text model: a non-finite F16 logit throws (naming the BF16 switch); BF16 has no guard; the cost of the check is printed.</summary>
    [SkippableFact]
    public void Guard_NonFiniteLogits_ThrowLoudlyOnF16_AndCostIsNegligible()
    {
        SkipUnlessGpu();
        string? gguf = Inkling2RealModelTests.FindGguf(); Skip.If(gguf is null, "Inkling-2-Mini-NVFP4.gguf not in the HF cache");
        using var file = GgufFile.Open(gguf!);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        using var dev = VulkanDevice.Create();
        using var vk = VulkanInkling2TransformerModel.BuildFromGguf(dev, file, cfg, Inkling2VulkanTests.SpvDir(), new Inkling2VulkanOptions());
        Assert.Equal(Inkling2ActFormat.F16, vk.Options.BaseAct);   // the default
        int[] ids = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000];
        using (vk.Forward(ids, Enumerable.Range(0, ids.Length).ToArray(), -1, null, true)) { }   // clean run: no throw
        foreach (float bad in new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity })
        {
            vk.BeforeFiniteCheck = l => l[12345] = bad;
            vk.ResetSequenceState();
            var ex = Assert.Throws<InvalidOperationException>(() => { using var t = vk.Forward(ids, Enumerable.Range(0, ids.Length).ToArray(), -1, null, true); });
            Assert.Contains("DOTLLM_INKLING2_ACT=bf16", ex.Message);
        }
        vk.BeforeFiniteCheck = null;
        // the check itself: one SumOfMagnitudes over a downloaded row of logits (200,058 floats)
        var row = new float[cfg.VocabSize]; new Random(1).NextBytes(System.Runtime.InteropServices.MemoryMarshal.AsBytes(row.AsSpan())); for (int i = 0; i < row.Length; i++) if (!float.IsFinite(row[i])) row[i] = 1f;
        float sink = 0; var sw = Stopwatch.StartNew();
        for (int i = 0; i < 2000; i++) sink += System.Numerics.Tensors.TensorPrimitives.SumOfMagnitudes(row.AsSpan());
        double us = sw.Elapsed.TotalMilliseconds * 1000 / 2000;
        Log($"F16GUARD text: non-finite logits throw (NaN, +inf, -inf); the finite check over one {cfg.VocabSize}-float row costs {us:F1} us (decode step ~ 50 ms) [{sink > 0}]");
        // BF16 has the full exponent range and no guard: a poisoned BF16 run does not throw
        using var bf = VulkanInkling2TransformerModel.BuildFromGguf(dev, file, cfg, Inkling2VulkanTests.SpvDir(), new Inkling2VulkanOptions(ActivationF16: false));
        Assert.Equal(Inkling2ActFormat.Bf16, bf.Options.BaseAct);
        bf.BeforeFiniteCheck = l => l[1] = float.NaN;
        using (bf.Forward(ids, Enumerable.Range(0, ids.Length).ToArray(), -1, null, true)) { }
    }

    /// <summary>Gate 3, ViT: a non-finite F16 embedding makes the encoder recompute that image in BF16 (stateless), counted, and the result equals the BF16 arm's.</summary>
    [SkippableFact]
    public void Guard_Vit_NonFiniteF16_RecomputesInBf16()
    {
        SkipUnlessGpu();
        string? mm = Inkling2VisionFixtures.FindMmproj(); Skip.If(mm is null, "mmproj not in the HF cache");
        var o = Inkling2VisionEndToEndTests.LoadOracle("rainbow80"); Skip.If(o is null, "vision oracle missing");
        using var dev = VulkanDevice.Create();
        using var enc = VulkanInkling2VisionEncoder.Load(dev, mm!, Inkling2VulkanTests.SpvDir());
        Assert.Equal(Inkling2ActFormat.F16, enc.ActFormat);   // the default
        using var clean = enc.Encode(o!.Image);
        Assert.Equal(0, enc.Bf16Fallbacks);
        enc.AfterDownload = r => r[5] = Half.NaN;
        using var guarded = enc.Encode(o.Image);
        enc.AfterDownload = null;
        Assert.Equal(1, enc.Bf16Fallbacks);
        Assert.Equal(Inkling2ActFormat.F16, enc.ActFormat);   // the fallback is per call
        enc.ActFormat = Inkling2ActFormat.Bf16;
        using var bf16 = enc.Encode(o.Image);
        var g = Inkling2VulkanVisionEncoderTests.Floats(guarded);
        Assert.True(g.All(float.IsFinite));
        Assert.Equal(Inkling2VulkanVisionEncoderTests.Floats(bf16), g);   // identical to a plain BF16 encode
        double d = Inkling2VulkanVisionEncoderTests.RelRms(Inkling2VulkanVisionEncoderTests.Floats(clean), g);
        Log($"F16GUARD vit: poisoned F16 result recomputed in BF16 (fallbacks 1); F16 vs BF16 encode relRMS {d:E2}");
    }

    private static int[] LongPrompt(GgufFile file, int tokens)
    {
        string corpus = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache", "corpora", "wikitext-2-raw", "wiki.test.lf.raw");
        Skip.IfNot(File.Exists(corpus), "wikitext-2-raw corpus missing");
        var tok = GgufTokenizerFactory.Load(file.Metadata);
        string text = File.ReadAllText(corpus);
        var ids = tok.Encode(text[..Math.Min(text.Length, tokens * 6)]);
        Skip.If(ids.Length < tokens, $"corpus tokenises to {ids.Length} < {tokens} tokens");
        return ids[..tokens];
    }

    /// <summary>Runs <paramref name="prompt"/> in 128-token chunks with every row's logits checked on the F16 arm; reports non-finite counts and the logit range.</summary>
    private (long NonFinite, float MaxAbsLogit, double Seconds) Sweep(VulkanInkling2TransformerModel vk, int[] prompt, IReadOnlyList<ImageSpan>? images)
    {
        long bad = 0; float maxAbs = 0;
        vk.BeforeFiniteCheck = l =>
        {
            foreach (float v in l) { if (!float.IsFinite(v)) bad++; else { float a = MathF.Abs(v); if (a > maxAbs) maxAbs = a; } }
        };
        vk.ResetSequenceState();
        var sw = Stopwatch.StartNew();
        for (int at = 0; at < prompt.Length; at += 128)
        {
            int len = Math.Min(128, prompt.Length - at);
            var pos = Enumerable.Range(at, len).ToArray();
            using var t = images is null ? vk.Forward(prompt.AsSpan(at, len), pos, -1) : vk.ForwardWithImages(prompt.AsSpan(at, len), pos, -1, images, null, false);
        }
        vk.BeforeFiniteCheck = null;
        return (bad, maxAbs, sw.Elapsed.TotalSeconds);
    }

    /// <summary>Gate 2, text: a 16,640-token wikitext prompt on the F16 arm; every logit of every row must be finite (the guard would throw on the first).</summary>
    [SkippableFact]
    public void Sweep_Text16K_F16_NoNonFinite()
    {
        SweepOnly(); SkipUnlessGpu();
        string? gguf = Inkling2RealModelTests.FindGguf(); Skip.If(gguf is null, "model missing");
        using var file = GgufFile.Open(gguf!);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        int[] prompt = LongPrompt(file, 16_640);
        using var dev = VulkanDevice.Create();
        using var vk = VulkanInkling2TransformerModel.BuildFromGguf(dev, file, cfg, Inkling2VulkanTests.SpvDir(), new Inkling2VulkanOptions());
        var (bad, maxAbs, secs) = Sweep(vk, prompt, null);
        Log($"F16SWEEP text {prompt.Length} tokens (128-token chunks, every row's logits): non-finite {bad}, max |logit| {maxAbs:F2}, {secs:F0} s; global K/V {vk.KvBytes / 1e6:F0} MB");
        Assert.Equal(0, bad);
    }

    /// <summary>Gate 2, multi-image: the 80, 190x110 and 640 images in one prompt on the F16 arm (CPU encoder), every row finite.</summary>
    [SkippableFact]
    public void Sweep_MultiImage_F16_NoNonFinite()
    {
        SweepOnly(); SkipUnlessGpu();
        string? gguf = Inkling2RealModelTests.FindGguf(); Skip.If(gguf is null, "model missing");
        string? mm = Inkling2VisionFixtures.FindMmproj(); Skip.If(mm is null, "mmproj missing");
        using var file = GgufFile.Open(gguf!);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        using var enc = Inkling2VisionEncoder.Load(mm!, ThreadingConfig.Auto);
        var tokens = new List<int>(); var spans = new List<ImageSpan>(); var embs = new List<ImageEmbeddings>();
        try
        {
            foreach (string name in new[] { "rainbow80", "rainbow190x110", "rainbow640", "rainbow190x110", "rainbow80" })
            {
                var o = Inkling2VisionEndToEndTests.LoadOracle(name); Skip.If(o is null, "vision oracle missing");
                var emb = enc.Encode(o!.Image); embs.Add(emb);
                tokens.AddRange(o.Before); spans.Add(new ImageSpan(tokens.Count, emb)); tokens.AddRange(Enumerable.Repeat(200054, emb.Tokens)); tokens.AddRange(o.After);
            }
            using var dev = VulkanDevice.Create();
            using var vk = VulkanInkling2TransformerModel.BuildFromGguf(dev, file, cfg, Inkling2VulkanTests.SpvDir(), new Inkling2VulkanOptions());
            var (bad, maxAbs, secs) = Sweep(vk, tokens.ToArray(), spans);
            Log($"F16SWEEP multi-image {tokens.Count} tokens, {spans.Count} images ({string.Join("+", embs.Select(e => e.Tokens))} image tokens), every row's logits: non-finite {bad}, max |logit| {maxAbs:F2}, {secs:F1} s");
            Assert.Equal(0, bad);
        }
        finally { foreach (var e in embs) e.Dispose(); }
    }
}
