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
/// Image prompts through the Inkling 2 Vulkan text model (#2062, PR A: the CPU encoder feeds the Vulkan model) against the cached llama.cpp b12780 CPU oracle
/// (<c>~/.dotllm/test-cache/token-oracles/inkling2/vision/</c>), teacher-forced over the oracle's own greedy continuation. The CPU reference's numbers are in
/// docs/INKLING2.md section 12 (mean KL 2.6e-3 to 3.4e-3 on the three images with the BF16 embedding default, gate 0.015). Hold the Strix GPU lock AND the cpu-bench
/// lock (the CPU encoder runs in this process, never beside another iGPU job).
/// </summary>
public sealed class Inkling2VulkanVisionTests : IDisposable
{
    private const double Gate = 0.015;
    private readonly ITestOutputHelper _output;
    private GgufFile? _file;
    private VulkanDevice? _dev;
    private VulkanInkling2TransformerModel? _model;
    private Inkling2VisionEncoder? _encoder;
    private VulkanInkling2VisionEncoder? _vkEncoder;

    public Inkling2VulkanVisionTests(ITestOutputHelper output) => _output = output;

    public void Dispose() { _vkEncoder?.Dispose(); _encoder?.Dispose(); _model?.Dispose(); _dev?.Dispose(); _file?.Dispose(); }

    private (VulkanInkling2TransformerModel Model, Inkling2VisionEncoder Encoder) Models()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available on this host.");
        string? gguf = Inkling2RealModelTests.FindGguf();
        Skip.If(gguf is null, "Inkling-2-Mini-NVFP4.gguf not in the HF cache");
        string? mm = Inkling2VisionFixtures.FindMmproj();
        Skip.If(mm is null, "mmproj-Inkling-2-Mini-Q8_0.gguf not in the HF cache");
        if (_model is null)
        {
            _file = GgufFile.Open(gguf!);
            var cfg = GgufModelConfigExtractor.Extract(_file.Metadata);
            _dev = VulkanDevice.Create();
            _model = VulkanInkling2TransformerModel.BuildFromGguf(_dev, _file, cfg, Inkling2VulkanTests.SpvDir(), new Inkling2VulkanOptions());
            _encoder = Inkling2VisionEncoder.Load(mm!, ThreadingConfig.Auto);   // the shipping default (BF16 embeddings)
        }
        return (_model, _encoder!);
    }

    private static unsafe float[] Row(ITensor t, int row, int vocab)
    {
        var a = new float[vocab];
        new ReadOnlySpan<float>((float*)t.DataPointer + (long)row * t.Shape[1], vocab).CopyTo(a);
        return a;
    }

    /// <summary>The image prompt, prefilled in <paramref name="chunk"/>-token calls, then the oracle's continuation teacher-forced; every scored row's logits.</summary>
    private (float[][] Rows, int GreedyMismatch, double PrefillSeconds) RunRows(Inkling2VisionEndToEndTests.Oracle o, VulkanInkling2TransformerModel model,
        ImageEmbeddings emb, int chunk)
    {
        var prompt = o.Before.Concat(Enumerable.Repeat(200054, emb.Tokens)).Concat(o.After).ToArray();
        var spans = new[] { new ImageSpan(o.Before.Length, emb) };
        int V = o.NVocab, mism = 0;
        var rows = new List<float[]>();
        model.ResetSequenceState();
        float[] ours = [];
        var sw = Stopwatch.StartNew();
        for (int at = 0; at < prompt.Length; at += chunk)
        {
            int len = Math.Min(chunk, prompt.Length - at);
            using var t = model.ForwardWithImages(prompt.AsSpan(at, len), Enumerable.Range(at, len).ToArray(), -1, spans, null, lastTokenLogitsOnly: true);
            if (at + len == prompt.Length) ours = Row(t, 0, model.Config.VocabSize);
        }
        double prefill = sw.Elapsed.TotalSeconds;
        for (int r = 0; r < o.Gen.Length + 1; r++)
        {
            rows.Add(ours);
            if (r < o.Gen.Length)
            {
                int arg = 0; for (int i = 1; i < V; i++) if (ours[i] > ours[arg]) arg = i;
                if (arg != o.Gen[r]) mism++;
                using var step = model.ForwardWithImages([o.Gen[r]], [prompt.Length + r], -1, spans, null, lastTokenLogitsOnly: true);
                ours = Row(step, 0, model.Config.VocabSize);
            }
        }
        return ([.. rows], mism, prefill);
    }

    private (double Mean, double Max, int Top1) Score(Inkling2VisionEndToEndTests.Oracle o, float[][] rows)
    {
        int V = o.NVocab; double sum = 0, max = 0; int top = 0;
        for (int r = 0; r < rows.Length; r++)
        {
            var (kl, same) = Inkling2VisionEndToEndTests.FullKl(o.Logits.AsSpan(r * V, V), rows[r]);
            sum += kl; max = Math.Max(max, kl); if (same) top++;
        }
        return (sum / rows.Length, max, top);
    }

    /// <summary>Teacher-forced KL of the text logits after an image prompt against llama.cpp (gate 0.015; CPU reference 2.6e-3 to 3.4e-3). 32-token prefill calls.</summary>
    [SkippableTheory]
    [InlineData("rainbow80")]
    [InlineData("rainbow190x110")]
    [InlineData("rainbow640")]
    public void ImagePrompt_Logits_MatchLlamaCpp(string name)
    {
        var o = Inkling2VisionEndToEndTests.LoadOracle(name);
        Skip.If(o is null, "vision oracle not generated");
        var (model, enc) = Models();
        using var emb = enc.Encode(o!.Image);
        Assert.Equal(o.ImageTokens, emb.Tokens);
        var (rows, mism, _) = RunRows(o, model, emb, 32);
        var s = Score(o, rows);
        _output.WriteLine($"{name} (Vulkan text, CPU encoder): mean KL {s.Mean:E2}, max KL {s.Max:E2}, top-1 {s.Top1}/{rows.Length}, teacher-forced greedy mismatches {mism}/{o.Gen.Length}");
        Assert.True(s.Mean < Gate, $"mean KL {s.Mean:E2} (gate {Gate:E1})");
        Assert.True(s.Top1 >= rows.Length - 2, $"top-1 agreement {s.Top1}/{rows.Length}");
    }

    /// <summary>The engine's default prefill (the whole prompt in one forward call, 128-token internal chunks) against llama.cpp, with the prefill time.</summary>
    [SkippableTheory]
    [InlineData("rainbow80")]
    [InlineData("rainbow190x110")]
    [InlineData("rainbow640")]
    public void ImagePrompt_SinglePassPrefill_MatchesLlamaCpp(string name)
    {
        var o = Inkling2VisionEndToEndTests.LoadOracle(name);
        Skip.If(o is null, "vision oracle not generated");
        var (model, enc) = Models();
        using var emb = enc.Encode(o!.Image);
        var (rows, mism, secs) = RunRows(o, model, emb, 100_000);
        var s = Score(o, rows);
        _output.WriteLine($"{name} single pass: mean KL {s.Mean:E2}, max KL {s.Max:E2}, top-1 {s.Top1}/{rows.Length}, greedy mismatches {mism}; Vulkan prefill of {o.Before.Length + emb.Tokens + o.After.Length} tokens {secs:F2} s");
        Assert.True(s.Mean < Gate, $"mean KL {s.Mean:E2} (gate {Gate:E1})");
        Assert.True(s.Top1 >= rows.Length - 2, $"top-1 agreement {s.Top1}/{rows.Length}");
    }

    /// <summary>
    /// Gate: the final-position logits of an image prompt do not depend on the prefill chunk size (CPU: exactly 0 after the conv-history fix). Chunks 4, 5 and 7
    /// put the image rows across chunk boundaries at offsets different from the 32-token run (and 4 / 5 / 7 do not divide the 128-token internal chunk).
    /// </summary>
    [SkippableTheory]
    [InlineData("rainbow80", 5)]
    [InlineData("rainbow190x110", 7)]
    [InlineData("rainbow190x110", 4)]
    public void ChunkedPrefill_ImageLogits_AreChunkSizeInvariant(string name, int chunk)
    {
        var o = Inkling2VisionEndToEndTests.LoadOracle(name);
        Skip.If(o is null, "vision oracle not generated");
        var (model, enc) = Models();
        using var emb = enc.Encode(o!.Image);
        var reference = RunRows(o, model, emb, 32).Rows[0];
        var chunked = RunRows(o, model, emb, chunk).Rows[0];
        var (kl, same) = Inkling2VisionEndToEndTests.FullKl(reference, chunked);
        _output.WriteLine($"{name} chunk {chunk}: image KL vs chunk 32 {kl:E2} (same top-1 {same})");
        Assert.True(same, "the top-1 token must not depend on the chunk size");
        Assert.True(kl < 1e-5, $"chunk {chunk} moved the image prompt logits by KL {kl:E2}");
    }

    /// <summary>
    /// Exact, noise-free addressing check: the <c>embed</c> trace (BF16 activations) of every chunk, for chunk sizes 4, 5 and 7, equals the one-pass trace bit for
    /// bit; image positions hold the encoder's F16 row rounded once to BF16 (no norm), text positions the normalised token embedding.
    /// </summary>
    [SkippableTheory]
    [InlineData("rainbow80", 5)]
    [InlineData("rainbow190x110", 7)]
    [InlineData("rainbow190x110", 4)]
    public void ChunkedPrefill_ImageRows_AreAddressedByAbsolutePosition(string name, int chunk)
    {
        var o = Inkling2VisionEndToEndTests.LoadOracle(name);
        Skip.If(o is null, "vision oracle not generated");
        var (model, enc) = Models();
        using var emb = enc.Encode(o!.Image);
        var prompt = o.Before.Concat(Enumerable.Repeat(200054, emb.Tokens)).Concat(o.After).ToArray();
        var spans = new[] { new ImageSpan(o.Before.Length, emb) };
        int H = model.Config.HiddenSize;

        float[] Embeds(int ch)
        {
            var all = new float[prompt.Length * H];
            int at = 0, cursor = 0;   // a long call is split into 128-token internal chunks, each traced separately
            model.Trace = (n, data, rows, cols) => { if (n == "embed") { data.CopyTo(all.AsSpan(cursor * H, rows * cols)); cursor += rows; } };
            try
            {
                model.ResetSequenceState();
                for (at = 0; at < prompt.Length; at += ch)
                {
                    int len = Math.Min(ch, prompt.Length - at);
                    cursor = at;
                    using var t = model.ForwardWithImages(prompt.AsSpan(at, len), Enumerable.Range(at, len).ToArray(), -1, spans, null, lastTokenLogitsOnly: true);
                }
            }
            finally { model.Trace = null; }
            return all;
        }
        var reference = Embeds(prompt.Length);
        var chunked = Embeds(chunk);
        Assert.Equal(reference, chunked);
        static float Bf(float f) { uint b = BitConverter.SingleToUInt32Bits(f); b += 0x7fffu + ((b >> 16) & 1u); return BitConverter.UInt32BitsToSingle(b & 0xFFFF0000u); }
        var row = new float[H];
        for (int i = 0; i < emb.Tokens; i++)
        {
            emb.ReadRow(i, row);
            Assert.Equal(row.Select(Bf).ToArray(), chunked.AsSpan((o.Before.Length + i) * H, H).ToArray());
        }
        Assert.NotEqual(chunked.AsSpan(0, H).ToArray(), chunked.AsSpan(o.Before.Length * H, H).ToArray());
        // The token rows outside the image are untouched by the injection: the same trace of a text-only call over the prompt prefix.
        var textOnly = new float[o.Before.Length * H];
        model.Trace = (n, data, rows, cols) => { if (n == "embed") data.CopyTo(textOnly, 0); };
        try
        {
            model.ResetSequenceState();
            using var t = model.Forward(prompt.AsSpan(0, o.Before.Length), Enumerable.Range(0, o.Before.Length).ToArray(), -1, null, true);
        }
        finally { model.Trace = null; }
        Assert.Equal(textOnly, chunked.AsSpan(0, o.Before.Length * H).ToArray());
    }

    /// <summary>Text-only prompts are unchanged by an empty image list, and an image list that does not overlap the chunk runs the plain path (bit-identical logits).</summary>
    [SkippableFact]
    public void NoImageRows_IsTheTextPath()
    {
        var (model, enc) = Models();
        int[] ids = Enumerable.Range(1000, 40).ToArray(); int[] pos = Enumerable.Range(0, 40).ToArray();
        model.ResetSequenceState();
        using var a = model.Forward(ids, pos, -1, null, true);
        var la = Row(a, 0, model.Config.VocabSize);
        var o = Inkling2VisionEndToEndTests.LoadOracle("rainbow80");
        Skip.If(o is null, "vision oracle not generated");
        using var emb = enc.Encode(o!.Image);
        model.ResetSequenceState();
        using var b = model.ForwardWithImages(ids, pos, -1, [new ImageSpan(5000, emb)], null, true);   // span far past this chunk
        Assert.Equal(la, Row(b, 0, model.Config.VocabSize));
        model.ResetSequenceState();
        using var c = model.ForwardWithImages(ids, pos, -1, [], null, true);
        Assert.Equal(la, Row(c, 0, model.Config.VocabSize));
    }

    // ───────────── part B: the Vulkan ViT encoder feeding the Vulkan text model ─────────────

    private VulkanInkling2VisionEncoder VkEncoder()
    {
        Models();   // device + model (skips when unavailable)
        return _vkEncoder ??= VulkanInkling2VisionEncoder.Load(_dev!, Inkling2VisionFixtures.FindMmproj()!, Inkling2VulkanTests.SpvDir());
    }

    /// <summary>Teacher-forced KL of the text logits after an image prompt encoded ON THE GPU, against llama.cpp (gate 0.015; CPU encoder + Vulkan text: 4.9e-3 to 6.6e-3 in section 12a).</summary>
    [SkippableTheory]
    [InlineData("rainbow80")]
    [InlineData("rainbow190x110")]
    [InlineData("rainbow640")]
    public void VulkanEncoder_ImagePrompt_Logits_MatchLlamaCpp(string name)
    {
        var o = Inkling2VisionEndToEndTests.LoadOracle(name);
        Skip.If(o is null, "vision oracle not generated");
        var (model, _) = Models();
        using var emb = VkEncoder().Encode(o!.Image);
        Assert.Equal(o.ImageTokens, emb.Tokens);
        var (rows, mism, _) = RunRows(o, model, emb, 32);
        var s = Score(o, rows);
        _output.WriteLine($"{name} (Vulkan ViT + Vulkan text): mean KL {s.Mean:E2}, max KL {s.Max:E2}, top-1 {s.Top1}/{rows.Length}, teacher-forced greedy mismatches {mism}/{o.Gen.Length}");
        Assert.True(s.Mean < Gate, $"mean KL {s.Mean:E2} (gate {Gate:E1})");
        Assert.True(s.Top1 >= rows.Length - 2, $"top-1 agreement {s.Top1}/{rows.Length}");
    }

    [SkippableTheory]
    [InlineData("rainbow80")]
    [InlineData("rainbow190x110")]
    [InlineData("rainbow640")]
    public void VulkanEncoder_SinglePassPrefill_MatchesLlamaCpp(string name)
    {
        var o = Inkling2VisionEndToEndTests.LoadOracle(name);
        Skip.If(o is null, "vision oracle not generated");
        var (model, _) = Models();
        using var emb = VkEncoder().Encode(o!.Image);
        var (rows, mism, secs) = RunRows(o, model, emb, 100_000);
        var s = Score(o, rows);
        _output.WriteLine($"{name} (Vulkan ViT) single pass: mean KL {s.Mean:E2}, max KL {s.Max:E2}, top-1 {s.Top1}/{rows.Length}, greedy mismatches {mism}; prefill {secs:F2} s");
        Assert.True(s.Mean < Gate, $"mean KL {s.Mean:E2} (gate {Gate:E1})");
        Assert.True(s.Top1 >= rows.Length - 2, $"top-1 agreement {s.Top1}/{rows.Length}");
    }

    /// <summary>Chunk-size invariance with Vulkan-encoded embeddings: image-prompt logits KL vs the 32-token run must be 0 (the CPU is 0 after the conv fix).</summary>
    [SkippableTheory]
    [InlineData("rainbow80", 5)]
    [InlineData("rainbow190x110", 7)]
    [InlineData("rainbow190x110", 4)]
    public void VulkanEncoder_ChunkedPrefill_ImageLogits_AreChunkSizeInvariant(string name, int chunk)
    {
        var o = Inkling2VisionEndToEndTests.LoadOracle(name);
        Skip.If(o is null, "vision oracle not generated");
        var (model, _) = Models();
        using var emb = VkEncoder().Encode(o!.Image);
        var reference = RunRows(o, model, emb, 32).Rows[0];
        var chunked = RunRows(o, model, emb, chunk).Rows[0];
        var (kl, same) = Inkling2VisionEndToEndTests.FullKl(reference, chunked);
        _output.WriteLine($"{name} chunk {chunk}: image KL vs chunk 32 {kl:E2} (same top-1 {same})");
        Assert.True(same);
        Assert.True(kl < 1e-5, $"chunk {chunk} moved the image prompt logits by KL {kl:E2}");
    }

    /// <summary>
    /// Time to first token, Vulkan encoder vs CPU encoder, text model on Vulkan in both: TTFT = encode + whole-prompt prefill (+ greedy pick). The CPU encode times are the cached wall times of
    /// <c>CpuCache_Populate</c> (the CPU and the iGPU are never busy together); the prefill is measured here (warm, best of runs) and shared by both columns.
    /// </summary>
    [SkippableFact]
    public void Ttft_VulkanEncoderVsCpuEncoder()
    {
        string tf = Path.Combine(Inkling2VulkanVisionEncoderTests.CacheDir, "cpu_times.txt");
        Skip.IfNot(File.Exists(tf), "CPU cache not populated");
        var cpuTimes = File.ReadAllLines(tf).Select(l => l.Split(' ')).Where(p => p[0] == "cpu_bf16").ToDictionary(p => p[1], p => double.Parse(p[3], System.Globalization.CultureInfo.InvariantCulture));
        var (model, _) = Models();
        var enc = VkEncoder();
        var template = Inkling2VisionEndToEndTests.LoadOracle("rainbow80");
        Skip.If(template is null, "vision oracle not generated");
        foreach (string name in new[] { "rainbow80", "rainbow190x110", "rainbow640", "noise2560" })
        {
            var image = Inkling2VulkanVisionEncoderTests.LoadImage(name);
            var o = name == "noise2560" ? template! : Inkling2VisionEndToEndTests.LoadOracle(name)!;
            using (var warm = enc.Encode(image)) { }
            double bestEnc = double.MaxValue, bestPrefill = double.MaxValue; int tokens = 0, promptLen = 0;
            for (int rep = 0; rep < (name == "noise2560" ? 1 : 2); rep++)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                using var emb = enc.Encode(image);
                double e = sw.Elapsed.TotalSeconds;
                tokens = emb.Tokens;
                var prompt = o.Before.Concat(Enumerable.Repeat(200054, emb.Tokens)).Concat(o.After).ToArray(); promptLen = prompt.Length;
                model.ResetSequenceState();
                sw.Restart();
                using var t = model.ForwardWithImages(prompt, Enumerable.Range(0, prompt.Length).ToArray(), -1, [new ImageSpan(o.Before.Length, emb)], null, lastTokenLogitsOnly: true);
                _ = Row(t, 0, model.Config.VocabSize);
                double pf = sw.Elapsed.TotalSeconds;
                bestEnc = Math.Min(bestEnc, e); bestPrefill = Math.Min(bestPrefill, pf);
            }
            double cpu = cpuTimes.TryGetValue(name, out var c) ? c : double.NaN;
            _output.WriteLine($"TTFT {name}: {tokens} image tokens, prompt {promptLen}; encode Vulkan {bestEnc:F3} s vs CPU {cpu:F3} s; prefill+first token {bestPrefill:F2} s; TTFT Vulkan encoder {bestEnc + bestPrefill:F2} s vs CPU encoder {cpu + bestPrefill:F2} s");
        }
    }

    /// <summary>
    /// MEASUREMENT: the floor for any encoder that does not replicate llama.cpp's Q8_0 ACTIVATION quantisation. The same Vulkan text model, fed the cached embeddings of the CPU encoder on the F32-weights
    /// control (same weight values, F32 activations: the exact tower, no activation noise), scored against the llama.cpp logits. Compare with the shipping CPU encoder (which shares the reference's Q8_0
    /// activation rounding, hence its low KL) and with the Vulkan encoder (BF16 activations, no activation quantisation).
    /// </summary>
    [SkippableTheory]
    [InlineData("rainbow80")]
    [InlineData("rainbow190x110")]
    [InlineData("rainbow640")]
    public void Report_ExactTowerEmbeddings_KlFloor(string name)
    {
        string f = Path.Combine(Inkling2VulkanVisionEncoderTests.CacheDir, $"cpu_ctrl_f32_{name}.f32");
        Skip.IfNot(File.Exists(f), "CPU cache not populated");
        var o = Inkling2VisionEndToEndTests.LoadOracle(name);
        Skip.If(o is null, "vision oracle not generated");
        var (model, _) = Models();
        var bytes = File.ReadAllBytes(f);
        var vals = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, vals, 0, bytes.Length);
        int gwTok = (o!.Image.Width + 39) / 40, ghTok = (o.Image.Height + 39) / 40;
        Assert.Equal(gwTok * ghTok * 2304, vals.Length);
        using var emb = new ImageEmbeddings(gwTok, ghTok, 2304, "ctrl");
        for (int i = 0; i < vals.Length; i++) emb.Rows[i] = (Half)vals[i];
        var (rows, mism, _) = RunRows(o, model, emb, 32);
        var s = Score(o, rows);
        _output.WriteLine($"{name} (exact-tower embeddings from the F32 control, Vulkan text): mean KL {s.Mean:E2}, max KL {s.Max:E2}, top-1 {s.Top1}/{rows.Length}, greedy mismatches {mism}");
    }
}
