using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Architectures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Teacher-forced oracle gates, KV-format comparison and throughput of the Inkling 2 Vulkan forward (#2034) against the cached llama.cpp b12780 CPU
/// oracle (<c>~/.dotllm/test-cache/token-oracles/inkling2/</c>). The CPU reference's numbers are in docs/INKLING2.md section 10. Hold the Strix GPU lock.
/// </summary>
public sealed class Inkling2VulkanGateTests
{
    private readonly ITestOutputHelper _output;
    public Inkling2VulkanGateTests(ITestOutputHelper output) => _output = output;

    private static void SkipUnlessGpu()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available on this host.");
    }

    private static unsafe float[] Rows(ITensor t, int rows, int vocab)
    {
        var a = new float[rows * vocab];
        new ReadOnlySpan<float>((void*)t.DataPointer, a.Length).CopyTo(a);
        return a;
    }

    private static VulkanInkling2TransformerModel Build(VulkanDevice dev, GgufFile file, ModelConfig cfg, Inkling2VulkanOptions opt)
        => VulkanInkling2TransformerModel.BuildFromGguf(dev, file, cfg, Inkling2VulkanTests.SpvDir(), opt);

    private static int IntEnv(string name, int def) => int.TryParse(Environment.GetEnvironmentVariable(name), out int v) && v > 0 ? v : def;

    private static Inkling2KvFormat Kv(string kv) => kv == "q8_0" ? Inkling2KvFormat.Q8_0 : Inkling2KvFormat.F16;

    private (double Mean, double Max, double Top1, int N, double MeanLow, double MeanHigh) ScoreKl(VulkanInkling2TransformerModel model, Inkling2RealModelTests.KlRef r, int sub)
    {
        int first = r.NCtx / 2, nScored = r.NCtx - 1 - first, V = model.Config.VocabSize;
        double sum = 0, max = 0, lo = 0, hi = 0; int top = 0, n = 0, nlo = 0, nhi = 0;
        for (int chunk = 0; chunk < r.NChunk; chunk++)
        {
            var baseRows = Inkling2RealModelTests.ReadChunk(r, chunk);
            var ids = r.Tokens.AsSpan(chunk * r.NCtx, r.NCtx).ToArray();
            model.ResetSequenceState();
            var got = new float[r.NCtx * V];
            for (int at = 0; at < r.NCtx; at += sub)
            {
                int len = Math.Min(sub, r.NCtx - at);
                using var part = model.Forward(ids.AsSpan(at, len), Enumerable.Range(at, len).ToArray(), -1);
                Rows(part, len, V).CopyTo(got, at * V);
            }
            for (int i = 0; i < nScored; i++)
            {
                var (kl, same) = Inkling2RealModelTests.Kl(baseRows[i], got.AsSpan((first + i) * V, V));
                sum += kl; max = Math.Max(max, kl); if (same) top++; n++;
                if (first + i < 1024) { lo += kl; nlo++; } else { hi += kl; nhi++; }
            }
        }
        return (sum / n, max, (double)top / n, n, nlo > 0 ? lo / nlo : 0, nhi > 0 ? hi / nhi : 0);
    }

    /// <summary>CPU reference: mean KL 4.10e-3, top-1 98.8% on the passage; 1.76e-3 below / 1.59e-3 above position 1024 on the 1280 windows; floor-1 log scale 4.52e-3.</summary>
    [SkippableTheory]
    [InlineData("nvfp4-passage-c256-fa-off.kld", "f16", false, 0)]
    [InlineData("nvfp4-passage-c256-fa-off.kld", "q8_0", false, 0)]
    [InlineData("nvfp4-passage-c256-fa-off.kld", "f16", true, 0)]
    [InlineData("nvfp4-passage-c256-fa-off-logscale-floor1.kld", "f16", false, 1)]
    [InlineData("nvfp4-wikitext-c1280-fa-off.kld", "f16", false, 0)]
    [InlineData("nvfp4-wikitext-c1280-fa-off.kld", "q8_0", false, 0)]
    public void Oracle_Gate(string kld, string kv, bool f32Act, int logFloor)
    {
        SkipUnlessGpu();
        string? gguf = Inkling2RealModelTests.FindGguf(); var r = Inkling2RealModelTests.LoadKlRef(kld);
        Skip.If(gguf is null || r is null, "model or oracle missing");
        using var file = GgufFile.Open(gguf!);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        if (logFloor > 0) cfg = cfg with { Inkling2 = cfg.Inkling2! with { LogScaleFloor = logFloor } };
        using var dev = VulkanDevice.Create();
        var opt = new Inkling2VulkanOptions(f32Act, Kv(kv), IntEnv("DOTLLM_INKLING2_VK_CHUNK", 128));
        using var vk = Build(dev, file, cfg, opt);
        int sub = IntEnv("DOTLLM_INKLING2_KL_SUBCHUNK", 32);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var s = ScoreKl(vk, r!, sub);
        _output.WriteLine($"GATE {kld} kv={kv} f32act={f32Act} chunk={opt.MaxChunk} sub={sub}: mean KL {s.Mean:E3} max {s.Max:E3} top-1 {s.Top1:P1} n={s.N}; KL pos<1024 {s.MeanLow:E3}, >=1024 {s.MeanHigh:E3}; {sw.Elapsed.TotalSeconds:F0}s");
        _output.WriteLine($"KV now {vk.KvBytes / 1e6:F1} MB; per global position {vk.KvBytesPerPosition} B x {vk.GlobalSlots} global slots; local ring {vk.RingPositions} positions x {vk.LocalSlots} slots; conv {vk.ConvStateBytes / 1e6:F2} MB");
        Assert.True(s.Mean < 8e-3, $"mean KL {s.Mean:E3}");
        if (!kld.Contains("1280")) Assert.True(s.Top1 > 0.95, $"top-1 {s.Top1:P1}");
    }

    [SkippableTheory]
    [InlineData("f16")]
    [InlineData("q8_0")]
    public void Greedy_TeacherForced(string kv)
    {
        SkipUnlessGpu();
        string? gguf = Inkling2RealModelTests.FindGguf();
        Skip.If(gguf is null, "model missing");
        string path = Path.Combine(Inkling2RealModelTests.OracleDir, "greedy-nvfp4-fa-off.json");
        Skip.IfNot(File.Exists(path), "greedy oracle missing");
        using var file = GgufFile.Open(gguf!);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        using var dev = VulkanDevice.Create();
        using var vk = Build(dev, file, cfg, new Inkling2VulkanOptions(false, Kv(kv)));
        using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(path));
        int steps = 0, mism = 0, mismConf = 0, conf = 0;
        foreach (var p in doc.RootElement.EnumerateObject())
        {
            int[] prompt = p.Value.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            int[] want = p.Value.GetProperty("gen_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            double[] margin = p.Value.GetProperty("margins").EnumerateArray().Select(e => e.GetDouble()).ToArray();
            vk.ResetSequenceState();
            using var pre = vk.Forward(prompt, Enumerable.Range(0, prompt.Length).ToArray(), -1, null, true);
            int arg = Inkling2RealModelTests.ArgMax(Rows(pre, 1, cfg.VocabSize));
            for (int j = 0; j < want.Length; j++)
            {
                steps++; bool c = margin[j] >= 0.3; if (c) conf++;
                if (arg != want[j]) { mism++; if (c) mismConf++; }
                if (j + 1 == want.Length) break;
                using var st = vk.Forward([want[j]], [prompt.Length + j], -1, null, true);
                arg = Inkling2RealModelTests.ArgMax(Rows(st, 1, cfg.VocabSize));
            }
        }
        _output.WriteLine($"GREEDY kv={kv}: teacher-forced mismatches {mism}/{steps} ({mismConf}/{conf} at margin >= 0.3); CPU reference 2/384 (0/345)");
        Assert.Equal(0, mismConf);
    }

    /// <summary>Prefill (300 tokens) and decode throughput, printed per round (round 0 is the warm-up).</summary>
    [SkippableTheory]
    [InlineData("f16")]
    [InlineData("q8_0")]
    public void Throughput(string kv)
    {
        SkipUnlessGpu();
        string? gguf = Inkling2RealModelTests.FindGguf(); var r = Inkling2RealModelTests.LoadKlRef("nvfp4-wikitext-c1280-fa-off.kld");
        Skip.If(gguf is null || r is null, "model or oracle missing");
        using var file = GgufFile.Open(gguf!);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        using var dev = VulkanDevice.Create();
        using var vk = Build(dev, file, cfg, new Inkling2VulkanOptions(false, Kv(kv), IntEnv("DOTLLM_INKLING2_VK_CHUNK", 128)));
        int[] prompt = r!.Tokens.AsSpan(0, 300).ToArray();
        for (int round = 0; round < 4; round++)
        {
            vk.ResetSequenceState();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            using (vk.Forward(prompt, Enumerable.Range(0, prompt.Length).ToArray(), -1, null, true)) { }
            double pre = sw.Elapsed.TotalSeconds;
            int next = 1234, steps = 32;
            sw.Restart();
            for (int j = 0; j < steps; j++)
            {
                using var t = vk.Forward([next], [prompt.Length + j], -1, null, true);
                next = Inkling2RealModelTests.ArgMax(Rows(t, 1, cfg.VocabSize));
            }
            double dec = sw.Elapsed.TotalSeconds;
            _output.WriteLine($"THROUGHPUT kv={kv} round {round}{(round == 0 ? " (warm-up)" : "")}: prefill {prompt.Length} tok {prompt.Length / pre:F1} tok/s; decode {steps / dec:F2} tok/s ({dec / steps * 1000:F0} ms/token)");
        }
    }
}
