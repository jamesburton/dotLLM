using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Architectures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Inkling-2-Mini-NVFP4 Vulkan forward (#2034) against the CPU reference and the cached llama.cpp CPU oracle. Heavy (17.6 GB model); skipped without the
/// model, the oracle or a Vulkan device. The caller holds the Strix GPU lock.
/// </summary>
public sealed class Inkling2VulkanTests
{
    private readonly ITestOutputHelper _output;
    public Inkling2VulkanTests(ITestOutputHelper output) => _output = output;

    internal static string SpvDir()
    {
        string dir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "spv"));
        if (!Directory.Exists(dir) || Directory.GetFiles(dir, "*.spv").Length == 0)
            dir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"));
        return dir;
    }

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

    private static double KlRows(ReadOnlySpan<float> p, ReadOnlySpan<float> q)
    {
        double mp = double.MinValue, mq = double.MinValue;
        for (int i = 0; i < p.Length; i++) { mp = Math.Max(mp, p[i]); mq = Math.Max(mq, q[i]); }
        double sp = 0, sq = 0;
        for (int i = 0; i < p.Length; i++) { sp += Math.Exp(p[i] - mp); sq += Math.Exp(q[i] - mq); }
        double lp = mp + Math.Log(sp), lq = mq + Math.Log(sq), kl = 0;
        for (int i = 0; i < p.Length; i++) { double pp = Math.Exp(p[i] - lp); if (pp > 1e-12) kl += pp * ((p[i] - lp) - (q[i] - lq)); }
        return kl;
    }

    /// <summary>The Vulkan forward against the CPU reference on the same token ids (no llama.cpp noise). The F32-activation arm is a control.</summary>
    [SkippableTheory]
    [InlineData(true, 32)]
    [InlineData(false, 32)]
    public void Real_Prefill_MatchesCpuReference(bool f32Act, int n)
    {
        SkipUnlessGpu();
        string? gguf = Inkling2RealModelTests.FindGguf(); Skip.If(gguf is null, "Inkling-2-Mini-NVFP4.gguf not in the HF cache");
        using var file = GgufFile.Open(gguf!);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        string passage = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache", "token-oracles", "inkling2", "greedy-nvfp4-fa-off.json");
        Skip.IfNot(File.Exists(passage), "oracle ids missing");
        using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(passage));
        int[] ids = doc.RootElement.EnumerateObject().First().Value.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32()).Take(n).ToArray();
        int[] pos = Enumerable.Range(0, ids.Length).ToArray();
        var cpuTrace = new Dictionary<string, float[]>();
        float[] cpuLogits;
        using (var cpu = Inkling2TransformerModel.LoadFromGguf(file, cfg, ThreadingConfig.Auto))
        {
            cpu.Trace = (name, data, r, c) => { if (name == "embed" || name == "result_norm" || name.StartsWith("pass")) cpuTrace[name] = data.ToArray(); };
            using var t = cpu.Forward(ids, pos, -1);
            cpuLogits = Rows(t, ids.Length, cfg.VocabSize);
        }
        using var dev = VulkanDevice.Create();
        using var vk = VulkanInkling2TransformerModel.BuildFromGguf(dev, file, cfg, SpvDir(), new Inkling2VulkanOptions(ActivationF32: f32Act));
        var vkTrace = new Dictionary<string, float[]>();
        vk.Trace = (name, data, r, c) => vkTrace[name] = data;
        using var g = vk.Forward(ids, pos, -1);
        var got = Rows(g, ids.Length, cfg.VocabSize);
        foreach (var kv in vkTrace)
        {
            if (!cpuTrace.TryGetValue(kv.Key, out var a)) continue;
            double num = 0, den = 0;
            for (int i = 0; i < a.Length; i++) { double d = a[i] - kv.Value[i]; num += d * d; den += (double)a[i] * a[i]; }
            _output.WriteLine($"trace {kv.Key}: rel L2 {Math.Sqrt(num / den):E2}");
        }
        double worst = 0, sum = 0; int top = 0;
        for (int t = 0; t < ids.Length; t++)
        {
            var p = cpuLogits.AsSpan(t * cfg.VocabSize, cfg.VocabSize); var q = got.AsSpan(t * cfg.VocabSize, cfg.VocabSize);
            double kl = KlRows(p, q); worst = Math.Max(worst, kl); sum += kl;
            if (p.ToArray().Select((v, i) => (v, i)).MaxBy(x => x.v).i == q.ToArray().Select((v, i) => (v, i)).MaxBy(x => x.v).i) top++;
        }
        _output.WriteLine($"f32Act={f32Act}: mean KL vs CPU {sum / ids.Length:E3}, max {worst:E3}, top-1 {top}/{ids.Length}");
        Assert.True(sum / ids.Length < (f32Act ? 1e-3 : 8e-3));
    }
}
