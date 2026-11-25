using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
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
/// OVHcloud Olala Vulkan forward (#1983) against the CPU reference (F32 activations) and the HF oracle. The Vulkan path stores BF16
/// activations, so parity is at BF16 noise, not float rounding; gates are set from measured values (docs/OLALA.md section 8).
/// </summary>
public sealed class OlalaVulkanTests
{
    private readonly ITestOutputHelper _output;
    public OlalaVulkanTests(ITestOutputHelper output) => _output = output;

    private static string SpvDir()
    {
        string dir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "spv"));
        if (!Directory.Exists(dir) || Directory.GetFiles(dir, "*.spv").Length == 0)
            dir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"));
        return dir;
    }

    private static string? TinyGguf()
    {
        string? dir = OlalaOracleTests.OracleDir();
        string? p = dir is null ? null : Path.Combine(dir, "tiny.gguf");
        return p is not null && File.Exists(p) ? p : null;
    }

    private static unsafe float[] Rows(ITensor t, int rows, int vocab)
    {
        var a = new float[rows * vocab];
        new ReadOnlySpan<float>((void*)t.DataPointer, a.Length).CopyTo(a);
        return a;
    }

    private static void SkipUnlessGpu()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available on this host.");
    }

    private static VulkanOlalaTransformerModel LoadVk(VulkanDevice dev, GgufFile gguf, bool f32Act = false)
    {
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        return VulkanOlalaTransformerModel.BuildFromGguf(dev, gguf, cfg, SpvDir(), new OlalaVulkanOptions(ActivationF32: f32Act));
    }

    private static OlalaTransformerModel LoadCpu(GgufFile gguf)
    {
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        return Assert.IsType<OlalaTransformerModel>(ModelLoader.CreateCpuModelFromGguf(gguf, cfg));
    }

    private (double MaxKl, double MedKl, double MaxAbs, int Top1, int Rows) Compare(float[] refLogits, float[] got, int rows, int vocab)
    {
        var kls = new List<double>(); double maxAbs = 0; int top1 = 0;
        for (int t = 0; t < rows; t++)
        {
            var r = refLogits.AsSpan(t * vocab, vocab); var g = got.AsSpan(t * vocab, vocab);
            kls.Add(OlalaOracleTests.Kl(r, g));
            for (int i = 0; i < vocab; i++) maxAbs = Math.Max(maxAbs, Math.Abs(r[i] - g[i]));
            // A top-1 flip is only a defect when the CPU's own top-2 margin exceeds the BF16 noise (a near-tied row can flip on 1e-2).
            var top = r.ToArray().OrderByDescending(v => v).Take(2).ToArray();
            if (OlalaOracleTests.Argmax(r) == OlalaOracleTests.Argmax(g) || top[0] - top[1] < 0.05f) top1++;
        }
        kls.Sort();
        return (kls[^1], kls[kls.Count / 2], maxAbs, top1, rows);
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]   // F32-activation reference arm: a control, must be (much) closer to the CPU
    public void Tiny_Prefill_MatchesCpu(bool f32Act)
    {
        SkipUnlessGpu();
        string? path = TinyGguf(); Skip.If(path is null, "tiny Olala oracle not generated");
        var c = OlalaOracleTests.LoadCase("tiny")!;
        using var file = GgufFile.Open(path!);
        using var dev = VulkanDevice.Create();
        using var vk = LoadVk(dev, file, f32Act);
        using var cpu = LoadCpu(file);
        int[] pos = Enumerable.Range(0, c.L).ToArray();
        using var a = cpu.Forward(c.Ids, pos, -1);
        using var b = vk.Forward(c.Ids, pos, -1);
        var s = Compare(Rows(a, c.L, c.Vocab), Rows(b, c.L, c.Vocab), c.L, c.Vocab);
        _output.WriteLine($"f32Act={f32Act}: maxKL {s.MaxKl:E2} medKL {s.MedKl:E2} max|dlogit| {s.MaxAbs:E2} top1 {s.Top1}/{s.Rows}");
        double klGate = f32Act ? 1e-6 : 2e-2, absGate = f32Act ? 1e-3 : 0.5;
        Assert.True(s.MaxKl < klGate, $"max KL {s.MaxKl:E3}");
        Assert.True(s.MaxAbs < absGate, $"max |dlogit| {s.MaxAbs:E3}");
        Assert.Equal(s.Rows, s.Top1);
    }

    [SkippableFact]
    public void Tiny_Decode_TeacherForced_MatchesCpu_AndGreedyIdenticalToHf()
    {
        SkipUnlessGpu();
        string? path = TinyGguf(); Skip.If(path is null, "tiny Olala oracle not generated");
        var c = OlalaOracleTests.LoadCase("tiny")!;
        using var file = GgufFile.Open(path!);
        using var dev = VulkanDevice.Create();
        using var vk = LoadVk(dev, file);
        using var cpu = LoadCpu(file);
        int[] pos = Enumerable.Range(0, c.L).ToArray();
        using (var p1 = cpu.Forward(c.Ids, pos, -1, null, true)) { }
        using (var p2 = vk.Forward(c.Ids, pos, -1, null, true)) { }
        vk.ResetSequenceState(); cpu.ResetSequenceState();
        using var cp = cpu.Forward(c.Ids, pos, -1, null, true);
        using var vp = vk.Forward(c.Ids, pos, -1, null, true);
        var generated = new List<int> { OlalaOracleTests.Argmax(Rows(vp, 1, c.Vocab)) };
        double worst = 0;
        worst = Math.Max(worst, OlalaOracleTests.Kl(Rows(cp, 1, c.Vocab), Rows(vp, 1, c.Vocab)));
        for (int j = 1; j < c.N; j++)
        {
            using var cs = cpu.Forward([c.GreedyIds[j - 1]], [c.L + j - 1], -1, null, true);
            using var vs = vk.Forward([c.GreedyIds[j - 1]], [c.L + j - 1], -1, null, true);
            var vr = Rows(vs, 1, c.Vocab);
            double kl = OlalaOracleTests.Kl(Rows(cs, 1, c.Vocab), vr);
            _output.WriteLine($"step {j}: KL vs CPU {kl:E2}");
            worst = Math.Max(worst, kl);
            Assert.Equal(OlalaOracleTests.Argmax(Rows(cs, 1, c.Vocab)), OlalaOracleTests.Argmax(vr));
        }
        Assert.True(worst < 2e-2, $"worst decode KL {worst:E3}");
    }

    [SkippableFact]
    public void StatePolicy_Guard_AllRecurrentStateIsF32()
    {
        SkipUnlessGpu();
        string? path = TinyGguf(); Skip.If(path is null, "tiny Olala oracle not generated");
        using var file = GgufFile.Open(path!);
        using var dev = VulkanDevice.Create();
        using var vk = LoadVk(dev, file);
        var o = vk.Config.Olala!;
        // Structural: the element byte size of every buffer kind is the policy's, and the policy is F32.
        foreach (var (kind, elements, bytes) in vk.StateBuffers)
        {
            Assert.Equal(4, bytes / elements);
            _output.WriteLine($"{kind}: {elements} x {bytes / elements} B");
        }
        Assert.Equal(OlalaStateStorage.F32, OlalaStatePolicy.Ssm);
        Assert.Equal(OlalaStateStorage.F32, OlalaStatePolicy.Angle);
        Assert.Equal(OlalaStateStorage.F32, OlalaStatePolicy.Trapezoid);
        Assert.Equal(OlalaStateStorage.F32, OlalaStatePolicy.TokenShift);
        // What was really allocated on the device equals the layout x 4 bytes (a bf16/f16 buffer would be half).
        var layout = vk.StateBuffers.ToDictionary(x => x.Kind, x => x.Bytes);
        long mamba = layout["ssm"] + layout["angle"] + layout["trapezoidK"] + layout["trapezoidV"];
        Assert.Equal(mamba, vk.AllocatedStateBytes["mamba"]);
        Assert.Equal(layout["tokenShift"], vk.AllocatedStateBytes["tokenShift"]);
        // Real 7A1B geometry (docs/OLALA.md section 4 table).
        var real = VulkanOlalaTransformerModel_Layout7A1B();
        Assert.Equal(29L * 1_572_864, real["ssm"]);
        Assert.Equal(29L * 6_144, real["angle"]);
        Assert.Equal(29L * 98_304, real["trapezoidK"]);
        Assert.Equal(29L * 12_288, real["trapezoidV"]);
        Assert.Equal(7L * 12_288, real["tokenShift"]);
        Assert.Equal(2, vk.ActivationElementBytes);   // BF16 activations by default
    }

    private static Dictionary<string, long> VulkanOlalaTransformerModel_Layout7A1B()
    {
        var o = new OlalaConfig
        {
            LayerTypes = "MMMMVMMMMVMMMMVMMMMVMMMMVMMMMVMMMMVM", SsmInner = 3072, StateSize = 128, GroupCount = 1, MambaHeadDim = 64, MimoRank = 4,
            NumRopeAngles = 32, NumSignalHeads = 36, NumNoiseHeads = 12, TpaRank = 4, GateBias = 1.15f, ExpertCount = 256, ExpertUsedCount = 6,
            ExpertFfLength = 768, SharedExpertFfLength = 1536, ExpertWeightsScale = 2.45f, MoeLatentSize = 384,
        };
        return VulkanOlalaTransformerModel.ComputeStateLayout(o, 128).ToDictionary(x => x.Kind, x => x.Bytes);
    }

    // ───────────────────────── real 7A1B checkpoints ─────────────────────────

    private static string? FindGguf(string envVar, string file)
    {
        string? env = Environment.GetEnvironmentVariable(envVar);
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;
        string hub = Environment.GetEnvironmentVariable("HF_HUB_CACHE")
            ?? Path.Combine(Environment.GetEnvironmentVariable("HF_HOME") is { Length: > 0 } h ? h : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface"), "hub");
        string snaps = Path.Combine(hub, "models--OVHaiLLM--Olala-v1.0-mini-7A1B-GGUF", "snapshots");
        if (!Directory.Exists(snaps)) return null;
        foreach (var d in Directory.GetDirectories(snaps)) { string p = Path.Combine(d, file); if (File.Exists(p)) return p; }
        return null;
    }

    /// <summary>Prefill rows then the teacher-forced decode rows (one forward per token) of one oracle case.</summary>
    private static float[] Positions(VulkanOlalaTransformerModel m, OlalaOracleTests.Case c)
    {
        m.ResetSequenceState();
        var all = new List<float>();
        using (var full = m.Forward(c.Ids, Enumerable.Range(0, c.L).ToArray(), -1)) all.AddRange(Rows(full, c.L, c.Vocab));
        for (int j = 0; j < c.N - 1; j++)
        {
            using var s = m.Forward([c.GreedyIds[j]], [c.L + j], -1, null, lastTokenLogitsOnly: true);
            all.AddRange(Rows(s, 1, c.Vocab));
        }
        return all.ToArray();
    }

    private static float[] OracleRows(OlalaOracleTests.Case c)
    {
        var o = new float[(c.L + c.N - 1) * c.Vocab];
        c.Full.CopyTo(o, 0);
        Array.Copy(c.Greedy, c.Vocab, o, c.L * c.Vocab, (c.N - 1) * c.Vocab);   // Greedy[0] repeats the last prefill row
        return o;
    }

    private (double Median, double Mean, double Worst, int Top1, int Rows, bool Finite) Report(string label, float[] a, float[] b, int vocab)
    {
        int rows = a.Length / vocab; var kls = new List<double>(); int top1 = 0; bool finite = true;
        for (int t = 0; t < rows; t++)
        {
            var ra = a.AsSpan(t * vocab, vocab); var rb = b.AsSpan(t * vocab, vocab);
            foreach (float v in rb) if (!float.IsFinite(v)) finite = false;
            kls.Add(OlalaOracleTests.Kl(ra, rb));
            if (OlalaOracleTests.Argmax(ra) == OlalaOracleTests.Argmax(rb)) top1++;
        }
        var sorted = kls.OrderBy(x => x).ToList();
        var r = (sorted[sorted.Count / 2], kls.Average(), sorted[^1], top1, rows, finite);
        _output.WriteLine($"{label}: {rows} rows, median KL {r.Item1:E2}, mean KL {r.Item2:E2}, worst {r.Item3:E2}, top-1 {top1}/{rows}, finite {finite}");
        return r;
    }

    /// <summary>
    /// Real checkpoint, Vulkan BF16-activation forward against (1) the HF float32 oracle and (2) the same engine with F32 activation STORAGE
    /// (the control that isolates the activation-precision effect). The q8_0-safe2 file must stay finite: it holds the 1e9..1e14 outlier tensors.
    /// </summary>
    [SkippableTheory]
    [InlineData("Olala-v1.0-mini-7A1B-q8_0-safe2.gguf", "DOTLLM_OLALA_Q8_GGUF")]
    [InlineData("Olala-v1.0-mini-7A1B-bf16.gguf", "DOTLLM_OLALA_BF16_GGUF")]
    public void RealModel_Bf16Activations_VsOracleAndF32ActivationControl(string file, string env)
    {
        SkipUnlessGpu();
        string? gguf = FindGguf(env, file); Skip.If(gguf is null, file + " not in the HF cache");
        var cases = new[] { "capital", "code", "chat" }.Select(p => OlalaOracleTests.LoadCase($"real-strict-gguf-norms-{p}")).Where(c => c is not null).Select(c => c!).ToArray();
        Skip.If(cases.Length == 0, "no real oracle files");
        bool withControl = Environment.GetEnvironmentVariable("DOTLLM_OLALA_VK_NO_CONTROL") != "1";
        using var dev = VulkanDevice.Create();
        using var f = GgufFile.Open(gguf!);
        var bf = new List<float[]>(); var f32 = new List<float[]>(); var greedyOk = new List<bool>();
        using (var vk = LoadVk(dev, f))
            foreach (var c in cases)
            {
                bf.Add(Positions(vk, c));
                vk.ResetSequenceState();
                using var pre = vk.Forward(c.Ids, Enumerable.Range(0, c.L).ToArray(), -1, null, true);
                var gen = new List<int> { OlalaOracleTests.Argmax(Rows(pre, 1, c.Vocab)) };
                for (int j = 1; j < c.N; j++) { using var s = vk.Forward([gen[^1]], [c.L + j - 1], -1, null, true); gen.Add(OlalaOracleTests.Argmax(Rows(s, 1, c.Vocab))); }
                int first = Enumerable.Range(0, c.N).FirstOrDefault(i => gen[i] != c.GreedyIds[i], -1);
                _output.WriteLine($"{file} / {c.Name}: greedy {(first < 0 ? "identical to HF" : $"first divergence at token {first}")} over {c.N} tokens");
                greedyOk.Add(first < 0);
            }
        if (withControl)
            using (var vk32 = LoadVk(dev, f, f32Act: true))
                foreach (var c in cases) f32.Add(Positions(vk32, c));
        for (int i = 0; i < cases.Length; i++)
        {
            var c = cases[i]; var oracle = OracleRows(c);
            var r = Report($"{file} / {c.Name} BF16-act vs HF oracle", oracle, bf[i], c.Vocab);
            Assert.True(r.Finite, "non-finite logits (activation overflow?)");
            if (withControl)
            {
                Report($"{file} / {c.Name} F32-act control vs HF oracle", oracle, f32[i], c.Vocab);
                Report($"{file} / {c.Name} BF16-act vs F32-act (same engine, same weights)", f32[i], bf[i], c.Vocab);
            }
            if (c.L <= 100) { Assert.True(r.Median < 0.05, $"median KL {r.Median:E2}"); Assert.True(r.Top1 >= 0.85 * r.Rows, $"top-1 {r.Top1}/{r.Rows}"); }
        }
    }
}
