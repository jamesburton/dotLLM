using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Architectures;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
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
        // The F32-activation arm is float-rounding identical to the CPU (measured max KL 7e-14, max |dlogit| 1.1e-6), so it is gated near that
        // floor: the K token-shift-unmasked-at-position-0 mutant (only RMSNorm's eps tells it apart) gives KL 1.4e-8 / |dlogit| 4.4e-4.
        double klGate = f32Act ? 1e-9 : 2e-2, absGate = f32Act ? 1e-4 : 0.5;
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
            else
            {
                // The long chat prompt is the only case past 128 keys, i.e. the only one that exercises the cross-block online-softmax
                // rescale of olala_tpa_attn. It is chaotic (docs/OLALA.md 5.1), so it is gated at its noise floor only. Measured: median
                // 1.8e-2 / top-1 281 of 323 (q8_0-safe2), 1.1e-2 / 295 (bf16). The `acc *= corr` mutant gives 0.84-0.93 / 165 of 323.
                Assert.True(r.Median < 0.1, $"long-prompt median KL {r.Median:E2}");
                Assert.True(r.Top1 >= 0.75 * r.Rows, $"long-prompt top-1 {r.Top1}/{r.Rows}");
            }
        }
    }

    // ───────────────────────── kernel-level tests (outlier magnitudes the tiny model cannot produce) ─────────────────────────

    private static ushort Bf16(float v) => OlalaOps.ToBf16(v);
    private static float FromBf16(ushort b) => OlalaOps.FromBf16(b);

    private static VulkanDevice.Buffer UploadBf16(VulkanDevice dev, ReadOnlySpan<float> v)
    {
        var bits = new ushort[v.Length]; for (int i = 0; i < bits.Length; i++) bits[i] = Bf16(v[i]);
        var b = dev.Allocate(Math.Max(16, bits.Length * 2));
        dev.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes<ushort>(bits), b);
        return b;
    }

    private static float[] DownloadBf16(VulkanDevice dev, VulkanDevice.Buffer b, int count)
    {
        var raw = new float[(count + 1) / 2]; dev.Download(b, raw);
        var u = System.Runtime.InteropServices.MemoryMarshal.Cast<float, ushort>(raw.AsSpan());
        var r = new float[count]; for (int i = 0; i < count; i++) r[i] = FromBf16(u[i]); return r;
    }

    /// <summary>
    /// The geodesic update fed mixer outputs of 1e12 (the real model reaches ~1e14): must agree with the CPU double-precision update and stay
    /// finite. An F16-storage activation path turns every such value into inf/NaN, which the tiny model (no outliers) cannot show.
    /// </summary>
    [SkippableTheory]
    [InlineData(1e12f)]
    [InlineData(1e14f)]
    [InlineData(1f)]
    public void GeodesicKernel_WithOutlierMixerOutput_MatchesCpuAndStaysFinite(float outlier)
    {
        SkipUnlessGpu();
        using var dev = VulkanDevice.Create();
        const int H = 1536, T = 3;
        var rng = new Random(7);
        var x = new float[T * H]; var g = new float[T * H];
        for (int t = 0; t < T; t++)
        {
            double ss = 0; for (int i = 0; i < H; i++) { x[t * H + i] = (float)(rng.NextDouble() * 2 - 1); ss += x[t * H + i] * x[t * H + i]; }
            for (int i = 0; i < H; i++) { x[t * H + i] = FromBf16(Bf16(x[t * H + i] / (float)Math.Sqrt(ss))); g[t * H + i] = FromBf16(Bf16((float)(rng.NextDouble() * 2 - 1) * outlier)); }
        }
        using var bx = UploadBf16(dev, x); using var bg = UploadBf16(dev, g);
        using var k = new OlalaKernel(dev, SpvDir(), "olala_geodesic", 2, 4);
        const float scale = 0.9f, bias = 0.05f; const int layer = 5;
        using (var ctx = dev.CreateSubmitContext())
        {
            ctx.Begin();
            k.Record(ctx.CommandBuffer, [bx, bg], [(uint)H, (uint)layer + 1, BitConverter.SingleToUInt32Bits(scale), BitConverter.SingleToUInt32Bits(bias)], T, 1, 1);
            ctx.SubmitAndWait();
        }
        var got = DownloadBf16(dev, bx, T * H);
        double worst = 0;
        for (int t = 0; t < T; t++)
        {
            var want = new float[H];
            OlalaOps.GeodesicUpdate(x.AsSpan(t * H, H), g.AsSpan(t * H, H), want, scale, bias, layer);
            for (int i = 0; i < H; i++)
            {
                Assert.True(float.IsFinite(got[t * H + i]), $"non-finite output at token {t} element {i}");
                worst = Math.Max(worst, Math.Abs(got[t * H + i] - want[i]));
            }
        }
        _output.WriteLine($"outlier {outlier:E0}: max |GPU - CPU| = {worst:E2}");
        Assert.True(worst < 2e-2, $"max |GPU - CPU| {worst:E2}");   // BF16 rounding of a unit vector is ~4e-3 per element
    }

    /// <summary>A BF16-weight GEMV with 1e12-scale BF16 activations (the ReLU^2 / mixer-output inputs of safe2) equals a double reference: no FP16 scale anywhere.</summary>
    [SkippableFact]
    public void GemvKernel_WithOutlierActivations_MatchesDoubleReference()
    {
        SkipUnlessGpu();
        using var dev = VulkanDevice.Create();
        const int M = 40, K = 256, T = 3;
        var rng = new Random(3);
        var w = new float[M * K]; var x = new float[T * K];
        for (int i = 0; i < w.Length; i++) w[i] = FromBf16(Bf16((float)(rng.NextDouble() * 2 - 1)));
        for (int i = 0; i < x.Length; i++) x[i] = FromBf16(Bf16((float)(rng.NextDouble() * 2 - 1) * 1e12f));
        using var bw = UploadBf16(dev, w); using var bx = UploadBf16(dev, x);
        using var by = dev.Allocate(T * M * 2 + 16); using var dummy = dev.Allocate(256);
        using var k = new OlalaKernel(dev, SpvDir(), "olala_gemv_bf16", 5, 4, [8u, 0u, 0u]);
        using (var ctx = dev.CreateSubmitContext())
        {
            ctx.Begin();
            k.Record(ctx.CommandBuffer, [bw, bx, by, dummy, dummy], [(uint)M, (uint)K, (uint)T, 0u], M, 1, 1);
            ctx.SubmitAndWait();
        }
        var got = DownloadBf16(dev, by, T * M);
        for (int t = 0; t < T; t++)
            for (int m = 0; m < M; m++)
            {
                double want = 0; for (int i = 0; i < K; i++) want += (double)w[m * K + i] * x[t * K + i];
                Assert.True(float.IsFinite(got[t * M + m]), $"non-finite at col {t} row {m}");
                Assert.True(Math.Abs(got[t * M + m] - want) <= 1e-2 * Math.Abs(want) + 1e9, $"col {t} row {m}: {got[t * M + m]:E4} vs {want:E4}");
            }
    }

    /// <summary>
    /// The in-place Q8_0 / Q6_K row decoders (<c>olala_wdot.glsl</c>, shared by the GEMV and both MoE kernels) on REAL rows of q8_0-safe2 against
    /// <see cref="Dequantize"/> + a double dot product. The tiny oracle model is BF16, so without this test nothing but the loosely gated real-model
    /// parity sees these decoders (it does catch a swapped Q6_K sub-block scale, but only through the minutes-long q8_0-safe2 run; this test catches it, and a Q8_0 element-order mutant, per type in seconds). 1e9 is the ReLU^2 magnitude feeding ffn_down_exps (Q6_K).
    /// </summary>
    [SkippableTheory]
    [InlineData("blk.0.ffn_down_exps.weight", 1f)]
    [InlineData("blk.0.ffn_down_exps.weight", 1e9f)]
    [InlineData("blk.0.ffn_up_exps.weight", 1f)]
    [InlineData("blk.0.ssm_in.weight", 1f)]
    public unsafe void QuantRowDecoders_OnRealSafe2Rows_MatchDequantizeReference(string tensor, float scale)
    {
        SkipUnlessGpu();
        string? gguf = FindGguf("DOTLLM_OLALA_Q8_GGUF", "Olala-v1.0-mini-7A1B-q8_0-safe2.gguf"); Skip.If(gguf is null, "q8_0-safe2 not in the HF cache");
        using var f = GgufFile.Open(gguf!);
        var d = f.TensorsByName[tensor];
        int K = (int)d.Shape[0];
        const int M = 96, T = 3;
        long rowBytes = Dequantize.RowByteSize(K, d.QuantizationType);
        nint src = f.TensorDataPointer(d);
        var w = new float[(long)M * K];
        Dequantize.ToFloat32(src, (long)M * K, d.QuantizationType, w);
        var rng = new Random(11);
        var x = new float[T * K];
        for (int i = 0; i < x.Length; i++) x[i] = FromBf16(Bf16((float)(rng.NextDouble() * 2 - 1) * scale));
        using var dev = VulkanDevice.Create();
        using var bw = dev.Allocate(rowBytes * M);
        dev.Upload(new ReadOnlySpan<byte>((void*)src, checked((int)(rowBytes * M))), bw);   // M rows only, a test fixture
        using var bx = UploadBf16(dev, x);
        using var byf = dev.Allocate(T * M * 4); using var dummy = dev.Allocate(256);
        string kname = "olala_gemv_" + (d.QuantizationType == QuantizationType.Q6_K ? "q6_k" : "q8_0");
        using var k = new OlalaKernel(dev, SpvDir(), kname, 5, 4, [8u, 0u, 1u]);
        using (var ctx = dev.CreateSubmitContext())
        {
            ctx.Begin();
            k.Record(ctx.CommandBuffer, [bw, bx, dummy, dummy, byf], [(uint)M, (uint)K, (uint)T, 0u], M, 1, 1);
            ctx.SubmitAndWait();
        }
        var got = new float[T * M]; dev.Download(byf, got);
        double worstRel = 0;
        for (int t = 0; t < T; t++)
            for (int m = 0; m < M; m++)
            {
                double want = 0, mag = 0;
                for (int i = 0; i < K; i++) { double p = (double)w[(long)m * K + i] * x[t * K + i]; want += p; mag += Math.Abs(p); }
                double rel = Math.Abs(got[t * M + m] - want) / mag;
                worstRel = Math.Max(worstRel, rel);
                Assert.True(float.IsFinite(got[t * M + m]), $"non-finite at col {t} row {m}");
                Assert.True(rel < 1e-5, $"{tensor} col {t} row {m}: {got[t * M + m]:E6} vs {want:E6} (|err| / sum|w x| = {rel:E2})");
            }
        _output.WriteLine($"{tensor} ({d.QuantizationType}, K {K}) x scale {scale:E0}: worst |err| / sum|w x| = {worstRel:E2}");
    }

    /// <summary>
    /// Warm same-session throughput (opt-in: <c>DOTLLM_OLALA_VK_BENCH=1</c>; hold the GPU lock, no CPU work alongside). Prefill = the 300-token chat
    /// prompt, last-row logits; decode = greedy one-token steps. UMA timing swings ~40% between sessions, so only same-session ranges are quoted.
    /// </summary>
    [SkippableFact]
    public void RealModel_Throughput_WarmRanges()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_OLALA_VK_BENCH") != "1", "set DOTLLM_OLALA_VK_BENCH=1");
        SkipUnlessGpu();
        string file = Environment.GetEnvironmentVariable("DOTLLM_OLALA_BENCH_FILE") ?? "Olala-v1.0-mini-7A1B-q8_0-safe2.gguf";
        string? gguf = FindGguf("DOTLLM_OLALA_Q8_GGUF", file); Skip.If(gguf is null, file + " not in the HF cache");
        var c = OlalaOracleTests.LoadCase("real-strict-gguf-norms-chat")!;
        using var dev = VulkanDevice.Create();
        using var f = GgufFile.Open(gguf!);
        bool f32 = Environment.GetEnvironmentVariable("DOTLLM_OLALA_BENCH_F32ACT") == "1";
        using var vk = LoadVk(dev, f, f32);
        // Rule 2/3 evidence: how the weights (routers included) reached the device on this load.
        _output.WriteLine($"weights: imported {VulkanWeightImportPolicy.ImportedTensorCount} ({VulkanWeightImportPolicy.ImportedBytes / 1048576.0:F0} MiB), " +
                          $"direct {VulkanWeightImportPolicy.DirectTensorCount}, staged {VulkanWeightImportPolicy.StagedTensorCount} " +
                          $"({VulkanWeightImportPolicy.StagedBytes / 1048576.0:F0} MiB), last fallback '{VulkanWeightImportPolicy.LastFallbackReason}'");
        int[] pos = Enumerable.Range(0, c.L).ToArray();
        var pre = new List<double>(); var dec = new List<double>();
        for (int round = 0; round < 4; round++)
        {
            vk.ResetSequenceState();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            using var p = vk.Forward(c.Ids, pos, -1, null, true);
            double ms = sw.Elapsed.TotalMilliseconds;
            int next = OlalaOracleTests.Argmax(Rows(p, 1, c.Vocab));
            var steps = new List<double>();
            for (int j = 0; j < 48; j++)
            {
                sw.Restart();
                using var s = vk.Forward([next], [c.L + j], -1, null, true);
                steps.Add(sw.Elapsed.TotalMilliseconds);
                next = OlalaOracleTests.Argmax(Rows(s, 1, c.Vocab));
            }
            if (round == 0) { _output.WriteLine($"cold round: prefill {c.L / (ms / 1000):F0} tok/s ({ms:F0} ms), decode {1000 / steps.Average():F1} tok/s"); continue; }
            pre.Add(c.L / (ms / 1000)); dec.Add(1000 / steps.Skip(4).Average());
            _output.WriteLine($"warm round {round}: prefill {pre[^1]:F0} tok/s ({ms:F0} ms), decode {dec[^1]:F1} tok/s (median step {steps.OrderBy(x => x).ElementAt(steps.Count / 2):F1} ms)");
        }
        _output.WriteLine($"{file} act={(f32 ? "F32" : "BF16")}: warm prefill {pre.Min():F0}-{pre.Max():F0} tok/s, decode {dec.Min():F1}-{dec.Max():F1} tok/s");
    }

    /// <summary>
    /// Split-submit per-stage profile (opt-in: <c>DOTLLM_OLALA_VK_PROFILE=1</c>; hold the GPU lock). Every stage is its own submit + fence wait, so each
    /// mark carries a sync cost (measured here with an empty-stage probe is not available: pass <c>DOTLLM_OLALA_VK_SYNC_US</c>, default 50) that is subtracted
    /// to give the "net" column. Prefill = the 300-token chat prompt (256 + 44 chunks); decode = 40 greedy steps.
    /// </summary>
    [SkippableFact]
    public void RealModel_StageProfile()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_OLALA_VK_PROFILE") != "1", "set DOTLLM_OLALA_VK_PROFILE=1");
        SkipUnlessGpu();
        string file = Environment.GetEnvironmentVariable("DOTLLM_OLALA_BENCH_FILE") ?? "Olala-v1.0-mini-7A1B-q8_0-safe2.gguf";
        string? gguf = FindGguf("DOTLLM_OLALA_Q8_GGUF", file); Skip.If(gguf is null, file + " not in the HF cache");
        var c = OlalaOracleTests.LoadCase("real-strict-gguf-norms-chat")!;
        double syncUs = double.Parse(Environment.GetEnvironmentVariable("DOTLLM_OLALA_VK_SYNC_US") ?? "50", System.Globalization.CultureInfo.InvariantCulture);
        using var dev = VulkanDevice.Create();
        using var f = GgufFile.Open(gguf!);
        using var vk = LoadVk(dev, f);
        int[] pos = Enumerable.Range(0, c.L).ToArray();
        // warm up (2 full rounds), then profile
        int next = 0;
        for (int w = 0; w < 2; w++)
        {
            vk.ResetSequenceState();
            using var p0 = vk.Forward(c.Ids, pos, -1, null, true);
            next = OlalaOracleTests.Argmax(Rows(p0, 1, c.Vocab));
            for (int j = 0; j < 8; j++) { using var s0 = vk.Forward([next], [c.L + j], -1, null, true); next = OlalaOracleTests.Argmax(Rows(s0, 1, c.Vocab)); }
        }
        void Dump(string title, int tokens)
        {
            var prof = vk.StageProfile!;
            double total = prof.Values.Sum(v => v.Ms), net = prof.Values.Sum(v => v.Ms - v.Count * syncUs / 1000.0);
            _output.WriteLine($"== {title}: {total:F1} ms wall in marks, {net:F1} ms net of {prof.Values.Sum(v => v.Count)} syncs x {syncUs:F0} us");
            foreach (var kv in prof.OrderByDescending(k => k.Value.Ms - k.Value.Count * syncUs / 1000.0))
            {
                double n = kv.Value.Ms - kv.Value.Count * syncUs / 1000.0;
                _output.WriteLine($"  {kv.Key,-20} n={kv.Value.Count,4} net {n,8:F2} ms ({100 * n / net,5:F1}%)  {n / Math.Max(1, kv.Value.Count) * 1000,8:F0} us/call");
            }
        }
        for (int round = 0; round < 3; round++)
        {
            vk.ResetSequenceState();
            vk.StageProfile = new();
            using var p = vk.Forward(c.Ids, pos, -1, null, true);
            Dump($"prefill {c.L} tok (round {round})", c.L);
            next = OlalaOracleTests.Argmax(Rows(p, 1, c.Vocab));
            vk.StageProfile = new();
            const int steps = 40;
            for (int j = 0; j < steps; j++) { using var s = vk.Forward([next], [c.L + j], -1, null, true); next = OlalaOracleTests.Argmax(Rows(s, 1, c.Vocab)); }
            Dump($"decode, {steps} steps summed (round {round}); per-step = /{steps}", steps);
            vk.StageProfile = null;
        }
    }

    /// <summary>
    /// Attribution of a greedy divergence (opt-in: <c>DOTLLM_OLALA_DIAG=1</c>; needs the gpu AND cpu locks): free-running greedy and teacher-forced
    /// top-2 margins of Vulkan BF16-act, Vulkan F32-act and the CPU model on the same q8_0-safe2 file, against the HF oracle.
    /// </summary>
    [SkippableFact]
    public void Diag_GreedyDivergenceAttribution_Capital()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_OLALA_DIAG") != "1", "set DOTLLM_OLALA_DIAG=1");
        SkipUnlessGpu();
        string? gguf = FindGguf("DOTLLM_OLALA_Q8_GGUF", "Olala-v1.0-mini-7A1B-q8_0-safe2.gguf"); Skip.If(gguf is null, "q8 not in cache");
        var c = OlalaOracleTests.LoadCase("real-strict-gguf-norms-capital")!;
        using var f = GgufFile.Open(gguf!);
        int[] pos = Enumerable.Range(0, c.L).ToArray();

        void Arm(string name, Func<int[], int[], bool, ITensor> fwd, Action reset)
        {
            reset();
            using var pre = fwd(c.Ids, pos, true);
            var gen = new List<int>(); var rows = new List<float[]> { Rows(pre, 1, c.Vocab) };
            gen.Add(OlalaOracleTests.Argmax(rows[0]));
            for (int j = 1; j < c.N; j++) { using var s = fwd([gen[^1]], [c.L + j - 1], true); var r = Rows(s, 1, c.Vocab); rows.Add(r); gen.Add(OlalaOracleTests.Argmax(r)); }
            int first = Enumerable.Range(0, c.N).FirstOrDefault(i => gen[i] != c.GreedyIds[i], -1);
            _output.WriteLine($"{name}: free-running greedy {(first < 0 ? "identical to HF" : $"first divergence at token {first} (HF {c.GreedyIds[first]}, got {gen[first]})")}");
            // teacher-forced margins on the HF tokens
            reset();
            using var pre2 = fwd(c.Ids, pos, true);
            var tf = Rows(pre2, 1, c.Vocab);
            for (int j = 0; j < c.N; j++)
            {
                if ((j >= 3 && j <= 5) || (j >= 10 && j <= 13))   // step 4: the F32-act / CPU q8 divergence; step 12: the BF16-act one
                {
                    var top = tf.Select((v, i) => (v, i)).OrderByDescending(x => x.v).Take(2).ToArray();
                    var hf = c.Greedy.AsSpan(j * c.Vocab, c.Vocab).ToArray().Select((v, i) => (v, i)).OrderByDescending(x => x.v).Take(2).ToArray();
                    _output.WriteLine($"  {name} step {j}: top1 {top[0].i} ({top[0].v:F3}) top2 {top[1].i} ({top[1].v:F3}) margin {top[0].v - top[1].v:F4} | HF top1 {hf[0].i} top2 {hf[1].i} margin {hf[0].v - hf[1].v:F4} | HF token {c.GreedyIds[j]}");
                }
                if (j + 1 < c.N) { using var s = fwd([c.GreedyIds[j]], [c.L + j], true); tf = Rows(s, 1, c.Vocab); }
            }
        }

        using var dev = VulkanDevice.Create();
        using (var vk = LoadVk(dev, f)) Arm("Vulkan BF16-act", (i, p, l) => vk.Forward(i, p, -1, null, l), vk.ResetSequenceState);
        using (var vk32 = LoadVk(dev, f, f32Act: true)) Arm("Vulkan F32-act", (i, p, l) => vk32.Forward(i, p, -1, null, l), vk32.ResetSequenceState);
        if (Environment.GetEnvironmentVariable("DOTLLM_OLALA_DIAG_CPU") == "1")   // needs the cpu lock
        using (var cpu = LoadCpu(f)) Arm("CPU", (i, p, l) => cpu.Forward(i, p, -1, null, l), cpu.ResetSequenceState);
    }
}
