using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Qwen4Exp;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>One CPU-oracle + Vulkan model pair loaded from the same random-weight checkpoint.</summary>
internal sealed unsafe class Q4eRig : IDisposable
{
    private readonly string _dir = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "dotllm-q4e-vk-" + Guid.NewGuid().ToString("N"));
    private readonly GgufFile _cpuGguf, _vkGguf;
    public ModelConfig Config { get; }
    public Qwen4ExpTransformerModel Cpu { get; }
    public VulkanDevice Device { get; }
    public VulkanQwen4ExpTransformerModel Vk { get; }
    public string FilePath { get; }

    public Q4eRig(byte[] gguf, string spvDir)
    {
        Directory.CreateDirectory(_dir);
        FilePath = System.IO.Path.Combine(_dir, "q4e.gguf");
        File.WriteAllBytes(FilePath, gguf);
        var (cpu, cg, config) = ModelLoader.LoadFromGguf(FilePath);
        _cpuGguf = cg; Cpu = (Qwen4ExpTransformerModel)cpu; Config = config;
        _vkGguf = GgufFile.Open(FilePath);
        Device = VulkanDevice.Create();
        Vk = VulkanQwen4ExpTransformerModel.BuildFromGguf(Device, _vkGguf, GgufModelConfigExtractor.Extract(_vkGguf.Metadata), spvDir);
    }

    public static float[] Row(ITensor t, int row)
    {
        int v = t.Shape[1];
        return new ReadOnlySpan<float>((void*)(t.DataPointer + (nint)((long)row * v * 4)), v).ToArray();
    }

    public void Dispose()
    {
        Vk.Dispose(); Device.Dispose(); Cpu.Dispose(); _cpuGguf.Dispose(); _vkGguf.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }
}

/// <summary>
/// Whole-model parity of the Vulkan Qwen4-Exp forward (#818 V1) against the CPU oracle <c>Qwen4ExpTransformerModel</c> (#816), which is
/// itself HF-validated. Random-weight checkpoints, never the real file (not yet available): degenerate-shape-proof geometry (2 GDN key
/// heads vs 4 value heads, 4 query heads over 2 KV heads, 3 indexer heads), a 512-expert top-10 router variant, and quantised variants
/// that mirror the real file's mix (Q8_0 projections, Q8_0 / Q5_1 expert banks that take the F32-widened fallback, BF16 indexer, F16
/// table; and a 256-wide geometry whose Q4_K / Q5_K banks take the resident MMVQ / grouped-coopmat kernels).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen4ExpParityTests
{
    private readonly ITestOutputHelper _out;
    public VulkanQwen4ExpParityTests(ITestOutputHelper output) => _out = output;

    public static IEnumerable<object[]> Variants() =>
    [
        ["tiny-f32", 0],
        ["tiny-q8q51", 1],
        ["e512-f32", 2],
        ["e512-q8q51", 3],
        ["kq256-kquant", 4],
        ["kq256-f32", 5],
        ["hd256-f32", 6],       // the released attention geometry: head_dim 256, 64 rotary dims, 4 query heads over 2 KV heads
        ["hd256-q8q51", 7],
    ];

    internal static byte[] Build(int variant) => variant switch
    {
        0 => Qwen4ExpRandomGguf.Build(Qwen4ExpRandomGguf.Tiny, Q4eQuant.F32),
        1 => Qwen4ExpRandomGguf.Build(Qwen4ExpRandomGguf.Tiny, Q4eQuant.Q8Q51),
        2 => Qwen4ExpRandomGguf.Build(Qwen4ExpRandomGguf.Experts512, Q4eQuant.F32),
        3 => Qwen4ExpRandomGguf.Build(Qwen4ExpRandomGguf.Experts512, Q4eQuant.Q8Q51),
        4 => Qwen4ExpRandomGguf.Build(Qwen4ExpRandomGguf.KQuant256, Q4eQuant.KQuant),
        5 => Qwen4ExpRandomGguf.Build(Qwen4ExpRandomGguf.KQuant256, Q4eQuant.F32),
        6 => Qwen4ExpRandomGguf.Build(Qwen4ExpRandomGguf.Hd256, Q4eQuant.F32),
        7 => Qwen4ExpRandomGguf.Build(Qwen4ExpRandomGguf.Hd256, Q4eQuant.Q8Q51),
        _ => throw new ArgumentOutOfRangeException(nameof(variant)),
    };

    internal static bool IsQuantised(int variant) => variant is 1 or 3 or 4 or 7;

    internal static int[] Ids(int count, int vocab, int seed = 7)
    {
        var rng = new Random(seed);
        return Enumerable.Range(0, count).Select(_ => rng.Next(0, vocab)).ToArray();
    }

    internal static (double RelL2, double Kl, bool Top1) Compare(float[] reference, float[] actual)
    {
        Assert.Equal(reference.Length, actual.Length);
        double num = 0, den = 0;
        for (int i = 0; i < reference.Length; i++) { double d = reference[i] - actual[i]; num += d * d; den += (double)reference[i] * reference[i]; }
        return (Math.Sqrt(num / Math.Max(den, 1e-30)), Kl(reference, actual), Argmax(reference) == Argmax(actual));
    }

    private static int Argmax(float[] v) { int b = 0; for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i; return b; }

    private static double Kl(float[] p, float[] q)
    {
        double[] lp = LogSoftmax(p), lq = LogSoftmax(q);
        double kl = 0;
        for (int i = 0; i < p.Length; i++) kl += Math.Exp(lp[i]) * (lp[i] - lq[i]);
        return kl;
    }

    private static double[] LogSoftmax(float[] x)
    {
        double max = x.Max(), sum = 0;
        foreach (float v in x) sum += Math.Exp(v - max);
        double lse = max + Math.Log(sum);
        return x.Select(v => v - lse).ToArray();
    }

    [SkippableTheory]
    [MemberData(nameof(Variants))]
    public void Prefill_MatchesOracle_PerLayerAndLogits(string name, int variant)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Q4eRig(Build(variant), spvDir);
        int T = 40;
        var ids = Ids(T, rig.Config.VocabSize);
        var pos = Enumerable.Range(0, T).ToArray();

        var cpuTrace = new Dictionary<string, float[]>();
        rig.Cpu.Trace = (n, d, r, c) => cpuTrace[n] = d.ToArray();
        var vkTrace = new Dictionary<string, float[]>();
        rig.Vk.Trace = (n, d, r, c) => vkTrace[n] = d;

        var cpuLogits = Q4eRig.Row(rig.Cpu.Forward(ids, pos, -1), T - 1);
        var vkLogits = Q4eRig.Row(rig.Vk.Forward(ids, pos, -1), 0);

        bool quant = IsQuantised(variant);
        double layerTol = quant ? 0.08 : 3e-3;
        for (int il = 0; il < rig.Config.NumLayers; il++)
        {
            var a = cpuTrace[$"blk.{il}.l_out"]; var b = vkTrace[$"blk.{il}.l_out"];
            double num = 0, den = 0;
            for (int i = 0; i < a.Length; i++) { double d = a[i] - b[i]; num += d * d; den += (double)a[i] * a[i]; }
            double rel = Math.Sqrt(num / den);
            _out.WriteLine($"{name} layer {il}: relL2 = {rel:E3}");
            Assert.True(rel < layerTol, $"{name} layer {il} residual relL2 {rel:E3} >= {layerTol}");
        }
        var (rl2, kl, top1) = Compare(cpuLogits, vkLogits);
        _out.WriteLine($"{name} first-token logits: relL2 = {rl2:E3}, KL = {kl:E3}, top1 agree = {top1}");
        Assert.True(kl < (quant ? 0.02 : 1e-4), $"{name}: KL {kl:E3}");
        Assert.True(rl2 < (quant ? 0.08 : 3e-3), $"{name}: logits relL2 {rl2:E3}");
    }

    [SkippableTheory]
    [MemberData(nameof(Variants))]
    public void GreedyDecode_FollowsOracle(string name, int variant)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Q4eRig(Build(variant), spvDir);
        int prompt = 24, steps = 24;
        var ids = Ids(prompt, rig.Config.VocabSize, seed: 11);
        var cpuLogits = Q4eRig.Row(rig.Cpu.Forward(ids, Enumerable.Range(0, prompt).ToArray(), -1), prompt - 1);
        var vkLogits = Q4eRig.Row(rig.Vk.Forward(ids, Enumerable.Range(0, prompt).ToArray(), -1), 0);

        int agree = 0; double worstKl = 0, worstRel = 0;
        for (int s = 0; s < steps; s++)
        {
            var (rl2, kl, top1) = Compare(cpuLogits, vkLogits);
            if (top1) agree++;
            worstKl = Math.Max(worstKl, kl); worstRel = Math.Max(worstRel, rl2);
            int next = Array.IndexOf(cpuLogits, cpuLogits.Max());       // the ORACLE drives both, so the trajectories cannot drift apart
            int p = prompt + s;
            cpuLogits = Q4eRig.Row(rig.Cpu.Forward([next], [p], -1), 0);
            vkLogits = Q4eRig.Row(rig.Vk.Forward([next], [p], -1), 0);
        }
        _out.WriteLine($"{name}: {agree}/{steps} argmax agree, worst KL {worstKl:E3}, worst relL2 {worstRel:E3}");
        bool quant = IsQuantised(variant);
        Assert.True(worstKl < (quant ? 0.05 : 1e-3), $"{name}: worst KL {worstKl:E3}");
        Assert.True(agree >= (quant ? steps * 3 / 4 : steps - 1), $"{name}: only {agree}/{steps} greedy tokens agree");
    }

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(17)]
    public void ChunkedPrefill_EqualsSingleShot_OnVulkan(int chunk)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Q4eRig(Build(0), spvDir);
        int T = 36;
        var ids = Ids(T, rig.Config.VocabSize, seed: 3);
        var whole = Q4eRig.Row(rig.Vk.Forward(ids, Enumerable.Range(0, T).ToArray(), -1), 0);

        using var state = rig.Vk.CreateState();
        float[] last = [];
        for (int i = 0; i < T; i += chunk)
        {
            int n = Math.Min(chunk, T - i);
            last = Q4eRig.Row(rig.Vk.Forward(ids.AsSpan(i, n), Enumerable.Range(i, n).ToArray(), -1, state), 0);
        }
        var (rl2, kl, _) = Compare(whole, last);
        _out.WriteLine($"chunk {chunk}: relL2 = {rl2:E3}, KL = {kl:E3}");
        Assert.True(rl2 < 2e-3, $"chunked({chunk}) diverges from single shot: relL2 {rl2:E3}");
    }

    [SkippableFact]
    public void SyntheticFixture_MatchesOracle()
    {
        // The in-tree SyntheticQwen4ExpGguf (named by the issue): a different GQA ratio (2 query heads over 1 KV head) and a single GDN key
        // head, so it is the degenerate-shape arm next to the NK != NV rigs above - it must still agree.
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Q4eRig(SyntheticQwen4ExpGguf.Build(), spvDir);
        int V = rig.Config.VocabSize;
        Assert.Equal(11, rig.Vk.DenseContextLimit);   // indexer budget 8 + block 4 - 1
        var ids = Ids(7, V, seed: 5);
        var cpu = Q4eRig.Row(rig.Cpu.Forward(ids, Enumerable.Range(0, 7).ToArray(), -1), 6);
        var vk = Q4eRig.Row(rig.Vk.Forward(ids, Enumerable.Range(0, 7).ToArray(), -1), 0);
        var (rl2, kl, _) = Compare(cpu, vk);
        _out.WriteLine($"synthetic prefill: relL2 {rl2:E3}, KL {kl:E3}");
        Assert.True(rl2 < 3e-3);
        for (int s = 0; s < 4; s++)
        {
            int next = Array.IndexOf(cpu, cpu.Max());
            cpu = Q4eRig.Row(rig.Cpu.Forward([next], [7 + s], -1), 0);
            vk = Q4eRig.Row(rig.Vk.Forward([next], [7 + s], -1), 0);
            (rl2, kl, _) = Compare(cpu, vk);
            Assert.True(rl2 < 3e-3, $"synthetic decode step {s}: relL2 {rl2:E3}");
        }
    }
}
