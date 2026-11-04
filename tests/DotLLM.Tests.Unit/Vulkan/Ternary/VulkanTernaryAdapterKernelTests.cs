using DotLLM.Cpu.Kernels;
using DotLLM.Models.Adapters.Ternary;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan.Ternary;

/// <summary>
/// Vulkan ternary residual adapter (#1618) against the CPU reference (<see cref="TernaryResidualHook"/>, scalar ISA, itself
/// bit-identical to the NeuroSymbolicAI reference). The device residual is F32 (as the Vulkan forward holds it), the integer
/// pipeline is exact and the float ops are single IEEE ops in the CPU's order, so the comparison is BIT-EXACT, not toleranced.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanTernaryAdapterKernelTests
{
    private readonly ITestOutputHelper _out;
    public VulkanTernaryAdapterKernelTests(ITestOutputHelper output) => _out = output;

    // ------------------------------------------------------------------ fixtures (same generators as TernaryAdapterTests) ----

    internal sealed class XorShift
    {
        private ulong _s;
        public XorShift(ulong seed) => _s = seed;
        public ulong Next() { _s ^= _s << 13; _s ^= _s >> 7; _s ^= _s << 17; return _s; }
        public float F() => ((Next() >> 40) / 8388608f) - 1f;
    }

    internal static float[] Row(ulong seed, int d)
    {
        var g = new XorShift(seed);
        var h = new float[d];
        for (int i = 0; i < d; i++) h[i] = g.F() * (i % 97 == 0 ? 8f : 1f);
        return h;
    }

    internal static NsAdapterPackage Synth(int rank, int d, ulong seed, float alpha, float thr, int layer = 3, bool halfScales = false)
    {
        var g = new XorShift(seed);
        sbyte W() => (sbyte)((int)(g.Next() % 3) - 1);
        var down = new PackedTernaryTensor(rank, d);
        var up = new PackedTernaryTensor(d, rank);
        for (int r = 0; r < rank; r++) for (int c = 0; c < d; c++) down[r, c] = W();
        for (int r = 0; r < d; r++) for (int c = 0; c < rank; c++) up[r, c] = W();
        float[] pool = [0.25f, 0.5f, 0.75f, 1f, 1.25f, 1.5f, 2f, 3f];
        for (int r = 0; r < rank; r++) down.Scales[r] = pool[(int)(g.Next() % 8)];
        for (int r = 0; r < d; r++) up.Scales[r] = pool[(int)(g.Next() % 8)];
        var m = new NsAdapterManifest { TargetModel = "synthetic", TargetLayer = layer, InputDimension = d, Rank = rank, Alpha = alpha, ActivationThreshold = thr, Domain = "synthetic" };
        return new NsAdapterPackage(m, down, up);
    }

    internal static float[] Rows(int n, int d, ulong seed0)
    {
        var all = new float[n * d];
        for (int t = 0; t < n; t++) Row(seed0 + (ulong)t, d).CopyTo(all, t * d);
        return all;
    }

    private static string FixturePath(string name) => Path.Combine(AppContext.BaseDirectory, "Adapters", "Fixtures", name + ".nsadapter");

    /// <summary>CPU reference over n rows (scalar ISA).</summary>
    internal static float[] CpuApply(IReadOnlyList<NsAdapterPackage> pkgs, float[] rows, int n, int layer)
    {
        using var hook = new TernaryResidualHook(pkgs[0].Manifest.InputDimension) { Isa = TernaryKernelIsa.Scalar };
        foreach (var p in pkgs) hook.Add(p);
        var r = (float[])rows.Clone();
        hook.ModifyResidual(r, layer, n);
        return r;
    }

    internal static float[] GpuApply(VulkanDevice dev, VulkanTernaryResidualModifier m, float[] rows, int n, int d, int layer)
    {
        using var buf = dev.Allocate((long)n * d * sizeof(float));
        dev.Upload(rows, buf);
        using (var ctx = dev.CreateSubmitContext())
        {
            ctx.Begin();
            m.Record(ctx.CommandBuffer, buf, layer, n, d);
            ctx.SubmitAndWait();
        }
        var o = new float[n * d];
        dev.Download(buf, o);
        return o;
    }

    internal static void AssertBitEqual(float[] expected, float[] actual, string ctx)
    {
        Assert.Equal(expected.Length, actual.Length);
        if (Environment.GetEnvironmentVariable("DOTLLM_TERNARY_DIFFDUMP") == "1")
            for (int i = 0; i < expected.Length; i++)
                if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                    Console.Error.WriteLine($"DIFF {ctx} i={i} cpu={expected[i]:R} gpu={actual[i]:R} ulps={BitConverter.SingleToInt32Bits(expected[i]) - BitConverter.SingleToInt32Bits(actual[i])}");
        for (int i = 0; i < expected.Length; i++)
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                Assert.Fail($"{ctx}: element {i}: cpu {expected[i]:R} gpu {actual[i]:R}");
    }

    internal static float MaxRelu(NsAdapterPackage p, float[] h)
    {
        float maxH = h.Max(Math.Abs);
        float qs = 127f / maxH;
        var x = h.Select(v => (sbyte)Math.Clamp((int)Math.Round(v * qs), -128, 127)).ToArray();
        float m = 0f;
        for (int j = 0; j < p.Manifest.Rank; j++)
        {
            long s = 0;
            for (int c = 0; c < p.Manifest.InputDimension; c++) s += p.Down[j, c] * x[c];
            m = Math.Max(m, Math.Max(0f, s * p.Down.Scales[j]));
        }
        return m;
    }

    private static VulkanTernaryResidualModifier Make(VulkanDevice dev, string spv, IReadOnlyList<NsAdapterPackage> pk, TernaryScaleStorage s = TernaryScaleStorage.F32)
        => VulkanTernaryResidualModifier.Create(dev, spv, pk, s);

    // ------------------------------------------------------------------ the three reference sample adapters (D = 4096) ----

    [SkippableTheory]
    [InlineData("bracket-parity")]
    [InlineData("csharp14-syntax-attractor")]
    [InlineData("modulo97-counter")]
    public void SampleAdapter_BitExactVsCpu_FourRows(string name)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spv);
        var pkg = NsAdapterPackage.Load(FixturePath(name));
        int d = pkg.Manifest.InputDimension;
        Assert.Equal(4096, d);
        int layer = pkg.Manifest.TargetLayer;
        var rows = Rows(4, d, 100);
        var expect = CpuApply([pkg], rows, 4, layer);
        Assert.NotEqual(rows, expect);   // the adapter really acts on these rows
        using var dev = VulkanDevice.Create();
        using var m = Make(dev, spv, [pkg]);
        AssertBitEqual(expect, GpuApply(dev, m, rows, 4, d, layer), name);
    }

    [SkippableFact]
    public void SampleAdapters_StackedAtOneLayer_BitExactVsCpu()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spv);
        var pk = new[] { "bracket-parity", "csharp14-syntax-attractor", "modulo97-counter" }
            .Select(n => NsAdapterPackage.Load(FixturePath(n))).ToList();
        foreach (var p in pk) p.Manifest.TargetLayer = 9;
        var rows = Rows(3, 4096, 700);
        var expect = CpuApply(pk, rows, 3, 9);
        using var dev = VulkanDevice.Create();
        using var m = Make(dev, spv, pk);
        Assert.Equal(pk.Sum(p => p.Manifest.Rank), m.RankAt(9));
        AssertBitEqual(expect, GpuApply(dev, m, rows, 3, 4096, 9), "stacked samples");
    }

    // ------------------------------------------------------------------ synthetic shape classes ----

    public static IEnumerable<object[]> ShapeCases()
    {
        // (d, rank, alpha, thr, rows). D off the 32/64/256 grid and at the boundaries; rank 1..64; alpha 0/1/other; gate on/off.
        int[] ds = [1, 31, 33, 63, 64, 65, 100, 255, 256, 257, 1000, 4096, 4099];
        foreach (int d in ds)
            foreach (int rank in new[] { 1, 2, 5, 32, 64 })
            {
                if ((d == 1 || d == 31) && rank > 5) continue;
                yield return new object[] { d, rank, 1f, 0f, d >= 1000 ? 2 : 3 };
            }
        yield return new object[] { 100, 7, 0f, 0f, 3 };       // alpha 0: exact identity
        yield return new object[] { 100, 7, -2f, 0f, 3 };      // alpha < 0: identity
        yield return new object[] { 257, 7, 0.37f, 0f, 3 };    // other alpha
        yield return new object[] { 257, 7, 2.5f, 0f, 3 };
        yield return new object[] { 100, 8, 1f, -1f, 3 };      // threshold: -1 -> only the 1e-6 guard
        yield return new object[] { 100, 8, 1f, 1e9f, 3 };     // gate closed on every row
    }

    [SkippableTheory]
    [MemberData(nameof(ShapeCases))]
    public void Synthetic_BitExactVsCpu(int d, int rank, float alpha, float thr, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spv);
        var pkg = Synth(rank, d, (ulong)(d * 131 + rank * 7 + 11), alpha, thr, layer: 3);
        var rows = Rows(n, d, 5000 + (ulong)d);
        var expect = CpuApply([pkg], rows, n, 3);
        if (alpha > 0 && thr < 1e8f && d >= 100 && rank >= 5) Assert.NotEqual(rows, expect);   // the case actually exercises the adapter
        using var dev = VulkanDevice.Create();
        using var m = Make(dev, spv, [pkg]);
        AssertBitEqual(expect, GpuApply(dev, m, rows, n, d, 3), $"d={d} rank={rank} alpha={alpha} thr={thr}");
    }

    [SkippableFact]
    public void Threshold_PerRowGate_OpenAndClosedRowsInOneDispatch()
    {
        // A threshold between two rows' largest ReLU value: one row opens, the other stays bit-identical.
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spv);
        const int d = 200, rank = 8, n = 4;
        var probe = Synth(rank, d, 3, 1f, 0f);
        var rows = Rows(n, d, 9100);
        var ms = Enumerable.Range(0, n).Select(t => MaxRelu(probe, rows.AsSpan(t * d, d).ToArray())).ToArray();
        var sorted = ms.Order().ToArray();
        float thr = (sorted[1] + sorted[2]) / 2f;   // two rows above, two below
        var pkg = Synth(rank, d, 3, 1f, thr);
        var expect = CpuApply([pkg], rows, n, 3);
        int changed = 0;
        for (int t = 0; t < n; t++)
            if (!rows.AsSpan(t * d, d).SequenceEqual(expect.AsSpan(t * d, d))) changed++;
        Assert.Equal(2, changed);
        using var dev = VulkanDevice.Create();
        using var m = Make(dev, spv, [pkg]);
        AssertBitEqual(expect, GpuApply(dev, m, rows, n, d, 3), "per-row gate");
    }

    [SkippableTheory]
    [InlineData(100, new[] { 3, 5, 2 })]
    [InlineData(1000, new[] { 1, 32, 7 })]
    [InlineData(257, new[] { 64, 64, 64, 64 })]   // total rank 256 = the cap
    public void MultiAdapter_SameLayer_RankConcat_BitExact(int d, int[] ranks)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spv);
        float[] alphas = [1f, 0.5f, 2f, 0.25f];
        var pk = ranks.Select((r, i) => Synth(r, d, (ulong)(77 + i * 13), alphas[i], i == 1 ? 0.5f : 0f, layer: 4)).ToList();
        var rows = Rows(3, d, 321);
        var expect = CpuApply(pk, rows, 3, 4);
        using var dev = VulkanDevice.Create();
        using var m = Make(dev, spv, pk);
        Assert.Equal(ranks.Sum(), m.RankAt(4));
        AssertBitEqual(expect, GpuApply(dev, m, rows, 3, d, 4), "rank-concat");
        // order matters (float add order of the deltas): reversed registration must match the CPU reversed registration
        var rev = pk.AsEnumerable().Reverse().ToList();
        using var mr = Make(dev, spv, rev);
        AssertBitEqual(CpuApply(rev, rows, 3, 4), GpuApply(dev, mr, rows, 3, d, 4), "rank-concat reversed order");
    }

    [SkippableFact]
    public void RankAboveCap_Rejected()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spv);
        var pk = new[] { Synth(200, 64, 1, 1f, 0f), Synth(57, 64, 2, 1f, 0f) };
        using var dev = VulkanDevice.Create();
        Assert.Throws<NotSupportedException>(() => Make(dev, spv, pk));
    }

    [SkippableFact]
    public void TwoLayers_EachLayerUsesItsOwnPack_AndOtherLayersAreUntouched()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spv);
        const int d = 300;
        var a = Synth(6, d, 41, 1f, 0f, layer: 2);
        var b = Synth(9, d, 42, 1f, 0f, layer: 7);
        var rows = Rows(2, d, 55);
        using var dev = VulkanDevice.Create();
        using var m = Make(dev, spv, [a, b]);
        Assert.True(m.AppliesTo(2)); Assert.True(m.AppliesTo(7)); Assert.False(m.AppliesTo(3));
        var ea = CpuApply([a], rows, 2, 2); var eb = CpuApply([b], rows, 2, 7);
        Assert.NotEqual(ea, eb);
        AssertBitEqual(ea, GpuApply(dev, m, rows, 2, d, 2), "layer 2");
        AssertBitEqual(eb, GpuApply(dev, m, rows, 2, d, 7), "layer 7");
        AssertBitEqual(rows, GpuApply(dev, m, rows, 2, d, 3), "layer 3 untouched");
    }

    [SkippableFact]
    public void ZeroRow_And_TinyRow_StayUntouched_NoNaN()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spv);
        const int d = 128;
        var pkg = Synth(8, d, 5, 1f, 0f);
        var rows = new float[3 * d];
        for (int i = 0; i < d; i++) rows[2 * d + i] = (i % 2 == 0 ? 1e-7f : -1e-7f);   // max|h| <= 1e-6: quantizer scale 0
        var expect = CpuApply([pkg], rows, 3, 3);
        using var dev = VulkanDevice.Create();
        using var m = Make(dev, spv, [pkg]);
        var got = GpuApply(dev, m, rows, 3, d, 3);
        AssertBitEqual(expect, got, "zero/tiny rows");
        Assert.True(got.All(float.IsFinite));
    }

    [SkippableFact]
    public void F16Scales_BitIdenticalToF32_WhenScalesAreHalfRepresentable()
    {
        // The synthetic scale pool is exactly representable in F16, so the F16 variant must reproduce the F32 output bit for bit.
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spv);
        const int d = 1000;
        var pk = new[] { Synth(7, d, 1, 1f, 0f), Synth(12, d, 2, 0.5f, 0f) };
        foreach (var p in pk) p.Manifest.TargetLayer = 1;
        var rows = Rows(3, d, 88);
        using var dev = VulkanDevice.Create();
        using var m32 = Make(dev, spv, pk, TernaryScaleStorage.F32);
        using var m16 = Make(dev, spv, pk, TernaryScaleStorage.F16);
        var o32 = GpuApply(dev, m32, rows, 3, d, 1);
        AssertBitEqual(CpuApply(pk, rows, 3, 1), o32, "f32 vs cpu");
        AssertBitEqual(o32, GpuApply(dev, m16, rows, 3, d, 1), "f16 vs f32");
    }

    // ------------------------------------------------------------------ latency (GPU timestamps) ----

    [SkippableFact]
    public unsafe void Latency_D4096_Rank_1_2_32_GpuTimestamps()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spv);
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_TERNARY_LATENCY") == "1", "set DOTLLM_TERNARY_LATENCY=1 (holds the GPU for a few ms per case)");
        const int d = 4096;
        using var dev = VulkanDevice.Create();
        float period = dev.TimestampPeriodNs;
        Skip.If(period <= 0f, "no timestamp support");
        var qci = new DotLLM.Vulkan.Interop.VkQueryPoolCreateInfo { sType = 11, queryType = 2, queryCount = 4 };
        Assert.True(DotLLM.Vulkan.Interop.VulkanApi.vkCreateQueryPool(dev.Handle, qci, 0, out nint pool) >= 0);
        try
        {
            foreach (int rank in new[] { 1, 2, 32 })
                foreach (int n in new[] { 1, 8 })
                {
                    var pkg = Synth(rank, d, 99 + (ulong)rank, 1f, 0f, layer: 0);
                    using var m = Make(dev, spv, [pkg]);
                    var rows = Rows(n, d, 4242);
                    using var buf = dev.Allocate((long)n * d * sizeof(float));
                    dev.Upload(rows, buf);
                    double Time(int reps)
                    {
                        using var ctx = dev.CreateSubmitContext();
                        ctx.Begin();
                        DotLLM.Vulkan.Interop.VulkanApi.vkCmdResetQueryPool(ctx.CommandBuffer, pool, 0, 4);
                        DotLLM.Vulkan.Interop.VulkanApi.vkCmdWriteTimestamp(ctx.CommandBuffer, DotLLM.Vulkan.Interop.VkPipelineStageFlags.BottomOfPipe, pool, 0);
                        for (int i = 0; i < reps; i++)
                        {
                            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                            m.Record(ctx.CommandBuffer, buf, 0, n, d);
                        }
                        KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                        DotLLM.Vulkan.Interop.VulkanApi.vkCmdWriteTimestamp(ctx.CommandBuffer, DotLLM.Vulkan.Interop.VkPipelineStageFlags.BottomOfPipe, pool, 1);
                        ctx.SubmitAndWait();
                        ulong* ts = stackalloc ulong[2];
                        Assert.True(DotLLM.Vulkan.Interop.VulkanApi.vkGetQueryPoolResults(dev.Handle, pool, 0, 2, 2 * sizeof(ulong), (nint)ts, sizeof(ulong), flags: 0x1 | 0x2) >= 0);
                        return (ts[1] - ts[0]) * (double)period / 1000.0;   // us
                    }
                    Time(8);   // warm: pipeline compile, clocks
                    var one = new List<double>(); var amort = new List<double>();
                    for (int rep = 0; rep < 7; rep++) { one.Add(Time(1)); amort.Add(Time(64) / 64.0); }
                    one.Sort(); amort.Sort();
                    _out.WriteLine($"D={d} rank={rank} rows={n}: single-dispatch (incl. barrier+launch) min/med/max = {one[0]:F1}/{one[3]:F1}/{one[^1]:F1} us; " +
                                   $"back-to-back x64 per dispatch min/med/max = {amort[0]:F1}/{amort[3]:F1}/{amort[^1]:F1} us");
                }
        }
        finally { DotLLM.Vulkan.Interop.VulkanApi.vkDestroyQueryPool(dev.Handle, pool, 0); }
    }
}
