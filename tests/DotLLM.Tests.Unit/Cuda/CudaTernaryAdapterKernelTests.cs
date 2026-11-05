using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Adapters.Ternary;
using DotLLM.Tests.Unit.Vulkan.Ternary;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// CUDA ternary residual adapter (#1618) against the CPU reference (<see cref="TernaryResidualHook"/>, scalar ISA). Same
/// fixtures and BIT-EXACT standard as <c>VulkanTernaryAdapterKernelTests</c>. Skips when no CUDA GPU or the PTX is absent.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaTernaryAdapterKernelTests
{
    private readonly ITestOutputHelper _out;
    public CudaTernaryAdapterKernelTests(ITestOutputHelper output) => _out = output;

    private static string FixturePath(string name) => Path.Combine(AppContext.BaseDirectory, "Adapters", "Fixtures", name + ".nsadapter");

    private static string Prereq()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = Environment.GetEnvironmentVariable("DOTLLM_TERNARY_PTX_DIR") ?? FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        Skip.IfNot(File.Exists(Path.Combine(ptxDir!, "ternary_adapter_residual_f32.ptx")), "ternary_adapter_residual_f32.ptx not generated");
        return ptxDir!;
    }

    private static unsafe float[] GpuApply(CudaTernaryResidualModifier m, CudaStream s, float[] rows, int n, int d, int layer)
    {
        long bytes = (long)n * d * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out nint buf, (nuint)bytes).ThrowOnError();
        try
        {
            fixed (float* p = rows) CudaDriverApi.cuMemcpyHtoD_v2(buf, (nint)p, (nuint)bytes).ThrowOnError();
            m.Apply(buf, layer, n, s.Handle);
            s.Synchronize();
            var o = new float[n * d];
            fixed (float* p = o) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, buf, (nuint)bytes).ThrowOnError();
            return o;
        }
        finally { CudaDriverApi.cuMemFree_v2(buf); }
    }

    private static CudaTernaryResidualModifier Make(string ptx, IReadOnlyList<NsAdapterPackage> pk, TernaryScaleStorage s = TernaryScaleStorage.F32)
    {
        var m = CudaTernaryResidualModifier.TryLoad(ptx, pk, s);
        Assert.NotNull(m);   // present but unloadable is a failure
        return m!;
    }

    private static float[] Cpu(IReadOnlyList<NsAdapterPackage> pk, float[] rows, int n, int layer) => VulkanTernaryAdapterKernelTests.CpuApply(pk, rows, n, layer);
    private static void Eq(float[] e, float[] a, string c) => VulkanTernaryAdapterKernelTests.AssertBitEqual(e, a, c);

    [SkippableTheory]
    [InlineData("bracket-parity")]
    [InlineData("csharp14-syntax-attractor")]
    [InlineData("modulo97-counter")]
    public void SampleAdapter_BitExactVsCpu_FourRows(string name)
    {
        string ptx = Prereq();
        var pkg = NsAdapterPackage.Load(FixturePath(name));
        int d = pkg.Manifest.InputDimension, layer = pkg.Manifest.TargetLayer;
        var rows = VulkanTernaryAdapterKernelTests.Rows(4, d, 100);
        var expect = Cpu([pkg], rows, 4, layer);
        Assert.NotEqual(rows, expect);
        using var ctx = CudaContext.Create(0);
        using var s = CudaStream.Create();
        using var m = Make(ptx, [pkg]);
        Eq(expect, GpuApply(m, s, rows, 4, d, layer), name);
    }

    [SkippableFact]
    public void SampleAdapters_StackedAtOneLayer_BitExactVsCpu()
    {
        string ptx = Prereq();
        var pk = new[] { "bracket-parity", "csharp14-syntax-attractor", "modulo97-counter" }
            .Select(n => NsAdapterPackage.Load(FixturePath(n))).ToList();
        foreach (var p in pk) p.Manifest.TargetLayer = 9;
        var rows = VulkanTernaryAdapterKernelTests.Rows(3, 4096, 700);
        using var ctx = CudaContext.Create(0);
        using var s = CudaStream.Create();
        using var m = Make(ptx, pk);
        Eq(Cpu(pk, rows, 3, 9), GpuApply(m, s, rows, 3, 4096, 9), "stacked samples");
    }

    public static IEnumerable<object[]> ShapeCases() => VulkanTernaryAdapterKernelTests.ShapeCases();

    [SkippableTheory]
    [MemberData(nameof(ShapeCases))]
    public void Synthetic_BitExactVsCpu(int d, int rank, float alpha, float thr, int n)
    {
        string ptx = Prereq();
        var pkg = VulkanTernaryAdapterKernelTests.Synth(rank, d, (ulong)(d * 131 + rank * 7 + 11), alpha, thr, layer: 3);
        var rows = VulkanTernaryAdapterKernelTests.Rows(n, d, 5000 + (ulong)d);
        var expect = Cpu([pkg], rows, n, 3);
        if (alpha > 0 && thr < 1e8f && d >= 100 && rank >= 5) Assert.NotEqual(rows, expect);
        using var ctx = CudaContext.Create(0);
        using var s = CudaStream.Create();
        using var m = Make(ptx, [pkg]);
        Eq(expect, GpuApply(m, s, rows, n, d, 3), $"d={d} rank={rank} alpha={alpha} thr={thr}");
    }

    [SkippableTheory]
    [InlineData(100, new[] { 3, 5, 2 })]
    [InlineData(1000, new[] { 1, 32, 7 })]
    [InlineData(257, new[] { 64, 64, 64, 64 })]
    [InlineData(8192, new[] { 16, 8 })]   // the CUDA kernel's D cap
    public void MultiAdapter_SameLayer_RankConcat_BitExact(int d, int[] ranks)
    {
        string ptx = Prereq();
        float[] alphas = [1f, 0.5f, 2f, 0.25f];
        var pk = ranks.Select((r, i) => VulkanTernaryAdapterKernelTests.Synth(r, d, (ulong)(77 + i * 13), alphas[i], i == 1 ? 0.5f : 0f, layer: 4)).ToList();
        var rows = VulkanTernaryAdapterKernelTests.Rows(3, d, 321);
        using var ctx = CudaContext.Create(0);
        using var s = CudaStream.Create();
        using var m = Make(ptx, pk);
        Eq(Cpu(pk, rows, 3, 4), GpuApply(m, s, rows, 3, d, 4), "rank-concat");
        var rev = pk.AsEnumerable().Reverse().ToList();
        using var mr = Make(ptx, rev);
        Eq(Cpu(rev, rows, 3, 4), GpuApply(mr, s, rows, 3, d, 4), "rank-concat reversed");
    }

    [SkippableFact]
    public void TwoLayers_OwnPack_OtherLayersUntouched_ZeroTinyRows_ManyRows()
    {
        string ptx = Prereq();
        const int d = 300;
        var a = VulkanTernaryAdapterKernelTests.Synth(6, d, 41, 1f, 0f, layer: 2);
        var b = VulkanTernaryAdapterKernelTests.Synth(9, d, 42, 1f, 0f, layer: 7);
        var rows = VulkanTernaryAdapterKernelTests.Rows(64, d, 55);   // prefill-like: 64 independent blocks
        for (int i = 0; i < d; i++) { rows[i] = 0f; rows[d + i] = i % 2 == 0 ? 1e-7f : -1e-7f; }
        using var ctx = CudaContext.Create(0);
        using var s = CudaStream.Create();
        using var m = Make(ptx, [a, b]);
        Assert.True(m.AppliesTo(2)); Assert.True(m.AppliesTo(7)); Assert.False(m.AppliesTo(3));
        Eq(Cpu([a], rows, 64, 2), GpuApply(m, s, rows, 64, d, 2), "layer 2");
        Eq(Cpu([b], rows, 64, 7), GpuApply(m, s, rows, 64, d, 7), "layer 7");
        Eq(rows, GpuApply(m, s, rows, 64, d, 3), "layer 3 untouched");
    }

    [SkippableFact]
    public void F16Scales_BitIdenticalToF32()
    {
        string ptx = Prereq();
        const int d = 1000;
        var pk = new[] { VulkanTernaryAdapterKernelTests.Synth(7, d, 1, 1f, 0f), VulkanTernaryAdapterKernelTests.Synth(12, d, 2, 0.5f, 0f) };
        foreach (var p in pk) p.Manifest.TargetLayer = 1;
        var rows = VulkanTernaryAdapterKernelTests.Rows(3, d, 88);
        using var ctx = CudaContext.Create(0);
        using var s = CudaStream.Create();
        using var m32 = Make(ptx, pk, TernaryScaleStorage.F32);
        using var m16 = Make(ptx, pk, TernaryScaleStorage.F16);
        var o32 = GpuApply(m32, s, rows, 3, d, 1);
        Eq(Cpu(pk, rows, 3, 1), o32, "f32 vs cpu");
        Eq(o32, GpuApply(m16, s, rows, 3, d, 1), "f16 vs f32");
    }

    [SkippableFact]
    public unsafe void Latency_D4096_Rank_1_2_32_CudaEvents()
    {
        string ptx = Prereq();
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_TERNARY_LATENCY") == "1", "set DOTLLM_TERNARY_LATENCY=1");
        const int d = 4096;
        using var ctx = CudaContext.Create(0);
        using var s = CudaStream.Create();
        CudaDriverApi.cuEventCreate(out nint e0, 0).ThrowOnError();
        CudaDriverApi.cuEventCreate(out nint e1, 0).ThrowOnError();
        try
        {
            foreach (int rank in new[] { 1, 2, 32 })
                foreach (int n in new[] { 1, 8 })
                {
                    var rows = VulkanTernaryAdapterKernelTests.Rows(n, d, 4242);
                    NsAdapterPackage pkg;
                    for (ulong seed = 99 + (ulong)rank; ; seed += 1000)
                    {
                        pkg = VulkanTernaryAdapterKernelTests.Synth(rank, d, seed, 1f, 0f, layer: 0);
                        if (Enumerable.Range(0, n).All(r => VulkanTernaryAdapterKernelTests.MaxRelu(pkg, rows.AsSpan(r * d, d).ToArray()) > 1e-6f)) break;
                    }
                    using var m = Make(ptx, [pkg]);
                    long bytes = (long)n * d * sizeof(float);
                    CudaDriverApi.cuMemAlloc_v2(out nint buf, (nuint)bytes).ThrowOnError();
                    try
                    {
                        fixed (float* p = rows) CudaDriverApi.cuMemcpyHtoD_v2(buf, (nint)p, (nuint)bytes).ThrowOnError();
                        double Time(int reps)
                        {
                            CudaDriverApi.cuEventRecord(e0, s.Handle).ThrowOnError();
                            for (int i = 0; i < reps; i++) m.Apply(buf, 0, n, s.Handle);
                            CudaDriverApi.cuEventRecord(e1, s.Handle).ThrowOnError();
                            CudaDriverApi.cuEventSynchronize(e1).ThrowOnError();
                            CudaDriverApi.cuEventElapsedTime(out float ms, e0, e1).ThrowOnError();
                            return ms * 1000.0;
                        }
                        Time(16);
                        var one = new List<double>(); var amort = new List<double>();
                        for (int rep = 0; rep < 7; rep++) { one.Add(Time(1)); amort.Add(Time(64) / 64.0); }
                        one.Sort(); amort.Sort();
                        _out.WriteLine($"CUDA D={d} rank={rank} rows={n}: single launch min/med/max = {one[0]:F1}/{one[3]:F1}/{one[^1]:F1} us; " +
                                       $"back-to-back x64 per launch min/med/max = {amort[0]:F1}/{amort[3]:F1}/{amort[^1]:F1} us");
                    }
                    finally { CudaDriverApi.cuMemFree_v2(buf); }
                }
        }
        finally { CudaDriverApi.cuEventDestroy_v2(e0); CudaDriverApi.cuEventDestroy_v2(e1); }
    }

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    private static string? FindPtxDir()
    {
        foreach (var dir in new[]
                 {
                     Path.Combine(AppContext.BaseDirectory, "ptx"),
                     Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
                 })
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        return null;
    }
}
