using System;
using System.IO;
using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Engine.KvCache.Codecs;
using DotLLM.Tests.Unit.Engine.KvCache;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// #1892 Task 3: the CUDA Q/O rotation kernels (<c>tq_rotate_q_f16</c>, <c>tq_rotate_o_f32</c> and their <c>_dyn</c> twins)
/// against <see cref="TurboQuantCodec"/>'s <c>Rotate</c> / <c>InverseRotate</c>. Runs on framework (RTX 3060).
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaTurboQuantRotationTests
{
    private readonly ITestOutputHelper _out;
    public CudaTurboQuantRotationTests(ITestOutputHelper output) => _out = output;

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

    private static nint Alloc(long bytes) { CudaDriverApi.cuMemAlloc_v2(out nint p, (nuint)bytes).ThrowOnError(); return p; }
    private static void Free(nint p) { if (p != 0) CudaDriverApi.cuMemFree_v2(p); }
    private static void Up<T>(nint dst, T[] src) where T : unmanaged
    { fixed (T* p = src) CudaDriverApi.cuMemcpyHtoD_v2(dst, (nint)p, (nuint)(src.Length * sizeof(T))).ThrowOnError(); }
    private static void Down<T>(T[] dst, nint src) where T : unmanaged
    { fixed (T* p = dst) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, src, (nuint)(dst.Length * sizeof(T))).ThrowOnError(); }

    public static TheoryData<int, int, int> Geometries => new()
    {
        { 64, 3, 2 }, { 128, 8, 4 }, { 256, 2, 3 },
    };

    [SkippableTheory]
    [MemberData(nameof(Geometries))]
    public void RotateQuery_MatchesCodecRotate(int headDim, int numHeads, int rows)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.TqRotateAvailable, "turboquant_rotate.ptx not present/loaded");

        var codecK = new TurboQuantCodec(headDim, 4, TurboQuantRotationTests.Seed);
        float[] signs = codecK.RotationSigns.ToArray();
        var rng = new Random(headDim * 31 + numHeads);
        int n = rows * numHeads * headDim;
        var q = new Half[n];
        float[] qf = TurboQuantRotationTests.Gaussian(rng, n, 3f);
        for (int i = 0; i < n; i++) { q[i] = (Half)qf[i]; qf[i] = (float)q[i]; }
        float[] expected = (float[])qf.Clone();
        for (int v = 0; v < rows * numHeads; v++) codecK.Rotate(expected.AsSpan(v * headDim, headDim));

        foreach (bool dyn in new[] { false, true })
        {
            nint dQ = 0, dSign = 0, dRows = 0;
            try
            {
                dQ = Alloc(n * sizeof(ushort)); dSign = Alloc(signs.Length * sizeof(float)); dRows = Alloc(sizeof(int));
                Up(dQ, q); Up(dSign, signs); Up(dRows, new[] { rows });
                if (dyn) kernels.LaunchTqRotateQDyn(dQ, dRows, rows + 3, numHeads, headDim, dSign, stream.Handle);
                else kernels.LaunchTqRotateQ(dQ, rows, numHeads, headDim, dSign, stream.Handle);
                stream.Synchronize();
                var got = new Half[n];
                Down(got, dQ);
                for (int i = 0; i < n; i++)
                {
                    float e = expected[i], g = (float)got[i];
                    Assert.True(MathF.Abs(e - g) <= 1e-3f * MathF.Abs(e) + 1e-3f, $"dyn={dyn} i={i}: codec={e} gpu={g}");
                }
            }
            finally { Free(dQ); Free(dSign); Free(dRows); }
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Geometries))]
    public void RotateOutput_InvertsCodec(int headDim, int numHeads, int rows)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.TqRotateAvailable, "turboquant_rotate.ptx not present/loaded");

        var codecV = new TurboQuantCodec(headDim, 4, TurboQuantRotationTests.Seed ^ TurboQuantRotationTests.VSeedXor);
        float[] signs = codecV.RotationSigns.ToArray();
        var rng = new Random(headDim * 17 + rows);
        int n = rows * numHeads * headDim;
        float[] a = TurboQuantRotationTests.Gaussian(rng, n, 2f);
        float[] expected = (float[])a.Clone();
        for (int v = 0; v < rows * numHeads; v++) codecV.InverseRotate(expected.AsSpan(v * headDim, headDim));

        foreach (bool dyn in new[] { false, true })
        {
            nint dO = 0, dSign = 0, dRows = 0;
            try
            {
                dO = Alloc((n + 5 * numHeads * headDim) * sizeof(float));
                dSign = Alloc(signs.Length * sizeof(float)); dRows = Alloc(sizeof(int));
                // rows beyond *dRows carry a sentinel the dyn kernel must leave alone
                var init = new float[n + 5 * numHeads * headDim];
                Array.Copy(a, init, n);
                for (int i = n; i < init.Length; i++) init[i] = 123.5f;
                Up(dO, init); Up(dSign, signs); Up(dRows, new[] { rows });
                if (dyn) kernels.LaunchTqRotateODyn(dO, dRows, rows + 5, numHeads, headDim, dSign, stream.Handle);
                else kernels.LaunchTqRotateO(dO, rows, numHeads, headDim, dSign, stream.Handle);
                stream.Synchronize();
                var got = new float[init.Length];
                Down(got, dO);
                double max = 0;
                for (int i = 0; i < n; i++)
                {
                    max = Math.Max(max, Math.Abs(expected[i] - got[i]));
                    Assert.True(MathF.Abs(expected[i] - got[i]) <= 1e-5f * MathF.Max(1f, MathF.Abs(expected[i])),
                        $"dyn={dyn} i={i}: codec={expected[i]} gpu={got[i]}");
                }
                for (int i = n; i < init.Length; i++) Assert.Equal(123.5f, got[i]);
                _out.WriteLine($"d={headDim} dyn={dyn}: maxAbsDiff={max:E3}");
            }
            finally { Free(dO); Free(dSign); Free(dRows); }
        }
    }

    /// <summary>The dyn twin must read the row count from the device pointer at run time (graph-capture safety): shrinking it leaves later rows untouched.</summary>
    [SkippableFact]
    public void Dyn_ReadsRowCountFromDevicePointer()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.TqRotateAvailable, "turboquant_rotate.ptx not present/loaded");

        const int hd = 128, heads = 2, maxRows = 4;
        var codec = new TurboQuantCodec(hd, 4, TurboQuantRotationTests.Seed ^ TurboQuantRotationTests.VSeedXor);
        float[] signs = codec.RotationSigns.ToArray();
        float[] a = TurboQuantRotationTests.Gaussian(new Random(5), maxRows * heads * hd, 1f);
        nint dO = Alloc(a.Length * sizeof(float)), dSign = Alloc(signs.Length * sizeof(float)), dRows = Alloc(sizeof(int));
        try
        {
            Up(dO, a); Up(dSign, signs); Up(dRows, new[] { 1 });
            kernels.LaunchTqRotateODyn(dO, dRows, maxRows, heads, hd, dSign, stream.Handle);
            stream.Synchronize();
            var got = new float[a.Length];
            Down(got, dO);
            int rotated = heads * hd;
            Assert.True(TurboQuantRotationTests.MaxAbsDiff(a[..rotated], got[..rotated]) > 0.1f, "row 0 was not rotated");
            for (int i = rotated; i < a.Length; i++) Assert.Equal(a[i], got[i]);
        }
        finally { Free(dO); Free(dSign); Free(dRows); }
    }
}
