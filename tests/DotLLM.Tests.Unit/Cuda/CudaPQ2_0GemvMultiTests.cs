using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Kernel parity for the small-S multi-column PQ2_0 GEMV (<c>pq2_0_gemv_multi_f16x_f32y_{S}</c>,
/// issue #482) against <c>S</c> independent launches of the production single-column kernel
/// (<see cref="CudaKernels.LaunchPQ2_0GemvF32Native"/>).
/// </summary>
/// <remarks>
/// <para>
/// <b>Two oracles.</b> (A) the single-column GPU kernel, one launch per column at
/// <c>x + s·k</c> / <c>y + s·n</c>. Both kernels round the activations to half the same way
/// (convert_f32_to_f16 vs the single-column kernel's staging), so the only difference is FP32
/// reassociation and the bound is tight. (B) the CPU scalar reference per column at the existing
/// single-column tolerance, so the two GPU kernels cannot agree on the same wrong answer.
/// </para>
/// <para>
/// <b>What a mutant looks like.</b> Reading column 1's activations from the wrong window —
/// in <c>pq2_0_gemv_multi.cu</c>, <c>x + (size_t)s * k + xElem</c> →
/// <c>x + (size_t)s * k + xElem + (s == 1 ? 16 : 0)</c> (16 halfs keeps the 16-byte alignment the
/// <c>uint4</c> loads need; an odd offset would fault with "misaligned address" instead of failing
/// numerically) — moves column 1 by O(|y|) while every other column still passes; oracle A's bound
/// is ~1e-4·max|y|, and oracle B fails it too. Swapping two columns' output rows or mis-mapping the
/// 16 codes of a lane's word fails the same way; k = 640 (5 groups) exercises the tail-group
/// predicate and n = 37/513 the row clamp.
/// </para>
/// <para>
/// Shapes: k ∈ {5120, 17408} (Bonsai 2 hidden / ffn), k = 640 (5 groups, so the last warp step
/// has only one live group of four), n ∈ {37, 513} (neither a multiple of the 16-row block).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaPQ2_0GemvMultiTests
{
    private readonly ITestOutputHelper _out;
    public CudaPQ2_0GemvMultiTests(ITestOutputHelper output) => _out = output;

    public static TheoryData<int, int, int> Shapes()
    {
        var data = new TheoryData<int, int, int>();
        for (int s = 1; s <= CudaKernels.Pq2_0GemvMultiMaxColumns; s++)
        {
            data.Add(s, 37, 5120);
            data.Add(s, 513, 17408);
        }
        data.Add(3, 513, 5120);
        data.Add(5, 37, 17408);
        data.Add(4, 37, 640);
        data.Add(7, 513, 640);
        return data;
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void MultiColumn_MatchesIndependentSingleColumnLaunches(int columns, int n, int k)
    {
        string ptxDir = SkipUnlessMultiKernel();
        Run(ptxDir, columns, n, k);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void Run(string ptxDir, int columns, int n, int k)
    {
        var rng = new Random(482 + 31 * columns + n + k);
        byte[] packed = RandomPackedPQ2_0(rng, n, k);
        float[] x = new float[columns * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;

        float[] single = new float[columns * n];
        float[] multi = new float[columns * n];
        {
            using var ctx = CudaContext.Create(0);
            using var stream = CudaStream.Create();
            using var kernels = new CudaKernels(ptxDir);
            Assert.True(kernels.HasPQ2_0GemvMulti, kernels.PQ2_0GemvMultiUnavailableReason);
            nint s = stream.Handle;

            long packedLen = packed.LongLength;
            long splitLen = CudaKernels.PQ2_0SplitLayoutBytes(n, k);
            nint devW = Alloc(packedLen), devWSplit = Alloc(splitLen);
            nint devX = Alloc((long)x.Length * sizeof(float));
            nint devXh = Alloc((long)x.Length * sizeof(ushort));
            nint devYSingle = Alloc((long)columns * n * sizeof(float));
            nint devYMulti = Alloc((long)columns * n * sizeof(float));
            try
            {
                fixed (byte* w = packed)
                    CudaDriverApi.cuMemcpyHtoD_v2(devW, (nint)w, (nuint)packedLen).ThrowOnError();
                fixed (float* px = x)
                    CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)px, (nuint)((long)x.Length * sizeof(float))).ThrowOnError();
                kernels.LaunchPQ2_0RepackSplitF16(devW, devWSplit, n, k, s);

                for (int c = 0; c < columns; c++)
                    kernels.LaunchPQ2_0GemvF32Native(devWSplit,
                        devX + (nint)((long)c * k * sizeof(float)),
                        devYSingle + (nint)((long)c * n * sizeof(float)), n, k, s);

                // Same staging as CudaQwen3HybridDenseTransformerModel.TryPQ2_0GemvMulti.
                kernels.LaunchConvertF32ToF16(devX, devXh, x.Length, s);
                kernels.LaunchPQ2_0GemvMulti(devWSplit, devXh, devYMulti, n, k, columns, s);
                stream.Synchronize();

                fixed (float* p = single)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devYSingle, (nuint)((long)single.Length * sizeof(float))).ThrowOnError();
                fixed (float* p = multi)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devYMulti, (nuint)((long)multi.Length * sizeof(float))).ThrowOnError();
            }
            finally
            {
                CudaDriverApi.cuMemFree_v2(devW);
                CudaDriverApi.cuMemFree_v2(devWSplit);
                CudaDriverApi.cuMemFree_v2(devX);
                CudaDriverApi.cuMemFree_v2(devXh);
                CudaDriverApi.cuMemFree_v2(devYSingle);
                CudaDriverApi.cuMemFree_v2(devYMulti);
            }
        }

        // Oracle A — per column, so a single-column fault is named.
        float maxRef = 0;
        foreach (float v in single) maxRef = MathF.Max(maxRef, MathF.Abs(v));
        float tol = 1e-4f * MathF.Max(1f, maxRef);
        for (int c = 0; c < columns; c++)
        {
            float maxDiff = 0;
            int worst = 0;
            for (int r = 0; r < n; r++)
            {
                float d = MathF.Abs(single[c * n + r] - multi[c * n + r]);
                if (d > maxDiff) { maxDiff = d; worst = r; }
            }
            _out.WriteLine($"S={columns} n={n} k={k} col {c}: max|multi-single|={maxDiff:E3} (row {worst}), tol {tol:E3}, max|y|={maxRef:F3}");
            Assert.True(maxDiff <= tol,
                $"S={columns} n={n} k={k}: column {c} row {worst} multi={multi[c * n + worst]} single={single[c * n + worst]} " +
                $"(|diff| {maxDiff:E3} > {tol:E3})");
        }

        // Oracle B — CPU scalar reference, same bar as CudaPQ2_0GemvTest's single-column tests.
        float[] cpu = new float[n];
        fixed (byte* w = packed)
        fixed (float* px = x, py = cpu)
        {
            for (int c = 0; c < columns; c++)
            {
                MatMul.GemvPQ2_0Scalar(w, px + (long)c * k, py, n, k);
                float maxAbs = 0, sumAbs = 0;
                for (int r = 0; r < n; r++)
                {
                    float d = MathF.Abs(cpu[r] - multi[c * n + r]);
                    sumAbs += d;
                    maxAbs = MathF.Max(maxAbs, d);
                }
                Assert.True(maxAbs <= 5e-2f, $"S={columns} column {c}: max |multi-cpu| {maxAbs} exceeds 5e-2");
                Assert.True(sumAbs / n <= 1e-2f, $"S={columns} column {c}: mean |multi-cpu| {sumAbs / n} exceeds 1e-2");
            }
        }
    }

    /// <summary>
    /// Random interleaved (on-disk) PQ2_0 rows: per 128-element group an fp16 scale then 32 code
    /// bytes, every code in {0, 1, 2}.
    /// </summary>
    internal static byte[] RandomPackedPQ2_0(Random rng, int n, int k)
    {
        int groupsPerRow = k / 128;
        byte[] buf = new byte[(long)n * groupsPerRow * 34];
        for (long g = 0; g < (long)n * groupsPerRow; g++)
        {
            long gb = g * 34;
            Half scale = (Half)(0.01f + rng.NextSingle() * 0.05f);
            ushort bits = BitConverter.HalfToUInt16Bits(scale);
            buf[gb] = (byte)bits;
            buf[gb + 1] = (byte)(bits >> 8);
            for (int b = 0; b < 32; b++)
                buf[gb + 2 + b] = (byte)(rng.Next(3) | (rng.Next(3) << 2) | (rng.Next(3) << 4) | (rng.Next(3) << 6));
        }
        return buf;
    }

    private static nint Alloc(long bytes)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint p, (nuint)bytes).ThrowOnError();
        return p;
    }

    /// <summary>
    /// Skips without a CUDA GPU or without <c>pq2_0_gemv_multi.ptx</c>; a present-but-stale PTX is
    /// a real failure (asserted inside the test).
    /// </summary>
    internal static string SkipUnlessMultiKernel()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        Skip.IfNot(File.Exists(Path.Combine(ptxDir!, "pq2_0_gemv_multi.ptx")),
            "pq2_0_gemv_multi.ptx not generated (run native/build_ptx.bat on a CUDA box)");
        return ptxDir!;
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
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }
}
