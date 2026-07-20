using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Validates the CUDA PQ2_0 (PrismML Bonsai ternary) GEMV against the CPU float reference
/// (<see cref="MatMul.GemvPQ2_0"/>). The F32-in kernel is an exact analog of the CPU path, so it
/// must match to fp32 reduction-order error. Uses synthetic packed tensors at real Bonsai-27B
/// dims (hidden=5120, ffn=17408 — see <c>qwen35.embedding_length</c>/<c>qwen35.feed_forward_length</c>
/// in the real GGUF) — no model file needed.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public class CudaPQ2_0GemvTest
{
    private readonly ITestOutputHelper _out;
    public CudaPQ2_0GemvTest(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    [SkippableTheory]
    [InlineData(512, 5120)]   // attention-shaped, hidden=5120 (Bonsai qwen35.embedding_length)
    [InlineData(512, 17408)]  // FFN-shaped, k=17408 (Bonsai qwen35.feed_forward_length)
    public void PQ2_0GemvF32In_MatchesCpuFloatReference(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        Run(n, k);
    }

    /// <summary>
    /// Validates the v2 F16 production decode kernel (shared-x staging + warp-per-row,
    /// PQ2_0_ROWS_PER_BLOCK=16) against the CPU float reference. Covers real Bonsai attention
    /// and FFN shapes plus n values that are NOT multiples of 16 (37, 3) to exercise the
    /// tail-row clamp path (<c>min(rowBase+rr, n-1)</c> when a block's last warp only partially
    /// overlaps valid rows, and the n=3 case where a single block covers far more rows than
    /// exist).
    /// </summary>
    [SkippableTheory]
    [InlineData(512, 5120)]
    [InlineData(512, 17408)]
    [InlineData(37, 5120)]
    [InlineData(3, 5120)]
    public void PQ2_0GemvF16In_MatchesCpuFloatReference(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunF16(n, k);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void RunF16(int n, int k)
    {
        var rng = new Random(5678);
        int groupsPerRow = k / 128;
        int rowBytes = groupsPerRow * 34;
        long packedLen = (long)n * rowBytes;

        sbyte[] ternary = new sbyte[(long)n * k];
        for (long i = 0; i < ternary.LongLength; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float[] groupScales = new float[n * groupsPerRow];
        for (int i = 0; i < groupScales.Length; i++) groupScales[i] = 0.01f + rng.NextSingle() * 0.05f;
        byte[] packed = Pack(ternary, groupScales, n, k);

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
        Half[] xh = new Half[k];
        for (int i = 0; i < k; i++) xh[i] = (Half)x[i];

        // CPU reference — full-precision float, same math the kernel approximates in F16.
        float[] cpu = new float[n];
        fixed (byte* w = packed)
        fixed (float* px = x, py = cpu)
            MatMul.GemvPQ2_0(w, px, py, n, k, null);

        // GPU — F16 in/out production path.
        Half[] gpu;
        {
            using var ctx = CudaContext.Create(0);
            using var stream = CudaStream.Create();
            string? ptxDir = FindPtxDir();
            Skip.If(ptxDir == null, "PTX files not found");
            using var kernels = new CudaKernels(ptxDir!);
            nint s = stream.Handle;

            CudaDriverApi.cuMemAlloc_v2(out nint devW, (nuint)packedLen).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint devX, (nuint)((long)k * sizeof(ushort))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint devY, (nuint)((long)n * sizeof(ushort))).ThrowOnError();
            try
            {
                fixed (byte* w = packed)
                    CudaDriverApi.cuMemcpyHtoD_v2(devW, (nint)w, (nuint)packedLen).ThrowOnError();
                fixed (Half* px = xh)
                    CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)px, (nuint)((long)k * sizeof(ushort))).ThrowOnError();

                kernels.LaunchPQ2_0GemvF16In(devW, devX, devY, n, k, s);
                stream.Synchronize();

                gpu = new Half[n];
                fixed (Half* py = gpu)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)py, devY, (nuint)((long)n * sizeof(ushort))).ThrowOnError();
            }
            finally
            {
                CudaDriverApi.cuMemFree_v2(devW);
                CudaDriverApi.cuMemFree_v2(devX);
                CudaDriverApi.cuMemFree_v2(devY);
            }
        }

        float maxAbsDiff = 0, sumAbsDiff = 0;
        for (int i = 0; i < n; i++)
        {
            float d = MathF.Abs(cpu[i] - (float)gpu[i]);
            sumAbsDiff += d;
            if (d > maxAbsDiff) maxAbsDiff = d;
        }
        float meanAbsDiff = sumAbsDiff / n;
        _out.WriteLine($"PQ2_0 GEMV F16In {n}×{k}: max abs diff={maxAbsDiff:E4}, mean={meanAbsDiff:E4}");

        // F16 has ~3 decimal digits of precision; the CPU reference is full float32. Looser than
        // the F32In exact-match test, but still tight — max magnitude here is O(1-10) given the
        // synthetic scale/x ranges, so an absolute 5e-2 bar comfortably separates "F16 rounding"
        // from "wrong row/tail-clamp/shared-stage logic".
        Assert.True(maxAbsDiff <= 5e-2f, $"max abs diff {maxAbsDiff} exceeds 5e-2 (F16 vs CPU float32)");
        Assert.True(meanAbsDiff <= 1e-2f, $"mean abs diff {meanAbsDiff} exceeds 1e-2");
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void Run(int n, int k)
    {
        var rng = new Random(1234);
        int groupsPerRow = k / 128;
        int rowBytes = groupsPerRow * 34;
        long packedLen = (long)n * rowBytes;

        sbyte[] ternary = new sbyte[(long)n * k];
        for (long i = 0; i < ternary.LongLength; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float[] groupScales = new float[n * groupsPerRow];
        for (int i = 0; i < groupScales.Length; i++) groupScales[i] = 0.01f + rng.NextSingle() * 0.05f;
        byte[] packed = Pack(ternary, groupScales, n, k);

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;

        // CPU reference.
        float[] cpu = new float[n];
        fixed (byte* w = packed)
        fixed (float* px = x, py = cpu)
            MatMul.GemvPQ2_0(w, px, py, n, k, null);

        // GPU.
        float[] gpu;
        {
            using var ctx = CudaContext.Create(0);
            using var stream = CudaStream.Create();
            string? ptxDir = FindPtxDir();
            Skip.If(ptxDir == null, "PTX files not found");
            using var kernels = new CudaKernels(ptxDir!);
            nint s = stream.Handle;

            CudaDriverApi.cuMemAlloc_v2(out nint devW, (nuint)packedLen).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint devX, (nuint)((long)k * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint devY, (nuint)((long)n * sizeof(float))).ThrowOnError();
            try
            {
                fixed (byte* w = packed)
                    CudaDriverApi.cuMemcpyHtoD_v2(devW, (nint)w, (nuint)packedLen).ThrowOnError();
                fixed (float* px = x)
                    CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)px, (nuint)((long)k * sizeof(float))).ThrowOnError();

                kernels.LaunchPQ2_0GemvF32In(devW, devX, devY, n, k, s);
                stream.Synchronize();

                gpu = new float[n];
                fixed (float* py = gpu)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)py, devY, (nuint)((long)n * sizeof(float))).ThrowOnError();
            }
            finally
            {
                CudaDriverApi.cuMemFree_v2(devW);
                CudaDriverApi.cuMemFree_v2(devX);
                CudaDriverApi.cuMemFree_v2(devY);
            }
        }

        float maxDiff = 0, sumDiff = 0;
        for (int i = 0; i < n; i++)
        {
            float d = MathF.Abs(cpu[i] - gpu[i]);
            sumDiff += d;
            if (d > maxDiff) maxDiff = d;
        }
        float meanDiff = sumDiff / n;
        _out.WriteLine($"PQ2_0 GEMV {n}×{k}: max abs diff={maxDiff:E4}, mean={meanDiff:E4}");

        Assert.True(maxDiff <= 1e-3f, $"max abs diff {maxDiff} exceeds 1e-3 (CPU vs GPU should match to fp32)");
        Assert.True(meanDiff <= 1e-4f, $"mean abs diff {meanDiff} exceeds 1e-4");
    }

    /// <summary>
    /// Validates the fused 2-way GEMV (<see cref="CudaKernels.LaunchPQ2_0Gemv2F16In"/>, used for
    /// decode-time dense-FFN gate+up and full-attention K+V) produces the same output as two
    /// separate <see cref="CudaKernels.LaunchPQ2_0GemvF16In"/> calls. Uses odd, unequal n0/n1
    /// (mirroring the I2_S fused-decode test) to exercise the row-selection/tail-clamp boundary
    /// between the two virtually-concatenated weight matrices.
    /// </summary>
    [SkippableFact]
    public void PQ2_0GemvFusedDecode_MatchesSeparateLaunches()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunFusedDecode();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void RunFusedDecode()
    {
        const int k = 5120;
        const int n0 = 37;
        const int n1 = 53;

        var rng = new Random(2468);
        byte[] w0 = RandomPacked(rng, n0, k);
        byte[] w1 = RandomPacked(rng, n1, k);

        Half[] x = new Half[k];
        for (int i = 0; i < k; i++)
            x[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.5f);

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var kernels = new CudaKernels(ptxDir!);
        nint s = stream.Handle;

        int groupsPerRow = k / 128;
        int rowBytes = groupsPerRow * 34;
        static nuint PackedBytes(int n, int rowBytes) => (nuint)((long)n * rowBytes);

        CudaDriverApi.cuMemAlloc_v2(out nint devW0, PackedBytes(n0, rowBytes)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devW1, PackedBytes(n1, rowBytes)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devX, (nuint)(k * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint sep0, (nuint)(n0 * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint sep1, (nuint)(n1 * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint fus0, (nuint)(n0 * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint fus1, (nuint)(n1 * sizeof(ushort))).ThrowOnError();
        try
        {
            fixed (byte* p = w0) CudaDriverApi.cuMemcpyHtoD_v2(devW0, (nint)p, PackedBytes(n0, rowBytes)).ThrowOnError();
            fixed (byte* p = w1) CudaDriverApi.cuMemcpyHtoD_v2(devW1, (nint)p, PackedBytes(n1, rowBytes)).ThrowOnError();
            fixed (Half* p = x) CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)p, (nuint)(k * sizeof(ushort))).ThrowOnError();

            kernels.LaunchPQ2_0GemvF16In(devW0, devX, sep0, n0, k, s);
            kernels.LaunchPQ2_0GemvF16In(devW1, devX, sep1, n1, k, s);
            kernels.LaunchPQ2_0Gemv2F16In(devW0, devW1, devX, fus0, fus1, n0, n1, k, s);
            stream.Synchronize();

            AssertHalfClose(sep0, fus0, n0);
            AssertHalfClose(sep1, fus1, n1);
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devW0); CudaDriverApi.cuMemFree_v2(devW1);
            CudaDriverApi.cuMemFree_v2(devX);
            CudaDriverApi.cuMemFree_v2(sep0); CudaDriverApi.cuMemFree_v2(sep1);
            CudaDriverApi.cuMemFree_v2(fus0); CudaDriverApi.cuMemFree_v2(fus1);
        }
    }

    private static unsafe void AssertHalfClose(nint expectedDevice, nint actualDevice, int n)
    {
        Half[] expected = new Half[n];
        Half[] actual = new Half[n];
        fixed (Half* p = expected)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, expectedDevice, (nuint)(n * sizeof(ushort))).ThrowOnError();
        fixed (Half* p = actual)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, actualDevice, (nuint)(n * sizeof(ushort))).ThrowOnError();

        for (int i = 0; i < n; i++)
        {
            float diff = MathF.Abs((float)expected[i] - (float)actual[i]);
            Assert.True(diff <= 1e-3f, $"index {i}: expected {(float)expected[i]}, actual {(float)actual[i]}");
        }
    }

    private static byte[] RandomPacked(Random rng, int n, int k)
    {
        int groupsPerRow = k / 128;
        sbyte[] ternary = new sbyte[(long)n * k];
        for (long i = 0; i < ternary.LongLength; i++)
            ternary[i] = (sbyte)(rng.Next(3) - 1);
        float[] groupScales = new float[n * groupsPerRow];
        for (int i = 0; i < groupScales.Length; i++)
            groupScales[i] = 0.01f + rng.NextSingle() * 0.05f;
        return Pack(ternary, groupScales, n, k);
    }

    /// <summary>Packs ternary {-1,0,+1} into dotLLM PQ2_0 layout: per-128-group Half scale + codes.</summary>
    private static byte[] Pack(sbyte[] ternary, float[] groupScales, int n, int k)
    {
        int groupsPerRow = k / 128;
        int rowBytes = groupsPerRow * 34;
        byte[] buf = new byte[(long)n * rowBytes];

        for (int r = 0; r < n; r++)
        {
            long rowBase = (long)r * rowBytes;
            for (int g = 0; g < groupsPerRow; g++)
            {
                long groupBase = rowBase + (long)g * 34;
                Half scale = (Half)groupScales[r * groupsPerRow + g];
                BitConverter.GetBytes(BitConverter.HalfToUInt16Bits(scale)).CopyTo(buf, (int)groupBase);

                int baseIdx = r * k + g * 128;
                for (int gp = 0; gp < 32; gp++)
                {
                    int code0 = ternary[baseIdx + gp] + 1;
                    int code1 = ternary[baseIdx + gp + 32] + 1;
                    int code2 = ternary[baseIdx + gp + 64] + 1;
                    int code3 = ternary[baseIdx + gp + 96] + 1;
                    buf[groupBase + 2 + gp] = (byte)((code0 << 6) | (code1 << 4) | (code2 << 2) | code3);
                }
            }
        }
        return buf;
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
