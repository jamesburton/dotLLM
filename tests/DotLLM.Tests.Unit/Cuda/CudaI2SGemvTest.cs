using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Validates the CUDA I2_S (BitNet ternary) GEMV against the CPU float reference (MatMul.GemvI2_S).
/// Variant A (W2A16, here the FP32-in twin) is an exact analog of the CPU path, so it must match to
/// fp32 reduction-order error. Uses synthetic packed tensors at real BitNet 2B4T dims — no model file.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public class CudaI2SGemvTest
{
    private readonly ITestOutputHelper _out;
    public CudaI2SGemvTest(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    [SkippableTheory]
    [InlineData(2560, 2560)]  // attention projection
    [InlineData(2560, 6912)]  // FFN down (k = 6912)
    public void I2SGemvF32In_MatchesCpuFloatReference(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        Run(n, k);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void Run(int n, int k)
    {
        var rng = new Random(1234);
        int rowBytes = k / 4;
        long packedLen = (long)n * rowBytes + 4;

        // Random ternary weights + a small positive per-tensor scale at the tensor tail.
        sbyte[] ternary = new sbyte[(long)n * k];
        for (long i = 0; i < ternary.LongLength; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float scale = 0.02f + (float)rng.NextDouble() * 0.03f;
        byte[] packed = Pack(ternary, n, k, scale);

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;

        // CPU reference.
        float[] cpu = new float[n];
        fixed (byte* w = packed)
        fixed (float* px = x, py = cpu)
            MatMul.GemvI2_S(w, px, py, n, k, null);

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

                kernels.LaunchI2_SGemvF32In(devW, devX, devY, n, k, s);
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
        _out.WriteLine($"I2_S GEMV {n}×{k}: max abs diff={maxDiff:E4}, mean={meanDiff:E4}");

        Assert.True(maxDiff <= 1e-3f, $"max abs diff {maxDiff} exceeds 1e-3 (CPU vs GPU should match to fp32)");
        Assert.True(meanDiff <= 1e-4f, $"mean abs diff {meanDiff} exceeds 1e-4");
    }

    [SkippableTheory]
    [InlineData(2560, 2560)]  // attention projection
    [InlineData(2560, 6912)]  // FFN down (k = 6912)
    public void I2SGemvA8_MatchesInt8Reference(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunA8(n, k);
    }

    [SkippableFact]
    public void I2SGemvFusedDecode_MatchesSeparateLaunches()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunFusedDecode();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void RunFusedDecode()
    {
        const int k = 256;
        const int n0 = 37;
        const int n1 = 53;
        const int n2 = 71;

        var rng = new Random(2468);
        byte[] w0 = RandomPacked(rng, n0, k, 0.021f);
        byte[] w1 = RandomPacked(rng, n1, k, 0.034f);
        byte[] w2 = RandomPacked(rng, n2, k, 0.047f);

        Half[] x = new Half[k];
        for (int i = 0; i < k; i++)
            x[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.5f);

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var kernels = new CudaKernels(ptxDir!);
        nint s = stream.Handle;

        static nuint PackedBytes(int n, int k) => (nuint)((long)n * (k / 4) + 4);
        CudaDriverApi.cuMemAlloc_v2(out nint devW0, PackedBytes(n0, k)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devW1, PackedBytes(n1, k)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devW2, PackedBytes(n2, k)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devX, (nuint)(k * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint sep0, (nuint)(n0 * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint sep1, (nuint)(n1 * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint sep2, (nuint)(n2 * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint fus0, (nuint)(n0 * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint fus1, (nuint)(n1 * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint fus2, (nuint)(n2 * sizeof(ushort))).ThrowOnError();
        try
        {
            fixed (byte* p = w0) CudaDriverApi.cuMemcpyHtoD_v2(devW0, (nint)p, PackedBytes(n0, k)).ThrowOnError();
            fixed (byte* p = w1) CudaDriverApi.cuMemcpyHtoD_v2(devW1, (nint)p, PackedBytes(n1, k)).ThrowOnError();
            fixed (byte* p = w2) CudaDriverApi.cuMemcpyHtoD_v2(devW2, (nint)p, PackedBytes(n2, k)).ThrowOnError();
            fixed (Half* p = x) CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)p, (nuint)(k * sizeof(ushort))).ThrowOnError();

            kernels.LaunchI2_SGemvF16In(devW0, devX, sep0, n0, k, s);
            kernels.LaunchI2_SGemvF16In(devW1, devX, sep1, n1, k, s);
            kernels.LaunchI2_SGemvF16In(devW2, devX, sep2, n2, k, s);
            kernels.LaunchI2_SGemv2F16In(devW0, devW1, devX, fus0, fus1, n0, n1, k, s);
            kernels.LaunchI2_SGemv3F16In(devW0, devW1, devW2, devX, fus0, fus1, fus2, n0, n1, n2, k, s);
            stream.Synchronize();

            AssertHalfClose(sep0, fus0, n0);
            AssertHalfClose(sep1, fus1, n1);
            AssertHalfClose(sep2, fus2, n2);
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devW0); CudaDriverApi.cuMemFree_v2(devW1); CudaDriverApi.cuMemFree_v2(devW2);
            CudaDriverApi.cuMemFree_v2(devX);
            CudaDriverApi.cuMemFree_v2(sep0); CudaDriverApi.cuMemFree_v2(sep1); CudaDriverApi.cuMemFree_v2(sep2);
            CudaDriverApi.cuMemFree_v2(fus0); CudaDriverApi.cuMemFree_v2(fus1); CudaDriverApi.cuMemFree_v2(fus2);
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

    private static byte[] RandomPacked(Random rng, int n, int k, float scale)
    {
        sbyte[] ternary = new sbyte[(long)n * k];
        for (long i = 0; i < ternary.LongLength; i++)
            ternary[i] = (sbyte)(rng.Next(3) - 1);
        return Pack(ternary, n, k, scale);
    }

    /// <summary>
    /// Validates Variant B (W2A8, __dp4a) against an INT8 reference computed on the CPU.
    ///
    /// The kernel quantizes nothing itself — activations are quantized on the host (per-token symmetric
    /// absmax, s_act = 127/absmax, xq = round(x·s_act)), and the kernel computes the exact int32 dot
    /// Σ xq_i·(code_i−1), scaled by weightScale·invActScale (invActScale = absmax/127 = 1/s_act). The CPU
    /// int8 reference does the SAME integer dot with the SAME int8 inputs, so the two must match to
    /// integer-math precision (the only float work is the final scale·invActScale multiply + the fp32
    /// reduction, hence ≤ 1e-4, not bit-exact).
    ///
    /// NOTE: the dp4a result vs the *float* CPU reference (MatMul.GemvI2_S over full-precision x) will
    /// differ — that gap is the expected per-token activation-quant error, NOT a kernel bug. We therefore
    /// assert against the int8 reference here, exactly as the spec (§2a, §7 row B0) prescribes.
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void RunA8(int n, int k)
    {
        var rng = new Random(4321);
        int rowBytes = k / 4;
        long packedLen = (long)n * rowBytes + 4;

        sbyte[] ternary = new sbyte[(long)n * k];
        for (long i = 0; i < ternary.LongLength; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float scale = 0.02f + (float)rng.NextDouble() * 0.03f;
        byte[] packed = Pack(ternary, n, k, scale);

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;

        // Host per-token int8 quant: symmetric absmax. s_act = 127/absmax, invActScale = absmax/127.
        float absmax = 0f;
        for (int i = 0; i < k; i++) absmax = MathF.Max(absmax, MathF.Abs(x[i]));
        if (absmax == 0f) absmax = 1f;            // degenerate guard (matches a real quantizer)
        float sAct = 127f / absmax;
        float invActScale = absmax / 127f;
        sbyte[] xq = new sbyte[k];
        for (int i = 0; i < k; i++)
        {
            int q = (int)MathF.Round(x[i] * sAct);
            if (q > 127) q = 127;
            if (q < -127) q = -127;               // symmetric range, mirror BitNet
            xq[i] = (sbyte)q;
        }

        // INT8 CPU reference: out = scale · invActScale · Σ xq_i·(code_i−1). Exact integer dot in long,
        // then the single float epilogue — mirrors the kernel's math.
        float[] cpu = new float[n];
        for (int r = 0; r < n; r++)
        {
            long iacc = 0;
            long rowOff = (long)r * k;
            for (int col = 0; col < k; col++)
                iacc += (long)xq[col] * ternary[rowOff + col]; // ternary already == code-1
            cpu[r] = (float)iacc * scale * invActScale;
        }

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
            CudaDriverApi.cuMemAlloc_v2(out nint devX, (nuint)((long)k * sizeof(sbyte))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint devY, (nuint)((long)n * sizeof(float))).ThrowOnError();
            try
            {
                fixed (byte* w = packed)
                    CudaDriverApi.cuMemcpyHtoD_v2(devW, (nint)w, (nuint)packedLen).ThrowOnError();
                fixed (sbyte* px = xq)
                    CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)px, (nuint)((long)k * sizeof(sbyte))).ThrowOnError();

                kernels.LaunchI2_SGemvA8(devW, devX, devY, n, k, invActScale, s);
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
        _out.WriteLine($"I2_S A8 GEMV {n}×{k}: max abs diff={maxDiff:E4}, mean={meanDiff:E4}");

        // Integer math on both sides → only the float epilogue/reduction differs. Tight tolerance.
        Assert.True(maxDiff <= 1e-4f, $"max abs diff {maxDiff} exceeds 1e-4 (dp4a vs int8 reference)");
        Assert.True(meanDiff <= 1e-5f, $"mean abs diff {meanDiff} exceeds 1e-5");
    }

    /// <summary>
    /// Microbenchmark: Variant A (W2A16, FP16 activations) vs Variant B (W2A8, dp4a int8 activations)
    /// at the GEMV level on the real BitNet projection shapes. Informational — always passes; run with
    /// <c>--logger "console;verbosity=detailed"</c> to see the timings. Decides which variant the
    /// forward integration should dispatch.
    /// </summary>
    [SkippableFact]
    public void Benchmark_I2SGemv_AvsB()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var kernels = new CudaKernels(ptxDir!);
        Bench(kernels, stream, 2560, 2560);
        Bench(kernels, stream, 2560, 6912);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void Bench(CudaKernels kernels, CudaStream stream, int n, int k)
    {
        const int Iters = 500, Warmup = 30;
        var rng = new Random(7);
        int rowBytes = k / 4;
        long packedLen = (long)n * rowBytes + 4;

        sbyte[] ternary = new sbyte[(long)n * k];
        for (long i = 0; i < ternary.LongLength; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        byte[] packed = Pack(ternary, n, k, 0.03f);

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
        Half[] xh = new Half[k];
        for (int i = 0; i < k; i++) xh[i] = (Half)x[i];
        float absmax = 0f; for (int i = 0; i < k; i++) absmax = MathF.Max(absmax, MathF.Abs(x[i]));
        float invAct = absmax / 127f;
        sbyte[] xq = new sbyte[k];
        for (int i = 0; i < k; i++) xq[i] = (sbyte)Math.Clamp((int)MathF.Round(x[i] * 127f / absmax), -127, 127);

        nint s = stream.Handle;
        CudaDriverApi.cuMemAlloc_v2(out nint devW, (nuint)packedLen).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devXh, (nuint)((long)k * 2)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devXq, (nuint)k).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devYh, (nuint)((long)n * 2)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devYf, (nuint)((long)n * 4)).ThrowOnError();
        try
        {
            fixed (byte* w = packed) CudaDriverApi.cuMemcpyHtoD_v2(devW, (nint)w, (nuint)packedLen).ThrowOnError();
            fixed (Half* p = xh) CudaDriverApi.cuMemcpyHtoD_v2(devXh, (nint)p, (nuint)((long)k * 2)).ThrowOnError();
            fixed (sbyte* p = xq) CudaDriverApi.cuMemcpyHtoD_v2(devXq, (nint)p, (nuint)k).ThrowOnError();

            for (int i = 0; i < Warmup; i++) kernels.LaunchI2_SGemvF16In(devW, devXh, devYh, n, k, s);
            stream.Synchronize();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < Iters; i++) kernels.LaunchI2_SGemvF16In(devW, devXh, devYh, n, k, s);
            stream.Synchronize(); sw.Stop();
            double aUs = sw.Elapsed.TotalMilliseconds * 1000.0 / Iters;

            for (int i = 0; i < Warmup; i++) kernels.LaunchI2_SGemvA8(devW, devXq, devYf, n, k, invAct, s);
            stream.Synchronize();
            sw.Restart();
            for (int i = 0; i < Iters; i++) kernels.LaunchI2_SGemvA8(devW, devXq, devYf, n, k, invAct, s);
            stream.Synchronize(); sw.Stop();
            double bUs = sw.Elapsed.TotalMilliseconds * 1000.0 / Iters;

            double wBytes = (double)n * rowBytes; // packed weight bytes read per GEMV
            _out.WriteLine($"I2_S GEMV {n}x{k}:  A(W2A16)={aUs:F1}us {wBytes / aUs / 1e3:F1}GB/s   " +
                           $"B(dp4a)={bUs:F1}us {wBytes / bUs / 1e3:F1}GB/s   B/A={bUs / aUs:F2}x");
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devW); CudaDriverApi.cuMemFree_v2(devXh); CudaDriverApi.cuMemFree_v2(devXq);
            CudaDriverApi.cuMemFree_v2(devYh); CudaDriverApi.cuMemFree_v2(devYf);
        }
    }

    /// <summary>Packs ternary {-1,0,+1} into dotLLM I2_S layout + trailing per-tensor float32 scale.</summary>
    private static byte[] Pack(sbyte[] ternary, int n, int k, float scale)
    {
        int rowBytes = k / 4;
        byte[] buf = new byte[(long)n * rowBytes + 4];
        for (int r = 0; r < n; r++)
        {
            long rowBase = (long)r * rowBytes;
            for (int col = 0; col < k; col++)
            {
                int blk = col / 128, j = col % 128, gi = j / 32, gp = j % 32;
                int code = ternary[(long)r * k + col] + 1; // {-1,0,+1} → {0,1,2}
                buf[rowBase + blk * 32 + gp] |= (byte)(code << (6 - 2 * gi));
            }
        }
        BitConverter.GetBytes(scale).CopyTo(buf, (int)((long)n * rowBytes));
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
