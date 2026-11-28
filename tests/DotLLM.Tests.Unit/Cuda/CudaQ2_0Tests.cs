using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Upstream ggml Q2_0 on CUDA (#2074): the F16 dequant (<c>dequant_q2_0_f16</c>) and the packed 1..8-column GEMV (<c>q2_0_gemv_f32io</c>)
/// against the CPU dequantiser over RANDOM blocks (all four codes - real Bonsai 2.5 tensors never carry code 3). Skipped without a CUDA GPU;
/// as of this PR neither box with an NVIDIA GPU (framework's eGPU is absent, T5500 is offline) could run it, so the kernels are build-verified only.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaQ2_0Tests
{
    private static byte[] RandomBlocks(int rows, int k, int seed)
    {
        var rng = new Random(seed);
        byte[] w = new byte[(long)rows * (k / 64) * 18];
        for (int b = 0; b < rows * (k / 64); b++)
        {
            BitConverter.TryWriteBytes(w.AsSpan(b * 18, 2), (Half)(0.001f + rng.NextSingle() * 0.01f));
            rng.NextBytes(w.AsSpan(b * 18 + 2, 16));
        }
        return w;
    }

    private static float[] CpuDequant(byte[] w, int rows, int k)
    {
        float[] wf = new float[(long)rows * k];
        fixed (byte* p = w)
            for (int r = 0; r < rows; r++)
                Dequantize.ToFloat32((nint)(p + (long)r * (k / 64) * 18), k, QuantizationType.Q2_0, wf.AsSpan(r * k, k));
        return wf;
    }

    [SkippableFact]
    public void Dequant_MatchesCpu()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        const int rows = 5, k = 192;
        var w = RandomBlocks(rows, k, 3);
        var expected = CpuDequant(w, rows, k);
        using var context = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ResolvePtxDir());
        nint dSrc = 0, dDst = 0;
        try
        {
            CudaDriverApi.cuMemAlloc_v2(out dSrc, (nuint)w.Length).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dDst, (nuint)(rows * k * sizeof(ushort))).ThrowOnError();
            fixed (byte* p = w) CudaDriverApi.cuMemcpyHtoD_v2(dSrc, (nint)p, (nuint)w.Length).ThrowOnError();
            kernels.LaunchDequantToF16(dSrc, QuantizationType.Q2_0, dDst, rows * k, stream.Handle);
            stream.Synchronize();
            ushort[] f16 = new ushort[rows * k];
            fixed (ushort* p = f16) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dDst, (nuint)(rows * k * sizeof(ushort))).ThrowOnError();
            for (int i = 0; i < f16.Length; i++)
                Assert.Equal(expected[i], (float)BitConverter.UInt16BitsToHalf(f16[i]), 3);   // d*{-1,0,1,2} is exact in F16
        }
        finally { if (dSrc != 0) CudaDriverApi.cuMemFree_v2(dSrc); if (dDst != 0) CudaDriverApi.cuMemFree_v2(dDst); }
    }

    [SkippableTheory]
    [InlineData(1, 64, 1)]
    [InlineData(37, 640, 1)]
    [InlineData(37, 640, 3)]
    [InlineData(70, 5120, 8)]
    public void Gemv_MatchesCpu(int n, int k, int cols)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        var w = RandomBlocks(n, k, 5 + n + k);
        var wf = CpuDequant(w, n, k);
        var rng = new Random(9);
        float[] x = new float[cols * k];
        for (int i = 0; i < x.Length; i++) x[i] = rng.NextSingle() * 2f - 1f;
        using var context = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ResolvePtxDir());
        nint dW = 0, dX = 0, dY = 0;
        try
        {
            CudaDriverApi.cuMemAlloc_v2(out dW, (nuint)w.Length).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dX, (nuint)(x.Length * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dY, (nuint)(cols * n * sizeof(float))).ThrowOnError();
            fixed (byte* p = w) CudaDriverApi.cuMemcpyHtoD_v2(dW, (nint)p, (nuint)w.Length).ThrowOnError();
            fixed (float* p = x) CudaDriverApi.cuMemcpyHtoD_v2(dX, (nint)p, (nuint)(x.Length * sizeof(float))).ThrowOnError();
            kernels.LaunchQ2_0GemvF32(dW, dX, dY, n, k, cols, stream.Handle);
            stream.Synchronize();
            float[] y = new float[cols * n];
            fixed (float* p = y) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dY, (nuint)(y.Length * sizeof(float))).ThrowOnError();
            for (int c = 0; c < cols; c++)
                for (int r = 0; r < n; r++)
                {
                    double e = 0;
                    for (int j = 0; j < k; j++) e += (double)wf[r * k + j] * x[c * k + j];
                    Assert.True(Math.Abs(y[c * n + r] - e) <= 1e-3 + 1e-4 * Math.Abs(e), $"col {c} row {r}: {y[c * n + r]} vs {e}");
                }
        }
        finally { if (dW != 0) CudaDriverApi.cuMemFree_v2(dW); if (dX != 0) CudaDriverApi.cuMemFree_v2(dX); if (dY != 0) CudaDriverApi.cuMemFree_v2(dY); }
    }

    private static string ResolvePtxDir()
    {
        string? envDir = Environment.GetEnvironmentVariable("DOTLLM_PTX_DIR");
        if (envDir is not null && Directory.Exists(envDir)) return envDir;
        string root = Path.GetDirectoryName(typeof(CudaQ2_0Tests).Assembly.Location)!;
        while (root.Length > 3 && !File.Exists(Path.Combine(root, "dotLLM.slnx"))) root = Path.GetDirectoryName(root)!;
        return Path.Combine(root, "native", "ptx");
    }
}
