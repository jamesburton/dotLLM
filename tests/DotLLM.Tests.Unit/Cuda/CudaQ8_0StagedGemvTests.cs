using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #482: the shared-memory-staged Q8_0 GEMV (<c>q8_0_gemv_f32in_staged</c>) must be
/// <b>bit-identical</b> to the kernel it replaces on the MTP path
/// (<see cref="CudaKernels.LaunchQuantizedGemvF32In"/>, <c>quantized_gemv_q8_0_f32in</c>) — it keeps
/// the per-thread block ownership, fma order and reduction tree and only changes how the operands
/// reach the threads. The comparison is exact (bit patterns), not a tolerance, because that is the
/// claim that lets the MTP draft logits stay bit-identical.
/// </summary>
/// <remarks>
/// Shapes cover every Bonsai 2 MTP-head projection, plus geometry edge cases: an odd row count (the
/// second row slot of the last thread block is inactive), a block count that is odd (unaligned row
/// stride — byte staging path) and one that leaves a 1-block final chunk. Also prints the per-launch
/// time of both kernels for the Bonsai 2 shapes. Skips when the staged PTX has not been generated.
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaQ8_0StagedGemvTests
{
    private readonly ITestOutputHelper _out;

    public CudaQ8_0StagedGemvTests(ITestOutputHelper output) => _out = output;

    [SkippableTheory]
    [InlineData(5120, 10240, true)]   // nextn.eh_proj
    [InlineData(12288, 5120, true)]   // attn_q (gated: 2 * 24 * 256)
    [InlineData(1024, 5120, true)]    // attn_k / attn_v
    [InlineData(5120, 6144, true)]    // attn_output
    [InlineData(17408, 5120, true)]   // ffn_gate / ffn_up
    [InlineData(5120, 17408, true)]   // ffn_down (bpr 544: 256 + 256 + 32)
    [InlineData(3, 96, false)]        // odd n; bpr 3 -> 102-byte row stride, byte staging
    [InlineData(7, 32 * 257, false)]  // odd n; 1-block final chunk, 2-byte-aligned rows
    [InlineData(1, 32, false)]        // single block
    [InlineData(65, 32 * 300, false)] // bpr 300: row stride 8 mod 16 -> mixed 16-/4-byte staging
    public void Staged_IsBitIdentical_ToQuantizedGemvF32In(int n, int k, bool time)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        Skip.IfNot(File.Exists(Path.Combine(ptxDir!, CudaQ8_0StagedGemv.PtxFileName)),
            $"{CudaQ8_0StagedGemv.PtxFileName} not generated (nvcc -ptx -arch=compute_75 on a CUDA box)");
        Skip.If(CudaQ8_0StagedGemv.DisabledByEnv, "DOTLLM_CUDA_Q8_STAGED_GEMV=0");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        using var staged = CudaQ8_0StagedGemv.TryLoad(ptxDir!);
        // Present but unloadable is a real failure, not a skip.
        Assert.NotNull(staged);

        var rng = new Random(0x482 + n * 7 + k);
        int bpr = k / 32;
        byte[] w = new byte[(long)n * bpr * 34];
        for (long b = 0; b < (long)n * bpr; b++)
        {
            long o = b * 34;
            ushort d = BitConverter.HalfToUInt16Bits((Half)(float)(rng.NextDouble() * 0.02 + 1e-4));
            w[o] = (byte)d;
            w[o + 1] = (byte)(d >> 8);
            for (int j = 0; j < 32; j++) w[o + 2 + j] = (byte)(sbyte)rng.Next(-128, 128);
        }
        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

        nint dW = 0, dX = 0, dY0 = 0, dY1 = 0;
        try
        {
            dW = Upload(w);
            dX = Upload(x);
            CudaDriverApi.cuMemAlloc_v2(out dY0, (nuint)(n * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dY1, (nuint)(n * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemsetD8_v2(dY1, 0xFF, (nuint)(n * sizeof(float))).ThrowOnError(); // NaN poison

            kernels.LaunchQuantizedGemvF32In(dW, dX, dY0, n, k, stream.Handle);
            staged!.Launch(dW, dX, dY1, n, k, stream.Handle);
            stream.Synchronize();

            float[] y0 = Download(dY0, n), y1 = Download(dY1, n);
            int exact = 0, firstBad = -1;
            for (int i = 0; i < n; i++)
            {
                if (BitConverter.SingleToInt32Bits(y0[i]) == BitConverter.SingleToInt32Bits(y1[i])) exact++;
                else if (firstBad < 0) firstBad = i;
            }
            _out.WriteLine($"n={n} k={k}: {exact}/{n} bit-identical");

            if (time)
            {
                double t0 = TimeMs(() => kernels.LaunchQuantizedGemvF32In(dW, dX, dY0, n, k, stream.Handle), stream);
                double t1 = TimeMs(() => staged.Launch(dW, dX, dY1, n, k, stream.Handle), stream);
                double mb = w.Length / 1e6;
                _out.WriteLine($"  original {t0:F3} ms ({mb / t0:F0} GB/s)  staged {t1:F3} ms ({mb / t1:F0} GB/s)  " +
                               $"speedup {t0 / t1:F2}x");
            }

            Assert.True(firstBad < 0,
                firstBad < 0 ? "" : $"row {firstBad}: original {y0[firstBad]:R} vs staged {y1[firstBad]:R}");
        }
        finally
        {
            if (dW != 0) CudaDriverApi.cuMemFree_v2(dW);
            if (dX != 0) CudaDriverApi.cuMemFree_v2(dX);
            if (dY0 != 0) CudaDriverApi.cuMemFree_v2(dY0);
            if (dY1 != 0) CudaDriverApi.cuMemFree_v2(dY1);
        }
    }

    private static double TimeMs(Action launch, CudaStream stream)
    {
        for (int i = 0; i < 3; i++) launch();
        stream.Synchronize();
        const int iters = 20;
        long t = Stopwatch.GetTimestamp();
        for (int i = 0; i < iters; i++) launch();
        stream.Synchronize();
        return (Stopwatch.GetTimestamp() - t) * 1000.0 / Stopwatch.Frequency / iters;
    }

    private static unsafe nint Upload<T>(T[] data) where T : unmanaged
    {
        long bytes = (long)data.Length * sizeof(T);
        CudaDriverApi.cuMemAlloc_v2(out nint d, (nuint)bytes).ThrowOnError();
        fixed (T* p = data) CudaDriverApi.cuMemcpyHtoD_v2(d, (nint)p, (nuint)bytes).ThrowOnError();
        return d;
    }

    private static unsafe float[] Download(nint d, int n)
    {
        var r = new float[n];
        fixed (float* p = r) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, d, (nuint)(n * sizeof(float))).ThrowOnError();
        return r;
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
