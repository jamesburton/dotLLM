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
