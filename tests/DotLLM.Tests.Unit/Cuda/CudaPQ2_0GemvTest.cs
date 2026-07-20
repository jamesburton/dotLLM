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
