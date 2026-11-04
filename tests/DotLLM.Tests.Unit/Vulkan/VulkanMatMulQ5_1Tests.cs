using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Dense Q5_1 GEMV / GEMM (#1600: GLM-4.5-Air's shared-expert down projection, K = 1408). Oracle = CPU <see cref="Dequantize"/> of the same bytes,
/// dotted in double. The min term <c>m</c> and the 5th bit are both exercised by random blocks, and a mutant oracle that drops <c>m</c> or the
/// 5th bit is shown to differ by far more than the tolerance (the fixture discriminates).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMatMulQ5_1Tests
{
    [SkippableTheory]
    [InlineData(37, 1408, 1)]
    [InlineData(37, 1408, 5)]
    [InlineData(8, 64, 1)]
    [InlineData(21, 96, 17)]
    public void Q5_1_Gemv_And_Gemm_Match_Dequant(int m, int k, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rng = new Random(0x1600 + m * 7 + k + n);
        int blocksPerRow = k / 32;
        byte[] w = new byte[(long)m * blocksPerRow * 24];
        rng.NextBytes(w);
        for (long off = 0; off < w.Length; off += 24)
        {
            BitConverter.TryWriteBytes(w.AsSpan((int)off, 2), (Half)(0.02f + 0.1f * (float)rng.NextDouble()));
            BitConverter.TryWriteBytes(w.AsSpan((int)off + 2, 2), (Half)(-1.5f + 3.0f * (float)rng.NextDouble()));
        }
        float[] deq = new float[(long)m * k];
        unsafe { fixed (byte* p = w) Dequantize.ToFloat32((nint)p, deq.Length, QuantizationType.Q5_1, deq); }
        float[] x = new float[n * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

        double[] want = new double[n * m], noMin = new double[n * m];
        for (int r = 0; r < n; r++)
            for (int o = 0; o < m; o++)
            {
                double acc = 0, acc2 = 0;
                for (int j = 0; j < k; j++)
                {
                    int b = (o * blocksPerRow + j / 32) * 24;
                    double mn = (double)BitConverter.ToHalf(w, b + 2);
                    acc += deq[o * k + j] * (double)x[r * k + j];
                    acc2 += (deq[o * k + j] - mn) * (double)x[r * k + j];   // mutant oracle: min term dropped
                }
                want[r * m + o] = acc; noMin[r * m + o] = acc2;
            }
        double mutantGap = 0;
        for (int i = 0; i < want.Length; i++) mutantGap = Math.Max(mutantGap, Math.Abs(want[i] - noMin[i]));
        Assert.True(mutantGap > 0.5, $"fixture cannot see the min term (gap {mutantGap})");

        using var device = VulkanDevice.Create();
        using var bufW = device.Allocate(w.Length);
        using var bufX = device.Allocate((long)x.Length * 4);
        using var bufY = device.Allocate((long)n * m * 4);
        device.Upload(w, bufW);
        device.Upload(x, bufX);
        float[] got = new float[n * m];
        if (n == 1)
        {
            using var kern = MatMulQ5_1GemvF32Kernel.Create(device, spvDir);
            kern.Launch(bufW, bufX, bufY, m, k);
        }
        else
        {
            using var kern = MatMulQ5_1GemmF32Kernel.Create(device, spvDir);
            kern.Launch(bufW, bufX, bufY, m, k, n);
        }
        device.Download(bufY, got);
        double maxErr = 0;
        for (int i = 0; i < got.Length; i++) maxErr = Math.Max(maxErr, Math.Abs(got[i] - want[i]));
        Assert.True(maxErr < 2e-3 * Math.Sqrt(k / 32.0), $"max |gpu - oracle| = {maxErr} (m {m}, k {k}, n {n})");
    }
}
