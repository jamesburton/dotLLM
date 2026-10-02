using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// <see cref="IqF16PrefillMatmul"/> (dequantise to F16 scratch, then the F16 coopmat GEMM) against a double-precision CPU matmul over
/// the CPU-dequantised weights. The weights pass through F16, so the tolerance is the F16 rounding of a unit-scale accumulation.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanIqF16PrefillMatmulTests
{
    [SkippableTheory]
    [InlineData(QuantizationType.IQ2_XS, 256, 512, 64)]
    [InlineData(QuantizationType.IQ3_S, 192, 768, 48)]
    [InlineData(QuantizationType.IQ1_S, 128, 512, 33)]
    [InlineData(QuantizationType.IQ2_XXS, 130, 256, 40)]
    public unsafe void TryRecord_MatchesCpuReference(QuantizationType qt, int m, int k, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasCooperativeMatrix, "needs cooperative matrix");
        using var gemm = MatMulF16GemmCoopmatKernel.Create(device, spvDir);
        using var path = IqF16PrefillMatmul.TryCreate(device, spvDir, gemm, (long)m * k)!;
        Assert.NotNull(path);

        int blockBytes = (int)QuantFormat.TryGetInfo(qt)!.Value.BlockBytes;
        int blocks = m * k / 256;
        var rng = new Random(0x621 + (int)qt + m);
        byte[] w = new byte[blocks * blockBytes];
        rng.NextBytes(w);
        for (int b = 0; b < blocks; b++)
            BitConverter.TryWriteBytes(new Span<byte>(w, b * blockBytes, 2), (Half)((rng.NextSingle() * 2f - 1f) * 0.01f));

        float[] deq = new float[m * k];
        fixed (byte* p = w) Dequantize.ToFloat32((nint)p, (long)m * k, qt, deq);
        float[] x = new float[n * k];
        for (int i = 0; i < x.Length; i++) x[i] = rng.NextSingle() * 2f - 1f;

        using var bufW = device.Allocate(((long)w.Length + 3) & ~3L);
        using var bufX = device.Allocate((long)x.Length * sizeof(float));
        using var bufY = device.Allocate((long)n * m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(w), bufW);
        device.Upload(new ReadOnlySpan<float>(x), bufX);

        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            Assert.True(path.TryRecord(ctx.CommandBuffer, qt, bufW, bufX, bufY, m, k, n));
            ctx.SubmitAndWait();
        }

        float[] y = new float[n * m];
        device.Download(bufY, y);

        double worst = 0;
        for (int t = 0; t < n; t++)
            for (int r = 0; r < m; r++)
            {
                double want = 0, mag = 0;
                for (int i = 0; i < k; i++)
                {
                    double term = (double)deq[r * k + i] * x[t * k + i];
                    want += term; mag += Math.Abs(term);
                }
                worst = Math.Max(worst, Math.Abs(y[t * m + r] - want) / Math.Max(mag, 1e-6));
            }
        Assert.True(worst < 5e-3, $"{qt}: worst error relative to sum|w*x| = {worst:E3}");
    }

    [Fact]
    public void Handles_CoversExactlyTheCodebookIqFormats()
    {
        Assert.True(IqF16PrefillMatmul.Handles(QuantizationType.IQ2_XS));
        Assert.True(IqF16PrefillMatmul.Handles(QuantizationType.IQ1_S));
        Assert.False(IqF16PrefillMatmul.Handles(QuantizationType.IQ4_XS));   // has its own blocked coopmat GEMM (#601)
        Assert.False(IqF16PrefillMatmul.Handles(QuantizationType.Q4_K));
    }
}
