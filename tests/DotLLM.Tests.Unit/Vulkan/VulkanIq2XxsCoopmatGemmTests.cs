using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1069: the native IQ2_XXS coopmat GEMM against the CPU dequantizer's double-precision product, and against #1060's F16 path
/// (dequantise to F16 scratch, F16 coopmat GEMM), which stages bit-identical F16 operands and so must agree to accumulation order.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanIq2XxsCoopmatGemmTests
{
    [SkippableTheory]
    [InlineData(70, 256, 32)]     // partial M and N tiles, one super-block
    [InlineData(130, 512, 129)]   // M and N each cross a 128 tile by one
    [InlineData(64, 4096, 40)]    // Bolt's hidden as K
    [InlineData(33, 5376, 64)]    // Bolt's shared-expert width as K (21 super-blocks)
    [InlineData(256, 1024, 256)]  // whole tiles only (the direct-store path)
    [InlineData(70, 256, 2)]      // the smallest prefill batch
    public void NativeGemm_MatchesCpuReference_AndF16Path(int m, int k, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(MatMulIq2XxsGemmCoopmatKernel.IsSupportedOn(device, spvDir), "no wave64 cooperative-matrix device");

        var rng = new Random(m * 131 + k * 7 + n);
        byte[] w = VulkanNemotronHTransformerModelIq3ForwardTests.RandomIq2XxsRows(rng, m, k);
        float[] x = new float[n * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] wf = new float[m * k];
        unsafe
        {
            fixed (byte* pw = w)
                Dequantize.ToFloat32((nint)pw, wf.Length, QuantizationType.IQ2_XXS, wf);
        }
        float[] expected = new float[n * m];
        for (int t = 0; t < n; t++)
            for (int r = 0; r < m; r++)
            {
                double acc = 0;
                for (int c = 0; c < k; c++) acc += (double)wf[r * k + c] * x[t * k + c];
                expected[t * m + r] = (float)acc;
            }

        using var native = MatMulIq2XxsGemmCoopmatKernel.Create(device, spvDir);
        using var f16Gemm = MatMulF16GemmCoopmatKernel.Create(device, spvDir);
        using var f16Path = IqF16PrefillMatmul.TryCreate(device, spvDir, f16Gemm, (long)m * k)
            ?? throw new InvalidOperationException("F16 prefill helper unavailable");
        using var bufW = device.Allocate(w.Length);
        using var bufX = device.Allocate((long)x.Length * sizeof(float));
        using var bufY = device.Allocate((long)n * m * sizeof(float));
        using var bufY16 = device.Allocate((long)n * m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(w), bufW);
        device.Upload(x, bufX);
        native.Launch(bufW, bufX, bufY, m, k, n);
        bool compareF16 = n >= f16Path.MinTokens;   // below its threshold the F16 path declines by design
        if (compareF16)
        {
            using var ctx = device.CreateSubmitContext();
            ctx.Begin();
            Assert.True(f16Path.TryRecord(ctx.CommandBuffer, QuantizationType.IQ2_XXS, bufW, bufX, bufY16, m, k, n), "the F16 path declined");
            ctx.SubmitAndWait();
        }
        float[] actual = new float[n * m], viaF16 = new float[n * m];
        device.Download(bufY, actual);
        device.Download(bufY16, viaF16);

        float maxAbs = expected.Max(MathF.Abs);
        for (int i = 0; i < actual.Length; i++)
        {
            Assert.True(MathF.Abs(actual[i] - expected[i]) <= 4e-3f * maxAbs,
                $"m={m} k={k} n={n} [{i}]: expected {expected[i]}, native {actual[i]} (max |y| {maxAbs})");
            if (compareF16)
                Assert.True(MathF.Abs(actual[i] - viaF16[i]) <= 1e-4f * maxAbs,
                    $"m={m} k={k} n={n} [{i}]: F16 path {viaF16[i]}, native {actual[i]} (same F16 operands)");
        }
    }
}
