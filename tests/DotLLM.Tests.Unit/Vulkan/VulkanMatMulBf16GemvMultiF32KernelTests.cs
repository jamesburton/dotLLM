using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity for the BF16 multi-column GEMV (issue #706): every column equals the single-column GEMV's result to a few ULP
/// (same accumulation and tree-reduce order; the compiler contracts mul+add differently).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulBf16GemvMultiF32KernelTests
{
    [SkippableTheory]
    [InlineData(48, 5120, 1)]
    [InlineData(48, 5120, 2)]
    [InlineData(48, 5120, 4)]
    [InlineData(48, 5120, 8)]
    [InlineData(576, 768, 3)]
    [InlineData(17, 128, 5)]
    public void Launch_EachColumnMatchesSingleColumnGemv(int m, int k, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x707 + m * 7 + k * 11 + n);
        float[] weightsF32 = F16Bf16Fixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] x = F16Bf16Fixture.RandomFloats(rng, n * k, range: 1.0f);
        byte[] weightsBf16 = F16Bf16Fixture.QuantizeRowsBf16(weightsF32, m, k);

        using var device = VulkanDevice.Create();
        using var multi = MatMulBf16GemvMultiF32Kernel.Create(device, spvDir);
        using var single = MatMulBf16GemvF32Kernel.Create(device, spvDir);

        using var bufW = device.Allocate(((long)weightsBf16.Length + 3) & ~3L);
        using var bufX = device.Allocate((long)n * k * sizeof(float));
        using var bufY = device.Allocate((long)n * m * sizeof(float));
        using var bufX1 = device.Allocate((long)k * sizeof(float));
        using var bufY1 = device.Allocate((long)m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(weightsBf16), bufW);
        device.Upload(x, bufX);

        multi.Launch(bufW, bufX, bufY, m, k, n);
        float[] actual = new float[n * m];
        device.Download(bufY, actual);

        for (int c = 0; c < n; c++)
        {
            device.Upload(x.AsSpan(c * k, k).ToArray(), bufX1);
            single.Launch(bufW, bufX1, bufY1, m, k);
            float[] expected = new float[m];
            device.Download(bufY1, expected);
            for (int i = 0; i < m; i++)
                Assert.InRange(Math.Abs(expected[i] - actual[c * m + i]), 0f, 1e-5f + 2e-5f * Math.Abs(expected[i]));   // FMA contraction differs by a few ULP
        }
    }
}
