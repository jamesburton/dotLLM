using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Tolerance + argmax parity for the dp4a IQ3_XXS MMVQ decode GEMV (issue #339):
/// Q8_1-quantized activation × IQ3_XXS codebook-grid weights (two 4-byte grid
/// lookups per pair, sign·grid → int8 → dp4a), compared to the IQ3_XXS-byte-
/// identical CPU oracle <see cref="Iq3Fixture.CpuGemvIq3Xxs"/>.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulIq3XxsMmvqKernelTests
{
    private const int GroupSize = 256;
    private const int BlockBytes = 98;
    private const float RelTol = 5e-2f;

    [SkippableTheory]
    [InlineData(1, 256)]
    [InlineData(8, 256)]
    [InlineData(4, 512)]
    [InlineData(16, 768)]
    [InlineData(512, 1024)]
    public void Mmvq_MatchesF32Oracle_ArgmaxAndTolerance(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "No integer-dot-product support.");

        using var quant = QuantizeQ8_1Kernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1.spv missing.");
        using var codebooks = Iq3Codebooks.Create(device);
        using var mmvq = MatMulIq3XxsMmvqKernel.TryCreate(device, spvDir, codebooks)
            ?? throw new Xunit.Sdk.XunitException("matmul_iq3_xxs_mmvq.spv missing or unsupported.");

        var rng = new Random(0x33 + m * 7 + k * 11);
        float[] weightsF32 = Iq3Fixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] x = Iq3Fixture.RandomFloats(rng, k, range: 1.0f);

        int totalBytes = m * (k / GroupSize) * BlockBytes;
        byte[] wq = Iq3Fixture.QuantizeRowsIq3Xxs(weightsF32, m, k);
        Assert.Equal(totalBytes, wq.Length);
        Iq3Fixture.AssertFixtureRoundtripIq3Xxs(weightsF32, wq, m, k);
        float[] expected = Iq3Fixture.CpuGemvIq3Xxs(wq, x, m, k);

        using var bufW = device.Allocate(((long)totalBytes + 3) & ~3L);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(k));
        using var bufXds = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(k));
        using var bufY = device.Allocate((long)m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(wq), bufW);
        device.Upload(x, bufX);

        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, bufX, bufXq, bufXds, k);
            DotLLM.Vulkan.Kernels.KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            mmvq.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufY, m, k);
            ctx.SubmitAndWait();
        }
        float[] actual = new float[m];
        device.Download(bufY, actual);
        Iq4MmvqParity.Assert(expected, actual, m, k, RelTol, "IQ3_XXS");
    }
}
