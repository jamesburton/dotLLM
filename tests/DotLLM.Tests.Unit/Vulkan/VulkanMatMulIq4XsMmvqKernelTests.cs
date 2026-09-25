using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Tolerance + argmax parity for the dp4a IQ4_XS MMVQ decode GEMV (issue #339):
/// Q8_1-quantized activation × IQ4_XS codebook weights, compared to the IQ4_XS-
/// byte-identical CPU oracle <see cref="Iq4Fixture.CpuGemvIq4Xs"/>.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulIq4XsMmvqKernelTests
{
    private const int GroupSize = 256;
    private const int BlockBytes = 136;
    private const float RelTol = 4e-2f;

    [SkippableTheory]
    [InlineData(1, 256)]
    [InlineData(8, 256)]
    [InlineData(4, 512)]
    [InlineData(16, 768)]
    [InlineData(2048, 2048)]
    [InlineData(4096, 1024)]
    [InlineData(1024, 4096)]
    public void Mmvq_MatchesF32Oracle_ArgmaxAndTolerance(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "No integer-dot-product support.");

        using var quant = QuantizeQ8_1Kernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1.spv missing.");
        using var mmvq = MatMulIq4XsMmvqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_iq4_xs_mmvq.spv missing or unsupported.");

        var rng = new Random(0x58 + m * 7 + k * 11);
        float[] weightsF32 = Iq4Fixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] x = Iq4Fixture.RandomFloats(rng, k, range: 1.0f);

        int totalBytes = m * (k / GroupSize) * BlockBytes;
        byte[] wq = Iq4Fixture.QuantizeRowsIq4Xs(weightsF32, m, k);
        Assert.Equal(totalBytes, wq.Length);
        Iq4Fixture.AssertFixtureRoundtripIq4Xs(weightsF32, wq, m, k);
        float[] expected = Iq4Fixture.CpuGemvIq4Xs(wq, x, m, k);

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
        Iq4MmvqParity.Assert(expected, actual, m, k, RelTol, "IQ4_XS");
    }
}
