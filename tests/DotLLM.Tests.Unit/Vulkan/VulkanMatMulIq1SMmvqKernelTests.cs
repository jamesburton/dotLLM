using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Tolerance + argmax parity for the dp4a IQ1_S MMVQ decode GEMV (issue #339):
/// Q8_1-quantized activation × IQ1_S ternary-grid weights (grid lanes → dp4a, plus
/// the per-sub-block delta·Σx term), compared to the IQ1_S-byte-identical CPU
/// oracle <see cref="Iq1Fixture.CpuGemvIq1S"/>.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulIq1SMmvqKernelTests
{
    private const int GroupSize = 256;
    private const int BlockBytes = 50;
    private const float RelTol = 6e-2f;

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
        using var mmvq = MatMulIq1SMmvqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_iq1_s_mmvq.spv missing or unsupported.");

        var rng = new Random(0x15 + m * 7 + k * 11);
        float[] weightsF32 = Iq1Fixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] x = Iq1Fixture.RandomFloats(rng, k, range: 1.0f);

        int totalBytes = m * (k / GroupSize) * BlockBytes;
        byte[] wq = Iq1Fixture.QuantizeRowsIq1S(weightsF32, m, k);
        Assert.Equal(totalBytes, wq.Length);
        Iq1Fixture.AssertFixtureRoundtripIq1S(weightsF32, wq, m, k);
        float[] expected = Iq1Fixture.CpuGemvIq1S(wq, x, m, k);

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
        Iq4MmvqParity.Assert(expected, actual, m, k, RelTol, "IQ1_S");
    }
}
