using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Tolerance + argmax parity for the dp4a IQ2_XS MMVQ decode GEMV (issue #339):
/// Q8_1-quantized activation × IQ2_XS codebook-grid weights (sign·grid → int8 →
/// dp4a), compared to the IQ2_XS-byte-identical CPU oracle
/// <see cref="Iq2Fixture.CpuGemvIq2Xs"/>.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulIq2XsMmvqKernelTests
{
    private const int GroupSize = 256;
    private const int BlockBytes = 74;
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
        using var codebooks = Iq2Codebooks.Create(device);
        using var mmvq = MatMulIq2XsMmvqKernel.TryCreate(device, spvDir, codebooks)
            ?? throw new Xunit.Sdk.XunitException("matmul_iq2_xs_mmvq.spv missing or unsupported.");

        var rng = new Random(0x2C + m * 7 + k * 11);
        float[] weightsF32 = Iq2Fixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] x = Iq2Fixture.RandomFloats(rng, k, range: 1.0f);

        int totalBytes = m * (k / GroupSize) * BlockBytes;
        byte[] wq = Iq2Fixture.QuantizeRowsIq2Xs(weightsF32, m, k);
        Assert.Equal(totalBytes, wq.Length);
        Iq2Fixture.AssertFixtureRoundtripIq2Xs(weightsF32, wq, m, k);
        float[] expected = Iq2Fixture.CpuGemvIq2Xs(wq, x, m, k);

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
        Iq4MmvqParity.Assert(expected, actual, m, k, RelTol, "IQ2_XS");
    }

    /// <summary>
    /// Production-scale coverage (issue #278): <see cref="Mmvq_MatchesF32Oracle_ArgmaxAndTolerance"/>'s
    /// largest case (m=512, k=1024) already costs the exhaustive nearest-pair
    /// fixture ~4m49s of CPU time, per the issue's own follow-up measurement,
    /// which is why the kernel was previously "unproven at the shapes
    /// production actually uses" (k=2048, m up to the 128256 vocab
    /// projection). <see cref="Iq2Fixture.QuantizeRowsIq2XsFast"/> packs
    /// random-but-valid block bytes directly (no search), so both the
    /// fixture and the CPU dequant oracle stay O(m*k) and this can actually
    /// reach those shapes.
    /// </summary>
    [SkippableTheory]
    [InlineData(4096, 2048)]   // FFN-scale k, large m.
    [InlineData(128256, 2048)] // Llama-3.2 vocab-sized output projection at real hidden_size.
    public void Mmvq_MatchesF32Oracle_AtProductionScale(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "No integer-dot-product support.");

        using var quant = QuantizeQ8_1Kernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1.spv missing.");
        using var codebooks = Iq2Codebooks.Create(device);
        using var mmvq = MatMulIq2XsMmvqKernel.TryCreate(device, spvDir, codebooks)
            ?? throw new Xunit.Sdk.XunitException("matmul_iq2_xs_mmvq.spv missing or unsupported.");

        var rng = new Random(0x278 + m * 7 + k * 11);
        byte[] wq = Iq2Fixture.QuantizeRowsIq2XsFast(rng, m, k);
        float[] x = Iq2Fixture.RandomFloats(rng, k, range: 1.0f);
        float[] expected = Iq2Fixture.CpuGemvIq2Xs(wq, x, m, k);

        using var bufW = device.Allocate(((long)wq.Length + 3) & ~3L);
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
        Iq4MmvqParity.Assert(expected, actual, m, k, RelTol, "IQ2_XS-production-scale");
    }
}
