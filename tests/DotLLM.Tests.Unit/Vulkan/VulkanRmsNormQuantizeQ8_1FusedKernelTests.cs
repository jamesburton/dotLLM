using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity tests for the fused (RMSNorm + Q8_1 activation quantize) decode
/// kernel (issue #145): the fused single dispatch must produce BIT-IDENTICAL
/// outputs to the standalone <see cref="RmsNormF32Kernel"/> (subgroup path) →
/// <see cref="QuantizeQ8_1Kernel"/> two-dispatch reference — all three outputs
/// (normalized F32 row, packed int8 xq, per-block (scale, sum) xds) are
/// compared exactly, not with an epsilon. Quantization rounds, so even a 1-ulp
/// drift in the fused rinv would flip int8 values on adversarial inputs; the
/// shapes below include the real model hidden sizes plus a K large enough to
/// exercise the grid-strided block loop (K &gt; 256·32).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanRmsNormQuantizeQ8_1FusedKernelTests
{
    private const float Eps = 1e-5f;

    [SkippableTheory]
    [InlineData(32, 0)]        // minimum: one block
    [InlineData(576, 1)]       // SmolLM-135M hidden
    [InlineData(960, 2)]       // block count not a multiple of subgroup width
    [InlineData(2048, 3)]      // Llama-3.2-1B hidden
    [InlineData(3072, 4)]      // Llama-3.2-3B hidden
    [InlineData(4096, 5)]      // Llama-3.1-8B hidden
    [InlineData(9216, 6)]      // > 256 blocks: exercises the grid-strided loop
    public void Launch_BitIdenticalToStandalonePair(int k, int seedSalt)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasSubgroupArithmetic, "Device lacks subgroup arithmetic.");

        var fused = RmsNormQuantizeQ8_1FusedKernel.TryCreate(device, spvDir);
        Skip.If(fused is null, "Fused rmsnorm+quantize SPV not present.");
        var quantize = QuantizeQ8_1Kernel.TryCreate(device, spvDir);
        Skip.If(quantize is null, "quantize_q8_1 SPV not present.");
        var rmsnorm = RmsNormF32Kernel.Create(device, spvDir);
        Skip.IfNot(rmsnorm.UsesSubgroupReduce,
            "Standalone rmsnorm not on the subgroup path; fused kernel would not be created in production.");

        try
        {
            var rng = new Random(0x145 + seedSalt * 7919);
            float[] hidden = new float[k];
            float[] normWeight = new float[k];
            for (int i = 0; i < k; i++)
            {
                // Mixed magnitudes + exact zeros: stress the amax / round paths.
                hidden[i] = (i % 97 == 0) ? 0f : (float)((rng.NextDouble() * 2.0 - 1.0) * (1 + (i % 13)));
                normWeight[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.5);
            }

            int blocks = k / QuantizeQ8_1Kernel.GroupSize;
            long xqBytes = QuantizeQ8_1Kernel.PackedBytes(k);
            long xdsBytes = QuantizeQ8_1Kernel.ScaleBytes(k);

            using var bufHidden = device.Allocate((long)k * sizeof(float));
            using var bufNormW = device.Allocate((long)k * sizeof(float));
            using var bufNormRef = device.Allocate((long)k * sizeof(float));
            using var bufNormFused = device.Allocate((long)k * sizeof(float));
            using var bufXqRef = device.Allocate(xqBytes);
            using var bufXdsRef = device.Allocate(xdsBytes);
            using var bufXqFused = device.Allocate(xqBytes);
            using var bufXdsFused = device.Allocate(xdsBytes);

            device.Upload(hidden, bufHidden);
            device.Upload(normWeight, bufNormW);

            // ── Reference: standalone rmsnorm → quantize (two submits). ──
            rmsnorm.Launch(bufHidden, bufNormW, bufNormRef, rowCount: 1, n: k, eps: Eps);
            using (var ctx = device.CreateSubmitContext())
            {
                ctx.Begin();
                quantize!.Record(ctx.CommandBuffer, bufNormRef, bufXqRef, bufXdsRef, k);
                ctx.SubmitAndWait();
            }

            // ── Fused single dispatch. ──
            fused!.Launch(bufHidden, bufNormW, bufNormFused, bufXqFused, bufXdsFused, k, Eps);

            float[] normRef = new float[k];
            float[] normFused = new float[k];
            device.Download(bufNormRef, normRef);
            device.Download(bufNormFused, normFused);

            // VulkanDevice.Download only exposes a float span; the packed uint
            // words are compared via bit-cast (exactness is unaffected).
            float[] xqRefF = new float[k / 4];
            float[] xqFusedF = new float[k / 4];
            device.Download(bufXqRef, xqRefF);
            device.Download(bufXqFused, xqFusedF);
            uint[] xqRef = Array.ConvertAll(xqRefF, BitConverter.SingleToUInt32Bits);
            uint[] xqFused = Array.ConvertAll(xqFusedF, BitConverter.SingleToUInt32Bits);

            float[] xdsRef = new float[blocks * 2];
            float[] xdsFused = new float[blocks * 2];
            device.Download(bufXdsRef, xdsRef);
            device.Download(bufXdsFused, xdsFused);

            for (int i = 0; i < k; i++)
                Assert.True(normRef[i].Equals(normFused[i]),
                    $"normOut[{i}] differs (k={k}): ref={normRef[i]:G9} fused={normFused[i]:G9}");
            for (int i = 0; i < xqRef.Length; i++)
                Assert.True(xqRef[i] == xqFused[i],
                    $"xq[{i}] differs (k={k}): ref=0x{xqRef[i]:X8} fused=0x{xqFused[i]:X8}");
            for (int i = 0; i < xdsRef.Length; i++)
                Assert.True(xdsRef[i].Equals(xdsFused[i]),
                    $"xds[{i}] differs (k={k}): ref={xdsRef[i]:G9} fused={xdsFused[i]:G9}");
        }
        finally
        {
            rmsnorm.Dispose();
            quantize?.Dispose();
            fused?.Dispose();
        }
    }

    [SkippableFact]
    public void TryCreate_ReturnsNull_WhenSpvMissing()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();
        var nonexistent = Path.Combine(Path.GetTempPath(), "dotllm-no-spv-here-" + Guid.NewGuid());
        var kernel = RmsNormQuantizeQ8_1FusedKernel.TryCreate(device, nonexistent);
        Assert.Null(kernel);
    }
}
