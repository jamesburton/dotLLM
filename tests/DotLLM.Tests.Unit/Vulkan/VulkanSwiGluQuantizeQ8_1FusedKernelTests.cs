using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity tests for the fused (SwiGLU activation + Q8_1 activation quantize)
/// decode kernel (issue #71, sibling of #145's RMSNorm+quantize fusion): the
/// fused single dispatch must produce BIT-IDENTICAL outputs to the standalone
/// <see cref="SwiGluF32Kernel"/> → <see cref="QuantizeQ8_1Kernel"/>
/// two-dispatch reference — all three outputs (gated F32 row, packed int8 xq,
/// per-block (scale, sum) xds) are compared exactly, not with an epsilon.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanSwiGluQuantizeQ8_1FusedKernelTests
{
    [SkippableTheory]
    [InlineData(32, 0)]         // minimum: one block
    [InlineData(1536, 1)]       // SmolLM-135M intermediate size
    [InlineData(2048, 2)]       // block count not a multiple of subgroup width
    [InlineData(5632, 3)]       // Llama-3.2-1B-ish intermediate
    [InlineData(8192, 4)]       // Llama-3.2-3B intermediate
    [InlineData(14336, 5)]      // Llama-3.1-8B intermediate
    [InlineData(9216, 6)]       // > 256 blocks: exercises the grid-strided loop
    public void Launch_BitIdenticalToStandalonePair(int n, int seedSalt)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();

        var fused = SwiGluQuantizeQ8_1FusedKernel.TryCreate(device, spvDir);
        Skip.If(fused is null, "Fused swiglu+quantize SPV not present.");
        var quantize = QuantizeQ8_1Kernel.TryCreate(device, spvDir);
        Skip.If(quantize is null, "quantize_q8_1 SPV not present.");
        var swiglu = SwiGluF32Kernel.Create(device, spvDir);

        try
        {
            var rng = new Random(0x71 + seedSalt * 7919);
            float[] gate = new float[n];
            float[] up = new float[n];
            for (int i = 0; i < n; i++)
            {
                // Mixed magnitudes + exact zeros: stress the amax / round paths.
                gate[i] = (i % 97 == 0) ? 0f : (float)((rng.NextDouble() * 2.0 - 1.0) * (1 + (i % 13)));
                up[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.5);
            }

            int blocks = n / QuantizeQ8_1Kernel.GroupSize;
            long xqBytes = QuantizeQ8_1Kernel.PackedBytes(n);
            long xdsBytes = QuantizeQ8_1Kernel.ScaleBytes(n);

            using var bufGate = device.Allocate((long)n * sizeof(float));
            using var bufUp = device.Allocate((long)n * sizeof(float));
            using var bufSiluRef = device.Allocate((long)n * sizeof(float));
            using var bufSiluFused = device.Allocate((long)n * sizeof(float));
            using var bufXqRef = device.Allocate(xqBytes);
            using var bufXdsRef = device.Allocate(xdsBytes);
            using var bufXqFused = device.Allocate(xqBytes);
            using var bufXdsFused = device.Allocate(xdsBytes);

            device.Upload(gate, bufGate);
            device.Upload(up, bufUp);

            // ── Reference: standalone swiglu -> quantize (two submits). ──
            swiglu.Launch(bufGate, bufUp, bufSiluRef, n);
            using (var ctx = device.CreateSubmitContext())
            {
                ctx.Begin();
                quantize!.Record(ctx.CommandBuffer, bufSiluRef, bufXqRef, bufXdsRef, n);
                ctx.SubmitAndWait();
            }

            // ── Fused single dispatch. ──
            fused!.Launch(bufGate, bufUp, bufSiluFused, bufXqFused, bufXdsFused, n);

            float[] siluRef = new float[n];
            float[] siluFused = new float[n];
            device.Download(bufSiluRef, siluRef);
            device.Download(bufSiluFused, siluFused);

            // VulkanDevice.Download only exposes a float span; the packed uint
            // words are compared via bit-cast (exactness is unaffected).
            float[] xqRefF = new float[n / 4];
            float[] xqFusedF = new float[n / 4];
            device.Download(bufXqRef, xqRefF);
            device.Download(bufXqFused, xqFusedF);
            uint[] xqRef = Array.ConvertAll(xqRefF, BitConverter.SingleToUInt32Bits);
            uint[] xqFused = Array.ConvertAll(xqFusedF, BitConverter.SingleToUInt32Bits);

            float[] xdsRef = new float[blocks * 2];
            float[] xdsFused = new float[blocks * 2];
            device.Download(bufXdsRef, xdsRef);
            device.Download(bufXdsFused, xdsFused);

            for (int i = 0; i < n; i++)
                Assert.True(siluRef[i].Equals(siluFused[i]),
                    $"siluOut[{i}] differs (n={n}): ref={siluRef[i]:G9} fused={siluFused[i]:G9}");
            for (int i = 0; i < xqRef.Length; i++)
                Assert.True(xqRef[i] == xqFused[i],
                    $"xq[{i}] differs (n={n}): ref=0x{xqRef[i]:X8} fused=0x{xqFused[i]:X8}");
            for (int i = 0; i < xdsRef.Length; i++)
                Assert.True(xdsRef[i].Equals(xdsFused[i]),
                    $"xds[{i}] differs (n={n}): ref={xdsRef[i]:G9} fused={xdsFused[i]:G9}");
        }
        finally
        {
            swiglu.Dispose();
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
        var kernel = SwiGluQuantizeQ8_1FusedKernel.TryCreate(device, nonexistent);
        Assert.Null(kernel);
    }
}
