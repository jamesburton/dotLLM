using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Tolerance + argmax parity test for the dp4a Q5_K MMVQ decode GEMV (issue #338):
/// <c>QuantizeQ8_1Kernel</c> (F32 activation → Q8_1) followed by
/// <c>MatMulQ5KMmvqKernel</c> (integer-dot GEMV against the 5-bit Q5_K weights).
/// Sibling of <see cref="VulkanMatMulQ4KMmvqKernelTests"/> /
/// <see cref="VulkanMatMulQ6KMmvqKernelTests"/> — compares against the Q5_K-byte-
/// identical CPU oracle (<see cref="Q5KFixture.CpuGemvQ5K"/>) with argmax-exact
/// (near-tie tolerant) + a loose abs/rel tolerance sized to the int8-activation
/// quant drift stacked on Q5_K weight rounding.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulQ5KMmvqKernelTests
{
    private const int Q5KGroupSize = 256;
    private const int Q5KBlockBytes = 176;
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
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — MMVQ path is unavailable here.");

        using var quant = QuantizeQ8_1Kernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1.spv missing.");
        using var mmvq = MatMulQ5KMmvqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q5_k_mmvq.spv missing or unsupported.");

        var rng = new Random(0x5c + m * 7 + k * 11);
        float[] weightsF32 = Q5KFixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] x = Q5KFixture.RandomFloats(rng, k, range: 1.0f);

        int blocksPerRow = k / Q5KGroupSize;
        int totalBytes = m * blocksPerRow * Q5KBlockBytes;
        byte[] weightsQ5K = Q5KFixture.QuantizeRows(weightsF32, m, k);
        Assert.Equal(totalBytes, weightsQ5K.Length);

        Q5KFixture.AssertFixtureRoundtrip(weightsF32, weightsQ5K, m, k);

        float[] expected = Q5KFixture.CpuGemvQ5K(weightsQ5K, x, m, k);

        long weightsBufBytes = ((long)totalBytes + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(k));
        using var bufXds = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(k));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ5K), bufW);
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

        AssertParity(expected, actual, m, k);
    }

    private static void AssertParity(float[] expected, float[] actual, int m, int k)
    {
        Assert.Equal(expected.Length, actual.Length);

        int argE = 0, argA = 0;
        for (int i = 1; i < m; i++)
        {
            if (expected[i] > expected[argE]) argE = i;
            if (actual[i] > actual[argA]) argA = i;
        }

        double ss = 0;
        for (int i = 0; i < m; i++) ss += (double)expected[i] * expected[i];
        float rms = (float)Math.Sqrt(ss / Math.Max(1, m));
        float absTol = MathF.Max(rms, 1e-6f) * RelTol;

        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < m; i++)
        {
            float e = expected[i], a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > absTol && rel > RelTol) errors++;
        }
        Assert.True(errors == 0,
            $"Q5_K MMVQ drift exceeded tolerance (m={m},k={k}): errors={errors}/{m}, " +
            $"maxAbs={maxAbs:G9}, maxRel={maxRel:G9}, absTol={absTol:G9}.");

        // Argmax-exact unless the kernel picked a near-tie (within absTol of the
        // true max) — activation int8 quant can flip a near-tied winner; a broken
        // kernel fails the tolerance check above first.
        bool nearTie = MathF.Abs(expected[argE] - expected[argA]) <= absTol;
        Assert.True(argE == argA || nearTie,
            $"Argmax mismatch beyond near-tie (m={m},k={k}): oracle={argE} ({expected[argE]:G6}), " +
            $"mmvq={argA} (oracle there {expected[argA]:G6}), absTol={absTol:G9}.");
    }
}
