using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Tolerance + argmax parity test for the dp4a Q2_K MMVQ decode GEMV (issue #339):
/// <c>QuantizeQ8_1Kernel</c> (F32 activation → Q8_1) then <c>MatMulQ2KMmvqKernel</c>
/// (integer-dot GEMV against the Q2_K weights), compared to the Q2_K-byte-identical
/// CPU oracle <c>Q2KFixture.CpuGemvQ2K</c> with argmax-exact (near-tie tolerant)
/// + a loose abs/rel tolerance sized to the int8-activation quant drift.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulQ2KMmvqKernelTests
{
    private const int GroupSize = 256;
    private const int BlockBytes = 84;
    private const float RelTol = 6e-2f; // Q2_K/Q3_K are the coarsest quants (~2.6/3.4 bpw); the int8-activation drift on a tiny single-output dot runs slightly higher than Q4_K+. Still far below a structural-bug signature (which shifts outputs O(1)).

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
        using var mmvq = MatMulQ2KMmvqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul__mmvq.spv missing or unsupported.");

        var rng = new Random(0x2a + m * 7 + k * 11);
        float[] weightsF32 = Q2KFixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] x = Q2KFixture.RandomFloats(rng, k, range: 1.0f);

        int blocksPerRow = k / GroupSize;
        int totalBytes = m * blocksPerRow * BlockBytes;
        byte[] weightsQ = Q2KFixture.QuantizeRows(weightsF32, m, k);
        Assert.Equal(totalBytes, weightsQ.Length);

        Q2KFixture.AssertFixtureRoundtrip(weightsF32, weightsQ, m, k);

        float[] expected = Q2KFixture.CpuGemvQ2K(weightsQ, x, m, k);

        long weightsBufBytes = ((long)totalBytes + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(k));
        using var bufXds = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(k));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ), bufW);
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
            $"Q2_K MMVQ drift exceeded tolerance (m={m},k={k}): errors={errors}/{m}, " +
            $"maxAbs={maxAbs:G9}, maxRel={maxRel:G9}, absTol={absTol:G9}.");

        bool nearTie = MathF.Abs(expected[argE] - expected[argA]) <= absTol;
        Assert.True(argE == argA || nearTie,
            $"Argmax mismatch beyond near-tie (m={m},k={k}): oracle={argE} ({expected[argE]:G6}), " +
            $"mmvq={argA} (oracle there {expected[argA]:G6}), absTol={absTol:G9}.");
    }
}
