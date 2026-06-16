using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Tolerance + argmax parity test for the dp4a Q4_K MMVQ decode GEMV (issue #52):
/// <c>QuantizeQ8_1Kernel</c> (F32 activation → Q8_1) followed by
/// <c>MatMulQ4KMmvqKernel</c> (integer-dot GEMV against the 4-bit Q4_K weights).
/// </summary>
/// <remarks>
/// <para>
/// Like the Q8_0 MMVQ path, this is NOT bit-exact vs the F32-in Q4_K GEMV: the
/// activation vector is quantized to int8 (Q8_1) first. We therefore compare
/// against the same Q4_K-byte-identical CPU oracle the F32-in GEMV test uses
/// (<see cref="Q4KFixture.CpuGemvQ4K"/> — Q4_K weights dequantized on the fly,
/// dotted against FULL-precision FP32 <c>x</c>) with:
/// </para>
/// <list type="bullet">
///   <item><b>argmax-exact</b> — a structurally broken kernel (wrong nibble
///     selection, 6-bit scale unpack, min term, or dp4a packing) shifts the
///     argmax; int8-activation quant does not.</item>
///   <item><b>loose abs/rel tolerance</b> — sized to the int8-activation-quant
///     error floor stacked on top of Q4_K's ~4.5-bit weight rounding. Tight
///     enough to catch a broken kernel, loose enough to admit the expected
///     activation-quant drift.</item>
/// </list>
/// <para>
/// Skipped when the device does not advertise
/// <c>VK_KHR_shader_integer_dot_product</c> — <c>TryCreate</c> returns null
/// there and the model falls back to the F32-in Q4_K GEMV.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulQ4KMmvqKernelTests
{
    private const int Q4KGroupSize = 256;
    private const int Q4KBlockBytes = 144;

    // Activation-quant drift (~1/127 per element, RMS-averaged over K) stacked
    // on Q4_K weight rounding. 4e-2 rel / magnitude-scaled abs sits comfortably
    // above that floor and far below what a broken kernel produces.
    private const float RelTol = 4e-2f;

    [SkippableTheory]
    [InlineData(1, 256)]                  // minimum: 1 super-block per row
    [InlineData(8, 256)]                  // 8 rows, 1 super-block — sanity
    [InlineData(4, 512)]                  // 2 super-blocks per row
    [InlineData(16, 768)]                 // 3 super-blocks per row, non-power-of-2
    [InlineData(2048, 2048)]              // Mistral-7B q/o projection family
    [InlineData(4096, 1024)]              // wide M, exercises workgroup-per-row dispatch
    [InlineData(1024, 4096)]              // long K, 16 super-blocks per row
    public void Mmvq_MatchesF32Oracle_ArgmaxAndTolerance(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — MMVQ path is unavailable here.");

        using var quant = QuantizeQ8_1Kernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1.spv missing.");
        using var mmvq = MatMulQ4KMmvqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q4_k_mmvq.spv missing or unsupported.");

        var rng = new Random(0x52 + m * 7 + k * 11);
        float[] weightsF32 = Q4KFixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] x = Q4KFixture.RandomFloats(rng, k, range: 1.0f);

        int blocksPerRow = k / Q4KGroupSize;
        int rowBytes = blocksPerRow * Q4KBlockBytes;
        int totalBytes = m * rowBytes;
        byte[] weightsQ4K = Q4KFixture.QuantizeRows(weightsF32, m, k);
        Assert.Equal(totalBytes, weightsQ4K.Length);

        // Structural safeguard: the fixture quantiser must agree with the CPU
        // oracle's dequant before the kernel comparison means anything.
        Q4KFixture.AssertFixtureRoundtrip(weightsF32, weightsQ4K, m, k);

        // F32-in oracle: Q4_K weights · FULL-precision x (the result MMVQ approximates).
        float[] expected = Q4KFixture.CpuGemvQ4K(weightsQ4K, x, m, k);

        long weightsBufBytes = ((long)totalBytes + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(k));
        using var bufXds = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(k));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ4K), bufW);
        device.Upload(x, bufX);

        // Quantize x → Q8_1, then MMVQ GEMV — two dispatches in one submit so
        // the barrier between them is exercised exactly as in the forward pass.
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

        // Argmax-exact: a structurally broken kernel moves the argmax.
        int argE = 0, argA = 0;
        for (int i = 1; i < m; i++)
        {
            if (expected[i] > expected[argE]) argE = i;
            if (actual[i] > actual[argA]) argA = i;
        }
        Assert.True(argE == argA,
            $"Argmax mismatch (m={m},k={k}): oracle={argE} ({expected[argE]:G6}), " +
            $"mmvq={argA} ({actual[argA]:G6}).");

        // Output magnitude scale for the abs tolerance (RMS of the oracle).
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
            // Pass when within EITHER the relative OR the magnitude-scaled abs
            // tolerance — small-magnitude outputs have large relative error from
            // activation quant but tiny absolute error, and vice versa.
            if (diff > absTol && rel > RelTol) errors++;
        }
        Assert.True(errors == 0,
            $"Q4_K MMVQ drift exceeded tolerance (m={m},k={k}): errors={errors}/{m}, " +
            $"maxAbs={maxAbs:G9}, maxRel={maxRel:G9}, absTol={absTol:G9}.");
    }
}
