using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Tolerance + argmax parity test for the dp4a Q4_K MMQ prefill GEMM (issue #340):
/// <c>QuantizeQ8_1RowsKernel</c> (F32 B[N,K] → Q8_1 row-wise) followed by
/// <c>MatMulQ4KMmqKernel</c> (integer-dot GEMM against Q4_K weights, nibble-decoded
/// to int8 with the per-sub-block scale and the asymmetric min term).
/// </summary>
/// <remarks>
/// MMQ is NOT bit-exact vs the F32-in Q4_K GEMM (the activation rows are Q8_1-quantized
/// to int8 first). Compared against the byte-identical CPU F32 oracle
/// <see cref="Q4KFixture.CpuGemmQ4K"/> with per-token argmax (a wrong nibble-unpack /
/// scale / min term / row index shifts the argmax; int8-activation quant does not) plus
/// a loose abs/rel tolerance sized to the activation-quant floor. Skipped when the device
/// lacks <c>VK_KHR_shader_integer_dot_product</c>.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulQ4KMmqKernelTests
{
    private const int Q4KGroupSize = 256;
    private const float RelTol = 3e-2f;

    [SkippableTheory]
    [InlineData(2, 4, 256)]          // one super-block per row, partial tile (bounds)
    [InlineData(1, 1, 256)]          // single-cell output
    [InlineData(17, 33, 512)]        // non-multiple-of-tile N/M, 2 super-blocks
    [InlineData(8, 2048, 2048)]      // Llama-3.2-1B q/o projection (small prefill batch)
    [InlineData(16, 8192, 2048)]     // Llama-3.2-1B gate/up projection
    [InlineData(16, 2048, 8192)]     // Llama-3.2-1B down projection
    [InlineData(7, 4, 768)]          // K=768, blocksPerRow=3 (odd)
    public void Mmq_MatchesF32Oracle_ArgmaxAndTolerance(int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — MMQ path is unavailable here.");

        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var mmq = MatMulQ4KMmqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q4_k_mmq.spv missing or unsupported.");

        var rng = new Random(0x4C + n * 13 + m * 7 + k * 3);
        float[] weightsF32 = Q4KFixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] inputB = Q4KFixture.RandomFloats(rng, n * k, range: 1.0f);

        byte[] weightsQ4K = Q4KFixture.QuantizeRows(weightsF32, m, k);
        float[] expected = Q4KFixture.CpuGemmQ4K(weightsQ4K, inputB, m, k, n);

        long weightsBufBytes = ((long)weightsQ4K.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k));
        using var bufXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ4K), bufW);
        device.Upload(inputB, bufB);

        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, bufB, bufXq, bufXds, n, k);
            DotLLM.Vulkan.Kernels.KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            mmq.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufC, m, k, n);
            ctx.SubmitAndWait();
        }

        float[] actual = new float[n * m];
        device.Download(bufC, actual);

        AssertParity(expected, actual, m, k, n);
    }

    private static void AssertParity(float[] expected, float[] actual, int m, int k, int n)
    {
        Assert.Equal(expected.Length, actual.Length);

        double ss = 0;
        for (int i = 0; i < expected.Length; i++) ss += (double)expected[i] * expected[i];
        float rms = (float)Math.Sqrt(ss / Math.Max(1, expected.Length));
        float absTol = MathF.Max(rms, 1e-6f) * RelTol;

        // Per-token argmax; near-tied maxima can flip under activation quant, so a
        // mismatch only fails when the oracle ranks the kernel's pick materially lower.
        for (int t = 0; t < n; t++)
        {
            int argE = 0, argA = 0;
            for (int i = 1; i < m; i++)
            {
                if (expected[t * m + i] > expected[t * m + argE]) argE = i;
                if (actual[t * m + i] > actual[t * m + argA]) argA = i;
            }
            float oracleMax = expected[t * m + argE];
            float oracleAtMmqArg = expected[t * m + argA];
            Assert.True(argE == argA || (oracleMax - oracleAtMmqArg) <= absTol,
                $"Argmax mismatch (n={n},m={m},k={k}) row {t}: oracle={argE} " +
                $"({oracleMax:G6}), mmq={argA} (oracle@{argA}={oracleAtMmqArg:G6}, " +
                $"gap={oracleMax - oracleAtMmqArg:G6} > absTol={absTol:G6}).");
        }

        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > absTol && rel > RelTol) errors++;
        }
        Assert.True(errors == 0,
            $"MMQ drift exceeded tolerance (n={n},m={m},k={k}): errors={errors}/{expected.Length}, " +
            $"maxAbs={maxAbs:G9}, maxRel={maxRel:G9}, absTol={absTol:G9}.");
    }
}
