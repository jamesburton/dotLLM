using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Tolerance + argmax parity test for the dp4a IQ4_NL MMQ prefill GEMM:
/// <c>QuantizeQ8_1RowsKernel</c> (F32 B[N,K] → Q8_1 row-wise) followed by
/// <c>MatMulIq4NlMmqKernel</c> (integer-dot GEMM against IQ4_NL weights — nibbles
/// codebook-decoded to int8 with a per-block fp16 scale, no min term). Compared
/// against the byte-identical CPU F32 oracle <see cref="Iq4Fixture.CpuGemmIq4Nl"/>.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulIq4NlMmqKernelTests
{
    // IQ4_NL's activation-quant + codebook-decode noise floor on the degenerate 1×1×32
    // case (a single block, single output) sits just above 3e-2; its MMVQ decode test
    // (VulkanMatMulIq4NlMmvqKernelTests) uses the same 4e-2 bound. The real projection
    // shapes pass well inside 3e-2; this only relaxes the tiny-output tail.
    private const float RelTol = 4e-2f;

    [SkippableTheory]
    [InlineData(2, 4, 32)]
    [InlineData(1, 1, 32)]
    [InlineData(17, 33, 64)]
    [InlineData(8, 2048, 2048)]
    [InlineData(16, 8192, 2048)]
    [InlineData(16, 2048, 8192)]
    [InlineData(7, 4, 96)]
    public void Mmq_MatchesF32Oracle_ArgmaxAndTolerance(int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — MMQ path is unavailable here.");

        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var mmq = MatMulIq4NlMmqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_iq4_nl_mmq.spv missing or unsupported.");

        var rng = new Random(0x4E + n * 13 + m * 7 + k * 3);
        float[] weightsF32 = Iq4Fixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] inputB = Iq4Fixture.RandomFloats(rng, n * k, range: 1.0f);

        byte[] weightsIq4Nl = Iq4Fixture.QuantizeRowsIq4Nl(weightsF32, m, k);
        float[] expected = Iq4Fixture.CpuGemmIq4Nl(weightsIq4Nl, inputB, m, k, n);

        long weightsBufBytes = ((long)weightsIq4Nl.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k));
        using var bufXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsIq4Nl), bufW);
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
