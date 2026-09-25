using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Tolerance + argmax parity test for the dp4a MoE indexed Q4_K expert-bank
/// matmul (issue #383): <see cref="QuantizeQ8_1RowsKernel"/> (F32 x[N,K] → Q8_1
/// row-wise) followed by <see cref="MoeIndexedMatmulQ4KMmqKernel"/> (integer-dot
/// indexed matmul against Q4_K expert weights).
/// </summary>
/// <remarks>
/// Combines <see cref="VulkanMoeIndexedMatmulQ4_KF32KernelTests"/>'s expert-bank
/// setup (random Q4_K-quantized experts, per-row expert indices, CPU oracle reading
/// the same bytes) with <see cref="VulkanMatMulQ4KMmqKernelTests"/>'s tolerance
/// pattern (MMQ is not bit-exact vs the F32-in kernel — activations are int8-quantized
/// first — so parity is argmax-exact-or-within-tolerance plus a loose abs/rel bound).
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMoeIndexedMatmulQ4KMmqKernelTests
{
    private const float RelTol = 3e-2f;

    [SkippableTheory]
    [InlineData(3, 4, 8, 256, 3)]     // smallest: 1 super-block per row, 3 indexed rows
    [InlineData(2, 4, 16, 256, 2)]    // n=2 rows, both pick the same expert
    [InlineData(4, 4, 16, 512, 3)]    // 2 super-blocks per row
    [InlineData(8, 8, 32, 768, 4)]    // 3 super-blocks per row
    [InlineData(5, 16, 48, 256, 8)]   // wider expert bank, indices spanning more experts
    [InlineData(8, 16, 704, 2816, 8)] // real 26B gate/up shape: K=2816 (11 super-blocks), Ie=704
    public void Launch_MatchesDequantizedCpuReference(int n, int numExperts, int m, int k, int activeExperts)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — MoE MMQ path is unavailable here.");

        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var kernel = MoeIndexedMatmulQ4KMmqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("moe_indexed_matmul_q4_k_q8_1.spv missing or unsupported.");

        var rng = new Random(0x4CB4C + n * 31 + numExperts * 17 + m * 11 + k * 7);
        float[] bankF32 = Q4KFixture.RandomFloats(rng, numExperts * m * k, range: 0.1f);
        float[] x = Q4KFixture.RandomFloats(rng, n * k, range: 1.0f);
        int[] indices = RandomIndices(rng, n, numExperts, activeExperts);

        byte[] bankQ4K = Q4KFixture.QuantizeRows(bankF32, numExperts * m, k);
        Q4KFixture.AssertFixtureRoundtrip(bankF32, bankQ4K, numExperts * m, k);

        // CPU oracle: same F32-in dequant+dot reference the F32 sibling kernel test
        // uses (byte-identical bank read), NOT an int8-quantized reference — the MMQ
        // kernel's own activation quantization is exactly what we're validating the
        // tolerance against, same pattern as VulkanMatMulQ4KMmqKernelTests.
        float[] expected = VulkanMoeIndexedMatmulQ4_KF32KernelTests.CpuIndexedMatmulQ4K(
            bankQ4K, x, indices, m, k, n, numExperts);

        long bankBufBytes = ((long)bankQ4K.Length + 3) & ~3L;
        using var bankBuf = device.Allocate(bankBufBytes);
        using var xBuf = device.Allocate((long)x.Length * sizeof(float));
        using var xqBuf = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k));
        using var xdsBuf = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
        using var idxBuf = device.Allocate((long)indices.Length * sizeof(int));
        using var yBuf = device.Allocate((long)expected.Length * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(bankQ4K), bankBuf);
        device.Upload(x, xBuf);
        device.Upload(MemoryMarshal.AsBytes<int>(indices), idxBuf);

        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, xBuf, xqBuf, xdsBuf, n, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            kernel.Record(ctx.CommandBuffer, bankBuf, xqBuf, xdsBuf, idxBuf, yBuf, m, k, n, numExperts);
            ctx.SubmitAndWait();
        }

        float[] actual = new float[expected.Length];
        device.Download(yBuf, actual);

        AssertParity(expected, actual, m, n);
    }

    private static void AssertParity(float[] expected, float[] actual, int m, int n)
    {
        Assert.Equal(expected.Length, actual.Length);

        double ss = 0;
        for (int i = 0; i < expected.Length; i++) ss += (double)expected[i] * expected[i];
        float rms = (float)Math.Sqrt(ss / Math.Max(1, expected.Length));
        float absTol = MathF.Max(rms, 1e-6f) * RelTol;

        // Per-row argmax; near-tied maxima can flip under activation quant, so a
        // mismatch only fails when the oracle ranks the kernel's pick materially lower.
        for (int row = 0; row < n; row++)
        {
            int argE = 0, argA = 0;
            for (int i = 1; i < m; i++)
            {
                if (expected[row * m + i] > expected[row * m + argE]) argE = i;
                if (actual[row * m + i] > actual[row * m + argA]) argA = i;
            }
            float oracleMax = expected[row * m + argE];
            float oracleAtArg = expected[row * m + argA];
            Assert.True(argE == argA || (oracleMax - oracleAtArg) <= absTol,
                $"Argmax mismatch (m={m},n={n}) row {row}: oracle={argE} " +
                $"({oracleMax:G6}), mmq={argA} (oracle@{argA}={oracleAtArg:G6}, " +
                $"gap={oracleMax - oracleAtArg:G6} > absTol={absTol:G6}).");
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
            $"MoE MMQ drift exceeded tolerance (m={m},n={n}): errors={errors}/{expected.Length}, " +
            $"maxAbs={maxAbs:G9}, maxRel={maxRel:G9}, absTol={absTol:G9}.");
    }

    private static int[] RandomIndices(Random rng, int count, int numExperts, int activePool)
    {
        int pool = Math.Min(activePool, numExperts);
        var unique = new HashSet<int>();
        while (unique.Count < pool)
            unique.Add(rng.Next(numExperts));
        var poolArr = unique.ToArray();
        var indices = new int[count];
        for (int i = 0; i < count; i++)
            indices[i] = poolArr[rng.Next(poolArr.Length)];
        return indices;
    }
}
