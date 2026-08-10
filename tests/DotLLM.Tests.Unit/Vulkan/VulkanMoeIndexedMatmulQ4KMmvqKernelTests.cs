using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity tests for <c>moe_indexed_matmul_q4_k_mmvq</c> — the dp4a MoE indexed
/// Q4_K decode GEMV (issue #137/#383), i.e. what Gemma-4-26B-A4B and
/// Qwen3.6-35B-A3B actually execute for their gate/up expert projections at
/// decode. Closes the coverage hole reported in issue #309.
/// </summary>
/// <remarks>
/// <para>Two independent assertions:</para>
/// <list type="number">
///   <item><b>Same-tier numerical parity</b> — the <c>xq</c>/<c>xds</c> the shader
///     actually read are downloaded and fed to a scalar C# reimplementation of the
///     shader's integer-dot math (<see cref="CpuIndexedMatmulQ4KQ8_1"/>). Both
///     sides compute the same exact-arithmetic quantity, so the bound is fp32
///     reordering only (2e-4) rather than the cross-tier 3e-2 the F32-oracle
///     siblings must use. Plus argmax agreement with a full-precision oracle.</item>
///   <item><b>Per-row expert indexing</b> — the n-row batched launch must agree
///     BIT-EXACTLY with n separate single-row launches at the same expert. Each
///     output cell is computed independently by one subgroup and the activation
///     quantization is per row, so equality is exact. This discriminates
///     broadcast-style index bugs (every row taking <c>indices[0]</c>), a dropped
///     expert stride, and row/column transposition — none of which a tolerance
///     bound reliably catches.</item>
/// </list>
/// <para>
/// Shapes are deliberately non-degenerate: <c>n</c>, <c>m</c>, <c>numExperts</c>
/// and <c>blocksPerRow</c> are pairwise distinct and mutually non-dividing in the
/// small cases, so an index computed as <c>m</c>, <c>n</c>, <c>i/M</c> or
/// <c>i%M</c> instead of the intended one lands somewhere different.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMoeIndexedMatmulQ4KMmvqKernelTests
{
    private const int Q4KGroupSize = 256;
    private const int Q4KBlockBytes = 144;

    [SkippableTheory]
    [InlineData(5, 7, 12, 256, 4)]     // 1 super-block/row; n≠m≠E, 4 distinct experts
    [InlineData(6, 8, 12, 512, 5)]     // 2 super-blocks/row
    [InlineData(3, 4, 9, 768, 3)]      // 3 super-blocks/row, odd m
    [InlineData(9, 16, 20, 1024, 11)]  // wide bank, 11-expert pool over 9 rows
    [InlineData(8, 16, 704, 2816, 8)]  // real 26B gate/up shape: Ie=704, K=2816
    public void Launch_MatchesSameTierCpuOracle(int n, int numExperts, int m, int k, int activeExperts)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — MoE MMVQ path is unavailable here.");

        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var kernel = MoeIndexedMatmulQ4KMmvqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("moe_indexed_matmul_q4_k_mmvq.spv missing or unsupported.");

        var rng = new Random(0x4C4B4 + n * 31 + numExperts * 17 + m * 11 + k * 7);
        float[] bankF32 = Q4KFixture.RandomFloats(rng, numExperts * m * k, range: 0.1f);
        float[] x = Q4KFixture.RandomFloats(rng, n * k, range: 1.0f);
        int[] indices = MoeMmvqParitySupport.RandomIndices(rng, n, numExperts, activeExperts);

        byte[] bankQ4K = Q4KFixture.QuantizeRows(bankF32, numExperts * m, k);
        Q4KFixture.AssertFixtureRoundtrip(bankF32, bankQ4K, numExperts * m, k);

        long bankBufBytes = ((long)bankQ4K.Length + 3) & ~3L;
        using var bankBuf = device.Allocate(bankBufBytes);
        using var xBuf = device.Allocate((long)x.Length * sizeof(float));
        using var xqBuf = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k));
        using var xdsBuf = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
        using var idxBuf = device.Allocate((long)indices.Length * sizeof(int));
        using var yBuf = device.Allocate((long)n * m * sizeof(float));

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

        float[] actual = new float[(long)n * m];
        device.Download(yBuf, actual);

        // Same-tier oracle over the EXACT int8 activations the shader consumed.
        sbyte[] xq = MoeMmvqParitySupport.DownloadActivationBytes(device, xqBuf, n, k);
        float[] xds = MoeMmvqParitySupport.DownloadActivationScales(device, xdsBuf, n, k);
        float[] sameTier = CpuIndexedMatmulQ4KQ8_1(bankQ4K, xq, xds, indices, m, k, n, numExperts);
        MoeMmvqParitySupport.AssertSameTierParity(sameTier, actual, m, n, "Q4_K MoE MMVQ");

        // Cross-tier sanity: argmax vs the full-precision F32-activation oracle.
        float[] f32 = VulkanMoeIndexedMatmulQ4_KF32KernelTests.CpuIndexedMatmulQ4K(
            bankQ4K, x, indices, m, k, n, numExperts);
        MoeMmvqParitySupport.AssertArgmaxAgreement(f32, actual, m, n, "Q4_K MoE MMVQ");
    }

    [SkippableTheory]
    [InlineData(5, 7, 12, 256)]
    [InlineData(6, 8, 12, 512)]
    [InlineData(9, 16, 20, 1024)]
    public void PerRowExpertIndex_MatchesSingleRowLaunches(int n, int numExperts, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — MoE MMVQ path is unavailable here.");

        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var kernel = MoeIndexedMatmulQ4KMmvqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("moe_indexed_matmul_q4_k_mmvq.spv missing or unsupported.");

        var rng = new Random(0xB4C4B + n * 31 + numExperts * 17 + m * 11 + k * 7);
        float[] bankF32 = Q4KFixture.RandomFloats(rng, numExperts * m * k, range: 0.1f);
        float[] x = Q4KFixture.RandomFloats(rng, n * k, range: 1.0f);
        byte[] bankQ4K = Q4KFixture.QuantizeRows(bankF32, numExperts * m, k);

        // Every row a DIFFERENT expert (cycling the whole bank) — the strongest
        // possible discrimination of the per-row lookup.
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = (i * 3 + 1) % numExperts;

        float[] batched = LaunchBatched(device, quant, kernel, bankQ4K, x, indices, m, k, n, numExperts);

        for (int row = 0; row < n; row++)
        {
            float[] xRow = x.AsSpan(row * k, k).ToArray();
            float[] single = LaunchBatched(
                device, quant, kernel, bankQ4K, xRow, new[] { indices[row] }, m, k, 1, numExperts);

            for (int col = 0; col < m; col++)
                Assert.True(batched[row * m + col].Equals(single[col]),
                    $"Row {row} (expert {indices[row]}), col {col}: batched={batched[row * m + col]:G9} " +
                    $"vs single-row={single[col]:G9} — the per-row expert lookup is not row-independent.");
        }
    }

    private static float[] LaunchBatched(
        VulkanDevice device, QuantizeQ8_1RowsKernel quant, MoeIndexedMatmulQ4KMmvqKernel kernel,
        byte[] bankQ4K, float[] x, int[] indices, int m, int k, int n, int numExperts)
    {
        long bankBufBytes = ((long)bankQ4K.Length + 3) & ~3L;
        using var bankBuf = device.Allocate(bankBufBytes);
        using var xBuf = device.Allocate((long)x.Length * sizeof(float));
        using var xqBuf = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k));
        using var xdsBuf = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
        using var idxBuf = device.Allocate((long)indices.Length * sizeof(int));
        using var yBuf = device.Allocate((long)n * m * sizeof(float));

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

        float[] y = new float[(long)n * m];
        device.Download(yBuf, y);
        return y;
    }

    /// <summary>
    /// Scalar C# reimplementation of <c>moe_indexed_matmul_q4_k_mmvq.comp</c>:
    /// per Q4_K sub-block <c>j</c>, <c>d·sc_j·(d_x·Σ q·xq) − dmin·mn_j·s_j</c>,
    /// with the weight row selected by the per-row expert index. Consumes the
    /// SAME int8 activations the shader read, so the only admissible divergence
    /// is fp32 accumulation order.
    /// </summary>
    private static float[] CpuIndexedMatmulQ4KQ8_1(
        byte[] bankQ4K, sbyte[] xq, float[] xds, int[] indices, int m, int k, int n, int numExperts)
    {
        int superBlocks = k / Q4KGroupSize;
        long rowBytes = (long)superBlocks * Q4KBlockBytes;
        long matrixBytes = m * rowBytes;
        int dsBlocksPerRow = k / MoeMmvqParitySupport.Q8_1GroupSize;
        var y = new float[(long)n * m];

        for (int row = 0; row < n; row++)
        {
            int expert = indices[row];
            if ((uint)expert >= (uint)numExperts) continue; // shader leaves y untouched
            long expertBase = (long)expert * matrixBytes;

            for (int outIdx = 0; outIdx < m; outIdx++)
            {
                long rowBase = expertBase + outIdx * rowBytes;
                float acc = 0f;

                for (int b = 0; b < superBlocks; b++)
                {
                    long blockBase = rowBase + (long)b * Q4KBlockBytes;
                    float d = MoeMmvqParitySupport.ReadHalf(bankQ4K, blockBase);
                    float dmin = MoeMmvqParitySupport.ReadHalf(bankQ4K, blockBase + 2);
                    long scalesBase = blockBase + 4;
                    long qsBase = blockBase + 16;

                    for (int j = 0; j < 8; j++)
                    {
                        UnpackScaleMin(bankQ4K, scalesBase, j, out int sc, out int mn);

                        int pairIdx = j >> 1;      // which 32-byte qs half
                        bool high = (j & 1) != 0;  // odd sub-blocks take the high nibble
                        int dot = 0;
                        int xBase = b * Q4KGroupSize + j * 32;
                        for (int i = 0; i < 32; i++)
                        {
                            byte packed = bankQ4K[qsBase + pairIdx * 32 + i];
                            int nib = high ? (packed >> 4) : (packed & 0xF);
                            dot += nib * xq[(long)row * k + xBase + i];
                        }

                        int dsIdx = (row * dsBlocksPerRow + b * 8 + j) * 2;
                        float dx = xds[dsIdx];
                        float s = xds[dsIdx + 1];
                        acc += (d * sc) * (dx * dot) - (dmin * mn) * s;
                    }
                }
                y[(long)row * m + outIdx] = acc;
            }
        }
        return y;
    }

    /// <summary>get_scale_min_k4 — the 6-bit (scale, min) pair <paramref name="j"/> from the 12 packed bytes.</summary>
    private static void UnpackScaleMin(byte[] blob, long scalesBase, int j, out int sc, out int mn)
    {
        if (j < 4)
        {
            sc = blob[scalesBase + j] & 63;
            mn = blob[scalesBase + j + 4] & 63;
        }
        else
        {
            sc = (blob[scalesBase + j + 4] & 0xF) | (((blob[scalesBase + j - 4] >> 6) & 0x3) << 4);
            mn = ((blob[scalesBase + j + 4] >> 4) & 0xF) | (((blob[scalesBase + j] >> 6) & 0x3) << 4);
        }
    }
}
