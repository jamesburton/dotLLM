using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity tests for <c>moe_indexed_matmul_q5_1_mmvq</c> — the dp4a MoE indexed
/// Q5_1 decode GEMV with the per-expert output-scale fold
/// (Gemma-4 <c>ffn_down_exps.scale[e]</c>). Closes the coverage hole reported in
/// issue #309.
/// </summary>
/// <remarks>
/// Same two-assertion design as
/// <see cref="VulkanMoeIndexedMatmulQ4KMmvqKernelTests"/> plus a third axis the
/// other two formats do not have: the per-expert <c>downScale</c>. The scales are
/// deliberately DISTINCT per expert, so a kernel that dropped the fold, applied
/// it before the reduction, or looked it up with the wrong expert index fails.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMoeIndexedMatmulQ5_1MmvqKernelTests
{
    private const int Q5_1GroupSize = 32;
    private const int Q5_1BlockBytes = 24;

    [SkippableTheory]
    [InlineData(5, 7, 12, 32, 4)]     // 1 block/row; n≠m≠E, 4 distinct experts
    [InlineData(6, 8, 12, 96, 5)]     // 3 blocks/row (odd) — window-tail path
    [InlineData(3, 4, 9, 256, 3)]     // 8 blocks/row = exactly one lane window
    [InlineData(9, 16, 20, 288, 11)]  // 9 blocks/row — window + 1-block tail
    [InlineData(8, 16, 2816, 704, 8)] // real 26B down shape: M=hidden 2816, K=Ie 704
    public void Launch_MatchesSameTierCpuOracle(int n, int numExperts, int m, int k, int activeExperts)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — MoE MMVQ path is unavailable here.");

        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var kernel = MoeIndexedMatmulQ5_1MmvqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("moe_indexed_matmul_q5_1_mmvq.spv missing or unsupported.");

        var rng = new Random(0x5C15C + n * 31 + numExperts * 17 + m * 11 + k * 7);
        float[] bankF32 = RandomFloats(rng, numExperts * m * k, range: 0.25f);
        float[] x = RandomFloats(rng, n * k, range: 1.0f);
        int[] indices = MoeMmvqParitySupport.RandomIndices(rng, n, numExperts, activeExperts);

        // DISTINCT per-expert scales — discriminates a dropped / mis-indexed fold.
        float[] downScale = new float[numExperts];
        for (int e = 0; e < numExperts; e++) downScale[e] = 0.5f + 0.13f * e;

        byte[] bankQ5_1 = Quantize.FromFloat32(bankF32, (long)numExperts * m * k, QuantizationType.Q5_1);

        float[] actual = Launch(device, quant, kernel, bankQ5_1, x, indices, downScale,
            m, k, n, numExperts, out sbyte[] xq, out float[] xds);

        float[] sameTier = CpuIndexedMatmulQ5_1Q8_1(bankQ5_1, xq, xds, indices, downScale, m, k, n, numExperts);
        MoeMmvqParitySupport.AssertSameTierParity(sameTier, actual, m, n, "Q5_1 MoE MMVQ");

        float[] f32 = CpuIndexedMatmulQ5_1F32(bankQ5_1, x, indices, downScale, m, k, n, numExperts);
        MoeMmvqParitySupport.AssertArgmaxAgreement(f32, actual, m, n, "Q5_1 MoE MMVQ");
    }

    [SkippableTheory]
    [InlineData(5, 7, 12, 32)]
    [InlineData(6, 8, 12, 96)]
    [InlineData(9, 16, 20, 288)]
    public void PerRowExpertIndex_MatchesSingleRowLaunches(int n, int numExperts, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — MoE MMVQ path is unavailable here.");

        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var kernel = MoeIndexedMatmulQ5_1MmvqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("moe_indexed_matmul_q5_1_mmvq.spv missing or unsupported.");

        var rng = new Random(0xC15C1 + n * 31 + numExperts * 17 + m * 11 + k * 7);
        float[] bankF32 = RandomFloats(rng, numExperts * m * k, range: 0.25f);
        byte[] bankQ5_1 = Quantize.FromFloat32(bankF32, (long)numExperts * m * k, QuantizationType.Q5_1);
        float[] x = RandomFloats(rng, n * k, range: 1.0f);
        float[] downScale = new float[numExperts];
        for (int e = 0; e < numExperts; e++) downScale[e] = 0.5f + 0.13f * e;

        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = (i * 3 + 1) % numExperts;

        float[] batched = Launch(device, quant, kernel, bankQ5_1, x, indices, downScale,
            m, k, n, numExperts, out _, out _);

        for (int row = 0; row < n; row++)
        {
            float[] xRow = x.AsSpan(row * k, k).ToArray();
            float[] single = Launch(device, quant, kernel, bankQ5_1, xRow, new[] { indices[row] },
                downScale, m, k, 1, numExperts, out _, out _);

            for (int col = 0; col < m; col++)
                Assert.True(batched[row * m + col].Equals(single[col]),
                    $"Row {row} (expert {indices[row]}), col {col}: batched={batched[row * m + col]:G9} " +
                    $"vs single-row={single[col]:G9} — the per-row expert lookup / scale fold is not row-independent.");
        }
    }

    private static float[] Launch(
        VulkanDevice device, QuantizeQ8_1RowsKernel quant, MoeIndexedMatmulQ5_1MmvqKernel kernel,
        byte[] bankQ5_1, float[] x, int[] indices, float[] downScale,
        int m, int k, int n, int numExperts, out sbyte[] xqBytes, out float[] xdsPairs)
    {
        long bankBufBytes = ((long)bankQ5_1.Length + 3) & ~3L;
        using var bankBuf = device.Allocate(bankBufBytes);
        using var xBuf = device.Allocate((long)x.Length * sizeof(float));
        using var xqBuf = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k));
        using var xdsBuf = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
        using var idxBuf = device.Allocate((long)indices.Length * sizeof(int));
        using var yBuf = device.Allocate((long)n * m * sizeof(float));
        using var scaleBuf = device.Allocate((long)downScale.Length * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(bankQ5_1), bankBuf);
        device.Upload(x, xBuf);
        device.Upload(MemoryMarshal.AsBytes<int>(indices), idxBuf);
        device.Upload(downScale, scaleBuf);

        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, xBuf, xqBuf, xdsBuf, n, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            kernel.Record(ctx.CommandBuffer, bankBuf, xqBuf, xdsBuf, idxBuf, yBuf, scaleBuf, m, k, n, numExperts);
            ctx.SubmitAndWait();
        }

        float[] y = new float[(long)n * m];
        device.Download(yBuf, y);
        xqBytes = MoeMmvqParitySupport.DownloadActivationBytes(device, xqBuf, n, k);
        xdsPairs = MoeMmvqParitySupport.DownloadActivationScales(device, xdsBuf, n, k);
        return y;
    }

    /// <summary>
    /// Scalar C# reimplementation of <c>moe_indexed_matmul_q5_1_mmvq.comp</c>:
    /// per 32-element block <c>d·d_x·Σ(q·xq) + m·s</c> with
    /// <c>q_e = nibble_e | (qh bit e) &lt;&lt; 4</c>, then the per-expert output
    /// scale folded onto the reduced sum — the shader's exact fold order.
    /// </summary>
    private static float[] CpuIndexedMatmulQ5_1Q8_1(
        byte[] bank, sbyte[] xq, float[] xds, int[] indices, float[] downScale,
        int m, int k, int n, int numExperts)
    {
        int blocksPerRow = k / Q5_1GroupSize;
        long rowBytes = (long)blocksPerRow * Q5_1BlockBytes;
        long matrixBytes = m * rowBytes;
        var y = new float[(long)n * m];

        for (int row = 0; row < n; row++)
        {
            int expert = indices[row];
            if ((uint)expert >= (uint)numExperts) continue;
            long expertBase = (long)expert * matrixBytes;

            for (int outIdx = 0; outIdx < m; outIdx++)
            {
                long rowBase = expertBase + outIdx * rowBytes;
                float acc = 0f;
                for (int b = 0; b < blocksPerRow; b++)
                {
                    long block = rowBase + (long)b * Q5_1BlockBytes;
                    float d = MoeMmvqParitySupport.ReadHalf(bank, block);
                    float mn = MoeMmvqParitySupport.ReadHalf(bank, block + 2);
                    uint qh = (uint)(bank[block + 4] | (bank[block + 5] << 8)
                                   | (bank[block + 6] << 16) | (bank[block + 7] << 24));
                    long qs = block + 8;

                    int dot = 0;
                    for (int e = 0; e < Q5_1GroupSize; e++)
                    {
                        int nib = e < 16 ? (bank[qs + e] & 0xF) : (bank[qs + e - 16] >> 4);
                        int q = nib | (int)(((qh >> e) & 1u) << 4);
                        dot += q * xq[(long)row * k + b * Q5_1GroupSize + e];
                    }

                    float dx = xds[(row * blocksPerRow + b) * 2];
                    float s = xds[(row * blocksPerRow + b) * 2 + 1];
                    acc += d * (dx * dot) + mn * s;
                }
                y[(long)row * m + outIdx] = downScale[expert] * acc;
            }
        }
        return y;
    }

    /// <summary>Full-precision (F32-activation) oracle via the production scalar dequant — argmax cross-check only.</summary>
    private static unsafe float[] CpuIndexedMatmulQ5_1F32(
        byte[] bank, float[] x, int[] indices, float[] downScale, int m, int k, int n, int numExperts)
    {
        int blocksPerRow = k / Q5_1GroupSize;
        long rowBytes = (long)blocksPerRow * Q5_1BlockBytes;
        long matrixBytes = m * rowBytes;
        var y = new float[(long)n * m];
        var rowDequant = new float[k];

        fixed (byte* bankPtr = bank)
        {
            for (int row = 0; row < n; row++)
            {
                int expert = Math.Clamp(indices[row], 0, numExperts - 1);
                byte* expertBase = bankPtr + (long)expert * matrixBytes;
                for (int outIdx = 0; outIdx < m; outIdx++)
                {
                    byte* rowBase = expertBase + outIdx * rowBytes;
                    Dequantize.DequantizeQ5_1Scalar((nint)rowBase, k, rowDequant);
                    float sum = 0;
                    for (int i = 0; i < k; i++)
                        sum += rowDequant[i] * x[row * k + i];
                    y[(long)row * m + outIdx] = sum * downScale[expert];
                }
            }
        }
        return y;
    }

    private static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }
}
