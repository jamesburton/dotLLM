using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity tests for <c>moe_indexed_matmul_q8_0_mmvq</c> — the dp4a MoE indexed
/// Q8_0 decode GEMV (issue #137/#383). Closes the coverage hole reported in
/// issue #309.
/// </summary>
/// <remarks>
/// Same two-assertion design as
/// <see cref="VulkanMoeIndexedMatmulQ4KMmvqKernelTests"/>: a tight SAME-TIER
/// oracle (fed the exact int8 activations the shader read, so only fp32
/// accumulation order can differ) plus a bit-exact per-row expert-indexing
/// equivalence against single-row launches. The Q8_0 shader's 34-byte block
/// stride forces awkward uint16 addressing, which is exactly the kind of
/// arithmetic the byte-identical oracle pins down.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMoeIndexedMatmulQ8_0MmvqKernelTests
{
    private const int Q8_0GroupSize = 32;
    private const int Q8_0BlockBytes = 34;

    [SkippableTheory]
    [InlineData(5, 7, 12, 32, 4)]     // 1 block/row; n≠m≠E, 4 distinct experts
    [InlineData(6, 8, 12, 96, 5)]     // 3 blocks/row (odd) — window-tail path
    [InlineData(3, 4, 9, 256, 3)]     // 8 blocks/row = exactly one lane window
    [InlineData(9, 16, 20, 288, 11)]  // 9 blocks/row — window + 1-block tail
    [InlineData(8, 16, 704, 2816, 8)] // real 26B gate/up shape: Ie=704, K=2816
    public void Launch_MatchesSameTierCpuOracle(int n, int numExperts, int m, int k, int activeExperts)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — MoE MMVQ path is unavailable here.");

        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var kernel = MoeIndexedMatmulQ8_0MmvqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("moe_indexed_matmul_q8_0_mmvq.spv missing or unsupported.");

        var rng = new Random(0x8C08C + n * 31 + numExperts * 17 + m * 11 + k * 7);
        float[] bankF32 = RandomFloats(rng, numExperts * m * k, range: 0.1f);
        float[] x = RandomFloats(rng, n * k, range: 1.0f);
        int[] indices = MoeMmvqParitySupport.RandomIndices(rng, n, numExperts, activeExperts);

        byte[] bankQ8 = QuantizeRows(bankF32, numExperts * m, k);

        float[] actual = Launch(device, quant, kernel, bankQ8, x, indices, m, k, n, numExperts,
            out sbyte[] xq, out float[] xds);

        float[] sameTier = CpuIndexedMatmulQ8_0Q8_1(bankQ8, xq, xds, indices, m, k, n, numExperts);
        MoeMmvqParitySupport.AssertSameTierParity(sameTier, actual, m, n, "Q8_0 MoE MMVQ");

        float[] f32 = CpuIndexedMatmulQ8_0F32(bankQ8, x, indices, m, k, n, numExperts);
        MoeMmvqParitySupport.AssertArgmaxAgreement(f32, actual, m, n, "Q8_0 MoE MMVQ");
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
        using var kernel = MoeIndexedMatmulQ8_0MmvqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("moe_indexed_matmul_q8_0_mmvq.spv missing or unsupported.");

        var rng = new Random(0xC08C0 + n * 31 + numExperts * 17 + m * 11 + k * 7);
        byte[] bankQ8 = QuantizeRows(RandomFloats(rng, numExperts * m * k, range: 0.1f), numExperts * m, k);
        float[] x = RandomFloats(rng, n * k, range: 1.0f);

        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = (i * 3 + 1) % numExperts;

        float[] batched = Launch(device, quant, kernel, bankQ8, x, indices, m, k, n, numExperts, out _, out _);

        for (int row = 0; row < n; row++)
        {
            float[] xRow = x.AsSpan(row * k, k).ToArray();
            float[] single = Launch(device, quant, kernel, bankQ8, xRow,
                new[] { indices[row] }, m, k, 1, numExperts, out _, out _);

            for (int col = 0; col < m; col++)
                Assert.True(batched[row * m + col].Equals(single[col]),
                    $"Row {row} (expert {indices[row]}), col {col}: batched={batched[row * m + col]:G9} " +
                    $"vs single-row={single[col]:G9} — the per-row expert lookup is not row-independent.");
        }
    }

    private static float[] Launch(
        VulkanDevice device, QuantizeQ8_1RowsKernel quant, MoeIndexedMatmulQ8_0MmvqKernel kernel,
        byte[] bankQ8, float[] x, int[] indices, int m, int k, int n, int numExperts,
        out sbyte[] xqBytes, out float[] xdsPairs)
    {
        long bankBufBytes = ((long)bankQ8.Length + 3) & ~3L;
        using var bankBuf = device.Allocate(bankBufBytes);
        using var xBuf = device.Allocate((long)x.Length * sizeof(float));
        using var xqBuf = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k));
        using var xdsBuf = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
        using var idxBuf = device.Allocate((long)indices.Length * sizeof(int));
        using var yBuf = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(bankQ8), bankBuf);
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
        xqBytes = MoeMmvqParitySupport.DownloadActivationBytes(device, xqBuf, n, k);
        xdsPairs = MoeMmvqParitySupport.DownloadActivationScales(device, xdsBuf, n, k);
        return y;
    }

    /// <summary>
    /// Scalar C# reimplementation of <c>moe_indexed_matmul_q8_0_mmvq.comp</c>:
    /// <c>Σ_blocks d_w · d_x · Σ(qs · xq)</c>, weight row selected by the per-row
    /// expert index. Q8_0 is symmetric, so the Q8_1 block sum is unused — a
    /// kernel that erroneously added a min/sum term fails here.
    /// </summary>
    private static float[] CpuIndexedMatmulQ8_0Q8_1(
        byte[] bankQ8, sbyte[] xq, float[] xds, int[] indices, int m, int k, int n, int numExperts)
    {
        int blocksPerRow = k / Q8_0GroupSize;
        long rowBytes = (long)blocksPerRow * Q8_0BlockBytes;
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
                    long block = rowBase + (long)b * Q8_0BlockBytes;
                    float dW = MoeMmvqParitySupport.ReadHalf(bankQ8, block);
                    float dX = xds[(row * blocksPerRow + b) * 2];
                    int dot = 0;
                    for (int j = 0; j < Q8_0GroupSize; j++)
                        dot += unchecked((sbyte)bankQ8[block + 2 + j]) * xq[(long)row * k + b * Q8_0GroupSize + j];
                    acc += (dW * dX) * dot;
                }
                y[(long)row * m + outIdx] = acc;
            }
        }
        return y;
    }

    /// <summary>Full-precision (F32-activation) oracle — used for the argmax cross-check only.</summary>
    private static unsafe float[] CpuIndexedMatmulQ8_0F32(
        byte[] bankQ8, float[] x, int[] indices, int m, int k, int n, int numExperts)
    {
        int blocksPerRow = k / Q8_0GroupSize;
        long rowBytes = (long)blocksPerRow * Q8_0BlockBytes;
        long matrixBytes = m * rowBytes;
        var y = new float[(long)n * m];

        fixed (byte* bankPtr = bankQ8)
        {
            for (int row = 0; row < n; row++)
            {
                int expert = Math.Clamp(indices[row], 0, numExperts - 1);
                byte* expertBase = bankPtr + (long)expert * matrixBytes;
                for (int outIdx = 0; outIdx < m; outIdx++)
                {
                    byte* rowBase = expertBase + outIdx * rowBytes;
                    float sum = 0;
                    for (int b = 0; b < blocksPerRow; b++)
                    {
                        byte* block = rowBase + b * Q8_0BlockBytes;
                        float d = (float)Unsafe.ReadUnaligned<Half>(block);
                        sbyte* qs = (sbyte*)(block + 2);
                        float blockSum = 0;
                        for (int j = 0; j < Q8_0GroupSize; j++)
                            blockSum += qs[j] * x[row * k + b * Q8_0GroupSize + j];
                        sum += d * blockSum;
                    }
                    y[(long)row * m + outIdx] = sum;
                }
            }
        }
        return y;
    }

    private static unsafe byte[] QuantizeRows(float[] src, int rows, int k)
    {
        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;
        var dst = new byte[(long)rows * rowBytes];
        fixed (float* srcPtr = src)
        fixed (byte* dstPtr = dst)
        {
            for (int row = 0; row < rows; row++)
                MatMul.QuantizeF32ToQ8_0(srcPtr + (long)row * k, dstPtr + (long)row * rowBytes, k);
        }
        return dst;
    }

    private static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }
}
