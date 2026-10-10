using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity tests for the dp4a MoE indexed IQ MMVQ decode GEMV (<c>moe_indexed_matmul_iq{3_s,4_xs,4_nl}_mmvq</c>, issue #823).
/// </summary>
/// <remarks>
/// The bank is RANDOM BLOCK BYTES (valid for every IQ encoding: any code/sign/scale byte decodes to something), with only the fp16 super-scale
/// forced sane. That is a stricter layout oracle than a fitted quantizer round-trip: every byte of a block feeds the result, so a kernel
/// that mis-reads a qh/sign/scale field changes the answer. The oracle is the production CPU dequant (<see cref="Dequantize.ToFloat32"/>)
/// dotted in double with the SAME Q8_1-quantized activations the kernel read back (the integer dot is exact, so only fp32 accumulation
/// order differs). Experts get DISTINCT bytes and indices mix, so a wrong expert base offset or a broadcast-style index bug fails.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMoeIndexedMatmulIqMmvqKernelTests
{
    [SkippableTheory]
    [InlineData(QuantizationType.IQ3_S, 5, 7, 12, 256, 4)]      // 1 super-block/row, n != m != E
    [InlineData(QuantizationType.IQ3_S, 6, 8, 12, 768, 5)]      // 3 super-blocks
    [InlineData(QuantizationType.IQ3_S, 10, 512, 48, 2560, 10)] // qwen4exp gate/up: 512 experts, top-10, K=hidden 2560 (10 super-blocks)
    [InlineData(QuantizationType.IQ4_XS, 5, 7, 12, 256, 4)]
    [InlineData(QuantizationType.IQ4_XS, 6, 8, 12, 768, 5)]
    [InlineData(QuantizationType.IQ4_XS, 10, 512, 48, 2560, 10)]
    [InlineData(QuantizationType.IQ4_NL, 5, 7, 12, 32, 4)]      // 1 block/row
    [InlineData(QuantizationType.IQ4_NL, 6, 8, 12, 96, 5)]      // 3 blocks/row - window-tail path
    [InlineData(QuantizationType.IQ4_NL, 3, 4, 9, 256, 3)]      // exactly one 8-block lane window
    [InlineData(QuantizationType.IQ4_NL, 9, 16, 20, 288, 11)]   // 9 blocks/row - window + 1-block tail
    [InlineData(QuantizationType.IQ4_NL, 10, 512, 48, 640, 10)] // qwen4exp down: K=intermediate 640 (not a multiple of 256), 20 blocks/row
    [InlineData(QuantizationType.IQ2_XXS, 5, 7, 12, 256, 4)]
    [InlineData(QuantizationType.IQ2_XXS, 10, 512, 48, 2560, 10)]
    [InlineData(QuantizationType.IQ2_XS, 6, 8, 12, 768, 5)]
    [InlineData(QuantizationType.IQ2_XS, 10, 512, 48, 2560, 10)]
    [InlineData(QuantizationType.IQ2_S, 6, 8, 12, 768, 5)]
    [InlineData(QuantizationType.IQ2_S, 10, 512, 48, 2560, 10)]
    [InlineData(QuantizationType.IQ3_XXS, 6, 8, 12, 768, 5)]
    [InlineData(QuantizationType.IQ3_XXS, 10, 512, 48, 2560, 10)]
    [InlineData(QuantizationType.Q2_0, 5, 7, 12, 64, 4)]       // 1 block/row
    [InlineData(QuantizationType.Q2_0, 6, 8, 12, 192, 5)]      // 3 blocks/row - window tail
    [InlineData(QuantizationType.Q2_0, 3, 4, 9, 256, 3)]       // exactly one 4-block window
    [InlineData(QuantizationType.Q2_0, 10, 512, 48, 640, 10)]  // ISTA down: 10 blocks/row
    public void Launch_MatchesSameTierCpuOracle(QuantizationType qt, int n, int numExperts, int m, int k, int activeExperts)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "Device does not advertise VK_KHR_shader_integer_dot_product.");

        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var codebooks = Iq3Codebooks.Create(device);
        using var codebooks2 = Iq2Codebooks.Create(device);
        var iq = MoeIndexedMatmulIqMmvqKernel.FromQuantizationType(qt)!.Value;
        using var kernel = MoeIndexedMatmulIqMmvqKernel.TryCreate(device, spvDir, iq, codebooks, codebooks2)
            ?? throw new Xunit.Sdk.XunitException(MoeIndexedMatmulIqMmvqKernel.ShaderName(iq) + ".spv missing or unsupported.");

        var rng = new Random(0x1B5A + (int)qt * 1009 + n * 31 + numExperts * 17 + m * 11 + k * 7);
        var (blockBytes, group) = MoeIndexedMatmulIqMmvqKernel.Describe(iq);
        long rowBytes = (long)(k / group) * blockBytes;
        byte[] bank = RandomBank(rng, (long)numExperts * m, (int)rowBytes, blockBytes);
        float[] x = new float[n * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 1.0);
        int[] indices = MoeMmvqParitySupport.RandomIndices(rng, n, numExperts, activeExperts);

        float[] actual = Launch(device, quant, kernel, bank, x, indices, m, k, n, numExperts, out sbyte[] xq, out float[] xds);
        float[] expected = CpuOracle(bank, qt, xq, xds, indices, m, k, n, numExperts, rowBytes);
        MoeMmvqParitySupport.AssertSameTierParity(expected, actual, m, n, qt + " MoE MMVQ");
    }

    [SkippableTheory]
    [InlineData(QuantizationType.IQ3_S, 6, 8, 12, 768)]
    [InlineData(QuantizationType.IQ4_XS, 6, 8, 12, 768)]
    [InlineData(QuantizationType.IQ4_NL, 9, 16, 20, 288)]
    [InlineData(QuantizationType.IQ2_XXS, 6, 8, 12, 768)]
    [InlineData(QuantizationType.IQ2_XS, 6, 8, 12, 768)]
    [InlineData(QuantizationType.IQ2_S, 6, 8, 12, 768)]
    [InlineData(QuantizationType.IQ3_XXS, 6, 8, 12, 768)]
    [InlineData(QuantizationType.Q2_0, 6, 8, 12, 192)]
    public void PerRowExpertIndex_MatchesSingleRowLaunches(QuantizationType qt, int n, int numExperts, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "Device does not advertise VK_KHR_shader_integer_dot_product.");

        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var codebooks = Iq3Codebooks.Create(device);
        using var codebooks2 = Iq2Codebooks.Create(device);
        var iq = MoeIndexedMatmulIqMmvqKernel.FromQuantizationType(qt)!.Value;
        using var kernel = MoeIndexedMatmulIqMmvqKernel.TryCreate(device, spvDir, iq, codebooks, codebooks2)
            ?? throw new Xunit.Sdk.XunitException("spv missing.");

        var rng = new Random(0xC15C1 + (int)qt + n * 31 + numExperts * 17 + m * 11 + k * 7);
        var (blockBytes, group) = MoeIndexedMatmulIqMmvqKernel.Describe(iq);
        byte[] bank = RandomBank(rng, (long)numExperts * m, (k / group) * blockBytes, blockBytes);
        float[] x = new float[n * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = (i * 3 + 1) % numExperts;

        float[] batched = Launch(device, quant, kernel, bank, x, indices, m, k, n, numExperts, out _, out _);
        for (int row = 0; row < n; row++)
        {
            float[] single = Launch(device, quant, kernel, bank, x.AsSpan(row * k, k).ToArray(), new[] { indices[row] },
                m, k, 1, numExperts, out _, out _);
            for (int col = 0; col < m; col++)
                Assert.True(batched[row * m + col].Equals(single[col]),
                    $"Row {row} (expert {indices[row]}), col {col}: batched={batched[row * m + col]:G9} vs single-row={single[col]:G9}.");
        }
    }

    /// <summary>Random block bytes with the leading fp16 of every block forced to a sane positive scale.</summary>
    private static byte[] RandomBank(Random rng, long rows, int rowBytes, int blockBytes)
    {
        long total = rows * rowBytes;
        var bytes = new byte[total];
        rng.NextBytes(bytes);
        for (long b = 0; b + blockBytes <= total; b += blockBytes)
        {
            ushort d = BitConverter.HalfToUInt16Bits((Half)(0.002f + (float)rng.NextDouble() * 0.02f));
            bytes[b] = (byte)(d & 0xFF);
            bytes[b + 1] = (byte)(d >> 8);
        }
        return bytes;
    }

    private static float[] Launch(
        VulkanDevice device, QuantizeQ8_1RowsKernel quant, MoeIndexedMatmulIqMmvqKernel kernel,
        byte[] bank, float[] x, int[] indices, int m, int k, int n, int numExperts,
        out sbyte[] xqBytes, out float[] xdsPairs)
    {
        long bankBufBytes = ((long)bank.Length + 3) & ~3L;
        using var bankBuf = device.Allocate(bankBufBytes);
        using var xBuf = device.Allocate((long)x.Length * sizeof(float));
        using var xqBuf = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k));
        using var xdsBuf = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
        using var idxBuf = device.Allocate((long)indices.Length * sizeof(int));
        using var yBuf = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(bank), bankBuf);
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

    /// <summary>CPU dequant of the expert row dotted (double) with the kernel's own Q8_1 activations (xq * d_x).</summary>
    private static unsafe float[] CpuOracle(
        byte[] bank, QuantizationType qt, sbyte[] xq, float[] xds, int[] indices,
        int m, int k, int n, int numExperts, long rowBytes)
    {
        var y = new float[(long)n * m];
        var w = new float[k];
        int blocksPerRow = k / 32;
        fixed (byte* bankPtr = bank)
        {
            for (int row = 0; row < n; row++)
            {
                int expert = indices[row];
                if ((uint)expert >= (uint)numExperts) continue;
                for (int outIdx = 0; outIdx < m; outIdx++)
                {
                    byte* rowPtr = bankPtr + ((long)expert * m + outIdx) * rowBytes;
                    Dequantize.ToFloat32((nint)rowPtr, k, qt, w);
                    double acc = 0;
                    for (int i = 0; i < k; i++)
                        acc += (double)w[i] * xq[(long)row * k + i] * xds[(row * blocksPerRow + i / 32) * 2];
                    y[(long)row * m + outIdx] = (float)acc;
                }
            }
        }
        return y;
    }
}
