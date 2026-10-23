using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Core.Configuration;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan MoE indexed Q5_1 expert-bank matmul,
/// including the per-expert output-scale fold (Gemma-4 ffn_down_exps.scale[e]).
/// </summary>
/// <remarks>
/// <para>
/// Mirrors <see cref="VulkanMoeIndexedMatmulQ6_KF32KernelTests"/>: random FP32
/// weights are quantised to Q5_1 via the production
/// <see cref="Quantize.FromFloat32(ReadOnlySpan{float}, long, QuantizationType)"/>
/// (the same path the GGUF loader produced), then the GPU indexed matmul is
/// compared against a scalar CPU reference reading the SAME bytes via
/// <see cref="Dequantize.DequantizeQ5_1Scalar"/>. This catches shader bugs in
/// the 5th-bit (qh) extraction, the low/high-nibble→element-index mapping, the
/// fp16 d/m reads, the per-row expert lookup, AND the per-expert scale fold.
/// </para>
/// <para>
/// A DISCRIMINATING case uses distinct per-expert scales so a kernel that
/// dropped the scale (or used the wrong expert's scale) fails — a uniform-scale
/// test would not.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMoeIndexedMatmulQ5_1F32KernelTests
{
    private const int Q5_1GroupSize = 32;
    private const int Q5_1BlockBytes = 24;
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(3, 4, 8, 32, 3)]     // smallest: 1 block per row, 3 indexed rows
    [InlineData(2, 4, 16, 32, 2)]    // n=2 rows, both pick the same expert
    [InlineData(4, 4, 16, 64, 3)]    // 2 blocks per row
    [InlineData(8, 8, 32, 96, 4)]    // 3 blocks per row
    [InlineData(5, 16, 48, 32, 8)]   // wider expert bank, indices spanning more experts
    [InlineData(6, 8, 24, 704, 5)]   // real-26B expert FF width (704 = 22 blocks/row)
    [InlineData(8, 16, 2816, 704, 8)] // real-26B down shape: M=hidden 2816, K=Ie 704
    public void Launch_MatchesDequantizedCpuReference(int n, int numExperts, int m, int k, int activeExperts)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x5C51C + n * 31 + numExperts * 17 + m * 11 + k * 7);
        float[] bankF32 = RandomFloats(rng, numExperts * m * k, range: 0.25f);
        float[] x = RandomFloats(rng, n * k, range: 1.0f);
        int[] indices = RandomIndices(rng, n, numExperts, activeExperts);

        // Distinct per-expert scales — discriminates a dropped / mis-indexed scale.
        float[] downScale = new float[numExperts];
        for (int e = 0; e < numExperts; e++)
            downScale[e] = 0.5f + 0.13f * e;

        // Quantise the whole bank as one (numExperts*m, k) flat blob — Q5_1 rows
        // are contiguous so a single call yields the [numExperts, m, rowBytes] layout.
        byte[] bankQ5_1 = Quantize.FromFloat32(bankF32, (long)numExperts * m * k, QuantizationType.Q5_1);

        float[] expected = CpuIndexedMatmulQ5_1(bankQ5_1, x, indices, downScale, m, k, n, numExperts);

        using var device = VulkanDevice.Create();
        using var kernel = MoeIndexedMatmulQ5_1F32Kernel.Create(device, spvDir);

        long bankBufBytes = ((long)bankQ5_1.Length + 3) & ~3L;
        using var bankBuf = device.Allocate(bankBufBytes);
        using var xBuf = device.Allocate((long)x.Length * sizeof(float));
        using var idxBuf = device.Allocate((long)indices.Length * sizeof(int));
        using var yBuf = device.Allocate((long)expected.Length * sizeof(float));
        using var scaleBuf = device.Allocate((long)downScale.Length * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(bankQ5_1), bankBuf);
        device.Upload(x, xBuf);
        device.Upload(MemoryMarshal.AsBytes<int>(indices), idxBuf);
        device.Upload(downScale, scaleBuf);

        kernel.Launch(bankBuf, xBuf, idxBuf, yBuf, scaleBuf, m, k, n, numExperts);

        float[] actual = new float[expected.Length];
        device.Download(yBuf, actual);

        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float bar = AbsTol + RelTol * MathF.Abs(expected[i]);
            Assert.True(diff <= bar,
                $"row={i / m}, col={i % m}: cpu={expected[i]:F6} vs vulkan={actual[i]:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    /// <summary>
    /// CPU reference: per-row Q5_1 dequant (via the production scalar oracle) +
    /// dot-product against x, then a per-expert output scale — exactly the
    /// kernel's fold order.
    /// </summary>
    private static unsafe float[] CpuIndexedMatmulQ5_1(
        byte[] bankQ5_1, float[] x, int[] indices, float[] downScale,
        int m, int k, int n, int numExperts)
    {
        int blocksPerRow = k / Q5_1GroupSize;
        int rowBytes = blocksPerRow * Q5_1BlockBytes;
        int matrixBytes = m * rowBytes;
        var y = new float[n * m];

        var rowDequant = new float[k];

        fixed (byte* bankPtr = bankQ5_1)
        {
            for (int row = 0; row < n; row++)
            {
                int idx = Math.Clamp(indices[row], 0, numExperts - 1);
                byte* expertBase = bankPtr + (long)idx * matrixBytes;
                int xRowBase = row * k;
                int yRowBase = row * m;
                float scale = downScale[idx];

                for (int outIdx = 0; outIdx < m; outIdx++)
                {
                    byte* rowBase = expertBase + (long)outIdx * rowBytes;
                    Dequantize.DequantizeQ5_1Scalar((nint)rowBase, k, rowDequant);
                    float sum = 0;
                    for (int i = 0; i < k; i++)
                        sum += rowDequant[i] * x[xRowBase + i];
                    y[yRowBase + outIdx] = sum * scale;
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
