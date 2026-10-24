using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan MoE indexed Q6_K expert-bank matmul.
/// </summary>
/// <remarks>
/// <para>
/// Validation strategy mirrors <see cref="VulkanMatMulQ6KGemvF32KernelTests"/>:
/// generate random FP32 weights, quantise them to Q6_K via <see cref="Q6KFixture.QuantizeRows"/>
/// (verified against the CPU oracle <c>DequantizeQ6_KScalar</c> with a
/// round-trip assertion at the start of each test), then compare the GPU
/// indexed matmul against a scalar CPU reference that reads the same bytes.
/// Comparing against a Q6_K-byte-identical reference (rather than against the
/// pre-quantisation FP32 weights) catches shader bugs in the (ql, qh)
/// bit-extraction, the int8 scale sign-extension, the fp16 d straddle read,
/// and the per-row expert lookup — bugs that a quantise-then-compare-to-FP32
/// reference would mask.
/// </para>
/// <para>
/// Tolerance: absolute 5e-3 / relative 1e-3 — same as the dense Q6_K GEMV
/// parity bar. The dominant drift is reduction-order; per-row expert lookup
/// adds no rounding.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMoeIndexedMatmulQ6_KF32KernelTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(3, 4, 8, 256, 3)]      // smallest: 1 super-block per row, 3 indexed rows
    [InlineData(2, 4, 16, 256, 2)]     // n=2 rows, both pick the same expert
    [InlineData(4, 4, 16, 512, 3)]     // 2 super-blocks per row (row stride 420 — 4-aligned)
    [InlineData(8, 8, 32, 768, 4)]     // 3 super-blocks per row, row stride 630 — NOT 4-aligned
    [InlineData(16, 8, 64, 1024, 4)]   // 4 super-blocks per row (row stride 840 — 4-aligned)
    [InlineData(5, 16, 48, 256, 8)]    // wider expert bank, indices spanning more experts
    public void Launch_MatchesDequantizedCpuReference(int n, int numExperts, int m, int k, int activeExperts)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x6CB6C + n * 31 + numExperts * 17 + m * 11 + k * 7);
        // Use Q6KFixture's range=0.1 to keep amax small and well-conditioned —
        // matches the dense Q6_K test's choice and makes the round-trip drift
        // dominate over numerical noise from random outliers.
        float[] bankF32 = Q6KFixture.RandomFloats(rng, numExperts * m * k, range: 0.1f);
        float[] x = Q6KFixture.RandomFloats(rng, n * k, range: 1.0f);
        int[] indices = RandomIndices(rng, n, numExperts, activeExperts);

        // Quantise the whole bank as one (numExperts*m, k) matrix — the byte
        // layout is per-row, so a flat call produces the exact contiguous
        // [numExperts, m, rowBytes] blob the shader expects.
        byte[] bankQ6K = Q6KFixture.QuantizeRows(bankF32, numExperts * m, k);

        // Sanity-check the fixture quantiser against the CPU oracle on the
        // full bank — same structural safeguard as VulkanMatMulQ6KGemvF32KernelTests.
        Q6KFixture.AssertFixtureRoundtrip(bankF32, bankQ6K, numExperts * m, k);

        float[] expected = CpuIndexedMatmulQ6K(bankQ6K, x, indices, m, k, n, numExperts);

        using var device = VulkanDevice.Create();
        using var kernel = MoeIndexedMatmulQ6_KF32Kernel.Create(device, spvDir);

        long bankBufBytes = ((long)bankQ6K.Length + 3) & ~3L;
        using var bankBuf = device.Allocate(bankBufBytes);
        using var xBuf = device.Allocate((long)x.Length * sizeof(float));
        using var idxBuf = device.Allocate((long)indices.Length * sizeof(int));
        using var yBuf = device.Allocate((long)expected.Length * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(bankQ6K), bankBuf);
        device.Upload(x, xBuf);
        device.Upload(MemoryMarshal.AsBytes<int>(indices), idxBuf);

        kernel.Launch(bankBuf, xBuf, idxBuf, yBuf, m, k, n, numExperts);

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

    [SkippableFact]
    public void Launch_AllRowsSameExpert_MatchesDenseQ6KGemv()
    {
        // Sanity check: when every row picks expert 0, the kernel must
        // produce the same output as a plain Q6_K GEMV against bank[0] for
        // each row of x — this isolates the per-row expert lookup from the
        // dequant arithmetic.
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int n = 4, m = 32, k = 256, numExperts = 4;
        var rng = new Random(unchecked((int)0xC0FFEE6Cu));

        float[] bankF32 = Q6KFixture.RandomFloats(rng, numExperts * m * k, range: 0.1f);
        float[] x = Q6KFixture.RandomFloats(rng, n * k, range: 1.0f);
        int[] indices = new int[n]; // all zeros → all rows pick expert 0
        byte[] bankQ6K = Q6KFixture.QuantizeRows(bankF32, numExperts * m, k);

        // Reference: per-row GEMV against bank[0] (the first m*k bytes of bankQ6K
        // when we slice off expert 0's slab).
        int rowBytes = (k / Q6KFixture.Q6KGroupSize) * Q6KFixture.Q6KBlockBytes;
        byte[] expert0 = bankQ6K.AsSpan(0, m * rowBytes).ToArray();
        float[] expected = new float[n * m];
        for (int row = 0; row < n; row++)
        {
            float[] xRow = x.AsSpan(row * k, k).ToArray();
            float[] yRow = Q6KFixture.CpuGemvQ6K(expert0, xRow, m, k);
            yRow.CopyTo(expected.AsSpan(row * m));
        }

        using var device = VulkanDevice.Create();
        using var kernel = MoeIndexedMatmulQ6_KF32Kernel.Create(device, spvDir);

        long bankBufBytes = ((long)bankQ6K.Length + 3) & ~3L;
        using var bankBuf = device.Allocate(bankBufBytes);
        using var xBuf = device.Allocate((long)x.Length * sizeof(float));
        using var idxBuf = device.Allocate((long)indices.Length * sizeof(int));
        using var yBuf = device.Allocate((long)expected.Length * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(bankQ6K), bankBuf);
        device.Upload(x, xBuf);
        device.Upload(MemoryMarshal.AsBytes<int>(indices), idxBuf);

        kernel.Launch(bankBuf, xBuf, idxBuf, yBuf, m, k, n, numExperts);

        float[] actual = new float[expected.Length];
        device.Download(yBuf, actual);

        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float bar = AbsTol + RelTol * MathF.Abs(expected[i]);
            Assert.True(diff <= bar,
                $"i={i}: gemv-expert0={expected[i]:F6} vs vulkan-indexed={actual[i]:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    /// <summary>
    /// CPU reference: per-row Q6_K dequant + dot-product against x. Reads the
    /// same bytes the GPU shader sees from the bank slab of the routed expert.
    /// </summary>
    private static unsafe float[] CpuIndexedMatmulQ6K(
        byte[] bankQ6K, float[] x, int[] indices, int m, int k, int n, int numExperts)
    {
        if ((k % Q6KFixture.Q6KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q6KFixture.Q6KGroupSize}", nameof(k));

        int blocksPerRow = k / Q6KFixture.Q6KGroupSize;
        int rowBytes = blocksPerRow * Q6KFixture.Q6KBlockBytes;
        int matrixBytes = m * rowBytes;
        var y = new float[n * m];

        fixed (byte* bankPtr = bankQ6K)
        {
            for (int row = 0; row < n; row++)
            {
                int idx = Math.Clamp(indices[row], 0, numExperts - 1);
                byte* expertBase = bankPtr + (long)idx * matrixBytes;
                int xRowBase = row * k;
                int yRowBase = row * m;

                for (int outIdx = 0; outIdx < m; outIdx++)
                {
                    byte* rowBase = expertBase + (long)outIdx * rowBytes;
                    float sum = 0;
                    for (int b = 0; b < blocksPerRow; b++)
                    {
                        byte* block = rowBase + b * Q6KFixture.Q6KBlockBytes;
                        byte* ql = block;
                        byte* qh = block + 128;
                        sbyte* scales = (sbyte*)(block + 192);
                        float d = (float)System.Runtime.CompilerServices.Unsafe.ReadUnaligned<Half>(block + 208);
                        int xBase = xRowBase + b * Q6KFixture.Q6KGroupSize;

                        // Mirrors DequantizeQ6_KScalar exactly.
                        for (int hf = 0; hf < 2; hf++)
                        {
                            int qlOff = hf * 64;
                            int qhOff = hf * 32;
                            int scOff = hf * 8;
                            int outHalfBase = hf * 128;
                            for (int l = 0; l < 32; l++)
                            {
                                int isc = l / 16;
                                int q1 = ((ql[qlOff + l]      & 0xF) | (((qh[qhOff + l] >> 0) & 3) << 4)) - 32;
                                int q2 = ((ql[qlOff + l + 32] & 0xF) | (((qh[qhOff + l] >> 2) & 3) << 4)) - 32;
                                int q3 = ((ql[qlOff + l]      >> 4) | (((qh[qhOff + l] >> 4) & 3) << 4)) - 32;
                                int q4 = ((ql[qlOff + l + 32] >> 4) | (((qh[qhOff + l] >> 6) & 3) << 4)) - 32;

                                float w1 = d * scales[scOff + isc]     * q1;
                                float w2 = d * scales[scOff + isc + 2] * q2;
                                float w3 = d * scales[scOff + isc + 4] * q3;
                                float w4 = d * scales[scOff + isc + 6] * q4;

                                sum += w1 * x[xBase + outHalfBase + l]
                                     + w2 * x[xBase + outHalfBase + l + 32]
                                     + w3 * x[xBase + outHalfBase + l + 64]
                                     + w4 * x[xBase + outHalfBase + l + 96];
                            }
                        }
                    }
                    y[yRowBase + outIdx] = sum;
                }
            }
        }
        return y;
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
