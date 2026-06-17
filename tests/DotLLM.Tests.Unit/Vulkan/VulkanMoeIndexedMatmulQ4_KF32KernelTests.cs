using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Core.Configuration;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan MoE indexed Q4_K expert-bank matmul
/// (Gemma-4 fused gate_up experts, kept quantized on device).
/// </summary>
/// <remarks>
/// Mirrors <see cref="VulkanMoeIndexedMatmulQ6_KF32KernelTests"/>: generate
/// random FP32 weights, quantise to Q4_K via <see cref="Q4KFixture.QuantizeRows"/>
/// (round-trip-verified at the top of each case), then compare the GPU indexed
/// matmul against a scalar CPU reference reading the SAME bytes. Comparing
/// against a Q4_K-byte-identical reference catches shader bugs in the nibble
/// extraction, the 6-bit (scale, min) unpack, the fp16 d/dmin reads, and the
/// per-row expert lookup — bugs a quantise-then-compare-to-FP32 reference masks.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMoeIndexedMatmulQ4_KF32KernelTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(3, 4, 8, 256, 3)]    // smallest: 1 super-block per row, 3 indexed rows
    [InlineData(2, 4, 16, 256, 2)]   // n=2 rows, both pick the same expert
    [InlineData(4, 4, 16, 512, 3)]   // 2 super-blocks per row
    [InlineData(8, 8, 32, 768, 4)]   // 3 super-blocks per row
    [InlineData(5, 16, 48, 256, 8)]  // wider expert bank, indices spanning more experts
    public void Launch_MatchesDequantizedCpuReference(int n, int numExperts, int m, int k, int activeExperts)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x4CB4C + n * 31 + numExperts * 17 + m * 11 + k * 7);
        float[] bankF32 = Q4KFixture.RandomFloats(rng, numExperts * m * k, range: 0.1f);
        float[] x = Q4KFixture.RandomFloats(rng, n * k, range: 1.0f);
        int[] indices = RandomIndices(rng, n, numExperts, activeExperts);

        byte[] bankQ4K = Q4KFixture.QuantizeRows(bankF32, numExperts * m, k);
        Q4KFixture.AssertFixtureRoundtrip(bankF32, bankQ4K, numExperts * m, k);

        float[] expected = CpuIndexedMatmulQ4K(bankQ4K, x, indices, m, k, n, numExperts);

        using var device = VulkanDevice.Create();
        using var kernel = MoeIndexedMatmulQ4_KF32Kernel.Create(device, spvDir);

        long bankBufBytes = ((long)bankQ4K.Length + 3) & ~3L;
        using var bankBuf = device.Allocate(bankBufBytes);
        using var xBuf = device.Allocate((long)x.Length * sizeof(float));
        using var idxBuf = device.Allocate((long)indices.Length * sizeof(int));
        using var yBuf = device.Allocate((long)expected.Length * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(bankQ4K), bankBuf);
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

    /// <summary>
    /// CPU reference: per-row Q4_K dequant + dot-product against x. Reads the
    /// same bytes the GPU shader sees from the bank slab of the routed expert.
    /// </summary>
    private static unsafe float[] CpuIndexedMatmulQ4K(
        byte[] bankQ4K, float[] x, int[] indices, int m, int k, int n, int numExperts)
    {
        int blocksPerRow = k / Q4KFixture.Q4KGroupSize;
        int rowBytes = blocksPerRow * Q4KFixture.Q4KBlockBytes;
        int matrixBytes = m * rowBytes;
        var y = new float[n * m];

        Span<byte> scBuf = stackalloc byte[8];
        Span<byte> mnBuf = stackalloc byte[8];

        fixed (byte* bankPtr = bankQ4K)
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
                        byte* block = rowBase + b * Q4KFixture.Q4KBlockBytes;
                        float d = (float)Unsafe.ReadUnaligned<Half>(block);
                        float dmin = (float)Unsafe.ReadUnaligned<Half>(block + 2);
                        fixed (byte* sc = scBuf)
                        fixed (byte* mn = mnBuf)
                        {
                            Dequantize.UnpackQ4Q5Scales(block + 4, sc, mn);
                        }
                        byte* qs = block + 16;
                        int xBase = xRowBase + b * Q4KFixture.Q4KGroupSize;

                        for (int j = 0; j < 8; j++)
                        {
                            float scF = d * scBuf[j];
                            float mnF = dmin * mnBuf[j];
                            int pairIdx = j / 2;
                            int nibbleHalf = j % 2;
                            int outBase = xBase + j * Q4KFixture.SubBlockSize;
                            for (int i = 0; i < Q4KFixture.SubBlockSize; i++)
                            {
                                int qsByte = pairIdx * Q4KFixture.SubBlockSize + i;
                                int nib = nibbleHalf == 0 ? (qs[qsByte] & 0xF) : (qs[qsByte] >> 4);
                                float w = scF * nib - mnF;
                                sum += w * x[outBase + i];
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
