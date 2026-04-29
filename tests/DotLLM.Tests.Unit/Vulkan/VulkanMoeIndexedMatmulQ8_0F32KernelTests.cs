using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>Numerical parity for the Q8_0 MoE indexed expert matmul kernel.</summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMoeIndexedMatmulQ8_0F32KernelTests
{
    [SkippableTheory]
    [InlineData(3, 4, 8, 32, 3)]
    [InlineData(40, 4, 32, 32, 4)]
    [InlineData(5, 3, 16, 64, 2)]
    public void Launch_MatchesDequantizedCpuReference(int n, int numExperts, int m, int k, int activeExperts)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x456789 + n * 31 + numExperts * 17 + m * 11 + k * 7);
        float[] bank = RandomFloats(rng, numExperts * m * k);
        float[] x = RandomFloats(rng, n * k);
        int[] indices = RandomIndices(rng, n, numExperts, activeExperts);

        byte[] bankQ8 = QuantizeBankToQ8(bank, numExperts, m, k, out float[] dequantBank);
        float[] expected = CpuIndexedMatmul(dequantBank, x, indices, m, k, n, numExperts);

        using var device = VulkanDevice.Create();
        using var kernel = MoeIndexedMatmulQ8_0F32Kernel.Create(device, spvDir);
        using var bankBuf = device.Allocate(bankQ8.Length);
        using var xBuf = device.Allocate((long)x.Length * sizeof(float));
        using var idxBuf = device.Allocate((long)indices.Length * sizeof(int));
        using var yBuf = device.Allocate((long)expected.Length * sizeof(float));

        device.Upload(bankQ8, bankBuf);
        device.Upload(x, xBuf);
        device.Upload(MemoryMarshal.AsBytes<int>(indices), idxBuf);

        kernel.Launch(bankBuf, xBuf, idxBuf, yBuf, m, k, n, numExperts);

        float[] actual = new float[expected.Length];
        device.Download(yBuf, actual);

        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float bar = 2e-4f + 1e-3f * MathF.Abs(expected[i]);
            Assert.True(diff <= bar,
                $"row={i / m}, col={i % m}: cpu={expected[i]:F6} vs vulkan={actual[i]:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    private static byte[] QuantizeBankToQ8(
        float[] bank, int numExperts, int m, int k, out float[] dequantBank)
    {
        if ((k % 32) != 0) throw new ArgumentException("k must be a multiple of 32.", nameof(k));

        int blocksPerRow = k / 32;
        int rowBytes = blocksPerRow * 34;
        byte[] q8 = new byte[numExperts * m * rowBytes];
        dequantBank = new float[bank.Length];

        for (int row = 0; row < numExperts * m; row++)
        {
            int srcBase = row * k;
            int dstBase = row * rowBytes;
            for (int block = 0; block < blocksPerRow; block++)
            {
                int blockSrc = srcBase + block * 32;
                int blockDst = dstBase + block * 34;
                float maxAbs = 0f;
                for (int i = 0; i < 32; i++)
                    maxAbs = MathF.Max(maxAbs, MathF.Abs(bank[blockSrc + i]));

                float d = maxAbs == 0f ? 0f : maxAbs / 127f;
                ushort half = BitConverter.HalfToUInt16Bits((Half)d);
                q8[blockDst] = (byte)(half & 0xFF);
                q8[blockDst + 1] = (byte)(half >> 8);
                float dRoundTrip = (float)BitConverter.UInt16BitsToHalf(half);

                for (int i = 0; i < 32; i++)
                {
                    int q = d == 0f ? 0 : (int)MathF.Round(bank[blockSrc + i] / d);
                    q = Math.Clamp(q, -128, 127);
                    q8[blockDst + 2 + i] = unchecked((byte)(sbyte)q);
                    dequantBank[blockSrc + i] = dRoundTrip * q;
                }
            }
        }

        return q8;
    }

    private static float[] CpuIndexedMatmul(
        float[] bank, float[] x, int[] indices, int m, int k, int n, int numExperts)
    {
        var y = new float[n * m];
        int bankRow = m * k;
        for (int row = 0; row < n; row++)
        {
            int idx = Math.Clamp(indices[row], 0, numExperts - 1);
            int expertBase = idx * bankRow;
            int xRowBase = row * k;
            int yRowBase = row * m;
            for (int outIdx = 0; outIdx < m; outIdx++)
            {
                int weightRowBase = expertBase + outIdx * k;
                float acc = 0f;
                for (int j = 0; j < k; j++)
                    acc += bank[weightRowBase + j] * x[xRowBase + j];
                y[yRowBase + outIdx] = acc;
            }
        }
        return y;
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
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
