using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the MoE indexed expert matmul Vulkan kernel
/// (per-row weight selection from a packed expert weight bank).
/// </summary>
/// <remarks>
/// Reference is a self-contained scalar CPU loop:
/// <c>y[n, m] = sum_k bank[indices[n], m, k] * x[n, k]</c>. Tolerance is
/// abs 1e-4 / rel 1e-3 — same as the standalone <c>matmul_f32</c> parity
/// bar; the only arithmetic difference vs that kernel is the per-row index
/// lookup, which adds no rounding.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMoeIndexedMatmulF32KernelTests
{
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(1, 4, 16, 8, 2)]      // smallest: 1 token, k=1 → n=1; tiny shapes
    [InlineData(2, 4, 16, 8, 2)]      // n=2 rows, both pick the same expert
    [InlineData(2, 4, 16, 8, 3)]      // n=2 rows, picking different experts
    [InlineData(8, 8, 128, 64, 4)]    // Mixtral-tiny-ish: hidden=64, intermediate=128
    [InlineData(16, 8, 64, 128, 4)]   // down-projection shape (M=hidden, K=intermediate)
    [InlineData(1, 64, 96, 48, 8)]    // wider expert bank, k=8
    public void Launch_MatchesCpuReference(int n, int numExperts, int m, int k, int activeExperts)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xBADF00D + n * 31 + numExperts * 17 + m * 11 + k * 7 + activeExperts);

        float[] bank = RandomFloats(rng, numExperts * m * k);
        float[] x = RandomFloats(rng, n * k);
        int[] indices = RandomIndices(rng, n, numExperts, activeExperts);

        float[] expected = CpuIndexedMatmul(bank, x, indices, m, k, n, numExperts);

        using var device = VulkanDevice.Create();
        using var kernel = MoeIndexedMatmulF32Kernel.Create(device, spvDir);

        using var bufBank = device.Allocate((long)bank.Length * sizeof(float));
        using var bufX = device.Allocate((long)x.Length * sizeof(float));
        using var bufIdx = device.Allocate((long)indices.Length * sizeof(int));
        using var bufY = device.Allocate((long)expected.Length * sizeof(float));

        device.Upload(bank, bufBank);
        device.Upload(x, bufX);
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(indices.AsSpan()), bufIdx);

        kernel.Launch(bufBank, bufX, bufIdx, bufY, m, k, n, numExperts);

        float[] actual = new float[expected.Length];
        device.Download(bufY, actual);

        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float bar = AbsTol + RelTol * MathF.Abs(expected[i]);
            Assert.True(diff <= bar,
                $"row={(int)(i / m)}, col={(int)(i % m)}: cpu={expected[i]:F6} vs vulkan={actual[i]:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    [SkippableFact]
    public void Launch_AllRowsSameExpert_MatchesPlainMatmul()
    {
        // Sanity check: when every row picks expert 0, the kernel must
        // produce the same output as a plain matmul against bank[0].
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int n = 4, m = 32, k = 16, numExperts = 4;
        var rng = new Random(unchecked((int)0xDEADBEEF));

        float[] bank = RandomFloats(rng, numExperts * m * k);
        float[] x = RandomFloats(rng, n * k);
        int[] indices = new int[n]; // all zeros → all rows pick expert 0

        float[] expectedFromIdx = CpuIndexedMatmul(bank, x, indices, m, k, n, numExperts);
        float[] expectedFromPlain = CpuPlainMatmul(bank.AsSpan(0, m * k).ToArray(), x, m, k, n);

        for (int i = 0; i < expectedFromIdx.Length; i++)
            Assert.Equal(expectedFromPlain[i], expectedFromIdx[i]);

        using var device = VulkanDevice.Create();
        using var kernel = MoeIndexedMatmulF32Kernel.Create(device, spvDir);
        using var bufBank = device.Allocate((long)bank.Length * sizeof(float));
        using var bufX = device.Allocate((long)x.Length * sizeof(float));
        using var bufIdx = device.Allocate((long)indices.Length * sizeof(int));
        using var bufY = device.Allocate((long)expectedFromPlain.Length * sizeof(float));

        device.Upload(bank, bufBank);
        device.Upload(x, bufX);
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(indices.AsSpan()), bufIdx);

        kernel.Launch(bufBank, bufX, bufIdx, bufY, m, k, n, numExperts);

        float[] actual = new float[expectedFromPlain.Length];
        device.Download(bufY, actual);

        for (int i = 0; i < expectedFromPlain.Length; i++)
        {
            float diff = MathF.Abs(expectedFromPlain[i] - actual[i]);
            Assert.True(diff <= AbsTol + RelTol * MathF.Abs(expectedFromPlain[i]),
                $"i={i}: plain={expectedFromPlain[i]:F6} vs vulkan-indexed={actual[i]:F6} (|diff|={diff:E3})");
        }
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

    private static float[] CpuPlainMatmul(float[] weight, float[] x, int m, int k, int n)
    {
        var y = new float[n * m];
        for (int row = 0; row < n; row++)
            for (int outIdx = 0; outIdx < m; outIdx++)
            {
                float acc = 0f;
                int wBase = outIdx * k;
                int xBase = row * k;
                for (int j = 0; j < k; j++)
                    acc += weight[wBase + j] * x[xBase + j];
                y[row * m + outIdx] = acc;
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
        // Pick `activePool` distinct expert IDs and randomly draw rows from them.
        // This exercises the per-row selection without making every row unique.
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
