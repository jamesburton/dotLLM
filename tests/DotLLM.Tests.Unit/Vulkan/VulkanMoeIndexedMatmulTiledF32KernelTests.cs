using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the tiled (shared-memory) MoE indexed expert
/// matmul Vulkan kernel. Mirrors <see cref="VulkanMoeIndexedMatmulF32KernelTests"/>
/// against the same scalar CPU reference.
/// </summary>
/// <remarks>
/// The tiled variant differs from the scalar one only in:
/// (1) workgroup shape — one WG per output row, TILE_M = TILE_K = 16 — and
/// (2) reduction order along K (chunked by TILE_K inside shared mem before
/// summing into the per-thread accumulator). Pure F32; no quant, no
/// precision change. The reduction-order shuffle adds at most a few ULPs
/// at these shapes, so we keep the same abs 1e-4 / rel 1e-3 tolerance as
/// the scalar variant.
///
/// Shapes intentionally include odd M and odd K (not multiples of TILE_M
/// or TILE_K) to exercise the in-shader bounds checks on both axes.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMoeIndexedMatmulTiledF32KernelTests
{
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(1, 4, 16, 8, 2)]      // smallest: 1 row, M=16=TILE_M, K=8 (< TILE_K → tail-only)
    [InlineData(2, 4, 16, 8, 2)]      // n=2 rows, both pick the same expert
    [InlineData(2, 4, 16, 8, 3)]      // n=2 rows, picking different experts
    [InlineData(8, 8, 128, 64, 4)]    // Mixtral-tiny-ish: M=128 (8×TILE_M), K=64 (4×TILE_K)
    [InlineData(16, 8, 64, 128, 4)]   // down-projection: M=64 (4×TILE_M), K=128 (8×TILE_K)
    [InlineData(1, 64, 96, 48, 8)]    // M=96 (6×TILE_M), K=48 (3×TILE_K)
    [InlineData(4, 4, 24, 20, 2)]     // odd shapes: M=24 (TILE_M+8), K=20 (TILE_K+4) — tail on both axes
    [InlineData(8, 4, 17, 33, 3)]     // very ragged: M=17 (TILE_M+1), K=33 (2*TILE_K+1)
    public void Launch_MatchesCpuReference(int n, int numExperts, int m, int k, int activeExperts)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xC0FFEE + n * 31 + numExperts * 17 + m * 11 + k * 7 + activeExperts);

        float[] bank = RandomFloats(rng, numExperts * m * k);
        float[] x = RandomFloats(rng, n * k);
        int[] indices = RandomIndices(rng, n, numExperts, activeExperts);

        float[] expected = CpuIndexedMatmul(bank, x, indices, m, k, n, numExperts);

        using var device = VulkanDevice.Create();
        using var kernel = MoeIndexedMatmulTiledF32Kernel.Create(device, spvDir);

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
        // Sanity check: when every row picks expert 0, the tiled kernel must
        // produce the same output as a plain matmul against bank[0].
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int n = 4, m = 32, k = 16, numExperts = 4;
        var rng = new Random(unchecked((int)0xCAFEBABE));

        float[] bank = RandomFloats(rng, numExperts * m * k);
        float[] x = RandomFloats(rng, n * k);
        int[] indices = new int[n]; // all zeros → all rows pick expert 0

        float[] expected = CpuPlainMatmul(bank.AsSpan(0, m * k).ToArray(), x, m, k, n);

        using var device = VulkanDevice.Create();
        using var kernel = MoeIndexedMatmulTiledF32Kernel.Create(device, spvDir);
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
            Assert.True(diff <= AbsTol + RelTol * MathF.Abs(expected[i]),
                $"i={i}: plain={expected[i]:F6} vs vulkan-indexed-tiled={actual[i]:F6} (|diff|={diff:E3})");
        }
    }

    [SkippableFact]
    public void Launch_MatchesScalarVariant()
    {
        // Cross-check: tiled vs scalar on the same inputs must agree within
        // the same parity bar. This is the strongest test that the only
        // difference between the two kernels is reduction order.
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int n = 64, numExperts = 8, m = 64, k = 64, activeExperts = 4;
        var rng = new Random(0x12345678);

        float[] bank = RandomFloats(rng, numExperts * m * k);
        float[] x = RandomFloats(rng, n * k);
        int[] indices = RandomIndices(rng, n, numExperts, activeExperts);

        using var device = VulkanDevice.Create();
        using var scalarKernel = MoeIndexedMatmulF32Kernel.Create(device, spvDir);
        using var tiledKernel = MoeIndexedMatmulTiledF32Kernel.Create(device, spvDir);

        using var bufBank = device.Allocate((long)bank.Length * sizeof(float));
        using var bufX = device.Allocate((long)x.Length * sizeof(float));
        using var bufIdx = device.Allocate((long)indices.Length * sizeof(int));
        using var bufYScalar = device.Allocate((long)n * m * sizeof(float));
        using var bufYTiled = device.Allocate((long)n * m * sizeof(float));

        device.Upload(bank, bufBank);
        device.Upload(x, bufX);
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(indices.AsSpan()), bufIdx);

        scalarKernel.Launch(bufBank, bufX, bufIdx, bufYScalar, m, k, n, numExperts);
        tiledKernel.Launch(bufBank, bufX, bufIdx, bufYTiled, m, k, n, numExperts);

        float[] scalar = new float[n * m];
        float[] tiled = new float[n * m];
        device.Download(bufYScalar, scalar);
        device.Download(bufYTiled, tiled);

        for (int i = 0; i < scalar.Length; i++)
        {
            float diff = MathF.Abs(scalar[i] - tiled[i]);
            float bar = AbsTol + RelTol * MathF.Abs(scalar[i]);
            Assert.True(diff <= bar,
                $"i={i}: scalar={scalar[i]:F6} vs tiled={tiled[i]:F6} (|diff|={diff:E3} > {bar:E3})");
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
