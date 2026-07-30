using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #232 — bit-exactness gate for the 4x4 register-blocked I2_S W2A8 GEMM tile.
///
/// <para>The tile performs the identical per-cell operation sequence in the identical order as the
/// per-cell baseline (only the interleaving across cells changes), so the requirement is
/// <b>bitwise</b> equality, not approximate parity. These tests compare raw
/// <see cref="BitConverter.SingleToInt32Bits(float)"/> patterns.</para>
///
/// <para><b>Why the large-N cases matter.</b> Upstream kkokosa/dotLLM#416 recorded a formulation
/// that passed at N=4 and N=32 and failed at N=256 purely because the larger output count samples
/// further into the error tail. Small-N parity is therefore not evidence; the headline case here
/// produces 655,360 outputs.</para>
/// </summary>
public sealed unsafe class I2SRegisterBlockedGemmTests
{
    /// <summary>
    /// Runs both kernels over the same random I2_S weights / activations and returns
    /// (mismatchCount, firstMismatchIndex, baselineValue, tiledValue).
    /// </summary>
    private static (long Mismatches, long FirstIndex, float Baseline, float Tiled) CompareKernels(
        int m, int k, int n, int seed, ComputeThreadPool? pool)
    {
        var rng = new Random(seed);
        int rowBytes = k / 4;
        long weightBytes = (long)m * rowBytes + 4;
        long outCount = (long)m * n;

        byte* weights = (byte*)NativeMemory.AllocZeroed((nuint)weightBytes);
        float* b = (float*)NativeMemory.AllocZeroed((nuint)((long)n * k * sizeof(float)));
        float* cBase = (float*)NativeMemory.AllocZeroed((nuint)(outCount * sizeof(float)));
        float* cTile = (float*)NativeMemory.AllocZeroed((nuint)(outCount * sizeof(float)));
        try
        {
            // Any byte is a valid pack of four 2-bit ternary codes, so random bytes exercise the
            // full {-1,0,+1} distribution including the all-zero and all-saturated patterns.
            byte[] rand = new byte[weightBytes];
            rng.NextBytes(rand);
            Marshal.Copy(rand, 0, (nint)weights, (int)weightBytes);
            *(float*)(weights + (long)m * rowBytes) = 0.0217f; // per-tensor scale

            for (long i = 0; i < (long)n * k; i++)
                b[i] = rng.NextSingle() * 2f - 1f;

            MatMul.BenchGemmI2_SW2A8(weights, b, cBase, m, k, n, tiled: false, pool);
            MatMul.BenchGemmI2_SW2A8(weights, b, cTile, m, k, n, tiled: true, pool);

            long mismatches = 0, firstIdx = -1;
            float fb = 0f, ft = 0f;
            for (long i = 0; i < outCount; i++)
            {
                if (BitConverter.SingleToInt32Bits(cBase[i]) == BitConverter.SingleToInt32Bits(cTile[i]))
                    continue;
                if (mismatches == 0) { firstIdx = i; fb = cBase[i]; ft = cTile[i]; }
                mismatches++;
            }

            return (mismatches, firstIdx, fb, ft);
        }
        finally
        {
            NativeMemory.Free(weights);
            NativeMemory.Free(b);
            NativeMemory.Free(cBase);
            NativeMemory.Free(cTile);
        }
    }

    /// <summary>
    /// The headline exactness gate: 2560x256 = 655,360 outputs, well past the point at which
    /// #416's non-exact formulation broke down.
    /// </summary>
    [Fact]
    public void Tile4x4_IsBitExact_AtLargeOutputCount()
    {
        var (mismatches, idx, fb, ft) = CompareKernels(m: 2560, k: 1024, n: 256, seed: 232, pool: null);
        Assert.True(mismatches == 0,
            $"{mismatches}/655360 outputs differ; first at {idx}: baseline={fb:R} tiled={ft:R}");
    }

    /// <summary>
    /// Ragged shapes: <c>m % 4 != 0</c> exercises the row remainder and <c>n % 4 != 0</c> the token
    /// remainder, both of which fall back to the per-cell kernel inside the tiled path.
    /// </summary>
    [Theory]
    [InlineData(2560, 1024, 253)]   // n % 4 == 1
    [InlineData(2559, 1024, 254)]   // m % 4 == 3, n % 4 == 2
    [InlineData(2557, 1024, 255)]   // m % 4 == 1, n % 4 == 3
    [InlineData(2560, 6912, 130)]   // wide k (BitNet ffn_down row length)
    [InlineData(6912, 2560, 33)]    // tall m (BitNet ffn_up), n just past one tile
    public void Tile4x4_IsBitExact_ForRaggedTileRemainders(int m, int k, int n)
    {
        var (mismatches, idx, fb, ft) = CompareKernels(m, k, n, seed: m * 31 + n, pool: null);
        Assert.True(mismatches == 0,
            $"m={m} k={k} n={n}: {mismatches}/{(long)m * n} outputs differ; first at {idx}: baseline={fb:R} tiled={ft:R}");
    }

    /// <summary>
    /// Threaded run: <c>PartitionRows</c> hands each worker an arbitrary row count, so the tiled
    /// path's row remainder is exercised per worker rather than only once at the end of M.
    /// </summary>
    [Fact]
    public void Tile4x4_IsBitExact_UnderThreadPoolPartitioning()
    {
        using var pool = new ComputeThreadPool(6); // 2560 % 6 != 0 → uneven, non-multiple-of-4 chunks
        var (mismatches, idx, fb, ft) = CompareKernels(m: 2560, k: 1024, n: 256, seed: 4321, pool: pool);
        Assert.True(mismatches == 0,
            $"{mismatches} outputs differ under 6-thread partitioning; first at {idx}: baseline={fb:R} tiled={ft:R}");
    }

    /// <summary>Sanity: the tile actually produces non-trivial output (guards against an all-zero pass).</summary>
    [Fact]
    public void Tile4x4_ProducesNonZeroOutput()
    {
        const int m = 256, k = 512, n = 8;
        var rng = new Random(9);
        int rowBytes = k / 4;
        byte* weights = (byte*)NativeMemory.AllocZeroed((nuint)((long)m * rowBytes + 4));
        float* b = (float*)NativeMemory.AllocZeroed((nuint)((long)n * k * sizeof(float)));
        float* c = (float*)NativeMemory.AllocZeroed((nuint)((long)m * n * sizeof(float)));
        try
        {
            byte[] rand = new byte[(long)m * rowBytes + 4];
            rng.NextBytes(rand);
            Marshal.Copy(rand, 0, (nint)weights, rand.Length);
            *(float*)(weights + (long)m * rowBytes) = 0.05f;
            for (long i = 0; i < (long)n * k; i++) b[i] = rng.NextSingle() * 2f - 1f;

            MatMul.BenchGemmI2_SW2A8(weights, b, c, m, k, n, tiled: true, null);

            int nonZero = 0;
            for (long i = 0; i < (long)m * n; i++) if (c[i] != 0f) nonZero++;
            Assert.True(nonZero > (long)m * n / 2, $"only {nonZero}/{(long)m * n} outputs non-zero");
        }
        finally
        {
            NativeMemory.Free(weights);
            NativeMemory.Free(b);
            NativeMemory.Free(c);
        }
    }
}
