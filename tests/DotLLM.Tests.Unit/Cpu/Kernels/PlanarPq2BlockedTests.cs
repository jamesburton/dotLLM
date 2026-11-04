using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>Serialises tests that flip the process-wide planar toggle (they must not overlap any other PQ2_0 test).</summary>
[CollectionDefinition("PQ2PlanarToggle", DisableParallelization = true)]
public sealed class PQ2PlanarToggleCollection;

/// <summary>
/// Token-blocked planar PQ2_0 GEMM (#1620): per-token results are bit-identical to the n = 1 planar GEMV (same per-token op order),
/// the F64 oracle bounds the error, every remainder shape (n not a multiple of the 4-token block) is covered, and the
/// <c>GemmPQ2_0</c> dispatch honours the toggle and the n limit.
/// </summary>
[Collection("PQ2PlanarToggle")]
public sealed unsafe class PlanarPq2BlockedTests
{
    private static byte[] RandomPq2Rows(Random rng, int rows, int k)
    {
        int groups = k / 128;
        var b = new byte[(long)rows * groups * 34];
        for (long g = 0; g < (long)rows * groups; g++)
        {
            ushort half = BitConverter.HalfToUInt16Bits((Half)(0.01f + (float)rng.NextDouble() * 0.2f));
            b[g * 34] = (byte)half; b[g * 34 + 1] = (byte)(half >> 8);
            for (int i = 0; i < 32; i++)
            {
                int packed = 0;
                for (int f = 0; f < 4; f++) packed |= rng.Next(3) << (2 * f);   // real PQ2_0 never stores code 3
                b[g * 34 + 2 + i] = (byte)packed;
            }
        }
        return b;
    }

    private static float[] Acts(Random rng, int n, int k)
    {
        var x = new float[n * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * (i % 53 == 0 ? 5f : 1f);
        return x;
    }

    [Theory]
    [InlineData(128, 5, 2)]
    [InlineData(256, 9, 3)]
    [InlineData(1280, 33, 4)]
    [InlineData(1280, 33, 5)]
    [InlineData(640, 17, 7)]
    [InlineData(512, 20, 8)]
    [InlineData(384, 11, 9)]
    [InlineData(384, 11, 13)]
    public void Blocked_IsBitIdentical_ToSingleTokenPlanar_ForEveryRemainder(int k, int m, int n)
    {
        var rng = new Random(1620 + n);
        byte[] w = RandomPq2Rows(rng, m, k);
        float[] x = Acts(rng, n, k);
        const float Canary = 12345.5f;
        var blocked = new float[(n + 4) * m]; var single = new float[n * m];   // 4 canary rows past the n real ones: a block that over-runs the remainder writes there
        Array.Fill(blocked, Canary);
        fixed (byte* wp = w) fixed (float* xp = x) fixed (float* cb = blocked) fixed (float* cs = single)
        {
            MatMul.GemmPQ2_0PlanarBlocked(wp, xp, cb, m, k, n, null);
            for (int t = 0; t < n; t++) MatMul.GemvPQ2_0Planar(wp, xp + t * k, cs + t * m, m, k, null);
        }
        for (int i = n * m; i < blocked.Length; i++) Assert.True(blocked[i] == Canary, $"blocked kernel wrote past token n-1 (element {i})");
        for (int i = 0; i < single.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(blocked[i]) == BitConverter.SingleToInt32Bits(single[i]),
                $"token {i / m} row {i % m}: blocked {blocked[i]:R} != single {single[i]:R}");
    }

    [Theory]
    [InlineData(1280, 33, 4)]
    [InlineData(1280, 33, 6)]
    public void Blocked_MatchesF64Truth_AsWellAsProductionGemm(int k, int m, int n)
    {
        var rng = new Random(77 + n);
        byte[] w = RandomPq2Rows(rng, m, k);
        float[] x = Acts(rng, n, k);
        var blocked = new float[n * m]; var prod = new float[n * m]; var truth = new double[n * m];
        fixed (byte* wp = w) fixed (float* xp = x) fixed (float* cb = blocked) fixed (float* cp = prod)
        {
            var tmp = new float[k];
            long rb = (long)(k / 128) * 34;
            for (int r = 0; r < m; r++)
            {
                Dequantize.ToFloat32((nint)(wp + r * rb), k, QuantizationType.PQ2_0, tmp);
                for (int t = 0; t < n; t++)
                {
                    double s = 0; for (int i = 0; i < k; i++) s += (double)tmp[i] * x[t * k + i];
                    truth[t * m + r] = s;
                }
            }
            MatMul.GemmPQ2_0PlanarBlocked(wp, xp, cb, m, k, n, null);
            bool saved = MatMul.PQ2_0PlanarEnabled;
            MatMul.PQ2_0PlanarEnabled = false;
            try { MatMul.GemmPQ2_0(wp, xp, cp, m, k, n, null); } finally { MatMul.PQ2_0PlanarEnabled = saved; }
        }
        double rms = Math.Sqrt(truth.Sum(v => v * v) / truth.Length);
        double Err(float[] c) => Math.Sqrt(Enumerable.Range(0, c.Length).Sum(i => Math.Pow(c[i] - truth[i], 2)) / c.Length) / rms;
        Assert.True(Err(blocked) < 0.02, $"blocked rel-RMS {Err(blocked)}");
        Assert.True(Err(blocked) < 3 * Err(prod) + 1e-4, $"blocked {Err(blocked)} vs production {Err(prod)}");
    }

    [Fact]
    public void GemmDispatch_UsesBlockedPlanarOnlyWhenEnabledAndWithinNMax()
    {
        const int k = 1280, m = 33;
        var rng = new Random(5);
        byte[] w = RandomPq2Rows(rng, m, k);
        bool saved = MatMul.PQ2_0PlanarEnabled; int savedMax = MatMul.PQ2_0PlanarMaxN;
        try
        {
            foreach (int n in new[] { 2, 4, 5 })
            {
                float[] x = Acts(rng, n, k);
                var off = new float[n * m]; var on = new float[n * m]; var direct = new float[n * m];
                fixed (byte* wp = w) fixed (float* xp = x) fixed (float* co = off) fixed (float* cn = on) fixed (float* cd = direct)
                {
                    MatMul.PQ2_0PlanarMaxN = 4;
                    MatMul.PQ2_0PlanarEnabled = false;
                    MatMul.GemmPQ2_0(wp, xp, co, m, k, n, null);
                    if (!global::System.Runtime.Intrinsics.X86.Avx2.IsSupported) continue;   // toggle ignored without AVX2 (scalar CI leg)
                    MatMul.PQ2_0PlanarEnabled = true;
                    long calls0 = MatMul.PQ2_0PlanarCalls;
                    MatMul.GemmPQ2_0(wp, xp, cn, m, k, n, null);
                    long calls = MatMul.PQ2_0PlanarCalls - calls0;
                    if (n <= 4)
                    {
                        Assert.Equal(1, calls);
                        MatMul.GemmPQ2_0PlanarBlocked(wp, xp, cd, m, k, n, null);
                        Assert.Equal(direct, on);
                        Assert.NotEqual(off, on);   // different quantization grid: cannot be bit-identical to production
                    }
                    else
                    {
                        Assert.Equal(0, calls);
                        Assert.Equal(off, on);     // above NMAX: production path, bit-identical
                    }
                }
            }
        }
        finally { MatMul.PQ2_0PlanarEnabled = saved; MatMul.PQ2_0PlanarMaxN = savedMax; }
    }
}
