using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Upstream ggml Q2_0 x Q8_0 kernels (#2074). Real Bonsai 2.5 tensors never use code 3 (+2) - measured: 0 of ~1.3M codes - so every test
/// here uses SYNTHETIC blocks that exercise all four codes; a parity test over the real file alone could not tell the ggml map
/// (-1,0,+1,+2) from the ternary map or from a clamp.
/// </summary>
public sealed unsafe class MatMulQ2_0Tests
{
    private static byte[] RandomWeights(int rows, int k, int seed, bool allCodes = true)
    {
        var rng = new Random(seed);
        int blocks = k / 64;
        var w = new byte[rows * blocks * 18];
        for (int b = 0; b < rows * blocks; b++)
        {
            ushort d = BitConverter.HalfToUInt16Bits((Half)(0.01f + (float)rng.NextDouble() * 0.2f));
            w[b * 18] = (byte)d; w[b * 18 + 1] = (byte)(d >> 8);
            for (int i = 0; i < 16; i++)
            {
                int v = 0;
                for (int s = 0; s < 4; s++) v |= (allCodes ? rng.Next(4) : rng.Next(3)) << (2 * s);
                w[b * 18 + 2 + i] = (byte)v;
            }
        }
        return w;
    }

    private static float[] RandomActs(int n, int k, int seed)
    {
        var rng = new Random(seed);
        var x = new float[n * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * (i % 97 == 0 ? 8f : 1f);
        return x;
    }

    /// <summary>Oracle independent of the kernel: dequantized weights (Dequantize.ToFloat32) dot the Q8_0-dequantized activation, in double.</summary>
    private static double[] Oracle(byte[] w, float[] x, int m, int k, int n)
    {
        var res = new double[n * m];
        var wf = new float[k];
        var xq = new byte[(k / 32) * 34];
        for (int t = 0; t < n; t++)
        {
            var xd = new float[k];
            fixed (float* px = x) fixed (byte* pq = xq)
            {
                MatMul.QuantizeF32ToQ8_0(px + t * k, pq, k);
                for (int b = 0; b < k / 32; b++)
                {
                    float d = (float)BitConverter.UInt16BitsToHalf((ushort)(pq[b * 34] | (pq[b * 34 + 1] << 8)));
                    for (int j = 0; j < 32; j++) xd[b * 32 + j] = d * (sbyte)pq[b * 34 + 2 + j];
                }
            }
            for (int r = 0; r < m; r++)
            {
                fixed (byte* pw = w) Dequantize.ToFloat32((nint)(pw + (long)r * (k / 64) * 18), k, QuantizationType.Q2_0, wf);
                double s = 0;
                for (int j = 0; j < k; j++) s += (double)wf[j] * xd[j];
                res[t * m + r] = s;
            }
        }
        return res;
    }

    [Fact]
    public void KnownAnswer_AllFourCodes_PinsTheGgmlMap()
    {
        // d = 1; elements 0..3 carry codes 0,1,2,3 (= -1, 0, +1, +2); activations are ones => dot = -1+0+1+2 = 2 for the first four, then 15 more groups of the same.
        byte[] w = new byte[18];
        w[0] = (byte)BitConverter.HalfToUInt16Bits((Half)1f); w[1] = (byte)(BitConverter.HalfToUInt16Bits((Half)1f) >> 8);
        for (int i = 0; i < 16; i++) w[2 + i] = 0b11_10_01_00;
        byte[] q8 = new byte[2 * 34];
        for (int b = 0; b < 2; b++)
        {
            q8[b * 34] = (byte)BitConverter.HalfToUInt16Bits((Half)1f); q8[b * 34 + 1] = (byte)(BitConverter.HalfToUInt16Bits((Half)1f) >> 8);
            for (int j = 0; j < 32; j++) q8[b * 34 + 2 + j] = 1;
        }
        fixed (byte* pw = w) fixed (byte* py = q8)
        {
            Assert.Equal(16 * 2f, MatMul.VecDotQ2_0Q8_0Scalar(pw, py, 1));
            if (System.Runtime.Intrinsics.X86.Avx2.IsSupported) Assert.Equal(16 * 2f, MatMul.VecDotQ2_0Q8_0Avx2(pw, py, 1));
        }
    }

    [Fact]
    public void KnownAnswer_PositionWithinBlock_IsElementOrder()
    {
        // Activation is nonzero only at element 37; weight code there is 3 (+2): result = 2 * y * d. Pins byte/shift placement for the high half.
        byte[] w = new byte[18];
        ushort one = BitConverter.HalfToUInt16Bits((Half)1f);
        w[0] = (byte)one; w[1] = (byte)(one >> 8);
        for (int i = 0; i < 16; i++) w[2 + i] = 0b01_01_01_01;   // all zero weights
        w[2 + 37 / 4] = (byte)(0b01_01_01_01 & ~(3 << (2 * (37 % 4))) | (3 << (2 * (37 % 4))));
        byte[] q8 = new byte[2 * 34];
        for (int b = 0; b < 2; b++) { q8[b * 34] = (byte)one; q8[b * 34 + 1] = (byte)(one >> 8); }
        q8[34 + 2 + 5] = 7;   // element 32 + 5 = 37
        fixed (byte* pw = w) fixed (byte* py = q8)
        {
            Assert.Equal(14f, MatMul.VecDotQ2_0Q8_0Scalar(pw, py, 1));
            if (System.Runtime.Intrinsics.X86.Avx2.IsSupported) Assert.Equal(14f, MatMul.VecDotQ2_0Q8_0Avx2(pw, py, 1));
        }
    }

    [Theory]
    [InlineData(64, 11)]
    [InlineData(320, 12)]
    [InlineData(5120, 13)]
    public void ScalarAndAvx2_AreBitIdentical_AndMatchOracle(int k, int seed)
    {
        var w = RandomWeights(1, k, seed);
        var x = RandomActs(1, k, seed + 100);
        var xq = new byte[(k / 32) * 34];
        fixed (byte* pw = w) fixed (byte* py = xq) fixed (float* px = x)
        {
            MatMul.QuantizeF32ToQ8_0(px, py, k);
            float s = MatMul.VecDotQ2_0Q8_0Scalar(pw, py, k / 64);
            double o = Oracle(w, x, 1, k, 1)[0];
            Assert.True(Math.Abs(s - o) <= 1e-4 * Math.Max(1.0, Math.Abs(o)), $"scalar {s} vs oracle {o}");
            if (System.Runtime.Intrinsics.X86.Avx2.IsSupported)
                Assert.Equal(BitConverter.SingleToInt32Bits(s), BitConverter.SingleToInt32Bits(MatMul.VecDotQ2_0Q8_0Avx2(pw, py, k / 64)));
        }
    }

    [Theory]
    [InlineData(1, false)]
    [InlineData(1, true)]
    [InlineData(7, true)]
    [InlineData(33, true)]
    public void Gemm_MatchesOracle_SerialAndPooled(int n, bool pooled)
    {
        const int m = 70, k = 384;   // m >= 32 so the pooled path really splits rows
        var w = RandomWeights(m, k, 5);
        var x = RandomActs(n, k, 6);
        var c = new float[n * m];
        using var pool = pooled ? new ComputeThreadPool(4) : null;
        fixed (byte* pw = w) fixed (float* px = x) fixed (float* pc = c)
            MatMul.GemmQ2_0(pw, px, pc, m, k, n, pool);
        var o = Oracle(w, x, m, k, n);
        for (int i = 0; i < c.Length; i++)
            Assert.True(Math.Abs(c[i] - o[i]) <= 2e-4 * Math.Max(1.0, Math.Abs(o[i])), $"[{i}] {c[i]} vs {o[i]}");
    }

    [Fact]
    public void RealisticTernaryBlocks_NeverUseCodeThree_ButStillMatch()
    {
        // The Bonsai-shaped distribution (codes 0..2 only) must also agree; this alone could not discriminate the code-3 map.
        const int m = 8, k = 256;
        var w = RandomWeights(m, k, 9, allCodes: false);
        var x = RandomActs(1, k, 10);
        var c = new float[m];
        fixed (byte* pw = w) fixed (float* px = x) fixed (float* pc = c) MatMul.GemvQ2_0(pw, px, pc, m, k, null);
        var o = Oracle(w, x, m, k, 1);
        for (int i = 0; i < m; i++) Assert.True(Math.Abs(c[i] - o[i]) <= 2e-4 * Math.Max(1.0, Math.Abs(o[i])));
    }
}
