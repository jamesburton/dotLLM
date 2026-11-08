using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// TERN_B64_F16 (issue #1706): layout/size, the dequant oracle, invalid-code rejection and the two CPU GEMM kernels (F32-activation scalar reference and
/// int8-activation production kernel). The expected values come from an INDEPENDENT decoder written straight from the spec (<see cref="SpecDecode"/>),
/// never from the code under test, so a self-consistent pack/unpack pair cannot hide a layout error.
/// </summary>
public sealed unsafe class TernB64F16Tests
{
    // ───────────────────────── spec-derived oracle (independent of TernB64F16) ─────────────────────────

    /// <summary>w[row, 64*b + j] per the spec: scale f16 LE at block start; byte k of 16 holds columns 4k..4k+3, LSB first; 00=0, 01=+1, 10=-1.</summary>
    internal static double[] SpecDecode(byte[] packed, int m, int k)
    {
        int bpr = k / 64;
        var w = new double[m * k];
        for (int r = 0; r < m; r++)
            for (int b = 0; b < bpr; b++)
            {
                int off = (r * bpr + b) * 18;
                double scale = (double)BitConverter.ToHalf(packed, off);
                for (int j = 0; j < 64; j++)
                {
                    int code = (packed[off + 2 + j / 4] >> (2 * (j % 4))) & 3;
                    Assert.NotEqual(3, code);
                    w[r * k + b * 64 + j] = (code == 1 ? 1 : code == 2 ? -1 : 0) * scale;
                }
            }
        return w;
    }

    internal static (sbyte[] Trits, float[] Scales, byte[] Packed) RandomTensor(int m, int k, int seed, double zeroFraction = 0.42, bool zeroSomeScales = false)
    {
        var rng = new Random(seed);
        var trits = new sbyte[m * k];
        for (int i = 0; i < trits.Length; i++)
            trits[i] = rng.NextDouble() < zeroFraction ? (sbyte)0 : (rng.Next(2) == 0 ? (sbyte)1 : (sbyte)-1);
        var scales = new float[m * (k / 64)];
        for (int i = 0; i < scales.Length; i++)
            scales[i] = zeroSomeScales && i % 5 == 0 ? 0f : (float)(Math.Exp(rng.NextDouble() * 4 - 6)); // ~0.0025 .. 0.135, like real expert scales
        return (trits, scales, TernB64F16.Pack(trits, scales));
    }

    private static float[] RandomX(int n, int k, int seed)
    {
        var rng = new Random(seed);
        var x = new float[n * k];
        for (int i = 0; i < x.Length; i++)
        {
            // Box-Muller N(0,1) with a few outliers (activation-like).
            double u1 = 1 - rng.NextDouble(), u2 = rng.NextDouble();
            x[i] = (float)(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
            if (rng.Next(97) == 0) x[i] *= 8;
        }
        return x;
    }

    private static double[] RefGemm(double[] w, float[] x, int m, int k, int n)
    {
        var c = new double[n * m];
        for (int t = 0; t < n; t++)
            for (int r = 0; r < m; r++)
            {
                double acc = 0;
                for (int j = 0; j < k; j++) acc += w[r * k + j] * x[t * k + j];
                c[t * m + r] = acc;
            }
        return c;
    }

    private static float[] RunF32(byte[] packed, float[] x, int m, int k, int n, ComputeThreadPool? pool = null)
    {
        var c = new float[n * m];
        fixed (byte* pw = packed) fixed (float* px = x) fixed (float* pc = c)
            TernB64F16.GemmF32Act(pw, px, pc, m, k, n, pool);
        return c;
    }

    private static float[] RunQ8(byte[] packed, float[] x, int m, int k, int n, ComputeThreadPool? pool = null)
    {
        var c = new float[n * m];
        fixed (byte* pw = packed) fixed (float* px = x) fixed (float* pc = c)
            TernB64F16.GemmQ8(pw, px, pc, m, k, n, pool);
        return c;
    }

    // ───────────────────────── geometry ─────────────────────────

    [Theory]
    [InlineData(64, 18)]
    [InlineData(2048, 2048 / 64 * 18)]
    [InlineData(576, 576 / 64 * 18)]
    public void ByteCounts_AreEighteenBytesPerSixtyFourWeights(long n, long expected)
    {
        Assert.Equal(expected, QuantizationType.TERN_B64_F16.ComputeByteCount(n));
        Assert.Equal(expected, Dequantize.RowByteSize(n, QuantizationType.TERN_B64_F16));
        Assert.Equal(expected, QuantFormat.TryGetInfo(QuantizationType.TERN_B64_F16)!.Value.RowByteSize(n));
        Assert.Equal(expected, TernB64F16.RowBytes((int)n));
        Assert.Equal(2.25, QuantFormat.TryGetInfo(QuantizationType.TERN_B64_F16)!.Value.BitsPerWeight);
    }

    [Fact]
    public void EnumId_IsDotllmPrivate243_AndDistinctFromQ2_0()
    {
        Assert.Equal(243, (int)QuantizationType.TERN_B64_F16);
        // Same 18 B/64 geometry as upstream Q2_0, but a different code map: the type must not be conflated with it.
        Assert.Equal(QuantizationType.Q2_0.ComputeByteCount(64), QuantizationType.TERN_B64_F16.ComputeByteCount(64));
        Assert.NotEqual((int)QuantizationType.Q2_0, (int)QuantizationType.TERN_B64_F16);
    }

    // ───────────────────────── bit order (hand-checkable) ─────────────────────────

    [Fact]
    public void BitOrder_HandCheckedBlock()
    {
        // scale = 0.5; byte0 = trits for cols 0..3 = [+1, -1, 0, +1] -> codes 01,10,00,01 -> LSB first: 01 | 10<<2 | 00<<4 | 01<<6 = 0b01_00_10_01 = 0x49
        // byte1 = cols 4..7 = [-1, 0, 0, +1] -> 10 | 00<<2 | 00<<4 | 01<<6 = 0b01_00_00_10 = 0x42 ; byte 15 = cols 60..63 = [+1,0,0,-1] -> 01 | 10<<6 = 0x81
        var blk = new byte[18];
        BitConverter.TryWriteBytes(blk.AsSpan(0, 2), (Half)0.5f);
        blk[2] = 0x49; blk[3] = 0x42; blk[17] = 0x81;
        var dst = new float[64];
        fixed (byte* p = blk) TernB64F16.Dequantize(p, 64, dst);
        var expected = new float[64];
        expected[0] = 0.5f; expected[1] = -0.5f; expected[2] = 0f; expected[3] = 0.5f;
        expected[4] = -0.5f; expected[7] = 0.5f;
        expected[60] = 0.5f; expected[63] = -0.5f;
        Assert.Equal(expected, dst);

        // Pack must produce exactly these bytes (writer and reader agree with the hand derivation, not just each other).
        var trits = new sbyte[64];
        trits[0] = 1; trits[1] = -1; trits[3] = 1; trits[4] = -1; trits[7] = 1; trits[60] = 1; trits[63] = -1;
        Assert.Equal(blk, TernB64F16.Pack(trits, [0.5f]));

        // And the F32 kernel multiplies by the right columns: x = column index + 1.
        var x = Enumerable.Range(1, 64).Select(i => (float)i).ToArray();
        float[] y = RunF32(blk, x, 1, 64, 1);
        // 0.5*(1 - 2 + 4 - 5 + 8 + 61 - 64) = 0.5 * 3
        Assert.Equal(1.5f, y[0], 5);
    }

    [Fact]
    public void ScaleIsLittleEndianFp16_AndAppliesPerBlockPerRow()
    {
        // Two rows, two blocks; all trits +1; x = 1 => y[r] = 64*(s[r,0] + s[r,1]) exactly representable.
        var trits = Enumerable.Repeat((sbyte)1, 2 * 128).ToArray();
        float[] scales = [0.5f, 0.25f, 1f, 2f];
        var packed = TernB64F16.Pack(trits, scales);
        Assert.Equal((byte)0x00, packed[0]); Assert.Equal((byte)0x38, packed[1]); // 0.5 = 0x3800 LE
        var y = RunF32(packed, Enumerable.Repeat(1f, 128).ToArray(), 2, 128, 1);
        Assert.Equal(48f, y[0]); Assert.Equal(192f, y[1]);
    }

    // ───────────────────────── dequant oracle ─────────────────────────

    [Theory]
    [InlineData(4, 64)]
    [InlineData(3, 576)]
    [InlineData(2, 2048)]
    public void Dequantize_MatchesSpecDecoder(int m, int k)
    {
        var (trits, scales, packed) = RandomTensor(m, k, seed: 7 + k);
        var spec = SpecDecode(packed, m, k);
        var dst = new float[m * k];
        fixed (byte* p = packed) TernB64F16.Dequantize(p, m * k, dst);
        for (int i = 0; i < dst.Length; i++) Assert.Equal(spec[i], (double)dst[i]);
        // and it is trit * fp16(scale)
        for (int i = 0; i < dst.Length; i++)
            Assert.Equal(trits[i] * (double)(Half)scales[i / 64], (double)dst[i]);
    }

    // ───────────────────────── invalid code ─────────────────────────

    private static void ValidateArr(string name, byte[] data, int m, int k) { fixed (byte* p = data) TernB64F16.Validate(name, p, m, k); }
    private static long FindArr(byte[] data) { fixed (byte* p = data) return TernB64F16.FindInvalidBlock(p, data.Length); }
    private static void DequantArr(byte[] data, int n) { fixed (byte* p = data) TernB64F16.Dequantize(p, n, new float[n]); }

    [Fact]
    public void ValidCodes_NeverFlagged_AndEveryInvalidPosition_Flagged()
    {
        var (_, _, clean) = RandomTensor(3, 128, seed: 5);
        Assert.Equal(-1, FindArr(clean));
        Assert.Equal(-1, TernB64F16.FindInvalidBlock(clean));
        ValidateArr("t", clean, 3, 128);
        for (int byteIdx = 0; byteIdx < 16; byteIdx++)
            for (int slot = 0; slot < 4; slot++)
            {
                var bad = (byte[])clean.Clone();
                int block = 4; // row 2, block 0
                bad[block * 18 + 2 + byteIdx] |= (byte)(3 << (2 * slot));
                Assert.Equal(block, FindArr(bad));
                Assert.Equal(block, TernB64F16.FindInvalidBlock(bad));
                var ex = Assert.Throws<InvalidDataException>(() => ValidateArr("blk.0.ffn_gate.3.weight", bad, 3, 128));
                Assert.Contains("row 2", ex.Message);
                Assert.Contains("blk.0.ffn_gate.3.weight", ex.Message);
                Assert.Throws<InvalidDataException>(() => DequantArr(bad, 3 * 128));
            }
    }

    [Fact]
    public void ScaleBytes_AreNotMistakenForCodes()
    {
        // A scale whose bytes contain 0b11 pairs (fp16 0x03FF = subnormal) must not trip the invalid-code scan.
        var packed = new byte[18];
        packed[0] = 0xFF; packed[1] = 0x03;
        Assert.Equal(-1, FindArr(packed));
    }

    [Fact]
    public void InDimension_NotMultipleOf64_IsRejected()
    {
        var packed = new byte[18 * 2];
        Assert.Throws<InvalidDataException>(() => ValidateArr("t", packed, 2, 96));
        Assert.Throws<ArgumentException>(() => RunQ8(packed, new float[96], 1, 96, 1));
    }

    // ───────────────────────── F32-activation kernel vs double oracle ─────────────────────────

    [Theory]
    [InlineData(8, 64, 1)]
    [InlineData(5, 2048, 3)]
    [InlineData(7, 576, 2)]
    [InlineData(3, 1024, 1)]
    [InlineData(70, 128, 2)]     // above the parallel threshold
    public void F32Act_MatchesDoubleOracle_RandomTrits(int m, int k, int n)
    {
        var (_, _, packed) = RandomTensor(m, k, seed: m * 31 + k);
        var x = RandomX(n, k, seed: 11 + k);
        var w = SpecDecode(packed, m, k);
        var reference = RefGemm(w, x, m, k, n);
        var y = RunF32(packed, x, m, k, n);
        for (int t = 0; t < n; t++)
            for (int r = 0; r < m; r++)
            {
                // Bound scales with the reduction (sqrt(K) * float eps * sum|w||x|), not a fixed tolerance.
                double abs = 0;
                for (int j = 0; j < k; j++) abs += Math.Abs(w[r * k + j] * x[t * k + j]);
                Assert.True(Math.Abs(y[t * m + r] - reference[t * m + r]) <= 1e-5 * Math.Max(abs, 1e-3),
                    $"[{t},{r}] {y[t * m + r]} vs {reference[t * m + r]} (sum|wx|={abs})");
            }
    }

    [Theory]
    [InlineData("alltpos")]
    [InlineData("allneg")]
    [InlineData("alternating")]
    [InlineData("allzero")]
    public void F32Act_StructuredTrits(string pattern)
    {
        const int m = 4, k = 576;
        var trits = new sbyte[m * k];
        for (int i = 0; i < trits.Length; i++)
            trits[i] = pattern switch { "alltpos" => (sbyte)1, "allneg" => (sbyte)-1, "alternating" => (i % 2 == 0 ? (sbyte)1 : (sbyte)-1), _ => (sbyte)0 };
        var scales = Enumerable.Range(0, m * (k / 64)).Select(i => 0.01f * (1 + i % 7)).ToArray();
        var packed = TernB64F16.Pack(trits, scales);
        var x = RandomX(1, k, 99);
        var w = SpecDecode(packed, m, k);
        var reference = RefGemm(w, x, m, k, 1);
        var y = RunF32(packed, x, m, k, 1);
        for (int r = 0; r < m; r++)
        {
            double abs = 0; for (int j = 0; j < k; j++) abs += Math.Abs(w[r * k + j] * x[j]);
            Assert.True(Math.Abs(y[r] - reference[r]) <= 1e-5 * Math.Max(abs, 1e-3), $"{pattern} row {r}: {y[r]} vs {reference[r]}");
        }
    }

    // ───────────────────────── int8-activation kernel ─────────────────────────

    /// <summary>
    /// Rigorous per-row error bound of activation quantization: each activation is rounded to the Q8_0 grid of its 32-block (step s = absmax/127, half-step
    /// error plus the fp16 rounding of s), so |dy| &lt;= sum_j |w_j| * 0.6 * s_blk(j). A kernel that mis-orders trits, drops a block or swaps the two
    /// Q8_0 halves of a 64-block blows through it by orders of magnitude.
    /// </summary>
    private static double Q8ErrorBound(double[] w, float[] x, int row, int t, int k)
    {
        double bound = 0;
        for (int b = 0; b < k / 32; b++)
        {
            double amax = 0;
            for (int j = 0; j < 32; j++) amax = Math.Max(amax, Math.Abs(x[t * k + b * 32 + j]));
            double s = amax / 127.0;
            for (int j = 0; j < 32; j++) bound += Math.Abs(w[row * k + b * 32 + j]) * 0.6 * s;
        }
        return bound;
    }

    [Theory]
    [InlineData(8, 64, 1, false)]
    [InlineData(6, 2048, 3, false)]
    [InlineData(9, 576, 2, true)]
    [InlineData(3, 1024, 1, false)]
    [InlineData(70, 128, 2, true)]
    public void Q8_WithinActivationQuantBound_AndCloseToF32Kernel(int m, int k, int n, bool zeroSomeScales)
    {
        var (_, _, packed) = RandomTensor(m, k, seed: 100 + m + k, zeroSomeScales: zeroSomeScales);
        var x = RandomX(n, k, seed: 21 + k);
        var w = SpecDecode(packed, m, k);
        var reference = RefGemm(w, x, m, k, n);
        var y = RunQ8(packed, x, m, k, n);
        double worstRatio = 0;
        for (int t = 0; t < n; t++)
            for (int r = 0; r < m; r++)
            {
                double err = Math.Abs(y[t * m + r] - reference[t * m + r]);
                double bound = Q8ErrorBound(w, x, r, t, k) + 1e-5 * Math.Abs(reference[t * m + r]) + 1e-7;
                Assert.True(err <= bound, $"[{t},{r}] err {err} > bound {bound} ({y[t * m + r]} vs {reference[t * m + r]})");
                if (bound > 1e-6) worstRatio = Math.Max(worstRatio, err / bound);
            }
        // The bound is a real constraint, not slack: a correct kernel uses a visible share of it only when activations are coarse; assert the kernel is
        // not accidentally exact-by-fallback (e.g. silently routed to the F32 kernel) on a shape with non-trivial activations.
        if (!TernB64F16.UseF32ActivationReference && k >= 576)
        {
            var f32 = RunF32(packed, x, m, k, n);
            Assert.NotEqual(f32, y);
        }
    }

    [Fact]
    public void Q8_ZeroScaleBlocks_ContributeExactlyZero_EvenWithNonzeroTrits()
    {
        const int m = 2, k = 128;
        var trits = Enumerable.Repeat((sbyte)1, m * k).ToArray();
        float[] scales = [0f, 0f, 0f, 0f];
        var packed = TernB64F16.Pack(trits, scales);
        var x = RandomX(1, k, 5);
        Assert.All(RunQ8(packed, x, m, k, 1), v => Assert.Equal(0f, v));
        Assert.All(RunF32(packed, x, m, k, 1), v => Assert.Equal(0f, v));
        // One live block among zero ones: only that block's columns contribute.
        scales = [0f, 0.5f, 0f, 0f];
        packed = TernB64F16.Pack(trits, scales);
        var xs = new float[k];
        for (int j = 0; j < k; j++) xs[j] = j < 64 ? 100f : 1f;
        var y = RunQ8(packed, xs, m, k, 1);
        Assert.Equal(32f, y[0], 1);   // 64 cols * 1 * 0.5
        Assert.Equal(0f, y[1]);
    }

    [Fact]
    public void Gemm_EqualsPerTokenGemv_BitExact_AndSerialEqualsParallel()
    {
        const int m = 130, k = 576, n = 5;
        var (_, _, packed) = RandomTensor(m, k, seed: 42);
        var x = RandomX(n, k, seed: 43);
        var all = RunQ8(packed, x, m, k, n);
        for (int t = 0; t < n; t++)
        {
            var one = RunQ8(packed, x.AsSpan(t * k, k).ToArray(), m, k, 1);
            Assert.Equal(one, all.AsSpan(t * m, m).ToArray());
        }
        using var pool = new ComputeThreadPool(4);
        Assert.Equal(all, RunQ8(packed, x, m, k, n, pool));
        Assert.Equal(RunF32(packed, x, m, k, n), RunF32(packed, x, m, k, n, pool));
    }

    [Fact]
    public void Q8_And_F32Act_SeeColumnOrderCorrectly_OneHotProbe()
    {
        // One-hot activations expose any column permutation: y[r] must equal w[r, j] for x = e_j (within the Q8 grid, exact for one-hot).
        const int m = 3, k = 192;
        var (_, _, packed) = RandomTensor(m, k, seed: 8);
        var w = SpecDecode(packed, m, k);
        for (int j = 0; j < k; j += 5)
        {
            var x = new float[k]; x[j] = 1f;
            var yq = RunQ8(packed, x, m, k, 1);
            var yf = RunF32(packed, x, m, k, 1);
            for (int r = 0; r < m; r++)
            {
                Assert.Equal(w[r * k + j], yf[r], 6);
                Assert.Equal(w[r * k + j], yq[r], 4);
            }
        }
    }
}
