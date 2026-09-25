using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// MXFP4 dequantization + matmul tests. Oracle: llama.cpp's
/// <c>dequantize_row_mxfp4</c> / <c>ggml_e8m0_to_fp32_half</c> semantics —
/// block = 1 E8M0 scale byte + 16 nibble bytes, low nibbles → elements 0..15,
/// high nibbles → 16..31, value = kvalues[nibble] * e8m0_half(e) with
/// kvalues = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12}.
/// </summary>
public sealed unsafe class Mxfp4Tests
{
    private static readonly sbyte[] KValues = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

    // ──────────────────── E8M0 scale ────────────────────

    [Theory]
    [InlineData((byte)0, -128)]     // denormal: 2^-128
    [InlineData((byte)1, -127)]     // denormal: 2^-127
    [InlineData((byte)2, -126)]     // 0.5 * 2^-125
    [InlineData((byte)127, -1)]     // 0.5 * 2^0
    [InlineData((byte)128, 0)]      // 0.5 * 2^1 = 1.0
    [InlineData((byte)129, 1)]      // 2.0
    [InlineData((byte)254, 126)]    // 0.5 * 2^127
    public void E8M0ToFloatHalf_MatchesHalvedPowerOfTwo(byte e, int expectedExp)
    {
        float expected = MathF.Pow(2.0f, expectedExp);
        Assert.Equal(expected, Dequantize.E8M0ToFloatHalf(e));
    }

    [Fact]
    public void E8M0ToFloatHalf_IsHalfOfFullE8M0_ForNormalRange()
    {
        // For e >= 2 the half-scale must equal 2^(e-127) / 2 exactly.
        for (int e = 2; e <= 254; e++)
        {
            double full = Math.Pow(2.0, e - 127);
            Assert.Equal((float)(full / 2.0), Dequantize.E8M0ToFloatHalf((byte)e));
        }
    }

    // ──────────────────── Dequantize ────────────────────

    /// <summary>Builds one 17-byte MXFP4 block from a scale byte + 32 4-bit codes.</summary>
    private static byte[] BuildBlock(byte e, byte[] codes)
    {
        Assert.Equal(32, codes.Length);
        byte[] block = new byte[17];
        block[0] = e;
        for (int j = 0; j < 16; j++)
            block[1 + j] = (byte)((codes[j] & 0x0F) | (codes[j + 16] << 4));
        return block;
    }

    [Fact]
    public void DequantizeMxfp4_KnownBlock_AllSixteenCodes()
    {
        // e = 130 → scale = 0.5 * 2^3 = 4.0. Codes 0..15 in elements 0..15,
        // reversed in elements 16..31.
        byte[] codes = new byte[32];
        for (int i = 0; i < 16; i++) { codes[i] = (byte)i; codes[16 + i] = (byte)(15 - i); }
        byte[] block = BuildBlock(130, codes);

        float[] result = new float[32];
        fixed (byte* p = block)
            Dequantize.ToFloat32((nint)p, 32, QuantizationType.MXFP4, result);

        for (int i = 0; i < 32; i++)
            Assert.Equal(KValues[codes[i]] * 4.0f, result[i]);
    }

    [Fact]
    public void DequantizeMxfp4_MultiBlock_PerBlockScales()
    {
        // Two blocks with different scales: e=128 (1.0) and e=126 (0.25).
        byte[] codes = new byte[32];
        for (int i = 0; i < 32; i++) codes[i] = (byte)(i % 16);
        byte[] b0 = BuildBlock(128, codes);
        byte[] b1 = BuildBlock(126, codes);
        byte[] data = new byte[34];
        b0.CopyTo(data, 0);
        b1.CopyTo(data, 17);

        float[] result = new float[64];
        fixed (byte* p = data)
            Dequantize.ToFloat32((nint)p, 64, QuantizationType.MXFP4, result);

        for (int i = 0; i < 32; i++)
        {
            Assert.Equal(KValues[codes[i]] * 1.0f, result[i]);
            Assert.Equal(KValues[codes[i]] * 0.25f, result[32 + i]);
        }
    }

    [Fact]
    public void RowByteSize_Mxfp4_Is17BytesPer32Elements()
    {
        Assert.Equal(17, Dequantize.RowByteSize(32, QuantizationType.MXFP4));
        Assert.Equal(17L * 90, Dequantize.RowByteSize(2880, QuantizationType.MXFP4));
        Assert.Equal(17L * 90, QuantizationType.MXFP4.ComputeByteCount(2880));
    }

    // ──────────────────── VecDot / GEMV ────────────────────

    private static byte[] BuildRandomMxfp4(int elementCount, Random rng)
    {
        int blocks = elementCount / 32;
        byte[] data = new byte[blocks * 17];
        for (int b = 0; b < blocks; b++)
        {
            // Scales near 1.0 (e in [120, 136]) keep values in a sane range.
            data[b * 17] = (byte)rng.Next(120, 137);
            for (int j = 0; j < 16; j++)
                data[b * 17 + 1 + j] = (byte)rng.Next(256);
        }
        return data;
    }

    [Fact]
    public void VecDotMxfp4F32_Avx2_MatchesScalar()
    {
        if (!Avx2.IsSupported || !Ssse3.IsSupported) return;

        var rng = new Random(42);
        const int k = 256;
        byte[] w = BuildRandomMxfp4(k, rng);

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

        fixed (float* xp = x)
        fixed (byte* wp = w)
        {
            float scalar = MatMul.VecDotMxfp4F32Scalar(wp, xp, k / 32);
            float avx2 = MatMul.VecDotMxfp4F32Avx2(wp, xp, k / 32);
            Assert.Equal(scalar, avx2, 3);
        }
    }

    /// <summary>
    /// Issue #275 regression: on <c>main</c> before the fix, <c>GemvMxfp4</c> quantized the
    /// activation to Q8_0 before dotting (mirroring llama.cpp's
    /// <c>ggml_vec_dot_mxfp4_q8_0_generic</c>), which added ~0.4%-per-block noise on top of the
    /// dequantized-dot reference that dotLLM's CUDA/Vulkan MXFP4 path does NOT have (both
    /// backends dequantize to F16 and run a full-precision GEMM/GEMV with no activation
    /// quantization step at all — see <c>native/kernels/dequant_bf16_mxfp4.cu</c> and
    /// <c>CudaKernels.cs</c>'s MXFP4 case). That asymmetry was the entire source of the #256
    /// gate's clean 4x MXFP4 outlier (1.2e-2 logit cosine vs every other type's 1e-5–3e-3).
    /// <see cref="MatMul.GemvMxfp4(byte*, float*, float*, int, int)"/> now dots directly
    /// against the f32 activation, so its output must equal (up to ordinary float
    /// summation-order noise) dot(dequantized MXFP4 row, f32 x) computed in double precision —
    /// a MUCH tighter tolerance than the pre-fix code could pass, since the pre-fix Q8_0
    /// round-trip introduced real per-element error (not just summation-order noise) that this
    /// test's tolerance is tight enough to catch.
    /// </summary>
    [Fact]
    public void GemvMxfp4_MatchesDequantizedDot()
    {
        var rng = new Random(1234);
        const int m = 48, k = 128;

        byte[] w = BuildRandomMxfp4(m * k, rng);
        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] wF32 = new float[m * k];
        fixed (byte* wp = w)
            Dequantize.ToFloat32((nint)wp, m * k, QuantizationType.MXFP4, wF32);

        float[] actual = new float[m];
        fixed (float* xp = x)
        fixed (float* ap = actual)
        fixed (byte* wp = w)
        {
            MatMul.GemvMxfp4(wp, xp, ap, m, k);
        }

        for (int i = 0; i < m; i++)
        {
            double expected = 0;
            double magnitude = 0;
            for (int j = 0; j < k; j++)
            {
                expected += (double)wF32[i * k + j] * x[j];
                magnitude += Math.Abs((double)wF32[i * k + j] * x[j]);
            }
            // Ordinary float summation-order tolerance only — no quantization noise remains.
            double tol = magnitude * 1e-6 + 1e-5;
            Assert.True(Math.Abs(expected - actual[i]) <= tol,
                $"row {i}: expected {expected}, got {actual[i]} (tol {tol})");
        }
    }

    [Fact]
    public void GemmMxfp4_MatchesGemvPerRow()
    {
        var rng = new Random(7);
        const int m = 16, k = 64, n = 3;

        byte[] w = BuildRandomMxfp4(m * k, rng);
        float[] b = new float[n * k];
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] c = new float[n * m];
        float[] expected = new float[n * m];

        fixed (byte* wp = w)
        fixed (float* bp = b)
        fixed (float* cp = c)
        fixed (float* ep = expected)
        {
            MatMul.GemmMxfp4(wp, bp, cp, m, k, n);
            for (int t = 0; t < n; t++)
                MatMul.GemvMxfp4(wp, bp + t * k, ep + t * m, m, k);
        }

        for (int i = 0; i < n * m; i++)
            Assert.Equal(expected[i], c[i], 4);
    }

    /// <summary>
    /// Issue #275 discriminating regression, hand-packed (no randomness): a single MXFP4 block
    /// with known codes/scale dotted against a known integer activation vector. The expected
    /// value is a closed-form sum of small integers — exactly representable in float, no
    /// summation-order ambiguity. Before the #275 fix, <c>GemvMxfp4</c> first quantized
    /// <paramref name="x"/>-equivalent activations to Q8_0 (scale = amax/127 = 32/127 ≈
    /// 0.2520...), which rounds every element to the nearest multiple of that scale and
    /// reproduces a value close to but NOT equal to the true integer dot — e.g. x[0]=1 quantizes
    /// to round(1/0.2520)=4 then dequantizes to 4×0.2520≈1.008, a ~0.8% per-element error that
    /// this test's tight tolerance catches. After the fix (dot directly against f32 x, no Q8_0
    /// round-trip) the result matches the exact expected value up to ordinary float rounding.
    /// </summary>
    [Fact]
    public void GemvMxfp4_HandPackedBlock_MatchesExactIntegerDot()
    {
        // e = 128 -> scale = 0.5 * 2^(128-127) = 1.0.
        // Low nibble j (elements 0..15) = code j; high nibble j (elements 16..31) = code j too,
        // i.e. weight[j] = weight[16+j] = KValues[j].
        byte[] codes = new byte[32];
        for (int j = 0; j < 16; j++) { codes[j] = (byte)j; codes[16 + j] = (byte)j; }
        byte[] block = BuildBlock(128, codes);

        // x[j] = j + 1 for j in 0..31 (integers 1..32) -> exactly representable, and Q8_0's
        // block scale (amax/127 = 32/127) is irrational relative to these integers, guaranteeing
        // the old Q8_0-round-tripped path would NOT reproduce the exact dot.
        float[] x = new float[32];
        for (int j = 0; j < 32; j++) x[j] = j + 1;

        double expected = 0;
        for (int j = 0; j < 16; j++)
        {
            expected += (double)KValues[j] * x[j];
            expected += (double)KValues[j] * x[16 + j];
        }

        float[] result = new float[1];
        fixed (byte* wp = block)
        fixed (float* xp = x)
        fixed (float* rp = result)
        {
            MatMul.GemvMxfp4(wp, xp, rp, 1, 32);
        }

        Assert.Equal(expected, result[0], 3);
    }
}
