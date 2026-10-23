using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// NVFP4 (GGUF type 40, issue #1038) dequantization and NVFP4 x Q8_0 dot tests.
/// </summary>
/// <remarks>
/// <para>Three independent oracles: (1) hand-computed blocks written from the spec in #1038;
/// (2) <b>llama.cpp b11739's own outputs</b>: the <c>Oracle*</c> arrays below were produced by calling
/// <c>ggml_get_type_traits(NVFP4)->to_float</c> and <c>ggml_get_type_traits_cpu(NVFP4)->vec_dot</c> in the
/// shipped <c>ggml-base.dll</c> / <c>ggml-cpu-alderlake.dll</c> through ctypes, on four real blocks of
/// <c>blk.0.ssm_in.weight</c> from NVIDIA's <c>NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf</c> and on two
/// crafted blocks covering the UE4M3 edge bytes (<c>0x00</c>, <c>0x7F</c>, <c>0xFF</c>, <c>0x01</c>) and
/// every nibble; the Q8_0 activation bytes are ggml's own <c>quantize_row_q8_0</c> output, so the dot
/// test does not depend on dotLLM's quantizer; (3) the scalar transcription of
/// <c>ggml_vec_dot_nvfp4_q8_0_generic</c> for the SIMD and GEMM paths.</para>
/// </remarks>
public sealed unsafe class Nvfp4Tests
{
    private static readonly sbyte[] KValues = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

    // ──────────────────── UE4M3 scale ────────────────────

    [Theory]
    [InlineData((byte)0x00, 0.0f)]          // zero
    [InlineData((byte)0x7F, 0.0f)]          // E4M3 NaN pattern decodes as 0 in ggml
    [InlineData((byte)0x7E, 224.0f)]        // max finite 448, halved
    [InlineData((byte)0x01, 0.0009765625f)] // min subnormal 2^-9, halved = 2^-10
    [InlineData((byte)0x07, 0.0068359375f)] // max subnormal 7·2^-9, halved
    [InlineData((byte)0x08, 0.0078125f)]    // min normal 2^-6, halved
    [InlineData((byte)0x38, 0.5f)]          // 1.0 (exp 7), halved
    [InlineData((byte)0x3C, 0.75f)]         // 1.5, halved
    [InlineData((byte)0x62, 20.0f)]         // 1.25·2^5 = 40, halved (the commonest byte in Bolt's ssm_in)
    [InlineData((byte)0x80, 0.0f)]          // bit 7 ignored: same as 0x00
    [InlineData((byte)0xB8, 0.5f)]          // bit 7 ignored: same as 0x38
    [InlineData((byte)0xFF, 240.0f)]        // bit 7 ignored and != 0x7F: exp 15, man 7 = 480, halved
    public void Ue4m3ToFloatHalf_MatchesGgmlForEdgeBytes(byte x, float expected)
    {
        Assert.Equal(expected, Dequantize.Ue4m3ToFloatHalfReference(x));
        Assert.Equal(expected, Dequantize.Ue4m3ToFloatHalf(x));
    }

    [Fact]
    public void Ue4m3Table_MatchesReferenceForAll256Bytes()
    {
        for (int i = 0; i < 256; i++)
            Assert.Equal(Dequantize.Ue4m3ToFloatHalfReference((byte)i), Dequantize.Ue4m3ToFloatHalf((byte)i));
    }

    // ──────────────────── Dequantize ────────────────────

    /// <summary>
    /// Builds one 36-byte block from four scale bytes and 64 codes given in ELEMENT order: element
    /// <c>16s + j</c> (j &lt; 8) goes in the low nibble of byte <c>4 + 8s + j</c>, element <c>16s + 8 + j</c>
    /// in its high nibble (#1038 spec).
    /// </summary>
    private static byte[] BuildBlock(byte[] scales, int[] codes)
    {
        Assert.Equal(4, scales.Length);
        Assert.Equal(64, codes.Length);
        byte[] block = new byte[36];
        scales.CopyTo(block, 0);
        for (int s = 0; s < 4; s++)
            for (int j = 0; j < 8; j++)
                block[4 + 8 * s + j] = (byte)((codes[16 * s + j] & 0xF) | ((codes[16 * s + 8 + j] & 0xF) << 4));
        return block;
    }

    [Fact]
    public void DequantizeNvfp4_HandComputedBlock_ElementOrderAndPerSubBlockScale()
    {
        // Sub-block scales 0.5, 1.0, 0.25, 0 (0x7F) — each sub-block must use its own.
        byte[] scales = [0x38, 0x40, 0x30, 0x7F];
        int[] codes = new int[64];
        for (int i = 0; i < 64; i++) codes[i] = (i * 5 + 1) % 16;
        byte[] block = BuildBlock(scales, codes);

        float[] got = Dequant(block);

        float[] d = [0.5f, 1.0f, 0.25f, 0.0f];
        for (int i = 0; i < 64; i++)
            Assert.Equal(KValues[codes[i]] * d[i / 16], got[i]);

        // Spot values written out by hand: element 0 = code 1 -> 1 * 0.5; element 8 = code 9 -> -1 * 0.5;
        // element 16 = code 1 (81 % 16) -> 1 * 1.0; element 17 = code 6 -> 8 * 1.0.
        Assert.Equal(0.5f, got[0]);
        Assert.Equal(-0.5f, got[8]);
        Assert.Equal(1.0f, got[16]);
        Assert.Equal(8.0f, got[17]);
    }

    [Fact]
    public void DequantizeNvfp4_MatchesLlamaCppToFloat_RealBoltBlocks()
        => AssertBits(OracleRealDequantBits, Dequant(OracleRealBlocks));

    [Fact]
    public void DequantizeNvfp4_MatchesLlamaCppToFloat_EdgeScaleBlocks()
        => AssertBits(OracleCraftedDequantBits, Dequant(OracleCraftedBlocks));

    [Fact]
    public void RowByteSize_Is36BytesPer64Elements()
    {
        Assert.Equal(36, Dequantize.RowByteSize(64, QuantizationType.NVFP4));
        Assert.Equal(4096 / 64 * 36, Dequantize.RowByteSize(4096, QuantizationType.NVFP4));
        Assert.Equal(4096L / 64 * 36 * 18560, QuantizationType.NVFP4.ComputeByteCount(4096L * 18560));
    }

    [Fact]
    public void Dequantize_RejectsElementCountNotMultipleOf64()
    {
        byte[] block = new byte[36];
        Assert.Throws<ArgumentException>(() => DequantRaw(block, 32));
    }

    // ──────────────────── NVFP4 x Q8_0 dot ────────────────────

    [Fact]
    public void VecDotScalar_MatchesLlamaCppVecDot_RealBoltBlocks()
        => AssertDot(OracleRealDotBits, OracleRealBlocks, OracleRealQ8, avx2: false);

    [Fact]
    public void VecDotScalar_MatchesLlamaCppVecDot_EdgeScaleBlocks()
        => AssertDot(OracleCraftedDotBits, OracleCraftedBlocks, OracleCraftedQ8, avx2: false);

    [SkippableFact]
    public void VecDotAvx2_MatchesLlamaCppVecDot_RealBoltBlocks()
    {
        Skip.IfNot(System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported);
        AssertDot(OracleRealDotBits, OracleRealBlocks, OracleRealQ8, avx2: true);
    }

    [SkippableFact]
    public void VecDotAvx2_MatchesLlamaCppVecDot_EdgeScaleBlocks()
    {
        Skip.IfNot(System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported);
        AssertDot(OracleCraftedDotBits, OracleCraftedBlocks, OracleCraftedQ8, avx2: true);
    }

    [SkippableFact]
    public void VecDotAvx2_MatchesScalar_RandomRows()
    {
        Skip.IfNot(System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported);
        var rng = new Random(1038);
        foreach (int blocks in new[] { 1, 2, 3, 64, 128 })
        {
            byte[] w = RandomBlocks(rng, blocks);
            byte[] q8 = Quantize(RandomVector(rng, blocks * 64));
            fixed (byte* pw = w, pq = q8)
            {
                float s = MatMul.VecDotNvfp4Q8_0Scalar(pw, pq, blocks);
                float a = MatMul.VecDotNvfp4Q8_0Avx2(pw, pq, blocks);
                Assert.True(Math.Abs(s - a) <= 1e-5f * Math.Max(1f, Math.Abs(s)), $"blocks {blocks}: scalar {s} vs avx2 {a}");
            }
        }
    }

    [Fact]
    public void VecDot_EqualsDequantizedDotOfDequantizedQ8_WithinRounding()
    {
        // Independent of llama.cpp: sum_i dequant(w)_i * dequant(q8)_i in double.
        var rng = new Random(7);
        const int blocks = 16;
        byte[] w = RandomBlocks(rng, blocks);
        byte[] q8 = Quantize(RandomVector(rng, blocks * 64));
        float[] wf = Dequant(w);
        double expected = 0;
        for (int i = 0; i < blocks * 64; i++)
        {
            int b = i / 32;
            float dy = (float)BitConverter.ToHalf(q8, b * 34);
            expected += (double)wf[i] * dy * (sbyte)q8[b * 34 + 2 + i % 32];
        }

        fixed (byte* pw = w, pq = q8)
        {
            float got = MatMul.VecDotNvfp4Q8_0(pw, pq, blocks);
            Assert.True(Math.Abs(expected - got) <= 1e-5 * Math.Max(1.0, Math.Abs(expected)), $"expected {expected}, got {got}");
        }
    }

    // ──────────────────── GEMM ────────────────────

    [Theory]
    [InlineData(1, false)]
    [InlineData(1, true)]
    [InlineData(5, false)]
    [InlineData(5, true)]
    public void GemmNvfp4_EqualsPerRowVecDot_SerialAndPooled(int n, bool pooled)
    {
        const int m = 70, k = 192; // m not a multiple of the thread count, k = 3 blocks
        var rng = new Random(n * 31 + (pooled ? 1 : 0));
        byte[] w = RandomBlocks(rng, m * (k / 64));
        float[] b = RandomVector(rng, n * k);
        float[] c = new float[n * m];

        using ComputeThreadPool? pool = pooled ? new ComputeThreadPool(3) : null;
        fixed (byte* pw = w)
        fixed (float* pb = b, pc = c)
            MatMul.GemmNvfp4(pw, pb, pc, m, k, n, pool);

        for (int t = 0; t < n; t++)
        {
            byte[] q8 = Quantize(b.AsSpan(t * k, k).ToArray());
            fixed (byte* pw = w, pq = q8)
                for (int row = 0; row < m; row++)
                {
                    float expected = MatMul.VecDotNvfp4Q8_0(pw + row * (k / 64) * 36, pq, k / 64);
                    Assert.Equal(expected, c[t * m + row]);
                }
        }
    }

    [Fact]
    public void GemmNvfp4_RejectsKNotMultipleOf64()
    {
        byte[] w = new byte[36];
        float[] b = new float[32], c = new float[1];
        Assert.Throws<ArgumentException>(() =>
        {
            fixed (byte* pw = w)
            fixed (float* pb = b, pc = c)
                MatMul.GemmNvfp4(pw, pb, pc, 1, 32, 1, null);
        });
    }

    // ──────────────────── helpers ────────────────────

    private static float[] Dequant(byte[] blocks)
    {
        float[] y = new float[blocks.Length / 36 * 64];
        fixed (byte* p = blocks)
            Dequantize.ToFloat32((nint)p, y.Length, QuantizationType.NVFP4, y);
        return y;
    }

    private static void DequantRaw(byte[] blocks, int count)
    {
        float[] y = new float[64];
        fixed (byte* p = blocks)
            Dequantize.ToFloat32((nint)p, count, QuantizationType.NVFP4, y);
    }

    private static void AssertBits(uint[] expectedBits, float[] actual)
    {
        Assert.Equal(expectedBits.Length, actual.Length);
        for (int i = 0; i < actual.Length; i++)
            Assert.True(expectedBits[i] == BitConverter.SingleToUInt32Bits(actual[i]),
                $"element {i}: llama.cpp {BitConverter.UInt32BitsToSingle(expectedBits[i])} (0x{expectedBits[i]:X8}), " +
                $"dotLLM {actual[i]} (0x{BitConverter.SingleToUInt32Bits(actual[i]):X8})");
    }

    private static void AssertDot(uint expectedBits, byte[] blocks, byte[] q8, bool avx2)
    {
        float expected = BitConverter.UInt32BitsToSingle(expectedBits);
        fixed (byte* pw = blocks, pq = q8)
        {
            int n = blocks.Length / 36;
            float got = avx2 ? MatMul.VecDotNvfp4Q8_0Avx2(pw, pq, n) : MatMul.VecDotNvfp4Q8_0Scalar(pw, pq, n);
            // Integer sub-block sums are exact; only float accumulation order differs from ggml's.
            Assert.True(Math.Abs(expected - got) <= 2e-6f * Math.Abs(expected), $"llama.cpp {expected}, dotLLM {got}");
        }
    }

    private static byte[] RandomBlocks(Random rng, int blocks)
    {
        byte[] w = new byte[blocks * 36];
        rng.NextBytes(w);
        for (int b = 0; b < blocks; b++)
            for (int s = 0; s < 4; s++)
                w[b * 36 + s] = (byte)rng.Next(0x20, 0x58); // realistic UE4M3 scales (~0.03 .. 6)
        return w;
    }

    private static float[] RandomVector(Random rng, int n)
    {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() * 2 - 1) * 3f;
        return v;
    }

    private static byte[] Quantize(float[] x)
    {
        byte[] q = new byte[x.Length / 32 * 34];
        fixed (float* px = x)
        fixed (byte* pq = q)
            MatMul.QuantizeF32ToQ8_0(px, pq, x.Length);
        return q;
    }

    // ──────────────────── llama.cpp b11739 oracle (generated, see class remarks) ────────────────────

    // Real: 4 blocks
    private static readonly byte[] OracleRealBlocks =
    [
        0x62, 0x65, 0x64, 0x61, 0xD5, 0xE6, 0x44, 0x10, 0x65, 0xCB, 0xFD, 0x6F, 0x0C, 0x95, 0x7D, 0xC0,
        0x41, 0x4E, 0x29, 0x51, 0x50, 0xBA, 0xFE, 0x2C, 0x66, 0x29, 0x33, 0x95, 0xEE, 0xB5, 0xCF, 0x12,
        0xC2, 0xE7, 0x1D, 0x73, 0x61, 0x62, 0x64, 0x61, 0x49, 0xA6, 0x7F, 0x97, 0xD5, 0x6E, 0x45, 0x96,
        0xCE, 0xDA, 0x64, 0x55, 0xC7, 0x51, 0xD3, 0x96, 0x02, 0x2E, 0x1C, 0x16, 0xDB, 0x4B, 0xE7, 0xB2,
        0xF0, 0xFC, 0xD7, 0x9E, 0x9A, 0x92, 0xC6, 0x49, 0x62, 0x64, 0x5F, 0x62, 0x6F, 0x65, 0xDC, 0xCC,
        0xF1, 0x66, 0x6C, 0x3B, 0xFD, 0xE3, 0x0E, 0x04, 0x4B, 0x51, 0x2D, 0x04, 0x6C, 0x35, 0x56, 0x55,
        0xD7, 0x69, 0x50, 0xF3, 0xA3, 0x27, 0x6D, 0x04, 0xE0, 0xCA, 0xFD, 0xAE, 0x61, 0x63, 0x65, 0x63,
        0xAE, 0x71, 0xA5, 0x3E, 0x57, 0x93, 0xC1, 0x75, 0xA0, 0x24, 0x03, 0x60, 0x37, 0xD6, 0x2D, 0x5D,
        0x32, 0x3E, 0xCC, 0xF4, 0x49, 0x90, 0xCC, 0x94, 0x35, 0xBF, 0x31, 0xB6, 0x55, 0xD9, 0xA9, 0x2F,
    ];

    private static readonly uint[] OracleRealDequantBits =
    [
        0x42F00000u, 0x43200000u, 0x42A00000u, 0x00000000u, 0x42F00000u, 0xC2700000u, 0xC2F00000u, 0xC3700000u,
        0xC2F00000u, 0xC3200000u, 0x42A00000u, 0x41A00000u, 0x43200000u, 0xC2A00000u, 0xC3700000u, 0x43200000u,
        0xC2D00000u, 0x431C0000u, 0xC31C0000u, 0x00000000u, 0x41D00000u, 0xC3500000u, 0xC1D00000u, 0x41D00000u,
        0x00000000u, 0xC1D00000u, 0x439C0000u, 0xC2D00000u, 0x42D00000u, 0x42D00000u, 0x42500000u, 0x431C0000u,
        0x00000000u, 0xC2400000u, 0xC3400000u, 0xC2C00000u, 0x43400000u, 0xC1C00000u, 0x42900000u, 0x43100000u,
        0x43100000u, 0xC2900000u, 0xC3900000u, 0x42400000u, 0x43400000u, 0x42400000u, 0x42900000u, 0xC1C00000u,
        0xC3100000u, 0x42D80000u, 0xC3580000u, 0x42100000u, 0x42100000u, 0x43580000u, 0xC2D80000u, 0x42580000u,
        0xC3100000u, 0xC2580000u, 0xC2900000u, 0x41900000u, 0xC2900000u, 0xC3100000u, 0x41900000u, 0x43580000u,
        0xC1900000u, 0x43100000u, 0xC3580000u, 0x43580000u, 0x42D80000u, 0xC3100000u, 0x42D80000u, 0x43100000u,
        0x42900000u, 0xC2100000u, 0x43580000u, 0xC1900000u, 0xC2D80000u, 0x43100000u, 0x42900000u, 0xC1900000u,
        0xC3200000u, 0xC2200000u, 0x42A00000u, 0x42F00000u, 0x43700000u, 0x41A00000u, 0x42700000u, 0x43200000u,
        0xC2A00000u, 0xC2F00000u, 0x43200000u, 0x42F00000u, 0xC2A00000u, 0x42F00000u, 0xC2F00000u, 0xC1A00000u,
        0x42400000u, 0xC3400000u, 0xC2C00000u, 0x43400000u, 0xC2900000u, 0xC2900000u, 0x43900000u, 0x42400000u,
        0x00000000u, 0x42400000u, 0x41C00000u, 0x41C00000u, 0xC3100000u, 0x42C00000u, 0xC3400000u, 0xC2900000u,
        0x00000000u, 0xC2900000u, 0x43580000u, 0xC3100000u, 0xC2100000u, 0x42100000u, 0x43100000u, 0xC1900000u,
        0xC3580000u, 0xC3580000u, 0xC2D80000u, 0xC1900000u, 0xC1900000u, 0xC1900000u, 0xC2900000u, 0x42900000u,
        0xC3700000u, 0x42F00000u, 0xC2A00000u, 0xC2A00000u, 0x41A00000u, 0x43200000u, 0xC2A00000u, 0xC2700000u,
        0x43200000u, 0x43200000u, 0xC2F00000u, 0xC2A00000u, 0xC3700000u, 0x43200000u, 0x43200000u, 0x42700000u,
        0xC3100000u, 0x42900000u, 0xC3400000u, 0x42C00000u, 0xC2900000u, 0x41C00000u, 0xC3100000u, 0x42C00000u,
        0xC3900000u, 0xC3400000u, 0x00000000u, 0x00000000u, 0x42C00000u, 0x43100000u, 0x42400000u, 0x00000000u,
        0xC2700000u, 0x42B40000u, 0x42F00000u, 0x42B40000u, 0x43340000u, 0xC1700000u, 0x00000000u, 0x42340000u,
        0x42F00000u, 0x42340000u, 0x42B40000u, 0x42B40000u, 0xC2B40000u, 0x42F00000u, 0x42B40000u, 0xC3340000u,
        0x42700000u, 0x43700000u, 0xC2F00000u, 0x42A00000u, 0x00000000u, 0xC2200000u, 0xC2F00000u, 0xC3200000u,
        0xC2200000u, 0x42200000u, 0x43200000u, 0x00000000u, 0xC3200000u, 0xC2A00000u, 0xC3700000u, 0xC2200000u,
        0xC3100000u, 0x41900000u, 0x42D80000u, 0xC3100000u, 0x43580000u, 0x42580000u, 0x41900000u, 0x42D80000u,
        0xC2100000u, 0x43580000u, 0xC2100000u, 0x42580000u, 0x42D80000u, 0xC1900000u, 0xC2900000u, 0x43580000u,
        0x00000000u, 0x42B00000u, 0x42840000u, 0x00000000u, 0x43840000u, 0x43300000u, 0xC3040000u, 0xC3040000u,
        0xC2300000u, 0x42300000u, 0x00000000u, 0x43300000u, 0x42840000u, 0xC3040000u, 0x42300000u, 0x43040000u,
        0x42500000u, 0xC3500000u, 0xC2D00000u, 0x42D00000u, 0xC1D00000u, 0x00000000u, 0xC2D00000u, 0x42D00000u,
        0x429C0000u, 0x429C0000u, 0xC2D00000u, 0xC39C0000u, 0x42D00000u, 0xC1D00000u, 0xC2D00000u, 0xC1D00000u,
        0x43040000u, 0xC3840000u, 0x41B00000u, 0x43300000u, 0x43040000u, 0xC1B00000u, 0xC1B00000u, 0xC3840000u,
        0x42840000u, 0xC2840000u, 0x42840000u, 0xC2840000u, 0x43040000u, 0xC3040000u, 0xC2300000u, 0x42300000u,
    ];

    private static readonly byte[] OracleRealQ8 =
    [
        0xEC, 0x25, 0xF3, 0x1D, 0x30, 0x3D, 0x41, 0x88, 0x2F, 0x1C, 0x05, 0xEE, 0x4F, 0xC8, 0xC0, 0xC0,
        0xC9, 0x4D, 0xEF, 0x07, 0x1E, 0x31, 0x86, 0x41, 0x3C, 0x2F, 0x1C, 0xF6, 0xED, 0xD8, 0xC8, 0xC0,
        0x7F, 0xCA, 0x0B, 0x26, 0xDB, 0xF0, 0x07, 0xC5, 0x30, 0x3C, 0x40, 0x3B, 0xA4, 0x1B, 0x04, 0xED,
        0xD8, 0x6E, 0xC1, 0xC2, 0xCB, 0xDB, 0x1F, 0x08, 0x1E, 0x30, 0x3C, 0x81, 0x3B, 0x2E, 0x1B, 0x04,
        0x27, 0xD8, 0xC9, 0xC1, 0xFC, 0x25, 0xC1, 0x6B, 0xDB, 0xF1, 0x08, 0x1F, 0x9E, 0x3D, 0x40, 0x3B,
        0x2E, 0xCB, 0x03, 0xEC, 0xD7, 0xC8, 0x7F, 0xC1, 0xCB, 0xDB, 0xF1, 0xEF, 0x1F, 0x31, 0x3D, 0x40,
        0x8A, 0x2D, 0x1A, 0x03, 0xEC, 0x52, 0xE0, 0x25, 0xC7, 0xBF, 0xC0, 0xCA, 0x4A, 0xF1, 0x09, 0x20,
        0x32, 0x84, 0x41, 0x3C, 0x2E, 0x1A, 0xFB, 0xEB, 0xD6, 0xC7, 0xBF, 0x7F, 0xCA, 0xDC, 0xF2, 0x0A,
        0xBF, 0x33, 0x3E, 0x41, 0x3B, 0xA5, 0x1A, 0x02, 0x09, 0x26, 0xEB, 0xD7, 0x70, 0xC1, 0xC2, 0xCC,
        0xDD, 0x1B, 0x0A, 0x20, 0x32, 0x3D, 0x81, 0x3A, 0x2C, 0x19, 0x02, 0x2B, 0xD6, 0xC8, 0xC1, 0xC2,
        0x67, 0xDD, 0xF3, 0x0A, 0x20, 0x9C, 0x3D, 0x3F, 0x3A, 0x2C, 0x03, 0x26, 0xCF, 0x01, 0xEA, 0xD6,
        0xC7, 0x7F, 0xC2, 0xCC, 0xDD, 0xF3, 0xEB, 0x21, 0x32, 0x3D, 0x40, 0x8D, 0x2C, 0x18, 0x01, 0xEA,
        0x55, 0xC7, 0xC0, 0xC2, 0xCD, 0x44, 0xF4, 0x0B, 0x21, 0x33, 0x85, 0x40, 0xD3, 0x25, 0x3B, 0x2D,
        0x18, 0xFF, 0xE9, 0xD4, 0xC5, 0xBE, 0x7F, 0xCB, 0xDD, 0xF4, 0x0C, 0xBB, 0x35, 0x3F, 0x42, 0x3B,
        0xA7, 0x18, 0x00, 0xE8, 0xD4, 0x76, 0xBE, 0xC1, 0xCB, 0xDD, 0x18, 0x0C, 0x23, 0x35, 0x08, 0x26,
        0x3D, 0x81, 0x39, 0x2B, 0x17, 0x00, 0x2F, 0xD5, 0xC7, 0xC1, 0xC3, 0x65, 0xDF, 0xF5, 0x0C, 0x22,
        0x99, 0x3D, 0x3F, 0x39, 0x2A, 0xD4, 0xFF, 0xE8, 0xD4, 0xC7, 0x7F, 0xC3, 0xCE, 0xDF, 0xF5, 0xE6,
    ];

    private const uint OracleRealDotBits = 0xC4520F22u; // -840.2364501953125

    // Crafted: 2 blocks
    private static readonly byte[] OracleCraftedBlocks =
    [
        0x00, 0x7F, 0xFF, 0x01, 0x03, 0x0A, 0x11, 0x18, 0x1F, 0x26, 0x2D, 0x34, 0x3B, 0x42, 0x49, 0x50,
        0x57, 0x5E, 0x65, 0x6C, 0x73, 0x7A, 0x81, 0x88, 0x8F, 0x96, 0x9D, 0xA4, 0xAB, 0xB2, 0xB9, 0xC0,
        0xC7, 0xCE, 0xD5, 0xDC, 0x38, 0x40, 0x30, 0x7E, 0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87,
        0x78, 0x69, 0x5A, 0x4B, 0x3C, 0x2D, 0x1E, 0x0F, 0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87,
        0x78, 0x69, 0x5A, 0x4B, 0x3C, 0x2D, 0x1E, 0x0F,
    ];

    private static readonly uint[] OracleCraftedDequantBits =
    [
        0x00000000u, 0x80000000u, 0x00000000u, 0x00000000u, 0x80000000u, 0x00000000u, 0x80000000u, 0x00000000u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
        0x80000000u, 0x00000000u, 0x80000000u, 0x00000000u, 0x00000000u, 0x80000000u, 0x00000000u, 0x80000000u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
        0x44340000u, 0xC3F00000u, 0x43700000u, 0x00000000u, 0xC5340000u, 0x44F00000u, 0xC4B40000u, 0x44700000u,
        0x45340000u, 0x45340000u, 0x00000000u, 0x00000000u, 0x00000000u, 0xC3700000u, 0xC3700000u, 0xC3F00000u,
        0xBB400000u, 0x3B000000u, 0xBA800000u, 0x00000000u, 0x3C400000u, 0xBC000000u, 0x3BC00000u, 0xBB800000u,
        0xBB000000u, 0xBB400000u, 0xBB400000u, 0xBB800000u, 0xBB800000u, 0xBB800000u, 0xBBC00000u, 0xBBC00000u,
        0x00000000u, 0x3F000000u, 0x3F800000u, 0x3FC00000u, 0x40000000u, 0x40400000u, 0x40800000u, 0x40C00000u,
        0xC0C00000u, 0xC0800000u, 0xC0400000u, 0xC0000000u, 0xBFC00000u, 0xBF800000u, 0xBF000000u, 0x00000000u,
        0x00000000u, 0xBF800000u, 0xC0000000u, 0xC0400000u, 0xC0800000u, 0xC0C00000u, 0xC1000000u, 0xC1400000u,
        0x41400000u, 0x41000000u, 0x40C00000u, 0x40800000u, 0x40400000u, 0x40000000u, 0x3F800000u, 0x00000000u,
        0x00000000u, 0x3E800000u, 0x3F000000u, 0x3F400000u, 0x3F800000u, 0x3FC00000u, 0x40000000u, 0x40400000u,
        0xC0400000u, 0xC0000000u, 0xBFC00000u, 0xBF800000u, 0xBF400000u, 0xBF000000u, 0xBE800000u, 0x00000000u,
        0x00000000u, 0xC3600000u, 0xC3E00000u, 0xC4280000u, 0xC4600000u, 0xC4A80000u, 0xC4E00000u, 0xC5280000u,
        0x45280000u, 0x44E00000u, 0x44A80000u, 0x44600000u, 0x44280000u, 0x43E00000u, 0x43600000u, 0x00000000u,
    ];

    private static readonly byte[] OracleCraftedQ8 =
    [
        0xEC, 0x25, 0xF3, 0x1D, 0x30, 0x3D, 0x41, 0x88, 0x2F, 0x1C, 0x05, 0xEE, 0x4F, 0xC8, 0xC0, 0xC0,
        0xC9, 0x4D, 0xEF, 0x07, 0x1E, 0x31, 0x86, 0x41, 0x3C, 0x2F, 0x1C, 0xF6, 0xED, 0xD8, 0xC8, 0xC0,
        0x7F, 0xCA, 0x0B, 0x26, 0xDB, 0xF0, 0x07, 0xC5, 0x30, 0x3C, 0x40, 0x3B, 0xA4, 0x1B, 0x04, 0xED,
        0xD8, 0x6E, 0xC1, 0xC2, 0xCB, 0xDB, 0x1F, 0x08, 0x1E, 0x30, 0x3C, 0x81, 0x3B, 0x2E, 0x1B, 0x04,
        0x27, 0xD8, 0xC9, 0xC1, 0xFC, 0x25, 0xC1, 0x6B, 0xDB, 0xF1, 0x08, 0x1F, 0x9E, 0x3D, 0x40, 0x3B,
        0x2E, 0xCB, 0x03, 0xEC, 0xD7, 0xC8, 0x7F, 0xC1, 0xCB, 0xDB, 0xF1, 0xEF, 0x1F, 0x31, 0x3D, 0x40,
        0x8A, 0x2D, 0x1A, 0x03, 0xEC, 0x52, 0xE0, 0x25, 0xC7, 0xBF, 0xC0, 0xCA, 0x4A, 0xF1, 0x09, 0x20,
        0x32, 0x84, 0x41, 0x3C, 0x2E, 0x1A, 0xFB, 0xEB, 0xD6, 0xC7, 0xBF, 0x7F, 0xCA, 0xDC, 0xF2, 0x0A,
        0xBF, 0x33, 0x3E, 0x41, 0x3B, 0xA5, 0x1A, 0x02,
    ];

    private const uint OracleCraftedDotBits = 0xC544E46Eu; // -3150.27685546875
}
