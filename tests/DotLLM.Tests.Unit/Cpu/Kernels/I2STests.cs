using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Tests for the BitNet I2_S ternary quantization format (ggml type 36).
/// Layout: 4 ternary codes packed per byte (2 bits each), block size 128; codes map
/// 0→-1, 1→0, 2→+1; a single per-tensor float32 scale is stored after the packed data
/// at byte offset <c>n/4</c>.
/// </summary>
public sealed unsafe class I2STests
{
    // ──────────────────── Byte-size accounting ────────────────────

    [Theory]
    [InlineData(128, 32)]      // one block → 32 bytes packed
    [InlineData(2560, 640)]    // BitNet attn row → 640 bytes packed
    [InlineData(6912, 1728)]   // BitNet ffn row → 1728 bytes packed
    public void RowByteSize_IsPackedStride_NoScale(long elementCount, long expected)
    {
        // Per-row stride is the packed 2-bit payload only; the scale is per-tensor, not per-row.
        Assert.Equal(expected, Dequantize.RowByteSize(elementCount, QuantizationType.I2_S));
    }

    [Theory]
    [InlineData(128, 36)]            // 32 packed + 4 scale
    [InlineData(256, 68)]            // 64 packed + 4 scale
    [InlineData(17694720, 4423684)]  // ffn_down whole tensor: n/4 + 4
    public void ComputeByteCount_IncludesPerTensorScale(long elementCount, long expected)
    {
        // Whole-tensor size includes the single trailing float32 scale (used by the GGUF
        // bounds check), so it differs from RowByteSize by exactly 4 bytes.
        Assert.Equal(expected, QuantizationType.I2_S.ComputeByteCount(elementCount));
    }

    // ──────────────────── Bit layout (hand-packed) ────────────────────

    /// <summary>
    /// Pins the exact bit layout independent of any packing helper. Within a 128-element block,
    /// byte at group_pos holds elements {gp, gp+32, gp+64, gp+96} at bit offsets {6,4,2,0}.
    /// Byte 0x92 = 0b10_01_00_10 → codes {2,1,0,2} → ternary {+1,0,-1,+1} for elements {0,32,64,96}.
    /// </summary>
    [Fact]
    public void DequantizeI2S_HandPackedByte_DecodesTernaryTimesScale()
    {
        const int n = 128;
        const float scale = 0.5f;
        // 32 packed bytes + 4-byte scale.
        byte* buf = (byte*)NativeMemory.AllocZeroed((nuint)(n / 4 + 4));
        try
        {
            buf[0] = 0x92;                                  // elements 0,32,64,96
            *(float*)(buf + n / 4) = scale;                 // per-tensor scale at tail

            float[] dest = new float[n];
            Dequantize.ToFloat32((nint)buf, n, QuantizationType.I2_S, dest);

            Assert.Equal(+scale, dest[0], 1e-6f);  // code 2 → +1
            Assert.Equal(0f, dest[32], 1e-6f);     // code 1 →  0
            Assert.Equal(-scale, dest[64], 1e-6f); // code 0 → -1
            Assert.Equal(+scale, dest[96], 1e-6f); // code 2 → +1
            Assert.Equal(-scale, dest[1], 1e-6f);  // unset byte → code 0 → -1
        }
        finally { NativeMemory.Free(buf); }
    }

    [Fact]
    public void DequantizeI2S_RoundTripsTestPacker()
    {
        var rng = new Random(7);
        const int n = 256; // two blocks
        sbyte[] ternary = new sbyte[n];
        for (int i = 0; i < n; i++) ternary[i] = (sbyte)(rng.Next(3) - 1); // {-1,0,1}
        const float scale = 0.0123f;

        byte* buf = PackI2S(ternary, scale);
        try
        {
            float[] dest = new float[n];
            Dequantize.ToFloat32((nint)buf, n, QuantizationType.I2_S, dest);
            for (int i = 0; i < n; i++)
                Assert.Equal(ternary[i] * scale, dest[i], 1e-6f);
        }
        finally { NativeMemory.Free(buf); }
    }

    // ──────────────────── Ternary GEMV ────────────────────

    [Fact]
    public void GemvI2S_MatchesDotOfDequantizedRows()
    {
        var rng = new Random(99);
        const int m = 5;    // output rows
        const int k = 256;  // input dim (2 blocks)
        sbyte[] ternary = new sbyte[m * k];
        for (int i = 0; i < ternary.Length; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        const float scale = 0.05f;

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

        byte* w = PackI2S(ternary, scale);
        try
        {
            float[] y = new float[m];
            fixed (float* xp = x)
            fixed (float* yp = y)
                MatMul.GemvI2_S(w, xp, yp, m, k, null);

            // Reference: dot of (ternary * scale) row with x.
            for (int r = 0; r < m; r++)
            {
                float acc = 0f;
                for (int c = 0; c < k; c++) acc += ternary[r * k + c] * scale * x[c];
                Assert.Equal(acc, y[r], 1e-3f);
            }
        }
        finally { NativeMemory.Free(w); }
    }

    // ──────────────────── W2A8 (int8-activation) path ────────────────────

    /// <summary>
    /// The W2A8 SIMD path (AVX2/AVX-VNNI) quantizes activations to int8 before the dot, so its
    /// result differs from the full-precision float reference by the activation-quant error only.
    /// On non-AVX2 hardware this same entry point falls back to the float path (exact), so the
    /// tolerance below holds on every box. We assert a small absolute+relative tolerance rather
    /// than bit-exactness. Q8_0 is per-32-block absmax int8 (~1/127 relative quant step), so a
    /// 1e-2 absolute / 2% relative envelope comfortably covers k=256 accumulation.
    /// </summary>
    [Fact]
    public void GemvI2S_W2A8_MatchesFloatReference_WithinQuantTolerance()
    {
        var rng = new Random(1234);
        const int m = 6;     // output rows
        const int k = 512;   // input dim (4 I2_S blocks / 16 Q8_0 blocks)
        sbyte[] ternary = new sbyte[m * k];
        for (int i = 0; i < ternary.Length; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        const float scale = 0.037f;

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

        byte* w = PackI2S(ternary, scale);
        try
        {
            float[] y = new float[m];
            fixed (float* xp = x)
            fixed (float* yp = y)
                MatMul.GemvI2_S(w, xp, yp, m, k, null);

            for (int r = 0; r < m; r++)
            {
                float acc = 0f;
                for (int c = 0; c < k; c++) acc += ternary[r * k + c] * scale * x[c];
                AssertWithinQuantTolerance(acc, y[r]);
            }
        }
        finally { NativeMemory.Free(w); }
    }

    /// <summary>
    /// W2A8 GEMM (prefill): each weight row is unpacked once and dotted against all N tokens.
    /// Validates the multi-token output layout C[t*m + r] against the float reference within
    /// activation-quant tolerance (or exactly, on the float-fallback path).
    /// </summary>
    [Fact]
    public void GemmI2S_W2A8_MatchesFloatReference_WithinQuantTolerance()
    {
        var rng = new Random(4321);
        const int m = 5;     // output rows (weight rows)
        const int k = 256;   // input dim
        const int n = 3;     // tokens
        sbyte[] ternary = new sbyte[m * k];
        for (int i = 0; i < ternary.Length; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        const float scale = 0.021f;

        float[] b = new float[n * k];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;

        byte* w = PackI2S(ternary, scale);
        try
        {
            float[] c = new float[n * m];
            fixed (float* bp = b)
            fixed (float* cp = c)
                MatMul.GemmI2_S(w, bp, cp, m, k, n, null);

            for (int t = 0; t < n; t++)
            for (int r = 0; r < m; r++)
            {
                float acc = 0f;
                for (int col = 0; col < k; col++)
                    acc += ternary[r * k + col] * scale * b[t * k + col];
                AssertWithinQuantTolerance(acc, c[t * m + r]);
            }
        }
        finally { NativeMemory.Free(w); }
    }

    // ──────────────────── SIMD unpack parity (decode-bound stage) ────────────────────

    /// <summary>
    /// The AVX2 SIMD ternary unpack (<see cref="MatMul.UnpackRowI8Avx2"/>) must be BIT-EXACT
    /// against the scalar reference (<see cref="MatMul.UnpackRowI8Scalar"/>) for every packed
    /// byte pattern. The packed payload is filled with a deterministic cycling pattern so all
    /// 256 byte values (hence every 4-code quartet combination) are exercised, across several
    /// K sizes (1 block … BitNet ffn-row width). This pins the exact layout the W2A8 int8 dot
    /// consumes, so a divergent SIMD path can never ship silently.
    /// </summary>
    [Theory]
    [InlineData(128)]    // 1 block
    [InlineData(256)]    // 2 blocks
    [InlineData(2560)]   // BitNet attn row (20 blocks)
    [InlineData(6912)]   // BitNet ffn row (54 blocks)
    public void UnpackRowI8_Simd_MatchesScalar_BitExact(int k)
    {
        // The cross-platform Vector256 path runs on every host (hardware-accelerated where AVX2
        // exists, software fallback otherwise) with identical numeric results, so parity verified
        // here (SSE4.2-only) transfers to the AVX2 hardware lowering.
        int rowBytes = k / 4;
        byte* packed = (byte*)NativeMemory.Alloc((nuint)rowBytes);
        sbyte* refBuf = (sbyte*)NativeMemory.Alloc((nuint)k);
        sbyte* simdBuf = (sbyte*)NativeMemory.Alloc((nuint)k);
        try
        {
            // Deterministic pattern; gcd(31,256)=1 so all 256 byte values appear once rowBytes≥256.
            for (int i = 0; i < rowBytes; i++) packed[i] = (byte)((i * 31 + 7) & 0xFF);

            MatMul.UnpackRowI8Scalar(packed, refBuf, k);
            MatMul.UnpackRowI8Simd(packed, simdBuf, k);

            for (int i = 0; i < k; i++)
                Assert.True(refBuf[i] == simdBuf[i],
                    $"mismatch at element {i} (k={k}): scalar={refBuf[i]} simd={simdBuf[i]}");
        }
        finally
        {
            NativeMemory.Free(packed);
            NativeMemory.Free(refBuf);
            NativeMemory.Free(simdBuf);
        }
    }

    private static void AssertWithinQuantTolerance(float expected, float actual)
    {
        // Absolute floor for near-zero sums; relative band for larger magnitudes.
        float tol = 1e-2f + 0.02f * MathF.Abs(expected);
        Assert.True(MathF.Abs(expected - actual) <= tol,
            $"expected {expected}, got {actual}, |Δ|={MathF.Abs(expected - actual)} > tol {tol}");
    }

    /// <summary>
    /// Test-side reference packer for I2_S: ternary {-1,0,+1} → codes {0,1,2}, 4 per byte
    /// in 128-element blocks, followed by a single float32 per-tensor scale. Caller frees.
    /// </summary>
    private static byte* PackI2S(sbyte[] ternary, float scale)
    {
        int n = ternary.Length;
        byte* buf = (byte*)NativeMemory.AllocZeroed((nuint)(n / 4 + 4));
        for (int e = 0; e < n; e++)
        {
            int block = e / 128, j = e % 128;
            int groupIdx = j / 32, groupPos = j % 32;
            int code = ternary[e] + 1; // -1→0, 0→1, +1→2
            buf[block * 32 + groupPos] |= (byte)(code << (6 - 2 * groupIdx));
        }
        *(float*)(buf + n / 4) = scale;
        return buf;
    }
}
