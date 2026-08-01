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

    // ──────────────────── Float reference tier (GemvI2_SScalar) ────────────────────

    /// <summary>
    /// <see cref="MatMul.GemvI2_SScalar"/> is the float tier used as the ground-truth reference by
    /// the F32-in GPU parity tests (CUDA + Vulkan). It must agree with a double-accumulated exact
    /// dot to fp32 rounding — far tighter than the activation-quant envelope the dispatching
    /// <c>GemvI2_S</c> needs — otherwise those parity bounds are calibrated against a moving target.
    /// </summary>
    [Fact]
    public void GemvI2SScalar_MatchesExactDot_ToFp32Rounding()
    {
        var rng = new Random(31337);
        const int m = 6;
        const int k = 512;
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
                MatMul.GemvI2_SScalar(w, xp, yp, m, k);

            for (int r = 0; r < m; r++)
            {
                // Double accumulation, so the reference itself contributes no fp32 error.
                double acc = 0;
                for (int c = 0; c < k; c++) acc += ternary[r * k + c] * (double)x[c];
                Assert.Equal((float)(acc * scale), y[r], 1e-5f);
            }
        }
        finally { NativeMemory.Free(w); }
    }

    /// <summary>
    /// Issue #229 regression guard: <c>GemvI2_S</c> and <see cref="MatMul.GemvI2_SScalar"/> are NOT
    /// interchangeable. On an AVX2 host the dispatching entry takes the W2A8 int8-activation tier,
    /// so it diverges from the float tier by orders of magnitude more than fp32 rounding. The GPU
    /// parity tests were previously pointed at the dispatching entry and so measured that
    /// activation-quant error instead of the kernel divergence they assert on.
    ///
    /// <para>This test <b>discriminates</b>: swapping either call back to the other entry makes it
    /// fail. On a pre-AVX2 host both entries take the float tier and the difference vanishes
    /// legitimately — which is exactly why the defect was invisible on the T5500 — so the
    /// divergence assertion is gated on <see cref="Avx2.IsSupported"/>.</para>
    /// </summary>
    [Fact]
    public void GemvI2S_DispatchingEntry_IsNotTheFloatTier_OnAvx2()
    {
        var rng = new Random(20229);
        const int m = 8;
        const int k = 2560;   // a real BitNet projection width, where the effect is clearly visible
        sbyte[] ternary = new sbyte[m * k];
        for (int i = 0; i < ternary.Length; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        const float scale = 0.024f;

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

        byte* w = PackI2S(ternary, scale);
        try
        {
            float[] dispatched = new float[m];
            float[] floatTier = new float[m];
            fixed (float* xp = x)
            fixed (float* dp = dispatched)
            fixed (float* fp = floatTier)
            {
                MatMul.GemvI2_S(w, xp, dp, m, k, null);
                MatMul.GemvI2_SScalar(w, xp, fp, m, k);
            }

            float maxDiff = 0f;
            for (int r = 0; r < m; r++) maxDiff = MathF.Max(maxDiff, MathF.Abs(dispatched[r] - floatTier[r]));

            // The float tier's own fp32-rounding envelope at this k (cf. the GPU parity bound
            // 5e-7·√k ≈ 2.5e-5). Anything above this is a different algorithm, not rounding.
            const float floatTierEnvelope = 1e-4f;

            if (Avx2.IsSupported)
            {
                Assert.True(maxDiff > floatTierEnvelope,
                    $"expected the dispatching entry to take the W2A8 tier on AVX2 and diverge from " +
                    $"the float tier by more than {floatTierEnvelope}, but max |Δ| was {maxDiff}. " +
                    $"If the W2A8 gate moved, the GPU parity tests' reference choice needs revisiting.");
            }
            else
            {
                Assert.True(maxDiff <= floatTierEnvelope,
                    $"without AVX2 both entries take the float tier, so they should agree; max |Δ| {maxDiff}");
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

    // ──────────────────── Ragged K (k % 128 != 0) — issue #206 ────────────────────
    //
    // Most published I2_S checkpoints have k an exact multiple of 128 (e.g. BitNet-2B-4T:
    // 2560/6912), which is what MatMul.I2S.cs's dedicated 128-block-interleave layout above
    // assumes. At least one real checkpoint (1bitLLM-style bitnet_b1_58-large/-xl: hidden=2048,
    // intermediate=5460, 5460 % 128 == 84) genuinely has a ragged row length on ffn_down. Before
    // this fix, GemvI2_S/GemmI2_S threw ArgumentException for any k % 128 != 0.
    //
    // PackI2S (below) already packs via the flattened-index formula `block = e/128, j = e%128`
    // (matching the real on-disk layout — the block interleave is computed over the tensor's
    // flattened m*k stream, NOT reset per row; see MatMul.I2S.cs's class remarks), so it's the
    // correct reference packer for ragged k too — no changes needed there. Test tensor totals
    // (m*k) are chosen as exact multiples of 128 so PackI2S's zero-alloc buffer sizing
    // (n/4 + 4 bytes) stays exact, matching how every real BitNet GGUF is shaped in practice.

    [Fact]
    public void GemvI2S_RaggedK_SmallSynthetic_MatchesDotOfDequantizedRows()
    {
        var rng = new Random(206);
        const int m = 16;   // 16 rows exercises multiple distinct row-start bit-phases
        const int k = 200;  // NOT a multiple of 128 (200 % 128 == 72) — m*k = 3200 = 25*128
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

            for (int r = 0; r < m; r++)
            {
                float acc = 0f;
                for (int c = 0; c < k; c++) acc += ternary[r * k + c] * scale * x[c];
                Assert.Equal(acc, y[r], 1e-3f);
            }
        }
        finally { NativeMemory.Free(w); }
    }

    /// <summary>Same shape as the "explicit scale" overload used by the indexed-MoE path.</summary>
    [Fact]
    public void GemvI2S_RaggedK_ExplicitScaleOverload_MatchesDotOfDequantizedRows()
    {
        var rng = new Random(207);
        const int m = 16;
        const int k = 200;
        sbyte[] ternary = new sbyte[m * k];
        for (int i = 0; i < ternary.Length; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        const float scale = 0.05f;

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

        // Pack WITHOUT the tail scale (explicit-scale overload reads the packed payload only).
        byte* w = (byte*)NativeMemory.AllocZeroed((nuint)(m * k / 4));
        for (int e = 0; e < ternary.Length; e++)
        {
            int block = e / 128, j = e % 128;
            int groupIdx = j / 32, groupPos = j % 32;
            int code = ternary[e] + 1;
            w[block * 32 + groupPos] |= (byte)(code << (6 - 2 * groupIdx));
        }
        try
        {
            float[] y = new float[m];
            fixed (float* xp = x)
            fixed (float* yp = y)
                MatMul.GemvI2_S(w, xp, yp, m, k, scale, null);

            for (int r = 0; r < m; r++)
            {
                float acc = 0f;
                for (int c = 0; c < k; c++) acc += ternary[r * k + c] * scale * x[c];
                Assert.Equal(acc, y[r], 1e-3f);
            }
        }
        finally { NativeMemory.Free(w); }
    }

    [Fact]
    public void GemmI2S_RaggedK_SmallSynthetic_MatchesDotOfDequantizedRows()
    {
        var rng = new Random(208);
        const int m = 16;
        const int k = 200;
        const int n = 3; // tokens
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
                Assert.Equal(acc, c[t * m + r], 1e-3f);
            }
        }
        finally { NativeMemory.Free(w); }
    }

    /// <summary>
    /// The real bitnet_b1_58-xl shape (hidden=2048, intermediate=5460 → ffn_down k=5460).
    /// m=32 is chosen deliberately: for k=5460, gcd(5460,128)=4, so consecutive row starts cycle
    /// through 128/4=32 distinct bit-phases before repeating (row r starts at flattened bit
    /// r·5460, and r·5460 mod 128 only returns to 0 every 32 rows) — using exactly 32 rows
    /// exercises every one of those phases once, the strongest test this shape admits for
    /// "does the ragged decoder handle a row that doesn't start on a block boundary".
    /// </summary>
    [Fact]
    public void GemvI2S_RaggedK_RealBitNetXlFfnDownShape_MatchesDotOfDequantizedRows()
    {
        var rng = new Random(209);
        const int m = 32;
        const int k = 5460;
        Assert.Equal(84, k % 128); // sanity: pin the real checkpoint's raggedness
        Assert.Equal(0, (m * k) % 128); // sanity: PackI2S's exact-byte-sizing precondition

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
                Assert.Equal(acc, y[r], 1e-2f);
            }
        }
        finally { NativeMemory.Free(w); }
    }

    /// <summary>Same shape via the thread-pool-parallel dispatch path (GemvI2_SRaggedWorker).</summary>
    [Fact]
    public void GemvI2S_RaggedK_WithThreadPool_MatchesDotOfDequantizedRows()
    {
        var rng = new Random(210);
        const int m = 64; // above ParallelMinRows so the pool path actually engages
        const int k = 200;
        sbyte[] ternary = new sbyte[m * k];
        for (int i = 0; i < ternary.Length; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        const float scale = 0.05f;

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

        byte* w = PackI2S(ternary, scale);
        using var pool = new DotLLM.Cpu.Threading.ComputeThreadPool(4);
        try
        {
            float[] y = new float[m];
            fixed (float* xp = x)
            fixed (float* yp = y)
                MatMul.GemvI2_S(w, xp, yp, m, k, pool);

            for (int r = 0; r < m; r++)
            {
                float acc = 0f;
                for (int c = 0; c < k; c++) acc += ternary[r * k + c] * scale * x[c];
                Assert.Equal(acc, y[r], 1e-3f);
            }
        }
        finally { NativeMemory.Free(w); }
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
