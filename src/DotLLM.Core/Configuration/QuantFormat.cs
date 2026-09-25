namespace DotLLM.Core.Configuration;

/// <summary>
/// Single source of truth for GGUF quantization block geometry (issue #351).
/// Every backend's C# code must reference these constants instead of declaring
/// its own copy — before this table, <c>Q8_0BlockBytes = 34</c> was declared
/// independently 25 times across CPU/CUDA/Vulkan, the exact duplication shape
/// that let the transposed Q3_K decode (#311) ship certified by parity between
/// two equally-wrong backends.
/// </summary>
/// <remarks>
/// <para>
/// Values are anchored to llama.cpp's block structs in <c>ggml/src/ggml-common.h</c>
/// (block bytes = <c>sizeof(block_*)</c>, group size = <c>QK*</c>); the anchor test
/// (<c>QuantFormatTests</c>) restates each layout field-by-field so the single
/// source is itself pinned, not merely single.
/// </para>
/// <para>
/// Everything here is compile-time <c>const</c> or a branch-free <c>switch</c> —
/// no allocation, no virtual dispatch, safe on any hot path. GLSL/CUDA kernel
/// <i>sources</i> necessarily keep their own literals (they cannot consume C#);
/// their compiled wrappers reference these constants.
/// </para>
/// </remarks>
public static class QuantFormat
{
    // ── Legacy 32-element block formats (llama.cpp QK4_0 … QK8_1 = 32) ──
    /// <summary>Elements per block for all legacy 32-group formats.</summary>
    public const int LegacyGroupSize = 32;
    /// <summary>block_q4_0: d(f16) + 16 nibble bytes.</summary>
    public const int Q4_0BlockBytes = 18;
    /// <summary>block_q4_1: d(f16) + m(f16) + 16 nibble bytes.</summary>
    public const int Q4_1BlockBytes = 20;
    /// <summary>block_q5_0: d(f16) + qh(4) + 16 nibble bytes.</summary>
    public const int Q5_0BlockBytes = 22;
    /// <summary>block_q5_1: d(f16) + m(f16) + qh(4) + 16 nibble bytes.</summary>
    public const int Q5_1BlockBytes = 24;
    /// <summary>block_q8_0: d(f16) + 32 int8.</summary>
    public const int Q8_0BlockBytes = 34;
    /// <summary>block_q8_1: d(f16) + s(f16) + 32 int8. Activation-side format.</summary>
    public const int Q8_1BlockBytes = 36;
    /// <summary>block_mxfp4: E8M0 scale byte + 16 nibble bytes.</summary>
    public const int Mxfp4BlockBytes = 17;
    /// <summary>block_iq4_nl: d(f16) + 16 nibble bytes (non-linear codebook).</summary>
    public const int IQ4_NLBlockBytes = 18;

    // ── K-quant / IQ 256-element super-block formats (llama.cpp QK_K = 256) ──
    /// <summary>Elements per super-block for all K-quant and IQ formats.</summary>
    public const int KQuantGroupSize = 256;
    /// <summary>block_q2_K: scales[16] + qs[64] + d(f16) + dmin(f16).</summary>
    public const int Q2_KBlockBytes = 84;
    /// <summary>block_q3_K: hmask[32] + qs[64] + scales[12] + d(f16).</summary>
    public const int Q3_KBlockBytes = 110;
    /// <summary>block_q4_K: d(f16) + dmin(f16) + scales[12] + qs[128].</summary>
    public const int Q4_KBlockBytes = 144;
    /// <summary>block_q5_K: d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128].</summary>
    public const int Q5_KBlockBytes = 176;
    /// <summary>block_q6_K: ql[128] + qh[64] + scales[16] + d(f16).</summary>
    public const int Q6_KBlockBytes = 210;
    /// <summary>block_q8_K: d(f32) + qs[256] + bsums[16×i16]. Activation-side format.</summary>
    public const int Q8_KBlockBytes = 292;
    /// <summary>block_iq1_s: d(f16) + qs[32] + qh[16×i16... packed 8×u16].</summary>
    public const int IQ1_SBlockBytes = 50;
    /// <summary>block_iq2_xxs: d(f16) + qs[32×u16].</summary>
    public const int IQ2_XXSBlockBytes = 66;
    /// <summary>block_iq2_xs: d(f16) + qs[32×u16] + scales[8].</summary>
    public const int IQ2_XSBlockBytes = 74;
    /// <summary>block_iq2_s: d(f16) + qs[32] + qh[8] + scales[8] + high bits[32].</summary>
    public const int IQ2_SBlockBytes = 82;
    /// <summary>block_iq3_xxs: d(f16) + qs[96].</summary>
    public const int IQ3_XXSBlockBytes = 98;
    /// <summary>block_iq3_s: d(f16) + qs[64] + qh[8] + signs[32] + scales[4].</summary>
    public const int IQ3_SBlockBytes = 110;
    /// <summary>block_iq4_xs: d(f16) + scales_h(u16) + scales_l[4] + qs[128].</summary>
    public const int IQ4_XSBlockBytes = 136;

    // ── Ternary / 2-bit 128-element group formats ──
    /// <summary>Elements per group for the 2-bit 128-code packings (I2_S, PQ2_0).</summary>
    public const int TernaryGroupSize = 128;
    /// <summary>
    /// I2_S: 128 2-bit codes packed 4/byte = 32 bytes. The scale is NOT in the
    /// block — one f32 per tensor at the tensor tail (offset m·K/4), so row
    /// strides use these 32 bytes only.
    /// </summary>
    public const int I2_SBlockBytes = 32;
    /// <summary>PQ2_0: scale(f16) + 32 packed code bytes; scale is per-group, in-block.</summary>
    public const int PQ2_0BlockBytes = 34;

    /// <summary>
    /// Static geometry for one quantization format. A <c>readonly record struct</c>
    /// returned by value from a const-folding switch — no table allocation.
    /// </summary>
    /// <param name="Type">The format.</param>
    /// <param name="BlockBytes">Bytes per block (packed, as stored in GGUF).</param>
    /// <param name="GroupSize">Elements per block.</param>
    /// <param name="HasMin">
    /// True when the block carries a min/offset term (<c>d·q + m</c> family:
    /// Q4_1, Q5_1, Q8_1's sum term, Q2_K/Q4_K/Q5_K's dmin); false for the
    /// symmetric <c>d·q</c>/<c>d·(q−z)</c> family.
    /// </param>
    public readonly record struct Info(QuantizationType Type, int BlockBytes, int GroupSize, bool HasMin)
    {
        /// <summary>Effective stored bits per weight (block bytes ÷ elements × 8).</summary>
        public double BitsPerWeight => BlockBytes * 8.0 / GroupSize;

        /// <summary>
        /// Byte size of one row of <paramref name="elementCount"/> elements.
        /// <paramref name="elementCount"/> must be a multiple of <see cref="GroupSize"/>.
        /// </summary>
        public long RowByteSize(long elementCount) => elementCount / GroupSize * BlockBytes;
    }

    /// <summary>
    /// Returns the block geometry for <paramref name="type"/>, or null for
    /// non-block formats (F32/F16/BF16) and unknown values. F32/F16/BF16 are
    /// deliberately excluded: they have no block structure, and callers that
    /// treat "bytes per element" uniformly hide exactly the packed/unpacked
    /// distinction this table exists to make explicit.
    /// </summary>
    public static Info? TryGetInfo(QuantizationType type) => type switch
    {
        QuantizationType.Q4_0 => new Info(type, Q4_0BlockBytes, LegacyGroupSize, HasMin: false),
        QuantizationType.Q4_1 => new Info(type, Q4_1BlockBytes, LegacyGroupSize, HasMin: true),
        QuantizationType.Q5_0 => new Info(type, Q5_0BlockBytes, LegacyGroupSize, HasMin: false),
        QuantizationType.Q5_1 => new Info(type, Q5_1BlockBytes, LegacyGroupSize, HasMin: true),
        QuantizationType.Q8_0 => new Info(type, Q8_0BlockBytes, LegacyGroupSize, HasMin: false),
        QuantizationType.MXFP4 => new Info(type, Mxfp4BlockBytes, LegacyGroupSize, HasMin: false),
        QuantizationType.IQ4_NL => new Info(type, IQ4_NLBlockBytes, LegacyGroupSize, HasMin: false),
        QuantizationType.Q2_K => new Info(type, Q2_KBlockBytes, KQuantGroupSize, HasMin: true),
        QuantizationType.Q3_K => new Info(type, Q3_KBlockBytes, KQuantGroupSize, HasMin: false),
        QuantizationType.Q4_K => new Info(type, Q4_KBlockBytes, KQuantGroupSize, HasMin: true),
        QuantizationType.Q5_K => new Info(type, Q5_KBlockBytes, KQuantGroupSize, HasMin: true),
        QuantizationType.Q6_K => new Info(type, Q6_KBlockBytes, KQuantGroupSize, HasMin: false),
        QuantizationType.IQ1_S => new Info(type, IQ1_SBlockBytes, KQuantGroupSize, HasMin: false),
        QuantizationType.IQ2_XXS => new Info(type, IQ2_XXSBlockBytes, KQuantGroupSize, HasMin: false),
        QuantizationType.IQ2_XS => new Info(type, IQ2_XSBlockBytes, KQuantGroupSize, HasMin: false),
        QuantizationType.IQ2_S => new Info(type, IQ2_SBlockBytes, KQuantGroupSize, HasMin: false),
        QuantizationType.IQ3_XXS => new Info(type, IQ3_XXSBlockBytes, KQuantGroupSize, HasMin: false),
        QuantizationType.IQ3_S => new Info(type, IQ3_SBlockBytes, KQuantGroupSize, HasMin: false),
        QuantizationType.IQ4_XS => new Info(type, IQ4_XSBlockBytes, KQuantGroupSize, HasMin: false),
        QuantizationType.I2_S => new Info(type, I2_SBlockBytes, TernaryGroupSize, HasMin: false),
        QuantizationType.PQ2_0 => new Info(type, PQ2_0BlockBytes, TernaryGroupSize, HasMin: false),
        _ => null,
    };

    /// <summary>
    /// All block-quantized formats the table describes, for exhaustive tests
    /// and diagnostics sweeps.
    /// </summary>
    public static ReadOnlySpan<QuantizationType> BlockFormats =>
    [
        QuantizationType.Q4_0, QuantizationType.Q4_1, QuantizationType.Q5_0,
        QuantizationType.Q5_1, QuantizationType.Q8_0, QuantizationType.MXFP4,
        QuantizationType.IQ4_NL,
        QuantizationType.Q2_K, QuantizationType.Q3_K, QuantizationType.Q4_K,
        QuantizationType.Q5_K, QuantizationType.Q6_K,
        QuantizationType.IQ1_S, QuantizationType.IQ2_XXS, QuantizationType.IQ2_XS,
        QuantizationType.IQ2_S, QuantizationType.IQ3_XXS, QuantizationType.IQ3_S,
        QuantizationType.IQ4_XS,
        QuantizationType.I2_S, QuantizationType.PQ2_0,
    ];
}
