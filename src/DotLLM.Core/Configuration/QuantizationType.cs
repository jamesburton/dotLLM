namespace DotLLM.Core.Configuration;

/// <summary>
/// GGUF quantization type identifiers. Values match the GGUF spec.
/// </summary>
public enum QuantizationType
{
    /// <summary>32-bit IEEE float.</summary>
    F32 = 0,

    /// <summary>16-bit IEEE float.</summary>
    F16 = 1,

    /// <summary>4-bit quantization, group size 32, no min.</summary>
    Q4_0 = 2,

    /// <summary>
    /// Brain float (truncated FP32, 1 sign / 8 exponent / 7 mantissa bits) — same exponent
    /// range as F32, narrower mantissa than F16. Common in HuggingFace-trained transformer
    /// checkpoints. GGUF reserves enum value 30 for BF16 (added in llama.cpp ggml-quants);
    /// dotLLM uses the same value so on-disk GGUF BF16 tensors round-trip cleanly.
    /// </summary>
    BF16 = 30,

    /// <summary>4-bit quantization, group size 32, with min.</summary>
    Q4_1 = 3,

    /// <summary>5-bit quantization, group size 32, no min.</summary>
    Q5_0 = 6,

    /// <summary>5-bit quantization, group size 32, with min.</summary>
    Q5_1 = 7,

    /// <summary>8-bit quantization, group size 32.</summary>
    Q8_0 = 8,

    /// <summary>2-bit K-quant, super-block of 256.</summary>
    Q2_K = 10,

    /// <summary>3-bit K-quant, super-block of 256.</summary>
    Q3_K = 11,

    /// <summary>4-bit K-quant, super-block of 256.</summary>
    Q4_K = 12,

    /// <summary>5-bit K-quant, super-block of 256.</summary>
    Q5_K = 13,

    /// <summary>6-bit K-quant, super-block of 256.</summary>
    Q6_K = 14,

    /// <summary>4-bit non-linear importance quantization, block size 32.</summary>
    IQ4_NL = 20,

    /// <summary>4-bit extra-small non-linear importance quantization, super-block of 256.</summary>
    IQ4_XS = 23,

    /// <summary>
    /// 2-bit importance quantization, "extra extra small" (2.0625 bpw). Super-block
    /// of 256 elements stored in 66 bytes: <c>d(Half@0)</c> + <c>qs[32](uint16)@2</c>.
    /// Each pair of <c>uint16</c> values encodes 32 dequantized elements via four
    /// 256-entry codebook lookups + four sign patterns + a 4-bit shared scale.
    /// </summary>
    IQ2_XXS = 16,

    /// <summary>
    /// 2-bit importance quantization, "small" (2.3125 bpw). Super-block of 256
    /// elements stored in 74 bytes. Distinct from <see cref="IQ2_S"/>: this format
    /// uses 9-bit codebook indices packed into <c>uint16 qs[32]</c> with 7-bit sign
    /// codes, and 4-bit scales in <c>scales[8]</c>. dotLLM ships dequant + CPU
    /// fallback only — no dedicated CUDA GEMV today.
    /// </summary>
    IQ2_XS = 17,

    /// <summary>
    /// 2-bit importance quantization, "standard" (2.5625 bpw). Super-block of 256
    /// elements stored in 82 bytes. This is the on-disk format used by both the
    /// <c>MOSTLY_IQ2_S</c> and <c>MOSTLY_IQ2_M</c> file-type recipes — the
    /// "IQ2_M" 11-12 GB GGUFs (e.g. Qwen3.6-A3B-IQ2_M) store their 2-bit tensors as
    /// IQ2_S blocks and use higher-precision K-quant tensors for attention. Block
    /// layout: <c>d(Half)+qs[64]+qh[8]+scales[8]</c>; signs are stored in the upper
    /// half of <c>qs</c> (bytes 32-63), inline grid indices in the lower half.
    /// </summary>
    IQ2_S = 22,

    /// <summary>
    /// 1.5-bit importance quantization (~1.5625 bpw — the smallest GGUF quant
    /// type). Super-block of 256 elements stored in 50 bytes:
    /// <c>d(Half@0) + qs[32]@2 + qh[8](uint16)@34</c>. Each 32-element sub-block
    /// uses one <c>qh</c> uint16 to encode a 3-bit per-block scale (top 3 bits),
    /// a sign-of-delta bit (bit 15), and four 3-bit grid-index high parts (bits
    /// 0..11). The 11-bit grid index per group of 8 elements selects from a
    /// 2048-entry signed-int8 codebook (each entry packs 8 ternary {-1, 0, +1}
    /// values into a uint64). Per-element decode:
    /// <c>y = dl * (grid[j] + delta)</c> with <c>delta = +/-0.125</c>.
    /// </summary>
    IQ1_S = 19,

    /// <summary>
    /// 3-bit importance quantization, "extra extra small" (3.0625 bpw).
    /// Super-block of 256 elements stored in 98 bytes:
    /// <c>d(Half@0) + qs[64]@2 + scales_and_signs[32]@66</c>. The first 64
    /// <c>qs</c> bytes hold 64 unsigned 8-bit indices into the 256-entry
    /// <c>iq3xxs_grid</c> (one grid row = 4 unsigned int8 values). The 32-byte
    /// <c>scales_and_signs</c> tail is 8 little-endian <c>uint32</c>s — one
    /// per 32-element sub-block — packing four 7-bit sign indices (low 28 bits)
    /// plus a 4-bit shared sub-scale (top nibble). Per-pair-of-4 decode:
    /// <c>db = d * (0.5 + s4) * 0.5</c>, sign byte =
    /// <c>ksigns_iq2xs[(aux32 &gt;&gt; 7*l) &amp; 0x7f]</c>.
    /// </summary>
    IQ3_XXS = 18,

    /// <summary>
    /// 3-bit importance quantization, "standard" (3.4375 bpw). Super-block of
    /// 256 elements stored in 110 bytes:
    /// <c>d(Half@0) + qs[64]@2 + qh[8]@66 + signs[32]@74 + scales[4]@106</c>.
    /// 9-bit codebook indices are split: low 8 bits in <c>qs</c>, high 1 bit
    /// in <c>qh</c> (4 bits per sub-block). The 32-byte <c>signs[]</c> table
    /// stores a full 8-bit sign mask per pair (matches IQ2_S — no ksigns
    /// indirection). Sub-blocks come in pairs that share one
    /// <c>scales[ib32/2]</c> byte (low nibble = first sub, high = second);
    /// per-pair scale is <c>db = d * (1 + 2 * sub_scale)</c>.
    /// </summary>
    IQ3_S = 21
}
