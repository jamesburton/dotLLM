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
    IQ2_S = 22
}
