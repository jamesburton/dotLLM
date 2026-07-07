using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// BitNet b1.58 weight quantization: converts full-precision (f32, typically up-cast from the HF
/// bf16 checkpoint) weights into the I2_S ternary packed format consumed by
/// <see cref="Dequantize.DequantizeI2_S"/>.
/// <para>
/// Mirrors transformers' per-tensor absmean <c>WeightQuant</c>: <c>scale = mean(|w|)</c> (clamped
/// ≥ 1e-5), ternary <c>t = clamp(round(w / scale), -1, 1)</c>, dequantized value <c>t · scale</c>.
/// Packing is the exact inverse of the dequant kernel: 128-element blocks → 32 bytes; within a
/// block the byte at <c>gp</c> (0..31) holds elements {gp, +32, +64, +96} at bit offsets {6,4,2,0};
/// the 2-bit code is <c>t + 1</c> (so decode <c>(code - 1) · scale</c>). A single f32 per-tensor
/// scale is appended at byte offset <c>count / 4</c>.
/// </para>
/// </summary>
public static class BitNetQuantize
{
    /// <summary>
    /// Quantizes <paramref name="count"/> row-major f32 weights into I2_S packed bytes written to
    /// <paramref name="dest"/>. <paramref name="count"/> must be a multiple of 128;
    /// <paramref name="dest"/> length must be at least <c>count / 4 + sizeof(float)</c>.
    /// </summary>
    /// <param name="weights">Source f32 weights (length ≥ <paramref name="count"/>).</param>
    /// <param name="count">Number of weights to quantize (multiple of 128).</param>
    /// <param name="dest">Destination for packed codes (count/4 bytes) + trailing f32 scale.</param>
    public static void QuantizeToI2S(ReadOnlySpan<float> weights, long count, Span<byte> dest)
    {
        if (count % Dequantize.I2SBlockSize != 0)
            throw new ArgumentException(
                $"I2_S element count must be a multiple of {Dequantize.I2SBlockSize}, got {count}",
                nameof(count));
        if (weights.Length < count)
            throw new ArgumentException($"weights too small: {weights.Length} < {count}", nameof(weights));

        int n = checked((int)count);
        int packedBytes = n / 4;
        if (dest.Length < packedBytes + sizeof(float))
            throw new ArgumentException(
                $"dest too small: {dest.Length} < {packedBytes + sizeof(float)}", nameof(dest));

        // Per-tensor absmean scale (clamped like transformers WeightQuant).
        double sum = 0;
        for (int i = 0; i < n; i++) sum += Math.Abs(weights[i]);
        float scale = (float)Math.Max(sum / count, 1e-5);

        dest[..packedBytes].Clear(); // codes are OR-ed in, so start from zero

        for (int i = 0; i < n; i++)
        {
            int ternary = Math.Clamp((int)MathF.Round(weights[i] / scale), -1, 1);
            int code = ternary + 1; // {-1,0,+1} → {0,1,2}
            int block = i / Dequantize.I2SBlockSize;
            int p = i % Dequantize.I2SBlockSize;
            int gp = p % 32;
            int slot = p / 32;            // 0..3
            int bitOffset = 6 - 2 * slot; // {6,4,2,0}
            dest[block * 32 + gp] |= (byte)(code << bitOffset);
        }

        Unsafe.WriteUnaligned(ref dest[packedBytes], scale);
    }
}
