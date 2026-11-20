using System.Buffers.Binary;
using DotLLM.Core.Attention;

namespace DotLLM.Engine.KvCache.Codecs;

/// <summary>
/// Encoders/decoders for the per-head-vector TurboQuant norm (see <see cref="TqNormFormat"/>).
/// Norms are non-negative, finite L2 norms. Values above a format's range <b>saturate</b> to its
/// largest finite value (never Inf/NaN) and are counted; NaN or negative input is a caller bug and throws.
/// </summary>
public static class TurboQuantNormCodec
{
    /// <summary>Lower end of the Log8 grid (code 1 decodes near this; exact 0 is code 0).</summary>
    public const float Log8Min = TurboQuantNormDecoder.Log8Min;
    /// <summary>Upper end of the Log8 grid (code 255).</summary>
    public const float Log8Max = TurboQuantNormDecoder.Log8Max;

    private static readonly float Log2Min = MathF.Log2(Log8Min);
    private static readonly float Log2Span = MathF.Log2(Log8Max) - MathF.Log2(Log8Min);

    /// <summary>Bytes per stored norm for <paramref name="f"/>.</summary>
    public static int BytesPerNorm(TqNormFormat f) => new TurboQuantLayout(1, 1, 1, f).NormBytes;

    /// <summary>Encodes <paramref name="norms"/> into <paramref name="dst"/> (<c>norms.Length * BytesPerNorm</c> bytes).</summary>
    public static void Encode(TqNormFormat f, ReadOnlySpan<float> norms, Span<byte> dst)
        => Encode(f, norms, dst, out _);

    /// <summary>As <see cref="Encode(TqNormFormat, ReadOnlySpan{float}, Span{byte})"/>, also returning how many values saturated.</summary>
    public static void Encode(TqNormFormat f, ReadOnlySpan<float> norms, Span<byte> dst, out int saturated)
    {
        int nb = BytesPerNorm(f);
        if (dst.Length < norms.Length * nb)
            throw new ArgumentException($"dst too small: need {norms.Length * nb}, got {dst.Length}.", nameof(dst));
        saturated = 0;
        for (int i = 0; i < norms.Length; i++)
        {
            float x = norms[i];
            if (float.IsNaN(x)) throw new ArgumentException($"norm[{i}] is NaN.", nameof(norms));
            if (x < 0f) throw new ArgumentOutOfRangeException(nameof(norms), $"norm[{i}] = {x} is negative.");
            Span<byte> d = dst.Slice(i * nb, nb);
            switch (f)
            {
                case TqNormFormat.F32Reference:
                    BinaryPrimitives.WriteSingleLittleEndian(d, x);
                    break;
                case TqNormFormat.F16:
                    {
                        if (x > 65504f) { x = 65504f; saturated++; }
                        BinaryPrimitives.WriteHalfLittleEndian(d, (Half)x); // IEEE round-to-nearest-even
                        break;
                    }
                case TqNormFormat.Bf16:
                    {
                        const float bf16Max = 3.38953139e38f; // 0x7F7F0000
                        if (x > bf16Max) { x = bf16Max; saturated++; }
                        uint bits = BitConverter.SingleToUInt32Bits(x);
                        bits += 0x7FFFu + ((bits >> 16) & 1u);  // RNE
                        ushort h = (ushort)(bits >> 16);
                        if (h > 0x7F7F) h = 0x7F7F;             // rounding carried into Inf: saturate
                        BinaryPrimitives.WriteUInt16LittleEndian(d, h);
                        break;
                    }
                case TqNormFormat.Fp8E4M3:
                    d[0] = EncodeFp8(x, expBits: 4, manBits: 3, bias: 7, maxCode: 0x7E, maxValue: 448f, ref saturated);
                    break;
                case TqNormFormat.Fp8E5M2:
                    d[0] = EncodeFp8(x, expBits: 5, manBits: 2, bias: 15, maxCode: 0x7B, maxValue: 57344f, ref saturated);
                    break;
                case TqNormFormat.Log8:
                    d[0] = EncodeLog8(x, ref saturated);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(f), f, "unknown norm format");
            }
        }
    }

    /// <summary>Decodes the <paramref name="index"/>-th norm from <paramref name="src"/>.</summary>
    public static float Decode(TqNormFormat f, ReadOnlySpan<byte> src, int index)
    {
        int nb = BytesPerNorm(f);
        return TurboQuantNormDecoder.Decode(f, src.Slice(index * nb, nb));
    }

    // Non-negative FP8, round-to-nearest-even, saturating. A subnormal-range value keeps the format's
    // absolute subnormal spacing (relative error is unbounded below the smallest normal).
    private static byte EncodeFp8(float x, int expBits, int manBits, int bias, int maxCode, float maxValue, ref int saturated)
    {
        if (x == 0f) return 0;
        if (x > maxValue) { saturated++; return (byte)maxCode; }

        int e = (int)((BitConverter.SingleToUInt32Bits(x) >> 23) & 0xFF) - 127; // x is a normal float here
        if (BitConverter.SingleToUInt32Bits(x) >> 23 == 0) e = -127;            // float subnormal -> flushes below
        int eMin = 1 - bias;
        if (e < eMin) e = eMin;                                                  // subnormal range
        // quantum = 2^(e - manBits); x / quantum is exact (power-of-two scaling).
        float scaled = MathF.ScaleB(x, manBits - e);
        int q = (int)MathF.Round(scaled, MidpointRounding.ToEven);
        int hidden = 1 << manBits;
        int code;
        if (q >= hidden)
        {
            if (q == 2 * hidden) { e++; q = hidden; } // mantissa carry
            code = ((e + bias) << manBits) | (q - hidden);
        }
        else
        {
            code = q; // subnormal (exponent field 0); q == hidden handled above equals exp field 1, mantissa 0
        }
        if (code > maxCode) { code = maxCode; saturated++; }
        return (byte)code;
    }

    private static byte EncodeLog8(float x, ref int saturated)
    {
        if (x == 0f) return 0;
        if (x > Log8Max) { saturated++; return 255; }
        float t = 255f * (MathF.Log2(x) - Log2Min) / Log2Span;
        int code = (int)MathF.Round(t, MidpointRounding.ToEven);
        // Only exact zero owns code 0; a non-zero value below the grid floor is clamped to the first step.
        return (byte)Math.Clamp(code, 1, 255);
    }
}
