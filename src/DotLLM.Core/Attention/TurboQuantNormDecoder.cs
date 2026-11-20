using System.Runtime.CompilerServices;
using System.Buffers.Binary;

namespace DotLLM.Core.Attention;

/// <summary>
/// Decoder for the per-head-vector TurboQuant norm (<see cref="TqNormFormat"/>). Lives in Core so the CPU attention kernel
/// and the engine's encoder share one definition of what a stored norm means.
/// </summary>
public static class TurboQuantNormDecoder
{
    /// <summary>Lower end of the Log8 grid (code 1 decodes near this; exact 0 is code 0).</summary>
    public const float Log8Min = 1e-4f;

    /// <summary>Upper end of the Log8 grid (code 255).</summary>
    public const float Log8Max = 1e4f;

    private static readonly float Log2Min = MathF.Log2(Log8Min);
    private static readonly float Log2Span = MathF.Log2(Log8Max) - MathF.Log2(Log8Min);

    /// <summary>Decodes the norm stored at the start of <paramref name="src"/> (at least <see cref="TurboQuantLayout.NormBytes"/> bytes).</summary>
    public static float Decode(TqNormFormat f, ReadOnlySpan<byte> src)
    {
        switch (f)
        {
            case TqNormFormat.F32Reference:
                return BinaryPrimitives.ReadSingleLittleEndian(src);
            case TqNormFormat.F16:
                return (float)BinaryPrimitives.ReadHalfLittleEndian(src);
            case TqNormFormat.Bf16:
                return BitConverter.UInt32BitsToSingle((uint)BinaryPrimitives.ReadUInt16LittleEndian(src) << 16);
            case TqNormFormat.Fp8E4M3:
                return DecodeFp8(src[0], manBits: 3, bias: 7);
            case TqNormFormat.Fp8E5M2:
                return DecodeFp8(src[0], manBits: 2, bias: 15);
            case TqNormFormat.Log8:
                {
                    int code = src[0];
                    if (code == 0) return 0f;
                    return MathF.Pow(2f, Log2Min + code * (Log2Span / 255f));
                }
            default:
                throw new ArgumentOutOfRangeException(nameof(f), f, "unknown norm format");
        }
    }

    /// <summary>Pointer form of <see cref="Decode(TqNormFormat, ReadOnlySpan{byte})"/> for kernels.</summary>
    public static unsafe float Decode(TqNormFormat f, byte* src)
        => f == TqNormFormat.F16
            ? (float)Unsafe.ReadUnaligned<Half>(src)
            : Decode(f, new ReadOnlySpan<byte>(src, new TurboQuantLayout(1, 1, 1, f).NormBytes));

    private static float DecodeFp8(byte code, int manBits, int bias)
    {
        int man = code & ((1 << manBits) - 1);
        int exp = code >> manBits;
        if (exp == 0) return MathF.ScaleB(man, 1 - bias - manBits);
        return MathF.ScaleB(1f + man / (float)(1 << manBits), exp - bias);
    }
}
