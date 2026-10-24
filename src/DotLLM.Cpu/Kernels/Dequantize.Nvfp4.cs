using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// NVFP4 dequantization (issue #1038). Block layout, as llama.cpp's <c>block_nvfp4</c>
/// (<c>ggml-common.h</c>): <c>d[4]</c> (one unsigned UE4M3 scale byte per 16-element sub-block)
/// at offset 0, then <c>qs[32]</c> (E2M1 nibbles) at offset 4; 36 bytes per 64 elements.
/// </summary>
/// <remarks>
/// <para>Element order differs from MXFP4: sub-block <c>s</c> owns <c>qs[8s .. 8s+7]</c>, and byte
/// <c>j</c> of it holds element <c>16s + j</c> in its low nibble and <c>16s + 8 + j</c> in its high
/// nibble (<c>dequantize_row_nvfp4</c>).</para>
/// <para>The per-tensor second-level scale (the sibling <c>X.scale</c> tensor) is NOT part of the
/// block and is not applied here: llama.cpp multiplies it into the matmul output
/// (<c>build_lora_mm</c>), and so do dotLLM's callers.</para>
/// </remarks>
public static unsafe partial class Dequantize
{
    /// <summary>NVFP4 block size in bytes: 4 (UE4M3 sub-block scales) + 32 (packed nibbles) = 36.</summary>
    internal const int Nvfp4BlockBytes = QuantFormat.Nvfp4BlockBytes;

    /// <summary>Elements per NVFP4 block.</summary>
    internal const int Nvfp4GroupSize = QuantFormat.Nvfp4GroupSize;

    /// <summary>Elements per NVFP4 sub-block (one UE4M3 scale each).</summary>
    internal const int Nvfp4SubBlockSize = QuantFormat.Nvfp4SubBlockSize;

    /// <summary>Byte offset of <c>qs</c> inside an NVFP4 block (after the four scale bytes).</summary>
    internal const int Nvfp4QsOffset = Nvfp4GroupSize / Nvfp4SubBlockSize;

    /// <summary>
    /// Literal transcription of llama.cpp's <c>ggml_ue4m3_to_fp32</c> (<c>ggml-impl.h</c>): an unsigned
    /// E4M3 (bias 7, 3 mantissa bits, subnormals at exponent 0) returned <b>halved</b>, to pair with the
    /// doubled E2M1 table <see cref="Mxfp4Values"/>. <c>0x00</c> and <c>0x7F</c> (the E4M3 NaN) decode
    /// to 0, and bit 7 is ignored rather than treated as a sign, so <c>0x80</c> is 0 and <c>0xFF</c> is
    /// 240. This is the oracle the lookup table <see cref="Ue4m3ToFloatHalf"/> is built from and
    /// tested against.
    /// </summary>
    /// <param name="x">The raw scale byte.</param>
    /// <returns>Half the decoded scale value.</returns>
    internal static float Ue4m3ToFloatHalfReference(byte x)
    {
        if (x == 0 || x == 0x7F)
            return 0.0f;

        int exp = (x >> 3) & 0xF;
        int man = x & 0x7;
        float raw = exp == 0
            ? MathF.ScaleB(man, -9)
            : MathF.ScaleB(1.0f + man / 8.0f, exp - 7);
        return raw * 0.5f;
    }

    /// <summary>All 256 UE4M3 bytes decoded by <see cref="Ue4m3ToFloatHalfReference"/>.</summary>
    private static readonly float[] Ue4m3HalfTable = BuildUe4m3HalfTable();

    private static float[] BuildUe4m3HalfTable()
    {
        var table = new float[256];
        for (int i = 0; i < table.Length; i++)
            table[i] = Ue4m3ToFloatHalfReference((byte)i);
        return table;
    }

    /// <summary>
    /// Table lookup of <see cref="Ue4m3ToFloatHalfReference"/> for the hot paths (one load, no
    /// branches). A byte always indexes inside the 256-entry table, so the bounds check is skipped.
    /// </summary>
    /// <param name="x">The raw scale byte.</param>
    /// <returns>Half the decoded scale value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float Ue4m3ToFloatHalf(byte x)
        => Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(Ue4m3HalfTable), x);

    /// <summary>
    /// Scalar NVFP4 dequantization — port of llama.cpp's <c>dequantize_row_nvfp4</c>:
    /// <c>y[16s + j] = kvalues[qs[8s+j] &amp; 0xF] * d_s</c>, <c>y[16s + j + 8] = kvalues[qs[8s+j] &gt;&gt; 4] * d_s</c>
    /// with <c>d_s = ue4m3_half(d[s])</c>.
    /// </summary>
    /// <param name="src">Pointer to NVFP4 blocks.</param>
    /// <param name="elementCount">Elements to decode; a multiple of 64.</param>
    /// <param name="dest">Destination, at least <paramref name="elementCount"/> long.</param>
    /// <exception cref="ArgumentException"><paramref name="elementCount"/> is not a multiple of 64.</exception>
    [SkipLocalsInit]
    internal static void DequantizeNvfp4(nint src, long elementCount, Span<float> dest)
    {
        if (elementCount % Nvfp4GroupSize != 0)
            throw new ArgumentException(
                $"NVFP4 element count must be a multiple of {Nvfp4GroupSize}, got {elementCount}",
                nameof(elementCount));

        long blockCount = elementCount / Nvfp4GroupSize;
        byte* block = (byte*)src;
        int outIdx = 0;
        ReadOnlySpan<sbyte> kvalues = Mxfp4Values;

        for (long b = 0; b < blockCount; b++)
        {
            byte* qs = block + Nvfp4QsOffset;
            for (int s = 0; s < Nvfp4QsOffset; s++)
            {
                float d = Ue4m3ToFloatHalf(block[s]);
                byte* sq = qs + s * (Nvfp4SubBlockSize / 2);
                int o = outIdx + s * Nvfp4SubBlockSize;
                for (int j = 0; j < Nvfp4SubBlockSize / 2; j++)
                {
                    dest[o + j] = kvalues[sq[j] & 0x0F] * d;
                    dest[o + j + Nvfp4SubBlockSize / 2] = kvalues[sq[j] >> 4] * d;
                }
            }

            outIdx += Nvfp4GroupSize;
            block += Nvfp4BlockBytes;
        }
    }
}
