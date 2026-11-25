using System.Numerics.Tensors;
using DotLLM.Core.Configuration;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Row decoder of the qwen4exp n-gram (PLE) table: every table row goes through <see cref="Decode"/>, so the three gather sites
/// (<see cref="Qwen4ExpPle.GatherRows"/>, and the cache-hit and read-through arms of <see cref="PleRowPrefetcher"/>) cannot disagree.
/// </summary>
/// <remarks>
/// <para>Besides every <see cref="Dequantize"/> format, the table may be the FP8 table as Qwen ships it (Edge0's <c>ple-fp8.gguf</c>):
/// a GGUF <see cref="QuantizationType.I8"/> tensor <c>[160, rows]</c> whose bytes are <b>FP8 E4M3FN</b> (1 sign, 4 exponent bits with
/// bias 7, 3 mantissa bits, no infinities, <c>0x7F</c>/<c>0xFF</c> are NaN), with ONE table-wide F32 scale from the GGUF metadata
/// (<c>strata.ple.format = f8_e4m3</c>, <c>strata.ple.scale</c>): <c>value = e4m3fn(byte) * scale</c>. The NaN codes never occur in a
/// real table; they decode to 0, like edge0-strata's <c>ngram.cpp</c> (<c>kFp8</c>).</para>
/// <para>The table is never widened: a row is decoded into the caller's scratch when it is gathered (1 byte per element on disk and
/// in the page cache, 160 B per row), so the 51.2 GB table stays a mapped file and only the touched rows are ever read.</para>
/// </remarks>
public static unsafe class PleRowCodec
{
    /// <summary>
    /// The FP8 value of every byte, before the table scale. Built from the bit fields (exponent rebuilt into an F32 exponent field,
    /// subnormals as <c>m * 2^-9</c>); the test oracle is built the other way round (a <c>Math.Pow</c> formula).
    /// </summary>
    private static readonly float[] s_e4m3fn = BuildE4M3FnTable();

    private static float[] BuildE4M3FnTable()
    {
        var t = new float[256];
        for (int b = 0; b < 256; b++)
        {
            int exp = (b >> 3) & 15, man = b & 7;
            float mag;
            if (exp == 15 && man == 7) mag = 0f;                                              // NaN encoding, never present in a real table
            else if (exp == 0) mag = man * (1f / 512f);                                       // subnormal: (m / 8) * 2^-6
            else mag = BitConverter.Int32BitsToSingle(((exp - 7 + 127) << 23) | (man << 20)); // normal: (1 + m/8) * 2^(e-7)
            t[b] = (b & 0x80) != 0 ? -mag : mag;
        }
        return t;
    }

    /// <summary>The FP8 E4M3FN byte <paramref name="code"/> as F32 (the table scale is not applied; NaN codes read as 0).</summary>
    public static float E4M3FnToSingle(byte code) => s_e4m3fn[code];

    /// <summary>
    /// True when <paramref name="quantType"/> is the FP8 table container (<see cref="QuantizationType.I8"/>: one FP8 byte per element,
    /// scaled by the table scale). I8 is otherwise only the tensor-overlay container, which is never a PLE table.
    /// </summary>
    public static bool IsFp8(QuantizationType quantType) => quantType == QuantizationType.I8;

    /// <summary>
    /// Validates a (table type, scale) pair at construction time: an FP8 table needs a positive finite scale (without it every row would
    /// silently decode at 1.0).
    /// </summary>
    /// <param name="quantType">Table storage type.</param>
    /// <param name="fp8Scale">The table scale (<c>strata.ple.scale</c>); ignored for every other type.</param>
    public static void Validate(QuantizationType quantType, float fp8Scale)
    {
        if (IsFp8(quantType) && !(fp8Scale > 0f && float.IsFinite(fp8Scale)))
            throw new ArgumentException(
                "An FP8 (I8) PLE table needs a positive finite table scale (GGUF metadata strata.ple.scale).", nameof(fp8Scale));
    }

    /// <summary>Decodes one table row of <paramref name="rowDim"/> elements at <paramref name="src"/> into F32.</summary>
    /// <param name="src">First byte of the row.</param>
    /// <param name="rowDim">Elements per row.</param>
    /// <param name="quantType">Table storage type (<see cref="QuantizationType.I8"/> = FP8 E4M3FN).</param>
    /// <param name="fp8Scale">Table scale for FP8 rows (ignored otherwise).</param>
    /// <param name="dest">Destination (at least <paramref name="rowDim"/> floats).</param>
    public static void Decode(nint src, int rowDim, QuantizationType quantType, float fp8Scale, Span<float> dest)
    {
        if (quantType != QuantizationType.I8)
        {
            Dequantize.ToFloat32(src, rowDim, quantType, dest);
            return;
        }
        var codes = new ReadOnlySpan<byte>((void*)src, rowDim);
        var d = dest.Slice(0, rowDim);
        float[] lut = s_e4m3fn;
        for (int i = 0; i < codes.Length; i++) d[i] = lut[codes[i]];
        TensorPrimitives.Multiply(d, fp8Scale, d);   // one F32 multiply per element, exactly strata's `v[b] * scale`
    }
}
