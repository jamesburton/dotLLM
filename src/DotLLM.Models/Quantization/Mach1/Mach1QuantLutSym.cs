// Ported from SyzygyResearch/Mach-1-Additive-35B's decode.py (Apache License 2.0):
// https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B/blob/main/decode.py
// Apache-2.0 is one-way compatible with dotLLM's GPLv3 (see CLAUDE.md). Ported for
// issue #266 (Phase A: codec decoder).
namespace DotLLM.Models.Quantization.Mach1;

/// <summary>
/// Expands the persisted <c>[2^tlutBits, V]</c> codebook to the full
/// <c>[2^L, V]</c> decoder table, per the hashed-symmetric-LUT rule
/// ("quantlut_sym"). Mirrors decode.py's <c>_np_full_lut</c>.
/// </summary>
/// <remarks>
/// For register state <c>s</c>, with <c>p = s*(s+1)</c> (exact in integer
/// arithmetic): <c>row(s) = (p &gt;&gt; (16 - tlutBits - 1)) &amp; (2^tlutBits - 1)</c>,
/// and <c>vec(s) = tlut[row(s)]</c> with component 0 negated iff bit 15 of
/// <c>p</c> is set.
/// </remarks>
public static class Mach1QuantLutSym
{
    /// <summary>
    /// Expands the small persisted table to the full <c>[2^L, V]</c> table.
    /// </summary>
    /// <param name="smallTlut">
    /// The persisted table, row-major, length <c>2^tlutBits * v</c>.
    /// </param>
    /// <param name="v">LUT vector width (columns).</param>
    /// <param name="l">Shift-register width in bits.</param>
    /// <param name="tlutBits">log2 of the persisted table's row count.</param>
    /// <returns>The expanded <c>[2^L, V]</c> table, row-major.</returns>
    public static float[] ExpandFullLut(ReadOnlySpan<float> smallTlut, int v, int l, int tlutBits)
    {
        if (l < 1 || l > 24)
            throw new ArgumentOutOfRangeException(nameof(l), l, "L must be in [1, 24].");
        if (tlutBits < 0 || tlutBits > l)
            throw new ArgumentOutOfRangeException(nameof(tlutBits), tlutBits, "tlutBits must be in [0, L].");
        if (v <= 0)
            throw new ArgumentOutOfRangeException(nameof(v));

        int smallRows = 1 << tlutBits;
        if (smallTlut.Length != smallRows * v)
            throw new ArgumentException(
                $"small tlut length {smallTlut.Length} does not match 2^tlutBits*V = {smallRows * v}");

        int shift = 16 - tlutBits - 1;
        if (shift < 0)
            throw new ArgumentException(
                $"tlutBits={tlutBits} implies a negative hash shift (16 - tlutBits - 1 = {shift}); " +
                "the quantlut_sym hash assumes a 16-bit register (L<=16 with this bit layout).");

        int fullRows = 1 << l;
        var table = new float[fullRows * v];
        int rowMask = smallRows - 1;

        for (int s = 0; s < fullRows; s++)
        {
            long p = (long)s * (s + 1); // exact; overflows int32 for s just above ~46340
            int row = (int)((p >> shift) & rowMask);
            bool signFlip = ((p >> 15) & 1L) != 0;

            int srcBase = row * v;
            int dstBase = s * v;
            for (int c = 0; c < v; c++)
            {
                float val = smallTlut[srcBase + c];
                table[dstBase + c] = (c == 0 && signFlip) ? -val : val;
            }
        }
        return table;
    }
}
