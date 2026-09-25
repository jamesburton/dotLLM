// Ported from SyzygyResearch/Mach-1-Additive-35B's decode.py (Apache License 2.0):
// https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B/blob/main/decode.py
// Apache-2.0 is one-way compatible with dotLLM's GPLv3 (see CLAUDE.md). Ported for
// issue #266 (Phase A: codec decoder).
using System.Runtime.CompilerServices;

namespace DotLLM.Models.Quantization.Mach1;

/// <summary>
/// Power-of-two padding used throughout the Mach-1 codec: every weight
/// matrix is decoded at <c>(padto(m0), padto(n0))</c> before the final crop
/// back to <c>(m0, n0)</c>. Mirrors decode.py's <c>padto</c>.
/// </summary>
public static class Mach1Padding
{
    /// <summary>
    /// Returns <paramref name="dim"/> unchanged if it is already a power of
    /// two, otherwise the next power of two above it.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int PadToPowerOfTwo(int dim)
    {
        if (dim <= 0)
            throw new ArgumentOutOfRangeException(nameof(dim), dim, "Dimension must be positive.");
        if ((dim & (dim - 1)) == 0)
            return dim;

        // Bit-twiddling next-power-of-two (avoids float log2 precision pitfalls).
        int v = dim - 1;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return v + 1;
    }
}
