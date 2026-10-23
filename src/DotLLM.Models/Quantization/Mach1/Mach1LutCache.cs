// Ported from SyzygyResearch/Mach-1-Additive-35B's decode.py (Apache License 2.0):
// https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B/blob/main/decode.py
// Apache-2.0 is one-way compatible with dotLLM's GPLv3 (see CLAUDE.md). Ported for
// issue #266 (Phase A: codec decoder).
using System.Collections.Concurrent;
using System.Runtime.InteropServices;

namespace DotLLM.Models.Quantization.Mach1;

/// <summary>
/// Process-wide cache for <see cref="Mach1QuantLutSym.ExpandFullLut"/>,
/// keyed by <c>(L, tlutBits, content-hash-of-small-tlut)</c> — mirrors
/// decode.py's <c>_np_full_lut_cached</c> (keyed by
/// <c>(L, tlut_bits, tlut.tobytes())</c>). Expanding a <c>[2^16, 8]</c> LUT
/// touches 512K rows; every expert projection in a layer shares the same
/// small table, so caching avoids redoing that work per projection.
/// </summary>
public static class Mach1LutCache
{
    private static readonly ConcurrentDictionary<(int L, int TlutBits, ulong Hash), float[]> Cache = new();

    /// <summary>
    /// Returns the cached expansion for this <c>(smallTlut, v, l, tlutBits)</c>
    /// if one exists, otherwise computes, caches, and returns it.
    /// </summary>
    public static float[] GetOrExpand(ReadOnlySpan<float> smallTlut, int v, int l, int tlutBits)
    {
        ulong hash = Fnv1a64(MemoryMarshal.AsBytes(smallTlut));
        var key = (l, tlutBits, hash);
        if (Cache.TryGetValue(key, out float[]? cached))
            return cached;

        float[] table = Mach1QuantLutSym.ExpandFullLut(smallTlut, v, l, tlutBits);
        Cache[key] = table;
        return table;
    }

    /// <summary>Clears the cache. Exposed for test isolation.</summary>
    public static void Clear() => Cache.Clear();

    private static ulong Fnv1a64(ReadOnlySpan<byte> data)
    {
        const ulong offsetBasis = 14695981039346656037UL;
        const ulong prime = 1099511628211UL;
        ulong hash = offsetBasis;
        foreach (byte b in data)
        {
            hash ^= b;
            hash *= prime;
        }
        return hash;
    }
}
