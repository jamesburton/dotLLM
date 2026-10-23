// Ported from SyzygyResearch/Mach-1-Additive-35B's decode.py (Apache License 2.0):
// https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B/blob/main/decode.py
// Apache-2.0 is one-way compatible with dotLLM's GPLv3 (see CLAUDE.md). Ported for
// issue #266 (Phase A: codec decoder).
namespace DotLLM.Models.Quantization.Mach1;

/// <summary>
/// Tile-grid wavefront indexing and per-tile gamma scaling for the v3t
/// expert container. Mirrors decode.py's <c>wave_index_map</c> /
/// <c>apply_wave_gamma</c>.
/// </summary>
/// <remarks>
/// The encoder's wavefront schedule starts one wave per anti-diagonal
/// beginning at the bottom-right column (<c>(Mb-i-1, Nb-1)</c> for
/// <c>i in [0, Mb)</c>), then one more per anti-diagonal starting along the
/// top row (<c>(0, Nb-i-1)</c> for <c>i in [0, Nb)</c>). Both schedules
/// include a start point at tile <c>(0, Nb-1)</c> (the top-right corner) —
/// once from the end of the first group, once from the start of the second —
/// so that tile is written twice, and the wavefront index is a single
/// counter that increments across <i>both</i> groups without resetting. The
/// second (later) write wins. Resetting the counter between groups, or
/// deduplicating the corner, silently corrupts an entire tile row (issue
/// #266) — this port keeps the two loops sharing one counter to avoid that.
/// </remarks>
public static class Mach1WaveGamma
{
    /// <summary>
    /// Builds the <c>[Mb, Nb]</c> tile grid (row-major) of the <i>last</i>
    /// wavefront index that wrote each tile.
    /// </summary>
    public static int[] BuildWaveIndexMap(int mb, int nb)
    {
        if (mb <= 0 || nb <= 0)
            throw new ArgumentOutOfRangeException(mb <= 0 ? nameof(mb) : nameof(nb));

        var idx = new int[mb * nb];
        int w = 0;
        // First group: anti-diagonals starting down the last column.
        for (int i = 0; i < mb; i++)
            WriteWavefront(idx, mb, nb, mb - i - 1, nb - 1, w++);
        // Second group: anti-diagonals starting along the first row —
        // continues the SAME counter (deliberately overwrites the corner
        // tile shared with the first group's final entry).
        for (int i = 0; i < nb; i++)
            WriteWavefront(idx, mb, nb, 0, nb - i - 1, w++);
        return idx;
    }

    private static void WriteWavefront(int[] idx, int mb, int nb, int jm, int jn, int w)
    {
        while (jm >= 0 && jm < mb && jn >= 0 && jn < nb)
        {
            idx[jm * nb + jn] = w;
            jm++;
            jn--;
        }
    }

    /// <summary>
    /// Multiplies every <c>td x td</c> tile of <paramref name="weights"/>
    /// (row-major <c>[m, n]</c>) by <c>gamma[wave(tile)]</c>, in place.
    /// </summary>
    public static void Apply(Span<float> weights, int m, int n, ReadOnlySpan<float> gamma, int td)
    {
        if (m % td != 0 || n % td != 0)
            throw new ArgumentException($"[{m},{n}] must be divisible by tile size {td}");
        if (weights.Length != m * n)
            throw new ArgumentException($"weights length {weights.Length} must equal m*n={m * n}");

        int mb = m / td, nb = n / td;
        int[] waveIndex = BuildWaveIndexMap(mb, nb);
        if (gamma.Length < mb + nb)
            throw new ArgumentException(
                $"gamma length {gamma.Length} is shorter than the wavefront count Mb+Nb={mb + nb}");

        for (int bm = 0; bm < mb; bm++)
        {
            for (int bn = 0; bn < nb; bn++)
            {
                float g = gamma[waveIndex[bm * nb + bn]];
                int rowBase = bm * td;
                int colBase = bn * td;
                for (int ti = 0; ti < td; ti++)
                {
                    int rowOffset = (rowBase + ti) * n + colBase;
                    for (int tj = 0; tj < td; tj++)
                        weights[rowOffset + tj] *= g;
                }
            }
        }
    }
}
