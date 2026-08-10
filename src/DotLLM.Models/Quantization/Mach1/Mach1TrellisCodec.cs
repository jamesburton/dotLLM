// Ported from SyzygyResearch/Mach-1-Additive-35B's decode.py (Apache License 2.0):
// https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B/blob/main/decode.py
// Apache-2.0 is one-way compatible with dotLLM's GPLv3 (see CLAUDE.md). Ported for
// issue #266 (Phase A: codec decoder).
namespace DotLLM.Models.Quantization.Mach1;

/// <summary>
/// Tail-biting trellis bit-unpack and tile reconstruction, mirroring
/// decode.py's <c>_np_unpack_trellis</c> / <c>_np_recons</c> exactly.
/// </summary>
/// <remarks>
/// <para>
/// Per tile: an <c>L</c>-bit shift register emits one <c>V</c>-vector per
/// step (<c>T/V</c> steps total, <c>T = td_x*td_y</c>), then shifts in
/// <c>K*V</c> fresh bits: <c>reg_i = ((reg_{i-1} &lt;&lt; K*V) | fresh_i) &amp;
/// (2^L - 1)</c>.
/// </para>
/// <para>
/// The bitstream is packed MSB-first per 16-bit word ("big-endian" in the
/// vendor's terminology — bit 15 of each <see cref="ushort"/> word is the
/// next bit of the logical stream, not a byte-order concern): <c>T*K</c>
/// bits total. The stream is <b>tail-biting</b> — the final <c>L - K*V</c>
/// register bits are not stored; they wrap around to the start of the
/// stream. Getting this wrap wrong "produces plausible-looking garbage
/// rather than an error" (issue #266), so every step here is a literal,
/// unparaphrased port of <c>_np_unpack_trellis</c>.
/// </para>
/// </remarks>
public static class Mach1TrellisCodec
{
    /// <summary>
    /// Computes (bits per shift step, bits per tile) for a given tile element
    /// count. <paramref name="k"/> need not be an integer — the only rate
    /// constraint is that <c>k*v</c> and <c>k*tileElementCount</c> are whole
    /// numbers of bits (mirrors decode.py's <c>_np_rate_bits</c>).
    /// </summary>
    public static (int StepBits, int TileBits) ComputeRateBits(int tileElementCount, double k, int v)
    {
        double stepD = k * v;
        double tileBitsD = k * tileElementCount;
        int step = (int)Math.Round(stepD, MidpointRounding.AwayFromZero);
        int tileBits = (int)Math.Round(tileBitsD, MidpointRounding.AwayFromZero);
        if (Math.Abs(stepD - step) > 1e-6 || Math.Abs(tileBitsD - tileBits) > 1e-6)
            throw new ArgumentException(
                $"rate K={k} with V={v}, T={tileElementCount} needs whole-bit steps " +
                $"(K*V={stepD}) and a whole-bit tile (K*T={tileBitsD})");
        return (step, tileBits);
    }

    /// <summary>
    /// Unpacks one tile's packed trellis words into register states.
    /// </summary>
    /// <param name="tileWords">
    /// The tile's packed bitstream, MSB-first per word, length
    /// <c>tileElementCount * K / 16</c>.
    /// </param>
    /// <param name="tileElementCount">Tile element count (<c>td_x * td_y</c>).</param>
    /// <param name="cb">Codebook/trellis parameters (<c>K</c>, <c>L</c>, <c>V</c>).</param>
    /// <param name="states">
    /// Destination for the <c>tileElementCount / V</c> register states, one
    /// per emission step.
    /// </param>
    public static void UnpackTileStates(
        ReadOnlySpan<ushort> tileWords, int tileElementCount, Mach1CbParams cb, Span<int> states)
    {
        int l = cb.L;
        (int step, int tileBits) = ComputeRateBits(tileElementCount, cb.K, cb.V);
        if (step > l)
            throw new ArgumentException($"K*V={step} exceeds register width L={l}");

        int nstep = tileElementCount / cb.V;
        if (states.Length != nstep)
            throw new ArgumentException(
                $"states span length {states.Length} must equal T/V={nstep}");

        int wrapBits = l - step;
        int totalBits = tileBits + wrapBits;

        // Extract tileBits MSB-first bits from the packed words, then append
        // the tail-biting wrap (the first wrapBits bits of the stream, again).
        Span<byte> bits = totalBits <= 2048 ? stackalloc byte[totalBits] : new byte[totalBits];
        int bi = 0;
        int wordIdx = 0;
        while (bi < tileBits)
        {
            if (wordIdx >= tileWords.Length)
                throw new ArgumentException(
                    $"tileWords (length {tileWords.Length}) too short for tileBits={tileBits}");
            ushort w = tileWords[wordIdx++];
            for (int b = 15; b >= 0 && bi < tileBits; b--)
                bits[bi++] = (byte)((w >> b) & 1);
        }
        for (int i = 0; i < wrapBits; i++)
            bits[tileBits + i] = bits[i];

        // Seed register = first L bits, MSB first.
        int reg = 0;
        for (int i = 0; i < l; i++)
            reg = (reg << 1) | bits[i];
        states[0] = reg;

        int mask = (1 << l) - 1;
        int pos = l;
        for (int i = 1; i < nstep; i++)
        {
            int fresh = 0;
            for (int j = 0; j < step; j++)
                fresh = (fresh << 1) | bits[pos + j];
            pos += step;
            reg = ((reg << step) & mask) | fresh;
            states[i] = reg;
        }
    }

    /// <summary>
    /// Unpacks every tile in a whole tensor's trellis stream into register
    /// states, tile-major (matches the row-major tile grid layout).
    /// </summary>
    public static void UnpackAllTiles(
        ReadOnlySpan<ushort> trellisWords, int ntiles, int wordsPerTile, int tileElementCount,
        Mach1CbParams cb, Span<int> states)
    {
        int stepsPerTile = tileElementCount / cb.V;
        if (states.Length != ntiles * stepsPerTile)
            throw new ArgumentException(
                $"states span length {states.Length} must equal ntiles*T/V={ntiles * stepsPerTile}");
        if (trellisWords.Length != ntiles * wordsPerTile)
            throw new ArgumentException(
                $"trellisWords length {trellisWords.Length} must equal ntiles*wordsPerTile={ntiles * wordsPerTile}");

        for (int t = 0; t < ntiles; t++)
        {
            var wordsSlice = trellisWords.Slice(t * wordsPerTile, wordsPerTile);
            var stateSlice = states.Slice(t * stepsPerTile, stepsPerTile);
            UnpackTileStates(wordsSlice, tileElementCount, cb, stateSlice);
        }
    }

    /// <summary>
    /// Register states + full LUT -&gt; codebook-unit weights <c>[m, n]</c>,
    /// row-major, dense. Mirrors decode.py's <c>_np_recons</c>: <c>V</c>
    /// scalars per state, tiles laid out row-major over the
    /// <c>(m/td_x, n/td_y)</c> grid, and row-major within each tile.
    /// </summary>
    /// <param name="states">Register states, tile-major, <c>ntiles * (T/V)</c> long.</param>
    /// <param name="ntiles">Tile count (<c>(m/td_x) * (n/td_y)</c>).</param>
    /// <param name="m">Padded row count.</param>
    /// <param name="n">Padded column count.</param>
    /// <param name="tdX">Tile height.</param>
    /// <param name="tdY">Tile width.</param>
    /// <param name="v">LUT vector width (columns emitted per state).</param>
    /// <param name="fullLut">The expanded <c>[2^L, V]</c> LUT, row-major.</param>
    /// <param name="dest">Destination for the dense <c>[m, n]</c> unit weights, row-major.</param>
    public static void ReconstructWeights(
        ReadOnlySpan<int> states, int ntiles, int m, int n, int tdX, int tdY, int v,
        ReadOnlySpan<float> fullLut, Span<float> dest)
    {
        if (m % tdX != 0 || n % tdY != 0)
            throw new ArgumentException($"[{m},{n}] must be divisible by tile size [{tdX},{tdY}]");
        if (dest.Length != m * n)
            throw new ArgumentException($"dest length {dest.Length} must equal m*n={m * n}");

        int nb = n / tdY;
        int stepsPerTile = (tdX * tdY) / v;
        if (states.Length != ntiles * stepsPerTile)
            throw new ArgumentException(
                $"states length {states.Length} must equal ntiles*T/V={ntiles * stepsPerTile}");

        for (int tile = 0; tile < ntiles; tile++)
        {
            int tileRow = tile / nb;
            int tileCol = tile % nb;
            int rowBase = tileRow * tdX;
            int colBase = tileCol * tdY;
            int tileStateBase = tile * stepsPerTile;

            for (int si = 0; si < stepsPerTile; si++)
            {
                int state = states[tileStateBase + si];
                int lutOffset = state * v;
                int eBase = si * v;
                for (int vc = 0; vc < v; vc++)
                {
                    int e = eBase + vc;
                    int localRow = e / tdY;
                    int localCol = e % tdY;
                    dest[(rowBase + localRow) * n + (colBase + localCol)] = fullLut[lutOffset + vc];
                }
            }
        }
    }
}
