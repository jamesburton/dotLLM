// Ported from SyzygyResearch/Mach-1-Additive-35B's decode.py (Apache License 2.0):
// https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B/blob/main/decode.py
// Apache-2.0 is one-way compatible with dotLLM's GPLv3 (see CLAUDE.md). Ported for
// issue #266 (Phase A: codec decoder).
namespace DotLLM.Models.Quantization.Mach1;

/// <summary>
/// Codebook / trellis parameters for one Mach-1 tier, mirroring decode.py's
/// <c>CB2</c>/<c>CB1</c>/<c>CB4</c>/<c>CB_V3T</c> dicts and the per-file
/// <c>cb_params</c> metadata blob.
/// </summary>
/// <param name="K">
/// Bits per weight (may be fractional, e.g. <c>1.5</c> for the v3t expert
/// tier — the only constraint is that <c>K*V</c> and <c>K*T</c> are whole
/// numbers of bits).
/// </param>
/// <param name="L">Shift-register width in bits (both known tiers use 16).</param>
/// <param name="V">Vector width emitted per register state (LUT columns).</param>
/// <param name="TlutBits">
/// log2 of the persisted (small) codebook's row count; the full LUT has
/// <c>2^L</c> rows, hashed down from the <c>2^TlutBits</c>-row persisted one.
/// </param>
/// <param name="TdX">Tile height (rows per 16x16-style tile).</param>
/// <param name="TdY">Tile width (columns per tile).</param>
public readonly record struct Mach1CbParams(double K, int L, int V, int TlutBits, int TdX, int TdY)
{
    /// <summary>Element count of one tile (<c>TdX * TdY</c>).</summary>
    public int TileElementCount => TdX * TdY;
}
