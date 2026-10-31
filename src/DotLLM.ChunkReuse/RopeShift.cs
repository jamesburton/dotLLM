using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;

namespace DotLLM.ChunkReuse;

/// <summary>
/// Moves post-RoPE keys to a new position. dotLLM stores K after RoPE in every cache, so a key computed at position
/// <c>p</c> holds <c>R(p)·k</c>; rotating it by <c>R(Δ)</c> gives <c>R(p+Δ)·k</c>, the key the model writes at <c>p+Δ</c>
/// (exact for plain RoPE, see <see cref="RopeParameters.FromConfig"/>).
/// </summary>
public static class RopeShift
{
    /// <summary>
    /// Rotates every head vector in <paramref name="keys"/> by <paramref name="positionDelta"/> positions in place.
    /// </summary>
    /// <remarks>
    /// The cos/sin row is computed the way the CPU model builds row <c>Δ</c> of its table (float frequency
    /// <c>1 / θ^(2i/ropeDim)</c>, float angle <c>Δ·freq</c>, cos/sin evaluated in double and rounded once), and applied with the
    /// model's own rotation kernels, so the result differs from the model's key at <c>p+Δ</c> only by float rounding of the
    /// composed rotation. That rounding grows with the angle (about <c>|p·freq|·2⁻²⁴</c>): small for test positions, not bitwise.
    /// </remarks>
    /// <param name="keys">K rows, any number of tokens, heads contiguous within a row: length must be a multiple of <paramref name="headDim"/>.</param>
    /// <param name="headDim">Dimension per head; the leading <see cref="RopeParameters.RopeDim"/> dims of each head rotate.</param>
    /// <param name="positionDelta">Positions to move by (target minus source). 0 is a no-op; negative moves towards 0.</param>
    /// <param name="rope">RoPE parameters of the model that produced <paramref name="keys"/>.</param>
    /// <exception cref="ArgumentException">The shapes are inconsistent.</exception>
    public static void Apply(Span<float> keys, int headDim, int positionDelta, in RopeParameters rope)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);
        if (keys.Length % headDim != 0)
            throw new ArgumentException($"keys length {keys.Length} is not a multiple of headDim {headDim}.", nameof(keys));
        int ropeDim = rope.RopeDim;
        if (ropeDim <= 0 || ropeDim % 2 != 0 || ropeDim > headDim)
            throw new ArgumentException($"RopeDim {ropeDim} must be positive, even and <= headDim {headDim}.", nameof(rope));
        if (rope.Type is not (RoPEType.Norm or RoPEType.NeoX))
            throw new ArgumentException($"Unknown RoPE pairing {rope.Type}.", nameof(rope));
        if (positionDelta == 0 || keys.IsEmpty)
            return;

        int half = ropeDim / 2;
        Span<float> cos = half <= 256 ? stackalloc float[half] : new float[half];
        Span<float> sin = half <= 256 ? stackalloc float[half] : new float[half];
        for (int i = 0; i < half; i++)
        {
            // Same float expressions as RoPE.PrecomputeFrequencyTable / PortableMath (double evaluation, one rounding).
            float freq = 1.0f / (float)Math.Pow(rope.Theta, 2.0f * i / ropeDim);
            float angle = positionDelta * freq;
            cos[i] = (float)Math.Cos(angle);
            sin[i] = (float)Math.Sin(angle);
        }

        bool neoX = rope.Type == RoPEType.NeoX;
        for (int offset = 0; offset < keys.Length; offset += headDim)
        {
            Span<float> head = keys.Slice(offset, ropeDim);
            if (neoX)
                RoPE.ApplyRotationNeoX(head, cos, sin, ropeDim);
            else
                RoPE.ApplyRotation(head, cos, sin, ropeDim);
        }
    }
}
