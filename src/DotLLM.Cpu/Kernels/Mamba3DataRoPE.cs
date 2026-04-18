using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Data-dependent 2-D rotation applied to the Mamba-3 <c>B</c> and <c>C</c>
/// coefficients. Absorbs the block-diagonal complex rotation matrix <c>R_t</c>
/// from Proposition 3 of Lahoti et al. (ICLR 2026, arXiv 2603.15569) into the
/// real-valued <c>B, C</c> projections so the SSM scan kernel itself stays
/// identical to Mamba-2.
/// </summary>
/// <remarks>
/// <para>
/// Unlike standard position-based RoPE (<see cref="RoPE"/>) whose cos/sin
/// tables depend only on sequence index, the Mamba-3 rotation angles are
/// <b>data-dependent</b>: each per-token timestep <c>dt[t, h]</c> modulates a
/// per-token frequency projection <c>θ[t, k]</c> (projected from the input,
/// not a learned-per-head table), and the angles accumulate along time with a
/// <b>negative</b> cumulative sum:
/// </para>
/// <code>
/// raw_angles[t, h, k] =  dt[t, h] * theta[t, k]           // outer product
/// cum_angles[t, h, k] = -sum_{s=0..t} raw_angles[s, h, k] // NEGATIVE cumsum
/// </code>
/// <para>
/// Each <c>d_state</c>-dim slice of <c>B</c> and <c>C</c> is then split into
/// adjacent even/odd pairs <c>(v[2k], v[2k+1])</c> and rotated by the 2-D
/// matrix <c>R(φ) = [[cos φ, −sin φ], [sin φ, cos φ]]</c>:
/// </para>
/// <code>
/// v'[2k]   = cos·v[2k] − sin·v[2k+1]
/// v'[2k+1] = sin·v[2k] + cos·v[2k+1]
/// </code>
/// <para>
/// <b>Shape conventions (matched to <c>VikramKarLex/mamba3-minimal</c>'s
/// <c>apply_rope</c>):</b>
/// </para>
/// <list type="bullet">
///   <item><description><c>b</c>, <c>c</c> have shape <c>[T, n_head, d_state]</c> row-major.
///         The caller broadcasts any <c>[T, n_group, d_state]</c> pre-QkNorm tensor to
///         the per-head layout before calling (typically via the learnable <c>B_bias</c>
///         / <c>C_bias</c> add of shape <c>[n_head, d_state]</c>, which implicitly
///         performs the broadcast).</description></item>
///   <item><description><c>dt</c> is <c>[T, n_head]</c> post-softplus (same convention
///         as <see cref="Mamba2SelectiveScan"/>).</description></item>
///   <item><description><c>theta</c> is <c>[T, d_state/2]</c> — projected from the
///         input token (shared across heads). The task brief assumed a learned
///         <c>[n_head, d_state/2]</c> table; the reference impl uses a per-token
///         projection instead, and this kernel follows the reference.</description></item>
///   <item><description>Pair ordering is <b>interleaved</b> (even/odd adjacent), matching
///         <see cref="RoPE.ApplyRotation"/> (the "Norm" pairing) rather than GPT-NeoX
///         half-split pairing. The reference's <c>x[..., 0::2]</c> / <c>x[..., 1::2]</c>
///         corresponds exactly to this.</description></item>
///   <item><description>Angle sign is <b>negative</b> cumulative sum, locking the
///         VikramKarLex/mamba3-minimal convention.</description></item>
/// </list>
/// <para>
/// <b>Alias safety.</b> <c>b</c> and <c>c</c> are rotated in place; each
/// element is read and overwritten within the same inner-pair iteration
/// (even and odd of the same pair are read into locals before either is
/// written), so in-place is safe. <c>b</c> and <c>c</c> may be the same span
/// — the two rotations operate identically on their own buffer without
/// cross-interference. <c>dt</c> and <c>theta</c> are read-only and must not
/// overlap the output buffers.
/// </para>
/// </remarks>
public static class Mamba3DataRoPE
{
    /// <summary>Stackalloc threshold in floats for the per-sequence scratch buffer.</summary>
    private const int StackAllocFloatThreshold = 2048;

    /// <summary>
    /// Applies Mamba-3 data-dependent RoPE to <paramref name="b"/> and <paramref name="c"/>
    /// in place. Both are shape <c>[T, n_head, d_state]</c> row-major.
    /// </summary>
    /// <param name="b">
    /// Mamba-3 <c>B</c> coefficient, shape <c>[T, n_head, d_state]</c> row-major,
    /// length <c>T * n_head * d_state</c>. Modified in place.
    /// </param>
    /// <param name="c">
    /// Mamba-3 <c>C</c> coefficient, shape <c>[T, n_head, d_state]</c> row-major,
    /// length <c>T * n_head * d_state</c>. Modified in place.
    /// </param>
    /// <param name="dt">
    /// Post-softplus timestep, shape <c>[T, n_head]</c> row-major, length <c>T * n_head</c>.
    /// </param>
    /// <param name="theta">
    /// Per-token frequency projection, shape <c>[T, d_state/2]</c> row-major,
    /// length <c>T * d_state / 2</c>. Shared across heads.
    /// </param>
    /// <param name="seqLen">Number of tokens <c>T</c>.</param>
    /// <param name="nHead">Number of SSM heads.</param>
    /// <param name="dState">State width (must be even).</param>
    [SkipLocalsInit]
    public static void Execute(
        Span<float> b,
        Span<float> c,
        ReadOnlySpan<float> dt,
        ReadOnlySpan<float> theta,
        int seqLen,
        int nHead,
        int dState)
    {
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead));
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));
        if ((dState & 1) != 0)
            throw new ArgumentException($"d_state must be even for 2-D pair rotation, got {dState}.", nameof(dState));

        int halfDState = dState >> 1;
        long bcLen = (long)seqLen * nHead * dState;
        long dtLen = (long)seqLen * nHead;
        long thetaLen = (long)seqLen * halfDState;

        if (b.Length < bcLen)
            throw new ArgumentException($"b length {b.Length} < T*n_head*d_state = {bcLen}.", nameof(b));
        if (c.Length < bcLen)
            throw new ArgumentException($"c length {c.Length} < T*n_head*d_state = {bcLen}.", nameof(c));
        if (dt.Length < dtLen)
            throw new ArgumentException($"dt length {dt.Length} < T*n_head = {dtLen}.", nameof(dt));
        if (theta.Length < thetaLen)
            throw new ArgumentException($"theta length {theta.Length} < T*d_state/2 = {thetaLen}.", nameof(theta));

        if (seqLen == 0) return;

        // Scratch for cum_angles of ONE time step at a time: [n_head, halfDState].
        // The recurrence is along time, so we can't vectorize cumsum across t without
        // materializing the whole table. We maintain the running cumulative angles in
        // a single [n_head * halfDState] buffer and reuse it across tokens.
        int scratchLen = nHead * halfDState;
        int trigLen = scratchLen; // cos/sin computed from the same buffer shape per step

        // Stack vs pool based on total bytes needed (cum + cos + sin).
        int totalFloats = scratchLen + 2 * trigLen;
        if (totalFloats <= StackAllocFloatThreshold)
        {
            Span<float> scratch = stackalloc float[totalFloats];
            ExecuteInto(b, c, dt, theta, seqLen, nHead, halfDState, scratch);
        }
        else
        {
            float[] rented = ArrayPool<float>.Shared.Rent(totalFloats);
            try
            {
                ExecuteInto(b, c, dt, theta, seqLen, nHead, halfDState,
                            rented.AsSpan(0, totalFloats));
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    [SkipLocalsInit]
    private static void ExecuteInto(
        Span<float> b,
        Span<float> c,
        ReadOnlySpan<float> dt,
        ReadOnlySpan<float> theta,
        int seqLen,
        int nHead,
        int halfDState,
        Span<float> scratch)
    {
        int scratchLen = nHead * halfDState;
        Span<float> cumAngles = scratch.Slice(0, scratchLen);
        Span<float> cosTable  = scratch.Slice(scratchLen, scratchLen);
        Span<float> sinTable  = scratch.Slice(2 * scratchLen, scratchLen);
        cumAngles.Clear();

        int dState = halfDState * 2;
        int bcRowStride = nHead * dState;    // stride to the next token in B / C
        int bcHeadStride = dState;           // stride to the next head within a token
        int dtRowStride = nHead;
        int thetaRowStride = halfDState;

        for (int t = 0; t < seqLen; t++)
        {
            ReadOnlySpan<float> dtRow = dt.Slice(t * dtRowStride, nHead);
            ReadOnlySpan<float> thetaRow = theta.Slice(t * thetaRowStride, halfDState);

            // Update cumulative angles:
            //   cum_angles[t, h, k] = cum_angles[t-1, h, k] - dt[t, h] * theta[t, k]
            // Per-head: fused multiply-subtract across k lanes; theta[t, :] is the vector
            // and dt[t, h] is the broadcast scalar.
            for (int h = 0; h < nHead; h++)
            {
                Span<float> cumRow = cumAngles.Slice(h * halfDState, halfDState);
                float scale = -dtRow[h];
                // cum += scale * theta  (scale is negative, implementing the NEGATIVE cumsum).
                TensorPrimitives.MultiplyAdd(thetaRow, scale, cumRow, cumRow);
            }

            // Trig: cos/sin over the entire [n_head, halfDState] block at once.
            TensorPrimitives.Cos(cumAngles, cosTable);
            TensorPrimitives.Sin(cumAngles, sinTable);

            // Apply rotation to B and C for this token, per head.
            int bcTokenBase = t * bcRowStride;
            for (int h = 0; h < nHead; h++)
            {
                int bcBase = bcTokenBase + h * bcHeadStride;
                int trigBase = h * halfDState;

                Span<float> bHead = b.Slice(bcBase, dState);
                Span<float> cHead = c.Slice(bcBase, dState);
                ReadOnlySpan<float> cosSlice = cosTable.Slice(trigBase, halfDState);
                ReadOnlySpan<float> sinSlice = sinTable.Slice(trigBase, halfDState);

                ApplyPairRotation(bHead, cosSlice, sinSlice, halfDState);
                ApplyPairRotation(cHead, cosSlice, sinSlice, halfDState);
            }
        }
    }

    /// <summary>
    /// Applies interleaved pair rotation in place to a single <c>[d_state]</c> slice.
    /// Same convention as <see cref="RoPE.ApplyRotation"/> (pairs are <c>(v[2k], v[2k+1])</c>).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    [SkipLocalsInit]
    private static void ApplyPairRotation(
        Span<float> vec,
        ReadOnlySpan<float> cos,
        ReadOnlySpan<float> sin,
        int halfDim)
    {
        // Read both halves into locals before writing — alias-safe on the same buffer.
        for (int k = 0; k < halfDim; k++)
        {
            float e = vec[2 * k];
            float o = vec[2 * k + 1];
            float co = cos[k];
            float si = sin[k];
            vec[2 * k]     = co * e - si * o;
            vec[2 * k + 1] = si * e + co * o;
        }
    }

    /// <summary>
    /// Scalar reference implementation kept for unit-test pinning. Identical numerics
    /// to <see cref="Execute"/> but without the <see cref="TensorPrimitives"/>
    /// fused-multiply-add or batched trig calls — every operation is a plain scalar
    /// floating-point op, so this is the ground truth.
    /// </summary>
    [SkipLocalsInit]
    internal static void ExecuteScalar(
        Span<float> b,
        Span<float> c,
        ReadOnlySpan<float> dt,
        ReadOnlySpan<float> theta,
        int seqLen,
        int nHead,
        int dState)
    {
        if ((dState & 1) != 0)
            throw new ArgumentException($"d_state must be even for 2-D pair rotation, got {dState}.", nameof(dState));
        int halfDState = dState >> 1;

        // Per-head per-lane running cumulative sum (negative). Scalar reference is
        // only invoked from unit tests on tiny inputs, but we still honor the
        // project-wide "no managed tensor allocations" rule via ArrayPool fallback.
        int scratchLen = nHead * halfDState;
        float[] cumAnglesArr = System.Buffers.ArrayPool<float>.Shared.Rent(scratchLen);
        Span<float> cumAngles = cumAnglesArr.AsSpan(0, scratchLen);
        cumAngles.Clear();
        try
        {
            ExecuteScalarInto(b, c, dt, theta, seqLen, nHead, halfDState, cumAngles);
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(cumAnglesArr);
        }
    }

    [SkipLocalsInit]
    private static void ExecuteScalarInto(
        Span<float> b,
        Span<float> c,
        ReadOnlySpan<float> dt,
        ReadOnlySpan<float> theta,
        int seqLen,
        int nHead,
        int halfDState,
        Span<float> cumAngles)
    {
        int dState = halfDState * 2;

        for (int t = 0; t < seqLen; t++)
        {
            int dtBase = t * nHead;
            int thetaBase = t * halfDState;
            int bcTokenBase = t * nHead * dState;

            for (int h = 0; h < nHead; h++)
            {
                float dtv = dt[dtBase + h];
                int cumBase = h * halfDState;
                int bcBase = bcTokenBase + h * dState;

                for (int k = 0; k < halfDState; k++)
                {
                    // Running negative cumulative sum: angle_{t,h,k} = -sum_{s<=t}(dt[s,h] * theta[s,k])
                    float thk = theta[thetaBase + k];
                    cumAngles[cumBase + k] -= dtv * thk;
                    float phi = cumAngles[cumBase + k];
                    float co = MathF.Cos(phi);
                    float si = MathF.Sin(phi);

                    // Rotate B pair.
                    float be = b[bcBase + 2 * k];
                    float bo = b[bcBase + 2 * k + 1];
                    b[bcBase + 2 * k]     = co * be - si * bo;
                    b[bcBase + 2 * k + 1] = si * be + co * bo;

                    // Rotate C pair (same angle).
                    float ce = c[bcBase + 2 * k];
                    float co2 = c[bcBase + 2 * k + 1];
                    c[bcBase + 2 * k]     = co * ce - si * co2;
                    c[bcBase + 2 * k + 1] = si * ce + co * co2;
                }
            }
        }
    }
}
