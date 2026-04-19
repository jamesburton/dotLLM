using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Canonical Mamba-3 attention-style SSD scan. Mirrors the inner loop of the
/// <c>state-spaces/mamba</c> Triton kernels <c>mamba3_siso_combined</c> (SISO)
/// and <c>mamba3_mimo_combined</c> (MIMO) from commit <c>7438488</c>.
/// </summary>
/// <remarks>
/// <para>
/// Structurally this is <b>not</b> the trapezoidal-rule α/β/γ recurrence used by
/// the minimal reference (<see cref="Mamba3SelectiveScan"/>). The canonical
/// kernel instead runs an attention-like decay-state update fused with a
/// pre-rotation QK-dot skip:
/// </para>
/// <code>
///   γ_t        = DT_t · trap_t                                          // trapezoidal weight
///   shift_γ_t  = DT_{t+1} · (1 - trap_{t+1})    // 0 at t = L-1          // next-step weight
///   scale_t    = γ_t + shift_γ_t                                        // applied to K only
///
///   Q_t        = rope(C_t + C_bias, cum_angles_t)        // already rotated by caller
///   K_t        = rope(B_t + B_bias, cum_angles_t)        // already rotated by caller
///   K_scaled_t = K_t · scale_t                                          // per-head scalar
///
///   h_t[p, n]  = exp(ADT_t) · h_{t-1}[p, n]  +  V_t[p] · K_scaled_t[n]   // SSM state update
///   y_scan_t[p] = Σ_n Q_t[n] · h_t[p, n]                                // state readout
///   skip_t     = D_h + γ_t · (C_t + C_bias) · (B_t + B_bias)            // pre-RoPE dot
///   y_t[p]     = y_scan_t[p] + skip_t · V_t[p]
///   y_t       *= silu(Z_t)                                              // only if hasZ
/// </code>
/// <para>
/// This is a one-to-one port of the pure-Python reference in
/// <c>tests/.../Fixtures/Mamba3/capture_fixtures_canonical.py</c>
/// (<c>canonical_siso_scan</c> / <c>canonical_mimo_scan</c>), which was
/// validated algebraically against the Triton kernel's Phase-1 / Phase-2
/// inner loops (see <c>mamba3_siso_fwd.py:256-425</c>).
/// </para>
/// <para>
/// MIMO adds a rank axis R to B, C (pre-RoPE), expands V through <c>mimo_x</c>,
/// gates through <c>mimo_z</c> per-rank, and contracts through <c>mimo_o</c>
/// after the per-rank y is computed. The hidden state <c>h</c> stays
/// rank-contracted — canonical handles this by summing <c>K_scaled</c> over
/// the rank axis inside the state update; V enters unexpanded (the rank
/// expansion of V is folded into the scan via the K.sum trick, matching the
/// Triton kernel's internal contract).
/// </para>
/// <para>
/// <b>Decode continuity.</b> <c>state</c> is threaded in / out to let the
/// block resume mid-sequence. The canonical scan has no β-term so it does
/// not maintain a <c>prev_Bx</c> buffer like
/// the trapezoidal recurrence; the second recurrent slot is the
/// <c>shifted_gamma</c> boundary, which the block supplies already folded
/// into <c>scale</c>. For single-shot prefill this lookahead comes from the
/// next token within the same call; for streaming decode the caller must
/// pass a <c>scale</c> series that encodes the right boundary condition
/// (typically <c>scale = γ</c> at the final token of the current chunk and
/// <c>scale = shifted_γ + γ</c> stitched across the chunk boundary by the
/// caller).
/// </para>
/// <para>
/// <b>Alias safety.</b> <c>b</c> and <c>c</c> are read-only (post-RoPE).
/// <c>y</c> must not overlap any input. <c>state</c> is read/written per
/// token; caller owns its storage. Scalar reference only — SIMD deferred.
/// </para>
/// </remarks>
public static class Mamba3CanonicalSsd
{
    /// <summary>
    /// Runs the canonical SISO SSD scan over <paramref name="seqLen"/> tokens.
    /// </summary>
    /// <param name="state">
    /// SSM hidden state, shape <c>[nHead, headDim, dState]</c> row-major.
    /// Updated in place. Zero on first call of a fresh sequence.
    /// </param>
    /// <param name="v">
    /// Value / <c>x</c> per head, shape <c>[T, nHead, headDim]</c> row-major.
    /// </param>
    /// <param name="qRoped">
    /// Rotated biased C, shape <c>[T, nHead, dState]</c> row-major.
    /// </param>
    /// <param name="kRoped">
    /// Rotated biased B, shape <c>[T, nHead, dState]</c> row-major.
    /// </param>
    /// <param name="qkPreDot">
    /// Pre-RoPE dot product <c>Σ_n (C+C_bias)_n · (B+B_bias)_n</c>,
    /// shape <c>[T, nHead]</c>.
    /// </param>
    /// <param name="scale">
    /// <c>γ_t + shifted_γ_t</c>, shape <c>[T, nHead]</c>.
    /// </param>
    /// <param name="gamma">
    /// <c>DT_t · trap_t</c>, shape <c>[T, nHead]</c>.
    /// </param>
    /// <param name="adt">
    /// <c>_A_t · DT_t</c> (already negative-clamped), shape <c>[T, nHead]</c>.
    /// </param>
    /// <param name="d">Per-head skip coefficient, shape <c>[nHead]</c>.</param>
    /// <param name="z">
    /// Gate <c>Z</c> per head, shape <c>[T, nHead, headDim]</c>, or empty span
    /// to skip the silu(z) gate.
    /// </param>
    /// <param name="y">
    /// Output, shape <c>[T, nHead, headDim]</c> row-major. Written.
    /// </param>
    /// <param name="seqLen">Number of tokens T.</param>
    /// <param name="nHead">Number of SSM heads.</param>
    /// <param name="headDim">Channels per head.</param>
    /// <param name="dState">State width.</param>
    [SkipLocalsInit]
    public static void ExecuteSiso(
        Span<float> state,
        ReadOnlySpan<float> v,
        ReadOnlySpan<float> qRoped,
        ReadOnlySpan<float> kRoped,
        ReadOnlySpan<float> qkPreDot,
        ReadOnlySpan<float> scale,
        ReadOnlySpan<float> gamma,
        ReadOnlySpan<float> adt,
        ReadOnlySpan<float> d,
        ReadOnlySpan<float> z,
        Span<float> y,
        int seqLen,
        int nHead,
        int headDim,
        int dState)
    {
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));

        long stateElems = (long)nHead * headDim * dState;
        long vElems = (long)seqLen * nHead * headDim;
        long bcElems = (long)seqLen * nHead * dState;
        long hdrElems = (long)seqLen * nHead;

        if (state.Length < stateElems) throw new ArgumentException("state span too small.", nameof(state));
        if (v.Length < vElems) throw new ArgumentException("v span too small.", nameof(v));
        if (qRoped.Length < bcElems) throw new ArgumentException("qRoped span too small.", nameof(qRoped));
        if (kRoped.Length < bcElems) throw new ArgumentException("kRoped span too small.", nameof(kRoped));
        if (qkPreDot.Length < hdrElems) throw new ArgumentException("qkPreDot span too small.", nameof(qkPreDot));
        if (scale.Length < hdrElems) throw new ArgumentException("scale span too small.", nameof(scale));
        if (gamma.Length < hdrElems) throw new ArgumentException("gamma span too small.", nameof(gamma));
        if (adt.Length < hdrElems) throw new ArgumentException("adt span too small.", nameof(adt));
        if (d.Length < nHead) throw new ArgumentException("d span too small.", nameof(d));
        if (!z.IsEmpty && z.Length < vElems) throw new ArgumentException("z span too small.", nameof(z));
        if (y.Length < vElems) throw new ArgumentException("y span too small.", nameof(y));

        if (seqLen == 0) return;

        bool hasZ = !z.IsEmpty;
        int bcHeadStride = dState;
        int vHeadStride = headDim;

        for (int t = 0; t < seqLen; t++)
        {
            int vTokBase = t * nHead * headDim;
            int bcTokBase = t * nHead * dState;
            int hdrTokBase = t * nHead;

            for (int h = 0; h < nHead; h++)
            {
                float decay = MathF.Exp(adt[hdrTokBase + h]);
                float scl = scale[hdrTokBase + h];
                float gm = gamma[hdrTokBase + h];
                float qkp = qkPreDot[hdrTokBase + h];
                float skip = d[h] + gm * qkp;

                int vBase = vTokBase + h * vHeadStride;
                int bcBase = bcTokBase + h * bcHeadStride;
                int stateBase = h * headDim * dState;
                ReadOnlySpan<float> q = qRoped.Slice(bcBase, dState);
                ReadOnlySpan<float> k = kRoped.Slice(bcBase, dState);
                ReadOnlySpan<float> vSlice = v.Slice(vBase, headDim);

                // Per-head h update + readout. State layout: h[p, n].
                for (int p = 0; p < headDim; p++)
                {
                    float vp = vSlice[p];
                    int stateRowBase = stateBase + p * dState;
                    float yScan = 0f;
                    for (int n = 0; n < dState; n++)
                    {
                        // h_new[p,n] = decay * h_old[p,n] + V[p] * K[n] * scale
                        float newState = decay * state[stateRowBase + n] + vp * (k[n] * scl);
                        state[stateRowBase + n] = newState;
                        yScan += q[n] * newState;
                    }

                    float yOut = yScan + skip * vp;
                    if (hasZ)
                    {
                        float zv = z[vBase + p];
                        // silu(z) = z * sigmoid(z) = z / (1 + exp(-z)).
                        float silu = zv / (1f + MathF.Exp(-zv));
                        yOut *= silu;
                    }
                    y[vBase + p] = yOut;
                }
            }
        }
    }

    /// <summary>
    /// Runs the canonical MIMO SSD scan over <paramref name="seqLen"/> tokens.
    /// Rank-expands <paramref name="z"/> through <paramref name="mimoZ"/> per
    /// rank, computes a per-rank y, then contracts the rank axis away through
    /// <paramref name="mimoO"/>.
    /// </summary>
    /// <remarks>
    /// The <paramref name="v"/> input is <em>not</em> pre-expanded — canonical
    /// folds the V rank expansion into the state update by summing the
    /// per-rank K over the rank axis and using the unexpanded V. This matches
    /// the Triton kernel contract exactly (h is shape <c>[nHead, headDim, dState]</c>,
    /// no rank dim).
    /// </remarks>
    /// <param name="state">SSM hidden state, <c>[nHead, headDim, dState]</c>. In-place.</param>
    /// <param name="v">V per head, <c>[T, nHead, headDim]</c>.</param>
    /// <param name="qRoped">Rotated biased C per rank, <c>[T, nRank, nHead, dState]</c>.</param>
    /// <param name="kRoped">Rotated biased B per rank, <c>[T, nRank, nHead, dState]</c>.</param>
    /// <param name="qkPreDotSum">Sum over rank of the pre-rotation QK dot, <c>[T, nHead]</c>.</param>
    /// <param name="scale">Per-token per-head scale, <c>[T, nHead]</c>.</param>
    /// <param name="gamma">DT·trap per-token per-head, <c>[T, nHead]</c>.</param>
    /// <param name="adt">_A·DT per-token per-head, <c>[T, nHead]</c>.</param>
    /// <param name="d">Per-head skip coefficient, <c>[nHead]</c>.</param>
    /// <param name="z">Gate, <c>[T, nHead, headDim]</c> or empty.</param>
    /// <param name="mimoZ">Gate rank expansion, <c>[nHead, nRank, headDim]</c>.</param>
    /// <param name="mimoO">Output rank contraction, <c>[nHead, nRank, headDim]</c>.</param>
    /// <param name="y">Output, <c>[T, nHead, headDim]</c>. Written.</param>
    /// <param name="yPerRank">
    /// Optional per-rank output buffer, <c>[T, nRank, nHead, headDim]</c>.
    /// Pass empty span to skip. Useful for fixture-level comparators (canonical
    /// exposes <c>y_pre_contract</c>).
    /// </param>
    /// <param name="seqLen">Token count T.</param>
    /// <param name="nRank">MIMO rank R.</param>
    /// <param name="nHead">Number of SSM heads.</param>
    /// <param name="headDim">Channels per head.</param>
    /// <param name="dState">State width.</param>
    [SkipLocalsInit]
    public static void ExecuteMimo(
        Span<float> state,
        ReadOnlySpan<float> v,
        ReadOnlySpan<float> qRoped,
        ReadOnlySpan<float> kRoped,
        ReadOnlySpan<float> qkPreDotSum,
        ReadOnlySpan<float> scale,
        ReadOnlySpan<float> gamma,
        ReadOnlySpan<float> adt,
        ReadOnlySpan<float> d,
        ReadOnlySpan<float> z,
        ReadOnlySpan<float> mimoZ,
        ReadOnlySpan<float> mimoO,
        Span<float> y,
        Span<float> yPerRank,
        int seqLen,
        int nRank,
        int nHead,
        int headDim,
        int dState)
    {
        if (nRank <= 0) throw new ArgumentOutOfRangeException(nameof(nRank));
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));

        long stateElems = (long)nHead * headDim * dState;
        long vElems = (long)seqLen * nHead * headDim;
        long bcElems = (long)seqLen * nRank * nHead * dState;
        long hdrElems = (long)seqLen * nHead;
        long mimoElems = (long)nHead * nRank * headDim;
        long yPerRankElems = (long)seqLen * nRank * nHead * headDim;

        if (state.Length < stateElems) throw new ArgumentException("state too small.", nameof(state));
        if (v.Length < vElems) throw new ArgumentException("v too small.", nameof(v));
        if (qRoped.Length < bcElems) throw new ArgumentException("qRoped too small.", nameof(qRoped));
        if (kRoped.Length < bcElems) throw new ArgumentException("kRoped too small.", nameof(kRoped));
        if (qkPreDotSum.Length < hdrElems) throw new ArgumentException("qkPreDotSum too small.", nameof(qkPreDotSum));
        if (scale.Length < hdrElems) throw new ArgumentException("scale too small.", nameof(scale));
        if (gamma.Length < hdrElems) throw new ArgumentException("gamma too small.", nameof(gamma));
        if (adt.Length < hdrElems) throw new ArgumentException("adt too small.", nameof(adt));
        if (d.Length < nHead) throw new ArgumentException("d too small.", nameof(d));
        if (!z.IsEmpty && z.Length < vElems) throw new ArgumentException("z too small.", nameof(z));
        if (mimoZ.Length < mimoElems) throw new ArgumentException("mimoZ too small.", nameof(mimoZ));
        if (mimoO.Length < mimoElems) throw new ArgumentException("mimoO too small.", nameof(mimoO));
        if (y.Length < vElems) throw new ArgumentException("y too small.", nameof(y));
        if (!yPerRank.IsEmpty && yPerRank.Length < yPerRankElems)
            throw new ArgumentException("yPerRank too small.", nameof(yPerRank));

        if (seqLen == 0) return;

        bool hasZ = !z.IsEmpty;
        bool writePerRank = !yPerRank.IsEmpty;
        int bcHeadStride = dState;
        int bcRankStride = nHead * dState;
        int bcTokStride = nRank * bcRankStride;

        // Strides into mimoZ / mimoO (shape [H, R, P] row-major).
        int mimoHeadStride = nRank * headDim;
        int mimoRankStride = headDim;

        // Inv-rank for evenly distributing the skip across ranks (canonical kernel
        // folds D / qk_dot into the per-rank output before mimo_o contraction;
        // dividing by R matches the reference canonical_mimo_scan).
        float invRank = 1f / nRank;

        for (int t = 0; t < seqLen; t++)
        {
            int vTokBase = t * nHead * headDim;
            int bcTokBase = t * bcTokStride;
            int hdrTokBase = t * nHead;
            int perRankTokBase = t * nRank * nHead * headDim;

            for (int h = 0; h < nHead; h++)
            {
                float decay = MathF.Exp(adt[hdrTokBase + h]);
                float scl = scale[hdrTokBase + h];
                float gm = gamma[hdrTokBase + h];
                float qkp = qkPreDotSum[hdrTokBase + h];
                float skip = d[h] + gm * qkp;

                int vBase = vTokBase + h * headDim;
                int stateBase = h * headDim * dState;
                ReadOnlySpan<float> vSlice = v.Slice(vBase, headDim);

                // h update: h_new[p,n] = decay * h_old[p,n] + V[p] * (Σ_r K_r[n]) * scale
                for (int p = 0; p < headDim; p++)
                {
                    float vp = vSlice[p];
                    int stateRowBase = stateBase + p * dState;
                    for (int n = 0; n < dState; n++)
                    {
                        float kSum = 0f;
                        for (int r = 0; r < nRank; r++)
                        {
                            int kIdx = bcTokBase + r * bcRankStride + h * bcHeadStride + n;
                            kSum += kRoped[kIdx];
                        }
                        float newState = decay * state[stateRowBase + n] + vp * (kSum * scl);
                        state[stateRowBase + n] = newState;
                    }
                }

                // Per-rank readout and contraction.
                for (int p = 0; p < headDim; p++)
                {
                    float vp = vSlice[p];
                    int stateRowBase = stateBase + p * dState;

                    float contracted = 0f;
                    for (int r = 0; r < nRank; r++)
                    {
                        int qBase = bcTokBase + r * bcRankStride + h * bcHeadStride;
                        // y_scan_r = Σ_n Q_r[n] * h[p, n]
                        float yScanR = 0f;
                        for (int n = 0; n < dState; n++)
                        {
                            yScanR += qRoped[qBase + n] * state[stateRowBase + n];
                        }
                        // per-rank y = y_scan_r + (skip / R) * V[p]
                        float yR = yScanR + skip * invRank * vp;

                        // Gate: silu(z_t * mimo_z[h, r, p]) — per-rank z via mimo_z rank expand.
                        if (hasZ)
                        {
                            int zIdx = vBase + p;
                            int mimoZIdx = h * mimoHeadStride + r * mimoRankStride + p;
                            float zGated = z[zIdx] * mimoZ[mimoZIdx];
                            float silu = zGated / (1f + MathF.Exp(-zGated));
                            yR *= silu;
                        }

                        if (writePerRank)
                        {
                            int perRankIdx = perRankTokBase + r * nHead * headDim + h * headDim + p;
                            yPerRank[perRankIdx] = yR;
                        }

                        // mimo_o contraction: y[h, p] += y_r * mimo_o[h, r, p]
                        int mimoOIdx = h * mimoHeadStride + r * mimoRankStride + p;
                        contracted += yR * mimoO[mimoOIdx];
                    }
                    y[vBase + p] = contracted;
                }
            }
        }
    }
}
