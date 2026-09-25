using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Single Mamba-3 SSM block — canonical <c>state-spaces/mamba</c> semantics
/// (Dao AI Lab / Goombalab <c>mamba_ssm.modules.mamba3</c> commit <c>7438488</c>).
/// Composes <see cref="Mamba3QkNorm"/>, <see cref="Mamba3DataRoPE.ExecuteCanonical"/>,
/// and <see cref="Mamba3CanonicalSsd"/> into an end-to-end
/// <c>u [T, d_model] → y [T, d_model]</c> forward.
/// </summary>
/// <remarks>
/// <para>
/// <b>Canonical forward pipeline</b> (mirrors
/// <c>capture_fixtures_canonical.py</c> line-for-line):
/// </para>
/// <code>
///   proj = u @ in_proj.weight^T                                   // GEMM
///   split(proj) → [z_raw, x_raw, B_raw, C_raw, dd_dt, dd_A, trap_raw, angles_raw]
///     widths:       d_inner, d_inner, d_state·G·R, d_state·G·R,
///                   nheads, nheads, nheads, num_rope_angles
///
///   z = reshape(z_raw, [T, H, P]);   x = reshape(x_raw, [T, H, P])
///   B = reshape(B_raw, [T, R, G, N]); C = reshape(C_raw, [T, R, G, N])
///
///   _A  = -softplus(dd_A);  _A = clamp(_A, max = -A_floor)        // [T, H]
///   DT  = softplus(dd_dt + dt_bias)                               // [T, H]
///   ADT = _A · DT
///   trap = sigmoid(trap_raw)                                      // [T, H]
///   γ            = DT · trap                                       // [T, H]
///   shifted_γ[t] = DT[t+1] · (1 − trap[t+1])     (0 at boundary)   // [T, H]
///   scale        = γ + shifted_γ                                   // [T, H]
///
///   B_norm = RMSNorm(B, B_norm_weight)  per last-axis slice
///   C_norm = RMSNorm(C, C_norm_weight)  per last-axis slice
///
///   B_biased = B_norm + B_bias   (broadcast G→H when G=1)         // [T, R, H, N]
///   C_biased = C_norm + C_bias
///
///   qk_pre_dot = Σ_n (C_biased · B_biased)      per (t,r,h)        // [T, R, H]
///
///   angles = tanh(angles_raw) · π                                  // inline
///   cum[t,h,s] = cum[t-1,h,s] + angles[t,s] · DT[t,h]   (mod 2π)   // [T, H, S]
///   (B_roped, C_roped) = RoPE(B_biased, C_biased, cum)             // per-rank, pairwise(SISO)/halved(MIMO)
///
///   SISO: y, state = CanonicalSsd_SISO(state, x, C_roped, B_roped,
///                       qk_pre_dot, scale, γ, ADT, D, z)           // silu(z) inside kernel
///   MIMO: y, state, y_pre = CanonicalSsd_MIMO(..., qk_pre_dot_sum, mimo_z, mimo_o)
///
///   y_final = y @ out_proj.weight^T                                // GEMM
/// </code>
/// <para>
/// <b>Decode continuity.</b> Two persistent buffers cross call boundaries:
/// <c>ssm_state [H, P, N]</c> and <c>cum_angle [H, S]</c>. Both are threaded
/// in / out so the block can resume mid-sequence. There is no <c>prev_Bx</c>
/// buffer in the canonical algorithm (the trapezoidal boundary at the chunk
/// edge is encoded via <c>shifted_γ[T-1] = 0</c>; callers that want to stitch
/// chunks without this boundary artefact must supply a lookahead via
/// <c>shifted_γ</c> themselves — see <see cref="Mamba3CanonicalSsd"/> XML).
/// </para>
/// <para>
/// <b>Alias safety.</b> All <c>ReadOnlySpan</c> inputs are read-only. <c>y</c>
/// must not overlap any input. <c>ssmState</c> and <c>cumAngle</c> are
/// read/written in place. Per-call temporaries (in_proj output, split slices,
/// DT / ADT / gamma / scale, biased B / C, qk_pre_dot, yScan) are drawn from
/// the caller-owned <see cref="Mamba3ForwardScratch"/>; the scratch is
/// allocated once by the model and reused across every layer / every Forward
/// call, eliminating the per-call managed-array allocations from earlier
/// stages.
/// </para>
/// </remarks>
public static class Mamba3Block
{
    // ────────────────────────────────────────────────────────────────────────
    // SISO
    // ────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Overload of <see cref="Forward(Mamba3ForwardScratch, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, Span{float}, Span{float}, Span{float}, Span{float}, Span{float}, int, int, int, int, int, int, int, int, float, float)"/>
    /// that omits the <c>kState</c>/<c>vState</c> streaming-boundary buffers.
    /// Equivalent to passing empty spans for both — so one-shot prefill (or
    /// split prefills that accept the canonical chunk-edge shifted_γ=0 drift)
    /// can keep using the shorter call signature.
    /// </summary>
    [SkipLocalsInit]
    public static void Forward(
        Mamba3ForwardScratch scratch,
        ReadOnlySpan<float> u,
        ReadOnlySpan<float> inProjWeight,
        ReadOnlySpan<float> outProjWeight,
        ReadOnlySpan<float> dtBias,
        ReadOnlySpan<float> bNormWeight,
        ReadOnlySpan<float> cNormWeight,
        ReadOnlySpan<float> bBias,
        ReadOnlySpan<float> cBias,
        ReadOnlySpan<float> d,
        Span<float> y,
        Span<float> ssmState,
        Span<float> cumAngle,
        int seqLen,
        int dModel,
        int dInner,
        int nHead,
        int headDim,
        int dState,
        int numBcHeads,
        int numRopeAngles,
        float aFloor,
        float normEps = 1e-5f)
        => Forward(
            scratch, u, inProjWeight, outProjWeight,
            dtBias, bNormWeight, cNormWeight, bBias, cBias, d,
            y, ssmState, cumAngle,
            kState: Span<float>.Empty, vState: Span<float>.Empty,
            seqLen, dModel, dInner, nHead, headDim, dState,
            numBcHeads, numRopeAngles, aFloor, normEps);

    /// <summary>
    /// Runs one canonical Mamba-3 SISO block forward.
    /// </summary>
    /// <param name="u">Input, shape <c>[T, d_model]</c> row-major.</param>
    /// <param name="inProjWeight">In-projection, shape <c>[d_in_proj, d_model]</c> row-major.
    /// <c>d_in_proj = 2·d_inner + 2·(d_state·num_bc_heads) + 3·n_head + num_rope_angles</c>.</param>
    /// <param name="outProjWeight">Out-projection, shape <c>[d_model, d_inner]</c> row-major.</param>
    /// <param name="dtBias">Per-head <c>dt</c> bias added pre-softplus, length <c>n_head</c>.</param>
    /// <param name="bNormWeight">RMSNorm weight applied to each <c>d_state</c> slice of B, length <c>d_state</c>.</param>
    /// <param name="cNormWeight">RMSNorm weight applied to each <c>d_state</c> slice of C, length <c>d_state</c>.</param>
    /// <param name="bBias">B bias, shape <c>[n_head, num_bc_heads, d_state]</c> row-major.</param>
    /// <param name="cBias">C bias, shape <c>[n_head, num_bc_heads, d_state]</c> row-major.</param>
    /// <param name="d">Per-head D skip coefficient, length <c>n_head</c>.</param>
    /// <param name="y">Output, shape <c>[T, d_model]</c> row-major. Written.</param>
    /// <param name="ssmState">SSM hidden state, shape <c>[n_head, head_dim, d_state]</c>. In-place.</param>
    /// <param name="cumAngle">Cumulative RoPE angle, shape <c>[n_head, num_rope_angles]</c>. In-place.
    /// Pass an empty span to start from zero and discard the final angle.</param>
    /// <param name="kState">
    /// Previous chunk's last-token post-RoPE (pre-scale) K, shape
    /// <c>[n_head, d_state]</c>. In-place. When non-empty, the block applies
    /// the canonical chunk-boundary adjustment
    /// <c>ssm_state += v_state · k_state · DT[0] · (1 - trap[0])</c>
    /// BEFORE running the SSD scan (matches
    /// <c>mamba3_siso_fwd.py:341-352</c>), and writes this chunk's last-token
    /// post-RoPE K on exit. Pass an empty span to disable streaming-decode
    /// boundary handling — equivalent to a one-shot forward with no prior
    /// chunk. On the first chunk of a sequence <c>kState</c> is all-zero, so
    /// the adjustment is a no-op by construction.
    /// </param>
    /// <param name="vState">
    /// Previous chunk's last-token V (= <c>x</c>), shape
    /// <c>[n_head, head_dim]</c>. Paired with <paramref name="kState"/> in the
    /// boundary adjustment. Same empty-span semantics. Updated to this chunk's
    /// last-token V on exit.
    /// </param>
    /// <param name="seqLen">Token count T.</param>
    /// <param name="dModel">Model dimension.</param>
    /// <param name="dInner">Inner dimension (<c>n_head · head_dim</c>).</param>
    /// <param name="nHead">Number of SSM heads.</param>
    /// <param name="headDim">Channels per head.</param>
    /// <param name="dState">State width.</param>
    /// <param name="numBcHeads">B/C group count G (typically 1 in known checkpoints).</param>
    /// <param name="numRopeAngles">Number of rotated pairs S (rotates first <c>2·S</c> channels).</param>
    /// <param name="aFloor">Floor for <c>-A</c> clamp (<c>A ≤ -aFloor</c>). Typical <c>1e-4</c>.</param>
    /// <param name="normEps">RMSNorm stabilising constant (default 1e-5).</param>
    /// <param name="scratch">
    /// Caller-owned pooled scratch. Sized on demand via
    /// <see cref="Mamba3ForwardScratch.EnsureCapacity(int)"/>. Safe to share a
    /// single instance across every layer and every Forward call for a given
    /// model.
    /// </param>
    [SkipLocalsInit]
    public static void Forward(
        Mamba3ForwardScratch scratch,
        ReadOnlySpan<float> u,
        ReadOnlySpan<float> inProjWeight,
        ReadOnlySpan<float> outProjWeight,
        ReadOnlySpan<float> dtBias,
        ReadOnlySpan<float> bNormWeight,
        ReadOnlySpan<float> cNormWeight,
        ReadOnlySpan<float> bBias,
        ReadOnlySpan<float> cBias,
        ReadOnlySpan<float> d,
        Span<float> y,
        Span<float> ssmState,
        Span<float> cumAngle,
        Span<float> kState,
        Span<float> vState,
        int seqLen,
        int dModel,
        int dInner,
        int nHead,
        int headDim,
        int dState,
        int numBcHeads,
        int numRopeAngles,
        float aFloor,
        float normEps = 1e-5f)
    {
        ArgumentNullException.ThrowIfNull(scratch);
        ValidateCommon(seqLen, dModel, dInner, nHead, headDim, dState,
                       numBcHeads, numRopeAngles);

        const int R = 1;                                              // SISO
        int bcPerToken = dState * numBcHeads * R;
        int dInProj = 2 * dInner + 2 * bcPerToken + 3 * nHead + numRopeAngles;

        // ── Scratch ─────────────────────────────────────────────────────────
        // Pooled, 64-byte-aligned, owned by the caller. Span views are sized
        // to the current seqLen — the backing allocation may be larger
        // (power-of-two growth) but that does not affect any slice maths.
        scratch.EnsureCapacity(seqLen);
        Span<float> proj = scratch.Proj(seqLen);
        Span<float> xBuf = scratch.X(seqLen);                          // [T, H*P]
        Span<float> zBuf = scratch.Z(seqLen);                          // [T, H*P]
        Span<float> dt = scratch.Dt(seqLen);                           // DT
        Span<float> adt = scratch.Adt(seqLen);                         // _A · DT
        Span<float> trap = scratch.Trap(seqLen);
        Span<float> gamma = scratch.Gamma(seqLen);
        Span<float> scale = scratch.Scale(seqLen);
        Span<float> anglesRaw = scratch.AnglesRaw(seqLen);
        // Expanded to nHead during SSD — canonical forward keeps B/C as
        // [T, R=1, H, N] because num_bc_heads==1 is expanded to H by the SSD
        // kernel. We materialise the expanded form here (R=1 collapsed away).
        // Scratch sizes B / C at T·max(1,mimoRank)·H·N — for a SISO model
        // this is exactly T·H·N so we take the span as-is; a SISO Forward on
        // a scratch built for a MIMO model would never run (MIMO dispatches
        // via ForwardMimo).
        Span<float> bHRN = scratch.B(seqLen).Slice(0, seqLen * nHead * dState);
        Span<float> cHRN = scratch.C(seqLen).Slice(0, seqLen * nHead * dState);
        Span<float> qkPreDot = scratch.QkPreDot(seqLen);               // [T, H]
        Span<float> yScan = scratch.YScan(seqLen);                     // [T, H*P]

        // ── Step 1: in_proj GEMM ─────────────────────────────────────────────
        GemmF32(inProjWeight, u, proj, m: dInProj, k: dModel, n: seqLen);

        // ── Step 2: split + per-token preprocess ─────────────────────────────
        //   in_proj slice layout (per token):
        //     [0, d_inner)                     z_raw
        //     [d_inner, 2·d_inner)             x_raw
        //     [2·d_inner, +bcPerToken)         B_raw  (flat [R, G, N] row-major)
        //     [... +bcPerToken)                C_raw
        //     [... +n_head)                    dd_dt
        //     [... +n_head)                    dd_A
        //     [... +n_head)                    trap_raw
        //     [... +num_rope_angles)           angles_raw
        int ofsZ = 0;
        int ofsX = dInner;
        int ofsB = 2 * dInner;
        int ofsC = ofsB + bcPerToken;
        int ofsDdDt = ofsC + bcPerToken;
        int ofsDdA = ofsDdDt + nHead;
        int ofsTrap = ofsDdA + nHead;
        int ofsAngles = ofsTrap + nHead;

        // We need the B_raw / C_raw per-token slices broadcast to [T, H, N] AND
        // the biased versions. Capture the RMSNormed + biased tensors directly
        // into bHRN / cHRN (canonical: norm over each [N] slice, then add bias,
        // then take the (C+bias)·(B+bias) dot for qk_pre_dot, then RoPE).
        for (int t = 0; t < seqLen; t++)
        {
            int src = t * dInProj;

            // z, x copies for the current token.
            proj.Slice(src + ofsZ, dInner).CopyTo(zBuf.Slice(t * dInner, dInner));
            proj.Slice(src + ofsX, dInner).CopyTo(xBuf.Slice(t * dInner, dInner));

            // DT[t,h] = softplus(dd_dt[t,h] + dt_bias[h])
            // _A[t,h] = max(-aFloor, -softplus(dd_A[t,h]))
            // ADT[t,h] = _A · DT
            // trap[t,h] = sigmoid(trap_raw[t,h])
            for (int h = 0; h < nHead; h++)
            {
                float ddDt = proj[src + ofsDdDt + h];
                float ddA = proj[src + ofsDdA + h];
                float trp = proj[src + ofsTrap + h];

                float dtv = SoftPlus(ddDt + dtBias[h]);
                float aVal = -SoftPlus(ddA);
                if (aVal > -aFloor) aVal = -aFloor;    // clamp(max = -aFloor)

                dt[t * nHead + h] = dtv;
                adt[t * nHead + h] = aVal * dtv;
                float tv = Sigmoid(trp);
                trap[t * nHead + h] = tv;
                gamma[t * nHead + h] = dtv * tv;       // γ = DT · trap
            }

            // angles_raw[t,:] (shared across heads; rotation multiplies by DT per head
            // inside ExecuteCanonical).
            proj.Slice(src + ofsAngles, numRopeAngles)
                .CopyTo(anglesRaw.Slice(t * numRopeAngles, numRopeAngles));

            // B/C per-(G,N) slice normalization + broadcast to [H, N] + bias add.
            // Canonical layout of B_raw within a token is [R=1, G, N] row-major.
            // With G == num_bc_heads == 1 we have a single [N] slice per token.
            // We RMS-normalize each [N] slice, then broadcast G → H and add bias.
            for (int g = 0; g < numBcHeads; g++)
            {
                int bSrcBase = src + ofsB + g * dState;
                int cSrcBase = src + ofsC + g * dState;
                // RMSNorm(slice[N]) · norm_weight
                RmsNormInto(proj.Slice(bSrcBase, dState), bNormWeight, normEps,
                            out float bInvRms);
                RmsNormInto(proj.Slice(cSrcBase, dState), cNormWeight, normEps,
                            out float cInvRms);

                // Broadcast every group slice to the H heads it maps to
                // (canonical: expand(-1, -1, R, H, N) from ngroups=G with H = nHead).
                // With G = 1 this duplicates to all heads. With G > 1 the mapping
                // would be groups-of-heads (H / G per group) — canonical currently
                // ships with G == 1 on the 370M checkpoint; defer multi-group support.
                int headsPerGroup = nHead / numBcHeads;
                for (int hInGroup = 0; hInGroup < headsPerGroup; hInGroup++)
                {
                    int h = g * headsPerGroup + hInGroup;
                    int biasBase = (h * numBcHeads + g) * dState;     // bBias [H, G, N]
                    int dstBase = (t * nHead + h) * dState;

                    for (int n = 0; n < dState; n++)
                    {
                        float bv = proj[bSrcBase + n] * bInvRms * bNormWeight[n] + bBias[biasBase + n];
                        float cv = proj[cSrcBase + n] * cInvRms * cNormWeight[n] + cBias[biasBase + n];
                        bHRN[dstBase + n] = bv;
                        cHRN[dstBase + n] = cv;
                    }
                }
            }
        }

        // ── Step 3: qk_pre_dot[t, h] = Σ_n (C_biased · B_biased) pre-RoPE ───
        for (int t = 0; t < seqLen; t++)
        {
            int baseT = t * nHead * dState;
            for (int h = 0; h < nHead; h++)
            {
                ReadOnlySpan<float> bh = bHRN.Slice(baseT + h * dState, dState);
                ReadOnlySpan<float> ch = cHRN.Slice(baseT + h * dState, dState);
                qkPreDot[t * nHead + h] = TensorPrimitives.Dot(ch, bh);
            }
        }

        // ── Step 4: shifted_γ[t, h] = DT[t+1] · (1 - trap[t+1]);   0 at T-1 ─
        //           scale[t, h]     = γ[t, h] + shifted_γ[t, h]
        for (int t = 0; t < seqLen; t++)
        {
            for (int h = 0; h < nHead; h++)
            {
                float sh = 0f;
                if (t + 1 < seqLen)
                {
                    int next = (t + 1) * nHead + h;
                    sh = dt[next] * (1f - trap[next]);
                }
                scale[t * nHead + h] = gamma[t * nHead + h] + sh;
            }
        }

        // ── Step 5: canonical data-RoPE (pairwise for SISO) ─────────────────
        Mamba3DataRoPE.ExecuteCanonical(
            bHRN, cHRN, anglesRaw, dt,
            cumAnglePrev: cumAngle,
            cumAngleOut: cumAngle,
            seqLen, nRank: R, nHead, dState, numRopeAngles,
            Mamba3RoPEMode.Pairwise);

        // ── Step 5.5: streaming-decode chunk-boundary adjustment ────────────
        // Canonical mamba3_siso_fwd.py:341-352: at the start of a chunk that
        // carries a prior (k_state, v_state) pair, inject
        //   ssm_state += v_state · k_state · DT[0] · (1 - trap[0])
        // BEFORE the scan. This is the deferred shifted_γ[T_prev-1] term from
        // the previous chunk's last token — a one-shot forward would have
        // folded it in at token T_prev via scale[T_prev-1] = γ + shifted_γ;
        // the split forward sees shifted_γ[T_prev-1] = 0 (no lookahead across
        // the call) and compensates here. On the first chunk both buffers are
        // zero so the update is a no-op.
        if (!kState.IsEmpty && !vState.IsEmpty && seqLen > 0)
        {
            ApplyChunkBoundaryAdjustment(
                ssmState, kState, vState, dt, trap,
                nHead, headDim, dState);
        }

        // ── Step 6: SISO SSD scan ───────────────────────────────────────────
        // xBuf layout is [T, dInner] row-major == [T, H, P] row-major.
        Mamba3CanonicalSsd.ExecuteSiso(
            ssmState, xBuf, cHRN, bHRN, qkPreDot,
            scale, gamma, adt, d, zBuf, yScan,
            seqLen, nHead, headDim, dState);

        // ── Step 6.5: persist chunk-boundary buffers for the next call ──────
        // Canonical final_k_state (mamba3_siso_fwd.py:318-322) stores the last
        // token's POST-RoPE, PRE-SCALE K. Our bHRN is exactly that — the SSD
        // kernel applies `scale` inline as `k[n] * scl` without mutating bHRN.
        // Canonical final_v_state stores the raw V (x) of the last token; our
        // xBuf holds that directly.
        if (!kState.IsEmpty && !vState.IsEmpty && seqLen > 0)
        {
            int lastTok = seqLen - 1;
            // bHRN[lastTok, h, :] → kState[h, :]  (layout: [T, H, N] → [H, N]).
            ReadOnlySpan<float> lastK = bHRN.Slice(lastTok * nHead * dState, nHead * dState);
            lastK.CopyTo(kState);
            // xBuf[lastTok, h, :] → vState[h, :]  (layout: [T, H, P] → [H, P]).
            ReadOnlySpan<float> lastV = xBuf.Slice(lastTok * nHead * headDim, nHead * headDim);
            lastV.CopyTo(vState);
        }

        // ── Step 7: out_proj GEMM ───────────────────────────────────────────
        GemmF32(outProjWeight, yScan, y, m: dModel, k: dInner, n: seqLen);
    }

    // ────────────────────────────────────────────────────────────────────────
    // MIMO
    // ────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Runs one canonical Mamba-3 MIMO block forward (rank-R B/C expansion,
    /// halved-rotary RoPE, rank-expanded gate via <paramref name="mimoZ"/> and
    /// rank-contracted output via <paramref name="mimoO"/>). V is not
    /// rank-expanded — the canonical kernel folds the V rank fan-in into the
    /// state update.
    /// </summary>
    /// <param name="u">Input, shape <c>[T, d_model]</c> row-major.</param>
    /// <param name="inProjWeight">In-projection, shape <c>[d_in_proj, d_model]</c> row-major.
    /// <c>d_in_proj = 2·d_inner + 2·(d_state·num_bc_heads·R) + 3·n_head + num_rope_angles</c>.</param>
    /// <param name="outProjWeight">Out-projection, shape <c>[d_model, d_inner]</c> row-major.</param>
    /// <param name="dtBias">Per-head <c>dt</c> bias, length <c>n_head</c>.</param>
    /// <param name="bNormWeight">RMSNorm weight, length <c>d_state</c>.</param>
    /// <param name="cNormWeight">RMSNorm weight, length <c>d_state</c>.</param>
    /// <param name="bBias">B bias, shape <c>[n_head, R, d_state]</c>. (Note: <c>num_bc_heads</c> axis collapsed into R=mimo_rank per canonical layout; G=1 assumed.)</param>
    /// <param name="cBias">C bias, shape <c>[n_head, R, d_state]</c>.</param>
    /// <param name="d">Per-head D skip coefficient, length <c>n_head</c>.</param>
    /// <param name="mimoZ">Gate rank expansion, shape <c>[n_head, R, head_dim]</c>.</param>
    /// <param name="mimoO">Output rank contraction, shape <c>[n_head, R, head_dim]</c>.</param>
    /// <param name="y">Output, shape <c>[T, d_model]</c>. Written.</param>
    /// <param name="ssmState">SSM state, <c>[n_head, head_dim, d_state]</c>. In-place.</param>
    /// <param name="cumAngle">Cumulative RoPE angle, <c>[n_head, num_rope_angles]</c>. In-place, empty for prefill.</param>
    /// <param name="seqLen">Token count.</param>
    /// <param name="dModel">Model dimension.</param>
    /// <param name="dInner">Inner dimension.</param>
    /// <param name="nHead">Number of SSM heads.</param>
    /// <param name="headDim">Channels per head.</param>
    /// <param name="dState">State width.</param>
    /// <param name="numBcHeads">B/C group count G (typically 1).</param>
    /// <param name="numRopeAngles">Rotated-pair count S.</param>
    /// <param name="mimoRank">MIMO rank R ≥ 2.</param>
    /// <param name="aFloor">Floor for <c>-A</c> clamp.</param>
    /// <param name="normEps">RMSNorm stabilising constant.</param>
    /// <param name="scratch">
    /// Caller-owned pooled scratch. Must be sized to at least the widest
    /// <c>mimoRank</c> the model will use (the constructor takes this from
    /// <see cref="Mamba3Config"/>). Shared across layers and Forward calls.
    /// </param>
    [SkipLocalsInit]
    public static void ForwardMimo(
        Mamba3ForwardScratch scratch,
        ReadOnlySpan<float> u,
        ReadOnlySpan<float> inProjWeight,
        ReadOnlySpan<float> outProjWeight,
        ReadOnlySpan<float> dtBias,
        ReadOnlySpan<float> bNormWeight,
        ReadOnlySpan<float> cNormWeight,
        ReadOnlySpan<float> bBias,
        ReadOnlySpan<float> cBias,
        ReadOnlySpan<float> d,
        ReadOnlySpan<float> mimoZ,
        ReadOnlySpan<float> mimoO,
        Span<float> y,
        Span<float> ssmState,
        Span<float> cumAngle,
        int seqLen,
        int dModel,
        int dInner,
        int nHead,
        int headDim,
        int dState,
        int numBcHeads,
        int numRopeAngles,
        int mimoRank,
        float aFloor,
        float normEps = 1e-5f)
        => ForwardMimo(
            scratch, u, inProjWeight, outProjWeight,
            dtBias, bNormWeight, cNormWeight, bBias, cBias, d, mimoZ, mimoO,
            y, ssmState, cumAngle,
            kState: Span<float>.Empty, vState: Span<float>.Empty,
            seqLen, dModel, dInner, nHead, headDim, dState,
            numBcHeads, numRopeAngles, mimoRank, aFloor, normEps);

    /// <summary>
    /// Streaming-aware overload of <see cref="ForwardMimo(Mamba3ForwardScratch, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, Span{float}, Span{float}, Span{float}, int, int, int, int, int, int, int, int, int, float, float)"/>.
    /// Threads <paramref name="kState"/> (<c>[mimoRank, nHead, dState]</c>) and
    /// <paramref name="vState"/> (<c>[nHead, headDim]</c>) across calls so that
    /// a split schedule reproduces a one-shot MIMO forward to F32-reorder
    /// noise. See <see cref="Mamba3CanonicalSsd.ExecuteMimoStreaming"/> for
    /// the boundary math.
    /// </summary>
    [SkipLocalsInit]
    public static void ForwardMimo(
        Mamba3ForwardScratch scratch,
        ReadOnlySpan<float> u,
        ReadOnlySpan<float> inProjWeight,
        ReadOnlySpan<float> outProjWeight,
        ReadOnlySpan<float> dtBias,
        ReadOnlySpan<float> bNormWeight,
        ReadOnlySpan<float> cNormWeight,
        ReadOnlySpan<float> bBias,
        ReadOnlySpan<float> cBias,
        ReadOnlySpan<float> d,
        ReadOnlySpan<float> mimoZ,
        ReadOnlySpan<float> mimoO,
        Span<float> y,
        Span<float> ssmState,
        Span<float> cumAngle,
        Span<float> kState,
        Span<float> vState,
        int seqLen,
        int dModel,
        int dInner,
        int nHead,
        int headDim,
        int dState,
        int numBcHeads,
        int numRopeAngles,
        int mimoRank,
        float aFloor,
        float normEps = 1e-5f)
    {
        ArgumentNullException.ThrowIfNull(scratch);
        ValidateCommon(seqLen, dModel, dInner, nHead, headDim, dState,
                       numBcHeads, numRopeAngles);
        if (mimoRank < 1) throw new ArgumentOutOfRangeException(nameof(mimoRank));

        int R = mimoRank;
        int bcPerToken = dState * numBcHeads * R;
        int dInProj = 2 * dInner + 2 * bcPerToken + 3 * nHead + numRopeAngles;

        // ── Scratch ─────────────────────────────────────────────────────────
        // Pooled, 64-byte-aligned, owned by the caller. Span views are sized
        // to the current seqLen — the backing allocation may be larger
        // (power-of-two growth) but that does not affect any slice maths.
        scratch.EnsureCapacity(seqLen);
        Span<float> proj = scratch.Proj(seqLen);
        Span<float> xBuf = scratch.X(seqLen);                          // [T, H, P]
        Span<float> zBuf = scratch.Z(seqLen);                          // [T, H, P]
        Span<float> dt = scratch.Dt(seqLen);
        Span<float> adt = scratch.Adt(seqLen);
        Span<float> trap = scratch.Trap(seqLen);
        Span<float> gamma = scratch.Gamma(seqLen);
        Span<float> scale = scratch.Scale(seqLen);
        Span<float> anglesRaw = scratch.AnglesRaw(seqLen);
        // Canonical layout for the SSD kernel: [T, R, H, N]. Scratch sizes
        // B / C at T·R_max·H·N — sliced here to the exact T·R·H·N footprint
        // the current MIMO rank demands.
        Span<float> bRHN = scratch.B(seqLen).Slice(0, seqLen * R * nHead * dState);
        Span<float> cRHN = scratch.C(seqLen).Slice(0, seqLen * R * nHead * dState);
        Span<float> qkPreDotSum = scratch.QkPreDot(seqLen);            // Σ_r per (t,h)
        Span<float> yScan = scratch.YScan(seqLen);

        // ── Step 1: in_proj GEMM ─────────────────────────────────────────────
        GemmF32(inProjWeight, u, proj, m: dInProj, k: dModel, n: seqLen);

        int ofsZ = 0;
        int ofsX = dInner;
        int ofsB = 2 * dInner;
        int ofsC = ofsB + bcPerToken;
        int ofsDdDt = ofsC + bcPerToken;
        int ofsDdA = ofsDdDt + nHead;
        int ofsTrap = ofsDdA + nHead;
        int ofsAngles = ofsTrap + nHead;

        // ── Step 2: split + per-token preprocess + norm + bias ─────────────
        for (int t = 0; t < seqLen; t++)
        {
            int src = t * dInProj;

            proj.Slice(src + ofsZ, dInner).CopyTo(zBuf.Slice(t * dInner, dInner));
            proj.Slice(src + ofsX, dInner).CopyTo(xBuf.Slice(t * dInner, dInner));

            for (int h = 0; h < nHead; h++)
            {
                float ddDt = proj[src + ofsDdDt + h];
                float ddA = proj[src + ofsDdA + h];
                float trp = proj[src + ofsTrap + h];

                float dtv = SoftPlus(ddDt + dtBias[h]);
                float aVal = -SoftPlus(ddA);
                if (aVal > -aFloor) aVal = -aFloor;

                dt[t * nHead + h] = dtv;
                adt[t * nHead + h] = aVal * dtv;
                float tv = Sigmoid(trp);
                trap[t * nHead + h] = tv;
                gamma[t * nHead + h] = dtv * tv;
            }

            proj.Slice(src + ofsAngles, numRopeAngles)
                .CopyTo(anglesRaw.Slice(t * numRopeAngles, numRopeAngles));

            // B_raw / C_raw per-token layout is [R, G, N] row-major.
            // We RMS-normalize each (R, G) [N] slice, broadcast G → H, and
            // add per-(H, R, N) bias. (Canonical bias shape is [H, R, N] when
            // G = 1 — we follow the capture script which emits B_bias with
            // shape [H, R, N] for MIMO.)
            int headsPerGroup = nHead / numBcHeads;
            for (int r = 0; r < R; r++)
            {
                for (int g = 0; g < numBcHeads; g++)
                {
                    int bSrcBase = src + ofsB + (r * numBcHeads + g) * dState;
                    int cSrcBase = src + ofsC + (r * numBcHeads + g) * dState;
                    RmsNormInto(proj.Slice(bSrcBase, dState), bNormWeight, normEps,
                                out float bInvRms);
                    RmsNormInto(proj.Slice(cSrcBase, dState), cNormWeight, normEps,
                                out float cInvRms);

                    for (int hInGroup = 0; hInGroup < headsPerGroup; hInGroup++)
                    {
                        int h = g * headsPerGroup + hInGroup;
                        int biasBase = (h * R + r) * dState;           // bBias [H, R, N]
                        int dstBase = ((t * R + r) * nHead + h) * dState;

                        for (int n = 0; n < dState; n++)
                        {
                            float bv = proj[bSrcBase + n] * bInvRms * bNormWeight[n] + bBias[biasBase + n];
                            float cv = proj[cSrcBase + n] * cInvRms * cNormWeight[n] + cBias[biasBase + n];
                            bRHN[dstBase + n] = bv;
                            cRHN[dstBase + n] = cv;
                        }
                    }
                }
            }
        }

        // ── Step 3: qk_pre_dot_sum[t, h] = Σ_r Σ_n (C_biased · B_biased) ───
        for (int t = 0; t < seqLen; t++)
        {
            for (int h = 0; h < nHead; h++)
            {
                float sum = 0f;
                for (int r = 0; r < R; r++)
                {
                    int baseIdx = ((t * R + r) * nHead + h) * dState;
                    ReadOnlySpan<float> bh = bRHN.Slice(baseIdx, dState);
                    ReadOnlySpan<float> ch = cRHN.Slice(baseIdx, dState);
                    sum += TensorPrimitives.Dot(ch, bh);
                }
                qkPreDotSum[t * nHead + h] = sum;
            }
        }

        // ── Step 4: shifted_γ + scale (identical to SISO) ───────────────────
        for (int t = 0; t < seqLen; t++)
        {
            for (int h = 0; h < nHead; h++)
            {
                float sh = 0f;
                if (t + 1 < seqLen)
                {
                    int next = (t + 1) * nHead + h;
                    sh = dt[next] * (1f - trap[next]);
                }
                scale[t * nHead + h] = gamma[t * nHead + h] + sh;
            }
        }

        // ── Step 5: canonical data-RoPE (halved for MIMO) ──────────────────
        Mamba3DataRoPE.ExecuteCanonical(
            bRHN, cRHN, anglesRaw, dt,
            cumAnglePrev: cumAngle,
            cumAngleOut: cumAngle,
            seqLen, nRank: R, nHead, dState, numRopeAngles,
            Mamba3RoPEMode.Halved);

        // ── Step 6: MIMO SSD scan (streaming-aware) ────────────────────────
        // ExecuteMimoStreaming applies the canonical chunk-boundary correction
        // at entry (when kState/vState are non-empty) and writes the last-token
        // K/V at exit. With empty kState/vState it is bit-equal to ExecuteMimo.
        Mamba3CanonicalSsd.ExecuteMimoStreaming(
            ssmState, xBuf, cRHN, bRHN, qkPreDotSum,
            scale, gamma, adt, dt, trap, d, zBuf, mimoZ, mimoO,
            kState, vState,
            yScan, yPerRank: Span<float>.Empty,
            seqLen, R, nHead, headDim, dState);

        // ── Step 7: out_proj GEMM ───────────────────────────────────────────
        GemmF32(outProjWeight, yScan, y, m: dModel, k: dInner, n: seqLen);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Helpers
    // ────────────────────────────────────────────────────────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ValidateCommon(
        int seqLen, int dModel, int dInner, int nHead, int headDim, int dState,
        int numBcHeads, int numRopeAngles)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (dModel <= 0) throw new ArgumentOutOfRangeException(nameof(dModel));
        if (dInner != nHead * headDim)
            throw new ArgumentException($"dInner {dInner} != nHead*headDim {nHead * headDim}");
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));
        if (numBcHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numBcHeads));
        if (nHead % numBcHeads != 0)
            throw new ArgumentException($"nHead {nHead} not divisible by numBcHeads {numBcHeads}");
        if (numRopeAngles <= 0) throw new ArgumentOutOfRangeException(nameof(numRopeAngles));
        if (2 * numRopeAngles > dState)
            throw new ArgumentException($"2*numRopeAngles ({2 * numRopeAngles}) > dState ({dState})");
    }

    /// <summary>
    /// Scalar row-major GEMM: <c>c[n, m] = Σ_k weight[m, k] · b[n, k]</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void GemmF32(ReadOnlySpan<float> weight, ReadOnlySpan<float> b, Span<float> c,
                                 int m, int k, int n)
    {
        for (int row = 0; row < n; row++)
        {
            ReadOnlySpan<float> bRow = b.Slice(row * k, k);
            Span<float> cRow = c.Slice(row * m, m);
            for (int j = 0; j < m; j++)
            {
                ReadOnlySpan<float> wRow = weight.Slice(j * k, k);
                cRow[j] = TensorPrimitives.Dot(wRow, bRow);
            }
        }
    }

    /// <summary>
    /// Computes <c>1 / sqrt(mean(slice²) + eps)</c> for a one-dim slice —
    /// the scalar factor half of RMSNorm. Caller applies
    /// <c>slice · invRms · norm_weight</c> lane-wise.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RmsNormInto(ReadOnlySpan<float> slice, ReadOnlySpan<float> _,
                                    float eps, out float invRms)
    {
        // F32 accumulator — matches canonical rms_norm_ref's upcast semantics
        // even though our base type is already F32 here.
        double acc = 0.0;
        for (int i = 0; i < slice.Length; i++)
        {
            double v = slice[i];
            acc += v * v;
        }
        float mean = (float)(acc / slice.Length);
        invRms = 1f / MathF.Sqrt(mean + eps);
    }

    /// <summary>
    /// Applies the canonical chunk-boundary state adjustment:
    /// <c>ssm_state[h, p, n] += v_state[h, p] · k_state[h, n] · DT[0, h] · (1 - trap[0, h])</c>.
    /// Mirrors <c>mamba3_siso_fwd.py:352</c>. No-op when <paramref name="kState"/>
    /// and <paramref name="vState"/> are all-zero (first chunk of a sequence).
    /// </summary>
    /// <param name="ssmState">SSM state <c>[H, P, N]</c>, mutated in place.</param>
    /// <param name="kState">Previous chunk's last-token post-RoPE K, <c>[H, N]</c>.</param>
    /// <param name="vState">Previous chunk's last-token V, <c>[H, P]</c>.</param>
    /// <param name="dt">Per-(T, H) DT table. Only <c>dt[0, :]</c> is read.</param>
    /// <param name="trap">Per-(T, H) trap (sigmoid) table. Only <c>trap[0, :]</c> is read.</param>
    /// <param name="nHead">Head count H.</param>
    /// <param name="headDim">Channels per head P.</param>
    /// <param name="dState">State width N.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyChunkBoundaryAdjustment(
        Span<float> ssmState,
        ReadOnlySpan<float> kState,
        ReadOnlySpan<float> vState,
        ReadOnlySpan<float> dt,
        ReadOnlySpan<float> trap,
        int nHead, int headDim, int dState)
    {
        // dt / trap laid out as [T, H]; we only need the t=0 slice.
        for (int h = 0; h < nHead; h++)
        {
            float coef = dt[h] * (1f - trap[h]);
            if (coef == 0f) continue;           // trap≈1 at t=0 → pure self term, no carry.

            int kBase = h * dState;
            int vBase = h * headDim;
            int stateBase = h * headDim * dState;

            for (int p = 0; p < headDim; p++)
            {
                float vpC = vState[vBase + p] * coef;
                if (vpC == 0f) continue;
                int row = stateBase + p * dState;
                for (int n = 0; n < dState; n++)
                {
                    ssmState[row + n] += vpC * kState[kBase + n];
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float SoftPlus(float x)
    {
        if (x > 20f) return x;
        if (x < -20f) return MathF.Exp(x);
        return MathF.Log(1f + MathF.Exp(x));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
}
