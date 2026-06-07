using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
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
/// read/written in place. Scratch is allocated per-call (<c>new float[]</c>) —
/// a pooled-scratch follow-up is tracked against a future stage, not this one.
/// </para>
/// </remarks>
public static class Mamba3Block
{
    // ────────────────────────────────────────────────────────────────────────
    // SISO
    // ────────────────────────────────────────────────────────────────────────

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
    [SkipLocalsInit]
    public static void Forward(
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
    {
        ValidateCommon(seqLen, dModel, dInner, nHead, headDim, dState,
                       numBcHeads, numRopeAngles);

        const int R = 1;                                              // SISO
        int bcPerToken = dState * numBcHeads * R;
        int dInProj = 2 * dInner + 2 * bcPerToken + 3 * nHead + numRopeAngles;

        // ── Scratch ─────────────────────────────────────────────────────────
        float[] proj = new float[seqLen * dInProj];
        float[] xBuf = new float[seqLen * dInner];                    // [T, H*P]
        float[] zBuf = new float[seqLen * dInner];                    // [T, H*P]
        float[] dt = new float[seqLen * nHead];                       // DT
        float[] adt = new float[seqLen * nHead];                      // _A · DT
        float[] trap = new float[seqLen * nHead];
        float[] gamma = new float[seqLen * nHead];
        float[] scale = new float[seqLen * nHead];
        float[] anglesRaw = new float[seqLen * numRopeAngles];
        // Expanded to nHead during SSD — canonical forward keeps B/C as
        // [T, R=1, H, N] because num_bc_heads==1 is expanded to H by the SSD
        // kernel. We materialise the expanded form here (R=1 collapsed away).
        float[] bHRN = new float[seqLen * nHead * dState];            // [T, H, N]
        float[] cHRN = new float[seqLen * nHead * dState];
        float[] qkPreDot = new float[seqLen * nHead];                 // [T, H]
        float[] yScan = new float[seqLen * dInner];                   // [T, H*P]

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
            new Span<float>(proj, src + ofsZ, dInner).CopyTo(zBuf.AsSpan(t * dInner, dInner));
            new Span<float>(proj, src + ofsX, dInner).CopyTo(xBuf.AsSpan(t * dInner, dInner));

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
            new Span<float>(proj, src + ofsAngles, numRopeAngles)
                .CopyTo(anglesRaw.AsSpan(t * numRopeAngles, numRopeAngles));

            // B/C per-(G,N) slice normalization + broadcast to [H, N] + bias add.
            // Canonical layout of B_raw within a token is [R=1, G, N] row-major.
            // With G == num_bc_heads == 1 we have a single [N] slice per token.
            // We RMS-normalize each [N] slice, then broadcast G → H and add bias.
            for (int g = 0; g < numBcHeads; g++)
            {
                int bSrcBase = src + ofsB + g * dState;
                int cSrcBase = src + ofsC + g * dState;
                // RMSNorm(slice[N]) · norm_weight
                RmsNormInto(proj.AsSpan(bSrcBase, dState), bNormWeight, normEps,
                            out float bInvRms);
                RmsNormInto(proj.AsSpan(cSrcBase, dState), cNormWeight, normEps,
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
                ReadOnlySpan<float> bh = bHRN.AsSpan(baseT + h * dState, dState);
                ReadOnlySpan<float> ch = cHRN.AsSpan(baseT + h * dState, dState);
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

        // ── Step 6: SISO SSD scan ───────────────────────────────────────────
        // xBuf layout is [T, dInner] row-major == [T, H, P] row-major.
        Mamba3CanonicalSsd.ExecuteSiso(
            ssmState, xBuf, cHRN, bHRN, qkPreDot,
            scale, gamma, adt, d, zBuf, yScan,
            seqLen, nHead, headDim, dState);

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
    [SkipLocalsInit]
    public static void ForwardMimo(
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
    {
        ValidateCommon(seqLen, dModel, dInner, nHead, headDim, dState,
                       numBcHeads, numRopeAngles);
        if (mimoRank < 1) throw new ArgumentOutOfRangeException(nameof(mimoRank));

        int R = mimoRank;
        int bcPerToken = dState * numBcHeads * R;
        int dInProj = 2 * dInner + 2 * bcPerToken + 3 * nHead + numRopeAngles;

        // ── Scratch ─────────────────────────────────────────────────────────
        float[] proj = new float[seqLen * dInProj];
        float[] xBuf = new float[seqLen * dInner];                    // [T, H, P]
        float[] zBuf = new float[seqLen * dInner];                    // [T, H, P]
        float[] dt = new float[seqLen * nHead];
        float[] adt = new float[seqLen * nHead];
        float[] trap = new float[seqLen * nHead];
        float[] gamma = new float[seqLen * nHead];
        float[] scale = new float[seqLen * nHead];
        float[] anglesRaw = new float[seqLen * numRopeAngles];
        // Canonical layout for the SSD kernel: [T, R, H, N].
        float[] bRHN = new float[(long)seqLen * R * nHead * dState];
        float[] cRHN = new float[(long)seqLen * R * nHead * dState];
        float[] qkPreDotSum = new float[seqLen * nHead];              // Σ_r per (t,h)
        float[] yScan = new float[seqLen * dInner];

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

            new Span<float>(proj, src + ofsZ, dInner).CopyTo(zBuf.AsSpan(t * dInner, dInner));
            new Span<float>(proj, src + ofsX, dInner).CopyTo(xBuf.AsSpan(t * dInner, dInner));

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

            new Span<float>(proj, src + ofsAngles, numRopeAngles)
                .CopyTo(anglesRaw.AsSpan(t * numRopeAngles, numRopeAngles));

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
                    RmsNormInto(proj.AsSpan(bSrcBase, dState), bNormWeight, normEps,
                                out float bInvRms);
                    RmsNormInto(proj.AsSpan(cSrcBase, dState), cNormWeight, normEps,
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
                    ReadOnlySpan<float> bh = bRHN.AsSpan(baseIdx, dState);
                    ReadOnlySpan<float> ch = cRHN.AsSpan(baseIdx, dState);
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

        // ── Step 6: MIMO SSD scan ───────────────────────────────────────────
        Mamba3CanonicalSsd.ExecuteMimo(
            ssmState, xBuf, cRHN, bRHN, qkPreDotSum,
            scale, gamma, adt, d, zBuf, mimoZ, mimoO,
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
