using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Single Mamba-3 SSM block (Lahoti et al., arXiv 2603.15569). Composes the five
/// Mamba-3 kernels (<see cref="Mamba3Discretize"/>, <see cref="Mamba3QkNorm"/>,
/// <see cref="Mamba3DataRoPE"/>, <see cref="Mamba3SelectiveScan"/>, and the
/// optional MIMO projection) into one end-to-end forward that maps
/// <c>u [T, d_model]</c> → <c>y [T, d_model]</c>.
/// </summary>
/// <remarks>
/// <para>
/// The order of operations mirrors <c>VikramKarLex/mamba3-minimal</c>'s SISO
/// forward path:
/// </para>
/// <code>
///   proj           = u @ in_proj.weight^T                          // GEMM
///   z, x, B, C, dt_raw, lam_raw, theta = split(proj, ...)
///   dt             = softplus(dt_raw + dt_bias)
///   lam            = sigmoid(lam_raw)
///   α, β, γ        = Mamba3Discretize(dt, A, lam)
///   B, C           = Mamba3QkNorm(B|C, B|C_norm_weight)
///   B              = B[:, None, :] + B_bias      // broadcast (T, n_head, d_state)
///   C              = C[:, None, :] + C_bias
///   B, C           = Mamba3DataRoPE(B, C, dt, theta)
///   y, state, prev = Mamba3SelectiveScan(state, prev, x, α, β, γ, B, C)
///   y              = y + x · D                                     // D skip
///   y              = y · silu(z)                                   // output gate
///   y_final        = y @ out_proj.weight^T                         // GEMM
/// </code>
/// <para>
/// This is a composable primitive, not a full model. Model loading / GGUF
/// dispatch is Stage D and is blocked on upstream checkpoint availability
/// (see <c>DESIGN_MAMBA_3.md</c>).
/// </para>
/// <para>
/// Scalar-first implementation — no SIMD on the glue code; kernels handle their
/// own SIMD. Scratch buffers are allocated per-call; a long-lived scratch pool
/// (analogous to <c>NemotronHForwardState</c>) is a straightforward follow-up.
/// </para>
/// </remarks>
public static class Mamba3Block
{
    /// <summary>
    /// Runs one Mamba-3 SSM block forward over <paramref name="seqLen"/> tokens.
    /// Advances <paramref name="state"/> and <paramref name="prevBx"/> in place.
    /// </summary>
    /// <param name="u">Input, shape <c>[T, d_model]</c> row-major.</param>
    /// <param name="inProjWeight">
    /// Input projection, shape <c>[d_in_proj, d_model]</c> row-major.
    /// <c>d_in_proj = 2·d_inner + 2·d_state + 2·n_head + d_state/2</c> (SISO).
    /// </param>
    /// <param name="outProjWeight">Output projection, shape <c>[d_model, d_inner]</c> row-major.</param>
    /// <param name="a">Per-head decay <c>A = -exp(A_log)</c>, length <c>n_head</c>. Already negative.</param>
    /// <param name="dtBias">Per-head <c>dt</c> bias added pre-softplus, length <c>n_head</c>.</param>
    /// <param name="bNormWeight">QK-Norm weight for B, length <c>d_state</c>.</param>
    /// <param name="cNormWeight">QK-Norm weight for C, length <c>d_state</c>.</param>
    /// <param name="bBias">Per-head B bias, shape <c>[n_head, d_state]</c> row-major.</param>
    /// <param name="cBias">Per-head C bias, shape <c>[n_head, d_state]</c> row-major.</param>
    /// <param name="d">Per-head D skip coefficient, length <c>n_head</c>.</param>
    /// <param name="y">Output, shape <c>[T, d_model]</c> row-major.</param>
    /// <param name="state">SSM recurrent state, shape <c>[n_head, head_dim, d_state]</c>. In-place.</param>
    /// <param name="prevBx">Previous-step B̄⊗x, shape <c>[n_head, head_dim, d_state]</c>. In-place.</param>
    /// <param name="seqLen">Number of tokens <c>T</c>.</param>
    /// <param name="dModel">Model dimension.</param>
    /// <param name="dInner">Inner dimension (<c>n_head · head_dim</c>).</param>
    /// <param name="nHead">Number of SSM heads.</param>
    /// <param name="headDim">Channels per head.</param>
    /// <param name="dState">SSM state width.</param>
    /// <param name="normEps">RMSNorm stabilising constant for QK-Norm (default 1e-5).</param>
    [SkipLocalsInit]
    public static void Forward(
        ReadOnlySpan<float> u,
        ReadOnlySpan<float> inProjWeight,
        ReadOnlySpan<float> outProjWeight,
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> dtBias,
        ReadOnlySpan<float> bNormWeight,
        ReadOnlySpan<float> cNormWeight,
        ReadOnlySpan<float> bBias,
        ReadOnlySpan<float> cBias,
        ReadOnlySpan<float> d,
        Span<float> y,
        Span<float> state,
        Span<float> prevBx,
        int seqLen,
        int dModel,
        int dInner,
        int nHead,
        int headDim,
        int dState,
        float normEps = 1e-5f)
        => Forward(u, inProjWeight, outProjWeight, a, dtBias, bNormWeight, cNormWeight,
                   bBias, cBias, d, y, state, prevBx, cumAngle: Span<float>.Empty,
                   seqLen, dModel, dInner, nHead, headDim, dState, normEps);

    /// <summary>
    /// Decode-aware overload: threads a third persistent state <paramref name="cumAngle"/>
    /// (the DataRoPE cumulative angle) alongside <paramref name="state"/> and
    /// <paramref name="prevBx"/>. Pass an empty span to get the original prefill
    /// behaviour (start from zeros, do not export final). Pass a buffer of length
    /// <c>n_head · d_state/2</c> for autoregressive decode: the kernel reads it
    /// at entry (as the starting angle) and overwrites it at exit with the
    /// final-token angle, so the next call resumes the rotation phase rather
    /// than resetting to 0.
    /// </summary>
    /// <remarks>
    /// Mirrors <c>VikramKarLex/mamba3-minimal</c>'s <c>InferenceCache.cum_angle</c>
    /// field, which has shape <c>(batch, n_head, d_state/2)</c>. Batch is 1 here.
    /// </remarks>
    [SkipLocalsInit]
    public static void Forward(
        ReadOnlySpan<float> u,
        ReadOnlySpan<float> inProjWeight,
        ReadOnlySpan<float> outProjWeight,
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> dtBias,
        ReadOnlySpan<float> bNormWeight,
        ReadOnlySpan<float> cNormWeight,
        ReadOnlySpan<float> bBias,
        ReadOnlySpan<float> cBias,
        ReadOnlySpan<float> d,
        Span<float> y,
        Span<float> state,
        Span<float> prevBx,
        Span<float> cumAngle,
        int seqLen,
        int dModel,
        int dInner,
        int nHead,
        int headDim,
        int dState,
        float normEps = 1e-5f)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (dInner != nHead * headDim)
            throw new ArgumentException($"dInner {dInner} != nHead*headDim {nHead * headDim}");
        if ((dState & 1) != 0)
            throw new ArgumentException($"dState {dState} must be even for data-RoPE pairs");

        int halfState = dState / 2;
        int dInProj = 2 * dInner + 2 * dState + 2 * nHead + halfState;

        // ── Scratch ──────────────────────────────────────────────────────────
        float[] proj = new float[seqLen * dInProj];
        float[] dt = new float[seqLen * nHead];
        float[] lam = new float[seqLen * nHead];
        float[] alpha = new float[seqLen * nHead];
        float[] beta = new float[seqLen * nHead];
        float[] gamma = new float[seqLen * nHead];
        float[] bBroad = new float[seqLen * nHead * dState];
        float[] cBroad = new float[seqLen * nHead * dState];
        float[] yScan = new float[seqLen * dInner];

        // ── Step 1: in_proj GEMM  u @ inProjWeight^T  →  proj ───────────────
        //   u          : [T, dModel]
        //   inProj.T   : [dModel, dInProj]   (stored as [dInProj, dModel] row-major)
        //   proj       : [T, dInProj]
        GemmF32(inProjWeight, u, proj, m: dInProj, k: dModel, n: seqLen);

        // ── Step 2: split the projection into 7 components ──────────────────
        // Layout along d_in_proj axis, per token:
        //   [0, dInner)                    z        gate
        //   [dInner, 2·dInner)             x        SSM input
        //   [2·dInner, 2·dInner+dState)    B_raw
        //   [2·dInner+dState,              C_raw
        //      2·dInner+2·dState)
        //   [...,  + nHead)                dt_raw
        //   [...,  + nHead)                lam_raw
        //   [...,  + halfState)            theta
        int bOff = 2 * dInner;
        int cOff = bOff + dState;
        int dtOff = cOff + dState;
        int lamOff = dtOff + nHead;
        int thetaOff = lamOff + nHead;

        float[] bRaw = new float[seqLen * dState];
        float[] cRaw = new float[seqLen * dState];
        float[] theta = new float[seqLen * halfState];

        for (int t = 0; t < seqLen; t++)
        {
            int src = t * dInProj;
            // dt = softplus(dt_raw + dt_bias)
            // lam = sigmoid(lam_raw)
            for (int h = 0; h < nHead; h++)
            {
                dt[t * nHead + h] = SoftPlus(proj[src + dtOff + h] + dtBias[h]);
                lam[t * nHead + h] = Sigmoid(proj[src + lamOff + h]);
            }
            // B_raw, C_raw, theta copies (contiguous slices)
            new Span<float>(proj, src + bOff, dState).CopyTo(bRaw.AsSpan(t * dState, dState));
            new Span<float>(proj, src + cOff, dState).CopyTo(cRaw.AsSpan(t * dState, dState));
            new Span<float>(proj, src + thetaOff, halfState).CopyTo(theta.AsSpan(t * halfState, halfState));
        }

        // ── Step 3: Discretize α, β, γ ──────────────────────────────────────
        Mamba3Discretize.Execute(dt, a, lam, alpha, beta, gamma, seqLen, nHead);

        // ── Step 4: QK-Norm on B, C ─────────────────────────────────────────
        Mamba3QkNorm.Execute(bRaw, bNormWeight, normEps, seqLen, nGroup: 1, dState);
        Mamba3QkNorm.Execute(cRaw, cNormWeight, normEps, seqLen, nGroup: 1, dState);

        // ── Step 5: broadcast B, C to per-head shape and add BC-bias ────────
        // Shape change: [T, dState] → [T, nHead, dState]
        // Per (t, h, k): broad[t, h, k] = Bqkn[t, k] + bBias[h, k]
        for (int t = 0; t < seqLen; t++)
        {
            int bqBase = t * dState;
            int bBroadBase = t * nHead * dState;
            for (int h = 0; h < nHead; h++)
            {
                int biasBase = h * dState;
                int headBase = bBroadBase + h * dState;
                for (int k = 0; k < dState; k++)
                {
                    bBroad[headBase + k] = bRaw[bqBase + k] + bBias[biasBase + k];
                    cBroad[headBase + k] = cRaw[bqBase + k] + cBias[biasBase + k];
                }
            }
        }

        // ── Step 6: data-dependent RoPE on B, C ─────────────────────────────
        // Mamba3DataRoPE (per agent B's doc) takes theta shape [T, dState/2],
        // rotates B/C treated as [T, nHead, dState] per-head. If the caller
        // passed a cum_angle persistent-state buffer, thread it through so the
        // rotation phase continues from the previous call's last token.
        Mamba3DataRoPE.Execute(
            bBroad, cBroad, dt, theta,
            cumAnglePrev: cumAngle,
            cumAngleOut: cumAngle,
            seqLen, nHead, dState);

        // ── Step 7: Selective scan ──────────────────────────────────────────
        // x slice from proj: [T, dInner] at offset dInner per token
        float[] x = new float[seqLen * dInner];
        for (int t = 0; t < seqLen; t++)
            new Span<float>(proj, t * dInProj + dInner, dInner).CopyTo(x.AsSpan(t * dInner, dInner));

        Mamba3SelectiveScan.Execute(state, prevBx, x, alpha, beta, gamma, bBroad, cBroad, yScan,
            nHead, headDim, dState, nGroup: nHead, seqLen);

        // ── Step 8: D skip and silu(z) gate ────────────────────────────────
        // y_with_D[t, h, p] = yScan[t, h, p] + x[t, h, p] * D[h]
        // Then y_gated[t, :] = y_with_D[t, :].flatten() * silu(z[t, :])
        float[] yGated = new float[seqLen * dInner];
        for (int t = 0; t < seqLen; t++)
        {
            int zSrc = t * dInProj;  // z is at proj[t, 0..dInner)
            for (int h = 0; h < nHead; h++)
            {
                for (int p = 0; p < headDim; p++)
                {
                    int i = h * headDim + p;
                    float ySk = yScan[t * dInner + i] + x[t * dInner + i] * d[h];
                    float zv = proj[zSrc + i];
                    yGated[t * dInner + i] = ySk * Silu(zv);
                }
            }
        }

        // ── Step 9: out_proj GEMM ──────────────────────────────────────────
        //   yGated     : [T, dInner]
        //   outProj.T  : [dInner, dModel]   (stored as [dModel, dInner] row-major)
        //   y          : [T, dModel]
        GemmF32(outProjWeight, yGated, y, m: dModel, k: dInner, n: seqLen);
    }

    /// <summary>
    /// MIMO variant of <c>Forward</c> (Lahoti et al., §3.3 + Appendix D).
    /// Rank-<paramref name="mimoRank"/> factorization of <c>B</c>, <c>C</c>, and <c>x</c>
    /// improves hardware utilisation and yields +1.2 accuracy pts over SISO at the
    /// same state width (paper, Table 4).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Order of operations mirrors the MIMO branch of
    /// <c>VikramKarLex/mamba3-minimal</c>'s <c>Mamba3.forward</c>:
    /// </para>
    /// <code>
    ///   proj   = u @ in_proj.weight^T                                         // GEMM
    ///   z, x, B_raw, C_raw, dt_raw, lam_raw, theta = split(proj, [...])      // bc_dim = d_state·R
    ///   dt     = softplus(dt_raw + dt_bias);  lam = sigmoid(lam_raw)
    ///   α, β, γ = Mamba3Discretize(dt, A, lam)
    ///   B, C   = Mamba3QkNorm(B|C, weight, ε)            // over bc_dim channels
    ///   B, C   = reshape [T, d_state·R] → per-rank slices [R][T, n_head, d_state]
    ///              via broadcast-over-heads + B_bias[h, d_state, R] / C_bias[h, d_state, R]
    ///   B_r, C_r = Mamba3DataRoPE(...)                   // per rank, same angles
    ///   x_mimo = Mamba3MimoProject.ExpandInput(x, mimo_x_proj)    // [T, h, p, R]
    ///   y_mimo, state, prev_Bx = MIMO trapezoidal scan
    ///   y_with_d[t,h,p,r] = y_mimo[t,h,p,r] + x[t,h,p]·D[h]       // D skip (broadcast over R)
    ///   z_mimo  = z_heads.unsqueeze(-1) · mimo_z_proj[h, p, R]
    ///   y_gated = y_with_d · silu(z_mimo)
    ///   y_rank  = Mamba3MimoProject.ContractOutput(y_gated, mimo_down)       // [T, h, p]
    ///   y_final = y_rank @ out_proj.weight^T                                  // GEMM
    /// </code>
    /// <para>
    /// State/prev_Bx shapes match SISO — rank is contracted before the recurrence
    /// by the MIMO einsum <c>Σ_r B[..,r] · x_mimo[..,r]</c>, so there is no
    /// per-rank state. <c>y_mimo</c> retains rank dim until the <c>mimo_down</c>
    /// contraction.
    /// </para>
    /// <para>
    /// Per-call scratch allocation (<c>new float[]</c>) matches the SISO path;
    /// a pooled scratch-state pass is a separate, straightforward follow-up.
    /// </para>
    /// </remarks>
    /// <param name="u">Input, shape <c>[T, d_model]</c> row-major.</param>
    /// <param name="inProjWeight">
    /// Input projection, shape <c>[d_in_proj, d_model]</c> row-major.
    /// <c>d_in_proj = 2·d_inner + 2·(d_state·R) + 2·n_head + d_state/2</c>.
    /// </param>
    /// <param name="outProjWeight">Output projection, shape <c>[d_model, d_inner]</c> row-major.</param>
    /// <param name="a">Per-head decay <c>A = -exp(A_log)</c>, length <c>n_head</c>.</param>
    /// <param name="dtBias">Per-head <c>dt</c> bias, length <c>n_head</c>.</param>
    /// <param name="bNormWeight">QK-Norm weight for B, length <c>d_state·R</c>.</param>
    /// <param name="cNormWeight">QK-Norm weight for C, length <c>d_state·R</c>.</param>
    /// <param name="bBias">B bias, shape <c>[n_head, d_state, R]</c> row-major.</param>
    /// <param name="cBias">C bias, shape <c>[n_head, d_state, R]</c> row-major.</param>
    /// <param name="d">Per-head D skip coefficient, length <c>n_head</c>.</param>
    /// <param name="mimoXProj">Rank-expansion weight for <c>x</c>, shape <c>[n_head, head_dim, R]</c>.</param>
    /// <param name="mimoZProj">Rank-expansion weight for <c>z</c>, shape <c>[n_head, head_dim, R]</c>.</param>
    /// <param name="mimoDown">Rank-contraction weight, shape <c>[n_head, head_dim, R]</c>.</param>
    /// <param name="y">Output, shape <c>[T, d_model]</c> row-major.</param>
    /// <param name="state">SSM recurrent state, shape <c>[n_head, head_dim, d_state]</c>. In-place.</param>
    /// <param name="prevBx">Previous-step <c>Σ_r B̄⊗x_mimo</c> (rank-contracted), shape <c>[n_head, head_dim, d_state]</c>. In-place.</param>
    /// <param name="seqLen">Number of tokens <c>T</c>.</param>
    /// <param name="dModel">Model dimension.</param>
    /// <param name="dInner">Inner dimension (<c>n_head · head_dim</c>).</param>
    /// <param name="nHead">Number of SSM heads.</param>
    /// <param name="headDim">Channels per head.</param>
    /// <param name="dState">SSM state width (must be even).</param>
    /// <param name="mimoRank">MIMO rank <c>R ≥ 1</c> (typically 2–4).</param>
    /// <param name="normEps">RMSNorm stabilising constant for QK-Norm (default 1e-5).</param>
    [SkipLocalsInit]
    public static void ForwardMimo(
        ReadOnlySpan<float> u,
        ReadOnlySpan<float> inProjWeight,
        ReadOnlySpan<float> outProjWeight,
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> dtBias,
        ReadOnlySpan<float> bNormWeight,
        ReadOnlySpan<float> cNormWeight,
        ReadOnlySpan<float> bBias,
        ReadOnlySpan<float> cBias,
        ReadOnlySpan<float> d,
        ReadOnlySpan<float> mimoXProj,
        ReadOnlySpan<float> mimoZProj,
        ReadOnlySpan<float> mimoDown,
        Span<float> y,
        Span<float> state,
        Span<float> prevBx,
        int seqLen,
        int dModel,
        int dInner,
        int nHead,
        int headDim,
        int dState,
        int mimoRank,
        float normEps = 1e-5f)
        => ForwardMimo(u, inProjWeight, outProjWeight, a, dtBias, bNormWeight, cNormWeight,
                       bBias, cBias, d, mimoXProj, mimoZProj, mimoDown,
                       y, state, prevBx, cumAngle: Span<float>.Empty,
                       seqLen, dModel, dInner, nHead, headDim, dState, mimoRank, normEps);

    /// <summary>
    /// Decode-aware MIMO overload: threads <paramref name="cumAngle"/> (shape
    /// <c>[n_head, d_state/2]</c>) through the forward alongside
    /// <paramref name="state"/> and <paramref name="prevBx"/>. See the SISO
    /// <see cref="Forward(ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, Span{float}, Span{float}, Span{float}, Span{float}, int, int, int, int, int, int, float)"/>
    /// overload for the cum_angle contract; the MIMO path broadcasts the same
    /// cum_angle across the rank axis, so the state shape is identical to SISO.
    /// </summary>
    [SkipLocalsInit]
    public static void ForwardMimo(
        ReadOnlySpan<float> u,
        ReadOnlySpan<float> inProjWeight,
        ReadOnlySpan<float> outProjWeight,
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> dtBias,
        ReadOnlySpan<float> bNormWeight,
        ReadOnlySpan<float> cNormWeight,
        ReadOnlySpan<float> bBias,
        ReadOnlySpan<float> cBias,
        ReadOnlySpan<float> d,
        ReadOnlySpan<float> mimoXProj,
        ReadOnlySpan<float> mimoZProj,
        ReadOnlySpan<float> mimoDown,
        Span<float> y,
        Span<float> state,
        Span<float> prevBx,
        Span<float> cumAngle,
        int seqLen,
        int dModel,
        int dInner,
        int nHead,
        int headDim,
        int dState,
        int mimoRank,
        float normEps = 1e-5f)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (dInner != nHead * headDim)
            throw new ArgumentException($"dInner {dInner} != nHead*headDim {nHead * headDim}");
        if ((dState & 1) != 0)
            throw new ArgumentException($"dState {dState} must be even for data-RoPE pairs");
        if (mimoRank <= 0) throw new ArgumentOutOfRangeException(nameof(mimoRank));

        int R = mimoRank;
        int halfState = dState / 2;
        int bcDim = dState * R;
        int dInProj = 2 * dInner + 2 * bcDim + 2 * nHead + halfState;

        // ── Scratch ──────────────────────────────────────────────────────────
        float[] proj = new float[seqLen * dInProj];
        float[] dt = new float[seqLen * nHead];
        float[] lam = new float[seqLen * nHead];
        float[] alpha = new float[seqLen * nHead];
        float[] beta = new float[seqLen * nHead];
        float[] gamma = new float[seqLen * nHead];
        // Per-rank B/C post-RoPE: layout [R, T, nHead, dState] row-major.
        // Rank-major keeps each rank slice contiguous so we can call Mamba3DataRoPE.Execute
        // on each slice directly and feed the MimoScan a familiar [T, nHead, dState] per rank.
        float[] bMimo = new float[(long)R * seqLen * nHead * dState];
        float[] cMimo = new float[(long)R * seqLen * nHead * dState];
        // x_mimo: [T, nHead, headDim, R] per Mamba3MimoProject.ExpandInput contract.
        float[] xMimo = new float[(long)seqLen * nHead * headDim * R];
        float[] yMimo = new float[(long)seqLen * nHead * headDim * R];

        // ── Step 1: in_proj GEMM  u @ inProjWeight^T  →  proj ───────────────
        GemmF32(inProjWeight, u, proj, m: dInProj, k: dModel, n: seqLen);

        // ── Step 2: split the projection into 7 components ──────────────────
        // Layout along d_in_proj axis, per token:
        //   [0, dInner)                z
        //   [dInner, 2·dInner)         x
        //   [2·dInner, 2·dInner+bcDim) B_raw     (bc_dim = dState * R)
        //   [+bcDim)                   C_raw
        //   [+nHead)                   dt_raw
        //   [+nHead)                   lam_raw
        //   [+halfState)               theta
        int bOff = 2 * dInner;
        int cOff = bOff + bcDim;
        int dtOff = cOff + bcDim;
        int lamOff = dtOff + nHead;
        int thetaOff = lamOff + nHead;

        float[] bRaw = new float[seqLen * bcDim];
        float[] cRaw = new float[seqLen * bcDim];
        float[] theta = new float[seqLen * halfState];

        for (int t = 0; t < seqLen; t++)
        {
            int src = t * dInProj;
            for (int h = 0; h < nHead; h++)
            {
                dt[t * nHead + h] = SoftPlus(proj[src + dtOff + h] + dtBias[h]);
                lam[t * nHead + h] = Sigmoid(proj[src + lamOff + h]);
            }
            new Span<float>(proj, src + bOff, bcDim).CopyTo(bRaw.AsSpan(t * bcDim, bcDim));
            new Span<float>(proj, src + cOff, bcDim).CopyTo(cRaw.AsSpan(t * bcDim, bcDim));
            new Span<float>(proj, src + thetaOff, halfState).CopyTo(theta.AsSpan(t * halfState, halfState));
        }

        // ── Step 3: Discretize α, β, γ ──────────────────────────────────────
        Mamba3Discretize.Execute(dt, a, lam, alpha, beta, gamma, seqLen, nHead);

        // ── Step 4: QK-Norm on B, C across the full bc_dim = d_state·R ──────
        // Reference uses a single RMSNorm(bc_dim) over the rank-expanded vector;
        // our kernel handles this as nGroup=1, dState=bcDim.
        Mamba3QkNorm.Execute(bRaw, bNormWeight, normEps, seqLen, nGroup: 1, bcDim);
        Mamba3QkNorm.Execute(cRaw, cNormWeight, normEps, seqLen, nGroup: 1, bcDim);

        // ── Step 5: reshape [T, d_state·R] → per-rank [R][T, n_head, d_state]
        //           with per-head bias add. Reference layout of B_qkn within a
        //           token is (d_state, R) row-major (innermost = R), so
        //           element [t, n, r] is bRaw[t*bcDim + n*R + r].
        //           Reference B_bias has shape (n_head, d_state, R) row-major:
        //             B_bias[h, n, r] = bBias[h*dState*R + n*R + r].
        //           bMimo[r, t, h, n] = bRaw[t*bcDim + n*R + r] + bBias[h*dState*R + n*R + r]
        int rankStride = seqLen * nHead * dState;       // stride between ranks in bMimo/cMimo
        for (int t = 0; t < seqLen; t++)
        {
            int bRawTokBase = t * bcDim;
            for (int h = 0; h < nHead; h++)
            {
                int biasHeadBase = h * dState * R;
                for (int n = 0; n < dState; n++)
                {
                    int bRawNR = bRawTokBase + n * R;
                    int biasNR = biasHeadBase + n * R;
                    for (int r = 0; r < R; r++)
                    {
                        int rankBase = r * rankStride;
                        int outIdx = rankBase + (t * nHead + h) * dState + n;
                        bMimo[outIdx] = bRaw[bRawNR + r] + bBias[biasNR + r];
                        cMimo[outIdx] = cRaw[bRawNR + r] + cBias[biasNR + r];
                    }
                }
            }
        }

        // ── Step 6: data-dependent RoPE on each rank slice independently ────
        // Reference broadcasts cum_angles over the rank axis, so the rotation
        // per-(t, h, k) is identical across ranks. We reuse Mamba3DataRoPE once
        // per rank. R is small (typically 2–4) so the cost of re-running the
        // cum_angles recurrence R times is negligible.
        //
        // Decode continuity: every rank starts from the SAME cumAngle (the
        // previous call's last-token angle). Only the last rank writes the
        // final angle back out — all ranks produce identical final angles so
        // it doesn't matter which one writes, but avoiding the redundant
        // writes keeps the semantics crisp.
        for (int r = 0; r < R; r++)
        {
            int offset = r * rankStride;
            Mamba3DataRoPE.Execute(
                bMimo.AsSpan(offset, rankStride),
                cMimo.AsSpan(offset, rankStride),
                dt, theta,
                cumAnglePrev: cumAngle,
                cumAngleOut: r == R - 1 ? cumAngle : Span<float>.Empty,
                seqLen, nHead, dState);
        }

        // ── Step 7: expand x to rank-R via Mamba3MimoProject ────────────────
        // x slice from proj: [T, dInner]
        float[] x = new float[seqLen * dInner];
        for (int t = 0; t < seqLen; t++)
            new Span<float>(proj, t * dInProj + dInner, dInner).CopyTo(x.AsSpan(t * dInner, dInner));

        Mamba3MimoProject.ExpandInput(x, mimoXProj, xMimo, seqLen, nHead, headDim, R);

        // ── Step 8: MIMO trapezoidal scan ───────────────────────────────────
        MimoScan(
            state, prevBx,
            xMimo, alpha, beta, gamma, bMimo, cMimo, yMimo,
            seqLen, nHead, headDim, dState, R);

        // ── Step 9: D skip + gated(z_mimo) in rank space ────────────────────
        // y_with_d[t,h,p,r] = yMimo[t,h,p,r] + x[t,h,p]·D[h]
        // z_mimo[t,h,p,r]   = z[t,h,p] * mimo_z_proj[h,p,r]
        // y_gated[t,h,p,r]  = y_with_d[t,h,p,r] * silu(z_mimo[t,h,p,r])
        for (int t = 0; t < seqLen; t++)
        {
            int zSrc = t * dInProj;  // z at proj[t, 0..dInner)
            for (int h = 0; h < nHead; h++)
            {
                float dh = d[h];
                int headBase = h * headDim;
                for (int p = 0; p < headDim; p++)
                {
                    int hp = headBase + p;
                    float xv = x[t * dInner + hp];
                    float zv = proj[zSrc + hp];
                    int rankWeightBase = (h * headDim + p) * R;   // mimoZProj/mimoDown index base
                    int yMimoBase = ((t * nHead + h) * headDim + p) * R;
                    for (int r = 0; r < R; r++)
                    {
                        float withD = yMimo[yMimoBase + r] + xv * dh;
                        float zMimo = zv * mimoZProj[rankWeightBase + r];
                        yMimo[yMimoBase + r] = withD * Silu(zMimo);
                    }
                }
            }
        }

        // ── Step 10: contract rank via mimo_down ────────────────────────────
        // y_contracted[t, h, p] = Σ_r yMimo[t, h, p, r] * mimoDown[h, p, r]
        float[] yContracted = new float[seqLen * dInner];
        Mamba3MimoProject.ContractOutput(yMimo, mimoDown, yContracted, seqLen, nHead, headDim, R);

        // ── Step 11: out_proj GEMM ─────────────────────────────────────────
        GemmF32(outProjWeight, yContracted, y, m: dModel, k: dInner, n: seqLen);
    }

    /// <summary>
    /// MIMO trapezoidal selective scan. Advances <paramref name="state"/> and
    /// <paramref name="prevBx"/> in place, producing per-rank output <paramref name="yMimo"/>.
    /// </summary>
    /// <remarks>
    /// Implements the rank-contracting recurrence from Appendix D of the paper
    /// (and <c>ssd_mimo</c> in the reference) as a sequential scalar scan:
    /// <code>
    ///   cur_Bx[h, p, n] = Σ_r  bMimo[r, t, h, n] · xMimo[t, h, p, r]
    ///   state[h, p, n]  = α · state  +  β · prev_Bx  +  γ · cur_Bx
    ///   prev_Bx[h, p, n] = cur_Bx
    ///   yMimo[t, h, p, r] = Σ_n state[h, p, n] · cMimo[r, t, h, n]
    /// </code>
    /// Both <c>state</c> and <c>prev_Bx</c> have the rank already contracted —
    /// same shape as SISO — which matches the reference's <c>ssm_state</c>
    /// shape <c>(b, n_head, head_dim, d_state)</c>.
    /// </remarks>
    [SkipLocalsInit]
    private static void MimoScan(
        Span<float> state,
        Span<float> prevBx,
        ReadOnlySpan<float> xMimo,
        ReadOnlySpan<float> alpha,
        ReadOnlySpan<float> beta,
        ReadOnlySpan<float> gamma,
        ReadOnlySpan<float> bMimo,
        ReadOnlySpan<float> cMimo,
        Span<float> yMimo,
        int seqLen,
        int nHead,
        int headDim,
        int dState,
        int R)
    {
        int stateStrideHead = headDim * dState;
        int rankStride = seqLen * nHead * dState;        // stride between ranks in bMimo/cMimo

        for (int t = 0; t < seqLen; t++)
        {
            int tokBcBase = t * nHead * dState;          // per-rank B/C row base within a rank slice
            for (int h = 0; h < nHead; h++)
            {
                float a_th = alpha[t * nHead + h];
                float b_th = beta[t * nHead + h];
                float g_th = gamma[t * nHead + h];

                int bcHeadBase = tokBcBase + h * dState;
                int stateHeadBase = h * stateStrideHead;
                int xmTokHeadBase = ((t * nHead) + h) * headDim * R;  // [t,h,0,0] in xMimo

                for (int p = 0; p < headDim; p++)
                {
                    int stateRowBase = stateHeadBase + p * dState;
                    int xmBase = xmTokHeadBase + p * R;               // xMimo[t, h, p, 0..R)
                    int yMimoBase = xmBase;                            // yMimo matches xMimo layout

                    // Write zeros to the per-rank output slots before accumulating.
                    for (int r = 0; r < R; r++)
                        yMimo[yMimoBase + r] = 0f;

                    for (int n = 0; n < dState; n++)
                    {
                        // cur_Bx[n] = Σ_r bMimo[r, t, h, n] * xMimo[t, h, p, r]
                        float curBx = 0f;
                        int bcIdxN = bcHeadBase + n;
                        for (int r = 0; r < R; r++)
                        {
                            curBx += bMimo[r * rankStride + bcIdxN] * xMimo[xmBase + r];
                        }

                        // Trapezoidal recurrence — rank-contracted state update.
                        float s = state[stateRowBase + n] * a_th
                                + prevBx[stateRowBase + n] * b_th
                                + curBx * g_th;

                        state[stateRowBase + n] = s;
                        prevBx[stateRowBase + n] = curBx;

                        // Scatter into per-rank outputs: yMimo[t,h,p,r] += s * cMimo[r, t, h, n]
                        for (int r = 0; r < R; r++)
                        {
                            yMimo[yMimoBase + r] += s * cMimo[r * rankStride + bcIdxN];
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Scalar row-major GEMM: <c>c[n, m] = Σ_k weight[m, k] · b[n, k]</c>.
    /// Matches the convention of <c>MatMul.GemmF32</c> without requiring a
    /// pointer variant. Used for in_proj and out_proj where we own F32 spans.
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float SoftPlus(float x)
    {
        if (x > 20f) return x;
        if (x < -20f) return MathF.Exp(x);
        return MathF.Log(1f + MathF.Exp(x));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Silu(float x) => x * Sigmoid(x);
}
