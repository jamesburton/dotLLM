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
        // rotates B/C treated as [T, nHead, dState] per-head.
        Mamba3DataRoPE.Execute(bBroad, cBroad, dt, theta, seqLen, nHead, dState);

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
