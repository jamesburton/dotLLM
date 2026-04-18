using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Mamba-3 trapezoidal selective scan (Lahoti et al., arXiv 2603.15569, Prop. 1 / Eq. 9):
/// <code>
///   state[t] = α_t · state[t-1]  +  β_t · (B̄_{t-1} ⊗ x_{t-1})  +  γ_t · (B̄_t ⊗ x_t)
///   y[t]     = C_t · state[t]
/// </code>
/// Unlike Mamba-2's first-order recurrence which uses only the current <c>B·x</c>, the
/// trapezoidal form also reads the <em>previous</em> step's <c>B·x</c>. That <c>prev_Bx</c>
/// per-head-per-channel-per-state tensor is the second recurrent state and is advanced
/// in-place alongside the SSM state. At the start of a fresh sequence both buffers should
/// be zero-initialised.
/// </summary>
/// <remarks>
/// <para>
/// The paper and the <c>VikramKarLex/mamba3-minimal</c> reference implement this as two
/// back-to-back SSD calls (a γ-path and a β-path) that share the same α-decay, followed
/// by an element-wise sum. In a sequential CPU scalar loop the two accumulations fuse
/// trivially in the innermost dimension, so we inline them here instead of orchestrating
/// two <see cref="Mamba2SelectiveScan"/> calls — the result is identical, one-pass memory
/// access, and there's no need to materialise an intermediate <c>BX</c> tensor of shape
/// <c>[T, n_head, head_dim, d_state]</c>.
/// </para>
/// <para>
/// α, β, γ are expected pre-computed (see <see cref="Mamba3Discretize"/>). B and C are
/// expected post-RoPE / post-QkNorm / post-BC-bias (see <see cref="Mamba3DataRoPE"/>,
/// <see cref="Mamba3QkNorm"/>). The D skip and the output SiLU(z) gate are applied by the
/// surrounding block, not here.
/// </para>
/// <para>
/// Scalar reference only — correctness first. SIMD of the innermost <c>d_state</c> loop is
/// straightforward (<c>Vector256</c> pipelines two FMAs) once profiling justifies it.
/// </para>
/// </remarks>
public static class Mamba3SelectiveScan
{
    /// <summary>
    /// Runs the trapezoidal scan over <paramref name="seqLen"/> tokens, advancing both
    /// <paramref name="state"/> and <paramref name="prevBx"/> in place.
    /// </summary>
    /// <param name="state">
    /// SSM hidden state, shape <c>[n_head, head_dim, d_state]</c> row-major.
    /// Updated in place. Zero on first call.
    /// </param>
    /// <param name="prevBx">
    /// Previous-step <c>B̄ ⊗ x</c>, shape <c>[n_head, head_dim, d_state]</c> row-major.
    /// Updated in place. Zero on first call (no β contribution at t=0 of a fresh sequence).
    /// </param>
    /// <param name="x">SSM input value, shape <c>[T, d_inner]</c> row-major. <c>d_inner = n_head · head_dim</c>.</param>
    /// <param name="alpha">Decay coefficient α_t = exp(dt·A), shape <c>[T, n_head]</c> row-major.</param>
    /// <param name="beta">Trapezoidal left-endpoint β_t = (1-λ)·dt·α_t, shape <c>[T, n_head]</c> row-major.</param>
    /// <param name="gamma">Trapezoidal right-endpoint γ_t = λ·dt, shape <c>[T, n_head]</c> row-major.</param>
    /// <param name="b">Input projection coefficient B̄, shape <c>[T, n_group, d_state]</c> row-major. Post-RoPE, post-QkNorm, post-BC-bias.</param>
    /// <param name="c">Output projection coefficient C, shape <c>[T, n_group, d_state]</c> row-major. Post-RoPE, post-QkNorm, post-BC-bias.</param>
    /// <param name="y">Output, shape <c>[T, d_inner]</c> row-major. Written.</param>
    /// <param name="nHead">Number of heads.</param>
    /// <param name="headDim">Channels per head (<c>d_inner / n_head</c>).</param>
    /// <param name="dState">SSM state width.</param>
    /// <param name="nGroup">Number of B/C groups. Must divide <paramref name="nHead"/>.</param>
    /// <param name="seqLen">Number of tokens in this step.</param>
    [SkipLocalsInit]
    public static void Execute(
        Span<float> state,
        Span<float> prevBx,
        ReadOnlySpan<float> x,
        ReadOnlySpan<float> alpha,
        ReadOnlySpan<float> beta,
        ReadOnlySpan<float> gamma,
        ReadOnlySpan<float> b,
        ReadOnlySpan<float> c,
        Span<float> y,
        int nHead,
        int headDim,
        int dState,
        int nGroup,
        int seqLen)
    {
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));
        if (nGroup <= 0) throw new ArgumentOutOfRangeException(nameof(nGroup));
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nHead % nGroup != 0)
            throw new ArgumentException($"n_head ({nHead}) must be divisible by n_group ({nGroup}).");

        int dInner = nHead * headDim;
        int headsPerGroup = nHead / nGroup;
        long stateElems = (long)nHead * headDim * dState;

        if (state.Length < stateElems)
            throw new ArgumentException("state span too small.", nameof(state));
        if (prevBx.Length < stateElems)
            throw new ArgumentException("prevBx span too small.", nameof(prevBx));
        if (x.Length < (long)seqLen * dInner)
            throw new ArgumentException("x span too small.", nameof(x));
        if (alpha.Length < (long)seqLen * nHead)
            throw new ArgumentException("alpha span too small.", nameof(alpha));
        if (beta.Length < (long)seqLen * nHead)
            throw new ArgumentException("beta span too small.", nameof(beta));
        if (gamma.Length < (long)seqLen * nHead)
            throw new ArgumentException("gamma span too small.", nameof(gamma));
        if (b.Length < (long)seqLen * nGroup * dState)
            throw new ArgumentException("b span too small.", nameof(b));
        if (c.Length < (long)seqLen * nGroup * dState)
            throw new ArgumentException("c span too small.", nameof(c));
        if (y.Length < (long)seqLen * dInner)
            throw new ArgumentException("y span too small.", nameof(y));

        int stateStrideHead = headDim * dState;

        for (int t = 0; t < seqLen; t++)
        {
            ReadOnlySpan<float> xRow = x.Slice(t * dInner, dInner);
            ReadOnlySpan<float> alphaRow = alpha.Slice(t * nHead, nHead);
            ReadOnlySpan<float> betaRow = beta.Slice(t * nHead, nHead);
            ReadOnlySpan<float> gammaRow = gamma.Slice(t * nHead, nHead);
            ReadOnlySpan<float> bRow = b.Slice(t * nGroup * dState, nGroup * dState);
            ReadOnlySpan<float> cRow = c.Slice(t * nGroup * dState, nGroup * dState);
            Span<float> yRow = y.Slice(t * dInner, dInner);

            for (int h = 0; h < nHead; h++)
            {
                float alpha_th = alphaRow[h];
                float beta_th = betaRow[h];
                float gamma_th = gammaRow[h];

                int g = h / headsPerGroup;
                ReadOnlySpan<float> bGroup = bRow.Slice(g * dState, dState);
                ReadOnlySpan<float> cGroup = cRow.Slice(g * dState, dState);

                int xHeadOffset = h * headDim;
                int stateHeadOffset = h * stateStrideHead;

                for (int p = 0; p < headDim; p++)
                {
                    float x_thp = xRow[xHeadOffset + p];
                    int stateRowOffset = stateHeadOffset + p * dState;
                    Span<float> stateRow = state.Slice(stateRowOffset, dState);
                    Span<float> prevBxRow = prevBx.Slice(stateRowOffset, dState);

                    float sumf = 0f;
                    for (int k = 0; k < dState; k++)
                    {
                        // Current B̄_t ⊗ x_t contribution at this (h, p, k).
                        float curBx_k = bGroup[k] * x_thp;

                        // Trapezoidal recurrence (Eq. 9):
                        //   state = α · state + β · prev_Bx + γ · curBx
                        float s = stateRow[k] * alpha_th
                                + prevBxRow[k] * beta_th
                                + curBx_k * gamma_th;

                        stateRow[k] = s;
                        prevBxRow[k] = curBx_k;   // memoise for the next step's β term

                        sumf += s * cGroup[k];
                    }
                    yRow[xHeadOffset + p] = sumf;
                }
            }
        }
    }
}
