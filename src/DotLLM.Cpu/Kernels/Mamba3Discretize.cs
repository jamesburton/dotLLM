using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Per-token, per-head computation of the trapezoidal-discretization coefficients
/// <c>α, β, γ</c> from Mamba-3 (Lahoti et al., ICLR 2026, Eq. 9 / Prop. 1).
/// </summary>
/// <remarks>
/// <para>
/// For each <c>(t, h)</c> pair with <c>t ∈ [0, T)</c>, <c>h ∈ [0, n_head)</c>:
/// </para>
/// <code>
/// α[t, h] = exp(dt[t, h] * A[h])              // decay, same as Mamba-2
/// γ[t, h] = λ[h] * dt[t, h]                   // current-step (right-endpoint) weight
/// β[t, h] = (1 - λ[h]) * dt[t, h] * α[t, h]   // previous-step (left-endpoint) weight
/// </code>
/// <para>
/// Inputs are all post-activation: <c>dt</c> is already softplus'd by the caller, and
/// <c>λ</c> is already sigmoid'd into <c>[0, 1]</c>. <c>A</c> is the learned
/// per-head decay parameter (negative-valued by convention so that <c>exp(dt·A)</c>
/// decays toward 0).
/// </para>
/// <para>
/// <b>Alias safety:</b> the three output buffers (<c>alpha</c>,
/// <c>beta</c>, <c>gamma</c>) MUST be distinct from each
/// other and from the inputs. The kernel writes <c>alpha</c> first and
/// then reads it back while computing <c>beta</c>, so aliasing output
/// buffers will produce silently-wrong results. This is the common case — callers
/// pre-allocate three separate scratch slots.
/// </para>
/// </remarks>
public static class Mamba3Discretize
{
    /// <summary>
    /// Computes trapezoidal discretization coefficients for Mamba-3.
    /// </summary>
    /// <param name="dt">
    /// Post-softplus timestep, shape <c>[T, n_head]</c> row-major, length <c>T * n_head</c>.
    /// </param>
    /// <param name="a">
    /// Per-head decay parameter <c>A</c>, length <c>n_head</c>. Negative-valued by
    /// convention so that <c>exp(dt · A)</c> decays toward 0.
    /// </param>
    /// <param name="lambda_">
    /// Per-head trapezoidal interpolation parameter <c>λ</c> in <c>[0, 1]</c>
    /// (post-sigmoid), length <c>n_head</c>.
    /// </param>
    /// <param name="alpha">
    /// Destination for <c>α</c>, shape <c>[T, n_head]</c> row-major. MUST NOT alias
    /// any input or other output buffer.
    /// </param>
    /// <param name="beta">
    /// Destination for <c>β</c>, shape <c>[T, n_head]</c> row-major. MUST NOT alias
    /// any input or other output buffer.
    /// </param>
    /// <param name="gamma">
    /// Destination for <c>γ</c>, shape <c>[T, n_head]</c> row-major. MUST NOT alias
    /// any input or other output buffer.
    /// </param>
    /// <param name="seqLen">Number of tokens <c>T</c>.</param>
    /// <param name="nHead">Number of SSM heads.</param>
    [SkipLocalsInit]
    public static void Execute(
        ReadOnlySpan<float> dt,
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> lambda_,
        Span<float> alpha,
        Span<float> beta,
        Span<float> gamma,
        int seqLen,
        int nHead)
    {
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead));

        long expected = (long)seqLen * nHead;
        if (dt.Length < expected)
            throw new ArgumentException(
                $"dt length {dt.Length} < T*n_head = {expected}.", nameof(dt));
        if (a.Length < nHead)
            throw new ArgumentException(
                $"a length {a.Length} < n_head = {nHead}.", nameof(a));
        if (lambda_.Length < nHead)
            throw new ArgumentException(
                $"lambda length {lambda_.Length} < n_head = {nHead}.", nameof(lambda_));
        if (alpha.Length < expected)
            throw new ArgumentException(
                $"alpha length {alpha.Length} < T*n_head = {expected}.", nameof(alpha));
        if (beta.Length < expected)
            throw new ArgumentException(
                $"beta length {beta.Length} < T*n_head = {expected}.", nameof(beta));
        if (gamma.Length < expected)
            throw new ArgumentException(
                $"gamma length {gamma.Length} < T*n_head = {expected}.", nameof(gamma));

        if (seqLen == 0) return;

        // Step 1: alpha[t, h] = exp(dt[t, h] * a[h])
        //   Do this row-by-row so we can use the per-row a[h] as a vector.
        //   Multiply(dt_row, a, alpha_row) then Exp(alpha_row, alpha_row) — both
        //   TensorPrimitives calls are element-wise and the in-place aliasing on
        //   alpha_row itself is safe (each lane reads then writes the same slot).
        //
        // Step 2: gamma[t, h] = lambda_[h] * dt[t, h]
        //   Again row-by-row so lambda_ acts as a vector.
        //
        // Step 3: beta[t, h] = dt[t, h] * alpha[t, h] - gamma[t, h] * alpha[t, h]
        //                    = (dt - gamma) * alpha
        //                    = ((1 - lambda) * dt) * alpha
        //   We fold this into a scalar inner loop — the compound op is simpler to
        //   reason about than three separate TP calls and has identical FMA shape.
        for (int t = 0; t < seqLen; t++)
        {
            int rowBase = t * nHead;
            ReadOnlySpan<float> dtRow = dt.Slice(rowBase, nHead);
            Span<float> alphaRow = alpha.Slice(rowBase, nHead);
            Span<float> betaRow = beta.Slice(rowBase, nHead);
            Span<float> gammaRow = gamma.Slice(rowBase, nHead);

            // alpha = exp(dt * a)
            TensorPrimitives.Multiply(dtRow, a[..nHead], alphaRow);
            TensorPrimitives.Exp(alphaRow, alphaRow);

            // gamma = lambda * dt
            TensorPrimitives.Multiply(dtRow, lambda_[..nHead], gammaRow);

            // beta = (1 - lambda) * dt * alpha  =  (dt - gamma) * alpha
            // Scalar inner loop keeps the (1-λ)·dt·α compound in a single FMA-shaped pass
            // and avoids needing a temporary buffer for the (1-λ)·dt intermediate.
            for (int h = 0; h < nHead; h++)
            {
                float dtv = dtRow[h];
                float gv = gammaRow[h];
                float av = alphaRow[h];
                betaRow[h] = (dtv - gv) * av;
            }
        }
    }

    /// <summary>
    /// Scalar reference implementation kept for unit-test pinning against any future
    /// SIMD variant. Identical semantics to <see cref="Execute"/>; pure element-wise
    /// loop with no TensorPrimitives calls.
    /// </summary>
    [SkipLocalsInit]
    internal static void ExecuteScalar(
        ReadOnlySpan<float> dt,
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> lambda_,
        Span<float> alpha,
        Span<float> beta,
        Span<float> gamma,
        int seqLen,
        int nHead)
    {
        for (int t = 0; t < seqLen; t++)
        {
            int rowBase = t * nHead;
            for (int h = 0; h < nHead; h++)
            {
                float dtv = dt[rowBase + h];
                float av = MathF.Exp(dtv * a[h]);
                float lam = lambda_[h];
                float gv = lam * dtv;
                float bv = (1.0f - lam) * dtv * av;
                alpha[rowBase + h] = av;
                beta[rowBase + h] = bv;
                gamma[rowBase + h] = gv;
            }
        }
    }
}
