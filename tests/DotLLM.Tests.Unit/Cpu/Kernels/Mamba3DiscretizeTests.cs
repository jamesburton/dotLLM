using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Unit tests for <see cref="Mamba3Discretize"/> — the trapezoidal discretization
/// coefficient kernel for Mamba-3 (Lahoti et al., ICLR 2026).
/// </summary>
/// <remarks>
/// Reference math (per token-head):
/// <code>
/// α = exp(dt · A)
/// γ = λ · dt
/// β = (1 - λ) · dt · α
/// </code>
/// Inputs are post-activation: <c>dt</c> is post-softplus (so &gt;= 0) and
/// <c>λ ∈ [0, 1]</c> post-sigmoid. <c>A</c> is negative-valued by convention.
/// </remarks>
public sealed class Mamba3DiscretizeTests
{
    /// <summary>
    /// λ = 0 is the trapezoidal degenerate where the current-step weight γ vanishes
    /// and the recurrence pushes all mass onto the *previous* step via β. In that
    /// regime: γ = 0 exactly, β = dt · α.
    /// </summary>
    [Fact]
    public void LambdaZero_GammaZero_BetaIsDtAlpha()
    {
        const int T = 3;
        const int H = 2;

        float[] dt = [0.25f, 0.5f, 0.4f, 0.1f, 0.6f, 0.8f]; // [T, H]
        float[] a = [-0.5f, -1.0f];
        float[] lambda_ = [0f, 0f];
        float[] alpha = new float[T * H];
        float[] beta = new float[T * H];
        float[] gamma = new float[T * H];

        Mamba3Discretize.Execute(dt, a, lambda_, alpha, beta, gamma, T, H);

        for (int i = 0; i < T * H; i++)
            Assert.Equal(0f, gamma[i], 1e-6f);

        for (int t = 0; t < T; t++)
            for (int h = 0; h < H; h++)
            {
                int idx = t * H + h;
                float expectedAlpha = MathF.Exp(dt[idx] * a[h]);
                Assert.Equal(expectedAlpha, alpha[idx], 1e-6f);
                // β = (1 - 0) · dt · α = dt · α
                Assert.Equal(dt[idx] * expectedAlpha, beta[idx], 1e-6f);
            }
    }

    /// <summary>
    /// λ = 1 is the other trapezoidal degenerate where the previous-step weight β
    /// vanishes and the recurrence is purely first-order (γ = dt). This is the
    /// setting where Mamba-3 reduces to Mamba-2-style discretization.
    /// </summary>
    [Fact]
    public void LambdaOne_BetaZero_GammaEqualsDt()
    {
        const int T = 2;
        const int H = 3;

        float[] dt = [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f];
        float[] a = [-0.25f, -0.5f, -1.0f];
        float[] lambda_ = [1f, 1f, 1f];
        float[] alpha = new float[T * H];
        float[] beta = new float[T * H];
        float[] gamma = new float[T * H];

        Mamba3Discretize.Execute(dt, a, lambda_, alpha, beta, gamma, T, H);

        for (int i = 0; i < T * H; i++)
        {
            Assert.Equal(0f, beta[i], 1e-6f);
            Assert.Equal(dt[i], gamma[i], 1e-6f);
        }

        // α is still the decay regardless of λ.
        for (int t = 0; t < T; t++)
            for (int h = 0; h < H; h++)
            {
                int idx = t * H + h;
                Assert.Equal(MathF.Exp(dt[idx] * a[h]), alpha[idx], 1e-6f);
            }
    }

    /// <summary>
    /// Hand-computed small case — T=2, H=2. Verifies the full (α, β, γ) tuple against
    /// a line-by-line manual calculation.
    /// </summary>
    [Fact]
    public void HandComputed_T2H2_MatchesReference()
    {
        const int T = 2;
        const int H = 2;

        // dt[t, h]
        float[] dt =
        [
            0.5f, 0.25f,  // t=0
            1.0f, 0.5f,   // t=1
        ];
        float[] a = [-1.0f, -2.0f];
        float[] lambda_ = [0.3f, 0.7f];

        float[] alpha = new float[T * H];
        float[] beta = new float[T * H];
        float[] gamma = new float[T * H];

        Mamba3Discretize.Execute(dt, a, lambda_, alpha, beta, gamma, T, H);

        // Hand-computed:
        // t=0, h=0: dt=0.5, a=-1.0, λ=0.3
        //   α = exp(-0.5)             ≈ 0.60653066
        //   γ = 0.3 · 0.5              = 0.15
        //   β = 0.7 · 0.5 · 0.60653066 ≈ 0.21228573
        // t=0, h=1: dt=0.25, a=-2.0, λ=0.7
        //   α = exp(-0.5)             ≈ 0.60653066
        //   γ = 0.7 · 0.25             = 0.175
        //   β = 0.3 · 0.25 · α        ≈ 0.04548980
        // t=1, h=0: dt=1.0, a=-1.0, λ=0.3
        //   α = exp(-1.0)             ≈ 0.36787944
        //   γ = 0.3 · 1.0              = 0.3
        //   β = 0.7 · 1.0 · α         ≈ 0.25751561
        // t=1, h=1: dt=0.5, a=-2.0, λ=0.7
        //   α = exp(-1.0)             ≈ 0.36787944
        //   γ = 0.7 · 0.5              = 0.35
        //   β = 0.3 · 0.5 · α         ≈ 0.05518192

        float[] expectedAlpha =
        [
            MathF.Exp(-0.5f), MathF.Exp(-0.5f),
            MathF.Exp(-1.0f), MathF.Exp(-1.0f),
        ];
        float[] expectedGamma =
        [
            0.15f, 0.175f,
            0.30f, 0.35f,
        ];
        float[] expectedBeta =
        [
            0.7f * 0.5f * MathF.Exp(-0.5f),
            0.3f * 0.25f * MathF.Exp(-0.5f),
            0.7f * 1.0f * MathF.Exp(-1.0f),
            0.3f * 0.5f * MathF.Exp(-1.0f),
        ];

        for (int i = 0; i < T * H; i++)
        {
            Assert.Equal(expectedAlpha[i], alpha[i], 1e-6f);
            Assert.Equal(expectedBeta[i], beta[i], 1e-6f);
            Assert.Equal(expectedGamma[i], gamma[i], 1e-6f);
        }
    }

    /// <summary>
    /// Decay sanity: with A negative and dt positive (both paper conventions), α must
    /// lie strictly in (0, 1). Also pins Execute to ExecuteScalar element-wise so
    /// the SIMD path is numerically equivalent to the reference loop.
    /// </summary>
    [Fact]
    public void NegativeA_AlphaInZeroOne_MatchesScalar()
    {
        const int T = 16;
        const int H = 8;

        var rng = new Random(42);
        float[] dt = new float[T * H];
        for (int i = 0; i < dt.Length; i++)
            dt[i] = rng.NextSingle() * 1.5f + 0.01f; // post-softplus range, positive
        float[] a = new float[H];
        for (int h = 0; h < H; h++)
            a[h] = -(rng.NextSingle() * 2.0f + 0.1f); // strictly negative
        float[] lambda_ = new float[H];
        for (int h = 0; h < H; h++)
            lambda_[h] = rng.NextSingle(); // in [0, 1)

        float[] alphaExec = new float[T * H];
        float[] betaExec = new float[T * H];
        float[] gammaExec = new float[T * H];
        float[] alphaRef = new float[T * H];
        float[] betaRef = new float[T * H];
        float[] gammaRef = new float[T * H];

        Mamba3Discretize.Execute(dt, a, lambda_, alphaExec, betaExec, gammaExec, T, H);
        Mamba3Discretize.ExecuteScalar(dt, a, lambda_, alphaRef, betaRef, gammaRef, T, H);

        for (int i = 0; i < T * H; i++)
        {
            Assert.InRange(alphaExec[i], 0.0f, 1.0f);
            Assert.NotEqual(0.0f, alphaExec[i]); // strictly > 0 since dt·A is finite
            Assert.Equal(alphaRef[i], alphaExec[i], 1e-6f);
            Assert.Equal(betaRef[i], betaExec[i], 1e-6f);
            Assert.Equal(gammaRef[i], gammaExec[i], 1e-6f);
        }
    }

    /// <summary>
    /// Output-buffer independence: the three output buffers must be distinct, and
    /// the kernel must not stomp one while computing another. Pinning this prevents
    /// a regression where β computation reads a corrupted α.
    /// </summary>
    [Fact]
    public void OutputBuffersIndependent_NoStomping()
    {
        const int T = 4;
        const int H = 3;

        float[] dt = [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f];
        float[] a = [-0.3f, -0.6f, -1.2f];
        float[] lambda_ = [0.2f, 0.5f, 0.8f];

        // Three distinct buffers.
        float[] alpha = new float[T * H];
        float[] beta = new float[T * H];
        float[] gamma = new float[T * H];

        // Poison the buffers with a sentinel before the call to force the kernel to
        // actually write every slot (not just rely on default zeros).
        for (int i = 0; i < T * H; i++)
        {
            alpha[i] = float.NaN;
            beta[i] = float.NaN;
            gamma[i] = float.NaN;
        }

        Mamba3Discretize.Execute(dt, a, lambda_, alpha, beta, gamma, T, H);

        // Compare against the scalar reference computed into a fresh triple of
        // buffers. If Execute aliased its outputs internally, one or more of the
        // three would diverge from the scalar truth.
        float[] alphaRef = new float[T * H];
        float[] betaRef = new float[T * H];
        float[] gammaRef = new float[T * H];
        Mamba3Discretize.ExecuteScalar(dt, a, lambda_, alphaRef, betaRef, gammaRef, T, H);

        for (int i = 0; i < T * H; i++)
        {
            Assert.False(float.IsNaN(alpha[i]), $"alpha[{i}] not written");
            Assert.False(float.IsNaN(beta[i]), $"beta[{i}] not written");
            Assert.False(float.IsNaN(gamma[i]), $"gamma[{i}] not written");
            Assert.Equal(alphaRef[i], alpha[i], 1e-6f);
            Assert.Equal(betaRef[i], beta[i], 1e-6f);
            Assert.Equal(gammaRef[i], gamma[i], 1e-6f);
        }
    }
}
