using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Correctness anchor for <see cref="GatedDeltaNetScan"/>. The delta-rule recurrence
/// involves both a retrieve (read before write) and a write-then-read-out, so
/// sign errors and index-swap bugs produce plausible-looking but wrong output.
/// These tests hand-compute tiny cases and compare element-wise to catch them.
/// </summary>
public sealed class GatedDeltaNetScanTests
{
    private const float Tol = 1e-5f;

    /// <summary>
    /// Zero initial state, single token, 2 V-heads / 1 K-head (VHeadsPerKHead = 2).
    /// Both V-heads share the same K/Q head. Verifies the basic write then read path
    /// and the K-head broadcast grouping.
    ///
    /// Hand-computed reference (Python/numpy):
    ///   k = [.5, .1, .3, .2]   q = [.1, .2, .3, .4]   dot(k,q) = 0.24
    ///   VH0: beta=0.8, delta = 0.8*v0 = [.48,.56,.64,.72]
    ///        out0 = delta * dot(k,q) / sqrt(4) = delta * 0.12 = [.0576,.0672,.0768,.0864]
    ///   VH1: beta=0.7, delta = 0.7*v1 = [.35,.28,.21,.14]
    ///        out1 = delta * 0.12 = [.042,.0336,.0252,.0168]
    /// </summary>
    [Fact]
    public void ZeroState_SingleToken_TwoVHeadsOneKHead_MatchesHandComputed()
    {
        const int nVHead = 2, nKHead = 1, dState = 4, seqLen = 1;

        float[] state = new float[nVHead * dState * dState];  // zeros

        // q and k: [seqLen=1, nKHead=1, dState=4]
        float[] q = [0.1f, 0.2f, 0.3f, 0.4f];
        float[] k = [0.5f, 0.1f, 0.3f, 0.2f];

        // v: [seqLen=1, nVHead=2, dState=4]
        float[] v = [0.6f, 0.7f, 0.8f, 0.9f,   // vh=0
                     0.5f, 0.4f, 0.3f, 0.2f];   // vh=1

        // g, beta: [seqLen=1, nVHead=2]
        float[] g    = [0.9f, 0.85f];
        float[] beta = [0.8f, 0.7f];

        float[] output = new float[seqLen * nVHead * dState];

        GatedDeltaNetScan.Execute(state, q, k, v, g, beta, output, nVHead, nKHead, dState, seqLen);

        // VH0 expected output (zero initial state, g=0.9 has no effect)
        Assert.Equal(0.0576f, output[0], Tol);
        Assert.Equal(0.0672f, output[1], Tol);
        Assert.Equal(0.0768f, output[2], Tol);
        Assert.Equal(0.0864f, output[3], Tol);

        // VH1 expected output
        Assert.Equal(0.0420f,  output[4], Tol);
        Assert.Equal(0.0336f,  output[5], Tol);
        Assert.Equal(0.0252f,  output[6], Tol);
        Assert.Equal(0.0168f,  output[7], Tol);
    }

    /// <summary>
    /// Non-zero initial state, decay-only token (beta = 0 so no write), verifies:
    ///   1. State decays by g = 0.5 element-wise.
    ///   2. Output reads from the decayed state via the query.
    ///
    /// Setup: state[0] = identity × 1 (flat [1,0,0,1] for dState=2)
    ///        k=[1,0], q=[1,0], v=[0,0], g=0.5, beta=0.0
    ///
    /// After decay: state = 0.5 × I = [[.5,0],[0,.5]]
    /// Retrieve: S.T @ k = [.5, 0] (but no write since beta=0)
    /// Output: S.T @ q / √2 = [.5, 0] / √2 = [.3535..., 0]
    /// </summary>
    [Fact]
    public void NonZeroState_BetaZero_OnlyDecaysNoWrite()
    {
        const int nVHead = 1, nKHead = 1, dState = 2, seqLen = 1;

        // Identity matrix (row-major [DState, DState])
        float[] state = [1f, 0f, 0f, 1f];

        float[] q = [1f, 0f];
        float[] k = [1f, 0f];
        float[] v = [0f, 0f];
        float[] g    = [0.5f];
        float[] beta = [0.0f];

        float[] output = new float[nVHead * dState];

        GatedDeltaNetScan.Execute(state, q, k, v, g, beta, output, nVHead, nKHead, dState, seqLen);

        // State after: 0.5 × I
        Assert.Equal(0.5f, state[0], Tol);   // [0,0]
        Assert.Equal(0f,   state[1], Tol);   // [0,1]
        Assert.Equal(0f,   state[2], Tol);   // [1,0]
        Assert.Equal(0.5f, state[3], Tol);   // [1,1]

        // Output: S.T @ q / √2 = [.5, 0] / √2
        float expected0 = 0.5f / MathF.Sqrt(2f);
        Assert.Equal(expected0, output[0], Tol);
        Assert.Equal(0f,        output[1], Tol);
    }

    /// <summary>
    /// Two-token sequence. Verifies state carries forward correctly across token steps
    /// and that decayed prior context influences the second token's output.
    ///
    /// Setup: nVHead=1, nKHead=1, dState=2, seqLen=2
    ///
    /// Token 0: k=[1,0], q=[1,0], v=[2,3], g=1.0, beta=1.0
    ///   state = outer([1,0],[2,3]) = [[2,3],[0,0]]
    ///   out0  = state.T @ q / √2  = [2,3]/√2  ≈ [1.41421, 2.12132]
    ///
    /// Token 1: k=[0,1], q=[0,1], v=[4,5], g=0.5, beta=1.0
    ///   state  ×= 0.5 → [[1,1.5],[0,0]]
    ///   retrieved = state.T @ [0,1] = [0,0]   (k orthogonal to k0 → no retrieval)
    ///   delta  = [4,5]
    ///   state  += outer([0,1],[4,5]) → [[1,1.5],[4,5]]
    ///   out1   = state.T @ [0,1] / √2 = [4,5]/√2 ≈ [2.82843, 3.53553]
    /// </summary>
    [Fact]
    public void TwoTokens_OrthogonalKeys_StateCarriesForward()
    {
        const int nVHead = 1, nKHead = 1, dState = 2, seqLen = 2;

        float[] state = new float[dState * dState];  // zeros

        // [seqLen=2, nKHead=1, dState=2]
        float[] q = [1f, 0f,   // token 0
                     0f, 1f];  // token 1
        float[] k = [1f, 0f,
                     0f, 1f];
        // [seqLen=2, nVHead=1, dState=2]
        float[] v = [2f, 3f,
                     4f, 5f];
        // [seqLen=2, nVHead=1]
        float[] g    = [1.0f, 0.5f];
        float[] beta = [1.0f, 1.0f];

        float[] output = new float[seqLen * nVHead * dState];

        GatedDeltaNetScan.Execute(state, q, k, v, g, beta, output, nVHead, nKHead, dState, seqLen);

        float inv_sqrt2 = 1f / MathF.Sqrt(2f);

        // Token 0 output: [2,3] / √2
        Assert.Equal(2f * inv_sqrt2, output[0], Tol);
        Assert.Equal(3f * inv_sqrt2, output[1], Tol);

        // Token 1 output: [4,5] / √2
        Assert.Equal(4f * inv_sqrt2, output[2], Tol);
        Assert.Equal(5f * inv_sqrt2, output[3], Tol);

        // Final state: [[1, 1.5], [4, 5]]
        Assert.Equal(1.0f, state[0], Tol);   // [0,0]
        Assert.Equal(1.5f, state[1], Tol);   // [0,1]
        Assert.Equal(4.0f, state[2], Tol);   // [1,0]
        Assert.Equal(5.0f, state[3], Tol);   // [1,1]
    }

    /// <summary>
    /// Retrieval test: write a known pattern then read it back at the same key.
    /// With unit-norm k=[1,0] writing v=[7,8] (beta=1, g=1), a subsequent query
    /// q=[1,0] must return [7,8]/√DState (perfect retrieval for unit-norm key).
    /// </summary>
    [Fact]
    public void Retrieve_UnitNormKey_PerfectRetrieval()
    {
        const int nVHead = 1, nKHead = 1, dState = 2, seqLen = 2;

        float[] state = new float[dState * dState];

        float[] q = [1f, 0f,   // token 0 (write)
                     1f, 0f];  // token 1 (read with same query)
        float[] k = [1f, 0f,
                     1f, 0f];
        float[] v = [7f, 8f,   // written at token 0
                     0f, 0f];  // token 1 value (no new write since beta=0)
        float[] g    = [1.0f, 1.0f];
        float[] beta = [1.0f, 0.0f];  // write at token 0, no write at token 1

        float[] output = new float[seqLen * nVHead * dState];

        GatedDeltaNetScan.Execute(state, q, k, v, g, beta, output, nVHead, nKHead, dState, seqLen);

        float inv_sqrt2 = 1f / MathF.Sqrt(2f);

        // Token 0: zero state, writes [7,8] → out0 = [7,8]/√2
        Assert.Equal(7f * inv_sqrt2, output[0], Tol);
        Assert.Equal(8f * inv_sqrt2, output[1], Tol);

        // Token 1: beta=0, no new write. Retrieval at same k=[1,0] cancels the
        // corrective term; output reads updated state (which holds the token-0 write).
        // State after token 0: S=[[7,8],[0,0]].
        // Token 1: g=1, so no decay. retrieved = S.T@[1,0] = [7,8].
        // delta = 0*(v-retrieved) = 0. S unchanged.
        // out1 = S.T@[1,0]/√2 = [7,8]/√2.
        Assert.Equal(7f * inv_sqrt2, output[2], Tol);
        Assert.Equal(8f * inv_sqrt2, output[3], Tol);
    }

    /// <summary>
    /// Verifies <see cref="GatedDeltaNetScan.L2NormalizeHeads"/> normalises each
    /// [dState]-length head slice to unit L2 norm independently.
    /// </summary>
    [Fact]
    public void L2NormalizeHeads_NormalisesEachSliceToUnitNorm()
    {
        const int dState = 2;

        // Two heads: [3,4] (norm=5) and [1,1] (norm=√2)
        float[] heads = [3f, 4f, 1f, 1f];
        GatedDeltaNetScan.L2NormalizeHeads(heads, dState);

        // Head 0: [3/5, 4/5]
        Assert.Equal(0.6f, heads[0], Tol);
        Assert.Equal(0.8f, heads[1], Tol);

        // Head 1: [1/√2, 1/√2]
        float inv_sqrt2 = 1f / MathF.Sqrt(2f);
        Assert.Equal(inv_sqrt2, heads[2], Tol);
        Assert.Equal(inv_sqrt2, heads[3], Tol);
    }

    /// <summary>
    /// Degenerate input: seqLen=0 must return immediately without touching output.
    /// </summary>
    [Fact]
    public void ZeroSeqLen_DoesNothing()
    {
        const int nVHead = 2, nKHead = 1, dState = 4;

        float[] state = new float[nVHead * dState * dState];
        float sentinel = 42f;
        float[] output = Enumerable.Repeat(sentinel, nVHead * dState).ToArray();

        GatedDeltaNetScan.Execute(
            state, [], [], [], [], [], output, nVHead, nKHead, dState, seqLen: 0);

        Assert.All(output, v => Assert.Equal(sentinel, v));
    }
}
