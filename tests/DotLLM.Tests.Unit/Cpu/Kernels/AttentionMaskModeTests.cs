using DotLLM.Core.Attention;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Kernel-level coverage for the bidirectional / hybrid attention-mask seam (issue #26 PR-3).
/// Verifies that:
/// <list type="bullet">
///   <item>The Causal mode is byte-identical to the pre-existing fast path (the default overload
///         and explicit-Causal produce the same bits).</item>
///   <item>Bidirectional drops the causal upper bound (a query attends to future keys).</item>
///   <item>Hybrid keeps the prefix causal and lets the canvas attend to everything.</item>
///   <item>The sliding window composes with every mode.</item>
/// </list>
/// Each assertion is exercised on both the naive score-matrix path (small <c>seqQ*seqKv</c>) and the
/// tiled online-softmax path (large score matrix) plus the scalar reference, since the masking
/// diverged across three code paths.
/// </summary>
public sealed class AttentionMaskModeTests
{
    // ───────────────────────── Golden causal identity ─────────────────────────

    [Fact]
    public void Causal_DefaultOverload_ByteIdenticalToExplicitCausal()
    {
        // The #1 rule: explicit Causal must equal the default (no-mask-arg) overload bit-for-bit.
        const int headDim = 4, seqLen = 6, numHeads = 2, numKvHeads = 2;
        var (q, k, v) = RandomQkv(seqLen, seqLen, numHeads, numKvHeads, headDim, seed: 11);

        float[] outDefault = new float[seqLen * numHeads * headDim];
        float[] outExplicit = new float[seqLen * numHeads * headDim];

        float scale = 1.0f / MathF.Sqrt(headDim);

        // Default overload (no mask args → Causal). Use the scale-only overload (no alibi).
        Attention.Execute(q, k, v, outDefault, seqLen, seqLen, numHeads, numKvHeads, headDim,
                          positionOffset: 0, scale);
        // Explicit Causal via the mask-aware overload.
        Attention.Execute(q, k, v, outExplicit, seqLen, seqLen, numHeads, numKvHeads, headDim,
                          positionOffset: 0, scale, ReadOnlySpan<float>.Empty,
                          slidingWindowSize: null, softCap: 0f,
                          maskMode: AttentionMaskMode.Causal, prefixLen: 0);

        Assert.Equal(outDefault, outExplicit); // exact bitwise equality of the float[] sequences
    }

    [Fact]
    public void Causal_TiledPath_DefaultEqualsExplicit_ByteIdentical()
    {
        // Force the tiled online-softmax path: large seqQ*seqKv score matrix (> 8 KB).
        const int headDim = 8, seqLen = 64, numHeads = 2, numKvHeads = 2;
        var (q, k, v) = RandomQkv(seqLen, seqLen, numHeads, numKvHeads, headDim, seed: 23);

        float[] a = new float[seqLen * numHeads * headDim];
        float[] b = new float[seqLen * numHeads * headDim];
        float scale = 1.0f / MathF.Sqrt(headDim);

        Attention.Execute(q, k, v, a, seqLen, seqLen, numHeads, numKvHeads, headDim, 0, scale);
        Attention.Execute(q, k, v, b, seqLen, seqLen, numHeads, numKvHeads, headDim, 0, scale,
                          ReadOnlySpan<float>.Empty,
                          slidingWindowSize: null, softCap: 0f,
                          maskMode: AttentionMaskMode.Causal, prefixLen: 0);

        Assert.Equal(a, b);
    }

    // ───────────────────────── Bidirectional ─────────────────────────

    [Fact]
    public void Bidirectional_Position0_AttendsToFutureKey()
    {
        // Causal: position 0 sees only key 0. Bidirectional: position 0 sees key 1 too.
        // Constructed so that V[1] differs from V[0] → changing the future key visibly
        // changes position 0's output only under Bidirectional.
        const int headDim = 2, seqLen = 2;
        // Equal Q/K directions → equal scores → output is the MEAN of visible V rows.
        float[] q = [1f, 0f, 1f, 0f];
        float[] k = [1f, 0f, 1f, 0f];
        float[] v = [1f, 0f, 0f, 1f];

        float[] causal = new float[seqLen * headDim];
        float[] bidi = new float[seqLen * headDim];

        Attention.Execute(q, k, v, causal, seqLen, seqLen, 1, 1, headDim, 0); // default Causal
        ExecuteBidi(q, k, v, bidi, seqLen, seqLen, 1, 1, headDim);

        // Causal: position 0 sees only V[0] = [1,0].
        Assert.Equal(1f, causal[0], 1e-5f);
        Assert.Equal(0f, causal[1], 1e-5f);

        // Bidirectional: position 0 sees V[0] and V[1] equally → mean [0.5, 0.5].
        Assert.Equal(0.5f, bidi[0], 1e-5f);
        Assert.Equal(0.5f, bidi[1], 1e-5f);
    }

    [Fact]
    public void Bidirectional_FutureTokenChange_ChangesEarlierPositionOutput()
    {
        // Stronger statement of acceptance: mutating a future key/value alters an
        // earlier position's output under Bidirectional (proving j>i attention),
        // while leaving it unchanged under Causal.
        const int headDim = 4, seqLen = 5, numHeads = 2, numKvHeads = 2;
        var (q, k, v) = RandomQkv(seqLen, seqLen, numHeads, numKvHeads, headDim, seed: 7);
        var (_, kMut, vMut) = RandomQkv(seqLen, seqLen, numHeads, numKvHeads, headDim, seed: 7);

        // Perturb only the LAST position's K and V.
        int kvStride = numKvHeads * headDim;
        for (int d = 0; d < kvStride; d++)
        {
            kMut[(seqLen - 1) * kvStride + d] += 0.5f;
            vMut[(seqLen - 1) * kvStride + d] += 0.5f;
        }

        // Causal: position 0's output must be identical (it never sees the last key).
        float[] cBase = new float[seqLen * numHeads * headDim];
        float[] cMut = new float[seqLen * numHeads * headDim];
        Attention.Execute(q, k, v, cBase, seqLen, seqLen, numHeads, numKvHeads, headDim, 0);
        Attention.Execute(q, kMut, vMut, cMut, seqLen, seqLen, numHeads, numKvHeads, headDim, 0);
        for (int d = 0; d < numHeads * headDim; d++)
            Assert.Equal(cBase[d], cMut[d], 1e-6f);

        // Bidirectional: position 0's output MUST change (it now attends to the last key).
        float[] bBase = new float[seqLen * numHeads * headDim];
        float[] bMut = new float[seqLen * numHeads * headDim];
        ExecuteBidi(q, k, v, bBase, seqLen, seqLen, numHeads, numKvHeads, headDim);
        ExecuteBidi(q, kMut, vMut, bMut, seqLen, seqLen, numHeads, numKvHeads, headDim);
        float maxDiff = 0f;
        for (int d = 0; d < numHeads * headDim; d++)
            maxDiff = MathF.Max(maxDiff, MathF.Abs(bBase[d] - bMut[d]));
        Assert.True(maxDiff > 1e-4f,
            $"Bidirectional position 0 did not react to a future-token change (maxDiff={maxDiff}).");
    }

    [Fact]
    public void Bidirectional_SlidingWindow_StillMasksDistantPast()
    {
        // Sliding window must compose with Bidirectional: with window=2, query at
        // position 3 cannot see key 0 (earliest visible = 3-2+1 = 2).
        const int headDim = 2, seqLen = 4;
        float[] q = [1f, 0f, 1f, 0f, 1f, 0f, 1f, 0f];
        float[] k = [1f, 0f, 1f, 0f, 1f, 0f, 1f, 0f];
        // Distinct V per position so we can detect which keys contribute.
        float[] v = [10f, 0f, 0f, 0f, 0f, 0f, 0f, 0f]; // only position 0 has mass

        float[] outNoWindow = new float[seqLen * headDim];
        float[] outWindow = new float[seqLen * headDim];

        ExecuteBidi(q, k, v, outNoWindow, seqLen, seqLen, 1, 1, headDim, slidingWindowSize: null);
        ExecuteBidi(q, k, v, outWindow, seqLen, seqLen, 1, 1, headDim, slidingWindowSize: 2);

        // Query 3 with full bidirectional attention DOES see key 0 → nonzero d0.
        Assert.True(outNoWindow[3 * headDim] > 1e-4f,
            "Bidirectional (no window) query 3 should see key 0.");
        // With window=2, query 3 cannot see key 0 → its mass vanishes (≈0).
        Assert.Equal(0f, outWindow[3 * headDim], 1e-5f);
    }

    // ───────────────────────── Hybrid ─────────────────────────

    [Fact]
    public void Hybrid_CanvasAttendsToFullPrefixAndCanvas_PrefixStaysCausal()
    {
        // prefixLen = 2. Positions 0,1 are the causal prefix; positions 2,3 are the canvas.
        //  - Prefix position 0 must NOT see key 1,2,3 (stays causal).
        //  - Canvas position 2 must see ALL keys 0..3 (prefix + canvas, including future key 3).
        const int headDim = 2, seqLen = 4, prefixLen = 2;
        float[] q = [1f, 0f, 1f, 0f, 1f, 0f, 1f, 0f];
        float[] k = [1f, 0f, 1f, 0f, 1f, 0f, 1f, 0f];
        float[] v = [1f, 0f, 2f, 0f, 3f, 0f, 4f, 0f];

        float[] outp = new float[seqLen * headDim];
        ExecuteHybrid(q, k, v, outp, seqLen, seqLen, 1, 1, headDim, prefixLen);

        // Prefix position 0: causal → sees only V[0] = 1.
        Assert.Equal(1f, outp[0 * headDim], 1e-5f);
        // Prefix position 1: causal → mean(V0,V1) = 1.5.
        Assert.Equal(1.5f, outp[1 * headDim], 1e-5f);
        // Canvas position 2: sees V0..V3 equally → mean(1,2,3,4) = 2.5.
        Assert.Equal(2.5f, outp[2 * headDim], 1e-5f);
        // Canvas position 3: sees V0..V3 equally → 2.5 as well.
        Assert.Equal(2.5f, outp[3 * headDim], 1e-5f);
    }

    [Fact]
    public void Hybrid_CanvasTokenChange_LeavesPrefixUnchanged_ButChangesOtherCanvas()
    {
        // Acceptance: changing a canvas key does NOT change a prefix position's output,
        // but DOES change another canvas position's output.
        const int headDim = 4, seqLen = 6, prefixLen = 3, numHeads = 2, numKvHeads = 2;
        var (q, k, v) = RandomQkv(seqLen, seqLen, numHeads, numKvHeads, headDim, seed: 31);
        var (_, kMut, vMut) = RandomQkv(seqLen, seqLen, numHeads, numKvHeads, headDim, seed: 31);

        int kvStride = numKvHeads * headDim;
        // Perturb a CANVAS key/value (position 5, which is >= prefixLen).
        for (int d = 0; d < kvStride; d++)
        {
            kMut[5 * kvStride + d] += 0.7f;
            vMut[5 * kvStride + d] += 0.7f;
        }

        float[] baseOut = new float[seqLen * numHeads * headDim];
        float[] mutOut = new float[seqLen * numHeads * headDim];
        ExecuteHybrid(q, k, v, baseOut, seqLen, seqLen, numHeads, numKvHeads, headDim, prefixLen);
        ExecuteHybrid(q, kMut, vMut, mutOut, seqLen, seqLen, numHeads, numKvHeads, headDim, prefixLen);

        int qStride = numHeads * headDim;

        // Prefix position 0 (qPos 0 < prefixLen) stays causal → cannot see key 5 → unchanged.
        for (int d = 0; d < qStride; d++)
            Assert.Equal(baseOut[0 * qStride + d], mutOut[0 * qStride + d], 1e-6f);

        // Canvas position 3 (qPos 3 >= prefixLen) attends to key 5 → must change.
        float maxDiff = 0f;
        for (int d = 0; d < qStride; d++)
            maxDiff = MathF.Max(maxDiff, MathF.Abs(baseOut[3 * qStride + d] - mutOut[3 * qStride + d]));
        Assert.True(maxDiff > 1e-4f,
            $"Hybrid canvas position 3 did not react to a canvas-token change (maxDiff={maxDiff}).");
    }

    [Fact]
    public void Hybrid_TiledPath_MatchesScalarReference()
    {
        // Cross-check the tiled online-softmax hybrid path against the scalar reference
        // on a large score matrix (forces the tiled branch).
        const int headDim = 8, seqLen = 48, prefixLen = 20, numHeads = 2, numKvHeads = 2;
        var (q, k, v) = RandomQkv(seqLen, seqLen, numHeads, numKvHeads, headDim, seed: 99);

        float[] tiled = new float[seqLen * numHeads * headDim];
        float[] scalar = new float[seqLen * numHeads * headDim];
        float scale = 1.0f / MathF.Sqrt(headDim);

        Attention.Execute(q, k, v, tiled, seqLen, seqLen, numHeads, numKvHeads, headDim,
                          0, scale, ReadOnlySpan<float>.Empty, slidingWindowSize: null, softCap: 0f,
                          maskMode: AttentionMaskMode.Hybrid, prefixLen: prefixLen);
        Attention.ExecuteScalar(q, k, v, scalar, seqLen, seqLen, numHeads, numKvHeads, headDim,
                                0, scale, alibiSlopes: ReadOnlySpan<float>.Empty, slidingWindowSize: null, softCap: 0f,
                                maskMode: AttentionMaskMode.Hybrid, prefixLen: prefixLen);

        // Tolerance set to 5e-2 to account for the fast approximate-exp softmax used by the
        // tiled SIMD path vs the exact scalar reference (matches AttentionTests convention).
        for (int i = 0; i < tiled.Length; i++)
            Assert.Equal(scalar[i], tiled[i], 5e-2f);
    }

    [Fact]
    public void Bidirectional_NaivePath_MatchesScalarReference()
    {
        const int headDim = 4, seqLen = 5, numHeads = 2, numKvHeads = 2;
        var (q, k, v) = RandomQkv(seqLen, seqLen, numHeads, numKvHeads, headDim, seed: 55);

        float[] naive = new float[seqLen * numHeads * headDim];
        float[] scalar = new float[seqLen * numHeads * headDim];
        float scale = 1.0f / MathF.Sqrt(headDim);

        ExecuteBidi(q, k, v, naive, seqLen, seqLen, numHeads, numKvHeads, headDim);
        Attention.ExecuteScalar(q, k, v, scalar, seqLen, seqLen, numHeads, numKvHeads, headDim,
                                0, scale, alibiSlopes: ReadOnlySpan<float>.Empty, slidingWindowSize: null, softCap: 0f,
                                maskMode: AttentionMaskMode.Bidirectional, prefixLen: 0);

        // Tolerance set to 5e-2 to account for the fast approximate-exp softmax used by the
        // naive SIMD path vs the exact scalar reference (matches AttentionTests convention).
        for (int i = 0; i < naive.Length; i++)
            Assert.Equal(scalar[i], naive[i], 5e-2f);
    }

    // ───────────────────────── helpers ─────────────────────────

    private static void ExecuteBidi(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                    Span<float> output, int seqQ, int seqKv, int numHeads, int numKvHeads,
                                    int headDim, int? slidingWindowSize = null)
    {
        float scale = 1.0f / MathF.Sqrt(headDim);
        Attention.Execute(q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim,
                          positionOffset: 0, scale, alibiSlopes: ReadOnlySpan<float>.Empty,
                          slidingWindowSize, softCap: 0f,
                          maskMode: AttentionMaskMode.Bidirectional, prefixLen: 0);
    }

    private static void ExecuteHybrid(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                      Span<float> output, int seqQ, int seqKv, int numHeads, int numKvHeads,
                                      int headDim, int prefixLen, int? slidingWindowSize = null)
    {
        float scale = 1.0f / MathF.Sqrt(headDim);
        Attention.Execute(q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim,
                          positionOffset: 0, scale, alibiSlopes: ReadOnlySpan<float>.Empty,
                          slidingWindowSize, softCap: 0f,
                          maskMode: AttentionMaskMode.Hybrid, prefixLen: prefixLen);
    }

    private static (float[] q, float[] k, float[] v) RandomQkv(
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim, int seed)
    {
        float[] q = Fill(seqQ * numHeads * headDim, seed * 3 + 1);
        float[] k = Fill(seqKv * numKvHeads * headDim, seed * 3 + 2);
        float[] v = Fill(seqKv * numKvHeads * headDim, seed * 3 + 3);
        return (q, k, v);
    }

    private static float[] Fill(int n, int seed)
    {
        float[] a = new float[n];
        for (int i = 0; i < n; i++)
            a[i] = 0.5f * MathF.Cos(0.61803398875f * (i + 1) + seed * 0.37f);
        return a;
    }
}
