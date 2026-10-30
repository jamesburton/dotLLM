using DotLLM.Core.Attention;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// <see cref="AttentionMaskMode.SymmetricWindow"/> (EmbeddingGemma, issue #1399): a layer with window <c>W</c> sees keys
/// <c>q - W/2 .. q + W/2</c> inclusive (llama.cpp <c>LLAMA_SWA_TYPE_SYMMETRIC</c>), a layer with no window is fully
/// bidirectional. The reference below is an independent double-precision brute force that states the rule directly.
/// </summary>
public sealed class SymmetricWindowAttentionTests
{
    private const int HeadDim = 8, NumHeads = 2, NumKvHeads = 1;

    [Theory]
    [InlineData(40, 8)]      // short rows: one-shot path
    [InlineData(40, 9)]      // odd window: W/2 rounds down
    [InlineData(700, 64)]    // long rows: tiled path (more keys than the one-shot limit)
    public void Matches_bruteforce_reference(int seq, int window)
    {
        var (q, k, v) = Qkv(seq);
        float[] expected = Reference(q, k, v, seq, window);

        float[] actual = new float[seq * NumHeads * HeadDim];
        Attention.Execute(q, k, v, actual, seq, seq, NumHeads, NumKvHeads, HeadDim, positionOffset: 0, Scale,
            ReadOnlySpan<float>.Empty, window, softCap: 0f, AttentionMaskMode.SymmetricWindow, prefixLen: 0);
        AssertClose(expected, actual);

        float[] scalar = new float[actual.Length];
        Attention.ExecuteScalar(q, k, v, scalar, seq, seq, NumHeads, NumKvHeads, HeadDim, 0, Scale,
            ReadOnlySpan<float>.Empty, window, 0f, AttentionMaskMode.SymmetricWindow, 0);
        AssertClose(expected, scalar);
    }

    [Fact]
    public void No_window_is_fully_bidirectional()
    {
        const int seq = 40;
        var (q, k, v) = Qkv(seq);
        float[] sym = new float[seq * NumHeads * HeadDim];
        float[] bidi = new float[sym.Length];
        Attention.Execute(q, k, v, sym, seq, seq, NumHeads, NumKvHeads, HeadDim, 0, Scale,
            ReadOnlySpan<float>.Empty, null, 0f, AttentionMaskMode.SymmetricWindow, 0);
        Attention.Execute(q, k, v, bidi, seq, seq, NumHeads, NumKvHeads, HeadDim, 0, Scale,
            ReadOnlySpan<float>.Empty, null, 0f, AttentionMaskMode.Bidirectional, 0);
        Assert.Equal(bidi, sym);
    }

    [Fact]
    public void Window_changes_the_result_and_differs_from_a_causal_lookback_window()
    {
        // Discriminates the three plausible implementations: symmetric (right), lookback-only window with no future
        // keys (what Bidirectional + sliding window computes, its VisibleStart has no upper bound twin) and no window.
        const int seq = 40, window = 8;
        var (q, k, v) = Qkv(seq);
        float[] sym = new float[seq * NumHeads * HeadDim];
        float[] lookbackOnly = new float[sym.Length];
        float[] noWindow = new float[sym.Length];
        Attention.Execute(q, k, v, sym, seq, seq, NumHeads, NumKvHeads, HeadDim, 0, Scale,
            ReadOnlySpan<float>.Empty, window, 0f, AttentionMaskMode.SymmetricWindow, 0);
        Attention.Execute(q, k, v, lookbackOnly, seq, seq, NumHeads, NumKvHeads, HeadDim, 0, Scale,
            ReadOnlySpan<float>.Empty, window, 0f, AttentionMaskMode.Bidirectional, 0);
        Attention.Execute(q, k, v, noWindow, seq, seq, NumHeads, NumKvHeads, HeadDim, 0, Scale,
            ReadOnlySpan<float>.Empty, null, 0f, AttentionMaskMode.Bidirectional, 0);
        Assert.True(MaxDiff(sym, lookbackOnly) > 1e-3f);
        Assert.True(MaxDiff(sym, noWindow) > 1e-3f);
    }

    private static float Scale => 1f / MathF.Sqrt(HeadDim);

    private static float[] Reference(float[] q, float[] k, float[] v, int seq, int window)
    {
        int half = window / 2;
        int qStride = NumHeads * HeadDim, kvStride = NumKvHeads * HeadDim;
        var o = new float[seq * qStride];
        for (int h = 0; h < NumHeads; h++)
        {
            int kvh = h / (NumHeads / NumKvHeads);
            for (int i = 0; i < seq; i++)
            {
                var scores = new double[seq];
                double max = double.NegativeInfinity;
                for (int j = 0; j < seq; j++)
                {
                    if (Math.Abs(i - j) > half) { scores[j] = double.NegativeInfinity; continue; }
                    double dot = 0;
                    for (int d = 0; d < HeadDim; d++) dot += (double)q[i * qStride + h * HeadDim + d] * k[j * kvStride + kvh * HeadDim + d];
                    scores[j] = dot * Scale;
                    max = Math.Max(max, scores[j]);
                }
                double sum = 0;
                for (int j = 0; j < seq; j++) { scores[j] = double.IsNegativeInfinity(scores[j]) ? 0 : Math.Exp(scores[j] - max); sum += scores[j]; }
                for (int d = 0; d < HeadDim; d++)
                {
                    double acc = 0;
                    for (int j = 0; j < seq; j++) acc += scores[j] / sum * v[j * kvStride + kvh * HeadDim + d];
                    o[i * qStride + h * HeadDim + d] = (float)acc;
                }
            }
        }
        return o;
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        Assert.True(MaxDiff(expected, actual) < 2e-3f, $"max abs diff {MaxDiff(expected, actual)}");
    }

    private static float MaxDiff(float[] a, float[] b)
    {
        float m = 0;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    private static (float[] q, float[] k, float[] v) Qkv(int seq)
    {
        float[] Fill(int n, int seed)
        {
            var a = new float[n];
            for (int i = 0; i < n; i++) a[i] = 0.9f * MathF.Cos(0.61803398875f * (i + 1) + seed * 0.37f);
            return a;
        }
        return (Fill(seq * NumHeads * HeadDim, 1), Fill(seq * NumKvHeads * HeadDim, 2), Fill(seq * NumKvHeads * HeadDim, 3));
    }
}
