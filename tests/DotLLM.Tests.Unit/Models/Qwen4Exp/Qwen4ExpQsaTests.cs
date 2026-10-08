using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Qwen4Exp;

/// <summary>
/// Qwen4-Exp QSA (#816): pooled-key cache, block selection (pool-then-rope, block-causal, relu-sum, top-k + tail) and
/// GQA over the selection, against the HF eager <c>Qwen4ExpTextAttention</c> (fixture from <c>Reference/gen_qsa.py</c>:
/// T=40 sparse with a 4-block budget, T=14 exactly dense), plus the &gt; 2051-token sparse-vs-dense discrimination.
/// </summary>
public sealed unsafe class Qwen4ExpQsaTests
{
    private static readonly Qwen4ExpReferenceFixture Fx = Qwen4ExpReferenceFixture.Load("qsa_attention.json");
    private readonly int _hidden = Fx.Int("hidden_size"), _nH = Fx.Int("num_heads"), _nKv = Fx.Int("num_kv_heads"), _d = Fx.Int("head_dim");
    private readonly int _ropeDim = Fx.Int("rope_dim"), _idxH = Fx.Int("idx_heads"), _idxD = Fx.Int("idx_dim"), _block = Fx.Int("block"), _budget = Fx.Int("budget_tokens");
    private readonly float _eps = (float)Fx.Meta.GetProperty("eps").GetDouble();

    private static void AssertClose(float[] expected, float[] actual, string what, float tol = 3e-5f)
    {
        Assert.Equal(expected.Length, actual.Length);
        float worst = 0; int at = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float d = MathF.Abs(expected[i] - actual[i]) / (1f + MathF.Abs(expected[i]));
            if (d > worst) { worst = d; at = i; }
        }
        Assert.True(worst <= tol, $"{what}: worst rel diff {worst:E3} at {at} (expected {expected[at]}, got {actual[at]})");
    }

    private static Qwen4ExpProjection F32Proj(float[] weight, int rowOffset, int outDim, int inDim) =>
        (input, output, tokens) =>
        {
            fixed (float* w = weight) fixed (float* i = input) fixed (float* o = output)
                MatMul.GemmF32(w + (long)rowOffset * inDim, i, o, outDim, inDim, tokens);
        };

    private (float[] Cos, float[] Sin) Rope(int maxPos)
    {
        float theta = (float)Fx.Meta.GetProperty("rope_theta").GetDouble();
        var cos = new float[maxPos * _ropeDim / 2]; var sin = new float[cos.Length];
        RoPE.PrecomputeFrequencyTable(maxPos, _ropeDim, theta, cos, sin);
        return (cos, sin);
    }

    private Qwen4ExpQsaLayer BuildLayer(int maxPos, int budgetTokens)
    {
        var (cos, sin) = Rope(maxPos);
        var qkv = Fx.F32("index_qk_proj");
        return new Qwen4ExpQsaLayer(_hidden, _nH, _nKv, _d, _ropeDim, _idxH, _idxD, _block, budgetTokens, _eps,
            Fx.F32("q_norm"), Fx.F32("k_norm"), Fx.F32("idx_q_norm"), Fx.F32("idx_k_norm"), cos, sin,
            F32Proj(Fx.F32("q_proj"), 0, 2 * _nH * _d, _hidden),
            F32Proj(Fx.F32("k_proj"), 0, _nKv * _d, _hidden),
            F32Proj(Fx.F32("v_proj"), 0, _nKv * _d, _hidden),
            F32Proj(Fx.F32("o_proj"), 0, _hidden, _nH * _d),
            F32Proj(qkv, 0, _idxH * _idxD, _hidden),            // HF fused index_qk_proj: q rows first ...
            F32Proj(qkv, _idxH * _idxD, _idxD, _hidden));       // ... then the single key head
    }

    private float[] RunLayer(Qwen4ExpQsaLayer layer, float[] x, int T, int chunk, Qwen4ExpQsaState? state = null)
    {
        state ??= layer.CreateState();
        var y = new float[T * _hidden];
        for (int i = 0; i < T; i += chunk)
        {
            int n = Math.Min(chunk, T - i);
            layer.Forward(x.AsSpan(i * _hidden, n * _hidden), n, state, y.AsSpan(i * _hidden, n * _hidden));
        }
        return y;
    }

    // ── indexer pieces ──────────────────────────────────────────────────────

    [Theory]
    [InlineData(40, 40)]
    [InlineData(40, 1)]
    [InlineData(40, 3)]
    [InlineData(40, 7)]
    [InlineData(14, 5)]
    public void IndexerCache_PooledKeys_MatchHf_AnyChunking(int T, int chunk)
    {
        var raw = Fx.F32($"t{T}.raw_keys");
        var (cos, sin) = Rope(64);
        var cache = new Qwen4ExpIndexerCache(_idxD, _block);
        for (int i = 0; i < T; i += chunk)
        {
            int n = Math.Min(chunk, T - i);
            cache.Append(raw.AsSpan(i * _idxD, n * _idxD), n, Fx.F32("idx_k_norm"), _eps, cos, sin, _ropeDim);
        }
        Assert.Equal(T / _block, cache.CompleteBlocks);
        AssertClose(Fx.F32($"t{T}.pooled"), cache.Pooled.ToArray(), "pooled keys vs HF");
    }

    [Fact]
    public void IndexerCache_PartialBlockIsNotPooledUntilComplete()
    {
        var raw = Fx.F32("t40.raw_keys");
        var (cos, sin) = Rope(64);
        var cache = new Qwen4ExpIndexerCache(_idxD, _block);
        cache.Append(raw.AsSpan(0, 3 * _idxD), 3, Fx.F32("idx_k_norm"), _eps, cos, sin, _ropeDim);
        Assert.Equal(0, cache.CompleteBlocks);
        cache.Append(raw.AsSpan(3 * _idxD, _idxD), 1, Fx.F32("idx_k_norm"), _eps, cos, sin, _ropeDim);
        Assert.Equal(1, cache.CompleteBlocks);
        AssertClose(Fx.F32("t40.pooled").AsSpan(0, _idxD).ToArray(), cache.Pooled.ToArray(), "block 0");
    }

    [Fact]
    public void SelectBlocks_ScoresAndSelection_MatchHf()
    {
        const int T = 40;
        var q = Fx.F32($"t{T}.idx_q");            // [T, heads, D]
        var pooled = Fx.F32($"t{T}.pooled");
        var scoresHf = Fx.F32($"t{T}.scores");    // [T, nb], -1 where invisible
        var mask = Fx.F32($"t{T}.sel_mask");      // [T, T]
        int nbTotal = T / _block, budgetBlocks = _budget / _block;
        bool sawSparse = false;
        for (int t = 0; t < T; t++)
        {
            int vis = (t + 1) / _block;
            var qt = q.AsSpan(t * _idxH * _idxD, _idxH * _idxD);
            var sel = new int[budgetBlocks];
            var sc = new float[nbTotal + 1];
            int n = Qwen4ExpQsa.SelectBlocks(qt, _idxH, _idxD, pooled, vis, budgetBlocks, sel, sc);
            Assert.Equal(Math.Min(budgetBlocks, vis), n);
            sawSparse |= vis > budgetBlocks;

            if (vis > budgetBlocks)
            {
                for (int b = 0; b < vis; b++)
                    Assert.True(Math.Abs(scoresHf[t * nbTotal + b] - sc[b]) <= 3e-5f * (1 + Math.Abs(sc[b])), $"score t={t} b={b}");
                var scalar = new float[nbTotal];
                Qwen4ExpQsa.ScoreBlocksScalar(qt, _idxH, _idxD, pooled, vis, scalar);
                for (int b = 0; b < vis; b++) Assert.True(Math.Abs(scalar[b] - sc[b]) <= 1e-5f * (1 + Math.Abs(sc[b])));
            }

            // Key list == HF mask row.
            var keyIdx = new int[t + 1];
            int count = Qwen4ExpQsa.BuildKeyList(t, _block, false, sel.AsSpan(0, n), keyIdx);
            var got = new float[T];
            for (int i = 0; i < count; i++) got[keyIdx[i]] = 1;
            Assert.Equal(mask.AsSpan(t * T, T).ToArray(), got);
            Assert.Equal(keyIdx.AsSpan(0, count).ToArray().OrderBy(i => i).ToArray(), keyIdx.AsSpan(0, count).ToArray());   // ascending
        }
        Assert.True(sawSparse);
    }

    [Fact]
    public void SelectBlocks_TiesBreakTowardLowerIndex()
    {
        // Every pooled key identical -> equal scores -> the lowest block ids win, deterministically.
        int heads = 2, dim = 8, blocks = 10, keep = 4;
        var q = Enumerable.Repeat(1f, heads * dim).ToArray();
        var pooled = Enumerable.Repeat(0.5f, blocks * dim).ToArray();
        var sel = new int[keep];
        Qwen4ExpQsa.SelectBlocks(q, heads, dim, pooled, blocks, keep, sel, new float[blocks]);
        Assert.Equal([0, 1, 2, 3], sel);
        // All-negative dot products rectify to 0 for every block (ties at 0) and still select deterministically.
        var neg = Enumerable.Repeat(-0.5f, blocks * dim).ToArray();
        Qwen4ExpQsa.SelectBlocks(q, heads, dim, neg, blocks, keep, sel, new float[blocks]);
        Assert.Equal([0, 1, 2, 3], sel);
    }

    // ── layer ───────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(40)]   // sparse engaged (10 complete blocks > 4-block budget)
    [InlineData(14)]   // 3 blocks <= budget: exactly dense
    public void Layer_MatchesHf(int T)
    {
        var layer = BuildLayer(64, _budget);
        var y = RunLayer(layer, Fx.F32($"t{T}.x"), T, T);
        AssertClose(Fx.F32($"t{T}.out"), y, $"layer T={T} vs HF");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(7)]
    [InlineData(16)]
    public void Layer_Chunked_EqualsSingleShot(int chunk)
    {
        var layer = BuildLayer(64, _budget);
        var x = Fx.F32("t40.x");
        var whole = RunLayer(layer, x, 40, 40);
        var chunked = RunLayer(layer, x, 40, chunk);
        AssertClose(whole, chunked, "chunked vs single-shot", 1e-5f);
    }

    [Fact]
    public void Layer_ForceDense_MatchesHfDense_AndDiffersFromSparse()
    {
        var x = Fx.F32("t40.x");
        var sparse = RunLayer(BuildLayer(64, _budget), x, 40, 40);

        var denseLayer = BuildLayer(64, _budget);
        denseLayer.ForceDense = true;
        var dense = RunLayer(denseLayer, x, 40, 40);
        AssertClose(Fx.F32("t40.dense_out"), dense, "forced-dense vs HF (budget huge)");

        // Large budget (indexer never prunes) is bitwise the same as forced dense.
        var bigBudget = RunLayer(BuildLayer(64, 4000), x, 40, 40);
        Assert.Equal(dense, bigBudget);

        // Rows whose context fits the budget agree exactly (first 19 queries: <= 4 complete blocks); later rows differ.
        int denseRows = _budget + _block - 1;   // 19 tokens: positions 0..18
        Assert.Equal(dense.AsSpan(0, denseRows * _hidden).ToArray(), sparse.AsSpan(0, denseRows * _hidden).ToArray());
        float maxDiff = 0;
        for (int i = denseRows * _hidden; i < sparse.Length; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(sparse[i] - dense[i]));
        Assert.True(maxDiff > 1e-3f, $"sparse and dense coincide beyond the budget (max diff {maxDiff})");
    }

    [Fact]
    public void Layer_PositionBeyondRopeTable_Throws()
    {
        var layer = BuildLayer(16, _budget);
        var x = Fx.F32("t40.x");
        Assert.Throws<ArgumentOutOfRangeException>(() => RunLayer(layer, x, 40, 40));
    }

    [Fact]
    public void Layer_ContextBeyondRealBudget_SparseDiffersFromDense_ExactlyBelow2051()
    {
        // Real budget (2048 tokens / block 4 = 512 blocks) at tiny width: T = 2200 > 2051.
        const int T = 2200, budget = 2048;
        var layer = BuildLayer(T, budget);
        var rng = new Random(77);
        var x = new float[T * _hidden];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * 1.5f;

        var sparse = RunLayer(layer, x, T, 256);
        var denseLayer = BuildLayer(T, budget);
        denseLayer.ForceDense = true;
        var dense = RunLayer(denseLayer, x, T, 256);

        int exactRows = budget + _block - 1;   // positions 0..2050 (n <= 2051) are exactly dense
        Assert.Equal(dense.AsSpan(0, exactRows * _hidden).ToArray(), sparse.AsSpan(0, exactRows * _hidden).ToArray());

        float maxDiff = 0;
        for (int i = exactRows * _hidden; i < sparse.Length; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(sparse[i] - dense[i]));
        Assert.True(maxDiff > 1e-4f, $"QSA never pruned beyond 2051 tokens (max diff {maxDiff})");

        // Chunking invariance holds across the budget boundary too.
        var oneShot = RunLayer(BuildLayer(T, budget), x, T, T);
        AssertClose(oneShot, sparse, "sparse chunk 256 vs single shot", 1e-5f);
    }
}
