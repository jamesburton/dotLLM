using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Selectors;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Recompute selectors (Task 7). Mechanics (exact budget, sorted, distinct, determinism, first-k shape, contiguous runs) use a
/// token-only <see cref="ChunkedContext"/>; the deviation and query-attention score sources run the synthetic dense
/// <c>qwen3</c> model (<see cref="SyntheticDenseGguf"/>) and are checked against independent oracles (full prefill K/V,
/// the model's own attention output).
/// </summary>
public sealed class SelectorTests : IDisposable
{
    private const string Fingerprint = "test-fingerprint";

    // Four chunks (L = 22) + a 3-token query; same ids as SelectiveRecomputeTests. Context + query fits Context = 64.
    private static readonly int[] ChunkA = [3, 17, 5, 29, 11, 40];
    private static readonly int[] ChunkB = [8, 22, 1, 35];
    private static readonly int[] ChunkC = [12, 7, 44, 2, 19];
    private static readonly int[] ChunkD = [31, 26, 0, 45, 16, 23, 10];
    private static readonly int[] Query = [9, 14, 38];

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-selectors-" + Guid.NewGuid().ToString("N"));
    private readonly ITestOutputHelper _output;

    public SelectorTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); }
        catch (IOException) { /* best effort: a still-mapped file on Windows */ }
        catch (UnauthorizedAccessException) { }
    }

    private static ChunkedContext Context() => new([ChunkA, ChunkB, ChunkC, ChunkD], Query);

    /// <summary>Scores that are distinct, non-monotone in position, and highest inside chunk A (which every score selector must skip).</summary>
    private static float[] PlantedScores(int n)
    {
        var s = new float[n];
        for (int i = 0; i < n; i++) s[i] = (i * 7919 % 23) + 0.5f * i / n;
        s[1] = 1000f; // chunk A: exact under naive concat, must be ranked after every stale token
        return s;
    }

    private sealed class FixedScores(float[] scores) : ITokenScoreSource
    {
        public int Calls { get; private set; }

        public float[] Score(ChunkedContext context)
        {
            Calls++;
            return scores;
        }
    }

    private static IRecomputeSelector MakeSelector(string name)
    {
        var src = new FixedScores(PlantedScores(Context().Length));
        return name switch
        {
            "first_k" => new FirstKSelector(),
            "first_k_2" => new FirstKSelector(k: 2),
            "chunk_starts" => new ChunkStartsSelector(),
            "deviation" => new DeviationTopKSelector(src),
            "random" => new RandomSelector(),
            "random_all" => new RandomSelector(includeFirstChunk: true),
            "query" => new QueryRelevanceSelector(src),
            "combined" => new CombinedSelector(src, startTokens: 2),
            "runs_deviation" => new ContiguousRuns(new DeviationTopKSelector(src), window: 3),
            "runs_random" => new ContiguousRuns(new RandomSelector(), window: 4),
            "runs_first_k" => new ContiguousRuns(new FirstKSelector(k: 2)),
            _ => throw new ArgumentException(name),
        };
    }

    public static TheoryData<string, int> SelectorsAndBudgets()
    {
        var data = new TheoryData<string, int>();
        string[] names = ["first_k", "first_k_2", "chunk_starts", "deviation", "random", "random_all", "query", "combined",
            "runs_deviation", "runs_random", "runs_first_k"];
        foreach (string n in names)
            foreach (int b in new[] { 0, 1, 5, 9, 16, 17, 21, 22, 30 })
                data.Add(n, b);
        return data;
    }

    [Theory]
    [MemberData(nameof(SelectorsAndBudgets))]
    public void AllSelectors_ReturnExactBudget_SortedDistinct(string selectorName, int budget)
    {
        ChunkedContext ctx = Context();
        IRecomputeSelector selector = MakeSelector(selectorName);
        foreach (int seed in new[] { 0, 1, 7 })
        {
            int[] picked = selector.Select(ctx, budget, seed);
            Assert.Equal(Math.Min(budget, ctx.Length), picked.Length);
            for (int i = 0; i < picked.Length; i++)
            {
                Assert.InRange(picked[i], 0, ctx.Length - 1);
                if (i > 0) Assert.True(picked[i] > picked[i - 1], $"{selectorName} b={budget}: not strictly ascending at {i}: {string.Join(',', picked)}");
            }
            // Stale tokens first: while the budget fits in chunks after the first, no chunk-A token (exact under naive concat)
            // is picked, except by the control that is told to include it.
            if (selectorName != "random_all" && budget <= ctx.Length - ChunkA.Length)
                Assert.DoesNotContain(picked, p => p < ChunkA.Length);
        }
        Assert.Throws<ArgumentOutOfRangeException>(() => selector.Select(ctx, -1, 0));
    }

    [Fact]
    public void AllSelectors_Nested_SmallerBudgetIsPrefixOfLarger()
    {
        // Budget curves must not be seed noise: each selector's pick at budget b is contained in its pick at b+1.
        ChunkedContext ctx = Context();
        foreach (string name in new[] { "first_k", "chunk_starts", "deviation", "random", "query", "combined", "runs_deviation", "runs_random" })
        {
            IRecomputeSelector s = MakeSelector(name);
            for (int b = 0; b < ctx.Length; b++)
            {
                var small = s.Select(ctx, b, 3).ToHashSet();
                Assert.True(small.IsSubsetOf(s.Select(ctx, b + 1, 3)), $"{name}: budget {b} is not nested in {b + 1}");
            }
        }
    }

    [Fact]
    public void RandomSelector_SameSeed_SameOutput_DifferentSeed_DifferentOutput()
    {
        ChunkedContext ctx = Context();
        var sel = new RandomSelector();
        int[] a = sel.Select(ctx, 6, seed: 42);
        int[] b = new RandomSelector().Select(ctx, 6, seed: 42);
        Assert.Equal(a, b);
        // Over several seeds the picks are not all equal (one coincidence is possible, eight are not for C(16,6) = 8008 sets).
        var distinct = Enumerable.Range(0, 8).Select(s => string.Join(',', sel.Select(ctx, 6, s))).Distinct().Count();
        Assert.True(distinct >= 6, $"only {distinct} distinct picks over 8 seeds");
        Assert.NotEqual(a, sel.Select(ctx, 6, seed: 43));
        // And it is not a fixed pattern: the picks are spread, not the first/last stale tokens.
        Assert.NotEqual(Enumerable.Range(ChunkA.Length, 6).ToArray(), a);
    }

    [Fact]
    public void FirstKSelector_PicksFirstKTokensOfEachChunk()
    {
        ChunkedContext ctx = Context();
        int[] starts = ctx.ChunkStarts;
        Assert.Equal([0, 6, 10, 15], starts);

        // Budget = k per chunk after the first (EPIC/LegoLink skips chunk 0, which is exact under naive concat).
        var sel = new FirstKSelector(k: 2);
        int[] expected = [6, 7, 10, 11, 15, 16];
        Assert.Equal(expected, sel.Select(ctx, 6, seed: 0));
        Assert.Equal(expected, sel.Select(ctx, 6, seed: 9)); // static: seed-independent

        // Smaller budget: offset 0 of every stale chunk first (the attention-sink token), then offset 1.
        Assert.Equal([6, 10, 15], sel.Select(ctx, 3, 0));
        Assert.Equal([6, 7, 10, 15], sel.Select(ctx, 4, 0));

        // k = 3 at budget 9 (chunk B has 4 tokens, so its first 3 fit).
        Assert.Equal([6, 7, 8, 10, 11, 12, 15, 16, 17], new FirstKSelector(k: 3).Select(ctx, 9, 0));

        // Beyond k per chunk the remainder is spread over the rest of the stale chunks, not appended at one chunk start.
        int[] more = sel.Select(ctx, 9, 0);
        Assert.Superset(expected.ToHashSet(), more.ToHashSet());
        int[] extra = more.Except(expected).ToArray();
        Assert.True(extra.Select(ctx.ChunkOf).Distinct().Count() >= 2, $"remainder not spread: {string.Join(',', extra)}");
    }

    [Fact]
    public void ChunkStartsSelector_SplitsBudgetAsEqualPrefixesOfStaleChunks()
    {
        ChunkedContext ctx = Context();
        var sel = new ChunkStartsSelector();
        Assert.Equal([6, 7, 8, 10, 11, 12, 15, 16, 17], sel.Select(ctx, 9, 0));
        // Chunk B (4 tokens) runs out first; the rest goes on deeper into C and D.
        Assert.Equal([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], sel.Select(ctx, 13, 0));
    }

    [Fact]
    public void ScoreSelectors_RankByScoreDescending_SkippingFirstChunk_ScoreComputedOnce()
    {
        ChunkedContext ctx = Context();
        float[] scores = PlantedScores(ctx.Length);
        int[] expectedOrder = Enumerable.Range(ChunkA.Length, ctx.Length - ChunkA.Length)
            .OrderByDescending(i => scores[i]).ThenBy(i => i).ToArray();
        foreach (Func<ITokenScoreSource, IRecomputeSelector> make in new Func<ITokenScoreSource, IRecomputeSelector>[]
                 { s => new DeviationTopKSelector(s), s => new QueryRelevanceSelector(s) })
        {
            var src = new FixedScores(scores);
            IRecomputeSelector sel = make(src);
            for (int b = 1; b <= 8; b++)
                Assert.Equal(expectedOrder.Take(b).Order().ToArray(), sel.Select(ctx, b, 0));
            Assert.True(src.Calls >= 1);
        }
    }

    [Fact]
    public void CombinedSelector_TakesChunkStartsThenDeviationTopK()
    {
        ChunkedContext ctx = Context();
        float[] scores = PlantedScores(ctx.Length);
        var sel = new CombinedSelector(new FixedScores(scores), startTokens: 1);
        int[] starts = [6, 10, 15];
        int[] rest = Enumerable.Range(ChunkA.Length, ctx.Length - ChunkA.Length).Except(starts)
            .OrderByDescending(i => scores[i]).ThenBy(i => i).ToArray();
        Assert.Equal(starts, sel.Select(ctx, 3, 0));
        Assert.Equal(starts.Concat(rest.Take(4)).Order().ToArray(), sel.Select(ctx, 7, 0));
    }

    [Fact]
    public void ContiguousRuns_ExpandsEachPickForwardIntoRun_WithinBudget()
    {
        ChunkedContext ctx = Context();
        float[] scores = new float[ctx.Length];
        scores[12] = 10; scores[19] = 9; scores[7] = 8; scores[20] = 7; // planted order: 12, 19, 7, 20, ...
        var runs = new ContiguousRuns(new DeviationTopKSelector(new FixedScores(scores)), window: 3);
        Assert.Equal([12, 13, 14], runs.Select(ctx, 3, 0));
        Assert.Equal([12, 13, 14, 19, 20], runs.Select(ctx, 5, 0));               // second run truncated at the budget
        Assert.Equal([12, 13, 14, 19, 20, 21], runs.Select(ctx, 6, 0));           // 19..21 clipped at the context end (L = 22)
        Assert.Equal([7, 8, 9, 12, 13, 14, 19, 20, 21], runs.Select(ctx, 9, 0));  // 20 already covered: skipped as an anchor
        // An already contiguous selector is unchanged by the decorator.
        var firstK = new FirstKSelector(k: 3);
        Assert.Equal(firstK.Select(ctx, 9, 0), new ContiguousRuns(firstK, window: 3).Select(ctx, 9, 0));
    }

    // ---- model-backed score sources ----

    [Fact]
    public void DeviationTopKSelector_RanksByLayer1KvDeviation()
    {
        using var tiny = TinyModel.Load(_scratch);
        ChunkedContext ctx = Context();
        var chunks = Encode(tiny, ChunkA, ChunkB, ChunkC, ChunkD);
        using SimpleKvCache stale = Assembled(tiny, chunks);

        // Plant: token 13 (chunk C, otherwise ordinary) gets a large perturbation of its stale layer-1 K and V rows. The
        // fresh layer-1 K/V the probe recomputes do not depend on the stale layer-1 rows, so token 13 must rank first.
        const int planted = 13;
        PerturbRow(stale, layer: 1, planted, delta: 25f);
        float[] keys0Before = stale.KeysSpan(0).ToArray();

        var source = new Layer1KvDeviationSource(tiny.Model, stale);
        float[] dev = source.Score(ctx);
        _output.WriteLine("layer-1 deviation: " + string.Join(' ', dev.Select(d => d.ToString("F3"))));

        Assert.Equal(ctx.Length, dev.Length);
        Assert.Equal(planted, Array.IndexOf(dev, dev.Max()));
        var sel = new DeviationTopKSelector(source);
        Assert.Equal([planted], sel.Select(ctx, 1, 0));
        // The probe does not touch the stale cache it reads (no free fix-up for the deviation selector).
        Assert.Equal(keys0Before, stale.KeysSpan(0).ToArray());
        Assert.Equal(ctx.Length, stale.CurrentLength);

        // Without the plant the ranking follows the true deviation: top-4 equals the full-prefill oracle's top-4.
        using SimpleKvCache clean = Assembled(tiny, chunks);
        float[] oracle = Layer1DeviationFromFullPrefill(tiny, ctx, clean);
        int[] oracleTop = Enumerable.Range(ChunkA.Length, ctx.Length - ChunkA.Length)
            .OrderByDescending(i => oracle[i]).Take(4).Order().ToArray();
        Assert.Equal(oracleTop, new DeviationTopKSelector(new Layer1KvDeviationSource(tiny.Model, clean)).Select(ctx, 4, 0));
    }

    [Fact]
    public void Layer1KvDeviationSource_EqualsFullPrefillLayer1Deviation_AndIsZeroOnFirstChunk()
    {
        using var tiny = TinyModel.Load(_scratch);
        ChunkedContext ctx = Context();
        using SimpleKvCache stale = Assembled(tiny, Encode(tiny, ChunkA, ChunkB, ChunkC, ChunkD));

        float[] dev = new Layer1KvDeviationSource(tiny.Model, stale).Score(ctx);
        float[] oracle = Layer1DeviationFromFullPrefill(tiny, ctx, stale);
        for (int i = 0; i < ctx.Length; i++)
            Assert.True(MathF.Abs(dev[i] - oracle[i]) <= 1e-4f * MathF.Max(1f, oracle[i]), $"token {i}: probe {dev[i]} vs oracle {oracle[i]}");
        for (int i = 0; i < ChunkA.Length; i++)
            Assert.True(dev[i] <= 1e-5f, $"chunk-0 token {i} has deviation {dev[i]}");
        Assert.True(dev.Skip(ChunkA.Length).Min() > 1e-3f, "fixture cannot discriminate: a stale token has no layer-1 deviation");
    }

    [Fact]
    public void OracleKvDeviationSource_SumsDeviationOverAllLayers()
    {
        using var tiny = TinyModel.Load(_scratch);
        ChunkedContext ctx = Context();
        using SimpleKvCache stale = Assembled(tiny, Encode(tiny, ChunkA, ChunkB, ChunkC, ChunkD));
        using SimpleKvCache full = NewCache(tiny);
        using (tiny.Model.Forward(ctx.TokenIds, Enumerable.Range(0, ctx.Length).ToArray(), -1, full, lastTokenLogitsOnly: true)) { }

        float[] dev = new OracleKvDeviationSource(full, stale).Score(ctx);
        for (int i = 0; i < ctx.Length; i++)
        {
            double sq = 0;
            for (int l = 0; l < full.NumLayers; l++) sq += RowSqDiff(full, stale, l, i);
            Assert.True(Math.Abs(dev[i] - Math.Sqrt(sq)) <= 1e-4 * Math.Max(1, Math.Sqrt(sq)), $"token {i}");
        }
        Assert.True(dev.Take(ChunkA.Length).Max() <= 1e-5f);
        // Deeper layers add deviation beyond layer 1 (layers 1..2 of the 3-layer fixture both count).
        float[] l1 = Layer1DeviationFromFullPrefill(tiny, ctx, stale);
        Assert.Contains(Enumerable.Range(ChunkA.Length, ctx.Length - ChunkA.Length), i => dev[i] > l1[i] + 1e-3f);
    }

    [Fact]
    public void QueryAttentionSource_ProbabilitiesReproduceModelAttentionOutput()
    {
        // Oracle: for every layer and query row, Σ_j p_j · V_j (our probabilities, the cache's V) must equal the attention
        // output the model computed. That pins the scale, the causal mask, the GQA head mapping and the softmax.
        using var tiny = TinyModel.Load(_scratch);
        ChunkedContext ctx = Context();
        using SimpleKvCache stale = Assembled(tiny, Encode(tiny, ChunkA, ChunkB, ChunkC, ChunkD));
        ModelConfig cfg = tiny.Config;
        int heads = cfg.NumAttentionHeads, kvHeads = cfg.NumKvHeads, hd = cfg.HeadDim;

        float worst = 0;
        int checkedRows = 0;
        tiny.Model.AttentionQueryProbe = (int layer, ReadOnlySpan<float> q, ReadOnlySpan<float> attnOut, ReadOnlySpan<int> positions,
            int numHeads, int numKvHeads, int headDim, float scale, DotLLM.Core.Attention.IKvCache cache) =>
        {
            var c = (SimpleKvCache)cache;
            ReadOnlySpan<float> keys = c.KeysSpan(layer), values = c.ValuesSpan(layer);
            int kvStride = numKvHeads * headDim;
            for (int r = 0; r < positions.Length; r++)
            {
                int visible = positions[r] + 1;
                var probs = new float[numHeads * visible];
                QueryAttention.Probabilities(q.Slice(r * numHeads * headDim, numHeads * headDim), keys, visible,
                    numHeads, numKvHeads, headDim, scale, probs);
                for (int h = 0; h < numHeads; h++)
                {
                    int kvh = h / (numHeads / numKvHeads);
                    for (int d = 0; d < headDim; d++)
                    {
                        double o = 0;
                        for (int j = 0; j < visible; j++) o += probs[h * visible + j] * values[j * kvStride + kvh * headDim + d];
                        worst = MathF.Max(worst, MathF.Abs((float)o - attnOut[r * numHeads * headDim + h * headDim + d]));
                    }
                }
                checkedRows++;
            }
        };
        try
        {
            using (tiny.Model.Forward(Query, Enumerable.Range(ctx.Length, Query.Length).ToArray(), -1, stale)) { }
        }
        finally { tiny.Model.AttentionQueryProbe = null; }

        _output.WriteLine($"max |Σ p·V − attention output| = {worst:E3} over {checkedRows} rows");
        Assert.Equal(cfg.NumLayers * Query.Length, checkedRows);
        Assert.True(worst <= 1e-4f, $"probabilities do not reproduce the model's attention output: {worst}");
        Assert.Equal(heads, cfg.NumAttentionHeads);
        Assert.NotEqual(heads, kvHeads); // GQA fixture: the head mapping is exercised
        Assert.True(hd > 0);
    }

    [Fact]
    public void QueryAttentionSource_ScoresAreMeanAttentionMassOnContext_AndRestoreTheCache()
    {
        using var tiny = TinyModel.Load(_scratch);
        ChunkedContext ctx = Context();
        using SimpleKvCache stale = Assembled(tiny, Encode(tiny, ChunkA, ChunkB, ChunkC, ChunkD));
        float[][] keysBefore = Enumerable.Range(0, stale.NumLayers).Select(l => stale.KeysSpan(l).ToArray()).ToArray();

        float[] s = new QueryAttentionSource(tiny.Model, stale).Score(ctx);

        Assert.Equal(ctx.Length, s.Length);
        Assert.All(s, v => Assert.InRange(v, 0f, 1f));
        float mass = s.Sum();
        // Mean over layers, heads and query rows of the probability mass on context keys: < 1 (query keys take the rest), > 0.
        Assert.InRange(mass, 0.05f, 1f - 1e-4f);
        Assert.Equal(ctx.Length, stale.CurrentLength);
        for (int l = 0; l < stale.NumLayers; l++) Assert.Equal(keysBefore[l], stale.KeysSpan(l).ToArray());
        Assert.Null(tiny.Model.AttentionQueryProbe);

        // Independent check of the aggregation: recompute the same mean from the probe directly.
        var expect = new double[ctx.Length];
        int rows = 0;
        tiny.Model.AttentionQueryProbe = (int layer, ReadOnlySpan<float> q, ReadOnlySpan<float> attnOut, ReadOnlySpan<int> positions,
            int numHeads, int numKvHeads, int headDim, float scale, DotLLM.Core.Attention.IKvCache cache) =>
        {
            ReadOnlySpan<float> keys = ((SimpleKvCache)cache).KeysSpan(layer);
            for (int r = 0; r < positions.Length; r++)
            {
                int visible = positions[r] + 1;
                var probs = new float[numHeads * visible];
                QueryAttention.Probabilities(q.Slice(r * numHeads * headDim, numHeads * headDim), keys, visible,
                    numHeads, numKvHeads, headDim, scale, probs);
                for (int h = 0; h < numHeads; h++)
                    for (int j = 0; j < ctx.Length; j++) expect[j] += probs[h * visible + j] / numHeads;
                rows++;
            }
        };
        try
        {
            using (tiny.Model.Forward(Query, Enumerable.Range(ctx.Length, Query.Length).ToArray(), -1, stale)) { }
        }
        finally { tiny.Model.AttentionQueryProbe = null; }
        for (int j = 0; j < ctx.Length; j++)
            Assert.True(Math.Abs(s[j] - expect[j] / rows) <= 1e-5, $"token {j}: {s[j]} vs {expect[j] / rows}");
    }

    // ---- helpers ----

    private static double RowSqDiff(SimpleKvCache a, SimpleKvCache b, int layer, int row)
    {
        int stride = a.KeysSpan(layer).Length / a.CurrentLength;
        double sq = 0;
        ReadOnlySpan<float> ka = a.KeysSpan(layer).Slice(row * stride, stride), kb = b.KeysSpan(layer).Slice(row * stride, stride);
        ReadOnlySpan<float> va = a.ValuesSpan(layer).Slice(row * stride, stride), vb = b.ValuesSpan(layer).Slice(row * stride, stride);
        for (int i = 0; i < stride; i++)
        {
            double dk = ka[i] - kb[i], dv = va[i] - vb[i];
            sq += dk * dk + dv * dv;
        }
        return sq;
    }

    /// <summary>‖(K,V)_full − (K,V)_stale‖ at layer 1 per token, from an independent full prefill of the context.</summary>
    private static float[] Layer1DeviationFromFullPrefill(TinyModel tiny, ChunkedContext ctx, SimpleKvCache stale)
    {
        using SimpleKvCache full = NewCache(tiny);
        using (tiny.Model.Forward(ctx.TokenIds, Enumerable.Range(0, ctx.Length).ToArray(), -1, full, lastTokenLogitsOnly: true)) { }
        return Enumerable.Range(0, ctx.Length).Select(i => (float)Math.Sqrt(RowSqDiff(full, stale, 1, i))).ToArray();
    }

    private static unsafe void PerturbRow(SimpleKvCache cache, int layer, int row, float delta)
    {
        int stride = cache.KeysSpan(layer).Length / cache.CurrentLength;
        float[] k = cache.KeysSpan(layer).Slice(row * stride, stride).ToArray();
        float[] v = cache.ValuesSpan(layer).Slice(row * stride, stride).ToArray();
        for (int i = 0; i < stride; i++) { k[i] += delta; v[i] -= delta; }
        fixed (float* kp = k)
        fixed (float* vp = v)
        {
            cache.Update(new TensorRef(1, stride, DType.Float32, -1, (nint)kp), new TensorRef(1, stride, DType.Float32, -1, (nint)vp),
                [row], layer);
        }
    }

    private static List<ChunkKvEntry> Encode(TinyModel tiny, params int[][] chunks)
    {
        var encoder = new ChunkEncoder(Fingerprint);
        return chunks.Select(c => encoder.Encode(tiny.Model, c)).ToList();
    }

    private static SimpleKvCache NewCache(TinyModel tiny)
        => new(KvGeometry.FromConfig(tiny.Config, compactSlidingWindows: false), SyntheticDenseGguf.Context);

    private static SimpleKvCache Assembled(TinyModel tiny, IReadOnlyList<ChunkKvEntry> chunks)
    {
        var cache = NewCache(tiny);
        new ChunkAssembler(tiny.Config).Assemble(cache, chunks, Fingerprint);
        return cache;
    }

    /// <summary>A loaded synthetic qwen3 model; keeps the mapped GGUF alive for the model's lifetime.</summary>
    private sealed class TinyModel : IDisposable
    {
        private readonly GgufFile _gguf;

        private TinyModel(GgufFile gguf, ModelConfig config, TransformerModel model)
        {
            _gguf = gguf;
            Config = config;
            Model = model;
        }

        public ModelConfig Config { get; }

        public TransformerModel Model { get; }

        public static TinyModel Load(string scratch)
        {
            string path = Path.Combine(scratch, "qwen3.gguf");
            File.WriteAllBytes(path, SyntheticDenseGguf.Build("qwen3"));
            var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            return new TinyModel(gguf, config, TransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded));
        }

        public void Dispose()
        {
            Model.Dispose();
            _gguf.Dispose();
        }
    }
}
