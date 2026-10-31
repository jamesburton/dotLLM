using ChunkKvLab;
using DotLLM.ChunkReuse;
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
/// Selective recompute (CacheBlend-style fix-up) over a naive-concat chunk cache, on the synthetic dense <c>qwen3</c> model
/// (<see cref="SyntheticDenseGguf"/>, F32, CPU). The quality oracle is a full prefill of the concatenated context; the error
/// of a cache is measured on the logits of a suffix forwarded after the context (the E1 flow).
/// </summary>
public sealed class SelectiveRecomputeTests : IDisposable
{
    private const string Fingerprint = "test-fingerprint";

    // Four chunks (L = 22) and a suffix; distinct ids so rows are distinguishable. The context + suffix fits Context = 64.
    private static readonly int[] ChunkA = [3, 17, 5, 29, 11, 40];
    private static readonly int[] ChunkB = [8, 22, 1, 35];
    private static readonly int[] ChunkC = [12, 7, 44, 2, 19];
    private static readonly int[] ChunkD = [31, 26, 0, 45, 16, 23, 10];
    private static readonly int[] Suffix = [9, 14, 38];

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-recompute-" + Guid.NewGuid().ToString("N"));
    private readonly ITestOutputHelper _output;

    public SelectiveRecomputeTests(ITestOutputHelper output)
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

    [Fact]
    public void Recompute_AllTokensSelected_MatchesFullPrefill()
    {
        using var tiny = TinyModel.Load(_scratch);
        var chunks = Encode(tiny, ChunkA, ChunkB, ChunkC, ChunkD);
        int[] context = Concat(ChunkA, ChunkB, ChunkC, ChunkD);
        int n = context.Length;

        float[] reference = FullPrefillSuffixRows(tiny, context, out float[] referenceLastContext);
        float[] naive = NaiveSuffixRows(tiny, chunks, n);
        Assert.True(MaxAbsDiff(naive, reference) > 1e-2f, "fixture cannot discriminate: naive concat already matches full prefill");

        using var cache = Assembled(tiny, chunks);
        SelectiveRecomputeResult result = SelectiveRecomputer.Run(tiny.Model, cache, chunks, Enumerable.Range(0, n).ToArray());

        Assert.Equal(1, result.ForwardCalls); // all positions = one contiguous run = a normal prefill over the cache
        Assert.Equal(n, result.TokensRecomputed);
        Assert.Equal(n, cache.CurrentLength);

        // The last context position was recomputed, so its next-token logits are returned.
        Assert.NotNull(result.LastPositionLogits);
        float lastErr = MaxAbsDiff(result.LastPositionLogits!, referenceLastContext);
        Assert.True(lastErr <= 1e-4f, $"last-position logits differ from full prefill by {lastErr}");

        float err = MaxAbsDiff(SuffixRows(tiny, cache, n), reference);
        _output.WriteLine($"max |recompute(all) - full prefill| = {err:E3}; naive = {MaxAbsDiff(naive, reference):E3}");
        Assert.True(err <= 1e-4f, $"all-selected recompute differs from full prefill by {err}");
    }

    [Fact]
    public void Recompute_NoTokensSelected_EqualsNaiveConcat()
    {
        using var tiny = TinyModel.Load(_scratch);
        var chunks = Encode(tiny, ChunkA, ChunkB, ChunkC, ChunkD);
        int n = chunks.Sum(c => c.TokenIds.Length);

        using var cache = Assembled(tiny, chunks);
        float[][] keysBefore = Snapshot(cache, keys: true);
        float[][] valuesBefore = Snapshot(cache, keys: false);

        SelectiveRecomputeResult result = SelectiveRecomputer.Run(tiny.Model, cache, chunks, ReadOnlySpan<int>.Empty);

        Assert.Equal(0, result.ForwardCalls);
        Assert.Equal(0, result.TokensRecomputed);
        Assert.Null(result.LastPositionLogits);
        Assert.Equal(n, cache.CurrentLength);
        AssertCacheEqual(keysBefore, valuesBefore, cache);
        Assert.Equal(NaiveSuffixRows(tiny, chunks, n), SuffixRows(tiny, cache, n));
    }

    [Fact]
    public void Recompute_Monotone_ErrorShrinksAsMoreTokensSelected()
    {
        using var tiny = TinyModel.Load(_scratch);
        var chunks = Encode(tiny, ChunkA, ChunkB, ChunkC, ChunkD);
        int[] context = Concat(ChunkA, ChunkB, ChunkC, ChunkD);
        int n = context.Length;
        float[] reference = FullPrefillSuffixRows(tiny, context, out _);

        // Nested selections: the first k of a fixed order over the stale positions (chunk A sits at offset 0 and is already
        // exact, so its positions come last). 10% ⊂ 50% ⊂ 100%.
        int[] order = Enumerable.Range(ChunkA.Length, n - ChunkA.Length).ToArray();
        new Random(7).Shuffle(order);
        order = [.. order, .. Enumerable.Range(0, ChunkA.Length)];

        double[] fractions = [0, 0.1, 0.5, 1.0];
        var kl = new double[fractions.Length];
        for (int f = 0; f < fractions.Length; f++)
        {
            int k = (int)Math.Round(fractions[f] * n);
            int[] selected = order[..k];
            Array.Sort(selected);
            using var cache = Assembled(tiny, chunks);
            SelectiveRecomputer.Run(tiny.Model, cache, chunks, selected);
            kl[f] = Metrics.MeanKl(reference, SuffixRows(tiny, cache, n), tiny.Config.VocabSize);
            _output.WriteLine($"{fractions[f]:P0} ({k} tokens): mean KL = {kl[f]:E3}");
        }

        Assert.True(kl[0] > 1e-3, $"fixture cannot discriminate: naive-concat KL {kl[0]:E3} is ~0");
        Assert.True(kl[^1] < 1e-8, $"100% selection must reproduce full prefill (KL {kl[^1]:E3})");
        const double noise = 1e-6;
        for (int f = 1; f < kl.Length; f++)
            Assert.True(kl[f] <= kl[f - 1] + noise, $"KL rose from {kl[f - 1]:E3} at {fractions[f - 1]:P0} to {kl[f]:E3} at {fractions[f]:P0}");
        Assert.True(kl[2] < kl[0], "50% selection did not reduce the error at all");
    }

    [Fact]
    public void Recompute_StaleChunkOnly_MatchesFullPrefill()
    {
        // [A; B]: A at offset 0 is exact (causal, same positions), only B is stale. Recomputing B's positions alone must give
        // the full-prefill cache. Catches wrong positions passed to Forward.
        using var tiny = TinyModel.Load(_scratch);
        var chunks = Encode(tiny, ChunkA, ChunkB);
        int[] context = Concat(ChunkA, ChunkB);
        int n = context.Length;
        float[] reference = FullPrefillSuffixRows(tiny, context, out float[] referenceLast);
        Assert.True(MaxAbsDiff(NaiveSuffixRows(tiny, chunks, n), reference) > 1e-2f, "fixture cannot discriminate");

        using var cache = Assembled(tiny, chunks);
        var result = SelectiveRecomputer.Run(tiny.Model, cache, chunks, Enumerable.Range(ChunkA.Length, ChunkB.Length).ToArray());

        Assert.Equal(1, result.ForwardCalls);
        Assert.Equal(ChunkB.Length, result.TokensRecomputed);
        Assert.True(MaxAbsDiff(result.LastPositionLogits!, referenceLast) <= 1e-4f);
        float err = MaxAbsDiff(SuffixRows(tiny, cache, n), reference);
        Assert.True(err <= 1e-4f, $"recomputing the stale chunk differs from full prefill by {err}");
    }

    [Fact]
    public void Recompute_TwoRuns_EqualsOnePassFixUp()
    {
        // [A; B; C; D] with B ∪ D selected (two runs). One-pass layer-wise CacheBlend gives B fresh KV attending to A, then D
        // fresh KV attending to A, fresh B and stale C. The same cache is built independently by encoding [A;B] as ONE chunk
        // (exact) and recomputing D alone on [AB; C; D]. Processing D before B (descending) differs.
        using var tiny = TinyModel.Load(_scratch);
        var chunks = Encode(tiny, ChunkA, ChunkB, ChunkC, ChunkD);
        int n = chunks.Sum(c => c.TokenIds.Length);
        int bStart = ChunkA.Length, dStart = ChunkA.Length + ChunkB.Length + ChunkC.Length;
        int[] selected = [.. Enumerable.Range(bStart, ChunkB.Length), .. Enumerable.Range(dStart, ChunkD.Length)];
        int[] dOnly = Enumerable.Range(dStart, ChunkD.Length).ToArray();

        var expectedChunks = new List<ChunkKvEntry>
        {
            new ChunkEncoder(Fingerprint).Encode(tiny.Model, Concat(ChunkA, ChunkB)), chunks[2], chunks[3],
        };
        using var expected = Assembled(tiny, expectedChunks);
        SelectiveRecomputer.Run(tiny.Model, expected, expectedChunks, dOnly);

        using var cache = Assembled(tiny, chunks);
        var result = SelectiveRecomputer.Run(tiny.Model, cache, chunks, selected);
        Assert.Equal(2, result.ForwardCalls);
        Assert.Equal(selected.Length, result.TokensRecomputed);

        // Every layer's K/V must match the independently built cache, not only the logits (compared before any suffix is
        // forwarded, which would extend both caches).
        for (int l = 0; l < tiny.Config.NumLayers; l++)
        {
            Assert.True(MaxAbsDiff(cache.KeysSpan(l), expected.KeysSpan(l)) <= 1e-4f, $"layer {l} keys differ");
            Assert.True(MaxAbsDiff(cache.ValuesSpan(l), expected.ValuesSpan(l)) <= 1e-4f, $"layer {l} values differ");
        }
        float[] expectedRows = SuffixRows(tiny, expected, n);

        // Fixture guard: with B left stale (D only) the result must be visibly different.
        using (var staleB = Assembled(tiny, chunks))
        {
            SelectiveRecomputer.Run(tiny.Model, staleB, chunks, dOnly);
            Assert.True(MaxAbsDiff(SuffixRows(tiny, staleB, n), expectedRows) > 1e-2f, "fixture cannot discriminate stale vs fresh B");
        }

        float err = MaxAbsDiff(SuffixRows(tiny, cache, n), expectedRows);
        Assert.True(err <= 1e-4f, $"two-run recompute differs from the one-pass fix-up by {err}");
    }

    [Fact]
    public void Recompute_MidCacheRun_LeavesOtherRowsAndLengthIntact()
    {
        // A run in the middle of the cache must overwrite only its own rows: CurrentLength unchanged, every row before and
        // after the run bit-identical (the later stale rows are what positions after the run attend to).
        using var tiny = TinyModel.Load(_scratch);
        var chunks = Encode(tiny, ChunkA, ChunkB, ChunkC, ChunkD);
        int n = chunks.Sum(c => c.TokenIds.Length);
        int bStart = ChunkA.Length, bEnd = ChunkA.Length + ChunkB.Length;

        using var cache = Assembled(tiny, chunks);
        float[][] keysBefore = Snapshot(cache, keys: true);
        float[][] valuesBefore = Snapshot(cache, keys: false);

        var result = SelectiveRecomputer.Run(tiny.Model, cache, chunks, Enumerable.Range(bStart, ChunkB.Length).ToArray());
        Assert.Equal(1, result.ForwardCalls);
        Assert.Null(result.LastPositionLogits); // position n-1 was not recomputed
        Assert.Equal(n, cache.CurrentLength);

        int stride = tiny.KvStride;
        bool runChanged = false;
        for (int l = 0; l < tiny.Config.NumLayers; l++)
        {
            float[] k = cache.KeysSpan(l).ToArray(), v = cache.ValuesSpan(l).ToArray();
            Assert.Equal(keysBefore[l][..(bStart * stride)], k[..(bStart * stride)]);
            Assert.Equal(valuesBefore[l][..(bStart * stride)], v[..(bStart * stride)]);
            Assert.Equal(keysBefore[l][(bEnd * stride)..], k[(bEnd * stride)..]);
            Assert.Equal(valuesBefore[l][(bEnd * stride)..], v[(bEnd * stride)..]);
            if (l > 0) runChanged |= MaxAbsDiff(keysBefore[l][(bStart * stride)..(bEnd * stride)], k[(bStart * stride)..(bEnd * stride)]) > 1e-3f;
        }
        Assert.True(runChanged, "the selected rows were not refreshed");
    }

    [Fact]
    public void Recompute_NonContiguousSelection_ErrorBetweenNoneAndAll()
    {
        using var tiny = TinyModel.Load(_scratch);
        var chunks = Encode(tiny, ChunkA, ChunkB, ChunkC, ChunkD);
        int[] context = Concat(ChunkA, ChunkB, ChunkC, ChunkD);
        int n = context.Length;
        float[] reference = FullPrefillSuffixRows(tiny, context, out _);
        int vocab = tiny.Config.VocabSize;

        // Scattered: first token of each stale chunk plus a few singles -> several runs.
        int[] selected = [6, 7, 10, 13, 15, 18, 21];

        double klNone = Metrics.MeanKl(reference, NaiveSuffixRows(tiny, chunks, n), vocab);
        using var cache = Assembled(tiny, chunks);
        var result = SelectiveRecomputer.Run(tiny.Model, cache, chunks, selected);
        double klSome = Metrics.MeanKl(reference, SuffixRows(tiny, cache, n), vocab);
        using var all = Assembled(tiny, chunks);
        SelectiveRecomputer.Run(tiny.Model, all, chunks, Enumerable.Range(0, n).ToArray());
        double klAll = Metrics.MeanKl(reference, SuffixRows(tiny, all, n), vocab);
        _output.WriteLine($"KL none = {klNone:E3}, scattered = {klSome:E3}, all = {klAll:E3}; runs = {result.ForwardCalls}");

        Assert.Equal(6, result.ForwardCalls); // runs: [6,7] [10] [13] [15] [18] [21]
        Assert.Equal(selected.Length, result.TokensRecomputed);
        Assert.NotNull(result.LastPositionLogits); // 21 = n-1
        Assert.True(klAll < klSome && klSome < klNone, $"expected all < scattered < none, got {klAll:E3}, {klSome:E3}, {klNone:E3}");
    }

    [Theory]
    [InlineData(new[] { 3, 2 })]     // not ascending
    [InlineData(new[] { 4, 4 })]     // duplicate
    [InlineData(new[] { -1 })]       // below range
    [InlineData(new[] { 0, 22 })]    // past the context (L = 22)
    public void Recompute_InvalidSelection_ThrowsAndLeavesCacheUnchanged(int[] selected)
    {
        using var tiny = TinyModel.Load(_scratch);
        var chunks = Encode(tiny, ChunkA, ChunkB, ChunkC, ChunkD);
        Assert.Equal(22, chunks.Sum(c => c.TokenIds.Length));
        using var cache = Assembled(tiny, chunks);
        float[][] keysBefore = Snapshot(cache, keys: true);
        float[][] valuesBefore = Snapshot(cache, keys: false);

        Assert.Throws<ArgumentException>(() => SelectiveRecomputer.Run(tiny.Model, cache, chunks, selected));
        AssertCacheEqual(keysBefore, valuesBefore, cache);
    }

    [Fact]
    public void Recompute_CacheShorterThanContext_Throws()
    {
        using var tiny = TinyModel.Load(_scratch);
        var chunks = Encode(tiny, ChunkA, ChunkB);
        using var cache = Assembled(tiny, chunks.Take(1).ToList()); // only A assembled
        Assert.Throws<ArgumentException>(() => SelectiveRecomputer.Run(tiny.Model, cache, chunks, [ChunkA.Length]));
        Assert.Equal(ChunkA.Length, cache.CurrentLength);
    }

    // ---- helpers ----

    private static int[] Concat(params int[][] parts) => parts.SelectMany(p => p).ToArray();

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

    /// <summary>Forwards <see cref="Suffix"/> at <c>n..</c> on <paramref name="cache"/>; returns all suffix logit rows. Mutates the cache.</summary>
    private static unsafe float[] SuffixRows(TinyModel tiny, IKvCache cache, int n)
    {
        using ITensor logits = tiny.Model.Forward(Suffix, Enumerable.Range(n, Suffix.Length).ToArray(), -1, cache);
        long count = (long)logits.Shape[0] * logits.Shape[1];
        Assert.Equal(Suffix.Length, logits.Shape[0]);
        return new ReadOnlySpan<float>((float*)logits.DataPointer, (int)count).ToArray();
    }

    private static float[] NaiveSuffixRows(TinyModel tiny, IReadOnlyList<ChunkKvEntry> chunks, int n)
    {
        using var cache = Assembled(tiny, chunks);
        return SuffixRows(tiny, cache, n);
    }

    /// <summary>Full prefill of <paramref name="context"/> then the suffix; also returns the last context row's logits.</summary>
    private static unsafe float[] FullPrefillSuffixRows(TinyModel tiny, int[] context, out float[] lastContextLogits)
    {
        using var cache = NewCache(tiny);
        using (ITensor logits = tiny.Model.Forward(context, Enumerable.Range(0, context.Length).ToArray(), -1, cache, lastTokenLogitsOnly: true))
        {
            int vocab = logits.Shape[1];
            lastContextLogits = new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(logits.Shape[0] - 1) * vocab, vocab).ToArray();
        }
        return SuffixRows(tiny, cache, context.Length);
    }

    private static float[][] Snapshot(SimpleKvCache cache, bool keys)
        => Enumerable.Range(0, cache.NumLayers).Select(l => (keys ? cache.KeysSpan(l) : cache.ValuesSpan(l)).ToArray()).ToArray();

    private static void AssertCacheEqual(float[][] keys, float[][] values, SimpleKvCache cache)
    {
        for (int l = 0; l < keys.Length; l++)
        {
            Assert.Equal(keys[l], cache.KeysSpan(l).ToArray());
            Assert.Equal(values[l], cache.ValuesSpan(l).ToArray());
        }
    }

    private static float MaxAbsDiff(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        Assert.Equal(a.Length, b.Length);
        float m = 0;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
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

        public int KvStride => Config.NumKvHeads * Config.HeadDim;

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
