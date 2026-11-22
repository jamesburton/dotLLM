using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using Xunit;
using static DotLLM.Tests.Unit.Engine.GenerationSessionFixture;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>#1919 T3: <see cref="GenerationSession.RollbackTo"/>, recurrent boundary records and KV-cache gating (CPU, three families).</summary>
public sealed class GenerationSessionRollbackTests
{
    private const float Tol = 1e-4f;

    private static float[] Logits(GenerationSession s)
    {
        var buf = new float[s.VocabSize];
        s.CopyNextLogits(buf);
        return buf;
    }

    private static void AssertLogitsEqual(float[] expected, float[] actual)
    {
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= Tol, $"logit {i}: {expected[i]} vs {actual[i]}");
    }

    /// <summary>Fresh session over exactly <paramref name="tokens"/>, then 5 greedy tokens: the oracle for a rolled-back session.</summary>
    private static async Task<(float[] Logits, int[] Next)> Fresh(GenerationSessionFixture oracle, IEnumerable<int> tokens)
    {
        await using var s = oracle.NewGenerator().CreateSession(Greedy(), Tag());
        await s.PrefillAsync(tokens.ToArray());
        var logits = Logits(s);
        var r = await s.GenerateUntilBoundaryAsync(new StopAfter(5));
        return (logits, [.. r.TokenIds]);
    }

    /// <summary>
    /// Builds [prefill A, generate 4, append B] and returns the committed token stream. Positions |A|, |A|+4 and |A|+4+|B| are
    /// the pauses (recorded boundaries on a recurrent model).
    /// </summary>
    private static async Task<int[]> Build(GenerationSession s, GenerationSessionFixture fx)
    {
        await s.PrefillAsync(fx.Prompt);
        var r = await s.GenerateUntilBoundaryAsync(new StopAfter(4));
        await s.AppendObservationTokensAsync(fx.Observation);
        return [.. fx.Prompt, .. r.TokenIds, .. fx.Observation];
    }

    /// <summary>
    /// THE rollback gate: after RollbackTo(k) the next-token logits and the generated continuation equal those of a fresh session
    /// that prefilled tokens[0..k) - nothing at or beyond k survives, in KV or in recurrent state.
    /// </summary>
    [Theory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task RollbackToABoundary_EqualsFreshPrefillOfThePrefix(string family)
    {
        using var fx = Create(family);
        using var oracle = Create(family);   // separate model: recurrent state is model-owned
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        int[] all = await Build(s, fx);
        int a = fx.Prompt.Length;

        foreach (int k in new[] { a + 4, a })
        {
            Assert.True(s.CanRollbackTo(k));
            s.RollbackTo(k);
            Assert.Equal(k, s.CurrentPosition);
            Assert.Equal(k, s.KvLength);
            var (freshLogits, freshNext) = await Fresh(oracle, all.Take(k));
            AssertLogitsEqual(freshLogits, Logits(s));
            var r = await s.GenerateUntilBoundaryAsync(new StopAfter(5));
            Assert.Equal(freshNext, r.TokenIds);
            s.RollbackTo(k);   // back to the boundary for the next iteration; a no-op position-wise
        }
    }

    /// <summary>A rolled-back session continues with different observations exactly like a fresh one.</summary>
    [Theory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task AfterRollback_AppendingOtherTokens_MatchesFresh(string family)
    {
        using var fx = Create(family);
        using var oracle = Create(family);   // separate model: recurrent state is model-owned
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await Build(s, fx);
        int a = fx.Prompt.Length;
        s.RollbackTo(a);
        int[] other = [7, 5, 6, 4];
        await s.AppendObservationTokensAsync(other);
        var (freshLogits, freshNext) = await Fresh(oracle, fx.Prompt.Concat(other));
        AssertLogitsEqual(freshLogits, Logits(s));
        var r = await s.GenerateUntilBoundaryAsync(new StopAfter(5));
        Assert.Equal(freshNext, r.TokenIds);
    }

    [Theory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task RollbackToZero_ResetsEverything(string family)
    {
        using var fx = Create(family);
        using var oracle = Create(family);   // separate model: recurrent state is model-owned
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await Build(s, fx);
        s.RollbackTo(0);
        Assert.Equal(0, s.CurrentPosition);
        Assert.Equal(0, s.KvLength);
        Assert.Throws<InvalidOperationException>(() => s.NextTokenArgMax());

        int[] other = [4, 6, 3, 5, 7];
        await s.PrefillAsync(other);
        var (freshLogits, freshNext) = await Fresh(oracle, other);
        AssertLogitsEqual(freshLogits, Logits(s));
        var r = await s.GenerateUntilBoundaryAsync(new StopAfter(5));
        Assert.Equal(freshNext, r.TokenIds);
    }

    /// <summary>An attention-only model can roll back to any position (it re-forwards one token for the logits, never the prefix).</summary>
    [Fact]
    public async Task AttentionModel_RollsBackToAnyPosition_WithoutReprefill()
    {
        using var fx = Create(Llama);
        using var oracle = Create(Llama);   // separate model: recurrent state is model-owned
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        int[] all = await Build(s, fx);
        foreach (int k in new[] { all.Length - 1, fx.Prompt.Length + 2, fx.Prompt.Length - 3, 1 })
        {
            Assert.True(s.CanRollbackTo(k));
            fx.Counter.Reset();
            s.RollbackTo(k);
            Assert.Equal(1, fx.Counter.Tokens);   // the one token whose logits are needed
            Assert.Equal(k, s.KvLength);
            var (freshLogits, freshNext) = await Fresh(oracle, all.Take(k));
            AssertLogitsEqual(freshLogits, Logits(s));
            var r = await s.GenerateUntilBoundaryAsync(new StopAfter(5));
            Assert.Equal(freshNext, r.TokenIds);
            // back to the full stream for the next k
            s.RollbackTo(0);
            await s.PrefillAsync(all);
        }
    }

    [Fact]
    public async Task Recurrent_RollbackBetweenBoundaries_IsRefused_AndBoundariesAreListed()
    {
        using var fx = Create(HybridDense);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        int[] all = await Build(s, fx);
        int a = fx.Prompt.Length;

        Assert.Equal([a, a + 4, all.Length], s.RollbackBoundaries);
        Assert.False(s.CanRollbackTo(a + 2));
        var ex = Assert.Throws<NotSupportedException>(() => s.RollbackTo(a + 2));
        Assert.Contains("boundar", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(all.Length, s.CurrentPosition);   // refusal changes nothing

        s.RollbackTo(a);
        Assert.Equal([a], s.RollbackBoundaries);       // boundaries beyond the target are dropped
        Assert.False(s.CanRollbackTo(a + 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => s.RollbackTo(a + 4));
    }

    [Theory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task RollbackTo_RejectsOutOfRangePositions(string family)
    {
        using var fx = Create(family);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);
        Assert.Throws<ArgumentOutOfRangeException>(() => s.RollbackTo(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => s.RollbackTo(fx.Prompt.Length + 1));
        Assert.False(s.CanRollbackTo(-1));
        Assert.False(s.CanRollbackTo(fx.Prompt.Length + 1));
        s.RollbackTo(fx.Prompt.Length);   // the current position: a no-op that keeps the logits
        Assert.Equal(fx.Prompt.Length, s.CurrentPosition);
        s.NextTokenArgMax();
    }

    /// <summary>Boundary records are bounded (16): the oldest are evicted, the newest kept.</summary>
    [Fact]
    public async Task Recurrent_BoundaryRecords_AreBounded_OldestEvicted()
    {
        using var fx = Create(HybridDense);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);
        for (int i = 0; i < 20; i++)
            await s.AppendObservationTokensAsync(new[] { 3 + i % 5 });

        Assert.Equal(GenerationSession.MaxRollbackBoundaries, s.RollbackBoundaries.Count);
        Assert.Equal(s.CurrentPosition, s.RollbackBoundaries[^1]);
        Assert.False(s.CanRollbackTo(fx.Prompt.Length));                    // evicted
        Assert.True(s.CanRollbackTo(s.CurrentPosition - 5));                // recent: kept
        s.RollbackTo(s.CurrentPosition - 5);
        Assert.Equal(s.CurrentPosition, s.KvLength);
    }

    /// <summary>A KV-cache that has already dropped the rows a cut needs refuses the rollback loudly, naming the cache type.</summary>
    [Fact]
    public async Task NonTruncatableKvCache_RefusesRollback_NamingTheCacheType()
    {
        using var fx = Create(HybridDense, headDim: 32);   // Q8_0 KV blocks are 32 wide
        var gen = new TextGenerator(fx.Model, fx.Tokenizer, kvCacheFactory: (cfg, size) =>
            new QuantizedKvCache(KvGeometry.FromConfig(cfg, false), size, KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize: 2));
        await using var s = gen.CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);
        int a = fx.Prompt.Length;
        await s.GenerateUntilBoundaryAsync(new StopAfter(3));
        Assert.Contains(a, s.RollbackBoundaries);   // a recorded boundary: only the KV-cache stands in the way

        Assert.False(s.CanRollbackTo(a));
        var ex = Assert.Throws<NotSupportedException>(() => s.RollbackTo(a));
        Assert.Contains(nameof(QuantizedKvCache), ex.Message);
        Assert.Equal(a + 3, s.CurrentPosition);
        s.RollbackTo(0);   // a full reset never needs the cut
        Assert.Equal(0, s.CurrentPosition);
    }

    /// <summary>A failed forward on a recurrent model is undone from the boundary recorded at the start of the operation.</summary>
    [Theory]
    [InlineData(HybridDense)]
    [InlineData(MoeHybrid)]
    public async Task Recurrent_FailedAppend_IsUndoneFromTheBoundary(string family)
    {
        using var fx = Create(family);
        using var oracle = Create(family);   // separate model: recurrent state is model-owned
        var gen = fx.NewGenerator();
        await using var s = gen.CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);
        var before = Logits(s);

        fx.Counter.FailNextForward();
        await Assert.ThrowsAsync<IOException>(async () => await s.AppendObservationTokensAsync(fx.Observation));
        Assert.Equal(fx.Prompt.Length, s.CurrentPosition);
        Assert.Equal(fx.Prompt.Length, s.KvLength);
        AssertLogitsEqual(before, Logits(s));

        await s.AppendObservationTokensAsync(fx.Observation);
        var (freshLogits, _) = await Fresh(oracle, fx.Prompt.Concat(fx.Observation));
        // Recurrent state was restored (not left advanced by the aborted forward): same logits as a clean prefill.
        AssertLogitsEqual(freshLogits, Logits(s));
    }
}
