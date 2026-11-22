using System.Reflection;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using Xunit;
using static DotLLM.Tests.Unit.Engine.GenerationSessionFixture;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// The #1919 backend-parity assertions, written once and run on every backend by a derived class that supplies
/// <see cref="Open"/>: CPU always, Vulkan when a device is present, CUDA only where a CUDA GPU is (written, not run, on the
/// Strix Halo dev box). Families: attention-only Llama, GDN-hybrid dense, GDN-hybrid MoE.
/// </summary>
public abstract partial class GenerationSessionBackendSuite
{
    internal abstract GenerationSessionFixture Open(string family, int? headDim = null);

    private static float[] Logits(GenerationSession s)
    {
        var buf = new float[s.VocabSize];
        s.CopyNextLogits(buf);
        return buf;
    }

    private static void AssertLogitsEqual(GenerationSessionFixture fx, float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= fx.LogitTolerance + fx.LogitRelativeTolerance * Math.Abs(expected[i]), $"logit {i}: {expected[i]} vs {actual[i]}");
    }

    private static async Task<(float[] Logits, int[] Next)> Fresh(GenerationSessionFixture oracle, IEnumerable<int> tokens)
    {
        await using var s = oracle.NewGenerator().CreateSession(Greedy(), Tag());
        await s.PrefillAsync(tokens.ToArray());
        var logits = Logits(s);
        var r = await s.GenerateUntilBoundaryAsync(new StopAfter(5));
        return (logits, [.. r.TokenIds]);
    }

    private static async Task<int[]> Build(GenerationSession s, GenerationSessionFixture fx)
    {
        await s.PrefillAsync(fx.Prompt);
        var r = await s.GenerateUntilBoundaryAsync(new StopAfter(4));
        await s.AppendObservationTokensAsync(fx.Observation);
        return [.. fx.Prompt, .. r.TokenIds, .. fx.Observation];
    }

    /// <summary>[prefill A, generate, append B] leaves the same distribution and continuation as one prefill of A+g+B; the KV length tracks the position.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task MultiSegment_MatchesSinglePass(string family)
    {
        using var fx = Open(family);
        using var oracle = Open(family);   // recurrent state is model-owned: the oracle needs its own model
        int[] g1, g2, g2Single;
        float[] multiLogits;
        int multiArgMax;
        await using (var multi = fx.NewGenerator().CreateSession(Greedy(), Tag()))
        {
            await multi.PrefillAsync(fx.Prompt);
            Assert.Equal(fx.Prompt.Length, multi.KvLength);
            var r1 = await multi.GenerateUntilBoundaryAsync(new StopAfter(3));
            g1 = [.. r1.TokenIds];
            Assert.Equal(3, g1.Length);
            Assert.Equal(multi.CurrentPosition, multi.KvLength);
            await multi.AppendObservationTokensAsync(fx.Observation);
            Assert.Equal(multi.CurrentPosition, multi.KvLength);
            multiLogits = Logits(multi);
            multiArgMax = multi.NextTokenArgMax();
            g2 = [.. (await multi.GenerateUntilBoundaryAsync(new StopAfter(4))).TokenIds];
        }

        await using (var single = oracle.NewGenerator().CreateSession(Greedy(), Tag()))
        {
            await single.PrefillAsync(fx.Prompt.Concat(g1).Concat(fx.Observation).ToArray());
            AssertLogitsEqual(fx, multiLogits, Logits(single));
            Assert.Equal(multiArgMax, single.NextTokenArgMax());
            g2Single = [.. (await single.GenerateUntilBoundaryAsync(new StopAfter(4))).TokenIds];
        }
        Assert.Equal(g2Single, g2);
    }

    /// <summary>After RollbackTo(k) the logits and the continuation equal a fresh session over tokens[0..k): nothing at or past k survives in KV or recurrent state.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task RollbackToABoundary_EqualsFreshPrefillOfThePrefix(string family)
    {
        using var fx = Open(family);
        using var oracle = Open(family);
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
            AssertLogitsEqual(fx, freshLogits, Logits(s));
            var r = await s.GenerateUntilBoundaryAsync(new StopAfter(5));
            Assert.Equal(freshNext, r.TokenIds);
            s.RollbackTo(k);
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task AfterRollback_AppendingOtherTokens_MatchesFresh(string family)
    {
        using var fx = Open(family);
        using var oracle = Open(family);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await Build(s, fx);
        s.RollbackTo(fx.Prompt.Length);
        int[] other = [7, 5, 6, 4];
        await s.AppendObservationTokensAsync(other);
        var (freshLogits, freshNext) = await Fresh(oracle, fx.Prompt.Concat(other));
        AssertLogitsEqual(fx, freshLogits, Logits(s));
        Assert.Equal(freshNext, (await s.GenerateUntilBoundaryAsync(new StopAfter(5))).TokenIds);
    }

    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task RollbackToZero_ResetsEverything(string family)
    {
        using var fx = Open(family);
        using var oracle = Open(family);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await Build(s, fx);
        s.RollbackTo(0);
        Assert.Equal(0, s.CurrentPosition);
        Assert.Equal(0, s.KvLength);
        int[] other = [4, 6, 3, 5, 7];
        await s.PrefillAsync(other);
        var (freshLogits, freshNext) = await Fresh(oracle, other);
        AssertLogitsEqual(fx, freshLogits, Logits(s));
        Assert.Equal(freshNext, (await s.GenerateUntilBoundaryAsync(new StopAfter(5))).TokenIds);
    }

    /// <summary>An attention-only model rolls back to any position (not just a pause) by re-forwarding one token.</summary>
    [SkippableFact]
    public async Task AttentionModel_RollsBackToAnyPosition()
    {
        using var fx = Open(Llama);
        using var oracle = Open(Llama);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        int[] all = await Build(s, fx);
        int k = fx.Prompt.Length + 2;   // mid-segment: never a pause
        Assert.True(s.CanRollbackTo(k));
        s.RollbackTo(k);
        Assert.Equal(k, s.KvLength);
        var (freshLogits, freshNext) = await Fresh(oracle, all.Take(k));
        AssertLogitsEqual(fx, freshLogits, Logits(s));
        Assert.Equal(freshNext, (await s.GenerateUntilBoundaryAsync(new StopAfter(5))).TokenIds);
    }

    /// <summary>
    /// Path-matched exactness of an attention rollback: RollbackTo(k) re-forwards token k-1 as one decode row over rows [0, k-1) kept from
    /// the earlier forwards, which is exactly what a fresh session does for [prefill k-1 tokens, append token k-1]. Both go through the
    /// same kernels, so they agree to 5e-4 on every backend even where prefill-vs-decode noise (see <c>LogitTolerance</c>) is larger.
    /// </summary>
    [SkippableFact]
    public async Task AttentionModel_Rollback_MatchesAPathMatchedOracle()
    {
        using var fx = Open(Llama);
        using var oracle = Open(Llama);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        int[] all = await Build(s, fx);
        foreach (int k in new[] { fx.Prompt.Length + 2, fx.Prompt.Length })
        {
            s.RollbackTo(k);
            await using var o = oracle.NewGenerator().CreateSession(Greedy(), Tag());
            await o.PrefillAsync(all.Take(k - 1).ToArray());
            await o.AppendObservationTokensAsync(new[] { all[k - 1] });
            var a = Logits(s);
            var b = Logits(o);
            for (int i = 0; i < a.Length; i++)
                Assert.True(Math.Abs(a[i] - b[i]) <= 5e-4f, $"k={k} logit {i}: rolled back {a[i]} vs path-matched oracle {b[i]}");
        }
    }

    /// <summary>A recurrent model refuses a position that is not a recorded pause, changing nothing.</summary>
    [SkippableTheory]
    [InlineData(HybridDense)]
    [InlineData(MoeHybrid)]
    public async Task Recurrent_RollbackBetweenBoundaries_IsRefused(string family)
    {
        using var fx = Open(family);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await Build(s, fx);
        int between = fx.Prompt.Length + 2;
        Assert.False(s.CanRollbackTo(between));
        Assert.Throws<NotSupportedException>(() => s.RollbackTo(between));
        Assert.Equal(s.CurrentPosition, s.KvLength);
    }

    /// <summary>The sampler's history is session-wide: two segments of 4 sample what one segment of 8 does under a repetition penalty.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task SamplerHistory_IsSessionWide(string family)
    {
        using var fx = Open(family);
        var opts = Greedy() with { RepetitionPenalty = 50f };
        int[] split, whole;
        await using (var s = fx.NewGenerator().CreateSession(opts, Tag()))
        {
            await s.PrefillAsync(fx.Prompt);
            var a = await s.GenerateUntilBoundaryAsync(new StopAfter(12));
            var b = await s.GenerateUntilBoundaryAsync(new StopAfter(12));
            split = [.. a.TokenIds, .. b.TokenIds];
        }
        await using (var s = fx.NewGenerator().CreateSession(opts, Tag()))
        {
            await s.PrefillAsync(fx.Prompt);
            whole = [.. (await s.GenerateUntilBoundaryAsync(new StopAfter(24))).TokenIds];
        }
        Assert.Equal(whole, split);
    }

    /// <summary>Provenance drift refuses every operation on this backend's session too, and restoring the identity makes it usable again.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task Provenance_DriftRefusesEveryOperation(string family)
    {
        using var fx = Open(family);
        var live = Tag();
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag(), () => live);
        await s.PrefillAsync(fx.Prompt);
        live = live with { ActiveModifiersDigest = "mods-1" };
        Assert.Throws<InvalidOperationException>(() => s.NextTokenArgMax());
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await s.AppendObservationTokensAsync(fx.Observation));
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await s.GenerateUntilBoundaryAsync(new StopAfter(2)));
        Assert.Throws<InvalidOperationException>(() => s.RollbackTo(0));
        live = Tag();
        s.NextTokenArgMax();   // identity restored: usable again, nothing was disturbed
    }

    /// <summary>
    /// #1919 T6: disposal releases the owned KV-cache and every boundary checkpoint, lets a new session open on the same model, and
    /// the new session starts from clean recurrent state (it reproduces the first session's tokens).
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task Dispose_ReleasesKvAndCheckpoints_AndASecondSessionStartsClean(string family)
    {
        using var fx = Open(family);
        var gen = fx.NewGenerator();
        var s = gen.CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);
        int[] first = [.. (await s.GenerateUntilBoundaryAsync(new StopAfter(5))).TokenIds];
        IKvCache kv = s.KvForTest;
        IReadOnlyList<object?> checkpoints = s.CheckpointsForTest;
        if (fx.IsRecurrent) Assert.NotEmpty(checkpoints);
        await s.DisposeAsync();
        await s.DisposeAsync();   // idempotent

        Assert.NotEqual(false, IsDisposed(kv));   // true, or null for a length-only handle that owns nothing
        foreach (object? cp in checkpoints)
            Assert.Throws<ObjectDisposedException>(() => fx.Model.RestoreRecurrentState(cp));
        Assert.Throws<ObjectDisposedException>(() => s.NextTokenArgMax());

        // The generator is usable again: a Generate call is no longer blocked by an open session.
        if (family == Llama) Assert.NotNull(gen.Generate("ab", Greedy(2)));

        await using var s2 = gen.CreateSession(Greedy(), Tag());
        await s2.PrefillAsync(fx.Prompt);
        Assert.Equal(first, (await s2.GenerateUntilBoundaryAsync(new StopAfter(5))).TokenIds);
    }

    /// <summary>true/false from the cache's private disposed flag; null when the cache has none (a length-only handle that owns no storage).</summary>
    internal static bool? IsDisposed(IKvCache kv)
    {
        var f = kv.GetType().GetField("_disposed", BindingFlags.Instance | BindingFlags.NonPublic);
        return f is null ? null : (bool)f.GetValue(kv)!;
    }
}

/// <summary>CPU run of <see cref="GenerationSessionBackendSuite"/>.</summary>
public sealed class GenerationSessionCpuSuiteTests : GenerationSessionBackendSuite
{
    internal override GenerationSessionFixture Open(string family, int? headDim = null)
        => Create(family, headDim, SessionBackend.Cpu);
}
