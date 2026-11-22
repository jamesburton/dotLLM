using DotLLM.Core.Configuration;
using DotLLM.Core.Sampling;
using DotLLM.Engine;
using Xunit;
using static DotLLM.Tests.Unit.Engine.GenerationSessionFixture;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>#1919 T2: session core (prefill, observation append, segment generation, probes) on CPU, three families.</summary>
public sealed class GenerationSessionCoreTests
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
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= Tol, $"logit {i}: {expected[i]} vs {actual[i]}");
    }

    /// <summary>
    /// THE parity gate: [prefill A, generate g1, append B] leaves the same next-token distribution as one prefill of A+g1+B, and
    /// the continuation then generates identical tokens.
    /// </summary>
    [Theory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task MultiSegment_MatchesSinglePass(string family)
    {
        using var fx = Create(family);
        var gen = fx.NewGenerator();

        int[] g1, g2, g2Single;
        float[] multiLogits;
        int multiArgMax;
        await using (var multi = gen.CreateSession(Greedy(), Tag()))
        {
            await multi.PrefillAsync(fx.Prompt);
            var r1 = await multi.GenerateUntilBoundaryAsync(new StopAfter(3));
            g1 = [.. r1.TokenIds];
            Assert.Equal(3, g1.Length);
            Assert.Equal(fx.Prompt.Length, r1.StartPosition);
            Assert.Equal(fx.Prompt.Length + 3, r1.EndPosition);
            Assert.Equal(r1.EndPosition, multi.CurrentPosition);
            await multi.AppendObservationTokensAsync(fx.Observation);
            multiLogits = Logits(multi);
            multiArgMax = multi.NextTokenArgMax();
            var r2 = await multi.GenerateUntilBoundaryAsync(new StopAfter(4));
            g2 = [.. r2.TokenIds];
        }

        await using (var single = gen.CreateSession(Greedy(), Tag()))
        {
            await single.PrefillAsync(fx.Prompt.Concat(g1).Concat(fx.Observation).ToArray());
            Assert.Equal(multi_position(fx, g1), single.CurrentPosition);
            AssertLogitsEqual(multiLogits, Logits(single));
            Assert.Equal(multiArgMax, single.NextTokenArgMax());
            var r2 = await single.GenerateUntilBoundaryAsync(new StopAfter(4));
            g2Single = [.. r2.TokenIds];
        }

        Assert.Equal(g2Single, g2);
    }

    private static int multi_position(GenerationSessionFixture fx, int[] g1) => fx.Prompt.Length + g1.Length + fx.Observation.Length;

    /// <summary>Pausing and resuming never re-forwards the prefix: the model sees each token exactly once (+ the boundary token).</summary>
    [Theory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task Resume_ForwardsEveryTokenExactlyOnce(string family)
    {
        using var fx = Create(family);
        var gen = fx.NewGenerator();
        await using var s = gen.CreateSession(Greedy(), Tag());

        fx.Counter.Reset();
        await s.PrefillAsync(fx.Prompt);
        Assert.Equal(fx.Prompt.Length, fx.Counter.Tokens);

        await s.GenerateUntilBoundaryAsync(new StopAfter(3));
        // 3 sampled tokens, each committed AND forwarded (the boundary one too, so logits are valid at the pause).
        Assert.Equal(fx.Prompt.Length + 3, fx.Counter.Tokens);

        await s.AppendObservationTokensAsync(fx.Observation);
        Assert.Equal(fx.Prompt.Length + 3 + fx.Observation.Length, fx.Counter.Tokens);

        await s.GenerateUntilBoundaryAsync(new StopAfter(4));
        Assert.Equal(fx.Prompt.Length + 3 + fx.Observation.Length + 4, fx.Counter.Tokens);
        Assert.Equal(fx.Counter.Tokens, s.CurrentPosition);
    }

    [Theory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task NextTokenArgMax_IsTheArgMaxOfCopiedLogits_AndFailsBeforePrefill(string family)
    {
        using var fx = Create(family);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        Assert.Throws<InvalidOperationException>(() => s.NextTokenArgMax());
        Assert.Throws<InvalidOperationException>(() => s.CopyNextLogits(new float[s.VocabSize]));

        await s.PrefillAsync(fx.Prompt);
        var logits = Logits(s);
        int best = 0;
        for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
        Assert.Equal(best, s.NextTokenArgMax());
        Assert.Throws<ArgumentException>(() => s.CopyNextLogits(new float[s.VocabSize - 1]));
    }

    /// <summary>A <see cref="StopResult.Stop"/> token is not committed: position and logits stay at the pre-token state.</summary>
    [Theory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task StopToken_IsNotCommitted_AndIsTheNextArgMax(string family)
    {
        using var fx = Create(family);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);
        var r = await s.GenerateUntilBoundaryAsync(new StopAfter(3, StopResult.Stop));

        Assert.Equal(FinishReason.Stop, r.FinishReason);
        Assert.Equal(2, r.TokenIds.Count);
        Assert.Equal(fx.Prompt.Length + 2, s.CurrentPosition);
        Assert.Equal(s.CurrentPosition, r.EndPosition);

        // Greedy: the refused third token is exactly the next argmax.
        int refused = s.NextTokenArgMax();
        await using var ref2 = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await ref2.PrefillAsync(fx.Prompt);
        var three = await ref2.GenerateUntilBoundaryAsync(new StopAfter(3));
        Assert.Equal(three.TokenIds[2], refused);
        Assert.Equal(three.TokenIds[0], r.TokenIds[0]);
        Assert.Equal(three.TokenIds[1], r.TokenIds[1]);
    }

    [Theory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task MaxTokens_BoundsEachSegment_AndReportsLength(string family)
    {
        using var fx = Create(family);
        await using var s = fx.NewGenerator().CreateSession(Greedy(maxTokens: 5), Tag());
        await s.PrefillAsync(fx.Prompt);
        var a = await s.GenerateUntilBoundaryAsync(new StopAfter(1000));
        Assert.Equal(5, a.TokenIds.Count);
        Assert.Equal(FinishReason.Length, a.FinishReason);
        var b = await s.GenerateUntilBoundaryAsync(new StopAfter(1000));
        Assert.Equal(5, b.TokenIds.Count);
        Assert.Equal(fx.Prompt.Length + 10, s.CurrentPosition);
    }

    [Fact]
    public async Task Text_IsTheDecodedCommittedTokens()
    {
        using var fx = Create(Llama);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);
        var r = await s.GenerateUntilBoundaryAsync(new StopAfter(6));
        Assert.Equal(fx.Tokenizer.Decode(r.TokenIds.ToArray(), stripBosSpace: false), r.Text);
    }

    [Theory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task KvCapacity_IsEnforced(string family)
    {
        using var fx = Create(family);
        int cap = fx.Prompt.Length + 4;
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag(), maxContextTokens: cap);
        await s.PrefillAsync(fx.Prompt);
        var r = await s.GenerateUntilBoundaryAsync(new StopAfter(100));
        Assert.Equal(FinishReason.Length, r.FinishReason);
        Assert.Equal(cap, s.CurrentPosition);
        Assert.Equal(4, r.TokenIds.Count);
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await s.AppendObservationTokensAsync(new int[] { 5 }));
        Assert.Equal(cap, s.CurrentPosition);   // refused before any forward
    }

    [Theory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task CancelledAppend_LeavesSessionUsableAndUnchanged(string family)
    {
        using var fx = Create(family);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);
        var before = Logits(s);
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await s.AppendObservationTokensAsync(fx.Observation, cts.Token));
        Assert.Equal(fx.Prompt.Length, s.CurrentPosition);
        AssertLogitsEqual(before, Logits(s));

        await s.AppendObservationTokensAsync(fx.Observation);
        Assert.Equal(fx.Prompt.Length + fx.Observation.Length, s.CurrentPosition);
    }

    /// <summary>A forward that throws mid-append (after the KV advanced) is undone: position, KV and logits are as before.</summary>
    [Fact]
    public async Task FailedAppendForward_IsUndone_ForAnAttentionModel()
    {
        using var fx = Create(Llama);
        var gen = fx.NewGenerator();
        await using var s = gen.CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);
        var before = Logits(s);

        fx.Counter.FailNextForward();
        await Assert.ThrowsAsync<IOException>(async () => await s.AppendObservationTokensAsync(fx.Observation));
        Assert.Equal(fx.Prompt.Length, s.CurrentPosition);
        Assert.Equal(s.CurrentPosition, s.KvLength);   // the abandoned append did not leave KV rows behind
        AssertLogitsEqual(before, Logits(s));

        await s.AppendObservationTokensAsync(fx.Observation);
        await using var fresh = gen.CreateSession(Greedy(), Tag());
        await fresh.PrefillAsync(fx.Prompt.Concat(fx.Observation).ToArray());
        AssertLogitsEqual(Logits(fresh), Logits(s));
    }

    [Fact]
    public void ResponseFormat_IsNotSupportedYet()
    {
        using var fx = Create(Llama);
        var opts = new InferenceOptions { ResponseFormat = new ResponseFormat.JsonObject() };
        Assert.Throws<NotSupportedException>(() => fx.NewGenerator().CreateSession(opts, Tag()));
    }

    [Fact]
    public async Task RecurrentModel_AllowsOneLiveSession_AndBlocksGenerateWhileOpen()
    {
        using var fx = Create(HybridDense);
        var gen = fx.NewGenerator();
        await using var s = gen.CreateSession(Greedy(), Tag());
        Assert.Throws<InvalidOperationException>(() => gen.CreateSession(Greedy(), Tag()));
        Assert.Throws<InvalidOperationException>(() => gen.Generate("x", Greedy(2)));
    }

    [Fact]
    public async Task ConcurrentOperation_IsRejected()
    {
        using var fx = Create(Llama);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);
        int hits = 0;
        var stop = new ReentrantStop(() =>
        {
            hits++;
            Assert.Throws<InvalidOperationException>(() => s.AppendObservationTokensAsync(fx.Observation).AsTask().GetAwaiter().GetResult());
        });
        await s.GenerateUntilBoundaryAsync(stop);
        Assert.True(hits > 0);
    }

    private sealed class ReentrantStop(Action onCall) : IStopCondition
    {
        public StopResult ShouldStop(int tokenId, IReadOnlyList<int> generatedTokens, string decodedText)
        {
            onCall();
            return StopResult.StopInclude;
        }
    }
}
