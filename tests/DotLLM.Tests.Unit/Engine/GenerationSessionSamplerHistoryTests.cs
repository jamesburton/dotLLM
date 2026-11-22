using DotLLM.Core.Configuration;
using DotLLM.Engine;
using Xunit;
using static DotLLM.Tests.Unit.Engine.GenerationSessionFixture;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>#1919 T5: the sampler's previous-token history is session-wide (every committed token), not per segment.</summary>
public sealed class GenerationSessionSamplerHistoryTests
{
    private static InferenceOptions Penalised(int maxTokens = 64) => Greedy(maxTokens) with { RepetitionPenalty = 50f };

    /// <summary>Two segments of 4 sample exactly what one segment of 8 does: the penalty sees the first segment's tokens.</summary>
    [Theory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task TwoSegments_EqualOneLongSegment_UnderRepetitionPenalty(string family)
    {
        using var fx = Create(family);
        var gen = fx.NewGenerator();

        int[] split;
        await using (var s = gen.CreateSession(Penalised(), Tag()))
        {
            await s.PrefillAsync(fx.Prompt);
            var a = await s.GenerateUntilBoundaryAsync(new StopAfter(4));
            var b = await s.GenerateUntilBoundaryAsync(new StopAfter(4));
            split = [.. a.TokenIds, .. b.TokenIds];
        }

        int[] whole;
        await using (var s = gen.CreateSession(Penalised(), Tag()))
        {
            await s.PrefillAsync(fx.Prompt);
            whole = [.. (await s.GenerateUntilBoundaryAsync(new StopAfter(8))).TokenIds];
        }
        Assert.Equal(whole, split);

        int[] unpenalised;
        await using (var s = gen.CreateSession(Greedy(), Tag()))
        {
            await s.PrefillAsync(fx.Prompt);
            unpenalised = [.. (await s.GenerateUntilBoundaryAsync(new StopAfter(8))).TokenIds];
        }
        Assert.NotEqual(unpenalised, whole);   // the penalty must be observable, else the equality above proves nothing
    }

    /// <summary>
    /// Prompt, generated and observation tokens are all history: [prefill A, generate, append B, generate] samples what
    /// [prefill A+g+B, generate] does.
    /// </summary>
    [Theory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task ObservationTokensAreHistory(string family)
    {
        using var fx = Create(family);
        var gen = fx.NewGenerator();

        int[] g1, g2;
        await using (var s = gen.CreateSession(Penalised(), Tag()))
        {
            await s.PrefillAsync(fx.Prompt);
            g1 = [.. (await s.GenerateUntilBoundaryAsync(new StopAfter(3))).TokenIds];
            await s.AppendObservationTokensAsync(fx.Observation);
            g2 = [.. (await s.GenerateUntilBoundaryAsync(new StopAfter(5))).TokenIds];
        }

        int[] g2Single;
        await using (var s = gen.CreateSession(Penalised(), Tag()))
        {
            await s.PrefillAsync(fx.Prompt.Concat(g1).Concat(fx.Observation).ToArray());
            g2Single = [.. (await s.GenerateUntilBoundaryAsync(new StopAfter(5))).TokenIds];
        }
        Assert.Equal(g2Single, g2);
    }
}
