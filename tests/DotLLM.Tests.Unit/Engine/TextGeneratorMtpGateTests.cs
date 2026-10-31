using System.Globalization;
using System.Text;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// <see cref="TextGenerator"/> + <see cref="MtpAdaptiveGate"/> + <see cref="MtpRoundController"/> end to end on a latency-simulating MTP mock (#900):
/// the generator must (a) keep the output identical to plain decode through a mid-request switch to plain decode, (b) actually stop speculating when
/// the measured rounds lose, (c) keep speculating when they win, and (d) speculate on sampled requests too. Each assertion fails if the path is skipped.
/// </summary>
public sealed class TextGeneratorMtpGateTests : IDisposable
{
    private const int Vocab = 16;

    // Chain over 2..13 so the EOS id (15) is never produced: successor of t.
    private static int Next(int t) => t >= 13 ? 2 : t + 1;

    private static MtpSpeculativeDecoderTests.MockMtpModel NewModel(bool draftRight, int plainMs, int rowMs, int fixedMs)
        => new(Next, t => draftRight ? Next(t) : (Next(t) + 4) % 14 + 2, supportsMtp: true)
        {
            ComputeMemoryBytes = 8L << 30,   // a "big" model: the gate's prior tries MTP first
            OnTrunkForward = rows => Spin(rows == 1 ? plainMs : fixedMs + rowMs * rows),
            OnDraftStep = () => Spin(1),
        };

    // The gate compares millisecond figures measured by the engine, so the engine reads a per-thread VIRTUAL clock (EngineClock) that only the
    // mock's simulated latencies advance. Wall time and scheduler preemption cannot reach the measured figures (#1373).
    private static void Spin(int ms) => t_virtualTicks += ms * System.Diagnostics.Stopwatch.Frequency / 1000;

    [ThreadStatic] private static long t_virtualTicks;
    private readonly IDisposable _clock;

    public TextGeneratorMtpGateTests()
    {
        t_virtualTicks = 0;
        _clock = EngineClock.Override(() => t_virtualTicks);
    }

    public void Dispose() => _clock.Dispose();

    private static InferenceOptions Greedy(int maxTokens) => new() { MaxTokens = maxTokens, Temperature = 0f };

    [Fact]
    public void LosingRounds_FallBackToPlainMidRequest_WithIdenticalOutput_AndDriftOnlyForTheProbeRounds()
    {
        using var plainModel = NewModel(draftRight: false, plainMs: 2, rowMs: 6, fixedMs: 6);
        var plainGen = new TextGenerator(plainModel, new Tok(), mtpEnabled: false);
        var reference = plainGen.Generate("p", Greedy(64));

        // Drafts always miss: every round costs 6 + 6 * (K + 1) + K ms for ONE token against 2 ms plain.
        using var model = NewModel(draftRight: false, plainMs: 2, rowMs: 6, fixedMs: 6);
        var gen = new TextGenerator(model, new Tok(), mtpEnabled: true, mtpAdaptive: true, speculativeCandidates: 3);
        Assert.NotNull(gen.MtpGate);
        gen.MtpGate!.RecordSegment(usedMtp: false, decodeTokens: 40, decodeMs: 40 * 2.0);   // plain reference: 2 ms / token
        var resp = gen.Generate("p", Greedy(64));

        Assert.Equal(reference.Text, resp.Text);                       // identical text through the switch
        Assert.True(resp.Timings.SpeculativeDraftTokens > 0, "the request must have started on MTP (prior = try MTP first)");
        Assert.True(resp.Timings.SpeculativeDraftTokens <= (MtpRoundController.MinRoundsForFallback + 2) * 3,
            $"fall-back must stop the drafting after a handful of rounds, got {resp.Timings.SpeculativeDraftTokens} drafts");
        Assert.True(model.TrunkForwardCount > 10, "the tail must have run as single-token plain forwards");
    }

    [Fact]
    public void ColdStart_LosingConfig_IsCaughtByTheRequestsOwnCalibration_NotAfterAFullRequest()
    {
        // No plain sample exists (first request of a process): before #820 nothing could say "MTP is slower", so a losing configuration ran the whole
        // request on MTP. The request now opens with zero-draft rounds, which are plain steps, and judges its own rounds against them.
        using var plainModel = NewModel(draftRight: false, plainMs: 2, rowMs: 6, fixedMs: 6);
        var reference = new TextGenerator(plainModel, new Tok(), mtpEnabled: false).Generate("p", Greedy(160));

        using var model = NewModel(draftRight: false, plainMs: 2, rowMs: 6, fixedMs: 6);
        var gen = new TextGenerator(model, new Tok(), mtpEnabled: true, mtpAdaptive: true, speculativeCandidates: 3);
        Assert.Equal(0, gen.MtpGate!.PlainMsPerToken);
        var resp = gen.Generate("p", Greedy(160));

        Assert.Equal(reference.Text, resp.Text);   // calibration rounds and the switch commit exactly the plain tokens
        Assert.True(resp.Timings.SpeculativeDraftTokens > 0, "the request must have tried MTP after calibrating");
        Assert.True(resp.Timings.SpeculativeDraftTokens <= (MtpRoundController.MinRoundsForFallback + 2) * 3,
            $"a losing config must drop MTP after a handful of rounds, drafted {resp.Timings.SpeculativeDraftTokens}");
        Assert.True(gen.MtpGate.PlainMsPerToken > 0, "the calibration rounds must hand the gate its first plain sample");
    }

    [Fact]
    public void ColdStart_WinningConfig_KeepsSpeculating_AndBothArmsAreMeasuredAfterOneRequest()
    {
        using var plainModel = NewModel(draftRight: true, plainMs: 20, rowMs: 2, fixedMs: 6);
        var reference = new TextGenerator(plainModel, new Tok(), mtpEnabled: false).Generate("p", Greedy(100));

        using var model = NewModel(draftRight: true, plainMs: 20, rowMs: 2, fixedMs: 6);
        var gen = new TextGenerator(model, new Tok(), mtpEnabled: true, mtpAdaptive: true, speculativeCandidates: 3);
        var resp = gen.Generate("p", Greedy(100));

        Assert.Equal(reference.Text, resp.Text);
        Assert.True(resp.Timings.SpeculativeAcceptedTokens >= 60, $"accepted {resp.Timings.SpeculativeAcceptedTokens}: the win must survive the calibration prefix");
        Assert.True(gen.MtpGate!.PlainMsPerToken > 0 && gen.MtpGate.MtpMsPerToken > 0, "one request must leave a sample for each arm");
        Assert.True(gen.MtpGate.MtpMsPerToken < gen.MtpGate.PlainMsPerToken);
        // With both arms measured the second request goes straight to the winner instead of spending a request exploring plain decode.
        Assert.True(gen.Generate("p", Greedy(100)).Timings.SpeculativeDraftTokens > 0);
    }

    [Fact]
    public void SpeculativeStats_AcceptedNeverExceedsDrafted_AndTheTargetTokenIsReportedSeparately()
    {
        // #1243: the MTP loop used to count every emitted token (accepted drafts AND each round's own target token) as "accepted", so with perfect
        // drafts accepted > drafted. Accepted is now drafts only; the per-round target token is SpeculativeSteps.
        using var model = NewModel(draftRight: true, plainMs: 20, rowMs: 2, fixedMs: 6);
        var gen = new TextGenerator(model, new Tok(), mtpEnabled: true, mtpAdaptive: true, speculativeCandidates: 3);
        gen.MtpGate!.RecordSegment(false, 40, 40 * 20.0);
        var resp = gen.Generate("p", Greedy(48));
        var t = resp.Timings;

        Assert.True(t.SpeculativeDraftTokens > 0 && t.SpeculativeSteps > 0);
        Assert.True(t.SpeculativeAcceptedTokens <= t.SpeculativeDraftTokens, $"accepted {t.SpeculativeAcceptedTokens} > drafted {t.SpeculativeDraftTokens}");
        Assert.Equal(t.SpeculativeDraftTokens, t.SpeculativeAcceptedTokens);   // perfect drafts: every draft is accepted
        // Every emitted token except the prefill one is an accepted draft or one round's target token.
        Assert.InRange(t.SpeculativeAcceptedTokens + t.SpeculativeSteps, resp.GeneratedTokenIds.Length - 1, resp.GeneratedTokenIds.Length);   // a final round may overshoot max_tokens by one
        Assert.Equal(1f, t.SpeculativeAcceptanceRate);
    }

    [Fact]
    public void Calibration_IsSkipped_ForShortRequests_AndWhenAPlainSampleExists()
    {
        using var model = NewModel(draftRight: true, plainMs: 20, rowMs: 2, fixedMs: 6);
        var gen = new TextGenerator(model, new Tok(), mtpEnabled: true, mtpAdaptive: true, speculativeCandidates: 3);
        Assert.False(gen.MtpGate!.NeedsPlainReference(MtpRoundController.MinTokensToCalibrate - 1));
        Assert.True(gen.MtpGate.NeedsPlainReference(MtpRoundController.MinTokensToCalibrate));
        gen.MtpGate.RecordSegment(false, 40, 40 * 20.0);
        Assert.False(gen.MtpGate.NeedsPlainReference(1000));
    }

    [Fact]
    public void WinningRounds_KeepSpeculating_WithIdenticalOutput()
    {
        // Perfect drafts: a K=3 round commits 4 tokens for 6 + 2*4 + 3 = 17 ms against 4 * 20 = 80 ms plain (a wide margin: the suite runs this
        // beside hundreds of CPU-bound tests and the spin clocks are wall time).
        using var plainModel = NewModel(draftRight: true, plainMs: 20, rowMs: 2, fixedMs: 6);
        var reference = new TextGenerator(plainModel, new Tok(), mtpEnabled: false).Generate("p", Greedy(48));

        using var model = NewModel(draftRight: true, plainMs: 20, rowMs: 2, fixedMs: 6);
        var gen = new TextGenerator(model, new Tok(), mtpEnabled: true, mtpAdaptive: true, speculativeCandidates: 3);
        gen.MtpGate!.RecordSegment(false, 40, 40 * 20.0);
        var resp = gen.Generate("p", Greedy(48));

        Assert.Equal(reference.Text, resp.Text);
        Assert.True(resp.Timings.SpeculativeAcceptedTokens >= 30, $"accepted {resp.Timings.SpeculativeAcceptedTokens}");
        Assert.True(model.TrunkForwardCount < 20, $"{model.TrunkForwardCount} trunk forwards: MTP rounds must carry several tokens each");
        Assert.True(gen.MtpGate.MtpMsPerToken > 0 && gen.MtpGate.MtpMsPerToken < gen.MtpGate.PlainMsPerToken);
    }

    [Fact]
    public void SampledRequests_Speculate_UnderTheirOwnGate()
    {
        using var model = NewModel(draftRight: true, plainMs: 8, rowMs: 2, fixedMs: 6);
        var gen = new TextGenerator(model, new Tok(), mtpEnabled: true, mtpAdaptive: true, speculativeCandidates: 3);
        // A sampled request with a sharp distribution: top_k=1 makes the pipeline stochastic-in-form but deterministic in result.
        var resp = gen.Generate("p", new InferenceOptions { MaxTokens = 32, Temperature = 0.8f, TopK = 1, Seed = 7 });

        Assert.True(resp.Timings.SpeculativeDraftTokens > 0, "a sampled request must take the MTP path");
        Assert.True(gen.MtpGate!.MtpMsPerToken == 0, "the greedy gate is not touched by a sampled request");
    }

    [Fact]
    public void Sampled_PlainAndMtp_HaveSeparateReferences()
    {
        using var model = NewModel(draftRight: true, plainMs: 8, rowMs: 2, fixedMs: 6);
        var gen = new TextGenerator(model, new Tok(), mtpEnabled: true, mtpAdaptive: true, speculativeCandidates: 3);
        gen.Generate("p", Greedy(32));
        gen.Generate("p", new InferenceOptions { MaxTokens = 32, Temperature = 0.8f, TopK = 1, Seed = 7 });
        Assert.True(gen.MtpGate!.MtpMsPerToken > 0);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void WinningMtp_IsChosenAgain_AfterTheOtherArmWasExplored(bool sampled)
    {
        // Serve log, UD-IQ4_XS: sampled request 1 ran MTP (24.1 tok/s), request 2 explored plain (22.5), then requests 3 and 4 ran plain even though MTP
        // had measured faster. After both arms are sampled the faster one must be picked every time (until a re-probe).
        using var model = NewModel(draftRight: true, plainMs: 20, rowMs: 2, fixedMs: 6);
        var gen = new TextGenerator(model, new Tok(), mtpEnabled: true, mtpAdaptive: true, speculativeCandidates: 3);
        var opts = sampled ? new InferenceOptions { MaxTokens = 40, Temperature = 0.8f, TopK = 1, Seed = 7 } : Greedy(40);
        var used = new List<bool>();
        for (int i = 0; i < 5; i++)
            used.Add(gen.Generate("p", opts).Timings.SpeculativeDraftTokens > 0);
        Assert.Equal(new[] { true, false, true, true, true }, used);
    }

    [Fact]
    public void Warmup_DoesNotLeaveAVerdictInTheGate()
    {
        // Warm-up requests are tiny and cold; a bad MTP warm-up sample once made `serve` run plain decode for its first real requests (measured on
        // UD-IQ4_XS: four greedy requests at plain speed while sampled requests, on a gate warm-up never touched, ran at 1.5x).
        using var model = NewModel(draftRight: true, plainMs: 8, rowMs: 2, fixedMs: 6);
        var gen = new TextGenerator(model, new Tok(), mtpEnabled: true, mtpAdaptive: true, speculativeCandidates: 3);
        WarmupRunner.Run(gen, new Tok(), new WarmupOptions { Iterations = 3, MaxTokens = 16 });
        Assert.Equal(0, gen.MtpGate!.PlainMsPerToken);
        Assert.Equal(0, gen.MtpGate.MtpMsPerToken);
        Assert.Equal(0, gen.MtpGate.CostModel.TotalRounds);
        Assert.True(gen.MtpGate.ShouldUseMtp(64), "after warm-up the first real request must follow the size prior (MTP first)");
    }

    private sealed class Tok : ITokenizer
    {
        public int VocabSize => Vocab;
        public int BosTokenId => 1;
        public int EosTokenId => 15;
        public int[] Encode(string text) => [3];
        public string Decode(ReadOnlySpan<int> tokenIds)
        {
            var sb = new StringBuilder();
            foreach (int id in tokenIds) sb.Append(DecodeToken(id));
            return sb.ToString();
        }
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
        public string DecodeToken(int tokenId) => "t" + tokenId.ToString(CultureInfo.InvariantCulture) + " ";
        public int CountTokens(string text) => 1;
    }
}
