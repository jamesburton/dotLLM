using DotLLM.Engine;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Policy of <see cref="MtpAdaptiveGate"/>: on Tev1-4B (Vulkan) MTP ran at 18-20 tok/s against 35-37 plain, so a fixed
/// "MTP on" default halves throughput there while it wins on large models. The gate must never engage MTP for short
/// requests, must explore each arm once, then follow whichever measured faster, and must re-probe the loser.
/// </summary>
public sealed class MtpAdaptiveGateTests
{
    private const long Small = 3L << 30;   // 3 GiB  -> plain first
    private const long Large = 7L << 30;   // 7 GiB  -> MTP first

    [Fact]
    public void ShortRequests_NeverUseMtp()
    {
        var g = new MtpAdaptiveGate(Large);
        for (int i = 0; i < 5; i++)
            Assert.False(g.ShouldUseMtp(MtpAdaptiveGate.MinMaxTokens - 1));
        Assert.True(g.ShouldUseMtp(MtpAdaptiveGate.MinMaxTokens));
    }

    [Fact]
    public void FirstArm_FollowsTheModelSizePrior_ThenExploresTheOther()
    {
        var big = new MtpAdaptiveGate(Large);
        Assert.True(big.ShouldUseMtp(64));
        big.Record(usedMtp: true, generatedTokens: 65, decodeMs: 650);      // 10 ms/token
        Assert.False(big.ShouldUseMtp(64));                                  // now explore plain once

        var small = new MtpAdaptiveGate(Small);
        Assert.False(small.ShouldUseMtp(64));
        small.Record(usedMtp: false, generatedTokens: 65, decodeMs: 1600);  // 25 ms/token
        Assert.True(small.ShouldUseMtp(64));                                 // explore MTP once
    }

    [Fact]
    public void AfterBothArmsSampled_ChoosesTheFasterOne()
    {
        // Tev1-4B-like: plain 27 ms/token, MTP 55 ms/token  -> plain wins.
        var g = new MtpAdaptiveGate(Small);
        g.Record(false, 65, 27 * 64);
        g.Record(true, 65, 55 * 64);
        Assert.False(g.ShouldUseMtp(64));

        // Bonsai-27B-like: plain 45 ms/token, MTP 22 ms/token -> MTP wins.
        var h = new MtpAdaptiveGate(Large);
        h.Record(false, 65, 45 * 64);
        h.Record(true, 65, 22 * 64);
        Assert.True(h.ShouldUseMtp(64));
    }

    [Fact]
    public void TheLosingArm_IsReprobedPeriodically()
    {
        var g = new MtpAdaptiveGate(Small);
        g.Record(false, 65, 27 * 64);
        g.Record(true, 65, 55 * 64);

        int mtpPicks = 0;
        for (int i = 0; i < MtpAdaptiveGate.ReprobeEvery * 3; i++)
            if (g.ShouldUseMtp(64)) mtpPicks++;
        Assert.Equal(3, mtpPicks);   // exactly one re-probe per ReprobeEvery decisions
    }

    [Fact]
    public void TinyOrEmptySamples_AreIgnored()
    {
        var g = new MtpAdaptiveGate(Small);
        g.Record(false, generatedTokens: 1, decodeMs: 25);        // 1-token classifier answer
        g.Record(false, generatedTokens: MtpAdaptiveGate.MinSampleTokens, decodeMs: 100);   // 5 decode tokens
        g.Record(true, generatedTokens: 64, decodeMs: 0);          // no timing
        Assert.Equal(0, g.PlainMsPerToken);
        Assert.Equal(0, g.MtpMsPerToken);
    }

    [Fact]
    public void Measurements_AreSmoothed_NotLastWriterWins()
    {
        var g = new MtpAdaptiveGate(Small);
        g.Record(false, 65, 20 * 64);
        g.Record(false, 65, 40 * 64);
        Assert.InRange(g.PlainMsPerToken, 20.5, 39.5);
    }
}
