using DotLLM.Server;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Tests for <see cref="ServerOptions.Parse"/> argument handling — focused on the scheduler
/// config surface (the rest of the flags are covered indirectly by server integration tests).
/// </summary>
public class ServerOptionsParseTests
{
    [Fact]
    public void Default_NoSchedulerSection()
    {
        // Without the flag, no scheduler options are produced — the scheduler uses its defaults
        // (fairness off). A host can still set ServerOptions.Scheduler directly (e.g. from appsettings).
        var opts = ServerOptions.Parse(new[] { "--model", "m.gguf" });
        Assert.Null(opts.Scheduler);
    }

    [Fact]
    public void SchedulerFairnessFlag_EnablesFairness()
    {
        var opts = ServerOptions.Parse(new[] { "--model", "m.gguf", "--scheduler-fairness" });
        Assert.NotNull(opts.Scheduler);
        Assert.True(opts.Scheduler!.EnableFairness);
    }

    [Fact]
    public void Default_NoSpeculativeAndNoPrefillChunk()
    {
        // Zero behavior change when unset: no draft model, K default, chunking off.
        var opts = ServerOptions.Parse(new[] { "--model", "m.gguf" });
        Assert.Null(opts.SpeculativeModel);
        Assert.Equal(5, opts.SpeculativeCandidates);
        Assert.Equal(0, opts.PrefillChunkSize);
    }

    [Theory]
    [InlineData("--speculative-model")]
    [InlineData("--draft-model")]
    public void SpeculativeModelFlag_SetsDraftModel(string flag)
    {
        var opts = ServerOptions.Parse(new[] { "--model", "m.gguf", flag, "draft.gguf" });
        Assert.Equal("draft.gguf", opts.SpeculativeModel);
    }

    [Theory]
    [InlineData("--speculative-k")]
    [InlineData("--draft-tokens")]
    public void SpeculativeKFlag_SetsCandidates(string flag)
    {
        var opts = ServerOptions.Parse(new[] { "--model", "m.gguf", flag, "3" });
        Assert.Equal(3, opts.SpeculativeCandidates);
    }

    [Theory]
    [InlineData("--prefill-chunk-size")]
    [InlineData("--ubatch-size")]
    public void PrefillChunkSizeFlag_SetsChunkSize(string flag)
    {
        var opts = ServerOptions.Parse(new[] { "--model", "m.gguf", flag, "256" });
        Assert.Equal(256, opts.PrefillChunkSize);
    }

    [Fact]
    public void ResolveSchedulerOptions_PrefillChunkSize_MapsToPerStepCap()
    {
        var opts = ServerOptions.Parse(new[] { "--model", "m.gguf", "--prefill-chunk-size", "128" });
        var scheduler = ServerStartup.ResolveSchedulerOptions(opts);
        Assert.NotNull(scheduler);
        Assert.Equal(128, scheduler!.MaxPrefillTokensPerStep);
    }

    [Fact]
    public void ResolveSchedulerOptions_ExplicitSchedulerCap_Wins()
    {
        // An explicit Scheduler.MaxPrefillTokensPerStep (e.g. bound from appsettings) is not
        // overridden by the CLI-level prefill chunk size.
        var opts = ServerOptions.Parse(new[] { "--model", "m.gguf", "--prefill-chunk-size", "128" }) with
        {
            Scheduler = new DotLLM.Engine.Scheduler.ContinuousBatchSchedulerOptions { MaxPrefillTokensPerStep = 512 },
        };
        var scheduler = ServerStartup.ResolveSchedulerOptions(opts);
        Assert.Equal(512, scheduler!.MaxPrefillTokensPerStep);
    }

    [Fact]
    public void ResolveSchedulerOptions_Unset_ReturnsNull()
    {
        var opts = ServerOptions.Parse(new[] { "--model", "m.gguf" });
        Assert.Null(ServerStartup.ResolveSchedulerOptions(opts));
    }

    [Fact]
    public void ResolveSchedulerOptions_FairnessPreserved_WithChunkSize()
    {
        var opts = ServerOptions.Parse(new[]
            { "--model", "m.gguf", "--scheduler-fairness", "--prefill-chunk-size", "64" });
        var scheduler = ServerStartup.ResolveSchedulerOptions(opts);
        Assert.True(scheduler!.EnableFairness);
        Assert.Equal(64, scheduler.MaxPrefillTokensPerStep);
    }
}
