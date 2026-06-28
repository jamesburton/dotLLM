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
}
