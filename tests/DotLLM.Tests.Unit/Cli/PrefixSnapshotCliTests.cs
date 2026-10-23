using DotLLM.Cli.Commands;
using Xunit;

namespace DotLLM.Tests.Unit.Cli;

public sealed class PrefixSnapshotCliTests
{
    [Theory]
    [InlineData(0, "8GiB")]
    [InlineData(-1, "0")]
    [InlineData(20, "-1GiB")]
    [InlineData(20, "invalid")]
    public void ServeAndChat_RejectInvalidSnapshotLimits(int entries, string budget)
    {
        Assert.False(new ServeCommand.Settings { PrefixSnapshots = entries, PrefixSnapshotHostBudget = budget }.Validate().Successful);
        Assert.False(new ChatCommand.Settings { PrefixSnapshots = entries, PrefixSnapshotHostBudget = budget }.Validate().Successful);
    }

    [Fact]
    public void ServeAndChat_AcceptMultiSessionHostLimitsAndDefaults()
    {
        Assert.True(new ServeCommand.Settings { PrefixSnapshots = 20, PrefixSnapshotHostBudget = "8GiB" }.Validate().Successful);
        Assert.True(new ChatCommand.Settings { PrefixSnapshots = 20, PrefixSnapshotHostBudget = "8GiB" }.Validate().Successful);
        Assert.True(new ServeCommand.Settings().Validate().Successful);
        Assert.True(new ChatCommand.Settings().Validate().Successful);
    }
}
