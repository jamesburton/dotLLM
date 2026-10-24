using DotLLM.Server;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

public sealed class PrefixSnapshotConfigurationTests
{
    [Theory]
    [InlineData("--prefix-snapshots", "0")]
    [InlineData("--prefix-snapshots", "-1")]
    [InlineData("--prefix-snapshot-host-budget", "-1GiB")]
    [InlineData("--prefix-snapshot-host-budget", "nonsense")]
    [InlineData("--prefix-snapshot-host-budget", "999999999999999999GiB")]
    public void InvalidSnapshotFlags_AreRejectedBeforeLoadingAModel(string flag, string value)
        => Assert.ThrowsAny<ArgumentException>(() => ServerOptions.Parse(["--model", "m.gguf", flag, value]));

    [Fact]
    public void SnapshotFlags_ReachTheEngineOptionsWithBinaryByteBudget()
    {
        var options = ServerOptions.Parse(["--model", "m.gguf", "--prefix-snapshots", "20", "--prefix-snapshot-host-budget", "8GiB"]);
        var engine = options.GetPrefixSnapshotOptions();
        Assert.Equal(20, engine.MaxEntries);
        Assert.Equal(8L << 30, engine.HostBudgetBytes);
    }

    [Fact]
    public void DefaultSnapshotOptions_PreserveOneEntryAndNoHostTier()
    {
        var engine = new ServerOptions { Model = "m.gguf" }.GetPrefixSnapshotOptions();
        Assert.Equal(1, engine.MaxEntries);
        Assert.Equal(0, engine.HostBudgetBytes);
    }

    [Theory]
    [InlineData("1024", 1024L)]
    [InlineData("1.5KiB", 1536L)]
    [InlineData("8GB", 8000000000L)]
    [InlineData(" 2 mib ", 2097152L)]
    [InlineData("0", 0L)]
    public void HostBudgets_ParseExactly(string text, long expected)
        => Assert.Equal(expected, ServerOptions.ParsePrefixSnapshotHostBudget(text));

    [Fact]
    public void FractionalBytesAndNegativeProgrammaticBudgets_AreRejected()
    {
        Assert.Throws<ArgumentException>(() => ServerOptions.ParsePrefixSnapshotHostBudget("0.1B"));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ServerOptions { Model = "m", PrefixSnapshotHostBudgetBytes = -1 }.GetPrefixSnapshotOptions());
    }
}
