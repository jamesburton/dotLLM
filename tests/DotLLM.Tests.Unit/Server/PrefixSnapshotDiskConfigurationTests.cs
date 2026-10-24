using DotLLM.Server;
using DotLLM.Cli.Commands;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

public sealed class PrefixSnapshotDiskConfigurationTests
{
    [Theory]
    [InlineData("--prefix-snapshot-disk-budget", "-1GiB")]
    [InlineData("--prefix-snapshot-disk-budget", "invalid")]
    [InlineData("--prefix-snapshot-disk-budget", "1GiB")]
    [InlineData("--prefix-snapshot-disk-dir", "cache")]
    public void DiskFlags_RejectInvalidOrIncompleteLimitsBeforeModelLoad(string flag, string value)
        => Assert.ThrowsAny<ArgumentException>(() => ServerOptions.Parse(["--model", "m.gguf", flag, value]));
    [Fact]
    public void DiskFlags_ResolveUnitsAndAbsoluteDirectoryIntoEngineOptions()
    {
        var options = ServerOptions.Parse(["--model", "m.gguf", "--prefix-snapshots", "20",
            "--prefix-snapshot-host-budget", "8GiB", "--prefix-snapshot-disk-dir", "prefix-cache",
            "--prefix-snapshot-disk-budget", "32GiB"]);
        var engine = options.GetPrefixSnapshotOptions();
        Assert.Equal(20, engine.MaxEntries);
        Assert.Equal(8L << 30, engine.HostBudgetBytes);
        Assert.Equal(32L << 30, engine.DiskBudgetBytes);
        Assert.Equal(Path.GetFullPath("prefix-cache"), engine.DiskDirectory);
    }

    [Fact]
    public void DefaultDiskOptions_AreDisabledWithoutCreatingADirectory()
    {
        var options = new ServerOptions { Model = "m.gguf" }.GetPrefixSnapshotOptions();
        Assert.Null(options.DiskDirectory);
        Assert.Equal(0, options.DiskBudgetBytes);
    }

    [Theory]
    [InlineData(null, "1GiB", false)]
    [InlineData("cache", null, false)]
    [InlineData("cache", "0", false)]
    [InlineData(" ", "1GiB", false)]
    [InlineData("cache", "-1GiB", false)]
    [InlineData("cache", "32GiB", true)]
    [InlineData(null, null, true)]
    public void ServeAndChat_ValidatePairedDiskLimits(string? directory, string? budget, bool valid)
    {
        Assert.Equal(valid, new ServeCommand.Settings
            { PrefixSnapshotDiskDirectory = directory, PrefixSnapshotDiskBudget = budget }.Validate().Successful);
        Assert.Equal(valid, new ChatCommand.Settings
            { PrefixSnapshotDiskDirectory = directory, PrefixSnapshotDiskBudget = budget }.Validate().Successful);
    }

    [Fact]
    public void DiskBudgetErrors_NameTheDiskFlag()
    {
        var error = Assert.Throws<ArgumentException>(() => ServerOptions.ParsePrefixSnapshotDiskBudget("0.1B"));
        Assert.Contains("--prefix-snapshot-disk-budget", error.Message, StringComparison.Ordinal);
        Assert.Equal(1536, ServerOptions.ParsePrefixSnapshotDiskBudget("1.5KiB"));
    }
}
