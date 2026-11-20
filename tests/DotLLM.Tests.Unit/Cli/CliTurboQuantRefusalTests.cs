using DotLLM.Cli.Helpers;
using DotLLM.Server;
using Xunit;

namespace DotLLM.Tests.Unit.Cli;

/// <summary>
/// #1888: <c>run</c> and <c>chat</c> had no TurboQuant branch at all, dropped the bit width at parse time and served tq with the default
/// cache. Until #1887 step 2 they refuse every TurboQuant form before the model loads.
/// </summary>
public class CliTurboQuantRefusalTests
{
    [Theory]
    [InlineData("tq4", "tq4")]
    [InlineData("tq4q", "tq4q")]
    [InlineData("turboquant", "turboquant")]
    [InlineData(null, "tq4")]
    [InlineData("tq4", null)]
    public void Run_TurboQuant_IsRefused_BeforeLoad(string? k, string? v)
    {
        string? reason = KvStorageCli.TurboQuantRefusal(KvCommand.Run, k, v);
        Assert.NotNull(reason);
        Assert.Contains("#1887", reason);
    }

    [Fact]
    public void Run_Tq4_PointsAtTheRealPath()
        => Assert.Contains("dotllm serve --device cpu --no-paged", KvStorageCli.TurboQuantRefusal(KvCommand.Run, "tq4", "tq4"));

    [Fact]
    public void Chat_Tq4_IsRefused_BeforeLoad()
        => Assert.Contains("dotllm serve --device cpu --no-paged", KvStorageCli.TurboQuantRefusal(KvCommand.Chat, "tq4", "tq4"));

    [Theory]
    [InlineData(null, null)]
    [InlineData("q8_0", "q8_0")]
    [InlineData("f16", "q4_0")]
    public void Run_NonTurboQuant_IsNotRefused(string? k, string? v)
        => Assert.Null(KvStorageCli.TurboQuantRefusal(KvCommand.Run, k, v));

    [Fact]
    public void Run_UnknownKvType_IsLeftToTheFlagParser()
        => Assert.Null(KvStorageCli.TurboQuantRefusal(KvCommand.Run, "bogus", null));
}

/// <summary>#1888: the real command tree refuses before resolving or loading the model (kills a removed call site in run / chat).</summary>
[Collection(CliConsoleCollection.Name)]
public sealed class CliTurboQuantCommandTests
{
    /// <summary>Runs the real command tree, capturing Console and AnsiConsole (chat writes errors through AnsiConsole, which binds the
    /// console it first saw, so swapping Console.Out alone captures nothing once another test has used it).</summary>
    internal static (int Code, string Output) Run(params string[] args)
    {
        var (out0, err0, ansi0) = (Console.Out, Console.Error, Spectre.Console.AnsiConsole.Console);
        var sw = new StringWriter();
        Console.SetOut(sw);
        Console.SetError(sw);
        Spectre.Console.AnsiConsole.Console = Spectre.Console.AnsiConsole.Create(new Spectre.Console.AnsiConsoleSettings
        {
            Out = new Spectre.Console.AnsiConsoleOutput(sw),
            Ansi = Spectre.Console.AnsiSupport.No,
        });
        try { return (CliCommandModel.Run(args).ExitCode, sw.ToString()); }
        finally { Console.SetOut(out0); Console.SetError(err0); Spectre.Console.AnsiConsole.Console = ansi0; }
    }

    [Fact]
    public void Run_Tq4_ExitsNonZero_WithTheRefusal()
    {
        var (code, output) = Run("run", "no-such-model-1888", "-p", "hi", "--cache-type-k", "tq4", "--cache-type-v", "tq4", "--json");
        Assert.NotEqual(0, code);
        Assert.Contains("#1887", output);
    }

    [Fact]
    public void Chat_Tq4_ExitsNonZero_WithTheRefusal()
    {
        var (code, output) = Run("chat", "no-such-model-1888", "--cache-type-k", "tq4", "--cache-type-v", "tq4");
        Assert.NotEqual(0, code);
        Assert.Contains("#1887", output);
    }
}

