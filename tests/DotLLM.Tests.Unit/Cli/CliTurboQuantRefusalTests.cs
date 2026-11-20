using DotLLM.Cli.Helpers;
using DotLLM.Core.Configuration;
using DotLLM.Server;
using Xunit;

namespace DotLLM.Tests.Unit.Cli;

/// <summary>
/// #1888, inverted by #1892: <c>run</c>, <c>chat</c> and <c>bench</c> run TurboQuant in place like <c>serve</c>. What is still refused before
/// the model loads is what the flags alone decide: mixed formats, QJL on a named GPU, <c>--cache-window</c>, the paged cache.
/// </summary>
public class CliTurboQuantRefusalTests
{
    [Theory]
    [InlineData("tq4", "tq4")]
    [InlineData("tq4q", "tq4q")]
    [InlineData("turboquant", "turboquant")]
    [InlineData("tq2", "tq2")]
    [InlineData("tq8", "tq8")]
    public void Run_Chat_Bench_TurboQuant_IsNoLongerRefusedBeforeLoad(string k, string v)
    {
        foreach (var command in new[] { KvCommand.Run, KvCommand.Chat, KvCommand.Bench })
            Assert.Null(KvStorageCli.TurboQuantRefusal(command, k, v));
    }

    [Theory]
    [InlineData(null, "tq4")]
    [InlineData("tq4", null)]
    [InlineData("tq3", "tq4")]
    [InlineData("tq4", "tq4q")]
    public void MixedOrOneSidedTq_IsRefused_BeforeLoad(string? k, string? v)
    {
        string? reason = KvStorageCli.TurboQuantRefusal(KvCommand.Run, k, v);
        Assert.NotNull(reason);
        Assert.Contains("#1895", reason);
    }

    [Fact]
    public void CacheWindow_WithTq_IsRefused_BeforeLoad_ForRunAndChat()
    {
        foreach (var command in new[] { KvCommand.Run, KvCommand.Chat })
        {
            string? reason = KvStorageCli.TurboQuantRefusal(command, "tq4", "tq4", cacheWindow: 32);
            Assert.NotNull(reason);
            Assert.Contains("--cache-window 32", reason);
        }
    }

    [Theory]
    [InlineData("vulkan")]
    [InlineData("gpu:0")]
    public void Tq4q_OnANamedGpu_IsRefused_BeforeLoad(string device)
    {
        string? reason = KvStorageCli.TurboQuantRefusal(KvCommand.Run, "tq4q", "tq4q", device: device);
        Assert.NotNull(reason);
        Assert.Contains("#1894", reason);
        Assert.Null(KvStorageCli.TurboQuantRefusal(KvCommand.Run, "tq4q", "tq4q", device: "cpu"));
        Assert.Null(KvStorageCli.TurboQuantRefusal(KvCommand.Run, "tq4q", "tq4q", device: "auto"));
    }

    [Fact]
    public void Paged_WithTq_IsRefused_BeforeLoad()
        => Assert.Contains("--no-paged", KvStorageCli.TurboQuantRefusal(KvCommand.Chat, "tq4", "tq4", paged: true));

    [Theory]
    [InlineData(null, null)]
    [InlineData("q8_0", "q8_0")]
    [InlineData("f16", "q4_0")]
    public void Run_NonTurboQuant_IsNotRefused(string? k, string? v)
        => Assert.Null(KvStorageCli.TurboQuantRefusal(KvCommand.Run, k, v, cacheWindow: 16));

    [Fact]
    public void Run_UnknownKvType_IsLeftToTheFlagParser()
        => Assert.Null(KvStorageCli.TurboQuantRefusal(KvCommand.Run, "bogus", null));

    [Theory]
    [InlineData("tq3", 3, false)]
    [InlineData("tq8", 8, false)]
    [InlineData("tq4q", 4, true)]
    [InlineData("tq", 4, false)]
    public void BuildKvConfig_KeepsBitsAndQjl_PerSide(string flag, int bits, bool qjl)
    {
        // run / chat / bench used to build `new KvCacheConfig(Key, Value, window)` and drop both.
        Assert.True(KvStorageCli.TryBuild(flag, flag, null, null, 0, out var request, out _));
        var kv = KvStorageCli.BuildKvConfig(request, 0, out string? mismatch);
        Assert.Null(mismatch);
        Assert.Equal(KvCacheDType.TurboQuant, kv.KeyDType);
        Assert.Equal(bits, kv.TurboQuantBits);
        Assert.Equal(qjl, kv.TurboQuantUseQjl);
    }
}

/// <summary>#1888 / #1892: the real command tree refuses before resolving or loading the model (kills a removed call site in run / chat).</summary>
[Collection(CliConsoleCollection.Name)]
public sealed class CliTurboQuantCommandTests
{
    /// <summary>Runs the real command tree, capturing Console and AnsiConsole (chat writes errors through AnsiConsole, which binds the
    /// console it first saw, so swapping Console.Out alone captures nothing once another test has used it).</summary>
    internal static (int Code, string Output) Run(params string[] args) => RunWithInput(null, args);

    /// <summary>Scripted keyboard for <c>AnsiConsole.Prompt</c>: each line is typed, then Enter; after the last line the input is exhausted.</summary>
    private sealed class ScriptedInput(string text) : Spectre.Console.IAnsiConsoleInput
    {
        private readonly Queue<ConsoleKeyInfo> _keys = new(text.Select(ch => ch == (char)10
            ? new ConsoleKeyInfo((char)13, ConsoleKey.Enter, false, false, false)
            : new ConsoleKeyInfo(ch, ConsoleKey.A, false, false, false)));
        public bool IsKeyAvailable() => _keys.Count > 0;
        public ConsoleKeyInfo? ReadKey(bool intercept) => _keys.Count > 0 ? _keys.Dequeue() : null;
        public Task<ConsoleKeyInfo?> ReadKeyAsync(bool intercept, CancellationToken cancellationToken) => Task.FromResult(ReadKey(intercept));
    }

    /// <summary>The captured console with the scripted keyboard in front of it.</summary>
    private sealed class ConsoleWithInput(Spectre.Console.IAnsiConsole inner, Spectre.Console.IAnsiConsoleInput input) : Spectre.Console.IAnsiConsole
    {
        public Spectre.Console.Profile Profile => inner.Profile;
        public Spectre.Console.IAnsiConsoleCursor Cursor => inner.Cursor;
        public Spectre.Console.IAnsiConsoleInput Input => input;
        public Spectre.Console.IExclusivityMode ExclusivityMode => inner.ExclusivityMode;
        public Spectre.Console.Rendering.RenderPipeline Pipeline => inner.Pipeline;
        public void Clear(bool home) => inner.Clear(home);
        public void Write(Spectre.Console.Rendering.IRenderable renderable) => inner.Write(renderable);
    }

    /// <summary>As <see cref="Run"/> with <paramref name="stdin"/> typed at the console (a chat turn; lines end with a newline).</summary>
    internal static (int Code, string Output) RunWithInput(string? stdin, params string[] args)
    {
        var (out0, err0, ansi0) = (Console.Out, Console.Error, Spectre.Console.AnsiConsole.Console);
        var sw = new StringWriter();
        Console.SetOut(sw);
        Console.SetError(sw);
        var captured = Spectre.Console.AnsiConsole.Create(new Spectre.Console.AnsiConsoleSettings
        {
            Out = new Spectre.Console.AnsiConsoleOutput(sw),
            Ansi = Spectre.Console.AnsiSupport.No,
            Interactive = Spectre.Console.InteractionSupport.Yes,
        });
        Spectre.Console.AnsiConsole.Console = stdin is null ? captured : new ConsoleWithInput(captured, new ScriptedInput(stdin));
        try { return (CliCommandModel.Run(args).ExitCode, sw.ToString()); }
        finally { Console.SetOut(out0); Console.SetError(err0); Spectre.Console.AnsiConsole.Console = ansi0; }
    }

    [Fact]
    public void Run_Tq4_WithCacheWindow_ExitsNonZero_WithTheRefusal()
    {
        var (code, output) = Run("run", "no-such-model-1892", "-p", "hi", "--cache-type-k", "tq4", "--cache-type-v", "tq4", "--cache-window", "16", "--json");
        Assert.NotEqual(0, code);
        Assert.Contains("--cache-window 16", output);
        Assert.Contains("#1887", output);
    }

    [Fact]
    public void Chat_Tq4_WithCacheWindow_ExitsNonZero_WithTheRefusal()
    {
        var (code, output) = Run("chat", "no-such-model-1892", "--cache-type-k", "tq4", "--cache-type-v", "tq4", "--cache-window", "16");
        Assert.NotEqual(0, code);
        Assert.Contains("--cache-window 16", output);
    }

    [Fact]
    public void Run_Tq4_IsNotRefusedBeforeTheModelResolves()
    {
        // Used to exit with #1887 before resolving anything; now it reaches model resolution, which fails for a different reason.
        var (code, output) = Run("run", "no-such-model-1892", "-p", "hi", "--cache-type-k", "tq4", "--cache-type-v", "tq4", "--json");
        Assert.NotEqual(0, code);
        Assert.DoesNotContain("does not run TurboQuant", output);
        Assert.DoesNotContain("--device cpu --no-paged", output);
    }
}
