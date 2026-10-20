using DotLLM.Cli.Commands;
using DotLLM.Server;
using Spectre.Console.Cli;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Issue #1068: the CUDA hybrid's served context is fitted to free VRAM at load (or set with <c>--ctx-size</c>) and reported
/// in <c>/props</c>, so clients compact before the KV cache outgrows the card. This covers the choice and the flag; the
/// measurement itself is <c>CudaQwen3HybridDenseContextFitTests</c> (GPU).
/// </summary>
public sealed class ServedContextFitTests
{
    [Fact]
    public void NoRequest_ServesTheFit()
    {
        Assert.Equal(61440, ServerStartup.ChooseServedContext(null, fit: 61440, native: 262144, out string? warning));
        Assert.Null(warning);
    }

    [Fact]
    public void NoRequest_FitAboveTheModel_ServesTheModelWindow()
        => Assert.Equal(32768, ServerStartup.ChooseServedContext(null, fit: 262144, native: 32768, out _));

    [Fact]
    public void SmallFit_IsServed_WithAWarning_AndNeverBelowOneGranule()
    {
        Assert.Equal(4096, ServerStartup.ChooseServedContext(null, fit: 4096, native: 262144, out string? w1));
        Assert.Contains("only 4096 tokens", w1);
        Assert.Equal(1024, ServerStartup.ChooseServedContext(null, fit: 0, native: 262144, out string? w2));
        Assert.NotNull(w2);
    }

    [Fact]
    public void ExplicitRequest_OverridesTheFit_CappedAtTheModel_AndWarnsWhenItExceedsTheFit()
    {
        Assert.Equal(16384, ServerStartup.ChooseServedContext(16384, fit: 61440, native: 262144, out string? below));
        Assert.Null(below);

        Assert.Equal(98304, ServerStartup.ChooseServedContext(98304, fit: 61440, native: 262144, out string? above));
        Assert.Contains("exceeds the 61440 tokens", above);

        // The boundary: exactly the fit is silent, one token past it warns.
        ServerStartup.ChooseServedContext(61440, fit: 61440, native: 262144, out string? atFit);
        Assert.Null(atFit);
        ServerStartup.ChooseServedContext(61441, fit: 61440, native: 262144, out string? pastFit);
        Assert.NotNull(pastFit);

        Assert.Equal(262144, ServerStartup.ChooseServedContext(1_000_000, fit: 300_000, native: 262144, out _));
    }

    [Theory]
    [InlineData(new[] { "model.gguf", "--ctx-size", "32768" }, 32768)]
    [InlineData(new[] { "model.gguf", "-c", "4096" }, 4096)]
    [InlineData(new[] { "model.gguf" }, null)]
    public void Serve_ParsesCtxSize(string[] args, int? expected)
    {
        Capture.Last = null;
        var app = new CommandApp();
        app.Configure(c => c.AddCommand<Capture>("x"));
        Assert.Equal(0, app.Run(new[] { "x" }.Concat(args).ToArray()));
        Assert.Equal(expected, Capture.Last!.CtxSize);
    }

    private sealed class Capture : Command<ServeCommand.Settings>
    {
        public static ServeCommand.Settings? Last;
        public override int Execute(CommandContext context, ServeCommand.Settings settings) { Last = settings; return 0; }
    }
}
