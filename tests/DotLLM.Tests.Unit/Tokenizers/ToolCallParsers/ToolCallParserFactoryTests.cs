using DotLLM.Core.Configuration;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ToolCallParsers;

/// <summary>
/// Covers <see cref="ToolCallParserFactory.ForToolChoice"/> — the single place that encodes
/// "constrained tool-call output is bare JSON, so it must be parsed markerlessly" (#325).
/// </summary>
public class ToolCallParserFactoryTests
{
    /// <summary>Exactly what the constraint emits under <c>tool_choice=required</c>.</summary>
    private const string ConstrainedOutput = """{"name": "get_weather", "arguments": {"city": "Tokyo"}}""";

    [Fact]
    public void ForToolChoice_Auto_KeepsModelParser()
    {
        var model = new HermesToolCallParser();
        Assert.Same(model, ToolCallParserFactory.ForToolChoice(new ToolChoice.Auto(), model));
    }

    [Fact]
    public void ForToolChoice_None_KeepsModelParser()
    {
        var model = new LlamaToolCallParser();
        Assert.Same(model, ToolCallParserFactory.ForToolChoice(new ToolChoice.None(), model));
    }

    [Theory]
    [InlineData("required")]
    [InlineData("function")]
    public void ForToolChoice_Constrained_UsesMarkerlessParser(string choice)
    {
        ToolChoice toolChoice = choice == "required"
            ? new ToolChoice.Required()
            : new ToolChoice.Function("get_weather");

        var model = new HermesToolCallParser();
        var parser = ToolCallParserFactory.ForToolChoice(toolChoice, model);

        Assert.IsType<GenericToolCallParser>(parser);
    }

    /// <summary>
    /// The regression this exists for: every marker-based parser returns <c>null</c> on the
    /// constrained bare-JSON output, and the swapped parser recovers the call.
    /// </summary>
    [Theory]
    [MemberData(nameof(MarkerParsers))]
    public void MarkerParser_CannotParse_ConstrainedOutput_ButSwappedParserCan(IToolCallParser modelParser)
    {
        Assert.Null(modelParser.TryParse(ConstrainedOutput));

        var parser = ToolCallParserFactory.ForToolChoice(new ToolChoice.Required(), modelParser);
        var calls = parser.TryParse(ConstrainedOutput);

        Assert.NotNull(calls);
        var call = Assert.Single(calls);
        Assert.Equal("get_weather", call.FunctionName);
        Assert.Contains("Tokyo", call.Arguments, StringComparison.Ordinal);
    }

    /// <summary>
    /// <see cref="LlamaToolCallParser"/> is the one exception: it already falls back to
    /// whole-response bare JSON (for Llama 3.2 lightweight models that omit
    /// <c>&lt;|python_tag|&gt;</c>), so constrained output survives it. It still needs the swap
    /// for streaming — its <c>IsToolCallStart</c> is marker-only.
    /// </summary>
    [Fact]
    public void LlamaParser_AlreadyTolerates_ConstrainedOutput()
    {
        var calls = new LlamaToolCallParser().TryParse(ConstrainedOutput);
        Assert.NotNull(calls);
        Assert.Equal("get_weather", Assert.Single(calls).FunctionName);

        // ...but streaming suppression still requires the markerless parser.
        Assert.False(new LlamaToolCallParser().IsToolCallStart(ConstrainedOutput));
        Assert.True(ToolCallParserFactory
            .ForToolChoice(new ToolChoice.Required(), new LlamaToolCallParser())
            .IsToolCallStart(ConstrainedOutput));
    }

    public static TheoryData<IToolCallParser> MarkerParsers()
    {
        var data = new TheoryData<IToolCallParser>();
        data.Add(new HermesToolCallParser());
        data.Add(new XmlToolCallParser());
        data.Add(new MistralToolCallParser());
        data.Add(new PythonicToolCallParser());
        return data;
    }

    /// <summary>
    /// Streaming suppression is marker-driven; the swapped parser must also recognise the
    /// constrained output as a tool-call start or the raw JSON leaks to the console.
    /// </summary>
    [Fact]
    public void SwappedParser_DetectsToolCallStart_OnBareJson()
    {
        var parser = ToolCallParserFactory.ForToolChoice(
            new ToolChoice.Required(), new HermesToolCallParser());

        Assert.False(new HermesToolCallParser().IsToolCallStart("""{"name": "get_"""));
        Assert.True(parser.IsToolCallStart("""{"name": "get_"""));
    }
}
