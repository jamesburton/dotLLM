using DotLLM.Core.Configuration;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ToolCallParsers;

/// <summary>#798: Harmony (gpt-oss) tool-call parsing.</summary>
public class HarmonyToolCallParserTests
{
    private readonly HarmonyToolCallParser _p = new();

    private static readonly ToolDefinition[] Tools =
    [
        new("get_weather", "w", """{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}"""),
    ];

    [Fact]
    public void ParsesAChannelRecipientCall_WithConstrain()
    {
        var calls = _p.TryParse("<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Paris\"}<|call|>");
        var c = Assert.Single(calls!);
        Assert.Equal("get_weather", c.FunctionName);
        Assert.Equal("{\"city\":\"Paris\"}", c.Arguments);
    }

    [Fact]
    public void ParsesFromRawFullTurn_IgnoringAnalysisAndFinal()
    {
        var calls = _p.TryParse(
            "<|channel|>analysis<|message|>Need weather. Call get_weather to=functions.nothing.<|end|>"
            + "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Paris\"}<|call|>");
        var c = Assert.Single(calls!);
        Assert.Equal("get_weather", c.FunctionName);
    }

    [Fact]
    public void RecipientInTheRoleHeader()
    {
        var calls = _p.TryParse("<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{\"city\":\"Rome\"}<|call|>");
        Assert.Equal("Rome", System.Text.Json.JsonDocument.Parse(Assert.Single(calls!).Arguments).RootElement.GetProperty("city").GetString());

        // As generated after a prompt that already ends in <|start|>assistant.
        var calls2 = _p.TryParse(" to=functions.get_weather<|channel|>commentary json<|message|>{\"city\":\"Rome\"}");
        Assert.Single(calls2!);
    }

    [Fact]
    public void TerminatorIsOptional_BecauseCallIsAnEndOfGenerationToken()
    {
        var calls = _p.TryParse("<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Paris\"}");
        Assert.Single(calls!);
    }

    [Fact]
    public void ParallelCalls_AreSequentiallyIdentified()
    {
        var calls = _p.TryParse(
            "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Paris\"}<|call|>"
            + "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Rome\"}<|call|>");
        Assert.Equal(2, calls!.Length);
        Assert.Equal(["call_0", "call_1"], calls.Select(c => c.Id));
        Assert.Contains("Rome", calls[1].Arguments, StringComparison.Ordinal);
    }

    [Fact]
    public void TruncatedArguments_AreNotReported()
    {
        Assert.Null(_p.TryParse("<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Par"));
    }

    [Fact]
    public void BuiltinToolRecipients_AreNotOurs()
    {
        Assert.Null(_p.TryParse("<|channel|>analysis to=browser.search code<|message|>{\"query\":\"x\"}<|call|>"));
        Assert.Null(_p.TryParse("<|channel|>commentary to=python <|constrain|>code<|message|>print(1)<|call|>"));
    }

    [Fact]
    public void PlainTextAndFinalChannel_AreNotCalls()
    {
        Assert.Null(_p.TryParse("Paris."));
        Assert.Null(_p.TryParse("<|channel|>final<|message|>The weather is sunny.<|return|>"));
        Assert.Null(_p.TryParse(""));
    }

    [Fact]
    public void EmptyBody_IsAnEmptyArgumentObject()
    {
        var c = Assert.Single(_p.TryParse("<|channel|>commentary to=functions.ping <|constrain|>json<|message|><|call|>")!);
        Assert.Equal("{}", c.Arguments);
    }

    [Fact]
    public void ArgumentsAreCoercedToTheSchema()
    {
        var c = Assert.Single(_p.TryParse(
            "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Paris\",\"days\":\"3\"}<|call|>", Tools)!);
        Assert.Contains("\"days\":3", c.Arguments.Replace(" ", ""), StringComparison.Ordinal);
    }

    [Fact]
    public void IsToolCallStart_NeedsTheRecipientAndTheMessageMarker()
    {
        Assert.True(_p.IsToolCallStart("<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>"));
        Assert.True(_p.IsToolCallStart("I'll check.<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"));
        Assert.False(_p.IsToolCallStart("<|channel|>commentary to=functions.get_wea"));
        Assert.False(_p.IsToolCallStart("The answer is Paris."));
        Assert.False(_p.IsToolCallStart("<|channel|>analysis<|message|>hmm"));
    }

    [Fact]
    public void Factory_PicksHarmony_ForTheTemplate_AndTheArchitecture()
    {
        const string template = "{{- '<|start|>assistant<|channel|>final<|message|>' }}";
        Assert.IsType<HarmonyToolCallParser>(ToolCallParserFactory.Create(Architecture.Llama, template));
        Assert.IsType<HarmonyToolCallParser>(ToolCallParserFactory.Create(Architecture.GptOss));
    }
}
