using System.Text.Json;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ToolCallParsers;

/// <summary>
/// Issue #771 (Llama part): the non-standard JSON shapes Llama 3.x emits after <c>&lt;|python_tag|&gt;</c>.
/// REAL capture: Llama-3.2-1B-Instruct Q8_0 via scripts/harness-smoke.ps1.
/// </summary>
public class LlamaToolCallShapeTests
{
    private readonly LlamaToolCallParser _parser = new();

    // REAL: "function" is the NAME (a string), not an object.
    private const string RealLlama32_1B =
        "<|python_tag|>{\"type\": \"function\", \"function\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}";

    private static JsonElement Args(ToolCall c) => JsonDocument.Parse(c.Arguments).RootElement;

    [Fact]
    public void Real_Llama32_1B_FunctionAsString_Parses()
    {
        var call = Assert.Single(_parser.TryParse(RealLlama32_1B)!);

        Assert.Equal("get_weather", call.FunctionName);
        Assert.Equal("Paris", Args(call).GetProperty("city").GetString());
    }

    [Fact]
    public void OpenAiEnvelope_FunctionObjectWithArguments_Parses()
    {
        var call = Assert.Single(_parser.TryParse(
            """<|python_tag|>{"type":"function","function":{"name":"get_weather","arguments":{"city":"Paris"}}}""")!);

        Assert.Equal("get_weather", call.FunctionName);
        Assert.Equal("Paris", Args(call).GetProperty("city").GetString());
    }

    [Fact]
    public void OpenAiEnvelope_FunctionObjectWithParametersAndStringArguments_Parses()
    {
        var call = Assert.Single(_parser.TryParse(
            """<|python_tag|>{"type":"function","function":{"name":"f","arguments":"{\"a\": 1}"}}""")!);

        Assert.Equal("f", call.FunctionName);
        Assert.Equal(1, Args(call).GetProperty("a").GetInt32());
    }

    [Fact]
    public void Documented_NameParameters_StillParses()
    {
        var call = Assert.Single(_parser.TryParse(
            """<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}""")!);
        Assert.Equal("get_weather", call.FunctionName);
    }

    [Fact]
    public void TypeFunction_WithTopLevelName_Parses()
    {
        var call = Assert.Single(_parser.TryParse(
            """<|python_tag|>{"type": "function", "name": "get_weather", "parameters": {"city": "Paris"}}""")!);
        Assert.Equal("get_weather", call.FunctionName);
    }

    [Fact]
    public void ParallelCalls_AsJsonArray()
    {
        var calls = _parser.TryParse(
            """<|python_tag|>[{"name":"a","parameters":{}},{"type":"function","function":"b","parameters":{"x":1}}]""")!;
        Assert.Equal(["a", "b"], calls.Select(c => c.FunctionName));
        Assert.Equal(["call_0", "call_1"], calls.Select(c => c.Id));
    }

    [Theory]
    [InlineData("; ")]
    [InlineData("\n")]
    [InlineData(", ")]
    [InlineData("")]
    public void ParallelCalls_SeparatedByTextual_Delimiters(string sep)
    {
        var calls = _parser.TryParse(
            "<|python_tag|>{\"name\":\"a\",\"parameters\":{}}" + sep + "{\"name\":\"b\",\"parameters\":{\"x\":2}}")!;
        Assert.Equal(["a", "b"], calls.Select(c => c.FunctionName));
        Assert.Equal(["call_0", "call_1"], calls.Select(c => c.Id));
    }

    [Fact]
    public void BareJson_WholeResponse_StillAccepted_WithoutMarker()
    {
        var call = Assert.Single(_parser.TryParse(
            """{"type": "function", "function": "get_weather", "parameters": {"city": "Paris"}}""")!);
        Assert.Equal("get_weather", call.FunctionName);
    }

    [Fact]
    public void BareJson_AfterProse_IsNotAToolCall()
    {
        Assert.Null(_parser.TryParse("""Use this: {"name": "get_weather", "parameters": {"city": "Paris"}}"""));
    }

    [Fact]
    public void BareJson_FollowedByProse_IsNotAToolCall()
    {
        Assert.Null(_parser.TryParse("""{"name": "get_weather", "parameters": {}} is the shape I would use."""));
    }

    [Fact]
    public void BuiltinPythonicCall_Parses()
    {
        var call = Assert.Single(_parser.TryParse(
            "<|python_tag|>brave_search.call(query=\"weather in Paris\", count=3)")!);

        Assert.Equal("brave_search", call.FunctionName);
        Assert.Equal("weather in Paris", Args(call).GetProperty("query").GetString());
        Assert.Equal(3, Args(call).GetProperty("count").GetInt32());
    }

    [Theory]
    [InlineData("<|python_tag|>")]
    [InlineData("<|python_tag|>{")]
    [InlineData("<|python_tag|>{\"type\": \"function\", \"function\": \"get_weather\", \"parameters\": {\"city\": \"Par")]
    [InlineData("<|python_tag|>{\"type\": \"function\"}")]
    [InlineData("<|python_tag|>{\"function\": {\"nope\": 1}}")]
    [InlineData("<|python_tag|>[1,2,3]")]
    [InlineData("<|python_tag|>just words")]
    public void MalformedOrPartial_ReturnsNull(string text)
        => Assert.Null(_parser.TryParse(text));

    [Fact]
    public void EveryPrefixOfRealCapture_NeverThrows()
    {
        for (int i = 0; i <= RealLlama32_1B.Length; i++)
        {
            var calls = _parser.TryParse(RealLlama32_1B[..i]);
            if (i == RealLlama32_1B.Length)
                Assert.NotNull(calls);
            else if (i < RealLlama32_1B.Length - 3)
                Assert.Null(calls); // outer object not yet closed
        }
    }

    [Fact]
    public void SchemaCoercion_QuotedNumberBecomesNumber()
    {
        var tools = new[] { new ToolDefinition("f", "d", """{"type":"object","properties":{"n":{"type":"integer"}}}""") };
        IToolCallParser p = _parser;

        var a = Args(p.TryParse("""<|python_tag|>{"name":"f","parameters":{"n":"17"}}""", tools)!.Single());

        Assert.Equal(17, a.GetProperty("n").GetInt32());
    }

    [Fact]
    public void IsToolCallStart_IsMarkerOnly()
    {
        Assert.True(_parser.IsToolCallStart("<|python_tag|>{"));
        Assert.False(_parser.IsToolCallStart("""{"name":"f"}"""));
    }
}
