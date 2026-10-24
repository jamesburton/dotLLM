using System.Text.Json;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ToolCallParsers;

/// <summary>
/// Issue #776: Gemma-4 tool calls (<c>&lt;|tool_call&gt;call:name{k:&lt;|"|&gt;v&lt;|"|&gt;}&lt;tool_call|&gt;</c>) were
/// returned as plain content. The REAL capture comes from gemma-4-E4B-it Q4_K_M via scripts/harness-smoke.ps1.
/// </summary>
public class Gemma4ToolCallParserTests
{
    private readonly Gemma4ToolCallParser _parser = new();

    // REAL (note the runaway <eos> text: the tokenizer's declared EOS is <turn|>, not <eos>).
    private const string RealCapture =
        "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos><eos>";

    private static JsonElement Args(ToolCall c) => JsonDocument.Parse(c.Arguments).RootElement;

    [Fact]
    public void RealCapture_ParsesAndIgnoresTrailingEos()
    {
        var calls = _parser.TryParse(RealCapture)!;

        var call = Assert.Single(calls);
        Assert.Equal("call_0", call.Id);
        Assert.Equal("get_weather", call.FunctionName);
        Assert.Equal("""{"city":"Paris"}""", call.Arguments);
    }

    [Fact]
    public void ParallelCalls()
    {
        var calls = _parser.TryParse(
            "<|tool_call>call:a{x:1}<tool_call|><|tool_call>call:b{y:<|\"|>z<|\"|>}<tool_call|>")!;

        Assert.Equal(["a", "b"], calls.Select(c => c.FunctionName));
        Assert.Equal(["call_0", "call_1"], calls.Select(c => c.Id));
        Assert.Equal(1, Args(calls[0]).GetProperty("x").GetInt32());
    }

    [Fact]
    public void ScalarTypes_NumbersBoolsNull()
    {
        var a = Args(_parser.TryParse(
            "<|tool_call>call:f{i:42,f:-3.5,t:true,n:false,z:null,neg:-7}<tool_call|>")!.Single());

        Assert.Equal(42, a.GetProperty("i").GetInt32());
        Assert.Equal(-3.5, a.GetProperty("f").GetDouble());
        Assert.True(a.GetProperty("t").GetBoolean());
        Assert.False(a.GetProperty("n").GetBoolean());
        Assert.Equal(JsonValueKind.Null, a.GetProperty("z").ValueKind);
        Assert.Equal(-7, a.GetProperty("neg").GetInt32());
    }

    [Fact]
    public void NestedObjectsAndArrays()
    {
        var a = Args(_parser.TryParse(
            "<|tool_call>call:f{opts:{units:<|\"|>metric<|\"|>,days:3},tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>],grid:[[1,2],[3]]}<tool_call|>")!.Single());

        Assert.Equal("metric", a.GetProperty("opts").GetProperty("units").GetString());
        Assert.Equal(3, a.GetProperty("opts").GetProperty("days").GetInt32());
        Assert.Equal(["a", "b"], a.GetProperty("tags").EnumerateArray().Select(e => e.GetString()));
        Assert.Equal(3, a.GetProperty("grid")[1][0].GetInt32());
    }

    [Fact]
    public void StringValues_MayContainStructuralCharacters()
    {
        var a = Args(_parser.TryParse(
            "<|tool_call>call:f{q:<|\"|>a, b: {c} [d]<|\"|>,after:1}<tool_call|>")!.Single());

        Assert.Equal("a, b: {c} [d]", a.GetProperty("q").GetString());
        Assert.Equal(1, a.GetProperty("after").GetInt32());
    }

    [Fact]
    public void EmptyArguments_AndWhitespace()
    {
        Assert.Equal("{}", _parser.TryParse("<|tool_call>call:ping{}<tool_call|>")!.Single().Arguments);
        var a = Args(_parser.TryParse("<|tool_call>call:f{ a: 1 , b: <|\"|>x<|\"|> }<tool_call|>")!.Single());
        Assert.Equal(1, a.GetProperty("a").GetInt32());
        Assert.Equal("x", a.GetProperty("b").GetString());
    }

    [Fact]
    public void MissingCloseTag_IsTolerated_WhenArgumentsClosed()
    {
        var calls = _parser.TryParse("<|tool_call>call:f{a:1}");
        Assert.Equal("f", Assert.Single(calls!).FunctionName);
    }

    [Fact]
    public void ContentBeforeCall_IsIgnored()
    {
        var calls = _parser.TryParse("Let me look that up.<|tool_call>call:f{a:1}<tool_call|>");
        Assert.Equal("f", Assert.Single(calls!).FunctionName);
    }

    [Fact]
    public void SchemaCoercion_FixesBareNumberForStringParameter()
    {
        var tools = new[] { new ToolDefinition("f", "d", """{"type":"object","properties":{"zip":{"type":"string"}}}""") };
        IToolCallParser p = _parser;

        var a = Args(p.TryParse("<|tool_call>call:f{zip:12345}<tool_call|>", tools)!.Single());

        Assert.Equal("12345", a.GetProperty("zip").GetString());
    }

    [Theory]
    [InlineData("")]
    [InlineData("Paris is the capital of France.")]
    [InlineData("<|tool_call>")]
    [InlineData("<|tool_call>call:")]
    [InlineData("<|tool_call>call:get_weather")]
    [InlineData("<|tool_call>call:get_weather{")]
    [InlineData("<|tool_call>call:get_weather{city:")]
    [InlineData("<|tool_call>call:get_weather{city:<|\"|>Par")]
    [InlineData("<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>")]
    [InlineData("<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,")]
    [InlineData("<|tool_call>call:{a:1}<tool_call|>")]
    [InlineData("<|tool_call>get_weather{a:1}<tool_call|>")]
    [InlineData("<|tool_call>call:f{a}<tool_call|>")]
    [InlineData("<|tool_call>call:f{a:1<tool_call|>")]
    [InlineData("<|tool_call>call:f{a:[1,2}<tool_call|>")]
    public void MalformedOrTruncated_ReturnsNull_AndNeverThrows(string text)
        => Assert.Null(_parser.TryParse(text));

    [Fact]
    public void EveryPrefixOfRealCapture_NeverThrows_AndOnlyCompleteArgumentsParse()
    {
        int closeBrace = RealCapture.IndexOf('}');
        for (int i = 0; i <= RealCapture.Length; i++)
        {
            var calls = _parser.TryParse(RealCapture[..i]);
            if (i <= closeBrace)
                Assert.Null(calls);
            else
                Assert.Equal("Paris", Args(Assert.Single(calls!)).GetProperty("city").GetString());
        }
    }

    [Fact]
    public void DeeplyNestedInput_DoesNotOverflowTheStack()
    {
        string text = "<|tool_call>call:f{a:" + new string('[', 5000) + new string(']', 5000) + "}<tool_call|>";
        Assert.Null(_parser.TryParse(text));
    }

    [Theory]
    [InlineData("<|tool_call>", true)]
    [InlineData("hello <|tool_call>call:f{", true)]
    [InlineData("hello", false)]
    [InlineData("<|channel>thought\n", false)]
    public void IsToolCallStart(string text, bool expected)
        => Assert.Equal(expected, _parser.IsToolCallStart(text));
}
