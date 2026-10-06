using System.Text.Json;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ToolCallParsers;

/// <summary>
/// Issue #771: Qwen3.5 / 3.6 / 3.8 / Ornith answer a tool request in the Qwen3-Coder XML form and the
/// server used to return it as plain content. Golden strings marked REAL were captured verbatim from the
/// models by scripts/harness-smoke.ps1 (Ornith-1.5-9B and Qwen3.6-35B-A3B, dev 0.3.0-dev.2493, Vulkan).
/// </summary>
public class QwenXmlToolCallParserTests
{
    private readonly QwenXmlToolCallParser _parser = new();

    // REAL: the generation is cut at the "</tool_call>" stop sequence, and the reasoning close leads it.
    private const string RealCapture =
        "</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n";

    private static ToolDefinition Tool(string name, string propsJson, string required = "[]") =>
        new(name, "d", $$"""{"type":"object","properties":{{propsJson}},"required":{{required}}}""");

    private static JsonElement Args(ToolCall c) => JsonDocument.Parse(c.Arguments).RootElement;

    [Fact]
    public void RealCapture_WithoutCloseTag_ParsesAndIgnoresLeadingThinkClose()
    {
        var calls = _parser.TryParse(RealCapture);

        var call = Assert.Single(calls!);
        Assert.Equal("call_0", call.Id);
        Assert.Equal("get_weather", call.FunctionName);
        Assert.Equal("""{"city":"Paris"}""", call.Arguments);
    }

    [Fact]
    public void WithCloseTag_Parses()
    {
        var calls = _parser.TryParse(RealCapture + "</tool_call>");
        Assert.Equal("""{"city":"Paris"}""", Assert.Single(calls!).Arguments);
    }

    [Fact]
    public void ProseBeforeCall_IsIgnored()
    {
        var calls = _parser.TryParse("I'll check that for you.\n\n" + RealCapture + "</tool_call>");
        Assert.Equal("get_weather", Assert.Single(calls!).FunctionName);
    }

    [Fact]
    public void ParallelCalls_AreAllReturnedWithSequentialIds()
    {
        const string text =
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n" +
            "<tool_call>\n<function=get_time>\n<parameter=tz>\nCET\n</parameter>\n</function>\n</tool_call>";

        var calls = _parser.TryParse(text)!;

        Assert.Equal(2, calls.Length);
        Assert.Equal(("call_0", "get_weather"), (calls[0].Id, calls[0].FunctionName));
        Assert.Equal(("call_1", "get_time"), (calls[1].Id, calls[1].FunctionName));
        Assert.Equal("""{"tz":"CET"}""", calls[1].Arguments);
    }

    [Fact]
    public void ParallelCalls_FirstWithoutCloseTag_DoesNotSwallowSecond()
    {
        const string text =
            "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n" +
            "<tool_call>\n<function=b>\n</function>\n</tool_call>";

        var calls = _parser.TryParse(text)!;

        Assert.Equal(["a", "b"], calls.Select(c => c.FunctionName));
    }

    [Fact]
    public void NoParameters_YieldsEmptyObject()
    {
        var calls = _parser.TryParse("<tool_call>\n<function=ping>\n</function>\n</tool_call>")!;
        Assert.Equal("{}", Assert.Single(calls).Arguments);
    }

    [Fact]
    public void BareFunctionBlock_WithoutToolCallWrapper_Parses()
    {
        // Qwen3-Coder occasionally omits <tool_call> (llama.cpp handles it too).
        var calls = _parser.TryParse("<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n</function>");
        Assert.Equal("""{"city":"Rome"}""", Assert.Single(calls!).Arguments);
    }

    [Fact]
    public void HermesJsonInsideToolCall_FallsBackToJson()
    {
        var calls = _parser.TryParse("""<tool_call>{"name":"get_weather","arguments":{"city":"Oslo"}}</tool_call>""")!;
        var call = Assert.Single(calls);
        Assert.Equal("get_weather", call.FunctionName);
        Assert.Equal("Oslo", Args(call).GetProperty("city").GetString());
    }

    [Fact]
    public void MultilineStringValue_KeepsInnerNewlinesAndIndentation()
    {
        var tools = new[] { Tool("write", """{"body":{"type":"string"}}""") };
        var calls = _parser.TryParse(
            "<tool_call>\n<function=write>\n<parameter=body>\nline1\n  line2\n\nline4\n</parameter>\n</function>\n</tool_call>", tools)!;

        Assert.Equal("line1\n  line2\n\nline4", Args(calls[0]).GetProperty("body").GetString());
    }

    [Fact]
    public void ValueWithoutTemplateNewlines_IsTakenAsWritten()
    {
        var calls = _parser.TryParse("<tool_call><function=f><parameter=k>v</parameter></function></tool_call>")!;
        Assert.Equal("""{"k":"v"}""", calls[0].Arguments);
    }

    [Fact]
    public void ValueContainingAngleBrackets_IsNotMistakenForMarkup()
    {
        var tools = new[] { Tool("write", """{"code":{"type":"string"}}""") };
        var calls = _parser.TryParse(
            "<tool_call>\n<function=write>\n<parameter=code>\n#include <functional>\nif (a < b) {}\n</parameter>\n</function>\n</tool_call>", tools)!;

        Assert.Equal("#include <functional>\nif (a < b) {}", Args(calls[0]).GetProperty("code").GetString());
    }

    [Fact]
    public void Types_FollowTheSchema()
    {
        var tools = new[]
        {
            Tool("f", """
                {"s":{"type":"string"},"i":{"type":"integer"},"n":{"type":"number"},"b":{"type":"boolean"},
                 "o":{"type":"object"},"a":{"type":"array","items":{"type":"integer"}},"opt":{"type":["integer","null"]}}
                """),
        };
        const string text = """
            <tool_call>
            <function=f>
            <parameter=s>
            123
            </parameter>
            <parameter=i>
            42
            </parameter>
            <parameter=n>
            3.5
            </parameter>
            <parameter=b>
            true
            </parameter>
            <parameter=o>
            {"k": [1, 2]}
            </parameter>
            <parameter=a>
            [1, 2, 3]
            </parameter>
            <parameter=opt>
            null
            </parameter>
            </function>
            </tool_call>
            """;

        var a = Args(_parser.TryParse(text, tools)!.Single());

        Assert.Equal(JsonValueKind.String, a.GetProperty("s").ValueKind);   // string stays string even if numeric-looking
        Assert.Equal("123", a.GetProperty("s").GetString());
        Assert.Equal(42, a.GetProperty("i").GetInt32());
        Assert.Equal(3.5, a.GetProperty("n").GetDouble());
        Assert.True(a.GetProperty("b").GetBoolean());
        Assert.Equal(2, a.GetProperty("o").GetProperty("k")[1].GetInt32());
        Assert.Equal(3, a.GetProperty("a").GetArrayLength());
        Assert.Equal(JsonValueKind.Null, a.GetProperty("opt").ValueKind);
    }

    [Fact]
    public void NonStringParameter_WithUnparseableText_FallsBackToString()
    {
        var tools = new[] { Tool("f", """{"i":{"type":"integer"}}""") };
        var calls = _parser.TryParse("<tool_call><function=f><parameter=i>many</parameter></function></tool_call>", tools)!;
        Assert.Equal("many", Args(calls[0]).GetProperty("i").GetString());
    }

    [Fact]
    public void WithoutSchema_ValuesAreJsonWhenValidElseStrings()
    {
        var calls = _parser.TryParse(
            "<tool_call><function=f><parameter=n>7</parameter><parameter=s>Paris</parameter><parameter=b>false</parameter></function></tool_call>")!;

        var a = Args(calls[0]);
        Assert.Equal(7, a.GetProperty("n").GetInt32());
        Assert.Equal("Paris", a.GetProperty("s").GetString());
        Assert.False(a.GetProperty("b").GetBoolean());
    }

    [Fact]
    public void UnknownTool_StillParses_WithHeuristicTypes()
    {
        var tools = new[] { Tool("other", """{"x":{"type":"string"}}""") };
        var calls = _parser.TryParse("<tool_call><function=f><parameter=n>7</parameter></function></tool_call>", tools)!;
        Assert.Equal(7, Args(calls[0]).GetProperty("n").GetInt32());
    }

    // ---- malformed / partial -------------------------------------------------

    [Theory]
    [InlineData("")]
    [InlineData("The weather in Paris is sunny.")]
    [InlineData("</think>\n\nNo tool needed.")]
    [InlineData("<tool_call>")]
    [InlineData("<tool_call>\n")]
    [InlineData("<tool_call>\n<function=")]
    [InlineData("<tool_call>\n<function=get_weather")]
    [InlineData("<tool_call>\n<function=get_weather>\n")]
    [InlineData("<tool_call>\n<function=get_weather>\n<parameter=city>\nPar")]
    [InlineData("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n")]
    [InlineData("<tool_call>\n<function=>\n</function>\n</tool_call>")]
    [InlineData("<tool_call>{not json}</tool_call>")]
    [InlineData("#include <functional>")]
    public void MalformedOrTruncated_ReturnsNull_AndNeverThrows(string text)
    {
        Assert.Null(_parser.TryParse(text));
    }

    [Fact]
    public void EveryPrefixOfRealCapture_NeverThrows_AndOnlyCompletePrefixParses()
    {
        string full = RealCapture;
        int completeAt = full.IndexOf("</function>", StringComparison.Ordinal) + "</function>".Length;
        for (int i = 0; i <= full.Length; i++)
        {
            var calls = _parser.TryParse(full[..i]);
            if (i < completeAt)
                Assert.Null(calls);
            else
                Assert.Equal("Paris", Args(Assert.Single(calls!)).GetProperty("city").GetString());
        }
    }

    [Fact]
    public void TruncatedSecondCall_ReturnsOnlyTheCompleteFirst()
    {
        const string text =
            "<tool_call>\n<function=a>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n<parameter=x>\n1";
        Assert.Equal(["a"], _parser.TryParse(text)!.Select(c => c.FunctionName));
    }

    [Fact]
    public void BareJson_WithoutMarkers_IsNotAToolCall()
    {
        Assert.Null(_parser.TryParse("""{"name":"get_weather","arguments":{"city":"Paris"}}"""));
    }

    // ---- streaming detection -------------------------------------------------

    [Theory]
    [InlineData("<tool_call>", true)]
    [InlineData("</think>\n\n<tool_call>\n<function=get_w", true)]
    [InlineData("<function=get_weather>", true)]
    [InlineData("Sure! ", false)]
    [InlineData("<tool", false)]
    public void IsToolCallStart(string text, bool expected)
        => Assert.Equal(expected, _parser.IsToolCallStart(text));

    [Fact]
    public void ThroughTheInterface_NoSchemaOverload_Works()
    {
        IToolCallParser p = _parser;
        Assert.NotNull(p.TryParse(RealCapture, null));
    }
}
