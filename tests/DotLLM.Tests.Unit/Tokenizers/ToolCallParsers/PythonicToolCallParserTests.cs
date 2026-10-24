using System.Text.Json;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ToolCallParsers;

/// <summary>
/// Tests for <see cref="PythonicToolCallParser"/> — SmolLM3's
/// <c>python_tools</c> chat-template branch parser. Coverage includes
/// the canonical positive case (kwargs of mixed types), the canonical
/// negative case (prose-only output), wrapper handling, parallel calls,
/// and the Python-vs-JSON keyword aliasing.
/// </summary>
public sealed class PythonicToolCallParserTests
{
    private readonly PythonicToolCallParser _parser = new();

    [Fact]
    public void TryParse_SimpleKwargs_ReturnsParsedToolCall()
    {
        const string text = "get_weather(city=\"Berlin\", units=\"metric\")";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Single(calls);
        Assert.Equal("get_weather", calls![0].FunctionName);

        using var doc = JsonDocument.Parse(calls[0].Arguments);
        Assert.Equal("Berlin", doc.RootElement.GetProperty("city").GetString());
        Assert.Equal("metric", doc.RootElement.GetProperty("units").GetString());
    }

    [Fact]
    public void TryParse_BareIdentifierNotFollowedByParen_ReturnsNull()
    {
        // Negative: pure prose with no Pythonic call shape.
        const string text =
            "The weather in Berlin is sunny. I won't call any tool right now.";

        var calls = _parser.TryParse(text);

        Assert.Null(calls);
    }

    [Fact]
    public void TryParse_PythonLiterals_ConvertedToJson()
    {
        const string text =
            "configure(threshold=0.75, enabled=True, retries=None, tags=[\"a\", \"b\"])";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Single(calls);
        using var doc = JsonDocument.Parse(calls![0].Arguments);
        var root = doc.RootElement;
        Assert.Equal(0.75, root.GetProperty("threshold").GetDouble(), 3);
        Assert.True(root.GetProperty("enabled").GetBoolean());
        Assert.Equal(JsonValueKind.Null, root.GetProperty("retries").ValueKind);
        var tags = root.GetProperty("tags");
        Assert.Equal(2, tags.GetArrayLength());
        Assert.Equal("a", tags[0].GetString());
        Assert.Equal("b", tags[1].GetString());
    }

    [Fact]
    public void TryParse_WrappedInToolCallTag_StillParses()
    {
        const string text = "<tool_call>search(query=\"dotllm\")</tool_call>";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Single(calls);
        Assert.Equal("search", calls![0].FunctionName);
        using var doc = JsonDocument.Parse(calls[0].Arguments);
        Assert.Equal("dotllm", doc.RootElement.GetProperty("query").GetString());
    }

    [Fact]
    public void TryParse_TwoParallelCalls_BothReturned()
    {
        const string text =
            """
            get_time(timezone="UTC")
            get_weather(city="Tokyo")
            """;

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Equal(2, calls!.Length);
        Assert.Equal("get_time", calls[0].FunctionName);
        Assert.Equal("get_weather", calls[1].FunctionName);
        Assert.Equal("call_0", calls[0].Id);
        Assert.Equal("call_1", calls[1].Id);
    }

    [Fact]
    public void TryParse_SingleQuoteStrings_StillParse()
    {
        // SmolLM3's python_tools branch occasionally emits single-quoted
        // strings (Python idiom). The parser must accept both.
        const string text = "lookup(key='alpha', other='beta')";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Single(calls);
        using var doc = JsonDocument.Parse(calls![0].Arguments);
        Assert.Equal("alpha", doc.RootElement.GetProperty("key").GetString());
        Assert.Equal("beta", doc.RootElement.GetProperty("other").GetString());
    }

    [Fact]
    public void TryParse_NestedDict_RecursivelyConverted()
    {
        const string text = "send(payload={\"id\": 7, \"ok\": True})";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Single(calls);
        using var doc = JsonDocument.Parse(calls![0].Arguments);
        var payload = doc.RootElement.GetProperty("payload");
        Assert.Equal(7, payload.GetProperty("id").GetInt32());
        Assert.True(payload.GetProperty("ok").GetBoolean());
    }

    [Fact]
    public void TryParse_UnterminatedString_ReturnsNull()
    {
        // Defensive: the parser must not produce a tool call when the
        // payload is truncated mid-string (streaming stop-token case).
        const string text = "search(query=\"unterminated";

        var calls = _parser.TryParse(text);

        Assert.Null(calls);
    }

    [Fact]
    public void IsToolCallStart_IdentifierThenParen_True()
    {
        Assert.True(_parser.IsToolCallStart("get_weather("));
    }

    [Fact]
    public void IsToolCallStart_WrapperPresent_True()
    {
        Assert.True(_parser.IsToolCallStart("<tool_call>"));
    }

    [Fact]
    public void IsToolCallStart_PlainWord_False()
    {
        Assert.False(_parser.IsToolCallStart("just a word"));
    }

    // ── Liquid AI LFM2 / LFM2.5 (#1015): <|tool_call_start|>[fn(a='x'), g(b=2)]<|tool_call_end|> ──

    [Fact]
    public void TryParse_Lfm2Wrapper_SingleCall_SingleQuotedString()
    {
        const string text = "<|tool_call_start|>[get_candidate_status(candidate_id='12345')]<|tool_call_end|>Checking.";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Single(calls);
        Assert.Equal("get_candidate_status", calls![0].FunctionName);
        using var doc = JsonDocument.Parse(calls[0].Arguments);
        Assert.Equal("12345", doc.RootElement.GetProperty("candidate_id").GetString());
    }

    [Fact]
    public void TryParse_Lfm2Wrapper_ParallelCalls_AndEscapedQuote()
    {
        const string text = "<|tool_call_start|>[search(query='it\\'s', n=3), now()]<|tool_call_end|>";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Equal(2, calls!.Length);
        Assert.Equal("search", calls[0].FunctionName);
        using var doc = JsonDocument.Parse(calls[0].Arguments);
        Assert.Equal("it's", doc.RootElement.GetProperty("query").GetString());
        Assert.Equal(3, doc.RootElement.GetProperty("n").GetInt32());
        Assert.Equal("now", calls[1].FunctionName);
    }

    [Fact]
    public void TryParse_Lfm2Wrapper_UnterminatedStream_StillParsesOpenList()
    {
        // Stop token (<|im_end|>) can cut the stream before <|tool_call_end|>.
        var calls = _parser.TryParse("<|tool_call_start|>[get_time(zone='UTC')]");

        Assert.NotNull(calls);
        Assert.Equal("get_time", calls![0].FunctionName);
    }

    [Fact]
    public void IsToolCallStart_Lfm2Wrapper_True()
    {
        Assert.True(_parser.IsToolCallStart("<|tool_call_start|>"));
    }

    [Fact]
    public void Factory_Lfm2Template_And_Arch_RouteToPythonic()
    {
        var byTemplate = ToolCallParserFactory.Create(DotLLM.Core.Configuration.Architecture.Llama,
            "{{ bos_token }}...<|tool_call_start|>[...]<|tool_call_end|>");
        Assert.IsType<PythonicToolCallParser>(byTemplate);
        Assert.IsType<PythonicToolCallParser>(ToolCallParserFactory.Create(DotLLM.Core.Configuration.Architecture.Lfm2Moe));
    }
}
