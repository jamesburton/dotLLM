using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ToolCallParsers;

/// <summary>
/// Tests for <see cref="XmlToolCallParser"/> — the Hermes-compatible
/// <c>&lt;tool_call&gt;{...}&lt;/tool_call&gt;</c> wrapper used by SmolLM3's
/// <c>xml_tools</c> chat-template branch.
/// </summary>
/// <remarks>
/// Bytewise identical to <see cref="HermesToolCallParser"/> at the parse
/// level. These tests cover the SmolLM3-facing surface — the canonical
/// positive and negative cases that the parser factory will dispatch
/// against — without re-running every edge case the Hermes test suite
/// already covers.
/// </remarks>
public sealed class XmlToolCallParserTests
{
    private readonly XmlToolCallParser _parser = new();

    [Fact]
    public void TryParse_SingleSmolLM3StyleCall_ReturnsParsedToolCall()
    {
        const string text =
            """<tool_call>{"name": "get_weather", "arguments": {"city": "Berlin"}}</tool_call>""";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Single(calls);
        Assert.Equal("get_weather", calls![0].FunctionName);
        Assert.Contains("Berlin", calls[0].Arguments);
    }

    [Fact]
    public void TryParse_NoTag_ReturnsNull()
    {
        const string text =
            "Sure, let me look that up for you — but here's the answer without calling any tool.";

        var calls = _parser.TryParse(text);

        Assert.Null(calls);
    }

    [Fact]
    public void IsToolCallStart_TagPresent_True()
    {
        Assert.True(_parser.IsToolCallStart("Before <tool_call>{partial"));
    }

    [Fact]
    public void IsToolCallStart_NoTag_False()
    {
        Assert.False(_parser.IsToolCallStart("plain response"));
    }
}
