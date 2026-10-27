using System.Text.Json;
using DotLLM.Server;
using DotLLM.Server.Models;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>#1106: <c>/props</c> <c>reasoning</c> reports which thinking controls the loaded template reads.</summary>
public sealed class ReasoningCapabilitiesTests
{
    private static JinjaChatTemplate T(string src) => new(src, "", "");

    [Fact]
    public void QwenStyleTemplate_ReportsSwitchEffortAndBlock()
    {
        var c = ReasoningSupport.Capabilities(T(
            "{%- if enable_thinking is undefined or enable_thinking is true %}{{ reasoning_effort|default('xhigh') }}<think>\n{%- endif %}x"));
        Assert.NotNull(c);
        Assert.True(c!.EnableThinking);
        Assert.True(c.ReasoningEffort);
        Assert.True(c.ThinkingBlock);
    }

    [Fact]
    public void PlainTemplate_ReportsNothing()
    {
        var c = ReasoningSupport.Capabilities(T("{% for m in messages %}{{ m.content }}{% endfor %}"));
        Assert.NotNull(c);
        Assert.False(c!.EnableThinking);
        Assert.False(c.ReasoningEffort);
        Assert.False(c.ThinkingBlock);
    }

    [Fact]
    public void NoTemplate_IsNull() => Assert.Null(ReasoningSupport.Capabilities(null));

    [Fact]
    public void Json_UsesSnakeCaseNames()
    {
        var json = JsonSerializer.Serialize(new ReasoningCapabilitiesDto { EnableThinking = true, ThinkingBlock = true },
            ServerJsonContext.Default.ReasoningCapabilitiesDto);
        Assert.Contains("\"enable_thinking\":true", json);
        Assert.Contains("\"reasoning_effort\":false", json);
        Assert.Contains("\"thinking_block\":true", json);
    }
}
