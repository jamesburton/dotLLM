using System.Text.Json;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>#767: ollama's <c>think</c> option on <c>/api/chat</c> and <c>/api/generate</c>.</summary>
public sealed class ReasoningOllamaTests
{
    private static (bool? Think, string? Level) Parse(string json)
    {
        using var doc = JsonDocument.Parse(json);
        OllamaApiEndpoint.ParseThink(doc.RootElement, out var think, out var level);
        return (think, level);
    }

    [Theory]
    [InlineData("""{"model":"m"}""", null, null)]
    [InlineData("""{"think":true}""", true, null)]
    [InlineData("""{"think":false}""", false, null)]
    [InlineData("""{"think":"low"}""", true, "low")]
    [InlineData("""{"think":"high"}""", true, "high")]
    [InlineData("""{"think":null}""", null, null)]
    public void Think_MapsToEnableThinkingAndEffort(string json, bool? think, string? level)
        => Assert.Equal((think, level), Parse(json));

    [Fact]
    public void ThinkFalse_RendersAClosedThinkBlock_ThinkTrueOpensOne()
    {
        string src = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Tokenizers", "ChatTemplates", "Fixtures", "qwen3.8-27b-chat-template.jinja"));
        var template = new DotLLM.Tokenizers.ChatTemplates.JinjaChatTemplate(src, "<|endoftext|>", "<|im_end|>");
        DotLLM.Tokenizers.ChatMessage[] msgs = [new() { Role = "user", Content = "q" }];

        var off = ReasoningSupport.BuildTemplateOptions(null, false, null, null, constrained: false);
        Assert.True(DotLLM.Tokenizers.Reasoning.ReasoningFormats.PromptOpensThinking(
            template.Apply(msgs, ReasoningSupport.BuildTemplateOptions(null, true, null, null, false))));
        Assert.False(DotLLM.Tokenizers.Reasoning.ReasoningFormats.PromptOpensThinking(template.Apply(msgs, off)));
    }

    [Fact]
    public void FormatJson_WithoutThink_DefaultsThinkingOff_AsAnyConstrainedRequest()
    {
        Assert.False(ReasoningSupport.BuildTemplateOptions(null, null, null, null, constrained: true).EnableThinking);
        Assert.True(ReasoningSupport.BuildTemplateOptions(null, true, null, null, constrained: true).EnableThinking);
    }
}
