using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using DotLLM.Tokenizers.Reasoning;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.Reasoning;

/// <summary>#798: markup detection against the REAL chat templates (checked-in template text, no weights).</summary>
public class ReasoningMarkupRealTemplateTests
{
    private static string Fixture(string name)
        => File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Tokenizers", "ChatTemplates", "Fixtures", name));

    [Theory]
    [InlineData("gpt-oss-chat-template.jinja", ReasoningMarkup.Harmony)]
    [InlineData("gemma-4-e4b-chat-template.jinja", ReasoningMarkup.Gemma4Channel)]
    [InlineData("qwen3.8-27b-chat-template.jinja", ReasoningMarkup.Think)]
    [InlineData("qwen35-ornith-chat-template.jinja", ReasoningMarkup.Think)]
    [InlineData("nemotron-nano-9b-v2-chat-template.jinja", ReasoningMarkup.Think)]
    [InlineData("llama-3.1-chat-template.jinja", ReasoningMarkup.Think)]
    [InlineData("llama-3.2-chat-template.jinja", ReasoningMarkup.Think)]
    [InlineData("smollm3-chat-template.jinja", ReasoningMarkup.Think)]
    [InlineData("glm-4.7-flash-chat-template.jinja", ReasoningMarkup.Think)]
    public void RealTemplates_DetectTheirMarkup(string fixture, ReasoningMarkup expected)
    {
        string src = Fixture(fixture);
        Assert.Equal(expected, ReasoningMarkups.Detect(src));
        Assert.Equal(expected, new JinjaChatTemplate(src, "", "").ReasoningMarkup);
    }

    [Fact]
    public void GptOss_RealTemplate_GenerationPromptEndsInTheAssistantStart_AndCarriesTheTools()
    {
        var t = new JinjaChatTemplate(Fixture("gpt-oss-chat-template.jinja"), "", "");
        var tools = new[] { new ToolDefinition("get_weather", "Get weather", """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""") };
        string prompt = t.Apply([new ChatMessage { Role = "user", Content = "weather in Paris?" }], new ChatTemplateOptions { Tools = tools });

        Assert.EndsWith("<|start|>assistant", prompt, StringComparison.Ordinal);
        Assert.Contains("namespace functions", prompt, StringComparison.Ordinal);
        Assert.Contains("type get_weather", prompt, StringComparison.Ordinal);

        // No block is open in the prompt for Harmony: the model emits its own <|channel|>analysis header.
        Assert.False(ReasoningFormats.PromptOpensThinking(prompt, ReasoningMarkup.Harmony.OpenTag(), ReasoningMarkup.Harmony.CloseTag()));
    }

    [Fact]
    public void GptOss_RealTemplate_ToolRoundTrip_RendersTheCallAndResultInHarmonyForm()
    {
        var t = new JinjaChatTemplate(Fixture("gpt-oss-chat-template.jinja"), "", "");
        var msgs = new[]
        {
            new ChatMessage { Role = "user", Content = "weather in Paris?" },
            new ChatMessage { Role = "assistant", Content = "", ToolCalls = [new ToolCall("call_0", "get_weather", "{\"city\":\"Paris\"}")] },
            new ChatMessage { Role = "tool", Content = "{\"temp\":20}", ToolCallId = "call_0" },
        };
        string prompt = t.Apply(msgs, new ChatTemplateOptions());
        Assert.Contains("<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{", prompt, StringComparison.Ordinal);
        Assert.Contains("Paris", prompt, StringComparison.Ordinal);
        Assert.Contains("<|call|>", prompt, StringComparison.Ordinal);
        Assert.Contains("<|start|>functions.get_weather to=assistant<|channel|>commentary<|message|>", prompt, StringComparison.Ordinal);
        Assert.EndsWith("<|start|>assistant", prompt, StringComparison.Ordinal);
    }

    [Fact]
    public void Gemma4_RealTemplate_ThinkingOnAndOff_PromptShapes()
    {
        var t = new JinjaChatTemplate(Fixture("gemma-4-e4b-chat-template.jinja"), "<bos>", "<eos>");
        string on = t.Apply([new ChatMessage { Role = "user", Content = "hi" }], new ChatTemplateOptions { EnableThinking = true });
        Assert.Contains("<|think|>", on, StringComparison.Ordinal);
        Assert.EndsWith("<|turn>model\n", on, StringComparison.Ordinal);
        // The generation prompt does not itself open a channel: the MODEL emits `<|channel>thought`.
        Assert.False(ReasoningFormats.PromptOpensThinking(on, ReasoningMarkups.Gemma4Open, ReasoningMarkups.Gemma4Close));

        // After a tool response with thinking on, the template DOES end inside an open channel.
        var msgs = new[]
        {
            new ChatMessage { Role = "user", Content = "weather?" },
            new ChatMessage { Role = "assistant", Content = "", ToolCalls = [new ToolCall("c0", "get_weather", "{\"city\":\"Paris\"}")] },
            new ChatMessage { Role = "tool", Content = "{\"temp\":20}", ToolCallId = "c0" },
        };
        string afterTool = t.Apply(msgs, new ChatTemplateOptions { EnableThinking = true });
        Assert.True(ReasoningFormats.PromptOpensThinking(afterTool, ReasoningMarkups.Gemma4Open, ReasoningMarkups.Gemma4Close));
    }
}
