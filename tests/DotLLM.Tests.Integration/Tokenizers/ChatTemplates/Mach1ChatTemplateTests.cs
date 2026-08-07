using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Integration.Tokenizers.ChatTemplates;

/// <summary>
/// Verifies the real <c>SyzygyResearch/Mach-1-Additive-35B</c>
/// <c>chat_template.jinja</c> renders correctly end-to-end through the
/// existing Jinja2-subset interpreter (issue #266 Phase B item 4). The
/// template defines a <c>render_content</c> <c>{% macro %}</c>; macro support
/// itself was implemented and merged separately (issue #273) — this test is
/// verification only, not new interpreter work, per the Phase B scope note.
/// </summary>
/// <remarks>
/// The template also exercises <c>namespace()</c>, <c>raise_exception</c>,
/// negative-step slicing (<c>messages[::-1]</c>), <c>loop.previtem</c> /
/// <c>loop.nextitem</c>, and the <c>|items</c> / <c>|tojson</c> / <c>|trim</c>
/// filters — real-world coverage beyond any existing synthetic Jinja fixture
/// in this repo.
/// </remarks>
public sealed class Mach1ChatTemplateTests
{
    [SkippableFact]
    public void RealChatTemplate_SimpleUserAssistantTurn_RendersExpectedImTags()
    {
        string? root = ResolveFixtureRoot();
        Skip.If(root is null, SkipReason);
        string templatePath = Path.Combine(root!, "chat_template.jinja");
        Skip.If(!File.Exists(templatePath), "chat_template.jinja not staged.");

        string templateSource = File.ReadAllText(templatePath);
        var template = new JinjaChatTemplate(templateSource, bosToken: "", eosToken: "<|im_end|>");

        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "What is 2+2?" },
        };

        string rendered = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        Assert.Contains("<|im_start|>user\nWhat is 2+2?<|im_end|>\n", rendered, StringComparison.Ordinal);
        Assert.Contains("<|im_start|>assistant\n<think>\n", rendered, StringComparison.Ordinal);
    }

    [SkippableFact]
    public void RealChatTemplate_SystemUserAssistantHistory_RendersInOrder()
    {
        string? root = ResolveFixtureRoot();
        Skip.If(root is null, SkipReason);
        string templatePath = Path.Combine(root!, "chat_template.jinja");
        Skip.If(!File.Exists(templatePath), "chat_template.jinja not staged.");

        string templateSource = File.ReadAllText(templatePath);
        var template = new JinjaChatTemplate(templateSource, bosToken: "", eosToken: "<|im_end|>");

        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "You are a helpful assistant." },
            new() { Role = "user", Content = "Hello" },
            new() { Role = "assistant", Content = "Hi there!" },
            new() { Role = "user", Content = "What's the weather?" },
        };

        string rendered = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        int systemIdx = rendered.IndexOf("You are a helpful assistant.", StringComparison.Ordinal);
        int helloIdx = rendered.IndexOf("Hello", StringComparison.Ordinal);
        int hiIdx = rendered.IndexOf("Hi there!", StringComparison.Ordinal);
        int weatherIdx = rendered.IndexOf("What's the weather?", StringComparison.Ordinal);

        Assert.True(systemIdx >= 0 && systemIdx < helloIdx, "system message must render before user turn 1.");
        Assert.True(helloIdx < hiIdx, "user turn 1 must render before assistant reply.");
        Assert.True(hiIdx < weatherIdx, "assistant reply must render before user turn 2.");
        Assert.StartsWith("<|im_start|>system\n", rendered, StringComparison.Ordinal);
    }

    [SkippableFact]
    public void RealChatTemplate_ToolCallRoundTrip_RendersFunctionCallBlock()
    {
        string? root = ResolveFixtureRoot();
        Skip.If(root is null, SkipReason);
        string templatePath = Path.Combine(root!, "chat_template.jinja");
        Skip.If(!File.Exists(templatePath), "chat_template.jinja not staged.");

        string templateSource = File.ReadAllText(templatePath);
        var template = new JinjaChatTemplate(templateSource, bosToken: "", eosToken: "<|im_end|>");

        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "What's the weather in Paris?" },
            new()
            {
                Role = "assistant",
                Content = "",
                ToolCalls =
                [
                    new ToolCall(Id: "call_1", FunctionName: "get_weather", Arguments: """{"location":"Paris"}"""),
                ],
            },
            new() { Role = "tool", Content = "{\"temp_c\": 18}", ToolCallId = "call_1" },
        };

        var tools = new[]
        {
            new ToolDefinition(
                Name: "get_weather",
                Description: "Get current weather for a location.",
                ParametersSchema: """{"type":"object","properties":{"location":{"type":"string"}}}"""),
        };

        string rendered = template.Apply(messages,
            new ChatTemplateOptions { AddGenerationPrompt = true, Tools = tools });

        // Tool block rendered in the system prelude.
        Assert.Contains("# Tools", rendered, StringComparison.Ordinal);
        Assert.Contains("get_weather", rendered, StringComparison.Ordinal);
        // Tool-call round-trips as a <function=...> block inside <tool_call>.
        Assert.Contains("<tool_call>\n<function=get_weather>", rendered, StringComparison.Ordinal);
        Assert.Contains("<parameter=location>", rendered, StringComparison.Ordinal);
        // Tool response round-trips inside <tool_response>.
        Assert.Contains("<tool_response>", rendered, StringComparison.Ordinal);
        Assert.Contains("\"temp_c\": 18", rendered, StringComparison.Ordinal);
    }

    private const string SkipReason =
        "Mach-1-Additive-35B fixture not found. Set DOTLLM_MACH1_35B_DIR or populate " +
        "~/.dotllm/test-cache/SyzygyResearch/Mach-1-Additive-35B/ (see docs/QUANTIZATION.md).";

    private static string? ResolveFixtureRoot()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_MACH1_35B_DIR");
        if (!string.IsNullOrWhiteSpace(env) && Directory.Exists(env))
            return env;

        string conventional = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "test-cache", "SyzygyResearch", "Mach-1-Additive-35B");
        return Directory.Exists(conventional) ? conventional : null;
    }
}
