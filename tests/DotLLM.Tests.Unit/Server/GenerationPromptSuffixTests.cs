using DotLLM.Server;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// #1093: the server derives the generation-prompt suffix (the text the template appends after the
/// last message when <c>AddGenerationPrompt</c> is on) so the engine can place its recurrent-prefix
/// snapshot just before it.
/// </summary>
public class GenerationPromptSuffixTests
{
    private static string Fixture(string name) =>
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Tokenizers", "ChatTemplates", "Fixtures", name));

    private static readonly ChatMessage[] Messages =
    [
        new ChatMessage { Role = "system", Content = "You are terse." },
        new ChatMessage { Role = "user", Content = "Hello" },
    ];

    private static string? Suffix(JinjaChatTemplate template, ChatTemplateOptions options)
    {
        Assert.True(ReasoningSupport.TryApply(template, Messages, options, out string prompt, out _, out _));
        return ReasoningSupport.GenerationPromptSuffix(template, Messages, options, prompt);
    }

    // ── save point (issue #1654): the prompt up to the start of the last user message ──

    [Fact]
    public void SavePointPrefix_IsTheRenderOfEveryMessageButTheLastUser_AndAPrefixOfThePrompt()
    {
        var template = new JinjaChatTemplate(Fixture("qwen3-4b-instruct-2507-chat-template.jinja"), "<|endoftext|>", "<|im_end|>");
        var options = ReasoningSupport.BuildTemplateOptions(null, null, null, null, constrained: false);
        ChatMessage[] messages =
        [
            new ChatMessage { Role = "system", Content = "You are terse." },
            new ChatMessage { Role = "user", Content = "Hello" },
            new ChatMessage { Role = "assistant", Content = "Hi." },
            new ChatMessage { Role = "user", Content = "Second question" },
        ];
        Assert.True(ReasoningSupport.TryApply(template, messages, options, out string prompt, out _, out _));

        string? savePoint = ReasoningSupport.SavePointPrefix(template, messages, options, prompt);

        Assert.NotNull(savePoint);
        Assert.StartsWith(savePoint, prompt, StringComparison.Ordinal);
        Assert.Contains("Hi.", savePoint, StringComparison.Ordinal);
        Assert.EndsWith("<|im_start|>user\n", savePoint, StringComparison.Ordinal);
        Assert.DoesNotContain("Second question", savePoint, StringComparison.Ordinal);
        Assert.StartsWith("Second question", prompt[savePoint.Length..], StringComparison.Ordinal);
        // A head render (every message but the last) is NOT a prefix on this template: the assistant turn gains a thinking block
        // when it is last. That is why the save point is derived from a marked full render instead.
        string head = template.Apply(messages.Take(3).ToArray(), options with { AddGenerationPrompt = false });
        Assert.False(prompt.StartsWith(head, StringComparison.Ordinal));
    }

    [Fact]
    public void SavePointPrefix_ForAToolResultLast_IsTheStartOfThatToolResult()
    {
        // Agent loops (Claude Code via /v1/messages, OpenAI tool loops via /v1/chat/completions) end with a tool message, not a
        // user one; both endpoints share this helper. PR #1698 review: a user-only rule left them without a save point.
        var template = new JinjaChatTemplate(Fixture("qwen3-4b-instruct-2507-chat-template.jinja"), "<|endoftext|>", "<|im_end|>");
        var options = ReasoningSupport.BuildTemplateOptions(null, null, null, null, constrained: false);
        ChatMessage[] messages =
        [
            new ChatMessage { Role = "system", Content = "You are terse." },
            new ChatMessage { Role = "user", Content = "Read the file." },
            new ChatMessage { Role = "assistant", Content = "", ToolCalls = [new ToolCall("call_1", "read", "{\"path\":\"a.cs\"}")] },
            new ChatMessage { Role = "tool", Content = "namespace A; class B {}", ToolCallId = "call_1" },
        ];
        Assert.True(ReasoningSupport.TryApply(template, messages, options, out string prompt, out _, out _));

        string? savePoint = ReasoningSupport.SavePointPrefix(template, messages, options, prompt);

        Assert.NotNull(savePoint);
        Assert.StartsWith(savePoint, prompt, StringComparison.Ordinal);
        Assert.Contains("Read the file.", savePoint, StringComparison.Ordinal);
        Assert.DoesNotContain("namespace A", savePoint, StringComparison.Ordinal);
        Assert.StartsWith("namespace A; class B {}", prompt[savePoint.Length..], StringComparison.Ordinal);
    }

    [Fact]
    public void SavePointPrefix_IsNull_ForASingleMessage_OrWhenTheLastMessageIsNotTheUser()
    {
        var template = new JinjaChatTemplate(Fixture("qwen3-4b-instruct-2507-chat-template.jinja"), "<|endoftext|>", "<|im_end|>");
        var options = ReasoningSupport.BuildTemplateOptions(null, null, null, null, constrained: false);
        ChatMessage[] single = [new ChatMessage { Role = "user", Content = "Hello" }];
        ChatMessage[] assistantLast = [.. Messages, new ChatMessage { Role = "assistant", Content = "Partial" }];

        Assert.True(ReasoningSupport.TryApply(template, single, options, out string p1, out _, out _));
        Assert.Null(ReasoningSupport.SavePointPrefix(template, single, options, p1));
        Assert.True(ReasoningSupport.TryApply(template, assistantLast, options with { AddGenerationPrompt = false }, out string p2, out _, out _));
        Assert.Null(ReasoningSupport.SavePointPrefix(template, assistantLast, options, p2));
    }

    [Fact]
    public void Suffix_ForQwen35Template_IsTheAssistantHeaderAndThinkOpener()
    {
        var template = new JinjaChatTemplate(Fixture("qwen35-ornith-chat-template.jinja"), "<|endoftext|>", "<|im_end|>");
        var options = ReasoningSupport.BuildTemplateOptions(null, true, null, null, constrained: false);

        string? suffix = Suffix(template, options);

        Assert.NotNull(suffix);
        Assert.StartsWith("<|im_start|>assistant\n", suffix, StringComparison.Ordinal);
        Assert.Contains("<think>", suffix, StringComparison.Ordinal);
    }

    [Fact]
    public void Suffix_ForQwen38Template_WithPreserveThinking_MatchesItsGenerationPrompt()
    {
        var template = new JinjaChatTemplate(Fixture("qwen3.8-27b-chat-template.jinja"), "<|endoftext|>", "<|im_end|>");
        var options = ReasoningSupport.BuildTemplateOptions(null, true, null, null, constrained: false) with
        {
            TemplateKwargs = new Dictionary<string, System.Text.Json.JsonElement>(StringComparer.Ordinal)
            {
                ["preserve_thinking"] = System.Text.Json.JsonDocument.Parse("true").RootElement.Clone(),
            },
        };

        string? suffix = Suffix(template, options);

        string withGen = template.Apply(Messages, options);
        string noGen = template.Apply(Messages, options with { AddGenerationPrompt = false });
        Assert.NotNull(suffix);
        Assert.Equal(withGen[noGen.Length..], suffix);
        Assert.StartsWith("<|im_start|>assistant\n", suffix, StringComparison.Ordinal);
    }

    [Fact]
    public void Suffix_IsNull_WhenTheNoGenerationRenderIsNotAPrefix()
    {
        // The generation-prompt flag changes the body, not just the tail.
        var template = new JinjaChatTemplate(
            "{% for m in messages %}{% if add_generation_prompt %}G{% endif %}{{ m.content }}{% endfor %}", "", "");
        var options = ReasoningSupport.BuildTemplateOptions(null, null, null, null, constrained: false);

        Assert.Null(Suffix(template, options));
    }

    [Fact]
    public void Suffix_IsNull_WhenTheTemplateAddsNothing()
    {
        var template = new JinjaChatTemplate("{% for m in messages %}{{ m.content }}{% endfor %}", "", "");
        var options = ReasoningSupport.BuildTemplateOptions(null, null, null, null, constrained: false);

        Assert.Null(Suffix(template, options));
    }

    [Fact]
    public void Suffix_IsNull_WhenRenderingThrows()
    {
        var template = new JinjaChatTemplate(
            "{% for m in messages %}{{ m.content }}{% endfor %}{% if not add_generation_prompt %}{{ raise_exception('no') }}{% endif %}", "", "");
        var options = ReasoningSupport.BuildTemplateOptions(null, null, null, null, constrained: false);
        string prompt = template.Apply(Messages, options);

        Assert.Null(ReasoningSupport.GenerationPromptSuffix(template, Messages, options, prompt));
    }
}
