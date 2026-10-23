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
