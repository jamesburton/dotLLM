using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

/// <summary>
/// End-to-end acceptance test for issue #409: the real, unmodified
/// <c>Qwen/Qwen3.8-27B</c> <c>chat_template.jinja</c> must at least PARSE successfully once the
/// tuple-literal grouping bug is fixed. The fixture is byte-for-byte what HuggingFace serves at
/// https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/chat_template.jinja (fetched 2026-08-14).
/// </summary>
public class JinjaQwen3_8_27BAcceptanceTests
{
    private static string LoadFixture()
    {
        // Copied to the output dir by the csproj (CallerFilePath is path-mapped to "/_/" in CI).
        var path = Path.Combine(AppContext.BaseDirectory, "Tokenizers", "ChatTemplates", "Fixtures", "qwen3.8-27b-chat-template.jinja");
        return File.ReadAllText(path);
    }

    [Fact]
    public void FullTemplate_Parses_WithoutThrowing()
    {
        // This is the exact regression from #409: before the tuple-literal fix, this throws
        // "Line 48, Col 53: Expected RightParen, got Comma" on
        // `resolved_reasoning_effort not in ('xhigh', 'medium', 'low')`, which sits in the
        // reasoning-instructions prelude that executes unconditionally for every render.
        var source = LoadFixture();
        var tokens = new JinjaLexer(source).Tokenize();
        var parser = new JinjaParser(tokens);
        var ast = parser.Parse();
        Assert.NotEmpty(ast.Nodes);
    }

    [Fact]
    public void FullTemplate_ConstructsAsJinjaChatTemplate_WithoutThrowing()
    {
        // JinjaChatTemplate's constructor lexes+parses eagerly; this is the same assertion as
        // above but through the public API callers actually use.
        var source = LoadFixture();
        _ = new JinjaChatTemplate(source, bosToken: "<|endoftext|>", eosToken: "<|im_end|>");
    }

    [Fact]
    public void FullTemplate_Renders_ToolLessChat()
    {
        // Full render (not just parse): needs `is undefined`, loop.previtem/nextitem (#399/#411),
        // macros, namespace, slicing and `not in (tuple)` (#409). Golden comparison against
        // reference Jinja2 lives in JinjaQwen3_8_27BReferenceRenderTests.
        var template = new JinjaChatTemplate(LoadFixture(), bosToken: "<|endoftext|>", eosToken: "<|im_end|>");
        var messages = new[] { new ChatMessage { Role = "user", Content = "What is 2 + 2?" } };
        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        Assert.Contains("<|im_start|>user\nWhat is 2 + 2?<|im_end|>", result, StringComparison.Ordinal);
        Assert.EndsWith("<|im_start|>assistant\n<think>\n", result, StringComparison.Ordinal);
    }
}
