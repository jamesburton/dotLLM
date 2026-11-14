using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using DotLLM.Tokenizers.Reasoning;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

/// <summary>
/// DiffuSpace Clover-35B-A3B-Preview's chat template (#1794): a text-only rewrite of Qwen3.6's, embedded byte-identically in the
/// official GGUF. Golden renders come from transformers (<c>Fixtures/generate_clover_render_cases.py</c>). A parse failure here would
/// make the server silently fall back to the plain transcript template, so the template must parse and render exactly.
/// </summary>
public class JinjaCloverReferenceRenderTests
{
    private const string Template = "clover-35b-a3b-chat-template.jinja";
    private const string Cases = "clover-render-cases.json";

    public static IEnumerable<object[]> CaseNames() => ReferenceRenderCases.Names(Cases);

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Render_MatchesTransformers(string caseName) => ReferenceRenderCases.AssertMatches(Template, Cases, caseName);

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Render_MatchesTransformers_ThroughJinjaChatTemplate(string caseName)
        => ReferenceRenderCases.AssertMatches(Template, Cases, caseName, viaChatTemplate: true);

    [Fact]
    public void ServerPath_OpensThink_AndHonoursEnableThinkingFalse()
    {
        var t = new JinjaChatTemplate(ReferenceRenderCases.Fixture(Template), "<|endoftext|>", "<|im_end|>");
        ChatMessage[] messages = [new ChatMessage { Role = "user", Content = "Hi" }];

        Assert.Equal("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n", t.Apply(messages, new ChatTemplateOptions()));
        Assert.Equal("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            t.Apply(messages, new ChatTemplateOptions { EnableThinking = false }));
        Assert.Equal(ReasoningMarkup.Think, t.ReasoningMarkup);
    }

    [Fact]
    public void ToolCallParser_IsTheQwenXmlParser()
    {
        // Clover maps onto Architecture.Qwen3MoeHybrid, whose fallback is the Qwen XML (<function=...>) parser.
        Assert.IsType<QwenXmlToolCallParser>(ToolCallParserFactory.Create(Architecture.Qwen3MoeHybrid, ReferenceRenderCases.Fixture(Template)));
    }
}
