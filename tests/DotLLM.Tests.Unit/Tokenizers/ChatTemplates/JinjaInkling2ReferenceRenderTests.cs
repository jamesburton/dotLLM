using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

/// <summary>
/// Renders the Thinking Machines Inkling 2 chat template (<c>thinkingmachines/Inkling-2-Mini chat_template.jinja</c>, byte-identical to the template
/// embedded in <c>ggml-org/Inkling-2-Mini-GGUF</c>) and compares it with transformers' own renderer (<c>Fixtures/generate_inkling2_render_cases.py</c>,
/// #2026). The template reads <c>reasoning_effort</c> (names or numbers, default 0.9, range-checked), declares tools as sorted compact JSON, and
/// raises for unknown roles / audio; every case, including the raised errors, is replayed through <see cref="JinjaChatTemplate"/>'s normalised source,
/// i.e. exactly what the server renders.
/// </summary>
public class JinjaInkling2ReferenceRenderTests
{
    private const string Template = "inkling2-chat-template.jinja";
    private const string Cases = "inkling2-render-cases.json";

    public static IEnumerable<object[]> CaseNames() => ReferenceRenderCases.Names(Cases);

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Render_MatchesTransformers(string caseName) => ReferenceRenderCases.AssertMatches(Template, Cases, caseName, viaChatTemplate: true);

    [Fact]
    public void ReasoningEffort_FromChatTemplateOptions_ReachesTheTemplate_AndThereIsNoSilentFallback()
    {
        var tpl = new JinjaChatTemplate(ReferenceRenderCases.Fixture(Template), "", "");
        string low = tpl.Apply([new ChatMessage { Role = "user", Content = "Hi" }], new ChatTemplateOptions { AddGenerationPrompt = false, ReasoningEffort = "low" });
        Assert.StartsWith("<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>", low, StringComparison.Ordinal);
        string dflt = tpl.Apply([new ChatMessage { Role = "user", Content = "Hi" }], new ChatTemplateOptions { AddGenerationPrompt = false });
        Assert.StartsWith("<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>", dflt, StringComparison.Ordinal);
        // an unknown effort must surface as an error, not as a plain "role: content" transcript
        Assert.ThrowsAny<Exception>(() => tpl.Apply([new ChatMessage { Role = "user", Content = "Hi" }], new ChatTemplateOptions { AddGenerationPrompt = false, ReasoningEffort = "wild" }));
    }
}
