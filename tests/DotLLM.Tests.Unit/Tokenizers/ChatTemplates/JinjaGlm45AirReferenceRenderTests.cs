using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

/// <summary>
/// Renders the real GLM-4.5-Air chat template (<c>zai-org/GLM-4.5-Air</c> chat_template.jinja, verbatim) and compares it with
/// transformers' own renderer (<c>Fixtures/generate_glm45_render_cases.py</c>, #1604). Covers system / multi-turn, the
/// generation prompt, <c>enable_thinking</c> true/false and the <c>/nothink</c> suffix, <c>reasoning_content</c> in history
/// (kept only after the last user turn), inline <c>&lt;think&gt;</c> splitting, tool declarations, parallel tool calls and
/// results, string-vs-JSON argument rendering and content-part lists.
/// </summary>
public class JinjaGlm45AirReferenceRenderTests
{
    private const string Template = "glm-4.5-air-chat-template.jinja";
    /// <summary>The template as embedded in the bartowski REAP-82B GGUF (llama.cpp's lightly rewritten <c>/nothink</c> line).</summary>
    private const string GgufTemplate = "glm-4.5-air-gguf-embedded-chat-template.jinja";
    private const string Cases = "glm-4.5-air-render-cases.json";

    public static IEnumerable<object[]> CaseNames() => ReferenceRenderCases.Names(Cases);

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Render_MatchesTransformers(string caseName) => ReferenceRenderCases.AssertMatches(Template, Cases, caseName);

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Render_ViaChatTemplate_MatchesTransformers(string caseName)
        => ReferenceRenderCases.AssertMatches(Template, Cases, caseName, viaChatTemplate: true);

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void GgufEmbeddedTemplate_RendersIdenticallyToHf(string caseName)
        => ReferenceRenderCases.AssertMatches(GgufTemplate, Cases, caseName, viaChatTemplate: true);
}
