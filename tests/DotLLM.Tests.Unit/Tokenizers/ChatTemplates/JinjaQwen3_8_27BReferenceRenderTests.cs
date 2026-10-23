using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

/// <summary>
/// Renders the real Qwen3.x-style chat template (byte-identical to the one embedded in
/// <c>ggml-org/Qwen3.8-27B-GGUF</c> and <c>prism-ml/Ternary-Bonsai-2-27B-gguf</c>) and compares it
/// against golden output produced by reference Python Jinja2 (see
/// <c>Fixtures/generate_render_cases.py</c>; HF-style sandboxed env, trim_blocks + lstrip_blocks).
/// Covers tool-less chat with reasoning on/off/effort levels, multi-turn, tools and tool calls.
/// </summary>
public class JinjaQwen3_8_27BReferenceRenderTests
{
    private const string Template = "qwen3.8-27b-chat-template.jinja";
    private const string Cases = "qwen3.8-27b-render-cases.json";

    public static IEnumerable<object[]> CaseNames() => ReferenceRenderCases.Names(Cases);

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Render_MatchesReferenceJinja2(string caseName) => ReferenceRenderCases.AssertMatches(Template, Cases, caseName);
}
