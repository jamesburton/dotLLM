using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

/// <summary>
/// Liquid AI LFM3 chat template (#1560): <c>LiquidAI/LFM3-1.7B</c> chat_template.jinja verbatim, rendered through the production
/// <see cref="JinjaChatTemplate"/> normalisation and compared with reference jinja2 output
/// (<c>Fixtures/generate_lfm3_render_cases.py</c>). Covers the system/tools preamble (<c>tojson</c>), the <c>&lt;think&gt;</c> stripping rule
/// for earlier assistant turns, <c>enable_thinking</c> / <c>preserve_thinking</c>, XML tool calls and tool responses, and image placeholders.
/// </summary>
public class Lfm3ChatTemplateTests
{
    private const string Template = "lfm3-chat-template.jinja";
    private const string Cases = "lfm3-render-cases.json";

    public static IEnumerable<object[]> CaseNames() => ReferenceRenderCases.Names(Cases);

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Render_MatchesReference(string caseName)
        => ReferenceRenderCases.AssertMatches(Template, Cases, caseName, viaChatTemplate: true);

    [Fact]
    public void ToolCallParser_ForLfm3_ParsesTheXmlFormTheTemplateRenders()
    {
        var parser = ToolCallParserFactory.Create(Architecture.Lfm3, ReferenceRenderCases.Fixture(Template));
        var calls = parser.TryParse(
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>");
        var call = Assert.Single(calls!);
        Assert.Equal("get_weather", call.FunctionName);
        Assert.Contains("\"city\"", call.Arguments, StringComparison.Ordinal);
        Assert.Contains("Paris", call.Arguments, StringComparison.Ordinal);
        Assert.Contains("\"days\"", call.Arguments, StringComparison.Ordinal);
    }

    [SkippableFact]
    public void Fixture_IsTheTemplateEmbeddedInTheOfficialGguf()
    {
        string? path = TestFixtureResolver.ResolveFile("DOTLLM_LFM3_GGUF", "LiquidAI", "LFM3-1.7B-GGUF", "LFM3-1.7B-Q8_0.gguf", "LFM3-1.7B-Q4_K_M.gguf").Path;
        Skip.If(path is null, "LFM3 GGUF not found: set DOTLLM_LFM3_GGUF or `hf download LiquidAI/LFM3-1.7B-GGUF LFM3-1.7B-Q8_0.gguf`.");
        using var gguf = GgufFile.Open(path!);
        string embedded = gguf.Metadata.GetString("tokenizer.chat_template");
        Assert.Equal(ReferenceRenderCases.Fixture(Template).ReplaceLineEndings("\n").TrimEnd('\n'), embedded.ReplaceLineEndings("\n").TrimEnd('\n'));
    }
}
