using DotLLM.Tokenizers.ChatTemplates;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

/// <summary>
/// Renders the Recursive Flint chat template (<c>RecursiveSI/flint</c> chat_template.jinja, verbatim) and compares it with
/// transformers' own renderer (<c>Fixtures/generate_flint_render_cases.py</c>, #1158). Covers the system / developer / user /
/// assistant / tool roles, reasoning content, GLM-style tool calls, the tool preamble's <c>tojson(ensure_ascii=False)</c>, and the
/// <c>"\x00"</c> strip that a lexer without hex escapes silently turns into a strip of the literal text <c>x00</c>.
/// </summary>
public class JinjaFlintReferenceRenderTests
{
    private const string Template = "flint-chat-template.jinja";
    private const string Cases = "flint-render-cases.json";

    public static IEnumerable<object[]> CaseNames() => ReferenceRenderCases.Names(Cases);

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Render_MatchesTransformers(string caseName) => ReferenceRenderCases.AssertMatches(Template, Cases, caseName);

    [Theory]
    [InlineData("\"a\\x00b\"", "a\0b")]
    [InlineData("\"\\u2603\"", "\u2603")]
    [InlineData("\"\\U0001F600\"", "\U0001F600")]
    [InlineData("'\\x41\\x4a'", "AJ")]
    [InlineData("\"\\xZZ\"", "xZZ")]   // not hex: keeps the historical drop-the-backslash behaviour
    public void Lexer_Decodes_Hex_Escapes(string literal, string expected)
    {
        var tokens = new JinjaLexer("{{ " + literal + " }}").Tokenize();
        Assert.Contains(tokens, t => t.Type == JinjaTokenType.StringLiteral && (string?)t.Value == expected);
    }

    [Fact]
    public void ToolCallParser_For_Flint_Template_Is_Glm_And_Parses_A_Flint_Call()
    {
        var parser = ToolCallParserFactory.Create(Architecture.Flint, ReferenceRenderCases.Fixture(Template));
        Assert.IsType<GlmToolCallParser>(parser);

        // The shape the template itself renders for an assistant tool call (see the tool_call_roundtrip case).
        var calls = parser.TryParse(
            "<think>need weather</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
            + "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>");
        var call = Assert.Single(calls!);
        Assert.Equal("get_weather", call.FunctionName);
        Assert.Contains("\"city\"", call.Arguments, StringComparison.Ordinal);
        Assert.Contains("Paris", call.Arguments, StringComparison.Ordinal);
        Assert.Contains("\"days\"", call.Arguments, StringComparison.Ordinal);
    }
}
