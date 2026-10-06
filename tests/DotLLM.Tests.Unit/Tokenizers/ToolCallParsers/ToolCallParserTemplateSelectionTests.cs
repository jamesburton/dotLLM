using DotLLM.Core.Configuration;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ToolCallParsers;

/// <summary>
/// Parser selection (<see cref="ToolCallParserFactory.Create"/>, reached via
/// <c>GgufChatTemplateFactory.CreateToolCallParser</c>) against the REAL chat templates extracted from the
/// GGUFs the harness runs. The Qwen3.5+ template contains the bare <c>&lt;tool_call&gt;</c> the Hermes check keys on,
/// so a regression in the check ORDER silently routes Ornith/Qwen3.6 back to the JSON parser (#771).
/// </summary>
public class ToolCallParserTemplateSelectionTests
{
    private static string Template(string name) => File.ReadAllText(Path.Combine(
        AppContext.BaseDirectory, "Tokenizers", "ChatTemplates", "Fixtures", name));

    [Theory]
    [InlineData("qwen35-ornith-chat-template.jinja", Architecture.Qwen3HybridDense, typeof(QwenXmlToolCallParser))]
    [InlineData("qwen3.8-27b-chat-template.jinja", Architecture.Qwen3HybridDense, typeof(QwenXmlToolCallParser))]
    [InlineData("qwen35-ornith-chat-template.jinja", Architecture.Qwen3MoeHybrid, typeof(QwenXmlToolCallParser))]
    [InlineData("gemma-4-e4b-chat-template.jinja", Architecture.Gemma4, typeof(Gemma4ToolCallParser))]
    [InlineData("llama-3.1-chat-template.jinja", Architecture.Llama, typeof(LlamaToolCallParser))]
    // Llama-3.2's template has no python_tag mention: it must still land on the Llama parser by architecture.
    [InlineData("llama-3.2-chat-template.jinja", Architecture.Llama, typeof(LlamaToolCallParser))]
    // Controls that PASS tool calling today and must keep their parser.
    [InlineData("qwen3-4b-instruct-2507-chat-template.jinja", Architecture.Qwen, typeof(XmlToolCallParser))]
    [InlineData("nemotron-nano-9b-v2-chat-template.jinja", Architecture.NemotronH, typeof(GenericToolCallParser))]
    public void RealTemplate_SelectsFamilyParser(string fixture, Architecture arch, Type expected)
        => Assert.IsType(expected, ToolCallParserFactory.Create(arch, Template(fixture)));

    [Fact]
    public void Gemma4_IsSelected_EvenWithoutTemplate_ByArchitecture()
        => Assert.IsType<Gemma4ToolCallParser>(ToolCallParserFactory.Create(Architecture.Gemma4, null));

    [Theory]
    [InlineData(Architecture.Qwen3HybridDense)]
    [InlineData(Architecture.Qwen3MoeHybrid)]
    public void Qwen35Family_WithoutTemplate_FallsBackToXmlParser(Architecture arch)
        => Assert.IsType<QwenXmlToolCallParser>(ToolCallParserFactory.Create(arch, null));

    [Fact]
    public void QwenHermesTemplate_StillGetsHermesJson_NotTheXmlParser()
    {
        // Qwen3-4B-Instruct-2507's template shows JSON inside <tool_call>: no <function=.
        Assert.IsNotType<QwenXmlToolCallParser>(
            ToolCallParserFactory.Create(Architecture.Qwen, Template("qwen3-4b-instruct-2507-chat-template.jinja")));
    }

    [Fact]
    public void XmlParser_AlsoUnderstandsHermesJson_SoAFormatSwitchingFineTuneStillWorks()
    {
        var calls = new QwenXmlToolCallParser().TryParse("""<tool_call>{"name":"f","arguments":{"a":1}}</tool_call>""");
        Assert.Equal("f", Assert.Single(calls!).FunctionName);
    }

    /// <summary>The marker parsers must reject constrained bare JSON (see ForToolChoice, #325).</summary>
    [Theory]
    [MemberData(nameof(NewParsers))]
    public void NewParsers_RejectBareJson_SoConstrainedOutputGoesThroughGenericParser(IToolCallParser parser)
    {
        const string bare = """{"name": "get_weather", "arguments": {"city": "Tokyo"}}""";
        Assert.Null(parser.TryParse(bare));
        Assert.IsType<GenericToolCallParser>(ToolCallParserFactory.ForToolChoice(new ToolChoice.Required(), parser));
    }

    public static IEnumerable<object[]> NewParsers() =>
    [
        [new QwenXmlToolCallParser()],
        [new Gemma4ToolCallParser()],
    ];
}
