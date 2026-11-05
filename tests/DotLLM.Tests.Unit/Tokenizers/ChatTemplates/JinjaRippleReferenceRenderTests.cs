using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using DotLLM.Tokenizers.Reasoning;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

/// <summary>
/// Renders the ricursiveAI Ripple-65B-Preview chat template (<c>chat_template.jinja</c>, verbatim; embedded byte-identically in the
/// AtomicChat and prometheusAIR GGUFs) and compares it with transformers' own renderer (<c>Fixtures/generate_ripple_render_cases.py</c>,
/// #1643): plain / system / multi-turn with <c>reasoning_content</c> (and <c>clear_thinking</c>) / tools / parallel tool calls with merged
/// tool results. Then checks that the output side (Hermes <c>&lt;tool_call&gt;</c> + <c>&lt;think&gt;</c>) parses with the existing parsers.
/// </summary>
public class JinjaRippleReferenceRenderTests
{
    private const string Template = "ripple-65b-preview-chat-template.jinja";
    private const string Cases = "ripple-render-cases.json";

    public static IEnumerable<object[]> CaseNames() => ReferenceRenderCases.Names(Cases);

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Render_MatchesTransformers(string caseName) => ReferenceRenderCases.AssertMatches(Template, Cases, caseName);

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Render_MatchesTransformers_ThroughJinjaChatTemplate(string caseName)
        => ReferenceRenderCases.AssertMatches(Template, Cases, caseName, viaChatTemplate: true);

    [Fact]
    public void ServerPath_ParallelToolCalls_RendersTheVendorLayout()
    {
        // The same conversation as the parallel_tool_calls_and_results case, built through the public ChatMessage API (what the server
        // renders). Arguments arrive as JSON text and must be re-emitted as an object (tojson), not a quoted string.
        var t = new JinjaChatTemplate(ReferenceRenderCases.Fixture(Template), "<|endoftext|>", "<|im_end|>");
        string s = t.Apply(
        [
            new ChatMessage { Role = "user", Content = "pwd and read a.txt" },
            new ChatMessage
            {
                Role = "assistant", Content = "", ReasoningContent = "Two calls.",
                ToolCalls = [new ToolCall("call_0", "execute_bash", "{\"command\": \"pwd\"}"), new ToolCall("call_1", "read_file", "{\"path\": \"a.txt\", \"limit\": 10}")],
            },
            new ChatMessage { Role = "tool", Content = "/home/u", ToolCallId = "call_0" },
            new ChatMessage { Role = "tool", Content = "hello", ToolCallId = "call_1" },
        ], new ChatTemplateOptions());
        Assert.EndsWith(
            "<|im_start|>assistant\n<think>\nTwo calls.\n</think>\n\n<tool_call>\n{\"name\": \"execute_bash\", \"arguments\": {\"command\": \"pwd\"}}\n</tool_call>"
            + "\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.txt\", \"limit\": 10}}\n</tool_call><|im_end|>\n"
            + "<|im_start|>user\n<tool_response>\n/home/u\n</tool_response>\n<tool_response>\nhello\n</tool_response><|im_end|>\n<|im_start|>assistant\n",
            s);
        Assert.DoesNotContain("<|endoftext|>", s, StringComparison.Ordinal);   // no BOS anywhere in the prompt
        Assert.Equal(ReasoningMarkup.Think, t.ReasoningMarkup);
    }

    [Fact]
    public void ToolCallParser_IsHermes_AndParsesVendorParallelCalls()
    {
        // Template detection keys on "<tool_call>" -> XmlToolCallParser (a thin wrapper over Hermes); with no template the
        // architecture fallback is Hermes itself. Both must parse the vendor layout.
        var parser = ToolCallParserFactory.Create(Architecture.Ricursive, ReferenceRenderCases.Fixture(Template));
        Assert.IsType<XmlToolCallParser>(parser);
        Assert.IsType<HermesToolCallParser>(ToolCallParserFactory.Create(Architecture.Ricursive, null));

        // Exactly what the model is trained to emit (the template's own assistant rendering, after the reasoning block).
        const string output = "<think>\nNeed two calls.\n</think>\n\n<tool_call>\n{\"name\": \"execute_bash\", \"arguments\": {\"command\": \"pwd\"}}\n</tool_call>"
                            + "\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.txt\", \"limit\": 10}}\n</tool_call>";
        var (reasoning, content, _) = ReasoningSplitter.Split(output, startInReasoning: false);
        Assert.Equal("Need two calls.", reasoning);
        var calls = parser.TryParse(content);
        Assert.NotNull(calls);
        Assert.Equal(2, calls!.Length);
        Assert.Equal("execute_bash", calls[0].FunctionName);
        Assert.Contains("\"pwd\"", calls[0].Arguments, StringComparison.Ordinal);
        Assert.Equal("read_file", calls[1].FunctionName);
        Assert.Contains("\"limit\"", calls[1].Arguments, StringComparison.Ordinal);
    }

    [Fact]
    public void Reasoning_GenerationPromptDoesNotOpenThink_ModelOpensItItself()
    {
        // The generation prompt is "<|im_start|>assistant\n" (no <think>): the splitter must start in content mode and still pick up a
        // model-opened block, and leave a no-think answer untouched.
        var (r1, c1, _) = ReasoningSplitter.Split("<think>\nplan\n</think>\nThe answer.", startInReasoning: false);
        Assert.Equal(("plan", "The answer."), (r1, c1));
        var (r2, c2, _) = ReasoningSplitter.Split("Just 4.", startInReasoning: false);
        Assert.Equal(("", "Just 4."), (r2, c2));
    }
}
