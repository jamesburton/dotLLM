using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using DotLLM.Tokenizers.Reasoning;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

/// <summary>
/// GLM-4.5-Air (<c>glm4moe</c>) chat behaviour (#1604): the server-side path (<see cref="ChatMessage"/> →
/// <see cref="JinjaChatTemplate"/>, which builds the Jinja context itself) must produce the transformers goldens that
/// <see cref="JinjaGlm45AirReferenceRenderTests"/> pins at the raw-context level; the envelope the template renders for a history
/// tool call must parse back to the same arguments; and the reasoning / tool-call split of the model's wire format.
/// </summary>
public class Glm45AirChatBehaviourTests
{
    private const string Cases = "glm-4.5-air-render-cases.json";
    private const string GgufTemplate = "glm-4.5-air-gguf-embedded-chat-template.jinja";

    private static readonly ToolDefinition Weather = new("get_weather", "Météo ☃ lookup",
        """{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}""");
    private static readonly ToolDefinition Add = new("add", "Add numbers",
        """{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}""");

    private static ChatMessage U(string c) => new() { Role = "user", Content = c };
    private static ChatMessage A(string c, string? reasoning = null, params ToolCall[] calls)
        => new() { Role = "assistant", Content = c, ReasoningContent = reasoning, ToolCalls = calls.Length == 0 ? null : calls };
    private static ChatMessage T(string c) => new() { Role = "tool", Content = c };

    private static string Golden(string name)
    {
        using var doc = JsonDocument.Parse(ReferenceRenderCases.Fixture(Cases));
        return doc.RootElement.EnumerateArray().Single(e => e.GetProperty("name").GetString() == name).GetProperty("expected").GetString()!;
    }

    private static string Render(ChatMessage[] msgs, ChatTemplateOptions? o = null)
        => new JinjaChatTemplate(ReferenceRenderCases.Fixture(GgufTemplate), "", "").Apply(msgs, o ?? new ChatTemplateOptions());

    public static IEnumerable<object[]> ServerCases() =>
    [
        ["user_only", new ChatMessage[] { U("What is 2 + 2?") }, new ChatTemplateOptions()],
        ["system_and_user", new[] { new ChatMessage { Role = "system", Content = "You are terse." }, U("Hello") }, new ChatTemplateOptions()],
        ["multi_turn", new[] { U("Hi"), A("Hello!"), U("Again"), A("Welcome back."), U("Bye") }, new ChatTemplateOptions()],
        ["enable_thinking_false", new[] { U("What is 2 + 2?") }, new ChatTemplateOptions { EnableThinking = false }],
        ["enable_thinking_true", new[] { U("What is 2 + 2?") }, new ChatTemplateOptions { EnableThinking = true }],
        ["reasoning_content_in_history", new[] { U("Hi"), A("Hello!", "greet the user"), U("Again") }, new ChatTemplateOptions()],
        ["tools_declared", new[] { U("Weather in Paris?") }, new ChatTemplateOptions { Tools = [Weather] }],
        ["tool_call_roundtrip",
            new[] { U("Weather in Paris?"), A("", null, new ToolCall("c0", "get_weather", """{"city":"Paris","days":3}""")), T("sunny"), U("thanks") },
            new ChatTemplateOptions { Tools = [Weather] }],
        ["parallel_tool_calls_and_results",
            new[]
            {
                U("Weather in Paris and Rome?"),
                A("Checking both.", "two lookups", new ToolCall("c0", "get_weather", """{"city":"Paris"}"""), new ToolCall("c1", "get_weather", """{"city":"Rome"}""")),
                T("sunny"), T("rainy"),
            },
            new ChatTemplateOptions { Tools = [Weather] }],
        ["tool_result_then_answer_thinking_off",
            new[] { U("What is 1+2?"), A("", null, new ToolCall("c0", "add", """{"a":1,"b":2}""")), T("3") },
            new ChatTemplateOptions { Tools = [Add], EnableThinking = false }],
    ];

    [Theory]
    [MemberData(nameof(ServerCases))]
    public void ServerPath_RendersTheTransformersGolden(string name, ChatMessage[] msgs, ChatTemplateOptions opts)
        => Assert.Equal(Golden(name), Render(msgs, opts));

    [Fact]
    public void GenerationPrompt_EndsWithAssistantMarker_AndOnlyNothinkClosesTheThinkBlock()
    {
        Assert.EndsWith("<|assistant|>", Render([U("x")]), StringComparison.Ordinal);
        Assert.EndsWith("<|assistant|>\n<think></think>", Render([U("x")], new ChatTemplateOptions { EnableThinking = false }), StringComparison.Ordinal);
        Assert.StartsWith("[gMASK]<sop>", Render([U("x")]), StringComparison.Ordinal);
    }

    [Fact]
    public void ToolCallParser_IsGlm_ForTheGgufEmbeddedTemplate()
        => Assert.IsType<GlmToolCallParser>(ToolCallParserFactory.Create(Architecture.Glm4Moe, ReferenceRenderCases.Fixture(GgufTemplate)));

    [Fact]
    public void TemplateRenderedToolCall_ParsesBackToTheSameTypedArguments()
    {
        // The envelope the template writes into history is the envelope the model is trained to emit: parse it back.
        string prompt = Render(
        [
            U("go"),
            A("", null, new ToolCall("c0", "get_weather", """{"city":"Paris","days":3}"""), new ToolCall("c1", "add", """{"a":1.5,"b":2}""")),
        ], new ChatTemplateOptions { Tools = [Weather, Add], AddGenerationPrompt = false });
        string assistant = prompt[(prompt.IndexOf("<|assistant|>", StringComparison.Ordinal) + "<|assistant|>".Length)..];

        var calls = new GlmToolCallParser().TryParse(assistant, [Weather, Add]);
        Assert.NotNull(calls);
        Assert.Equal(2, calls.Length);
        Assert.Equal("get_weather", calls[0].FunctionName);
        using (var d = JsonDocument.Parse(calls[0].Arguments))
        {
            Assert.Equal("Paris", d.RootElement.GetProperty("city").GetString());
            Assert.Equal(3, d.RootElement.GetProperty("days").GetInt32());
        }
        Assert.Equal("add", calls[1].FunctionName);
        using (var d = JsonDocument.Parse(calls[1].Arguments))
        {
            Assert.Equal(1.5, d.RootElement.GetProperty("a").GetDouble());
            Assert.Equal(2, d.RootElement.GetProperty("b").GetDouble());
        }
    }

    [Fact]
    public void ArgumentOrder_KeyThenValue_IsNotSwapped()
    {
        var calls = new GlmToolCallParser().TryParse(
            "<tool_call>get_weather\n<arg_value>Paris</arg_value>\n<arg_key>city</arg_key>\n</tool_call>", [Weather]);
        // Value-before-key is not the envelope: it must not yield city=Paris.
        Assert.True(calls is null || !calls[0].Arguments.Contains("\"city\":\"Paris\"", StringComparison.Ordinal));
    }

    [Fact]
    public void WireFormat_ThinkThenToolCall_SplitsReasoningAndParsesTheCall()
    {
        // What GLM-4.5-Air emits after the bare `<|assistant|>` prompt, ended by <|observation|> (eom, not in the text).
        const string wire = "\n<think>The user wants Paris weather.</think>\n<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n<arg_key>days</arg_key>\n<arg_value>2</arg_value>\n</tool_call>";
        var (reasoning, content, state) = ReasoningSplitter.Split(wire, startInReasoning: false);
        Assert.Equal("The user wants Paris weather.", reasoning);
        Assert.True(state.SawReasoning);
        Assert.StartsWith("<tool_call>get_weather", content, StringComparison.Ordinal);

        var call = Assert.Single(new GlmToolCallParser().TryParse(content, [Weather])!);
        Assert.Equal("""{"city":"Paris","days":2}""", call.Arguments);
    }

    [Fact]
    public void WireFormat_NoThinkMode_IsPlainContent()
    {
        // After `<|assistant|>\n<think></think>` the model answers directly.
        var (reasoning, content, state) = ReasoningSplitter.Split("\nThe answer is 4.", startInReasoning: false);
        Assert.Equal("", reasoning);
        Assert.False(state.SawReasoning);
        Assert.Equal("\nThe answer is 4.", content);
    }

    [Fact]
    public void Streaming_ThinkTagSplitAcrossTokens_IsInvisible()
    {
        var s = new ReasoningSplitter(startInReasoning: false);
        string r = "", c = "";
        foreach (string piece in new[] { "\n", "<th", "ink>plan", " it</", "think>", "\nAnswer" })
        {
            var ch = s.Feed(piece);
            r += ch.Reasoning;
            c += ch.Content;
        }
        var fin = s.Finish();
        r += fin.Reasoning;
        c += fin.Content;
        Assert.Equal("plan it", r);
        Assert.Equal("Answer", c);
    }
}
