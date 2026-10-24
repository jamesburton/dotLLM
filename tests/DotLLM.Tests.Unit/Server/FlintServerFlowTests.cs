using System.Text;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Sampling;
using DotLLM.Engine;
using DotLLM.Engine.Samplers.StopConditions;
using DotLLM.Models.Gguf;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using DotLLM.Tokenizers.Reasoning;
using DotLLM.Tokenizers.ToolCallParsers;
using DotLLM.Tests.Unit.Tokenizers.ChatTemplates;
using Microsoft.AspNetCore.Http;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// #1173 (Flint F8): the OpenAI chat surface driven through Recursive Flint's own chat template — the <c>developer</c>
/// role, <c>reasoning_content</c> in both directions, GLM-style tool calls, and <c>&lt;|user|&gt;</c> /
/// <c>&lt;|observation|&gt;</c> as end-of-generation tokens.
/// </summary>
public sealed class FlintServerFlowTests
{
    private static readonly JinjaChatTemplate Template =
        new(ReferenceRenderCases.Fixture("flint-chat-template.jinja"), "", "<|endoftext|>");

    private static readonly ToolDefinition[] Tools =
        [new("get_weather", "Get the weather", """{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}""")];

    private static ChatCompletionRequest Parse(string json) =>
        JsonSerializer.Deserialize(json, ServerJsonContext.Default.ChatCompletionRequest)!;

    /// <summary>Renders a request the way <c>ChatCompletionEndpoint</c> does and returns the prompt and its reasoning plan.</summary>
    private static (string Prompt, ReasoningPlan Plan) Render(string json, bool constrained = false)
    {
        var request = Parse(json);
        var options = ReasoningSupport.BuildTemplateOptions(
            RequestConverter.ToTools(request.Tools), request.EnableThinking, request.ReasoningEffort, request.ChatTemplateKwargs, constrained);
        Assert.True(ReasoningSupport.TryApply(Template, RequestConverter.ToMessages(request.Messages), options, out string prompt, out string? error, out _), error);
        return (prompt, ReasoningSupport.Plan(ReasoningFormat.Auto, null, constrained, prompt, out _, Template));
    }

    // ---------------------------------------------------------------- request side

    [Fact]
    public void DeveloperRole_IsRenderedAsTheDeveloperMarker()
    {
        var (prompt, _) = Render("""
            {"messages":[{"role":"system","content":"S"},{"role":"developer","content":"Answer in French."},{"role":"user","content":"hi"}]}
            """);
        Assert.Equal("<|system|>S<|developer|>Answer in French.<|user|>hi<|assistant|>", prompt);
    }

    [Fact]
    public void PriorReasoningContent_IsRenderedInsideThinkTags_AndAToolTurnRoundTrips()
    {
        var (prompt, _) = Render("""
            {"messages":[
              {"role":"user","content":"weather in Paris?"},
              {"role":"assistant","content":"","reasoning_content":"need the tool",
               "tool_calls":[{"id":"call_0","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\",\"days\":3}"}}]},
              {"role":"tool","tool_call_id":"call_0","content":"{\"temp\":21}"}
            ]}
            """);
        Assert.EndsWith(
            "<|user|>weather in Paris?<|assistant|><think>need the tool</think>"
            + "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>"
            + "<|observation|><tool_response>{\"temp\":21}</tool_response><|assistant|>",
            prompt);
    }

    [Fact]
    public void Plan_UsesThinkMarkup_AndNeverAssumesTheTurnOpensInThinking()
    {
        // The generation prompt is a bare <|assistant|>: the model opens <think> itself (or not at all).
        var (_, plan) = Render("""{"messages":[{"role":"user","content":"hi"}]}""");
        Assert.True(plan.Enabled);
        Assert.Equal(ReasoningMarkup.Think, plan.Markup);
        Assert.False(plan.PromptOpened);
    }

    // ---------------------------------------------------------------- response side (non-streaming)

    private static InferenceResponse Response(string text) => new()
    {
        GeneratedTokenIds = [], Text = text, FinishReason = FinishReason.Stop, PromptTokenCount = 5, GeneratedTokenCount = 9,
    };

    private static ChatChoiceDto Build(string text, IToolCallParser? parser = null)
    {
        var (_, plan) = Render("""{"messages":[{"role":"user","content":"hi"}]}""");
        return ChatCompletionEndpoint.BuildChoice(
            Parse("""{"messages":[{"role":"user","content":"hi"}]}"""),
            new InferenceOptions(), parser is null ? null : Tools, parser, plan, null, Response(text), 0, out _);
    }

    private const string ThinkAnswer = "<think>The user greets me.</think>Hello! How can I help?";

    private const string ThinkToolCall =
        "<think>Need the weather for Paris.</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
        + "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>";

    [Fact]
    public void NonStreaming_ThinkBlock_GoesToReasoningContent()
    {
        var c = Build(ThinkAnswer);
        Assert.Equal("The user greets me.", c.Message.ReasoningContent);
        Assert.Equal("Hello! How can I help?", c.Message.Content);
    }

    [Fact]
    public void NonStreaming_NoThinkBlock_IsAllContent()
    {
        var c = Build("Hello!");
        Assert.Null(c.Message.ReasoningContent);
        Assert.Equal("Hello!", c.Message.Content);
    }

    [Fact]
    public void NonStreaming_GlmToolCall_AfterThinking_IsParsedWithTypedArguments()
    {
        var parser = ToolCallParserFactory.Create(Architecture.Flint, ReferenceRenderCases.Fixture("flint-chat-template.jinja"));
        Assert.IsType<GlmToolCallParser>(parser);
        var c = Build(ThinkToolCall, parser);
        Assert.Equal("Need the weather for Paris.", c.Message.ReasoningContent);
        var call = Assert.Single(c.Message.ToolCalls!);
        Assert.Equal("get_weather", call.Function.Name);
        Assert.Equal("{\"city\":\"Paris\",\"days\":3}", call.Function.Arguments);
        Assert.Equal("tool_calls", c.FinishReason);
    }

    [Fact]
    public void ForcedToolChoice_WithGlmParser_ConstrainsToBareJson_ParsedByTheMarkerlessParser()
    {
        // The constraint is format-agnostic: it emits {"name":...,"arguments":{...}}, not Flint's <arg_key> envelope.
        var options = new InferenceOptions { StopSequences = ["</tool_call>"] };
        var parser = ToolChoiceBinder.Apply(new ToolChoice.Required(), Tools, new GlmToolCallParser(), ref options, out bool forced);
        Assert.True(forced);
        Assert.IsType<ResponseFormat.JsonSchema>(options.ResponseFormat);
        Assert.IsType<GenericToolCallParser>(parser);

        // Thinking is off for a constrained request: the plan does not split, the JSON is the whole answer.
        var (_, plan) = Render("""{"messages":[{"role":"user","content":"hi"}]}""", constrained: true);
        Assert.False(plan.Enabled);
        var c = ChatCompletionEndpoint.BuildChoice(
            Parse("""{"messages":[{"role":"user","content":"hi"}]}"""), options, Tools, parser, plan, null,
            Response("""{"name":"get_weather","arguments":{"city":"Paris"}}"""), 0, out _);
        var call = Assert.Single(c.Message.ToolCalls!);
        Assert.Equal("get_weather", call.Function.Name);
        Assert.Contains("Paris", call.Function.Arguments, StringComparison.Ordinal);
    }

    // ---------------------------------------------------------------- streaming

    private static async IAsyncEnumerable<GenerationToken> Tokens(string[] pieces)
    {
        for (int i = 0; i < pieces.Length; i++)
        {
            await Task.Yield();
            yield return new GenerationToken(i, pieces[i], i == pieces.Length - 1 ? FinishReason.Stop : null);
        }
    }

    private static async Task<(string Reasoning, string Content, JsonElement? ToolCalls, string? Finish)> StreamAsync(string[] pieces, IToolCallParser? parser)
    {
        var (_, plan) = Render("""{"messages":[{"role":"user","content":"hi"}]}""");
        var ctx = new DefaultHttpContext();
        var body = new MemoryStream();
        ctx.Response.Body = body;
        await ChatCompletionEndpoint.WriteChatStreamAsync(
            Parse("""{"messages":[{"role":"user","content":"hi"}],"stream":true}"""),
            ctx, _ => Tokens(pieces), (work, _) => work(), "chatcmpl-test", "m",
            parser is null ? null : Tools, parser, plan, CancellationToken.None);

        var r = new StringBuilder();
        var c = new StringBuilder();
        JsonElement? calls = null;
        string? finish = null;
        foreach (string block in Encoding.UTF8.GetString(body.ToArray()).Split("\n\n", StringSplitOptions.RemoveEmptyEntries))
        {
            string data = block["data: ".Length..];
            if (data == "[DONE]") continue;
            foreach (var choice in JsonDocument.Parse(data).RootElement.Clone().GetProperty("choices").EnumerateArray())
            {
                var d = choice.GetProperty("delta");
                if (d.TryGetProperty("reasoning_content", out var rc)) r.Append(rc.GetString());
                if (d.TryGetProperty("content", out var cc)) c.Append(cc.GetString());
                if (d.TryGetProperty("tool_calls", out var tc)) calls = tc.Clone();
                if (choice.TryGetProperty("finish_reason", out var fr) && fr.ValueKind == JsonValueKind.String) finish = fr.GetString();
            }
        }
        return (r.ToString(), c.ToString(), calls, finish);
    }

    /// <summary>Flint-like tokenisation: tags are single tokens, text is small pieces.</summary>
    private static string[] Pieces(string s)
    {
        var parts = new List<string>();
        int i = 0;
        while (i < s.Length)
        {
            int end = s[i] == '<' ? s.IndexOf('>', i) : -1;
            if (end > 0)
            {
                parts.Add(s.Substring(i, end + 1 - i));
                i = end + 1;
                continue;
            }
            int j = i + 1;
            while (j < s.Length && s[j] != '<' && j - i < 3)
                j++;
            parts.Add(s.Substring(i, j - i));
            i = j;
        }
        return [.. parts];
    }

    [Fact]
    public async Task Streaming_ThinkBlock_MatchesNonStreaming()
    {
        var s = await StreamAsync(Pieces(ThinkAnswer), null);
        var c = Build(ThinkAnswer);
        Assert.Equal(c.Message.ReasoningContent, s.Reasoning);
        Assert.Equal(c.Message.Content, s.Content);
        Assert.DoesNotContain("think>", s.Reasoning + s.Content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Streaming_GlmToolCall_ArrivesAsToolCalls_NotContent()
    {
        var s = await StreamAsync(Pieces(ThinkToolCall), new GlmToolCallParser());
        Assert.Equal("Need the weather for Paris.", s.Reasoning);
        Assert.Equal("", s.Content);
        Assert.Equal("tool_calls", s.Finish);
        var fn = s.ToolCalls!.Value[0].GetProperty("function");
        Assert.Equal("get_weather", fn.GetProperty("name").GetString());
        Assert.Equal("{\"city\":\"Paris\",\"days\":3}", fn.GetProperty("arguments").GetString());
    }

    // ---------------------------------------------------------------- end of generation

    [Fact]
    public void GgufTokenizer_FlintPreType_ForcesUserAndObservationAsEndOfGeneration()
    {
        // llama.cpp (llama-vocab.cpp) forces both for pre-type "flint" even when the file lists no eot / eom / eog ids.
        string[] tokens = [.. Enumerable.Range(0, 256).Select(i => ((char)(0x100 + i)).ToString()), "<|endoftext|>", "<|user|>", "<|observation|>", "<|assistant|>"];
        var entries = new Dictionary<string, GgufMetadataValue>
        {
            ["tokenizer.ggml.model"] = new(GgufValueType.String, "gpt2"),
            ["tokenizer.ggml.tokens"] = new(GgufValueType.Array, tokens),
            ["tokenizer.ggml.merges"] = new(GgufValueType.Array, Array.Empty<string>()),
            ["tokenizer.ggml.bos_token_id"] = new(GgufValueType.UInt32, 256u),
            ["tokenizer.ggml.eos_token_id"] = new(GgufValueType.UInt32, 256u),
            ["tokenizer.ggml.pre"] = new(GgufValueType.String, "flint"),
        };
        var tok = GgufBpeTokenizerFactory.Load(new GgufMetadata(entries));

        var stop = EndOfGenerationTokens.CreateStopCondition(tok);
        Assert.Equal(StopResult.Stop, stop.ShouldStop(256, [], ""));   // <|endoftext|>
        Assert.Equal(StopResult.Stop, stop.ShouldStop(257, [], ""));   // <|user|>
        Assert.Equal(StopResult.Stop, stop.ShouldStop(258, [], ""));   // <|observation|>
        Assert.Equal(StopResult.Continue, stop.ShouldStop(259, [], "")); // <|assistant|> is not an end
    }
}
