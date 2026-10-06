using System.Text;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Tokenizers.Reasoning;
using DotLLM.Server.Models;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ToolCallParsers;
using Microsoft.AspNetCore.Http;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Wire-level tests for the OpenAI streaming surface (#771/#776): tool-call markup must not leak as
/// <c>delta.content</c>, the final chunk carries <c>tool_calls</c> with <c>finish_reason=tool_calls</c>, and
/// held-back text that turns out NOT to be a call is still delivered. No model is loaded.
/// </summary>
public sealed class OpenAiStreamingToolCallTests
{
    private static readonly ToolDefinition[] Tools =
    [
        new("get_weather", "Get weather", """{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}"""),
    ];

    private static async IAsyncEnumerable<GenerationToken> Script(params string[] pieces)
    {
        for (int i = 0; i < pieces.Length; i++)
        {
            await Task.Yield();
            yield return new GenerationToken(0, pieces[i], i == pieces.Length - 1 ? FinishReason.Stop : null);
        }
    }

    private static Task NoGate(Func<Task> work, CancellationToken ct) => work();

    private sealed record Result(string Content, JsonElement? FinalToolCalls, string? FinishReason, string Raw);

    private static async Task<Result> RunAsync(string[] pieces, IToolCallParser parser, bool forced = false, ToolDefinition[]? tools = null, ReasoningPlan? plan = null)
    {
        var ctx = new DefaultHttpContext();
        var body = new MemoryStream();
        ctx.Response.Body = body;
        var request = new ChatCompletionRequest { Messages = [], Stream = true };

        await ChatCompletionEndpoint.WriteChatStreamAsync(
            request, ctx, _ => Script(pieces), NoGate, "req_1", "m", tools ?? Tools, parser,
            plan ?? ReasoningPlan.Disabled, CancellationToken.None, forced);

        string raw = Encoding.UTF8.GetString(body.ToArray());
        var content = new StringBuilder();
        JsonElement? finalCalls = null;
        string? finish = null;
        foreach (string block in raw.Split("\n\n", StringSplitOptions.RemoveEmptyEntries))
        {
            string data = block["data: ".Length..];
            if (data == "[DONE]") continue;
            var choices = JsonDocument.Parse(data).RootElement.GetProperty("choices");
            if (choices.GetArrayLength() == 0) continue;
            var choice = choices[0];
            if (choice.GetProperty("delta").TryGetProperty("content", out var c) && c.ValueKind == JsonValueKind.String)
                content.Append(c.GetString());
            if (choice.GetProperty("delta").TryGetProperty("tool_calls", out var tc))
                finalCalls = tc.Clone();
            if (choice.TryGetProperty("finish_reason", out var fr) && fr.ValueKind == JsonValueKind.String)
                finish = fr.GetString();
        }
        return new Result(content.ToString(), finalCalls, finish, raw);
    }

    [Fact]
    public async Task QwenXml_CallIsNotLeakedAsContent_AndFinalChunkCarriesToolCalls()
    {
        var r = await RunAsync(
            ["</think>\n\n", "<tool_call>", "\n<function=get_weather>\n", "<parameter=city>\nParis\n</parameter>\n", "<parameter=days>\n3\n</parameter>\n", "</function>\n"],
            new QwenXmlToolCallParser());

        Assert.DoesNotContain("<tool_call>", r.Content);
        Assert.DoesNotContain("<function", r.Content);
        Assert.Equal("tool_calls", r.FinishReason);
        var call = Assert.Single(r.FinalToolCalls!.Value.EnumerateArray().ToArray());
        Assert.Equal("get_weather", call.GetProperty("function").GetProperty("name").GetString());
        // arguments is a JSON string; days is coerced to an integer by the schema.
        var args = JsonDocument.Parse(call.GetProperty("function").GetProperty("arguments").GetString()!).RootElement;
        Assert.Equal("Paris", args.GetProperty("city").GetString());
        Assert.Equal(3, args.GetProperty("days").GetInt32());
    }

    [Fact]
    public async Task Gemma4_CallIsNotLeakedAsContent()
    {
        var r = await RunAsync(
            ["<|tool_call>", "call:get_weather{city:", "<|\"|>Paris<|\"|>", "}<tool_call|>"],
            new Gemma4ToolCallParser());

        Assert.DoesNotContain("tool_call", r.Content);
        Assert.Equal("tool_calls", r.FinishReason);
        var args = JsonDocument.Parse(r.FinalToolCalls!.Value[0].GetProperty("function").GetProperty("arguments").GetString()!).RootElement;
        Assert.Equal("Paris", args.GetProperty("city").GetString());
    }

    [Fact]
    public async Task Llama_PythonTagCall_NotLeaked()
    {
        var r = await RunAsync(
            ["<|python_tag|>", "{\"type\": \"function\", \"function\": \"get_weather\", ", "\"parameters\": {\"city\": \"Paris\"}}"],
            new LlamaToolCallParser());

        Assert.DoesNotContain("python_tag", r.Content);
        Assert.Equal("tool_calls", r.FinishReason);
        Assert.Equal("get_weather", r.FinalToolCalls!.Value[0].GetProperty("function").GetProperty("name").GetString());
    }

    [Fact]
    public async Task ProseBeforeTheCall_StillStreams()
    {
        var r = await RunAsync(
            ["Let me check. ", "<tool_call>", "\n<function=get_weather>\n</function>\n"],
            new QwenXmlToolCallParser());

        Assert.Equal("Let me check. ", r.Content);
        Assert.Equal("tool_calls", r.FinishReason);
    }

    [Fact]
    public async Task PlainAnswer_StreamsUnchanged_WithToolsPresent()
    {
        var r = await RunAsync(["The capital ", "of France is Paris."], new QwenXmlToolCallParser());

        Assert.Equal("The capital of France is Paris.", r.Content);
        Assert.Null(r.FinalToolCalls);
        Assert.Equal("stop", r.FinishReason);
    }

    [Fact]
    public async Task TruncatedCall_IsDeliveredAsContent_NotSwallowed()
    {
        // The model ran out of tokens mid-call: nothing parses, so the held-back text must still reach the client.
        var r = await RunAsync(
            ["<tool_call>", "\n<function=get_weather>\n<parameter=city>\nPar"],
            new QwenXmlToolCallParser());

        Assert.Null(r.FinalToolCalls);
        Assert.Equal("<tool_call>\n<function=get_weather>\n<parameter=city>\nPar", r.Content);
    }

    [Fact]
    public async Task ReasoningThenToolCall_ReasoningStreamsSeparately_ToolMarkupSuppressed_AndOnlyAnswerIsParsed()
    {
        var plan = new ReasoningPlan(ReasoningFormat.Auto, promptOpened: true);
        var r = await RunAsync(
            ["I could call <tool_call> maybe", "</think>\n\n", "<tool_call>",
             "\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n"],
            new QwenXmlToolCallParser(), plan: plan);

        Assert.Equal("tool_calls", r.FinishReason);
        Assert.DoesNotContain("<function", r.Content);
        Assert.DoesNotContain("<tool_call>", r.Content);
        Assert.Contains("reasoning_content", r.Raw, StringComparison.Ordinal);
        Assert.Contains("maybe", r.Raw, StringComparison.Ordinal);
        Assert.Single(r.FinalToolCalls!.Value.EnumerateArray().ToArray());
    }

    [Fact]
    public async Task NoTools_NothingIsSuppressed()
    {
        var r = await RunAsync(["<tool_call>", "x"], new QwenXmlToolCallParser(), tools: []);
        Assert.Equal("<tool_call>x", r.Content);
    }

    [Fact]
    public async Task ForcedToolChoice_SuppressesBareJson_AndReportsTheCall()
    {
        var r = await RunAsync(
            ["{\"name\": \"get_weather\", ", "\"arguments\": {\"city\": \"Paris\"}}"],
            new GenericToolCallParser(), forced: true);

        Assert.Equal("", r.Content);
        Assert.Equal("tool_calls", r.FinishReason);
        Assert.Equal("get_weather", r.FinalToolCalls!.Value[0].GetProperty("function").GetProperty("name").GetString());
    }

    [Fact]
    public async Task GenericModelParser_UnderAuto_DoesNotSwallowJsonishProse()
    {
        var r = await RunAsync(["Use {\"name\": ", "\"x\"} as the shape."], new GenericToolCallParser());
        Assert.Equal("Use {\"name\": \"x\"} as the shape.", r.Content);
    }
}
