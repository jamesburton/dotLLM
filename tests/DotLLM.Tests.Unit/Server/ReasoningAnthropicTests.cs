using System.Text;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.Reasoning;
using DotLLM.Tokenizers.ToolCallParsers;
using Microsoft.AspNetCore.Http;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>#767: <c>thinking</c> blocks on the Anthropic Messages surface.</summary>
public sealed class ReasoningAnthropicTests
{
    private static AnthropicMessagesRequest Parse(string json) =>
        JsonSerializer.Deserialize(json, ServerJsonContext.Default.AnthropicMessagesRequest)!;

    private static ReasoningPlan Open() =>
        ReasoningSupport.Plan(ReasoningFormat.Auto, null, false, "<|im_start|>assistant\n<think>\n", out _);

    private readonly record struct Frame(string Event, JsonElement Data);

    private static async IAsyncEnumerable<GenerationToken> Tokens(params string[] pieces)
    {
        for (int i = 0; i < pieces.Length; i++)
        {
            await Task.Yield();
            yield return new GenerationToken(i, pieces[i], i == pieces.Length - 1 ? FinishReason.Stop : null);
        }
    }

    private sealed class MarkerParser : IToolCallParser
    {
        public ToolCall[]? TryParse(string t) =>
            t.Contains("<tool_call>", StringComparison.Ordinal) ? [new ToolCall("c1", "f", "{}")] : null;
        public bool IsToolCallStart(string t) => t.Contains("<tool_call>", StringComparison.Ordinal);
    }

    private static async Task<Frame[]> RunAsync(ReasoningPlan? plan, IToolCallParser? parser, params string[] pieces)
    {
        var ctx = new DefaultHttpContext();
        var body = new MemoryStream();
        ctx.Response.Body = body;
        await MessagesEndpoint.WriteMessageStreamAsync(
            ctx, _ => Tokens(pieces), (work, _) => work(), parser, null,
            "msg_t", "m", promptTokenCount: 3, CancellationToken.None, forcedToolCall: false, plan);

        var frames = new List<Frame>();
        foreach (string block in Encoding.UTF8.GetString(body.ToArray()).Split("\n\n", StringSplitOptions.RemoveEmptyEntries))
        {
            var lines = block.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            frames.Add(new Frame(lines[0]["event: ".Length..], JsonDocument.Parse(lines[1]["data: ".Length..]).RootElement.Clone()));
        }
        return [.. frames];
    }

    // ---------------------------------------------------------------- request side

    [Theory]
    [InlineData("""{"type":"enabled","budget_tokens":2048}""", true)]
    [InlineData("""{"type":"disabled"}""", false)]
    [InlineData("""{"type":"bogus"}""", null)]
    public void ThinkingConfig_MapsToEnableThinking(string thinking, bool? expected)
    {
        var r = Parse($$"""{"max_tokens":8,"messages":[{"role":"user","content":"hi"}],"thinking":{{thinking}}}""");
        Assert.Equal(expected, r.ThinkingEnabled);
    }

    [Fact]
    public void ThinkingConfig_Absent_LeavesTemplateDefault_AndNewFieldsAreNull()
    {
        var r = Parse("""{"max_tokens":8,"messages":[{"role":"user","content":"hi"}]}""");
        Assert.Null(r.ThinkingEnabled);
        Assert.Null(r.ChatTemplateKwargs);
        Assert.Null(r.ReasoningFormat);
    }

    [Fact]
    public void InputThinkingBlocks_BecomeReasoningContent_OnAssistantMessages()
    {
        var r = Parse("""
            {"max_tokens":8,"messages":[
              {"role":"user","content":"hi"},
              {"role":"assistant","content":[
                {"type":"thinking","thinking":"let me see","signature":"x"},
                {"type":"text","text":"Hello!"}]},
              {"role":"user","content":"again"}]}
            """);
        var msgs = AnthropicConverter.ToMessages(r);
        Assert.Equal("let me see", msgs[1].ReasoningContent);
        Assert.Equal("Hello!", msgs[1].Content);
        Assert.Null(msgs[0].ReasoningContent);
        Assert.Null(msgs[2].ReasoningContent);
    }

    [Fact]
    public void InputRedactedThinking_IsStillDropped()
    {
        var r = Parse("""
            {"max_tokens":8,"messages":[{"role":"assistant","content":[
                {"type":"redacted_thinking","data":"abc"},{"type":"text","text":"ok"}]}]}
            """);
        var m = AnthropicConverter.ToMessages(r).Single();
        Assert.Null(m.ReasoningContent);
        Assert.Equal("ok", m.Content);
    }

    // ---------------------------------------------------------------- streaming

    [Fact]
    public async Task Stream_ThinkingBlockThenTextBlock_WithSignature_AndShiftedIndices()
    {
        var f = await RunAsync(Open(), null, "Let me", " think</", "think>", "\n\n", "The answer", " is 4.");

        Assert.Equal(
            ["message_start", "ping",
             "content_block_start", "content_block_delta", "content_block_delta", "content_block_delta", "content_block_stop",
             "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop",
             "message_delta", "message_stop"],
            f.Select(x => x.Event));

        var thinkStart = f[2].Data;
        Assert.Equal(0, thinkStart.GetProperty("index").GetInt32());
        Assert.Equal("thinking", thinkStart.GetProperty("content_block").GetProperty("type").GetString());

        Assert.Equal("thinking_delta", f[3].Data.GetProperty("delta").GetProperty("type").GetString());
        string thinking = f[3].Data.GetProperty("delta").GetProperty("thinking").GetString()
            + f[4].Data.GetProperty("delta").GetProperty("thinking").GetString();
        Assert.Equal("Let me think", thinking);
        Assert.Equal("signature_delta", f[5].Data.GetProperty("delta").GetProperty("type").GetString());
        Assert.Equal(0, f[6].Data.GetProperty("index").GetInt32());

        var textStart = f[7].Data;
        Assert.Equal(1, textStart.GetProperty("index").GetInt32());
        Assert.Equal("text", textStart.GetProperty("content_block").GetProperty("type").GetString());
        Assert.Equal("The answer", f[8].Data.GetProperty("delta").GetProperty("text").GetString());
        Assert.Equal(" is 4.", f[9].Data.GetProperty("delta").GetProperty("text").GetString());
        Assert.Equal(1, f[10].Data.GetProperty("index").GetInt32());

        Assert.Equal("end_turn", f[11].Data.GetProperty("delta").GetProperty("stop_reason").GetString());
        Assert.Equal(6, f[11].Data.GetProperty("usage").GetProperty("output_tokens").GetInt32());
    }

    [Fact]
    public async Task Stream_NoPlan_KeepsTheOriginalEagerEventSequence()
    {
        var f = await RunAsync(null, null, "Hel", "lo");
        Assert.Equal(
            ["message_start", "content_block_start", "ping", "content_block_delta", "content_block_delta",
             "content_block_stop", "message_delta", "message_stop"],
            f.Select(x => x.Event));
    }

    [Fact]
    public async Task Stream_PlanButNoThinking_StillOpensTextBlockAtIndexZero()
    {
        var closedPlan = ReasoningSupport.Plan(ReasoningFormat.Auto, null, false, "<think>\n\n</think>\n\n", out _);
        var f = await RunAsync(closedPlan, null, "Plain ", "answer");
        Assert.DoesNotContain(f, x => x.Data.TryGetProperty("content_block", out var b) && b.GetProperty("type").GetString() == "thinking");
        var start = f.Single(x => x.Event == "content_block_start");
        Assert.Equal(0, start.Data.GetProperty("index").GetInt32());
        Assert.Equal("text", start.Data.GetProperty("content_block").GetProperty("type").GetString());
        Assert.Equal("Plain answer", string.Concat(f.Where(x => x.Event == "content_block_delta")
            .Select(x => x.Data.GetProperty("delta").GetProperty("text").GetString())));
    }

    [Fact]
    public async Task Stream_UnclosedThinkingAtMaxTokens_EndsWithJustTheThinkingBlock()
    {
        var f = await RunAsync(Open(), null, "still ", "thinking");
        Assert.Single(f, x => x.Event == "content_block_start");
        Assert.Equal("thinking", f.Single(x => x.Event == "content_block_start").Data.GetProperty("content_block").GetProperty("type").GetString());
        Assert.Contains(f, x => x.Event == "content_block_stop");
    }

    [Fact]
    public async Task Stream_ToolCallQuotedInThinking_IsNotEmittedAsToolUse()
    {
        var f = await RunAsync(Open(), new MarkerParser(), "maybe <tool_call>", " no", "</think>", "Sunny.");
        Assert.DoesNotContain(f, x => x.Event == "content_block_start"
            && x.Data.GetProperty("content_block").GetProperty("type").GetString() == "tool_use");
        Assert.Equal("end_turn", f.Last(x => x.Event == "message_delta").Data.GetProperty("delta").GetProperty("stop_reason").GetString());
    }

    [Fact]
    public async Task Stream_ToolUseAfterThinking_GetsTheNextIndex()
    {
        var f = await RunAsync(Open(), new MarkerParser(), "need it", "</think>", "<tool_call>x</tool_call>");
        var toolStart = f.Single(x => x.Event == "content_block_start"
            && x.Data.GetProperty("content_block").GetProperty("type").GetString() == "tool_use");
        // thinking = 0; the tool-call markup is withheld from the text block, so the next block is tool_use at 1.
        Assert.Equal(1, toolStart.Data.GetProperty("index").GetInt32());
        Assert.Equal("tool_use", f.Last(x => x.Event == "message_delta").Data.GetProperty("delta").GetProperty("stop_reason").GetString());
    }

    // ---------------------------------------------------------------- DTO shape

    [Fact]
    public void ThinkingBlockDto_SerializesTypeThinkingAndSignature()
    {
        var response = new AnthropicMessageResponse
        {
            Id = "msg_1", Model = "m", StopReason = "end_turn",
            Content = [new AnthropicContentBlockDto { Type = "thinking", Thinking = "hmm", Signature = "" },
                       new AnthropicContentBlockDto { Type = "text", Text = "hi" }],
            Usage = new AnthropicUsageDto { InputTokens = 1, OutputTokens = 2 },
        };
        using var doc = JsonDocument.Parse(JsonSerializer.Serialize(response, ServerJsonContext.Default.AnthropicMessageResponse));
        var block = doc.RootElement.GetProperty("content")[0];
        Assert.Equal("thinking", block.GetProperty("type").GetString());
        Assert.Equal("hmm", block.GetProperty("thinking").GetString());
        Assert.Equal("", block.GetProperty("signature").GetString());
        Assert.False(block.TryGetProperty("text", out _));
        Assert.False(doc.RootElement.GetProperty("content")[1].TryGetProperty("thinking", out _));
    }
}
