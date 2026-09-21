using System.Text;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Server.Endpoints;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ToolCallParsers;
using Microsoft.AspNetCore.Http;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Endpoint-level tests for the Anthropic streaming SSE surface: they drive
/// <see cref="MessagesEndpoint.WriteMessageStreamAsync"/> with a scripted token stream
/// and assert the emitted event names, ordering and JSON payload shapes. No model is
/// loaded — the token source and the model gate are injected.
/// </summary>
public sealed class AnthropicStreamingTests
{
    /// <summary>One SSE frame: the <c>event:</c> name and its parsed <c>data:</c> payload.</summary>
    private readonly record struct SseFrame(string Event, JsonElement Data);

    private static async IAsyncEnumerable<GenerationToken> Tokens(
        params (string Text, FinishReason? Finish)[] script)
    {
        foreach (var (text, finish) in script)
        {
            await Task.Yield();
            yield return new GenerationToken(0, text, finish);
        }
    }

    /// <summary>Pass-through gate standing in for <c>ServerState.ExecuteAsync</c>.</summary>
    private static Task NoGate(Func<Task> work, CancellationToken ct) => work();

    private sealed class FixedToolCallParser(ToolCall[]? result) : IToolCallParser
    {
        public ToolCall[]? TryParse(string generatedText) => result;
        public bool IsToolCallStart(string text) => false;
    }

    private static async Task<SseFrame[]> RunAsync(
        IAsyncEnumerable<GenerationToken> tokens,
        IToolCallParser? parser = null,
        string[]? stopSequences = null,
        bool suppressToolCallText = false)
    {
        var ctx = new DefaultHttpContext();
        var body = new MemoryStream();
        ctx.Response.Body = body;

        await MessagesEndpoint.WriteMessageStreamAsync(
            ctx, _ => tokens, NoGate, parser, stopSequences,
            messageId: "msg_test", modelId: "test-model", promptTokenCount: 7,
            CancellationToken.None, suppressToolCallText);

        Assert.Equal("text/event-stream", ctx.Response.ContentType);
        // Connection-specific headers are illegal over HTTP/2 and must not be emitted.
        Assert.False(ctx.Response.Headers.ContainsKey("Connection"));

        return ParseSse(Encoding.UTF8.GetString(body.ToArray()));
    }

    private static SseFrame[] ParseSse(string raw)
    {
        var frames = new List<SseFrame>();
        foreach (var block in raw.Split("\n\n", StringSplitOptions.RemoveEmptyEntries))
        {
            var lines = block.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            string name = lines[0]["event: ".Length..];
            string data = lines[1]["data: ".Length..];
            frames.Add(new SseFrame(name, JsonDocument.Parse(data).RootElement.Clone()));
        }
        return [.. frames];
    }

    // --- Text-only stream ---------------------------------------------------

    [Fact]
    public async Task Streaming_TextOnly_EmitsExpectedEventSequence()
    {
        var frames = await RunAsync(Tokens(("Hel", null), ("lo", FinishReason.Stop)));

        Assert.Equal(
            ["message_start", "content_block_start", "ping", "content_block_delta",
             "content_block_delta", "content_block_stop", "message_delta", "message_stop"],
            frames.Select(f => f.Event));
    }

    [Fact]
    public async Task Streaming_TextOnly_EmitsExpectedPayloadShapes()
    {
        var frames = await RunAsync(Tokens(("Hel", null), ("lo", FinishReason.Stop)));

        var start = frames[0].Data;
        Assert.Equal("message_start", start.GetProperty("type").GetString());
        var startMsg = start.GetProperty("message");
        Assert.Equal("msg_test", startMsg.GetProperty("id").GetString());
        Assert.Equal("message", startMsg.GetProperty("type").GetString());
        Assert.Equal("assistant", startMsg.GetProperty("role").GetString());
        Assert.Equal("test-model", startMsg.GetProperty("model").GetString());
        Assert.Equal(JsonValueKind.Null, startMsg.GetProperty("stop_reason").ValueKind);
        Assert.Equal(7, startMsg.GetProperty("usage").GetProperty("input_tokens").GetInt32());

        var blockStart = frames[1].Data;
        Assert.Equal(0, blockStart.GetProperty("index").GetInt32());
        Assert.Equal("text", blockStart.GetProperty("content_block").GetProperty("type").GetString());

        var delta = frames[3].Data.GetProperty("delta");
        Assert.Equal("text_delta", delta.GetProperty("type").GetString());
        Assert.Equal("Hel", delta.GetProperty("text").GetString());
        Assert.Equal("lo", frames[4].Data.GetProperty("delta").GetProperty("text").GetString());

        Assert.Equal(0, frames[5].Data.GetProperty("index").GetInt32());

        var messageDelta = frames[6].Data;
        Assert.Equal("end_turn", messageDelta.GetProperty("delta").GetProperty("stop_reason").GetString());
        Assert.Equal(JsonValueKind.Null, messageDelta.GetProperty("delta").GetProperty("stop_sequence").ValueKind);
        Assert.Equal(2, messageDelta.GetProperty("usage").GetProperty("output_tokens").GetInt32());

        Assert.Equal("message_stop", frames[7].Data.GetProperty("type").GetString());
    }

    [Fact]
    public async Task Streaming_MaxTokens_ReportsMaxTokensStopReason()
    {
        var frames = await RunAsync(Tokens(("hi", FinishReason.Length)));

        var messageDelta = frames.Single(f => f.Event == "message_delta").Data;
        Assert.Equal("max_tokens", messageDelta.GetProperty("delta").GetProperty("stop_reason").GetString());
    }

    [Fact]
    public async Task Streaming_MatchingStopSequence_ReportsStopSequence()
    {
        var frames = await RunAsync(
            Tokens(("all done", null), ("END", FinishReason.Stop)),
            stopSequences: ["END"]);

        var delta = frames.Single(f => f.Event == "message_delta").Data.GetProperty("delta");
        Assert.Equal("stop_sequence", delta.GetProperty("stop_reason").GetString());
        Assert.Equal("END", delta.GetProperty("stop_sequence").GetString());
    }

    // --- tool_use stream ----------------------------------------------------

    [Fact]
    public async Task Streaming_ToolUse_EmitsToolBlockAfterTextBlock()
    {
        var parser = new FixedToolCallParser(
            [new ToolCall("toolu_1", "get_weather", """{"city":"Paris"}""")]);
        var frames = await RunAsync(Tokens(("calling", FinishReason.Stop)), parser);

        Assert.Equal(
            ["message_start", "content_block_start", "ping", "content_block_delta",
             "content_block_stop", "content_block_start", "content_block_delta",
             "content_block_stop", "message_delta", "message_stop"],
            frames.Select(f => f.Event));

        // The tool_use block opens at index 1, after the text block at index 0.
        var toolStart = frames[5].Data;
        Assert.Equal(1, toolStart.GetProperty("index").GetInt32());
        var block = toolStart.GetProperty("content_block");
        Assert.Equal("tool_use", block.GetProperty("type").GetString());
        Assert.Equal("toolu_1", block.GetProperty("id").GetString());
        Assert.Equal("get_weather", block.GetProperty("name").GetString());
        // Anthropic opens a tool_use block with an empty input; arguments arrive as deltas.
        Assert.Equal(JsonValueKind.Object, block.GetProperty("input").ValueKind);
        Assert.Empty(block.GetProperty("input").EnumerateObject());

        var toolDelta = frames[6].Data.GetProperty("delta");
        Assert.Equal("input_json_delta", toolDelta.GetProperty("type").GetString());
        Assert.Equal("""{"city":"Paris"}""", toolDelta.GetProperty("partial_json").GetString());

        Assert.Equal(1, frames[7].Data.GetProperty("index").GetInt32());

        var messageDelta = frames[8].Data;
        Assert.Equal("tool_use", messageDelta.GetProperty("delta").GetProperty("stop_reason").GetString());
    }

    [Fact]
    public async Task Streaming_ToolParserFindsNothing_EmitsTextOnlySequence()
    {
        var frames = await RunAsync(Tokens(("plain", FinishReason.Stop)), new FixedToolCallParser(null));

        Assert.DoesNotContain(frames, f =>
            f.Event == "content_block_start" &&
            f.Data.GetProperty("content_block").GetProperty("type").GetString() == "tool_use");
        Assert.Equal("end_turn",
            frames.Single(f => f.Event == "message_delta").Data.GetProperty("delta")
                  .GetProperty("stop_reason").GetString());
    }

    // --- protocol invariants the official SDK relies on ----------------------

    [Fact]
    public async Task Streaming_EveryFrameCarriesATypeMatchingItsEventName()
    {
        // anthropic/_streaming.py dispatches on the SSE `event:` name and only fills in
        // `data.type` when the payload omits it — a payload whose `type` disagreed with the
        // event name would be routed as one event and parsed as another.
        var parser = new FixedToolCallParser([new ToolCall("toolu_1", "get_weather", "{}")]);
        var frames = await RunAsync(Tokens(("hi", FinishReason.Stop)), parser);

        foreach (var frame in frames)
            Assert.Equal(frame.Event, frame.Data.GetProperty("type").GetString());
    }

    [Fact]
    public async Task Streaming_MessageStartIsFirstAndMessageStopIsLast()
    {
        // accumulate_event() raises "Unexpected event order" for anything before message_start.
        var frames = await RunAsync(Tokens(("hi", FinishReason.Stop)));

        Assert.Equal("message_start", frames[0].Event);
        Assert.Equal("message_stop", frames[^1].Event);
        Assert.Single(frames, f => f.Event == "message_start");
        Assert.Single(frames, f => f.Event == "message_stop");
    }

    // --- mid-stream failure --------------------------------------------------

    private static async IAsyncEnumerable<GenerationToken> ThrowingTokens(int okTokens)
    {
        for (int i = 0; i < okTokens; i++)
        {
            await Task.Yield();
            yield return new GenerationToken(0, "tok", null);
        }
        await Task.Yield();
        throw new InvalidOperationException("backend exploded");
    }

    [Fact]
    public async Task Streaming_FailureAfterMessageStart_EmitsAnErrorEvent()
    {
        // Status headers are already flushed, so the only way to report the failure is the
        // Anthropic stream protocol's named `error` event; the SDK turns it into an
        // APIStatusError. Without it the client sees a truncated stream (#449).
        var frames = await RunAsync(ThrowingTokens(okTokens: 2));

        var error = Assert.Single(frames, f => f.Event == "error");
        Assert.Equal("error", error.Data.GetProperty("type").GetString());
        Assert.Equal("api_error", error.Data.GetProperty("error").GetProperty("type").GetString());
        Assert.Contains("backend exploded",
            error.Data.GetProperty("error").GetProperty("message").GetString()!);

        // A failed stream must not also claim to have finished normally.
        Assert.DoesNotContain(frames, f => f.Event == "message_delta");
        Assert.DoesNotContain(frames, f => f.Event == "message_stop");
        Assert.Equal("error", frames[^1].Event);

        // Whatever was generated before the failure still reached the client.
        Assert.Equal(2, frames.Count(f => f.Event == "content_block_delta"));
    }

    [Fact]
    public async Task Streaming_ClientDisconnect_DoesNotEmitAnErrorEvent()
    {
        // A cancelled request is not a server error; the SDK would surface a spurious
        // APIStatusError for a stream the caller itself abandoned.
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => RunAsync(CancelledTokens()));
    }

    private static async IAsyncEnumerable<GenerationToken> CancelledTokens()
    {
        await Task.Yield();
        yield return new GenerationToken(0, "tok", null);
        await Task.Yield();
        throw new OperationCanceledException();
    }

    // --- tool-call markup must not also be streamed as text ------------------

    // Bare tool-call JSON, split across two tokens the way a model emits it.
    private const string JsonHead = @"{""name"":""get_weather"",";
    private const string JsonTail = @"""arguments"":{""city"":""Paris""}}";

    private static string TextOf(SseFrame[] frames) => string.Concat(
        frames.Where(f => f.Event == "content_block_delta" &&
                          f.Data.GetProperty("delta").GetProperty("type").GetString() == "text_delta")
              .Select(f => f.Data.GetProperty("delta").GetProperty("text").GetString()));

    [Fact]
    public async Task Streaming_ForcedToolCall_DoesNotAlsoStreamTheJsonAsText()
    {
        // With a forced tool_choice the model emits bare tool-call JSON. Streaming it as
        // text_delta AND re-emitting it as a tool_use block reports the same payload twice:
        // the SDK's stream.text_stream would print raw JSON to the user, and the accumulated
        // message would carry a text block that the non-streaming route never produces.
        var frames = await RunAsync(
            Tokens((JsonHead, null), (JsonTail, FinishReason.Stop)),
            new GenericToolCallParser(),
            suppressToolCallText: true);

        Assert.DoesNotContain("get_weather", TextOf(frames));
        Assert.Contains(frames, f =>
            f.Event == "content_block_start" &&
            f.Data.GetProperty("content_block").GetProperty("type").GetString() == "tool_use");
        Assert.Equal("tool_use",
            frames.Single(f => f.Event == "message_delta").Data
                  .GetProperty("delta").GetProperty("stop_reason").GetString());
    }

    [Fact]
    public async Task Streaming_MarkerToolCall_StreamsThePreambleButNotTheMarkup()
    {
        // tool_choice=auto with a marker-based parser: text before the marker is genuine
        // assistant prose and must still reach the client; the <tool_call> envelope must not.
        var frames = await RunAsync(
            Tokens(("Let me check. ", null),
                   ("<tool_call>", null),
                   ("""{"name":"get_weather","arguments":{"city":"Paris"}}""", null),
                   ("</tool_call>", FinishReason.Stop)),
            new HermesToolCallParser());

        string text = TextOf(frames);
        Assert.Contains("Let me check.", text);
        Assert.DoesNotContain("tool_call", text);
        Assert.DoesNotContain("get_weather", text);
        Assert.Contains(frames, f =>
            f.Event == "content_block_start" &&
            f.Data.GetProperty("content_block").GetProperty("type").GetString() == "tool_use");
    }

    [Fact]
    public async Task Streaming_NoToolMarkup_StreamsEverythingAsText()
    {
        // The suppression must not eat ordinary prose from a model that never calls a tool.
        var frames = await RunAsync(
            Tokens(("The answer ", null), ("is 4.", FinishReason.Stop)),
            new HermesToolCallParser());

        Assert.Equal("The answer is 4.", TextOf(frames));
    }
}
