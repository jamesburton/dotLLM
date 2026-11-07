using System.Text;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
using DotLLM.Tests.Unit.Server;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ToolCallParsers;
using Microsoft.AspNetCore.Http;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ToolCallParsers;

/// <summary>#796: IBM Granite 3.x <c>&lt;|tool_call|&gt;[{...}]</c> tool calls.</summary>
public class GraniteToolCallParserTests
{
    private readonly GraniteToolCallParser _p = new();

    private static readonly ToolDefinition[] Tools =
        [new("get_weather", "w", """{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}""")];

    [Fact]
    public void ParsesTheJsonListAfterTheMarker()
    {
        var calls = _p.TryParse("<|tool_call|>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]");
        var c = Assert.Single(calls!);
        Assert.Equal("get_weather", c.FunctionName);
        Assert.Equal("Paris", JsonDocument.Parse(c.Arguments).RootElement.GetProperty("city").GetString());
    }

    [Fact]
    public void ParallelCallsInOneList()
    {
        var calls = _p.TryParse(
            "<|tool_call|>[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}},{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Rome\"}}]");
        Assert.Equal(2, calls!.Length);
        Assert.Equal(["call_0", "call_1"], calls.Select(c => c.Id));
    }

    [Fact]
    public void ProseBeforeTheMarker_AndARepeatedMarker()
    {
        var calls = _p.TryParse("Let me check. <|tool_call|>{\"name\":\"a\",\"arguments\":{}}<|tool_call|>{\"name\":\"b\",\"arguments\":{}}");
        Assert.Equal(["a", "b"], calls!.Select(c => c.FunctionName));
    }

    [Fact]
    public void NoMarker_IsNotACall_EvenIfTheTextQuotesJson()
    {
        Assert.Null(_p.TryParse("{\"name\":\"get_weather\",\"arguments\":{}}"));
        Assert.Null(_p.TryParse("It is sunny."));
    }

    [Fact]
    public void TruncatedJson_IsNotReported()
    {
        Assert.Null(_p.TryParse("<|tool_call|>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Par"));
    }

    [Fact]
    public void ArgumentsAreCoercedToTheSchema()
    {
        var c = Assert.Single(((IToolCallParser)_p).TryParse("<|tool_call|>[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\",\"days\":\"3\"}}]", Tools)!);
        Assert.Contains("\"days\":3", c.Arguments.Replace(" ", ""), StringComparison.Ordinal);
    }

    [Fact]
    public void IsToolCallStart_KeysOnTheMarker()
    {
        Assert.True(_p.IsToolCallStart("<|tool_call|>"));
        Assert.True(_p.IsToolCallStart("ok <|tool_call|>[{"));
        Assert.False(_p.IsToolCallStart("<|tool_call"));
        Assert.False(_p.IsToolCallStart("{\"name\": \"x\"}"));
    }

    [Fact]
    public void Factory_PicksGranite_ForTheTemplate_AndTheArchitecture()
    {
        const string template = "respond only with <|tool_call|> followed by a JSON list of tools used";
        Assert.IsType<GraniteToolCallParser>(ToolCallParserFactory.Create(Architecture.Llama, template));
        Assert.IsType<GraniteToolCallParser>(ToolCallParserFactory.Create(Architecture.Granite));
        Assert.IsType<GraniteToolCallParser>(ToolCallParserFactory.Create(Architecture.GraniteMoe, null));
        // Neighbouring token pairs still route to their own parsers.
        Assert.IsType<Gemma4ToolCallParser>(ToolCallParserFactory.Create(Architecture.Llama, "<|tool_call>call:x{}<tool_call|>"));
        Assert.IsType<XmlToolCallParser>(ToolCallParserFactory.Create(Architecture.Llama, "<tool_call>{}</tool_call>"));
    }

    // ---------------------------------------------------------------- streaming wire test

    private static async IAsyncEnumerable<GenerationToken> Script(string[] pieces)
    {
        for (int i = 0; i < pieces.Length; i++)
        {
            await Task.Yield();
            yield return new GenerationToken(i, pieces[i], i == pieces.Length - 1 ? FinishReason.Stop : null);
        }
    }

    /// <summary>The first token arrives late, as after a long prefill: the endpoint's keep-alive (#1025) fires meanwhile.</summary>
    private static async IAsyncEnumerable<GenerationToken> SlowFirstToken(string[] pieces, Task firstTokenGate)
    {
        await firstTokenGate;
        await foreach (var t in Script(pieces)) yield return t;
    }

    /// <summary>A response body that signals once an SSE comment line (<c>: waiting for ...</c>) has been written to it.</summary>
    private sealed class CommentSignallingStream : MemoryStream
    {
        private readonly Lock _gate = new();
        public TaskCompletionSource FirstComment { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);

        // On a subclass MemoryStream.Write(ReadOnlySpan) routes through Stream.Write(span) -> Write(byte[], int, int), so the
        // array overload is the one place that writes and checks; the span overload must only defer to base.
        public override void Write(byte[] buffer, int offset, int count)
        {
            lock (_gate)
            {
                base.Write(buffer, offset, count);
                if (!FirstComment.Task.IsCompleted && Encoding.UTF8.GetString(ToArray()).Contains("\n: waiting for ", StringComparison.Ordinal))
                    FirstComment.TrySetResult();
            }
        }

        public override void Write(ReadOnlySpan<byte> buffer) { lock (_gate) base.Write(buffer); }
        public override Task WriteAsync(byte[] buffer, int offset, int count, CancellationToken ct) { Write(buffer, offset, count); return Task.CompletedTask; }
        public override ValueTask WriteAsync(ReadOnlyMemory<byte> buffer, CancellationToken ct = default) { Write(buffer.Span); return ValueTask.CompletedTask; }
        public string Text { get { lock (_gate) return Encoding.UTF8.GetString(ToArray()); } }
    }

    /// <param name="afterAKeepAlive">When true the first token is held until the endpoint has written a prefill keep-alive comment
    /// (interval 1 ms), so the SSE body deterministically carries <c>: waiting for prompt processing (0s)</c> before the first delta.</param>
    private static async Task<(string Content, JsonElement? Calls, string? Finish, string Raw)> StreamAsync(
        string[] pieces, ToolDefinition[] tools, bool afterAKeepAlive = false)
    {
        var ctx = new DefaultHttpContext();
        var body = new CommentSignallingStream();
        ctx.Response.Body = body;
        await ChatCompletionEndpoint.WriteChatStreamAsync(
            new ChatCompletionRequest { Messages = [], Stream = true }, ctx,
            _ => afterAKeepAlive ? SlowFirstToken(pieces, body.FirstComment.Task) : Script(pieces), (work, _) => work(),
            "req", "m", tools, new GraniteToolCallParser(), ReasoningPlan.Disabled, CancellationToken.None,
            keepAliveInterval: afterAKeepAlive ? TimeSpan.FromMilliseconds(1) : null);

        string raw = body.Text;
        var content = new StringBuilder();
        JsonElement? calls = null;
        string? finish = null;
        foreach (var frame in SseTestReader.Json(raw))
        {
            foreach (var choice in frame.GetProperty("choices").EnumerateArray())
            {
                var d = choice.GetProperty("delta");
                if (d.TryGetProperty("content", out var c)) content.Append(c.GetString());
                if (d.TryGetProperty("tool_calls", out var tc)) calls = tc.Clone();
                if (choice.TryGetProperty("finish_reason", out var fr) && fr.ValueKind == JsonValueKind.String) finish = fr.GetString();
            }
        }
        return (content.ToString(), calls, finish, raw);
    }

    [Fact]
    public async Task Streaming_MarkerAndJson_DoNotLeakIntoContent_CallArrivesInFinalChunk()
    {
        string[] pieces = ["<|tool_call|>", "[{\"name\": ", "\"get_weather\", ", "\"arguments\": {\"city\": ", "\"Paris\"}}]"];
        var (content, calls, finish, _) = await StreamAsync(pieces, Tools);

        Assert.DoesNotContain("<|tool_call", content, StringComparison.Ordinal);
        Assert.Equal("", content);
        Assert.Equal("tool_calls", finish);
        Assert.Equal("get_weather", calls!.Value[0].GetProperty("function").GetProperty("name").GetString());
    }

    // ---------------------------------------------------------------- real output (granite-3.3-2b-instruct Q8_0, greedy, 2026-10-07)

    private static string Raw(string name)
        => File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Tokenizers", "ChatTemplates", "Fixtures", "real-output", name + ".raw.txt"))
            .Replace("\r\n", "\n", StringComparison.Ordinal);

    /// <summary>Realistic tokenisation: the special token is one piece, the JSON arrives in small pieces.</summary>
    private static string[] Tokens(string raw)
    {
        var parts = new List<string> { "<|tool_call|>" };
        string rest = raw["<|tool_call|>".Length..];
        for (int i = 0; i < rest.Length; i += 6)
            parts.Add(rest.Substring(i, Math.Min(6, rest.Length - i)));
        return [.. parts];
    }

    private static readonly ToolDefinition[] Alarm =
        [new("set_alarm", "a", """{"type":"object","properties":{"hour":{"type":"integer"},"label":{"type":"string"},"repeat":{"type":"boolean"},"days":{"type":"array","items":{"type":"string"}}},"required":["hour","label"]}""")];

    [Fact]
    public void RealOutput_NamedCall()
    {
        var c = Assert.Single(_p.TryParse(Raw("granite-toolcall"))!);
        Assert.Equal("get_weather", c.FunctionName);
        Assert.Equal("{\"city\": \"Paris\"}", c.Arguments);
    }

    [Fact]
    public void RealOutput_ParallelCallsInOneList()
    {
        var calls = _p.TryParse(Raw("granite-toolcall-parallel"))!;
        Assert.Equal(2, calls.Length);
        Assert.Contains("Paris", calls[0].Arguments, StringComparison.Ordinal);
        Assert.Contains("Tokyo", calls[1].Arguments, StringComparison.Ordinal);
    }

    [Fact]
    public void RealOutput_NamelessCall_IsAttributedToTheOnlyDeclaredTool_WithTypedArguments()
    {
        // The model really answered {"arguments": {...}} with no name, deterministically at temperature 0.
        string raw = Raw("granite-toolcall-typed");
        Assert.Null(_p.TryParse(raw));                                              // no tools known: nothing to attribute it to
        var c = Assert.Single(((IToolCallParser)_p).TryParse(raw, Alarm)!);
        Assert.Equal("set_alarm", c.FunctionName);
        using var doc = JsonDocument.Parse(c.Arguments);
        Assert.Equal(7, doc.RootElement.GetProperty("hour").GetInt32());
        Assert.True(doc.RootElement.GetProperty("repeat").GetBoolean());
        Assert.Equal(["Monday", "Friday"], doc.RootElement.GetProperty("days").EnumerateArray().Select(e => e.GetString()));

        // Two declared tools: ambiguous, so not reported.
        var two = new[] { Alarm[0], new ToolDefinition("other", "o", "{}") };
        Assert.Null(((IToolCallParser)_p).TryParse(raw, two));
    }

    [Fact]
    public void NamelessCall_Truncated_IsNotReported()
    {
        Assert.Null(((IToolCallParser)_p).TryParse("<|tool_call|>[{\"arguments\": {\"hour\": 7, \"lab", Alarm));
    }

    [Theory]
    [InlineData("granite-toolcall")]
    [InlineData("granite-toolcall-typed")]
    [InlineData("granite-toolcall-parallel")]
    public async Task RealOutput_Streamed_NoMarkerOrJsonInContent_CallsInFinalChunk(string fixture)
    {
        var tools = fixture.EndsWith("typed", StringComparison.Ordinal) ? Alarm : Tools;
        var (content, calls, finish, _) = await StreamAsync(Tokens(Raw(fixture)), tools);
        Assert.Equal("", content);
        Assert.Equal("tool_calls", finish);
        Assert.Equal(fixture.EndsWith("parallel", StringComparison.Ordinal) ? 2 : 1, calls!.Value.GetArrayLength());
    }

    /// <summary>
    /// #1498: the flaky CI failure (<c>JsonReaderException: 'i' is an invalid start of a value</c>) was the endpoint's prefill
    /// keep-alive comment <c>: waiting for prompt processing (10s)</c> (#1025), written when the first token is more than one
    /// interval late (a stalled runner), read by a test SSE reader that sliced <c>"data: ".Length</c> off every block. Here the
    /// comment is forced deterministically (1 ms interval, first token gated on the comment having been written) and the
    /// stream must still read as a clean tool call.
    /// </summary>
    [Theory]
    [InlineData("granite-toolcall")]
    [InlineData("granite-toolcall-typed")]
    [InlineData("granite-toolcall-parallel")]
    public async Task RealOutput_Streamed_AfterAPrefillKeepAliveComment_StillReadsAsACleanToolCall(string fixture)
    {
        var tools = fixture.EndsWith("typed", StringComparison.Ordinal) ? Alarm : Tools;
        var (content, calls, finish, raw) = await StreamAsync(Tokens(Raw(fixture)), tools, afterAKeepAlive: true);

        // The arm is live only if the keep-alive really went out (control-arm sensitivity).
        Assert.True(SseTestReader.HasComment(raw), "no keep-alive comment in the body: " + raw);
        Assert.Contains("\n: waiting for prompt processing (", raw, StringComparison.Ordinal);
        Assert.Equal("", content);
        Assert.Equal("tool_calls", finish);
        Assert.Equal(fixture.EndsWith("parallel", StringComparison.Ordinal) ? 2 : 1, calls!.Value.GetArrayLength());
    }

    /// <summary>
    /// #1498 control arm: the suspected cause was a chunk-boundary bug. Every single split point of every fixture (the marker
    /// token kept whole, then the JSON cut once at each character), plus one-character-per-token streaming, must read as the
    /// same clean call set. This ran green against the unfixed reader too: the boundary hypothesis is discharged, not just untested.
    /// </summary>
    [Theory]
    [InlineData("granite-toolcall")]
    [InlineData("granite-toolcall-typed")]
    [InlineData("granite-toolcall-parallel")]
    public async Task RealOutput_Streamed_EverySplitPoint_ReadsAsTheSameCalls(string fixture)
    {
        var tools = fixture.EndsWith("typed", StringComparison.Ordinal) ? Alarm : Tools;
        string raw = Raw(fixture);
        string rest = raw["<|tool_call|>".Length..];
        int expected = fixture.EndsWith("parallel", StringComparison.Ordinal) ? 2 : 1;

        async Task CheckAsync(string[] pieces, string arm)
        {
            var (content, calls, finish, _) = await StreamAsync(pieces, tools);
            Assert.True(content.Length == 0, $"{arm}: content leaked: {content}");
            Assert.True(finish == "tool_calls", $"{arm}: finish {finish}");
            Assert.True(calls is not null && calls.Value.GetArrayLength() == expected, $"{arm}: calls {calls?.GetArrayLength()}");
        }

        for (int cut = 1; cut < rest.Length; cut++)
            await CheckAsync(["<|tool_call|>", rest[..cut], rest[cut..]], $"cut@{cut}");
        await CheckAsync(["<|tool_call|>", .. rest.Select(ch => ch.ToString())], "per-char");
        await CheckAsync([raw], "whole");
        // Marker split across tokens, as a tokenizer without the special token would do.
        await CheckAsync(["<|tool", "_call|>", rest], "split-marker");
    }
}
