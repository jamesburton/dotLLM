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

    private static async IAsyncEnumerable<GenerationToken> Script(FinishReason last, Action<int>? beforePiece, string[] pieces)
    {
        for (int i = 0; i < pieces.Length; i++)
        {
            await Task.Yield();
            beforePiece?.Invoke(i);
            yield return new GenerationToken(0, pieces[i], i == pieces.Length - 1 ? last : null);
        }
    }

    private static Task NoGate(Func<Task> work, CancellationToken ct) => work();

    private sealed record Result(string Content, JsonElement? FinalToolCalls, string? FinishReason, string Raw);

    private static async Task<Result> RunAsync(string[] pieces, IToolCallParser parser, bool forced = false, ToolDefinition[]? tools = null, ReasoningPlan? plan = null,
        FinishReason last = FinishReason.Stop, Action<int, string>? beforePiece = null)
    {
        var ctx = new DefaultHttpContext();
        var body = new MemoryStream();
        ctx.Response.Body = body;
        var request = new ChatCompletionRequest { Messages = [], Stream = true };

        await ChatCompletionEndpoint.WriteChatStreamAsync(
            request, ctx, _ => Script(last, beforePiece is null ? null : i => beforePiece(i, Encoding.UTF8.GetString(body.ToArray())), pieces), NoGate, "req_1", "m", tools ?? Tools, parser,
            plan ?? ReasoningPlan.Disabled, CancellationToken.None, forced);

        string raw = Encoding.UTF8.GetString(body.ToArray());
        var content = new StringBuilder();
        // Reassemble tool_calls the way an OpenAI client does: deltas merge by index, arguments concatenate (#1342).
        var assembled = new SortedDictionary<int, (string? Id, string? Name, StringBuilder Args)>();
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
                foreach (var d in tc.EnumerateArray())
                {
                    int idx = d.GetProperty("index").GetInt32();
                    var cur = assembled.TryGetValue(idx, out var e) ? e : (null, null, new StringBuilder());
                    var fn = d.GetProperty("function");
                    string? id = d.TryGetProperty("id", out var idEl) ? idEl.GetString() : cur.Id;
                    string? name = fn.TryGetProperty("name", out var nEl) ? nEl.GetString() : cur.Name;
                    cur.Args.Append(fn.GetProperty("arguments").GetString());
                    assembled[idx] = (id, name, cur.Args);
                }
            if (choice.TryGetProperty("finish_reason", out var fr) && fr.ValueKind == JsonValueKind.String)
                finish = fr.GetString();
        }
        JsonElement? finalCalls = assembled.Count == 0 ? null : JsonSerializer.SerializeToElement(
            assembled.Select(kv => new
            {
                index = kv.Key, id = kv.Value.Id, type = "function",
                function = new { name = kv.Value.Name, arguments = kv.Value.Args.ToString() },
            }).ToArray());
        return new Result(content.ToString(), finalCalls, finish, raw);
    }

    [Fact]
    public async Task ParallelCalls_HaveDistinctStreamIndexes_AndPreserveArguments()
    {
        var r = await RunAsync(
            ["<tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>\n",
             "<tool_call>\n<function=get_weather>\n<parameter=city>London</parameter>\n</function>\n</tool_call>"],
            new QwenXmlToolCallParser());

        Assert.Equal("tool_calls", r.FinishReason);
        var calls = r.FinalToolCalls!.Value;
        Assert.Equal(2, calls.GetArrayLength());
        for (int i = 0; i < calls.GetArrayLength(); i++)
        {
            Assert.Equal(i, calls[i].GetProperty("index").GetInt32());
            Assert.Equal("function", calls[i].GetProperty("type").GetString());
            Assert.Equal("get_weather", calls[i].GetProperty("function").GetProperty("name").GetString());
            var arguments = JsonDocument.Parse(calls[i].GetProperty("function").GetProperty("arguments").GetString()!).RootElement;
            Assert.Equal(i == 0 ? "Paris" : "London", arguments.GetProperty("city").GetString());
        }
        Assert.NotEqual(calls[0].GetProperty("id").GetString(), calls[1].GetProperty("id").GetString());
        if (Environment.GetEnvironmentVariable("DOTLLM_STREAM_INDEX_CAPTURE") is { } capture)
            await File.WriteAllTextAsync(capture, r.Raw);
    }

    [Fact]
    public void NonstreamToolCalls_DoNotContainStreamIndexes()
    {
        var calls = RequestConverter.ToToolCallDtos([new ToolCall("call_0", "get_weather", "{\"city\":\"Paris\"}")]);
        var response = new ChatCompletionResponse
        {
            Id = "req_1", Model = "m", Usage = new UsageDto(),
            Choices = [new ChatChoiceDto { Message = new ChatMessageDto { Role = "assistant", ToolCalls = calls }, FinishReason = "tool_calls" }],
        };
        var json = JsonSerializer.SerializeToElement(response, ServerJsonContext.Default.ChatCompletionResponse);
        var call = json.GetProperty("choices")[0].GetProperty("message").GetProperty("tool_calls")[0];
        Assert.False(call.TryGetProperty("index", out _));
        Assert.Equal("{\"city\":\"Paris\"}", call.GetProperty("function").GetProperty("arguments").GetString());
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
        Assert.Equal(0, call.GetProperty("index").GetInt32());
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
    public async Task TruncatedUnparseableCall_BeforeAnyName_IsDeliveredAsContent_NotSwallowed()
    {
        // The model ran out of tokens before the function name completed: nothing streamed, so the held-back text must still reach the client.
        var r = await RunAsync(["<tool_call>", "\n<function=get_wea"], new QwenXmlToolCallParser());

        Assert.Null(r.FinalToolCalls);
        Assert.Equal("<tool_call>\n<function=get_wea", r.Content);
    }

    [Fact]
    public async Task StreamedCall_ArgumentsArriveBeforeTheCallIsComplete()
    {
        // 1342: each piece is fed only after everything the previous pieces released has hit the wire. The city value is
        // streamed as it arrives, and the id and name are out before any argument text.
        string[] pieces = ["<tool_call>\n<function=get_weather>\n", "<parameter=city>\n", "Par", "is, Fra", "nce", "\n</parameter>\n", "</function>\n</tool_call>"];
        var r = await RunAsync(pieces, new QwenXmlToolCallParser(), beforePiece: (i, wire) =>
        {
            if (i == 2) { Assert.Contains("\"name\":\"get_weather\"", wire); Assert.Contains("\"id\":\"call_0\"", wire); }
            if (i == 3) Assert.Contains("Par", wire);               // "Par" is already out while the value is still open
            if (i == 4) Assert.Contains("is, Fra", wire);
            if (i == 6) Assert.DoesNotContain("\"finish_reason\":\"tool_calls\"", wire);   // not finished yet
        });

        Assert.Equal("tool_calls", r.FinishReason);
        var call = Assert.Single(r.FinalToolCalls!.Value.EnumerateArray().ToArray());
        Assert.Equal("{\"city\":\"Paris, France\"}", call.GetProperty("function").GetProperty("arguments").GetString());
        Assert.Equal(0, call.GetProperty("index").GetInt32());
        // Continuations carry no id/type/name: only the first delta of a call does.
        Assert.Equal(r.Raw.IndexOf("\"id\":\"call_0\"", StringComparison.Ordinal), r.Raw.LastIndexOf("\"id\":\"call_0\"", StringComparison.Ordinal));
        Assert.Equal("", r.Content);
    }

    [Fact]
    public async Task StreamedCall_ThenSecondCall_SecondGetsIndexOneInTheFinalChunk()
    {
        var r = await RunAsync(
            ["<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n",
             "<tool_call>\n<function=get_weather>\n<parameter=city>\nLondon\n</parameter>\n</function>\n</tool_call>"],
            new QwenXmlToolCallParser());

        var calls = r.FinalToolCalls!.Value.EnumerateArray().ToArray();
        Assert.Equal(2, calls.Length);
        Assert.Equal(0, calls[0].GetProperty("index").GetInt32());
        Assert.Equal(1, calls[1].GetProperty("index").GetInt32());
        Assert.Equal("{\"city\":\"Paris\"}", calls[0].GetProperty("function").GetProperty("arguments").GetString());
        Assert.Equal("{\"city\":\"London\"}", calls[1].GetProperty("function").GetProperty("arguments").GetString());
        Assert.Equal("tool_calls", r.FinishReason);
    }

    [Fact]
    public async Task StreamedCall_CutOffByMaxTokens_KeepsLengthFinish_AndNeverLeaksCallTextAsContent()
    {
        // Documented outcome (1342): the client saw a prefix of the arguments. The stream ends with the real finish reason (never
        // tool_calls), the half call is not also delivered as content, and the partial arguments are not valid JSON.
        var r = await RunAsync(["<tool_call>\n<function=get_weather>\n", "<parameter=city>\nPar"],
            new QwenXmlToolCallParser(), last: FinishReason.Length);

        Assert.Equal("length", r.FinishReason);
        Assert.Equal("", r.Content);
        var call = Assert.Single(r.FinalToolCalls!.Value.EnumerateArray().ToArray());
        string args = call.GetProperty("function").GetProperty("arguments").GetString()!;
        Assert.Equal("{\"city\":\"Par", args);
        Assert.ThrowsAny<JsonException>(() => JsonDocument.Parse(args));
    }

    [Fact]
    public async Task HermesJsonBodyInsideQwenParser_IsNotStreamed_AndStillArrivesWhole()
    {
        var r = await RunAsync(["<tool_call>\n", "{\"name\": \"get_weather\", ", "\"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"],
            new QwenXmlToolCallParser());

        Assert.Equal("tool_calls", r.FinishReason);
        var call = Assert.Single(r.FinalToolCalls!.Value.EnumerateArray().ToArray());
        Assert.Equal("Paris", JsonDocument.Parse(call.GetProperty("function").GetProperty("arguments").GetString()!).RootElement.GetProperty("city").GetString());
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

    /// <summary>One prose-then-call completion per parser, in that parser's own wire format (#1287).</summary>
    public static TheoryData<string, string> ProseThenCall() => new()
    {
        { nameof(HermesToolCallParser), "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>" },
        { nameof(XmlToolCallParser), "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>" },
        { nameof(QwenXmlToolCallParser), "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>" },
        { nameof(LlamaToolCallParser), "<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}" },
        { nameof(MistralToolCallParser), "[TOOL_CALLS][{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]" },
        { nameof(Gemma4ToolCallParser), "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>" },
        { nameof(GlmToolCallParser), "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>" },
        { nameof(GraniteToolCallParser), "<|tool_call|>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]" },
        { nameof(PythonicToolCallParser), "<|tool_call_start|>[get_weather(city=\"Paris\")]<|tool_call_end|>" },
        { nameof(HarmonyToolCallParser), "to=functions.get_weather <|constrain|>json<|message|>{\"city\": \"Paris\"}<|call|>" },
        { nameof(GenericToolCallParser), "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}" },
    };

    private static IToolCallParser ParserNamed(string name) => name switch
    {
        nameof(HermesToolCallParser) => new HermesToolCallParser(),
        nameof(XmlToolCallParser) => new XmlToolCallParser(),
        nameof(QwenXmlToolCallParser) => new QwenXmlToolCallParser(),
        nameof(LlamaToolCallParser) => new LlamaToolCallParser(),
        nameof(MistralToolCallParser) => new MistralToolCallParser(),
        nameof(Gemma4ToolCallParser) => new Gemma4ToolCallParser(),
        nameof(GlmToolCallParser) => new GlmToolCallParser(),
        nameof(GraniteToolCallParser) => new GraniteToolCallParser(),
        nameof(PythonicToolCallParser) => new PythonicToolCallParser(),
        nameof(HarmonyToolCallParser) => new HarmonyToolCallParser(),
        _ => new GenericToolCallParser(),
    };

    /// <summary>
    /// #1287: with tools present, prose before a call streams as it arrives (several content chunks, not one flush at the
    /// end), the call is reported, and no fragment of the marker leaks into content, even when the marker arrives split
    /// across tokens (3-char pieces here).
    /// </summary>
    [Theory]
    [MemberData(nameof(ProseThenCall))]
    public async Task ProseBeforeCall_Streams_AndNoMarkerFragmentLeaks(string parserName, string call)
    {
        const string prose = "Let me look that up for you. ";
        string all = prose + call;
        var pieces = Enumerable.Range(0, (all.Length + 2) / 3).Select(i => all.Substring(i * 3, Math.Min(3, all.Length - i * 3))).ToArray();

        var r = await RunAsync(pieces, ParserNamed(parserName));

        Assert.Equal("tool_calls", r.FinishReason);
        Assert.Equal("get_weather", r.FinalToolCalls!.Value[0].GetProperty("function").GetProperty("name").GetString());
        Assert.Equal(prose, r.Content);
        int contentChunks = r.Raw.Split("\n\n").Count(b => b.Contains("\"content\":\"", StringComparison.Ordinal));
        Assert.True(contentChunks >= 5, $"prose should stream piece by piece, got {contentChunks} content chunk(s)");
    }

    /// <summary>#1287: a tail that only looks like the start of a marker is released once it stops matching.</summary>
    [Fact]
    public async Task MarkerLookalike_IsHeldOnlyUntilItDiverges()
    {
        var r = await RunAsync(["a <to", "ol>b <tool_", "x> done"], new HermesToolCallParser());

        Assert.Null(r.FinalToolCalls);
        Assert.Equal("a <tool>b <tool_x> done", r.Content);
    }

    /// <summary>
    /// The generic parser now holds from a <c>{</c> (#1287), but brace-bearing prose that never parses into a call is
    /// delayed, not swallowed: it is delivered once generation ends. (A <c>{"name": ...}</c> object, by contrast, is
    /// reported once as a call: before #1287 it went out as content AND as the final chunk's tool_calls.)
    /// </summary>
    [Fact]
    public async Task GenericModelParser_UnderAuto_DoesNotSwallowJsonishProse()
    {
        var r = await RunAsync(["Use {", "\"a\": 1} as ", "the shape."], new GenericToolCallParser());
        Assert.Null(r.FinalToolCalls);
        Assert.Equal("Use {\"a\": 1} as the shape.", r.Content);
    }
}
