using System.Text;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
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

    [Fact]
    public async Task Streaming_MarkerAndJson_DoNotLeakIntoContent_CallArrivesInFinalChunk()
    {
        string[] pieces = ["<|tool_call|>", "[{\"name\": ", "\"get_weather\", ", "\"arguments\": {\"city\": ", "\"Paris\"}}]"];
        var ctx = new DefaultHttpContext();
        var body = new MemoryStream();
        ctx.Response.Body = body;
        await ChatCompletionEndpoint.WriteChatStreamAsync(
            new ChatCompletionRequest { Messages = [], Stream = true }, ctx, _ => Script(pieces), (work, _) => work(),
            "req", "m", Tools, new GraniteToolCallParser(), ReasoningPlanFor(), CancellationToken.None);

        var content = new StringBuilder();
        JsonElement? calls = null;
        string? finish = null;
        foreach (string block in Encoding.UTF8.GetString(body.ToArray()).Split("\n\n", StringSplitOptions.RemoveEmptyEntries))
        {
            string data = block["data: ".Length..];
            if (data == "[DONE]") continue;
            foreach (var choice in JsonDocument.Parse(data).RootElement.GetProperty("choices").EnumerateArray())
            {
                var d = choice.GetProperty("delta");
                if (d.TryGetProperty("content", out var c)) content.Append(c.GetString());
                if (d.TryGetProperty("tool_calls", out var tc)) calls = tc.Clone();
                if (choice.TryGetProperty("finish_reason", out var fr) && fr.ValueKind == JsonValueKind.String) finish = fr.GetString();
            }
        }

        Assert.DoesNotContain("<|tool_call", content.ToString(), StringComparison.Ordinal);
        Assert.Equal("", content.ToString());
        Assert.Equal("tool_calls", finish);
        Assert.Equal("get_weather", calls!.Value[0].GetProperty("function").GetProperty("name").GetString());
    }

    private static ReasoningPlan ReasoningPlanFor() => ReasoningPlan.Disabled;
}
