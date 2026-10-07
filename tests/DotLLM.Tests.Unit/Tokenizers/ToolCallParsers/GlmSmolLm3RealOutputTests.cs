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

namespace DotLLM.Tests.Unit.Tokenizers.ToolCallParsers;

/// <summary>
/// #797 golden fixtures: decoded text the dotLLM server really produced (build of this branch, Vulkan, greedy,
/// <c>tool_choice: none</c> so the raw text comes back in <c>content</c>), captured 2026-10-07 from GLM-4.7-Flash Q4_K_M and
/// SmolLM3-3B Q8_0. No weights, only output text.
/// </summary>
public class GlmSmolLm3RealOutputTests
{
    private static string Raw(string name)
        => File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Tokenizers", "ChatTemplates", "Fixtures", "real-output", name + ".raw.txt"))
            .Replace("\r\n", "\n", StringComparison.Ordinal);

    private static readonly ToolDefinition[] Weather =
        [new("get_weather", "w", """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""")];

    [Fact]
    public void Glm_RealToolCall_StopTrimmed_Parses()
    {
        string raw = Raw("glm-toolcall");
        Assert.DoesNotContain("</tool_call>", raw, StringComparison.Ordinal);   // the stop sequence trimmed it, as in the report
        var c = Assert.Single(((IToolCallParser)new GlmToolCallParser()).TryParse(raw, Weather)!);
        Assert.Equal("get_weather", c.FunctionName);
        Assert.Equal("{\"city\":\"Paris\"}", c.Arguments);
    }

    [Fact]
    public void Glm_RealParallelRequest_StopTrimmedToOneCall_StillParses()
    {
        var calls = new GlmToolCallParser().TryParse(Raw("glm-toolcall-parallel"))!;
        Assert.Equal("get_weather", Assert.Single(calls).FunctionName);
    }

    [Fact]
    public void Glm_RealThinkingTurn_SplitsAtTheClosingThinkTag_BecauseTheTemplateOpenedTheBlock()
    {
        // The GLM generation prompt ends "<|assistant|><think>", so the output starts INSIDE the block.
        var (r, c, s) = ReasoningSplitter.Split(Raw("glm-think-plain"), startInReasoning: true);
        Assert.StartsWith("1.  **Analyze the Request:**", r, StringComparison.Ordinal);
        Assert.EndsWith("Output: Paris.", r, StringComparison.Ordinal);
        Assert.Equal("Paris", c);
        Assert.True(s.SawReasoning);
    }

    [Fact]
    public void SmolLm3_RealToolCall_ParsesViaTheXmlParser()
    {
        var c = Assert.Single(new XmlToolCallParser().TryParse(Raw("smollm3-toolcall"))!);
        Assert.Equal("get_weather", c.FunctionName);
        Assert.Contains("Paris", c.Arguments, StringComparison.Ordinal);
    }

    private static async IAsyncEnumerable<GenerationToken> Script(string[] pieces)
    {
        for (int i = 0; i < pieces.Length; i++)
        {
            await Task.Yield();
            yield return new GenerationToken(i, pieces[i], i == pieces.Length - 1 ? FinishReason.Stop : null);
        }
    }

    [Fact]
    public async Task Glm_RealToolCall_Streamed_NoMarkupInContent_ProseBeforeTheCallStillStreams()
    {
        // "<tool_call>" is a single token in GLM's vocabulary; the rest arrives in small pieces.
        string raw = Raw("glm-toolcall");
        int at = raw.IndexOf("<tool_call>", StringComparison.Ordinal);
        var pieces = new List<string> { raw[..at], "<tool_call>" };
        string rest = raw[(at + "<tool_call>".Length)..];
        for (int i = 0; i < rest.Length; i += 5)
            pieces.Add(rest.Substring(i, Math.Min(5, rest.Length - i)));

        var ctx = new DefaultHttpContext();
        var body = new MemoryStream();
        ctx.Response.Body = body;
        await ChatCompletionEndpoint.WriteChatStreamAsync(
            new ChatCompletionRequest { Messages = [], Stream = true }, ctx, _ => Script([.. pieces]), (work, _) => work(),
            "req", "m", Weather, new GlmToolCallParser(), ReasoningPlan.Disabled, CancellationToken.None);

        var content = new StringBuilder();
        JsonElement? calls = null;
        foreach (string block in Encoding.UTF8.GetString(body.ToArray()).Split("\n\n", StringSplitOptions.RemoveEmptyEntries))
        {
            string data = block["data: ".Length..];
            if (data == "[DONE]") continue;
            foreach (var choice in JsonDocument.Parse(data).RootElement.GetProperty("choices").EnumerateArray())
            {
                var d = choice.GetProperty("delta");
                if (d.TryGetProperty("content", out var c)) content.Append(c.GetString());
                if (d.TryGetProperty("tool_calls", out var tc)) calls = tc.Clone();
            }
        }

        Assert.DoesNotContain("<tool_call>", content.ToString(), StringComparison.Ordinal);
        Assert.DoesNotContain("arg_key", content.ToString(), StringComparison.Ordinal);
        Assert.Equal("get_weather", calls!.Value[0].GetProperty("function").GetProperty("name").GetString());
    }
}
