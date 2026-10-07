using System.Text;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Sampling;
using DotLLM.Engine;
using DotLLM.Engine.Samplers.StopConditions;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using DotLLM.Tokenizers.ToolCallParsers;
using Microsoft.AspNetCore.Http;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ToolCallParsers;

/// <summary>#797: GLM-4.5-style <c>&lt;tool_call&gt;name&lt;arg_key&gt;…</c> tool calls, GLM's end-of-turn tokens, SmolLM3's tool rendering.</summary>
public class GlmToolCallParserTests
{
    private readonly GlmToolCallParser _p = new();

    private static readonly ToolDefinition[] Tools =
    [
        new("get_weather", "w", """{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},"opts":{"type":"object"}}}"""),
    ];

    private static string Fixture(string name)
        => File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Tokenizers", "ChatTemplates", "Fixtures", name));

    // ---------------------------------------------------------------- parser

    [Fact]
    public void ParsesTheHarnessOutput_WithTheClosingTagTrimmedByTheStopSequence()
    {
        // Verbatim from the #797 report: the server's "</tool_call>" stop trims the tag.
        var calls = _p.TryParse("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>");
        var c = Assert.Single(calls!);
        Assert.Equal("get_weather", c.FunctionName);
        Assert.Equal("{\"city\":\"Paris\"}", c.Arguments);
    }

    [Fact]
    public void ClosedBlock_TypedArguments_ComeFromTheSchema()
    {
        var calls = ((IToolCallParser)_p).TryParse(
            "</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
            + "<arg_key>days</arg_key><arg_value>3</arg_value><arg_key>opts</arg_key><arg_value>{\"unit\": \"c\"}</arg_value></tool_call>", Tools);
        var c = Assert.Single(calls!);
        using var doc = JsonDocument.Parse(c.Arguments);
        Assert.Equal("Paris", doc.RootElement.GetProperty("city").GetString());
        Assert.Equal(3, doc.RootElement.GetProperty("days").GetInt32());
        Assert.Equal("c", doc.RootElement.GetProperty("opts").GetProperty("unit").GetString());
    }

    [Fact]
    public void StringParameterThatLooksNumeric_StaysAString()
    {
        var c = Assert.Single(((IToolCallParser)_p).TryParse(
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>12345</arg_value></tool_call>", Tools)!);
        Assert.Equal("{\"city\":\"12345\"}", c.Arguments);
    }

    [Fact]
    public void ParallelCalls_AreSequentiallyIdentified_WithOrWithoutClosingTags()
    {
        var calls = _p.TryParse(
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>\n"
            + "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Rome</arg_value>");
        Assert.Equal(2, calls!.Length);
        Assert.Equal(["call_0", "call_1"], calls.Select(c => c.Id));
        Assert.Contains("Rome", calls[1].Arguments, StringComparison.Ordinal);
    }

    [Fact]
    public void NoArguments_AndNewlineAfterName()
    {
        Assert.Equal("{}", Assert.Single(_p.TryParse("<tool_call>ping</tool_call>")!).Arguments);
        var c = Assert.Single(_p.TryParse("<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>")!);
        Assert.Equal("get_weather", c.FunctionName);
        Assert.Equal("{\"city\":\"Paris\"}", c.Arguments);
    }

    [Fact]
    public void TruncatedGeneration_IsNotReported()
    {
        Assert.Null(_p.TryParse("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Par"));
        Assert.Null(_p.TryParse("<tool_call>get_weather<arg_key>ci"));
        Assert.Null(_p.TryParse("<tool_call>get_weather<arg_key>city</arg_key>"));
    }

    [Fact]
    public void ProseAndHermesJson_AreNotGlmCalls()
    {
        Assert.Null(_p.TryParse("It is sunny in Paris."));
        Assert.Null(_p.TryParse(""));
        Assert.Null(_p.TryParse("<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>"));   // JSON body: name contains braces/space
    }

    [Fact]
    public void IsToolCallStart_KeysOnTheOpenTag()
    {
        Assert.True(_p.IsToolCallStart("<tool_call>get_weather"));
        Assert.False(_p.IsToolCallStart("<tool_c"));
    }

    [Fact]
    public void Factory_PicksGlm_BeforeTheHermesCheck_ForTheRealTemplate()
    {
        Assert.IsType<GlmToolCallParser>(ToolCallParserFactory.Create(Architecture.DeepSeekV2, Fixture("glm-4.7-flash-chat-template.jinja")));
        // The neighbours are untouched.
        Assert.IsType<XmlToolCallParser>(ToolCallParserFactory.Create(Architecture.Llama, "<tool_call>{}</tool_call>"));
        Assert.IsType<QwenXmlToolCallParser>(ToolCallParserFactory.Create(Architecture.Llama, "<tool_call><function=x><parameter=y>"));
    }

    // ---------------------------------------------------------------- the real template round trip

    [Fact]
    public void RealTemplate_RendersToolsAndTheGenerationPrompt()
    {
        var t = new JinjaChatTemplate(Fixture("glm-4.7-flash-chat-template.jinja"), "", "");
        string prompt = t.Apply([new ChatMessage { Role = "user", Content = "weather in Paris?" }],
            new ChatTemplateOptions { Tools = Tools });
        Assert.Contains("<tools>", prompt, StringComparison.Ordinal);
        Assert.Contains("get_weather", prompt, StringComparison.Ordinal);
        Assert.EndsWith("<|assistant|><think>", prompt, StringComparison.Ordinal);
    }

    // ---------------------------------------------------------------- stop set

    [Fact]
    public void ToolChoiceBinder_DropsTheToolCallStop_ForGlm_SoParallelCallsSurvive()
    {
        var options = new InferenceOptions { StopSequences = ["<|im_end|>", "</tool_call>"] };
        ToolChoiceBinder.Apply(new ToolChoice.Auto(), Tools, new GlmToolCallParser(), ref options, out _);
        Assert.DoesNotContain("</tool_call>", options.StopSequences);
        Assert.Contains("<|im_end|>", options.StopSequences);
    }

    private sealed class EogTokenizer(int eos, int[] extra, params (string Text, int Id)[] specials) : ITokenizer
    {
        public int[] Encode(string text) => throw new NotSupportedException();
        public string Decode(ReadOnlySpan<int> tokenIds) => throw new NotSupportedException();
        public string DecodeToken(int id) => specials.FirstOrDefault(s => s.Id == id).Text ?? "x";
        public int VocabSize => 160000;
        public int BosTokenId => 0;
        public int EosTokenId => eos;
        public IReadOnlyList<int> ExtraEndOfGenerationTokenIds => extra;
        public int CountTokens(string text) => text.Length;
    }

    [Fact]
    public void EndOfGeneration_IncludesTheEotAndEomIdsTheModelFileDeclares()
    {
        // GLM-4.7-Flash: eos=<|endoftext|> 154820, eot=<|user|> 154827, eom=<|observation|> 154829.
        var tok = new EogTokenizer(154820, [154827, 154829], ("<|endoftext|>", 154820), ("<|user|>", 154827), ("<|observation|>", 154829));
        int[] ids = EndOfGenerationTokens.Resolve(tok);
        Assert.Equal(154820, ids[0]);
        Assert.Contains(154827, ids);
        Assert.Contains(154829, ids);
        var cond = EndOfGenerationTokens.CreateStopCondition(tok);
        Assert.Equal(StopResult.Stop, cond.ShouldStop(154827, [], ""));
        Assert.Equal(StopResult.Stop, cond.ShouldStop(154829, [], ""));
    }

    [Fact]
    public void EndOfGeneration_HarmonyVocabulary_NeverStopsOnEnd_EvenIfTheFileDeclaresItAsEot()
    {
        // #798 guard (llama.cpp llama-vocab.cpp): <|end|> closes the analysis message; the turn continues.
        var harmony = new EogTokenizer(102, [107], ("<|return|>", 102), ("<|end|>", 107), ("<|call|>", 112));
        int[] ids = EndOfGenerationTokens.Resolve(harmony);
        Assert.DoesNotContain(107, ids);
        Assert.Contains(102, ids);

        // A vocabulary without the Harmony pair (Phi-3 style: eot = <|end|>) keeps it.
        var phi = new EogTokenizer(32000, [32007], ("<|endoftext|>", 32000), ("<|end|>", 32007));
        Assert.Contains(32007, EndOfGenerationTokens.Resolve(phi));
    }

    [Fact]
    public void EndOfGeneration_ExtraIdsAreDeduplicated_AndAbsentMeansNone()
    {
        var dup = new EogTokenizer(5, [5, 7, 7]);
        Assert.Equal([5, 7], EndOfGenerationTokens.Resolve(dup));
        Assert.Equal([9], EndOfGenerationTokens.Resolve(new EogTokenizer(9, [])));
    }

    // ---------------------------------------------------------------- SmolLM3 tool rendering

    [Fact]
    public void SmolLm3_OpenAiStyleTools_AreRenderedIntoThePrompt()
    {
        var t = new JinjaChatTemplate(Fixture("smollm3-chat-template.jinja"), "", "<|im_end|>");
        string prompt = t.Apply([new ChatMessage { Role = "user", Content = "weather in Paris?" }],
            new ChatTemplateOptions { Tools = Tools, EnableThinking = false });
        Assert.Contains("### Tools", prompt, StringComparison.Ordinal);
        Assert.Contains("<tools>", prompt, StringComparison.Ordinal);
        Assert.Contains("get_weather", prompt, StringComparison.Ordinal);
        Assert.Contains("<tool_call>", prompt, StringComparison.Ordinal);
    }

    [Fact]
    public void SmolLm3_WithoutTools_NoToolSection_AndAnExplicitPythonBranchWins()
    {
        var t = new JinjaChatTemplate(Fixture("smollm3-chat-template.jinja"), "", "<|im_end|>");
        string none = t.Apply([new ChatMessage { Role = "user", Content = "hi" }], new ChatTemplateOptions());
        Assert.DoesNotContain("### Tools", none, StringComparison.Ordinal);

        var kwargs = new Dictionary<string, JsonElement>
        {
            ["python_tools"] = JsonDocument.Parse("""[{"name":"f"}]""").RootElement.Clone(),
        };
        string py = t.Apply([new ChatMessage { Role = "user", Content = "hi" }],
            new ChatTemplateOptions { Tools = Tools, TemplateKwargs = kwargs });
        Assert.Contains("<code>", py, StringComparison.Ordinal);                    // the python branch the client chose
        Assert.DoesNotContain("<tool_call>", py, StringComparison.Ordinal);         // and no auto-added xml branch
    }

    [Fact]
    public void GenerationBlocks_RenderTheirBodyAndHonourWhitespaceControl()
    {
        // HF's {% generation %} (assistant-mask marker): previously a parse error, which silently replaced the
        // whole template with the plain transcript.
        var msgs = new[] { new ChatMessage { Role = "user", Content = "x" } };
        Assert.Equal("a b c", new JinjaChatTemplate("a {% generation %}b{% endgeneration %} c", "", "").Apply(msgs, new ChatTemplateOptions()));
        Assert.Equal("ab", new JinjaChatTemplate("a {%- generation -%} b {%- endgeneration -%}", "", "").Apply(msgs, new ChatTemplateOptions()));
    }

    [Fact]
    public void OtherTemplates_AreNotGivenAnXmlToolsAlias()
    {
        // A template that already reads `tools` must be rendered exactly as before.
        var t = new JinjaChatTemplate("{% if tools %}T:{{ tools | length }}{% endif %}{% if xml_tools %}X{% endif %}", "", "");
        Assert.Equal("T:1", t.Apply([new ChatMessage { Role = "user", Content = "hi" }], new ChatTemplateOptions { Tools = Tools }));
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
    public async Task Streaming_GlmCall_IsHeldBackFromContent_AndReportedInTheFinalChunk()
    {
        string[] pieces = ["<tool_call>", "get_weather", "<arg_key>", "city", "</arg_key>", "<arg_value>", "Paris", "</arg_value>"];
        var ctx = new DefaultHttpContext();
        var body = new MemoryStream();
        ctx.Response.Body = body;
        await ChatCompletionEndpoint.WriteChatStreamAsync(
            new ChatCompletionRequest { Messages = [], Stream = true }, ctx, _ => Script(pieces), (work, _) => work(),
            "req", "m", Tools, new GlmToolCallParser(), ReasoningPlan.Disabled, CancellationToken.None);

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
        Assert.Equal("", content.ToString());
        Assert.Equal("get_weather", calls!.Value[0].GetProperty("function").GetProperty("name").GetString());
        Assert.Equal("{\"city\":\"Paris\"}", calls!.Value[0].GetProperty("function").GetProperty("arguments").GetString());
    }
}
