using System.Text;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.Reasoning;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.Reasoning;

/// <summary>
/// #798 golden fixtures: the decoded text the dotLLM server really produced (dnx build of this branch, Vulkan, greedy,
/// <c>reasoning_format: none</c>, tools kept in the prompt with <c>tool_choice: none</c> so the raw text is returned) for
/// gpt-oss-20b (Harmony) and Gemma-4 E4B (<c>&lt;|channel&gt;thought</c>), captured 2026-10-07. No weights, only the output text.
/// </summary>
public class RealOutputFixtureTests
{
    private static string Raw(string name)
        => File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Tokenizers", "ChatTemplates", "Fixtures", "real-output", name + ".raw.txt"))
            .Replace("\r\n", "\n", StringComparison.Ordinal);

    private static readonly ToolDefinition[] Weather =
        [new("get_weather", "w", """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""")];

    private static readonly ToolDefinition[] Alarm =
        [new("set_alarm", "a", """{"type":"object","properties":{"hour":{"type":"integer"},"label":{"type":"string"},"repeat":{"type":"boolean"},"days":{"type":"array","items":{"type":"string"}}},"required":["hour","label"]}""")];

    // Chunkings: one char, a "token-like" 3, a prime 7, and the whole text.
    public static TheoryData<string, int> Cases()
    {
        var d = new TheoryData<string, int>();
        foreach (string f in new[]
                 {
                     "gptoss-plain", "gptoss-toolcall", "gptoss-toolcall-typed",
                     "gemma4-think-plain", "gemma4-nothink-plain", "gemma4-think-toolcall",
                 })
            foreach (int piece in new[] { 1, 3, 7, int.MaxValue })
                d.Add(f, piece);
        return d;
    }

    private static IReasoningSplitter NewSplitter(string fixture)
        => fixture.StartsWith("gptoss", StringComparison.Ordinal)
            ? new HarmonySplitter()
            : new ReasoningSplitter(false, false, ReasoningMarkups.Gemma4Open, ReasoningMarkups.Gemma4Close);

    private static (string R, string C, IReasoningSplitter S) Run(string fixture, string text, int piece)
    {
        var s = NewSplitter(fixture);
        var r = new StringBuilder();
        var c = new StringBuilder();
        for (int i = 0; i < text.Length; i += Math.Min(piece, text.Length - i))
        {
            var ch = s.Feed(text.Substring(i, Math.Min(piece, text.Length - i)));
            r.Append(ch.Reasoning);
            c.Append(ch.Content);
        }
        var fin = s.Finish();
        return (r.Append(fin.Reasoning).ToString(), c.Append(fin.Content).ToString(), s);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void RealOutput_StreamedAnyChunking_EqualsOneShot_AndNeverLeaksMarkup(string fixture, int piece)
    {
        string text = Raw(fixture);
        var (er, ec, _) = Run(fixture, text, int.MaxValue);
        var (r, c, _) = Run(fixture, text, piece);
        Assert.Equal(er, r);
        Assert.Equal(ec, c);

        // Reasoning never carries markup; content carries markup only as a raw tool segment.
        Assert.DoesNotContain("<|channel", r, StringComparison.Ordinal);
        Assert.DoesNotContain("<channel|>", r + c, StringComparison.Ordinal);
        Assert.DoesNotContain("<|channel>", r + c, StringComparison.Ordinal);
        Assert.DoesNotContain("<|message|>", r, StringComparison.Ordinal);
        if (!fixture.Contains("toolcall", StringComparison.Ordinal))
            Assert.DoesNotContain("<|", c, StringComparison.Ordinal);
    }

    [Fact]
    public void GptOss_Plain_AnalysisThenFinal()
    {
        var (r, c, s) = Run("gptoss-plain", Raw("gptoss-plain"), int.MaxValue);
        Assert.Equal("The user asks: \"What is the capital of France? Answer in one word.\" The answer is \"Paris\". Just output that.", r);
        Assert.Equal("Paris", c);
        Assert.True(s.SawReasoning);
    }

    [Fact]
    public void GptOss_ToolCall_ReasoningSplit_CallParsed()
    {
        string text = Raw("gptoss-toolcall");
        var (r, c, _) = Run("gptoss-toolcall", text, int.MaxValue);
        Assert.Equal("We need to call get_weather function with city \"Paris\".", r);
        Assert.StartsWith("<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>", c, StringComparison.Ordinal);

        var call = Assert.Single(new HarmonyToolCallParser().TryParse(c)!);
        Assert.Equal("get_weather", call.FunctionName);
        Assert.Equal("{\"city\":\"Paris\"}", call.Arguments);

        // The parser also reads the raw, unsplit text identically.
        Assert.Equal(call, Assert.Single(new HarmonyToolCallParser().TryParse(text)!));
    }

    [Fact]
    public void GptOss_ToolCallTyped_ArgumentsKeepTheirTypes()
    {
        var (_, c, _) = Run("gptoss-toolcall-typed", Raw("gptoss-toolcall-typed"), int.MaxValue);
        var call = Assert.Single(((IToolCallParser)new HarmonyToolCallParser()).TryParse(c, Alarm)!);
        Assert.Equal("set_alarm", call.FunctionName);
        using var doc = System.Text.Json.JsonDocument.Parse(call.Arguments);
        Assert.Equal(7, doc.RootElement.GetProperty("hour").GetInt32());
        Assert.True(doc.RootElement.GetProperty("repeat").GetBoolean());
        Assert.Equal(2, doc.RootElement.GetProperty("days").GetArrayLength());
    }

    [Fact]
    public void Gemma4_ThinkPlain_ThoughtSplitOff_AnswerClean()
    {
        var (r, c, s) = Run("gemma4-think-plain", Raw("gemma4-think-plain"), int.MaxValue);
        Assert.StartsWith("Let $B$ be the cost of the bat", r, StringComparison.Ordinal);
        Assert.EndsWith("The question asks for the number only.", r, StringComparison.Ordinal);
        Assert.Equal("0.05", c);
        Assert.True(s.SawReasoning);
    }

    [Fact]
    public void Gemma4_NoThinking_IsPlainContent()
    {
        var (r, c, s) = Run("gemma4-nothink-plain", Raw("gemma4-nothink-plain"), int.MaxValue);
        Assert.Equal("", r);
        Assert.Equal("Paris", c);
        Assert.False(s.SawReasoning);
    }

    [Fact]
    public void Gemma4_ThinkToolCall_ThoughtSplit_CallParsedFromContent()
    {
        var (r, c, _) = Run("gemma4-think-toolcall", Raw("gemma4-think-toolcall"), int.MaxValue);
        Assert.StartsWith("1. **Analyze the user's request:**", r, StringComparison.Ordinal);
        Assert.Equal("<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>", c);
        var call = Assert.Single(new Gemma4ToolCallParser().TryParse(c)!);
        Assert.Equal("get_weather", call.FunctionName);
        Assert.Equal("{\"city\":\"Paris\"}", call.Arguments);
    }
}
