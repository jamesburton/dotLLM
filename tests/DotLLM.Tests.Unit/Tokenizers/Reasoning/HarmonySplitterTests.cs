using System.Text;
using DotLLM.Tokenizers.Reasoning;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.Reasoning;

/// <summary>#798: OpenAI Harmony (gpt-oss) channel splitting.</summary>
public class HarmonySplitterTests
{
    private const string AnalysisThenFinal =
        "<|channel|>analysis<|message|>The user asks the capital of France. Answer: Paris.<|end|>"
        + "<|start|>assistant<|channel|>final<|message|>The capital of France is Paris.";

    private const string AnalysisThenToolCall =
        "<|channel|>analysis<|message|>Need weather. Use the tool.<|end|>"
        + "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Paris\"}<|call|>";

    private static (string Reasoning, string Content) Streamed(string text, int piece)
    {
        var s = new HarmonySplitter();
        var r = new StringBuilder();
        var c = new StringBuilder();
        for (int i = 0; i < text.Length; i += piece)
        {
            var chunk = s.Feed(text.Substring(i, Math.Min(piece, text.Length - i)));
            r.Append(chunk.Reasoning);
            c.Append(chunk.Content);
        }
        var fin = s.Finish();
        r.Append(fin.Reasoning);
        c.Append(fin.Content);
        return (r.ToString(), c.ToString());
    }

    [Fact]
    public void AnalysisGoesToReasoning_FinalToContent()
    {
        var (r, c, s) = HarmonySplitter.Split(AnalysisThenFinal);
        Assert.Equal("The user asks the capital of France. Answer: Paris.", r);
        Assert.Equal("The capital of France is Paris.", c);
        Assert.True(s.SawReasoning);
        Assert.Equal(AnalysisThenFinal.IndexOf("<|end|>", StringComparison.Ordinal) + "<|end|>".Length, s.ReasoningRawLength);
    }

    [Fact]
    public void ToolCall_PassesThroughRawIntoContent_ReasoningSplitOff()
    {
        var (r, c, _) = HarmonySplitter.Split(AnalysisThenToolCall);
        Assert.Equal("Need weather. Use the tool.", r);
        Assert.Equal(
            "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Paris\"}<|call|>", c);
    }

    [Fact]
    public void RecipientInRoleHeader_IsAlsoAToolSegment()
    {
        // llama.cpp accepts the recipient before the channel as well as after it. The prompt already ends in
        // "<|start|>assistant", so the generated text starts at " to=...".
        const string text = " to=functions.get_weather<|channel|>commentary json<|message|>{\"city\":\"Rome\"}<|call|>";
        var (r, c, _) = HarmonySplitter.Split(text);
        Assert.Equal("", r);
        Assert.Equal(text.TrimStart(), c);

        var full = "<|start|>assistant to=functions.x<|channel|>commentary json<|message|>{}<|call|>";
        Assert.Equal(full, HarmonySplitter.Split(full).Content);
    }

    [Fact]
    public void CommentaryPreamble_WithoutRecipient_IsContent()
    {
        const string text =
            "<|channel|>analysis<|message|>think<|end|>"
            + "<|start|>assistant<|channel|>commentary<|message|>I'll look that up.<|end|>"
            + "<|start|>assistant<|channel|>commentary to=functions.f <|constrain|>json<|message|>{}<|call|>";
        var (r, c, _) = HarmonySplitter.Split(text);
        Assert.Equal("think", r);
        Assert.StartsWith("I'll look that up.<|start|>assistant<|channel|>commentary to=functions.f", c, StringComparison.Ordinal);
    }

    [Fact]
    public void ConsecutiveSameStreamMessages_AreJoinedWithABlankLine()
    {
        const string text =
            "<|channel|>analysis<|message|>one<|end|><|start|>assistant<|channel|>analysis<|message|>two<|end|>"
            + "<|start|>assistant<|channel|>commentary<|message|>pre<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>post";
        var (r, c, _) = HarmonySplitter.Split(text);
        Assert.Equal("one\n\ntwo", r);
        Assert.Equal("pre\n\npost", c);
    }

    [Fact]
    public void CutOffMidAnalysis_IsAllReasoning_EmptyContent_RawLengthIsEverything()
    {
        const string text = "<|channel|>analysis<|message|>still thinking about it";
        var (r, c, s) = HarmonySplitter.Split(text);
        Assert.Equal("still thinking about it", r);
        Assert.Equal("", c);
        Assert.Equal(text.Length, s.ReasoningRawLength);
    }

    [Fact]
    public void CutOffMidHeader_EmitsNothing()
    {
        var (r, c, _) = HarmonySplitter.Split("<|channel|>commentary to=functions.get_we");
        Assert.Equal("", r);
        Assert.Equal("", c);
    }

    [Fact]
    public void NonHarmonyOutput_IsPlainContent()
    {
        var (r, c, s) = HarmonySplitter.Split("Paris is the capital.");
        Assert.Equal("", r);
        Assert.Equal("Paris is the capital.", c);
        Assert.False(s.SawReasoning);

        // A lone "<" that never becomes a marker is still just text.
        Assert.Equal("<b>hi</b>", HarmonySplitter.Split("<b>hi</b>").Content);
    }

    [Fact]
    public void FinalFollowsAnalysisWithoutAnEndMarker_StillSplits()
    {
        var (r, c, _) = HarmonySplitter.Split("<|channel|>analysis<|message|>hmm<|channel|>final<|message|>ok");
        Assert.Equal("hmm", r);
        Assert.Equal("ok", c);
    }

    [Fact]
    public void ReturnTerminator_IfPresentInText_IsDropped()
    {
        Assert.Equal("done", HarmonySplitter.Split("<|channel|>final<|message|>done<|return|>").Content);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(7)]
    [InlineData(11)]
    public void Streaming_AnyChunking_EqualsOneShot(int piece)
    {
        foreach (string text in new[]
                 {
                     AnalysisThenFinal, AnalysisThenToolCall,
                     "<|channel|>analysis<|message|>  padded thought  \n<|end|><|start|>assistant<|channel|>final<|message|>  keep  ",
                     "<|channel|>analysis<|message|>cut off  ", "plain <b>text</b> here",
                     " to=functions.get_weather<|channel|>commentary json<|message|>{\"a\":1}<|call|>",
                 })
        {
            var (er, ec, _) = HarmonySplitter.Split(text);
            var (r, c) = Streamed(text, piece);
            Assert.Equal(er, r);
            Assert.Equal(ec, c);
        }
    }

    [Fact]
    public void Streaming_NeverEmitsAPartialMarker()
    {
        var s = new HarmonySplitter();
        var sb = new StringBuilder();
        foreach (char ch in AnalysisThenToolCall)
        {
            var chunk = s.Feed(ch.ToString());
            sb.Append(chunk.Reasoning).Append(chunk.Content);
            // At every point the emitted text has no unterminated "<|" marker fragment.
            string seen = sb.ToString();
            int lt = seen.LastIndexOf("<|", StringComparison.Ordinal);
            if (lt >= 0)
                Assert.Contains("|>", seen[lt..], StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ToolHeader_IsEmittedAtomically_WhenMessageStarts()
    {
        var s = new HarmonySplitter();
        string[] pieces = ["<|channel|>", "commentary", " to=functions.get_weather", " <|constrain|>", "json", "<|message|>", "{\"city\":", "\"Paris\"}"];
        var contentChunks = new List<string>();
        foreach (var p in pieces)
        {
            var c = s.Feed(p);
            if (c.Content.Length > 0)
                contentChunks.Add(c.Content);
        }
        // Nothing was emitted until the header was complete; the first chunk carries the whole marker.
        Assert.StartsWith("<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>", contentChunks[0], StringComparison.Ordinal);
    }

    [Fact]
    public void InReasoning_TracksTheAnalysisMessage()
    {
        var s = new HarmonySplitter();
        s.Feed("<|channel|>");
        Assert.False(s.InReasoning);
        s.Feed("analysis");
        Assert.True(s.InReasoning);
        s.Feed("<|message|>thinking");
        Assert.True(s.InReasoning);
        s.Feed("<|end|>");
        Assert.False(s.InReasoning);
    }

    [Theory]
    [InlineData("{{- '<|start|>assistant<|channel|>final<|message|>' }}{{ '<|end|>' }}", ReasoningMarkup.Harmony)]
    [InlineData("<|channel>thought\n x <channel|>", ReasoningMarkup.Gemma4Channel)]
    [InlineData("<|im_start|>assistant\n<think>\n", ReasoningMarkup.Think)]
    [InlineData("", ReasoningMarkup.Think)]
    public void Detect_FromTemplateSource(string template, ReasoningMarkup expected)
        => Assert.Equal(expected, ReasoningMarkups.Detect(template));

    [Fact]
    public void FilterStops_RemovesEndOnlyForHarmony()
    {
        string[] stops = ["<|im_end|>", "<|end|>", "</s>"];
        Assert.Equal(["<|im_end|>", "</s>"], ReasoningMarkup.Harmony.FilterStops(stops));
        Assert.Same(stops, ReasoningMarkup.Think.FilterStops(stops));
        Assert.Same(stops, ReasoningMarkup.Gemma4Channel.FilterStops(stops));
    }
}
