using DotLLM.Tokenizers.Reasoning;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.Reasoning;

/// <summary>Unit tests for the streaming reasoning/content splitter (#767).</summary>
public class ReasoningSplitterTests
{
    private static (string Reasoning, string Content) Stream(
        IEnumerable<string> pieces, bool startInReasoning, bool detectAnywhere = false)
    {
        var s = new ReasoningSplitter(startInReasoning, detectAnywhere);
        string r = "", c = "";
        foreach (var p in pieces)
        {
            var ch = s.Feed(p);
            r += ch.Reasoning; c += ch.Content;
        }
        var fin = s.Finish();
        return (r + fin.Reasoning, c + fin.Content);
    }

    private static IEnumerable<string> Chars(string s) => s.Select(ch => ch.ToString());

    [Fact]
    public void TemplateOpenedThink_SplitsAtClose_AndDropsSeparators()
    {
        var (r, c, _) = ReasoningSplitter.Split("We need to answer.\n</think>\n\nThe capital is Paris.", startInReasoning: true);
        Assert.Equal("We need to answer.", r);
        Assert.Equal("The capital is Paris.", c);
    }

    [Fact]
    public void ModelEmitsThinkItself_WhenPromptDidNotOpenIt()
    {
        var (r, c, s) = ReasoningSplitter.Split("<think>\nhmm\n</think>\n\nAnswer", startInReasoning: false);
        Assert.Equal("hmm", r);
        Assert.Equal("Answer", c);
        Assert.True(s.SawReasoning);
    }

    [Fact]
    public void NoThinkAtAll_PassesThroughVerbatim_IncludingLeadingWhitespace()
    {
        var (r, c, s) = ReasoningSplitter.Split("  hello <think> not a block", startInReasoning: false);
        Assert.Equal("", r);
        Assert.Equal("  hello <think> not a block", c);
        Assert.False(s.SawReasoning);
    }

    [Fact]
    public void ThinkingDisabledTemplate_ClosedBlockInPrompt_AllContent()
    {
        Assert.False(ReasoningFormats.PromptOpensThinking("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        var (r, c, _) = ReasoningSplitter.Split("Paris.", startInReasoning: false);
        Assert.Equal("", r);
        Assert.Equal("Paris.", c);
    }

    [Fact]
    public void UnclosedThink_MaxTokens_AllReasoningNoContent()
    {
        var (r, c, s) = ReasoningSplitter.Split("still thinking about </thi", startInReasoning: true);
        Assert.Equal("still thinking about </thi", r);
        Assert.Equal("", c);
        Assert.Equal("still thinking about </thi".Length, s.ReasoningRawLength);
    }

    [Fact]
    public void ReasoningRawLength_CoversThroughClosingTag()
    {
        const string text = "abc</think>\n\nxyz";
        var (_, _, s) = ReasoningSplitter.Split(text, startInReasoning: true);
        Assert.Equal("abc</think>".Length, s.ReasoningRawLength);
    }

    [Fact]
    public void EmptyThinkBlock_ThinkingOffStyleOutput()
    {
        var (r, c, _) = ReasoningSplitter.Split("<think>\n\n</think>\n\nOK", startInReasoning: false);
        Assert.Equal("", r);
        Assert.Equal("OK", c);
    }

    public static IEnumerable<object[]> Cases()
    {
        yield return ["We need to answer.\n</think>\n\nThe capital is Paris.", true, false];
        yield return ["<think>\nhmm</think>Answer", false, false];
        yield return ["plain answer <think> literal", false, false];
        yield return ["r1 </think> a1 <think> r2 </think> a2", true, true];
        yield return ["a0 <think>r1</think>a1<think>r2</think>a2", false, true];
        yield return ["  \n  <think>x</think> y", false, false];
        yield return ["thinking with <b>html</b> and </thin and <think", true, false];
        yield return ["x</think>", true, false];
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void EverySplitPoint_ChunkedEqualsOneShot(string text, bool start, bool anywhere)
    {
        var (er, ec, _) = ReasoningSplitter.Split(text, start, anywhere);
        for (int cut = 0; cut <= text.Length; cut++)
        {
            var (r, c) = Stream([text[..cut], text[cut..]], start, anywhere);
            Assert.True(er == r && ec == c, $"cut {cut}: expected ({er}|{ec}) got ({r}|{c})");
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void CharByChar_ChunkedEqualsOneShot(string text, bool start, bool anywhere)
    {
        var (er, ec, _) = ReasoningSplitter.Split(text, start, anywhere);
        var (r, c) = Stream(Chars(text), start, anywhere);
        Assert.Equal(er, r);
        Assert.Equal(ec, c);
    }

    [Fact]
    public void CloseTagSplitAcrossTokens_NeverLeaksPartialTag()
    {
        var s = new ReasoningSplitter(startInReasoning: true);
        var a = s.Feed("deep thought </th");
        Assert.DoesNotContain("<", a.Reasoning);       // "</th" is held back
        Assert.Equal("deep thought", a.Reasoning);       // trailing space held too
        var b = s.Feed("ink>\n\nHi");
        Assert.Equal("", b.Reasoning);
        Assert.Equal("Hi", b.Content);
    }

    [Fact]
    public void FalseAlarmPartialTag_IsEventuallyEmitted()
    {
        var s = new ReasoningSplitter(startInReasoning: true);
        var a = s.Feed("a </th");
        var b = s.Feed("ought");
        Assert.Equal("a </thought", a.Reasoning + b.Reasoning + s.Finish().Reasoning);
    }

    [Fact]
    public void Auto_IgnoresThinkAfterAnswerStarted_Deepseek_Recognises()
    {
        const string text = "answer <think>aside</think> more";
        var (ar, ac, _) = ReasoningSplitter.Split(text, startInReasoning: false, detectAnywhere: false);
        Assert.Equal("", ar);
        Assert.Equal(text, ac);

        var (dr, dc, _) = ReasoningSplitter.Split(text, startInReasoning: false, detectAnywhere: true);
        Assert.Equal("aside", dr);
        Assert.Equal("answer more", dc);
    }

    [Theory]
    [InlineData("<|im_start|>assistant\n<think>\n", true)]
    [InlineData("<|im_start|>assistant\n<think>", true)]
    [InlineData("<|im_start|>assistant\n<think>\n\n</think>\n\n", false)]
    [InlineData("<|im_start|>assistant\n", false)]
    [InlineData("<|im_start|>user\nwhat does <think> mean?<|im_end|>\n<|im_start|>assistant\n", false)]
    [InlineData("<|im_start|>assistant\n<think>\nold</think>\n\nhi<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\n", true)]
    public void PromptOpensThinking(string prompt, bool expected)
        => Assert.Equal(expected, ReasoningFormats.PromptOpensThinking(prompt));

    [Fact]
    public void CreateSplitter_NoneYieldsNull()
    {
        Assert.Null(ReasoningFormats.CreateSplitter(ReasoningFormat.None, "x<think>\n"));
        Assert.NotNull(ReasoningFormats.CreateSplitter(ReasoningFormat.Auto, "x"));
    }

    [Theory]
    [InlineData("none", ReasoningFormat.None)]
    [InlineData("AUTO", ReasoningFormat.Auto)]
    [InlineData("deepseek", ReasoningFormat.Deepseek)]
    public void Parse_RoundTrips(string text, ReasoningFormat expected)
    {
        Assert.Equal(expected, ReasoningFormats.Parse(text));
        Assert.Equal(expected, ReasoningFormats.Parse(expected.ToWireString()));
    }

    [Fact]
    public void Parse_Unknown_Throws() => Assert.Throws<ArgumentException>(() => ReasoningFormats.Parse("bogus"));
}
