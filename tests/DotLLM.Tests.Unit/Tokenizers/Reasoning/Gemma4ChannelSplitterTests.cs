using System.Text;
using DotLLM.Tokenizers.Reasoning;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.Reasoning;

/// <summary>#798: Gemma-4 <c>&lt;|channel&gt;thought…&lt;channel|&gt;</c> markup, via the shared tag splitter.</summary>
public class Gemma4ChannelSplitterTests
{
    private const string Open = ReasoningMarkups.Gemma4Open;
    private const string Close = ReasoningMarkups.Gemma4Close;

    private static (string R, string C, ReasoningSplitter S) Split(string text, bool startInside = false, bool anywhere = false)
        => ReasoningSplitter.Split(text, startInside, anywhere, Open, Close);

    private static (string R, string C) Streamed(string text, int piece, bool startInside = false)
    {
        var s = new ReasoningSplitter(startInside, false, Open, Close);
        var r = new StringBuilder();
        var c = new StringBuilder();
        for (int i = 0; i < text.Length; i += piece)
        {
            var ch = s.Feed(text.Substring(i, Math.Min(piece, text.Length - i)));
            r.Append(ch.Reasoning);
            c.Append(ch.Content);
        }
        var fin = s.Finish();
        return (r.Append(fin.Reasoning).ToString(), c.Append(fin.Content).ToString());
    }

    [Fact]
    public void ThoughtChannel_IsReasoning_AnswerIsContent()
    {
        var (r, c, s) = Split("<|channel>thought\nThe user wants the capital.\n<channel|>Paris.");
        Assert.Equal("The user wants the capital.", r);
        Assert.Equal("Paris.", c);
        Assert.True(s.SawReasoning);
    }

    [Fact]
    public void EmptyThoughtChannel_YieldsOnlyContent()
    {
        // enable_thinking=false: some Gemma-4 builds still emit an empty channel.
        var (r, c, s) = Split("<|channel>thought\n<channel|>Paris.");
        Assert.Equal("", r);
        Assert.Equal("Paris.", c);
        Assert.True(s.SawReasoning);
    }

    [Fact]
    public void NoChannel_IsPlainContent()
    {
        var (r, c, s) = Split("Paris.");
        Assert.Equal("", r);
        Assert.Equal("Paris.", c);
        Assert.False(s.SawReasoning);
    }

    [Fact]
    public void ToolCallAfterThought_StaysInContentForTheParser()
    {
        var (r, c, _) = Split("<|channel>thought\nneed weather<channel|><|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>");
        Assert.Equal("need weather", r);
        Assert.Equal("<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>", c);
    }

    [Fact]
    public void PromptEndingInAnOpenChannel_StartsInsideReasoning()
    {
        // After a tool response with thinking on, the template's generation prompt is "<|channel>thought\n".
        Assert.True(ReasoningFormats.PromptOpensThinking("...<|turn>model\n<|channel>thought\n", Open, Close));
        Assert.False(ReasoningFormats.PromptOpensThinking("...<|channel>thought\nx<channel|>", Open, Close));
        Assert.False(ReasoningFormats.PromptOpensThinking("...<|turn>model\n", Open, Close));

        var (r, c, _) = Split("still going\n<channel|>Paris.", startInside: true);
        Assert.Equal("still going", r);
        Assert.Equal("Paris.", c);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(9)]
    public void Streaming_AnyChunking_EqualsOneShot_NoTagFragmentLeaks(int piece)
    {
        foreach (string text in new[]
                 {
                     "<|channel>thought\nThe user wants the capital.\n<channel|>Paris.",
                     "<|channel>thought\n<channel|>Paris.",
                     "<|channel>thought\nunfinished <chan",
                     "Paris is <|channel> not a block",
                 })
        {
            var (er, ec, _) = Split(text);
            var (r, c) = Streamed(text, piece);
            Assert.Equal(er, r);
            Assert.Equal(ec, c);
            Assert.DoesNotContain("<|channel>", r, StringComparison.Ordinal);
            Assert.DoesNotContain("<channel|>", r + c, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ThinkMarkupIsUnaffected()
    {
        var (r, c, _) = ReasoningSplitter.Split("hmm\n</think>\n\nOK", startInReasoning: true);
        Assert.Equal("hmm", r);
        Assert.Equal("OK", c);
    }

    [Fact]
    public void Tags_PerMarkup()
    {
        Assert.Equal("<think>", ReasoningMarkup.Think.OpenTag());
        Assert.Equal("</think>", ReasoningMarkup.Think.CloseTag());
        Assert.Equal("<|channel>thought", ReasoningMarkup.Gemma4Channel.OpenTag());
        Assert.Equal("<channel|>", ReasoningMarkup.Gemma4Channel.CloseTag());
    }
}
