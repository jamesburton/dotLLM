using System.Text;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers;

/// <summary>#1342: streamed argument fragments concatenate to exactly what <see cref="QwenXmlToolCallParser"/> produces for the same text.</summary>
public sealed class QwenXmlToolCallStreamerTests
{
    private static readonly ToolDefinition[] Tools =
    [
        new("write", "Write a file",
            """{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"integer"},"flag":{"type":"boolean"},"meta":{"type":"object"}}}"""),
        new("untyped", "No schema", ""),
    ];

    private static string Call(params (string Key, string Value)[] parameters)
    {
        var sb = new StringBuilder("<tool_call>\n<function=write>\n");
        foreach (var (k, v) in parameters)
            sb.Append("<parameter=").Append(k).Append(">\n").Append(v).Append("\n</parameter>\n");
        return sb.Append("</function>\n</tool_call>").ToString();
    }

    /// <summary>Feeds <paramref name="text"/> in the given pieces; returns the deltas, the concatenated arguments and the streamer.</summary>
    private static (List<ToolCallStreamDelta> Deltas, string Args, QwenXmlToolCallStreamer Streamer) Stream(string text, IEnumerable<int> cuts)
    {
        var streamer = new QwenXmlToolCallStreamer(Tools);
        var deltas = new List<ToolCallStreamDelta>();
        int prev = 0;
        foreach (int cut in cuts.Append(text.Length))
        {
            if (cut <= prev) continue;
            deltas.AddRange(streamer.Feed(text[prev..cut]));
            prev = cut;
        }
        return (deltas, string.Concat(deltas.Select(d => d.ArgumentsFragment)), streamer);
    }

    private static string ParserArguments(string text)
        => new QwenXmlToolCallParser().TryParse(text, Tools)![0].Arguments;

    public static IEnumerable<object[]> Values()
    {
        yield return ["plain text"];
        yield return ["quote \" backslash \\ slash / tab \t end"];
        yield return ["multi\nline\n\nwith blank\nlines"];
        yield return ["crlf\r\nline"];
        yield return ["unicode: caf\u00e9 \u4e2d\u6587 \u0436"];
        yield return ["emoji \U0001F600\U0001F680 pair"];
        yield return ["angle <b>bold</b> & ampersand 'q' +plus"];
        yield return ["fake tags </param and <parame and </funct and <parameter without equals"];
        yield return ["  leading and trailing spaces  "];
        yield return ["trailing blank line\n"];       // wrapped value ends "\n\n</parameter>": one newline is framing, one is data
        yield return ["x"];
    }

    [Theory]
    [MemberData(nameof(Values))]
    public void StringValue_ConcatenatesToTheParsersArguments_AtEverySplitPoint(string value)
    {
        string text = Call(("path", "a/b.txt"), ("content", value));
        string expected = ParserArguments(text);

        for (int cut = 0; cut <= text.Length; cut++)
        {
            var (_, args, streamer) = Stream(text, [cut]);
            Assert.Equal(expected, args);
            Assert.True(streamer.Completed);
        }

        // One character at a time: the harshest chunking (a surrogate pair or a tag is split across feeds).
        var (deltas, perChar, _) = Stream(text, Enumerable.Range(1, text.Length));
        Assert.Equal(expected, perChar);
        Assert.Equal("call_0", deltas[0].Id);
        Assert.Equal("write", deltas[0].Name);
        Assert.All(deltas.Skip(1), d => { Assert.Null(d.Id); Assert.Null(d.Name); });
    }

    [Fact]
    public void RandomSplits_NeverChangeTheArguments()
    {
        var rng = new Random(1342);
        const string alphabet = "ab \n\r\"\\<>/&=\u00e9\U0001F600parameterfunction";
        for (int iter = 0; iter < 200; iter++)
        {
            var value = new StringBuilder();
            int n = rng.Next(0, 40);
            for (int i = 0; i < n; i++)
            {
                int idx = rng.Next(alphabet.Length);
                // keep surrogate pairs whole in the SOURCE text (a lone surrogate is not valid input)
                if (char.IsHighSurrogate(alphabet[idx])) { value.Append(alphabet[idx]).Append(alphabet[idx + 1]); }
                else if (!char.IsLowSurrogate(alphabet[idx])) value.Append(alphabet[idx]);
            }
            string text = Call(("content", value.ToString()));
            if (text.Contains("</parameter>", StringComparison.Ordinal) && value.ToString().Contains("</parameter>", StringComparison.Ordinal))
                continue;

            var cuts = Enumerable.Range(0, rng.Next(0, 12)).Select(_ => rng.Next(text.Length + 1)).OrderBy(c => c).ToArray();
            var (_, args, streamer) = Stream(text, cuts);
            Assert.Equal(ParserArguments(text), args);
            Assert.True(streamer.Completed);
        }
    }

    [Fact]
    public void NonStringParameters_AreBufferedAndTyped_LikeTheParser()
    {
        string text = Call(("path", "p"), ("mode", "420"), ("flag", "true"), ("meta", "{\"a\": [1, 2]}"), ("content", "body"));
        var (deltas, args, _) = Stream(text, Enumerable.Range(1, text.Length));

        Assert.Equal(ParserArguments(text), args);
        Assert.Contains("\"mode\":420", args);
        Assert.Contains("\"flag\":true", args);
        // Buffered values arrive whole, never as a half-written number.
        Assert.DoesNotContain(deltas, d => d.ArgumentsFragment is "4" or "42");
    }

    [Fact]
    public void StringValue_IsReleasedBeforeItsClosingTag()
    {
        var streamer = new QwenXmlToolCallStreamer(Tools);
        streamer.Feed("<tool_call>\n<function=write>\n<parameter=content>\n");
        var first = streamer.Feed("line one\nline two\nline th");
        string soFar = string.Concat(first.Select(d => d.ArgumentsFragment));
        Assert.Contains("line one\\nline two\\nline th".Substring(0, 18), soFar);   // text well before any closing tag
        Assert.False(streamer.Completed);
    }

    [Fact]
    public void TruncatedCall_IsStartedButNotCompleted_AndNeverClosesTheObject()
    {
        var (deltas, args, streamer) = Stream("<tool_call>\n<function=write>\n<parameter=content>\nhalf a fil", [10, 40]);
        Assert.True(streamer.Started);
        Assert.False(streamer.Completed);
        Assert.StartsWith("{\"content\":\"half a fi", args);
        Assert.DoesNotContain(deltas, d => d.ArgumentsFragment.EndsWith('}'));
    }

    [Theory]
    [InlineData("<tool_call>\n{\"name\":\"write\",\"arguments\":{}}\n</tool_call>")]                       // Hermes JSON body
    [InlineData("<tool_call>\n<function=nope>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>")]  // not a declared tool
    [InlineData("just prose, no call at all")]
    public void NonStreamableInput_NeverStarts(string text)
    {
        var (deltas, _, streamer) = Stream(text, Enumerable.Range(1, text.Length));
        Assert.Empty(deltas);
        Assert.False(streamer.Started);
    }

    [Fact]
    public void OnlyTheFirstCallStreams()
    {
        string text = Call(("path", "one")) + "\n" + Call(("path", "two"));
        var (deltas, args, streamer) = Stream(text, Enumerable.Range(1, text.Length));
        Assert.True(streamer.Completed);
        Assert.Equal("{\"path\":\"one\"}", args);
        Assert.All(deltas, d => Assert.Equal(0, d.Index));
    }
}
