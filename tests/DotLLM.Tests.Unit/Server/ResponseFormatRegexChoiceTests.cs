using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Server;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// <c>response_format</c> extensions for classifier-style callers: <c>{"type":"regex","pattern":...}</c> and
/// <c>{"type":"choice","choices":[...]}</c>. Both reach the engine as a <see cref="ResponseFormat.Regex"/>, so the
/// output is the bare matched text and generation stops the moment the match is complete.
/// </summary>
public sealed class ResponseFormatRegexChoiceTests
{
    private static ResponseFormat? Parse(string json)
    {
        using var doc = JsonDocument.Parse(json);
        return RequestConverter.ParseResponseFormat(doc.RootElement.Clone());
    }

    [Fact]
    public void RegexType_ParsesThePattern()
    {
        var rf = Parse("""{"type":"regex","pattern":"[AB]"}""");
        var rx = Assert.IsType<ResponseFormat.Regex>(rf);
        Assert.Equal("[AB]", rx.Pattern);
    }

    [Fact]
    public void ChoiceType_BuildsAnAlternation()
    {
        var rf = Parse("""{"type":"choice","choices":["A","B","C"]}""");
        var rx = Assert.IsType<ResponseFormat.Regex>(rf);
        Assert.Equal("(A|B|C)", rx.Pattern);
    }

    [Fact]
    public void ChoiceType_EscapesRegexMetacharacters()
    {
        var rf = Parse("""{"type":"choice","choices":["a.b","C+","(x)"]}""");
        var rx = Assert.IsType<ResponseFormat.Regex>(rf);
        Assert.Equal(@"(a\.b|C\+|\(x\))", rx.Pattern);
    }

    [Theory]
    [InlineData("""{"type":"regex"}""")]
    [InlineData("""{"type":"regex","pattern":5}""")]
    [InlineData("""{"type":"choice"}""")]
    [InlineData("""{"type":"choice","choices":[]}""")]
    public void MalformedExtensions_AreIgnoredRatherThanThrowing(string json)
        => Assert.Null(Parse(json));

    [Fact]
    public void ExistingFormats_AreUnchanged()
    {
        Assert.IsType<ResponseFormat.JsonObject>(Parse("""{"type":"json_object"}"""));
        Assert.Null(Parse("""{"type":"text"}"""));
    }
}
