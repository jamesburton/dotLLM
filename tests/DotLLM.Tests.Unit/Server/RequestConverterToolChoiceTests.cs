using System.Text.Json;
using DotLLM.Server;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Coverage for <see cref="RequestConverter.IsToolChoiceSupported"/> — the 400 gate that prevents
/// clients from receiving silent <c>auto</c> behaviour when they ask for <c>required</c>, a
/// specific function, or any value other than <c>auto</c>/unset (issue #121, item 3).
/// </summary>
public sealed class RequestConverterToolChoiceTests
{
    [Fact]
    public void Unset_IsSupported()
    {
        Assert.True(RequestConverter.IsToolChoiceSupported(null, out var v));
        Assert.Equal(string.Empty, v);
    }

    [Fact]
    public void Auto_IsSupported()
    {
        var el = JsonDocument.Parse("\"auto\"").RootElement;
        Assert.True(RequestConverter.IsToolChoiceSupported(el, out var v));
        Assert.Equal(string.Empty, v);
    }

    [Theory]
    [InlineData("\"none\"", "none")]
    [InlineData("\"required\"", "required")]
    [InlineData("\"bogus\"", "bogus")]
    public void StringNonAuto_IsRejected(string json, string expectedRejected)
    {
        var el = JsonDocument.Parse(json).RootElement;
        Assert.False(RequestConverter.IsToolChoiceSupported(el, out var v));
        Assert.Equal(expectedRejected, v);
    }

    [Fact]
    public void SpecificFunction_IsRejected()
    {
        var el = JsonDocument.Parse(
            """{"type":"function","function":{"name":"get_weather"}}""").RootElement;
        Assert.False(RequestConverter.IsToolChoiceSupported(el, out var v));
        Assert.Equal("function:get_weather", v);
    }
}
