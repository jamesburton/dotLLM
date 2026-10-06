using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ToolCallParsers;

public class ToolArgumentCoercerTests
{
    private static ToolDefinition Tool(string propsJson)
        => new("f", "d", $$"""{"type":"object","properties":{{propsJson}}}""");

    private static string Coerce(string args, string propsJson)
        => ToolArgumentCoercer.Coerce([new ToolCall("c", "f", args)], [Tool(propsJson)])[0].Arguments;

    [Theory]
    [InlineData("""{"n":"17"}""", """{"n":{"type":"integer"}}""", """{"n":17}""")]
    [InlineData("""{"n":"2.5"}""", """{"n":{"type":"number"}}""", """{"n":2.5}""")]
    [InlineData("""{"n":3.0}""", """{"n":{"type":"integer"}}""", """{"n":3}""")]
    [InlineData("""{"b":"TRUE"}""", """{"b":{"type":"boolean"}}""", """{"b":true}""")]
    [InlineData("""{"s":12345}""", """{"s":{"type":"string"}}""", """{"s":"12345"}""")]
    [InlineData("""{"s":true}""", """{"s":{"type":"string"}}""", """{"s":"true"}""")]
    [InlineData("""{"a":"[1,2]"}""", """{"a":{"type":"array"}}""", """{"a":[1,2]}""")]
    [InlineData("""{"o":"{\"k\":1}"}""", """{"o":{"type":"object"}}""", """{"o":{"k":1}}""")]
    [InlineData("""{"n":"abc"}""", """{"n":{"type":"integer"}}""", """{"n":"abc"}""")]          // cannot coerce: untouched
    [InlineData("""{"n":"3.5"}""", """{"n":{"type":"integer"}}""", """{"n":"3.5"}""")]            // not integral: untouched
    [InlineData("""{"s":"17"}""", """{"s":{"type":["string","integer"]}}""", """{"s":"17"}""")]    // string allowed: verbatim
    [InlineData("""{"u":"5"}""", """{"u":{"anyOf":[{"type":"integer"},{"type":"null"}]}}""", """{"u":5}""")]
    [InlineData("""{"x":"5"}""", """{"y":{"type":"integer"}}""", """{"x":"5"}""")]                // unknown property
    public void Coerces_ByDeclaredType(string args, string props, string expected)
        => Assert.Equal(expected, Coerce(args, props));

    [Fact]
    public void Recurses_IntoNestedObjectsAndArrayItems()
    {
        string props = """{"o":{"type":"object","properties":{"n":{"type":"integer"}}},"a":{"type":"array","items":{"type":"number"}}}""";
        Assert.Equal("""{"o":{"n":4},"a":[1,2.5]}""", Coerce("""{"o":{"n":"4"},"a":["1","2.5"]}""", props));
    }

    [Fact]
    public void NoTools_OrUnknownTool_ReturnsSameInstance()
    {
        var calls = new[] { new ToolCall("c", "g", """{"n":"1"}""") };
        Assert.Same(calls, ToolArgumentCoercer.Coerce(calls, null));
        Assert.Same(calls, ToolArgumentCoercer.Coerce(calls, [Tool("""{"n":{"type":"integer"}}""")]));
    }

    [Theory]
    [InlineData("not json")]
    [InlineData("[1,2]")]
    [InlineData("")]
    public void UnusableArguments_AreReturnedUnchanged(string args)
        => Assert.Equal(args, Coerce(args, """{"n":{"type":"integer"}}"""));

    [Fact]
    public void BadSchema_IsIgnored()
    {
        var calls = new[] { new ToolCall("c", "f", """{"n":"1"}""") };
        var tools = new[] { new ToolDefinition("f", "d", "{not a schema") };
        Assert.Equal("""{"n":"1"}""", ToolArgumentCoercer.Coerce(calls, tools)[0].Arguments);
    }
}
