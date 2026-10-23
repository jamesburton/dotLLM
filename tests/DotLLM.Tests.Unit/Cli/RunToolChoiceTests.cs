using DotLLM.Engine.Constraints;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Cli;

/// <summary>
/// Verifies the constrained tool-calling round-trip used by <c>run --tool-choice required</c>:
/// <c>ToolCallSchemaBuilder</c> produces a schema for constrained decoding, and
/// <c>GenericToolCallParser</c> correctly parses the bare-JSON output the constrained decoder emits.
/// </summary>
public sealed class RunToolChoiceTests
{
    private static readonly ToolDefinition WeatherTool =
        new("get_weather", "Get current weather for a city",
            """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""");

    private static readonly ToolDefinition[] SingleTools = [WeatherTool];

    private static readonly ToolDefinition[] MultiTools =
    [
        new("get_weather", "Weather",  "{\"type\":\"object\"}"),
        new("search_web",  "Search",   "{\"type\":\"object\"}")
    ];

    [Fact]
    public void BuildForRequired_SingleTool_ProducesNonEmptySchema()
    {
        string schema = ToolCallSchemaBuilder.BuildForRequired(SingleTools, "arguments");
        Assert.NotEmpty(schema);
        Assert.Contains("get_weather", schema, StringComparison.Ordinal);
        Assert.Contains("arguments", schema, StringComparison.Ordinal);
    }

    [Fact]
    public void BuildForRequired_MultiTool_ContainsAllNames()
    {
        string schema = ToolCallSchemaBuilder.BuildForRequired(MultiTools, "arguments");
        Assert.Contains("get_weather", schema, StringComparison.Ordinal);
        Assert.Contains("search_web", schema, StringComparison.Ordinal);
    }

    [Fact]
    public void BuildForFunction_ProducesConstSchema()
    {
        string schema = ToolCallSchemaBuilder.BuildForFunction(WeatherTool, "arguments");
        Assert.Contains("get_weather", schema, StringComparison.Ordinal);
        // Single-function schema uses const for name
        Assert.Contains("const", schema, StringComparison.Ordinal);
    }

    [Fact]
    public void GenericToolCallParser_ParsesBareJson_WithArgumentsKey()
    {
        const string bareJson = """{"name":"get_weather","arguments":{"city":"Tokyo"}}""";
        var parser = new GenericToolCallParser();

        var calls = parser.TryParse(bareJson);

        Assert.NotNull(calls);
        Assert.Single(calls);
        Assert.Equal("get_weather", calls![0].FunctionName);
        Assert.Contains("Tokyo", calls[0].Arguments, StringComparison.Ordinal);
    }

    [Fact]
    public void GenericToolCallParser_ParsesBareJson_WithParametersKey()
    {
        const string bareJson = """{"name":"search_web","parameters":{"query":"dotnet"}}""";
        var parser = new GenericToolCallParser();

        var calls = parser.TryParse(bareJson);

        Assert.NotNull(calls);
        Assert.Single(calls);
        Assert.Equal("search_web", calls![0].FunctionName);
        Assert.Contains("dotnet", calls[0].Arguments, StringComparison.Ordinal);
    }
}
