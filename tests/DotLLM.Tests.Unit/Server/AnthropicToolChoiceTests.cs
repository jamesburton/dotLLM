using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Tests for Anthropic <c>tool_choice</c> handling on <c>/v1/messages</c> (#449).
/// </summary>
/// <remarks>
/// <see cref="AnthropicConverter.ParseToolChoice"/> existed from #448 but nothing called it, so
/// <c>{"type":"tool","name":…}</c> behaved exactly like <c>auto</c> and <c>none</c> did not stop
/// a tool call being reported. These tests pin the effect of the choice — the decoding constraint
/// that is installed and the parser that is used — not merely that the JSON parses.
/// </remarks>
public sealed class AnthropicToolChoiceTests
{
    private static readonly ToolDefinition[] Tools =
    [
        new("get_weather", "weather", """{"type":"object","properties":{"city":{"type":"string"}}}"""),
        new("get_time", "time", """{"type":"object","properties":{"tz":{"type":"string"}}}"""),
    ];

    private static InferenceOptions BaseOptions() => new() { MaxTokens = 16 };

    private static ToolChoice Choice(string json) =>
        AnthropicConverter.ParseToolChoice(JsonDocument.Parse(json).RootElement.Clone());

    [Fact]
    public void ToolChoice_Auto_LeavesDecodingUnconstrained()
    {
        var options = BaseOptions();
        var model = new HermesToolCallParser();

        var parser = MessagesEndpoint.ApplyToolChoice(Choice("""{"type":"auto"}"""), Tools, model, ref options, out bool forced);

        Assert.Same(model, parser);
        Assert.Null(options.ResponseFormat);
        Assert.False(forced);
    }

    [Fact]
    public void ToolChoice_None_SuppressesToolCallDetection()
    {
        var options = BaseOptions();

        var parser = MessagesEndpoint.ApplyToolChoice(
            Choice("""{"type":"none"}"""), Tools, new HermesToolCallParser(), ref options, out bool forced);

        // No parser means nothing the model emits can be reported as a tool_use block.
        Assert.Null(parser);
        Assert.Null(options.ResponseFormat);
        Assert.False(forced);
    }

    [Fact]
    public void ToolChoice_Tool_ConstrainsDecodingToThatToolsSchema()
    {
        var options = BaseOptions();

        var parser = MessagesEndpoint.ApplyToolChoice(
            Choice("""{"type":"tool","name":"get_time"}"""), Tools, new HermesToolCallParser(), ref options, out bool forced);

        var format = Assert.IsType<ResponseFormat.JsonSchema>(options.ResponseFormat);
        Assert.Contains("get_time", format.Schema);
        // The named tool is the only one the constraint admits.
        Assert.DoesNotContain("get_weather", format.Schema);
        // The constraint emits a bare JSON object, not the model's <tool_call> envelope, so the
        // markerless parser must be used or the call is parsed back as plain text.
        Assert.IsType<GenericToolCallParser>(parser);
        // The whole completion is the call, so none of it may be streamed as assistant text.
        Assert.True(forced);
    }

    [Fact]
    public void ToolChoice_Any_ConstrainsDecodingToAnyOfTheTools()
    {
        var options = BaseOptions();

        var parser = MessagesEndpoint.ApplyToolChoice(
            Choice("""{"type":"any"}"""), Tools, new HermesToolCallParser(), ref options, out bool forced);

        var format = Assert.IsType<ResponseFormat.JsonSchema>(options.ResponseFormat);
        Assert.Contains("get_weather", format.Schema);
        Assert.Contains("get_time", format.Schema);
        Assert.IsType<GenericToolCallParser>(parser);
    }

    [Fact]
    public void ToolChoice_LlamaParser_UsesTheParametersArgumentKey()
    {
        // The schema has to agree with the parser about the argument object's key, or the
        // constrained output parses into a tool call with no arguments.
        var options = BaseOptions();

        MessagesEndpoint.ApplyToolChoice(
            Choice("""{"type":"tool","name":"get_time"}"""), Tools, new LlamaToolCallParser(), ref options, out bool forced);

        var format = Assert.IsType<ResponseFormat.JsonSchema>(options.ResponseFormat);
        Assert.Contains("parameters", format.Schema);
    }

    [Fact]
    public void ToolChoice_WithoutTools_YieldsNoParser()
    {
        var options = BaseOptions();

        Assert.Null(MessagesEndpoint.ApplyToolChoice(
            Choice("""{"type":"any"}"""), tools: null, new HermesToolCallParser(), ref options, out bool forced));
        Assert.False(forced);
        Assert.Null(options.ResponseFormat);
    }

    [Fact]
    public void ValidateRequest_ForcedToolNotInTools_Fails()
    {
        // Unsatisfiable: there is no schema to constrain decoding with, so the request would
        // silently degrade into an ordinary completion.
        var req = JsonSerializer.Deserialize("""
        {"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}],
         "tools":[{"name":"get_weather","input_schema":{"type":"object"}}],
         "tool_choice":{"type":"tool","name":"get_time"}}
        """, ServerJsonContext.Default.AnthropicMessagesRequest)!;

        var error = MessagesEndpoint.ValidateRequest(req);
        Assert.NotNull(error);
        Assert.Contains("tool_choice", error);
    }

    [Fact]
    public void ValidateRequest_ForcedToolPresent_Passes()
    {
        var req = JsonSerializer.Deserialize("""
        {"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}],
         "tools":[{"name":"get_weather","input_schema":{"type":"object"}}],
         "tool_choice":{"type":"tool","name":"get_weather"}}
        """, ServerJsonContext.Default.AnthropicMessagesRequest)!;

        Assert.Null(MessagesEndpoint.ValidateRequest(req));
    }
}
