using System.Text;
using System.Text.Json;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
using Microsoft.AspNetCore.Http;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Request-validation tests for <see cref="MessagesEndpoint"/> (<c>POST /v1/messages</c>).
/// Pure structural validation — no model load required.
/// </summary>
public sealed class AnthropicMessagesEndpointTests
{
    private static AnthropicMessagesRequest Parse(string json) =>
        JsonSerializer.Deserialize(json, ServerJsonContext.Default.AnthropicMessagesRequest)!;



    [Fact]
    public void ValidateRequest_EmptyMessages_Fails()
    {
        var req = new AnthropicMessagesRequest { Messages = [], MaxTokens = 16 };
        Assert.NotNull(MessagesEndpoint.ValidateRequest(req));
    }

    [Fact]
    public void ValidateRequest_MissingMaxTokens_Fails()
    {
        // max_tokens is a required field of the Anthropic Messages API.
        var req = Parse("""{"model":"m","messages":[{"role":"user","content":"hi"}]}""");
        Assert.NotNull(MessagesEndpoint.ValidateRequest(req));
    }

    [Fact]
    public void ValidateRequest_NonPositiveMaxTokens_Fails()
    {
        var req = new AnthropicMessagesRequest
        {
            Messages = [new AnthropicMessageDto { Role = "user", Content = default }],
            MaxTokens = 0,
        };
        Assert.NotNull(MessagesEndpoint.ValidateRequest(req));
    }

    [Theory]
    [InlineData("system")]
    [InlineData("tool")]
    [InlineData("developer")]
    [InlineData("")]
    public void ValidateRequest_UnsupportedRole_Fails(string role)
    {
        // Roles flow straight into the chat template; only user/assistant are addressable
        // by a client (the system prompt is the top-level `system` field).
        var req = Parse($$"""
        {"model":"m","max_tokens":16,"messages":[{"role":"{{role}}","content":"hi"}]}
        """);
        Assert.NotNull(MessagesEndpoint.ValidateRequest(req));
    }

    [Theory]
    [InlineData("123")]
    [InlineData("null")]
    [InlineData("true")]
    [InlineData("""{"type":"text"}""")]
    public void ValidateRequest_NonStringNonArrayContent_Fails(string content)
    {
        var req = Parse($$"""
        {"model":"m","max_tokens":16,"messages":[{"role":"user","content":{{content}}}]}
        """);
        Assert.NotNull(MessagesEndpoint.ValidateRequest(req));
    }

    [Fact]
    public void ValidateRequest_MissingContent_Fails()
    {
        var req = Parse("""{"model":"m","max_tokens":16,"messages":[{"role":"user"}]}""");
        Assert.NotNull(MessagesEndpoint.ValidateRequest(req));
    }

    [Fact]
    public void ValidateRequest_BlockArrayContent_ReturnsNull()
    {
        var req = Parse("""
        {"model":"m","max_tokens":16,"messages":[
          {"role":"user","content":[{"type":"text","text":"hi"}]},
          {"role":"assistant","content":"yes"}]}
        """);
        Assert.Null(MessagesEndpoint.ValidateRequest(req));
    }

    [Fact]
    public void ValidateRequest_Valid_ReturnsNull()
    {
        var req = Parse("""
        {"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}
        """);
        Assert.Null(MessagesEndpoint.ValidateRequest(req));
    }

    // --- Error envelope -----------------------------------------------------

    [Fact]
    public async Task WriteErrorAsync_EmitsAnthropicEnvelopeNotOpenAiShape()
    {
        // The OpenAI surface answers with {"error":"<string>"}; an Anthropic SDK client
        // parses {"type":"error","error":{"type":..,"message":..}} and would throw on the
        // OpenAI shape. Every /v1/messages failure path must use this envelope.
        var ctx = new DefaultHttpContext();
        using var body = new MemoryStream();
        ctx.Response.Body = body;

        await MessagesEndpoint.WriteErrorAsync(ctx, 503, "api_error", "No model loaded");

        Assert.Equal(503, ctx.Response.StatusCode);
        using var doc = JsonDocument.Parse(Encoding.UTF8.GetString(body.ToArray()));
        var root = doc.RootElement;
        Assert.Equal("error", root.GetProperty("type").GetString());
        Assert.Equal(JsonValueKind.Object, root.GetProperty("error").ValueKind);
        Assert.Equal("api_error", root.GetProperty("error").GetProperty("type").GetString());
        Assert.Equal("No model loaded", root.GetProperty("error").GetProperty("message").GetString());
    }
}
