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

    // --- #449: content-block and count_tokens-mode validation ----------------

    [Theory]
    [InlineData("""{"type":"text","text":"hi"}""")]
    [InlineData("""{"type":"tool_use","id":"toolu_1","name":"f","input":{}}""")]
    [InlineData("""{"type":"tool_result","tool_use_id":"toolu_1","content":"42"}""")]
    [InlineData("""{"type":"thinking","thinking":"hmm","signature":"sig"}""")]
    [InlineData("""{"type":"redacted_thinking","data":"AAAA"}""")]
    public void ValidateRequest_SupportedContentBlock_Passes(string block)
    {
        // thinking/redacted_thinking are accepted (and dropped by the converter) so a client
        // replaying an extended-thinking transcript is not rejected.
        var req = Parse($$"""
        {"model":"m","max_tokens":16,"messages":[{"role":"user","content":[{{block}}]}]}
        """);
        Assert.Null(MessagesEndpoint.ValidateRequest(req));
    }

    [Theory]
    [InlineData("""{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBOR"}}""")]
    [InlineData("""{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVB"}}""")]
    [InlineData("""{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}""")]
    [InlineData("""{"type":"txet","text":"typo"}""")]
    public void ValidateRequest_UnsupportedContentBlock_Fails(string block)
    {
        // The converter drops anything it does not recognise. Dropping an image silently would
        // have the model answer about a picture it never saw, so the request is refused instead.
        var req = Parse($$"""
        {"model":"m","max_tokens":16,"messages":[{"role":"user","content":[{{block}}]}]}
        """);
        var error = MessagesEndpoint.ValidateRequest(req);
        Assert.NotNull(error);
        Assert.Contains("content[0]", error);
    }

    [Fact]
    public void ValidateRequest_ContentBlockWithoutType_Fails()
    {
        var req = Parse("""
        {"model":"m","max_tokens":16,"messages":[{"role":"user","content":[{"text":"hi"}]}]}
        """);
        Assert.NotNull(MessagesEndpoint.ValidateRequest(req));
    }

    [Theory]
    [InlineData("\"just a string\"")]
    [InlineData("42")]
    [InlineData("null")]
    public void ValidateRequest_NonObjectContentBlock_Fails(string block)
    {
        var req = Parse($$"""
        {"model":"m","max_tokens":16,"messages":[{"role":"user","content":[{{block}}]}]}
        """);
        Assert.NotNull(MessagesEndpoint.ValidateRequest(req));
    }

    [Fact]
    public void ValidateRequest_OffendingBlockIndexIsReported()
    {
        // The index must name the offending block, not the first one.
        var req = Parse("""
        {"model":"m","max_tokens":16,"messages":[{"role":"user","content":[
            {"type":"text","text":"ok"},
            {"type":"text","text":"also ok"},
            {"type":"image","source":{}}]}]}
        """);
        var error = MessagesEndpoint.ValidateRequest(req);
        Assert.NotNull(error);
        Assert.Contains("content[2]", error);
    }

    [Fact]
    public void ValidateRequest_CountTokensMode_AllowsMissingMaxTokens()
    {
        // /v1/messages/count_tokens has no max_tokens in its request body at all.
        var req = Parse("""{"model":"m","messages":[{"role":"user","content":"hi"}]}""");
        Assert.NotNull(MessagesEndpoint.ValidateRequest(req));
        Assert.Null(MessagesEndpoint.ValidateRequest(req, requireMaxTokens: false));
    }

    [Fact]
    public void ValidateRequest_CountTokensMode_StillRejectsNonPositiveMaxTokens()
    {
        // Relaxing "required" must not relax "positive when present".
        var req = Parse("""{"model":"m","max_tokens":0,"messages":[{"role":"user","content":"hi"}]}""");
        Assert.NotNull(MessagesEndpoint.ValidateRequest(req, requireMaxTokens: false));
    }

    [Fact]
    public void ValidateRequest_CountTokensMode_StillRejectsBadMessages()
    {
        var req = Parse("""{"model":"m","messages":[{"role":"system","content":"hi"}]}""");
        Assert.NotNull(MessagesEndpoint.ValidateRequest(req, requireMaxTokens: false));
    }
}
