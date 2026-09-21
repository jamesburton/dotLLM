using System.Text.Json;
using DotLLM.Server;
using DotLLM.Server.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// #452: the server must emit the SDK-shaped error envelope
/// <c>{"error": {"message", "type", "param", "code"}}</c>, not the flat
/// <c>{"error": "&lt;string&gt;"}</c> it used to. Neither the official OpenAI SDK nor the
/// Anthropic SDK can classify a failure off the flat form — <c>.code</c>, <c>.type</c> and
/// <c>.param</c> are all unreachable — so any client that keys retry/backoff on the body
/// rather than the status line is flying blind.
/// </summary>
public sealed class OpenAiErrorEnvelopeTests
{
    private static JsonElement Serialize(ErrorResponse response)
    {
        string json = JsonSerializer.Serialize(response, ServerJsonContext.Default.ErrorResponse);
        return JsonDocument.Parse(json).RootElement.Clone();
    }

    [Fact]
    public void ErrorIsAnObject_NotAString()
    {
        var root = Serialize(ErrorResponse.InvalidRequest("boom"));

        Assert.True(root.TryGetProperty("error", out var error));
        Assert.Equal(JsonValueKind.Object, error.ValueKind);
        Assert.Equal("boom", error.GetProperty("message").GetString());
    }

    [Theory]
    [InlineData("invalid_request_error")]
    [InlineData("rate_limit_error")]
    [InlineData("not_found_error")]
    [InlineData("api_error")]
    public void FactoriesProduceTheExpectedTypes(string expected)
    {
        ErrorResponse response = expected switch
        {
            "invalid_request_error" => ErrorResponse.InvalidRequest("m"),
            "rate_limit_error" => ErrorResponse.RateLimit("m"),
            "not_found_error" => ErrorResponse.NotFound("m"),
            _ => ErrorResponse.Internal("m"),
        };

        Assert.Equal(expected, Serialize(response).GetProperty("error").GetProperty("type").GetString());
    }

    /// <summary>
    /// <c>param</c> and <c>code</c> are emitted as explicit <c>null</c>s rather than omitted.
    /// The context sets <c>DefaultIgnoreCondition = WhenWritingNull</c>, so this only holds
    /// because <see cref="ErrorDetail"/> opts out per-property — assert it so it cannot regress
    /// silently into "sometimes present, sometimes absent".
    /// </summary>
    [Fact]
    public void ParamAndCodeAreAlwaysPresent_EvenWhenNull()
    {
        var error = Serialize(ErrorResponse.InvalidRequest("m")).GetProperty("error");

        Assert.True(error.TryGetProperty("param", out var param));
        Assert.Equal(JsonValueKind.Null, param.ValueKind);
        Assert.True(error.TryGetProperty("code", out var code));
        Assert.Equal(JsonValueKind.Null, code.ValueKind);
    }

    [Fact]
    public void ParamAndCodeRoundTrip_WhenSupplied()
    {
        var error = Serialize(ErrorResponse.NotFound("no such model", param: "model", code: "model_not_found"))
            .GetProperty("error");

        Assert.Equal("model", error.GetProperty("param").GetString());
        Assert.Equal("model_not_found", error.GetProperty("code").GetString());
    }

    /// <summary>
    /// The top-level <c>"type": "error"</c> discriminator is what Anthropic's envelope requires
    /// (#448) and is inert for OpenAI clients — one envelope serves both surfaces.
    /// </summary>
    [Fact]
    public void EnvelopeCarriesTopLevelErrorDiscriminator()
    {
        Assert.Equal("error", Serialize(ErrorResponse.Internal("m")).GetProperty("type").GetString());
    }
}
