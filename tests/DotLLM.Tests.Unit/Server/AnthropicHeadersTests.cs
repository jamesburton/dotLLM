using DotLLM.Server;
using Microsoft.AspNetCore.Http;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Tests for the Anthropic request headers on <c>/v1/messages</c> (#449):
/// <c>anthropic-version</c> must be honoured (unknown value rejected) and
/// <c>anthropic-beta</c> must never be a reason to reject a request.
/// </summary>
/// <remarks>
/// The supported version strings and the "missing header is accepted" rule are dotLLM's own
/// contract (<see cref="AnthropicHeaders"/> documents why). Only the status class and error
/// type are asserted here, never the wording of the real API's message — that text has not
/// been observed against the live service from this environment.
/// </remarks>
public sealed class AnthropicHeadersTests
{
    private static IHeaderDictionary Headers(params (string Name, string Value)[] pairs)
    {
        var ctx = new DefaultHttpContext();
        foreach (var (name, value) in pairs)
            ctx.Request.Headers.Append(name, value);
        return ctx.Request.Headers;
    }

    [Fact]
    public void Validate_MissingVersion_IsAccepted()
    {
        // Deliberate deviation from the real API: a curl user with no header is not blocked.
        Assert.Null(AnthropicHeaders.Validate(Headers()));
    }

    [Theory]
    [InlineData("2023-06-01")] // what every official SDK sends
    [InlineData("2023-01-01")]
    public void Validate_SupportedVersion_IsAccepted(string version)
    {
        Assert.Null(AnthropicHeaders.Validate(Headers(("anthropic-version", version))));
    }

    [Theory]
    [InlineData("2099-01-01")]
    [InlineData("2023-06-02")] // one day off a supported value — must not be accepted loosely
    [InlineData("v1")]
    [InlineData("latest")]
    public void Validate_UnknownVersion_IsRejected(string version)
    {
        var error = AnthropicHeaders.Validate(Headers(("anthropic-version", version)));
        Assert.NotNull(error);
        // The message names the header and echoes the offending value, so a client can act on it.
        Assert.Contains("anthropic-version", error);
        Assert.Contains(version, error);
    }

    [Fact]
    public void Validate_UnknownVersionAmongSupportedOnes_IsRejected()
    {
        // A comma-joined header (what a proxy produces from repeated headers) is split: one bad
        // member is still a rejection, otherwise a client could smuggle an unsupported pin
        // through by pairing it with a supported one.
        Assert.NotNull(AnthropicHeaders.Validate(Headers(("anthropic-version", "2023-06-01, 2099-01-01"))));
    }

    [Fact]
    public void Validate_UnknownVersionInRepeatedHeader_IsRejected()
    {
        Assert.NotNull(AnthropicHeaders.Validate(
            Headers(("anthropic-version", "2023-06-01"), ("anthropic-version", "2099-01-01"))));
    }

    [Fact]
    public void Validate_EmptyVersion_IsAccepted()
    {
        // An empty value is indistinguishable from an absent one for a client's purposes.
        Assert.Null(AnthropicHeaders.Validate(Headers(("anthropic-version", ""))));
    }

    [Theory]
    [InlineData("token-efficient-tools-2025-02-19")]
    [InlineData("some-beta-nobody-has-heard-of")]
    [InlineData("output-128k-2025-02-19,interleaved-thinking-2025-05-14")]
    public void Validate_UnknownBeta_IsNeverRejected(string beta)
    {
        // SDK helpers attach betas on their own; a 400 here would break otherwise-valid calls.
        Assert.Null(AnthropicHeaders.Validate(
            Headers(("anthropic-version", "2023-06-01"), ("anthropic-beta", beta))));
    }

    [Fact]
    public void RequestedBetas_FlattensCommasAndRepeatedHeaders()
    {
        var betas = AnthropicHeaders.RequestedBetas(Headers(
            ("anthropic-beta", "a-2025-01-01, b-2025-01-01"),
            ("anthropic-beta", "c-2025-01-01")));

        Assert.Equal(["a-2025-01-01", "b-2025-01-01", "c-2025-01-01"], betas);
    }

    [Fact]
    public void RequestedBetas_NoHeader_IsEmpty()
    {
        Assert.Empty(AnthropicHeaders.RequestedBetas(Headers()));
    }
}
