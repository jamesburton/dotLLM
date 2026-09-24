using Microsoft.AspNetCore.Http;

namespace DotLLM.Server;

/// <summary>
/// Handling for the two Anthropic-specific request headers (<c>anthropic-version</c>
/// and <c>anthropic-beta</c>) on the <c>/v1/messages</c> surface (#449).
/// </summary>
/// <remarks>
/// <para>
/// The official SDKs always send <c>anthropic-version: 2023-06-01</c>; the real API
/// rejects an unrecognised value with a 400 <c>invalid_request_error</c>, so an unknown
/// value is rejected here too — a client pinned to a version dotLLM does not implement
/// should learn that immediately rather than silently receive a different wire shape.
/// </para>
/// <para>
/// <b>Deliberate deviation:</b> the real API also rejects a <i>missing</i>
/// <c>anthropic-version</c>. dotLLM's server is a local development tool that people
/// drive with <c>curl</c> as well as the SDKs, so an absent header is accepted and
/// treated as <see cref="DefaultVersion"/>. The acceptance criterion in #449 only
/// covers rejecting an <i>unknown</i> value.
/// </para>
/// <para>
/// <c>anthropic-beta</c> is accepted and ignored: no beta feature is honoured today,
/// and the real API tolerates betas a given endpoint does not use. It must never 400,
/// because SDK helpers (token-efficient tools, long outputs, …) attach betas on their
/// own and a rejection would break otherwise-valid calls.
/// </para>
/// </remarks>
public static class AnthropicHeaders
{
    /// <summary>The <c>anthropic-version</c> request header name.</summary>
    public const string VersionHeader = "anthropic-version";

    /// <summary>The <c>anthropic-beta</c> request header name.</summary>
    public const string BetaHeader = "anthropic-beta";

    /// <summary>Version assumed when the request carries no <c>anthropic-version</c>.</summary>
    public const string DefaultVersion = "2023-06-01";

    // Both published API versions. The wire shape dotLLM emits is the 2023-06-01 one;
    // 2023-01-01 is accepted because the difference (the legacy Text Completions
    // response envelope) does not apply to /v1/messages.
    private static readonly string[] SupportedVersions = ["2023-06-01", "2023-01-01"];

    /// <summary>
    /// Validates the Anthropic request headers.
    /// </summary>
    /// <param name="headers">The incoming request headers.</param>
    /// <returns>
    /// An error message for a 400 <c>invalid_request_error</c>, or <see langword="null"/>
    /// when the headers are acceptable.
    /// </returns>
    public static string? Validate(IHeaderDictionary headers)
    {
        if (!headers.TryGetValue(VersionHeader, out var values))
            return null;

        // A header sent several times arrives as several values; any one of them being
        // unrecognised is a client error, so all are checked.
        foreach (var value in values)
        {
            if (string.IsNullOrWhiteSpace(value))
                continue;
            foreach (var part in value.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
            {
                if (Array.IndexOf(SupportedVersions, part) < 0)
                    return $"{VersionHeader}: unsupported version \"{part}\"";
            }
        }

        return null;
    }

    /// <summary>
    /// Returns the beta flags requested by the caller, flattened across repeated headers
    /// and comma-separated values. All of them are ignored; the accessor exists so the
    /// set can be logged or honoured later without changing the parsing rules.
    /// </summary>
    /// <param name="headers">The incoming request headers.</param>
    public static string[] RequestedBetas(IHeaderDictionary headers)
    {
        if (!headers.TryGetValue(BetaHeader, out var values))
            return [];

        List<string>? betas = null;
        foreach (var value in values)
        {
            if (string.IsNullOrWhiteSpace(value))
                continue;
            foreach (var part in value.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
                (betas ??= []).Add(part);
        }
        return betas?.ToArray() ?? [];
    }
}
