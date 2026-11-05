using System.Net;
using DotLLM.ChunkReuse.Store;
using DotLLM.Server.Models;
using Microsoft.AspNetCore.Http;
using Microsoft.Net.Http.Headers;

namespace DotLLM.Server.ChunkCache;

/// <summary>Why a caller may not use the chunk-cache control plane.</summary>
internal enum ChunkAuthFailure
{
    /// <summary>Admitted.</summary>
    None,

    /// <summary>Keys are configured and the request carries none, or an unknown one: <c>401 authentication_error</c>.</summary>
    Unauthorized,

    /// <summary>No keys are configured and the caller is not local: <c>403 permission_error</c>.</summary>
    Forbidden,
}

/// <summary>
/// Who is calling, for the chunk cache: whether the caller is authenticated for the chunk-cache surface and its store
/// namespace. The namespace is always server-assigned (<see cref="ChunkNamespace.ForApiKey"/> of the caller's key, or
/// <see cref="ChunkNamespace.Default"/>), never taken from a request body, and never printed: <see cref="ToString"/> leaves
/// it out.
/// </summary>
/// <param name="Authenticated">
/// True for a caller with a configured key, or (no keys configured) a local caller. Only an authenticated caller's
/// annotations are looked up in the store and may receive per-chunk status.
/// </param>
/// <param name="Namespace">The caller's store namespace.</param>
internal sealed record ChunkCaller(bool Authenticated, string Namespace)
{
    /// <summary>A caller whose key is marked <c>admin</c> in the keys file (store-wide figures and operations, later tasks).</summary>
    public bool IsAdmin { get; init; }

    /// <summary>An unauthenticated caller: the default namespace, no store lookups, counts only.</summary>
    public static ChunkCaller Anonymous { get; } = new(false, ChunkNamespace.Default);

    // Never print the namespace (spec §10): it is a digest of a credential.
    private bool PrintMembers(System.Text.StringBuilder builder)
    {
        builder.Append(System.Globalization.CultureInfo.InvariantCulture, $"Authenticated = {Authenticated}, IsAdmin = {IsAdmin}");
        return true;
    }
}

/// <summary>
/// Authentication for the chunk cache (plan Task 14; spec §10). dotLLM has no authentication anywhere else
/// (<c>docs/SERVER.md</c> § Security); this is opt-in and covers the chunk-cache surface only.
/// </summary>
/// <remarks>
/// <para>
/// <b>With <c>--chunk-cache-api-keys</c>:</b> the request must carry a configured key, in <c>X-API-Key</c> or
/// <c>Authorization: Bearer</c> (checked in that order; a request with two <c>X-API-Key</c> values is refused). The key is
/// matched in constant time (<see cref="ChunkCacheKeys.TryMatch(string, out bool, out string?)"/>). Otherwise the control plane answers
/// <c>401 authentication_error</c>. The caller's namespace is the key's <c>ns=</c> name, else <see cref="ChunkNamespace.ForApiKey"/> of the key.
/// </para>
/// <para>
/// <b>Without keys:</b> local callers only, all in <see cref="ChunkNamespace.Default"/>; anyone else gets
/// <c>403 permission_error</c>. Local means all of: the connection's remote address is a loopback address (an
/// IPv4-mapped one too; an unknown address, as on a Unix socket or in-memory host, is not local); the <c>Host</c> header names
/// <c>localhost</c> or a loopback address (a DNS-rebinding page reaches loopback under its own host name); and no
/// <c>Origin</c> header (browsers send one on cross-origin requests, and the server's CORS policy allows any origin, so a web
/// page could otherwise read a local caller's chunk status). Behind a reverse proxy on the same host every caller is local:
/// configure keys there.
/// </para>
/// <para>
/// <b>Inference requests</b> are never refused here (dotLLM inference stays unauthenticated); a caller that would be
/// refused is <see cref="ChunkCaller.Anonymous"/> for its annotations: no store lookups (every verified chunk is a miss)
/// and a summary report even when <c>per_chunk</c> was asked for. The same rule will gate store writes (Task 15).
/// </para>
/// <para>Nothing here logs: keys, their digests and namespaces never reach a log or a response body.</para>
/// </remarks>
internal static class ChunkCacheAuth
{
    /// <summary>The error type of a 401.</summary>
    public const string AuthenticationErrorType = "authentication_error";

    /// <summary>The error type of a 403.</summary>
    public const string PermissionErrorType = "permission_error";

    /// <summary>Resolves the caller of a request.</summary>
    /// <param name="context">The request.</param>
    /// <param name="options">The server's chunk-cache options.</param>
    /// <param name="failure">Why the caller may not use the control plane, or <see cref="ChunkAuthFailure.None"/>.</param>
    /// <returns>The caller; <see cref="ChunkCaller.Anonymous"/> whenever <paramref name="failure"/> is not None.</returns>
    public static ChunkCaller Resolve(HttpContext context, ChunkCacheOptions options, out ChunkAuthFailure failure)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(options);
        if (options.ApiKeys is { } keys)
        {
            // A blank X-API-Key gives no key here (no fall-back to Bearer); the rate limiter may still bucket by the Bearer
            // value. That divergence fails closed (401) and is harmless.
            string? key = PresentedKey(context.Request.Headers);
            if (key is not null && keys.TryMatch(key, out bool admin, out string? named))
            {
                failure = ChunkAuthFailure.None;
                return new ChunkCaller(true, named ?? ChunkNamespace.ForApiKey(key.Trim(' ', '\t'))) { IsAdmin = admin };
            }
            failure = ChunkAuthFailure.Unauthorized;
            return ChunkCaller.Anonymous;
        }

        if (IsLocal(context))
        {
            failure = ChunkAuthFailure.None;
            return new ChunkCaller(true, ChunkNamespace.Default);
        }
        failure = ChunkAuthFailure.Forbidden;
        return ChunkCaller.Anonymous;
    }

    /// <summary>The caller of an annotated inference request: never refused, <see cref="ChunkCaller.Anonymous"/> when the control plane would refuse it.</summary>
    /// <param name="context">The request.</param>
    /// <param name="state">The server state (its chunk-cache options; the resolution is counted on its runtime).</param>
    /// <returns>The caller.</returns>
    /// <remarks>Call only for an annotated request: this looks the key up and hashes it.</remarks>
    public static ChunkCaller ForInference(HttpContext context, ServerState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        state.ChunkCacheRuntime.CountInferenceCallerResolution();
        return Resolve(context, state.Options.ChunkCache, out _);
    }

    /// <summary>Writes the 401 or 403 for <paramref name="failure"/> (OpenAI-shaped envelope, as the other endpoints).</summary>
    /// <param name="context">The request.</param>
    /// <param name="failure">The failure (not None).</param>
    /// <returns>The write.</returns>
    public static Task WriteFailureAsync(HttpContext context, ChunkAuthFailure failure)
    {
        ArgumentNullException.ThrowIfNull(context);
        int status;
        ErrorResponse body;
        if (failure == ChunkAuthFailure.Unauthorized)
        {
            status = StatusCodes.Status401Unauthorized;
            context.Response.Headers[HeaderNames.WWWAuthenticate] = "Bearer";
            body = new ErrorResponse
            {
                Error = new ErrorDetail
                {
                    Message = "The chunk-cache API needs an API key (X-API-Key or Authorization: Bearer) listed in --chunk-cache-api-keys.",
                    Type = AuthenticationErrorType,
                    Code = "invalid_api_key",
                },
            };
        }
        else
        {
            status = StatusCodes.Status403Forbidden;
            body = new ErrorResponse
            {
                Error = new ErrorDetail
                {
                    Message = "The chunk-cache API answers local callers only (loopback, no Origin header) unless --chunk-cache-api-keys is set.",
                    Type = PermissionErrorType,
                    Code = "chunk_cache_local_only",
                },
            };
        }
        context.Response.StatusCode = status;
        return context.Response.WriteAsJsonAsync(body, ServerJsonContext.Default.ErrorResponse, cancellationToken: context.RequestAborted);
    }

    /// <summary>
    /// A browser request from another site (<c>Sec-Fetch-Site: cross-site</c>, sent by current browsers): refused on the
    /// control plane whatever the key, so a page can never drive it (CSRF), also once prewarm and evict exist.
    /// </summary>
    /// <param name="context">The request.</param>
    /// <returns>Whether the request says it is cross-site.</returns>
    internal static bool IsCrossSite(HttpContext context) =>
        context.Request.Headers.TryGetValue("Sec-Fetch-Site", out var v)
        && v.Any(x => string.Equals(x?.Trim(), "cross-site", StringComparison.OrdinalIgnoreCase));

    /// <summary>The key a request presents: <c>X-API-Key</c> (exactly one value), else <c>Authorization: Bearer</c>; null when none.</summary>
    internal static string? PresentedKey(IHeaderDictionary headers)
    {
        if (headers.TryGetValue("X-API-Key", out var values) && values.Count > 0)
        {
            if (values.Count != 1)
                return null; // ambiguous: refuse rather than pick one
            string? v = values[0];
            return string.IsNullOrWhiteSpace(v) ? null : v;
        }
        if (headers.TryGetValue(HeaderNames.Authorization, out var auth) && auth.Count == 1 && auth[0] is { } a)
        {
            const string bearer = "Bearer ";
            if (a.StartsWith(bearer, StringComparison.OrdinalIgnoreCase))
            {
                string token = a[bearer.Length..].Trim(' ', '\t');
                return token.Length == 0 ? null : token;
            }
        }
        return null;
    }

    /// <summary>Whether a request comes from this machine and not through a browser (see the class remarks).</summary>
    internal static bool IsLocal(HttpContext context)
    {
        IPAddress? remote = context.Connection.RemoteIpAddress;
        if (remote is null)
            return false;
        if (remote.IsIPv4MappedToIPv6)
            remote = remote.MapToIPv4();
        if (!IPAddress.IsLoopback(remote))
            return false;
        if (context.Request.Headers.ContainsKey(HeaderNames.Origin))
            return false;
        string host = context.Request.Host.Host;
        if (string.Equals(host, "localhost", StringComparison.OrdinalIgnoreCase))
            return true;
        return IPAddress.TryParse(host.Trim('[', ']'), out IPAddress? hostIp)
            && IPAddress.IsLoopback(hostIp.IsIPv4MappedToIPv6 ? hostIp.MapToIPv4() : hostIp);
    }
}
