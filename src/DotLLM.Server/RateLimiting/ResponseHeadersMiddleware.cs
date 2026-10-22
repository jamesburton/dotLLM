using System.Diagnostics;
using System.Globalization;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;

namespace DotLLM.Server.RateLimiting;

/// <summary>
/// Emits the observability headers the official OpenAI/Anthropic SDKs read (#452):
/// <c>x-request-id</c>, <c>openai-processing-ms</c>, and the <c>x-ratelimit-*</c> trio for both
/// requests and tokens.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this exists.</b> The server already had a real rate limiter, but none of its state
/// reached the client — so an SDK could only retry blind, and a failure had no id to correlate
/// against a server log. This middleware makes the limiter visible without changing what it
/// admits.
/// </para>
/// <para>
/// <b>Ordering.</b> Register this <i>outside</i> (before) <see cref="RateLimitMiddleware"/>, so
/// the headers also land on the 429 short-circuit. <see cref="RateLimitMiddleware"/> then
/// overwrites the <c>x-ratelimit-*</c> values on a rejection with the post-rejection state, which
/// is the state the client actually needs.
/// </para>
/// <para>
/// <b>Timing of the writes.</b> Everything that can be known up front is written <i>before</i>
/// <c>await _next</c>. That is what makes it safe on the SSE endpoints: their first flush starts
/// the response, and headers cannot be added after that. Only
/// <c>openai-processing-ms</c> — which is not known until the pipeline returns — is deferred to
/// <see cref="HttpResponse.OnStarting(System.Func{object, Task}, object)"/>, so it is present on
/// buffered responses and harmlessly absent on a stream that started before generation finished.
/// </para>
/// <para>
/// <b>Inbound vendor headers.</b> <c>OpenAI-Organization</c>, <c>OpenAI-Project</c>,
/// <c>OpenAI-Beta</c> and <c>anthropic-beta</c> name concepts this server has no equivalent for.
/// Nothing here (or anywhere else in the pipeline) inspects them, so they are accepted and
/// ignored rather than rejected — a client that always sends them must not get a 400.
/// </para>
/// </remarks>
public sealed class ResponseHeadersMiddleware
{
    /// <summary>Response/request header carrying the correlation id.</summary>
    public const string RequestIdHeader = "x-request-id";

    /// <summary>Wall-clock milliseconds spent in the pipeline, OpenAI's spelling.</summary>
    public const string ProcessingMsHeader = "openai-processing-ms";

    /// <summary>
    /// Longest inbound <see cref="RequestIdHeader"/> value echoed back verbatim. Anything longer
    /// is replaced with a server-generated id rather than reflected — an echoed header is
    /// attacker-controlled data landing in logs and proxies.
    /// </summary>
    public const int MaxRequestIdLength = 128;

    private static readonly HeaderApiKeyResolver DefaultResolver = new();

    private readonly RequestDelegate _next;
    private readonly RateLimitManager? _rateLimitManager;
    private readonly IApiKeyResolver _resolver;

    /// <param name="next">Next delegate in the pipeline.</param>
    /// <param name="rateLimitManager">Null when rate limiting is off; the id/timing headers are still emitted.</param>
    /// <param name="resolver">
    /// Must be the SAME <see cref="IApiKeyResolver"/> the limiter uses. A host that wires real auth
    /// registers its own, and this middleware runs before <see cref="RateLimitMiddleware"/> has
    /// stashed the resolved key — resolving with the default here would report another bucket's budget.
    /// </param>
    public ResponseHeadersMiddleware(RequestDelegate next, RateLimitManager? rateLimitManager,
        IApiKeyResolver? resolver = null)
    {
        _next = next;
        _rateLimitManager = rateLimitManager;
        _resolver = resolver ?? DefaultResolver;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        context.Response.Headers[RequestIdHeader] = ResolveRequestId(context);

        if (_rateLimitManager is { Config.Enabled: true })
        {
            var apiKey = ResolveApiKey(context);
            if (_rateLimitManager.GetSnapshot(apiKey) is { } snapshot)
                ApplyRateLimitHeaders(context.Response, snapshot);
        }

        var started = Stopwatch.GetTimestamp();
        context.Response.OnStarting(static state =>
        {
            var (response, startTimestamp) = ((HttpResponse, long))state;
            var elapsed = Stopwatch.GetElapsedTime(startTimestamp);
            response.Headers[ProcessingMsHeader] =
                ((long)elapsed.TotalMilliseconds).ToString(CultureInfo.InvariantCulture);
            return Task.CompletedTask;
        }, (context.Response, started));

        await _next(context);
    }

    /// <summary>
    /// Writes the <c>x-ratelimit-*</c> trio for each configured limiter. An unconfigured limiter
    /// emits nothing: a header advertising a limit of 0 with 0 remaining would make a well-behaved
    /// SDK back off indefinitely against a server that is not limiting it at all.
    /// </summary>
    internal static void ApplyRateLimitHeaders(HttpResponse response, in RateLimitSnapshot snapshot)
    {
        if (snapshot.RequestLimit > 0)
        {
            response.Headers["x-ratelimit-limit-requests"] = Fmt(snapshot.RequestLimit);
            response.Headers["x-ratelimit-remaining-requests"] = Fmt(snapshot.RequestsRemaining);
            response.Headers["x-ratelimit-reset-requests"] = Fmt(snapshot.RequestsResetSeconds);
        }

        if (snapshot.TokenLimit > 0)
        {
            response.Headers["x-ratelimit-limit-tokens"] = Fmt(snapshot.TokenLimit);
            response.Headers["x-ratelimit-remaining-tokens"] = Fmt(snapshot.TokensRemaining);
            response.Headers["x-ratelimit-reset-tokens"] = Fmt(snapshot.TokensResetSeconds);
        }

        static string Fmt(long v) => v.ToString(CultureInfo.InvariantCulture);
    }

    /// <summary>
    /// Echoes a sane inbound <see cref="RequestIdHeader"/>, otherwise falls back to the
    /// connection's trace identifier so the id in the response matches the one in the server log.
    /// </summary>
    private static string ResolveRequestId(HttpContext context)
    {
        if (context.Request.Headers.TryGetValue(RequestIdHeader, out var inbound))
        {
            var candidate = inbound.ToString();
            if (candidate.Length is > 0 and <= MaxRequestIdLength && !ContainsControlChars(candidate))
                return candidate;
        }

        return string.IsNullOrEmpty(context.TraceIdentifier)
            ? Guid.NewGuid().ToString("N")
            : context.TraceIdentifier;
    }

    private static bool ContainsControlChars(string value)
    {
        foreach (char c in value)
        {
            if (char.IsControl(c))
                return true;
        }
        return false;
    }

    /// <summary>
    /// Resolves the API key, preferring the value <see cref="RateLimitMiddleware"/> already stashed.
    /// On this middleware's inbound leg it will not have run yet, so the configured resolver is used —
    /// which is why it must be the same instance the limiter partitions on.
    /// </summary>
    private string ResolveApiKey(HttpContext context) =>
        context.Items.TryGetValue(RateLimitMiddleware.ApiKeyItemKey, out var stashed) && stashed is string s
            ? s
            : _resolver.Resolve(context);
}

/// <summary>Extension methods to wire <see cref="ResponseHeadersMiddleware"/> into a pipeline.</summary>
public static class ResponseHeadersMiddlewareExtensions
{
    /// <summary>
    /// Adds the SDK observability headers (<c>x-request-id</c>, <c>openai-processing-ms</c>,
    /// <c>x-ratelimit-*</c>). Register before <c>UseDotLLMRateLimiting</c> so the headers also
    /// land on a 429 short-circuit. <paramref name="manager"/> may be <c>null</c> when rate
    /// limiting is disabled — the id and timing headers are emitted either way.
    /// </summary>
    public static IApplicationBuilder UseDotLLMResponseHeaders(this IApplicationBuilder app,
        RateLimitManager? manager, IApiKeyResolver? resolver = null) =>
        // Constructed directly rather than through UseMiddleware: the manager is legitimately null
        // when rate limiting is off, and DI activation matches constructor arguments by type.
        app.Use(next => new ResponseHeadersMiddleware(next, manager, resolver).InvokeAsync);
}
