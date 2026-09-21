using System.Globalization;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using DotLLM.Server.Models;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;

namespace DotLLM.Server.RateLimiting;

/// <summary>
/// ASP.NET middleware that admits or rejects requests based on the
/// configured per-API-key limits. Sits before the endpoints, so a
/// rejection short-circuits with HTTP 429 and a <c>Retry-After</c>
/// header — the inference pipeline is never invoked.
/// </summary>
/// <remarks>
/// <para>
/// <b>Scope.</b> Every <c>/v1/</c> path is metered except an explicit exemption
/// list of control-plane routes — see <see cref="IsMeteredPath"/> for why that is
/// an exemption list rather than an allowlist. Health probes, <c>/props</c> and
/// the chat UI sit outside <c>/v1/</c> and are never metered.
/// </para>
/// <para>
/// <b>Token estimation.</b> For accurate per-token accounting the
/// middleware would need to tokenize the request body up front. Today it
/// uses a cheap approximation (~1 token per 4 chars of payload) plus
/// <c>max_tokens</c> from the parsed JSON.
/// On admission the estimate is reserved; the endpoint reports the
/// authoritative count via <see cref="RateLimitLease.ReportActualTokens"/>
/// for true-up.
/// </para>
/// </remarks>
public sealed class RateLimitMiddleware
{
    /// <summary>
    /// <see cref="HttpContext.Items"/> key under which the admitted lease
    /// is stashed. Endpoints retrieve it to report actual completion
    /// tokens before disposal.
    /// </summary>
    public const string LeaseItemKey = "__DotLLM_RateLimitLease";

    /// <summary>
    /// <see cref="HttpContext.Items"/> key under which the resolved API-key string is stashed for
    /// downstream endpoints (e.g. to set <see cref="DotLLM.Engine.InferenceRequest.ApiKey"/> for the
    /// scheduler's per-key admission fairness). Present only on metered paths when rate limiting is enabled.
    /// </summary>
    public const string ApiKeyItemKey = "__DotLLM_ApiKey";

    private readonly RequestDelegate _next;
    private readonly RateLimitManager _manager;
    private readonly IApiKeyResolver _resolver;

    public RateLimitMiddleware(RequestDelegate next, RateLimitManager manager, IApiKeyResolver resolver)
    {
        _next = next;
        _manager = manager;
        _resolver = resolver;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        // Pass-through when disabled — keep zero overhead on the hot path
        // beyond a single bool check.
        if (!_manager.Config.Enabled)
        {
            await _next(context);
            return;
        }

        // Only meter inference endpoints. Everything else (UI assets,
        // /health, /v1/models, /v1/lora, /v1/cache/clear, ...) is free.
        if (!IsMeteredPath(context.Request.Path))
        {
            await _next(context);
            return;
        }

        var apiKey = _resolver.Resolve(context);
        // Stash for downstream endpoints (scheduler per-key fairness reads this onto InferenceRequest.ApiKey).
        context.Items[ApiKeyItemKey] = apiKey;
        int estimated = await EstimateTotalTokensAsync(context, _manager.Config).ConfigureAwait(false);

        var result = await _manager.TryAcquireAsync(apiKey, estimated, context.RequestAborted)
            .ConfigureAwait(false);

        if (!result.IsAcquired)
        {
            await WriteRejection(context, result, apiKey).ConfigureAwait(false);
            return;
        }

        var lease = result.Lease!;
        context.Items[LeaseItemKey] = lease;

        // #452: ResponseHeadersMiddleware stamped a snapshot on the way in, but that predates this
        // request's own acquire — so a success response would advertise the remaining budget as it
        // was one request ago. Re-stamp now: still before the response starts, so this is safe on
        // the SSE paths, whose first flush happens inside _next.
        if (_manager.GetSnapshot(apiKey) is { } admitted)
            ResponseHeadersMiddleware.ApplyRateLimitHeaders(context.Response, admitted);

        try
        {
            await _next(context);
        }
        finally
        {
            // If the endpoint never reported actuals (e.g. exception path),
            // the lease still releases — the reservation is just consumed.
            lease.Dispose();
            context.Items.Remove(LeaseItemKey);
        }
    }

    /// <summary>
    /// Returns the admitted lease stashed on this request, or <c>null</c>
    /// when rate-limiting is disabled or the path is unmetered. Used by
    /// endpoints to call <see cref="RateLimitLease.ReportActualTokens"/>
    /// after generation completes.
    /// </summary>
    public static RateLimitLease? GetLease(HttpContext context) =>
        context.Items.TryGetValue(LeaseItemKey, out var v) ? v as RateLimitLease : null;

    /// <summary>
    /// Non-generative <c>/v1/</c> routes: model listing/management, adapter and prefix-cache
    /// administration, tokenizer utilities, config. Everything else under <c>/v1/</c> is metered.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Add a route here only when it genuinely does not consume inference budget. A
    /// segment-boundary prefix match is used, so <c>/v1/models</c> also covers
    /// <c>/v1/models/{**id}</c>.
    /// </para>
    /// <para>
    /// <b>Known inexactness:</b> <c>POST /v1/prompt-cache/{id}</c> <i>does</i> run a prefill
    /// through the model to populate the KV cache. It is exempt because it was unmetered before
    /// this list was inverted and metering it now would be a behaviour change, not because it is
    /// free. Revisit if prefix registration becomes a way to burn budget unbilled.
    /// </para>
    /// </remarks>
    private static readonly string[] UnmeteredV1Prefixes =
    [
        "/v1/models",
        "/v1/lora",
        "/v1/prompt-cache",
        "/v1/cache",
        "/v1/config",
        "/v1/tokenize",
        "/v1/detokenize",
    ];

    /// <summary>
    /// Decides whether a request path consumes inference budget.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>This is deliberately an exemption list, not an allowlist.</b> It used to name the three
    /// paths that <i>were</i> metered, which meant every future generative endpoint shipped
    /// unmetered by omission and nothing failed when someone forgot. That list was already
    /// drifting from reality: it named <c>/v1/embeddings</c>, which does not exist yet, while
    /// <c>/v1/messages</c> (#448) would have bypassed rate limiting entirely — and an unmetered
    /// path can never return 429, so its configured limits are simply unenforceable.
    /// </para>
    /// <para>
    /// Inverting it makes the failure mode safe: a new <c>/v1/</c> route is metered by default,
    /// and forgetting to classify one produces an over-metered control-plane route (visible,
    /// harmless) rather than a silent hole in the limiter.
    /// </para>
    /// <para>
    /// Non-<c>/v1/</c> paths — health probes, <c>/props</c>, the chat UI and its assets — are
    /// never metered.
    /// </para>
    /// </remarks>
    internal static bool IsMeteredPath(PathString path)
    {
        if (!path.HasValue) return false;
        var p = path.Value!;
        if (!p.StartsWith("/v1/", StringComparison.OrdinalIgnoreCase))
            return false;

        foreach (var exempt in UnmeteredV1Prefixes)
        {
            // Segment-boundary match so /v1/models and /v1/models/{id} are exempt but a
            // hypothetical /v1/modelsomething is not silently swept in with them.
            if (p.StartsWith(exempt, StringComparison.OrdinalIgnoreCase) &&
                (p.Length == exempt.Length || p[exempt.Length] == '/'))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Best-effort estimate of total tokens this request will consume.
    /// Reads <c>max_tokens</c> from the parsed body when present, plus a
    /// cheap char-count approximation for the prompt/messages. We buffer
    /// the body so the downstream endpoint can re-read it.
    /// </summary>
    private static async ValueTask<int> EstimateTotalTokensAsync(HttpContext context, RateLimitConfig config)
    {
        var req = context.Request;
        if (!HttpMethods.IsPost(req.Method))
            return config.EstimatedCompletionTokensFallback;

        // Buffer body so model binding still sees it.
        req.EnableBuffering();
        long startPos = req.Body.CanSeek ? req.Body.Position : 0;
        int promptChars = 0;
        int? maxTokens = null;

        try
        {
            using var reader = new StreamReader(req.Body, Encoding.UTF8, detectEncodingFromByteOrderMarks: false, bufferSize: 8192, leaveOpen: true);
            // Limit to a reasonable size — past this we just fall back to the cap.
            const int maxScanChars = 256 * 1024;
            var sb = new StringBuilder();
            char[] buf = new char[4096];
            int read;
            while ((read = await reader.ReadAsync(buf.AsMemory(), context.RequestAborted).ConfigureAwait(false)) > 0)
            {
                sb.Append(buf, 0, read);
                if (sb.Length >= maxScanChars) break;
            }
            string body = sb.ToString();
            promptChars = body.Length;
            maxTokens = TryReadMaxTokens(body);
        }
        catch (Exception) // malformed body — defer to endpoint validator
        {
            // Conservative fallback.
            maxTokens = null;
        }
        finally
        {
            if (req.Body.CanSeek)
                req.Body.Position = startPos;
        }

        // Heuristic: ~4 bytes/token is a common approximation across BPE
        // tokenizers. Cheap and stable; trued up after generation.
        int promptEstimate = Math.Max(1, promptChars / 4);
        int completionEstimate = maxTokens ?? config.EstimatedCompletionTokensFallback;
        return promptEstimate + Math.Max(0, completionEstimate);
    }

    private static int? TryReadMaxTokens(string body)
    {
        try
        {
            using var doc = JsonDocument.Parse(body);
            if (doc.RootElement.ValueKind != JsonValueKind.Object) return null;
            if (doc.RootElement.TryGetProperty("max_tokens", out var mt) &&
                mt.ValueKind == JsonValueKind.Number &&
                mt.TryGetInt32(out var v) && v > 0)
            {
                return v;
            }
        }
        catch (JsonException) { }
        return null;
    }

    private async Task WriteRejection(HttpContext context, AcquireResult result, string apiKey)
    {
        context.Response.StatusCode = StatusCodes.Status429TooManyRequests;
        context.Response.Headers["Retry-After"] = result.RetryAfter.ToString(CultureInfo.InvariantCulture);
        context.Response.Headers["X-RateLimit-Limiter"] = result.Rejected.ToString();

        // #452: re-stamp the x-ratelimit-* trio with the post-rejection state. ResponseHeadersMiddleware
        // already wrote a snapshot on the way in, but that one predates this request's attempt — the
        // client needs the budget as it stands now, which is what its backoff is computed against.
        if (_manager.GetSnapshot(apiKey) is { } snapshot)
            ResponseHeadersMiddleware.ApplyRateLimitHeaders(context.Response, snapshot);

        string reason = result.Rejected switch
        {
            LimiterKind.Requests => "requests-per-minute",
            LimiterKind.Tokens => "tokens-per-minute",
            LimiterKind.Concurrency => "max-concurrent",
            _ => "rate-limit",
        };
        // SDK-shaped envelope (#452): the official clients classify a failure from `error.type`,
        // so a flat string here leaves a 429 indistinguishable from any other error.
        // code is OpenAI's spelling; which limiter fired is carried by X-RateLimit-Limiter.
        var body = ErrorResponse.RateLimit(
            $"Rate limit exceeded ({reason}). Retry in {result.RetryAfter}s.",
            code: "rate_limit_exceeded");
        context.Response.ContentType = "application/json";
        await JsonSerializer.SerializeAsync(
            context.Response.Body, body,
            ServerJsonContext.Default.ErrorResponse,
            context.RequestAborted).ConfigureAwait(false);
    }
}

/// <summary>Extension methods to wire the middleware into an ASP.NET pipeline.</summary>
public static class RateLimitMiddlewareExtensions
{
    /// <summary>
    /// Add the dotLLM rate-limit middleware to the request pipeline. Must
    /// be placed before <c>MapDotLLMEndpoints</c>. No-op when the manager
    /// is null or disabled.
    /// </summary>
    public static IApplicationBuilder UseDotLLMRateLimiting(this IApplicationBuilder app,
        RateLimitManager manager, IApiKeyResolver resolver)
    {
        ArgumentNullException.ThrowIfNull(manager);
        ArgumentNullException.ThrowIfNull(resolver);
        return app.UseMiddleware<RateLimitMiddleware>(manager, resolver);
    }
}
