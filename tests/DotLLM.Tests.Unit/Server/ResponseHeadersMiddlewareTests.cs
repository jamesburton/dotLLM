using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Threading.Tasks;
using DotLLM.Server.RateLimiting;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// #452: the rate limiter exists but was invisible to clients — none of its state reached the
/// response. Official SDKs read <c>x-ratelimit-*</c> and <c>retry-after</c> to drive their own
/// backoff, and correlate a failure by <c>x-request-id</c>. These tests drive
/// <see cref="ResponseHeadersMiddleware"/> over a <see cref="DefaultHttpContext"/>.
/// </summary>
/// <remarks>
/// The deterministic headers are written <b>before</b> the inner pipeline runs, so they land even
/// on an SSE response that flushes its first chunk immediately. Only
/// <c>openai-processing-ms</c> — which cannot be known until the pipeline returns — goes through
/// <c>Response.OnStarting</c>; <see cref="StartableResponseFeature"/> stands in for the server's
/// response feature so that callback can actually be exercised (the stock
/// <see cref="DefaultHttpContext"/> feature drops <c>OnStarting</c> registrations on the floor).
/// </remarks>
public sealed class ResponseHeadersMiddlewareTests
{
    private sealed class StartableResponseFeature : IHttpResponseFeature
    {
        private readonly List<(Func<object, Task> Callback, object State)> _onStarting = [];

        public Stream Body { get; set; } = new MemoryStream();
        public bool HasStarted { get; private set; }
        public IHeaderDictionary Headers { get; set; } = new HeaderDictionary();
        public string? ReasonPhrase { get; set; }
        public int StatusCode { get; set; } = 200;

        public void OnCompleted(Func<object, Task> callback, object state) { }
        public void OnStarting(Func<object, Task> callback, object state) => _onStarting.Add((callback, state));

        /// <summary>Simulates the server beginning to write the response.</summary>
        public async Task StartAsync()
        {
            HasStarted = true;
            foreach (var (callback, state) in _onStarting)
                await callback(state);
        }
    }

    private static DefaultHttpContext NewContext(out StartableResponseFeature response,
        string path = "/v1/chat/completions")
    {
        response = new StartableResponseFeature();
        var features = new FeatureCollection();
        features.Set<IHttpRequestFeature>(new HttpRequestFeature { Method = HttpMethods.Post, Path = path });
        features.Set<IHttpResponseFeature>(response);
        features.Set<IHttpResponseBodyFeature>(new StreamResponseBodyFeature(response.Body));
        return new DefaultHttpContext(features);
    }

    private static ResponseHeadersMiddleware Build(RateLimitManager? manager = null) =>
        new(_ => Task.CompletedTask, manager);

    [Fact]
    public async Task GeneratesRequestId_WhenNoneSupplied()
    {
        var ctx = NewContext(out _);

        await Build().InvokeAsync(ctx);

        Assert.False(string.IsNullOrEmpty(ctx.Response.Headers["x-request-id"].ToString()));
    }

    [Fact]
    public async Task EchoesInboundRequestId()
    {
        var ctx = NewContext(out _);
        ctx.Request.Headers["x-request-id"] = "req_abc123";

        await Build().InvokeAsync(ctx);

        Assert.Equal("req_abc123", ctx.Response.Headers["x-request-id"].ToString());
    }

    /// <summary>An oversized inbound id is not reflected back verbatim — it is replaced.</summary>
    [Fact]
    public async Task RejectsOverlongInboundRequestId()
    {
        var ctx = NewContext(out _);
        ctx.Request.Headers["x-request-id"] = new string('x', 500);

        await Build().InvokeAsync(ctx);

        var id = ctx.Response.Headers["x-request-id"].ToString();
        Assert.False(string.IsNullOrEmpty(id));
        Assert.True(id.Length <= ResponseHeadersMiddleware.MaxRequestIdLength);
    }

    [Fact]
    public async Task EmitsProcessingMs_WhenTheResponseStarts()
    {
        var ctx = NewContext(out var response);

        await Build().InvokeAsync(ctx);
        await response.StartAsync();

        Assert.True(ctx.Response.Headers.ContainsKey("openai-processing-ms"));
        Assert.True(long.TryParse(ctx.Response.Headers["openai-processing-ms"].ToString(),
            NumberStyles.Integer, CultureInfo.InvariantCulture, out var ms) && ms >= 0);
    }

    /// <summary>
    /// Inbound <c>OpenAI-Organization</c>/<c>-Project</c>/<c>-Beta</c> and <c>anthropic-beta</c>
    /// are concepts this server has no equivalent for. They must be accepted and ignored — never
    /// a 400 (#452).
    /// </summary>
    [Fact]
    public async Task ToleratesVendorRequestHeaders()
    {
        var ctx = NewContext(out _);
        ctx.Request.Headers["OpenAI-Organization"] = "org-123";
        ctx.Request.Headers["OpenAI-Project"] = "proj_123";
        ctx.Request.Headers["OpenAI-Beta"] = "assistants=v2";
        ctx.Request.Headers["anthropic-beta"] = "token-counting-2024-11-01";

        await Build().InvokeAsync(ctx);

        Assert.Equal(200, ctx.Response.StatusCode);
    }

    [Fact]
    public async Task EmitsRateLimitHeaders_OnSuccess_WhenLimiterConfigured()
    {
        var cfg = new RateLimitConfig
        {
            Enabled = true,
            DefaultPolicy = new RateLimitPolicy { RequestsPerMinute = 120, TokensPerMinute = 6000 },
        };
        using var manager = new RateLimitManager(cfg);
        var ctx = NewContext(out _);
        ctx.Request.Headers[HeaderApiKeyResolver.ApiKeyHeader] = "alice";

        await Build(manager).InvokeAsync(ctx);

        Assert.Equal("120", ctx.Response.Headers["x-ratelimit-limit-requests"].ToString());
        Assert.Equal("6000", ctx.Response.Headers["x-ratelimit-limit-tokens"].ToString());
        Assert.True(ctx.Response.Headers.ContainsKey("x-ratelimit-remaining-requests"));
        Assert.True(ctx.Response.Headers.ContainsKey("x-ratelimit-remaining-tokens"));
        Assert.True(ctx.Response.Headers.ContainsKey("x-ratelimit-reset-requests"));
        Assert.True(ctx.Response.Headers.ContainsKey("x-ratelimit-reset-tokens"));
    }

    /// <summary>
    /// A limiter that is not configured must not produce a header claiming a limit of zero —
    /// an SDK reading <c>x-ratelimit-remaining-tokens: 0</c> would back off forever.
    /// </summary>
    [Fact]
    public async Task OmitsHeadersForUnconfiguredLimiters()
    {
        var cfg = new RateLimitConfig
        {
            Enabled = true,
            DefaultPolicy = new RateLimitPolicy { RequestsPerMinute = 10, TokensPerMinute = 0 },
        };
        using var manager = new RateLimitManager(cfg);
        var ctx = NewContext(out _);

        await Build(manager).InvokeAsync(ctx);

        Assert.True(ctx.Response.Headers.ContainsKey("x-ratelimit-limit-requests"));
        Assert.False(ctx.Response.Headers.ContainsKey("x-ratelimit-limit-tokens"));
    }

    [Fact]
    public async Task NoRateLimitHeaders_WhenLimitingDisabled()
    {
        var ctx = NewContext(out _);

        await Build().InvokeAsync(ctx);

        Assert.False(ctx.Response.Headers.ContainsKey("x-ratelimit-limit-requests"));
        // x-request-id is unconditional, though.
        Assert.True(ctx.Response.Headers.ContainsKey("x-request-id"));
    }

    /// <summary>
    /// The deterministic headers are written before the inner pipeline runs. That is what makes
    /// them safe on the SSE endpoints, whose first flush starts the response long before
    /// <c>await _next</c> returns.
    /// </summary>
    [Fact]
    public async Task HeadersAreSetBeforeTheInnerPipelineRuns()
    {
        string? seen = null;
        var mw = new ResponseHeadersMiddleware(
            c => { seen = c.Response.Headers["x-request-id"].ToString(); return Task.CompletedTask; },
            rateLimitManager: null);
        var ctx = NewContext(out _);

        await mw.InvokeAsync(ctx);

        Assert.False(string.IsNullOrEmpty(seen));
    }

    /// <summary>
    /// The snapshot must reflect consumption, not just the configured ceiling — otherwise
    /// <c>x-ratelimit-remaining-requests</c> is a constant and the SDK never sees pressure.
    /// </summary>
    [Fact]
    public async Task RemainingRequestsDecreasesAsBudgetIsConsumed()
    {
        var cfg = new RateLimitConfig
        {
            Enabled = true,
            DefaultPolicy = new RateLimitPolicy { RequestsPerMinute = 60 },
        };
        using var manager = new RateLimitManager(cfg);

        var before = manager.GetSnapshot("alice");
        Assert.NotNull(before);
        Assert.Equal(60, before!.Value.RequestsRemaining);

        using var lease = (await manager.TryAcquireAsync("alice", estimatedTokens: 0, default)).Lease;

        var after = manager.GetSnapshot("alice");
        Assert.NotNull(after);
        Assert.True(after!.Value.RequestsRemaining < before.Value.RequestsRemaining,
            $"remaining should drop after an admitted request (before={before.Value.RequestsRemaining}, after={after.Value.RequestsRemaining})");
    }
}
