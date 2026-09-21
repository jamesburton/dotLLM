using System;
using System.Linq;
using DotLLM.Server;
using DotLLM.Server.RateLimiting;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// #452 follow-up. The rate limiter used to decide what to meter from a hardcoded allowlist of
/// three paths. That is a design defect, not a missing line: a new generative endpoint ships
/// <b>unmetered by omission</b>, and nothing fails when someone forgets. The allowlist was
/// already drifting — it named <c>/v1/embeddings</c>, which does not exist yet, while
/// <c>/v1/messages</c> (#448) would have bypassed metering entirely.
/// </summary>
/// <remarks>
/// The policy is now inverted: everything under <c>/v1/</c> is metered <i>unless</i> it is on an
/// explicit exemption list of non-generative routes. Forgetting to classify a new route now fails
/// safe (it gets metered) instead of failing open.
/// </remarks>
public sealed class MeteredPathPolicyTests
{
    /// <summary>
    /// Generative routes — present and planned. <c>/v1/messages</c> (#448) and
    /// <c>/v1/embeddings</c> (#451) are deliberately included while absent from this worktree:
    /// the point is that the policy covers them the moment they land, without anyone editing a
    /// list.
    /// </summary>
    [Theory]
    [InlineData("/v1/chat/completions")]
    [InlineData("/v1/completions")]
    [InlineData("/v1/embeddings")]
    [InlineData("/v1/messages")]
    [InlineData("/v1/responses")]
    [InlineData("/v1/some/endpoint/invented/next/year")]
    public void GenerativeRoutesAreMetered(string path) =>
        Assert.True(RateLimitMiddleware.IsMeteredPath(path),
            $"{path} must be metered — an unmetered generative route can never return 429, so its rate limits are unenforceable.");

    [Theory]
    [InlineData("/health")]
    [InlineData("/ready")]
    [InlineData("/props")]
    [InlineData("/")]
    [InlineData("/app.js")]
    [InlineData("/v1/models")]
    [InlineData("/v1/models/available")]
    [InlineData("/v1/models/Qwen/Qwen3-4B")]
    [InlineData("/v1/lora")]
    [InlineData("/v1/lora/load")]
    [InlineData("/v1/config")]
    [InlineData("/v1/tokenize")]
    [InlineData("/v1/detokenize")]
    [InlineData("/v1/prompt-cache")]
    [InlineData("/v1/cache/clear")]
    public void ControlPlaneRoutesAreNotMetered(string path) =>
        Assert.False(RateLimitMiddleware.IsMeteredPath(path), $"{path} is a control-plane route and must stay free.");

    /// <summary>
    /// Structural guard: every route this server actually registers must be classified on purpose.
    /// A new route under <c>/v1/</c> is metered by default, so this test does not break when one
    /// is added — but if someone adds a control-plane route and forgets to exempt it, the
    /// exemption list below is where it surfaces, rather than a silent metering hole.
    /// </summary>
    [Fact]
    public void EveryRegisteredRouteIsClassified()
    {
        var builder = WebApplication.CreateSlimBuilder();
        using var state = new ServerState { Options = new ServerOptions { Model = "test" } };
        builder.Services.AddSingleton(state);
        var app = builder.Build();
        app.MapDotLLMEndpoints(serveUi: true);

        var registered = ((IEndpointRouteBuilder)app).DataSources
            .SelectMany(d => d.Endpoints)
            .OfType<RouteEndpoint>()
            .Select(e => "/" + e.RoutePattern.RawText!.TrimStart('/'))
            .Distinct(StringComparer.Ordinal)
            .ToArray();

        Assert.NotEmpty(registered);

        // No route registered today is generative except the two completion endpoints, so every
        // other registered route must be exempt. This is the assertion that would fire if a
        // control-plane route were added under /v1/ without being classified.
        string[] generative = ["/v1/chat/completions", "/v1/completions"];
        foreach (var path in registered)
        {
            bool expected = generative.Contains(path, StringComparer.Ordinal);
            Assert.Equal(expected, RateLimitMiddleware.IsMeteredPath(path));
        }
    }

    /// <summary>
    /// End-to-end through the middleware: a path the policy meters is actually gated, which is
    /// what makes a 429 (and therefore the whole #452 header/envelope surface) reachable on it.
    /// </summary>
    [Fact]
    public async System.Threading.Tasks.Task AMeteredPathIsActuallyGated()
    {
        var cfg = new RateLimitConfig
        {
            Enabled = true,
            DefaultPolicy = new RateLimitPolicy { RequestsPerMinute = 1 },
        };
        using var manager = new RateLimitManager(cfg);
        var mw = new RateLimitMiddleware(_ => System.Threading.Tasks.Task.CompletedTask, manager, new HeaderApiKeyResolver());

        static DefaultHttpContext Request(string path)
        {
            var ctx = new DefaultHttpContext();
            ctx.Request.Method = HttpMethods.Post;
            ctx.Request.Path = path;
            ctx.Request.Body = new System.IO.MemoryStream("{}"u8.ToArray());
            ctx.Response.Body = new System.IO.MemoryStream();
            return ctx;
        }

        // Use a route that exists only in the sibling Anthropic branch: metering must not depend
        // on anyone having remembered to add it to a list here.
        var first = Request("/v1/messages");
        await mw.InvokeAsync(first);
        Assert.Equal(200, first.Response.StatusCode);

        var second = Request("/v1/messages");
        await mw.InvokeAsync(second);
        Assert.Equal(StatusCodes.Status429TooManyRequests, second.Response.StatusCode);
    }
}
