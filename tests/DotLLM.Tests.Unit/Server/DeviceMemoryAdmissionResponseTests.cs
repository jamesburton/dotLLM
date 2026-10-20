using System.Text;
using DotLLM.Core.Configuration;
using DotLLM.Server;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Issue #1068: a request whose context does not fit in free GPU memory is refused by the CUDA hybrid's admission
/// (<see cref="DeviceMemoryAdmissionException"/>). The server reports it as <c>context_length_exceeded</c> so agent clients
/// compact and retry. Before this, the exception escaped the stream and the client saw the connection drop.
/// </summary>
public sealed class DeviceMemoryAdmissionResponseTests
{
    private const string Refusal = "Not enough free GPU memory for the KV cache (q8_0) for 81768 tokens: it needs 2715 MiB.";

    [Fact]
    public async Task BeforeTheResponseStarts_ItIsA400ContextLengthExceeded()
    {
        var (ctx, body) = Context("/v1/chat/completions", started: false);
        await Run(ctx);

        Assert.Equal(400, ctx.Response.StatusCode);
        string raw = Encoding.UTF8.GetString(body.ToArray());
        Assert.Contains("\"code\":\"context_length_exceeded\"", raw);
        Assert.Contains("81768 tokens", raw);
    }

    [Fact]
    public async Task OnAStartedOpenAiStream_ItIsAnErrorEvent_ThenDone()
    {
        var (ctx, body) = Context("/v1/chat/completions", started: true);
        await Run(ctx);

        string raw = Encoding.UTF8.GetString(body.ToArray());
        Assert.StartsWith("data: {\"type\":\"error\",\"error\":", raw);
        Assert.Contains("context_length_exceeded", raw);
        Assert.EndsWith("data: [DONE]\n\n", raw);
    }

    [Fact]
    public async Task OnAStartedMessagesStream_ItIsAnAnthropicInvalidRequestEvent()
    {
        var (ctx, body) = Context("/v1/messages", started: true);
        await Run(ctx);

        string raw = Encoding.UTF8.GetString(body.ToArray());
        Assert.StartsWith("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\"", raw);
        Assert.DoesNotContain("[DONE]", raw);
    }

    [Fact]
    public async Task ARefusalMidMessagesStream_ReachesTheHandler_AsInvalidRequest_NotApiError()
    {
        // The endpoint's own catch-all used to swallow the refusal as an api_error, which the Anthropic SDK retries.
        var (ctx, body) = Context("/v1/messages", started: true);
        await Run(ctx, c => DotLLM.Server.Endpoints.MessagesEndpoint.WriteMessageStreamAsync(
            c, _ => RefusingTokens(), (work, _) => work(), null, null, "msg_1", "m", 7, CancellationToken.None));

        string raw = Encoding.UTF8.GetString(body.ToArray());
        Assert.Contains("\"type\":\"invalid_request_error\"", raw);
        Assert.DoesNotContain("api_error", raw);
    }

    [Fact]
    public async Task ARefusalMidCompletionsStream_ReachesTheHandler_AsContextLengthExceeded_Not500()
    {
        var (ctx, body) = Context("/v1/completions", started: true);
        await Run(ctx, c => DotLLM.Server.Endpoints.CompletionEndpoint.WriteCompletionStreamAsync(
            c, _ => RefusingTokens(), (work, _) => work(), "cmpl-1", "m", includeUsageChunk: false, CancellationToken.None));

        string raw = Encoding.UTF8.GetString(body.ToArray());
        Assert.Contains("context_length_exceeded", raw);
        Assert.DoesNotContain("api_error", raw);
    }

    private static async IAsyncEnumerable<DotLLM.Engine.GenerationToken> RefusingTokens()
    {
        await Task.Yield();
        yield return new DotLLM.Engine.GenerationToken(0, "tok", null);
        throw new DeviceMemoryAdmissionException(Refusal);
    }

    private static Task Run(HttpContext ctx) => Run(ctx, _ => throw new DeviceMemoryAdmissionException(Refusal));

    private static async Task Run(HttpContext ctx, RequestDelegate endpoint)
    {
        var app = new ApplicationBuilder(new ServiceCollection().BuildServiceProvider());
        ModelUnavailableException.UseModelUnavailableHandler(app);
        app.Run(endpoint);
        await app.Build()(ctx);
    }

    private static (DefaultHttpContext Ctx, MemoryStream Body) Context(string path, bool started)
    {
        var ctx = new DefaultHttpContext();
        var body = new MemoryStream();
        if (started) ctx.Features.Set<IHttpResponseFeature>(new StartedResponseFeature());
        ctx.Features.Set<IHttpResponseBodyFeature>(new StreamResponseBodyFeature(body));
        ctx.Request.Path = path;
        return (ctx, body);
    }

    private sealed class StartedResponseFeature : HttpResponseFeature
    {
        public override bool HasStarted => true;
    }
}
