namespace DotLLM.Server.Endpoints;

/// <summary>
/// Shared response setup for the server's Server-Sent Events (SSE) streaming endpoints.
/// </summary>
internal static class SseResponse
{
    /// <summary>
    /// Applies the response headers an SSE stream needs: the <c>text/event-stream</c> content type
    /// and <c>Cache-Control: no-cache</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A <c>Connection: keep-alive</c> header is deliberately <b>not</b> set. <c>Connection</c> is a
    /// hop-by-hop, connection-specific header field: it is prohibited over HTTP/2 and HTTP/3
    /// (RFC 9113 §8.2.2 — an endpoint MUST treat a message containing connection-specific header
    /// fields as malformed), and it is redundant over HTTP/1.1, where persistent connections are
    /// already the default. Kestrel filters it on HTTP/2, but a strict intermediary in the request
    /// path need not be as forgiving. Please do not re-add it.
    /// </para>
    /// </remarks>
    /// <param name="httpContext">The HTTP context whose response is being turned into an SSE stream.</param>
    internal static void ApplyHeaders(HttpContext httpContext)
    {
        // Idempotent: a keep-alive (#1023) may already have committed the stream with these headers.
        if (httpContext.Response.HasStarted) return;
        httpContext.Response.ContentType = "text/event-stream";
        httpContext.Response.Headers.CacheControl = "no-cache";
    }

    /// <summary>
    /// Writes an OpenAI-shaped error (#1023): the status + JSON body, or - when a keep-alive already committed an SSE response - an
    /// in-stream <c>data: {"error":...}</c> event followed by <c>data: [DONE]</c>. Stops the request's keep-alive first.
    /// </summary>
    /// <param name="ctx">The request.</param>
    /// <param name="status">The HTTP status, when the response has not started.</param>
    /// <param name="error">The error.</param>
    /// <returns>A task that completes when the error is written.</returns>
    internal static async Task WriteOpenAiErrorAsync(HttpContext ctx, int status, Models.ErrorResponse error)
    {
        await SseKeepAlive.StopForAsync(ctx);
        if (!ctx.Response.HasStarted)
        {
            ctx.Response.StatusCode = status;
            await ctx.Response.WriteAsJsonAsync(error, ServerJsonContext.Default.ErrorResponse, contentType: null, ctx.RequestAborted);
            return;
        }
        await ctx.Response.WriteAsync("data: " + System.Text.Json.JsonSerializer.Serialize(error, ServerJsonContext.Default.ErrorResponse)
            + "\n\ndata: [DONE]\n\n", ctx.RequestAborted);
        await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
    }
}
