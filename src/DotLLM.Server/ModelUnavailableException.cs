using System.Text.Json;
using DotLLM.Server.Models;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;

namespace DotLLM.Server;

/// <summary>
/// The model a request was admitted for is no longer active when its turn comes, and could not be re-activated (#1023) - e.g. it
/// was swapped out while the request queued and reloading it failed. Mapped to a 503 by <see cref="UseModelUnavailableHandler"/>.
/// </summary>
public sealed class ModelUnavailableException : Exception
{
    /// <summary>Creates the exception.</summary>
    /// <param name="modelId">The model the request was admitted for.</param>
    /// <param name="reason">Why it could not be re-activated.</param>
    public ModelUnavailableException(string modelId, string reason)
        : base($"Model '{modelId}' is no longer loaded and could not be re-activated: {reason}")
    {
        ModelId = modelId;
    }

    /// <summary>The model the request was admitted for.</summary>
    public string ModelId { get; }

    /// <summary>
    /// Maps <see cref="ModelUnavailableException"/> to a response: a 503 JSON error before the response has started, or - once an SSE
    /// stream has begun (keep-alive comments) - an OpenAI-style <c>data: {"error":...}</c> event followed by <c>data: [DONE]</c>.
    /// </summary>
    /// <param name="app">The application.</param>
    /// <returns><paramref name="app"/>.</returns>
    public static IApplicationBuilder UseModelUnavailableHandler(IApplicationBuilder app) =>
        app.Use(async (HttpContext context, Func<Task> next) =>
        {
            try
            {
                await next();
            }
            catch (ModelUnavailableException ex) when (!context.RequestAborted.IsCancellationRequested)
            {
                ServerLog.Info(ex.Message);
                var error = ErrorResponse.Internal(ex.Message, code: "model_not_loaded");
                if (!context.Response.HasStarted)
                {
                    context.Response.StatusCode = 503;
                    await context.Response.WriteAsJsonAsync(error, ServerJsonContext.Default.ErrorResponse, contentType: null, context.RequestAborted);
                    return;
                }
                await context.Response.WriteAsync("data: " + JsonSerializer.Serialize(error, ServerJsonContext.Default.ErrorResponse) + "\n\ndata: [DONE]\n\n",
                    context.RequestAborted);
                await context.Response.Body.FlushAsync(context.RequestAborted);
            }
        });
}
