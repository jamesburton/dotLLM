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
    /// Maps <see cref="DotLLM.Core.Configuration.DeviceMemoryAdmissionException"/> the same way as a 400 <c>context_length_exceeded</c>
    /// (issue #1068).
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
                if (context.Request.Path.StartsWithSegments("/v1/messages"))
                {
                    // Anthropic surface: its envelope, and `event: error` once the stream has started (#1023 review).
                    await Endpoints.MessagesEndpoint.WriteErrorAsync(context, 503, "api_error", ex.Message);
                    return;
                }
                var error = ErrorResponse.Internal(ex.Message, code: "model_not_loaded");
                await Endpoints.SseResponse.WriteOpenAiErrorAsync(context, 503, error);
            }
            catch (DotLLM.Core.Configuration.DeviceMemoryAdmissionException ex) when (!context.RequestAborted.IsCancellationRequested)
            {
                // Issue #1068: the context this request needs does not fit in free GPU memory, and growing into shared system
                // memory was refused. Reported as context_length_exceeded so agent clients compact and retry, instead of the
                // connection dropping mid-stream.
                ServerLog.Info(ex.Message);
                if (context.Request.Path.StartsWithSegments("/v1/messages"))
                {
                    await Endpoints.MessagesEndpoint.WriteErrorAsync(context, 400, "invalid_request_error", ex.Message);
                    return;
                }
                var error = ErrorResponse.InvalidRequest(ex.Message, param: "messages", code: "context_length_exceeded");
                await Endpoints.SseResponse.WriteOpenAiErrorAsync(context, 400, error);
            }
        });
}
