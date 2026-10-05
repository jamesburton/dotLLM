using Microsoft.Extensions.Hosting;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// <c>POST /v1/admin/shutdown</c> (issue #722): asks the server to stop gracefully. Gated by <c>--allow-model-admin</c> like every other write route.
/// </summary>
/// <remarks>
/// <para>
/// The answer is <c>202</c> and the stop happens a moment later (so the response itself is delivered): the host stops accepting connections, waits
/// for in-flight requests within its shutdown timeout, and <c>serve</c> then disposes the model state - the same drain a Ctrl+C gets. This is the
/// route that lets a client (the tray, <c>dotllm stop --server</c>) stop a server it did not start, and stop one it did start without a hard kill.
/// </para>
/// </remarks>
public static class AdminEndpoint
{
    /// <summary>Delay between answering and stopping, so the 202 reaches the client first.</summary>
    internal static readonly TimeSpan StopDelay = TimeSpan.FromMilliseconds(250);

    public static void Map(WebApplication app) =>
        app.MapPost("/v1/admin/shutdown", (ServerState state, IHostApplicationLifetime lifetime) =>
        {
            if (!state.Options.AllowModelAdminApi)
                return AdminGate.Forbidden("POST /v1/admin/shutdown");

            _ = Task.Run(async () =>
            {
                await Task.Delay(StopDelay);
                lifetime.StopApplication();
            });
            return Results.Json(new Models.StatusResponse { Status = "shutting_down" }, ServerJsonContext.Default.StatusResponse, statusCode: StatusCodes.Status202Accepted);
        });
}
