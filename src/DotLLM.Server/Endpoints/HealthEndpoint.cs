using DotLLM.Server.Models;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// GET /health and /ready — health and readiness probes; GET /v1/status — what the server is doing (#1023).
/// </summary>
public static class HealthEndpoint
{
    public static void Map(WebApplication app)
    {
        app.MapGet("/health", () => Results.Ok(new StatusResponse { Status = "ok" }));

        app.MapGet("/ready", (ServerState state) =>
            state.IsReady
                ? Results.Ok(new StatusResponse { Status = "ready" })
                : Results.StatusCode(503));

        // (#1023) Cheap and lock-free, for polling: "loading X since T" vs "busy with req#N" vs "idle" vs "no_model".
        app.MapGet("/v1/status", (ServerState state) => Results.Ok(state.GetStatus()));
    }
}
