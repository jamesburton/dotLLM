using DotLLM.Engine.Scheduler;
using DotLLM.Server.Models;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// GET /health and /ready — health and readiness probes; GET /v1/status — what the server is doing (#1023).
/// </summary>
public static class HealthEndpoint
{
    public static void Map(WebApplication app)
    {
        // Always 200 (liveness); the context facts appear once a model is loaded (#1289).
        app.MapGet("/health", (ServerState state) => Results.Ok(BuildHealth(state)));

        app.MapGet("/ready", (ServerState state) =>
            state.IsReady
                ? Results.Ok(new StatusResponse { Status = "ready" })
                : Results.StatusCode(503));

        // (#1023) Cheap and lock-free, for polling: "loading X since T" vs "busy with req#N" vs "idle" vs "no_model".
        app.MapGet("/v1/status", (ServerState state) => Results.Ok(state.GetStatus()));
    }

    /// <summary>
    /// The <c>/health</c> body (#1289): llama.cpp/azllm field names for the served context, the trained context,
    /// concurrent slots and whether a prompt cache is active. Lock-free reads of the live state.
    /// </summary>
    /// <param name="state">Server state.</param>
    /// <returns>The health payload; only <c>status</c> when no model is loaded.</returns>
    internal static HealthResponse BuildHealth(ServerState state)
    {
        var config = state.Config;
        if (!state.IsReady || config is null)
            return new HealthResponse { Status = "ok" };

        return new HealthResponse
        {
            Status = "ok",
            MaxContext = config.MaxSequenceLength,
            NCtxTrain = ModelMeta.TrainedContext(config, state.CurrentGguf),
            Slots = state.Scheduler is null ? 1 : (ServerStartup.ResolveSchedulerOptions(state.Options) ?? new ContinuousBatchSchedulerOptions()).MaxActiveSequences,
            PromptCacheActive = state.PrefixCache is not null || state.PrefixTrieManager is not null,
        };
    }
}
