using DotLLM.Server.Models;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// <c>GET /v1/models</c> — list resident models (#369: active model plus any stashed-but-loaded
/// models, each with idle/expiry state for keep-alive observability) — and
/// <c>GET /v1/models/{id}</c> (#450), which is what the OpenAI SDK's
/// <c>client.models.retrieve()</c> calls.
/// </summary>
public static class ModelEndpoint
{
    public static void Map(WebApplication app)
    {
        app.MapGet("/v1/models", (ServerState state) => BuildList(state));

        // A catch-all segment, because model ids here are HuggingFace repo ids and therefore
        // routinely contain '/' ("Qwen/Qwen3-4B"). Literal routes registered elsewhere under
        // /v1/models (available, load, inspect) are more specific and still win.
        app.MapGet("/v1/models/{**id}", (string id, ServerState state) =>
        {
            var model = Retrieve(state, id);
            return model is null
                ? Results.NotFound(ErrorResponse.NotFound(
                    $"The model '{id}' does not exist", param: "model", code: "model_not_found"))
                : Results.Ok(model);
        });
    }

    /// <summary>
    /// The <c>GET /v1/models</c> payload: every resident model, or — on a bare server that has
    /// never loaded one — a single entry for the configured-but-unloaded model id, matching the
    /// original always-one-row behavior that pre-#369 callers depend on.
    /// </summary>
    internal static ModelListResponse BuildList(ServerState state)
    {
        var now = DateTimeOffset.UtcNow;
        var data = state.ListResidentModels().Select(m => new ModelInfoDto
        {
            Id = m.Key,
            Created = now.ToUnixTimeSeconds(),
            IsActive = m.IsActive,
            IdleSeconds = Math.Max(0, (now - m.LastUsedUtc).TotalSeconds),
            KeepAliveSeconds = m.EffectiveKeepAliveSeconds,
            ExpiresInSeconds = m.ExpiresInSeconds,
            SizeBytes = m.EstimatedBytes,
        }).ToArray();

        if (data.Length == 0)
        {
            data =
            [
                new ModelInfoDto
                {
                    Id = state.Options.ModelId,
                    Created = now.ToUnixTimeSeconds(),
                    IsActive = false,
                    KeepAliveSeconds = state.Residency.DefaultKeepAliveSeconds,
                }
            ];
        }

        return new ModelListResponse { Data = data };
    }

    /// <summary>
    /// Looks up one model by id for <c>GET /v1/models/{id}</c> (#450), or <c>null</c> when this
    /// server does not know it. Resolution goes through <see cref="BuildList"/> so retrieve can
    /// never disagree with list — including the bare-server fallback, where the configured
    /// model id must retrieve rather than 404.
    /// </summary>
    internal static ModelInfoDto? Retrieve(ServerState state, string id)
    {
        if (string.IsNullOrWhiteSpace(id))
            return null;

        foreach (var model in BuildList(state).Data)
        {
            if (string.Equals(model.Id, id, StringComparison.Ordinal))
                return model;
        }
        return null;
    }
}
