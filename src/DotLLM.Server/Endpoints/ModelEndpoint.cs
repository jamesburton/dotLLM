using DotLLM.Server.Models;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// GET /v1/models — list resident models (#369: active model plus any stashed-but-loaded models,
/// each with idle/expiry state for keep-alive observability).
/// </summary>
public static class ModelEndpoint
{
    public static void Map(WebApplication app) =>
        app.MapGet("/v1/models", (ServerState state) =>
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

            // Bare server (no model ever loaded): report the configured-but-unloaded model id so
            // pre-#369 callers still see an entry, matching the original always-one-row behavior.
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
        });
}
