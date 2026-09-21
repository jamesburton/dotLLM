using DotLLM.Server.Models;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// Runtime server settings (#454):
/// <list type="bullet">
///   <item><c>GET /v1/settings</c> — read keep-alive / residency / sweep settings (always available).</item>
///   <item><c>PUT /v1/settings</c> — partial update (gated by <see cref="ServerOptions.AllowModelAdminApi"/>).</item>
/// </list>
/// </summary>
/// <remarks>
/// Deliberately a new resource rather than an extension of <c>/v1/config</c>: that endpoint's
/// body is the mutable <i>sampling</i> defaults (temperature, top_p, …), a different concept with
/// a different audience. Overloading it would make a tray's "set keep-alive" round-trip carry —
/// and risk clobbering — every sampling default too.
/// </remarks>
public static class SettingsEndpoint
{
    public static void Map(WebApplication app)
    {
        app.MapGet("/v1/settings", (ServerState state) => Results.Ok(Read(state)));

        app.MapPut("/v1/settings", (SettingsUpdateRequest request, ServerState state) =>
        {
            if (!state.Options.AllowModelAdminApi)
                return AdminGate.Forbidden("PUT /v1/settings");

            if (request.MaxResidentModels is < 1)
                return Results.BadRequest(new ErrorResponse { Error = "max_resident_models must be >= 1" });
            if (request.ResidentMemoryBudgetBytes is < 0)
                return Results.BadRequest(new ErrorResponse { Error = "resident_memory_budget_bytes must be >= 0" });
            // Bounded to exactly the range RunIdleSweepLoopAsync clamps to. Accepting 0.05 and
            // then silently sweeping at 0.1 while GET reported 0.05 would make the endpoint lie.
            if (request.IdleSweepIntervalSeconds is < 0.1 or > 3600)
                return Results.BadRequest(new ErrorResponse
                {
                    Error = "idle_sweep_interval_seconds must be between 0.1 and 3600",
                });

            return Results.Ok(Apply(state, request));
        });
    }

    /// <summary>Current settings snapshot. Separated from the route so tests can call it directly.</summary>
    public static SettingsDto Read(ServerState state) => new()
    {
        KeepAliveSeconds = state.Residency.DefaultKeepAliveSeconds,
        MaxResidentModels = state.Residency.MaxResidentModels,
        ResidentMemoryBudgetBytes = state.Residency.MemoryBudgetBytes,
        IdleSweepIntervalSeconds = state.IdleSweepIntervalSeconds,
        ModelAdminApiEnabled = state.Options.AllowModelAdminApi,
        LoraAdminApiEnabled = state.Options.AllowLoraAdminApi,
        DisabledModels = state.Catalog.DisabledKeys().ToArray(),
    };

    /// <summary>
    /// Applies a partial settings update and reports what took effect.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Every mutable field is stored on <see cref="ModelResidencyManager"/> or on
    /// <see cref="ServerState.IdleSweepIntervalSeconds"/> — <b>never</b> on
    /// <see cref="ServerState.Options"/>, which is replaced wholesale on every model swap and
    /// would silently revert the change at the next load.
    /// </para>
    /// <para>
    /// Tightening <c>max_resident_models</c> or the byte budget runs
    /// <see cref="ModelResidencyManager.EnforceBudget"/> immediately, so "takes effect" means now
    /// rather than at the next load. Only stashed models are evicted; the active model is never
    /// dropped by a settings change.
    /// </para>
    /// <para>
    /// <c>restart_required</c> is currently always empty — every field on
    /// <see cref="SettingsUpdateRequest"/> is live-applicable. It is part of the contract so a
    /// future non-live setting can be reported rather than silently ignored.
    /// </para>
    /// </remarks>
    public static SettingsUpdateResponse Apply(ServerState state, SettingsUpdateRequest request)
    {
        var applied = new List<string>();
        var evicted = new List<string>();

        if (request.KeepAliveSeconds is { } keepAlive)
        {
            state.Residency.DefaultKeepAliveSeconds = keepAlive;
            applied.Add("keep_alive_seconds");
        }

        bool residencyChanged = false;
        if (request.MaxResidentModels is { } maxResident)
        {
            state.Residency.MaxResidentModels = Math.Max(1, maxResident);
            applied.Add("max_resident_models");
            residencyChanged = true;
        }

        if (request.ResidentMemoryBudgetBytes is { } budget)
        {
            state.Residency.MemoryBudgetBytes = budget;
            applied.Add("resident_memory_budget_bytes");
            residencyChanged = true;
        }

        if (request.IdleSweepIntervalSeconds is { } sweep)
        {
            state.IdleSweepIntervalSeconds = sweep;
            applied.Add("idle_sweep_interval_seconds");
        }

        if (residencyChanged)
            evicted.AddRange(state.Residency.EnforceBudget(state.EstimatedBytes).Select(s => s.Key));

        return new SettingsUpdateResponse
        {
            Settings = Read(state),
            Applied = applied.ToArray(),
            RestartRequired = [],
            Evicted = evicted.ToArray(),
        };
    }
}

/// <summary>
/// Shared 403 for the #454 admin-gated routes. Unlike the bare <c>403</c> the LoRA routes return,
/// this carries an <see cref="ErrorResponse"/> body naming the flag to set — a tray app surfaces
/// that directly instead of guessing why a button did nothing.
/// </summary>
internal static class AdminGate
{
    internal static IResult Forbidden(string route) =>
        Results.Json(
            new ErrorResponse
            {
                Error = $"{route} is disabled. Start the server with --allow-model-admin "
                      + "(ServerOptions.AllowModelAdminApi) to enable the model-administration API.",
            },
            statusCode: StatusCodes.Status403Forbidden);
}
