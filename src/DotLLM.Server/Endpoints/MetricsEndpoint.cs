using System.Text;
using DotLLM.Telemetry;
using Microsoft.AspNetCore.Http;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// <c>GET /metrics</c> — Prometheus text exposition (#1344). Backed by an in-process
/// <see cref="PrometheusMetricsCollector"/> over the <c>DotLLM.*</c> meters; no exporter package involved. Mapped only
/// when <see cref="ServerOptions.MetricsEnabled"/>. The route is outside <c>/v1/</c>, so it is never rate limited.
/// </summary>
public static class MetricsEndpoint
{
    /// <summary>Prometheus text exposition content type.</summary>
    public const string ContentType = "text/plain; version=0.0.4; charset=utf-8";

    public static void Map(WebApplication app)
    {
        var state = app.Services.GetRequiredService<ServerState>();
        if (!state.Options.MetricsEnabled) return;

        // Created at map time so counters accumulate from startup, not from the first scrape.
        var collector = new PrometheusMetricsCollector();
        app.Lifetime.ApplicationStopped.Register(collector.Dispose);

        app.MapGet("/metrics", () => Results.Text(collector.Render() + RenderServerInfo(state), ContentType));
    }

    /// <summary>Server-level series that are not meter instruments: readiness and the loaded model.</summary>
    internal static string RenderServerInfo(ServerState state)
    {
        var sb = new StringBuilder();
        sb.Append("# HELP dotllm_server_ready 1 when a model is loaded and serving.\n# TYPE dotllm_server_ready gauge\n");
        sb.Append("dotllm_server_ready ").Append(state.IsReady ? '1' : '0').Append('\n');
        if (state.IsReady)
        {
            sb.Append("# HELP dotllm_model_info The loaded model (value is always 1).\n# TYPE dotllm_model_info gauge\n");
            sb.Append("dotllm_model_info{model=\"").Append(PrometheusMetricsCollector.EscapeLabelValue(state.Options.ModelId)).Append("\"} 1\n");
        }
        return sb.ToString();
    }
}
