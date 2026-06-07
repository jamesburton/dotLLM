using Microsoft.Extensions.DependencyInjection;
using OpenTelemetry;
using OpenTelemetry.Metrics;
using OpenTelemetry.Resources;
using OpenTelemetry.Trace;

namespace DotLLM.Telemetry;

/// <summary>
/// Dependency-injection helpers that wire the dotLLM <see cref="EngineTelemetry"/> instruments
/// to OpenTelemetry with the OTLP exporter and ASP.NET Core instrumentation.
/// </summary>
public static class OpenTelemetryServiceExtensions
{
    /// <summary>
    /// Adds OpenTelemetry metrics + tracing pipelines for the dotLLM engine. The OTLP exporter
    /// is configured from the standard OpenTelemetry environment variables
    /// (<c>OTEL_EXPORTER_OTLP_ENDPOINT</c>, <c>OTEL_EXPORTER_OTLP_HEADERS</c>,
    /// <c>OTEL_EXPORTER_OTLP_PROTOCOL</c>, …) — see the OpenTelemetry specification for the
    /// full list. Enables ASP.NET Core HTTP instrumentation so incoming chat/completions
    /// requests get traced automatically; engine spans become children via
    /// <see cref="System.Diagnostics.Activity.Current"/> propagation.
    /// </summary>
    /// <param name="services">The DI service collection.</param>
    /// <param name="serviceName">Service name used for the OpenTelemetry <see cref="Resource"/>.</param>
    /// <returns>The same <see cref="IServiceCollection"/> for chaining.</returns>
    public static IServiceCollection AddDotLLMOpenTelemetry(this IServiceCollection services,
        string serviceName = "dotllm-server")
    {
        services.AddOpenTelemetry()
            .ConfigureResource(rb => rb.AddService(serviceName))
            .WithMetrics(builder =>
            {
                builder.AddMeter(EngineTelemetry.Name)
                       .AddAspNetCoreInstrumentation()
                       .AddOtlpExporter();
            })
            .WithTracing(builder =>
            {
                builder.AddSource(EngineTelemetry.Name)
                       .AddAspNetCoreInstrumentation()
                       .AddOtlpExporter();
            });

        return services;
    }
}
