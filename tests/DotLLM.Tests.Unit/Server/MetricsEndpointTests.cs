using System.Diagnostics.Metrics;
using System.Net;
using DotLLM.Server;
using DotLLM.Telemetry;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>#1344: in-process Prometheus exposition and the <c>GET /metrics</c> gate.</summary>
public sealed class MetricsEndpointTests
{
    [Fact]
    public void Exposition_HasExactLines_ForCounterGaugeAndHistogram()
    {
        using var meter = new Meter("DotLLM.Test.Exposition");
        var counter = meter.CreateCounter<long>("dotllm.test.requests", unit: "{request}", description: "Test requests.");
        var hist = meter.CreateHistogram<double>("dotllm.test.latency_ms", unit: "ms", description: "Latency.");
        long gaugeValue = 7;
        _ = meter.CreateObservableGauge("dotllm.test.depth", () => gaugeValue, description: "Depth.");

        using var collector = new PrometheusMetricsCollector(n => n == "DotLLM.Test.Exposition");

        counter.Add(2, new KeyValuePair<string, object?>("finish_reason", "Stop"));
        counter.Add(3, new KeyValuePair<string, object?>("finish_reason", "Stop"));
        counter.Add(1, new KeyValuePair<string, object?>("finish_reason", "Length"));
        // bounds: 1,5,10,25,50,100,... ; 5 is on a boundary (le is inclusive), 7 -> le=10, 1000000 -> +Inf only
        hist.Record(5);
        hist.Record(7);
        hist.Record(1_000_000);

        string text = collector.Render();
        var lines = text.Split('\n');

        Assert.Contains("# HELP dotllm_test_requests_total Test requests.", lines);
        Assert.Contains("# TYPE dotllm_test_requests_total counter", lines);
        Assert.Contains("dotllm_test_requests_total{finish_reason=\"Stop\"} 5", lines);
        Assert.Contains("dotllm_test_requests_total{finish_reason=\"Length\"} 1", lines);

        Assert.Contains("# TYPE dotllm_test_depth gauge", lines);
        Assert.Contains("dotllm_test_depth 7", lines);
        gaugeValue = 9;
        Assert.Contains("dotllm_test_depth 9", collector.Render().Split('\n'));

        Assert.Contains("# TYPE dotllm_test_latency_ms histogram", lines);
        Assert.Contains("dotllm_test_latency_ms_bucket{le=\"1\"} 0", lines);
        Assert.Contains("dotllm_test_latency_ms_bucket{le=\"5\"} 1", lines);      // boundary value lands in its own bucket
        Assert.Contains("dotllm_test_latency_ms_bucket{le=\"10\"} 2", lines);     // cumulative
        Assert.Contains("dotllm_test_latency_ms_bucket{le=\"120000\"} 2", lines); // 1e6 is beyond the last finite bound
        Assert.Contains("dotllm_test_latency_ms_bucket{le=\"+Inf\"} 3", lines);
        Assert.Contains("dotllm_test_latency_ms_sum 1000012", lines);
        Assert.Contains("dotllm_test_latency_ms_count 3", lines);

        Assert.EndsWith("\n", text);
        Assert.DoesNotContain("\r", text);
    }

    [Fact]
    public void LabelValues_AreEscaped_AndFamiliesWithoutSamplesAreOmitted()
    {
        using var meter = new Meter("DotLLM.Test.Escape");
        var counter = meter.CreateCounter<long>("dotllm.test.escaped");
        _ = meter.CreateCounter<long>("dotllm.test.never");
        using var collector = new PrometheusMetricsCollector(n => n == "DotLLM.Test.Escape");

        counter.Add(1, new KeyValuePair<string, object?>("model", "a\"b\\c\nd"));

        string text = collector.Render();
        Assert.Contains("dotllm_test_escaped_total{model=\"a\\\"b\\\\c\\nd\"} 1", text.Split('\n'));
        Assert.DoesNotContain("dotllm_test_never", text);
    }

    [Fact]
    public void EngineRecordCompletion_ProducesTheDocumentedSeries()
    {
        using var collector = new PrometheusMetricsCollector();
        EngineTelemetry.RecordCompletion("TestArch", promptTokens: 100, cachedTokens: 40, generatedTokens: 11,
            prefillMs: 50, decodeMs: 100, totalMs: 160, finishReason: "Stop");

        var lines = collector.Render().Split('\n');
        Assert.Contains("dotllm_engine_requests_total{finish_reason=\"Stop\",model=\"TestArch\"} 1", lines);
        Assert.Contains("dotllm_engine_tokens_prefill_total{model=\"TestArch\"} 60", lines);
        Assert.Contains("dotllm_engine_tokens_decode_total{model=\"TestArch\"} 10", lines);
        Assert.Contains("dotllm_engine_tokens_cached_total{model=\"TestArch\"} 40", lines);
        Assert.Contains("dotllm_engine_request_duration_ms_count{model=\"TestArch\"} 1", lines);
    }

    [Theory]
    [InlineData("localhost", true)]
    [InlineData("127.0.0.1", true)]
    [InlineData("::1", true)]
    [InlineData("[::1]", true)]
    [InlineData("0.0.0.0", false)]
    [InlineData("192.168.1.5", false)]
    [InlineData("myhost.lan", false)]
    public void DefaultIsOnForLoopbackOnly(string host, bool expected)
    {
        Assert.Equal(expected, new ServerOptions { Model = "m", Host = host }.MetricsEnabled);
        Assert.True(new ServerOptions { Model = "m", Host = host, Metrics = true }.MetricsEnabled);
        Assert.False(new ServerOptions { Model = "m", Host = host, Metrics = false }.MetricsEnabled);
    }

    [Fact]
    public void Parse_AcceptsMetricsFlags()
    {
        Assert.Null(ServerOptions.Parse(["--model", "m"]).Metrics);
        Assert.True(ServerOptions.Parse(["--model", "m", "--metrics"]).Metrics);
        Assert.False(ServerOptions.Parse(["--model", "m", "--no-metrics"]).Metrics);
    }

    private static async Task<(WebApplication App, ServerState State, HttpClient Http)> StartAsync(ServerOptions options)
    {
        var builder = WebApplication.CreateSlimBuilder();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        var state = ServerStartup.CreateBareState(options);
        builder.Services.AddSingleton(state);
        var app = builder.Build();
        app.MapDotLLMEndpoints(serveUi: false);
        await app.StartAsync();
        return (app, state, new HttpClient { BaseAddress = new Uri(app.Urls.First()) });
    }

    [Fact]
    public async Task Endpoint_ServesTextFormat_OnLoopback()
    {
        var (app, state, http) = await StartAsync(new ServerOptions { Model = "none", Host = "127.0.0.1" });
        try
        {
            EngineTelemetry.RecordCompletion("EndpointArch", 10, 0, 5, 5, 10, 20, "Length");
            var resp = await http.GetAsync("/metrics");
            Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
            Assert.Equal("text/plain; version=0.0.4; charset=utf-8", resp.Content.Headers.ContentType!.ToString());
            string body = await resp.Content.ReadAsStringAsync();
            Assert.Contains("dotllm_engine_requests_total{finish_reason=\"Length\",model=\"EndpointArch\"} 1", body.Split('\n'));
            Assert.Contains("dotllm_server_ready 0", body.Split('\n'));
        }
        finally { http.Dispose(); await app.DisposeAsync(); state.Dispose(); }
    }

    [Theory]
    [InlineData("0.0.0.0", null, HttpStatusCode.NotFound)]    // exposed bind: off unless asked for
    [InlineData("0.0.0.0", true, HttpStatusCode.OK)]
    [InlineData("127.0.0.1", false, HttpStatusCode.NotFound)] // explicit opt-out
    public async Task Endpoint_IsGated(string host, bool? metrics, HttpStatusCode expected)
    {
        var (app, state, http) = await StartAsync(new ServerOptions { Model = "none", Host = host, Metrics = metrics });
        try { Assert.Equal(expected, (await http.GetAsync("/metrics")).StatusCode); }
        finally { http.Dispose(); await app.DisposeAsync(); state.Dispose(); }
    }
}
