# Telemetry & Observability — dotLLM

dotLLM emits engine metrics and per-request tracing through the built-in
.NET diagnostics primitives — `System.Diagnostics.Metrics` and
`System.Diagnostics.Activity` — and ships an OpenTelemetry/OTLP wiring
out of the box on the server. Nothing is recorded when nothing is
subscribed: each call site is a single `Instrument.Enabled` /
`ActivitySource.HasListeners()` branch, zero allocations, ~10-20 ns
microbenched (`TelemetryOverheadBenchmark`).

The runtime surface lives in the `DotLLM.Telemetry` assembly:

| Symbol | Purpose |
|--------|---------|
| `EngineTelemetry.Meter` | `Meter` named `DotLLM.Engine` carrying every engine instrument |
| `EngineTelemetry.ActivitySource` | `ActivitySource` named `DotLLM.Engine` |
| `EngineActivities` | Span-name constants and zero-overhead `Start*` helpers |
| `TelemetryTags` | OpenTelemetry-style attribute key constants |
| `OpenTelemetryServiceExtensions.AddDotLLMOpenTelemetry` | DI helper that wires metrics + tracing to OTLP and ASP.NET instrumentation |

## Metrics

All instruments live on `Meter("DotLLM.Engine")` and are tagged by
`model` (the architecture name from `ModelConfig.Architecture`).

| Instrument | Type | Unit | Description |
|------------|------|------|-------------|
| `dotllm.engine.tokens.prefill` | Counter&lt;long&gt; | `tokens` | Prompt tokens processed during prefill (excludes prefix-cache hits) |
| `dotllm.engine.tokens.decode` | Counter&lt;long&gt; | `tokens` | Tokens generated during decode |
| `dotllm.engine.tokens_per_second.prefill` | Histogram&lt;double&gt; | `tokens/s` | Prefill throughput per request |
| `dotllm.engine.tokens_per_second.decode` | Histogram&lt;double&gt; | `tokens/s` | Decode throughput per request |
| `dotllm.engine.time_to_first_token_ms` | Histogram&lt;double&gt; | `ms` | Time from request start to first generated token |
| `dotllm.engine.request.queue_depth` | ObservableGauge&lt;long&gt; | `{request}` | Scheduler queue depth — emits `-1` until Step 35 wires the continuous-batching scheduler |
| `dotllm.engine.kvcache.utilization` | ObservableGauge&lt;double&gt; | `1` | Paged KV-cache utilization in `[0, 1]` — emits `-1` until Step 35 |

The two observable gauges accept a callback via
`EngineTelemetry.SetQueueDepthProvider` /
`SetKvCacheUtilizationProvider`. When unset they emit `-1` as a
sentinel so downstream dashboards can detect "not wired yet".

## Tracing

Per-request hierarchy emitted from the engine:

```
dotllm.request                    (root — TextGenerator.Generate / Stream)
 ├── dotllm.prefill               (one per call; prefill_token_count, prefill_duration_ms)
 ├── dotllm.sample                (around the sampler pipeline)
 └── dotllm.decode_step           (~1% sampled by step index)
```

`dotllm.decode_step` spans are emitted deterministically at 1% — every
100-th decode step — so trace volume stays bounded even for very long
generations.

### Attributes

`TelemetryTags` defines the keys used. Notable ones on the root span:

| Tag | Notes |
|-----|-------|
| `model` | `ModelConfig.Architecture.ToString()` |
| `dotllm.max_tokens` | `InferenceOptions.MaxTokens` |
| `dotllm.sampler.temperature` / `top_k` / `top_p` | Sampler config snapshot |
| `dotllm.prompt_tokens` / `dotllm.generated_tokens` / `dotllm.cached_tokens` | Set on completion |
| `dotllm.finish_reason` | `Stop`, `Length`, `ToolCalls`, … |

## Hot-path discipline

Every metric and span site is guarded so the cost when no listener is
subscribed is a single conditional branch and zero allocations.
`TelemetryOverheadBenchmark` is the regression test for this claim.

For example, the per-token sample site reduces to:

```csharp
using var sampleSpan = telemetry.StartSample(); // null when no listener
// existing sampler work
```

and the per-request counter site to:

```csharp
if (EngineTelemetry.DecodeTokens.Enabled)
    EngineTelemetry.DecodeTokens.Add(decoded, modelTag);
```

The `TelemetryRecorder` struct caches the model `KeyValuePair` once
per call so per-step writes don't allocate.

## Server integration

`DotLLM.Server.ServerStartup.BuildApp` calls
`services.AddDotLLMOpenTelemetry(state.Options.ModelId)` only when the
standard OpenTelemetry environment variable
`OTEL_EXPORTER_OTLP_ENDPOINT` is set. That single call:

- Adds `Meter("DotLLM.Engine")` and ASP.NET Core instrumentation to
  the metrics pipeline, exports via OTLP.
- Adds `ActivitySource("DotLLM.Engine")` and ASP.NET Core
  instrumentation to the tracing pipeline, exports via OTLP.

Standard OTLP configuration goes through the usual env vars
(`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_PROTOCOL`,
`OTEL_EXPORTER_OTLP_HEADERS`, …). Per-signal overrides
(`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, etc.) are honoured by the
exporter directly.

ASP.NET request spans become parents of the engine
`dotllm.request` span via the ambient `Activity.Current`, so traces
in Jaeger / Tempo / Honeycomb show the inference work nested under
each HTTP request.

### Example: ship traces and metrics to a local OTel collector

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
dotllm serve --model llama-3-8b-q4_k_m.gguf
```

### Prometheus scrape endpoint: `GET /metrics`

`dotllm serve` exposes the same instruments in the Prometheus text format (0.0.4) at `GET /metrics`, with no OTel
collector or exporter package: a `MeterListener` (`PrometheusMetricsCollector`) folds the `DotLLM.*` meters into
counters, gauges and fixed-bucket histograms. It is on by default for loopback binds only; `--metrics` forces it on,
`--no-metrics` off (see [SERVER.md](SERVER.md#health--readiness)). Starting the collector is what enables the
instruments, so with the endpoint off the hot path stays at the usual `Enabled` branch.

```yaml
scrape_configs:
  - job_name: dotllm
    static_configs: [{ targets: ["localhost:8080"] }]
```

Name mapping: dots become underscores, a unit suffix is appended unless the name already carries it (`By` is
`_bytes`), counters get `_total`. Histogram buckets: `ms` instruments use 1 ms to 120 s, `tokens/s` use 1 to 10 000.

| Series | Type | Labels |
|--------|------|--------|
| `dotllm_engine_tokens_prefill_total`, `dotllm_engine_tokens_decode_total`, `dotllm_engine_tokens_cached_total` | counter | `model` |
| `dotllm_engine_requests_total` | counter | `model`, `finish_reason` |
| `dotllm_engine_request_duration_ms` | histogram | `model` |
| `dotllm_engine_tokens_per_second_prefill`, `..._decode` | histogram | `model` |
| `dotllm_engine_time_to_first_token_ms` | histogram | `model` (TextGenerator path only) |
| `dotllm_engine_request_queue_depth`, `dotllm_engine_kvcache_utilization` | gauge | none (`-1` = provider not wired) |
| `dotllm_server_ready`, `dotllm_model_info{model=...}` | gauge | none / `model` |
| `dotllm_moe_*` | counter / gauge | none |

Both generation paths record the per-request series: `TextGenerator` through `TelemetryRecorder` and the
continuous-batch scheduler at request completion (`EngineTelemetry.RecordCompletion`).

Not yet exposed (no instrument exists): prefix-cache hit/miss by tier, snapshot bytes per tier, MTP drafted/accepted
counters, device memory used/free.

## Diagnostics versus telemetry

`DotLLM.Diagnostics` (hooks, logit lens, SAE) targets ML
interpretability — capturing tensors mid-forward-pass. This module
targets production observability — latency, throughput, error rates,
distributed tracing. They are independent and impose no overhead
when disabled.
