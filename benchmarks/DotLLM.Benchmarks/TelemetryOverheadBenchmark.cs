using System.Diagnostics;
using BenchmarkDotNet.Attributes;
using DotLLM.Telemetry;

namespace DotLLM.Benchmarks;

/// <summary>
/// Verifies the dotLLM engine telemetry honours its "zero overhead when no listener" claim.
/// <para>
/// Each call site replicates the engine's per-token hot path: ask for an <see cref="Activity"/>
/// from <see cref="EngineTelemetry.ActivitySource"/>, then dispose it. With no
/// <see cref="ActivityListener"/> attached the call should resolve to a single branch on
/// <c>HasListeners()</c>, no allocations, ~1 ns per call.
/// </para>
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10, invocationCount: 1_000_000)]
public class TelemetryOverheadBenchmark
{
    /// <summary>Baseline: do nothing — establishes the noise floor.</summary>
    [Benchmark(Baseline = true)]
    public int NoOp() => 0;

    /// <summary>
    /// Per-decode-step span site as currently written in TextGenerator. The
    /// <see cref="EngineActivities.StartDecodeStep(int)"/> helper short-circuits on
    /// <see cref="ActivitySource.HasListeners"/> before any other work.
    /// </summary>
    [Benchmark]
    public int DecodeStepSpan_NoListener()
    {
        using var span = EngineActivities.StartDecodeStep(0);
        return span is null ? 0 : 1;
    }

    /// <summary>Counter site — Counter&lt;long&gt;.Add becomes a no-op when no MeterListener is attached.</summary>
    [Benchmark]
    public void CounterAdd_NoListener()
    {
        EngineTelemetry.DecodeTokens.Add(1);
    }

    /// <summary>Histogram site — Histogram&lt;double&gt;.Record becomes a no-op when no listener is attached.</summary>
    [Benchmark]
    public void HistogramRecord_NoListener()
    {
        EngineTelemetry.DecodeTokensPerSecond.Record(123.0);
    }
}
