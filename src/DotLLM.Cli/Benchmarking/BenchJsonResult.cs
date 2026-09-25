namespace DotLLM.Cli.Benchmarking;

/// <summary>One repetition in the <c>bench --json</c> output.</summary>
public sealed class BenchJsonRep
{
    /// <summary>Prefill wall time, milliseconds.</summary>
    public double PrefillMs { get; init; }

    /// <summary>Total decode wall time, milliseconds.</summary>
    public double DecodeMs { get; init; }

    /// <summary>Prefill throughput, tokens/second.</summary>
    public double PrefillTokS { get; init; }

    /// <summary>Decode throughput, tokens/second.</summary>
    public double DecodeTokS { get; init; }

    /// <summary>Maps a measured <see cref="BenchRep"/> to its JSON shape.</summary>
    public static BenchJsonRep From(BenchRep rep) => new()
    {
        PrefillMs = Math.Round(rep.PrefillMs, 2),
        DecodeMs = Math.Round(rep.DecodeMs, 2),
        PrefillTokS = Math.Round(rep.PrefillTokS, 2),
        DecodeTokS = Math.Round(rep.DecodeTokS, 2),
    };
}

/// <summary>Machine-readable result of <c>dotllm bench --json</c>.</summary>
public sealed class BenchJsonResult
{
    /// <summary>Model name (file name without quant suffix / extension).</summary>
    public required string Model { get; init; }

    /// <summary>Resolved GGUF file path.</summary>
    public required string ModelPath { get; init; }

    /// <summary>Quantization label (e.g. Q8_0).</summary>
    public required string Quant { get; init; }

    /// <summary>Backend: cpu, vulkan, or cuda.</summary>
    public required string Backend { get; init; }

    /// <summary>Device label (GPU name or cpu-{N}t).</summary>
    public required string Device { get; init; }

    /// <summary>Host machine name (lowercase).</summary>
    public required string Host { get; init; }

    /// <summary>Point-in-time runtime version / commit (results.csv runtime_version).</summary>
    public required string Commit { get; init; }

    /// <summary>Model load wall time, milliseconds (excluded from reps).</summary>
    public double LoadMs { get; init; }

    /// <summary>Actual prompt token count per repetition.</summary>
    public int PromptTokens { get; init; }

    /// <summary>Greedy decode steps timed per repetition.</summary>
    public int DecodeTokens { get; init; }

    /// <summary>Extra untimed context tokens fed between prefill and decode.</summary>
    public int Depth { get; init; }

    /// <summary>Context depth the timed decode ran at (prompt + depth).</summary>
    public int DecodeCtxDepth { get; init; }

    /// <summary>The discarded warm-up repetition.</summary>
    public required BenchJsonRep Warmup { get; init; }

    /// <summary>Measured repetitions (warm-up excluded), in run order.</summary>
    public required BenchJsonRep[] Reps { get; init; }

    /// <summary>Median prefill throughput, tokens/second.</summary>
    public double PrefillTokSMedian { get; init; }

    /// <summary>Best (fastest-rep) prefill throughput, tokens/second.</summary>
    public double PrefillTokSBest { get; init; }

    /// <summary>Median decode throughput, tokens/second.</summary>
    public double DecodeTokSMedian { get; init; }

    /// <summary>Best (fastest-rep) decode throughput, tokens/second.</summary>
    public double DecodeTokSBest { get; init; }

    /// <summary>Median prefill wall time, milliseconds.</summary>
    public double PrefillMsMedian { get; init; }

    /// <summary>Minimum prefill wall time, milliseconds.</summary>
    public double PrefillMsMin { get; init; }

    /// <summary>Median decode wall time, milliseconds.</summary>
    public double DecodeMsMedian { get; init; }

    /// <summary>Minimum decode wall time, milliseconds.</summary>
    public double DecodeMsMin { get; init; }

    /// <summary>Ready-to-paste benchmarks/perf-matrix/results.csv row.</summary>
    public required string CsvRow { get; init; }
}
