using System.Text.Json;

namespace DotLLM.Server.ChunkCache;

/// <summary>
/// The cost model behind the <c>shadow</c> plan's predicted TTFT (spec §5.5, §5.9): per-backend constants for selective
/// recompute, assembly and prefill. The defaults are provisional (embedded <c>chunk-cost-model.json</c>, with their
/// sources); a later calibration task replaces them per model and kernel.
/// </summary>
/// <param name="UsPerRowKey">Microseconds per (recomputed row × visible key), whole forward.</param>
/// <param name="AssemblyMsPerKTokens">Milliseconds to assemble 1,000 stored chunk tokens into the KV cache.</param>
/// <param name="PrefillTokensPerSec">Ordinary prefill throughput, used when the request itself measured none.</param>
internal sealed record ChunkCostModel(double UsPerRowKey, double AssemblyMsPerKTokens, double PrefillTokensPerSec)
{
    private static readonly Lazy<IReadOnlyDictionary<string, ChunkCostModel>> Defaults = new(LoadDefaults);

    /// <summary>The model for the CPU backend when the embedded defaults are unreadable (never expected).</summary>
    internal static ChunkCostModel Fallback { get; } = new(0.9, 111.0, 140.0);

    /// <summary>The default model for a backend tag; the CPU model for an unknown tag.</summary>
    /// <param name="backend">Backend tag (<c>cpu</c>, <c>vulkan</c>).</param>
    /// <returns>The model.</returns>
    public static ChunkCostModel For(string? backend)
    {
        var table = Defaults.Value;
        if (backend is not null && table.TryGetValue(backend, out var model))
            return model;
        return table.TryGetValue("cpu", out var cpu) ? cpu : Fallback;
    }

    /// <summary>Predicted milliseconds to prefill <paramref name="tokens"/> at <paramref name="tokensPerSec"/>.</summary>
    public static double PrefillMs(long tokens, double tokensPerSec) =>
        tokens <= 0 || tokensPerSec <= 0 || !double.IsFinite(tokensPerSec) ? 0 : tokens * 1000.0 / tokensPerSec;

    /// <summary>Predicted milliseconds to assemble <paramref name="tokens"/> stored chunk tokens.</summary>
    public double AssemblyMs(long tokens) => tokens <= 0 ? 0 : AssemblyMsPerKTokens * tokens / 1000.0;

    /// <summary>
    /// Predicted milliseconds to recompute the first <paramref name="rows"/> rows of a chunk placed at absolute token
    /// <paramref name="start"/>: row <c>start + i</c> attends to <c>start + i + 1</c> keys.
    /// </summary>
    public double RecomputeMs(long start, long rows)
    {
        if (rows <= 0)
            return 0;
        double rowKeys = rows * (double)start + rows * (rows + 1) / 2.0;
        return UsPerRowKey * rowKeys / 1000.0;
    }

    private static IReadOnlyDictionary<string, ChunkCostModel> LoadDefaults()
    {
        var table = new Dictionary<string, ChunkCostModel>(StringComparer.Ordinal);
        try
        {
            using var stream = typeof(ChunkCostModel).Assembly.GetManifestResourceStream("DotLLM.Server.ChunkCache.chunk-cost-model.json");
            if (stream is null)
                return table;
            using var doc = JsonDocument.Parse(stream);
            foreach (var backend in doc.RootElement.GetProperty("backends").EnumerateObject())
            {
                var b = backend.Value;
                table[backend.Name] = new ChunkCostModel(
                    b.GetProperty("us_per_row_key").GetDouble(),
                    b.GetProperty("assembly_ms_per_ktokens").GetDouble(),
                    b.GetProperty("prefill_tokens_per_sec").GetDouble());
            }
        }
        catch (Exception ex) when (ex is JsonException or KeyNotFoundException or InvalidOperationException or FormatException)
        {
            table.Clear();
        }
        return table;
    }
}
