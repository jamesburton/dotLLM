using System.Text.Json;

namespace DotLLM.Models.SafeTensors;

/// <summary>
/// Parsed contents of a HuggingFace <c>model.safetensors.index.json</c>
/// sidecar, which maps every tensor name to the shard file that contains
/// it. Emitted by HF <c>save_pretrained</c> when a model is sharded
/// (checkpoint size &gt; <c>max_shard_size</c>, default 5 GiB).
/// </summary>
/// <remarks>
/// <para>
/// Schema (abbreviated):
/// </para>
/// <code>
/// {
///   "metadata": { "total_size": 16060522496 },
///   "weight_map": {
///     "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
///     "lm_head.weight":            "model-00004-of-00004.safetensors"
///   }
/// }
/// </code>
/// <para>
/// Only <c>weight_map</c> is strictly required. <c>metadata.total_size</c>
/// is surfaced as a nullable <see cref="long"/> so callers can
/// sanity-check downloads, but its absence is tolerated.
/// </para>
/// </remarks>
/// <param name="WeightMap">
/// Tensor name → shard filename (filename only; the containing directory
/// is the directory of the index.json file itself).
/// </param>
/// <param name="TotalSize">
/// Sum of all tensor byte counts as reported by HF's
/// <c>metadata.total_size</c>, or <c>null</c> when the field is absent.
/// </param>
public sealed record SafetensorsIndex(
    IReadOnlyDictionary<string, string> WeightMap,
    long? TotalSize)
{
    /// <summary>
    /// Parses an index JSON payload (already loaded into memory) into a
    /// <see cref="SafetensorsIndex"/>. Throws <see cref="InvalidDataException"/>
    /// on schema violations — the <c>weight_map</c> field is required and
    /// must be a non-empty object of string values.
    /// </summary>
    public static SafetensorsIndex Parse(string jsonContent)
    {
        ArgumentNullException.ThrowIfNull(jsonContent);

        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(jsonContent);
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException(
                $"model.safetensors.index.json is not valid JSON: {ex.Message}", ex);
        }

        using (doc)
        {
            if (doc.RootElement.ValueKind != JsonValueKind.Object)
                throw new InvalidDataException(
                    "model.safetensors.index.json root must be a JSON object.");

            if (!doc.RootElement.TryGetProperty("weight_map", out var wmEl)
                || wmEl.ValueKind != JsonValueKind.Object)
                throw new InvalidDataException(
                    "model.safetensors.index.json must declare an object 'weight_map'.");

            var weightMap = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var prop in wmEl.EnumerateObject())
            {
                if (prop.Value.ValueKind != JsonValueKind.String)
                    throw new InvalidDataException(
                        $"weight_map entry for '{prop.Name}' must be a string shard filename.");
                string shard = prop.Value.GetString()
                               ?? throw new InvalidDataException(
                                   $"weight_map entry for '{prop.Name}' is null.");
                weightMap[prop.Name] = shard;
            }

            if (weightMap.Count == 0)
                throw new InvalidDataException(
                    "model.safetensors.index.json weight_map is empty.");

            long? totalSize = null;
            if (doc.RootElement.TryGetProperty("metadata", out var metaEl)
                && metaEl.ValueKind == JsonValueKind.Object
                && metaEl.TryGetProperty("total_size", out var tsEl)
                && tsEl.ValueKind == JsonValueKind.Number
                && tsEl.TryGetInt64(out long ts))
            {
                totalSize = ts;
            }

            return new SafetensorsIndex(weightMap, totalSize);
        }
    }

    /// <summary>
    /// Reads and parses an index JSON file from disk.
    /// </summary>
    /// <exception cref="FileNotFoundException"><paramref name="indexFilePath"/> does not exist.</exception>
    /// <exception cref="InvalidDataException">The file is not valid index JSON.</exception>
    public static SafetensorsIndex Load(string indexFilePath)
    {
        ArgumentNullException.ThrowIfNull(indexFilePath);
        if (!File.Exists(indexFilePath))
            throw new FileNotFoundException(
                $"Safetensors index file not found: {indexFilePath}", indexFilePath);
        return Parse(File.ReadAllText(indexFilePath));
    }

    /// <summary>
    /// Returns the distinct shard filenames declared in the weight map,
    /// preserving first-seen order. Useful when opening every shard
    /// exactly once.
    /// </summary>
    public IReadOnlyList<string> DistinctShardFileNames()
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var ordered = new List<string>();
        foreach (var shard in WeightMap.Values)
        {
            if (seen.Add(shard))
                ordered.Add(shard);
        }
        return ordered;
    }
}
