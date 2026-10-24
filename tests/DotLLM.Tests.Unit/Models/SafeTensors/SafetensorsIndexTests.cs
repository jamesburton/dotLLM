using DotLLM.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

/// <summary>
/// Unit tests for <see cref="SafetensorsIndex"/>. Exercises the JSON
/// schema we need to consume from HuggingFace
/// <c>model.safetensors.index.json</c> sidecars.
/// </summary>
public sealed class SafetensorsIndexTests : IDisposable
{
    private readonly string _scratch;

    public SafetensorsIndexTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-stix-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void Parse_CanonicalIndex_PopulatesWeightMapAndTotalSize()
    {
        const string json = """
        {
          "metadata": { "total_size": 16060522496 },
          "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
            "lm_head.weight": "model-00004-of-00004.safetensors"
          }
        }
        """;

        var index = SafetensorsIndex.Parse(json);

        Assert.Equal(16060522496L, index.TotalSize);
        Assert.Equal(3, index.WeightMap.Count);
        Assert.Equal("model-00001-of-00004.safetensors", index.WeightMap["model.embed_tokens.weight"]);
        Assert.Equal("model-00004-of-00004.safetensors", index.WeightMap["lm_head.weight"]);
    }

    [Fact]
    public void Parse_MissingTotalSize_ReturnsNull()
    {
        const string json = """
        { "weight_map": { "a": "s1.safetensors" } }
        """;

        var index = SafetensorsIndex.Parse(json);

        Assert.Null(index.TotalSize);
        Assert.Single(index.WeightMap);
    }

    [Fact]
    public void Parse_MetadataPresentButTotalSizeMissing_ReturnsNull()
    {
        const string json = """
        {
          "metadata": { "format": "pt" },
          "weight_map": { "a": "s1.safetensors" }
        }
        """;

        var index = SafetensorsIndex.Parse(json);

        Assert.Null(index.TotalSize);
    }

    [Fact]
    public void Parse_MalformedJson_Throws()
    {
        Assert.Throws<InvalidDataException>(() =>
            SafetensorsIndex.Parse("{not-json"));
    }

    [Fact]
    public void Parse_MissingWeightMap_Throws()
    {
        const string json = """{ "metadata": { "total_size": 1 } }""";
        Assert.Throws<InvalidDataException>(() => SafetensorsIndex.Parse(json));
    }

    [Fact]
    public void Parse_EmptyWeightMap_Throws()
    {
        const string json = """{ "weight_map": {} }""";
        Assert.Throws<InvalidDataException>(() => SafetensorsIndex.Parse(json));
    }

    [Fact]
    public void Parse_WeightMapEntryNotString_Throws()
    {
        const string json = """{ "weight_map": { "a": 42 } }""";
        Assert.Throws<InvalidDataException>(() => SafetensorsIndex.Parse(json));
    }

    [Fact]
    public void Load_ReadsFileFromDisk()
    {
        string path = Path.Combine(_scratch, "model.safetensors.index.json");
        File.WriteAllText(path, """
        {
          "metadata": { "total_size": 42 },
          "weight_map": {
            "x": "s1.safetensors",
            "y": "s2.safetensors"
          }
        }
        """);

        var index = SafetensorsIndex.Load(path);

        Assert.Equal(42L, index.TotalSize);
        Assert.Equal(2, index.WeightMap.Count);
    }

    [Fact]
    public void Load_MissingFile_Throws()
    {
        Assert.Throws<FileNotFoundException>(() =>
            SafetensorsIndex.Load(Path.Combine(_scratch, "nope.json")));
    }

    [Fact]
    public void DistinctShardFileNames_PreservesFirstSeenOrder_Deduplicates()
    {
        const string json = """
        {
          "weight_map": {
            "a": "s2.safetensors",
            "b": "s1.safetensors",
            "c": "s2.safetensors",
            "d": "s1.safetensors",
            "e": "s3.safetensors"
          }
        }
        """;

        var index = SafetensorsIndex.Parse(json);
        var shards = index.DistinctShardFileNames();

        // Order reflects first-seen from Dictionary<string, string> iteration.
        // Dictionary preserves insertion order in .NET, so this is deterministic.
        Assert.Equal(3, shards.Count);
        Assert.Equal("s2.safetensors", shards[0]);
        Assert.Equal("s1.safetensors", shards[1]);
        Assert.Equal("s3.safetensors", shards[2]);
    }
}
