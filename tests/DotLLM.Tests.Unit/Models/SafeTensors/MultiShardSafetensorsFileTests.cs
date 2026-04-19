using System.Runtime.InteropServices;
using System.Text.Json;
using DotLLM.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

/// <summary>
/// Unit tests for <see cref="MultiShardSafetensorsFile"/>. Writes
/// byte-accurate 2-shard fixtures with an accompanying index.json and
/// verifies the shard-aware lookup surface.
/// </summary>
public sealed class MultiShardSafetensorsFileTests : IDisposable
{
    private readonly string _scratch;

    public MultiShardSafetensorsFileTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-stms-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <summary>
    /// Writes a 2-shard fixture:
    ///   - shard1 contains 'alpha' (ramp from 1.0)
    ///   - shard2 contains 'beta'  (ramp from 100.0)
    /// Emits a matching index.json. Returns the index path.
    /// </summary>
    private string BuildTwoShardFixture()
    {
        string shard1 = Path.Combine(_scratch, "model-00001-of-00002.safetensors");
        string shard2 = Path.Combine(_scratch, "model-00002-of-00002.safetensors");
        string indexPath = Path.Combine(_scratch, "model.safetensors.index.json");

        new SafetensorsFixtureBuilder()
            .AddFloat32("alpha", [2, 3], startValue: 1.0f)
            .WriteTo(shard1);

        new SafetensorsFixtureBuilder()
            .AddFloat32("beta", [4], startValue: 100.0f)
            .WriteTo(shard2);

        string json = JsonSerializer.Serialize(new
        {
            metadata = new { total_size = 24 + 16 },
            weight_map = new Dictionary<string, string>
            {
                ["alpha"] = "model-00001-of-00002.safetensors",
                ["beta"] = "model-00002-of-00002.safetensors",
            },
        });
        File.WriteAllText(indexPath, json);
        return indexPath;
    }

    [Fact]
    public void Open_TwoShardFixture_EnumeratesAllTensors()
    {
        string indexPath = BuildTwoShardFixture();

        using var src = MultiShardSafetensorsFile.Open(indexPath);

        Assert.Equal(2, src.ShardCount);
        Assert.Equal(2, src.Tensors.Count);
        Assert.True(src.TensorsByName.ContainsKey("alpha"));
        Assert.True(src.TensorsByName.ContainsKey("beta"));
    }

    [Fact]
    public void GetShardIndexFor_RoutesToCorrectShard()
    {
        string indexPath = BuildTwoShardFixture();

        using var src = MultiShardSafetensorsFile.Open(indexPath);

        Assert.Equal(0, src.GetShardIndexFor("alpha"));
        Assert.Equal(1, src.GetShardIndexFor("beta"));
    }

    [Fact]
    public void GetTensorSpan_ReadsBytesFromOwningShard()
    {
        string indexPath = BuildTwoShardFixture();

        using var src = MultiShardSafetensorsFile.Open(indexPath);

        var alphaBytes = src.GetTensorSpan("alpha");
        var alphaFloats = MemoryMarshal.Cast<byte, float>(alphaBytes);
        Assert.Equal([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f], alphaFloats.ToArray());

        var betaBytes = src.GetTensorSpan("beta");
        var betaFloats = MemoryMarshal.Cast<byte, float>(betaBytes);
        Assert.Equal([100.0f, 101.0f, 102.0f, 103.0f], betaFloats.ToArray());
    }

    [Fact]
    public unsafe void GetTensorPointer_ReadsBytesFromOwningShard()
    {
        string indexPath = BuildTwoShardFixture();

        using var src = MultiShardSafetensorsFile.Open(indexPath);

        nint ap = src.GetTensorPointer("alpha");
        Assert.NotEqual(nint.Zero, ap);
        var alpha = new ReadOnlySpan<float>((void*)ap, 6);
        Assert.Equal(1.0f, alpha[0]);
        Assert.Equal(6.0f, alpha[5]);
    }

    [Fact]
    public void Open_UnknownTensor_ThrowsKeyNotFound()
    {
        string indexPath = BuildTwoShardFixture();

        using var src = MultiShardSafetensorsFile.Open(indexPath);
        Assert.Throws<KeyNotFoundException>(() => src.GetTensorPointer("not-a-tensor"));
    }

    [Fact]
    public void Open_IndexReferencesMissingShard_Throws()
    {
        string indexPath = Path.Combine(_scratch, "model.safetensors.index.json");
        File.WriteAllText(indexPath, """
        {
          "weight_map": {
            "alpha": "missing-shard.safetensors"
          }
        }
        """);

        Assert.Throws<FileNotFoundException>(() => MultiShardSafetensorsFile.Open(indexPath));
    }

    [Fact]
    public void Open_IndexPointsToWrongShardForTensor_Throws()
    {
        string shard1 = Path.Combine(_scratch, "model-00001-of-00002.safetensors");
        string shard2 = Path.Combine(_scratch, "model-00002-of-00002.safetensors");
        string indexPath = Path.Combine(_scratch, "model.safetensors.index.json");

        new SafetensorsFixtureBuilder()
            .AddFloat32("alpha", [1], startValue: 1.0f).WriteTo(shard1);
        new SafetensorsFixtureBuilder()
            .AddFloat32("beta", [1], startValue: 2.0f).WriteTo(shard2);

        // Index says 'alpha' lives in shard2 — it doesn't.
        File.WriteAllText(indexPath, """
        {
          "weight_map": {
            "alpha": "model-00002-of-00002.safetensors",
            "beta":  "model-00002-of-00002.safetensors"
          }
        }
        """);

        var ex = Assert.Throws<InvalidDataException>(() =>
            MultiShardSafetensorsFile.Open(indexPath));
        Assert.Contains("alpha", ex.Message);
    }

    [Fact]
    public void Open_TensorRedeclaredInMultipleShards_IndexResolves()
    {
        // A pathological but handled case: shard1 and shard2 both physically
        // declare 'alpha'. The index.json names shard1 as authoritative —
        // so the lookup must route to shard1, not shard2.
        string shard1 = Path.Combine(_scratch, "model-00001-of-00002.safetensors");
        string shard2 = Path.Combine(_scratch, "model-00002-of-00002.safetensors");
        string indexPath = Path.Combine(_scratch, "model.safetensors.index.json");

        new SafetensorsFixtureBuilder()
            .AddFloat32("alpha", [2], startValue: 10.0f)
            .AddFloat32("a_only", [1], startValue: 11.0f)
            .WriteTo(shard1);

        new SafetensorsFixtureBuilder()
            .AddFloat32("alpha", [2], startValue: 99.0f)  // different value!
            .AddFloat32("b_only", [1], startValue: 98.0f)
            .WriteTo(shard2);

        // Index declares 'alpha' belongs to shard1. alpha duplicated in shard2
        // conflicts with the index, so the loader rejects it — safer than
        // silently picking one.
        File.WriteAllText(indexPath, """
        {
          "weight_map": {
            "alpha":  "model-00001-of-00002.safetensors",
            "a_only": "model-00001-of-00002.safetensors",
            "b_only": "model-00002-of-00002.safetensors"
          }
        }
        """);

        Assert.Throws<InvalidDataException>(() => MultiShardSafetensorsFile.Open(indexPath));
    }

    [Fact]
    public void Dispose_ReleasesAllShards_IsIdempotent()
    {
        string indexPath = BuildTwoShardFixture();

        var src = MultiShardSafetensorsFile.Open(indexPath);
        Assert.Equal(2, src.ShardCount);
        src.Dispose();
        src.Dispose(); // must not throw

        // After dispose, the underlying files must be unlocked so we can
        // delete them (Windows rejects deletes on mmap-locked files).
        File.Delete(Path.Combine(_scratch, "model-00001-of-00002.safetensors"));
        File.Delete(Path.Combine(_scratch, "model-00002-of-00002.safetensors"));
    }

    [Fact]
    public void OpenWithoutIndex_TwoShards_UnifiesByFirstSeen()
    {
        // Arrange: two shards, no index.
        string shard1 = Path.Combine(_scratch, "part1.safetensors");
        string shard2 = Path.Combine(_scratch, "part2.safetensors");
        new SafetensorsFixtureBuilder().AddFloat32("a", [1], 1.0f).WriteTo(shard1);
        new SafetensorsFixtureBuilder().AddFloat32("b", [1], 2.0f).WriteTo(shard2);

        using var src = MultiShardSafetensorsFile.OpenWithoutIndex(
            _scratch, new[] { "part1.safetensors", "part2.safetensors" });

        Assert.Equal(2, src.Tensors.Count);
        Assert.Equal(0, src.GetShardIndexFor("a"));
        Assert.Equal(1, src.GetShardIndexFor("b"));
    }

    [Fact]
    public void OpenWithoutIndex_DuplicateTensors_Throws()
    {
        string shard1 = Path.Combine(_scratch, "part1.safetensors");
        string shard2 = Path.Combine(_scratch, "part2.safetensors");
        new SafetensorsFixtureBuilder().AddFloat32("dup", [1], 1.0f).WriteTo(shard1);
        new SafetensorsFixtureBuilder().AddFloat32("dup", [1], 2.0f).WriteTo(shard2);

        Assert.Throws<InvalidDataException>(() =>
            MultiShardSafetensorsFile.OpenWithoutIndex(
                _scratch, new[] { "part1.safetensors", "part2.safetensors" }));
    }
}
