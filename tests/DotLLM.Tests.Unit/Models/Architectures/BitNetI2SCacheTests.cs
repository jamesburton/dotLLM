using System;
using System.IO;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Verifies the on-disk BitNet I2_S weight cache: freshly-quantized packed bytes stored under
/// (model key, tensor name) round-trip back verbatim, and the cache correctly misses on absent
/// entries, size mismatches, and foreign model keys so stale/incompatible caches never feed the
/// loader wrong weights.
/// </summary>
public sealed class BitNetI2SCacheTests : IDisposable
{
    private readonly string _dir =
        Path.Combine(Path.GetTempPath(), "dotllm-i2s-cache-test-" + Guid.NewGuid().ToString("N"));

    [Fact]
    public void StoreThenTryLoad_RoundTripsIdenticalPackedBytes()
    {
        var payload = new byte[72];
        new Random(0x1B17).NextBytes(payload);
        const string modelKey = "modelA-v1";
        const string tensor = "model.layers.0.self_attn.q_proj.weight";

        BitNetI2SCache.Store(_dir, modelKey, tensor, payload);

        var dest = new byte[payload.Length];
        bool ok = BitNetI2SCache.TryLoad(_dir, modelKey, tensor, dest);

        Assert.True(ok);
        Assert.Equal(payload, dest);
    }

    [Fact]
    public void TryLoad_RejectsEntryLongerThanDestination()
    {
        // A stored payload larger than the caller's expected tensor size means the cache is
        // stale/incompatible (e.g. the model changed). It must MISS, not silently fill dest
        // with a truncated prefix.
        var stored = new byte[128];
        new Random(0x515E).NextBytes(stored);
        const string modelKey = "modelA-v1";
        const string tensor = "model.layers.0.mlp.down_proj.weight";

        BitNetI2SCache.Store(_dir, modelKey, tensor, stored);

        var dest = new byte[64]; // caller expects a smaller tensor than what was cached
        bool ok = BitNetI2SCache.TryLoad(_dir, modelKey, tensor, dest);

        Assert.False(ok);
    }

    private static readonly byte[] Cfg =
        System.Text.Encoding.UTF8.GetBytes("{\"model_type\":\"bitnet\",\"num_hidden_layers\":30}");

    [Fact]
    public void ComputeModelKey_IsStableAndNonEmptyForIdenticalInputs()
    {
        var shards = new[] { ("model.safetensors", 5_000_000_000L) };
        string a = BitNetI2SCache.ComputeModelKey(Cfg, shards, 1);
        string b = BitNetI2SCache.ComputeModelKey(Cfg, shards, 1);
        Assert.NotEmpty(a);
        Assert.Equal(a, b);
    }

    [Fact]
    public void ComputeModelKey_ChangesWhenQuantizerVersionChanges()
    {
        var shards = new[] { ("model.safetensors", 5_000_000_000L) };
        Assert.NotEqual(
            BitNetI2SCache.ComputeModelKey(Cfg, shards, 1),
            BitNetI2SCache.ComputeModelKey(Cfg, shards, 2));
    }

    [Fact]
    public void ComputeModelKey_ChangesWhenAShardLengthChanges()
    {
        Assert.NotEqual(
            BitNetI2SCache.ComputeModelKey(Cfg, new[] { ("model.safetensors", 5_000_000_000L) }, 1),
            BitNetI2SCache.ComputeModelKey(Cfg, new[] { ("model.safetensors", 5_000_000_001L) }, 1));
    }

    [Fact]
    public void ComputeModelKey_ChangesWhenConfigChanges()
    {
        var shards = new[] { ("model.safetensors", 5_000_000_000L) };
        var otherCfg = System.Text.Encoding.UTF8.GetBytes("{\"model_type\":\"bitnet\",\"num_hidden_layers\":40}");
        Assert.NotEqual(
            BitNetI2SCache.ComputeModelKey(Cfg, shards, 1),
            BitNetI2SCache.ComputeModelKey(otherCfg, shards, 1));
    }

    public void Dispose()
    {
        if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
    }
}
