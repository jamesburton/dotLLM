using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Loaders;

/// <summary>
/// Verifies the I2_S disk cache wired into the BitNet HF safetensors loader: a cold load
/// quantizes every linear projection and stores it; a subsequent warm load with the same cache
/// key serves every projection from disk — skipping re-quantization entirely — so repeated loads
/// of the same checkpoint avoid the dominant online-quantization cost.
/// </summary>
public sealed class BitNetI2SCacheLoadTests : IDisposable
{
    private const int Hidden = 128;
    private const int NumHeads = 4;
    private const int NumKvHeads = 2;
    private const int HeadDim = 32;
    private const int Intermediate = 256;
    private const int Vocab = 64;
    private const int NumLayers = 2;

    // q,k,v,o,gate,up,down per layer = 7 I2_S-quantized projections.
    private const int I2SProjectionsPerLayer = 7;
    private const int ExpectedI2SProjections = I2SProjectionsPerLayer * NumLayers;

    private readonly string _scratch =
        Path.Combine(Path.GetTempPath(), $"dotllm-bitnet-i2scache-{Guid.NewGuid():N}");

    public BitNetI2SCacheLoadTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void ColdLoadStoresEveryProjection_WarmLoadServesAllFromCache()
    {
        WriteSafetensors();
        WriteConfigJson();
        string cacheDir = Path.Combine(_scratch, "i2s-cache");

        var (source, config) = ModelLoader.OpenSafetensorsAndConfig(_scratch);
        try
        {
            // Cold load: no cache entries exist yet → every projection is a miss + store.
            var cold = new BitNetI2SCacheContext(cacheDir, "test-model-key");
            var w1 = TransformerWeightsSafetensorsLoader.Load(source, config, cold);
            w1.Dispose();

            Assert.Equal(ExpectedI2SProjections, cold.Misses);
            Assert.Equal(ExpectedI2SProjections, cold.Stores);
            Assert.Equal(0, cold.Hits);

            // Warm load: identical key → every projection is served from disk, no misses.
            var warm = new BitNetI2SCacheContext(cacheDir, "test-model-key");
            var w2 = TransformerWeightsSafetensorsLoader.Load(source, config, warm);
            w2.Dispose();

            Assert.Equal(ExpectedI2SProjections, warm.Hits);
            Assert.Equal(0, warm.Misses);
            Assert.Equal(0, warm.Stores);
        }
        finally
        {
            source.Dispose();
        }
    }

    [Fact]
    public void ModelLoader_PopulatesI2SCacheBesideCheckpoint_OnFirstLoad()
    {
        WriteSafetensors();
        WriteConfigJson();
        string cacheDir = Path.Combine(_scratch, ".dotllm-i2s-cache");
        Assert.False(Directory.Exists(cacheDir));

        var (model, source, config) = ModelLoader.LoadFromSafetensors(_scratch);
        model.Dispose();
        source.Dispose();

        Assert.True(Directory.Exists(cacheDir), "expected an I2_S cache directory beside the checkpoint");
        int entries = Directory.GetFiles(cacheDir, "*.i2s", SearchOption.AllDirectories).Length;
        Assert.Equal(ExpectedI2SProjections, entries);
    }

    private void WriteConfigJson()
    {
        string json = $$"""
        {
          "architectures": ["BitNetForCausalLM"],
          "model_type": "bitnet",
          "hidden_act": "relu2",
          "hidden_size": {{Hidden}},
          "intermediate_size": {{Intermediate}},
          "num_hidden_layers": {{NumLayers}},
          "num_attention_heads": {{NumHeads}},
          "num_key_value_heads": {{NumKvHeads}},
          "head_dim": {{HeadDim}},
          "vocab_size": {{Vocab}},
          "max_position_embeddings": 128,
          "rms_norm_eps": 1e-5,
          "rope_theta": 500000.0,
          "tie_word_embeddings": true
        }
        """;
        File.WriteAllText(Path.Combine(_scratch, "config.json"), json);
    }

    private void WriteSafetensors()
    {
        var rng = new Random(126);
        var tensors = new List<(string Name, string DType, int[] Shape, byte[] Bytes)>
        {
            ("model.embed_tokens.weight", "F32", [Vocab, Hidden], F32(rng, Vocab * Hidden, 0.05f)),
            ("model.norm.weight", "F32", [Hidden], Ones(Hidden)),
        };

        for (int i = 0; i < NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            tensors.Add(($"{p}.input_layernorm.weight", "F32", [Hidden], Ones(Hidden)));
            tensors.Add(($"{p}.post_attention_layernorm.weight", "F32", [Hidden], Ones(Hidden)));

            tensors.Add(($"{p}.self_attn.q_proj.weight", "BF16", [NumHeads * HeadDim, Hidden], Bf16(rng, NumHeads * HeadDim * Hidden)));
            tensors.Add(($"{p}.self_attn.k_proj.weight", "BF16", [NumKvHeads * HeadDim, Hidden], Bf16(rng, NumKvHeads * HeadDim * Hidden)));
            tensors.Add(($"{p}.self_attn.v_proj.weight", "BF16", [NumKvHeads * HeadDim, Hidden], Bf16(rng, NumKvHeads * HeadDim * Hidden)));
            tensors.Add(($"{p}.self_attn.o_proj.weight", "BF16", [Hidden, NumHeads * HeadDim], Bf16(rng, Hidden * NumHeads * HeadDim)));
            tensors.Add(($"{p}.self_attn.attn_sub_norm.weight", "F32", [Hidden], Ones(Hidden)));

            tensors.Add(($"{p}.mlp.gate_proj.weight", "BF16", [Intermediate, Hidden], Bf16(rng, Intermediate * Hidden)));
            tensors.Add(($"{p}.mlp.up_proj.weight", "BF16", [Intermediate, Hidden], Bf16(rng, Intermediate * Hidden)));
            tensors.Add(($"{p}.mlp.down_proj.weight", "BF16", [Hidden, Intermediate], Bf16(rng, Hidden * Intermediate)));
            tensors.Add(($"{p}.mlp.ffn_sub_norm.weight", "F32", [Intermediate], Ones(Intermediate)));
        }

        using var ms = new MemoryStream();
        using (var w = new Utf8JsonWriter(ms, new JsonWriterOptions { Indented = false }))
        {
            w.WriteStartObject();
            long offset = 0;
            foreach (var (name, dtype, shape, bytes) in tensors)
            {
                w.WriteStartObject(name);
                w.WriteString("dtype", dtype);
                w.WritePropertyName("shape");
                w.WriteStartArray();
                foreach (var d in shape) w.WriteNumberValue(d);
                w.WriteEndArray();
                w.WritePropertyName("data_offsets");
                w.WriteStartArray();
                w.WriteNumberValue(offset);
                w.WriteNumberValue(offset + bytes.Length);
                w.WriteEndArray();
                w.WriteEndObject();
                offset += bytes.Length;
            }
            w.WriteEndObject();
        }

        byte[] headerJson = ms.ToArray();
        using var fs = new FileStream(
            Path.Combine(_scratch, "model.safetensors"), FileMode.Create, FileAccess.Write, FileShare.None);
        Span<byte> prefix = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(prefix, (ulong)headerJson.Length);
        fs.Write(prefix);
        fs.Write(headerJson);
        foreach (var (_, _, _, bytes) in tensors)
            fs.Write(bytes);
    }

    private static byte[] F32(Random rng, int n, float scale)
    {
        var bytes = new byte[(long)n * sizeof(float)];
        for (int i = 0; i < n; i++)
        {
            float v = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4, 4), v);
        }
        return bytes;
    }

    private static byte[] Ones(int n)
    {
        var bytes = new byte[(long)n * sizeof(float)];
        for (int i = 0; i < n; i++)
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4, 4), 1.0f);
        return bytes;
    }

    private static byte[] Bf16(Random rng, int n)
    {
        var bytes = new byte[(long)n * 2];
        for (int i = 0; i < n; i++)
        {
            float v = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.05);
            uint bits = Unsafe.As<float, uint>(ref v);
            ushort bf16 = (ushort)(bits >> 16);
            bytes[i * 2] = (byte)(bf16 & 0xFF);
            bytes[i * 2 + 1] = (byte)(bf16 >> 8);
        }
        return bytes;
    }
}
