using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Engine.KvCache;
using DotLLM.Hip;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Hip;

public sealed partial class HipTransformerModelTests
{
    [SkippableTheory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    [Trait("Category", "GPU")]
    public void SafeTensorsDirectLoading_MatchesCpuAcrossShardsAndTiedHeads(bool sharded, bool tied)
    {
        RequireGpu();
        Fixture fixture = CreateFixture(tied);
        string directory = Path.Combine(Path.GetTempPath(), $"dotllm-hip-hf-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            string path = WriteHfFixture(fixture, directory, sharded);
            HipTransformerModel model;
            using (var source = Open(path)) model = HipTransformerModel.LoadFromSafetensors(source, fixture.Config);
            using (model)
            using (var source = Open(path))
            using (var cpu = TransformerModel.LoadFromSafetensors(source, fixture.Config with { MaxSequenceLength = 128 },
                new ThreadingConfig(1, EnableCallerPinning: false)))
            using (var actualCache = model.CreateKvCache(128, 4))
            using (var expectedCache = new SimpleKvCache(Layers, KvHeads, HeadDim, 128))
            {
                Assert.Equal(fixture.StoredBytes, model.StoredWeightBytes);
                int[] tokens = Tokens(17, 3);
                using (var expected = cpu.Forward(tokens, Positions(0, tokens.Length), -1, expectedCache))
                using (var actual = model.Forward(tokens, Positions(0, tokens.Length), 0, actualCache, true))
                    Close(ReadLast(expected), ReadLast(actual), $"HF sharded={sharded}, tied={tied}, prefill");
                using var expectedDecode = cpu.Forward([47], [17], -1, expectedCache);
                using var actualDecode = model.Forward([47], [17], 0, actualCache);
                Close(ReadLast(expectedDecode), ReadLast(actualDecode), $"HF sharded={sharded}, tied={tied}, decode");
                Assert.Equal(18, actualCache.CurrentLength);
            }
        }
        finally { Directory.Delete(directory, recursive: true); }

        static ISafetensorsTensorSource Open(string path) => path.EndsWith(".json", StringComparison.Ordinal)
            ? MultiShardSafetensorsFile.Open(path) : SafetensorsFile.Open(path);
    }

    [Fact]
    public void SafeTensorsBf16IsRejectedBeforeDeviceInitialization()
    {
        Fixture fixture = CreateFixture(false);
        string directory = Path.Combine(Path.GetTempPath(), $"dotllm-hip-hf-reject-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            using var source = SafetensorsFile.Open(WriteHfFixture(fixture, directory, false, bf16Embedding: true));
            var error = Assert.Throws<NotSupportedException>(() =>
            {
                using var rejected = HipTransformerModel.LoadFromSafetensors(source, fixture.Config, deviceId: -1);
            });
            Assert.Contains("BF16", error.Message, StringComparison.Ordinal);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    private static unsafe string WriteHfFixture(Fixture fixture, string directory, bool sharded, bool bf16Embedding = false)
    {
        using var gguf = GgufFile.Open(fixture.Path);
        var builders = new[] { new SafetensorsFixtureBuilder(), new SafetensorsFixtureBuilder() };
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var tensor in gguf.TensorsByName.Values)
        {
            string name = tensor.Name switch
            {
                "token_embd.weight" => "model.embed_tokens.weight",
                "output_norm.weight" => "model.norm.weight",
                "output.weight" => "lm_head.weight",
                _ => LayerName(tensor.Name),
            };
            int[] shape = tensor.Shape.Dimensions.Reverse().ToArray();
            int bytes = checked(shape.Aggregate(1, (a, b) => a * b) * (tensor.QuantizationType == QuantizationType.F32 ? 4 : 2));
            string dtype = tensor.QuantizationType == QuantizationType.F32 ? "F32" : "F16";
            if (bf16Embedding && string.Equals(tensor.Name, "token_embd.weight", StringComparison.Ordinal)) dtype = "BF16";
            int shard = sharded ? map.Count % 2 : 0;
            builders[shard].AddRaw(name, dtype, shape, new ReadOnlySpan<byte>((void*)gguf.TensorDataPointer(tensor.Name), bytes).ToArray());
            map.Add(name, shard == 0 ? "a.safetensors" : "b.safetensors");
        }
        string aPath = Path.Combine(directory, "a.safetensors");
        builders[0].WriteTo(aPath);
        if (!sharded) return aPath;
        builders[1].WriteTo(Path.Combine(directory, "b.safetensors"));
        string index = Path.Combine(directory, "model.safetensors.index.json");
        File.WriteAllText(index, JsonSerializer.Serialize(new { weight_map = map }));
        return index;

        static string LayerName(string name)
        {
            string[] parts = name.Split('.');
            string suffix = parts[2] switch
            {
                "attn_norm" => "input_layernorm",
                "ffn_norm" => "post_attention_layernorm",
                "attn_q" => "self_attn.q_proj",
                "attn_k" => "self_attn.k_proj",
                "attn_v" => "self_attn.v_proj",
                "attn_output" => "self_attn.o_proj",
                "ffn_gate" => "mlp.gate_proj",
                "ffn_up" => "mlp.up_proj",
                "ffn_down" => "mlp.down_proj",
                _ => throw new InvalidOperationException(name),
            };
            return $"model.layers.{parts[1]}.{suffix}.weight";
        }
    }
}
