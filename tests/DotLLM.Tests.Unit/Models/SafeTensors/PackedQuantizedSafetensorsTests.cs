using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

public sealed unsafe class PackedQuantizedSafetensorsTests : IDisposable
{
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-packed-{Guid.NewGuid():N}");
    private const string Projection = "model.layers.0.mlp.gate_proj";

    [Theory]
    [InlineData(true, false, false)]
    [InlineData(true, true, true)]
    [InlineData(false, false, true)]
    [InlineData(false, true, false)]
    public void Storage_BorrowsAllComponentsAndOwnsOnlyDescriptors(bool awq, bool sharded, bool tied)
    {
        var fixture = new Fixture(awq, tied: tied);
        string path = fixture.Write(Path.Combine(_scratch, "storage"), sharded);
        var (source, config) = ModelLoader.OpenSafetensorsAndConfig(path);
        using (source)
        {
            var packed = Assert.IsType<PackedQuantizedSafetensorsSource>(source);
            Assert.Equal(64, sizeof(Packed4Weight));
            Assert.Equal(14, packed.ProjectionCount);
            Assert.Equal(fixture.Dense.Values.Sum(x => (long)x.Length), packed.PackedParameterCount);
            Assert.Equal(packed.PackedParameterCount + (tied ? 40 * 64 : 2 * 40 * 64) + 5 * 64, packed.LogicalParameterCount);
            Assert.Equal(fixture.Tensors.Where(x => x.Key.EndsWith("qweight", StringComparison.Ordinal)
                || x.Key.EndsWith("qzeros", StringComparison.Ordinal) || x.Key.EndsWith("scales", StringComparison.Ordinal)
                || x.Key.EndsWith("g_idx", StringComparison.Ordinal)).Sum(x => (long)x.Value.Bytes.Length), packed.PackedBytes);
            using (var weights = TransformerWeightsSafetensorsLoader.Load(source, config, allowPackedQuantization: true))
            {
                Assert.Equal(14, weights.LiveOwnedAllocationCount);
                var w = (Packed4Weight*)weights.Layers[0].GateWeight;
                Assert.Equal(96, w->OutputFeatures);
                Assert.Equal(64, w->InputFeatures);
                Assert.Equal(awq ? QuantizationType.Awq4 : QuantizationType.Gptq4, weights.Layers[0].GateQuantType);
                Assert.Equal(source.GetTensorPointer(Projection + ".qweight"), (nint)w->QWeight);
                Assert.Equal(source.GetTensorPointer(Projection + ".qzeros"), (nint)w->QZeros);
                Assert.Equal(source.GetTensorPointer(Projection + ".scales"), (nint)w->Scales);
                if (!awq) Assert.Equal(source.GetTensorPointer(Projection + ".g_idx"), (nint)w->GroupIndices);
                Assert.False(WeightRepacking.IsRepackable(w->Layout));
                Assert.False(MatMul.SupportsFusedDecode(w->Layout));
                Assert.Equal(QuantizationType.F16, weights.TokenEmbedQuantType);
                if (tied) Assert.Equal(weights.TokenEmbedWeight, weights.OutputWeight);
#pragma warning disable IDISP016 // Deliberately verify disposal bookkeeping and idempotence.
                weights.Dispose();
                Assert.Equal(0, weights.LiveOwnedAllocationCount);
#pragma warning restore IDISP016
            }
            // Weight handles own metadata only. Their disposal must not unmap a shard.
            Assert.NotEqual(0, source.GetTensorPointer(Projection + ".qweight"));
            using var reloaded = TransformerModel.LoadFromSafetensors(source, config,
                new ThreadingConfig(1, EnableCallerPinning: false));
            using var result = reloaded.Forward([1], [0], -1);
            Assert.Equal(40, result.Shape[1]);
        }
        Assert.Throws<ObjectDisposedException>(() => source.GetTensorPointer(Projection + ".qweight"));
        Assert.Throws<ObjectDisposedException>(() => source.GetTensorFileLocation(Projection + ".qweight"));
        source.Dispose();
    }

    [Theory]
    [InlineData(true, false, false, 1, false)]
    [InlineData(true, true, true, 2, false)]
    [InlineData(false, false, true, 2, true)]
    [InlineData(false, true, false, 1, true)]
    [InlineData(false, true, false, 2, false)]
    public void PrefillAndCachedDecode_MatchIndependentDenseWeights(bool awq, bool sharded, bool tied, int threads, bool actOrder)
    {
        var fixture = new Fixture(awq, tied: tied, actOrder: actOrder);
        string packedPath = fixture.Write(Path.Combine(_scratch, "packed"), sharded);
        string referencePath = fixture.Write(Path.Combine(_scratch, "dense"), false, dense: true);
        var threading = new ThreadingConfig(threads, EnableCallerPinning: false);
        var (model, source, config) = ModelLoader.LoadFromSafetensors(packedPath, threading);
        var (reference, referenceSource, _) = ModelLoader.LoadFromSafetensors(referencePath, threading);
        using (source) using (referenceSource) using (model) using (reference)
        using (var cache = new SimpleKvCache(2, 2, 16, 16))
        using (var referenceCache = new SimpleKvCache(2, 2, 16, 16))
        {
            Compare([1, 7, 13], [0, 1, 2]);
            Compare([4], [3]);
            Compare([17], [4]);
            void Compare(int[] tokens, int[] positions)
            {
                using var expected = reference.Forward(tokens, positions, -1, referenceCache);
                using var actual = model.Forward(tokens, positions, -1, cache);
                AssertClose(new ReadOnlySpan<float>((void*)expected.DataPointer, tokens.Length * config.VocabSize),
                    new ReadOnlySpan<float>((void*)actual.DataPointer, tokens.Length * config.VocabSize), 5e-5f);
            }
        }
    }

    [Theory]
    [InlineData(true, 32)]
    [InlineData(true, 64)]
    [InlineData(true, 128)]
    [InlineData(true, -1)]
    [InlineData(false, 32)]
    [InlineData(false, 64)]
    [InlineData(false, 128)]
    [InlineData(false, -1)]
    public void GoldenProjection_UsesProducerNibbleOrderZerosAndGroups(bool awq, int groupSize)
    {
        var fixture = new Fixture(awq, hidden: 128, intermediate: 256, groupSize: groupSize);
        // Explicit independent packed words: q(out,k)=(3*out+5*k+k/7)%16.
        var q = fixture.Tensors[Projection + ".qweight"].Bytes;
        Assert.Equal(awq ? 0x5f932c60u : 0x4e94fa50u, BinaryPrimitives.ReadUInt32LittleEndian(q));
        Assert.Equal(awq ? 0x86427531u : 0x76543210u,
            BinaryPrimitives.ReadUInt32LittleEndian(fixture.Tensors[Projection + ".qzeros"].Bytes));
        var (source, _) = ModelLoader.OpenSafetensorsAndConfig(fixture.Write(Path.Combine(_scratch, "golden"), true));
        using (source)
        {
            var owned = new List<nint>();
            try
            {
                var (pointer, _, m, k) = SafetensorsTensorResolver.ResolveLinear(source, Projection + ".weight", owned);
                float[] input = Enumerable.Range(0, 3 * k).Select(i => (i % 19 - 9) / 17f).ToArray();
                float[] expected = new float[3 * m], actual = new float[3 * m];
                // Double-accumulation oracle consumes logical values produced BEFORE packing.
                for (int token = 0; token < 3; token++)
                    for (int row = 0; row < m; row++)
                    {
                        double sum = 0;
                        for (int col = 0; col < k; col++) sum += (double)fixture.Dense[Projection][row * k + col] * input[token * k + col];
                        expected[token * m + row] = (float)sum;
                    }
                fixed (float* x = input, y = actual)
                {
                    Packed4MatMul.Gemm((Packed4Weight*)pointer, x, y, m, k, 3);
                    Assert.Throws<ArgumentException>(() => Packed4MatMul.Gemm((Packed4Weight*)pointer, (float*)1, (float*)1, k, m, 1));
                }
                AssertClose(expected, actual, 3e-6f);
            }
            finally { foreach (nint pointer in owned) NativeMemory.AlignedFree((void*)pointer); }
        }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void GpuAndUnvalidatedSources_AreRejectedBeforeWeightAllocation(bool awq)
    {
        var fixture = new Fixture(awq);
        string path = fixture.Write(Path.Combine(_scratch, "gate"), false);
        var (source, config) = ModelLoader.OpenSafetensorsAndConfig(path);
        using (source)
        {
            var error = Assert.Throws<NotSupportedException>(() => { using var rejected = TransformerWeightsSafetensorsLoader.Load(source, config); });
            Assert.Contains("CPU only", error.Message, StringComparison.Ordinal);
            using var raw = SafetensorsFile.Open(path);
            Assert.Throws<InvalidDataException>(() => { using var rejected = TransformerModel.LoadFromSafetensors(raw, config); });
        }
    }

    [Fact]
    public void GptqZeroBiasPadding_IsIgnoredAndNonzeroSubstitutionFails()
    {
        var fixture = new Fixture(awq: false);
        string biasName = Projection + ".bias";
        fixture.Tensors[biasName] = new("F16", [96], new byte[96 * 2]);
        var (source, config) = ModelLoader.OpenSafetensorsAndConfig(fixture.Write(Path.Combine(_scratch, "zero-bias"), true));
        using (source)
        using (var weights = TransformerWeightsSafetensorsLoader.Load(source, config, allowPackedQuantization: true))
        {
            Assert.Null(SafetensorsTensorResolver.ResolveOptionalBias(source, biasName, 96));
            Assert.Equal(14, weights.LiveOwnedAllocationCount);
            Assert.Equal(fixture.Dense.Values.Sum(x => (long)x.Length) + 2 * 40 * 64 + 5 * 64,
                Assert.IsType<PackedQuantizedSafetensorsSource>(source).LogicalParameterCount);
        }
        BinaryPrimitives.WriteUInt16LittleEndian(fixture.Tensors[biasName].Bytes, 0x0001); // Smallest positive F16.
        string path = fixture.Write(Path.Combine(_scratch, "nonzero-bias"), true);
        var error = Assert.Throws<NotSupportedException>(() => { var (rejected, _) = ModelLoader.OpenSafetensorsAndConfig(path); rejected.Dispose(); });
        Assert.Contains("exactly zero", error.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("bits")]
    [InlineData("group_size")]
    [InlineData("version")]
    [InlineData("zero_point")]
    [InlineData("missing_quant")]
    [InlineData("wrong_method")]
    [InlineData("architecture")]
    [InlineData("head_dim")]
    [InlineData("zero_dimension")]
    [InlineData("missing_member")]
    [InlineData("wrong_dtype")]
    [InlineData("wrong_shape")]
    [InlineData("byte_count")]
    [InlineData("zero_scale")]
    [InlineData("nan_scale")]
    [InlineData("orphan")]
    [InlineData("dense_duplicate")]
    [InlineData("bias")]
    [InlineData("fused")]
    [InlineData("packed_head")]
    public void MalformedAwq_FailsDuringOpen(string mutation) => Reject(mutation, awq: true);

    [Theory]
    [InlineData("bits")]
    [InlineData("group_size")]
    [InlineData("format")]
    [InlineData("missing_desc_act")]
    [InlineData("missing_sym")]
    [InlineData("missing_g_idx")]
    [InlineData("bad_g_idx")]
    [InlineData("unbalanced_g_idx")]
    [InlineData("contradictory_desc_act")]
    [InlineData("contradictory_sym")]
    [InlineData("missing_member")]
    [InlineData("wrong_dtype")]
    [InlineData("wrong_shape")]
    [InlineData("zero_scale")]
    public void MalformedGptq_FailsDuringOpen(string mutation) => Reject(mutation, awq: false);

    private void Reject(string mutation, bool awq)
    {
        var fixture = new Fixture(awq);
        string key = Projection + ".scales";
        switch (mutation)
        {
            case "bits": fixture.Quant["bits"] = 3; break;
            case "group_size": fixture.Quant["group_size"] = 48; break;
            case "version": fixture.Quant["version"] = "gemv"; break;
            case "zero_point": fixture.Quant["zero_point"] = false; break;
            case "missing_quant": fixture.Config.Remove("quantization_config"); break;
            case "wrong_method": fixture.Quant["quant_method"] = "mystery"; break;
            case "architecture": fixture.Config["model_type"] = "qwen2"; break;
            case "head_dim": fixture.Config["head_dim"] = 32; break;
            case "zero_dimension": fixture.Config["hidden_size"] = 0; break;
            case "format": fixture.Quant["checkpoint_format"] = "gptq_v2"; break;
            case "missing_desc_act": fixture.Quant.Remove("desc_act"); break;
            case "missing_sym": fixture.Quant.Remove("sym"); break;
            case "missing_g_idx": fixture.Tensors.Remove(Projection + ".g_idx"); break;
            case "contradictory_desc_act": fixture.Quant["desc_act"] = false; break;
            case "contradictory_sym": fixture.Quant["sym"] = true; break;
            case "bad_g_idx": BinaryPrimitives.WriteInt32LittleEndian(fixture.Tensors[Projection + ".g_idx"].Bytes, 2); break;
            case "unbalanced_g_idx": BinaryPrimitives.WriteInt32LittleEndian(fixture.Tensors[Projection + ".g_idx"].Bytes, 1); break;
            case "missing_member": fixture.Tensors.Remove(key); break;
            case "wrong_dtype": fixture.Tensors[key] = fixture.Tensors[key] with { DType = "BF16" }; break;
            case "wrong_shape": fixture.Tensors[key] = fixture.Tensors[key] with { Shape = [96, 2] }; break;
            case "byte_count": fixture.Tensors[key] = fixture.Tensors[key] with { Bytes = fixture.Tensors[key].Bytes[..^2] }; break;
            case "zero_scale": BinaryPrimitives.WriteUInt16LittleEndian(fixture.Tensors[key].Bytes, 0); break;
            case "nan_scale": BinaryPrimitives.WriteUInt16LittleEndian(fixture.Tensors[key].Bytes, 0x7e00); break;
            case "orphan": fixture.Tensors["unused.qzeros"] = fixture.Tensors[Projection + ".qzeros"]; break;
            case "dense_duplicate": fixture.Tensors[Projection + ".weight"] = fixture.Tensors[key]; break;
            case "bias": fixture.Tensors[Projection + ".bias"] = fixture.Tensors[key]; break;
            case "fused": fixture.Tensors["model.layers.0.self_attn.qkv_proj.weight"] = fixture.Tensors[key]; break;
            case "packed_head": fixture.Tensors["lm_head.qweight"] = fixture.Tensors[Projection + ".qweight"]; break;
            default: throw new ArgumentOutOfRangeException(nameof(mutation));
        }
        string path = fixture.Write(Path.Combine(_scratch, "invalid"), true);
        Exception? error = Record.Exception(() => { var (source, _) = ModelLoader.OpenSafetensorsAndConfig(path); source.Dispose(); });
        Assert.True(error is InvalidDataException or NotSupportedException, $"{mutation}: expected format rejection, got {error}");
        // Failed-open ownership unwinds all shards: on Windows these files can now be reopened exclusively.
        foreach (string shard in Directory.GetFiles(Path.GetDirectoryName(path)!, "*.safetensors"))
            using (new FileStream(shard, FileMode.Open, FileAccess.ReadWrite, FileShare.None)) { }
    }

    private static void AssertClose(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(Math.Abs(expected[i] - actual[i]), 0, tolerance * (1 + Math.Abs(expected[i])));
    }

    internal sealed record Tensor(string DType, int[] Shape, byte[] Bytes);
    internal sealed class Fixture
    {
        public Dictionary<string, Tensor> Tensors { get; } = new(StringComparer.Ordinal);
        public Dictionary<string, float[]> Dense { get; } = new(StringComparer.Ordinal);
        public Dictionary<string, object?> Config { get; }
        public Dictionary<string, object?> Quant { get; }
        public Fixture(bool awq, bool tied = false, int hidden = 64, int intermediate = 96, int groupSize = 32, bool actOrder = true)
        {
            Quant = new(StringComparer.Ordinal) { ["quant_method"] = awq ? "awq" : "gptq", ["bits"] = 4, ["group_size"] = groupSize,
                ["version"] = "gemm", ["zero_point"] = true, ["desc_act"] = actOrder, ["sym"] = false };
            Config = new(StringComparer.Ordinal) { ["model_type"] = "llama", ["hidden_size"] = hidden, ["intermediate_size"] = intermediate,
                ["num_hidden_layers"] = 2, ["num_attention_heads"] = 4, ["num_key_value_heads"] = 2,
                ["vocab_size"] = 40, ["max_position_embeddings"] = 16, ["rms_norm_eps"] = 1e-5,
                ["tie_word_embeddings"] = tied, ["quantization_config"] = Quant };
            AddFloat("model.embed_tokens.weight", [40, hidden]);
            AddFloat("model.norm.weight", [hidden], true);
            if (!tied) AddFloat("lm_head.weight", [40, hidden]);
            for (int layer = 0; layer < 2; layer++)
            {
                string p = $"model.layers.{layer}.";
                AddFloat(p + "input_layernorm.weight", [hidden], true);
                AddFloat(p + "post_attention_layernorm.weight", [hidden], true);
                AddPacked(p + "self_attn.q_proj", hidden, hidden);
                AddPacked(p + "self_attn.k_proj", hidden / 2, hidden);
                AddPacked(p + "self_attn.v_proj", hidden / 2, hidden);
                AddPacked(p + "self_attn.o_proj", hidden, hidden);
                AddPacked(p + "mlp.gate_proj", intermediate, hidden);
                AddPacked(p + "mlp.up_proj", intermediate, hidden);
                AddPacked(p + "mlp.down_proj", hidden, intermediate);
            }
            void AddPacked(string name, int m, int k)
            {
                int group = groupSize == -1 ? k : groupSize, groups = k / group;
                var codes = new int[k, m];
                var zeros = new int[groups, m];
                var scales = new Half[groups * m];
                var indices = Enumerable.Range(0, k).Select(i => awq || !actOrder ? i / group : (i * 5 % k) / group).ToArray();
                for (int g = 0; g < groups; g++) for (int row = 0; row < m; row++)
                {
                    zeros[g, row] = (row + g * 3) % 13 + 1;
                    scales[g * m + row] = (Half)((1 + (row * 7 + g * 3) % 9) / 2048f);
                }
                var dense = new float[m * k];
                for (int col = 0; col < k; col++) for (int row = 0; row < m; row++)
                {
                    codes[col, row] = (row * 3 + col * 5 + col / 7) % 16;
                    dense[row * k + col] = (codes[col, row] - zeros[indices[col], row]) * (float)scales[indices[col] * m + row];
                }
                Dense[name] = dense;
                // Pack in the producer's FORWARD order; reference above never unpacks these bytes.
                int[] order = awq ? [0, 2, 4, 6, 1, 3, 5, 7] : [0, 1, 2, 3, 4, 5, 6, 7];
                var qw = new uint[k * m / 8];
                var qz = new uint[groups * m / 8];
                for (int col = 0; col < k; col++) for (int row = 0; row < m; row++)
                    if (awq) qw[col * (m / 8) + row / 8] |= (uint)codes[col, (row / 8) * 8 + order[row % 8]] << ((row % 8) * 4);
                    else qw[(col / 8) * m + row] |= (uint)codes[col, row] << ((col % 8) * 4);
                for (int g = 0; g < groups; g++) for (int row = 0; row < m; row++)
                    qz[g * (m / 8) + row / 8] |= (uint)(zeros[g, (row / 8) * 8 + order[row % 8]] - (awq ? 0 : 1)) << ((row % 8) * 4);
                Tensors[name + ".qweight"] = new("I32", awq ? [k, m / 8] : [k / 8, m], MemoryMarshal.AsBytes(qw.AsSpan()).ToArray());
                Tensors[name + ".qzeros"] = new("I32", [groups, m / 8], MemoryMarshal.AsBytes(qz.AsSpan()).ToArray());
                Tensors[name + ".scales"] = new("F16", [groups, m], MemoryMarshal.AsBytes(scales.AsSpan()).ToArray());
                if (!awq) Tensors[name + ".g_idx"] = new("I32", [k], MemoryMarshal.AsBytes(indices.AsSpan()).ToArray());
            }
            void AddFloat(string name, int[] shape, bool norm = false)
            {
                int count = shape.Aggregate(1, (a, b) => a * b);
                Half[] values = Enumerable.Range(0, count).Select(i => (Half)(norm ? 1 : ((i * 13 + i / 7) % 31 - 15) / 128f)).ToArray();
                Tensors[name] = new("F16", shape, MemoryMarshal.AsBytes(values.AsSpan()).ToArray());
            }
        }

        public string Write(string directory, bool sharded, bool dense = false)
        {
            Directory.CreateDirectory(directory);
            var config = new Dictionary<string, object?>(Config, StringComparer.Ordinal);
            if (dense) config.Remove("quantization_config");
            File.WriteAllText(Path.Combine(directory, "config.json"), JsonSerializer.Serialize(config));
            var builders = new[] { new SafetensorsFixtureBuilder(), new SafetensorsFixtureBuilder() };
            var map = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var (name, tensor) in Tensors)
            {
                if (dense && (name.EndsWith(".qweight", StringComparison.Ordinal) || name.EndsWith(".qzeros", StringComparison.Ordinal)
                    || name.EndsWith(".scales", StringComparison.Ordinal) || name.EndsWith(".g_idx", StringComparison.Ordinal))) continue;
                int shard = sharded ? map.Count % 2 : 0;
                builders[shard].AddRaw(name, tensor.DType, tensor.Shape, tensor.Bytes);
                map[name] = shard == 0 ? "a.safetensors" : "b.safetensors";
            }
            if (dense)
                foreach (var (name, values) in Dense)
                {
                    int k = Tensors[name + ".qweight"].Shape[0] * (Equals(Quant["quant_method"], "awq") ? 1 : 8);
                    builders[0].AddFloat32(name + ".weight", [values.Length / k, k], values);
                }
            builders[0].WriteTo(Path.Combine(directory, "a.safetensors"));
            if (!sharded) return Path.Combine(directory, "a.safetensors");
            builders[1].WriteTo(Path.Combine(directory, "b.safetensors"));
            string path = Path.Combine(directory, "model.safetensors.index.json");
            File.WriteAllText(path, JsonSerializer.Serialize(new { weight_map = map }));
            return path;
        }
    }

    public void Dispose()
    {
        if (Directory.Exists(_scratch)) Directory.Delete(_scratch, recursive: true);
    }
}
