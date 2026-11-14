using System.Buffers.Binary;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Cpu.Kernels;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using Xunit;
using NativeMemory = System.Runtime.InteropServices.NativeMemory;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

public sealed unsafe class Bf16SafetensorsStorageTests : IDisposable
{
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-bf16-{Guid.NewGuid():N}");

    private static ModelConfig Config(bool tied) => new()
    {
        Architecture = Architecture.Llama,
        VocabSize = 29,
        HiddenSize = 24,
        IntermediateSize = 37,
        NumLayers = 2,
        NumAttentionHeads = 4,
        NumKvHeads = 2,
        HeadDim = 6,
        MaxSequenceLength = 16,
        NormEpsilon = 1e-5f,
        TiedEmbeddings = tied,
        RoPEConfig = new RoPEConfig(Theta: 10000, DimensionCount: 6, Type: RoPEType.NeoX),
    };

    [Fact]
    public void Bf16Values_KeepTheirFullExponentRangeAndBits()
    {
        // Include signed zero, subnormals and finite values beyond F16's exponent range.
        ushort[] bits = [0x0000, 0x8000, 0x0001, 0x8001, 0x0080, 0x8080,
            0x3F80, 0xBF80, 0x4780, 0xC780, 0x7F7F, 0xFF7F];
        var bytes = new byte[bits.Length * 2];
        for (int i = 0; i < bits.Length; i++)
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(i * 2), bits[i]);
        Directory.CreateDirectory(_scratch);
        string path = Path.Combine(_scratch, "range.safetensors");
        new SafetensorsFixtureBuilder().AddRaw("matrix", "BF16", [3, 4], bytes).WriteTo(path);
        using var file = SafetensorsFile.Open(path);
        var owned = new List<nint>();
        try
        {
            var (pointer, type, rows, columns) = SafetensorsTensorResolver.ResolveLinear(file, "matrix", owned);
            Span<float> values = stackalloc float[bits.Length];
            Dequantize.ToFloat32(pointer, rows * columns, type, values);
            for (int i = 0; i < bits.Length; i++)
                Assert.Equal((uint)bits[i] << 16, BitConverter.SingleToUInt32Bits(values[i]));
        }
        finally
        {
            foreach (nint pointer in owned) NativeMemory.AlignedFree((void*)pointer);
        }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void DenseWeights_BorrowBf16Mappings_WithoutOwnedCopies(bool sharded, bool tied)
    {
        using var file = WriteFixture("storage", bf16: true, sharded, tied);
        using (var weights = TransformerWeightsSafetensorsLoader.Load(file, Config(tied)))
        {
            Assert.Equal(0, weights.LiveOwnedAllocationCount);
            AssertMatrix("model.embed_tokens.weight", weights.TokenEmbedWeight, weights.TokenEmbedQuantType, 29, 24);
            AssertMatrix(tied ? "model.embed_tokens.weight" : "lm_head.weight",
                weights.OutputWeight, weights.OutputQuantType, weights.OutputOutputDim, weights.OutputInputDim);
            if (tied) Assert.Equal(weights.TokenEmbedWeight, weights.OutputWeight);

            for (int i = 0; i < weights.Layers.Length; i++)
            {
                ref readonly var layer = ref weights.Layers[i];
                string prefix = $"model.layers.{i}";
                AssertMatrix($"{prefix}.self_attn.q_proj.weight", layer.QWeight, layer.QQuantType, layer.QOutputDim, layer.QInputDim);
                AssertMatrix($"{prefix}.self_attn.k_proj.weight", layer.KWeight, layer.KQuantType, layer.KOutputDim, layer.KInputDim);
                AssertMatrix($"{prefix}.self_attn.v_proj.weight", layer.VWeight, layer.VQuantType, layer.VOutputDim, layer.VInputDim);
                AssertMatrix($"{prefix}.self_attn.o_proj.weight", layer.OWeight, layer.OQuantType, layer.OOutputDim, layer.OInputDim);
                AssertMatrix($"{prefix}.mlp.gate_proj.weight", layer.GateWeight, layer.GateQuantType, layer.GateOutputDim, layer.GateInputDim);
                AssertMatrix($"{prefix}.mlp.up_proj.weight", layer.UpWeight, layer.UpQuantType, layer.UpOutputDim, layer.UpInputDim);
                AssertMatrix($"{prefix}.mlp.down_proj.weight", layer.DownWeight, layer.DownQuantType, layer.DownOutputDim, layer.DownInputDim);
            }
            Assert.False(weights.TryReleaseOwnedHostAllocation(weights.TokenEmbedWeight));
        }

        // Disposing weights must leave the caller-owned mapping usable for another model.
        using var model = TransformerModel.LoadFromSafetensors(file, Config(tied));
        using var logits = model.Forward([0, 7], [0, 1], deviceId: -1);
        Assert.Equal(29, logits.Shape[1]);

        void AssertMatrix(string name, nint pointer, QuantizationType type, int rows, int columns)
        {
            var descriptor = file.TensorsByName[name];
            Assert.Equal(QuantizationType.BF16, type);
            Assert.Equal(file.GetTensorPointer(name), pointer);
            Assert.Equal(descriptor.Shape, new[] { rows, columns });
            Assert.Equal(descriptor.ElementCount * 2, Dequantize.RowByteSize(columns, type) * rows);
        }
    }

    [Theory]
    [InlineData(false, false, 1)]
    [InlineData(false, true, 2)]
    [InlineData(true, false, 2)]
    [InlineData(true, true, 1)]
    public void PrefillAndCachedDecode_MatchExactlyWidenedReference(bool sharded, bool tied, int threads)
    {
        using var bf16File = WriteFixture("bf16", bf16: true, sharded, tied);
        using var referenceFile = WriteFixture("reference", bf16: false, sharded: false, tied);
        var config = Config(tied);
        var threading = new ThreadingConfig(threads, EnableCallerPinning: false);
        using var actual = TransformerModel.LoadFromSafetensors(bf16File, config, threading);
        using var reference = TransformerModel.LoadFromSafetensors(referenceFile, config, threading);
        using var actualCache = new SimpleKvCache(2, 2, 6, 16);
        using var referenceCache = new SimpleKvCache(2, 2, 6, 16);

        Compare([0, 7, 13], [0, 1, 2]);
        Compare([4], [3]);
        Compare([17], [4]);

        void Compare(int[] tokens, int[] positions)
        {
            using var expected = reference.Forward(tokens, positions, -1, referenceCache);
            using var observed = actual.Forward(tokens, positions, -1, actualCache);
            int count = tokens.Length * config.VocabSize;
            var expectedValues = new ReadOnlySpan<float>((void*)expected.DataPointer, count);
            var observedValues = new ReadOnlySpan<float>((void*)observed.DataPointer, count);
            for (int i = 0; i < count; i++)
                Assert.InRange(Math.Abs(expectedValues[i] - observedValues[i]), 0,
                    2e-5f * (1 + Math.Abs(expectedValues[i])));
        }
    }

    private ISafetensorsTensorSource WriteFixture(string name, bool bf16, bool sharded, bool tied)
    {
        string directory = Path.Combine(_scratch, name);
        Directory.CreateDirectory(directory);
        var builders = new[] { new SafetensorsFixtureBuilder(), new SafetensorsFixtureBuilder() };
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        var rng = new Random(1143);
        Add("model.embed_tokens.weight", [29, 24]);
        Add("model.norm.weight", [24], norm: true);
        if (!tied) Add("lm_head.weight", [29, 24]);
        for (int layer = 0; layer < 2; layer++)
        {
            string prefix = $"model.layers.{layer}";
            Add($"{prefix}.input_layernorm.weight", [24], norm: true);
            Add($"{prefix}.post_attention_layernorm.weight", [24], norm: true);
            Add($"{prefix}.self_attn.q_proj.weight", [24, 24]);
            Add($"{prefix}.self_attn.k_proj.weight", [12, 24]);
            Add($"{prefix}.self_attn.v_proj.weight", [12, 24]);
            Add($"{prefix}.self_attn.o_proj.weight", [24, 24]);
            Add($"{prefix}.mlp.gate_proj.weight", [37, 24]);
            Add($"{prefix}.mlp.up_proj.weight", [37, 24]);
            Add($"{prefix}.mlp.down_proj.weight", [24, 37]);
        }
        builders[0].WriteTo(Path.Combine(directory, "a.safetensors"));
        if (!sharded) return SafetensorsFile.Open(Path.Combine(directory, "a.safetensors"));
        builders[1].WriteTo(Path.Combine(directory, "b.safetensors"));
        string index = Path.Combine(directory, "model.safetensors.index.json");
        File.WriteAllText(index, JsonSerializer.Serialize(new { weight_map = map }));
        return MultiShardSafetensorsFile.Open(index);

        void Add(string tensorName, int[] shape, bool norm = false)
        {
            int count = shape.Aggregate(1, (a, b) => a * b);
            var bytes = new byte[count * 2];
            var values = new float[count];
            for (int i = 0; i < count; i++)
            {
                float value = norm ? 1f : (rng.NextSingle() * 2 - 1) * 0.125f;
                ushort bits = (ushort)(BitConverter.SingleToUInt32Bits(value) >> 16);
                BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(i * 2), bits);
                values[i] = BitConverter.UInt32BitsToSingle((uint)bits << 16);
            }
            int shard = sharded ? map.Count % 2 : 0;
            map.Add(tensorName, shard == 0 ? "a.safetensors" : "b.safetensors");
            if (bf16) builders[shard].AddRaw(tensorName, "BF16", shape, bytes);
            else builders[shard].AddFloat32(tensorName, shape, values);
        }
    }

    public void Dispose()
    {
        if (Directory.Exists(_scratch)) Directory.Delete(_scratch, recursive: true);
    }
}
