using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models.Loaders;

/// <summary>
/// End-to-end verification that <see cref="ModelLoader.LoadFromSafetensors(string, System.Nullable{ThreadingConfig})"/>
/// can open a synthetic BitNet b1.58 HuggingFace checkpoint directory
/// (<c>config.json</c> + <c>model.safetensors</c>) and run a forward pass that
/// produces finite vocab-sized logits.
/// </summary>
/// <remarks>
/// <para>
/// No download — the fixture is written in-code (mirroring the byte-accurate
/// safetensors layout: LE u64 header length, UTF-8 JSON header, raw row-major
/// data region). The linear projections are bf16 (as in the real
/// <c>microsoft/bitnet-b1.58-2B-4T-bf16</c> checkpoint); the loader quantizes
/// each of them to ternary I2_S at load. Norms + Sub-LN weights are F32.
/// </para>
/// <para>
/// We are proving the loading plumbing end-to-end — HfConfigExtractor detects
/// BitNet + squared-ReLU, the tensor names resolve, every linear becomes I2_S,
/// the attn/ffn Sub-LN wire up, tie_word_embeddings aliases the LM head, and
/// the forward returns <c>[seq, vocab]</c> logits without NaN/Inf. We are NOT
/// asserting semantic output quality (random weights → random logits).
/// Numerical parity against the real bf16 checkpoint is out of scope here and
/// must be validated separately with the cached
/// <c>microsoft/bitnet-b1.58-2B-4T-bf16</c>.
/// </para>
/// </remarks>
public sealed class SyntheticBitNetSafetensorsLoadTests : IDisposable
{
    // hidden=128 keeps every linear's element count a multiple of 128, which
    // the I2_S ternary packer requires (128-element blocks).
    private const int Hidden = 128;
    private const int NumHeads = 4;
    private const int NumKvHeads = 2;
    private const int HeadDim = 32;
    private const int Intermediate = 256;
    private const int Vocab = 64;
    private const int NumLayers = 2;

    private readonly string _scratch;
    private readonly ITestOutputHelper _output;

    public SyntheticBitNetSafetensorsLoadTests(ITestOutputHelper output)
    {
        _output = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-bitnet-it-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void LoadAndForwardPass_ProducesFiniteVocabLogits()
    {
        WriteSafetensors();
        WriteConfigJson();

        var (model, source, config) = ModelLoader.LoadFromSafetensors(_scratch);
        try
        {
            _output.WriteLine(
                $"Config: arch={config.Architecture} act={config.ActivationFunction} "
              + $"vocab={config.VocabSize} hidden={config.HiddenSize} layers={config.NumLayers} "
              + $"heads={config.NumAttentionHeads} kv_heads={config.NumKvHeads} head_dim={config.HeadDim} "
              + $"intermediate={config.IntermediateSize} tied={config.TiedEmbeddings}");

            Assert.Equal(Architecture.BitNet, config.Architecture);
            Assert.Equal(ActivationFunction.ReluSquared, config.ActivationFunction);
            Assert.Equal(Vocab, config.VocabSize);
            Assert.Equal(Hidden, config.HiddenSize);
            Assert.Equal(NumLayers, config.NumLayers);
            Assert.True(config.TiedEmbeddings);

            int[] tokenIds = [0, 1, 2];
            int[] positions = [0, 1, 2];
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);

            Assert.Equal(2, logits.Shape.Rank);
            Assert.Equal(tokenIds.Length, logits.Shape[0]);
            Assert.Equal(config.VocabSize, logits.Shape[1]);
            AssertAllFinite(logits);
        }
        finally
        {
            model.Dispose();
            source.Dispose();
        }
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
        var tensors = new List<(string Name, string DType, int[] Shape, byte[] Bytes)>();

        tensors.Add(("model.embed_tokens.weight", "F32", [Vocab, Hidden], F32(rng, Vocab * Hidden, 0.05f)));
        tensors.Add(("model.norm.weight", "F32", [Hidden], Ones(Hidden)));

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
            ushort bf16 = (ushort)(bits >> 16); // truncate to high 16 bits
            bytes[i * 2] = (byte)(bf16 & 0xFF);
            bytes[i * 2 + 1] = (byte)(bf16 >> 8);
        }
        return bytes;
    }

    private static unsafe void AssertAllFinite(ITensor logits)
    {
        int n = 1;
        for (int i = 0; i < logits.Shape.Rank; i++) n *= logits.Shape[i];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, n);
        for (int i = 0; i < span.Length; i++)
            Assert.True(float.IsFinite(span[i]), $"Logit index {i} is non-finite ({span[i]}).");
    }
}
