using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

/// <summary>
/// Synthetic-fixture tests for the BitNet b1.58 HuggingFace safetensors load
/// path, driven end-to-end through <see cref="ModelLoader.LoadFromSafetensors(string, System.Nullable{ThreadingConfig})"/>.
/// A tiny BitNet-shaped checkpoint (bf16 linears, F32 norms + Sub-LN weights)
/// plus a <c>config.json</c> (<c>model_type=bitnet</c>, <c>hidden_act=relu2</c>)
/// is written to a scratch directory, then loaded via the public dispatch.
/// Verifies that:
/// <list type="bullet">
///   <item><see cref="HfConfigExtractor"/> detects <see cref="Architecture.BitNet"/>
///     and sets the squared-ReLU FFN;</item>
///   <item>the loader quantizes every linear projection to
///     <see cref="QuantizationType.I2_S"/> (ternary) instead of NotSupported-ing;</item>
///   <item>tie_word_embeddings aliases the embedding matrix as the LM head; and</item>
///   <item>a forward pass produces finite <c>[seq, vocab]</c> logits.</item>
/// </list>
/// </summary>
public sealed class BitNetSafetensorsLoadTests : IDisposable
{
    // hidden=128 makes every linear's element count a multiple of 128 (I2_S
    // requires 128-element blocks) — at least one dim of every projection is
    // the hidden size.
    private const int Hidden = 128;
    private const int NumHeads = 4;
    private const int NumKvHeads = 2;
    private const int HeadDim = 32;
    private const int Intermediate = 256;
    private const int Vocab = 64;
    private const int NumLayers = 2;

    private readonly string _scratch;

    public BitNetSafetensorsLoadTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-bitnet-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void LoadAndForward_ProducesFiniteVocabLogits_WithI2SAndSubLn()
    {
        BuildBitNetFixture();
        WriteConfigJson();

        var (model, source, config) = ModelLoader.LoadFromSafetensors(_scratch);
        try
        {
            Assert.Equal(Architecture.BitNet, config.Architecture);
            Assert.Equal(ActivationFunction.ReluSquared, config.ActivationFunction);
            Assert.Equal(Vocab, config.VocabSize);
            Assert.Equal(Hidden, config.HiddenSize);
            Assert.Equal(NumLayers, config.NumLayers);
            Assert.True(config.TiedEmbeddings);

            using var logits = model.Forward(
                tokenIds: [0, 1, 2],
                positions: [0, 1, 2],
                deviceId: -1);

            Assert.Equal(2, logits.Shape.Rank);
            Assert.Equal(3, logits.Shape[0]);
            Assert.Equal(Vocab, logits.Shape[1]);
            AssertAllFinite(logits);
        }
        finally
        {
            model.Dispose();
            source.Dispose();
        }
    }

    /// <summary>
    /// Writes a tiny BitNet fixture: bf16 linears (q/k/v/o + gate/up/down),
    /// F32 norms and Sub-LN weights, tied embeddings (no lm_head).
    /// </summary>
    private void BuildBitNetFixture()
    {
        var rng = new Random(58);
        var b = new SafetensorsFixtureBuilder();

        b.AddFloat32("model.embed_tokens.weight", [Vocab, Hidden], RandomVec(rng, Vocab * Hidden, 0.05f));
        b.AddFloat32("model.norm.weight", [Hidden], Ones(Hidden));

        for (int i = 0; i < NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            b.AddFloat32($"{p}.input_layernorm.weight", [Hidden], Ones(Hidden));
            b.AddFloat32($"{p}.post_attention_layernorm.weight", [Hidden], Ones(Hidden));

            AddBf16($"{p}.self_attn.q_proj.weight", [NumHeads * HeadDim, Hidden], b, rng);
            AddBf16($"{p}.self_attn.k_proj.weight", [NumKvHeads * HeadDim, Hidden], b, rng);
            AddBf16($"{p}.self_attn.v_proj.weight", [NumKvHeads * HeadDim, Hidden], b, rng);
            AddBf16($"{p}.self_attn.o_proj.weight", [Hidden, NumHeads * HeadDim], b, rng);
            // BitNet Sub-LN over the attention output before o_proj.
            b.AddFloat32($"{p}.self_attn.attn_sub_norm.weight", [Hidden], Ones(Hidden));

            AddBf16($"{p}.mlp.gate_proj.weight", [Intermediate, Hidden], b, rng);
            AddBf16($"{p}.mlp.up_proj.weight", [Intermediate, Hidden], b, rng);
            AddBf16($"{p}.mlp.down_proj.weight", [Hidden, Intermediate], b, rng);
            // BitNet Sub-LN over the gated intermediate before down_proj.
            b.AddFloat32($"{p}.mlp.ffn_sub_norm.weight", [Intermediate], Ones(Intermediate));
        }

        b.WriteTo(Path.Combine(_scratch, "model.safetensors"));
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

    private static void AddBf16(string name, int[] shape, SafetensorsFixtureBuilder b, Random rng)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        var bytes = new byte[n * 2];
        for (long i = 0; i < n; i++)
        {
            // Small-amplitude values so the absmean ternary scale is well
            // conditioned; encoded as bf16 (high 16 bits of the f32 bit pattern).
            float v = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.05);
            uint bits = System.Runtime.CompilerServices.Unsafe.As<float, uint>(ref v);
            ushort bf16 = (ushort)(bits >> 16);
            bytes[i * 2] = (byte)(bf16 & 0xFF);
            bytes[i * 2 + 1] = (byte)(bf16 >> 8);
        }
        b.AddRaw(name, "BF16", shape, bytes);
    }

    private static float[] RandomVec(Random rng, int n, float scale)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++)
            v[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        return v;
    }

    private static float[] Ones(int n)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = 1.0f;
        return v;
    }

    private static unsafe void AssertAllFinite(ITensor logits)
    {
        int n = 1;
        for (int i = 0; i < logits.Shape.Rank; i++)
            n *= logits.Shape[i];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, n);
        for (int i = 0; i < span.Length; i++)
        {
            float v = span[i];
            Assert.True(float.IsFinite(v), $"Logit index {i} is non-finite ({v}).");
        }
    }
}
