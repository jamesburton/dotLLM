using System.Buffers.Binary;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Loaders;

/// <summary>
/// CPU-only integration coverage that proves a real loaded transformer honors
/// <see cref="ModelConfig.SlidingWindowSize"/> during <see cref="TransformerModel.Forward"/>.
/// </summary>
public sealed class SlidingWindowSyntheticQwenForwardTests : IDisposable
{
    private const int Hidden = 8;
    private const int NumHeads = 2;
    private const int NumKvHeads = 1;
    private const int HeadDim = 4;
    private const int Intermediate = 16;
    private const int Vocab = 32;
    private const int SeqLen = 16;
    private const int SlidingWindow = 4;
    private const float Eps = 1e-5f;

    private readonly string _scratch;

    public SlidingWindowSyntheticQwenForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-sliding-qwen-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort cleanup */ }
    }

    [Fact]
    public unsafe void SyntheticQwen_ForwardLogitsMatchBruteForceSlidingWindowReference()
    {
        var fixture = BuildFixture();
        string path = Path.Combine(_scratch, "synthetic-qwen-sliding.safetensors");
        fixture.Builder.WriteTo(path);

        using var file = SafetensorsFile.Open(path);
        var config = BuildConfig(slidingWindowSize: SlidingWindow);
        using var model = TransformerModel.LoadFromSafetensors(file, config);

        int[] tokenIds = Enumerable.Range(0, SeqLen).Select(i => (i * 7 + 3) % Vocab).ToArray();
        int[] positions = Enumerable.Range(0, SeqLen).ToArray();

        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);

        Assert.Equal(SlidingWindow, model.Config.SlidingWindowSize);
        Assert.Equal(SeqLen, logits.Shape[0]);
        Assert.Equal(Vocab, logits.Shape[1]);

        float[] expected = ComputeReferenceLogits(tokenIds, fixture.Embeddings, fixture.LmHead, SlidingWindow);
        var actual = new ReadOnlySpan<float>((void*)logits.DataPointer, SeqLen * Vocab);

        for (int i = 0; i < actual.Length; i++)
            Assert.Equal(expected[i], actual[i], 1e-4f);

        float[] fullCausal = ComputeReferenceLogits(tokenIds, fixture.Embeddings, fixture.LmHead, slidingWindow: null);
        Assert.True(
            MathF.Abs(expected[(SeqLen - 1) * Vocab] - fullCausal[(SeqLen - 1) * Vocab]) > 1e-3f,
            "Synthetic fixture must distinguish sliding-window attention from full causal attention.");
    }

    private static (TinySafetensorsBuilder Builder, float[] Embeddings, float[] LmHead) BuildFixture()
    {
        float[] embeddings = Matrix(Vocab, Hidden, seed: 11, scale: 0.07f, bias: 0.13f);
        float[] lmHead = Matrix(Vocab, Hidden, seed: 29, scale: 0.05f, bias: 0.0f);
        var b = new TinySafetensorsBuilder()
            .AddFloat32("model.embed_tokens.weight", [Vocab, Hidden], embeddings)
            .AddFloat32("model.norm.weight", [Hidden], Ones(Hidden))
            .AddFloat32("lm_head.weight", [Vocab, Hidden], lmHead);

        b.AddFloat32("model.layers.0.input_layernorm.weight", [Hidden], Ones(Hidden));
        b.AddFloat32("model.layers.0.post_attention_layernorm.weight", [Hidden], Ones(Hidden));

        b.AddFloat32("model.layers.0.self_attn.q_proj.weight", [NumHeads * HeadDim, Hidden], new float[NumHeads * HeadDim * Hidden]);
        b.AddFloat32("model.layers.0.self_attn.k_proj.weight", [NumKvHeads * HeadDim, Hidden], new float[NumKvHeads * HeadDim * Hidden]);
        b.AddFloat32("model.layers.0.self_attn.v_proj.weight", [NumKvHeads * HeadDim, Hidden], VProjection());
        b.AddFloat32("model.layers.0.self_attn.o_proj.weight", [Hidden, NumHeads * HeadDim], OProjection());

        b.AddFloat32("model.layers.0.mlp.gate_proj.weight", [Intermediate, Hidden], new float[Intermediate * Hidden]);
        b.AddFloat32("model.layers.0.mlp.up_proj.weight", [Intermediate, Hidden], new float[Intermediate * Hidden]);
        b.AddFloat32("model.layers.0.mlp.down_proj.weight", [Hidden, Intermediate], new float[Hidden * Intermediate]);

        return (b, embeddings, lmHead);
    }

    private static ModelConfig BuildConfig(int? slidingWindowSize)
        => new()
        {
            Architecture = Architecture.Qwen,
            VocabSize = Vocab,
            HiddenSize = Hidden,
            IntermediateSize = Intermediate,
            NumLayers = 1,
            NumAttentionHeads = NumHeads,
            NumKvHeads = NumKvHeads,
            HeadDim = HeadDim,
            MaxSequenceLength = 64,
            NormEpsilon = Eps,
            TiedEmbeddings = false,
            SlidingWindowSize = slidingWindowSize,
            RoPEConfig = new RoPEConfig(Theta: 1_000_000.0f, DimensionCount: HeadDim, Type: RoPEType.NeoX),
        };

    private static float[] ComputeReferenceLogits(int[] tokenIds, float[] embeddings, float[] lmHead, int? slidingWindow)
    {
        float[][] tokenHidden = new float[SeqLen][];
        float[][] values = new float[SeqLen][];

        for (int t = 0; t < SeqLen; t++)
        {
            tokenHidden[t] = Row(embeddings, tokenIds[t], Hidden);
            float[] normed = RmsNorm(tokenHidden[t]);
            values[t] = normed[..HeadDim];
        }

        float[] logits = new float[SeqLen * Vocab];
        for (int t = 0; t < SeqLen; t++)
        {
            int start = slidingWindow is int w ? Math.Max(0, t - w + 1) : 0;
            int count = t - start + 1;
            float[] hiddenAfterAttention = (float[])tokenHidden[t].Clone();

            for (int d = 0; d < HeadDim; d++)
            {
                float sum = 0.0f;
                for (int src = start; src <= t; src++)
                    sum += values[src][d];
                hiddenAfterAttention[d] += sum / count;
            }

            float[] finalHidden = RmsNorm(hiddenAfterAttention);
            for (int vocab = 0; vocab < Vocab; vocab++)
                logits[t * Vocab + vocab] = Dot(Row(lmHead, vocab, Hidden), finalHidden);
        }

        return logits;
    }

    private static float[] VProjection()
    {
        var w = new float[NumKvHeads * HeadDim * Hidden];
        for (int d = 0; d < HeadDim; d++)
            w[d * Hidden + d] = 1.0f;
        return w;
    }

    private static float[] OProjection()
    {
        var w = new float[Hidden * NumHeads * HeadDim];
        for (int d = 0; d < HeadDim; d++)
            w[d * (NumHeads * HeadDim) + d] = 1.0f;
        return w;
    }

    private static float[] RmsNorm(float[] input)
    {
        float sumSq = 0.0f;
        for (int i = 0; i < input.Length; i++)
            sumSq += input[i] * input[i];

        float scale = 1.0f / MathF.Sqrt(sumSq / input.Length + Eps);
        var output = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
            output[i] = input[i] * scale;
        return output;
    }

    private static float Dot(float[] left, float[] right)
    {
        float sum = 0.0f;
        for (int i = 0; i < left.Length; i++)
            sum += left[i] * right[i];
        return sum;
    }

    private static float[] Row(float[] matrix, int row, int cols)
    {
        var result = new float[cols];
        Array.Copy(matrix, row * cols, result, 0, cols);
        return result;
    }

    private static float[] Ones(int count)
    {
        var values = new float[count];
        Array.Fill(values, 1.0f);
        return values;
    }

    private static float[] Matrix(int rows, int cols, int seed, float scale, float bias)
    {
        var values = new float[rows * cols];
        for (int i = 0; i < values.Length; i++)
        {
            int bucket = ((i + 1) * (seed + 17) + seed * 31) % 23;
            values[i] = bias + (bucket - 11) * scale;
        }
        return values;
    }

    private sealed class TinySafetensorsBuilder
    {
        private readonly List<(string Name, int[] Shape, byte[] Bytes)> _tensors = new();

        public TinySafetensorsBuilder AddFloat32(string name, int[] shape, ReadOnlySpan<float> values)
        {
            long count = 1;
            for (int i = 0; i < shape.Length; i++)
                count *= shape[i];
            if (values.Length != count)
                throw new ArgumentException($"Shape implies {count} values, got {values.Length}.", nameof(values));

            var bytes = new byte[count * sizeof(float)];
            for (int i = 0; i < values.Length; i++)
                BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * sizeof(float), sizeof(float)), values[i]);

            _tensors.Add((name, shape, bytes));
            return this;
        }

        public void WriteTo(string path)
        {
            using var header = new MemoryStream();
            using (var writer = new Utf8JsonWriter(header))
            {
                writer.WriteStartObject();
                long offset = 0;
                foreach (var (name, shape, bytes) in _tensors)
                {
                    writer.WriteStartObject(name);
                    writer.WriteString("dtype", "F32");
                    writer.WritePropertyName("shape");
                    writer.WriteStartArray();
                    foreach (int dim in shape)
                        writer.WriteNumberValue(dim);
                    writer.WriteEndArray();
                    writer.WritePropertyName("data_offsets");
                    writer.WriteStartArray();
                    writer.WriteNumberValue(offset);
                    writer.WriteNumberValue(offset + bytes.Length);
                    writer.WriteEndArray();
                    writer.WriteEndObject();
                    offset += bytes.Length;
                }
                writer.WriteEndObject();
            }

            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
            Span<byte> prefix = stackalloc byte[sizeof(ulong)];
            BinaryPrimitives.WriteUInt64LittleEndian(prefix, (ulong)header.Length);
            fs.Write(prefix);
            fs.Write(header.GetBuffer().AsSpan(0, (int)header.Length));
            foreach (var (_, _, bytes) in _tensors)
                fs.Write(bytes);
        }
    }
}
