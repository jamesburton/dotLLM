using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// A <see cref="TransformerModel"/> decoding through a <see cref="PagedKvCache"/> (attention reading the blocks in place,
/// #1268) emits the same greedy tokens as through a <see cref="SimpleKvCache"/>, and never allocates the legacy staging.
/// Synthetic F32 GQA fixture (group 2, two KV heads, head dim 32), so it runs in every configuration.
/// </summary>
public sealed class TransformerModelPagedKvDecodeTests : IDisposable
{
    private const int Hidden = 128;
    private const int Layers = 2;
    private const int Heads = 4;
    private const int KvHeads = 2;
    private const int HeadDim = Hidden / Heads;
    private const int Vocab = 48;
    private const int Intermediate = 96;
    private const int MaxSeq = 96;
    private const int Steps = 24;

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-paged-{Guid.NewGuid():N}");

    public TransformerModelPagedKvDecodeTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Theory]
    [InlineData("f32")]
    [InlineData("f16")]
    [InlineData("q8_0")]
    public void PagedDecodeIsTokenIdenticalToSimpleCache(string type)
    {
        DType elementType = type switch { "f32" => DType.Float32, "f16" => DType.Float16, _ => DType.Q8_0 };
        string path = Path.Combine(_scratch, "fixture.safetensors");
        WriteFixture(path);
        var cfg = BuildConfig();
        int[] prompt = [3, 17, 5, 41, 9, 22, 30];

        using var sf = SafetensorsFile.Open(path);
        using var model = TransformerModel.LoadFromSafetensors(sf, cfg);

        using var simple = new SimpleKvCache(Layers, KvHeads, HeadDim, MaxSeq);
        int[] expected = Greedy(model, simple, prompt);

        using var factory = new PagedKvCacheFactory(Layers, KvHeads, HeadDim, blockSize: 16, maxTotalTokens: 512, elementType);
        using var blocker = factory.Create(MaxSeq);
        using var paged = factory.Create(MaxSeq);
        int[] actual = Greedy(model, paged, prompt, blocker);

        Assert.Equal(expected, actual);
        Assert.Equal(0, paged.StagingBytes);
        Assert.Equal(elementType, factory.Pool.ElementType);
    }

    /// <summary>Prefill, then greedy decode; interleaves a block of <paramref name="blocker"/> so pages are fragmented.</summary>
    private static unsafe int[] Greedy(TransformerModel model, IKvCache cache, int[] prompt, PagedKvCache? blocker = null)
    {
        var tokens = new List<int>();
        int[] positions = Enumerable.Range(0, prompt.Length).ToArray();
        int next;
        using (var logits = model.Forward(prompt, positions, deviceId: -1, cache))
            next = ArgMaxLast(logits);
        for (int step = 0; step < Steps; step++)
        {
            tokens.Add(next);
            int pos = prompt.Length + step;
            if (blocker is not null && pos % 16 == 0)
                OccupyBlock(blocker, pos);
            using var logits = model.Forward([next], [pos], deviceId: -1, cache);
            next = ArgMaxLast(logits);
        }

        return tokens.ToArray();
    }

    private static unsafe void OccupyBlock(PagedKvCache blocker, int pos)
    {
        float* row = stackalloc float[KvHeads * HeadDim];
        new Span<float>(row, KvHeads * HeadDim).Clear();
        var r = new TensorRef(1, KvHeads * HeadDim, DType.Float32, -1, (nint)row);
        for (int layer = 0; layer < Layers; layer++)
            blocker.Update(r, r, [pos], layer);
    }

    private static unsafe int ArgMaxLast(ITensor logits)
    {
        int vocab = logits.Shape[1];
        var last = new ReadOnlySpan<float>((float*)logits.DataPointer + (logits.Shape[0] - 1) * vocab, vocab);
        return System.Numerics.Tensors.TensorPrimitives.IndexOfMax(last);
    }

    private static ModelConfig BuildConfig() => new()
    {
        Architecture = Architecture.Llama,
        VocabSize = Vocab,
        HiddenSize = Hidden,
        IntermediateSize = Intermediate,
        NumLayers = Layers,
        NumAttentionHeads = Heads,
        NumKvHeads = KvHeads,
        HeadDim = HeadDim,
        MaxSequenceLength = MaxSeq,
        NormEpsilon = 1e-5f,
        RoPEConfig = new RoPEConfig(Theta: 10000f, DimensionCount: HeadDim, Type: RoPEType.Norm),
    };

    private static void WriteFixture(string path)
    {
        var b = new SafetensorsFixtureBuilder();
        Add(b, "model.embed_tokens.weight", [Vocab, Hidden], 1.0f, 0);
        Add(b, "model.norm.weight", [Hidden], 0.05f, 1, center: 1.0f);
        Add(b, "lm_head.weight", [Vocab, Hidden], 0.2f, 2);
        for (int i = 0; i < Layers; i++)
        {
            int s = 10 * (i + 1);
            string p = $"model.layers.{i}";
            Add(b, $"{p}.input_layernorm.weight", [Hidden], 0.05f, s, center: 1.0f);
            Add(b, $"{p}.post_attention_layernorm.weight", [Hidden], 0.05f, s + 1, center: 1.0f);
            Add(b, $"{p}.self_attn.q_proj.weight", [Heads * HeadDim, Hidden], 0.3f, s + 2);
            Add(b, $"{p}.self_attn.k_proj.weight", [KvHeads * HeadDim, Hidden], 0.3f, s + 3);
            Add(b, $"{p}.self_attn.v_proj.weight", [KvHeads * HeadDim, Hidden], 0.3f, s + 4);
            Add(b, $"{p}.self_attn.o_proj.weight", [Hidden, Heads * HeadDim], 0.1f, s + 5);
            Add(b, $"{p}.mlp.gate_proj.weight", [Intermediate, Hidden], 0.1f, s + 6);
            Add(b, $"{p}.mlp.up_proj.weight", [Intermediate, Hidden], 0.1f, s + 7);
            Add(b, $"{p}.mlp.down_proj.weight", [Hidden, Intermediate], 0.1f, s + 8);
        }

        b.WriteTo(path);
    }

    /// <summary>Deterministic pseudo-random weights (hash-based so rows are decorrelated).</summary>
    private static void Add(SafetensorsFixtureBuilder b, string name, int[] shape, float amplitude, int seed, float center = 0f)
    {
        int n = 1;
        foreach (int d in shape) n *= d;
        var rng = new Random(1000 + seed);
        float[] values = new float[n];
        for (int i = 0; i < n; i++)
            values[i] = center + amplitude * (float)(rng.NextDouble() * 2 - 1);
        b.AddFloat32(name, shape, values);
    }
}
