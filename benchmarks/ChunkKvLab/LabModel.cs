using System.Diagnostics;
using DotLLM.ChunkReuse;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;

namespace ChunkKvLab;

/// <summary>A CPU model, its tokenizer and the cache/forward helpers the lab commands share.</summary>
internal sealed unsafe class LabModel : IDisposable
{
    private readonly GgufFile _gguf;

    private LabModel(GgufFile gguf, ModelConfig config, IModel model, ITokenizer tokenizer, string fingerprint, string modelId, string tokenizerId, string quant)
    {
        ModelId = modelId;
        TokenizerId = tokenizerId;
        Quant = quant;
        _gguf = gguf;
        Config = config;
        Model = model;
        Tokenizer = tokenizer;
        Fingerprint = fingerprint;
        Geometry = KvGeometry.FromConfig(config, compactSlidingWindows: false);
        long bytes = 0;
        for (int l = 0; l < Geometry.LayerCount; l++) bytes += 2L * Geometry.KvStrideOf(l) * sizeof(float);
        KvBytesPerToken = bytes;
    }

    public ModelConfig Config { get; }
    public IModel Model { get; }
    public ITokenizer Tokenizer { get; }
    public string Fingerprint { get; }

    /// <summary>Content-based model id (<see cref="GgufIdentity.ModelId"/>), not the file name.</summary>
    public string ModelId { get; }

    /// <summary>Quantization label hashed into the fingerprint (<c>general.file_type</c>).</summary>
    public string Quant { get; }

    /// <summary>Content-based tokenizer id (<see cref="GgufIdentity.TokenizerId"/>), not the file name.</summary>
    public string TokenizerId { get; }
    public KvGeometry Geometry { get; }

    /// <summary>F32 K+V bytes per token over all layers.</summary>
    public long KvBytesPerToken { get; }

    public static LabModel Load(string path, int threads)
    {
        var sw = Stopwatch.StartNew();
        var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        ITokenizer tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        IModel model = ModelLoader.CreateCpuModelFromGguf(gguf, config, new ThreadingConfig(threads));
        string file = Path.GetFileNameWithoutExtension(path);
        // Content-based identities: a renamed file keeps its chunks, a re-quantized or edited file with the same name does not.
        string modelId = GgufIdentity.ModelId(gguf);
        string tokenizerId = GgufIdentity.TokenizerId(gguf.Metadata);
        string quant = gguf.Metadata.TryGetValue("general.file_type", out var ft)
            ? "ftype-" + Convert.ToString(ft.Value, System.Globalization.CultureInfo.InvariantCulture) : "unknown";
        string fp = ChunkFingerprint.Compute(config, modelId, quant, tokenizerId);
        Console.WriteLine($"loaded {file} ({config.Architecture}, {config.NumLayers} layers, kv heads {config.NumKvHeads}, head dim {config.HeadDim}, "
            + $"vocab {config.VocabSize}) in {sw.Elapsed.TotalSeconds:F1} s; engine {ChunkFingerprint.EngineVersion}");
        Console.WriteLine($"model id {modelId}; tokenizer id {tokenizerId}; quant {quant}; fingerprint {fp}");
        return new LabModel(gguf, config, model, tokenizer, fp, modelId, tokenizerId, quant);
    }

    public SimpleKvCache NewCache(int capacity) => new(Geometry, capacity);

    /// <summary>Prefills <paramref name="ids"/> at positions <paramref name="firstPosition"/>.. in blocks of <paramref name="block"/> tokens.</summary>
    public void Prefill(IKvCache cache, ReadOnlySpan<int> ids, int firstPosition, int block)
    {
        for (int off = 0; off < ids.Length; off += block)
        {
            int n = Math.Min(block, ids.Length - off);
            int[] positions = new int[n];
            for (int i = 0; i < n; i++) positions[i] = firstPosition + off + i;
            using ITensor _ = Model.Forward(ids.Slice(off, n), positions, -1, cache, lastTokenLogitsOnly: true);
        }
    }

    /// <summary>Forwards <paramref name="ids"/> at <paramref name="firstPosition"/>.. and copies the logits (all rows or the last).</summary>
    public float[] ForwardLogits(IKvCache cache, ReadOnlySpan<int> ids, int firstPosition, bool allRows)
    {
        int[] positions = new int[ids.Length];
        for (int i = 0; i < ids.Length; i++) positions[i] = firstPosition + i;
        using ITensor logits = allRows
            ? Model.Forward(ids, positions, -1, cache)
            : Model.Forward(ids, positions, -1, cache, lastTokenLogitsOnly: true);
        int rows = logits.Shape[0];
        int vocab = logits.Shape[1];
        int firstRow = allRows ? 0 : rows - 1;
        int count = allRows ? rows : 1;
        return new ReadOnlySpan<float>((float*)logits.DataPointer + (long)firstRow * vocab, count * vocab).ToArray();
    }

    public void Dispose()
    {
        Model.Dispose();
        _gguf.Dispose();
    }
}
