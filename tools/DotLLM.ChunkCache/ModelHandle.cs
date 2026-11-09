using System.Diagnostics;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Store;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;

namespace DotLLM.ChunkCache;

/// <summary>
/// A GGUF model's identity (content-based model and tokenizer ids, fingerprint, store key function) and tokenizer; the CPU
/// weights are loaded only when a verb encodes or runs forwards (<see cref="Model"/>).
/// </summary>
/// <remarks>
/// The model id is <see cref="GgufIdentity.ModelId"/>: a SHA-256 over the header, the tensor table and the first and last
/// 4 KiB of every tensor (or of every byte with <c>--strict-model-id</c>), never the file name, so a renamed file keeps its
/// chunks and a re-quantised one does not. The store keys use <see cref="StoredContextAssembler.ForModel(DotLLM.Core.Models.ModelConfig, string, string, string)"/>, which refuses a
/// model whose configuration does not reproduce the fingerprint.
/// </remarks>
internal sealed class ModelHandle : IDisposable
{
    private readonly GgufFile _gguf;
    private readonly int _threads;
    private IModel? _model;

    private ModelHandle(string path, GgufFile gguf, ModelConfig config, ITokenizer tokenizer, string modelId, string tokenizerId, string quant, int threads)
    {
        Path = path;
        _gguf = gguf;
        _threads = threads;
        Config = config;
        Tokenizer = tokenizer;
        ModelId = modelId;
        TokenizerId = tokenizerId;
        Quant = quant;
        Assembler = StoredContextAssembler.ForModel(config, modelId, quant, tokenizerId);
        KeyOf = ids => Assembler.KeyOf(config, ids);
        Producer = KeyOf([]).Hex;
        Geometry = KvGeometry.FromConfig(config, compactSlidingWindows: false);
        long bytes = 0;
        for (int l = 0; l < Geometry.LayerCount; l++) bytes += 2L * Geometry.KvStrideOf(l) * sizeof(float);
        KvBytesPerToken = bytes;
    }

    /// <summary>The GGUF path.</summary>
    public string Path { get; }

    /// <summary>Model configuration.</summary>
    public ModelConfig Config { get; }

    /// <summary>The model's tokenizer.</summary>
    public ITokenizer Tokenizer { get; }

    /// <summary>Content-based model id (a server-backed run adopts the server's, see <see cref="AdoptServerIdentity"/>).</summary>
    public string ModelId { get; private set; }

    /// <summary>Content-based tokenizer id.</summary>
    public string TokenizerId { get; }

    /// <summary>Quantisation label (<c>general.file_type</c>).</summary>
    public string Quant { get; }

    /// <summary>The store assembler for this identity (keys, hit checks, encode of misses).</summary>
    public StoredContextAssembler Assembler { get; }

    /// <summary>Model fingerprint hashed into every key.</summary>
    public string Fingerprint => Assembler.ModelFingerprint;

    /// <summary>Store key of a chunk's token ids.</summary>
    public ChunkKeyFunction KeyOf { get; }

    /// <summary>The producer identity of this model's indexes (key of the empty sequence).</summary>
    public string Producer { get; }

    /// <summary>KV geometry (no compact sliding windows: the store keeps every row).</summary>
    public KvGeometry Geometry { get; }

    /// <summary>F32 K+V bytes per token over all layers (what the store writes per token).</summary>
    public long KvBytesPerToken { get; }

    /// <summary>F32 K+V bytes per token per layer (8,192 for Qwen3-4B: 8 KV heads × 128 × 2 × 4 B).</summary>
    public double KvBytesPerTokenPerLayer => (double)KvBytesPerToken / Geometry.LayerCount;

    /// <summary>The CPU model (weights loaded on first use).</summary>
    public IModel Model => _model ??= ModelLoader.CreateCpuModelFromGguf(_gguf, Config, new ThreadingConfig(_threads));

    /// <summary>Opens <paramref name="path"/> and computes its identity.</summary>
    public static ModelHandle Open(CliContext ctx, string path, bool fullHash, int threads)
    {
        string full = ctx.FullPath(path);
        if (!File.Exists(full)) throw new FileNotFoundException($"Model file not found: {full}", full);
        var sw = Stopwatch.StartNew();
        GgufFile gguf = GgufFile.Open(full);
        try
        {
            ModelConfig config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            ITokenizer tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
            string modelId = GgufIdentity.ModelId(gguf, fullHash);
            string tokenizerId = GgufIdentity.TokenizerId(gguf.Metadata);
            string quant = gguf.Metadata.TryGetValue("general.file_type", out var ft)
                ? "ftype-" + Convert.ToString(ft.Value, System.Globalization.CultureInfo.InvariantCulture) : "unknown";
            var handle = new ModelHandle(full, gguf, config, tokenizer, modelId, tokenizerId, quant, threads);
            ctx.Info($"model {System.IO.Path.GetFileName(full)}: {config.Architecture}, {config.NumLayers} layers, {handle.KvBytesPerTokenPerLayer:0} B/token/layer; "
                + $"id {Short(modelId)}{(fullHash ? " (full hash)" : "")}; fingerprint {Short(handle.Fingerprint)} [{sw.Elapsed.TotalSeconds:F1} s]");
            return handle;
        }
        catch
        {
            gguf.Dispose();
            throw;
        }
    }

    /// <summary>
    /// A server-backed run reads and writes the index of the server's own identity: the file name carries the server's
    /// <c>model.id</c> and the first 8 digits of its <c>model.producer</c>. The producer must be the one this GGUF computes (same
    /// weights, tokenizer, RoPE, backend and dtype): the index's chunk keys come from this handle, so an index under another
    /// producer's name would name keys the server never stores.
    /// </summary>
    public void AdoptServerIdentity(ServerIdentity server, string serverUrl)
    {
        if (string.IsNullOrEmpty(server.Producer) || string.IsNullOrEmpty(server.ModelId))
            throw new InvalidOperationException($"{serverUrl} reported no model identity (model.id / model.producer); is its model supported for chunk reuse?");
        if (!string.Equals(server.Producer, Producer, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException($"The server's chunk-key producer ({Short(server.Producer)}) is not this model file's ({Short(Producer)}): "
                + "another GGUF, quantisation, tokenizer, backend or K/V dtype. Use the GGUF the server serves (the keys of an index built here would never hit).");
        ModelId = server.ModelId;
    }

    /// <summary>A new exact host-F32 cache for <paramref name="capacity"/> positions.</summary>
    public SimpleKvCache NewCache(int capacity) => new(Geometry, capacity);

    /// <summary>The first 12 characters after any <c>prefix:</c> (for display).</summary>
    public static string Short(string id)
    {
        int colon = id.LastIndexOf(':');
        string s = colon >= 0 ? id[(colon + 1)..] : id;
        return s.Length <= 12 ? s : s[..12];
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _model?.Dispose();
        _gguf.Dispose();
    }
}
