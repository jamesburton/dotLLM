using System.Diagnostics;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Backends;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Vulkan;

namespace ChunkKvLab;

/// <summary>A model on the selected backend (CPU or Vulkan), its tokenizer and the cache/forward helpers the lab commands share.</summary>
internal sealed unsafe class LabModel : IDisposable
{
    private readonly GgufFile _gguf;
    private readonly VulkanDevice? _device;

    private LabModel(GgufFile gguf, ModelConfig config, IModel model, ITokenizer tokenizer, string fingerprint, IChunkBackend backend,
        VulkanDevice? device)
    {
        _gguf = gguf;
        _device = device;
        Config = config;
        Model = model;
        Tokenizer = tokenizer;
        Fingerprint = fingerprint;
        Backend = backend;
        DeviceId = device is null ? -1 : 0;
        Geometry = KvGeometry.FromConfig(config, compactSlidingWindows: false);
        long bytes = 0;
        for (int l = 0; l < Geometry.LayerCount; l++) bytes += 2L * Geometry.KvStrideOf(l) * sizeof(float);
        KvBytesPerToken = bytes;
    }

    public ModelConfig Config { get; }
    public IModel Model { get; }
    public ITokenizer Tokenizer { get; }
    public string Fingerprint { get; }
    public KvGeometry Geometry { get; }

    /// <summary>The chunk backend matching <see cref="Model"/> (<c>--backend cpu|vulkan</c>).</summary>
    public IChunkBackend Backend { get; }

    /// <summary>Vulkan device name, or "cpu".</summary>
    public string DeviceName => _device?.DeviceName ?? "cpu";

    /// <summary>Forward device id: -1 (CPU) or 0 (Vulkan).</summary>
    public int DeviceId { get; }

    /// <summary>Backend chosen with the global <c>--backend</c> option (set by <c>Program</c> before a command runs).</summary>
    public static string SelectedBackend { get; set; } = ChunkBackendTags.Cpu;

    /// <summary>F32 K+V bytes per token over all layers.</summary>
    public long KvBytesPerToken { get; }

    /// <summary>Loads <paramref name="path"/> on <see cref="SelectedBackend"/>.</summary>
    public static LabModel Load(string path, int threads) => Load(path, threads, SelectedBackend);

    /// <summary>
    /// Loads <paramref name="path"/> on <paramref name="backend"/>. Vulkan: the model and the chunk backend share one device (caches the
    /// backend creates are bound by the model's forward), and a model whose keys plain RoPE cannot re-rotate is refused before upload.
    /// </summary>
    public static LabModel Load(string path, int threads, string backend)
    {
        var sw = Stopwatch.StartNew();
        var gguf = GgufFile.Open(path);
        VulkanDevice? device = null;
        IModel? model = null;
        IChunkBackend? chunkBackend = null;
        try
        {
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            ITokenizer tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
            switch (backend)
            {
                case ChunkBackendTags.Cpu:
                    model = ModelLoader.CreateCpuModelFromGguf(gguf, config, new ThreadingConfig(threads));
                    chunkBackend = new CpuChunkBackend(config);
                    break;
                case ChunkBackendTags.Vulkan:
                {
                    var features = ChunkModelFeatures.FromGguf(gguf);
                    VulkanChunkBackend.CheckModelSupported(config, features);
                    string spvDir = VulkanModelLoader.ResolveSpvDir();
                    device = VulkanDevice.Create();
                    model = VulkanTransformerModel.LoadFromGguf(device, gguf, config, spvDir);
                    chunkBackend = new VulkanChunkBackend(device, spvDir, config, features);
                    break;
                }
                default:
                    throw new ArgumentException($"--backend must be cpu or vulkan; got '{backend}'.");
            }
            string file = Path.GetFileNameWithoutExtension(path);
            string quant = file[(file.LastIndexOf('-') + 1)..];
            string fp = ChunkFingerprint.Compute(config, file, quant, "gguf-embedded:" + file, chunkBackend.Tag);
            Console.WriteLine($"loaded {file} ({config.Architecture}, {config.NumLayers} layers, kv heads {config.NumKvHeads}, head dim {config.HeadDim}, "
                + $"vocab {config.VocabSize}) on {chunkBackend.Tag} in {sw.Elapsed.TotalSeconds:F1} s; engine {ChunkFingerprint.EngineVersion}");
            return new LabModel(gguf, config, model, tokenizer, fp, chunkBackend, device);
        }
        catch
        {
            // Release in reverse order of creation: backend kernel, model (device-resident weights), device, mapped file.
            (chunkBackend as IDisposable)?.Dispose();
            model?.Dispose();
            device?.Dispose();
            gguf.Dispose();
            throw;
        }
    }

    /// <summary>A host F32 cache for the runners that still read <see cref="SimpleKvCache"/> directly (CPU only until Phase G Task 3).</summary>
    public SimpleKvCache NewCache(int capacity)
    {
        if (Backend.Tag != ChunkBackendTags.Cpu)
            throw new NotSupportedException(
                $"This command still runs on SimpleKvCache; --backend {Backend.Tag} wires the model and chunk backend only (Phase G Task 3 ports the runners).");
        return new(Geometry, capacity);
    }

    /// <summary>An empty exact-F32 cache of the selected backend.</summary>
    public IKvCache NewBackendCache(int capacity) => Backend.CreateCache(capacity);

    /// <summary>Prefills <paramref name="ids"/> at positions <paramref name="firstPosition"/>.. in blocks of <paramref name="block"/> tokens.</summary>
    public void Prefill(IKvCache cache, ReadOnlySpan<int> ids, int firstPosition, int block)
    {
        for (int off = 0; off < ids.Length; off += block)
        {
            int n = Math.Min(block, ids.Length - off);
            int[] positions = new int[n];
            for (int i = 0; i < n; i++) positions[i] = firstPosition + off + i;
            using ITensor _ = Model.Forward(ids.Slice(off, n), positions, DeviceId, cache, lastTokenLogitsOnly: true);
        }
    }

    /// <summary>
    /// Largest batch <see cref="ForwardLogits"/> returns all-row logits for. All-row logits cost <c>rows × vocab × 4</c> bytes per call
    /// (Qwen3 vocab 151,936: ~0.6 GB at 1,024 rows, ~20 GB at 32K), and on Vulkan <c>TrySetAllRowLogitsLimit</c> sizes a device buffer to
    /// the largest request and never shrinks it, after which every prefill block up to that size also runs the full LM head.
    /// Runners score long spans in blocks of at most this many rows (prefill the rest with <see cref="Prefill"/>), never whole contexts.
    /// </summary>
    public const int MaxAllRowLogitsRows = 1024;

    /// <summary>
    /// Forwards <paramref name="ids"/> at <paramref name="firstPosition"/>.. and copies the logits (all rows or the last). All-row requests
    /// are limited to <see cref="MaxAllRowLogitsRows"/> rows.
    /// </summary>
    public float[] ForwardLogits(IKvCache cache, ReadOnlySpan<int> ids, int firstPosition, bool allRows)
    {
        if (allRows && ids.Length > MaxAllRowLogitsRows)
            throw new ArgumentException(
                $"All-row logits for {ids.Length} rows exceed the lab limit {MaxAllRowLogitsRows} ({(long)ids.Length * Config.VocabSize * 4 / (1 << 20)} MiB); "
                + "request them in bounded blocks.", nameof(ids));
        int[] positions = new int[ids.Length];
        for (int i = 0; i < ids.Length; i++) positions[i] = firstPosition + i;
        // The Vulkan model returns one logit row per position only up to its all-row limit (default 1): widen it (grow-only).
        if (allRows && ids.Length > 1 && Backend.Tag != ChunkBackendTags.Cpu && !Model.TrySetAllRowLogitsLimit(ids.Length))
            throw new NotSupportedException($"The {Backend.Tag} model cannot return all-row logits for {ids.Length} rows.");
        using ITensor logits = allRows
            ? Model.Forward(ids, positions, DeviceId, cache)
            : Model.Forward(ids, positions, DeviceId, cache, lastTokenLogitsOnly: true);
        int rows = logits.Shape[0];
        int vocab = logits.Shape[1];
        if (allRows && rows != ids.Length)
            throw new InvalidOperationException($"Asked for {ids.Length} logit rows; the model returned {rows}.");
        int firstRow = allRows ? 0 : rows - 1;
        int count = allRows ? rows : 1;
        return new ReadOnlySpan<float>((float*)logits.DataPointer + (long)firstRow * vocab, count * vocab).ToArray();
    }

    public void Dispose()
    {
        (Backend as IDisposable)?.Dispose();
        Model.Dispose();
        _device?.Dispose();
        _gguf.Dispose();
    }
}
