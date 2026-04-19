using System.Buffers.Binary;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Tokenizers.Hf;

namespace DotLLM.Models;

/// <summary>
/// Convenience helper encapsulating the format-open → config-extract → model-load
/// pattern. Single dispatch point for all architecture creation from either
/// GGUF or HuggingFace safetensors on-disk layouts.
/// </summary>
public static class ModelLoader
{
    /// <summary>
    /// Loads a model from a GGUF file path. Opens the file, extracts config,
    /// and creates the appropriate model instance.
    /// </summary>
    /// <param name="path">Path to the GGUF model file.</param>
    /// <param name="threading">Threading configuration. Null defaults to single-threaded.</param>
    /// <returns>The loaded model, GGUF file handle, and model configuration.</returns>
    public static (IModel Model, GgufFile Gguf, ModelConfig Config) LoadFromGguf(
        string path, ThreadingConfig? threading = null)
    {
        var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        IModel model = config.Architecture switch
        {
            Architecture.NemotronH => NemotronHTransformerModel.LoadFromGguf(gguf, config),
            _ => TransformerModel.LoadFromGguf(gguf, config, threading ?? ThreadingConfig.SingleThreaded),
        };
        return (model, gguf, config);
    }

    /// <summary>
    /// Loads a model from a HuggingFace safetensors checkpoint. Accepts any of
    /// three forms for <paramref name="safetensorsPath"/>:
    /// <list type="bullet">
    /// <item>A <c>*.safetensors</c> file — single-shard ingest, as before.</item>
    /// <item>A <c>model.safetensors.index.json</c> file — multi-shard ingest driven
    /// by the index's <c>weight_map</c>.</item>
    /// <item>A directory — probed for <c>model.safetensors.index.json</c> first
    /// (multi-shard), then a single <c>*.safetensors</c> (single-shard).</item>
    /// </list>
    /// In every case the directory containing the weights is scanned for a
    /// <c>config.json</c> which drives both architecture dispatch and
    /// <see cref="ModelConfig"/> population.
    /// </summary>
    /// <param name="safetensorsPath">
    /// A <c>*.safetensors</c> file, a <c>model.safetensors.index.json</c>, or a
    /// directory containing one of the above.
    /// </param>
    /// <param name="threading">Threading configuration. Null defaults to single-threaded.</param>
    /// <returns>The loaded model, safetensors source, and model configuration.</returns>
    /// <exception cref="FileNotFoundException">
    /// The path or an accompanying <c>config.json</c> is missing.
    /// </exception>
    /// <exception cref="InvalidDataException">
    /// <c>config.json</c> is malformed or declares an unsupported architecture,
    /// or a multi-shard index references files missing on disk.
    /// </exception>
    public static (IModel Model, ISafetensorsTensorSource Safetensors, ModelConfig Config) LoadFromSafetensors(
        string safetensorsPath, ThreadingConfig? threading = null)
    {
        ArgumentNullException.ThrowIfNull(safetensorsPath);

        (ISafetensorsTensorSource source, string weightsDir) = OpenSafetensorsSource(safetensorsPath);

        try
        {
            string configPath = Path.Combine(weightsDir, "config.json");
            if (!File.Exists(configPath))
                throw new FileNotFoundException(
                    $"Expected HuggingFace config.json next to the safetensors weights, but '{configPath}' does not exist.",
                    configPath);

            string configJson = File.ReadAllText(configPath);
            using var doc = JsonDocument.Parse(configJson);
            Architecture arch;
            try
            {
                arch = HfConfigExtractor.ResolveArchitecture(doc.RootElement);
            }
            catch (InvalidDataException)
            {
                // Fall through to Mamba-3 probe: model_type=mamba3 is handled by a
                // dedicated extractor, not HfConfigExtractor.
                string? modelType = doc.RootElement.TryGetProperty("model_type", out var mt)
                                    && mt.ValueKind == JsonValueKind.String
                    ? mt.GetString()
                    : null;
                if (!string.Equals(modelType, "mamba3", StringComparison.Ordinal))
                    throw;
                arch = Architecture.Mamba3;
            }

            ModelConfig config = arch switch
            {
                Architecture.Mamba3 => Mamba3ConfigExtractor.Extract(doc.RootElement),
                _ => HfConfigExtractor.Extract(doc.RootElement),
            };

            IModel model = config.Architecture switch
            {
                Architecture.Llama or Architecture.Mistral or Architecture.Phi or Architecture.Qwen
                    or Architecture.Mixtral
                    => TransformerModel.LoadFromSafetensors(source, config, threading ?? ThreadingConfig.SingleThreaded),
                Architecture.Mamba3
                    => Mamba3TransformerModel.LoadFromSafetensors(source, config),
                _ => throw new NotSupportedException(
                    $"Safetensors loader does not yet dispatch architecture {config.Architecture}. "
                    + "Supported today: Llama, Mistral, Phi, Qwen, Mixtral, Mamba3."),
            };

            return (model, source, config);
        }
        catch
        {
            source.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Resolves <paramref name="path"/> to an opened
    /// <see cref="ISafetensorsTensorSource"/> (single- or multi-shard) and the
    /// directory that contains it. Extracted from
    /// <see cref="LoadFromSafetensors"/> so the auto-detection logic is
    /// testable in isolation.
    /// </summary>
    private static (ISafetensorsTensorSource Source, string WeightsDir) OpenSafetensorsSource(string path)
    {
        // Case 1: directory — probe for index.json first, single shard second.
        if (Directory.Exists(path))
        {
            string indexPath = Path.Combine(path, "model.safetensors.index.json");
            if (File.Exists(indexPath))
                return (MultiShardSafetensorsFile.Open(indexPath), path);

            string[] candidates = Directory.GetFiles(path, "*.safetensors", SearchOption.TopDirectoryOnly);
            // Filter out any pre-shard artefact we don't want to treat as single-shard.
            // (e.g. stale 'model.safetensors' sitting next to an incomplete shard set.)
            if (candidates.Length == 1)
                return (SafetensorsFile.Open(candidates[0]), path);
            if (candidates.Length == 0)
                throw new FileNotFoundException(
                    $"Directory '{path}' contains no *.safetensors and no model.safetensors.index.json.",
                    indexPath);
            // Multiple .safetensors files but no index.json is unusual — we reject
            // rather than guess, because every candidate is a plausible single-shard
            // root and picking the wrong one would silently load partial weights.
            throw new InvalidDataException(
                $"Directory '{path}' contains {candidates.Length} *.safetensors files "
                + "but no model.safetensors.index.json to arbitrate between them.");
        }

        if (!File.Exists(path))
            throw new FileNotFoundException($"Safetensors path not found: {path}", path);

        // Case 2: an index.json file.
        if (path.EndsWith(".safetensors.index.json", StringComparison.OrdinalIgnoreCase)
            || string.Equals(Path.GetFileName(path), "model.safetensors.index.json", StringComparison.OrdinalIgnoreCase))
        {
            string dir = Path.GetDirectoryName(path)
                         ?? throw new InvalidDataException(
                             $"Could not determine parent directory of index file '{path}'.");
            return (MultiShardSafetensorsFile.Open(path), dir);
        }

        // Case 3: a single *.safetensors file. If a sibling index.json exists,
        // prefer the multi-shard path — it is authoritative and the caller just
        // happens to have pointed at one shard.
        string? weightsDir = Path.GetDirectoryName(path);
        if (string.IsNullOrEmpty(weightsDir))
            throw new InvalidDataException(
                $"Could not determine directory of safetensors path '{path}'.");
        string siblingIndex = Path.Combine(weightsDir, "model.safetensors.index.json");
        if (File.Exists(siblingIndex))
            return (MultiShardSafetensorsFile.Open(siblingIndex), weightsDir);

        return (SafetensorsFile.Open(path), weightsDir);
    }

    /// <summary>
    /// Top-level dispatcher that auto-detects GGUF vs safetensors by file
    /// extension, falling back to magic-byte probing when the extension is
    /// ambiguous. Returns an opaque file handle (either
    /// <see cref="GgufFile"/> or <see cref="SafetensorsFile"/>) plus the
    /// loaded model and its config.
    /// </summary>
    /// <remarks>
    /// Callers that need to force a specific format should call
    /// <see cref="LoadFromGguf"/> or <see cref="LoadFromSafetensors"/>
    /// directly — this entry point exists only as a convenience for
    /// generic "given a path, load a model" code paths.
    /// </remarks>
    public static (IModel Model, IDisposable File, ModelConfig Config) Load(
        string path, ThreadingConfig? threading = null)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Model file not found: {path}", path);

        LoadFormat format = DetectFormat(path);
        switch (format)
        {
            case LoadFormat.Gguf:
            {
                var (model, gguf, config) = LoadFromGguf(path, threading);
                return (model, gguf, config);
            }
            case LoadFormat.Safetensors:
            {
                var (model, st, config) = LoadFromSafetensors(path, threading);
                return (model, st, config);
            }
            default:
                throw new InvalidDataException(
                    $"Cannot determine model format for '{path}'. Expected .gguf or .safetensors.");
        }
    }

    /// <summary>
    /// Loads a HuggingFace <c>tokenizer.json</c> from a checkpoint directory
    /// and returns it as an <see cref="ITokenizer"/>. Accepts either the
    /// checkpoint directory path or a path to a file inside it (e.g. the
    /// <c>model.safetensors</c> path).
    /// </summary>
    /// <param name="directoryOrFilePath">
    /// A checkpoint directory or a path to a file in that directory. The
    /// parent directory is scanned for <c>tokenizer.json</c>.
    /// </param>
    /// <returns>
    /// A ready-to-use tokenizer, or <see langword="null"/> when the directory
    /// contains no <c>tokenizer.json</c>.
    /// </returns>
    /// <remarks>
    /// Pairs with <see cref="LoadFromSafetensors"/> — HF checkpoints ship the
    /// tokenizer alongside the weights, but we surface it via a separate call
    /// so existing <c>(IModel, IDisposable, ModelConfig)</c> tuple contracts
    /// do not change. Callers that want weights and tokenizer in one call
    /// invoke both and compose the result.
    /// </remarks>
    public static ITokenizer? LoadTokenizerFromHfDirectory(string directoryOrFilePath)
    {
        ArgumentNullException.ThrowIfNull(directoryOrFilePath);
        string dir = Directory.Exists(directoryOrFilePath)
            ? directoryOrFilePath
            : Path.GetDirectoryName(directoryOrFilePath) ?? directoryOrFilePath;
        return HfBpeTokenizerFactory.TryLoadFromDirectory(dir);
    }

    private enum LoadFormat { Unknown, Gguf, Safetensors }

    /// <summary>
    /// Detects the on-disk format of a model file. Extension check first
    /// (fast and unambiguous in practice), then a magic-byte probe for
    /// corner cases like extensionless files in a test harness.
    /// </summary>
    private static LoadFormat DetectFormat(string path)
    {
        string ext = Path.GetExtension(path).ToLowerInvariant();
        if (ext == ".gguf") return LoadFormat.Gguf;
        if (ext == ".safetensors") return LoadFormat.Safetensors;

        // Magic-byte sniff: GGUF starts with ASCII "GGUF" (0x47 0x47 0x55 0x46).
        // Safetensors starts with an 8-byte LE u64 header length — not a magic
        // sequence, but we can sanity-check that the first 8 bytes would be
        // a plausible header length (small but not tiny, not exceeding file size).
        try
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            Span<byte> buf = stackalloc byte[8];
            int read = fs.Read(buf);
            if (read < 4) return LoadFormat.Unknown;
            if (buf[0] == 0x47 && buf[1] == 0x47 && buf[2] == 0x55 && buf[3] == 0x46)
                return LoadFormat.Gguf;
            if (read == 8)
            {
                ulong headerLen = BinaryPrimitives.ReadUInt64LittleEndian(buf);
                long fileLen = fs.Length;
                // Plausibility: 2 <= headerLen <= fileLen - 8 and headerLen doesn't
                // exceed a few MB (HF headers top out around low MB in practice).
                if (headerLen >= 2 && (long)headerLen + 8 <= fileLen && headerLen < 64 * 1024 * 1024)
                    return LoadFormat.Safetensors;
            }
        }
        catch
        {
            // Fall through to Unknown.
        }
        return LoadFormat.Unknown;
    }
}
