// Loader glue for issue #266 Phase B (load path) on top of the Phase A codec
// decoder ported from SyzygyResearch/Mach-1-Additive-35B's decode.py
// (Apache License 2.0): https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B/blob/main/decode.py
// Apache-2.0 is one-way compatible with dotLLM's GPLv3 (see CLAUDE.md).
using System.Runtime.InteropServices;
using System.Text.Json;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Models.Quantization.Mach1;

/// <summary>
/// Orchestrates reading a Mach-1 additive-codec HF repo's <c>packed/</c>
/// directory tree (no <c>model.safetensors.index.json</c>; composite
/// <c>|</c>-delimited tensor keys) and decoding its four tiers (experts, NE
/// spine, LM head, embeddings) to dense fp32 via the Phase A decoders.
/// </summary>
/// <remarks>
/// <para>
/// <b>Format-churn awareness.</b> Per issue #266's explicit risk list,
/// <c>codec.json</c>'s <c>container</c> field is versioned and the vendor
/// repo has already changed shape more than once. This type does NOT assume
/// a fixed layout: expert-tier container kind is detected per-file via
/// <see cref="Mach1ExpertContainer.Detect"/> (v3t vs v2/manifest), NE-tier
/// codec is checked via <see cref="Mach1NeSpineDecoder.IsCanonRhtTrellisContainer"/>,
/// and an optional <c>__zsc__</c> zstd sidecar (mentioned in decode.py's
/// <c>_read_safetensors_np</c> as present on some v3 files) is detected and
/// rejected with a clear <see cref="NotSupportedException"/> rather than
/// silently mis-reading the file — none of the files in the real fixture
/// (<c>SyzygyResearch/Mach-1-Additive-35B</c> as of 2026-08) carry it, and
/// zstd decompression is not wired into this port.
/// </para>
/// <para>
/// <b>Memory discipline.</b> Per-layer files are opened lazily and only one
/// experts file + one NE file are held open at a time (closing the previous
/// layer's file when a caller moves to a new layer) — callers driving a
/// layer-by-layer load (e.g. <c>Qwen3MoeHybridTransformerModel.LoadFromMach1Packed</c>)
/// should decode and consume one layer's tensors before advancing, rather
/// than holding many decoded dense layers resident simultaneously (256
/// experts x 3 projections x 40 layers dense-decoded to fp32 is ~120 GB —
/// see issue #266's "Practical constraints" section).
/// </para>
/// </remarks>
public sealed class Mach1PackedCheckpoint : IDisposable
{
    private const string ExpectedExpertContainer = "trained_susv_wave_gamma_chunked_v1";

    private readonly string _root;
    private readonly Mach1CbParams _expertsCb;
    private readonly SafetensorsFile _expertsCodebook;
    private readonly float[] _expertsSmallTlut;

    private SafetensorsFile? _openExpertsLayerFile;
    private int _openExpertsLayerIdx = -1;
    private Mach1ExpertLayerDecoderV3T? _expertsDecoderV3T;
    private Mach1ExpertLayerDecoderV2? _expertsDecoderV2;

    private SafetensorsFile? _neTlutFile;
    private SafetensorsFile? _openNeLayerFile;
    private int _openNeLayerIdx = -1;
    private Mach1NeSpineDecoder? _neDecoder;

    private bool _disposed;

    private Mach1PackedCheckpoint(string root, Mach1CbParams expertsCb, SafetensorsFile expertsCodebook, float[] expertsSmallTlut)
    {
        _root = root;
        _expertsCb = expertsCb;
        _expertsCodebook = expertsCodebook;
        _expertsSmallTlut = expertsSmallTlut;
    }

    /// <summary>
    /// Opens a Mach-1 <c>packed/</c> checkpoint root (the directory containing
    /// <c>packed/experts/</c>, <c>packed/ne/</c>, <c>packed/head/</c>,
    /// alongside <c>config.json</c>, <c>codec.json</c> is read from
    /// <c>packed/experts/codec.json</c>).
    /// </summary>
    /// <param name="root">
    /// The checkpoint root directory (e.g.
    /// <c>~/.dotllm/test-cache/SyzygyResearch/Mach-1-Additive-35B</c>).
    /// </param>
    public static Mach1PackedCheckpoint Open(string root)
    {
        string expertsDir = Path.Combine(root, "packed", "experts");
        string codecJsonPath = Path.Combine(expertsDir, "codec.json");
        if (!File.Exists(codecJsonPath))
            throw new FileNotFoundException(
                $"Mach-1 packed checkpoint at '{root}' is missing 'packed/experts/codec.json'.", codecJsonPath);

        using JsonDocument codecDoc = JsonDocument.Parse(File.ReadAllText(codecJsonPath));
        JsonElement codecRoot = codecDoc.RootElement;

        string? container = codecRoot.TryGetProperty("container", out var containerEl) && containerEl.ValueKind == JsonValueKind.String
            ? containerEl.GetString()
            : null;
        if (!string.Equals(container, ExpectedExpertContainer, StringComparison.Ordinal))
        {
            throw new NotSupportedException(
                $"Mach-1 experts codec.json declares container='{container}', but this loader only " +
                $"implements '{ExpectedExpertContainer}' (Phase A/B scope). codec.json's container field " +
                "is versioned by design (issue #266 risk #5) — dispatch was refused rather than guessing.");
        }

        if (!codecRoot.TryGetProperty("cb_params", out JsonElement cbParamsEl))
            throw new InvalidDataException("packed/experts/codec.json is missing 'cb_params'.");
        var cb = new Mach1CbParams(
            K: cbParamsEl.GetProperty("K").GetDouble(),
            L: cbParamsEl.GetProperty("L").GetInt32(),
            V: cbParamsEl.GetProperty("V").GetInt32(),
            TlutBits: cbParamsEl.GetProperty("tlut_bits").GetInt32(),
            TdX: cbParamsEl.GetProperty("td_x").GetInt32(),
            TdY: cbParamsEl.GetProperty("td_y").GetInt32());

        string codebookPath = Path.Combine(expertsDir, "codebook.safetensors");
        SafetensorsFile codebookFile = SafetensorsFile.Open(codebookPath);
        RejectZstdSidecar(codebookFile, codebookPath);

        float[] smallTlut;
        try
        {
            var tlutSpan = MemoryMarshal.Cast<byte, float>(codebookFile.GetTensorSpan("tlut"));
            smallTlut = tlutSpan.ToArray();
        }
        catch
        {
            codebookFile.Dispose();
            throw;
        }

        return new Mach1PackedCheckpoint(root, cb, codebookFile, smallTlut);
    }

    /// <summary>
    /// Decodes one expert's one projection (<c>"gate"</c>, <c>"up"</c>, or
    /// <c>"down"</c>) at the given layer to dense <c>[m0, n0]</c> fp32.
    /// Opens (and, on layer change, disposes the previous) <c>packed/experts/L{LL}.safetensors</c>
    /// as needed — see the memory-discipline remarks on this type.
    /// </summary>
    public void DecodeExpertProjection(int layer, int expertIndex, string proj, int m0, int n0, Span<float> dest)
    {
        EnsureExpertsLayerOpen(layer);
        if (_expertsDecoderV3T is { } v3t)
        {
            v3t.DecodeExpertProjection(expertIndex, proj, m0, n0, dest);
        }
        else if (_expertsDecoderV2 is { } v2)
        {
            v2.DecodeExpertProjection(expertIndex, proj, dest);
        }
        else
        {
            throw new InvalidOperationException("No expert decoder resolved after EnsureExpertsLayerOpen.");
        }
    }

    private void EnsureExpertsLayerOpen(int layer)
    {
        if (_openExpertsLayerIdx == layer)
            return;

        _openExpertsLayerFile?.Dispose();
        _expertsDecoderV3T = null;
        _expertsDecoderV2 = null;

        string path = Path.Combine(_root, "packed", "experts", $"L{layer:D2}.safetensors");
        var file = SafetensorsFile.Open(path);
        RejectZstdSidecar(file, path);

        Mach1ExpertContainerKind kind = Mach1ExpertContainer.Detect(file.Metadata);
        switch (kind)
        {
            case Mach1ExpertContainerKind.ChunkedV3T:
                _expertsDecoderV3T = new Mach1ExpertLayerDecoderV3T(file, _expertsSmallTlut, _expertsCb);
                break;
            case Mach1ExpertContainerKind.ManifestV2:
                _expertsDecoderV2 = new Mach1ExpertLayerDecoderV2(file, _expertsCodebook);
                break;
            default:
                file.Dispose();
                throw new NotSupportedException(
                    $"'{path}' declares an unrecognized expert-tier container (metadata has neither a " +
                    "'wave_gamma' fields entry nor a 'manifest' key) — refusing to guess (issue #266 risk #5).");
        }

        _openExpertsLayerFile = file;
        _openExpertsLayerIdx = layer;
    }

    /// <summary>
    /// Returns the NE-tier <c>dims</c> metadata for the given layer's shard:
    /// tensor name -&gt; <c>[m0, n0, m, n]</c> (unpadded then padded dims).
    /// </summary>
    public IEnumerable<string> GetNeTensorNames(int layer)
    {
        EnsureNeLayerOpen(layer);
        return _neDecoder!.TensorNames;
    }

    /// <summary>
    /// Decodes one named NE-tier tensor (attention / linear-attn / shared-expert
    /// projection) at the given layer to dense <c>[m0, n0]</c> fp32. Tensor
    /// names are the full HF checkpoint keys, e.g.
    /// <c>model.language_model.layers.0.linear_attn.in_proj_qkv.weight</c>.
    /// </summary>
    public void DecodeNeTensor(int layer, string tensorName, Span<float> dest)
    {
        EnsureNeLayerOpen(layer);
        _neDecoder!.DecodeTensor(tensorName, dest);
    }

    private void EnsureNeLayerOpen(int layer)
    {
        if (_openNeLayerIdx == layer)
            return;

        if (_neTlutFile is null)
        {
            string tlutPath = Path.Combine(_root, "packed", "ne", "tlut.safetensors");
            _neTlutFile = SafetensorsFile.Open(tlutPath);
            RejectZstdSidecar(_neTlutFile, tlutPath);
        }

        _openNeLayerFile?.Dispose();
        _neDecoder = null;

        string path = Path.Combine(_root, "packed", "ne", $"L{layer:D2}.safetensors");
        var file = SafetensorsFile.Open(path);
        RejectZstdSidecar(file, path);

        if (!Mach1NeSpineDecoder.IsCanonRhtTrellisContainer(file.Metadata))
        {
            string? codec = file.Metadata.TryGetValue("codec", out string? c) ? c : null;
            file.Dispose();
            throw new NotSupportedException(
                $"'{path}' declares NE-tier codec='{codec}', but this loader only implements " +
                "'canon_rht_bitshift_trellis_intlattice' (Phase A/B scope). Older transform-free / " +
                "manifest-driven NE containers documented in decode.py are not ported — refusing to " +
                "guess (issue #266 risk #5).");
        }

        _neDecoder = new Mach1NeSpineDecoder(file, _neTlutFile);
        _openNeLayerFile = file;
        _openNeLayerIdx = layer;
    }

    /// <summary>
    /// Decodes the full embedding table (<c>packed/ne/embed_int4.safetensors</c>,
    /// affine int4 asym g64 + exact-overwrite exceptions) to dense
    /// <c>[vocabSize, hiddenSize]</c> fp32. Materializes the whole table (a few
    /// GB at 35B scale) — acceptable per issue #266's explicit "permitted
    /// stepping-stone" (spine/head/embeddings may decode to dense at load; only
    /// the routed experts are memory-critical).
    /// </summary>
    public void DecodeEmbedding(Span<float> dest)
    {
        string path = Path.Combine(_root, "packed", "ne", "embed_int4.safetensors");
        using var file = SafetensorsFile.Open(path);
        RejectZstdSidecar(file, path);
        Mach1EmbedDecoder.Decode(file, bits: 4, dest);
    }

    /// <summary>
    /// Decodes the full LM head (<c>packed/head/head_c{0..N-1}of{N}.safetensors</c>,
    /// int5-g64) to dense <c>[vocabSize, hiddenSize]</c> fp32.
    /// </summary>
    public void DecodeHead(Span<float> dest, int hiddenSize)
    {
        string headDir = Path.Combine(_root, "packed", "head");
        if (!Directory.Exists(headDir))
            throw new DirectoryNotFoundException($"Mach-1 checkpoint at '{_root}' has no 'packed/head/' directory.");

        string[] chunkFiles = Directory.GetFiles(headDir, "head_c*of*.safetensors")
            .OrderBy(f => f, StringComparer.Ordinal)
            .ToArray();
        if (chunkFiles.Length == 0)
            throw new FileNotFoundException($"No 'head_c*of*.safetensors' chunk files found under '{headDir}'.");

        foreach (string chunkPath in chunkFiles)
        {
            using var file = SafetensorsFile.Open(chunkPath);
            RejectZstdSidecar(file, chunkPath);
            Mach1HeadDecoder.DecodeChunkInto(file, dest, hiddenSize);
        }
    }

    /// <summary>
    /// Throws <see cref="NotSupportedException"/> if the file carries a
    /// <c>__zsc__</c> zstd sidecar tensor (decode.py's <c>_read_safetensors_np</c>
    /// documents this as present on "v3 files" holding every non-code-stream
    /// tensor byte-exactly). No files in the fixture this port was validated
    /// against carry one; zstd decompression is not wired into this loader.
    /// </summary>
    private static void RejectZstdSidecar(SafetensorsFile file, string path)
    {
        if (file.TensorsByName.ContainsKey("__zsc__"))
        {
            throw new NotSupportedException(
                $"'{path}' carries a '__zsc__' zstd sidecar (decode.py's v3-file convention). " +
                "This loader does not implement zstd decompression — refusing to silently drop the " +
                "sidecar-compressed tensors (issue #266 risk #5, format churn).");
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _openExpertsLayerFile?.Dispose();
        _expertsCodebook.Dispose();
        _openNeLayerFile?.Dispose();
        _neTlutFile?.Dispose();
    }
}
