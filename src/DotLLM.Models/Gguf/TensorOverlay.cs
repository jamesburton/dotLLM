using System.Security.Cryptography;
using System.Text.RegularExpressions;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;

namespace DotLLM.Models.Gguf;

/// <summary>
/// A tensor overlay (issue #1706, <c>--tensor-overlay overlay.gguf</c>): a small GGUF whose tensors REPLACE exactly the named FFN projections
/// of a base model, so an arbitrary subset of <c>(layer, expert, projection)</c> can run in another format (typically
/// <see cref="QuantizationType.TERN_B64_F16"/>) while everything else keeps its original precision.
/// </summary>
/// <remarks>
/// <para><b>Contract</b> (also in <c>docs/QUANTIZATION.md</c>; backend-independent, so the Vulkan loader reuses it unchanged):</para>
/// <list type="bullet">
/// <item>Tensor names: <c>blk.{L}.ffn_{gate,up,down}.{E}.weight</c> for a routed expert, <c>blk.{L}.ffn_{gate,up,down}.weight</c> for a dense FFN.
/// Any other name is an error.</item>
/// <item>Shapes are GGUF dim order <c>[in, out]</c> (dim 0 innermost = the reduction axis), i.e. row-major <c>[out, in]</c> data. They must equal the
/// base projection's shape exactly (gate/up: <c>[hidden, ffn]</c>; down: <c>[ffn, hidden]</c>); a mismatch is an error.</item>
/// <item>Metadata <c>dotllm.overlay.base_sha256</c> = lowercase hex SHA-256 of the base model file (first shard of a split set). A missing key or a
/// mismatch is an error unless hash verification is explicitly disabled (<c>--overlay-skip-base-hash</c>); hashing a multi-GB file takes seconds,
/// so the digest is cached per (path, length, mtime) under <c>~/.dotllm/hash-cache</c>.</item>
/// <item>Per-tensor dtype is the tensor's GGUF type. <see cref="QuantizationType.TERN_B64_F16"/> tensors are validated at open (in % 64 == 0, no
/// <c>0b11</c> trit code). Tensors are read in place from the overlay's mapping; nothing is copied or expanded.</item>
/// <item>Nothing is ever ignored: a name the base model does not have, or a loader that did not consume the overlay, fails
/// (<see cref="VerifyAllApplied"/>).</item>
/// </list>
/// </remarks>
public sealed partial class TensorOverlay : IDisposable
{
    /// <summary><c>general.architecture</c> of an overlay GGUF; only such files may carry raw I8 container tensors.</summary>
    public const string OverlayArchitecture = "dotllm-overlay";

    /// <summary>Metadata key holding the SHA-256 (lowercase hex) of the base model file.</summary>
    public const string BaseSha256Key = "dotllm.overlay.base_sha256";

    /// <summary>Projection indices used by <see cref="TryParseName"/>: 0 gate, 1 up, 2 down.</summary>
    public static readonly string[] ProjectionNames = ["gate", "up", "down"];

    [GeneratedRegex(@"^blk\.(\d+)\.ffn_(gate|up|down)(?:\.(\d+))?\.weight$", RegexOptions.CultureInvariant)]
    private static partial Regex NameRegex();

    private readonly HashSet<string> _applied = new(StringComparer.Ordinal);
    private readonly List<Entry> _entries = [];
    private readonly List<string> _notes = [];

    /// <summary>One overlay tensor in LOGICAL terms (the container may carry packed bytes as I8 with the real dtype/shape in metadata).</summary>
    private sealed record Entry(string Name, int In, int Out, QuantizationType Qt, nint Ptr);

    /// <summary>Metadata key prefix of the logical dtype of an I8-container tensor (dotllm.overlay.dtype.{tensor}).</summary>
    public const string DtypeKeyPrefix = "dotllm.overlay.dtype.";

    /// <summary>Metadata key prefix of the logical [out, in] shape of an I8-container tensor (dotllm.overlay.shape.{tensor}).</summary>
    public const string ShapeKeyPrefix = "dotllm.overlay.shape.";

    /// <summary>Base-hash value meaning "not recorded": the check is skipped with a note, not an error.</summary>
    public const string UnspecifiedHash = "UNSPECIFIED";

    private TensorOverlay(GgufFile file, string path) { File = file; Path = path; }

    /// <summary>Warnings and decisions made while opening (e.g. a skipped base-hash check); callers print them.</summary>
    public IReadOnlyList<string> Notes => _notes;

    /// <summary>The opened overlay GGUF (owned).</summary>
    public GgufFile File { get; }

    /// <summary>Path the overlay was opened from.</summary>
    public string Path { get; }

    /// <summary>Number of tensors in the overlay.</summary>
    public int Count => _entries.Count;

    /// <summary>Number of overlay tensors a loader has consumed.</summary>
    public int AppliedCount => _applied.Count;

    /// <summary>
    /// Parses an overlay tensor name.
    /// </summary>
    /// <param name="name">Tensor name.</param>
    /// <param name="layer">Layer index.</param>
    /// <param name="proj">0 gate, 1 up, 2 down.</param>
    /// <param name="expert">Expert index, or -1 for a dense FFN tensor.</param>
    /// <returns>False when the name is not in the overlay grammar.</returns>
    public static bool TryParseName(string name, out int layer, out int proj, out int expert)
    {
        layer = proj = 0; expert = -1;
        var m = NameRegex().Match(name);
        if (!m.Success) return false;
        if (!int.TryParse(m.Groups[1].ValueSpan, out layer)) return false;
        proj = m.Groups[2].Value switch { "gate" => 0, "up" => 1, _ => 2 };
        if (m.Groups[3].Success && !int.TryParse(m.Groups[3].ValueSpan, out expert)) return false;
        return true;
    }

    /// <summary>
    /// Opens and validates an overlay against its base file.
    /// </summary>
    /// <param name="overlayPath">Overlay GGUF path.</param>
    /// <param name="baseModelPath">Path of the base model file (hashed unless <paramref name="verifyBaseHash"/> is false).</param>
    /// <param name="verifyBaseHash">False = skip the base-hash check entirely (explicit opt-out; documented trade-off for multi-GB bases).</param>
    /// <returns>The overlay (caller owns; normally handed to <see cref="GgufFile.AttachOverlay"/>).</returns>
    /// <exception cref="InvalidDataException">Unknown tensor name, malformed ternary tensor, missing/mismatched base hash.</exception>
    public static TensorOverlay Open(string overlayPath, string? baseModelPath, bool verifyBaseHash = true)
    {
        var file = GgufFile.Open(overlayPath);
        try
        {
            var ov = new TensorOverlay(file, overlayPath);
            ov.ValidateNames();
            ov.ValidateTensors();
            if (verifyBaseHash) ov.VerifyBaseHash(baseModelPath);
            return ov;
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Opens <paramref name="overlayPath"/>, validates it against <paramref name="baseModelPath"/> and attaches it to <paramref name="baseFile"/>
    /// (which then owns it). The shared entry point of every CLI/server surface.
    /// </summary>
    public static TensorOverlay Attach(GgufFile baseFile, string? baseModelPath, string overlayPath, bool verifyBaseHash = true)
    {
        var ov = Open(overlayPath, baseModelPath, verifyBaseHash);
        baseFile.AttachOverlay(ov);
        return ov;
    }

    private void ValidateNames()
    {
        var bad = new List<string>();
        foreach (var t in File.Tensors)
            if (!TryParseName(t.Name, out _, out _, out _))
                bad.Add(t.Name);
        if (bad.Count > 0)
            throw new InvalidDataException(
                $"Tensor overlay '{Path}' has {bad.Count} tensor name(s) outside blk.{{L}}.ffn_{{gate,up,down}}[.{{E}}].weight: "
                + string.Join(", ", bad.Take(5)) + (bad.Count > 5 ? ", ..." : "") + ". Unknown tensors are never ignored.");
    }

    private static long[]? ReadLongs(GgufMetadata md, string key)
    {
        if (!md.TryGetValue(key, out var v) || v.Value is string || v.Value is not System.Collections.IEnumerable e) return null;
        var l = new List<long>();
        foreach (object? o in e) l.Add(Convert.ToInt64(o, System.Globalization.CultureInfo.InvariantCulture));
        return l.ToArray();
    }

    private unsafe void ValidateTensors()
    {
        foreach (var t in File.Tensors)
        {
            QuantizationType qt; int inDim, outDim;
            string dtypeKey = DtypeKeyPrefix + t.Name;
            if (File.Metadata.TryGetValue(dtypeKey, out var dv))
            {
                // I8 container: python gguf can only write I8, so a packed format travels as raw bytes with its logical dtype/shape in metadata.
                string dtype = (dv.Value as string) ?? "";
                if (!Enum.TryParse(dtype, ignoreCase: false, out qt) || !Enum.IsDefined(qt) || qt is QuantizationType.I8 or QuantizationType.LittleBit)
                    throw new InvalidDataException($"Tensor overlay '{Path}': '{t.Name}' declares unknown/unsupported dtype '{dtype}' in {dtypeKey}.");
                var shape = ReadLongs(File.Metadata, ShapeKeyPrefix + t.Name);
                if (shape is not { Length: 2 } || shape[0] <= 0 || shape[1] <= 0 || shape[0] > int.MaxValue || shape[1] > int.MaxValue)
                    throw new InvalidDataException($"Tensor overlay '{Path}': '{t.Name}' needs {ShapeKeyPrefix}{t.Name} = [out, in] (got {(shape is null ? "none" : "[" + string.Join(", ", shape) + "]")}).");
                outDim = (int)shape[0]; inDim = (int)shape[1];
                if (t.QuantizationType != QuantizationType.I8 && t.QuantizationType != qt)
                    throw new InvalidDataException($"Tensor overlay '{Path}': '{t.Name}' is stored as {t.QuantizationType} but declares logical dtype {dtype}; only an I8 container or the native type is accepted.");
            }
            else
            {
                if (t.Shape.Rank != 2)
                    throw new InvalidDataException($"Tensor overlay '{Path}': '{t.Name}' must be rank 2 ([in, out]), got {t.Shape}.");
                qt = t.QuantizationType; inDim = t.Shape[0]; outDim = t.Shape[1];
            }

            if (qt == QuantizationType.TERN_B64_F16 && inDim % TernB64F16.GroupSize != 0)
                throw new InvalidDataException($"Tensor overlay '{Path}': TERN_B64_F16 tensor '{t.Name}' has in-dimension {inDim}, not a multiple of {TernB64F16.GroupSize}.");
            long physical = t.QuantizationType.ComputeByteCount(t.Shape.ElementCount);
            long expected = qt.ComputeByteCount((long)inDim * outDim);
            if (physical != expected)
                throw new InvalidDataException(
                    $"Tensor overlay '{Path}': '{t.Name}' holds {physical} bytes but {qt} [out={outDim}, in={inDim}] needs {expected}.");

            nint ptr = File.TensorDataPointer(t);
            if (qt == QuantizationType.TERN_B64_F16)
                TernB64F16.Validate(t.Name, (byte*)ptr, outDim, inDim);
            _entries.Add(new Entry(t.Name, inDim, outDim, qt, ptr));
        }
    }

    private void VerifyBaseHash(string? baseModelPath)
    {
        if (!File.Metadata.TryGetValue(BaseSha256Key, out var v) || v.Value is not string expected || expected.Length == 0)
            throw new InvalidDataException(
                $"Tensor overlay '{Path}' has no '{BaseSha256Key}' metadata, so it cannot be tied to a base model. Re-export it with the base hash, "
                + "or pass --overlay-skip-base-hash to run it unchecked.");
        if (string.Equals(expected.Trim(), UnspecifiedHash, StringComparison.Ordinal))
        {
            _notes.Add($"WARNING: tensor overlay '{Path}' has {BaseSha256Key} = {UnspecifiedHash}; the base model was NOT verified.");
            return;
        }
        if (baseModelPath is null)
        {
            _notes.Add($"tensor overlay '{Path}': base is not a single GGUF file, so {BaseSha256Key} cannot be checked (skipped).");
            return;
        }
        string actual = BaseHash.Compute(baseModelPath);
        if (!string.Equals(expected.Trim(), actual, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException(
                $"Tensor overlay '{Path}' was built for a different base model: {BaseSha256Key} = {expected}, but '{baseModelPath}' hashes to {actual}. "
                + "(--overlay-skip-base-hash disables this check.)");
    }

    /// <summary>
    /// Applies the overlay to the loaded FFN projections of the CPU <c>TransformerModel</c>: dense layers have the projection pointer/format
    /// swapped; MoE layers get a per-expert side table (<see cref="ExpertOverlayTable"/>) beside their untouched stacked banks.
    /// </summary>
    internal void ApplyToLayers(TransformerLayerWeights[] layers, ModelConfig config)
    {
        // Collect per-layer tables first so a failure leaves the model untouched.
        foreach (var t in _entries)
        {
            TryParseName(t.Name, out int layer, out int proj, out int expert);
            string what = $"Tensor overlay '{Path}': '{t.Name}'";
            if (layer >= layers.Length)
                throw new InvalidDataException($"{what} names layer {layer}, but the model has {layers.Length} layers.");

            ref readonly var lw = ref layers[layer];
            nint ptr = t.Ptr;
            if (expert >= 0)
            {
                var moe = lw.Moe;
                if (moe is null)
                    throw new InvalidDataException($"{what} names an expert, but layer {layer} of the base model is a dense FFN layer.");
                if (!moe.UseQuantExperts)
                    AdoptF32ExpertBanks(moe, config, what, layer);
                if (expert >= moe.NumExperts)
                    throw new InvalidDataException($"{what} names expert {expert}, but the layer has {moe.NumExperts} experts.");
                int expIn = proj == 2 ? moe.IntermediateSize : moe.HiddenSize;
                int expOut = proj == 2 ? moe.HiddenSize : moe.IntermediateSize;
                CheckShape(what, t.In, t.Out, expIn, expOut);
                moe.ExpertOverlay ??= new ExpertOverlayTable(moe.NumExperts);
                moe.ExpertOverlay.Set(proj, expert, ptr, t.Qt);
            }
            else
            {
                if (lw.Moe is not null)
                    throw new InvalidDataException($"{what} is a dense-FFN name, but layer {layer} of the base model is a MoE layer (use blk.{layer}.ffn_{ProjectionNames[proj]}.{{E}}.weight).");
                if (lw.GateWeight == 0)
                    throw new InvalidDataException($"{what}: layer {layer} of the base model has no dense FFN projections.");
                int inDim = proj switch { 0 => lw.GateInputDim, 1 => lw.UpInputDim, _ => lw.DownInputDim };
                int outDim = proj switch { 0 => lw.GateOutputDim, 1 => lw.UpOutputDim, _ => lw.DownOutputDim };
                CheckShape(what, t.In, t.Out, inDim, outDim);
                layers[layer] = lw.WithDenseFfnOverride(proj, ptr, t.Qt);
            }
            _applied.Add(t.Name);
        }
    }

    /// <summary>One overlay tensor in logical terms, for GPU backends that upload the packed bytes themselves (#1706 S3).</summary>
    /// <param name="Name">Tensor name.</param>
    /// <param name="Layer">Layer index.</param>
    /// <param name="Proj">0 gate, 1 up, 2 down.</param>
    /// <param name="Expert">Expert index, or -1 for a dense FFN tensor.</param>
    /// <param name="In">Contraction length.</param>
    /// <param name="Out">Output rows.</param>
    /// <param name="Qt">Storage format.</param>
    /// <param name="Ptr">Address inside the overlay's mmap (valid while the overlay is open).</param>
    internal readonly record struct GpuEntry(string Name, int Layer, int Proj, int Expert, int In, int Out, QuantizationType Qt, nint Ptr);

    /// <summary>The overlay's tensors in logical terms (see <see cref="GpuEntry"/>).</summary>
    internal IEnumerable<GpuEntry> GpuEntries()
    {
        foreach (var t in _entries)
        {
            TryParseName(t.Name, out int layer, out int proj, out int expert);
            yield return new GpuEntry(t.Name, layer, proj, expert, t.In, t.Out, t.Qt, t.Ptr);
        }
    }

    /// <summary>Marks an overlay tensor consumed by a backend that handled it outside <see cref="ApplyToLayers"/> (the Vulkan expert upload).</summary>
    internal void MarkApplied(string name) => _applied.Add(name);

    /// <summary>
    /// Applies only the DENSE-FFN entries to the loaded layers (pointer/format swap, shape-checked; the Vulkan loader then uploads the swapped
    /// projection packed). Expert entries are left for the backend's own per-expert upload and are marked applied there.
    /// </summary>
    internal void ApplyDenseToLayers(TransformerLayerWeights[] layers)
    {
        foreach (var t in _entries)
        {
            TryParseName(t.Name, out int layer, out int proj, out int expert);
            if (expert >= 0) continue;
            string what = $"Tensor overlay '{Path}': '{t.Name}'";
            if (layer >= layers.Length)
                throw new InvalidDataException($"{what} names layer {layer}, but the model has {layers.Length} layers.");
            ref readonly var lw = ref layers[layer];
            if (lw.Moe is not null)
                throw new InvalidDataException($"{what} is a dense-FFN name, but layer {layer} of the base model is a MoE layer (use blk.{layer}.ffn_{ProjectionNames[proj]}.{{E}}.weight).");
            if (lw.GateWeight == 0)
                throw new InvalidDataException($"{what}: layer {layer} of the base model has no dense FFN projections.");
            int inDim = proj switch { 0 => lw.GateInputDim, 1 => lw.UpInputDim, _ => lw.DownInputDim };
            int outDim = proj switch { 0 => lw.GateOutputDim, 1 => lw.UpOutputDim, _ => lw.DownOutputDim };
            CheckShape(what, t.In, t.Out, inDim, outDim);
            layers[layer] = lw.WithDenseFfnOverride(proj, t.Ptr, t.Qt);
            _applied.Add(t.Name);
        }
    }

    /// <summary>
    /// A safetensors-loaded MoE layer (HF checkpoints: per-expert F32 matrices <c>W1</c> = gate, <c>W3</c> = up, <c>W2</c> = down) joins the same code path as
    /// the GGUF stacked banks: every expert gets an F32 entry in the side table (pointing at its existing matrix, no copy) and the layer runs
    /// <c>MoeQuantSwiGluMlp</c> with plain-SwiGLU / Mixtral gating, exactly what <c>MoeSwiGluMlp</c> computes for it. Overlay entries then replace individual
    /// (expert, projection) cells.
    /// </summary>
    private static void AdoptF32ExpertBanks(MoeLayerWeights moe, ModelConfig config, string what, int layer)
    {
        if (moe.W1.Length != moe.NumExperts || moe.W1[0] == 0 || moe.W2.Length != moe.NumExperts || moe.W3.Length != moe.NumExperts)
            throw new NotSupportedException($"{what}: layer {layer}'s experts are not in a layout the overlay path supports (stacked ffn_*_exps banks or per-expert F32 matrices).");
        if (moe.HasSharedExpert || moe.SigmoidGating || moe.WeightsScale != 1.0f || moe.SelectionBias is not null || moe.GateBias is not null
            || config.ActivationFunction != ActivationFunction.SiLU)
            throw new NotSupportedException($"{what}: layer {layer} has shared experts, sigmoid/biased routing or a non-SiLU activation, which the overlay path does not support yet.");
        var table = moe.ExpertOverlay ?? new ExpertOverlayTable(moe.NumExperts);
        for (int e = 0; e < moe.NumExperts; e++)
        {
            table.Set(0, e, moe.W1[e], QuantizationType.F32);
            table.Set(1, e, moe.W3[e], QuantizationType.F32);
            table.Set(2, e, moe.W2[e], QuantizationType.F32);
        }
        moe.ExpertOverlay = table;
        moe.UseQuantExperts = true;
        moe.SoftmaxAfterTopK = false;
        moe.UseSwiGluOai = false;
    }

    private static void CheckShape(string what, int gotIn, int gotOut, int expIn, int expOut)
    {
        if (gotIn != expIn || gotOut != expOut)
            throw new InvalidDataException(
                $"{what} has shape [in={gotIn}, out={gotOut}], but the base projection is [in={expIn}, out={expOut}].");
    }

    /// <summary>
    /// Throws when any overlay tensor was not consumed by the loaded model (a backend or architecture without overlay support must fail, not
    /// silently run the unmodified base).
    /// </summary>
    /// <exception cref="NotSupportedException">The overlay was not (fully) applied.</exception>
    public void VerifyAllApplied()
    {
        if (_applied.Count == Count) return;
        throw new NotSupportedException(
            $"--tensor-overlay '{Path}': {Count - _applied.Count} of {Count} overlay tensors were not applied. Overlays are supported on the CPU "
            + "backend (plain transformer / MoE) and on Vulkan for dense-FFN and plain-SwiGLU MoE GGUF models (TERN_B64_F16 tensors only); "
            + "use --device cpu or --device vulkan with a supported architecture.");
    }

    /// <inheritdoc/>
    public void Dispose() => File.Dispose();

    /// <summary>Cached SHA-256 of model files (hex), keyed by resolved path, length and mtime.</summary>
    public static class BaseHash
    {
        /// <summary>Digest of <paramref name="path"/>, computed (streamed) on first use and cached under <c>~/.dotllm/hash-cache</c>.</summary>
        public static string Compute(string path)
        {
            string real = path;
            try { if (new FileInfo(path).ResolveLinkTarget(true) is { } t) real = t.FullName; } catch { /* use as given */ }
            var info = new FileInfo(real);
            string key = Convert.ToHexString(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(
                $"{info.FullName.ToLowerInvariant()}|{info.Length}|{info.LastWriteTimeUtc.Ticks}"))).ToLowerInvariant();
            string dir = Environment.GetEnvironmentVariable("DOTLLM_HASH_CACHE_DIR")
                         ?? System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "hash-cache");
            string cache = System.IO.Path.Combine(dir, key + ".sha256");
            try
            {
                if (System.IO.File.Exists(cache) && System.IO.File.ReadAllText(cache).Trim() is { Length: 64 } hit) return hit;
            }
            catch { /* recompute */ }

            using var fs = new FileStream(real, FileMode.Open, FileAccess.Read, FileShare.ReadWrite, 1 << 20, FileOptions.SequentialScan);
            string hex = Convert.ToHexString(SHA256.HashData(fs)).ToLowerInvariant();
            try
            {
                Directory.CreateDirectory(dir);
                System.IO.File.WriteAllText(cache, hex);
            }
            catch { /* cache is best effort */ }
            return hex;
        }
    }
}
