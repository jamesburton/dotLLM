using System.Globalization;
using DotLLM.Core.Configuration;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Adapters.Ternary;

/// <summary>
/// GGUF container for ternary residual adapters (#1619): writes and reads the same circuits as the <c>.nsadapter</c> ZIP package, bit-identically.
/// Layout (docs/TERNARY_ADAPTER.md, "GGUF form"):
/// <list type="bullet">
/// <item>Metadata of adapter 0 is the reference's <c>neuro_symbolic.*</c> key set (<see cref="NsAdapterPackage.ToGgufMetadata"/>, string values, byte for byte);
/// adapter <c>i &gt;= 1</c> uses <c>neuro_symbolic.{i}.*</c>. <c>nsadapter.count</c> (uint32, absent = 1) is the number of adapters.
/// Exact binary32 <c>nsadapter.{i}.alpha</c> / <c>.activation_threshold</c> are written too and preferred on read (the reference's <c>G7</c> strings do not round-trip every float).</item>
/// <item>Tensors per adapter: <c>nsadapter.{i}.down</c> and <c>.up</c> hold the package's packed BitNet 2-bit codes verbatim (rows * ceil(cols/4) bytes, zero-padded to a multiple of 4)
/// as a 1-D <c>I32</c> tensor (the only existing 4-byte integer GGUF type dotLLM reads; no new ggml type is introduced), and <c>.down_scale</c> / <c>.up_scale</c> the per-row
/// scales as 1-D <c>F32</c> tensors. Shapes come from <c>input_dimension</c>/<c>rank</c> metadata (<c>down</c> = rank x D, <c>up</c> = D x rank).</item>
/// <item>A standalone adapter file sets <c>general.architecture = "nsadapter"</c>; the same keys/tensors may also be appended to a model GGUF.</item>
/// </list>
/// </summary>
public static class NsAdapterGguf
{
    /// <summary><c>general.architecture</c> of a standalone adapter file.</summary>
    public const string Architecture = "nsadapter";
    /// <summary>Metadata key holding the number of adapters (uint32; absent means 1).</summary>
    public const string CountKey = "nsadapter.count";
    /// <summary>Largest adapter count accepted from a file.</summary>
    public const int MaxAdapters = 256;

    /// <summary>Tensor name of adapter <paramref name="i"/>'s packed down codes.</summary>
    public static string DownName(int i) => $"nsadapter.{i}.down";
    /// <summary>Tensor name of adapter <paramref name="i"/>'s packed up codes.</summary>
    public static string UpName(int i) => $"nsadapter.{i}.up";
    /// <summary>Tensor name of adapter <paramref name="i"/>'s per-row down scales.</summary>
    public static string DownScaleName(int i) => $"nsadapter.{i}.down_scale";
    /// <summary>Tensor name of adapter <paramref name="i"/>'s per-row up scales.</summary>
    public static string UpScaleName(int i) => $"nsadapter.{i}.up_scale";

    private static string MetaKey(int i, string suffix) => i == 0 ? $"neuro_symbolic.{suffix}" : $"neuro_symbolic.{i}.{suffix}";

    /// <summary>True when <paramref name="path"/> starts with the GGUF magic.</summary>
    public static bool LooksLikeGguf(string path)
    {
        try
        {
            using var fs = File.OpenRead(path);
            Span<byte> m = stackalloc byte[4];
            return fs.Read(m) == 4 && m.SequenceEqual("GGUF"u8);
        }
        catch (IOException) { return false; }
    }

    /// <summary>Loads adapters from a <c>.nsadapter</c> ZIP or an adapter GGUF, detected by magic.</summary>
    public static IReadOnlyList<NsAdapterPackage> LoadAny(string path) =>
        LooksLikeGguf(path) ? Read(path) : [NsAdapterPackage.Load(path)];

    // --------------------------------------------------------------------------------------------------------- write ----

    /// <summary>Serialises packages to a standalone adapter GGUF.</summary>
    public static byte[] Write(IReadOnlyList<NsAdapterPackage> packages)
    {
        ArgumentNullException.ThrowIfNull(packages);
        if (packages.Count is 0 or > MaxAdapters) throw new ArgumentException($"adapter count must be 1..{MaxAdapters}", nameof(packages));
        var w = new GgufWriter().AddString("general.architecture", Architecture).AddUInt32(CountKey, (uint)packages.Count);
        for (int i = 0; i < packages.Count; i++)
        {
            NsAdapterPackage p = packages[i];
            p.ValidateDimensions();
            var meta = p.ToGgufMetadata();
            meta["neuro_symbolic.down_checksum"] = p.Down.DataChecksum();   // computed, not trusted from the manifest
            meta["neuro_symbolic.up_checksum"] = p.Up.DataChecksum();
            foreach (var (key, value) in meta.OrderBy(kv => kv.Key, StringComparer.Ordinal))
                w.AddString(MetaKey(i, key["neuro_symbolic.".Length..]), value);
            w.AddFloat32($"nsadapter.{i}.alpha", p.Manifest.Alpha);
            w.AddFloat32($"nsadapter.{i}.activation_threshold", p.Manifest.ActivationThreshold);
            AddCodes(w, DownName(i), p.Down);
            AddCodes(w, UpName(i), p.Up);
            AddScales(w, DownScaleName(i), p.Down.Scales);
            AddScales(w, UpScaleName(i), p.Up.Scales);
        }
        return w.Build();
    }

    /// <summary>Writes packages to <paramref name="path"/> as an adapter GGUF.</summary>
    public static void Write(string path, IReadOnlyList<NsAdapterPackage> packages) => File.WriteAllBytes(path, Write(packages));

    private static void AddCodes(GgufWriter w, string name, PackedTernaryTensor t)
    {
        int padded = (t.Data.Length + 3) & ~3;
        var blob = new byte[padded];
        t.Data.CopyTo(blob, 0);
        w.AddTensor(name, [padded / 4], (uint)QuantizationType.I32, blob);
    }

    private static void AddScales(GgufWriter w, string name, float[] scales)
    {
        var blob = new byte[scales.Length * 4];
        Buffer.BlockCopy(scales, 0, blob, 0, blob.Length);
        w.AddTensor(name, [scales.Length], (uint)QuantizationType.F32, blob);
    }

    // ---------------------------------------------------------------------------------------------------------- read ----

    /// <summary>Opens <paramref name="path"/> and reads every adapter in it.</summary>
    public static IReadOnlyList<NsAdapterPackage> Read(string path)
    {
        GgufFile gguf;
        try { gguf = GgufFile.Open(path); }
        catch (Exception ex) when (ex is InvalidDataException or EndOfStreamException or ArgumentException or NotSupportedException)
        { throw new InvalidDataException("not a valid GGUF file: " + ex.Message, ex); }
        using (gguf) return Read(gguf);
    }

    /// <summary>True when the GGUF carries at least one adapter (<c>nsadapter.0.down</c>).</summary>
    public static bool HasAdapters(GgufFile gguf) => gguf.TensorsByName.ContainsKey(DownName(0));

    /// <summary>
    /// Reads every adapter from an open GGUF and validates it like <see cref="NsAdapterPackage.Load(System.IO.Stream)"/>: dimension bounds, tensor types and
    /// exact element counts, zero padding, finite alpha/threshold, SHA-256 of both code tensors against the metadata. Throws <see cref="InvalidDataException"/>.
    /// </summary>
    public static IReadOnlyList<NsAdapterPackage> Read(GgufFile gguf)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        int count = 1;
        if (gguf.Metadata.TryGetValue(CountKey, out _))
        {
            uint c = gguf.Metadata.GetUInt32(CountKey);
            if (c is 0 or > MaxAdapters) throw new InvalidDataException($"{CountKey} = {c} out of range 1..{MaxAdapters}");
            count = (int)c;
        }
        if (!HasAdapters(gguf)) throw new InvalidDataException($"GGUF has no adapter tensors ('{DownName(0)}' missing)");
        var list = new List<NsAdapterPackage>(count);
        for (int i = 0; i < count; i++) list.Add(ReadOne(gguf, i));
        return list;
    }

    private static string Str(GgufFile g, int i, string suffix, bool required = true)
    {
        string key = MetaKey(i, suffix);
        if (g.Metadata.TryGetValue(key, out _))
        {
            try { return g.Metadata.GetString(key); }
            catch (InvalidOperationException ex) { throw new InvalidDataException(ex.Message, ex); }
        }
        if (required) throw new InvalidDataException($"adapter {i}: metadata key '{key}' missing");
        return string.Empty;
    }

    private static int Int(GgufFile g, int i, string suffix)
    {
        if (!int.TryParse(Str(g, i, suffix), NumberStyles.Integer, CultureInfo.InvariantCulture, out int v))
            throw new InvalidDataException($"adapter {i}: '{MetaKey(i, suffix)}' is not an integer");
        return v;
    }

    private static float Flt(GgufFile g, int i, string exactKey, string suffix)
    {
        if (g.Metadata.TryGetValue(exactKey, out _))
        {
            try { return g.Metadata.GetFloat32(exactKey); }
            catch (InvalidOperationException ex) { throw new InvalidDataException(ex.Message, ex); }
        }
        if (!float.TryParse(Str(g, i, suffix), NumberStyles.Float, CultureInfo.InvariantCulture, out float v))
            throw new InvalidDataException($"adapter {i}: '{MetaKey(i, suffix)}' is not a number");
        return v;
    }

    private static NsAdapterPackage ReadOne(GgufFile g, int i)
    {
        int rank = Int(g, i, "rank"), d = Int(g, i, "input_dimension");
        if (rank <= 0 || d <= 0 || rank > 4096 || d > 1 << 20) throw new InvalidDataException($"adapter {i}: Rank/InputDimension out of range ({rank}, {d})");
        int layer = Int(g, i, "target_layer");
        if (g.Metadata.TryGetValue(MetaKey(i, "layer"), out _) && Int(g, i, "layer") != layer)
            throw new InvalidDataException($"adapter {i}: neuro_symbolic layer and target_layer disagree");
        if (!DateTimeOffset.TryParse(Str(g, i, "created_at"), CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind, out var created))
            throw new InvalidDataException($"adapter {i}: created_at is not a round-trip timestamp");
        var manifest = new NsAdapterManifest
        {
            TargetModel = Str(g, i, "target_model", false), TargetLayer = layer, InputDimension = d, Rank = rank,
            Alpha = Flt(g, i, $"nsadapter.{i}.alpha", "alpha"), ActivationThreshold = Flt(g, i, $"nsadapter.{i}.activation_threshold", "activation_threshold"),
            Domain = Str(g, i, "domain", false), Description = Str(g, i, "description", false), CreatedAt = created,
            DownChecksum = Str(g, i, "down_checksum"), UpChecksum = Str(g, i, "up_checksum"),
        };
        var down = ReadTensor(g, i, DownName(i), DownScaleName(i), rank, d);
        var up = ReadTensor(g, i, UpName(i), UpScaleName(i), d, rank);
        VerifyChecksum(i, "down", manifest.DownChecksum, down);
        VerifyChecksum(i, "up", manifest.UpChecksum, up);
        return new NsAdapterPackage(manifest, down, up);   // re-validates dimensions, layer and finiteness
    }

    private static unsafe PackedTernaryTensor ReadTensor(GgufFile g, int i, string codesName, string scalesName, int rows, int cols)
    {
        long bytes = (long)rows * ((cols + 3) / 4);
        long padded = (bytes + 3) & ~3L;
        if (!g.TensorsByName.TryGetValue(codesName, out var cd)) throw new InvalidDataException($"adapter {i}: tensor '{codesName}' missing");
        if (!g.TensorsByName.TryGetValue(scalesName, out var sd)) throw new InvalidDataException($"adapter {i}: tensor '{scalesName}' missing");
        if (cd.QuantizationType != QuantizationType.I32 || cd.Shape.ElementCount != padded / 4)
            throw new InvalidDataException($"adapter {i}: '{codesName}' must be I32 with {padded / 4} elements (is {cd.QuantizationType} x {cd.Shape.ElementCount})");
        if (sd.QuantizationType != QuantizationType.F32 || sd.Shape.ElementCount != rows)
            throw new InvalidDataException($"adapter {i}: '{scalesName}' must be F32 with {rows} elements (is {sd.QuantizationType} x {sd.Shape.ElementCount})");
        var data = new byte[bytes];
        new ReadOnlySpan<byte>((void*)g.TensorDataPointer(in cd), (int)padded)[..(int)bytes].CopyTo(data);
        foreach (byte b in new ReadOnlySpan<byte>((byte*)g.TensorDataPointer(in cd) + bytes, (int)(padded - bytes)))
            if (b != 0) throw new InvalidDataException($"adapter {i}: '{codesName}' has non-zero padding");
        var scales = new float[rows];
        new ReadOnlySpan<float>((void*)g.TensorDataPointer(in sd), rows).CopyTo(scales);
        return new PackedTernaryTensor(rows, cols, data, scales);
    }

    private static void VerifyChecksum(int i, string name, string expected, PackedTernaryTensor t)
    {
        if (string.IsNullOrWhiteSpace(expected)) throw new InvalidDataException($"adapter {i}: {name} checksum missing");
        if (!string.Equals(expected, t.DataChecksum(), StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"adapter {i}: {name} tensor checksum mismatch (expected {expected}).");
    }
}
