using System.Globalization;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace DotLLM.Models.Adapters.Ternary;

/// <summary>
/// Manifest of a <c>.nsadapter</c> package (NeuroSymbolicAI <c>Ternary.Core.Packaging.NsAdapterManifest</c>, field for
/// field). Unknown JSON fields are ignored. Defaults use <c>set</c> accessors so the source-generated serializer honours them.
/// </summary>
public sealed class NsAdapterManifest
{
    /// <summary>Target foundation model family (informational).</summary>
    public string TargetModel { get; set; } = string.Empty;
    /// <summary>Transformer layer whose output the adapter is added to.</summary>
    public int TargetLayer { get; set; }
    /// <summary>Hidden dimension D of the residual stream.</summary>
    public int InputDimension { get; set; }
    /// <summary>Rank r of the bottleneck.</summary>
    public int Rank { get; set; }
    /// <summary>Multiplier on the residual delta (reference <c>TernaryExpert.Gate</c>); <c>&lt;= 0</c> disables the adapter.</summary>
    public float Alpha { get; set; } = 1.0f;
    /// <summary>Gate: the adapter contributes only if the largest ReLU'd down-projection value exceeds this (see docs).</summary>
    public float ActivationThreshold { get; set; }
    /// <summary>Domain tag.</summary>
    public string Domain { get; set; } = string.Empty;
    /// <summary>Human-readable description.</summary>
    public string Description { get; set; } = string.Empty;
    /// <summary>Creation timestamp.</summary>
    public DateTimeOffset CreatedAt { get; set; } = DateTimeOffset.UtcNow;
    /// <summary>SHA-256 (hex) of the down tensor's packed data bytes (or of the whole <c>down.bin</c> stream).</summary>
    public string DownChecksum { get; set; } = string.Empty;
    /// <summary>SHA-256 (hex) of the up tensor's packed data bytes (or of the whole <c>up.bin</c> stream).</summary>
    public string UpChecksum { get; set; } = string.Empty;
}

[JsonSourceGenerationOptions(WriteIndented = true, PropertyNameCaseInsensitive = true)]
[JsonSerializable(typeof(NsAdapterManifest))]
internal sealed partial class NsAdapterJsonContext : JsonSerializerContext;

/// <summary>
/// A matrix of ternary weights in the BitNet b1.58 2-bit packing of the reference <c>PackedTernaryTensor</c>:
/// 4 weights per byte, column <c>c</c> at bits <c>2*(c&amp;3)</c> of byte <c>c/4</c>, codes <c>00</c>=0, <c>01</c>=+1,
/// <c>11</c>=-1 (<c>10</c> reserved, read as 0), plus one F32 scale per row. Binary form:
/// <c>int32 rows, int32 cols, float32 scale[rows], byte data[rows*ceil(cols/4)]</c> (little-endian).
/// </summary>
public sealed class PackedTernaryTensor
{
    /// <summary>Rows.</summary>
    public int Rows { get; }
    /// <summary>Columns.</summary>
    public int Columns { get; }
    /// <summary>Bytes per row (<c>ceil(Columns/4)</c>).</summary>
    public int BytesPerRow { get; }
    /// <summary>Packed data, row-major.</summary>
    public byte[] Data { get; }
    /// <summary>Per-row F32 scales.</summary>
    public float[] Scales { get; }

    /// <summary>Creates an all-zero tensor with unit scales.</summary>
    public PackedTernaryTensor(int rows, int columns)
        : this(rows, columns, new byte[checked(rows * ((columns + 3) / 4))], Enumerable.Repeat(1.0f, rows).ToArray()) { }

    /// <summary>Wraps existing data/scales (lengths validated).</summary>
    public PackedTernaryTensor(int rows, int columns, byte[] data, float[] scales)
    {
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (columns <= 0) throw new ArgumentOutOfRangeException(nameof(columns));
        Rows = rows; Columns = columns; BytesPerRow = (columns + 3) / 4;
        if (data.Length != rows * BytesPerRow) throw new ArgumentException($"Data length {data.Length} != {rows * BytesPerRow}.", nameof(data));
        if (scales.Length != rows) throw new ArgumentException($"Scales length {scales.Length} != {rows}.", nameof(scales));
        Data = data; Scales = scales;
    }

    /// <summary>Gets the weight (-1, 0, +1) at (row, col).</summary>
    public sbyte this[int row, int col]
    {
        get => Decode((byte)((Data[row * BytesPerRow + (col >> 2)] >> ((col & 3) * 2)) & 3));
        set
        {
            if (value is < -1 or > 1) throw new ArgumentOutOfRangeException(nameof(value));
            int bi = row * BytesPerRow + (col >> 2), sh = (col & 3) * 2;
            byte code = value == 0 ? (byte)0 : value == 1 ? (byte)1 : (byte)3;
            Data[bi] = (byte)((Data[bi] & ~(3 << sh)) | (code << sh));
        }
    }

    internal static sbyte Decode(byte code) => code switch { 1 => 1, 3 => -1, _ => 0 };

    /// <summary>Writes the reference binary form.</summary>
    public void WriteTo(Stream stream)
    {
        using var w = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        w.Write(Rows);
        w.Write(Columns);
        foreach (float s in Scales) w.Write(s);
        w.Write(Data);
    }

    /// <summary>Reads the reference binary form.</summary>
    public static PackedTernaryTensor ReadFrom(Stream stream)
    {
        using var r = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        int rows = r.ReadInt32(), cols = r.ReadInt32();
        if (rows <= 0 || cols <= 0) throw new InvalidDataException($"Invalid tensor dimensions: {rows}x{cols}.");
        long expected = (long)rows * ((cols + 3) / 4);
        if (expected > int.MaxValue || (long)rows * 4 + expected > stream.Length - stream.Position)
            throw new InvalidDataException("Tensor payload shorter than its header claims.");
        var scales = new float[rows];
        for (int i = 0; i < rows; i++) scales[i] = r.ReadSingle();
        byte[] data = r.ReadBytes((int)expected);
        if (data.Length != expected) throw new EndOfStreamException($"Expected {expected} packed bytes, read {data.Length}.");
        return new PackedTernaryTensor(rows, cols, data, scales);
    }

    /// <summary>SHA-256 (lower-case hex) of the packed data bytes.</summary>
    public string DataChecksum() => Convert.ToHexStringLower(SHA256.HashData(Data));
}

/// <summary>
/// A <c>.nsadapter</c> package: a ZIP with <c>manifest.json</c>, <c>down.bin</c> (rank x D) and <c>up.bin</c> (D x rank),
/// exactly as written by the NeuroSymbolicAI reference. See docs/TERNARY_ADAPTER.md.
/// </summary>
public sealed class NsAdapterPackage
{
    private const string ManifestEntry = "manifest.json";
    private const int MaxManifestBytes = 64 * 1024;

    /// <summary>Package metadata.</summary>
    public NsAdapterManifest Manifest { get; }
    /// <summary>Down projection, <c>Rank x InputDimension</c>.</summary>
    public PackedTernaryTensor Down { get; }
    /// <summary>Up projection, <c>InputDimension x Rank</c>.</summary>
    public PackedTernaryTensor Up { get; }

    /// <summary>Creates a package; dimensions must match the manifest.</summary>
    public NsAdapterPackage(NsAdapterManifest manifest, PackedTernaryTensor down, PackedTernaryTensor up)
    {
        ArgumentNullException.ThrowIfNull(manifest); ArgumentNullException.ThrowIfNull(down); ArgumentNullException.ThrowIfNull(up);
        Manifest = manifest; Down = down; Up = up;
        if (manifest.Rank <= 0) manifest.Rank = down.Rows;
        if (manifest.InputDimension <= 0) manifest.InputDimension = down.Columns;
        ValidateDimensions();
    }

    /// <summary>Throws <see cref="InvalidDataException"/> when manifest and tensor dimensions disagree or values are out of range.</summary>
    public void ValidateDimensions()
    {
        int r = Manifest.Rank, d = Manifest.InputDimension;
        if (r <= 0 || d <= 0) throw new InvalidDataException($"manifest Rank/InputDimension must be positive ({r}, {d}).");
        if (Manifest.TargetLayer < 0) throw new InvalidDataException($"manifest TargetLayer {Manifest.TargetLayer} is negative.");
        if (!float.IsFinite(Manifest.Alpha) || !float.IsFinite(Manifest.ActivationThreshold))
            throw new InvalidDataException("manifest Alpha/ActivationThreshold must be finite.");
        if (Down.Rows != r || Down.Columns != d)
            throw new InvalidDataException($"down tensor {Down.Rows}x{Down.Columns} != Rank x InputDimension {r}x{d}.");
        if (Up.Rows != d || Up.Columns != r)
            throw new InvalidDataException($"up tensor {Up.Rows}x{Up.Columns} != InputDimension x Rank {d}x{r}.");
    }

    /// <summary>Writes the package (checksums are recomputed).</summary>
    public void Save(Stream stream)
    {
        ValidateDimensions();
        Manifest.DownChecksum = Down.DataChecksum();
        Manifest.UpChecksum = Up.DataChecksum();
        using var archive = new ZipArchive(stream, ZipArchiveMode.Create, leaveOpen: true);
        using (var s = archive.CreateEntry(ManifestEntry, CompressionLevel.Optimal).Open())
            JsonSerializer.Serialize(s, Manifest, NsAdapterJsonContext.Default.NsAdapterManifest);
        using (var s = archive.CreateEntry("down.bin", CompressionLevel.Optimal).Open()) Down.WriteTo(s);
        using (var s = archive.CreateEntry("up.bin", CompressionLevel.Optimal).Open()) Up.WriteTo(s);
    }

    /// <summary>Writes the package to a file.</summary>
    public void Save(string path)
    {
        using var fs = File.Create(path);
        Save(fs);
    }

    /// <summary>Loads and fully validates a package from a file.</summary>
    public static NsAdapterPackage Load(string path)
    {
        using var fs = File.OpenRead(path);
        return Load(fs);
    }

    /// <summary>
    /// Loads and validates a package: required entries (legacy <c>weights_down.bin</c>/<c>weights_up.bin</c> names accepted),
    /// size bounds derived from the manifest before any payload is read, dimension checks, SHA-256 verification.
    /// Throws <see cref="InvalidDataException"/> on any corruption or mismatch.
    /// </summary>
    public static NsAdapterPackage Load(Stream stream)
    {
        ZipArchive archive;
        try { archive = new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen: true); }
        catch (InvalidDataException ex) { throw new InvalidDataException("not a valid .nsadapter (zip) archive: " + ex.Message, ex); }
        using (archive)
        {
            var me = archive.GetEntry(ManifestEntry) ?? throw new InvalidDataException("missing manifest.json");
            if (me.Length > MaxManifestBytes) throw new InvalidDataException("manifest.json too large");
            NsAdapterManifest manifest;
            try
            {
                using var ms = me.Open();
                manifest = JsonSerializer.Deserialize(ms, NsAdapterJsonContext.Default.NsAdapterManifest)
                    ?? throw new InvalidDataException("empty manifest");
            }
            catch (JsonException ex) { throw new InvalidDataException("manifest.json is not valid: " + ex.Message, ex); }

            int r = manifest.Rank, d = manifest.InputDimension;
            if (r <= 0 || d <= 0 || r > 4096 || d > 1 << 20)
                throw new InvalidDataException($"manifest Rank/InputDimension out of range ({r}, {d}).");

            var (down, downRaw) = ReadTensor(archive, "down.bin", "weights_down.bin", r, d);
            var (up, upRaw) = ReadTensor(archive, "up.bin", "weights_up.bin", d, r);
            if (down.Rows != r || down.Columns != d) throw new InvalidDataException($"down tensor {down.Rows}x{down.Columns} != {r}x{d}.");
            if (up.Rows != d || up.Columns != r) throw new InvalidDataException($"up tensor {up.Rows}x{up.Columns} != {d}x{r}.");
            VerifyChecksum("down", manifest.DownChecksum, down, downRaw);
            VerifyChecksum("up", manifest.UpChecksum, up, upRaw);
            var pkg = new NsAdapterPackage(manifest, down, up);
            return pkg;
        }
    }

    /// <summary>
    /// The reference's GGUF-compatible metadata export (<c>NsAdapterPackage.ToGgufMetadata</c>), same keys and formatting.
    /// </summary>
    public Dictionary<string, string> ToGgufMetadata() => new(StringComparer.OrdinalIgnoreCase)
    {
        ["neuro_symbolic.domain"] = Manifest.Domain,
        ["neuro_symbolic.target_model"] = Manifest.TargetModel,
        ["neuro_symbolic.layer"] = Manifest.TargetLayer.ToString(CultureInfo.InvariantCulture),
        ["neuro_symbolic.target_layer"] = Manifest.TargetLayer.ToString(CultureInfo.InvariantCulture),
        ["neuro_symbolic.input_dimension"] = Manifest.InputDimension.ToString(CultureInfo.InvariantCulture),
        ["neuro_symbolic.rank"] = Manifest.Rank.ToString(CultureInfo.InvariantCulture),
        ["neuro_symbolic.alpha"] = Manifest.Alpha.ToString("G7", CultureInfo.InvariantCulture),
        ["neuro_symbolic.activation_threshold"] = Manifest.ActivationThreshold.ToString("G7", CultureInfo.InvariantCulture),
        ["neuro_symbolic.description"] = Manifest.Description,
        ["neuro_symbolic.created_at"] = Manifest.CreatedAt.ToString("O", CultureInfo.InvariantCulture),
        ["neuro_symbolic.down_checksum"] = Manifest.DownChecksum,
        ["neuro_symbolic.up_checksum"] = Manifest.UpChecksum,
    };

    private static (PackedTernaryTensor tensor, byte[] raw) ReadTensor(ZipArchive archive, string name, string legacy, int rows, int cols)
    {
        var e = archive.GetEntry(name) ?? archive.GetEntry(legacy) ?? throw new InvalidDataException($"missing {name}");
        long bytesPerRow = (cols + 3) / 4;
        long maxLen = 8 + 4L * rows + bytesPerRow * rows;   // header + scales + data
        if (e.Length > maxLen) throw new InvalidDataException($"{name} is {e.Length} bytes, expected at most {maxLen}");
        byte[] raw;
        using (var s = e.Open())
        using (var ms = new MemoryStream())
        {
            s.CopyTo(ms);
            raw = ms.ToArray();
        }
        try
        {
            using var ms = new MemoryStream(raw);
            return (PackedTernaryTensor.ReadFrom(ms), raw);
        }
        catch (Exception ex) when (ex is InvalidDataException or EndOfStreamException)
        {
            // Reference fallback: headerless raw packed bytes with unit scales.
            if (raw.Length == bytesPerRow * rows)
                return (new PackedTernaryTensor(rows, cols, raw, Enumerable.Repeat(1.0f, rows).ToArray()), raw);
            throw new InvalidDataException($"{name} is not a valid packed ternary tensor: {ex.Message}", ex);
        }
    }

    private static void VerifyChecksum(string name, string expected, PackedTernaryTensor t, byte[] raw)
    {
        if (string.IsNullOrWhiteSpace(expected)) throw new InvalidDataException($"{name} checksum missing in manifest");
        string a = t.DataChecksum(), b = Convert.ToHexStringLower(SHA256.HashData(raw));
        if (!string.Equals(expected, a, StringComparison.OrdinalIgnoreCase) && !string.Equals(expected, b, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"{name} tensor checksum mismatch (expected {expected}).");
    }
}
