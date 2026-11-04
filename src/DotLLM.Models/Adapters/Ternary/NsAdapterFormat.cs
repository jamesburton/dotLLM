using System.Buffers.Binary;
using System.Globalization;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Models.Adapters.Ternary;

/// <summary>Managed description of one adapted layer, used by <see cref="NsAdapterFile.Write"/> and tests.</summary>
/// <param name="Layer">Zero-based transformer layer whose output the circuit is added to.</param>
/// <param name="Rank">Rank r (K_i) of this layer's circuit.</param>
/// <param name="Alpha">Scalar gain applied to <c>sigma(W_down x)</c> per rank row.</param>
/// <param name="Down">Row-major <c>[rank x d_model]</c> ternary values in {-1,0,+1}.</param>
/// <param name="UpT">Row-major <c>[rank x d_model]</c> ternary values: row j is column j of W_up (d_model x rank).</param>
public sealed record TernaryLayerSpec(int Layer, int Rank, float Alpha, sbyte[] Down, sbyte[] UpT);

/// <summary>
/// Reader/writer for the standalone <c>.nsadapter</c> v0 package (docs/TERNARY_ADAPTER.md):
/// 64-byte binary header, a YAML-subset manifest, then a 64-byte-aligned payload of 2-bit packed
/// plus/minus bitmasks. The payload is memory-mapped and read in place.
/// </summary>
public static unsafe class NsAdapterFile
{
    private const int HeaderSize = 64;
    private static readonly byte[] Magic = "NSAD"u8.ToArray();

    /// <summary>Packs a row-major ternary matrix into the plus/minus bitmask layout.</summary>
    public static ulong[] Pack(ReadOnlySpan<sbyte> values, int rows, int cols)
    {
        if (values.Length != rows * cols) throw new ArgumentException("values.Length != rows*cols.");
        int wpr = TernaryAdapterKernel.WordsPerRow(cols);
        var words = new ulong[(long)rows * wpr];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
            {
                sbyte v = values[r * cols + c];
                if (v is < -1 or > 1) throw new ArgumentException($"Non-ternary value {v} at ({r},{c}).");
                if (v == 0) continue;
                int idx = r * wpr + (c >> 6) * 2 + (v < 0 ? 1 : 0);
                words[idx] |= 1UL << (c & 63);
            }
        return words;
    }

    /// <summary>Writes a <c>.nsadapter</c> package.</summary>
    public static void Write(string path, string name, int dModel, bool relu, IReadOnlyList<TernaryLayerSpec> layers)
    {
        ArgumentNullException.ThrowIfNull(layers);
        if (dModel <= 0) throw new ArgumentException("d_model must be positive.");
        int wpr = TernaryAdapterKernel.WordsPerRow(dModel);
        using var payload = new MemoryStream();
        var sb = new StringBuilder();
        sb.Append("format: nsadapter\nformat_version: 0\n");
        sb.Append(CultureInfo.InvariantCulture, $"name: {name}\n");
        sb.Append(CultureInfo.InvariantCulture, $"d_model: {dModel}\n");
        sb.Append(relu ? "sigma: relu\n" : "sigma: identity\n");
        sb.Append("dtype: ternary-bitmask-2bit\nblock_cols: 64\nmask_order: plus,minus\nlayers:\n");
        foreach (var l in layers)
        {
            if (l.Rank <= 0) throw new ArgumentException($"Layer {l.Layer}: rank must be positive.");
            if (l.Down.Length != l.Rank * dModel || l.UpT.Length != l.Rank * dModel)
                throw new ArgumentException($"Layer {l.Layer}: matrices must be rank*d_model long.");
            Align64(payload);
            long downOff = payload.Position;
            WriteWords(payload, Pack(l.Down, l.Rank, dModel));
            Align64(payload);
            long upOff = payload.Position;
            WriteWords(payload, Pack(l.UpT, l.Rank, dModel));
            sb.Append(CultureInfo.InvariantCulture, $"  - layer: {l.Layer}\n    rank: {l.Rank}\n    alpha: {l.Alpha.ToString("R", CultureInfo.InvariantCulture)}\n");
            sb.Append(CultureInfo.InvariantCulture, $"    down_offset: {downOff}\n    up_offset: {upOff}\n");
        }
        Align64(payload);
        byte[] manifest = Encoding.UTF8.GetBytes(sb.ToString());
        long payloadOffset = (HeaderSize + manifest.Length + 63) / 64 * 64;
        byte[] payloadBytes = payload.ToArray();
        byte[] header = new byte[HeaderSize];
        Magic.CopyTo(header, 0);
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(4), 0);       // version
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(6), 0);       // flags
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(8), (uint)manifest.Length);
        BinaryPrimitives.WriteUInt64LittleEndian(header.AsSpan(16), (ulong)payloadOffset);
        BinaryPrimitives.WriteUInt64LittleEndian(header.AsSpan(24), (ulong)payloadBytes.Length);
        SHA256.HashData(payloadBytes).CopyTo(header, 32);
        using var fs = File.Create(path);
        fs.Write(header);
        fs.Write(manifest);
        fs.Write(new byte[payloadOffset - HeaderSize - manifest.Length]);
        fs.Write(payloadBytes);
    }

    /// <summary>
    /// Opens a package. The payload is memory-mapped; the returned adapter owns the mapping.
    /// </summary>
    /// <param name="path">Path to the <c>.nsadapter</c> file.</param>
    /// <param name="verifyChecksum">Verify the SHA-256 of the payload (cheap: adapters are KB-MB).</param>
    /// <param name="isa">Kernel ISA (tests).</param>
    public static TernaryResidualAdapter Open(string path, bool verifyChecksum = true,
        TernaryKernelIsa isa = TernaryKernelIsa.Auto)
    {
        long fileLen = new FileInfo(path).Length;
        if (fileLen < HeaderSize) throw new InvalidDataException(".nsadapter: file shorter than header.");
        var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        MemoryMappedViewAccessor? view = null;
        byte* basePtr = null;
        try
        {
            view = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            view.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);
            basePtr += view.PointerOffset;
            var owner = new MappedOwner(mmf, view);
            try
            {
                return Build(path, basePtr, fileLen, owner, verifyChecksum, isa);
            }
            catch
            {
                owner.Dispose();
                throw;
            }
        }
        catch
        {
            if (view is null) mmf.Dispose();
            throw;
        }
    }

    private static TernaryResidualAdapter Build(string path, byte* basePtr, long fileLen, MappedOwner owner,
        bool verifyChecksum, TernaryKernelIsa isa)
    {
        var hdr = new ReadOnlySpan<byte>(basePtr, HeaderSize);
        if (!hdr[..4].SequenceEqual(Magic)) throw new InvalidDataException(".nsadapter: bad magic.");
        ushort version = BinaryPrimitives.ReadUInt16LittleEndian(hdr[4..]);
        if (version != 0) throw new InvalidDataException($".nsadapter: unsupported version {version}.");
        uint manLen = BinaryPrimitives.ReadUInt32LittleEndian(hdr[8..]);
        ulong payOff = BinaryPrimitives.ReadUInt64LittleEndian(hdr[16..]);
        ulong payLen = BinaryPrimitives.ReadUInt64LittleEndian(hdr[24..]);
        if (payOff % 64 != 0 || payOff < HeaderSize + manLen || payOff > (ulong)fileLen
            || payLen > (ulong)fileLen - payOff || HeaderSize + (long)manLen > fileLen)
            throw new InvalidDataException(".nsadapter: header offsets out of range.");
        byte* payload = basePtr + payOff;
        if (verifyChecksum)
        {
            Span<byte> actual = stackalloc byte[32];
            SHA256.HashData(new ReadOnlySpan<byte>(payload, checked((int)payLen)), actual);
            if (!actual.SequenceEqual(hdr.Slice(32, 32)))
                throw new InvalidDataException(".nsadapter: payload checksum mismatch.");
        }

        var m = Manifest.Parse(Encoding.UTF8.GetString(basePtr + HeaderSize, (int)manLen));
        if (m.Get("format") != "nsadapter") throw new InvalidDataException("manifest: format != nsadapter.");
        if (m.Get("format_version") != "0") throw new InvalidDataException("manifest: unsupported format_version.");
        if (m.Get("dtype") != "ternary-bitmask-2bit") throw new InvalidDataException("manifest: unsupported dtype.");
        if (m.Get("block_cols") != "64") throw new InvalidDataException("manifest: block_cols must be 64.");
        if (m.Get("mask_order") != "plus,minus") throw new InvalidDataException("manifest: mask_order must be plus,minus.");
        int d = m.GetInt("d_model");
        if (d <= 0) throw new InvalidDataException("manifest: d_model must be positive.");
        string sigma = m.Get("sigma");
        if (sigma is not ("relu" or "identity")) throw new InvalidDataException($"manifest: unknown sigma '{sigma}'.");
        if (m.Layers.Count == 0) throw new InvalidDataException("manifest: no layers.");

        int wpr = TernaryAdapterKernel.WordsPerRow(d);
        var layers = new List<TernaryLayerData>();
        var natives = new List<nint>();
        try
        {
            foreach (var le in m.Layers)
            {
                int layer = le.GetInt("layer");
                int rank = le.GetInt("rank");
                float alpha = float.Parse(le.Get("alpha"), NumberStyles.Float, CultureInfo.InvariantCulture);
                long downOff = le.GetLong("down_offset");
                long upOff = le.GetLong("up_offset");
                long bytes = (long)rank * wpr * 8;
                if (layer < 0 || rank <= 0 || !float.IsFinite(alpha))
                    throw new InvalidDataException($"manifest: bad layer entry (layer={layer}, rank={rank}, alpha={alpha}).");
                foreach (long off in new[] { downOff, upOff })
                    if (off < 0 || off % 64 != 0 || off + bytes > (long)payLen)
                        throw new InvalidDataException($"manifest: layer {layer} offset {off} out of payload.");
                ulong* down = (ulong*)(payload + downOff);
                ulong* up = (ulong*)(payload + upOff);
                ValidateMasks(down, rank, d, wpr, layer, "down");
                ValidateMasks(up, rank, d, wpr, layer, "up");
                float* ra = (float*)NativeMemory.AlignedAlloc((nuint)rank * sizeof(float), 64);
                natives.Add((nint)ra);
                for (int j = 0; j < rank; j++) ra[j] = alpha;
                layers.Add(new TernaryLayerData(layer, rank, down, up, ra));
            }
            return new TernaryResidualAdapter(m.Get("name"), d, sigma == "relu", layers.ToArray(),
                new List<IDisposable> { owner }, natives, isa);
        }
        catch
        {
            foreach (nint b in natives) NativeMemory.AlignedFree((void*)b);
            throw;
        }
    }

    private static void ValidateMasks(ulong* w, int rows, int cols, int wpr, int layer, string which)
    {
        int blocks = wpr / 2;
        int tailBits = cols & 63;
        ulong validLast = tailBits == 0 ? ulong.MaxValue : (1UL << tailBits) - 1;
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocks; b++)
            {
                ulong p = w[(long)r * wpr + b * 2], mi = w[(long)r * wpr + b * 2 + 1];
                if ((p & mi) != 0)
                    throw new InvalidDataException($"layer {layer} {which}: row {r} has a weight that is both +1 and -1.");
                if (b == blocks - 1 && ((p | mi) & ~validLast) != 0)
                    throw new InvalidDataException($"layer {layer} {which}: row {r} has mask bits beyond d_model.");
            }
    }

    private static void Align64(MemoryStream s)
    {
        long pad = (64 - s.Position % 64) % 64;
        if (pad != 0) s.Write(new byte[pad]);
    }

    private static void WriteWords(MemoryStream s, ulong[] words)
    {
        Span<byte> buf = stackalloc byte[8];
        foreach (ulong w in words)
        {
            BinaryPrimitives.WriteUInt64LittleEndian(buf, w);
            s.Write(buf);
        }
    }

    private sealed class MappedOwner : IDisposable
    {
        private readonly MemoryMappedFile _mmf;
        private readonly MemoryMappedViewAccessor _view;
        public MappedOwner(MemoryMappedFile mmf, MemoryMappedViewAccessor view) { _mmf = mmf; _view = view; }
        public void Dispose()
        {
            _view.SafeMemoryMappedViewHandle.ReleasePointer();
            _view.Dispose();
            _mmf.Dispose();
        }
    }

    /// <summary>Parsed manifest (restricted YAML: top-level scalars + a <c>layers:</c> block sequence of flat maps).</summary>
    internal sealed class Manifest
    {
        private readonly Dictionary<string, string> _top = new(StringComparer.Ordinal);
        public List<Entry> Layers { get; } = new();

        public string Get(string key) => _top.TryGetValue(key, out var v) ? v : throw new InvalidDataException($"manifest: missing '{key}'.");
        public int GetInt(string key) => int.Parse(Get(key), CultureInfo.InvariantCulture);

        public static Manifest Parse(string text)
        {
            var m = new Manifest();
            Entry? cur = null;
            bool inLayers = false;
            foreach (string raw in text.Split('\n'))
            {
                string line = raw.TrimEnd('\r');
                string t = line.Trim();
                if (t.Length == 0 || t[0] == '#') continue;
                if (!char.IsWhiteSpace(line[0]))
                {
                    inLayers = false;
                    (string k, string v) = KeyValue(t);
                    if (k == "layers") { inLayers = true; continue; }
                    m._top[k] = v;
                    continue;
                }
                if (!inLayers) throw new InvalidDataException($"manifest: unexpected indented line '{t}'.");
                if (t.StartsWith("- ", StringComparison.Ordinal))
                {
                    cur = new Entry();
                    m.Layers.Add(cur);
                    t = t[2..].Trim();
                }
                if (cur is null) throw new InvalidDataException("manifest: layer field before '- '.");
                (string ek, string ev) = KeyValue(t);
                cur.Fields[ek] = ev;
            }
            return m;
        }

        private static (string, string) KeyValue(string t)
        {
            int i = t.IndexOf(':');
            if (i <= 0) throw new InvalidDataException($"manifest: cannot parse '{t}'.");
            string v = t[(i + 1)..].Trim();
            if (v.Length >= 2 && v[0] == '"' && v[^1] == '"') v = v[1..^1];
            return (t[..i].Trim(), v);
        }

        internal sealed class Entry
        {
            public Dictionary<string, string> Fields { get; } = new(StringComparer.Ordinal);
            public string Get(string key) => Fields.TryGetValue(key, out var v) ? v : throw new InvalidDataException($"manifest layer: missing '{key}'.");
            public int GetInt(string key) => int.Parse(Get(key), CultureInfo.InvariantCulture);
            public long GetLong(string key) => long.Parse(Get(key), CultureInfo.InvariantCulture);
        }
    }
}
