using System.Buffers.Binary;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Text;

namespace DotLLM.ChunkReuse.Store;

/// <summary>
/// <see cref="IChunkStore"/> on a directory: one file per entry at <c>root/&lt;hex[0..2]&gt;/&lt;hex&gt;.ckv</c>.
/// <para>
/// File layout (little-endian): magic <c>CKVS</c>, int32 <see cref="FormatVersion"/>, int32 <see cref="KvDtype"/>, the 64-byte
/// ASCII key, int32 length + UTF-8 model fingerprint, int32 token count, int32 layer count, per layer int32 K and V element
/// counts, int64 payload byte count; then the payload: token ids (int32), K of every layer, V of every layer, each in the
/// entry's dtype. <see cref="ChunkKvEntry"/> holds F32 rows (the CPU oracle cache's native dtype), so entries are written as
/// <see cref="KvDtype.F32"/> without conversion; other dtypes are reserved in the header and read as misses until supported.
/// </para>
/// <para>
/// Writes go to a unique temp file in the shard directory, are flushed to disk and then renamed over the final name, so a crash
/// or a concurrent writer never leaves a partial <c>.ckv</c> at the final path. Reads memory-map the file and validate the
/// header and the exact file length; any mismatch (truncation, trailing bytes, other version, other key) is a miss.
/// </para>
/// </summary>
public sealed class FileChunkStore : IChunkStore
{
    /// <summary>On-disk format version; a file with any other version is a miss.</summary>
    public const int FormatVersion = 1;

    private const string Extension = ".ckv";
    private const int MaxFingerprintBytes = 4096;
    private const int MaxLayers = 4096;
    private const int MaxHeaderBytes = 16 + 64 + 4 + MaxFingerprintBytes + 8 + 8 * MaxLayers + 8;
    private static ReadOnlySpan<byte> Magic => "CKVS"u8;

    private readonly string _root;

    /// <summary>Opens (creating if needed) a store rooted at <paramref name="root"/>.</summary>
    /// <param name="root">Store directory.</param>
    public FileChunkStore(string root)
    {
        ArgumentException.ThrowIfNullOrEmpty(root);
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException("FileChunkStore writes raw little-endian rows; big-endian hosts are not supported.");
        _root = Path.GetFullPath(root);
        Directory.CreateDirectory(_root);
    }

    /// <summary>The store directory.</summary>
    public string Root => _root;

    /// <inheritdoc/>
    public long TotalBytes
    {
        get
        {
            long total = 0;
            foreach ((_, string path) in EnumerateEntries())
            {
                try { total += new FileInfo(path).Length; }
                catch (IOException) { /* removed meanwhile */ }
            }
            return total;
        }
    }

    /// <summary>The path an entry for <paramref name="key"/> lives at.</summary>
    /// <param name="key">Chunk key.</param>
    /// <returns>The path.</returns>
    public string PathOf(ChunkKey key)
    {
        if (!ChunkKey.IsWellFormed(key.Hex)) throw new ArgumentException($"Malformed chunk key '{key.Hex}'.", nameof(key));
        return Path.Combine(_root, key.Hex[..2], key.Hex + Extension);
    }

    /// <inheritdoc/>
    public bool Contains(ChunkKey key)
    {
        string path = PathOf(key);
        try
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete);
            long length = fs.Length;
            byte[] head = new byte[(int)Math.Min(length, MaxHeaderBytes)];
            fs.ReadExactly(head);
            return TryParseHeader(head, length, key, out _);
        }
        catch (IOException) { return false; }
        catch (UnauthorizedAccessException) { return false; }
    }

    /// <inheritdoc/>
    public bool TryGet(ChunkKey key, out ChunkKvEntry entry)
    {
        entry = null!;
        string path = PathOf(key);
        try
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete);
            long length = fs.Length;
            if (length == 0) return false; // an empty file cannot be mapped
            using var map = MemoryMappedFile.CreateFromFile(fs, null, 0, MemoryMappedFileAccess.Read, HandleInheritability.None, leaveOpen: true);
            using var view = map.CreateViewAccessor(0, length, MemoryMappedFileAccess.Read);

            byte[] head = new byte[(int)Math.Min(length, MaxHeaderBytes)];
            view.ReadArray(0, head, 0, head.Length);
            if (!TryParseHeader(head, length, key, out Header h)) return false;

            long offset = h.HeaderBytes;
            int[] tokens = new int[h.TokenCount];
            view.ReadArray(offset, tokens, 0, tokens.Length);
            offset += (long)tokens.Length * sizeof(int);
            var keys = new float[h.LayerCount][];
            for (int l = 0; l < h.LayerCount; l++)
            {
                keys[l] = new float[h.KLengths[l]];
                view.ReadArray(offset, keys[l], 0, keys[l].Length);
                offset += (long)keys[l].Length * sizeof(float);
            }
            var values = new float[h.LayerCount][];
            for (int l = 0; l < h.LayerCount; l++)
            {
                values[l] = new float[h.VLengths[l]];
                view.ReadArray(offset, values[l], 0, values[l].Length);
                offset += (long)values[l].Length * sizeof(float);
            }
            entry = new ChunkKvEntry(h.Fingerprint, tokens, keys, values);
            return true;
        }
        catch (IOException) { return false; }
        catch (UnauthorizedAccessException) { return false; }
    }

    /// <inheritdoc/>
    public void Put(ChunkKey key, ChunkKvEntry entry)
    {
        ArgumentNullException.ThrowIfNull(entry);
        Validate(entry);
        string path = PathOf(key);
        if (Contains(key)) return; // content-addressed: same key, same content

        string dir = Path.GetDirectoryName(path)!;
        Directory.CreateDirectory(dir);
        string temp = Path.Combine(dir, $"{key.Hex}.{Guid.NewGuid():N}.tmp");
        try
        {
            WriteEntry(temp, key, entry);
            for (int attempt = 0; ; attempt++)
            {
                try
                {
                    File.Move(temp, path, overwrite: true);
                    return;
                }
                catch (Exception e) when (e is IOException or UnauthorizedAccessException)
                {
                    // On Windows the destination cannot be replaced while a reader has it open; another writer's identical
                    // entry is as good as ours.
                    if (Contains(key)) return;
                    if (attempt >= 20) throw;
                    Thread.Sleep(5);
                }
            }
        }
        finally
        {
            try { File.Delete(temp); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }
    }

    /// <inheritdoc/>
    public void Evict(ChunkKey key)
    {
        try { File.Delete(PathOf(key)); }
        catch (DirectoryNotFoundException) { }
    }

    /// <inheritdoc/>
    public IEnumerable<ChunkKey> Keys() => EnumerateEntries().Select(e => e.Key).ToList();

    private List<(ChunkKey Key, string Path)> EnumerateEntries()
    {
        var result = new List<(ChunkKey, string)>();
        if (!Directory.Exists(_root)) return result;
        foreach (string shard in Directory.EnumerateDirectories(_root))
        {
            string shardName = Path.GetFileName(shard);
            if (shardName.Length != 2) continue;
            foreach (string file in Directory.EnumerateFiles(shard, "*" + Extension))
            {
                // "*.ckv" also matches longer extensions on Windows; require the exact one.
                if (!string.Equals(Path.GetExtension(file), Extension, StringComparison.Ordinal)) continue;
                string hex = Path.GetFileNameWithoutExtension(file);
                if (ChunkKey.IsWellFormed(hex) && hex.StartsWith(shardName, StringComparison.Ordinal))
                    result.Add((new ChunkKey(hex), file));
            }
        }
        return result;
    }

    private static void Validate(ChunkKvEntry entry)
    {
        ArgumentNullException.ThrowIfNull(entry.ModelFingerprint);
        ArgumentNullException.ThrowIfNull(entry.TokenIds);
        ArgumentNullException.ThrowIfNull(entry.KeysPerLayer);
        ArgumentNullException.ThrowIfNull(entry.ValuesPerLayer);
        int n = entry.TokenIds.Length;
        if (n == 0) throw new ArgumentException("An entry must have at least one token.", nameof(entry));
        int layers = entry.KeysPerLayer.Length;
        if (layers == 0 || layers > MaxLayers || entry.ValuesPerLayer.Length != layers)
            throw new ArgumentException($"Entry has {layers} K layers and {entry.ValuesPerLayer.Length} V layers.", nameof(entry));
        if (Encoding.UTF8.GetByteCount(entry.ModelFingerprint) > MaxFingerprintBytes)
            throw new ArgumentException("Model fingerprint is too long.", nameof(entry));
        for (int l = 0; l < layers; l++)
        {
            if (entry.KeysPerLayer[l] is not { } k || entry.ValuesPerLayer[l] is not { } v || k.Length % n != 0 || v.Length % n != 0)
                throw new ArgumentException($"Layer {l}: K/V lengths must be multiples of the token count {n}.", nameof(entry));
        }
    }

    private static void WriteEntry(string path, ChunkKey key, ChunkKvEntry entry)
    {
        int layers = entry.KeysPerLayer.Length;
        long payload = (long)entry.TokenIds.Length * sizeof(int);
        for (int l = 0; l < layers; l++)
            payload += ((long)entry.KeysPerLayer[l].Length + entry.ValuesPerLayer[l].Length) * sizeof(float);

        byte[] fp = Encoding.UTF8.GetBytes(entry.ModelFingerprint);
        var header = new byte[16 + 64 + 4 + fp.Length + 8 + 8 * layers + 8];
        var w = new SpanWriter(header);
        w.Bytes(Magic);
        w.Int32(FormatVersion);
        w.Int32((int)KvDtype.F32);
        w.Int32(0); // reserved
        w.Bytes(Encoding.ASCII.GetBytes(key.Hex));
        w.Int32(fp.Length);
        w.Bytes(fp);
        w.Int32(entry.TokenIds.Length);
        w.Int32(layers);
        for (int l = 0; l < layers; l++)
        {
            w.Int32(entry.KeysPerLayer[l].Length);
            w.Int32(entry.ValuesPerLayer[l].Length);
        }
        w.Int64(payload);

        using var fs = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None, 1 << 16);
        fs.SetLength(header.Length + payload);
        fs.Write(header);
        fs.Write(MemoryMarshal.AsBytes(entry.TokenIds.AsSpan()));
        for (int l = 0; l < layers; l++) fs.Write(MemoryMarshal.AsBytes(entry.KeysPerLayer[l].AsSpan()));
        for (int l = 0; l < layers; l++) fs.Write(MemoryMarshal.AsBytes(entry.ValuesPerLayer[l].AsSpan()));
        fs.Flush(flushToDisk: true);
    }

    private static bool TryParseHeader(ReadOnlySpan<byte> head, long fileLength, ChunkKey key, out Header header)
    {
        header = default;
        var r = new SpanReader(head);
        if (!r.Bytes(4, out ReadOnlySpan<byte> magic) || !magic.SequenceEqual(Magic)) return false;
        if (!r.Int32(out int version) || version != FormatVersion) return false;
        if (!r.Int32(out int dtype) || dtype != (int)KvDtype.F32) return false;
        if (!r.Int32(out _)) return false;
        if (!r.Bytes(64, out ReadOnlySpan<byte> hex) || !hex.SequenceEqual(Encoding.ASCII.GetBytes(key.Hex))) return false;
        if (!r.Int32(out int fpLen) || fpLen < 0 || fpLen > MaxFingerprintBytes) return false;
        if (!r.Bytes(fpLen, out ReadOnlySpan<byte> fpBytes)) return false;
        if (!r.Int32(out int tokens) || tokens <= 0) return false;
        if (!r.Int32(out int layers) || layers <= 0 || layers > MaxLayers) return false;
        var kLens = new int[layers];
        var vLens = new int[layers];
        long expected = (long)tokens * sizeof(int);
        for (int l = 0; l < layers; l++)
        {
            if (!r.Int32(out kLens[l]) || !r.Int32(out vLens[l])) return false;
            if (kLens[l] < 0 || vLens[l] < 0 || kLens[l] % tokens != 0 || vLens[l] % tokens != 0) return false;
            expected += ((long)kLens[l] + vLens[l]) * sizeof(float);
        }
        if (!r.Int64(out long payload) || payload != expected) return false;
        if (r.Position + payload != fileLength) return false; // truncated or trailing bytes

        header = new Header(Encoding.UTF8.GetString(fpBytes), tokens, layers, kLens, vLens, r.Position);
        return true;
    }

    private readonly record struct Header(string Fingerprint, int TokenCount, int LayerCount, int[] KLengths, int[] VLengths, int HeaderBytes);

    private ref struct SpanWriter(Span<byte> buffer)
    {
        private readonly Span<byte> _buffer = buffer;
        private int _pos;

        public void Bytes(ReadOnlySpan<byte> src) { src.CopyTo(_buffer[_pos..]); _pos += src.Length; }
        public void Int32(int v) { BinaryPrimitives.WriteInt32LittleEndian(_buffer[_pos..], v); _pos += 4; }
        public void Int64(long v) { BinaryPrimitives.WriteInt64LittleEndian(_buffer[_pos..], v); _pos += 8; }
    }

    private ref struct SpanReader(ReadOnlySpan<byte> buffer)
    {
        private readonly ReadOnlySpan<byte> _buffer = buffer;

        public int Position { get; private set; }

        public bool Bytes(int count, out ReadOnlySpan<byte> bytes)
        {
            bytes = default;
            if (count < 0 || _buffer.Length - Position < count) return false;
            bytes = _buffer.Slice(Position, count);
            Position += count;
            return true;
        }

        public bool Int32(out int v)
        {
            v = 0;
            if (!Bytes(4, out ReadOnlySpan<byte> b)) return false;
            v = BinaryPrimitives.ReadInt32LittleEndian(b);
            return true;
        }

        public bool Int64(out long v)
        {
            v = 0;
            if (!Bytes(8, out ReadOnlySpan<byte> b)) return false;
            v = BinaryPrimitives.ReadInt64LittleEndian(b);
            return true;
        }
    }
}
