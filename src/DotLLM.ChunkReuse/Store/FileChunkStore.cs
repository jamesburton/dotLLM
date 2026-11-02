using System.Buffers.Binary;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Text;

namespace DotLLM.ChunkReuse.Store;

/// <summary>
/// <see cref="IChunkStore"/> on a directory: one file per entry at <c>root/&lt;hex[0..2]&gt;/&lt;hex&gt;.ckv</c>.
/// <para>
/// File layout (little-endian): magic <c>CKVS</c>, int32 <see cref="FormatVersion"/>, int32 <see cref="KvDtype"/>, the 64-byte
/// ASCII key, int32 length + UTF-8 model fingerprint, int32 length + ASCII backend tag (<see cref="ChunkKvEntry.Backend"/>; since
/// format version 2), int32 token count, int32 layer count, per layer int32 K and V element counts, int64 payload byte count; then the payload: token ids (int32), K of every layer, V of every layer, each in the
/// entry's dtype. <see cref="ChunkKvEntry"/> holds F32 rows (the CPU oracle cache's native dtype), so entries are written as
/// <see cref="KvDtype.F32"/> without conversion; other dtypes are reserved in the header and read as misses until supported.
/// </para>
/// <para>
/// Writes go to a unique temp file in the shard directory, are flushed to disk and then renamed over the final name, so a crash
/// or a concurrent writer never leaves a partial <c>.ckv</c> at the final path. A crashed writer can leave its temp file behind;
/// <see cref="SweepTempFiles"/> removes old ones. Reads memory-map the file and validate the
/// header and the exact file length; any mismatch (truncation, trailing bytes, other version, other key, malformed backend tag)
/// is a miss. Version-1 files (no backend tag) are misses: the backend that produced them is unknown.
/// </para>
/// <para>
/// The backend tag is reported as written: the store cannot tell which backend a key was computed for, so a reader compares
/// <see cref="ChunkKvEntry.Backend"/> with its own backend (<see cref="StoredContextAssembler"/> rejects and replaces a mismatch).
/// CPU and Vulkan entries never mix: the backend is in the key and in the header.
/// </para>
/// </summary>
public sealed class FileChunkStore : IChunkStore
{
    /// <summary>On-disk format version (2: the header carries the backend tag); a file with any other version is a miss.</summary>
    public const int FormatVersion = 2;

    /// <summary>Longest backend tag, in bytes.</summary>
    public const int MaxBackendTagBytes = 64;

    private const string Extension = ".ckv";
    private const string TempExtension = ".tmp";
    private const int MaxFingerprintBytes = 4096;
    private const int MaxLayers = 4096;
    private const int MaxHeaderBytes = 16 + 64 + 4 + MaxFingerprintBytes + 4 + MaxBackendTagBytes + 8 + 8 * MaxLayers + 8;
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

    /// <summary>Header size of an entry with this fingerprint, backend tag and layer count; the file is this plus the payload (the token
    /// ids and the F32 K and V rows).</summary>
    /// <param name="fingerprint">Model fingerprint.</param>
    /// <param name="backend">Backend tag.</param>
    /// <param name="layers">KV layer count.</param>
    /// <returns>The header size in bytes.</returns>
    public static long HeaderBytes(string fingerprint, string backend, int layers)
    {
        ArgumentNullException.ThrowIfNull(fingerprint);
        ArgumentNullException.ThrowIfNull(backend);
        return 16 + 64 + 4 + Encoding.UTF8.GetByteCount(fingerprint) + 4 + Encoding.UTF8.GetByteCount(backend) + 8 + 8L * layers + 8;
    }

    /// <summary>True when <paramref name="tag"/> can be stored as a backend tag: 1 to <see cref="MaxBackendTagBytes"/> lowercase
    /// ASCII letters, digits, <c>.</c>, <c>_</c> or <c>-</c> (e.g. <c>cpu</c>, <c>vulkan</c>).</summary>
    /// <param name="tag">Candidate.</param>
    /// <returns>Whether it is well formed.</returns>
    public static bool IsWellFormedBackendTag(string? tag)
    {
        if (tag is null || tag.Length is 0 or > MaxBackendTagBytes) return false;
        foreach (char c in tag)
            if (!IsTagChar(c)) return false;
        return true;
    }

    private static bool IsTagChar(int c) => c is (>= 'a' and <= 'z') or (>= '0' and <= '9') or '.' or '_' or '-';

    /// <summary>The store directory.</summary>
    public string Root => _root;

    /// <inheritdoc/>
    public long TotalBytes => GetStats().EntryBytes;

    /// <summary>Test hook: runs after the temp file is written, just before the rename, with the destination path.</summary>
    internal Action<string>? BeforeRenameForTest { get; set; }

    /// <summary>Counts entries and leftover temp files (and their bytes) separately.</summary>
    /// <returns>The statistics.</returns>
    public StoreStats GetStats()
    {
        int entries = 0, temps = 0;
        long entryBytes = 0, tempBytes = 0;
        foreach ((_, string path) in EnumerateEntries())
        {
            if (TryLength(path, out long len)) { entries++; entryBytes += len; }
        }
        foreach (string path in EnumerateTempFiles())
        {
            if (TryLength(path, out long len)) { temps++; tempBytes += len; }
        }
        return new StoreStats(entries, entryBytes, temps, tempBytes);
    }

    /// <summary>
    /// Deletes store temp files (<c>&lt;key&gt;.&lt;guid&gt;.tmp</c>) last written more than <paramref name="olderThan"/> ago. Use an
    /// age well above the longest write, so a temp file of a writer that is still running is not removed.
    /// </summary>
    /// <param name="olderThan">Minimum age; <see cref="TimeSpan.Zero"/> removes all.</param>
    /// <returns>Number of files deleted.</returns>
    public int SweepTempFiles(TimeSpan olderThan)
    {
        DateTime cutoff = DateTime.UtcNow - olderThan;
        int deleted = 0;
        foreach (string path in EnumerateTempFiles())
        {
            try
            {
                if (File.GetLastWriteTimeUtc(path) > cutoff) continue;
                File.Delete(path);
                deleted++;
            }
            catch (IOException) { /* in use or already gone */ }
            catch (UnauthorizedAccessException) { }
        }
        return deleted;
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

    /// <summary>
    /// Reads only the header of <paramref name="key"/>'s entry and returns the model fingerprint it was written with (for
    /// per-model eviction without loading the K/V payload). A missing or invalid entry (the same checks as
    /// <see cref="Contains"/>) gives false.
    /// </summary>
    /// <param name="key">Chunk key.</param>
    /// <param name="fingerprint">The entry's model fingerprint when the entry is valid.</param>
    /// <returns>Whether a valid entry exists.</returns>
    public bool TryReadFingerprint(ChunkKey key, out string fingerprint)
    {
        fingerprint = "";
        string path = PathOf(key);
        try
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete);
            long length = fs.Length;
            byte[] head = new byte[(int)Math.Min(length, MaxHeaderBytes)];
            fs.ReadExactly(head);
            if (!TryParseHeader(head, length, key, out Header h)) return false;
            fingerprint = h.Fingerprint;
            return true;
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
            entry = new ChunkKvEntry(h.Fingerprint, tokens, keys, values, h.Backend);
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
        string temp = Path.Combine(dir, $"{key.Hex}.{Guid.NewGuid():N}{TempExtension}");
        try
        {
            WriteEntry(temp, key, entry);
            BeforeRenameForTest?.Invoke(path);
            try
            {
                File.Move(temp, path, overwrite: true);
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                // On Windows the destination cannot be replaced while a reader has it mapped. If it is a valid entry (another
                // writer stored the same key meanwhile) it is as good as ours; otherwise fail at once instead of waiting for
                // the reader.
                if (Contains(key)) return;
                throw new ChunkStoreBusyException(path, e);
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
        foreach ((string shardName, string file) in EnumerateShardFiles("*" + Extension))
        {
            // Defensive: require the exact extension (the .NET 10 matcher already does).
            if (!string.Equals(Path.GetExtension(file), Extension, StringComparison.Ordinal)) continue;
            string hex = Path.GetFileNameWithoutExtension(file);
            if (ChunkKey.IsWellFormed(hex) && hex.StartsWith(shardName, StringComparison.Ordinal))
                result.Add((new ChunkKey(hex), file));
        }
        return result;
    }

    // Store temp files only: "<64 hex key>.<32 hex guid>.tmp" in the shard named by the key's prefix.
    private List<string> EnumerateTempFiles()
    {
        var result = new List<string>();
        foreach ((string shardName, string file) in EnumerateShardFiles("*" + TempExtension))
        {
            string name = Path.GetFileName(file);
            if (name.Length != 64 + 1 + 32 + TempExtension.Length || name[64] != '.') continue;
            string hex = name[..64];
            string guid = name.Substring(65, 32);
            if (ChunkKey.IsWellFormed(hex) && ChunkKey.IsWellFormed(guid + guid) && hex.StartsWith(shardName, StringComparison.Ordinal))
                result.Add(file);
        }
        return result;
    }

    // Files in two-character shard directories; a root or shard removed meanwhile contributes nothing.
    private List<(string Shard, string File)> EnumerateShardFiles(string pattern)
    {
        var result = new List<(string, string)>();
        string[] shards;
        try { shards = Directory.GetDirectories(_root); }
        catch (DirectoryNotFoundException) { return result; }
        foreach (string shard in shards)
        {
            string shardName = Path.GetFileName(shard);
            if (shardName.Length != 2) continue;
            try
            {
                foreach (string file in Directory.GetFiles(shard, pattern)) result.Add((shardName, file));
            }
            catch (DirectoryNotFoundException) { /* shard removed meanwhile */ }
        }
        return result;
    }

    private static bool TryLength(string path, out long length)
    {
        try
        {
            length = new FileInfo(path).Length;
            return true;
        }
        catch (IOException)
        {
            length = 0;
            return false;
        }
    }

    private static void Validate(ChunkKvEntry entry)
    {
        ArgumentNullException.ThrowIfNull(entry.ModelFingerprint);
        ArgumentNullException.ThrowIfNull(entry.TokenIds);
        ArgumentNullException.ThrowIfNull(entry.KeysPerLayer);
        ArgumentNullException.ThrowIfNull(entry.ValuesPerLayer);
        ArgumentNullException.ThrowIfNull(entry.Backend);
        if (!IsWellFormedBackendTag(entry.Backend))
            throw new ArgumentException(
                $"Backend tag '{entry.Backend}' is not 1 to {MaxBackendTagBytes} lowercase ASCII letters, digits, '.', '_' or '-'.", nameof(entry));
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
        byte[] backend = Encoding.ASCII.GetBytes(entry.Backend); // well formed (Validate): ASCII only
        var header = new byte[HeaderBytes(entry.ModelFingerprint, entry.Backend, layers)];
        var w = new SpanWriter(header);
        w.Bytes(Magic);
        w.Int32(FormatVersion);
        w.Int32((int)KvDtype.F32);
        w.Int32(0); // reserved
        w.Bytes(Encoding.ASCII.GetBytes(key.Hex));
        w.Int32(fp.Length);
        w.Bytes(fp);
        w.Int32(backend.Length);
        w.Bytes(backend);
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
        if (!r.Int32(out int tagLen) || tagLen <= 0 || tagLen > MaxBackendTagBytes) return false;
        if (!r.Bytes(tagLen, out ReadOnlySpan<byte> tagBytes)) return false;
        foreach (byte b in tagBytes)
            if (!IsTagChar(b)) return false;
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

        header = new Header(Encoding.UTF8.GetString(fpBytes), Encoding.ASCII.GetString(tagBytes), tokens, layers, kLens, vLens, r.Position);
        return true;
    }

    private readonly record struct Header(string Fingerprint, string Backend, int TokenCount, int LayerCount, int[] KLengths, int[] VLengths, int HeaderBytes);

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
