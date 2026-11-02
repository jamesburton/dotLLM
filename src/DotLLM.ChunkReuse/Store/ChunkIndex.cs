using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using DotLLM.ChunkReuse.CodeCorpus;

namespace DotLLM.ChunkReuse.Store;

/// <summary>
/// Store key of a chunk from its token ids, with every producer field (model fingerprint, tokenizer, RoPE, backend, dtype)
/// fixed by the caller, e.g. <c>ids =&gt; ChunkKey.Create(fingerprint, tokenizerId, rope, "cpu", KvDtype.F32, ids)</c>.
/// </summary>
/// <param name="tokenIds">The chunk's token ids.</param>
/// <returns>The key.</returns>
public delegate ChunkKey ChunkKeyFunction(ReadOnlySpan<int> tokenIds);

/// <summary>
/// What an <see cref="ChunkIndex.Update"/> changed. A chunk is matched by its key within its file, not by position, so a
/// chunk whose tokens are unchanged is <see cref="Unchanged"/> even when an edit above it moved its lines (its K/V is
/// position-independent and is re-rotated at assembly).
/// </summary>
/// <param name="Added">Chunks (at their new position) whose key the file did not have before: these need their K/V.</param>
/// <param name="Removed">Chunks (at their old position) whose key the file no longer has.</param>
/// <param name="Unchanged">Chunks (at their new position) whose key the file already had.</param>
public sealed record IndexDiff(IReadOnlyList<ChunkRef> Added, IReadOnlyList<ChunkRef> Removed, IReadOnlyList<ChunkRef> Unchanged)
{
    /// <summary>
    /// Keys no chunk of the index references any more (eviction candidates). A key still referenced by another file (identical
    /// chunk text) is not orphaned.
    /// </summary>
    public IReadOnlyList<ChunkKey> OrphanedKeys { get; init; } = [];

    /// <summary>Sum of the token counts of <see cref="Added"/>.</summary>
    public long AddedTokens => Added.Sum(r => (long)(r.TokenCount ?? 0));
}

/// <summary>State of a file in a <see cref="StatusReport"/>.</summary>
public enum FileState
{
    /// <summary>Indexed and (if the current tree was given) unchanged.</summary>
    Indexed = 0,

    /// <summary>The file's text differs from the indexed text: its indexed chunks are stale; run an update.</summary>
    Stale = 1,

    /// <summary>In the current tree but not in the index.</summary>
    New = 2,

    /// <summary>In the index but no longer in the current tree.</summary>
    Deleted = 3,
}

/// <summary>One file with something to report.</summary>
/// <param name="Path">File path.</param>
/// <param name="State">Its state.</param>
/// <param name="Missing">Indexed chunks of the file whose key is not in the store, in text order.</param>
public sealed record FileStatus(string Path, FileState State, IReadOnlyList<ChunkRef> Missing);

/// <summary>Result of <see cref="ChunkIndex.Status"/>.</summary>
/// <param name="Files">Files that are not indexed-and-complete, in ordinal path order.</param>
/// <param name="TotalChunks">Chunks in the index.</param>
/// <param name="MissingChunks">Indexed chunks whose key is not in the store.</param>
/// <param name="MissingTokens">Sum of their token counts.</param>
public sealed record StatusReport(IReadOnlyList<FileStatus> Files, int TotalChunks, int MissingChunks, long MissingTokens)
{
    /// <summary>True when no file is listed: every indexed chunk is stored and no file is stale, new or deleted.</summary>
    public bool IsUpToDate => Files.Count == 0;

    /// <summary>One line per missing chunk or changed file, e.g. <c>src/A.cs:12-40 missing (154 tokens)</c>.</summary>
    /// <returns>The text.</returns>
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.Append(TotalChunks).Append(" chunks, ").Append(MissingChunks).Append(" missing (").Append(MissingTokens).Append(" tokens)\n");
        foreach (FileStatus f in Files)
        {
            if (f.State != FileState.Indexed) sb.Append(f.Path).Append(' ').Append(f.State.ToString().ToLowerInvariant()).Append('\n');
            foreach (ChunkRef r in f.Missing)
                sb.Append(r.FilePath).Append(':').Append(r.StartLine).Append('-').Append(r.EndLine).Append(" missing (")
                    .Append(r.TokenCount ?? 0).Append(" tokens)\n");
        }
        return sb.ToString();
    }
}

/// <summary>
/// Which chunks a source tree has, per file, with their store keys: the basis for incremental invalidation (an edit re-keys only
/// the chunks whose tokens changed) and for status (which chunks the store lacks). One index holds one chunker's chunks for one
/// producer (model, tokenizer, RoPE, backend, dtype). It persists as JSON next to the store (<see cref="DefaultPath"/>).
/// </summary>
/// <remarks>
/// <para>
/// A file whose text hash and chunker are unchanged is not re-chunked. A changed file is re-chunked and its new chunks are
/// matched to its old ones by key (as a multiset), so with symbol chunks an edit inside one method changes that method's chunk
/// only, and with fixed-size chunks an edit that changes the token count re-keys every chunk from the edit point to the end of
/// the file.
/// </para>
/// <para>
/// The producer identity is the key of the empty token sequence (<c>keyOf([])</c>): it hashes every producer field, so an
/// index saved for another model, tokenizer, RoPE setting, backend or dtype is detected on <see cref="Load"/> and discarded.
/// Not thread-safe.
/// </para>
/// </remarks>
public sealed class ChunkIndex
{
    /// <summary>JSON format version; any other version loads as an empty index.</summary>
    public const int FormatVersion = 1;

    private readonly ChunkKeyFunction _keyOf;
    private readonly string _producer;
    private readonly SortedDictionary<string, FileEntry> _files = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _refCounts = new(StringComparer.Ordinal);

    /// <summary>Creates an empty index.</summary>
    /// <param name="keyOf">Key of a chunk from its token ids (fixes the producer).</param>
    public ChunkIndex(ChunkKeyFunction keyOf)
    {
        ArgumentNullException.ThrowIfNull(keyOf);
        _keyOf = keyOf;
        _producer = keyOf([]).Hex;
    }

    /// <summary>Name of the chunker the index was built with (<see cref="IChunker.Name"/>), or null when empty.</summary>
    public string? ChunkerName { get; private set; }

    /// <summary>Why <see cref="Load"/> returned an empty index for an existing file (corrupt, other version, other producer), else null.</summary>
    public string? LoadNote { get; private set; }

    /// <summary>Indexed file paths in ordinal order.</summary>
    public IReadOnlyCollection<string> Paths => _files.Keys;

    /// <summary>Number of distinct keys the indexed chunks reference.</summary>
    public int DistinctKeyCount => _refCounts.Count;

    /// <summary>Number of indexed chunks.</summary>
    public int ChunkCount => _files.Values.Sum(f => f.Chunks.Count);

    /// <summary>Whether <paramref name="path"/> is indexed.</summary>
    /// <param name="path">File path.</param>
    /// <returns>Whether it is indexed.</returns>
    public bool Contains(string path) => _files.ContainsKey(path);

    /// <summary>Chunks of <paramref name="path"/> in text order (empty when not indexed).</summary>
    /// <param name="path">File path.</param>
    /// <returns>The chunks.</returns>
    public IReadOnlyList<ChunkRef> ChunksOf(string path) => _files.TryGetValue(path, out FileEntry? f) ? f.Chunks : [];

    /// <summary>
    /// Brings the index up to date with <paramref name="files"/>: re-chunks the files whose text (or the chunker) changed,
    /// keys their chunks, and reports what changed.
    /// </summary>
    /// <param name="files">Current files (unique paths).</param>
    /// <param name="chunker">The chunker; a different chunker than the index's re-chunks every file.</param>
    /// <param name="removeMissingFiles">
    /// True (default): <paramref name="files"/> is the whole tree, so indexed files not in it are removed. False: only the given
    /// files are updated and the others are kept (except after a chunker change, which drops every file not given, since one
    /// index holds one chunker's chunks).
    /// </param>
    /// <returns>The diff; lists are in ordinal path order, then text order.</returns>
    public IndexDiff Update(IReadOnlyList<SourceFile> files, IChunker chunker, bool removeMissingFiles = true)
    {
        ArgumentNullException.ThrowIfNull(files);
        ArgumentNullException.ThrowIfNull(chunker);
        ChunkKind kind = ChunkRef.KindOf(chunker);
        bool sameChunker = string.Equals(ChunkerName, chunker.Name, StringComparison.Ordinal);

        var given = new Dictionary<string, SourceFile>(StringComparer.Ordinal);
        foreach (SourceFile f in files)
            if (!given.TryAdd(f.Path, f)) throw new ArgumentException($"Duplicate file path '{f.Path}'.", nameof(files));

        var added = new List<ChunkRef>();
        var removed = new List<ChunkRef>();
        var unchanged = new List<ChunkRef>();
        var touchedKeys = new HashSet<string>(StringComparer.Ordinal);

        IEnumerable<string> paths = (removeMissingFiles || !sameChunker ? given.Keys.Concat(_files.Keys) : given.Keys)
            .Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToList();
        foreach (string path in paths)
        {
            _files.TryGetValue(path, out FileEntry? old);
            if (!given.TryGetValue(path, out SourceFile? file))
            {
                // Not in the tree (or, after a chunker change, chunked by the other chunker): drop its chunks.
                Release(old!, removed, touchedKeys);
                _files.Remove(path);
                continue;
            }

            string hash = HashText(file.Text);
            if (old is not null && sameChunker && old.TextHash == hash)
            {
                unchanged.AddRange(old.Chunks);
                continue;
            }

            var chunks = new List<ChunkRef>();
            foreach (Chunk c in chunker.Chunk(file))
                chunks.Add(ChunkRef.From(c, kind, _keyOf(c.TokenIds)));

            // Match new chunks to old ones by key, as a multiset (a file may hold the same chunk text twice).
            var oldByKey = new Dictionary<string, Queue<ChunkRef>>(StringComparer.Ordinal);
            foreach (ChunkRef r in old?.Chunks ?? [])
            {
                if (!oldByKey.TryGetValue(r.KeyHex!, out Queue<ChunkRef>? q)) oldByKey[r.KeyHex!] = q = new Queue<ChunkRef>();
                q.Enqueue(r);
            }
            foreach (ChunkRef r in chunks)
            {
                if (oldByKey.TryGetValue(r.KeyHex!, out Queue<ChunkRef>? q) && q.Count > 0)
                {
                    q.Dequeue();
                    unchanged.Add(r);
                }
                else
                {
                    added.Add(r);
                }
                AddRef(r.KeyHex!);
            }
            if (old is not null) Release(old, removed, touchedKeys, oldByKey);
            _files[path] = new FileEntry(hash, chunks);
        }

        ChunkerName = _files.Count > 0 || given.Count > 0 ? chunker.Name : ChunkerName;
        var orphaned = touchedKeys.Where(k => !_refCounts.ContainsKey(k)).Order(StringComparer.Ordinal).Select(k => new ChunkKey(k)).ToList();
        return new IndexDiff(added, removed, unchanged) { OrphanedKeys = orphaned };
    }

    /// <summary>
    /// Lists, by file and line range, the indexed chunks whose key is not in <paramref name="store"/>; with
    /// <paramref name="currentFiles"/>, also the files whose text changed since indexing (stale), new files and deleted files.
    /// </summary>
    /// <param name="store">The store.</param>
    /// <param name="currentFiles">The current tree, or null to check the store only.</param>
    /// <returns>The report.</returns>
    public StatusReport Status(IChunkStore store, IReadOnlyList<SourceFile>? currentFiles = null)
    {
        ArgumentNullException.ThrowIfNull(store);
        Dictionary<string, SourceFile>? current = currentFiles?.ToDictionary(f => f.Path, StringComparer.Ordinal);
        var present = new Dictionary<string, bool>(StringComparer.Ordinal);
        var report = new List<FileStatus>();
        int total = 0, missingChunks = 0;
        long missingTokens = 0;

        IEnumerable<string> paths = current is null ? _files.Keys : _files.Keys.Concat(current.Keys).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal);
        foreach (string path in paths)
        {
            if (!_files.TryGetValue(path, out FileEntry? entry))
            {
                report.Add(new FileStatus(path, FileState.New, []));
                continue;
            }
            var missing = new List<ChunkRef>();
            foreach (ChunkRef r in entry.Chunks)
            {
                total++;
                if (!present.TryGetValue(r.KeyHex!, out bool has)) present[r.KeyHex!] = has = store.Contains(new ChunkKey(r.KeyHex!));
                if (has) continue;
                missing.Add(r);
                missingChunks++;
                missingTokens += r.TokenCount ?? 0;
            }
            FileState state = current is null ? FileState.Indexed
                : !current.TryGetValue(path, out SourceFile? now) ? FileState.Deleted
                : HashText(now.Text) != entry.TextHash ? FileState.Stale
                : FileState.Indexed;
            if (state != FileState.Indexed || missing.Count > 0) report.Add(new FileStatus(path, state, missing));
        }
        return new StatusReport(report, total, missingChunks, missingTokens);
    }

    /// <summary>The index file for a store: <c>&lt;storeRoot&gt;/index-&lt;chunkerName&gt;.json</c> (one index per chunker).</summary>
    /// <param name="storeRoot">Store directory.</param>
    /// <param name="chunkerName">Chunker name, e.g. <c>symbol-512</c>.</param>
    /// <returns>The path.</returns>
    public static string DefaultPath(string storeRoot, string chunkerName)
    {
        ArgumentException.ThrowIfNullOrEmpty(storeRoot);
        ArgumentException.ThrowIfNullOrEmpty(chunkerName);
        if (chunkerName.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
            throw new ArgumentException($"Chunker name '{chunkerName}' is not a valid file name part.", nameof(chunkerName));
        return Path.Combine(Path.GetFullPath(storeRoot), $"index-{chunkerName}.json");
    }

    /// <summary>Writes the index as JSON (to a temp file, then one rename over <paramref name="path"/>).</summary>
    /// <param name="path">Destination file.</param>
    public void Save(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        var doc = new IndexDocument
        {
            FormatVersion = FormatVersion,
            Producer = _producer,
            Chunker = ChunkerName,
            Files = _files.Select(kv => new FileDocument
            {
                Path = kv.Key,
                Sha256 = kv.Value.TextHash,
                Chunks = kv.Value.Chunks.Select(r => new ChunkDocument
                {
                    StartLine = r.StartLine, EndLine = r.EndLine, StartOffset = r.StartOffset, EndOffset = r.EndOffset,
                    Tokens = r.TokenCount ?? 0, Key = r.KeyHex!,
                }).ToList(),
            }).ToList(),
        };
        string full = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(full)!);
        string temp = full + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            using (var stream = new FileStream(temp, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                JsonSerializer.Serialize(stream, doc, ChunkIndexJsonContext.Default.IndexDocument);
                stream.Flush(flushToDisk: true);
            }
            File.Move(temp, full, overwrite: true);
        }
        finally
        {
            if (File.Exists(temp)) File.Delete(temp);
        }
    }

    /// <summary>
    /// Loads an index saved by <see cref="Save"/>. A missing file gives an empty index; a corrupt file, another format version
    /// or another producer (<paramref name="keyOf"/> differs from the one it was saved with) gives an empty index with a
    /// <see cref="LoadNote"/>, so everything is re-keyed rather than trusted.
    /// </summary>
    /// <param name="path">Index file.</param>
    /// <param name="keyOf">Key of a chunk from its token ids.</param>
    /// <returns>The index.</returns>
    public static ChunkIndex Load(string path, ChunkKeyFunction keyOf)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        var index = new ChunkIndex(keyOf);
        if (!File.Exists(path)) return index;
        IndexDocument? doc;
        try
        {
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            doc = JsonSerializer.Deserialize(stream, ChunkIndexJsonContext.Default.IndexDocument);
        }
        catch (JsonException ex)
        {
            index.LoadNote = $"index file is not valid JSON ({ex.Message}); starting empty";
            return index;
        }
        if (doc is null || doc.FormatVersion != FormatVersion)
        {
            index.LoadNote = $"index format version {doc?.FormatVersion} is not {FormatVersion}; starting empty";
            return index;
        }
        if (!string.Equals(doc.Producer, index._producer, StringComparison.Ordinal))
        {
            index.LoadNote = "index was built for another producer (model, tokenizer, RoPE, backend or dtype); starting empty";
            return index;
        }
        ChunkKind kind;
        try
        {
            kind = doc.Chunker is null ? ChunkKind.Fixed : KindOfName(doc.Chunker);
        }
        catch (ArgumentException ex)
        {
            index.LoadNote = ex.Message + "; starting empty";
            return index;
        }
        foreach (FileDocument f in doc.Files ?? [])
        {
            if (f.Path is null || f.Sha256 is null || f.Chunks is null || f.Chunks.Any(c => !ChunkKey.IsWellFormed(c.Key)))
            {
                var empty = new ChunkIndex(keyOf) { LoadNote = $"index entry for '{f.Path}' is malformed; starting empty" };
                return empty;
            }
            var chunks = f.Chunks.Select(c => new ChunkRef(f.Path, c.StartLine, c.EndLine, c.StartOffset, c.EndOffset, kind)
            {
                TokenCount = c.Tokens,
                KeyHex = c.Key,
            }).ToList();
            index._files[f.Path] = new FileEntry(f.Sha256, chunks);
            foreach (ChunkRef r in chunks) index.AddRef(r.KeyHex!);
        }
        index.ChunkerName = doc.Chunker;
        return index;
    }

    private static ChunkKind KindOfName(string name) =>
        name.StartsWith("symbol-", StringComparison.Ordinal) ? ChunkKind.Symbol
        : name.StartsWith("fixed-", StringComparison.Ordinal) ? ChunkKind.Fixed
        : throw new ArgumentException($"index chunker '{name}' is neither symbol-* nor fixed-*");

    private void AddRef(string key) => CollectionsMarshal.GetValueRefOrAddDefault(_refCounts, key, out _)++;

    /// <summary>Drops the references of <paramref name="old"/>'s chunks; those not matched (all, when
    /// <paramref name="unmatched"/> is null) are reported as removed.</summary>
    private void Release(FileEntry old, List<ChunkRef> removed, HashSet<string> touchedKeys, Dictionary<string, Queue<ChunkRef>>? unmatched = null)
    {
        foreach (ChunkRef r in old.Chunks)
        {
            string key = r.KeyHex!;
            ref int count = ref CollectionsMarshal.GetValueRefOrNullRef(_refCounts, key);
            if (--count == 0) _refCounts.Remove(key);
            touchedKeys.Add(key);
        }
        if (unmatched is null)
        {
            removed.AddRange(old.Chunks);
            return;
        }
        // Old chunks left in the queues (in text order) had no new chunk with their key.
        var left = new HashSet<ChunkRef>(unmatched.Values.SelectMany(q => q), ReferenceEqualityComparer.Instance);
        removed.AddRange(old.Chunks.Where(left.Contains));
    }

    private static string HashText(string text) => Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(text)));

    private sealed record FileEntry(string TextHash, List<ChunkRef> Chunks);

    internal sealed class IndexDocument
    {
        public int FormatVersion { get; set; }
        public string? Producer { get; set; }
        public string? Chunker { get; set; }
        public List<FileDocument>? Files { get; set; }
    }

    internal sealed class FileDocument
    {
        public string? Path { get; set; }
        public string? Sha256 { get; set; }
        public List<ChunkDocument>? Chunks { get; set; }
    }

    internal sealed class ChunkDocument
    {
        public int StartLine { get; set; }
        public int EndLine { get; set; }
        public int StartOffset { get; set; }
        public int EndOffset { get; set; }
        public int Tokens { get; set; }
        public string Key { get; set; } = "";
    }
}

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
[JsonSerializable(typeof(ChunkIndex.IndexDocument))]
internal sealed partial class ChunkIndexJsonContext : JsonSerializerContext;
