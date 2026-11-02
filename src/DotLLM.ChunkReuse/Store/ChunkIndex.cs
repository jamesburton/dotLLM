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
/// <param name="Added">
/// Chunks (at their new position) that are new <i>to their file</i>. This is not "needs encoding": a chunk moved from another
/// file, or text that duplicates a chunk elsewhere, is here although its K/V may already be stored. Use
/// <see cref="NeedsEncoding"/> to get the chunks to encode.
/// </param>
/// <param name="Removed">Chunks (at their old position) whose key the file no longer has.</param>
/// <param name="Unchanged">Chunks (at their new position) whose key the file already had.</param>
public sealed record IndexDiff(IReadOnlyList<ChunkRef> Added, IReadOnlyList<ChunkRef> Removed, IReadOnlyList<ChunkRef> Unchanged)
{
    /// <summary>
    /// Keys no chunk of <i>this index</i> references any more. A key still referenced by another file of the index (identical
    /// chunk text) is not orphaned, but another index sharing the store (e.g. the other chunker's) may still use it: evict
    /// only <see cref="ChunkIndex.EvictableOrphans"/>, never these directly.
    /// </summary>
    public IReadOnlyList<ChunkKey> OrphanedKeys { get; init; } = [];

    /// <summary>Sum of the token counts of <see cref="Added"/>.</summary>
    public long AddedTokens => Added.Sum(r => (long)(r.TokenCount ?? 0));

    /// <summary>
    /// The chunks of <see cref="Added"/> whose key <paramref name="store"/> lacks, one per distinct key (first occurrence), in
    /// <see cref="Added"/> order: exactly what must be encoded and put.
    /// </summary>
    /// <param name="store">The store.</param>
    /// <returns>The chunks to encode.</returns>
    public IReadOnlyList<ChunkRef> NeedsEncoding(IChunkStore store)
    {
        ArgumentNullException.ThrowIfNull(store);
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var result = new List<ChunkRef>();
        foreach (ChunkRef r in Added)
            if (r.KeyHex is not null && seen.Add(r.KeyHex) && !store.Contains(new ChunkKey(r.KeyHex))) result.Add(r);
        return result;
    }
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
/// <param name="Files">Files that are not indexed-and-complete, in path order.</param>
/// <param name="TotalChunks">Chunks in the index.</param>
/// <param name="MissingChunks">Indexed chunks whose key is not in the store (a stale file's old chunks included).</param>
/// <param name="MissingTokens">Sum of their token counts.</param>
public sealed record StatusReport(IReadOnlyList<FileStatus> Files, int TotalChunks, int MissingChunks, long MissingTokens)
{
    /// <summary>
    /// True when no file is listed: every indexed chunk is stored and (if the current tree was given) no file is stale, new or
    /// deleted. Without the current tree, edits since the last update are not seen.
    /// </summary>
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
/// producer (model, tokenizer, RoPE, backend, dtype). It persists as JSON next to the store (<see cref="DefaultPath"/>); a
/// store can host one index per chunker.
/// </summary>
/// <remarks>
/// <para>
/// A file whose text hash and chunker are unchanged is not re-chunked. A changed file is re-chunked and its new chunks are
/// matched to its old ones by key (as a multiset), so with symbol chunks an edit inside one method changes that method's chunk
/// only, and with fixed-size chunks an edit that changes the token count re-keys every chunk from the edit point to the end of
/// the file. The chunker's identity is its <see cref="IChunker.Name"/> only: a chunker whose algorithm changes must change its
/// name, or files with unchanged text keep their old boundaries.
/// </para>
/// <para>
/// <b>Paths</b> are normalised (<see cref="NormalizePath"/>): relative, '/' separated, no <c>.</c> or <c>..</c> segments;
/// compared case-insensitively on Windows and ordinally elsewhere (<see cref="PathComparer"/>). The first spelling seen is
/// kept.
/// </para>
/// <para>
/// The producer identity is the key of the empty token sequence (<c>keyOf([])</c>): it hashes every producer field, so an
/// index saved for another model, tokenizer, RoPE setting, backend or dtype is detected on <see cref="Load"/> and discarded.
/// </para>
/// <para>
/// <b>Concurrency.</b> An instance is not thread-safe. Across processes, <see cref="Save"/> replaces the file with one
/// rename, so a reader (<see cref="Load"/>, <see cref="OpenForRead"/>) sees either the old or the new file, never a mix.
/// Readers share the file for read, write and delete; on Windows a rename over an open file is still refused, so Save retries
/// for up to two seconds (a Load holds the file for milliseconds) and then throws, leaving the old file intact. Two writers
/// do not merge: the last rename wins and the other's update is lost. A tool that updates an index from several processes
/// must serialise them (e.g. a lock file next to the index).
/// </para>
/// </remarks>
public sealed class ChunkIndex
{
    /// <summary>JSON format version; any other version loads as an empty index.</summary>
    public const int FormatVersion = 1;

    /// <summary>Path comparison: case-insensitive on Windows, ordinal elsewhere.</summary>
    public static readonly StringComparer PathComparer = OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;

    private const string IndexFilePrefix = "index-";
    private const string IndexFileSuffix = ".json";

    private readonly ChunkKeyFunction _keyOf;
    private readonly string _producer;
    private readonly SortedDictionary<string, FileEntry> _files = new(PathComparer);
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

    /// <summary>Indexed file paths in <see cref="PathComparer"/> order.</summary>
    public IReadOnlyCollection<string> Paths => _files.Keys;

    /// <summary>Number of distinct keys the indexed chunks reference.</summary>
    public int DistinctKeyCount => _refCounts.Count;

    /// <summary>Number of indexed chunks.</summary>
    public int ChunkCount => _files.Values.Sum(f => f.Chunks.Count);

    /// <summary>The distinct keys the indexed chunks reference.</summary>
    /// <returns>The keys.</returns>
    public IEnumerable<ChunkKey> ReferencedKeys() => _refCounts.Keys.Select(k => new ChunkKey(k));

    /// <summary>Whether <paramref name="path"/> is indexed.</summary>
    /// <param name="path">File path (normalised before lookup).</param>
    /// <returns>Whether it is indexed.</returns>
    public bool Contains(string path) => _files.ContainsKey(NormalizePath(path));

    /// <summary>Chunks of <paramref name="path"/> in text order (empty when not indexed).</summary>
    /// <param name="path">File path (normalised before lookup).</param>
    /// <returns>The chunks.</returns>
    public IReadOnlyList<ChunkRef> ChunksOf(string path) => _files.TryGetValue(NormalizePath(path), out FileEntry? f) ? f.Chunks : [];

    /// <summary>
    /// The index's form of a file path: '\' becomes '/', <c>.</c> segments and repeated separators are dropped. Rooted paths
    /// (<c>/x</c>, <c>C:/x</c>, <c>\\server\x</c>) and <c>..</c> segments are rejected, because index paths are relative to the
    /// tree root.
    /// </summary>
    /// <param name="path">A relative path.</param>
    /// <returns>The normalised path.</returns>
    /// <exception cref="ArgumentException">The path is empty, rooted or contains <c>..</c>.</exception>
    public static string NormalizePath(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        string p = path.Replace('\\', '/');
        if (p.StartsWith('/') || (p.Length >= 2 && p[1] == ':') || Path.IsPathRooted(p))
            throw new ArgumentException($"Index paths are relative to the tree root; '{path}' is rooted.", nameof(path));
        string[] parts = p.Split('/', StringSplitOptions.RemoveEmptyEntries).Where(s => s != ".").ToArray();
        if (parts.Contains("..", StringComparer.Ordinal))
            throw new ArgumentException($"Index paths may not contain '..': '{path}'.", nameof(path));
        if (parts.Length == 0) throw new ArgumentException($"'{path}' names no file.", nameof(path));
        return string.Join('/', parts);
    }

    /// <summary>
    /// Brings the index up to date with <paramref name="files"/>: re-chunks the files whose text (or the chunker) changed,
    /// keys their chunks, and reports what changed.
    /// </summary>
    /// <param name="files">Current files (unique paths after <see cref="NormalizePath"/>).</param>
    /// <param name="chunker">The chunker; a different chunker than the index's re-chunks every file.</param>
    /// <param name="removeMissingFiles">
    /// True (default): <paramref name="files"/> is the whole tree, so indexed files not in it are removed. False: only the given
    /// files are updated and the others are kept (except after a chunker change, which drops every file not given, since one
    /// index holds one chunker's chunks).
    /// </param>
    /// <returns>The diff; lists are in path order, then text order.</returns>
    public IndexDiff Update(IReadOnlyList<SourceFile> files, IChunker chunker, bool removeMissingFiles = true)
    {
        ArgumentNullException.ThrowIfNull(files);
        ArgumentNullException.ThrowIfNull(chunker);
        ChunkKind kind = ChunkRef.KindOf(chunker);
        bool sameChunker = string.Equals(ChunkerName, chunker.Name, StringComparison.Ordinal);

        var given = new Dictionary<string, SourceFile>(PathComparer);
        foreach (SourceFile f in files)
        {
            string norm = NormalizePath(f.Path);
            if (!given.TryAdd(norm, string.Equals(norm, f.Path, StringComparison.Ordinal) ? f : f with { Path = norm }))
                throw new ArgumentException($"Duplicate file path '{f.Path}'.", nameof(files));
        }

        var added = new List<ChunkRef>();
        var removed = new List<ChunkRef>();
        var unchanged = new List<ChunkRef>();
        var touchedKeys = new HashSet<string>(StringComparer.Ordinal);

        IEnumerable<string> paths = (removeMissingFiles || !sameChunker ? given.Keys.Concat(_files.Keys) : given.Keys)
            .Distinct(PathComparer).Order(PathComparer).ToList();
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
            // Keep the spelling already indexed (case may differ on Windows).
            if (old is not null) file = file with { Path = old.Path };

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
            _files[file.Path] = new FileEntry(file.Path, hash, chunks);
        }

        ChunkerName = _files.Count > 0 || given.Count > 0 ? chunker.Name : ChunkerName;
        var orphaned = touchedKeys.Where(k => !_refCounts.ContainsKey(k)).Order(StringComparer.Ordinal).Select(k => new ChunkKey(k)).ToList();
        return new IndexDiff(added, removed, unchanged) { OrphanedKeys = orphaned };
    }

    /// <summary>
    /// The keys of <paramref name="diff"/>'s <see cref="IndexDiff.OrphanedKeys"/> that are safe to evict from the store at
    /// <paramref name="storeRoot"/>: referenced neither by this index nor by any other index file under the store root
    /// (<c>index-*.json</c>, e.g. the other chunker's). This index's own file (<see cref="DefaultPath"/> for
    /// <see cref="ChunkerName"/>) is skipped, since it may still hold the references this update removed.
    /// </summary>
    /// <param name="diff">A diff returned by <see cref="Update"/> on this index.</param>
    /// <param name="storeRoot">The store directory.</param>
    /// <returns>Keys to evict, in ordinal order.</returns>
    /// <exception cref="InvalidDataException">Another index file under the store cannot be read: eviction cannot be decided.</exception>
    public IReadOnlyList<ChunkKey> EvictableOrphans(IndexDiff diff, string storeRoot)
    {
        ArgumentNullException.ThrowIfNull(diff);
        string? own = ChunkerName is null ? null : DefaultPath(storeRoot, ChunkerName);
        IReadOnlySet<ChunkKey> elsewhere = ReferencedKeysUnder(storeRoot, own);
        return diff.OrphanedKeys.Where(k => !_refCounts.ContainsKey(k.Hex) && !elsewhere.Contains(k)).ToList();
    }

    /// <summary>
    /// Union of the keys referenced by every index file (<c>index-*.json</c>) directly under <paramref name="storeRoot"/>,
    /// whatever its producer or chunker. Use it before evicting from a store that several indexes share.
    /// </summary>
    /// <param name="storeRoot">The store directory.</param>
    /// <param name="exceptPath">An index file to skip (typically the caller's own, about to be rewritten), or null.</param>
    /// <returns>The keys.</returns>
    /// <exception cref="InvalidDataException">An index file cannot be read or parsed (conservative: no guess).</exception>
    public static IReadOnlySet<ChunkKey> ReferencedKeysUnder(string storeRoot, string? exceptPath = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(storeRoot);
        var keys = new HashSet<ChunkKey>();
        string root = Path.GetFullPath(storeRoot);
        if (!Directory.Exists(root)) return keys;
        string? skip = exceptPath is null ? null : Path.GetFullPath(exceptPath);
        foreach (string file in Directory.EnumerateFiles(root, IndexFilePrefix + "*" + IndexFileSuffix, SearchOption.TopDirectoryOnly))
        {
            if (!file.EndsWith(IndexFileSuffix, StringComparison.OrdinalIgnoreCase)) continue;
            if (skip is not null && string.Equals(Path.GetFullPath(file), skip, StringComparison.OrdinalIgnoreCase)) continue;
            IndexDocument? doc;
            try
            {
                using FileStream stream = OpenForRead(file);
                doc = JsonSerializer.Deserialize(stream, ChunkIndexJsonContext.Default.IndexDocument);
            }
            catch (FileNotFoundException)
            {
                continue; // removed between enumeration and open
            }
            catch (Exception ex) when (ex is JsonException or IOException or UnauthorizedAccessException)
            {
                throw new InvalidDataException($"Index file '{file}' cannot be read ({ex.Message}); eviction cannot be decided.", ex);
            }
            if (doc?.Files is null)
                throw new InvalidDataException($"Index file '{file}' has no file list; eviction cannot be decided.");
            foreach (FileDocument? f in doc.Files)
            {
                foreach (ChunkDocument? c in f?.Chunks ?? [])
                {
                    if (c is null || !ChunkKey.IsWellFormed(c.Key))
                        throw new InvalidDataException($"Index file '{file}' has a malformed chunk entry; eviction cannot be decided.");
                    keys.Add(new ChunkKey(c.Key));
                }
            }
        }
        return keys;
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
        Dictionary<string, SourceFile>? current = null;
        if (currentFiles is not null)
        {
            current = new Dictionary<string, SourceFile>(PathComparer);
            foreach (SourceFile f in currentFiles)
                if (!current.TryAdd(NormalizePath(f.Path), f))
                    throw new ArgumentException($"Duplicate file path '{f.Path}'.", nameof(currentFiles));
        }
        var present = new Dictionary<string, bool>(StringComparer.Ordinal);
        var report = new List<FileStatus>();
        int total = 0, missingChunks = 0;
        long missingTokens = 0;

        IEnumerable<string> paths = current is null ? _files.Keys : _files.Keys.Concat(current.Keys).Distinct(PathComparer).Order(PathComparer);
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
            if (state != FileState.Indexed || missing.Count > 0) report.Add(new FileStatus(entry.Path, state, missing));
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
        return Path.Combine(Path.GetFullPath(storeRoot), IndexFilePrefix + chunkerName + IndexFileSuffix);
    }

    /// <summary>
    /// Opens an index file for reading the way <see cref="Load"/> does: sharing read, write and delete. Keep it open briefly:
    /// a concurrent <see cref="Save"/> waits (up to two seconds) for it to close.
    /// </summary>
    /// <param name="path">Index file.</param>
    /// <returns>The stream.</returns>
    public static FileStream OpenForRead(string path) =>
        new(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);

    /// <summary>
    /// Writes the index as JSON: to a unique temp file next to <paramref name="path"/>, flushed, then one rename over it. If two
    /// processes save the same index, the last rename wins (see the class remarks).
    /// </summary>
    /// <param name="path">Destination file.</param>
    public void Save(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        var doc = new IndexDocument
        {
            FormatVersion = FormatVersion,
            Producer = _producer,
            Chunker = ChunkerName,
            Files = _files.Values.Select(f => (FileDocument?)new FileDocument
            {
                Path = f.Path,
                Sha256 = f.TextHash,
                Chunks = f.Chunks.Select(r => (ChunkDocument?)new ChunkDocument
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
            MoveWithRetry(temp, full);
        }
        finally
        {
            if (File.Exists(temp)) File.Delete(temp);
        }
    }

    /// <summary>
    /// Loads an index saved by <see cref="Save"/>. A missing file gives an empty index. A file that is not valid JSON, has another
    /// format version, another producer (<paramref name="keyOf"/> differs from the one it was saved with), or any malformed
    /// entry (null entry, bad path or hash, duplicate path, bad key, chunks that do not tile the file in order, bad lines or
    /// token counts, no chunker) gives an empty index with a <see cref="LoadNote"/>, so everything is re-keyed rather than
    /// trusted. I/O errors (access denied, device errors) propagate.
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
            using FileStream stream = OpenForRead(path);
            doc = JsonSerializer.Deserialize(stream, ChunkIndexJsonContext.Default.IndexDocument);
        }
        catch (FileNotFoundException)
        {
            return index;
        }
        catch (JsonException ex)
        {
            return Empty(keyOf, $"index file is not valid JSON ({ex.Message})");
        }
        if (doc is null || doc.FormatVersion != FormatVersion)
            return Empty(keyOf, $"index format version {doc?.FormatVersion} is not {FormatVersion}");
        if (!string.Equals(doc.Producer, index._producer, StringComparison.Ordinal))
            return Empty(keyOf, "index was built for another producer (model, tokenizer, RoPE, backend or dtype)");
        if (doc.Files is null) return Empty(keyOf, "index has no file list");
        if (doc.Files.Count > 0 && string.IsNullOrEmpty(doc.Chunker)) return Empty(keyOf, "index has files but no chunker name");
        ChunkKind kind = doc.Chunker is null ? ChunkKind.Other : ChunkRef.KindOfName(doc.Chunker);

        foreach (FileDocument? f in doc.Files)
        {
            string? problem = Validate(f, out string norm);
            if (problem is null && index._files.ContainsKey(norm)) problem = $"duplicate path '{f!.Path}'";
            if (problem is not null) return Empty(keyOf, $"malformed index entry: {problem}");
            var chunks = f!.Chunks!.Select(c => new ChunkRef(norm, c!.StartLine, c.EndLine, c.StartOffset, c.EndOffset, kind)
            {
                TokenCount = c.Tokens,
                KeyHex = c.Key,
            }).ToList();
            index._files[norm] = new FileEntry(norm, f.Sha256!, chunks);
            foreach (ChunkRef r in chunks) index.AddRef(r.KeyHex!);
        }
        index.ChunkerName = doc.Chunker;
        return index;
    }

    /// <summary>How long <see cref="Save"/> keeps retrying the final rename while a reader holds the destination.</summary>
    internal static TimeSpan SaveRetryBudget { get; set; } = TimeSpan.FromSeconds(2);

    // Windows refuses to rename over a file that is open, even when the reader shares delete; readers hold the index only for
    // the length of one Load, so retry briefly, then fail loudly (the temp file is removed by the caller).
    private static void MoveWithRetry(string temp, string full)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        while (true)
        {
            try
            {
                File.Move(temp, full, overwrite: true);
                return;
            }
            catch (Exception ex) when (ex is UnauthorizedAccessException or IOException && File.Exists(temp))
            {
                if (sw.Elapsed >= SaveRetryBudget)
                    throw new IOException($"Index file '{full}' is held open by a reader; it was not replaced.", ex);
                Thread.Sleep(20);
            }
        }
    }

    private static ChunkIndex Empty(ChunkKeyFunction keyOf, string why) => new(keyOf) { LoadNote = why + "; starting empty" };

    /// <summary>Null when <paramref name="f"/> is well formed: a valid relative path, a SHA-256, and chunks that tile the text
    /// from offset 0 without gaps or overlaps, with 1-based ordered lines and non-negative token counts.</summary>
    private static string? Validate(FileDocument? f, out string norm)
    {
        norm = "";
        if (f is null) return "null file entry";
        if (f.Path is null) return "file entry without a path";
        try { norm = NormalizePath(f.Path); }
        catch (ArgumentException) { return $"bad path '{f.Path}'"; }
        if (!ChunkKey.IsWellFormed(f.Sha256)) return $"bad text hash for '{f.Path}'";
        if (f.Chunks is null) return $"no chunk list for '{f.Path}'";
        int offset = 0, line = 1;
        foreach (ChunkDocument? c in f.Chunks)
        {
            if (c is null) return $"null chunk in '{f.Path}'";
            if (!ChunkKey.IsWellFormed(c.Key)) return $"bad key in '{f.Path}'";
            if (c.StartOffset != offset || c.EndOffset <= c.StartOffset) return $"chunk offsets {c.StartOffset}-{c.EndOffset} in '{f.Path}' do not tile the file";
            if (c.StartLine < line || c.EndLine < c.StartLine) return $"chunk lines {c.StartLine}-{c.EndLine} in '{f.Path}' are out of order";
            if (c.Tokens < 0) return $"negative token count in '{f.Path}'";
            offset = c.EndOffset;
            line = c.EndLine;
        }
        return null;
    }

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

    private sealed record FileEntry(string Path, string TextHash, List<ChunkRef> Chunks);

    internal sealed class IndexDocument
    {
        public int FormatVersion { get; set; }
        public string? Producer { get; set; }
        public string? Chunker { get; set; }
        public List<FileDocument?>? Files { get; set; }
    }

    internal sealed class FileDocument
    {
        public string? Path { get; set; }
        public string? Sha256 { get; set; }
        public List<ChunkDocument?>? Chunks { get; set; }
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
