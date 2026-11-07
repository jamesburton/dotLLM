using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using DotLLM.ChunkReuse.CodeCorpus;

namespace DotLLM.ChunkReuse.Store;

/// <summary>
/// Store key of a chunk from its token ids, with every producer field (model fingerprint, tokenizer, RoPE, backend, dtype) and
/// the namespace fixed by the caller, e.g.
/// <c>ids =&gt; ChunkKey.Create(fingerprint, tokenizerId, rope, "cpu", KvDtype.F32, ids, @namespace: ns)</c>.
/// The index's producer identity is the key of no tokens, so it changes with any of them, the namespace included.
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
    /// <see cref="Added"/> order: what <i>this update</i> adds that the store lacks. Chunks in <see cref="Unchanged"/> (every
    /// chunk of a file whose text did not change) are not checked, so if the store was incomplete before the update
    /// (interrupted build, eviction by another index, a fresh store with an old index) they are not listed: use
    /// <see cref="ChunkIndex.Status"/> for everything missing, e.g. on a build's first update.
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
/// producer (model, tokenizer, RoPE, backend, dtype, namespace). It persists as JSON next to the store
/// (<see cref="DefaultPath(string, string, string, string, string)"/>: one index per chunker, model, producer and namespace, listed in the store
/// registry, <see cref="StoreRegistry"/>).
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
/// do not merge: the last rename wins and the other's update is lost, so writers serialise on the index lock.
/// </para>
/// <para>
/// <b>Lock model</b> (integration spec L3). Two kinds of exclusive lock files (<see cref="FileShare.None"/>, released on
/// dispose or when the process dies; the files themselves are never deleted, and a file left on disk is not a held lock):
/// <list type="bullet">
/// <item><b>Index lock</b> (<see cref="AcquireIndexLock(string, string, string, string, string, TimeSpan)"/>, one per index file): held from
/// <see cref="Load"/> through <see cref="Update"/> to <see cref="Save"/> by every writer of that index. Index-only updates
/// (a build without encoding) take only this lock.</item>
/// <item><b>Store lock</b> (<see cref="AcquireStoreLock"/>, one per store): held for short scopes only: one put batch
/// (<see cref="LockingChunkStore"/>); one server-ledger write (the batch's puts, <see cref="ServerLedger.Record"/> and
/// <see cref="ServerLedger.Save(StoreLockHandle)"/> in one scope, so no eviction sees an entry without its row); one registry save; one eviction
/// pass (<see cref="EvictStoreWideOrphans"/>, from reading the indexes and the ledger to the last delete).</item>
/// </list>
/// <b>Lock order:</b> an index lock, then the store lock; never an index lock while holding the store lock (refused with
/// <see cref="InvalidOperationException"/> for an index lock of the same store root while a live store-lock handle of that root
/// was acquired in this execution flow (handles acquired in a child flow and handed back are not seen; acquire and dispose in
/// one flow, and do not start long-lived work inside a scope); work started inside a store-lock scope inherits the
/// flow and is refused too). <b>Publication rule:</b> a writer saves its index (under the index lock) <i>before</i> putting the
/// entries it references, and decides which referenced entries are already stored (e.g. <see cref="Status"/>) under the store
/// lock <i>after</i> that save. An eviction pass that started earlier has then finished, and every later pass reads the new
/// index, so a key the writer saw stored is never evicted under it. A key evicted before that check is simply missing and is
/// encoded again; nothing is ever corrupt, only a miss. A lock timeout is <see cref="ChunkStoreLockTimeoutException"/>
/// (a <see cref="ChunkStoreBusyException"/>). The server's entries are referenced by the server ledger
/// (<see cref="ServerLedger"/>) instead of an index.
/// </para>
/// <para>
/// <b>Platforms.</b> Verified on Windows (NTFS): an exclusive handle refuses every other open, readers included, and the OS
/// closes a killed holder's handle. On Linux and macOS .NET emulates <see cref="FileShare.None"/> with an advisory
/// <c>flock</c>, which other .NET processes honour and the kernel releases on process exit, but this is not verified here, does
/// not bind non-.NET programs, and is disabled by <c>DOTNET_SYSTEM_IO_DISABLEFILELOCKING</c>; network file systems are not
/// supported.
/// </para>
/// <para>
/// <b>Known behaviour.</b> A case-only rename on Windows keeps the old spelling (the first one seen). An index saved on a
/// case-sensitive system with <c>A.cs</c> and <c>a.cs</c> loads empty on Windows (duplicate path). After a chunker change
/// the old <c>index-&lt;old&gt;.json</c> still protects its keys from eviction until it is deleted; delete it when switching
/// chunker.
/// </para>
/// </remarks>
public sealed partial class ChunkIndex
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

    /// <summary>
    /// Turns a solution scan's files into index input: one tree root and '/'-separated paths relative to it. Use it before
    /// <see cref="Update"/> on files from <c>SolutionScanner</c>, whose paths can start with <c>../</c> (a solution in a
    /// subfolder, linked files) or be absolute (another drive).
    /// </summary>
    /// <param name="scanFiles">Files whose paths are absolute, or relative to <paramref name="root"/>.</param>
    /// <param name="root">The scan's root directory (e.g. <c>ScanResult.RootDirectory</c>), or null when every path is absolute.</param>
    /// <returns>
    /// <c>Files</c>: the files with index paths, deduplicated by <see cref="PathComparer"/> (first occurrence kept), in input
    /// order. <c>Root</c>: the deepest common ancestor directory of <paramref name="root"/> (when given) and every kept file,
    /// absolute; persist it with the index, since index paths are relative to it and a different root re-paths every file.
    /// <c>Excluded</c>: input paths on another volume than the root's (no relative form), to report or fail on.
    /// </returns>
    /// <exception cref="ArgumentException"><paramref name="root"/> is null and a path is relative, or <paramref name="root"/> is relative.</exception>
    /// <remarks>
    /// The volume is <paramref name="root"/>'s when given, else the volume holding most files (the first one on a tie).
    /// Turning an index path back into a disk path is <c>Path.Combine(Root, path)</c>.
    /// </remarks>
    public static (IReadOnlyList<SourceFile> Files, string Root, IReadOnlyList<string> Excluded) RelativizeScan(IEnumerable<SourceFile> scanFiles, string? root)
    {
        ArgumentNullException.ThrowIfNull(scanFiles);
        if (root is not null && !Path.IsPathFullyQualified(root))
            throw new ArgumentException($"The scan root must be absolute: '{root}'.", nameof(root));
        string? rootFull = root is null ? null : Path.TrimEndingDirectorySeparator(Path.GetFullPath(root));

        var absolute = new List<(SourceFile File, string Full)>();
        foreach (SourceFile f in scanFiles)
        {
            string full;
            if (Path.IsPathFullyQualified(f.Path)) full = Path.GetFullPath(f.Path);
            else if (rootFull is not null) full = Path.GetFullPath(Path.Combine(rootFull, f.Path));
            else throw new ArgumentException($"'{f.Path}' is relative but no scan root was given.", nameof(scanFiles));
            absolute.Add((f, full));
        }

        string? volume = rootFull is not null ? Path.GetPathRoot(rootFull)
            : absolute.GroupBy(a => Path.GetPathRoot(a.Full)!, PathComparer).OrderByDescending(g => g.Count()).Select(g => g.Key).FirstOrDefault();
        var excluded = new List<string>();
        var kept = new List<(SourceFile File, string Full)>();
        foreach ((SourceFile f, string full) in absolute)
        {
            if (PathComparer.Equals(Path.GetPathRoot(full), volume)) kept.Add((f, full));
            else excluded.Add(f.Path);
        }

        // Deepest common ancestor directory, compared segment by segment with PathComparer.
        IEnumerable<string> dirs = kept.Select(k => Path.GetDirectoryName(k.Full)!);
        if (rootFull is not null) dirs = dirs.Prepend(rootFull);
        string[]? common = null;
        foreach (string d in dirs)
        {
            string[] parts = d.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            if (common is null) { common = parts; continue; }
            int n = 0;
            while (n < common.Length && n < parts.Length && PathComparer.Equals(common[n], parts[n])) n++;
            common = common[..n];
        }
        // A one-segment ancestor is the volume itself: "C:" -> "C:\", "" (POSIX) -> "/".
        string commonRoot = common is null || common.Length == 0 ? volume ?? ""
            : common.Length == 1 ? common[0] + Path.DirectorySeparatorChar
            : string.Join(Path.DirectorySeparatorChar, common);

        var files = new List<SourceFile>();
        var seen = new HashSet<string>(PathComparer);
        foreach ((SourceFile f, string full) in kept)
        {
            string rel = NormalizePath(Path.GetRelativePath(commonRoot, full));
            if (seen.Add(rel)) files.Add(f with { Path = rel });
        }
        return (files, commonRoot, excluded);
    }

    /// <summary>
    /// Mark-and-sweep: the keys in <paramref name="store"/> that neither an index file under <paramref name="storeRoot"/> nor
    /// the server ledger (<see cref="ServerLedger"/>) references (<see cref="ReferencedKeysUnder"/>). Unlike <see cref="EvictableOrphans"/>, this also finds orphans left by earlier
    /// updates that were saved but not evicted. To evict, use <see cref="EvictStoreWideOrphans"/>, which computes this and deletes
    /// the result inside one store-lock scope (or hold <see cref="AcquireStoreLock"/> yourself from this call to the last
    /// delete); see the class remarks for why concurrent index-only updates are then safe.
    /// </summary>
    /// <param name="store">The store.</param>
    /// <param name="storeRoot">The store directory (where the index files are).</param>
    /// <returns>Keys to evict, in ordinal order.</returns>
    /// <exception cref="InvalidDataException">An index file under the store cannot be read.</exception>
    /// <exception cref="LedgerUnreadableException">The server ledger exists but cannot be read (fail closed: nothing is an orphan).</exception>
    public static IReadOnlyList<ChunkKey> StoreWideOrphans(IChunkStore store, string storeRoot)
    {
        ArgumentNullException.ThrowIfNull(store);
        IReadOnlySet<ChunkKey> referenced = ReferencedKeysUnder(storeRoot);
        return store.Keys().Where(k => !referenced.Contains(k)).OrderBy(k => k.Hex, StringComparer.Ordinal).ToList();
    }

    /// <summary>Name of the store-wide lock file (<see cref="AcquireStoreLock"/>), in the store root.</summary>
    public const string StoreLockFileName = ".chunkstore.lock";

    /// <summary>Extension of an index lock file (<see cref="IndexLockPath"/>); not <c>.json</c>, so never read as an index.</summary>
    public const string IndexLockSuffix = ".lock";

    /// <summary>
    /// The store-lock handles acquired in the current execution flow (and inherited by flows started from it), for the
    /// lock-order check only: a disposed handle no longer counts, and only handles of the same store root count. Proof that the
    /// lock is held is the handle itself (<see cref="StoreLockHandle.EnsureHeldFor"/>), never this list.
    /// </summary>
    private static readonly AsyncLocal<StoreLockHandle[]?> s_flowStoreLocks = new();

    /// <summary>Test hook: how many store-lock handles (held or released) this flow still tracks; released ones are pruned on the next acquire.</summary>
    internal static int FlowStoreLocksTracked => (s_flowStoreLocks.Value ?? []).Length;

    /// <summary>
    /// Takes the store-wide writer lock: an exclusively opened (<see cref="FileShare.None"/>) <see cref="StoreLockFileName"/> in
    /// the store root, released on dispose or when the process dies. Hold it for <b>short scopes only</b> (integration spec L3):
    /// one put batch (<see cref="LockingChunkStore"/>), one server-ledger write (put, record and
    /// <see cref="ServerLedger.Save(StoreLockHandle)"/> of one batch in one scope), one store-registry save, one eviction pass
    /// (<see cref="EvictStoreWideOrphans"/>: from reading the indexes and the ledger to the last delete). Index-only updates do
    /// not take it (<see cref="AcquireIndexLock(string, string, string, string, string, TimeSpan)"/>). Readers (status, assembly,
    /// select) need no lock. See the class remarks for the full contract and the lock order.
    /// </summary>
    /// <param name="storeRoot">The store directory (created if missing).</param>
    /// <param name="timeout">How long to wait for another holder; <see cref="TimeSpan.Zero"/> tries once.</param>
    /// <returns>The lock; dispose to release (idempotent). Pass it to the saves that require it
    /// (<see cref="StoreRegistry.Save(StoreLockHandle)"/>, <see cref="ServerLedger.Save(StoreLockHandle)"/>). Use it in a
    /// <c>using</c> so an exception releases it too.</returns>
    /// <exception cref="ChunkStoreLockTimeoutException">Another holder (process, or handle in this process) kept the lock for
    /// <paramref name="timeout"/>. The store lock is not re-entrant: a second acquire in the same process waits and times out.</exception>
    public static StoreLockHandle AcquireStoreLock(string storeRoot, TimeSpan timeout)
    {
        ArgumentException.ThrowIfNullOrEmpty(storeRoot);
        ArgumentOutOfRangeException.ThrowIfLessThan(timeout, TimeSpan.Zero);
        string spelled = Path.TrimEndingDirectorySeparator(Path.GetFullPath(storeRoot));
        StorePaths.CreateStoreDirectory(spelled);
        string root = StorePaths.CanonicalRoot(spelled); // the real path: a link, junction or short name of the root is the same root
        FileStream stream = OpenLockFile(Path.Combine(root, StoreLockFileName), timeout);
        var handle = new StoreLockHandle(root, stream);
        StoreLockHandle[] live = (s_flowStoreLocks.Value ?? []).Where(h => h.IsHeld).ToArray();
        s_flowStoreLocks.Value = [.. live, handle];
        return handle;
    }

    /// <summary>
    /// Takes the index lock of the index file <paramref name="indexPath"/> (an <c>index-*.json</c> path; its lock is the same
    /// path with <see cref="IndexLockSuffix"/>), with the semantics of
    /// <see cref="AcquireIndexLock(string, string, string, string, string, TimeSpan)"/>. Use it for an index whose path is not a
    /// <see cref="DefaultPath(string, string, string, string, string)"/> (a legacy name being migrated, a test fixture).
    /// </summary>
    /// <param name="indexPath">The index file (need not exist).</param>
    /// <param name="timeout">How long to wait for another holder.</param>
    /// <returns>The lock; dispose to release (idempotent).</returns>
    /// <exception cref="ArgumentException">The path is not an <c>index-*.json</c> file name.</exception>
    /// <exception cref="ChunkStoreLockTimeoutException">Another holder kept this index's lock for <paramref name="timeout"/>.</exception>
    /// <exception cref="InvalidOperationException">The current flow holds the store lock of the same store root (lock order).</exception>
    public static IDisposable AcquireIndexLockFor(string indexPath, TimeSpan timeout)
    {
        ArgumentException.ThrowIfNullOrEmpty(indexPath);
        string full = Path.GetFullPath(indexPath);
        string name = Path.GetFileName(full);
        if (!name.StartsWith(IndexFilePrefix, StringComparison.Ordinal) || !name.EndsWith(IndexFileSuffix, StringComparison.Ordinal)
            || name.Length <= IndexFilePrefix.Length + IndexFileSuffix.Length)
            throw new ArgumentException($"'{indexPath}' is not an index file (index-*.json).", nameof(indexPath));
        return AcquireIndexLockAt(LockPathOf(full), timeout);
    }

    /// <summary>Takes the index lock at <paramref name="lockPath"/> (see <see cref="AcquireIndexLock(string, string, string, string, string, TimeSpan)"/>).</summary>
    private static IDisposable AcquireIndexLockAt(string lockPath, TimeSpan timeout)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(timeout, TimeSpan.Zero);
        string spelled = Path.GetDirectoryName(lockPath)!;
        if (FlowHoldsStoreLock(spelled))
            throw new InvalidOperationException($"Lock order violation: the index lock '{lockPath}' was requested while this flow holds "
                + "the store lock of the same store. Take index locks before the store lock (and release the store lock first).");
        StorePaths.CreateStoreDirectory(spelled);
        return new IndexLockHandle(OpenLockFile(lockPath, timeout));
    }

    /// <summary>Whether the current flow holds the store lock of <paramref name="storeRoot"/> (compared by real path).</summary>
    internal static bool FlowHoldsStoreLock(string storeRoot)
    {
        StoreLockHandle[] held = (s_flowStoreLocks.Value ?? []).Where(h => h.IsHeld).ToArray();
        if (held.Length == 0) return false;
        string root = StorePaths.CanonicalRoot(storeRoot);
        return held.Any(h => PathComparer.Equals(h.StoreRoot, root));
    }

    /// <summary>The lock file of an index file: its <c>.json</c> extension replaced by <see cref="IndexLockSuffix"/>.</summary>
    private static string LockPathOf(string indexPath) => Path.ChangeExtension(indexPath, IndexLockSuffix);

    /// <summary>
    /// Opens <paramref name="path"/> exclusively, retrying every 20 ms until <paramref name="timeout"/> while another handle holds
    /// it (a sharing violation). Access denied is NOT "busy": a read-only lock file or an unwritable store is a configuration
    /// error and surfaces at once as <see cref="UnauthorizedAccessException"/> instead of looking like a lock timeout (lock files
    /// are never deleted, so there is no pending-delete case to wait out). The file is created if missing, owner-only on POSIX,
    /// and never deleted: an existing lock file is not a held lock, only an open handle is (the OS closes a crashed holder's
    /// handle). Deleting a lock file would break the lock on POSIX (a new file at the same path is another inode).
    /// </summary>
    private static FileStream OpenLockFile(string path, TimeSpan timeout)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        while (true)
        {
            try
            {
                return StorePaths.OpenExclusive(path);
            }
            catch (IOException ex) when (StorePaths.IsSharingViolation(ex))
            {
                if (sw.Elapsed >= timeout) throw new ChunkStoreLockTimeoutException(path, timeout, ex);
                Thread.Sleep(20);
            }
        }
    }

    /// <summary>An acquired index lock: the exclusive handle.</summary>
    private sealed class IndexLockHandle(FileStream stream) : IDisposable
    {
        private int _disposed;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0) stream.Dispose();
        }
    }

    /// <summary>Test hook: runs inside <see cref="EvictStoreWideOrphans"/>, under the store lock, before the first delete.</summary>
    private static readonly AsyncLocal<Action<IReadOnlyList<ChunkKey>>?> s_beforeOrphanDeletes = new();

    /// <remarks>Per async flow, so a hook one test sets never runs inside another test's eviction on a parallel thread.</remarks>
    internal static Action<IReadOnlyList<ChunkKey>>? BeforeOrphanDeletes
    {
        get => s_beforeOrphanDeletes.Value;
        set => s_beforeOrphanDeletes.Value = value;
    }

    /// <summary>
    /// One eviction pass: takes the store lock, computes <see cref="StoreWideOrphans"/> and evicts them, all inside the one lock
    /// scope, so no put batch (which also takes the store lock) or registry or ledger save interleaves with it. An index-only
    /// update may run concurrently: it never references a key the pass deletes unless that key's entry was already stored, and
    /// a writer that relies on stored entries checks them under the store lock after saving its index (class remarks).
    /// </summary>
    /// <param name="store">The store at <paramref name="storeRoot"/> (unlocked; the pass holds the lock).</param>
    /// <param name="storeRoot">The store directory.</param>
    /// <param name="lockTimeout">How long to wait for the store lock.</param>
    /// <returns>The evicted keys, in ordinal order.</returns>
    /// <exception cref="ChunkStoreLockTimeoutException">The store lock was not obtained; nothing was deleted.</exception>
    /// <exception cref="InvalidDataException">An index file cannot be read; nothing was deleted.</exception>
    /// <exception cref="LedgerUnreadableException">The server ledger cannot be read; nothing was deleted.</exception>
    public static IReadOnlyList<ChunkKey> EvictStoreWideOrphans(IChunkStore store, string storeRoot, TimeSpan lockTimeout)
    {
        ArgumentNullException.ThrowIfNull(store);
        using IDisposable held = AcquireStoreLock(storeRoot, lockTimeout);
        IReadOnlyList<ChunkKey> victims = StoreWideOrphans(store, storeRoot);
        BeforeOrphanDeletes?.Invoke(victims);
        foreach (ChunkKey k in victims) store.Evict(k);
        return victims;
    }

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

        // Phase 1: decide and chunk every file, checking each chunker output, before anything changes.
        var plan = new List<(string Path, FileEntry? Old, FileEntry? New)>();
        IEnumerable<string> paths = (removeMissingFiles || !sameChunker ? given.Keys.Concat(_files.Keys) : given.Keys)
            .Distinct(PathComparer).Order(PathComparer).ToList();
        foreach (string path in paths)
        {
            _files.TryGetValue(path, out FileEntry? old);
            if (!given.TryGetValue(path, out SourceFile? file))
            {
                // Not in the tree (or, after a chunker change, chunked by the other chunker): drop its chunks.
                plan.Add((path, old, null));
                continue;
            }
            // Keep the spelling already indexed (case may differ on Windows).
            if (old is not null) file = file with { Path = old.Path };

            string hash = HashText(file.Text);
            if (old is not null && sameChunker && old.TextHash == hash && !old.Legacy)
            {
                plan.Add((path, old, old));
                continue;
            }

            var chunks = new List<ChunkRef>();
            foreach (Chunk c in chunker.Chunk(file))
                chunks.Add(ChunkRef.From(c, kind, _keyOf(c.TokenIds)));
            string? problem = CheckTiling(chunks.Select(r => (r.StartOffset, r.EndOffset, r.StartLine, r.EndLine, r.TokenCount ?? 0)), file.Text.Length);
            if (problem is not null)
                throw new InvalidOperationException($"Chunker '{chunker.Name}' produced chunks that do not tile '{file.Path}': {problem}. "
                    + "An index needs chunks that cover the file in order, from offset 0 to its end, without gaps or overlaps.");
            plan.Add((path, old, new FileEntry(file.Path, hash, chunks)));
        }

        // Phase 2: apply.
        var added = new List<ChunkRef>();
        var removed = new List<ChunkRef>();
        var unchanged = new List<ChunkRef>();
        var touchedKeys = new HashSet<string>(StringComparer.Ordinal);
        foreach ((string path, FileEntry? old, FileEntry? now) in plan)
        {
            if (now is null)
            {
                Release(old!, removed, touchedKeys);
                _files.Remove(path);
                continue;
            }
            if (ReferenceEquals(old, now))
            {
                unchanged.AddRange(old!.Chunks);
                continue;
            }

            // Match new chunks to old ones by key, as a multiset (a file may hold the same chunk text twice).
            var oldByKey = new Dictionary<string, Queue<ChunkRef>>(StringComparer.Ordinal);
            foreach (ChunkRef r in old?.Chunks ?? [])
            {
                if (!oldByKey.TryGetValue(r.KeyHex!, out Queue<ChunkRef>? q)) oldByKey[r.KeyHex!] = q = new Queue<ChunkRef>();
                q.Enqueue(r);
            }
            foreach (ChunkRef r in now.Chunks)
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
            _files[now.Path] = now;
        }

        ChunkerName = _files.Count > 0 || given.Count > 0 ? chunker.Name : ChunkerName;
        var orphaned = touchedKeys.Where(k => !_refCounts.ContainsKey(k)).Order(StringComparer.Ordinal).Select(k => new ChunkKey(k)).ToList();
        return new IndexDiff(added, removed, unchanged) { OrphanedKeys = orphaned };
    }

    /// <summary>
    /// The keys of <paramref name="diff"/>'s <see cref="IndexDiff.OrphanedKeys"/> that are safe to evict from the store at
    /// <paramref name="storeRoot"/>: referenced neither by this index nor by any other index file under the store root
    /// (<c>index-*.json</c>, e.g. the other chunker's). This index's own file (<paramref name="ownIndexPath"/>) is skipped, since
    /// it may still hold the references this update removed. Passing the wrong own path is safe but conservative (a stale own
    /// file keeps its keys).
    /// </summary>
    /// <param name="diff">A diff returned by <see cref="Update"/> on this index.</param>
    /// <param name="storeRoot">The store directory.</param>
    /// <param name="ownIndexPath">This index's file (e.g. <see cref="DefaultPath(string, string, string, string, string)"/>).</param>
    /// <returns>Keys to evict, in ordinal order.</returns>
    /// <exception cref="InvalidDataException">Another index file under the store cannot be read: eviction cannot be decided.</exception>
    public IReadOnlyList<ChunkKey> EvictableOrphans(IndexDiff diff, string storeRoot, string ownIndexPath)
    {
        ArgumentNullException.ThrowIfNull(diff);
        ArgumentException.ThrowIfNullOrEmpty(ownIndexPath);
        string own = ownIndexPath;
        IReadOnlySet<ChunkKey> elsewhere = ReferencedKeysUnder(storeRoot, own);
        return diff.OrphanedKeys.Where(k => !_refCounts.ContainsKey(k.Hex) && !elsewhere.Contains(k)).ToList();
    }

    /// <summary>
    /// Union of the keys referenced by every index file (<c>index-*.json</c>) directly under <paramref name="storeRoot"/>,
    /// whatever its producer or chunker, and by the server ledger (<see cref="ServerLedger"/>, every namespace). Use it
    /// before evicting from a store that several indexes or the server share.
    /// </summary>
    /// <param name="storeRoot">The store directory.</param>
    /// <param name="exceptPath">An index file to skip (typically the caller's own, about to be rewritten), or null. The
    /// ledger is never skipped.</param>
    /// <returns>The keys.</returns>
    /// <exception cref="InvalidDataException">An index file cannot be read or parsed (conservative: no guess).</exception>
    /// <exception cref="LedgerUnreadableException">The server ledger exists but cannot be read (conservative: no guess).</exception>
    public static IReadOnlySet<ChunkKey> ReferencedKeysUnder(string storeRoot, string? exceptPath = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(storeRoot);
        var keys = IndexKeysUnder(storeRoot, exceptPath);
        string root = Path.GetFullPath(storeRoot);
        if (!Directory.Exists(root)) return keys;
        keys.UnionWith(ServerLedger.ReadKeys(root));
        return keys;
    }

    /// <summary>
    /// The keys referenced by the index files (<c>index-*.json</c>) directly under <paramref name="storeRoot"/> only, without
    /// the server ledger: what the server's budget eviction must never delete the file of (Phase I Task 15, rule (c): a
    /// ledger row may go, but a key an index references keeps its entry file). For "may this key be deleted" use
    /// <see cref="ReferencedKeysUnder"/>, which adds the ledger.
    /// </summary>
    /// <param name="storeRoot">The store directory.</param>
    /// <returns>The keys (empty when the store root does not exist).</returns>
    /// <exception cref="InvalidDataException">An index file cannot be read or parsed (conservative: no guess).</exception>
    public static IReadOnlySet<ChunkKey> IndexReferencedKeysUnder(string storeRoot)
    {
        ArgumentException.ThrowIfNullOrEmpty(storeRoot);
        return IndexKeysUnder(storeRoot, null);
    }

    private static HashSet<ChunkKey> IndexKeysUnder(string storeRoot, string? exceptPath)
    {
        var keys = new HashSet<ChunkKey>();
        string root = Path.GetFullPath(storeRoot);
        if (!Directory.Exists(root)) return keys;
        string? skip = exceptPath is null ? null : Path.GetFullPath(exceptPath);
        foreach (string file in Directory.EnumerateFiles(root, IndexFilePrefix + "*" + IndexFileSuffix, SearchOption.TopDirectoryOnly))
        {
            if (!file.EndsWith(IndexFileSuffix, StringComparison.OrdinalIgnoreCase)) continue;
            if (skip is not null && PathComparer.Equals(Path.GetFullPath(file), skip)) continue;
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
                throw new InvalidDataException($"Index file '{file}' cannot be read ({ex.Message}); eviction cannot be decided. Repair or remove it if it is not an index of this store.", ex);
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

    /// <summary>
    /// The <b>legacy</b> (before Phase I Task 4b) index file name <c>&lt;storeRoot&gt;/index-&lt;chunkerName&gt;.json</c>, one
    /// per chunker for every model and namespace. New indexes use <see cref="DefaultPath(string, string, string, string, string)"/>; a
    /// legacy file is still an index (it protects its keys from eviction) and is renamed once by
    /// <see cref="MigrateLegacyIndex"/> when its producer matches.
    /// </summary>
    /// <param name="storeRoot">Store directory.</param>
    /// <param name="chunkerName">Chunker name, e.g. <c>symbol-512</c>; any plain file name part.</param>
    /// <returns>The path.</returns>
    public static string DefaultPath(string storeRoot, string chunkerName)
    {
        ArgumentException.ThrowIfNullOrEmpty(storeRoot);
        ArgumentException.ThrowIfNullOrEmpty(chunkerName);
        if (chunkerName.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 || chunkerName.IndexOfAny(['/', '\\', ':']) >= 0)
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
    /// processes save the same index, the last rename wins (see the class remarks). Temp files of this index older than an hour
    /// (left by a crashed writer) are removed first.
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
                    // A legacy entry not yet re-chunked stays legacy on disk (so a later load still re-chunks it).
                    Tokens = r.TokenCount ?? 0, Key = r.KeyHex!, WireEligible = f.Legacy ? null : r.WireEligible,
                }).ToList(),
            }).ToList(),
        };
        string full = Path.GetFullPath(path);
        string dir = Path.GetDirectoryName(full)!;
        StorePaths.CreateStoreDirectory(dir);
        SweepStaleTemps(dir, Path.GetFileName(full));
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
            // Best effort: a cleanup failure must not replace the save's own exception (a leftover temp is swept by a later save).
            try { if (File.Exists(temp)) File.Delete(temp); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
        }
    }

    internal static void SweepStaleTemps(string dir, string indexFileName)
    {
        DateTime cutoff = DateTime.UtcNow - TimeSpan.FromHours(1);
        foreach (string tmp in Directory.EnumerateFiles(dir, indexFileName + ".*.tmp", SearchOption.TopDirectoryOnly))
        {
            try
            {
                if (File.GetLastWriteTimeUtc(tmp) < cutoff) File.Delete(tmp);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // Best effort: a temp file still held by its writer is left for a later save.
            }
        }
    }

    /// <summary>
    /// Loads an index saved by <see cref="Save"/>. A missing file gives an empty index.
    /// <list type="bullet">
    /// <item>Not valid JSON, another format version, another producer (<paramref name="keyOf"/> differs from the one it was
    /// saved with), or a structurally malformed entry (null entry, bad path or hash, duplicate path, null chunk, bad key, no
    /// chunker): an empty index with a <see cref="LoadNote"/>, so everything is re-keyed rather than trusted.</item>
    /// <item>A file entry whose chunks do not tile the file (offset 0 first, contiguous, non-negative lengths, ordered 1-based
    /// lines, non-negative token counts): only that file is dropped (it is re-chunked on the next <see cref="Update"/>) and
    /// the <see cref="LoadNote"/> names it.</item>
    /// <item>A file entry written before Phase I Task 5 (chunks without <c>wireEligible</c>): whole-line <c>symbol-*</c> chunks
    /// load as wire-eligible and the file is re-chunked on the next <see cref="Update"/> even if its text is unchanged (it was
    /// tokenised before Task 5b); any other such entry (token slices, other chunkers) is dropped like a non-tiling one.</item>
    /// </list>
    /// I/O errors (access denied, device errors) propagate.
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

        var seen = new HashSet<string>(PathComparer);
        var dropped = new List<string>();
        foreach (FileDocument? f in doc.Files)
        {
            string? problem = Validate(f, out string norm);
            if (problem is null && !seen.Add(norm)) problem = $"duplicate path '{f!.Path}'";
            if (problem is not null) return Empty(keyOf, $"malformed index entry: {problem}");
            string? tiling = CheckTiling(f!.Chunks!.Select(c => (c!.StartOffset, c.EndOffset, c.StartLine, c.EndLine, c.Tokens)), textLength: null);
            if (tiling is not null)
            {
                dropped.Add($"'{norm}' ({tiling})");
                continue;
            }
            // Written before Phase I Task 5 (no wireEligible): whole-line symbol chunks are eligible; a file with token slices,
            // or of another chunker, cannot be told apart from text-first pieces, so it is dropped and re-chunked. Such an entry
            // was also tokenised before Task 5b (special-token spellings parsed), so the next update re-chunks it (Legacy).
            bool legacy = f.Chunks!.Any(c => c!.WireEligible is null);
            if (legacy && (kind != ChunkKind.Symbol || HasSharedLine(f.Chunks!)))
            {
                dropped.Add($"'{norm}' (written before wire eligibility was recorded, and not whole-line symbol chunks)");
                continue;
            }
            var chunks = f.Chunks!.Select(c => new ChunkRef(norm, c!.StartLine, c.EndLine, c.StartOffset, c.EndOffset, kind)
            {
                TokenCount = c.Tokens,
                KeyHex = c.Key,
                WireEligible = c.WireEligible ?? true,
            }).ToList();
            index._files[norm] = new FileEntry(norm, f.Sha256!, chunks) { Legacy = legacy };
            foreach (ChunkRef r in chunks) index.AddRef(r.KeyHex!);
        }
        index.ChunkerName = doc.Chunker;
        if (dropped.Count > 0)
            index.LoadNote = $"dropped {dropped.Count} file entr{(dropped.Count == 1 ? "y" : "ies")} whose chunks do not tile the file "
                + $"or predate wire eligibility (re-chunked on the next update): {string.Join("; ", dropped)}";
        return index;
    }

    /// <summary>How long <see cref="Save"/> keeps retrying the final rename while a reader holds the destination.</summary>
    internal static TimeSpan SaveRetryBudget
    {
        get => s_saveRetryBudget.Value ?? TimeSpan.FromSeconds(2);
        set => s_saveRetryBudget.Value = value;
    }

    // Per async flow (a test shortens it for itself only; a process-wide value raced between parallel test classes).
    private static readonly AsyncLocal<TimeSpan?> s_saveRetryBudget = new();

    private const int ErrorAccessDenied = unchecked((int)0x80070005);
    private const int ErrorSharingViolation = unchecked((int)0x80070020);
    private const int ErrorLockViolation = unchecked((int)0x80070021);

    // Windows refuses to rename over a file that is open, even when the reader shares delete; readers hold the index only for
    // the length of one Load, so retry briefly on sharing/access errors only, then fail (the caller removes the temp file).
    internal static void MoveWithRetry(string temp, string full)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        while (true)
        {
            try
            {
                File.Move(temp, full, overwrite: true);
                return;
            }
            catch (Exception ex) when (ex is UnauthorizedAccessException
                || (ex is IOException && ex.HResult is ErrorAccessDenied or ErrorSharingViolation or ErrorLockViolation))
            {
                if (sw.Elapsed >= SaveRetryBudget)
                    throw new IOException($"File '{full}' could not be replaced within {SaveRetryBudget.TotalSeconds:0.#} s "
                        + $"(sharing violation or access denied; e.g. a reader holds it open, or the file is read-only): {ex.Message}", ex);
                Thread.Sleep(20);
            }
        }
    }

    private static ChunkIndex Empty(ChunkKeyFunction keyOf, string why) => new(keyOf) { LoadNote = why + "; starting empty" };

    /// <summary>Null when <paramref name="f"/> is structurally well formed: a valid relative path, a SHA-256, a chunk list
    /// without null entries, and well-formed keys. Tiling is checked separately (<see cref="CheckTiling"/>).</summary>
    private static string? Validate(FileDocument? f, out string norm)
    {
        norm = "";
        if (f is null) return "null file entry";
        if (f.Path is null) return "file entry without a path";
        try { norm = NormalizePath(f.Path); }
        catch (ArgumentException) { return $"bad path '{f.Path}'"; }
        if (!ChunkKey.IsWellFormed(f.Sha256)) return $"bad text hash for '{f.Path}'";
        if (f.Chunks is null) return $"no chunk list for '{f.Path}'";
        foreach (ChunkDocument? c in f.Chunks)
        {
            if (c is null) return $"null chunk in '{f.Path}'";
            if (!ChunkKey.IsWellFormed(c.Key)) return $"bad key in '{f.Path}'";
        }
        return null;
    }

    /// <summary>
    /// The chunk contract an index relies on, checked on chunker output in <see cref="Update"/> and on every file entry in
    /// <see cref="Load"/>: the first chunk starts at offset 0, each next one starts where the previous ended, no chunk ends
    /// before it starts (zero-length chunks are allowed: decode-based offsets of a byte-level BPE can produce one at a
    /// multi-byte boundary), lines are 1-based, ordered and never go back, token counts are non-negative, and (when
    /// <paramref name="textLength"/> is known) the last chunk ends at the end of the text.
    /// </summary>
    /// <returns>Null when the chunks tile the file, else the first problem.</returns>
    private static string? CheckTiling(IEnumerable<(int StartOffset, int EndOffset, int StartLine, int EndLine, int Tokens)> chunks, int? textLength)
    {
        int offset = 0, line = 1, i = 0;
        foreach ((int so, int eo, int sl, int el, int tokens) in chunks)
        {
            if (so != offset) return $"chunk {i} starts at offset {so}, expected {offset}";
            if (eo < so) return $"chunk {i} ends at offset {eo} before it starts ({so})";
            if (sl < line || el < sl) return $"chunk {i} lines {sl}-{el} go back or are inverted (previous ended on line {line})";
            if (tokens < 0) return $"chunk {i} has a negative token count";
            offset = eo;
            line = el;
            i++;
        }
        if (textLength is int n && offset != n) return $"chunks end at offset {offset}, the text has {n} characters";
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

    /// <summary>Whether two neighbouring chunks share a line: token-id slices of one over-long line (whole-line chunks never do).</summary>
    private static bool HasSharedLine(List<ChunkDocument?> chunks)
    {
        for (int i = 1; i < chunks.Count; i++)
            if (chunks[i]!.StartLine == chunks[i - 1]!.EndLine) return true;
        return false;
    }

    /// <param name="Path">Indexed path.</param>
    /// <param name="TextHash">SHA-256 of the text the chunks were made from.</param>
    /// <param name="Chunks">The chunks, in text order.</param>
    private sealed record FileEntry(string Path, string TextHash, List<ChunkRef> Chunks)
    {
        /// <summary>Loaded from an index written before Phase I Task 5: re-chunked on the next update even if unchanged.</summary>
        public bool Legacy { get; init; }
    }

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

        /// <summary>Null in index files written before Phase I Task 5 (see <see cref="Load"/>).</summary>
        public bool? WireEligible { get; set; }
    }
}

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
[JsonSerializable(typeof(ChunkIndex.IndexDocument))]
internal sealed partial class ChunkIndexJsonContext : JsonSerializerContext;
