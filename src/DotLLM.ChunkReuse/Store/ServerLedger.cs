using System.Text.Json;
using System.Text.Json.Serialization;
using DotLLM.ChunkReuse.Tooling;

namespace DotLLM.ChunkReuse.Store;

/// <summary>One entry the server wrote (<see cref="ServerLedger"/>).</summary>
/// <param name="KeyHex">The entry's key (64 lowercase hex).</param>
/// <param name="Namespace">The namespace the key was made in (<see cref="ChunkNamespace"/>); a key belongs to exactly one.</param>
/// <param name="SourcePath">First-seen provenance: the tree-relative, '/'-separated <c>source.path</c> of the first sighting that
/// gave one, or null.</param>
/// <param name="Bytes">Size of the entry file in bytes, as the writer recorded it.</param>
/// <param name="FirstSeen">When the row was first recorded.</param>
/// <param name="LastUsed">Latest write or hit (never moves backwards).</param>
public sealed record LedgerRow(string KeyHex, string Namespace, string? SourcePath, long Bytes, DateTimeOffset FirstSeen, DateTimeOffset LastUsed);

/// <summary>
/// The server ledger (<see cref="ServerLedger.DefaultPath"/>) exists but cannot be read: not JSON, an unknown format version,
/// a malformed row, or an I/O error. Eviction must not be decided without it (the server's entries would look like orphans),
/// so <see cref="ChunkIndex.StoreWideOrphans"/> and <see cref="ChunkIndex.ReferencedKeysUnder"/> throw this and nothing is
/// deleted. The file is left as it is for repair. The message names the file but never a namespace value.
/// </summary>
/// <remarks>An <see cref="IOException"/> (<see cref="InvalidDataException"/>, which unreadable index files throw, is sealed).</remarks>
public sealed class LedgerUnreadableException : IOException
{
    /// <summary>Creates the exception.</summary>
    /// <param name="path">The ledger file.</param>
    /// <param name="reason">What is wrong with it.</param>
    /// <param name="inner">The underlying failure, if any.</param>
    public LedgerUnreadableException(string path, string reason, Exception? inner = null)
        : base($"Server ledger '{path}' cannot be read ({reason}); eviction cannot be decided and nothing was deleted. "
            + "Repair or restore the file; deleting it would expose every server-written entry to orphan eviction.", inner)
        => Path = path;

    /// <summary>The ledger file.</summary>
    public string Path { get; }
}

/// <summary>
/// The server's record of the store entries it wrote (spec §14 L2): <c>&lt;storeRoot&gt;/ledger-server.json</c>, one row per
/// key with its namespace, first-seen provenance, size and first / last use. Index files reference the CLI's entries; this
/// ledger references the server's, so <see cref="ChunkIndex.StoreWideOrphans"/> = store keys minus index keys minus ledger
/// keys. A ledger key stops being protected only when the ledger itself drops it (<see cref="Remove"/>: forget, budget
/// eviction), or when its entry file is gone at <see cref="Load"/>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Format</b> (version <see cref="FormatVersion"/>): <c>{"formatVersion": 1, "rows": [{"key", "namespace", "sourcePath",
/// "bytes", "firstSeen", "lastUsed"}]}</c>, camelCase, rows sorted by namespace then key (ordinal), so each namespace's rows
/// form one contiguous section and identical content gives identical bytes. Unknown properties are ignored (a later writer
/// of the same version may add fields); any other version, a missing field, a repeated property, a malformed or duplicate row,
/// a directory at the ledger path, or a probe failure other than "not found" is
/// <see cref="LedgerUnreadableException"/>, never an empty ledger: reading less than the writer wrote would make its entries
/// orphans.
/// </para>
/// <para>
/// <b>Writers.</b> The server that owns the store is the only writer (L1). <see cref="Save"/> writes a temp file next to the
/// ledger, flushes it and renames it over the ledger in one step, so a reader sees the old or the new ledger, never a part; a
/// crash leaves at most a temp file, which no reader looks at and a later save sweeps after an hour. Two instances saving
/// the same ledger do not merge (the last rename wins). Order: put the entry file first, then record and save the row, all
/// inside <b>one</b> <see cref="ChunkIndex.AcquireStoreLock"/> scope per batch (put, <see cref="Record"/>, <see cref="Save"/>;
/// never release the lock between the put and the save, and never encode while holding it): an eviction pass
/// (<see cref="ChunkIndex.EvictStoreWideOrphans"/>) holds the same lock from reading the ledger to its last delete, so it never
/// sees a new entry without its row. A row whose file is missing (crash between the two, or a manual delete) is dropped on
/// load. The ledger needs no index lock; a writer that also takes an index lock takes it before the store lock
/// (<see cref="ChunkIndex"/> remarks, lock order). The CLI never writes the ledger.
/// </para>
/// <para>
/// <b>Size.</b> No cap: the whole file is read on <see cref="Load"/> (one entry probe per row) and on every eviction pass.
/// The ledger is server-written and bounded by the store budget (~100k rows is tens of MB), which is acceptable.
/// </para>
/// <para>An instance is not thread-safe.</para>
/// </remarks>
public sealed class ServerLedger
{
    /// <summary>The ledger format version this build reads and writes.</summary>
    public const int FormatVersion = 1;

    /// <summary>File name of the ledger in the store root. Not <c>index-*</c>, so it is never read as an index.</summary>
    public const string FileName = "ledger-server.json";

    private readonly string _root;
    private readonly Dictionary<string, LedgerRow> _rows = new(StringComparer.Ordinal);

    private ServerLedger(string root) => _root = root;

    /// <summary>The ledger file of a store: <c>&lt;storeRoot&gt;/ledger-server.json</c>.</summary>
    /// <param name="storeRoot">Store directory.</param>
    /// <returns>The absolute path.</returns>
    public static string DefaultPath(string storeRoot)
    {
        ArgumentException.ThrowIfNullOrEmpty(storeRoot);
        return System.IO.Path.Combine(System.IO.Path.GetFullPath(storeRoot), FileName);
    }

    /// <summary>Rows dropped by <see cref="Load"/> because their entry file was missing; the next <see cref="Save"/> persists the drop.</summary>
    public int DroppedOnLoad { get; private set; }

    private static readonly AsyncLocal<Func<string, FileAttributes>?> s_probeHook = new();

    /// <summary>Test hook (per async flow): replaces <see cref="File.GetAttributes(string)"/> for the ledger and entry probes.</summary>
    internal static Func<string, FileAttributes>? ProbeHook
    {
        get => s_probeHook.Value;
        set => s_probeHook.Value = value;
    }

    private static FileAttributes Probe(string path) => (ProbeHook ?? File.GetAttributes)(path);

    /// <summary>
    /// Rows kept by <see cref="Load"/> although their entry file's state could not be determined (access denied, a sharing or
    /// device error, a directory at the entry path). Kept rather than dropped: dropping would make a live entry an orphan.
    /// </summary>
    public int UndeterminedOnLoad { get; private set; }

    /// <summary>Test hook: called with the flushed temp file just before the rename in <see cref="Save"/>.</summary>
    internal Action<string>? BeforeRename { get; set; }

    /// <summary>
    /// Loads the ledger of <paramref name="storeRoot"/>. A missing file (or store directory) is an empty ledger; nothing is
    /// created. Rows whose entry file (<c>&lt;root&gt;/&lt;xx&gt;/&lt;key&gt;.ckv</c>) is reported not found (file or shard
    /// directory) are dropped (<see cref="DroppedOnLoad"/>); rows whose entry state cannot be determined are kept
    /// (<see cref="UndeterminedOnLoad"/>).
    /// </summary>
    /// <param name="storeRoot">Store directory.</param>
    /// <returns>The ledger.</returns>
    /// <exception cref="LedgerUnreadableException">The file exists but cannot be read (see the class remarks).</exception>
    public static ServerLedger Load(string storeRoot)
    {
        string path = DefaultPath(storeRoot);
        var ledger = new ServerLedger(System.IO.Path.GetDirectoryName(path)!);
        foreach (LedgerRow row in ReadRows(path))
        {
            switch (EntryState(FileChunkStore.EntryPath(ledger._root, row.KeyHex)))
            {
                case false:
                    ledger.DroppedOnLoad++;
                    continue;
                case null:
                    ledger.UndeterminedOnLoad++;
                    break;
            }
            ledger._rows.Add(row.KeyHex, row);
        }
        return ledger;
    }

    /// <summary>The keys the ledger of <paramref name="storeRoot"/> protects (no entry-file check); empty when there is no ledger.</summary>
    /// <exception cref="LedgerUnreadableException">The file exists but cannot be read.</exception>
    internal static IEnumerable<ChunkKey> ReadKeys(string storeRoot) => ReadRows(DefaultPath(storeRoot)).Select(r => new ChunkKey(r.KeyHex));

    /// <summary>True: the entry file exists; false: it (or its shard directory) is not found; null: cannot be determined.</summary>
    private static bool? EntryState(string entryPath)
    {
        try
        {
            return (Probe(entryPath) & FileAttributes.Directory) == 0 ? true : null;
        }
        catch (Exception ex) when (ex is FileNotFoundException or DirectoryNotFoundException)
        {
            return false;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    private static List<LedgerRow> ReadRows(string path)
    {
        var rows = new List<LedgerRow>();
        // Only "not found" means no ledger; any other probe failure, or a directory at the path, is unreadable (fail closed).
        FileAttributes attributes;
        try
        {
            attributes = Probe(path);
        }
        catch (Exception ex) when (ex is FileNotFoundException or DirectoryNotFoundException)
        {
            return rows;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            throw new LedgerUnreadableException(path, "its state cannot be read: " + ex.Message, ex);
        }
        if ((attributes & FileAttributes.Directory) != 0) throw new LedgerUnreadableException(path, "it is a directory");
        LedgerDocument? doc;
        try
        {
            using FileStream stream = ChunkIndex.OpenForRead(path);
            doc = JsonSerializer.Deserialize(stream, ServerLedgerJsonContext.Default.LedgerDocument);
        }
        catch (Exception ex) when (ex is FileNotFoundException or DirectoryNotFoundException)
        {
            return rows; // removed between the check and the open
        }
        catch (Exception ex) when (ex is JsonException or IOException or UnauthorizedAccessException)
        {
            throw new LedgerUnreadableException(path, ex is JsonException ? "not valid ledger JSON: " + ex.Message : ex.Message, ex);
        }
        if (doc is null) throw new LedgerUnreadableException(path, "no ledger document");
        if (doc.FormatVersion != FormatVersion)
            throw new LedgerUnreadableException(path, doc.FormatVersion is null
                ? "no format version"
                : $"format version {doc.FormatVersion} is not supported; this build reads version {FormatVersion}");
        if (doc.Rows is null) throw new LedgerUnreadableException(path, "no row list");
        var seen = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < doc.Rows.Count; i++)
        {
            RowDocument? r = doc.Rows[i];
            string? problem = r is null ? "is null"
                : !ChunkKey.IsWellFormed(r.Key) ? "has a malformed key"
                : r.Namespace is null ? "has no namespace"
                : !ChunkNamespace.IsValid(r.Namespace) ? "has an invalid namespace"
                : r.Bytes is not >= 0 ? "has no or a negative byte count"
                : r.FirstSeen is null || r.LastUsed is null ? "has no first-seen or last-used time"
                : r.SourcePath is not null && !IsCanonicalPath(r.SourcePath) ? "has a source path that is not tree-relative and '/'-separated"
                : !seen.Add(r.Key!) ? "repeats an earlier key"
                : null;
            if (problem is not null) throw new LedgerUnreadableException(path, $"row {i} {problem}");
            rows.Add(new LedgerRow(r!.Key!, r.Namespace!, r.SourcePath, r.Bytes!.Value, r.FirstSeen!.Value, r.LastUsed!.Value));
        }
        return rows;
    }

    private static bool IsCanonicalPath(string path)
    {
        try { return ChunkIndex.NormalizePath(path) == path; }
        catch (ArgumentException) { return false; }
    }

    /// <summary>All rows, by namespace then key (ordinal).</summary>
    public IReadOnlyCollection<LedgerRow> Rows => Sorted().ToList();

    private IEnumerable<LedgerRow> Sorted() =>
        _rows.Values.OrderBy(r => r.Namespace, StringComparer.Ordinal).ThenBy(r => r.KeyHex, StringComparer.Ordinal);

    /// <summary>
    /// Records that the server wrote (or wrote again) the entry <paramref name="key"/>. A new row starts with
    /// <paramref name="now"/> as first and last use. An existing row keeps its first-seen time and provenance (a null
    /// provenance takes the first non-null one given), takes <paramref name="bytes"/>, and moves its last use to
    /// <paramref name="now"/> if that is later.
    /// </summary>
    /// <param name="key">The entry's key.</param>
    /// <param name="ns">The namespace the key was made in (<see cref="ChunkNamespace.Validate"/>).</param>
    /// <param name="sourcePath">Tree-relative path of the chunk's source, or null; normalised with <see cref="ChunkIndex.NormalizePath"/>.</param>
    /// <param name="bytes">Entry file size.</param>
    /// <param name="now">The time of the write.</param>
    /// <exception cref="ArgumentException">Malformed key, invalid namespace, or a rooted / <c>..</c> source path.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="bytes"/> is negative.</exception>
    /// <exception cref="InvalidOperationException">The key is already recorded under another namespace.</exception>
    public void Record(ChunkKey key, string ns, string? sourcePath, long bytes, DateTimeOffset now)
    {
        if (!ChunkKey.IsWellFormed(key.Hex)) throw new ArgumentException("Malformed chunk key.", nameof(key));
        ChunkNamespace.Validate(ns, nameof(ns));
        ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        string? source = sourcePath is null ? null : ChunkIndex.NormalizePath(sourcePath);
        if (_rows.TryGetValue(key.Hex, out LedgerRow? old))
        {
            if (!string.Equals(old.Namespace, ns, StringComparison.Ordinal))
                throw new InvalidOperationException("The key is already recorded under another namespace; a key belongs to the namespace it was made in.");
            _rows[key.Hex] = old with { SourcePath = old.SourcePath ?? source, Bytes = bytes, LastUsed = Max(old.LastUsed, now) };
        }
        else
        {
            _rows.Add(key.Hex, new LedgerRow(key.Hex, ns, source, bytes, now, now));
        }
    }

    /// <summary>Moves the last use of <paramref name="key"/>'s row to <paramref name="now"/> if that is later (a hit).</summary>
    /// <param name="key">The key.</param>
    /// <param name="now">The time of the use.</param>
    /// <returns>Whether the key has a row (no row is created).</returns>
    public bool Touch(ChunkKey key, DateTimeOffset now)
    {
        if (key.Hex is null || !_rows.TryGetValue(key.Hex, out LedgerRow? old)) return false;
        _rows[key.Hex] = old with { LastUsed = Max(old.LastUsed, now) };
        return true;
    }

    /// <summary>
    /// Drops <paramref name="key"/>'s row (forget, budget eviction). After the next <see cref="Save"/> the entry is no longer
    /// protected by the ledger, but it may still be in use: keys are content-addressed, so a server entry (notably in the
    /// default namespace) can be the very key a CLI index references. Delete the entry file only if
    /// <see cref="ChunkIndex.ReferencedKeysUnder"/> (computed after this save, under the store lock) no longer contains it;
    /// otherwise drop the row and keep the file. For the same reason a row's <see cref="LedgerRow.Bytes"/> can count an entry
    /// the server does not own alone.
    /// </summary>
    /// <param name="key">The key.</param>
    /// <returns>Whether a row was removed.</returns>
    public bool Remove(ChunkKey key) => key.Hex is not null && _rows.Remove(key.Hex);

    /// <summary>Rows by ascending last use, then key (ordinal): the budget eviction order.</summary>
    /// <returns>The rows.</returns>
    public IEnumerable<LedgerRow> LeastRecentlyUsed() =>
        _rows.Values.OrderBy(r => r.LastUsed).ThenBy(r => r.KeyHex, StringComparer.Ordinal).ToList();

    /// <summary>
    /// Rows whose first-seen provenance matches <paramref name="glob"/> (<see cref="PathGlob"/> semantics: case-insensitive
    /// on Windows, ordinal elsewhere, as <see cref="ChunkIndex.PathComparer"/>), across namespaces; rows without provenance
    /// never match. Intersect with <see cref="InNamespace"/> to scope it to one caller.
    /// </summary>
    /// <param name="glob">A path glob; a negated (<c>!</c>) glob selects the rows with provenance that do not match.</param>
    /// <returns>The rows, by namespace then key.</returns>
    /// <exception cref="ArgumentException">The glob is empty, rooted or has a <c>..</c> segment.</exception>
    public IEnumerable<LedgerRow> UnderPath(string glob)
    {
        PathGlob g = PathGlob.Parse(glob);
        return Sorted().Where(r => r.SourcePath is not null && g.IsMatch(r.SourcePath) != g.Negated).ToList();
    }

    /// <summary>The rows of one namespace, by key.</summary>
    /// <param name="ns">The namespace (<see cref="ChunkNamespace.Validate"/>).</param>
    /// <returns>The rows.</returns>
    public IEnumerable<LedgerRow> InNamespace(string ns)
    {
        ChunkNamespace.Validate(ns, nameof(ns));
        return Sorted().Where(r => string.Equals(r.Namespace, ns, StringComparison.Ordinal)).ToList();
    }

    /// <summary>
    /// Writes the ledger atomically (temp file, flush, one rename; see the class remarks), retrying the rename briefly while a
    /// reader holds the file. Call it under the store lock (<see cref="ChunkIndex.AcquireStoreLock"/>), in the same lock scope as
    /// the puts it records (one short scope per batch; see the class remarks), and pass that lock as proof.
    /// </summary>
    /// <param name="storeLock">The held store lock of this ledger's store root.</param>
    /// <exception cref="InvalidOperationException">The lock was released or is another store's.</exception>
    public void Save(StoreLockHandle storeLock)
    {
        ArgumentNullException.ThrowIfNull(storeLock);
        storeLock.EnsureHeldFor(_root);
        var doc = new LedgerDocument
        {
            FormatVersion = FormatVersion,
            Rows = Sorted().Select(r => (RowDocument?)new RowDocument
            {
                Key = r.KeyHex, Namespace = r.Namespace, SourcePath = r.SourcePath, Bytes = r.Bytes, FirstSeen = r.FirstSeen, LastUsed = r.LastUsed,
            }).ToList(),
        };
        string full = System.IO.Path.Combine(_root, FileName);
        StorePaths.CreateStoreDirectory(_root);
        ChunkIndex.SweepStaleTemps(_root, FileName);
        string temp = full + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            using (var stream = new FileStream(temp, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                JsonSerializer.Serialize(stream, doc, ServerLedgerJsonContext.Default.LedgerDocument);
                stream.Flush(flushToDisk: true);
            }
            BeforeRename?.Invoke(temp);
            ChunkIndex.MoveWithRetry(temp, full);
        }
        finally
        {
            // Best effort: a failed cleanup must not replace the save's own exception (the sweep removes the temp later).
            try { if (File.Exists(temp)) File.Delete(temp); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
        }
    }

    private static DateTimeOffset Max(DateTimeOffset a, DateTimeOffset b) => b > a ? b : a;

    internal sealed class LedgerDocument
    {
        public int? FormatVersion { get; set; }
        public List<RowDocument?>? Rows { get; set; }
    }

    internal sealed class RowDocument
    {
        public string? Key { get; set; }
        public string? Namespace { get; set; }
        public string? SourcePath { get; set; }
        public long? Bytes { get; set; }
        public DateTimeOffset? FirstSeen { get; set; }
        public DateTimeOffset? LastUsed { get; set; }
    }
}

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase, WriteIndented = true, AllowDuplicateProperties = false)]
[JsonSerializable(typeof(ServerLedger.LedgerDocument))]
internal sealed partial class ServerLedgerJsonContext : JsonSerializerContext;
