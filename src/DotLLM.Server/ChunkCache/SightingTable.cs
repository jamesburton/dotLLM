namespace DotLLM.Server.ChunkCache;

/// <summary>
/// How many requests each chunk text was seen in, per namespace (spec §5.10: a miss is queued for a background encode only
/// once its text has been seen in at least <see cref="AdmitAfter"/> requests). Keyed by <c>(namespace, text hash)</c>; rows
/// carry the first <c>source.path</c> seen, for forget (Task 16), capped at <see cref="StoredPathChars"/> (a longer path is
/// kept as a prefix plus a hash). Bounded: at most <see cref="Capacity"/> rows, the least recently seen dropped first, and
/// at most <see cref="PerNamespaceCapacity"/> per namespace (a namespace at its share drops its own least recent row, so
/// one tenant cannot push out another's sightings). Forget (<see cref="Remove"/>) drops rows by namespace and provenance,
/// with no marker. In memory only (never persisted, never logged: spec §10). Thread-safe.
/// </summary>
internal sealed class SightingTable
{
    /// <summary>Default row cap.</summary>
    public const int DefaultCapacity = 100_000;

    /// <summary>Sightings needed before a miss is admitted.</summary>
    public const int AdmitAfter = 2;

    /// <summary>Longest <c>source.path</c> admission accepts at all; a longer one is refused as unreadable.</summary>
    public const int MaxSourcePathChars = 1024;

    private readonly object _gate = new();
    private readonly Dictionary<(string Ns, string Hash), LinkedListNode<Row>> _rows = [];
    private readonly LinkedList<Row> _lru = new(); // most recently seen first
    private readonly Dictionary<string, LinkedList<Row>> _byNamespace = new(StringComparer.Ordinal); // per namespace, same order

    /// <summary>Creates a table.</summary>
    /// <param name="capacity">Most rows kept (at least 1).</param>
    public SightingTable(int capacity = DefaultCapacity) : this(capacity, Math.Max(1, capacity / 4))
    {
    }

    /// <summary>Creates a table with a per-namespace row cap.</summary>
    /// <param name="capacity">Most rows kept.</param>
    /// <param name="perNamespaceCapacity">Most rows one namespace may hold.</param>
    public SightingTable(int capacity, int perNamespaceCapacity)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(perNamespaceCapacity, 1);
        Capacity = capacity;
        PerNamespaceCapacity = Math.Min(perNamespaceCapacity, capacity);
    }

    /// <summary>Most rows kept.</summary>
    public int Capacity { get; }

    /// <summary>Most rows one namespace may hold.</summary>
    public int PerNamespaceCapacity { get; }

    /// <summary>Longest <c>source.path</c> stored verbatim; a longer one is stored as a prefix plus a hash.</summary>
    public const int StoredPathChars = 256;

    /// <summary>Rows held now.</summary>
    public int Count
    {
        get { lock (_gate) return _rows.Count; }
    }

    /// <summary>
    /// Records that one request carried the text with <paramref name="hash"/> in <paramref name="ns"/> (call once per request
    /// and text) and returns how many requests have now carried it (1 for a new or previously evicted row).
    /// </summary>
    /// <param name="ns">The caller's namespace.</param>
    /// <param name="hash">The verified wire hash of the text (<c>sha256:</c> + hex).</param>
    /// <param name="sourcePath">The annotation's <c>source.path</c>, or null.</param>
    /// <returns>The sighting count.</returns>
    public int See(string ns, string hash, string? sourcePath)
    {
        ArgumentNullException.ThrowIfNull(ns);
        ArgumentNullException.ThrowIfNull(hash);
        string? path = StoredPath(sourcePath);
        lock (_gate)
        {
            if (_rows.TryGetValue((ns, hash), out var node))
            {
                var row = node.Value;
                _lru.Remove(node);
                _lru.AddFirst(node);
                var own = _byNamespace[ns];
                own.Remove(row.NamespaceNode!);
                own.AddFirst(row.NamespaceNode!);
                row.Count = row.Count == int.MaxValue ? int.MaxValue : row.Count + 1;
                row.SourcePath ??= path;
                return row.Count;
            }
            if (!_byNamespace.TryGetValue(ns, out var mine))
                _byNamespace[ns] = mine = new LinkedList<Row>();
            if (mine.Count >= PerNamespaceCapacity && mine.Last is { } ownOldest)
                Drop(ownOldest.Value);
            else if (_rows.Count >= Capacity && _lru.Last is { } oldest)
                Drop(oldest.Value);
            var added = new Row(ns, hash) { Count = 1, SourcePath = path };
            added.NamespaceNode = mine.AddFirst(added);
            _rows[(ns, hash)] = _lru.AddFirst(added);
            return 1;
        }
    }

    /// <summary>The sighting count of a text (0 when unknown); does not count as a sighting.</summary>
    /// <param name="ns">Namespace.</param>
    /// <param name="hash">Text hash.</param>
    /// <returns>The count.</returns>
    public int CountOf(string ns, string hash)
    {
        lock (_gate)
            return _rows.TryGetValue((ns, hash), out var node) ? node.Value.Count : 0;
    }

    /// <summary>The first <c>source.path</c> recorded for a text, or null.</summary>
    /// <param name="ns">Namespace.</param>
    /// <param name="hash">Text hash.</param>
    /// <returns>The path or null.</returns>
    public string? SourcePathOf(string ns, string hash)
    {
        lock (_gate)
            return _rows.TryGetValue((ns, hash), out var node) ? node.Value.SourcePath : null;
    }

    /// <summary>
    /// Removes (or, with <paramref name="countOnly"/>, counts) the rows <paramref name="match"/> selects: forget (Task 16).
    /// No marker is kept: a later sighting of the same text starts again at 1.
    /// </summary>
    /// <param name="match">Selects a row by its namespace and stored <c>source.path</c> (null when none was given).</param>
    /// <param name="countOnly">Count without removing (a dry run).</param>
    /// <returns>How many rows matched.</returns>
    public int Remove(Func<string, string?, bool> match, bool countOnly = false)
    {
        ArgumentNullException.ThrowIfNull(match);
        lock (_gate)
        {
            var hits = _lru.Where(r => match(r.Ns, r.SourcePath)).ToList();
            if (!countOnly)
                foreach (var row in hits)
                    Drop(row);
            return hits.Count;
        }
    }

    private void Drop(Row row)
    {
        if (!_rows.Remove((row.Ns, row.Hash), out var node))
            return;
        _lru.Remove(node);
        var own = _byNamespace[row.Ns];
        own.Remove(row.NamespaceNode!);
        if (own.Count == 0)
            _byNamespace.Remove(row.Ns);
    }

    /// <summary>The path as stored: verbatim up to <see cref="StoredPathChars"/>, else a prefix, '#', and 16 hex digits of its SHA-256.</summary>
    internal static string? StoredPath(string? path)
    {
        if (path is null || path.Length <= StoredPathChars)
            return path;
        string digest = Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(path)))[..16];
        int cut = StoredPathChars - 17;
        if (char.IsHighSurrogate(path[cut - 1]))
            cut--; // never keep half a surrogate pair
        return path[..cut] + "#" + digest;
    }

    private sealed class Row(string ns, string hash)
    {
        public LinkedListNode<Row>? NamespaceNode { get; set; }
        public string Ns { get; } = ns;
        public string Hash { get; } = hash;
        public int Count { get; set; }
        public string? SourcePath { get; set; }
    }
}
