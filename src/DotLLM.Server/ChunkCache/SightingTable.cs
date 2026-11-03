namespace DotLLM.Server.ChunkCache;

/// <summary>
/// How many requests each chunk text was seen in, per namespace (spec §5.10: a miss is queued for a background encode only
/// once its text has been seen in at least <see cref="AdmitAfter"/> requests). Keyed by <c>(namespace, text hash)</c>; rows
/// carry the first <c>source.path</c> seen, for forget (Task 16). Bounded: at most <see cref="Capacity"/> rows, the least
/// recently seen dropped first. In memory only (never persisted, never logged: spec §10). Thread-safe.
/// </summary>
internal sealed class SightingTable
{
    /// <summary>Default row cap.</summary>
    public const int DefaultCapacity = 100_000;

    /// <summary>Sightings needed before a miss is admitted.</summary>
    public const int AdmitAfter = 2;

    /// <summary>Longest <c>source.path</c> kept on a row; a longer one is not kept.</summary>
    public const int MaxSourcePathChars = 1024;

    private readonly object _gate = new();
    private readonly Dictionary<(string Ns, string Hash), LinkedListNode<Row>> _rows = [];
    private readonly LinkedList<Row> _lru = new(); // most recently seen first

    /// <summary>Creates a table.</summary>
    /// <param name="capacity">Most rows kept (at least 1).</param>
    public SightingTable(int capacity = DefaultCapacity)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        Capacity = capacity;
    }

    /// <summary>Most rows kept.</summary>
    public int Capacity { get; }

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
        string? path = sourcePath is { Length: <= MaxSourcePathChars } ? sourcePath : null;
        lock (_gate)
        {
            if (_rows.TryGetValue((ns, hash), out var node))
            {
                _lru.Remove(node);
                _lru.AddFirst(node);
                node.Value.Count = node.Value.Count == int.MaxValue ? int.MaxValue : node.Value.Count + 1;
                node.Value.SourcePath ??= path;
                return node.Value.Count;
            }
            if (_rows.Count >= Capacity && _lru.Last is { } oldest)
            {
                _lru.RemoveLast();
                _rows.Remove((oldest.Value.Ns, oldest.Value.Hash));
            }
            var row = new Row(ns, hash) { Count = 1, SourcePath = path };
            _rows[(ns, hash)] = _lru.AddFirst(row);
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

    private sealed class Row(string ns, string hash)
    {
        public string Ns { get; } = ns;
        public string Hash { get; } = hash;
        public int Count { get; set; }
        public string? SourcePath { get; set; }
    }
}
