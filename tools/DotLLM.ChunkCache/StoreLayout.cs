using DotLLM.ChunkReuse.Store;

namespace DotLLM.ChunkCache;

/// <summary>One index the CLI keeps in a store (a row of the store registry, <see cref="StoreLayout.ConfigFileName"/>).</summary>
internal sealed record IndexRecord
{
    /// <summary>Content-based tokenizer id.</summary>
    public string? TokenizerId { get; init; }

    /// <summary>Backend tag of the index's keys.</summary>
    public string? Backend { get; init; }

    /// <summary>Index file name in the store root.</summary>
    public string IndexFile { get; init; } = "";

    /// <summary>Chunker name (<c>symbol-512</c>).</summary>
    public string Chunker { get; init; } = "";

    /// <summary>Producer identity of the index (<see cref="ModelHandle.Producer"/>).</summary>
    public string Producer { get; init; } = "";

    /// <summary>Model fingerprint the keys carry.</summary>
    public string ModelFingerprint { get; init; } = "";

    /// <summary>Content-based model id.</summary>
    public string ModelId { get; init; } = "";

    /// <summary>The GGUF path last used (informational).</summary>
    public string ModelPath { get; init; } = "";

    /// <summary>Source kind: <c>solution</c> or <c>directory</c>.</summary>
    public string SourceKind { get; init; } = "";

    /// <summary>Absolute solution/project file or directory.</summary>
    public string SourcePath { get; init; } = "";

    /// <summary>MSBuild configuration (solutions).</summary>
    public string? Configuration { get; init; }

    /// <summary>Absolute root the index paths are relative to (<see cref="ChunkIndex.RelativizeScan"/>).</summary>
    public string Root { get; init; } = "";

    /// <summary>Last build, UTC.</summary>
    public DateTime UpdatedUtc { get; init; }

    /// <summary>The source spec of this record.</summary>
    [System.Text.Json.Serialization.JsonIgnore]
    public SourceSpec Source => new(SourceKind, SourcePath, Configuration);
}


/// <summary>The CLI's view of a store directory: the entries (<see cref="FileChunkStore"/>), one index file per chunker, model
/// and namespace, and the store registry (<see cref="StoreRegistry"/>), whose rows carry the CLI's tree root and source.</summary>
/// <remarks>
/// <para>
/// <b>Index files</b> (integration spec L9): <c>index-&lt;chunker&gt;.&lt;model12&gt;.&lt;p8&gt;.default.json</c>
/// (<see cref="ChunkIndex.DefaultPath(string, string, string, string, string)"/>; the CLI writes the default namespace only), so two
/// models (or producers) sharing a store keep separate indexes and eviction (which unions every <c>index-*.json</c>) protects both. Indexes
/// written under older names (<c>index-&lt;chunker&gt;.&lt;producer12&gt;.json</c>, the Task 4b name without <c>p8</c>) are renamed by the next build of the
/// same producer (<see cref="ChunkIndex.MigrateLegacyIndex"/>) and read in place by read-only verbs until then.
/// </para>
/// <para>
/// <b>Registry.</b> <see cref="ConfigFileName"/> is the library's <see cref="StoreRegistry"/> (advisory; written only under the
/// store lock). The CLI keeps its own fields (tree root, source, model path and fingerprint) in each row's <c>tool</c> object.
/// </para>
/// </remarks>
internal sealed class StoreLayout
{
    /// <summary>The registry file in the store root.</summary>
    public const string ConfigFileName = StoreRegistry.FileName;

    private StoreLayout(string root, StoreRegistry registry, FileChunkStore? files)
    {
        Root = root;
        Registry = registry;
        Files = files;
    }

    /// <summary>Store directory (absolute).</summary>
    public string Root { get; }

    /// <summary>The store registry as loaded.</summary>
    public StoreRegistry Registry { get; }

    /// <summary>The entry store, or null when the directory does not exist (opened without creating it).</summary>
    public FileChunkStore? Files { get; }

    /// <summary>The entries: <see cref="Files"/>, or an empty store when the directory does not exist.</summary>
    public IChunkStore Entries => (IChunkStore?)Files ?? EmptyChunkStore.Instance;

    /// <summary>Entry statistics (zero when the directory does not exist).</summary>
    public StoreStats Stats => Files?.GetStats() ?? default;

    /// <summary>The entry store of a layout whose directory exists.</summary>
    public FileChunkStore Existing => Files ?? throw new DirectoryNotFoundException($"Store not found: {Root}");

    /// <summary>The indexes the registry lists, as CLI records.</summary>
    public IReadOnlyList<IndexRecord> Records => Registry.Entries.Select(ToRecord).ToList();

    /// <summary>
    /// Opens the store at <paramref name="root"/>. With <paramref name="create"/> the directory is created if needed (writers);
    /// without it a missing directory opens as an empty store and nothing is created (read-only verbs, dry runs).
    /// </summary>
    public static StoreLayout Open(string root, bool create)
    {
        string full = Path.GetFullPath(root);
        if (create) Directory.CreateDirectory(full);
        else if (!Directory.Exists(full)) return new StoreLayout(full, StoreRegistry.Load(full), null);
        return new StoreLayout(full, StoreRegistry.Load(full), new FileChunkStore(full));
    }

    /// <summary>The (current-form) index file of <paramref name="chunker"/> for the model <paramref name="model"/>.</summary>
    public string IndexPath(string chunker, ModelHandle model) => ChunkIndex.DefaultPath(Root, chunker, model.ModelId, model.Producer, ChunkNamespace.Default);

    /// <summary>The index file a read-only verb opens: the current one, or a not-yet-migrated legacy one of this producer.</summary>
    public string ReadPath(string chunker, ModelHandle model) => ChunkIndex.ResolveIndexPath(Root, chunker, model.ModelId, ChunkNamespace.Default, model.KeyOf);

    /// <summary>The record of <paramref name="chunker"/> for <paramref name="model"/> (current or legacy file name), or null.</summary>
    public IndexRecord? Find(string chunker, ModelHandle model)
    {
        string file = Path.GetFileName(IndexPath(chunker, model));
        if (Registry.Get(file) is RegistryEntry e) return ToRecord(e);
        string legacy = Path.GetFileName(ReadPath(chunker, model));
        return Registry.Get(legacy) is RegistryEntry l ? ToRecord(l) : null;
    }

    /// <summary>Adds or replaces the record with the same index file.</summary>
    public void Upsert(IndexRecord record) => Registry.Upsert(ToEntry(record));

    /// <summary>Writes the registry under the held store lock <paramref name="storeLock"/>.</summary>
    public void Save(StoreLockHandle storeLock) => Registry.Save(storeLock);

    /// <summary>
    /// Whether <paramref name="name"/> (from <see cref="ConfigFileName"/>) is a plain index file name of this store: no
    /// directory part, not rooted, no <c>..</c>, and of the form <c>index-*.json</c>. Names that fail are never opened or deleted.
    /// </summary>
    public bool IsIndexFileName(string? name)
    {
        if (!StoreRegistry.IsPlainIndexFileName(name)) return false;
        string full = Path.GetFullPath(Path.Combine(Root, name!));
        return string.Equals(Path.GetDirectoryName(full), Path.TrimEndingDirectorySeparator(Root), StringComparison.OrdinalIgnoreCase);
    }

    private static IndexRecord ToRecord(RegistryEntry e)
    {
        string? Tool(string k) => e.Tool is not null && e.Tool.TryGetValue(k, out string? v) ? v : null;
        return new IndexRecord
        {
            IndexFile = e.File, Chunker = e.Chunker, Producer = e.Producer ?? "", ModelId = e.ModelId ?? "", TokenizerId = e.TokenizerId, Backend = e.Backend,
            ModelFingerprint = Tool("model_fingerprint") ?? "", ModelPath = Tool("model_path") ?? "", SourceKind = Tool("source_kind") ?? "",
            SourcePath = Tool("source_path") ?? "", Configuration = Tool("configuration"), Root = Tool("root") ?? "",
            UpdatedUtc = e.Updated?.UtcDateTime ?? default,
        };
    }

    private static RegistryEntry ToEntry(IndexRecord r)
    {
        RegistryEntry fresh = StoreRegistry.EntryFor(r.Chunker, r.ModelId, r.Producer, ChunkNamespace.Default);
        if (!string.Equals(fresh.File, r.IndexFile, StringComparison.Ordinal))
            throw new InvalidOperationException($"Index file '{r.IndexFile}' is not the {r.Chunker} index of model {ModelHandle.Short(r.ModelId)}.");
        return fresh with
        {
            TokenizerId = r.TokenizerId, Backend = r.Backend,
            Updated = r.UpdatedUtc == default ? null : new DateTimeOffset(DateTime.SpecifyKind(r.UpdatedUtc, DateTimeKind.Utc)),
            Tool = new Dictionary<string, string?>(StringComparer.Ordinal)
            {
                ["model_fingerprint"] = r.ModelFingerprint, ["model_path"] = r.ModelPath, ["source_kind"] = r.SourceKind,
                ["source_path"] = r.SourcePath, ["configuration"] = r.Configuration, ["root"] = r.Root,
            },
        };
    }
}

/// <summary>A store with no entries (a store directory that does not exist, opened read-only).</summary>
internal sealed class EmptyChunkStore : IChunkStore
{
    public static readonly EmptyChunkStore Instance = new();

    public bool TryGet(ChunkKey key, out ChunkReuse.ChunkKvEntry entry)
    {
        entry = null!;
        return false;
    }

    public void Put(ChunkKey key, ChunkReuse.ChunkKvEntry entry) => throw new InvalidOperationException("The store does not exist (opened read-only).");

    public bool Contains(ChunkKey key) => false;

    public IEnumerable<ChunkKey> Keys() => [];

    public long TotalBytes => 0;

    public void Evict(ChunkKey key)
    {
    }
}
