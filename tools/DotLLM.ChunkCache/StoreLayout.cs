using System.Text.Json;
using DotLLM.ChunkReuse.Store;

namespace DotLLM.ChunkCache;

/// <summary>One index the CLI keeps in a store (a row of <see cref="StoreLayout.ConfigFileName"/>).</summary>
internal sealed record IndexRecord
{
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

/// <summary>The CLI's view of a store directory: the entries (<see cref="FileChunkStore"/>), one index file per chunker and
/// model, and a small config file that records each index's tree root and source.</summary>
/// <remarks>
/// <para>
/// <b>Index files.</b> <c>index-&lt;chunker&gt;.&lt;producer12&gt;.json</c>: one per chunker <i>and</i> producer (model,
/// tokenizer, RoPE, backend, dtype), so two models sharing a store keep separate indexes, and eviction (which unions every
/// <c>index-*.json</c>) protects both. A plain <c>index-&lt;chunker&gt;.json</c> would be overwritten by the second model's
/// build, orphaning the first model's entries.
/// </para>
/// <para>
/// <b>Config.</b> <see cref="ConfigFileName"/> is not named <c>index-*</c>, so <see cref="ChunkIndex.ReferencedKeysUnder"/>
/// ignores it. It is written (temp file + rename) only under the store lock.
/// </para>
/// </remarks>
internal sealed class StoreLayout
{
    /// <summary>The config file in the store root.</summary>
    public const string ConfigFileName = "chunkcache.json";

    /// <summary>Config format version.</summary>
    public const int ConfigVersion = 1;

    private StoreLayout(string root, List<IndexRecord> records, FileChunkStore? files)
    {
        Root = root;
        Records = records;
        Files = files;
    }

    /// <summary>Store directory (absolute).</summary>
    public string Root { get; }

    /// <summary>The entry store, or null when the directory does not exist (opened without creating it).</summary>
    public FileChunkStore? Files { get; }

    /// <summary>The entries: <see cref="Files"/>, or an empty store when the directory does not exist.</summary>
    public IChunkStore Entries => (IChunkStore?)Files ?? EmptyChunkStore.Instance;

    /// <summary>Entry statistics (zero when the directory does not exist).</summary>
    public StoreStats Stats => Files?.GetStats() ?? default;

    /// <summary>The entry store of a layout whose directory exists.</summary>
    public FileChunkStore Existing => Files ?? throw new DirectoryNotFoundException($"Store not found: {Root}");

    /// <summary>The indexes recorded in the config.</summary>
    public List<IndexRecord> Records { get; }

    /// <summary>
    /// Opens the store at <paramref name="root"/>. With <paramref name="create"/> the directory is created if needed (writers);
    /// without it a missing directory opens as an empty store and nothing is created (read-only verbs, dry runs).
    /// </summary>
    public static StoreLayout Open(string root, bool create)
    {
        if (create) Directory.CreateDirectory(root);
        else if (!Directory.Exists(root)) return new StoreLayout(Path.GetFullPath(root), [], null);
        string path = Path.Combine(root, ConfigFileName);
        var records = new List<IndexRecord>();
        if (File.Exists(path))
        {
            ConfigDocument? doc;
            try
            {
                using FileStream s = ChunkIndex.OpenForRead(path);
                doc = JsonSerializer.Deserialize<ConfigDocument>(s, CliContext.JsonOptions);
            }
            catch (JsonException ex)
            {
                throw new InvalidDataException($"Store config '{path}' is not valid JSON ({ex.Message}). Repair or remove it.", ex);
            }
            if (doc is null || doc.FormatVersion != ConfigVersion)
                throw new InvalidDataException($"Store config '{path}' has format version {doc?.FormatVersion}, not {ConfigVersion}.");
            records.AddRange(doc.Indexes ?? []);
        }
        return new StoreLayout(Path.GetFullPath(root), records, new FileChunkStore(root));
    }

    /// <summary>Index file of <paramref name="chunker"/> for <paramref name="producer"/>.</summary>
    public string IndexPath(string chunker, string producer) => ChunkIndex.DefaultPath(Root, $"{chunker}.{producer[..12]}");

    /// <summary>The record of <paramref name="chunker"/> for <paramref name="producer"/>, or null.</summary>
    public IndexRecord? Find(string chunker, string producer) =>
        Records.FirstOrDefault(r => r.Chunker == chunker && r.Producer == producer);

    /// <summary>Adds or replaces the record with the same index file.</summary>
    public void Upsert(IndexRecord record)
    {
        Records.RemoveAll(r => string.Equals(r.IndexFile, record.IndexFile, StringComparison.OrdinalIgnoreCase));
        Records.Add(record);
        Records.Sort((a, b) => string.CompareOrdinal(a.IndexFile, b.IndexFile));
    }

    /// <summary>Writes the config (call under the store lock).</summary>
    public void Save()
    {
        string path = Path.Combine(Root, ConfigFileName);
        string temp = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            File.WriteAllText(temp, JsonSerializer.Serialize(new ConfigDocument { FormatVersion = ConfigVersion, Indexes = Records }, CliContext.JsonOptions));
            for (int attempt = 0; ; attempt++)
            {
                try
                {
                    File.Move(temp, path, overwrite: true);
                    break;
                }
                catch (Exception e) when ((e is IOException or UnauthorizedAccessException) && attempt < 100)
                {
                    Thread.Sleep(20); // a reader holds it briefly (Windows refuses a rename over an open file)
                }
            }
        }
        finally
        {
            if (File.Exists(temp)) File.Delete(temp);
        }
    }

    /// <summary>
    /// Whether <paramref name="name"/> (from <see cref="ConfigFileName"/>) is a plain index file name of this store: no
    /// directory part, not rooted, no <c>..</c>, and of the form <c>index-*.json</c>. Names that fail are never opened or deleted.
    /// </summary>
    public bool IsIndexFileName(string? name)
    {
        if (string.IsNullOrEmpty(name) || name.IndexOfAny(['/', '\\', ':']) >= 0 || name.Contains("..", StringComparison.Ordinal)) return false;
        if (!name.StartsWith("index-", StringComparison.Ordinal) || !name.EndsWith(".json", StringComparison.OrdinalIgnoreCase)) return false;
        if (name.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 || Path.GetFileName(name) != name) return false;
        string full = Path.GetFullPath(Path.Combine(Root, name));
        return string.Equals(Path.GetDirectoryName(full), Path.TrimEndingDirectorySeparator(Root), StringComparison.OrdinalIgnoreCase);
    }

    private sealed class ConfigDocument
    {
        public int FormatVersion { get; set; }

        public List<IndexRecord>? Indexes { get; set; }
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
