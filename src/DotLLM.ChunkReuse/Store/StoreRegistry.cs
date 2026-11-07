using System.Text.Json;
using System.Text.Json.Serialization;

namespace DotLLM.ChunkReuse.Store;

/// <summary>One index of a store as listed in its registry (<see cref="StoreRegistry"/>).</summary>
/// <param name="File">Index file name in the store root (no directory), e.g. <c>index-symbol-512.0123456789ab.default.json</c>.</param>
/// <param name="Chunker">Chunker name.</param>
/// <param name="ModelTag">The file name's model component (<see cref="ChunkIndex.ModelTag"/>), or null for a legacy name whose
/// model is unknown.</param>
/// <param name="Namespace">Namespace (<see cref="ChunkNamespace.Default"/> is the empty string).</param>
public sealed record RegistryEntry(string File, string Chunker, string? ModelTag, string Namespace)
{
    /// <summary>Content model id (<c>GgufIdentity.ModelId</c> form), or null when not known (a rebuilt row).</summary>
    public string? ModelId { get; init; }

    /// <summary>Content tokenizer id, or null when not known.</summary>
    public string? TokenizerId { get; init; }

    /// <summary>Backend tag the index's keys were made for (<c>cpu</c>, <c>vulkan</c>), or null when not known.</summary>
    public string? Backend { get; init; }

    /// <summary>Producer identity of the index (<c>keyOf([]).Hex</c>), or null when not known.</summary>
    public string? Producer { get; init; }

    /// <summary>When the index was last saved by its writer, or null when not known.</summary>
    public DateTimeOffset? Updated { get; init; }

    /// <summary>
    /// Tool-specific string fields (e.g. the CLI's tree root and source), kept verbatim by every writer of the registry; null
    /// when there are none.
    /// </summary>
    public IReadOnlyDictionary<string, string?>? Tool { get; init; }

    /// <summary>The file name's producer component (<see cref="ChunkIndex.ProducerTag"/>), or null for an older name.</summary>
    public string? ProducerTag { get; init; }

    /// <summary>Whether the file name is of the current form (<see cref="ChunkIndex.TryParseIndexFileName"/>), not a legacy name.</summary>
    public bool IsCurrentName => ChunkIndex.TryParseIndexFileName(File, out _);

    /// <summary>Properties of the row this build does not know (a later writer of the same format), kept on save.</summary>
    internal Dictionary<string, JsonElement>? Extra { get; init; }
}

/// <summary>
/// The store registry <c>&lt;storeRoot&gt;/chunkcache.json</c> (integration spec L9): one row per index file of the store
/// (<see cref="RegistryEntry"/>: file, chunker, model, tokenizer, backend, namespace, updated, plus tool-specific fields).
/// </summary>
/// <remarks>
/// <para>
/// <b>Advisory, never authoritative.</b> Eviction (<see cref="ChunkIndex.ReferencedKeysUnder"/>,
/// <see cref="ChunkIndex.StoreWideOrphans"/>) reads every <c>index-*.json</c> in the store root whether or not the registry
/// lists it; the registry only answers "which index is for this chunker, model and namespace, and what else do we know about
/// it". <see cref="Load"/> therefore never fails on content: a missing, unreadable or newer-format registry is rebuilt from the
/// index file names on disk (chunker, model tag, producer tag and namespace) and the producer each index file records (the
/// other fields unknown), rows whose file is gone are dropped,
/// and index files the registry does not list are added. The name is not <c>index-*</c>, so it is never read as an index.
/// </para>
/// <para>
/// <b>Writes</b> (<see cref="Save"/>) happen only under the store lock (<see cref="ChunkIndex.AcquireStoreLock"/>, one short
/// scope; enforced for the current flow) as load, change, save (<see cref="Update"/>), so concurrent writers never lose each
/// other's rows. The file is replaced atomically (temp file, flush, one rename). An unreadable registry is copied to
/// <c>chunkcache.json.bad-&lt;utc time&gt;</c> before it is replaced (earlier copies kept); a registry of a newer format
/// version, or one that could not be read even after a short retry, is never overwritten by that instance
/// (<see cref="Save"/> throws; the next <see cref="Load"/> reads it again).
/// </para>
/// <para>
/// <b>Format</b> (snake_case JSON): version 2, <c>{"format_version": 2, "indexes": [{"file", "chunker", "model12", "producer8", "model_id",
/// "tokenizer_id", "backend", "namespace", "producer", "updated", "tool": {...}}]}</c>; unknown row properties are kept.
/// Version 1 (written by <c>chunkcache</c> before Phase I Task 4b: <c>index_file</c>, <c>producer</c>, <c>model_id</c>,
/// <c>updated_utc</c> and the CLI's fields) is read and rewritten as version 2, the CLI's fields moving into <c>tool</c>.
/// </para>
/// </remarks>
public sealed class StoreRegistry
{
    /// <summary>The registry file name in the store root.</summary>
    public const string FileName = "chunkcache.json";

    /// <summary>The format version this build writes.</summary>
    public const int FormatVersion = 2;

    /// <summary>
    /// Where an unreadable registry is copied before it is replaced: <c>chunkcache.json.bad-&lt;yyyyMMddTHHmmssfffZ&gt;</c>, one
    /// file per replacement (earlier copies are kept).
    /// </summary>
    public const string BadCopyPrefix = ".bad-";

    /// <summary>
    /// The first free path for a bad copy of <paramref name="registryPath"/>: the UTC-stamped name, then the same name with <c>~1</c>,
    /// <c>~2</c>, ... appended. The counter is appended to the stamped name, never cut out of the full path (a directory can
    /// contain '~', e.g. a Windows 8.3 short name such as <c>PROGRA~1</c>).
    /// </summary>
    internal static string BadCopyPath(string registryPath, DateTime utc, Func<string, bool> exists)
    {
        string stamped = registryPath + BadCopyPrefix + utc.ToString("yyyyMMdd'T'HHmmssfff'Z'", System.Globalization.CultureInfo.InvariantCulture);
        string bad = stamped;
        for (int n = 1; exists(bad); n++) bad = stamped + "~" + n;
        return bad;
    }

    private static readonly StringComparer FileComparer = ChunkIndex.PathComparer;
    private static readonly string[] V1ToolFields = ["model_fingerprint", "model_path", "source_kind", "source_path", "configuration", "root"];

    private readonly string _root;
    private readonly SortedDictionary<string, RegistryEntry> _rows = new(FileComparer);
    private byte[]? _unreadableBytes;
    private int? _newerVersion;

    private StoreRegistry(string root) => _root = root;

    /// <summary>The registry file of <paramref name="storeRoot"/>.</summary>
    /// <param name="storeRoot">Store directory.</param>
    /// <returns>The path.</returns>
    public static string DefaultPath(string storeRoot)
    {
        ArgumentException.ThrowIfNullOrEmpty(storeRoot);
        return Path.Combine(Path.GetFullPath(storeRoot), FileName);
    }

    /// <summary>The rows, by file name.</summary>
    public IReadOnlyList<RegistryEntry> Entries => _rows.Values.ToList();

    /// <summary>Why the registry was rebuilt or rows were dropped or added on <see cref="Load"/>, or null when it loaded as written.</summary>
    public string? LoadNote { get; private set; }

    /// <summary>Whether <see cref="Load"/> rebuilt the registry from the index files (missing, unreadable or newer format).</summary>
    public bool Rebuilt { get; private set; }

    /// <summary>
    /// Rows <see cref="Load"/> refused because their file name is not a plain <c>index-*.json</c> name in the store root (or
    /// they were malformed), as written. A non-empty list means someone edited the registry; destructive tools should refuse.
    /// </summary>
    public IReadOnlyList<string> InvalidRows { get; private set; } = [];

    /// <summary>Test hook: called with the flushed temp file just before the rename in <see cref="Save"/>.</summary>
    internal Action<string>? BeforeRename { get; set; }

    /// <summary>
    /// Loads the registry of <paramref name="storeRoot"/> and reconciles it with the index files on disk (see the class
    /// remarks). Never throws for the registry's content; nothing is written.
    /// </summary>
    /// <param name="storeRoot">Store directory (need not exist).</param>
    /// <returns>The registry.</returns>
    public static StoreRegistry Load(string storeRoot)
    {
        string path = DefaultPath(storeRoot);
        var registry = new StoreRegistry(Path.GetDirectoryName(path)!);
        var notes = new List<string>();
        var invalid = new List<string>();
        RegistryDocument? doc = null;
        bool exists = File.Exists(path);
        if (exists)
        {
            try
            {
                byte[] bytes = ReadWithRetry(path);
                try
                {
                    doc = JsonSerializer.Deserialize(bytes, StoreRegistryJsonContext.Default.RegistryDocument);
                    if (doc?.FormatVersion is not (1 or FormatVersion) || doc.Indexes is null)
                    {
                        if (doc?.FormatVersion is int v && v > FormatVersion) registry._newerVersion = v;
                        else registry._unreadableBytes = bytes;
                        notes.Add(doc?.FormatVersion is int w && w > FormatVersion
                            ? $"the registry has format version {w}, newer than {FormatVersion}; rebuilt from the index files for reading only"
                            : "the registry has no supported format version or no index list; rebuilt from the index files");
                        doc = null;
                    }
                }
                catch (JsonException ex)
                {
                    registry._unreadableBytes = bytes;
                    notes.Add($"the registry is not valid JSON ({ex.Message}); rebuilt from the index files");
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                notes.Add($"the registry cannot be read ({ex.Message}); rebuilt from the index files");
                registry._newerVersion = -1; // unknown content: never overwritten by this instance; the next Load reads again
            }
        }
        else
        {
            notes.Add("no registry; rebuilt from the index files");
        }
        registry.Rebuilt = doc is null;

        if (doc is not null)
        {
            for (int i = 0; i < doc.Indexes!.Count; i++)
            {
                RowDocument? r = doc.Indexes[i];
                RegistryEntry? entry = r is null ? null : FromRow(r, doc.FormatVersion!.Value);
                if (entry is null)
                {
                    invalid.Add($"row {i}: {(r is null ? "null" : "'" + (r.File ?? r.IndexFile) + "'")}");
                    continue;
                }
                if (!registry._rows.TryAdd(entry.File, entry)) invalid.Add($"row {i}: '{entry.File}' repeats an earlier row");
            }
        }

        // Reconcile with the disk: drop rows whose file is gone, add current-form index files the registry does not list.
        int dropped = 0, added = 0;
        foreach (RegistryEntry e in registry._rows.Values.ToList())
        {
            if (File.Exists(Path.Combine(registry._root, e.File))) continue;
            registry._rows.Remove(e.File);
            dropped++;
        }
        if (Directory.Exists(registry._root))
        {
            foreach (string file in Directory.EnumerateFiles(registry._root, "index-*.json", SearchOption.TopDirectoryOnly))
            {
                string name = Path.GetFileName(file);
                if (registry._rows.ContainsKey(name) || !ChunkIndex.TryParseIndexFileName(name, out IndexFileName parts)) continue;
                string? producer = ChunkIndex.ProducerOf(file);
                registry._rows.Add(name, new RegistryEntry(name, parts.Chunker, parts.ModelTag, parts.Namespace)
                {
                    ProducerTag = parts.ProducerTag,
                    Producer = producer is not null && producer.StartsWith(parts.ProducerTag, StringComparison.Ordinal) ? producer : null,
                });
                added++;
            }
        }
        if (dropped > 0) notes.Add($"dropped {dropped} row(s) whose index file is gone");
        if (added > 0 && !registry.Rebuilt) notes.Add($"added {added} index file(s) the registry did not list");
        if (invalid.Count > 0) notes.Add($"refused {invalid.Count} row(s): {string.Join("; ", invalid)}");
        registry.InvalidRows = invalid;
        registry.LoadNote = notes.Count == 0 || (!exists && registry._rows.Count == 0) ? null : string.Join("; ", notes);
        return registry;
    }

    /// <summary>How long <see cref="Load"/> retries a read refused by a sharing violation or access denied (an AV scanner, an
    /// indexer, a writer's rename in progress) before treating the registry as unreadable.</summary>
    internal static TimeSpan ReadRetryBudget
    {
        get => s_readRetryBudget.Value ?? TimeSpan.FromSeconds(2);
        set => s_readRetryBudget.Value = value;
    }

    // Per async flow (a test shortens it for itself only; a process-wide value raced between parallel test classes).
    private static readonly AsyncLocal<TimeSpan?> s_readRetryBudget = new();

    private static byte[] ReadWithRetry(string path)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        while (true)
        {
            try
            {
                return File.ReadAllBytes(path);
            }
            catch (Exception ex) when ((ex is IOException and not FileNotFoundException and not DirectoryNotFoundException) || ex is UnauthorizedAccessException)
            {
                if (sw.Elapsed >= ReadRetryBudget) throw;
                Thread.Sleep(20);
            }
        }
    }

    private static RegistryEntry? FromRow(RowDocument r, int version)
    {
        string? file = version == 1 ? r.IndexFile : r.File;
        if (!IsPlainIndexFileName(file) || string.IsNullOrEmpty(r.Chunker)) return null;
        string ns = r.Namespace ?? ChunkNamespace.Default;
        if (!ChunkNamespace.IsValid(ns)) return null;
        string? modelId = string.IsNullOrEmpty(r.ModelId) ? null : r.ModelId;
        string? tag = r.Model12 ?? (modelId is null ? null : ChunkIndex.ModelTag(modelId));
        string chunker = r.Chunker;
        string? producer = ChunkKey.IsWellFormed(r.Producer) ? r.Producer : null;
        string? p8 = null;
        if (ChunkIndex.TryParseIndexFileName(file, out IndexFileName parts))
        {
            // The file name is authoritative for the parts it encodes.
            chunker = parts.Chunker;
            tag = parts.ModelTag;
            p8 = parts.ProducerTag;
            ns = parts.Namespace;
            if (modelId is not null && ChunkIndex.ModelTag(modelId) != parts.ModelTag) modelId = null;
            if (producer is not null && !producer.StartsWith(p8, StringComparison.Ordinal)) producer = null;
        }
        Dictionary<string, string?>? tool = r.Tool is null ? null : new Dictionary<string, string?>(r.Tool, StringComparer.Ordinal);
        Dictionary<string, JsonElement>? extra = r.Extra is null ? null : new Dictionary<string, JsonElement>(r.Extra, StringComparer.Ordinal);
        DateTimeOffset? updated = r.Updated;
        if (version == 1 && extra is not null)
        {
            // Version 1 (the Phase S CLI's config): its own fields become tool fields; updated_utc becomes updated.
            foreach (string f in V1ToolFields)
            {
                if (!extra.Remove(f, out JsonElement v)) continue;
                tool ??= new Dictionary<string, string?>(StringComparer.Ordinal);
                tool[f] = v.ValueKind == JsonValueKind.String ? v.GetString() : null;
            }
            if (extra.Remove("updated_utc", out JsonElement u) && u.ValueKind == JsonValueKind.String && DateTimeOffset.TryParse(u.GetString(), System.Globalization.CultureInfo.InvariantCulture, System.Globalization.DateTimeStyles.AssumeUniversal, out DateTimeOffset t))
                updated ??= t;
            if (extra.Count == 0) extra = null;
        }
        return new RegistryEntry(file!, chunker, tag, ns)
        {
            ModelId = modelId, TokenizerId = r.TokenizerId, Backend = r.Backend, Producer = producer, ProducerTag = p8, Updated = updated, Tool = tool, Extra = extra,
        };
    }

    /// <summary>
    /// Whether <paramref name="name"/> is a plain index file name of a store root: no directory part, not rooted, no <c>..</c>,
    /// no invalid characters, of the form <c>index-*.json</c>. Names that fail are never opened or deleted.
    /// </summary>
    /// <param name="name">Candidate (from the registry, i.e. data).</param>
    /// <returns>Whether it is plain.</returns>
    public static bool IsPlainIndexFileName(string? name)
    {
        if (string.IsNullOrEmpty(name) || name.IndexOfAny(['/', '\\', ':']) >= 0 || name.Contains("..", StringComparison.Ordinal)) return false;
        if (!name.StartsWith("index-", StringComparison.Ordinal) || !name.EndsWith(".json", StringComparison.Ordinal)) return false;
        return name.IndexOfAny(Path.GetInvalidFileNameChars()) < 0 && Path.GetFileName(name) == name && !Path.IsPathRooted(name);
    }

    /// <summary>
    /// The index file for this chunker, model, producer and namespace
    /// (<see cref="ChunkIndex.DefaultPath(string, string, string, string, string)"/>) when the registry lists it or the file
    /// exists, else null.
    /// </summary>
    /// <param name="chunker">Chunker name.</param>
    /// <param name="modelId">Content model id.</param>
    /// <param name="producer">Producer identity (64 hex; a server-backed client takes it from the server).</param>
    /// <param name="ns">Namespace.</param>
    /// <returns>The path, or null.</returns>
    public string? Find(string chunker, string modelId, string producer, string ns)
    {
        string path = ChunkIndex.DefaultPath(_root, chunker, modelId, producer, ns);
        return _rows.ContainsKey(Path.GetFileName(path)) || File.Exists(path) ? path : null;
    }

    /// <summary>The row of <paramref name="file"/> (a file name), or null.</summary>
    /// <param name="file">Index file name.</param>
    /// <returns>The row.</returns>
    public RegistryEntry? Get(string file) => _rows.GetValueOrDefault(file);

    /// <summary>
    /// A new row for this chunker, model, producer and namespace (file name per
    /// <see cref="ChunkIndex.DefaultPath(string, string, string, string, string)"/>); add the other fields with <c>with</c>.
    /// </summary>
    /// <param name="chunker">Chunker name.</param>
    /// <param name="modelId">Content model id.</param>
    /// <param name="producer">Producer identity (64 hex).</param>
    /// <param name="ns">Namespace.</param>
    /// <returns>The row.</returns>
    public static RegistryEntry EntryFor(string chunker, string modelId, string producer, string ns)
    {
        string file = Path.GetFileName(ChunkIndex.DefaultPath(Path.GetTempPath(), chunker, modelId, producer, ns));
        return new RegistryEntry(file, chunker, ChunkIndex.ModelTag(modelId), ns)
        {
            ModelId = modelId, Producer = producer, ProducerTag = ChunkIndex.ProducerTag(producer),
        };
    }

    /// <summary>Adds or replaces the row of <paramref name="entry"/>'s file.</summary>
    /// <param name="entry">The row.</param>
    /// <exception cref="ArgumentException">The file name is not plain, or (current-form names) its parts disagree with the row.</exception>
    public void Upsert(RegistryEntry entry)
    {
        ArgumentNullException.ThrowIfNull(entry);
        if (!IsPlainIndexFileName(entry.File)) throw new ArgumentException($"'{entry.File}' is not a plain index-*.json file name.", nameof(entry));
        ChunkNamespace.Validate(entry.Namespace, nameof(entry));
        if (ChunkIndex.TryParseIndexFileName(entry.File, out IndexFileName parts)
            && (parts.Chunker != entry.Chunker || parts.ModelTag != entry.ModelTag || parts.Namespace != entry.Namespace
                || parts.ProducerTag != entry.ProducerTag
                || (entry.ModelId is not null && ChunkIndex.ModelTag(entry.ModelId) != parts.ModelTag)
                || (entry.Producer is not null && !entry.Producer.StartsWith(parts.ProducerTag, StringComparison.Ordinal))))
            throw new ArgumentException($"The row for '{entry.File}' disagrees with its file name (chunker, model, producer or namespace).", nameof(entry));
        _rows[entry.File] = entry;
    }

    /// <summary>Removes the row of <paramref name="file"/>.</summary>
    /// <param name="file">Index file name.</param>
    /// <returns>Whether a row was removed.</returns>
    public bool Remove(string file) => _rows.Remove(file);

    /// <summary>Moves the row of a renamed (migrated) index file to its new name, keeping its other fields.</summary>
    internal void Rename(string from, string to, string chunker, string modelId, string producer, string ns)
    {
        RegistryEntry? old = Get(from);
        _rows.Remove(from);
        RegistryEntry fresh = EntryFor(chunker, modelId, producer, ns);
        if (fresh.File != to) throw new InvalidOperationException($"'{to}' is not the index file of this chunker, model, producer and namespace.");
        _rows[to] = old is null
            ? fresh
            : old with
            {
                File = to, Chunker = chunker, ModelTag = fresh.ModelTag, Namespace = ns, ModelId = modelId, Producer = producer, ProducerTag = fresh.ProducerTag,
            };
    }

    /// <summary>
    /// Writes the registry atomically (temp file, flush, one rename), after copying an unreadable registry to
    /// <c>chunkcache.json.bad-&lt;utc time&gt;</c>. Call it holding the store lock and pass it as proof (<see cref="Update"/> does).
    /// </summary>
    /// <param name="storeLock">The held store lock of this registry's store root.</param>
    /// <exception cref="InvalidOperationException">The lock was released or is another store's.</exception>
    /// <exception cref="InvalidDataException">The registry on disk has a newer format version, or could not be read: it is
    /// never overwritten.</exception>
    public void Save(StoreLockHandle storeLock)
    {
        ArgumentNullException.ThrowIfNull(storeLock);
        storeLock.EnsureHeldFor(_root);
        string path = DefaultPath(_root);
        if (_newerVersion is int v)
            throw new InvalidDataException(v > 0
                ? $"Store registry '{path}' has format version {v}, newer than this build's {FormatVersion}; it is not overwritten. Use a newer tool."
                : $"Store registry '{path}' could not be read; it is not overwritten. Check its permissions.");
        StorePaths.CreateStoreDirectory(_root);
        if (_unreadableBytes is not null)
        {
            string bad = BadCopyPath(path, DateTime.UtcNow, File.Exists);
            File.WriteAllBytes(bad, _unreadableBytes);
            _unreadableBytes = null;
        }
        var doc = new RegistryDocument
        {
            FormatVersion = FormatVersion,
            Indexes = _rows.Values.Select(e => (RowDocument?)new RowDocument
            {
                File = e.File, Chunker = e.Chunker, Model12 = e.ModelTag, Producer8 = e.ProducerTag, ModelId = e.ModelId, TokenizerId = e.TokenizerId,
                Backend = e.Backend, Namespace = e.Namespace, Producer = e.Producer, Updated = e.Updated,
                Tool = e.Tool is null ? null : new Dictionary<string, string?>(e.Tool, StringComparer.Ordinal), Extra = e.Extra,
            }).ToList(),
        };
        ChunkIndex.SweepStaleTemps(_root, FileName);
        string temp = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            using (var stream = new FileStream(temp, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                JsonSerializer.Serialize(stream, doc, StoreRegistryJsonContext.Default.RegistryDocument);
                stream.Flush(flushToDisk: true);
            }
            BeforeRename?.Invoke(temp);
            ChunkIndex.MoveWithRetry(temp, path);
        }
        finally
        {
            try { if (File.Exists(temp)) File.Delete(temp); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
        }
    }

    /// <summary>
    /// One registry write: takes the store lock, loads, applies <paramref name="change"/>, saves, releases. Concurrent
    /// writers serialise on the lock, so none loses another's rows.
    /// </summary>
    /// <param name="storeRoot">Store directory.</param>
    /// <param name="lockTimeout">How long to wait for the store lock.</param>
    /// <param name="change">The change (runs under the lock; keep it short).</param>
    /// <returns>The saved registry.</returns>
    /// <exception cref="ChunkStoreLockTimeoutException">The store lock was not obtained; nothing was written.</exception>
    public static StoreRegistry Update(string storeRoot, TimeSpan lockTimeout, Action<StoreRegistry> change)
    {
        ArgumentNullException.ThrowIfNull(change);
        using StoreLockHandle held = ChunkIndex.AcquireStoreLock(storeRoot, lockTimeout);
        StoreRegistry registry = Load(storeRoot);
        change(registry);
        registry.Save(held);
        return registry;
    }

    internal sealed class RegistryDocument
    {
        public int? FormatVersion { get; set; }

        public List<RowDocument?>? Indexes { get; set; }
    }

    internal sealed class RowDocument
    {
        public string? File { get; set; }

        /// <summary>Version 1 only.</summary>
        public string? IndexFile { get; set; }

        public string? Chunker { get; set; }

        public string? Model12 { get; set; }

        public string? Producer8 { get; set; }

        public string? ModelId { get; set; }

        public string? TokenizerId { get; set; }

        public string? Backend { get; set; }

        public string? Namespace { get; set; }

        public string? Producer { get; set; }

        public DateTimeOffset? Updated { get; set; }

        public Dictionary<string, string?>? Tool { get; set; }

        [JsonExtensionData]
        public Dictionary<string, JsonElement>? Extra { get; set; }
    }
}

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower, WriteIndented = true,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull, AllowDuplicateProperties = false)]
[JsonSerializable(typeof(StoreRegistry.RegistryDocument))]
internal sealed partial class StoreRegistryJsonContext : JsonSerializerContext;
