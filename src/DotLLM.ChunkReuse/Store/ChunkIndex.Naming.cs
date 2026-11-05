using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace DotLLM.ChunkReuse.Store;

/// <summary>The parts of an index file name <c>index-&lt;chunker&gt;.&lt;model12&gt;.&lt;p8&gt;.&lt;ns-or-default&gt;.json</c>.</summary>
/// <param name="Chunker">Chunker name (<see cref="ChunkIndex.IsValidChunkerName"/>).</param>
/// <param name="ModelTag">First 12 hex digits of SHA-256 of the content model id (<see cref="ChunkIndex.ModelTag"/>).</param>
/// <param name="ProducerTag">First 8 hex digits of the producer identity (<see cref="ChunkIndex.ProducerTag"/>).</param>
/// <param name="Namespace">The namespace (<see cref="ChunkNamespace.Default"/> for the literal <c>default</c>).</param>
public sealed record IndexFileName(string Chunker, string ModelTag, string ProducerTag, string Namespace);

/// <summary>What <see cref="ChunkIndex.MigrateLegacyIndex"/> did.</summary>
/// <param name="IndexPath">The index file to use: <see cref="ChunkIndex.DefaultPath(string, string, string, string, string)"/>.</param>
/// <param name="MigratedFrom">The older file renamed to <paramref name="IndexPath"/>, or null when nothing was renamed.</param>
/// <param name="LeftBehind">Older files of this chunker that were not renamed (other producer, unreadable, or the current file
/// already exists), each with the reason. They stay on disk as indexes (their keys stay protected from eviction) until
/// someone deletes them.</param>
public sealed record IndexMigration(string IndexPath, string? MigratedFrom, IReadOnlyList<string> LeftBehind);

public sealed partial class ChunkIndex
{
    /// <summary>The file name component for <see cref="ChunkNamespace.Default"/> (reserved: no namespace may be called that).</summary>
    public const string DefaultNamespaceFileTag = ChunkNamespace.ReservedDefaultName;

    /// <summary>Length of <see cref="ModelTag"/>, in hex digits.</summary>
    public const int ModelTagLength = 12;

    /// <summary>Length of <see cref="ProducerTag"/>, in hex digits.</summary>
    public const int ProducerTagLength = 8;

    /// <summary>Longest chunker name usable in an index file name.</summary>
    public const int MaxChunkerNameLength = 64;

    /// <summary>
    /// Whether <paramref name="name"/> can name a chunker in an index file name: 1-64 characters of <c>[a-z0-9_-]</c> (no
    /// <c>.</c>, which separates the name's parts), e.g. <c>symbol-512</c>, <c>lines-512</c>.
    /// </summary>
    /// <param name="name">Candidate.</param>
    /// <returns>Whether it is valid.</returns>
    public static bool IsValidChunkerName(string? name)
    {
        if (string.IsNullOrEmpty(name) || name.Length > MaxChunkerNameLength) return false;
        foreach (char c in name)
            if (c is not ((>= 'a' and <= 'z') or (>= '0' and <= '9') or '_' or '-')) return false;
        return true;
    }

    /// <summary>
    /// The model component of an index file name: the first <see cref="ModelTagLength"/> lowercase hex digits of SHA-256 over
    /// the UTF-8 bytes of the content model id (<c>GgufIdentity.ModelId</c> form, e.g. the server's model identity), so a
    /// renamed GGUF keeps its index and a re-quantised one gets another, and all of a model's indexes group by it.
    /// </summary>
    /// <param name="modelId">The content model id; not empty.</param>
    /// <returns>12 hex digits.</returns>
    public static string ModelTag(string modelId)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelId);
        Span<byte> digest = stackalloc byte[SHA256.HashSizeInBytes];
        SHA256.HashData(Encoding.UTF8.GetBytes(modelId), digest);
        return Convert.ToHexStringLower(digest)[..ModelTagLength];
    }

    /// <summary>
    /// The producer component of an index file name: the first <see cref="ProducerTagLength"/> hex digits of the producer
    /// identity (<c>keyOf([]).Hex</c>, which hashes model fingerprint, tokenizer, RoPE, backend, dtype, positioning and
    /// namespace), so a CPU and a Vulkan producer of one model (or two KV dtypes) keep separate indexes. A server-backed
    /// client takes the producer from the server (<c>GET /v1/chunk-cache</c>, plan Task 14).
    /// </summary>
    /// <param name="producer">The producer identity: 64 lowercase hex digits.</param>
    /// <returns>8 hex digits.</returns>
    /// <exception cref="ArgumentException">Not 64 lowercase hex digits.</exception>
    public static string ProducerTag(string producer)
    {
        if (!ChunkKey.IsWellFormed(producer)) throw new ArgumentException("A producer identity is 64 lowercase hex digits (keyOf([]).Hex).", nameof(producer));
        return producer[..ProducerTagLength];
    }

    /// <summary>
    /// The index file of one chunker, model, producer and namespace (integration spec L9, review ruling S3):
    /// <c>&lt;storeRoot&gt;/index-&lt;chunker&gt;.&lt;model12&gt;.&lt;p8&gt;.&lt;ns-or-default&gt;.json</c>, where <c>model12</c>
    /// is <see cref="ModelTag"/>, <c>p8</c> is <see cref="ProducerTag"/> and the last part is the namespace, or <c>default</c>
    /// for <see cref="ChunkNamespace.Default"/>. Two models, chunkers, producers (backend, dtype, tokenizer, RoPE) or namespaces
    /// sharing a store root therefore never share (and clobber) an index.
    /// </summary>
    /// <param name="storeRoot">Store directory.</param>
    /// <param name="chunkerName">Chunker name (<see cref="IsValidChunkerName"/>).</param>
    /// <param name="modelId">Content model id (<see cref="ModelTag"/>).</param>
    /// <param name="producer">Producer identity, <c>keyOf([]).Hex</c> (<see cref="ProducerTag"/>).</param>
    /// <param name="ns">Namespace, validated with <see cref="ChunkNamespace.Validate"/> before it reaches the file name.</param>
    /// <returns>The path.</returns>
    /// <exception cref="ArgumentException">A part is not valid.</exception>
    /// <remarks>Only 32 bits of the producer are in the name. A writer still checks <see cref="ProducerOf"/> before saving over
    /// an existing file (a prefix collision is refused, never overwritten).</remarks>
    public static string DefaultPath(string storeRoot, string chunkerName, string modelId, string producer, string ns)
    {
        ArgumentException.ThrowIfNullOrEmpty(storeRoot);
        return Path.Combine(Path.GetFullPath(storeRoot), FileNameOf(chunkerName, modelId, producer, ns));
    }

    /// <summary>The file name of <see cref="DefaultPath(string, string, string, string, string)"/> (no directory).</summary>
    private static string FileNameOf(string chunkerName, string modelId, string producer, string ns)
    {
        if (!IsValidChunkerName(chunkerName))
            throw new ArgumentException($"Chunker name '{chunkerName}' is not 1-{MaxChunkerNameLength} characters of [a-z0-9_-].", nameof(chunkerName));
        string model = ModelTag(modelId);
        string p8 = ProducerTag(producer);
        return $"{IndexFilePrefix}{chunkerName}.{model}.{p8}.{NamespaceFileTag(ns)}{IndexFileSuffix}";
    }

    /// <summary>The namespace part of a file name: the validated namespace, or <see cref="DefaultNamespaceFileTag"/>.</summary>
    private static string NamespaceFileTag(string ns)
    {
        ChunkNamespace.Validate(ns, nameof(ns));
        return ns.Length == 0 ? DefaultNamespaceFileTag : ns;
    }

    /// <summary>
    /// Parses a file name of the <see cref="DefaultPath(string, string, string, string, string)"/> form. Older names
    /// (<c>index-&lt;chunker&gt;.json</c>, <c>index-&lt;chunker&gt;.&lt;producer12&gt;.json</c>, and the Task 4b
    /// <c>index-&lt;chunker&gt;.&lt;model12&gt;.&lt;ns-or-default&gt;.json</c>) and anything else return false (a Task 4b name whose
    /// namespace starts with 8 hex digits and a dot can parse as the current form; such files are only migrated by exact
    /// name, never by parsing).
    /// </summary>
    /// <param name="fileName">A file name (no directory).</param>
    /// <param name="parts">The parts when it parses.</param>
    /// <returns>Whether it is an index file name of the current form.</returns>
    public static bool TryParseIndexFileName(string? fileName, out IndexFileName parts)
    {
        parts = null!;
        if (fileName is null || !fileName.StartsWith(IndexFilePrefix, StringComparison.Ordinal) || !fileName.EndsWith(IndexFileSuffix, StringComparison.Ordinal))
            return false;
        string body = fileName[IndexFilePrefix.Length..^IndexFileSuffix.Length];
        int dot = body.IndexOf('.', StringComparison.Ordinal);
        int pAt = dot + 1 + ModelTagLength + 1, nsAt = pAt + ProducerTagLength + 1;
        if (dot <= 0 || body.Length <= nsAt || body[pAt - 1] != '.' || body[nsAt - 1] != '.') return false;
        string chunker = body[..dot];
        string model = body.Substring(dot + 1, ModelTagLength);
        string p8 = body.Substring(pAt, ProducerTagLength);
        string nsTag = body[nsAt..];
        if (!IsValidChunkerName(chunker) || !model.All(char.IsAsciiHexDigitLower) || !p8.All(char.IsAsciiHexDigitLower)) return false;
        string ns = nsTag == DefaultNamespaceFileTag ? ChunkNamespace.Default : nsTag;
        if (ns.Length > 0 && !ChunkNamespace.IsValid(ns)) return false;
        parts = new IndexFileName(chunker, model, p8, ns);
        return true;
    }

    /// <summary>
    /// The lock file of the index of one chunker, model, producer and namespace:
    /// <see cref="DefaultPath(string, string, string, string, string)"/> with <see cref="IndexLockSuffix"/>, e.g.
    /// <c>index-symbol-512.0123456789ab.89abcdef.default.lock</c>.
    /// </summary>
    /// <param name="storeRoot">The store directory.</param>
    /// <param name="chunkerName">Chunker name.</param>
    /// <param name="modelId">Content model id.</param>
    /// <param name="producer">Producer identity (64 hex).</param>
    /// <param name="ns">Namespace.</param>
    /// <returns>The path.</returns>
    public static string IndexLockPath(string storeRoot, string chunkerName, string modelId, string producer, string ns) =>
        LockPathOf(DefaultPath(storeRoot, chunkerName, modelId, producer, ns));

    /// <summary>
    /// Takes the writer lock of one index (<see cref="IndexLockPath"/>): exclusive like the store lock, but per index (chunker,
    /// model, producer and namespace), so index-only updates (load, <see cref="Update"/>, <see cref="Save"/> of an index
    /// without encoding) of different indexes never block each other, and never wait for the store lock (a server's put
    /// batches, an eviction pass). Every process that saves an index must hold its index lock from <see cref="Load"/> through
    /// <see cref="Save"/>, or two writers lose each other's updates (the last rename wins).
    /// </summary>
    /// <param name="storeRoot">The store directory (created if missing).</param>
    /// <param name="chunkerName">Chunker name.</param>
    /// <param name="modelId">Content model id.</param>
    /// <param name="producer">Producer identity (64 hex).</param>
    /// <param name="ns">Namespace.</param>
    /// <param name="timeout">How long to wait for another holder; <see cref="TimeSpan.Zero"/> tries once.</param>
    /// <returns>The lock; dispose to release (idempotent).</returns>
    /// <exception cref="ChunkStoreLockTimeoutException">Another holder kept this index's lock for <paramref name="timeout"/>.</exception>
    /// <exception cref="InvalidOperationException">The current flow holds this store's lock: the lock order is index lock first,
    /// then the store lock, never the reverse (a store-lock holder waiting for an index lock whose holder waits for the store
    /// lock is a deadlock). Refused at once, without waiting.</exception>
    public static IDisposable AcquireIndexLock(string storeRoot, string chunkerName, string modelId, string producer, string ns, TimeSpan timeout) =>
        AcquireIndexLockAt(IndexLockPath(storeRoot, chunkerName, modelId, producer, ns), timeout);

    /// <summary>
    /// The producer identity recorded in an index file (the key of no tokens it was saved with), or null when the file is
    /// missing, unreadable or has none.
    /// </summary>
    /// <param name="indexPath">An index file.</param>
    /// <returns>The producer hex, or null.</returns>
    public static string? ProducerOf(string indexPath)
    {
        ArgumentException.ThrowIfNullOrEmpty(indexPath);
        try
        {
            using FileStream stream = OpenForRead(indexPath);
            return JsonSerializer.Deserialize(stream, ChunkIndexJsonContext.Default.IndexDocument)?.Producer;
        }
        catch (Exception ex) when (ex is JsonException or IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <summary>
    /// Older names of one chunker's index that <see cref="MigrateLegacyIndex"/> considers, in order: the library's
    /// <c>index-&lt;chunker&gt;.json</c> (<see cref="DefaultPath(string, string)"/>), the Phase S CLI's
    /// <c>index-&lt;chunker&gt;.&lt;producer12&gt;.json</c>, and the Task 4b <c>index-&lt;chunker&gt;.&lt;model12&gt;.&lt;ns-or-default&gt;.json</c>.
    /// </summary>
    private static IEnumerable<string> LegacyPaths(string root, string chunkerName, string modelId, string producer, string ns)
    {
        yield return DefaultPath(root, chunkerName);
        yield return DefaultPath(root, chunkerName + "." + producer[..ModelTagLength]);
        yield return Path.Combine(root, $"{IndexFilePrefix}{chunkerName}.{ModelTag(modelId)}.{NamespaceFileTag(ns)}{IndexFileSuffix}");
    }

    /// <summary>
    /// Test hook: runs inside <see cref="MigrateLegacyIndex"/> under the store lock, after the rename, before the registry save.
    /// </summary>
    internal static Action? AfterMigrationRename { get; set; }

    /// <summary>
    /// Migrates an older index of this chunker, model, producer and namespace to
    /// <see cref="DefaultPath(string, string, string, string, string)"/>, once: when the current file does not exist and an older
    /// file (<c>index-&lt;chunker&gt;.json</c>, <c>index-&lt;chunker&gt;.&lt;producer12&gt;.json</c>, or the Task 4b
    /// <c>index-&lt;chunker&gt;.&lt;model12&gt;.&lt;ns&gt;.json</c>) was saved by this producer (<paramref name="keyOf"/>), the older
    /// file is renamed to the current name (no copy, so its keys never lose protection) and its store-registry row, if any, is
    /// moved with it. An older file of another producer, or an unreadable one, is left where it is: it is still an index
    /// (eviction keeps its keys) until someone deletes it. Afterwards the older name is gone, so it is read once.
    /// </summary>
    /// <param name="storeRoot">The store directory.</param>
    /// <param name="chunkerName">Chunker name.</param>
    /// <param name="modelId">Content model id.</param>
    /// <param name="ns">Namespace.</param>
    /// <param name="keyOf">This producer's key function (the producer is <c>keyOf([]).Hex</c>).</param>
    /// <param name="lockTimeout">How long to wait for the store lock (taken for the rename and the registry save).</param>
    /// <returns>What happened.</returns>
    /// <exception cref="InvalidDataException">The rename happened but the registry could not be saved (e.g. a newer-format
    /// registry); the message says the index was renamed. The index is usable; only its registry row is missing.</exception>
    /// <remarks>Call it holding the current index's lock
    /// (<see cref="AcquireIndexLock(string, string, string, string, string, TimeSpan)"/>), before <see cref="Load"/>; it takes the
    /// store lock itself (lock order: index lock, then store lock). Legacy writers held the store lock for their whole run, so
    /// the rename never interleaves with one.</remarks>
    public static IndexMigration MigrateLegacyIndex(string storeRoot, string chunkerName, string modelId, string ns, ChunkKeyFunction keyOf, TimeSpan lockTimeout)
    {
        ArgumentNullException.ThrowIfNull(keyOf);
        string producer = keyOf([]).Hex;
        string target = DefaultPath(storeRoot, chunkerName, modelId, producer, ns);
        string root = Path.GetDirectoryName(target)!;
        var left = new List<string>();
        if (!Directory.Exists(root)) return new IndexMigration(target, null, left);
        string[] candidates = LegacyPaths(root, chunkerName, modelId, producer, ns).Where(File.Exists).ToArray();
        if (candidates.Length == 0) return new IndexMigration(target, null, left);

        using StoreLockHandle held = AcquireStoreLock(root, lockTimeout);
        string? from = null;
        foreach (string legacy in candidates)
        {
            if (!File.Exists(legacy)) continue;
            string? p = ProducerOf(legacy);
            if (from is not null || File.Exists(target)) left.Add($"{Path.GetFileName(legacy)} (the current index {Path.GetFileName(target)} already exists)");
            else if (p is null) left.Add($"{Path.GetFileName(legacy)} (unreadable or no producer)");
            else if (!string.Equals(p, producer, StringComparison.Ordinal)) left.Add($"{Path.GetFileName(legacy)} (another producer)");
            else
            {
                StoreRegistry registry = StoreRegistry.Load(root);
                File.Move(legacy, target, overwrite: false);
                from = legacy;
                AfterMigrationRename?.Invoke();
                registry.Rename(Path.GetFileName(legacy), Path.GetFileName(target), chunkerName, modelId, producer, ns);
                try
                {
                    registry.Save(held);
                }
                catch (Exception ex) when (ex is InvalidDataException or IOException or UnauthorizedAccessException)
                {
                    throw new InvalidDataException($"Index '{Path.GetFileName(legacy)}' was renamed to '{Path.GetFileName(target)}' (usable as is), "
                        + $"but the store registry was not updated: {ex.Message}", ex);
                }
            }
        }
        return new IndexMigration(target, from, left);
    }

    /// <summary>
    /// The index file a read-only caller should open for this chunker, model, producer and namespace: the current name when it
    /// exists, else a not-yet-migrated older file of this producer, else the current name (which then loads empty). Never renames.
    /// </summary>
    /// <param name="storeRoot">The store directory.</param>
    /// <param name="chunkerName">Chunker name.</param>
    /// <param name="modelId">Content model id.</param>
    /// <param name="ns">Namespace.</param>
    /// <param name="keyOf">This producer's key function.</param>
    /// <returns>The path to load.</returns>
    public static string ResolveIndexPath(string storeRoot, string chunkerName, string modelId, string ns, ChunkKeyFunction keyOf)
    {
        ArgumentNullException.ThrowIfNull(keyOf);
        string producer = keyOf([]).Hex;
        string target = DefaultPath(storeRoot, chunkerName, modelId, producer, ns);
        if (File.Exists(target)) return target;
        foreach (string legacy in LegacyPaths(Path.GetDirectoryName(target)!, chunkerName, modelId, producer, ns))
            if (File.Exists(legacy) && string.Equals(ProducerOf(legacy), producer, StringComparison.Ordinal)) return legacy;
        return target;
    }
}
