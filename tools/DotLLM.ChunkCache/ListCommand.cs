using System.Text.Json;
using DotLLM.ChunkReuse.Store;

namespace DotLLM.ChunkCache;

/// <summary>One index of a store as <c>list</c> reports it (<c>stores[].indexes[]</c>).</summary>
internal sealed record ListIndexJson
{
    /// <summary>Index file name in the store root.</summary>
    public string File { get; init; } = "";

    /// <summary>Chunker name (<c>symbol-512</c>).</summary>
    public string Chunker { get; init; } = "";

    /// <summary>Content-based model id the index was built for.</summary>
    public string ModelId { get; init; } = "";

    /// <summary>Producer identity (the key of no tokens).</summary>
    public string Producer { get; init; } = "";

    /// <summary>The GGUF path last used (informational).</summary>
    public string? ModelPath { get; init; }

    /// <summary>Source kind (<c>solution</c> or <c>directory</c>).</summary>
    public string? SourceKind { get; init; }

    /// <summary>The solution or directory scanned.</summary>
    public string? SourcePath { get; init; }

    /// <summary>The root the index paths are relative to.</summary>
    public string? Root { get; init; }

    /// <summary>Last build, UTC (ISO 8601).</summary>
    public string? UpdatedUtc { get; init; }

    /// <summary><c>ok</c>, <c>missing</c> (the file vanished between the registry read, which drops rows of files that are gone, and the index read) or <c>unreadable</c>.</summary>
    public string State { get; init; } = "ok";

    /// <summary>Why the file is unreadable.</summary>
    public string? Error { get; init; }

    /// <summary>Indexed files.</summary>
    public int Files { get; init; }

    /// <summary>Chunks (a key that repeats counts each time).</summary>
    public int Chunks { get; init; }

    /// <summary>Tokens of all chunks.</summary>
    public long Tokens { get; init; }

    /// <summary>Distinct chunk keys.</summary>
    public int DistinctKeys { get; init; }

    /// <summary>Distinct keys the store holds an entry for.</summary>
    public int StoredKeys { get; init; }

    /// <summary>Distinct keys the store lacks.</summary>
    public int MissingKeys { get; init; }
}

/// <summary>One store as <c>list</c> reports it.</summary>
internal sealed record ListStoreJson
{
    /// <summary>The store directory.</summary>
    public string Path { get; init; } = "";

    /// <summary>Entry files (<c>.ckv</c>).</summary>
    public int Entries { get; init; }

    /// <summary>Their bytes.</summary>
    public long EntryBytes { get; init; }

    /// <summary>Leftover temp files of a crashed writer.</summary>
    public int TempFiles { get; init; }

    /// <summary>Their bytes.</summary>
    public long TempBytes { get; init; }

    /// <summary><c>held</c> (another holder owns the store, or the lock file cannot be probed) or <c>free</c>. Advisory: one probe, stale at once.</summary>
    public string OwnerLock { get; init; } = "free";

    /// <summary>The server ledger's state.</summary>
    public LedgerJson Ledger { get; init; } = new("absent", null, null, null, null);

    /// <summary>Why the registry was rebuilt from the index files or had rows dropped (never written back by <c>list</c>).</summary>
    public string? RegistryNote { get; init; }

    /// <summary>Registry rows refused (not a plain index file name).</summary>
    public IReadOnlyList<string>? InvalidRegistryRows { get; init; }

    /// <summary>The indexes, by file name.</summary>
    public IReadOnlyList<ListIndexJson> Indexes { get; init; } = [];
}

/// <summary>The server's own figures (<c>GET /v1/chunk-cache</c>) as <c>list --server</c> reports them.</summary>
internal sealed record ListServerJson(string Url, string? ModelId, string? Producer, string? TokenizerId, long? Entries, long? Bytes, long? BudgetBytes);

/// <summary>Result of <c>list</c>.</summary>
internal sealed record ListResult
{
    /// <summary>One per <c>--store</c>, in the order given.</summary>
    public IReadOnlyList<ListStoreJson> Stores { get; init; } = [];

    /// <summary>The server's figures (with <c>--server</c>).</summary>
    public ListServerJson? Server { get; init; }
}

/// <summary>
/// <c>chunkcache list --store DIR [--store DIR ...] [--server URL]</c>: a read-only inventory of stores. It opens each store with
/// <see cref="StoreLayout.Open"/> (<c>create: false</c>, so a missing directory is an error and nothing is created), probes the owner
/// lock once with <see cref="StoreOwnerLock.IsHeldByAnother"/> (advisory, issue 1664: a writer holds the lock for its whole run instead
/// of consulting this), and reads each index file directly. It takes no lock and writes no registry, ledger or index. The index
/// counts come from the files without a model: <see cref="ChunkIndex.Load"/> needs the model's key function and answers an empty index
/// for any other producer, so it cannot be used here.
/// </summary>
internal static class ListCommand
{
    public static int Run(CliContext ctx, CliArgs args)
    {
        IReadOnlyList<string> roots = args.List("store");
        if (roots.Count == 0) throw new UsageException("list: --store DIR is required (repeat it to list several stores).");

        // Every store is inspected before the server is asked, so a missing store fails fast and costs no request.
        var stores = roots.Select(root => Inspect(ctx.FullPath(root))).ToList();

        ListServerJson? server = null;
        if (args.Value("server") is { } url)
        {
            using ServerConnection connection = ServerConnection.Open(ctx, url);
            ServerIdentity identity = connection.GetIdentityAsync().GetAwaiter().GetResult();
            server = new ListServerJson(connection.BaseUrl, identity.ModelId, identity.Producer, identity.TokenizerId, identity.Entries, identity.Bytes, identity.BudgetBytes);
        }

        var result = new ListResult { Stores = stores, Server = server };
        if (ctx.Json)
        {
            ctx.WriteJson(result);
            return ExitCodes.Ok;
        }
        TextWriter o = ctx.Out;
        foreach (ListStoreJson s in stores) WriteText(o, s);
        if (server is not null)
            o.WriteLine($"server {server.Url}: model {Short(server.ModelId)}, producer {Short(server.Producer)}, {server.Entries?.ToString(System.Globalization.CultureInfo.InvariantCulture) ?? "?"} entries"
                + (server.Bytes is long b ? $", {Common.Bytes(b)}" : "") + (server.BudgetBytes is long budget and > 0 ? $", budget {Common.Bytes(budget)}" : ""));
        return ExitCodes.Ok;
    }

    private static ListStoreJson Inspect(string root)
    {
        StoreLayout layout = StoreLayout.Open(root, create: false);
        FileChunkStore files = layout.Files ?? throw new DirectoryNotFoundException($"Store not found: {layout.Root}");
        bool held = StoreOwnerLock.IsHeldByAnother(layout.Root);
        StoreStats stats = files.GetStats();
        HashSet<string> stored = files.Keys().Select(k => k.Hex).ToHashSet(StringComparer.Ordinal);

        var indexes = new List<ListIndexJson>();
        var invalid = new List<string>(layout.Registry.InvalidRows);
        foreach (IndexRecord record in layout.Records.OrderBy(r => r.IndexFile, StringComparer.Ordinal))
        {
            if (!layout.IsIndexFileName(record.IndexFile)) invalid.Add(record.IndexFile);
            else indexes.Add(ReadIndex(layout.Root, record, stored));
        }
        return new ListStoreJson
        {
            Path = layout.Root, Entries = stats.Entries, EntryBytes = stats.EntryBytes, TempFiles = stats.TempFiles, TempBytes = stats.TempBytes,
            OwnerLock = held ? "held" : "free", Ledger = StatusCommand.ReadLedger(layout.Root), RegistryNote = layout.Registry.LoadNote,
            InvalidRegistryRows = invalid.Count > 0 ? invalid : null, Indexes = indexes,
        };
    }

    private static ListIndexJson ReadIndex(string root, IndexRecord record, HashSet<string> stored)
    {
        var info = new ListIndexJson
        {
            File = record.IndexFile, Chunker = record.Chunker, ModelId = record.ModelId, Producer = record.Producer,
            ModelPath = NullIfEmpty(record.ModelPath), SourceKind = NullIfEmpty(record.SourceKind), SourcePath = NullIfEmpty(record.SourcePath),
            Root = NullIfEmpty(record.Root), UpdatedUtc = record.UpdatedUtc == default ? null : record.UpdatedUtc.ToString("O", System.Globalization.CultureInfo.InvariantCulture),
        };
        string path = System.IO.Path.Combine(root, record.IndexFile);
        if (!System.IO.File.Exists(path)) return info with { State = "missing" };
        try
        {
            using FileStream stream = ChunkIndex.OpenForRead(path);
            ChunkIndex.IndexDocument? doc = JsonSerializer.Deserialize(stream, ChunkIndexJsonContext.Default.IndexDocument);
            if (doc?.Files is null) return info with { State = "unreadable", Error = "the file is not an index document" };
            var keys = new HashSet<string>(StringComparer.Ordinal);
            int chunks = 0, files = 0;
            long tokens = 0;
            foreach (ChunkIndex.FileDocument? file in doc.Files)
            {
                if (file?.Chunks is null) continue;
                files++;
                foreach (ChunkIndex.ChunkDocument? chunk in file.Chunks)
                {
                    if (chunk is null) continue;
                    chunks++;
                    tokens += chunk.Tokens;
                    keys.Add(chunk.Key);
                }
            }
            int present = keys.Count(stored.Contains);
            return info with { Files = files, Chunks = chunks, Tokens = tokens, DistinctKeys = keys.Count, StoredKeys = present, MissingKeys = keys.Count - present };
        }
        catch (Exception ex) when (ex is JsonException or IOException or UnauthorizedAccessException)
        {
            return info with { State = "unreadable", Error = ex.Message };
        }
    }

    private static void WriteText(TextWriter o, ListStoreJson s)
    {
        o.WriteLine($"store {s.Path}: {s.Entries} entries, {Common.Bytes(s.EntryBytes)}"
            + (s.TempFiles > 0 ? $"; {s.TempFiles} leftover temp files ({Common.Bytes(s.TempBytes)})" : "")
            + $"; owner lock: {s.OwnerLock} (advisory: one probe, may change at once)");
        o.WriteLine(s.Ledger.State switch
        {
            "ok" => $"  server ledger: {s.Ledger.Rows} entries",
            "absent" => "  server ledger: none",
            _ => $"  server ledger UNREADABLE: {s.Ledger.Error}",
        });
        if (s.RegistryNote is not null) o.WriteLine($"  registry: {s.RegistryNote}");
        foreach (string row in s.InvalidRegistryRows ?? []) o.WriteLine($"  registry row refused: {row}");
        foreach (ListIndexJson i in s.Indexes)
        {
            o.WriteLine($"  {i.Chunker}  model {Short(i.ModelId)}  producer {Short(i.Producer)}  [{i.File}]");
            if (i.State != "ok")
            {
                o.WriteLine($"    {i.State}{(i.Error is null ? "" : ": " + i.Error)}");
                continue;
            }
            o.WriteLine($"    {i.Files} files, {i.Chunks} chunks, {i.Tokens} tokens; {i.StoredKeys} of {i.DistinctKeys} keys stored, {i.MissingKeys} missing");
            if (i.SourcePath is not null) o.WriteLine($"    source {i.SourceKind} {i.SourcePath}{(i.UpdatedUtc is null ? "" : "; built " + i.UpdatedUtc)}");
        }
    }

    private static string? NullIfEmpty(string? s) => string.IsNullOrEmpty(s) ? null : s;

    private static string Short(string? id) => string.IsNullOrEmpty(id) ? "?" : ModelHandle.Short(id);
}
