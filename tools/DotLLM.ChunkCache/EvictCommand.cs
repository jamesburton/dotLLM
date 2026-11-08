using DotLLM.ChunkCache.Client;
using DotLLM.ChunkReuse.Store;

namespace DotLLM.ChunkCache;

/// <summary>Result of <c>evict</c>: the <c>chunkcache_evict</c> output schema plus extras.</summary>
internal sealed record EvictResult
{
    public int Evicted { get; init; }
    public long FreedBytes { get; init; }
    public int KeptReferenced { get; init; }
    public bool DryRun { get; init; }

    public string Mode { get; init; } = "orphans";

    /// <summary>The server that evicted (server-backed runs).</summary>
    public string? Server { get; init; }

    /// <summary>Offline <c>--mode all</c>: entries whose header could not be read (damaged or partial); kept.</summary>
    public int UnreadableEntries { get; init; }
    public IReadOnlyList<string> Keys { get; init; } = [];
    public IReadOnlyList<string> DroppedIndexes { get; init; } = [];
    public int TempFilesRemoved { get; init; }
    public int StoreEntriesAfter { get; init; }
}

/// <summary>
/// <c>chunkcache evict</c> (integration spec §7.2): modes <c>orphans | path | namespace | all</c>, a dry run unless <c>--apply</c>.
/// Through <c>--server</c> the server evicts (it holds the write lock and the ledger; the CLI never touches either). With
/// <c>--offline</c> (a store no server owns) the CLI lists or deletes the store-wide orphans (entries that no <c>index-*.json</c> and
/// not the server ledger reference: <see cref="ChunkIndex.StoreWideOrphans"/>, a mark-and-sweep) or, with <c>--mode all --model M</c>,
/// one model's entries (by the fingerprint in each entry's header) and indexes. An applying offline run takes the store's owner
/// lock for its WHOLE pass and then the exclusive store lock (one eviction pass, integration spec L3), so no put interleaves.
/// </summary>
internal static class EvictCommand
{
    public static readonly IReadOnlyDictionary<string, OptionKind> Spec = Common.Merge(Common.ModelOptions, new Dictionary<string, OptionKind>
    {
        ["store"] = OptionKind.Single,
        ["apply"] = OptionKind.Flag,
        ["mode"] = OptionKind.Single,
        ["path"] = OptionKind.Single,
        ["namespace"] = OptionKind.Single,
        ["all-for-model"] = OptionKind.Flag, // the old CLI mode: rejected with a hint to --mode all
        ["lock-timeout"] = OptionKind.Single,
        ["show"] = OptionKind.Single,
    });

    public const string Help = """
        chunkcache evict (--server URL | --offline --store DIR) [--mode orphans|path|namespace|all] [--apply] [options]

          Default: a dry run. --mode orphans (default): entries no index and not the server ledger reference. --mode path --path G:
          forget a repo (entries first seen under the glob, their index references and sightings). --mode namespace [--namespace N]
          and --mode all: wipe a namespace / the store (across namespaces only for an admin key).
          --server URL    the server evicts (POST /v1/chunk-cache/evict); the API key is read from CHUNKCACHE_API_KEY only
          --offline       no server owns --store: orphans (and --mode all --model M.gguf: that model's entries and indexes) are
                          listed, and with --apply deleted while holding the store's owner lock for the whole pass (exit 1, store_owned,
                          while a server or another offline writer owns it). An unreadable index or ledger blocks eviction (exit 1).
          --apply         really evict (--lock-timeout S for the short store-lock scopes, default 30)
          --strict-model-id  identify --model by a full hash (must match how it was built)
          --json --show N (keys to print, default 20)
        The old --all-for-model is --mode all --model M.gguf --offline.
        """;

    /// <summary>Test hook, per async flow: runs inside an applying offline pass (owner lock and store lock held), before any deletion.</summary>
    internal static Action? TestHookInsideApply
    {
        get => s_insideApply.Value;
        set => s_insideApply.Value = value;
    }

    private static readonly AsyncLocal<Action?> s_insideApply = new();

    public static int Run(CliContext ctx, CliArgs args)
    {
        if (args.Flag("all-for-model"))
            throw new UsageException("evict: --all-for-model is now --mode all (with --model M.gguf and --offline: one model's entries and indexes; through --server: the caller's namespace).");
        string mode = args.Value("mode") ?? "orphans";
        if (mode is "all_for_model" or "all-for-model")
            throw new UsageException($"evict: mode '{mode}' is now 'all' (--mode all --model M.gguf --offline). Modes: orphans | path | namespace | all.");
        if (mode is not ("orphans" or "path" or "namespace" or "all"))
            throw new UsageException($"evict: unknown mode '{mode}'. Modes: orphans | path | namespace | all.");
        bool apply = args.Flag("apply"), offline = args.Flag("offline");
        string? serverUrl = args.Value("server");
        if (serverUrl is null && !offline)
            throw new UsageException("evict: give --server URL (the server evicts) or --offline --store DIR (a store no server owns).");
        if (serverUrl is not null && offline) throw new UsageException("evict: give --server or --offline, not both.");
        if (mode == "path" && args.Value("path") is null) throw new UsageException("evict: --mode path needs --path GLOB.");
        if (mode != "path" && args.Value("path") is not null) throw new UsageException("evict: --path goes with --mode path.");
        if (mode != "namespace" && args.Value("namespace") is not null) throw new UsageException("evict: --namespace goes with --mode namespace.");
        return serverUrl is not null ? RunServer(ctx, args, serverUrl, mode, apply) : RunOffline(ctx, args, mode, apply);
    }

    private static int RunServer(CliContext ctx, CliArgs args, string serverUrl, string mode, bool apply)
    {
        if (args.Value("model") is not null || args.Flag("strict-model-id") || args.Value("threads") is not null)
            throw new UsageException("evict: --model, --strict-model-id and --threads belong to --offline --mode all; the server knows its model.");
        using ServerConnection server = ServerConnection.Open(ctx, serverUrl);
        Client.EvictResult r = server.Client.EvictAsync(new EvictRequest
        {
            Mode = mode, Path = args.Value("path"), Namespace = args.Value("namespace"), DryRun = !apply,
        }, CancellationToken.None).AsTask().GetAwaiter().GetResult();
        var result = new EvictResult
        {
            Evicted = r.Evicted, FreedBytes = r.FreedBytes, KeptReferenced = r.KeptReferenced, DryRun = r.DryRun ?? !apply, Mode = mode, Server = server.BaseUrl,
        };
        if (ctx.Json)
        {
            ctx.WriteJson(result);
            return ExitCodes.Ok;
        }
        ctx.Out.WriteLine($"{(result.DryRun ? "dry run: would evict" : "evicted")} {result.Evicted} entries ({Common.Bytes(result.FreedBytes)}) in mode {mode} on {server.BaseUrl}; kept {result.KeptReferenced} referenced");
        if (result.DryRun) ctx.Out.WriteLine("run with --apply to evict");
        return ExitCodes.Ok;
    }

    private static int RunOffline(CliContext ctx, CliArgs args, string mode, bool apply)
    {
        string storeRoot = ctx.FullPath(args.Required("store"));
        if (mode is "path" or "namespace")
            throw new UsageException($"evict: --mode {mode} needs --server (sightings and provenance live in the server's ledger; the CLI never touches it).");
        bool allForModel = mode == "all";
        if (allForModel && args.Value("model") is null) throw new UsageException("evict: --offline --mode all needs --model (it removes that model's entries and indexes).");
        if (!allForModel && (args.Value("model") is not null || args.Flag("strict-model-id") || args.Value("threads") is not null))
            throw new UsageException("evict: --model, --strict-model-id and --threads are used only with --mode all.");
        TimeSpan lockTimeout = args.Seconds("lock-timeout", Common.DefaultLockTimeout);
        int show = args.Int("show", 20, 0);
        if (!Directory.Exists(storeRoot)) throw new DirectoryNotFoundException($"Store not found: {storeRoot}");

        // An applying run owns the store for its WHOLE pass (a check followed by deletes would race a server starting in between).
        using ModelHandle? model = allForModel ? Common.OpenModel(ctx, args) : null;
        using StoreOwnerLock? owner = apply ? Common.AcquireOwner(ctx, storeRoot) : null;
        using StoreLockHandle? held = apply ? Common.Lock(ctx, storeRoot, lockTimeout) : null;
        if (apply) TestHookInsideApply?.Invoke();
        StoreLayout layout = StoreLayout.Open(storeRoot, create: false);
        FileChunkStore store = layout.Existing;
        List<ChunkKey> all = store.Keys().ToList();

        IReadOnlyList<ChunkKey> victims;
        int unreadable = 0;
        var dropIndexes = new List<string>();
        if (model is null)
        {
            victims = ChunkIndex.StoreWideOrphans(store, storeRoot); // throws on an unreadable index or server ledger
        }
        else
        {
            var mine = new List<ChunkKey>();
            foreach (ChunkKey k in all)
            {
                if (!store.TryReadFingerprint(k, out string fp)) unreadable++;
                else if (fp == model.Fingerprint) mine.Add(k);
            }
            victims = mine.OrderBy(k => k.Hex, StringComparer.Ordinal).ToList();
            if (victims.Count == 0)
                ctx.Info("note: no entry carries this model's fingerprint; if the store was built with (or without) --strict-model-id, pass the same here.");
            // This model's indexes: current names by model tag (any chunker, any namespace), legacy names by producer, and the
            // registry rows recorded for this model.
            string tag = ChunkIndex.ModelTag(model.ModelId);
            dropIndexes.AddRange(Directory.EnumerateFiles(storeRoot, "index-*.json").Select(Path.GetFileName)
                .Where(f => ChunkIndex.TryParseIndexFileName(f, out IndexFileName p) && p.ModelTag == tag)!);
            dropIndexes.AddRange(Directory.EnumerateFiles(storeRoot, "index-*." + model.Producer[..12] + ".json").Select(Path.GetFileName)!);
            dropIndexes.AddRange(layout.Records.Where(r => r.ModelFingerprint == model.Fingerprint || r.ModelId == model.ModelId).Select(r => r.IndexFile));
            dropIndexes = dropIndexes.Distinct(StringComparer.OrdinalIgnoreCase).Order(StringComparer.Ordinal).ToList();
            // The registry is data: a name that is not a plain index-*.json in the store root is refused before anything is
            // deleted (e.g. "..\\x", an absolute path, a subdirectory); the registry refuses such rows on load, and a registry
            // with refused rows blocks this destructive mode.
            string[] bad = dropIndexes.Where(f => !layout.IsIndexFileName(f)).Concat(layout.Registry.InvalidRows).ToArray();
            if (bad.Length > 0)
                throw new InvalidDataException($"Store registry '{StoreLayout.ConfigFileName}' names index files outside the store or not of the form index-*.json: "
                    + string.Join(", ", bad.Select(b => $"'{b}'")) + ". Nothing was deleted; repair the registry.");
        }
        long freed = 0;
        foreach (ChunkKey k in victims)
        {
            try { freed += new FileInfo(store.PathOf(k)).Length; }
            catch (IOException) { }
        }

        int tempRemoved = 0;
        if (apply)
        {
            if (dropIndexes.Count > 0)
            {
                foreach (string f in dropIndexes)
                {
                    string p = Path.Combine(storeRoot, f);
                    if (File.Exists(p)) File.Delete(p);
                }
                foreach (string f in dropIndexes) layout.Registry.Remove(f);
                layout.Save(held!);
            }
            foreach (ChunkKey k in victims) store.Evict(k);
            tempRemoved = store.SweepTempFiles(TimeSpan.FromHours(1));
        }

        var result = new EvictResult
        {
            Evicted = victims.Count, FreedBytes = freed, KeptReferenced = all.Count - victims.Count - unreadable, DryRun = !apply, UnreadableEntries = unreadable,
            Mode = model is null ? "orphans" : "all", Keys = victims.Select(k => k.Hex).ToList(), DroppedIndexes = dropIndexes,
            TempFilesRemoved = tempRemoved, StoreEntriesAfter = apply ? store.Keys().Count() : all.Count,
        };
        if (ctx.Json)
        {
            ctx.WriteJson(result);
            return ExitCodes.Ok;
        }
        TextWriter o = ctx.Out;
        string verb = apply ? "evicted" : "would evict";
        o.WriteLine($"{(apply ? "" : "dry run: ")}{verb} {result.Evicted} {(model is null ? "orphaned" : "model")} entries ({Common.Bytes(freed)}); "
            + $"kept {result.KeptReferenced} entries{(model is null ? " referenced by an index or the server ledger" : " of other models")}"
            + (unreadable > 0 ? $" and {unreadable} unreadable entries" : ""));
        foreach (string k in result.Keys.Take(show)) o.WriteLine($"  {k}");
        if (result.Keys.Count > show) o.WriteLine($"  ... {result.Keys.Count - show} more (--show N)");
        foreach (string f in dropIndexes) o.WriteLine($"  {(apply ? "dropped" : "would drop")} index {f}");
        if (!apply) o.WriteLine("run with --apply to delete");
        return ExitCodes.Ok;
    }
}
