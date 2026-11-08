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

    /// <summary>All-for-model mode: entries whose header could not be read (damaged or partial); kept.</summary>
    public int UnreadableEntries { get; init; }
    public IReadOnlyList<string> Keys { get; init; } = [];
    public IReadOnlyList<string> DroppedIndexes { get; init; } = [];
    public int TempFilesRemoved { get; init; }
    public int StoreEntriesAfter { get; init; }
}

/// <summary>
/// <c>chunkcache evict</c>: by default a dry run that lists the store-wide orphans (entries that no <c>index-*.json</c> of the
/// store and not the server ledger reference: <see cref="ChunkIndex.StoreWideOrphans"/>, a mark-and-sweep); <c>--apply</c> deletes them while holding
/// the exclusive store lock for the whole pass (one eviction pass, integration spec L3), so no put interleaves; a build publishes its index before relying on stored entries (ChunkIndex remarks). <c>--all-for-model</c>
/// removes every entry written for one model (by the fingerprint in each entry's header) and that model's index files.
/// </summary>
internal static class EvictCommand
{
    public static readonly IReadOnlyDictionary<string, OptionKind> Spec = Common.Merge(Common.ModelOptions, new Dictionary<string, OptionKind>
    {
        ["store"] = OptionKind.Single,
        ["apply"] = OptionKind.Flag,
        ["all-for-model"] = OptionKind.Flag,
        ["lock-timeout"] = OptionKind.Single,
        ["show"] = OptionKind.Single,
    });

    public const string Help = """
        chunkcache evict --store DIR [--apply] [--all-for-model --model M.gguf] [options]

          Default: a dry run listing the store-wide orphans: entries that no index of the store (any chunker, any model)
          and not the server ledger (ledger-server.json, entries a server wrote) reference. An unreadable index-*.json or
          ledger blocks eviction (exit 1): nothing is guessed.
          --apply            delete them in one pass holding the exclusive store lock (puts wait; --lock-timeout S, default 30)
          --all-for-model    every entry written for --model (by the fingerprint in its header) and that model's index files
          --strict-model-id  identify --model by a full hash (must match how it was built)
          --json --show N (keys to print, default 20)
        """;

    public static int Run(CliContext ctx, CliArgs args)
    {
        string storeRoot = ctx.FullPath(args.Required("store"));
        bool apply = args.Flag("apply"), allForModel = args.Flag("all-for-model");
        if (allForModel && args.Value("model") is null) throw new UsageException("evict: --all-for-model needs --model.");
        if (!allForModel && (args.Value("model") is not null || args.Flag("strict-model-id") || args.Value("threads") is not null))
            throw new UsageException("evict: --model, --strict-model-id and --threads are used only with --all-for-model.");
        TimeSpan lockTimeout = args.Seconds("lock-timeout", Common.DefaultLockTimeout);
        int show = args.Int("show", 20, 0);
        if (!Directory.Exists(storeRoot)) throw new DirectoryNotFoundException($"Store not found: {storeRoot}");

        using ModelHandle? model = allForModel ? Common.OpenModel(ctx, args) : null;
        using StoreLockHandle? held = apply ? Common.Lock(ctx, storeRoot, lockTimeout) : null;
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
            Mode = model is null ? "orphans" : "all_for_model", Keys = victims.Select(k => k.Hex).ToList(), DroppedIndexes = dropIndexes,
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
