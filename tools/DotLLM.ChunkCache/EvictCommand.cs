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
/// <c>chunkcache evict</c>: by default a dry run that lists the store-wide orphans (entries no <c>index-*.json</c> of the
/// store references: <see cref="ChunkIndex.StoreWideOrphans"/>, a mark-and-sweep); <c>--apply</c> deletes them while holding
/// the exclusive store lock, so no build can reference a key between the check and the delete. <c>--all-for-model</c>
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
          references. An unreadable index-*.json blocks eviction (exit 1): nothing is guessed.
          --apply            delete them, holding the exclusive store lock (builds wait; --lock-timeout S, default 30)
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
        using IDisposable? held = apply ? Common.Lock(ctx, storeRoot, lockTimeout) : null;
        StoreLayout layout = StoreLayout.Open(storeRoot, create: false);
        FileChunkStore store = layout.Existing;
        List<ChunkKey> all = store.Keys().ToList();

        IReadOnlyList<ChunkKey> victims;
        int unreadable = 0;
        var dropIndexes = new List<string>();
        if (model is null)
        {
            victims = ChunkIndex.StoreWideOrphans(store, storeRoot); // throws InvalidDataException on an unreadable index
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
            string suffix = "." + model.Producer[..12] + ".json";
            dropIndexes.AddRange(Directory.EnumerateFiles(storeRoot, "index-*" + suffix).Select(Path.GetFileName)!);
            dropIndexes.AddRange(layout.Records.Where(r => r.ModelFingerprint == model.Fingerprint).Select(r => r.IndexFile));
            dropIndexes = dropIndexes.Distinct(StringComparer.OrdinalIgnoreCase).Order(StringComparer.Ordinal).ToList();
            // The registry is data: a name that is not a plain index-*.json in the store root is refused before anything is
            // deleted (e.g. "..\\x", an absolute path, a subdirectory).
            string[] bad = dropIndexes.Where(f => !layout.IsIndexFileName(f)).ToArray();
            if (bad.Length > 0)
                throw new InvalidDataException($"Store config '{StoreLayout.ConfigFileName}' names index files outside the store or not of the form index-*.json: "
                    + string.Join(", ", bad.Select(b => $"'{b}'")) + ". Nothing was deleted; repair the config.");
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
                layout.Records.RemoveAll(r => dropIndexes.Contains(r.IndexFile, StringComparer.OrdinalIgnoreCase));
                layout.Save();
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
            + $"kept {result.KeptReferenced} entries{(model is null ? " referenced by an index" : " of other models")}"
            + (unreadable > 0 ? $" and {unreadable} unreadable entries" : ""));
        foreach (string k in result.Keys.Take(show)) o.WriteLine($"  {k}");
        if (result.Keys.Count > show) o.WriteLine($"  ... {result.Keys.Count - show} more (--show N)");
        foreach (string f in dropIndexes) o.WriteLine($"  {(apply ? "dropped" : "would drop")} index {f}");
        if (!apply) o.WriteLine("run with --apply to delete");
        return ExitCodes.Ok;
    }
}
