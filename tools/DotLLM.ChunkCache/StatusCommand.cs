using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Store;

namespace DotLLM.ChunkCache;

/// <summary>A missing chunk (<c>chunkcache_status.missing[]</c>).</summary>
internal sealed record MissingJson(string Path, int StartLine, int EndLine, int Tokens);

/// <summary>
/// The store's server ledger (<see cref="ServerLedger"/>) as <c>status</c> sees it. <c>State</c>: <c>absent</c> (no ledger
/// file), <c>ok</c>, or <c>unreadable</c> (orphan eviction refuses until it is repaired; <c>Error</c> says why). Namespace
/// values are never reported.
/// </summary>
internal sealed record LedgerJson(string State, int? Rows, int? RowsWithoutEntry, int? RowsUndetermined, string? Error);

/// <summary>Result of <c>status</c>: the <c>chunkcache_status</c> output schema plus extras.</summary>
internal sealed record StatusResult
{
    public int TotalChunks { get; init; }
    public int MissingChunks { get; init; }
    public long MissingTokens { get; init; }
    public bool UpToDate { get; init; }
    public IReadOnlyList<string> StaleFiles { get; init; } = [];
    public IReadOnlyList<string> NewFiles { get; init; } = [];
    public IReadOnlyList<string> DeletedFiles { get; init; } = [];
    public IReadOnlyList<MissingJson>? Missing { get; init; }
    public long StoreBytes { get; init; }
    public long? BudgetBytes { get; init; }
    public int Entries { get; init; }

    public int TempFiles { get; init; }
    public long TempBytes { get; init; }
    public bool Scanned { get; init; }
    public string? IndexPath { get; init; }
    public string? IndexLoadNote { get; init; }
    public string? Root { get; init; }
    public string? Chunker { get; init; }
    public LedgerJson? Ledger { get; init; }
}

/// <summary>
/// <c>chunkcache status</c>: which indexed chunks the store lacks (by file and line range) and, after a rescan of the tree
/// (unless <c>--no-scan</c>), which files are stale, new or deleted since the last build; plus store statistics. Read-only:
/// takes no lock.
/// </summary>
internal static class StatusCommand
{
    public static readonly IReadOnlyDictionary<string, OptionKind> Spec = Common.Merge(Common.ModelOptions, Common.SourceOptions, Common.ChunkerOptions,
        new Dictionary<string, OptionKind>
        {
            ["store"] = OptionKind.Single,
            ["no-scan"] = OptionKind.Flag,
            ["strict"] = OptionKind.Flag,
            ["list-missing"] = OptionKind.Flag,
            ["show"] = OptionKind.Single,
        });

    public const string Help = """
        chunkcache status --model M.gguf --store DIR [options]

          Lists the chunks of the store's index (this model, this chunker) that are missing from the store, by file and line
          range, and, after rescanning the tree the index records (or --solution/--directory), the stale, new and deleted files.

          --no-scan          check the store against the index only (edits since the last build are not seen)
          --strict           exit 3 when not up to date (CI)
          --list-missing     include every missing chunk in the JSON (text output always lists up to --show, default 40)
          --chunker symbol|fixed --chunk-tokens N --configuration C --json
        """;

    public static int Run(CliContext ctx, CliArgs args)
    {
        string storeRoot = ctx.FullPath(args.Required("store"));
        string chunkerName = Common.CheckChunkerName(args, args.Value("chunker") ?? "symbol");
        int show = args.Int("show", 40, 0);
        bool noScan = args.Flag("no-scan");
        if (noScan && SourceTree.SpecFrom(ctx, args) is not null) throw new UsageException("status: --no-scan takes no source.");
        if (!Directory.Exists(storeRoot)) throw new DirectoryNotFoundException($"Store not found: {storeRoot}");

        using ModelHandle model = Common.OpenModel(ctx, args);
        IChunker chunker = Common.Chunker(chunkerName, args, model.Tokenizer);
        StoreLayout layout = StoreLayout.Open(storeRoot, create: false);
        string indexPath = layout.ReadPath(chunker.Name, model);
        IndexRecord? record = layout.Find(chunker.Name, model);
        if (record is null && !File.Exists(indexPath))
            throw new InvalidOperationException($"The store has no {chunker.Name} index for this model ({ModelHandle.Short(model.ModelId)}); run build first.");
        ChunkIndex index = ChunkIndex.Load(indexPath, model.KeyOf);
        if (index.LoadNote is not null) ctx.Info($"index note: {index.LoadNote}");

        SourceTree? tree = null;
        if (!noScan)
        {
            tree = SourceTree.Load(ctx, Common.SourceFor(ctx, args, record), model.Tokenizer);
            ctx.Info(tree.Summary);
            if (record is not null && !ChunkIndex.PathComparer.Equals(Path.TrimEndingDirectorySeparator(record.Root), Path.TrimEndingDirectorySeparator(tree.Root)))
                ctx.Info($"warning: the index paths are relative to {record.Root}; the scanned tree's root is {tree.Root}");
        }
        StatusReport report = index.Status(layout.Existing, tree?.Files);
        LedgerJson ledger = ReadLedger(storeRoot);
        StoreStats stats = layout.Stats;
        var missing = report.Files.SelectMany(f => f.Missing).Select(r => new MissingJson(r.FilePath, r.StartLine, r.EndLine, r.TokenCount ?? 0)).ToList();
        string[] Of(FileState s) => report.Files.Where(f => f.State == s).Select(f => f.Path).ToArray();
        var result = new StatusResult
        {
            TotalChunks = report.TotalChunks, MissingChunks = report.MissingChunks, MissingTokens = report.MissingTokens, UpToDate = report.IsUpToDate,
            StaleFiles = Of(FileState.Stale), NewFiles = Of(FileState.New), DeletedFiles = Of(FileState.Deleted),
            Missing = args.Flag("list-missing") || !ctx.Json ? missing : null, StoreBytes = stats.EntryBytes, Entries = stats.Entries,
            TempFiles = stats.TempFiles, TempBytes = stats.TempBytes, Scanned = tree is not null, IndexPath = indexPath, IndexLoadNote = index.LoadNote,
            Root = record?.Root, Chunker = chunker.Name, Ledger = ledger,
        };

        if (ctx.Json) ctx.WriteJson(result);
        else
        {
            TextWriter o = ctx.Out;
            o.WriteLine($"{chunker.Name} index {Path.GetFileName(indexPath)}: {result.TotalChunks} chunks, {result.MissingChunks} missing ({result.MissingTokens} tokens); "
                + (result.UpToDate ? "up to date" : "NOT up to date") + (tree is null ? " (store only; --no-scan)" : ""));
            foreach (string f in result.StaleFiles) o.WriteLine($"  stale    {f}");
            foreach (string f in result.NewFiles) o.WriteLine($"  new      {f}");
            foreach (string f in result.DeletedFiles) o.WriteLine($"  deleted  {f}");
            foreach (MissingJson m in missing.Take(show)) o.WriteLine($"  missing  {m.Path}:{m.StartLine}-{m.EndLine} ({m.Tokens} tokens)");
            if (missing.Count > show) o.WriteLine($"  ... {missing.Count - show} more missing chunks (--show N)");
            o.WriteLine($"store {storeRoot}: {result.Entries} entries, {Common.Bytes(result.StoreBytes)}"
                + (result.TempFiles > 0 ? $"; {result.TempFiles} leftover temp files ({Common.Bytes(result.TempBytes)})" : ""));
            o.WriteLine(ledger.State switch
            {
                "ok" => $"server ledger: {ledger.Rows} entries" + (ledger.RowsWithoutEntry > 0 ? $" ({ledger.RowsWithoutEntry} rows without an entry file, dropped on the next server save)" : "")
                    + (ledger.RowsUndetermined > 0 ? $" ({ledger.RowsUndetermined} rows whose entry file could not be checked; kept)" : ""),
                "absent" => "server ledger: none",
                _ => $"server ledger UNREADABLE: {ledger.Error} (evict refuses until it is repaired)",
            });
        }
        if (args.Flag("strict") && ledger.State == "unreadable")
        {
            ctx.Info("strict: the server ledger is unreadable; orphan eviction refuses until it is repaired.");
            return ExitCodes.StrictFailure;
        }
        if (args.Flag("strict") && !report.IsUpToDate)
        {
            ctx.Info("strict: the store is not up to date for this tree.");
            return ExitCodes.StrictFailure;
        }
        return ExitCodes.Ok;
    }

    /// <summary>Reads the ledger for reporting only: an unreadable ledger is a state here, not an error (status is read-only).</summary>
    private static LedgerJson ReadLedger(string storeRoot)
    {
        try
        {
            if (!Path.Exists(ServerLedger.DefaultPath(storeRoot))) return new LedgerJson("absent", null, null, null, null);
            ServerLedger ledger = ServerLedger.Load(storeRoot); // tri-state: an unprobeable or directory path is unreadable
            return new LedgerJson("ok", ledger.Rows.Count, ledger.DroppedOnLoad, ledger.UndeterminedOnLoad, null);
        }
        catch (LedgerUnreadableException ex)
        {
            return new LedgerJson("unreadable", null, null, null, ex.Message);
        }
    }
}
