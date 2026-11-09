using System.Diagnostics;
using DotLLM.ChunkCache.Client;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;
using DotLLM.ChunkReuse.Store;
using DotLLM.ChunkReuse.Tooling;

namespace DotLLM.ChunkCache;

/// <summary>A skipped project in the JSON result (<c>chunkcache_build.skipped_projects</c>).</summary>
internal sealed record SkippedJson(string Project, string Reason, string Message, bool FailsStrict);

/// <summary>A chunk listed by <c>build --dry-run</c>.</summary>
internal sealed record PlannedChunk(string Path, int StartLine, int EndLine, int Tokens, string Key);

/// <summary>
/// Result of <c>build</c>. The first fields are the <c>chunkcache_build</c> MCP output schema (integration spec §7.2); the rest
/// are CLI extras.
/// </summary>
internal sealed record BuildResult
{
    public int Files { get; init; }
    public int Chunks { get; init; }
    public long Tokens { get; init; }
    public int? WireIneligibleChunks { get; init; }
    public int DeniedFiles { get; init; }
    public int Added { get; init; }
    public int Removed { get; init; }
    public int Unchanged { get; init; }
    public long EncodeQueuedTokens { get; init; }
    public IReadOnlyList<SkippedJson> SkippedProjects { get; init; } = [];
    public long ElapsedMs { get; init; }

    public bool DryRun { get; init; }
    public bool StrictFailed { get; init; }
    public string? Store { get; init; }
    public string? Root { get; init; }
    public string? IndexPath { get; init; }
    public string? Chunker { get; init; }
    public string? ModelId { get; init; }
    public string? Fingerprint { get; init; }
    public string? IndexLoadNote { get; init; }
    public int DistinctKeys { get; init; }
    public long DistinctTokens { get; init; }
    public int AlreadyStored { get; init; }
    public int SelectedForEncoding { get; init; }
    public int ToEncodeChunks { get; init; }
    public int EncodedChunks { get; init; }
    public long EncodedTokens { get; init; }
    public int StoreHits { get; init; }
    public int StoreFailures { get; init; }
    public double EncodeMs { get; init; }
    public double TokensPerSecond { get; init; }
    public int MissingAfter { get; init; }
    public long MissingTokensAfter { get; init; }
    public long EstimatedBytes { get; init; }
    public long EstimatedFullStoreBytes { get; init; }
    public double BytesPerTokenPerLayer { get; init; }
    public int Layers { get; init; }
    public int StoreEntries { get; init; }
    public long StoreBytes { get; init; }
    public double ScanMs { get; init; }
    public double IndexMs { get; init; }
    public IReadOnlyList<PlannedChunk>? WouldEncode { get; init; }

    /// <summary>How missing chunks were handled: <c>prewarm</c> (queued on the server), <c>local</c> (<c>--offline</c>: encoded here),
    /// <c>index_only</c> or <c>dry_run</c> (nothing encoded).</summary>
    public string? Mode { get; init; }

    /// <summary>The server of a server-backed run (no user info).</summary>
    public string? Server { get; init; }

    /// <summary>The server's answers to the prewarm requests, summed.</summary>
    public PrewarmJson? Prewarm { get; init; }
}

/// <summary>
/// <c>chunkcache build</c>: scan a solution or directory, chunk it, update the store's index for this model and chunker, then
/// encode (CPU) and store every indexed chunk the store lacks, or only those a selection picks (<c>--path</c>,
/// <c>--symbol</c>, <c>--since</c>), at most <c>--max-chunks</c> of them.
/// </summary>
/// <remarks>
/// Encoding is driven from <see cref="ChunkIndex.Status"/> (everything missing, so an interrupted build resumes), not from the
/// diff's added chunks. Locks (integration spec L3): the index lock from the index load through its save and the publication
/// check; the store lock only around the registry save plus that check, then around each put (<see cref="LockingChunkStore"/>),
/// never across an encode. The index is saved before encoding starts, so an interrupted build leaves a correct index and a
/// partly filled store. Memory is bounded by one chunk's K/V: files are re-chunked one at a time
/// and each chunk is encoded, stored and dropped.
/// </remarks>
internal static class BuildCommand
{
    /// <summary>Test hook, per async flow: runs once the offline writer holds the store owner lock, before anything is read or written.</summary>
    internal static Action? TestHookAfterOwnerLock
    {
        get => s_afterOwnerLock.Value;
        set => s_afterOwnerLock.Value = value;
    }

    private static readonly AsyncLocal<Action?> s_afterOwnerLock = new();

    /// <summary>Test hook: runs after the index is saved (index lock held, store lock not), before the registry save and the
    /// publication check.</summary>
    internal static Action? TestHookAfterIndexSave { get; set; }

    /// <summary>Test hook: runs after the publication check, with no lock held, before the first encode.</summary>
    internal static Action? TestHookBeforeEncode { get; set; }

    /// <summary>How the missing chunks are handled.</summary>
    private enum EncodeMode
    {
        /// <summary>Nothing is encoded (<c>--index-only</c>, <c>--dry-run</c>).</summary>
        None,

        /// <summary>Queued on the server (<c>--server</c>): text through <c>prewarm</c>, never an entry write.</summary>
        Prewarm,

        /// <summary>Encoded here (<c>--offline</c>), holding the store owner lock for the whole run.</summary>
        Local,
    }

    public static int Run(CliContext ctx, CliArgs args)
    {
        var total = Stopwatch.StartNew();
        string storeRoot = ctx.FullPath(args.Required("store"));
        SourceSpec spec = SourceTree.SpecFrom(ctx, args) ?? throw new UsageException("build: give --solution or --directory.");
        string chunkerName = Common.ChunkerName(args);
        bool dryRun = args.Flag("dry-run"), strict = args.Flag("strict"), lab = args.Flag("lab");
        int? maxChunks = args.IntOrNull("max-chunks", 0);
        int? maxEncodeTokens = args.IntOrNull("max-encode-tokens", 0);
        int show = args.Int("show", 50, 0);
        TimeSpan lockTimeout = args.Seconds("lock-timeout", Common.DefaultLockTimeout);
        _ = args.IntOrNull("chunk-tokens", 1, 1 << 20);

        string? serverUrl = args.Value("server");
        bool offline = args.Flag("offline"), indexOnly = args.Flag("index-only");
        if (args.Flag("encode") && indexOnly) throw new UsageException("build: --encode and --index-only contradict each other.");
        if (offline && serverUrl is not null) throw new UsageException("build: give --server (the server encodes) or --offline (this machine encodes), not both.");
        EncodeMode mode = indexOnly || dryRun ? EncodeMode.None
            : serverUrl is not null ? EncodeMode.Prewarm
            : offline ? EncodeMode.Local
            : throw new UsageException("build: say who encodes: --server URL (queue the chunks on the server), --offline (encode into --store here) or --index-only (update the index only).");
        if (!lab && !WireChunking.IsWireChunker(chunkerName))
            throw new UsageException($"build: chunker '{chunkerName}' is not wire-eligible (the wire carries {string.Join(", ", WireChunking.Eligible)} only); "
                + "pass --lab for a research build of a store that never leaves this machine.");

        // Read-only preparation first (a bad --model or an unreachable server leaves the store untouched).
        using ModelHandle model = Common.OpenModel(ctx, args);
        using ServerConnection? server = Common.ConnectServer(ctx, args, model);

        IChunker chunker = Common.Chunker(args, model.Tokenizer);
        SelectionRequest? selection = Common.HasSelection(args) ? Common.Selection(args, chunker) : null;
        if (selection is null && (args.List("exclude").Count > 0 || args.Value("budget") is not null || args.Value("max-per-hop") is not null))
            throw new UsageException("build: --exclude, --budget and --max-per-hop refine a selection; give --path, --symbol or --since too.");

        var sw = Stopwatch.StartNew();
        SourceTree tree = SourceTree.Load(ctx, spec, model.Tokenizer);
        double scanMs = sw.Elapsed.TotalMilliseconds;
        ctx.Info(tree.Summary);
        foreach (SkippedEntry s in tree.Skipped)
            ctx.Info($"{(s.FailsStrict ? "warning" : "note")}: skipped {s.Project} ({s.Reason}): {FirstLine(s.Message)}");
        var skipped = tree.Skipped.Select(s => new SkippedJson(s.Project, s.Reason, s.Message, s.FailsStrict)).ToList();
        string modeName = dryRun ? "dry_run" : mode switch { EncodeMode.Prewarm => "prewarm", EncodeMode.Local => "local", _ => "index_only" };

        if (strict && tree.FailsStrict)
        {
            var failed = new BuildResult
            {
                Files = tree.Files.Count, DeniedFiles = tree.Denied.Count, SkippedProjects = skipped, StrictFailed = true, DryRun = dryRun,
                Store = storeRoot, Root = tree.Root, Chunker = chunker.Name, ModelId = model.ModelId, Fingerprint = model.Fingerprint,
                ElapsedMs = total.ElapsedMilliseconds, ScanMs = scanMs, Mode = modeName, Server = server?.BaseUrl,
            };
            Emit(ctx, failed, show);
            ctx.Info($"strict: {tree.Skipped.Count(s => s.FailsStrict)} skipped project(s) or excluded file(s); nothing was written.");
            return ExitCodes.StrictFailure;
        }

        // Wire eligibility of every chunk of the tree: allow-listed chunker AND not a token-id slice AND the text tokenised alone
        // (ChunkTokenization.Encode) gives the chunk's ids. One file at a time: only counts are kept.
        sw.Restart();
        int wireIneligible = 0;
        foreach (SourceFile f in tree.Files)
            wireIneligible += WireEligibility.CountIneligible(chunker.Name, chunker.Chunk(f), c => f.Text[c.StartOffset..c.EndOffset], model.Tokenizer);
        ctx.Info($"wire eligibility: {wireIneligible} chunk(s) cannot travel on the wire [{sw.Elapsed.TotalSeconds:F1} s]");

        // The selection (chunking, Roslyn graph) depends only on the tree, not on the index: compute it before taking the lock.
        HashSet<string>? selectedIds = null;
        if (selection is not null)
        {
            sw.Restart();
            Chunk[] chunks = tree.Files.SelectMany(chunker.Chunk).ToArray();
            ReferenceGraph graph = tree.BuildGraph(model.Tokenizer, chunks);
            SelectionResult sel = Common.Select(selection, tree, graph);
            selectedIds = sel.Select(s => s.Chunk.Id).ToHashSet(StringComparer.Ordinal);
            ctx.Info($"selection: {sel.Count} chunks / {sel.SelectedTokens} tokens selected ({sel.Trimmed.Count} trimmed) [{sw.Elapsed.TotalSeconds:F1} s]");
            foreach (string n in sel.Notes) ctx.Info($"  note: {n}");
            foreach (string u in sel.UnmatchedPatterns) ctx.Info($"  unmatched pattern: {u}");
        }

        // The offline writer owns the store for its WHOLE run: taken before the first write (the scan and --strict above write nothing)
        // and released at the end. (StoreOwnerLock.IsHeldByAnother would be a check followed by writes: advisory only, issue #1664.)
        using StoreOwnerLock? owner = mode == EncodeMode.Local ? Common.AcquireOwner(ctx, storeRoot) : null;
        if (owner is not null) TestHookAfterOwnerLock?.Invoke();

        // Lock model (integration spec L3; ChunkIndex remarks): the index lock is held from the index load to the publication
        // check; the store lock only for the registry save plus that check, and then per put. A dry run writes nothing, takes no
        // lock and does not create the store directory.
        StoreLayout layout = StoreLayout.Open(storeRoot, create: !dryRun);
        string indexPath = layout.IndexPath(chunker.Name, model); // index-<chunker>.<model12>.<p8>.default.json (L9)
        IDisposable? indexHeld = dryRun ? null : Common.IndexLock(ctx, indexPath, lockTimeout);
        ChunkIndex index;
        IndexDiff diff;
        StatusReport status;
        double indexMs;
        try
        {
            string loadPath = indexPath;
            if (dryRun)
            {
                loadPath = layout.ReadPath(chunker.Name, model); // a legacy index is read in place, never renamed
            }
            else
            {
                // A pre-L9 index of this producer is renamed to the current name once (under the store lock, briefly).
                IndexMigration migration = ChunkIndex.MigrateLegacyIndex(storeRoot, chunker.Name, model.ModelId, ChunkNamespace.Default, model.KeyOf, lockTimeout);
                if (migration.MigratedFrom is not null) ctx.Info($"index note: migrated {Path.GetFileName(migration.MigratedFrom)} to {Path.GetFileName(indexPath)}");
                foreach (string left in migration.LeftBehind) ctx.Info($"index note: legacy index left in place: {left}");
                layout = StoreLayout.Open(storeRoot, create: true); // the registry as of now, under the index lock
                // The name carries 32 bits of the producer: a prefix collision with another producer's index is refused, never
                // overwritten (another producer otherwise writes its own file).
                string? onDisk = ChunkIndex.ProducerOf(indexPath);
                if (onDisk is not null && !string.Equals(onDisk, model.Producer, StringComparison.Ordinal))
                    throw new InvalidOperationException($"Index '{Path.GetFileName(indexPath)}' was written by another producer whose identity shares its first 8 hex digits; "
                        + "it is not overwritten. Use another --store, or delete that index if it is no longer needed.");
            }
            IndexRecord? record = layout.Find(chunker.Name, model);
            if (record is not null && !ChunkIndex.PathComparer.Equals(Path.TrimEndingDirectorySeparator(record.Root), Path.TrimEndingDirectorySeparator(tree.Root)) && !args.Flag("reroot"))
                throw new InvalidOperationException($"The store's {chunker.Name} index for this model tracks the tree at '{record.Root}', not '{tree.Root}'. "
                    + "Use another --store for another tree, or --reroot to move the index (every file is re-pathed; nothing is re-encoded).");

            sw.Restart();
            index = ChunkIndex.Load(loadPath, model.KeyOf);
            if (index.LoadNote is not null) ctx.Info($"index note: {index.LoadNote}");
            diff = index.Update(tree.Files, chunker);
            if (dryRun)
            {
                status = index.Status(layout.Entries);
            }
            else
            {
                // Publish the index first (index lock only), then, in one short store-lock scope, record it in the registry
                // (read-modify-write, so other writers' rows survive) and decide what is already stored: an eviction pass that
                // read the old index has finished by then, and every later pass reads this one.
                index.Save(indexPath);
                TestHookAfterIndexSave?.Invoke();
                using (StoreLockHandle storeLock = Common.Lock(ctx, storeRoot, lockTimeout))
                {
                    StoreLayout registry = StoreLayout.Open(storeRoot, create: true);
                    registry.Upsert(new IndexRecord
                    {
                        IndexFile = Path.GetFileName(indexPath), Chunker = chunker.Name, Producer = model.Producer, ModelFingerprint = model.Fingerprint,
                        ModelId = model.ModelId, TokenizerId = model.TokenizerId, Backend = model.Assembler.Backend, ModelPath = model.Path,
                        SourceKind = spec.Kind, SourcePath = spec.Path, Configuration = spec.Configuration, Root = tree.Root, UpdatedUtc = DateTime.UtcNow,
                    });
                    registry.Save(storeLock);
                    status = index.Status(layout.Entries);
                }
            }
            indexMs = sw.Elapsed.TotalMilliseconds;
        }
        finally
        {
            indexHeld?.Dispose(); // encoding needs no index lock: the index is saved
        }

        // Everything the store lacks for this index, one chunk per distinct key, in path then text order.
        var all = index.Paths.SelectMany(index.ChunksOf).ToList();
        long tokens = all.Sum(r => (long)(r.TokenCount ?? 0));
        var distinct = all.DistinctBy(r => r.KeyHex, StringComparer.Ordinal).ToList();
        long distinctTokens = distinct.Sum(r => (long)(r.TokenCount ?? 0));
        // Missing chunks in path then text order; restricted to the selection FIRST, then one per distinct key, so a selected
        // chunk whose text also occurs in an unselected (earlier) file is still encoded.
        IEnumerable<ChunkRef> candidates = status.Files.SelectMany(f => f.Missing);
        int missingDistinct = candidates.Select(r => r.KeyHex).Distinct(StringComparer.Ordinal).Count();
        if (selectedIds is not null) candidates = candidates.Where(r => selectedIds.Contains(r.Id));
        var seen = new HashSet<string>(StringComparer.Ordinal);
        List<ChunkRef> missing = candidates.Where(r => seen.Add(r.KeyHex!)).ToList();
        int alreadyStored = distinct.Count - missingDistinct;
        ctx.Info($"index {Path.GetFileName(indexPath)}: {tree.Files.Count} files, {all.Count} chunks ({distinct.Count} distinct keys, {tokens} tokens); "
            + $"added {diff.Added.Count}, removed {diff.Removed.Count}, unchanged {diff.Unchanged.Count}; {missingDistinct} distinct chunks missing from the store"
            + (selectedIds is null ? "" : $", {missing.Count} of them selected") + $" [{indexMs / 1000:F1} s]");
        int selectedCount = missing.Count;
        if (maxChunks is int cap && missing.Count > cap) missing = missing.Take(cap).ToList();
        if (maxEncodeTokens is int tokenCap) missing = WithinTokenBudget(missing, tokenCap);
        long toEncodeTokens = missing.Sum(r => (long)(r.TokenCount ?? 0));
        long estimated = EstimateBytes(model, missing.Select(r => r.TokenCount ?? 0));
        long estimatedFull = EstimateBytes(model, distinct.Select(r => r.TokenCount ?? 0));
        ctx.Info($"to encode: {missing.Count} chunks, {toEncodeTokens} tokens, about {Common.Bytes(estimated)} "
            + $"({model.KvBytesPerTokenPerLayer:0} B/token/layer x {model.Geometry.LayerCount} layers, F32); the whole tree would need about {Common.Bytes(estimatedFull)}");

        var result = new BuildResult
        {
            Files = tree.Files.Count, Chunks = all.Count, Tokens = tokens, DeniedFiles = tree.Denied.Count, WireIneligibleChunks = wireIneligible,
            Added = diff.Added.Count, Removed = diff.Removed.Count, Unchanged = diff.Unchanged.Count, EncodeQueuedTokens = mode == EncodeMode.None ? 0 : toEncodeTokens,
            SkippedProjects = skipped, DryRun = dryRun, Store = storeRoot, Root = tree.Root, IndexPath = indexPath, Chunker = chunker.Name,
            ModelId = model.ModelId, Fingerprint = model.Fingerprint, IndexLoadNote = index.LoadNote, DistinctKeys = distinct.Count,
            DistinctTokens = distinctTokens, AlreadyStored = alreadyStored, SelectedForEncoding = selectedCount, ToEncodeChunks = missing.Count,
            EstimatedBytes = estimated, EstimatedFullStoreBytes = estimatedFull, BytesPerTokenPerLayer = model.KvBytesPerTokenPerLayer,
            Layers = model.Geometry.LayerCount, ScanMs = scanMs, IndexMs = indexMs, Mode = modeName, Server = server?.BaseUrl,
        };

        if (dryRun)
        {
            result = result with
            {
                WouldEncode = missing.Select(r => new PlannedChunk(r.FilePath, r.StartLine, r.EndLine, r.TokenCount ?? 0, r.KeyHex!)).ToList(),
                MissingAfter = status.MissingChunks, MissingTokensAfter = status.MissingTokens, ElapsedMs = total.ElapsedMilliseconds,
            };
            Emit(ctx, result, show);
            return ExitCodes.Ok;
        }

        var files = tree.Files.ToDictionary(f => f.Path, ChunkIndex.PathComparer);
        if (mode == EncodeMode.None)
        {
            result = result with { MissingAfter = status.MissingChunks, MissingTokensAfter = status.MissingTokens, ElapsedMs = total.ElapsedMilliseconds };
            Emit(ctx, result, show);
            return ExitCodes.Ok;
        }

        if (mode == EncodeMode.Prewarm)
        {
            // The server encodes: text only (never a hash, never K/V), each text with its path so the server's deny-list also applies.
            TestHookBeforeEncode?.Invoke();
            PrewarmJson sent = Prewarm(ctx, server!, chunker, model, files, missing);
            result = result with
            {
                EncodeQueuedTokens = sent.QueuedTokens, Prewarm = sent, MissingAfter = status.MissingChunks, MissingTokensAfter = status.MissingTokens,
                ElapsedMs = total.ElapsedMilliseconds,
            };
            Emit(ctx, result, show);
            return ExitCodes.Ok;
        }

        // Encode here (--offline): one file at a time (re-chunked to recover the token ids), one chunk at a time.
        var lockedStore = new LockingChunkStore(layout.Existing, storeRoot, lockTimeout); // each put takes the store lock briefly
        TestHookBeforeEncode?.Invoke();
        int encoded = 0, hits = 0, failures = 0, done = 0;
        long encodedTokens = 0, doneTokens = 0;
        var encodeSw = Stopwatch.StartNew();
        var progress = Stopwatch.StartNew();
        foreach (IGrouping<string, ChunkRef> group in missing.GroupBy(r => r.FilePath, ChunkIndex.PathComparer))
        {
            SourceFile file = files[group.Key];
            Dictionary<string, Chunk> byId = chunker.Chunk(file).ToDictionary(c => c.Id, StringComparer.Ordinal);
            foreach (ChunkRef r in group)
            {
                if (!byId.TryGetValue(r.Id, out Chunk? chunk))
                    throw new InvalidOperationException($"Chunk {r.Id} is in the index but the chunker no longer produces it; the file changed during the build?");
                if (chunk.TokenIds.Length == 0) continue;
                var assembly = model.Assembler.Assemble([chunk], model.Model, lockedStore);
                encoded += assembly.Stats.Misses;
                hits += assembly.Stats.Hits;
                encodedTokens += assembly.Stats.EncodedTokens;
                failures += assembly.StoreFailures;
                done++;
                doneTokens += chunk.TokenIds.Length;
                if (progress.Elapsed.TotalSeconds >= 15)
                {
                    progress.Restart();
                    double rate = encodedTokens / Math.Max(encodeSw.Elapsed.TotalSeconds, 1e-9);
                    long left = toEncodeTokens - doneTokens;
                    ctx.Info($"  encoded {done}/{missing.Count} chunks, {encodedTokens} tokens, {rate:F0} tokens/s, ~{left / Math.Max(rate, 1e-9) / 60:F1} min left");
                }
            }
        }
        double encodeMs = encodeSw.Elapsed.TotalMilliseconds;
        StatusReport after = index.Status(layout.Entries);
        StoreStats stats = layout.Stats;
        result = result with
        {
            EncodedChunks = encoded, EncodedTokens = encodedTokens, StoreHits = hits, StoreFailures = failures,
            EncodeMs = encodeMs, TokensPerSecond = encodedTokens / Math.Max(encodeMs / 1000, 1e-9), MissingAfter = after.MissingChunks,
            MissingTokensAfter = after.MissingTokens, StoreEntries = stats.Entries, StoreBytes = stats.EntryBytes, ElapsedMs = total.ElapsedMilliseconds,
        };
        Emit(ctx, result, show);
        if (failures > 0)
        {
            ctx.Info($"error: {failures} encoded chunk(s) could not be written to the store (busy or damaged entries); rerun build to repair.");
            return ExitCodes.Error;
        }
        return ExitCodes.Ok;
    }

    /// <summary>The longest prefix of <paramref name="missing"/> whose token counts sum to at most <paramref name="budget"/>.</summary>
    internal static List<ChunkRef> WithinTokenBudget(List<ChunkRef> missing, int budget)
    {
        var kept = new List<ChunkRef>();
        long sum = 0;
        foreach (ChunkRef r in missing)
        {
            sum += r.TokenCount ?? 0;
            if (sum > budget) break;
            kept.Add(r);
        }
        return kept;
    }

    // Queues the wire-eligible chunks of `missing` on the server, in batches. Every text is the chunk's text exactly as the
    // wire hash is taken over it (LF as scanned). Nothing else is sent: no hash, no K/V, no token ids.
    private static PrewarmJson Prewarm(CliContext ctx, ServerConnection server, IChunker chunker, ModelHandle model,
        Dictionary<string, SourceFile> files, List<ChunkRef> missing)
    {
        var sender = new PrewarmSender(server);
        foreach (IGrouping<string, ChunkRef> group in missing.GroupBy(r => r.FilePath, ChunkIndex.PathComparer))
        {
            SourceFile file = files[group.Key];
            Dictionary<string, Chunk> byId = chunker.Chunk(file).ToDictionary(c => c.Id, StringComparer.Ordinal);
            foreach (ChunkRef r in group)
            {
                if (!byId.TryGetValue(r.Id, out Chunk? chunk))
                    throw new InvalidOperationException($"Chunk {r.Id} is in the index but the chunker no longer produces it; the file changed during the build?");
                string text = file.Text[chunk.StartOffset..chunk.EndOffset];
                if (chunk.TokenIds.Length == 0 || !WireEligibility.IsEligible(chunker.Name, chunk, text, model.Tokenizer))
                {
                    sender.SkipIneligible();
                    continue;
                }
                sender.Add(new PrewarmText { Id = chunk.Id, Text = text, Source = new PrewarmSource { Path = chunk.FilePath } });
            }
        }
        PrewarmJson sent = sender.Finish();
        ctx.Info($"prewarm: {sent.Requests} request(s) to {server.BaseUrl}: requested {sent.Requested}, already stored {sent.AlreadyStored}, queued {sent.Queued} ({sent.QueuedTokens} tokens), refused {sent.Refused}"
            + (sent.SkippedIneligible > 0 ? $"; {sent.SkippedIneligible} wire-ineligible chunk(s) not sent" : ""));
        return sent;
    }

    /// <summary>
    /// Store bytes for chunks of these token counts: the F32 K and V rows (<see cref="ModelHandle.KvBytesPerToken"/>, i.e.
    /// bytes/token/layer × layers; 8,192 × 36 for Qwen3-4B) plus the token ids, plus a header estimate per entry.
    /// </summary>
    public static long EstimateBytes(ModelHandle model, IEnumerable<int> tokenCounts)
    {
        long header = FileChunkStore.HeaderBytes(model.Fingerprint, model.Assembler.Backend, model.Geometry.LayerCount);
        long total = 0;
        foreach (int n in tokenCounts) total += header + n * (model.KvBytesPerToken + sizeof(int));
        return total;
    }

    private static void Emit(CliContext ctx, BuildResult r, int show)
    {
        if (ctx.Json)
        {
            ctx.WriteJson(r);
            return;
        }
        TextWriter o = ctx.Out;
        o.WriteLine($"build {(r.DryRun ? "(dry run) " : "")}{r.Chunker} into {r.Store}");
        o.WriteLine($"  tree: {r.Files} files, {r.Chunks} chunks, {r.Tokens} tokens ({r.DistinctKeys} distinct chunks, {r.DistinctTokens} tokens); root {r.Root}");
        if (r.StrictFailed)
        {
            o.WriteLine($"  strict failure: {r.SkippedProjects.Count(s => s.FailsStrict)} skipped project(s) or excluded file(s):");
            foreach (SkippedJson s in r.SkippedProjects.Where(s => s.FailsStrict)) o.WriteLine($"    {s.Project} ({s.Reason})");
            return;
        }
        o.WriteLine($"  index: added {r.Added}, removed {r.Removed}, unchanged {r.Unchanged}; already stored {r.AlreadyStored} of {r.DistinctKeys} distinct chunks");
        if (r.SkippedProjects.Count > 0) o.WriteLine($"  skipped: {string.Join(", ", r.SkippedProjects.Select(s => $"{s.Project} ({s.Reason})"))}");
        o.WriteLine($"  to encode: {r.ToEncodeChunks} chunks, {r.EncodeQueuedTokens} tokens, estimated {Common.Bytes(r.EstimatedBytes)} "
            + $"({r.BytesPerTokenPerLayer:0} B/token/layer x {r.Layers} layers); whole tree {Common.Bytes(r.EstimatedFullStoreBytes)}");
        if (r.DryRun)
        {
            foreach (PlannedChunk c in (r.WouldEncode ?? []).Take(show))
                o.WriteLine($"    {c.Path}:{c.StartLine}-{c.EndLine}  {c.Tokens} tokens  {c.Key[..12]}");
            if ((r.WouldEncode?.Count ?? 0) > show) o.WriteLine($"    ... {r.WouldEncode!.Count - show} more (--show N)");
            return;
        }
        if (r.Prewarm is { } pw)
        {
            o.WriteLine($"  prewarm via {r.Server}: requested {pw.Requested}, already stored {pw.AlreadyStored}, queued {pw.Queued} ({pw.QueuedTokens} tokens), refused {pw.Refused}"
                + (pw.SkippedIneligible > 0 ? $"; {pw.SkippedIneligible} wire-ineligible not sent" : ""));
            o.WriteLine($"  wire-ineligible chunks in the tree: {r.WireIneligibleChunks}; denied files: {r.DeniedFiles}");
            return;
        }
        if (r.Mode == "index_only")
        {
            o.WriteLine($"  index only (nothing encoded); wire-ineligible {r.WireIneligibleChunks}, denied files {r.DeniedFiles}; still missing {r.MissingAfter} chunks ({r.MissingTokensAfter} tokens)");
            return;
        }
        o.WriteLine($"  encoded: {r.EncodedChunks} chunks, {r.EncodedTokens} tokens in {r.EncodeMs / 1000:F1} s ({r.TokensPerSecond:F0} tokens/s, cpu); "
            + $"store hits {r.StoreHits}, store failures {r.StoreFailures}, wire-ineligible {r.WireIneligibleChunks}");
        o.WriteLine($"  store: {r.StoreEntries} entries, {Common.Bytes(r.StoreBytes)}; still missing {r.MissingAfter} chunks ({r.MissingTokensAfter} tokens)");
        o.WriteLine($"  time: scan {r.ScanMs / 1000:F1} s, index {r.IndexMs / 1000:F1} s, total {r.ElapsedMs / 1000.0:F1} s");
    }

    private static string FirstLine(string s)
    {
        int nl = s.IndexOf('\n', StringComparison.Ordinal);
        return nl < 0 ? s : s[..nl] + " ...";
    }
}
