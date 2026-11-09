using System.Diagnostics;
using System.Text;
using System.Text.Json.Nodes;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;
using DotLLM.ChunkReuse.Store;
using DotLLM.ChunkReuse.Tooling;

namespace DotLLM.ChunkCache;

/// <summary>One selected chunk (<c>chunkcache_select.chunks[]</c>).</summary>
internal sealed record SelectedJson(string Id, string Path, int StartLine, int EndLine, int Tokens, string Hash, string Reason, string Detail,
    int? Hop, bool? Stored, string? Text)
{
    public string[] Reasons { get; init; } = [];
    public int Priority { get; init; }
    public string? Key { get; init; }
}

/// <summary>A symbol declaration found by the selection.</summary>
internal sealed record DefinitionJson(string QualifiedName, string Kind, string Path, int Line);

/// <summary>Result of <c>select</c>: the <c>chunkcache_select</c> output schema plus extras.</summary>
internal sealed record SelectResult
{
    public IReadOnlyList<SelectedJson> Chunks { get; init; } = [];
    public long SelectedTokens { get; init; }
    public long CandidateTokens { get; init; }
    public int Trimmed { get; init; }
    public IReadOnlyList<string> UnmatchedPatterns { get; init; } = [];
    public IReadOnlyList<string> Notes { get; init; } = [];

    public IReadOnlyList<string> UnmatchedChangedFiles { get; init; } = [];
    public IReadOnlyList<DefinitionJson> SymbolDefinitions { get; init; } = [];
    public string Order { get; init; } = "document";
    public string? Root { get; init; }
    public string? Chunker { get; init; }
    public int? StoredChunks { get; init; }
    public long ElapsedMs { get; init; }

    /// <summary>The server that answered <c>stored</c> (server-backed runs).</summary>
    public string? Server { get; init; }
}

/// <summary>
/// <c>chunkcache select</c>: <see cref="ContextSelector.Select"/> over the tree (paths, a symbol's neighbourhood, git changes),
/// trimmed to a token budget. Tokenizer only (no weights); with <c>--store</c>, says which selected chunks are stored.
/// </summary>
internal static class SelectCommand
{
    public static int Run(CliContext ctx, CliArgs args)
    {
        var total = Stopwatch.StartNew();
        string order = args.Value("order") ?? "document";
        if (order is not ("document" or "rank")) throw new UsageException("select: --order is document or rank.");
        Common.CheckChunkerName(args, args.Value("chunker") ?? "symbol");
        if (!Common.HasSelection(args)) throw new UsageException("select: give --path, --symbol or --since.");
        int show = args.Int("show", 40, 0);
        string? storeArg = args.Value("store");
        if (storeArg is null && SourceTree.SpecFrom(ctx, args) is null)
            throw new UsageException("select: give --store (to use its recorded source) or --solution/--directory.");

        using ModelHandle model = Common.OpenModel(ctx, args);
        using ServerConnection? server = Common.ConnectServer(ctx, args, model);
        IChunker chunker = Common.Chunker(args, model.Tokenizer);
        SelectionRequest request = Common.Selection(args, chunker);
        StoreLayout? layout = storeArg is null ? null : OpenExisting(ctx.FullPath(storeArg));
        IndexRecord? record = layout?.Find(chunker.Name, model);
        SourceTree tree = SourceTree.Load(ctx, Common.SourceFor(ctx, args, record), model.Tokenizer);
        ctx.Info(tree.Summary);

        var sw = Stopwatch.StartNew();
        Chunk[] chunks = tree.Files.SelectMany(chunker.Chunk).ToArray();
        ReferenceGraph graph = tree.BuildGraph(model.Tokenizer, chunks);
        ctx.Info($"{chunker.Name}: {chunks.Length} chunks, graph {graph.Edges.Count} edges [{sw.Elapsed.TotalSeconds:F1} s]");

        SelectionResult result = Common.Select(request, tree, graph);
        var files = tree.Files.ToDictionary(f => f.Path, RoslynGraphBuilder.PathComparer);
        IReadOnlyList<SelectedChunk> ordered = order == "rank" ? result.Selected : result.InDocumentOrder();
        var list = new List<SelectedJson>();
        int stored = 0;
        // Server-backed: the server answers `stored` for the selected texts (one explain request, text only; there is no hash lookup).
        string[] texts = ordered.Select(s => Common.TextOf(files[s.Chunk.FilePath], s.Chunk)).ToArray();
        bool?[]? answered = server is null ? null : StoredViaExplain(ctx, server, chunker.Name, texts);
        int at = 0;
        foreach (SelectedChunk s in ordered)
        {
            ChunkRef c = s.Chunk;
            string text = texts[at];
            Chunk g = graph.Chunks[graph.IndexOf(c.Id)];
            string key = model.KeyOf(g.TokenIds).Hex;
            bool? isStored = answered is not null ? answered[at] : layout is null ? null : layout.Entries.Contains(new ChunkKey(key));
            at++;
            if (isStored == true) stored++;
            list.Add(new SelectedJson(c.Id, c.FilePath, c.StartLine, c.EndLine, s.Tokens, Common.WireHash(text), Name(s.Reason), s.Detail,
                s.Hop >= 0 ? s.Hop : null, isStored, args.Flag("include-text") ? text : null)
            {
                Reasons = Enum.GetValues<SelectionReason>().Where(r => r != SelectionReason.None && s.Reasons.HasFlag(r)).Select(Name).ToArray(),
                Priority = s.Priority,
                Key = key,
            });
        }

        var output = new SelectResult
        {
            Chunks = list, SelectedTokens = result.SelectedTokens, CandidateTokens = result.CandidateTokens, Trimmed = result.Trimmed.Count,
            UnmatchedPatterns = result.UnmatchedPatterns, Notes = result.Notes, UnmatchedChangedFiles = result.UnmatchedChangedFiles,
            SymbolDefinitions = result.SymbolDefinitions.Select(d => new DefinitionJson(d.QualifiedName, d.Kind, d.FilePath, d.Line)).ToList(),
            Order = order, Root = tree.Root, Chunker = chunker.Name, StoredChunks = layout is null && answered is null ? null : stored, ElapsedMs = total.ElapsedMilliseconds,
            Server = server?.BaseUrl,
        };
        if (ctx.Json)
        {
            ctx.WriteJson(output);
            return ExitCodes.Ok;
        }
        TextWriter o = ctx.Out;
        foreach (DefinitionJson d in output.SymbolDefinitions) o.WriteLine($"definition {d.QualifiedName} ({d.Kind}) {d.Path}:{d.Line}");
        o.WriteLine($"selected {list.Count} chunks / {output.SelectedTokens} tokens of {output.CandidateTokens} candidate tokens; trimmed {output.Trimmed} chunks"
            + (layout is null && answered is null ? "" : $"; {stored} of {list.Count} stored"));
        foreach (SelectedJson c in list.Take(show))
            o.WriteLine($"  {c.Path}:{c.StartLine}-{c.EndLine}  {c.Tokens} tok  {c.Reason}{(c.Hop is int h ? $" hop {h}" : "")}{(c.Stored is bool b ? (b ? "  stored" : "  missing") : "")}  {c.Detail}");
        if (list.Count > show) o.WriteLine($"  ... {list.Count - show} more (--show N)");
        foreach (string n in output.Notes) o.WriteLine($"note: {n}");
        foreach (string u in output.UnmatchedPatterns) o.WriteLine($"unmatched pattern: {u}");
        if (output.UnmatchedChangedFiles.Count > 0) o.WriteLine($"changed files with no chunk: {string.Join(", ", output.UnmatchedChangedFiles.Take(20))}");
        return ExitCodes.Ok;
    }

    /// <summary>
    /// Whether the server holds each of <paramref name="texts"/>: ONE <c>POST /v1/chunk-cache/explain</c> per batch with
    /// <c>{"request": ...}</c> (contract ruling 15): an OpenAI chat body whose single user message has the texts as one string
    /// content (ruling 19) and a manifest span per text (UTF-8 byte offsets, the wire hash of the LF text). A span the server
    /// reports <c>hit</c> is stored; any other status, or no per-chunk report (an unauthenticated caller), is unknown (null) when
    /// the server gave no status and false otherwise. A chunker off the wire is not asked about (null).
    /// </summary>
    internal static bool?[] StoredViaExplain(CliContext ctx, ServerConnection server, string chunkerName, string[] texts)
    {
        var stored = new bool?[texts.Length];
        if (texts.Length == 0) return stored;
        if (!WireChunking.IsWireChunker(chunkerName))
        {
            ctx.Info($"note: chunker {chunkerName} is not wire-eligible; the server is not asked which chunks are stored.");
            return stored;
        }
        int maxSpans = Math.Min(server.Identity?.MaxChunksPerRequest ?? 4096, 4096);
        const long maxBytes = 2L * 1024 * 1024; // the endpoint refuses bodies over 4 MiB
        int from = 0;
        while (from < texts.Length)
        {
            int to = from;
            long bytes = 0;
            while (to < texts.Length && to - from < maxSpans && (to == from || bytes + Encoding.UTF8.GetByteCount(texts[to]) + 1 <= maxBytes))
                bytes += Encoding.UTF8.GetByteCount(texts[to++]) + 1;
            var content = new StringBuilder();
            var spans = new JsonArray();
            long offset = 0;
            for (int i = from; i < to; i++)
            {
                int n = Encoding.UTF8.GetByteCount(texts[i]);
                spans.Add(new JsonObject
                {
                    ["id"] = "c" + i.ToString(System.Globalization.CultureInfo.InvariantCulture),
                    ["hash"] = Common.WireHash(texts[i]),
                    ["chunker"] = chunkerName,
                    ["at"] = new JsonObject { ["message"] = 0, ["part"] = 0, ["start"] = offset, ["end"] = offset + n },
                });
                content.Append(texts[i]).Append('\n');
                offset += n + 1;
            }
            var body = new JsonObject
            {
                ["model"] = server.Identity?.ModelId ?? "chunkcache",
                ["messages"] = new JsonArray(new JsonObject { ["role"] = "user", ["content"] = content.ToString() }),
                ["max_tokens"] = 1,
                ["chunk_cache"] = new JsonObject { ["v"] = 1, ["policy"] = "shadow", ["report"] = "per_chunk", ["chunks"] = spans },
            };
            Client.ExplainResult r = server.Client.ExplainAsync(body, CancellationToken.None).AsTask().GetAwaiter().GetResult();
            var byId = (r.Chunks ?? []).Where(c => c.Id is not null).ToDictionary(c => c.Id!, c => c.Status, StringComparer.Ordinal);
            ctx.Info($"explain: {string.Join(", ", (r.Chunks ?? []).GroupBy(c => c.Status + (c.Why is null ? "" : " (" + c.Why + ")")).Select(g => $"{g.Count()} {g.Key}"))}"
                + (r.Chunks is null || r.Chunks.Count == 0 ? "no per-chunk report (the caller is not authenticated for its namespace?)" : ""));
            for (int i = from; i < to; i++)
                if (byId.TryGetValue("c" + i.ToString(System.Globalization.CultureInfo.InvariantCulture), out string? status) && status is not null)
                    stored[i] = status == "hit";
            from = to;
        }
        return stored;
    }

    private static StoreLayout OpenExisting(string root) =>
        Directory.Exists(root) ? StoreLayout.Open(root, create: false) : throw new DirectoryNotFoundException($"Store not found: {root}");

    private static string Name(SelectionReason r) => r switch
    {
        SelectionReason.Definition => "definition",
        SelectionReason.DefinitionBody => "definition_body",
        SelectionReason.Changed => "changed",
        SelectionReason.Path => "path",
        SelectionReason.Caller => "caller",
        SelectionReason.Callee => "callee",
        _ => r.ToString().ToLowerInvariant(),
    };
}
