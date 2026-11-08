using System.Diagnostics;
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
}

/// <summary>
/// <c>chunkcache select</c>: <see cref="ContextSelector.Select"/> over the tree (paths, a symbol's neighbourhood, git changes),
/// trimmed to a token budget. Tokenizer only (no weights); with <c>--store</c>, says which selected chunks are stored.
/// </summary>
internal static class SelectCommand
{
    public static readonly IReadOnlyDictionary<string, OptionKind> Spec = Common.Merge(Common.ModelOptions, Common.SourceOptions, Common.ChunkerOptions,
        Common.SelectionOptions, new Dictionary<string, OptionKind>
        {
            ["store"] = OptionKind.Single,
            ["include-text"] = OptionKind.Flag,
            ["order"] = OptionKind.Single,
            ["show"] = OptionKind.Single,
        });

    public const string Help = """
        chunkcache select --model M.gguf (--store DIR | --solution X | --directory D) [criteria] [options]

          Chooses chunks for a context (the union of the criteria), ranked definition > body > path > changed > neighbour,
          trimmed strictly to --budget. Tokenizer only. With --store, the source defaults to the one the store's index records
          and each chunk says whether it is stored.

          --path G                    a glob (repeat for more; '!' negates)   --exclude G   drop matching files everywhere (repeatable)
          --symbol S --hops H         a symbol's definition and its callers/callees within H hops (default 1)
          --max-per-hop N             keep only the N heaviest neighbours per hop
          --since REF [--merge-base]  files changed since REF (or since the merge base of REF and HEAD), plus untracked
          --budget T                  token budget
          --order document|rank       output order (default document)    --include-text   include each chunk's text
          --chunker symbol|fixed --chunk-tokens N --configuration C --json --show N (lines, default 40)
        """;

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
        foreach (SelectedChunk s in ordered)
        {
            ChunkRef c = s.Chunk;
            string text = Common.TextOf(files[c.FilePath], c);
            Chunk g = graph.Chunks[graph.IndexOf(c.Id)];
            string key = model.KeyOf(g.TokenIds).Hex;
            bool? isStored = layout is null ? null : layout.Entries.Contains(new ChunkKey(key));
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
            Order = order, Root = tree.Root, Chunker = chunker.Name, StoredChunks = layout is null ? null : stored, ElapsedMs = total.ElapsedMilliseconds,
        };
        if (ctx.Json)
        {
            ctx.WriteJson(output);
            return ExitCodes.Ok;
        }
        TextWriter o = ctx.Out;
        foreach (DefinitionJson d in output.SymbolDefinitions) o.WriteLine($"definition {d.QualifiedName} ({d.Kind}) {d.Path}:{d.Line}");
        o.WriteLine($"selected {list.Count} chunks / {output.SelectedTokens} tokens of {output.CandidateTokens} candidate tokens; trimmed {output.Trimmed} chunks"
            + (layout is null ? "" : $"; {stored} of {list.Count} stored"));
        foreach (SelectedJson c in list.Take(show))
            o.WriteLine($"  {c.Path}:{c.StartLine}-{c.EndLine}  {c.Tokens} tok  {c.Reason}{(c.Hop is int h ? $" hop {h}" : "")}{(c.Stored is bool b ? (b ? "  stored" : "  missing") : "")}  {c.Detail}");
        if (list.Count > show) o.WriteLine($"  ... {list.Count - show} more (--show N)");
        foreach (string n in output.Notes) o.WriteLine($"note: {n}");
        foreach (string u in output.UnmatchedPatterns) o.WriteLine($"unmatched pattern: {u}");
        if (output.UnmatchedChangedFiles.Count > 0) o.WriteLine($"changed files with no chunk: {string.Join(", ", output.UnmatchedChangedFiles.Take(20))}");
        return ExitCodes.Ok;
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
