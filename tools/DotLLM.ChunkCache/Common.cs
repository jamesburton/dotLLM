using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;
using DotLLM.ChunkReuse.Store;
using DotLLM.ChunkReuse.Tooling;
using DotLLM.Tokenizers;

namespace DotLLM.ChunkCache;

/// <summary>Options and helpers shared by the verbs.</summary>
internal static class Common
{
    /// <summary>Default lock wait.</summary>
    public static readonly TimeSpan DefaultLockTimeout = TimeSpan.FromSeconds(30);

    /// <summary>Options every model-using verb takes.</summary>
    public static readonly IReadOnlyDictionary<string, OptionKind> ModelOptions = new Dictionary<string, OptionKind>
    {
        ["model"] = OptionKind.Single,
        ["strict-model-id"] = OptionKind.Flag,
        ["threads"] = OptionKind.Single,
        ["json"] = OptionKind.Flag,
        ["help"] = OptionKind.Flag,
    };

    /// <summary>Source options (<c>--solution</c> or <c>--directory</c>).</summary>
    public static readonly IReadOnlyDictionary<string, OptionKind> SourceOptions = new Dictionary<string, OptionKind>
    {
        ["solution"] = OptionKind.Single,
        ["directory"] = OptionKind.Single,
        ["configuration"] = OptionKind.Single,
    };

    /// <summary>Chunker options.</summary>
    public static readonly IReadOnlyDictionary<string, OptionKind> ChunkerOptions = new Dictionary<string, OptionKind>
    {
        ["chunker"] = OptionKind.Single,
        ["chunk-tokens"] = OptionKind.Single,
    };

    /// <summary><see cref="ContextSelector"/> options.</summary>
    public static readonly IReadOnlyDictionary<string, OptionKind> SelectionOptions = new Dictionary<string, OptionKind>
    {
        ["path"] = OptionKind.List,
        ["exclude"] = OptionKind.List,
        ["symbol"] = OptionKind.Single,
        ["hops"] = OptionKind.Single,
        ["since"] = OptionKind.Single,
        ["merge-base"] = OptionKind.Flag,
        ["budget"] = OptionKind.Single,
        ["max-per-hop"] = OptionKind.Single,
    };

    /// <summary>Merges option tables.</summary>
    public static IReadOnlyDictionary<string, OptionKind> Merge(params IReadOnlyDictionary<string, OptionKind>[] tables)
    {
        var all = new Dictionary<string, OptionKind>(StringComparer.Ordinal);
        foreach (var t in tables)
            foreach (var kv in t) all[kv.Key] = kv.Value;
        return all;
    }

    /// <summary>CPU threads for forwards: <c>--threads</c>, default min(8, processors).</summary>
    public static int Threads(CliArgs args) => args.Int("threads", Math.Min(8, Environment.ProcessorCount), 1, 1024);

    /// <summary>Opens the model of <c>--model</c>.</summary>
    public static ModelHandle OpenModel(CliContext ctx, CliArgs args) =>
        ModelHandle.Open(ctx, args.Required("model"), args.Flag("strict-model-id"), Threads(args));

    /// <summary>The chunker of <c>--chunker</c> (<c>symbol</c> default, or <c>fixed</c>) and <c>--chunk-tokens</c> (512).</summary>
    public static IChunker Chunker(CliArgs args, ITokenizer tokenizer) => Chunker(args.Value("chunker") ?? "symbol", args, tokenizer);

    /// <summary>The named chunker with <c>--chunk-tokens</c>.</summary>
    public static IChunker Chunker(string name, CliArgs args, ITokenizer tokenizer)
    {
        int tokens = args.Int("chunk-tokens", 512, 1, 1 << 20);
        return name switch
        {
            "symbol" => new SymbolChunker(tokenizer, tokens),
            "fixed" => new FixedSizeChunker(tokenizer, tokens),
            _ => throw new UsageException($"{args.Verb}: unknown chunker '{name}' (symbol|fixed)."),
        };
    }

    /// <summary>The chunker kind name (<c>symbol</c>/<c>fixed</c>) validated, for a list option.</summary>
    public static string CheckChunkerName(CliArgs args, string name) => name is "symbol" or "fixed"
        ? name : throw new UsageException($"{args.Verb}: unknown chunker '{name}' (symbol|fixed).");

    /// <summary>Whether any selection criterion was given.</summary>
    public static bool HasSelection(CliArgs args) =>
        args.List("path").Count > 0 || args.Value("symbol") is not null || args.Value("since") is not null;

    /// <summary>The <see cref="SelectionRequest"/> of the selection options.</summary>
    public static SelectionRequest Selection(CliArgs args, IChunker chunker)
    {
        if (args.Flag("merge-base") && args.Value("since") is null) throw new UsageException($"{args.Verb}: --merge-base needs --since.");
        if (args.Value("hops") is not null && args.Value("symbol") is null) throw new UsageException($"{args.Verb}: --hops needs --symbol.");
        return new SelectionRequest
        {
            Paths = args.List("path"),
            Exclude = args.List("exclude"),
            Symbol = args.Value("symbol"),
            Hops = args.Int("hops", 1, 0, ContextSelector.MaxHops),
            ChangedSince = args.Value("since"),
            ChangedSinceMergeBase = args.Flag("merge-base"),
            TokenBudget = args.IntOrNull("budget", 0),
            MaxNeighboursPerHop = args.IntOrNull("max-per-hop", 0),
            ChunkKind = ChunkRef.KindOf(chunker),
        };
    }

    /// <summary>Runs <see cref="ContextSelector.Select"/>, with git only when the request needs it.</summary>
    public static SelectionResult Select(SelectionRequest request, SourceTree tree, ReferenceGraph graph) =>
        ContextSelector.Select(request, tree.Files, graph, request.ChangedSince is null ? null : new GitCliInfo(tree.Root), tree.Root);

    /// <summary>
    /// The source to scan: <c>--solution</c>/<c>--directory</c> when given, else the one recorded for this index in the store.
    /// </summary>
    public static SourceSpec SourceFor(CliContext ctx, CliArgs args, IndexRecord? record)
    {
        SourceSpec? given = SourceTree.SpecFrom(ctx, args);
        if (given is not null) return given;
        if (record is not null) return record.Source;
        throw new UsageException($"{args.Verb}: give --solution or --directory (the store has no index for this model and chunker to take the source from).");
    }

    /// <summary>Takes the store lock, mapping a timeout to a clear error.</summary>
    public static IDisposable Lock(CliContext ctx, string storeRoot, TimeSpan timeout)
    {
        ctx.Info($"waiting up to {timeout.TotalSeconds:0.#} s for the store lock ...");
        return ChunkIndex.AcquireStoreLock(storeRoot, timeout);
    }

    /// <summary>A chunk's text (the file text between its offsets).</summary>
    public static string TextOf(SourceFile file, ChunkRef chunk) => file.Text[chunk.StartOffset..chunk.EndOffset];

    /// <summary>The wire hash of a chunk text: <c>sha256:</c> + hex of its UTF-8 bytes (integration spec §5.2).</summary>
    public static string WireHash(string text) => WireChunking.Hash(text);

    /// <summary>Bytes as a short human string.</summary>
    public static string Bytes(double bytes)
    {
        string[] units = ["B", "KiB", "MiB", "GiB", "TiB"];
        int u = 0;
        while (Math.Abs(bytes) >= 1024 && u < units.Length - 1)
        {
            bytes /= 1024;
            u++;
        }
        return u == 0 ? $"{bytes:0} {units[u]}" : $"{bytes:0.##} {units[u]}";
    }
}
