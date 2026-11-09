using System.Text;
using System.Text.Json;
using DotLLM.ChunkCache.Client;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;
using DotLLM.ChunkReuse.Tooling;

namespace DotLLM.ChunkCache;

/// <summary>Summed answers of <c>POST /v1/chunk-cache/prewarm</c> (the <c>chunkcache_prewarm</c> output schema plus request counts).</summary>
internal sealed record PrewarmJson(int Requested, int AlreadyStored, int Queued, long QueuedTokens, int Refused, IReadOnlyList<string> RefusedReasons, int Requests, int SkippedIneligible);

/// <summary>
/// Sends chunk texts to the server's prewarm endpoint in batches (text only: never a hash, a token id or K/V) and sums the
/// answers. Texts the CLI refuses itself (deny-listed path, path outside the corpus root) are counted with their reason and never
/// sent.
/// </summary>
internal sealed class PrewarmSender(ServerConnection server, int? maxTokens = null)
{
    private const int BatchTexts = 256;
    private const long BatchBytes = 2L * 1024 * 1024;

    private readonly List<PrewarmText> _batch = [];
    private readonly List<string> _reasons = [];
    private long _batchBytes;
    private int _requested, _alreadyStored, _queued, _refused, _requests, _skipped;
    private long _queuedTokens;

    /// <summary>Queues <paramref name="text"/>; sends a batch when it is full.</summary>
    public void Add(PrewarmText text)
    {
        long bytes = Encoding.UTF8.GetByteCount(text.Text ?? "");
        if (_batch.Count >= BatchTexts || (_batch.Count > 0 && _batchBytes + bytes > BatchBytes)) Flush();
        _batch.Add(text);
        _batchBytes += bytes;
    }

    /// <summary>Counts a text the CLI itself did not send (<paramref name="reason"/> joins the refused reasons).</summary>
    public void Refuse(string reason)
    {
        _requested++;
        _refused++;
        if (!_reasons.Contains(reason)) _reasons.Add(reason);
    }

    /// <summary>Counts a chunk that is not wire-eligible and therefore not sent (not a refusal: it was never a request).</summary>
    public void SkipIneligible() => _skipped++;

    /// <summary>Sends the last batch and returns the totals.</summary>
    public PrewarmJson Finish()
    {
        Flush();
        return new PrewarmJson(_requested, _alreadyStored, _queued, _queuedTokens, _refused, _reasons, _requests, _skipped);
    }

    private void Flush()
    {
        if (_batch.Count == 0) return;
        PrewarmResult r = server.Client.PrewarmAsync(new PrewarmRequest { Texts = _batch.ToList(), MaxTokens = maxTokens }, CancellationToken.None)
            .AsTask().GetAwaiter().GetResult();
        _requests++;
        _requested += r.Requested;
        _alreadyStored += r.AlreadyStored;
        _queued += r.Queued;
        _queuedTokens += r.QueuedTokens;
        _refused += r.Refused;
        foreach (string why in r.RefusedReasons ?? []) if (!_reasons.Contains(why)) _reasons.Add(why);
        _batch.Clear();
        _batchBytes = 0;
    }
}

/// <summary>
/// <c>chunkcache prewarm --server URL</c> (integration spec §7.2): queues chunk TEXTS for the server to encode. The server
/// computes every key and writes every entry; there is no hash to key map, so a hash-only request does not exist. Input is a
/// <c>--texts</c> file or a selection over a tree (the same criteria as <c>select</c>).
/// </summary>
internal static class PrewarmCommand
{
    private const long MaxTextsFileBytes = 64L * 1024 * 1024;

    public static int Run(CliContext ctx, CliArgs args)
    {
        string serverUrl = args.Value("server") ?? throw new UsageException("prewarm: --server URL is required (the server encodes and stores; the CLI never writes entries).");
        if (args.Flag("offline")) throw new UsageException("prewarm: --offline has no meaning here; prewarm always goes through --server.");
        int? maxTokens = args.IntOrNull("max-tokens", 1);
        string? textsFile = args.Value("texts");
        bool selecting = Common.HasSelection(args) || SourceTree.SpecFrom(ctx, args) is not null;
        if (textsFile is null && !selecting) throw new UsageException("prewarm: give --texts FILE or a source with a selection (--directory/--solution and --path, --symbol or --since).");
        if (textsFile is not null && selecting) throw new UsageException("prewarm: give --texts or a selection, not both.");

        using ServerConnection server = ServerConnection.Open(ctx, serverUrl);
        var sender = new PrewarmSender(server, maxTokens);
        if (textsFile is not null) SendTexts(ctx, ctx.FullPath(textsFile), sender);
        else SendSelection(ctx, args, server, sender);
        PrewarmJson result = sender.Finish();

        if (ctx.Json)
        {
            ctx.WriteJson(result);
            return ExitCodes.Ok;
        }
        ctx.Out.WriteLine($"prewarm via {server.BaseUrl}: requested {result.Requested}, already stored {result.AlreadyStored}, queued {result.Queued} ({result.QueuedTokens} tokens), refused {result.Refused}");
        foreach (string why in result.RefusedReasons) ctx.Out.WriteLine($"  refused: {why}");
        return ExitCodes.Ok;
    }

    private static void SendTexts(CliContext ctx, string file, PrewarmSender sender)
    {
        var info = new FileInfo(file);
        if (!info.Exists) throw new FileNotFoundException($"Texts file not found: {file}", file);
        if (info.Length > MaxTextsFileBytes) throw new InvalidDataException($"Texts file {file} is larger than {MaxTextsFileBytes >> 20} MiB.");
        JsonElement root;
        try
        {
            using JsonDocument doc = JsonDocument.Parse(File.ReadAllBytes(file));
            root = doc.RootElement.Clone();
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException($"Texts file {file} is not JSON: {ex.Message}");
        }
        if (root.ValueKind == JsonValueKind.Object && root.TryGetProperty("texts", out JsonElement inner)) root = inner;
        if (root.ValueKind != JsonValueKind.Array) throw new InvalidDataException("Texts file: expected a JSON array of {id, text, source?} (or {\"texts\": [...]}).");
        foreach (JsonElement item in root.EnumerateArray())
        {
            string? id = item.ValueKind == JsonValueKind.Object && item.TryGetProperty("id", out JsonElement i) && i.ValueKind == JsonValueKind.String ? i.GetString() : null;
            string? text = item.ValueKind == JsonValueKind.Object && item.TryGetProperty("text", out JsonElement t) && t.ValueKind == JsonValueKind.String ? t.GetString() : null;
            if (string.IsNullOrEmpty(id) || string.IsNullOrEmpty(text)) throw new InvalidDataException("Texts file: every item needs a string id and a non-empty string text (text only, never a hash).");
            string? path = item.TryGetProperty("source", out JsonElement src) && src.ValueKind == JsonValueKind.Object
                && src.TryGetProperty("path", out JsonElement p) && p.ValueKind == JsonValueKind.String ? p.GetString() : null;
            if (path is not null && !IsRelativeInsideRoot(path)) { sender.Refuse("path_outside_corpus_root"); continue; }
            if (path is not null && ChunkDenyList.Default.IsDenied(path)) { sender.Refuse("denied_path"); continue; }
            sender.Add(new PrewarmText { Id = id, Text = text, Source = path is null ? null : new PrewarmSource { Path = path } });
        }
    }

    /// <summary>A corpus-relative path: not rooted, no drive or UNC, no NUL, and no <c>..</c> segment (either separator).</summary>
    internal static bool IsRelativeInsideRoot(string path)
    {
        if (path.Length == 0 || path.Contains('\0', StringComparison.Ordinal)) return false;
        if (Path.IsPathRooted(path) || path.StartsWith('/') || path.StartsWith('\\') || (path.Length > 1 && path[1] == ':')) return false;
        return !path.Split(['/', '\\']).Any(seg => seg == "..");
    }

    private static void SendSelection(CliContext ctx, CliArgs args, ServerConnection server, PrewarmSender sender)
    {
        if (!Common.HasSelection(args)) throw new UsageException("prewarm: give --path, --symbol or --since with the source.");
        string chunkerName = Common.ChunkerName(args);
        if (!WireChunking.IsWireChunker(chunkerName))
            throw new UsageException($"prewarm: chunker '{chunkerName}' is not wire-eligible (the server accepts {string.Join(", ", WireChunking.Eligible)}).");
        using ModelHandle model = Common.OpenModel(ctx, args);
        model.AdoptServerIdentity(server.GetIdentityAsync().GetAwaiter().GetResult(), server.BaseUrl);
        IChunker chunker = Common.Chunker(args, model.Tokenizer);
        SelectionRequest request = Common.Selection(args, chunker);
        SourceTree tree = SourceTree.Load(ctx, SourceTree.SpecFrom(ctx, args) ?? throw new UsageException("prewarm: give --solution or --directory."), model.Tokenizer);
        ctx.Info(tree.Summary);
        Chunk[] chunks = tree.Files.SelectMany(chunker.Chunk).ToArray();
        ReferenceGraph graph = tree.BuildGraph(model.Tokenizer, chunks);
        SelectionResult result = Common.Select(request, tree, graph);
        var files = tree.Files.ToDictionary(f => f.Path, RoslynGraphBuilder.PathComparer);
        foreach (DotLLM.ChunkReuse.Tooling.SelectedChunk s in result.InDocumentOrder())
        {
            Chunk g = graph.Chunks[graph.IndexOf(s.Chunk.Id)];
            string text = Common.TextOf(files[s.Chunk.FilePath], s.Chunk);
            if (!WireEligibility.IsEligible(chunker.Name, g, text, model.Tokenizer)) { sender.SkipIneligible(); continue; }
            sender.Add(new PrewarmText { Id = s.Chunk.Id, Text = text, Source = new PrewarmSource { Path = s.Chunk.FilePath } });
        }
    }
}
