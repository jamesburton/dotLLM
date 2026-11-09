using System.Text.Json;
using System.Text.Json.Nodes;

namespace DotLLM.ChunkCache;

/// <summary>One chunk of an explain answer (<c>chunkcache_explain.chunks[]</c>).</summary>
internal sealed record ExplainChunkJson(string? Id, string? Status, string? Why);

/// <summary>Predicted times of an explain answer.</summary>
internal sealed record ExplainPredictedJson(double? Reuse, double? FullPrefill);

/// <summary>Result of <c>explain</c>: the <c>chunkcache_explain</c> output schema (spec §7.2).</summary>
internal sealed record ExplainJson(string? PolicyWouldApply, string? Reason, IReadOnlyList<ExplainChunkJson> Chunks, ExplainPredictedJson? PredictedMs,
    IReadOnlyList<string> Warnings)
{
    /// <summary>The server asked (no user info).</summary>
    public string? Server { get; init; }
}

/// <summary>
/// <c>chunkcache explain --server URL --request FILE</c>: a dry run of the chunk plan for a request body
/// (<c>POST /v1/chunk-cache/explain</c> with <c>{"request": &lt;the file&gt;}</c>, contract ruling 15). Text only; the file is
/// sent as it is (it is the user's own request), and nothing is written anywhere.
/// </summary>
internal static class ExplainCommand
{
    private const long MaxRequestFileBytes = 4L * 1024 * 1024; // the endpoint refuses larger bodies

    public static readonly IReadOnlyDictionary<string, OptionKind> Spec = Common.Merge(Common.ModelOptions, new Dictionary<string, OptionKind>
    {
        ["request"] = OptionKind.Single,
        ["show"] = OptionKind.Single,
    });

    public const string Help = """
        chunkcache explain --server URL --request FILE.json [--json]

          What would the server do with this chat request's chunk_cache annotations? Posts {"request": <FILE>} to
          POST /v1/chunk-cache/explain and prints policy_would_apply, reason, per-chunk status (prefix | hit | miss | ignored, with
          the reason for an ignored one: only for an authenticated caller's own namespace), predicted times and warnings.
          The API key is read from CHUNKCACHE_API_KEY only. --show N chunk lines in text mode (default 40).
        """;

    public static int Run(CliContext ctx, CliArgs args)
    {
        string serverUrl = args.Value("server") ?? throw new UsageException("explain: --server URL is required.");
        if (args.Flag("offline") || args.Value("model") is not null || args.Flag("strict-model-id") || args.Value("threads") is not null)
            throw new UsageException("explain: it only asks the server (--server, --request); --model, --offline and --threads do not apply.");
        string file = ctx.FullPath(args.Required("request"));
        int show = args.Int("show", 40, 0);
        var info = new FileInfo(file);
        if (!info.Exists) throw new FileNotFoundException($"Request file not found: {file}", file);
        if (info.Length > MaxRequestFileBytes) throw new InvalidDataException($"Request file {file} is larger than {MaxRequestFileBytes >> 20} MiB (the server refuses it).");
        JsonObject body;
        try
        {
            body = JsonNode.Parse(File.ReadAllBytes(file)) as JsonObject ?? throw new InvalidDataException($"Request file {file}: expected a JSON object (a chat request body).");
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException($"Request file {file} is not JSON: {ex.Message}");
        }

        using ServerConnection server = ServerConnection.Open(ctx, serverUrl);
        Client.ExplainResult r = server.Client.ExplainAsync(body, CancellationToken.None).AsTask().GetAwaiter().GetResult();
        var result = new ExplainJson(r.PolicyWouldApply, r.Reason, (r.Chunks ?? []).Select(c => new ExplainChunkJson(c.Id, c.Status, c.Why)).ToList(),
            r.PredictedMs is null ? null : new ExplainPredictedJson(r.PredictedMs.Reuse, r.PredictedMs.FullPrefill), r.Warnings ?? [])
        { Server = server.BaseUrl };
        if (ctx.Json)
        {
            ctx.WriteJson(result);
            return ExitCodes.Ok;
        }
        TextWriter o = ctx.Out;
        o.WriteLine($"policy would apply: {result.PolicyWouldApply ?? "(none)"}{(result.Reason is null ? "" : $" ({result.Reason})")}");
        foreach (ExplainChunkJson c in result.Chunks.Take(show))
            o.WriteLine($"  {c.Id}  {c.Status}{(c.Why is null ? "" : $"  {c.Why}")}");
        if (result.Chunks.Count > show) o.WriteLine($"  ... {result.Chunks.Count - show} more (--show N)");
        if (result.PredictedMs is { } p) o.WriteLine($"predicted: reuse {p.Reuse?.ToString("0.#", System.Globalization.CultureInfo.InvariantCulture) ?? "-"} ms, full prefill {p.FullPrefill?.ToString("0.#", System.Globalization.CultureInfo.InvariantCulture) ?? "-"} ms");
        foreach (string w in result.Warnings) o.WriteLine($"warning: {w}");
        return ExitCodes.Ok;
    }
}
