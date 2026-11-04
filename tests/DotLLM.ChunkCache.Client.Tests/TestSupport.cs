using System.Net;
using System.Text;
using System.Text.Json.Nodes;

namespace DotLLM.ChunkCache.Client.Tests;

/// <summary>Paths and readers for the copied contract fixtures (Fixtures/v1).</summary>
internal static class Fixtures
{
    public static string Dir => Path.Combine(AppContext.BaseDirectory, "Fixtures", "v1");

    public static string PathOf(string name) => Path.Combine(Dir, name);

    public static byte[] Bytes(string name) => File.ReadAllBytes(PathOf(name));

    /// <summary>Parses a fixture with a real serializer, so JSON escapes are decoded exactly as a client or server would.</summary>
    public static JsonObject Json(string name) => JsonNode.Parse(Bytes(name))!.AsObject();

    /// <summary>One annotation of a request fixture and the decoded text it covers.</summary>
    public sealed record Annotation(string Id, string Hash, string SpanText);

    /// <summary>
    /// Every annotation of a request fixture in document order (inline first, then manifest), with its span text cut by
    /// the wire rule: manifest <c>start</c>/<c>end</c> are UTF-8 byte offsets into the decoded part text.
    /// </summary>
    public static IReadOnlyList<Annotation> Annotations(JsonObject request)
    {
        var messages = request["messages"]!.AsArray();
        var list = new List<Annotation>();
        foreach (JsonNode? m in messages)
        {
            if (m!["content"] is JsonArray parts)
            {
                foreach (JsonNode? p in parts)
                {
                    if (p!["chunk"] is JsonObject chunk)
                        list.Add(new Annotation((string)chunk["id"]!, (string)chunk["hash"]!, (string)p["text"]!));
                }
            }
        }

        if (request["chunk_cache"]?["chunks"] is JsonArray manifest)
        {
            foreach (JsonNode? c in manifest)
            {
                JsonNode at = c!["at"]!;
                string part = PartText(messages[(int)at["message"]!]!, (int)at["part"]!);
                byte[] utf8 = Encoding.UTF8.GetBytes(part);
                int start = at["start"] is { } s ? (int)s : 0;
                int end = at["end"] is { } e ? (int)e : utf8.Length;
                string span = Encoding.UTF8.GetString(utf8, start, end - start);
                list.Add(new Annotation((string)c["id"]!, (string)c["hash"]!, span));
            }
        }

        return list;
    }

    private static string PartText(JsonNode message, int part) => message["content"] switch
    {
        JsonValue v when part == 0 => (string)v!,
        JsonArray a => (string)a[part]!["text"]!,
        _ => throw new InvalidOperationException("no such text part"),
    };
}

/// <summary>A fake <see cref="HttpMessageHandler"/> that records requests (with their bodies read eagerly).</summary>
internal sealed class StubHandler(Func<HttpRequestMessage, HttpResponseMessage> respond) : HttpMessageHandler
{
    public List<(HttpRequestMessage Request, string? Body)> Requests { get; } = [];

    public static StubHandler Json(HttpStatusCode status, string json) =>
        new(_ => new HttpResponseMessage(status) { Content = new StringContent(json, Encoding.UTF8, "application/json") });

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        string? body = request.Content is null ? null : await request.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        Requests.Add((request, body));
        return respond(request);
    }
}

/// <summary>A fake <see cref="IProcessRunner"/> that records the call and returns a canned result.</summary>
internal sealed class FakeRunner(ProcessResult result) : IProcessRunner
{
    public string? Executable { get; private set; }

    public IReadOnlyList<string>? Arguments { get; private set; }

    public Task<ProcessResult> RunAsync(string executable, IReadOnlyList<string> arguments, CancellationToken ct)
    {
        Executable = executable;
        Arguments = arguments.ToArray();
        return Task.FromResult(result);
    }
}

/// <summary>A fake <see cref="IChunkIndexQueries"/>.</summary>
internal sealed class FakeQueries(SelectResult select) : IChunkIndexQueries
{
    public SelectRequest? LastSelect { get; private set; }

    public StatusRequest? LastStatus { get; private set; }

    public ValueTask<SelectResult> SelectAsync(SelectRequest r, CancellationToken ct)
    {
        LastSelect = r;
        return ValueTask.FromResult(select);
    }

    public ValueTask<StatusResult> StatusAsync(StatusRequest r, CancellationToken ct)
    {
        LastStatus = r;
        return ValueTask.FromResult(new StatusResult { TotalChunks = 7 });
    }
}
