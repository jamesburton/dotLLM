using System.ClientModel;
using System.ClientModel.Primitives;
using System.Collections.Concurrent;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;
using OpenAI;

// Test code: handlers, requests and helpers are handed to HttpClient / HttpMessageInvoker chains that each test
// disposes (ownership the analyzer cannot follow), and continuations may resume on xUnit's context.
#pragma warning disable IDISP001, IDISP004, IDISP017, MA0004

namespace DotLLM.ChunkCache.Client.Tests;

/// <summary>One request as the server saw it.</summary>
internal sealed record RecordedRequest(string Method, string Path, IReadOnlyDictionary<string, string> Headers, byte[] Body)
{
    public string BodyText => Encoding.UTF8.GetString(Body);

    public JsonObject Json => JsonNode.Parse(Body)!.AsObject();
}

/// <summary>
/// A minimal HTTP/1.1 server on a real loopback socket: the test decides what each request gets, including a response
/// written in pieces (for SSE). Every response closes its connection, so a streamed body ends at EOF.
/// </summary>
internal sealed class FakeHttpServer : IAsyncDisposable
{
    private readonly TcpListener _listener;
    private readonly Func<RecordedRequest, Stream, CancellationToken, Task> _respond;
    private readonly CancellationTokenSource _stop = new();
    private readonly Task _loop;

    public FakeHttpServer(Func<RecordedRequest, Stream, CancellationToken, Task> respond)
    {
        _respond = respond;
        _listener = new TcpListener(IPAddress.Loopback, 0);
        _listener.Start();
        BaseUri = new Uri($"http://127.0.0.1:{((IPEndPoint)_listener.LocalEndpoint).Port}/");
        _loop = Task.Run(AcceptLoopAsync);
    }

    public Uri BaseUri { get; }

    public ConcurrentQueue<RecordedRequest> Requests { get; } = new();

    public RecordedRequest[] Snapshot => Requests.ToArray();

    public static async Task WriteJsonAsync(Stream s, int status, string json, CancellationToken ct)
    {
        byte[] body = Encoding.UTF8.GetBytes(json);
        string head = $"HTTP/1.1 {status} {Reason(status)}\r\nContent-Type: application/json\r\nContent-Length: {body.Length}\r\nConnection: close\r\n\r\n";
        await s.WriteAsync(Encoding.ASCII.GetBytes(head), ct);
        await s.WriteAsync(body, ct);
        await s.FlushAsync(ct);
    }

    public static async Task WriteSseHeadAsync(Stream s, CancellationToken ct)
    {
        const string head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
        await s.WriteAsync(Encoding.ASCII.GetBytes(head), ct);
        await s.FlushAsync(ct);
    }

    public static async Task WriteRawAsync(Stream s, string text, CancellationToken ct)
    {
        await s.WriteAsync(Encoding.UTF8.GetBytes(text), ct);
        await s.FlushAsync(ct);
    }

    private static string Reason(int status) => status switch
    {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        503 => "Service Unavailable",
        _ => "Status",
    };

    private async Task AcceptLoopAsync()
    {
        while (!_stop.IsCancellationRequested)
        {
            TcpClient client;
            try
            {
                client = await _listener.AcceptTcpClientAsync(_stop.Token);
            }
            catch (Exception) when (_stop.IsCancellationRequested)
            {
                return;
            }
            catch (ObjectDisposedException)
            {
                return;
            }

            _ = Task.Run(() => ServeAsync(client));
        }
    }

    private async Task ServeAsync(TcpClient client)
    {
#pragma warning disable IDISP007 // the accepted connection belongs to this method
        using (client)
#pragma warning restore IDISP007
        {
            NetworkStream s = client.GetStream();
            try
            {
                RecordedRequest request = await ReadRequestAsync(s, _stop.Token);
                Requests.Enqueue(request);
                await _respond(request, s, _stop.Token);
            }
            catch (Exception) when (_stop.IsCancellationRequested)
            {
            }
            catch (IOException)
            {
            }
        }
    }

    private static async Task<RecordedRequest> ReadRequestAsync(Stream s, CancellationToken ct)
    {
        var head = new List<byte>();
        var one = new byte[1];
        while (true)
        {
            int n = await s.ReadAsync(one, ct);
            if (n == 0) throw new IOException("closed before the request head");
            head.Add(one[0]);
            if (head.Count >= 4 && head[^4] == '\r' && head[^3] == '\n' && head[^2] == '\r' && head[^1] == '\n')
                break;
        }

        string[] lines = Encoding.ASCII.GetString(head.ToArray()).Split("\r\n");
        string[] first = lines[0].Split(' ');
        var headers = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (string line in lines.Skip(1).Where(l => l.Length > 0))
        {
            int c = line.IndexOf(':', StringComparison.Ordinal);
            headers[line[..c].Trim()] = line[(c + 1)..].Trim();
        }

        byte[] body;
        if (headers.TryGetValue("Content-Length", out string? len))
        {
            body = new byte[int.Parse(len, CultureInfo.InvariantCulture)];
            await s.ReadExactlyAsync(body, ct);
        }
        else if (headers.TryGetValue("Transfer-Encoding", out string? te) && te.Contains("chunked", StringComparison.OrdinalIgnoreCase))
        {
            var ms = new MemoryStream();
            while (true)
            {
                string sizeLine = await ReadLineAsync(s, ct);
                int size = int.Parse(sizeLine.Split(';')[0], NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                if (size == 0)
                {
                    await ReadLineAsync(s, ct);
                    break;
                }

                var chunk = new byte[size];
                await s.ReadExactlyAsync(chunk, ct);
                ms.Write(chunk);
                await ReadLineAsync(s, ct);
            }

            body = ms.ToArray();
        }
        else
        {
            body = [];
        }

        return new RecordedRequest(first[0], first[1], headers, body);
    }

    private static async Task<string> ReadLineAsync(Stream s, CancellationToken ct)
    {
        var sb = new StringBuilder();
        var one = new byte[1];
        while (true)
        {
            await s.ReadExactlyAsync(one, ct);
            if (one[0] == '\n') return sb.ToString().TrimEnd('\r');
            sb.Append((char)one[0]);
        }
    }

    public async ValueTask DisposeAsync()
    {
        await _stop.CancelAsync();
        _listener.Stop();
        try { await _loop; } catch (OperationCanceledException) { }
        _stop.Dispose();
    }
}

/// <summary>An in-memory terminal handler that records request bytes and content headers.</summary>
internal sealed class CaptureHandler(Func<HttpRequestMessage, byte[]?, HttpResponseMessage> respond) : HttpMessageHandler
{
    public List<(HttpRequestMessage Request, byte[]? Body, string? ContentType, long? ContentLength)> Seen { get; } = [];

    public static CaptureHandler Ok(string json = "{}") =>
        new((_, _) => new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(json, Encoding.UTF8, "application/json") });

    public byte[] LastBody => Seen[^1].Body!;

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        byte[]? body = request.Content is null ? null : await request.Content.ReadAsByteArrayAsync(cancellationToken);
        Seen.Add((request, body, request.Content?.Headers.ContentType?.ToString(), request.Content?.Headers.ContentLength));
        return respond(request, body);
    }
}

#pragma warning disable IDISP011 // a 503 is disposed only on the retry path; the analyzer cannot see the loop exit
/// <summary>A retrying handler: resends the same request message after a 503, awaiting a delay in between.</summary>
internal sealed class RetryOn503Handler(int maxRetries, TimeSpan delay) : DelegatingHandler
{
    public int Attempts { get; private set; }

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        for (int i = 0; ; i++)
        {
            Attempts++;
            HttpResponseMessage response = await base.SendAsync(request, cancellationToken);
            if (response.StatusCode != HttpStatusCode.ServiceUnavailable || i >= maxRetries)
                return response;
            response.Dispose();
            await Task.Delay(delay, cancellationToken);
        }
    }
}

#pragma warning restore IDISP011

internal static class Pipeline
{
    public const string Model = "qwen3-4b";

    /// <summary>Chains <paramref name="handlers"/> (outermost first) over <paramref name="terminal"/>.</summary>
    public static HttpClient Client(HttpMessageHandler terminal, params DelegatingHandler[] handlers)
    {
        HttpMessageHandler next = terminal;
        for (int i = handlers.Length - 1; i >= 0; i--)
        {
            handlers[i].InnerHandler = next;
            next = handlers[i];
        }

        return new HttpClient(next);
    }

    /// <summary>A real MEAI OpenAI chat client over <paramref name="http"/>, with the SDK's own retries off.</summary>
    public static IChatClient OpenAiChat(HttpClient http, Uri serverBase)
    {
        var options = new OpenAIClientOptions
        {
            Endpoint = new Uri(serverBase, "v1/"),
            Transport = new HttpClientPipelineTransport(http),
            RetryPolicy = new ClientRetryPolicy(maxRetries: 0),
        };
        return new OpenAI.Chat.ChatClient(Model, new ApiKeyCredential("test-key"), options).AsIChatClient();
    }

    public static ChunkCacheCapabilities Caps(int? maxChunks = null) => new()
    {
        Versions = [1], Policies = ["exact", "shadow", "balanced", "fast", "auto"], Selectors = ["chunk_starts"],
        WireChunkers = ["symbol-512"], MaxChunksPerRequest = maxChunks,
    };

    /// <summary>UTF-16 index of a UTF-8 byte offset into <paramref name="text"/>.</summary>
    public static int Utf16Of(string text, int utf8Offset) =>
        Encoding.UTF8.GetString(Encoding.UTF8.GetBytes(text), 0, utf8Offset).Length;

    /// <summary>The part text at (message, part) of a body (string content is part 0).</summary>
    public static string PartText(JsonObject body, int message, int part) => body["messages"]![message]!["content"] switch
    {
        JsonValue v when part == 0 => (string)v!,
        JsonArray a => (string)a[part]!["text"]!,
        _ => throw new InvalidOperationException("no such text part"),
    };

    /// <summary>
    /// The fixture's raw bytes with its top-level <c>chunk_cache</c> member cut out textually, so every other byte
    /// (including the fixture's own <c>\uXXXX</c> escapes) is exactly the contract's.
    /// </summary>
    public static byte[] FixtureWithoutManifest(string name)
    {
        string text = Encoding.UTF8.GetString(Fixtures.Bytes(name));
        int cut = text.LastIndexOf(",\n  \"chunk_cache\"", StringComparison.Ordinal);
        if (cut < 0) throw new InvalidOperationException("no top-level chunk_cache in " + name);
        int close = text.LastIndexOf("\n}", StringComparison.Ordinal);
        string stripped = text[..cut] + text[close..];
        JsonObject expected = Fixtures.Json(name);
        expected.Remove("chunk_cache");
        if (!JsonNode.DeepEquals(expected, JsonNode.Parse(stripped)))
            throw new InvalidOperationException("textual cut changed the document");
        return Encoding.UTF8.GetBytes(stripped);
    }

    /// <summary>
    /// Asserts <paramref name="output"/> is <paramref name="input"/> with exactly one top-level member
    /// <c>,"chunk_cache":{...}</c> inserted before the final <c>}</c>, and returns the inserted manifest's raw text.
    /// </summary>
    public static string SplicedManifest(byte[] input, byte[] output)
    {
        int close = Array.LastIndexOf(input, (byte)'}');
        Xunit.Assert.True(output.Length > input.Length, "nothing was inserted");
        Xunit.Assert.True(output.AsSpan(0, close).SequenceEqual(input.AsSpan(0, close)), "bytes before the insertion point changed");
        int tail = input.Length - close;
        Xunit.Assert.True(output.AsSpan(output.Length - tail).SequenceEqual(input.AsSpan(close)), "bytes after the insertion point changed");
        string inserted = Encoding.UTF8.GetString(output, close, output.Length - tail - close);
        const string prefix = ",\"chunk_cache\":";
        Xunit.Assert.StartsWith(prefix, inserted, StringComparison.Ordinal);
        return inserted[prefix.Length..];
    }

    public static HttpRequestMessage Post(byte[] body, string path = "http://dotllm.test/v1/chat/completions")
    {
        var content = new ByteArrayContent(body);
        content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/json") { CharSet = "utf-8" };
        return new HttpRequestMessage(HttpMethod.Post, path) { Content = content };
    }

    /// <summary>A body with string contents only (the v1 OpenAI surface, ruling 19).</summary>
    public static byte[] StringBody(params (string Role, string Content)[] messages)
    {
        var arr = new JsonArray();
        foreach ((string role, string content) in messages)
            arr.Add(new JsonObject { ["role"] = role, ["content"] = content });
        return Encoding.UTF8.GetBytes(new JsonObject { ["model"] = Model, ["messages"] = arr, ["temperature"] = 0 }.ToJsonString());
    }
}

/// <summary>A test clock: only <see cref="GetUtcNow"/> is faked; timers stay real.</summary>
internal sealed class ManualTime : TimeProvider
{
    private DateTimeOffset _now = new(2026, 11, 2, 12, 0, 0, TimeSpan.Zero);

    public override DateTimeOffset GetUtcNow() => _now;

    public void Advance(TimeSpan by) => _now += by;
}

/// <summary>A scripted <see cref="IChunkCacheClient"/> for the capability probe.</summary>
internal sealed class ScriptedProbe(Func<int, CancellationToken, Task<ChunkCacheCapabilities?>> answer) : IChunkCacheClient
{
    private int _calls;

    public int Calls => Volatile.Read(ref _calls);

    public async ValueTask<ChunkCacheCapabilities?> GetCapabilitiesAsync(CancellationToken ct) =>
        await answer(Interlocked.Increment(ref _calls), ct);

    public ValueTask<SelectResult> SelectAsync(SelectRequest request, CancellationToken ct) => throw new NotSupportedException();

    public ValueTask<PrewarmResult> PrewarmAsync(PrewarmRequest request, CancellationToken ct) => throw new NotSupportedException();

    public ValueTask<StatusResult> StatusAsync(StatusRequest request, CancellationToken ct) => throw new NotSupportedException();

    public ValueTask<EvictResult> EvictAsync(EvictRequest request, CancellationToken ct) => throw new NotSupportedException();

    public ValueTask<ExplainResult> ExplainAsync(JsonObject requestBody, CancellationToken ct) => throw new NotSupportedException();
}
