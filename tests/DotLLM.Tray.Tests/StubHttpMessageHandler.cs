using System.Net;
using System.Text;

namespace DotLLM.Tray.Tests;

/// <summary>
/// An in-memory HTTP transport: the tray's client is exercised end to end over real
/// <see cref="HttpClient"/> plumbing, against canned responses, with no socket involved.
/// </summary>
internal sealed class StubHttpMessageHandler : HttpMessageHandler
{
    private readonly Func<HttpRequestMessage, HttpResponseMessage> _responder;

    internal StubHttpMessageHandler(Func<HttpRequestMessage, HttpResponseMessage> responder) =>
        _responder = responder;

    /// <summary>Requests seen, in order, with their bodies already read.</summary>
    internal List<RecordedRequest> Requests { get; } = [];

    protected override async Task<HttpResponseMessage> SendAsync(
        HttpRequestMessage request, CancellationToken cancellationToken)
    {
        var body = request.Content is null
            ? null
            : await request.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        Requests.Add(new RecordedRequest(request.Method, request.RequestUri!, body));
        return _responder(request);
    }

    internal static HttpResponseMessage Json(string json, HttpStatusCode status = HttpStatusCode.OK) =>
        new(status)
        {
            Content = new StringContent(json, Encoding.UTF8, "application/json"),
        };

    internal static HttpResponseMessage Sse(string body) =>
        new(HttpStatusCode.OK)
        {
            Content = new StringContent(body, Encoding.UTF8, "text/event-stream"),
        };

    internal static HttpClient Client(Func<HttpRequestMessage, HttpResponseMessage> responder, out StubHttpMessageHandler handler)
    {
        handler = new StubHttpMessageHandler(responder);
        return new HttpClient(handler, disposeHandler: true) { BaseAddress = new Uri("http://localhost:18080/") };
    }
}

/// <summary>One observed request.</summary>
/// <param name="Method">HTTP method.</param>
/// <param name="Uri">Absolute request URI.</param>
/// <param name="Body">Request body, or null.</param>
internal sealed record RecordedRequest(HttpMethod Method, Uri Uri, string? Body);
