using System.Net;

namespace DotLLM.Tray.Api;

/// <summary>
/// A non-success response from the dotLLM management API.
/// </summary>
/// <remarks>
/// Carries the parsed <c>error</c> body verbatim when the server sent one. That matters for the
/// #454 admin gate: its 403 body names the exact flag to start the server with, and the tray
/// surfaces that text rather than showing a generic "forbidden" that leaves the user guessing.
/// </remarks>
public sealed class DotLlmApiException : Exception
{
    /// <summary>Creates an exception for a failed request.</summary>
    /// <param name="statusCode">HTTP status the server returned.</param>
    /// <param name="message">The server's <c>error</c> text, or a synthesized description.</param>
    /// <param name="route">The route that failed, for logging.</param>
    public DotLlmApiException(HttpStatusCode statusCode, string message, string route)
        : base(message)
    {
        StatusCode = statusCode;
        Route = route;
    }

    /// <summary>The HTTP status code.</summary>
    public HttpStatusCode StatusCode { get; }

    /// <summary>The route that produced the failure.</summary>
    public string Route { get; }

    /// <summary>
    /// True when this failure is the #454 admin gate refusing a write route — i.e. the server is
    /// healthy but was not started with <c>--allow-model-admin</c>. The tray disables its
    /// administrative controls on this rather than treating the server as broken.
    /// </summary>
    public bool IsAdminGateRefusal => StatusCode == HttpStatusCode.Forbidden;
}
