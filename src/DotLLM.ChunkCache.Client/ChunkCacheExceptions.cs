using System.Net;

namespace DotLLM.ChunkCache.Client;

/// <summary>
/// A <c>/v1/chunk-cache</c> endpoint answered with an error status. <see cref="Code"/> carries the spec 5.8 code when
/// the server sent one (for example <c>chunk_store_unavailable</c> on a 503).
/// </summary>
public sealed class ChunkCacheRequestException : Exception
{
    /// <summary>Creates the exception.</summary>
    public ChunkCacheRequestException(HttpStatusCode statusCode, string? errorType, string? code, string message)
        : base(message)
    {
        StatusCode = statusCode;
        ErrorType = errorType;
        Code = code;
    }

    /// <summary>The HTTP status.</summary>
    public HttpStatusCode StatusCode { get; }

    /// <summary>The envelope's <c>error.type</c> (e.g. <c>invalid_request_error</c>, <c>authentication_error</c>).</summary>
    public string? ErrorType { get; }

    /// <summary>The envelope's <c>error.code</c>.</summary>
    public string? Code { get; }
}

/// <summary>The <c>chunkcache</c> CLI failed: a non-zero exit or output that is not the expected JSON.</summary>
public sealed class ChunkCacheCliException : Exception
{
    /// <summary>Creates the exception.</summary>
    public ChunkCacheCliException(int exitCode, string stderr, string message)
        : base(message)
    {
        ExitCode = exitCode;
        Stderr = stderr;
    }

    /// <summary>The process exit code.</summary>
    public int ExitCode { get; }

    /// <summary>The process's standard error.</summary>
    public string Stderr { get; }
}
