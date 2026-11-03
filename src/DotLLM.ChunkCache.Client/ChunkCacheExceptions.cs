// SPDX-License-Identifier: MIT
using System.Net;

namespace DotLLM.ChunkCache.Client;

/// <summary>
/// A <c>/v1/chunk-cache</c> endpoint answered with an error status, or with a success body that is not the expected
/// JSON. <see cref="Code"/> carries the spec 5.8 code when the server sent one (for example <c>chunk_store_unavailable</c>
/// on a 503).
/// </summary>
public sealed class ChunkCacheRequestException : Exception
{
    /// <summary>Creates the exception.</summary>
    public ChunkCacheRequestException(HttpStatusCode statusCode, string? errorType, string? code, string message,
        Exception? innerException = null)
        : base(message, innerException)
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

/// <summary>
/// The <c>chunkcache</c> CLI failed: not found on <c>PATH</c>, could not start, timed out (inner
/// <see cref="TimeoutException"/>), exited non-zero, or printed output that is not the expected JSON.
/// <see cref="ExitCode"/> is <c>-1</c> when the process never produced one.
/// </summary>
public sealed class ChunkCacheCliException : Exception
{
    /// <summary>Creates the exception.</summary>
    public ChunkCacheCliException(int exitCode, string stderr, string message, Exception? innerException = null)
        : base(message, innerException)
    {
        ExitCode = exitCode;
        Stderr = stderr;
    }

    /// <summary>The process exit code, or <c>-1</c>.</summary>
    public int ExitCode { get; }

    /// <summary>The process's standard error.</summary>
    public string Stderr { get; }
}
