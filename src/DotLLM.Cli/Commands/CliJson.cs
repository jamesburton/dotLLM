using System.Text.Json;

namespace DotLLM.Cli.Commands;

/// <summary>Shared conventions of the <c>--json</c> outputs of <c>list</c>, <c>model list</c>, <c>model info</c> and <c>ps</c> (issue #1721).</summary>
internal static class CliJson
{
    /// <summary>Value of the top-level <c>version</c> field. Bumped only for a breaking change; fields are only ever added within a version.</summary>
    internal const int SchemaVersion = 1;

    /// <summary>Failure-kind codes of the error document (<c>error.code</c>); the same style as chunkcache's (<c>store_busy</c>, <c>server_unreachable</c>, ...).</summary>
    internal static class Codes
    {
        /// <summary>No dotllm server answered (connection refused, timeout).</summary>
        internal const string ServerUnreachable = "server_unreachable";
        /// <summary>A server answered with something that is not the expected JSON shape.</summary>
        internal const string ServerBadResponse = "server_bad_response";
        /// <summary>Hugging Face could not be reached or answered an HTTP error (unknown repo, 5xx, offline).</summary>
        internal const string HubUnreachable = "hub_unreachable";
        /// <summary>Hugging Face answered with something that is not the expected JSON.</summary>
        internal const string HubBadResponse = "hub_bad_response";
        /// <summary>Any other failure (I/O, a bug); the message carries the exception text.</summary>
        internal const string Internal = "internal_error";
    }

    /// <summary>Maps an exception from a Hugging Face call to a <see cref="Codes"/> value.</summary>
    internal static string ClassifyHubFailure(Exception ex) => ex switch
    {
        HttpRequestException or TaskCanceledException => Codes.HubUnreachable,
        JsonException => Codes.HubBadResponse,
        _ => Codes.Internal,
    };

    /// <summary>Prints the error document <c>{"error":{"code","message","exit_code"}}</c> on stdout (same shape as chunkcache) and returns <paramref name="exitCode"/>.</summary>
    internal static int WriteError(string code, string message, int exitCode)
    {
        Console.WriteLine(JsonSerializer.Serialize(
            new CliJsonErrorResult { Error = new CliJsonError { Code = code, Message = message, ExitCode = exitCode } },
            CliJsonContext.Default.CliJsonErrorResult));
        return exitCode;
    }
}

/// <summary>Error document printed on stdout by a failed <c>--json</c> command.</summary>
internal sealed record CliJsonErrorResult
{
    public required CliJsonError Error { get; init; }
}

/// <summary>Body of <see cref="CliJsonErrorResult"/>.</summary>
internal sealed record CliJsonError
{
    public required string Code { get; init; }
    public required string Message { get; init; }
    public required int ExitCode { get; init; }
}

/// <summary>JSON output of <c>dotllm ps --json</c>.</summary>
internal sealed record PsJsonResult
{
    public required int Version { get; init; }
    public required string Url { get; init; }
    public required PsJsonModel[] Models { get; init; }
    public string? DeviceWarning { get; init; }
}

/// <summary>One resident model of <c>ps --json</c>. <c>Device</c> is only known for the active model.</summary>
internal sealed record PsJsonModel
{
    public required string Id { get; init; }
    public required bool Active { get; init; }
    public string? Device { get; init; }
    public required long SizeBytes { get; init; }
    public double? IdleSeconds { get; init; }
    /// <summary>Seconds until auto-unload; absent when the model never unloads.</summary>
    public double? ExpiresInSeconds { get; init; }
}

/// <summary>JSON output of <c>dotllm model info --json</c>.</summary>
internal sealed record ModelInfoJsonResult
{
    public required int Version { get; init; }
    public required string Repo { get; init; }
    public string? Author { get; init; }
    public required long Downloads { get; init; }
    public required long Likes { get; init; }
    public string? Pipeline { get; init; }
    public required string[] Tags { get; init; }
    /// <summary>Directory holding the already-downloaded files of this repo; absent when none is local.</summary>
    public string? LocalPath { get; init; }
    public required ModelInfoJsonFile[] GgufFiles { get; init; }
}

/// <summary>One GGUF file of a repository.</summary>
internal sealed record ModelInfoJsonFile
{
    public required string Filename { get; init; }
    public required long SizeBytes { get; init; }
    public required string Quant { get; init; }
    public required bool Local { get; init; }
}
