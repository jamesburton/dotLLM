using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace DotLLM.ChunkCache;

/// <summary>
/// Where a verb writes and how it resolves paths. With <c>--json</c> the result object is the only thing on
/// <see cref="Out"/>; progress and warnings go to <see cref="Err"/>, so a wrapper (e.g. an MCP server) can parse stdout.
/// </summary>
internal sealed class CliContext(TextWriter output, TextWriter error, string workingDirectory)
{
    /// <summary>JSON options of every result: snake_case names (the MCP tool schemas' names), nulls omitted, indented.</summary>
    public static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = true,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.SnakeCaseLower) },
    };

    /// <summary>Standard output.</summary>
    public TextWriter Out { get; } = output;

    /// <summary>Standard error.</summary>
    public TextWriter Err { get; } = error;

    /// <summary>Directory relative paths are resolved against.</summary>
    public string WorkingDirectory { get; } = Path.GetFullPath(workingDirectory);

    /// <summary>Whether <c>--json</c> was given.</summary>
    public bool Json { get; set; }

    /// <summary>The HTTP handler of server calls (null: the default socket handler); tests serve contract fixtures through it.</summary>
    public HttpMessageHandler? HttpHandler { get; init; }

    /// <summary>Reads an environment variable (the API key is read here, nowhere else); tests inject their own.</summary>
    public Func<string, string?> GetEnv { get; init; } = Environment.GetEnvironmentVariable;

    /// <summary>
    /// The text with the API key (if one is set) replaced by <c>***</c>. Every message the CLI prints (stderr and the JSON error
    /// object) goes through this, so a server that echoes the key back in an error body cannot leak it into logs.
    /// </summary>
    public string Redact(string text)
    {
        string? key = GetEnv(ServerConnection.ApiKeyVariable);
        return string.IsNullOrEmpty(key) || string.IsNullOrEmpty(text) ? text : text.Replace(key, "***", StringComparison.Ordinal);
    }

    /// <summary>Human-readable result text: stdout normally, stderr under <c>--json</c> (stdout carries only the JSON).</summary>
    public TextWriter Text => Json ? Err : Out;

    /// <summary>An absolute path for <paramref name="path"/> (relative to <see cref="WorkingDirectory"/>).</summary>
    public string FullPath(string path) => Path.GetFullPath(Path.IsPathFullyQualified(path) ? path : Path.Combine(WorkingDirectory, path));

    /// <summary>Progress or a warning line (stderr).</summary>
    public void Info(string line) => Err.WriteLine(Redact(line));

    /// <summary>Writes <paramref name="result"/> as the JSON document on stdout.</summary>
    public void WriteJson<T>(T result) => Out.WriteLine(Redact(JsonSerializer.Serialize(result, JsonOptions)));
}
