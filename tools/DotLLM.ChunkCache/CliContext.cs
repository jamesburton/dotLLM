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

    /// <summary>Human-readable result text: stdout normally, stderr under <c>--json</c> (stdout carries only the JSON).</summary>
    public TextWriter Text => Json ? Err : Out;

    /// <summary>An absolute path for <paramref name="path"/> (relative to <see cref="WorkingDirectory"/>).</summary>
    public string FullPath(string path) => Path.GetFullPath(Path.IsPathFullyQualified(path) ? path : Path.Combine(WorkingDirectory, path));

    /// <summary>Progress or a warning line (stderr).</summary>
    public void Info(string line) => Err.WriteLine(line);

    /// <summary>Writes <paramref name="result"/> as the JSON document on stdout.</summary>
    public void WriteJson<T>(T result) => Out.WriteLine(JsonSerializer.Serialize(result, JsonOptions));
}
