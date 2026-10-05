using DotLLM.HuggingFace;
using Spectre.Console;

namespace DotLLM.Cli.Commands;

/// <summary>
/// Resolves a GGUF file argument to a local file path. Accepts either a direct file path,
/// a HuggingFace repo ID (e.g., "Qwen/Qwen3-0.6B-GGUF"), or a repo ID with filename
/// (e.g., "bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q8_0.gguf").
/// </summary>
internal static class GgufFileResolver
{
    /// <summary>
    /// Resolves the argument to a local .gguf file path via <see cref="ModelResolver"/>: a path, <c>owner/repo[:tag]</c>,
    /// <c>owner/repo/file.gguf</c>, <c>hf.co/...</c> or a bare local name; models in the Hugging Face hub cache are found too.
    /// When nothing local matches and <paramref name="allowPull"/> is set, a Hub reference is downloaded (resumable, with progress).
    /// Returns null and prints an error if resolution fails.
    /// </summary>
    /// <param name="fileArg">File path, repo ID (optionally with :tag or /filename), or local model name.</param>
    /// <param name="quant">Optional quantization filter (e.g., "Q8_0"); equivalent to a <c>:tag</c>.</param>
    /// <param name="allowPull">Download a missing Hub model instead of failing (ollama-style <c>run</c> behaviour).</param>
    public static string? Resolve(string fileArg, string? quant = null, bool allowPull = false)
    {
        if (File.Exists(fileArg))
            return Path.GetFullPath(fileArg);

        string? local = ModelResolver.ResolveLocal(fileArg, quant);
        if (local is not null)
            return local;

        var reference = ModelResolver.Parse(fileArg);
        if (reference.IsRepo && allowPull)
        {
            try
            {
                return PullWithProgress(reference, quant);
            }
            catch (Exception ex) when (ex is HttpRequestException or InvalidOperationException or IOException)
            {
                AnsiConsole.MarkupLine($"[red]Could not pull {fileArg.EscapeMarkup()}:[/] {ex.Message.EscapeMarkup()}");
                return null;
            }
        }

        AnsiConsole.MarkupLine($"[red]Model not found locally:[/] {fileArg.EscapeMarkup()}");
        AnsiConsole.MarkupLine(reference.IsRepo
            ? $"[grey]Download it with:[/] dotllm model pull {fileArg.EscapeMarkup()}"
            : "[grey]Pass a .gguf path, an 'owner/repo[:quant]' Hugging Face reference, or the name of a model from 'dotllm model list'.[/]");
        return null;
    }

    private static string PullWithProgress(ModelReference reference, string? quant)
    {
        using var client = new HuggingFaceClient();
        using var downloader = new HuggingFaceDownloader();
        string tag = (quant ?? reference.Tag) is { } t ? ":" + t : "";
        AnsiConsole.MarkupLine($"[grey]Not found locally - pulling[/] [bold]{(reference.RepoId + tag).EscapeMarkup()}[/]");
        return AnsiConsole.Progress()
            .AutoClear(false)
            .Columns(new TaskDescriptionColumn(), new ProgressBarColumn(), new PercentageColumn(), new TransferSpeedColumn(), new RemainingTimeColumn())
            .Start(ctx =>
            {
                var task = ctx.AddTask("[green]downloading[/]", maxValue: 100);
                long? lastTotal = null;
                var progress = new Progress<(long bytesDownloaded, long? totalBytes)>(p =>
                {
                    if (!p.totalBytes.HasValue) return;
                    if (lastTotal != p.totalBytes.Value) { task.MaxValue = p.totalBytes.Value; lastTotal = p.totalBytes.Value; }
                    task.Value = p.bytesDownloaded;
                });
                return ModelResolver.PullAsync(reference, quant, client, downloader, progress, CancellationToken.None).GetAwaiter().GetResult();
            });
    }

    /// <summary>
    /// Tries to resolve an "owner/repo/file.gguf" path to a local file.
    /// Returns null if the argument doesn't match this form or the file doesn't exist.
    /// </summary>
    private static string? TryResolveRepoFile(string modelsDir, string fileArg)
    {
        // Need at least 3 segments: owner / repo / filename
        var parts = fileArg.Split('/');
        if (parts.Length < 3)
            return null;

        // owner/repo is the first two segments, the rest is the filename (could contain /).
        string owner = parts[0];
        string repo = parts[1];
        string fileName = string.Join(Path.DirectorySeparatorChar.ToString(), parts[2..]);

        string candidatePath = Path.Combine(modelsDir, owner, repo, fileName);
        return File.Exists(candidatePath) ? Path.GetFullPath(candidatePath) : null;
    }
}
