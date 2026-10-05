using System.ComponentModel;
using DotLLM.HuggingFace;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands;

/// <summary>
/// Downloads a GGUF file from a HuggingFace repository.
/// </summary>
internal sealed class ModelPullCommand : AsyncCommand<ModelPullCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<repo-id>")]
        [Description("HuggingFace repository ID (e.g. 'TheBloke/Llama-2-7B-GGUF').")]
        public string RepoId { get; set; } = string.Empty;

        [CommandOption("--file|-f")]
        [Description("Specific GGUF filename to download. If omitted, lists available files for selection.")]
        public string? Filename { get; set; }

        [CommandOption("--dir|-d")]
        [Description("Destination directory. Defaults to ~/.dotllm/models/.")]
        public string? Directory { get; set; }
    }

    public override async Task<int> ExecuteAsync(CommandContext context, Settings settings)
    {
        using var client = new HuggingFaceClient();
        using var downloader = new HuggingFaceDownloader();

        var filename = settings.Filename;
        var reference = ModelResolver.Parse(settings.RepoId);
        string repoId = reference.RepoId ?? settings.RepoId;
        if (filename is null && reference.Filename is not null) filename = reference.Filename;
        // "owner/repo:Q4_K_M" picks the matching file without a prompt.
        if (string.IsNullOrEmpty(filename) && reference.Tag is not null)
        {
            var listing = await client.ListGgufFilesAsync(repoId);
            filename = ModelResolver.ChooseRemoteFile(listing.Select(f => (f.Path, f.Size)), reference.Tag);
            if (filename is null)
            {
                AnsiConsole.MarkupLine($"[red]No GGUF matching '{reference.Tag.EscapeMarkup()}' in {repoId.EscapeMarkup()}.[/]");
                return 1;
            }
        }

        // If no filename specified, list GGUF files and let user pick
        if (string.IsNullOrEmpty(filename))
        {
            var ggufFiles = await AnsiConsole.Status()
                .StartAsync("Fetching file list...", async _ =>
                    await client.ListGgufFilesAsync(repoId));

            if (ggufFiles.Count == 0)
            {
                AnsiConsole.MarkupLine("[red]No GGUF files found in repository.[/]");
                return 1;
            }

            filename = AnsiConsole.Prompt(
                new SelectionPrompt<string>()
                    .Title("Select a GGUF file to download:")
                    .AddChoices(ggufFiles.Select(f => f.Path)));
        }

        AnsiConsole.MarkupLine($"Downloading [bold]{filename.EscapeMarkup()}[/] from [bold]{repoId.EscapeMarkup()}[/]...");

        var path = await AnsiConsole.Progress()
            .AutoClear(false)
            .Columns(
                new TaskDescriptionColumn(),
                new ProgressBarColumn(),
                new PercentageColumn(),
                new TransferSpeedColumn(),
                new RemainingTimeColumn())
            .StartAsync(async ctx =>
            {
                var task = ctx.AddTask($"[green]{filename.EscapeMarkup()}[/]", maxValue: 100);
                long? lastTotal = null;

                // Ticks are posted to the thread pool, so a stale one can land after the download
                // returns and leave the bar short — cosmetic only here, unlike the server job state
                // this same pattern corrupted (#521); the bar is torn down on return either way.
                var progress = new Progress<(long bytesDownloaded, long? totalBytes)>(p =>
                {
                    if (p.totalBytes.HasValue)
                    {
                        if (lastTotal != p.totalBytes.Value)
                        {
                            task.MaxValue = p.totalBytes.Value;
                            lastTotal = p.totalBytes.Value;
                        }
                        task.Value = p.bytesDownloaded;
                    }
                });

                // Hub cache (shared with huggingface_hub / hf download) + a mirror link; --dir keeps the old flat-directory behaviour.
                if (settings.Directory is not null)
                    return await downloader.DownloadFileAsync(repoId, filename, settings.Directory, progress);
                var r = await downloader.DownloadToHubCacheAsync(repoId, filename, progress: progress);
                return r.ModelPath;
            });

        AnsiConsole.MarkupLine($"[green]Saved to:[/] {path.EscapeMarkup()}");
        return 0;
    }
}
