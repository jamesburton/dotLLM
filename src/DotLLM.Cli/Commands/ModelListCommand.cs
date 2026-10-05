using DotLLM.Cli.Helpers;
using DotLLM.HuggingFace;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands;

/// <summary>
/// Lists locally downloaded GGUF models.
/// </summary>
internal sealed class ModelListCommand : Command<ModelListCommand.Settings>
{
    public sealed class Settings : CommandSettings;

    public override int Execute(CommandContext context, Settings settings)
    {
        var models = ModelResolver.EnumerateLocal();
        var profiles = ModelProfileStore.List();

        if (models.Count == 0 && profiles.Count == 0)
        {
            AnsiConsole.MarkupLine("[yellow]No locally downloaded models found.[/]");
            AnsiConsole.MarkupLine($"[dim]Models directory: {HuggingFaceDownloader.DefaultModelsDirectory.EscapeMarkup()}[/]");
            return 0;
        }

        var table = new Table();
        table.Border(TableBorder.Rounded);
        table.AddColumn("Repository");
        table.AddColumn("Filename");
        table.AddColumn(new TableColumn("Size").RightAligned());
        table.AddColumn("Downloaded");

        foreach (var model in models.OrderByDescending(m => m.DownloadedAt))
        {
            table.AddRow(
                $"[bold]{model.RepoId.EscapeMarkup()}[/]",
                model.Filename.EscapeMarkup(),
                FormatHelpers.FormatSize(model.SizeBytes),
                model.DownloadedAt.LocalDateTime.ToString("yyyy-MM-dd HH:mm"));
        }

        if (models.Count > 0) AnsiConsole.Write(table);

        if (profiles.Count > 0)
        {
            var pt = new Table().Border(TableBorder.Rounded).Title("Profiles (dotllm model create)");
            pt.AddColumn("Name"); pt.AddColumn("From"); pt.AddColumn("Settings");
            foreach (var (name, p) in profiles)
            {
                var bits = new List<string>();
                if (p.System is not null) bits.Add("system");
                if (p.Temperature is { } t) bits.Add($"temp {t}");
                if (p.Device is not null) bits.Add(p.Device);
                if (p.MaxTokens is { } mt) bits.Add($"max {mt}");
                if (p.KeepAlive is { } ka) bits.Add($"keep-alive {ka}s");
                pt.AddRow($"[bold]{name.EscapeMarkup()}[/]", (p.From ?? "").EscapeMarkup(), string.Join(", ", bits).EscapeMarkup());
            }
            AnsiConsole.Write(pt);
        }
        return 0;
    }

}
