using System.ComponentModel;
using DotLLM.Cli.Helpers;
using DotLLM.HuggingFace;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands;

/// <summary>
/// Deletes a locally downloaded GGUF model file.
/// </summary>
internal sealed class ModelDeleteCommand : Command<ModelDeleteCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<model>")]
        [Description("HuggingFace repo ID (e.g. 'Qwen/Qwen2.5-0.5B-Instruct-GGUF') or filename pattern.")]
        public string Model { get; set; } = string.Empty;

        [CommandOption("--quant|-q")]
        [Description("Quantization filter (e.g. Q4_K_M, Q8_0). Narrows match when repo has multiple files.")]
        public string? Quant { get; set; }

        [CommandOption("--yes|-y")]
        [Description("Skip confirmation prompt.")]
        [DefaultValue(false)]
        public bool Yes { get; set; }
    }

    public override int Execute(CommandContext context, Settings settings)
    {
        var models = ModelResolver.EnumerateLocal();

        // Find matches by repo ID or filename
        var matches = models.Where(m =>
            m.RepoId.Contains(settings.Model, StringComparison.OrdinalIgnoreCase) ||
            m.Filename.Contains(settings.Model, StringComparison.OrdinalIgnoreCase)).ToList();

        // Apply quant filter
        if (settings.Quant is not null)
        {
            matches = matches.Where(m =>
                m.Filename.Contains(settings.Quant, StringComparison.OrdinalIgnoreCase)).ToList();
        }

        if (matches.Count == 0)
        {
            AnsiConsole.MarkupLine($"[yellow]No matching models found for '{settings.Model.EscapeMarkup()}'.[/]");
            return 1;
        }

        // Show what will be deleted
        var table = new Table();
        table.Border(TableBorder.Rounded);
        table.AddColumn("Repository");
        table.AddColumn("Filename");
        table.AddColumn(new TableColumn("Size").RightAligned());

        foreach (var m in matches)
        {
            table.AddRow(
                m.RepoId.EscapeMarkup(),
                m.Filename.EscapeMarkup(),
                FormatHelpers.FormatSize(m.SizeBytes));
        }

        AnsiConsole.Write(table);

        // Confirm
        if (!settings.Yes)
        {
            var label = matches.Count == 1 ? "this model" : $"these {matches.Count} models";
            if (!AnsiConsole.Confirm($"Delete {label}?", defaultValue: false))
            {
                AnsiConsole.MarkupLine("[dim]Cancelled.[/]");
                return 0;
            }
        }

        // Remove every link (mirror + hub-cache snapshot) and the blob when unambiguous.
        foreach (var m in matches)
        {
            try
            {
                long freed = ModelResolver.DeleteLocal(m);
                AnsiConsole.MarkupLine(freed > 0
                    ? $"[red]Deleted[/] {m.Filename.EscapeMarkup()} ({FormatHelpers.FormatSize(freed)})"
                    : $"[yellow]Nothing removed for[/] {m.Filename.EscapeMarkup()}");
            }
            catch (Exception ex)
            {
                AnsiConsole.MarkupLine($"[red]Failed to delete {m.Filename.EscapeMarkup()}: {ex.Message.EscapeMarkup()}[/]");
            }
        }

        return 0;
    }
}
