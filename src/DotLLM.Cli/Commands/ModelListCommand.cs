using System.ComponentModel;
using System.Text.RegularExpressions;
using DotLLM.Cli.Helpers;
using DotLLM.HuggingFace;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands;

/// <summary>
/// Lists locally available GGUF models (dotllm store, Hugging Face hub cache, ollama store) and model profiles.
/// Registered twice: <c>dotllm list</c> and <c>dotllm model list</c>.
/// </summary>
internal sealed partial class ModelListCommand : Command<ModelListCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("--json")]
        [Description("Print one JSON object {\"models\": array, \"profiles\": array} instead of tables (model: repo, filename, quant, size_bytes, path, downloaded_at). Nothing else is written to stdout.")]
        [DefaultValue(false)]
        public bool Json { get; set; }

        [CommandOption("--paths")]
        [Description("Add a Path column (full file path) to the model table. Always present in --json output.")]
        [DefaultValue(false)]
        public bool Paths { get; set; }
    }

    [GeneratedRegex(@"[.\-_]((?:IQ|PQ|TQ|Q)\d\w*|BF16|F16|F32|MXFP4|NVFP4)(?=\.gguf$|-\d{5}-of-\d{5}\.gguf$)", RegexOptions.IgnoreCase)]
    private static partial Regex QuantRegex();

    /// <summary>Quantization label parsed from a GGUF file name (e.g. <c>Q4_K_M</c>), or <c>-</c> when the name carries none.</summary>
    internal static string InferQuant(string filename)
    {
        var m = QuantRegex().Match(filename);
        return m.Success ? m.Groups[1].Value.ToUpperInvariant() : "-";
    }

    public override int Execute(CommandContext context, Settings settings)
    {
        var models = ModelResolver.EnumerateLocal(includeOllama: true);
        var profiles = ModelProfileStore.List();

        if (settings.Json)
        {
            var result = new ModelListJsonResult
            {
                Models = models.OrderByDescending(m => m.DownloadedAt).Select(m => new ModelListJsonModel
                {
                    Repo = m.RepoId, Filename = m.Filename, Quant = InferQuant(m.Filename), SizeBytes = m.SizeBytes,
                    Path = m.FullPath, DownloadedAt = m.DownloadedAt.ToString("O"),
                }).ToArray(),
                Profiles = profiles.Select(p => new ModelListJsonProfile
                {
                    Name = p.Item1, From = p.Item2.From, Device = p.Item2.Device,
                }).ToArray(),
            };
            Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(result, CliJsonContext.Default.ModelListJsonResult));
            return 0;
        }

        if (models.Count == 0 && profiles.Count == 0)
        {
            AnsiConsole.MarkupLine("[yellow]No locally downloaded models found.[/]");
            AnsiConsole.MarkupLine($"[dim]Models directory: {HuggingFaceDownloader.DefaultModelsDirectory.EscapeMarkup()}[/]");
            AnsiConsole.MarkupLine("[dim]Get one with: dotllm model pull <owner/repo>. Models loaded in a running server: dotllm ps.[/]");
            return 0;
        }

        var table = new Table();
        table.Border(TableBorder.Rounded);
        table.AddColumn("Repository");
        table.AddColumn("Filename");
        table.AddColumn("Quant");
        table.AddColumn(new TableColumn("Size").RightAligned());
        table.AddColumn("Downloaded");
        if (settings.Paths) table.AddColumn("Path");

        foreach (var model in models.OrderByDescending(m => m.DownloadedAt))
        {
            var row = new List<string>
            {
                $"[bold]{model.RepoId.EscapeMarkup()}[/]",
                model.Filename.EscapeMarkup(),
                InferQuant(model.Filename).EscapeMarkup(),
                FormatHelpers.FormatSize(model.SizeBytes),
                model.DownloadedAt.LocalDateTime.ToString("yyyy-MM-dd HH:mm"),
            };
            if (settings.Paths) row.Add(model.FullPath.EscapeMarkup());
            table.AddRow(row.ToArray());
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
