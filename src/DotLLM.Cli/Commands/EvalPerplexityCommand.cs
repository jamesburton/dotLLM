using System.ComponentModel;
using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.Engine.Evaluation;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands;

/// <summary>
/// Evaluates language-model perplexity on a tokenized text corpus: loads a GGUF model on the
/// CPU backend, runs forward over non-overlapping context windows, and reports
/// perplexity = exp(mean per-token negative log-likelihood), along with mean NLL and token counts.
/// </summary>
/// <remarks>
/// Perplexity is the product-quality metric used to compare matmul/prefill kernel variants
/// (e.g. a BF16 Q8_0 path vs the exact-int path): run the same corpus through each configuration
/// and compare the resulting numbers. The harness runs whatever compute path the loaded model
/// is configured with — it does not select a kernel itself.
/// </remarks>
internal sealed class EvalPerplexityCommand : Command<EvalPerplexityCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<model>")]
        [Description("Path to a GGUF file or HuggingFace repo ID (e.g., QuantFactory/SmolLM-135M-GGUF).")]
        public string Model { get; set; } = string.Empty;

        [CommandOption("--text")]
        [Description("Path to a UTF-8 text file to evaluate perplexity over.")]
        public string? TextFile { get; set; }

        [CommandOption("--text-inline")]
        [Description("Inline text to evaluate (alternative to --text for short corpora).")]
        public string? TextInline { get; set; }

        [CommandOption("--max-window")]
        [Description("Maximum context window length in tokens. 0 = model's full context length (default). "
                     + "Each window allocates a [window x vocab] logits tensor and runs O(window^2) "
                     + "attention, so lower this on large-context models to cap memory and time.")]
        [DefaultValue(0)]
        public int MaxWindow { get; set; }

        [CommandOption("--quant|-q")]
        [Description("Quantization filter when multiple GGUF files exist (e.g., Q8_0, Q4_K_M).")]
        public string? Quant { get; set; }

        [CommandOption("--threads")]
        [Description("Number of CPU threads for inference. 0 = auto/all cores (default), 1 = single-threaded.")]
        [DefaultValue(0)]
        public int Threads { get; set; }
    }

    public override int Execute(CommandContext context, Settings settings)
    {
        // Resolve the corpus text up front so we fail fast before loading weights.
        string? corpus = ResolveCorpus(settings);
        if (corpus is null)
        {
            AnsiConsole.MarkupLine("[red]Provide exactly one of --text <file> or --text-inline <text>.[/]");
            return 1;
        }
        if (corpus.Length == 0)
        {
            AnsiConsole.MarkupLine("[red]The supplied corpus is empty.[/]");
            return 1;
        }

        var resolvedPath = GgufFileResolver.Resolve(settings.Model, settings.Quant);
        if (resolvedPath is null)
            return 1;

        var loadSw = Stopwatch.StartNew();
        using var gguf = GgufFile.Open(resolvedPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(
            gguf, config, new ThreadingConfig(settings.Threads));
        loadSw.Stop();

        int[] tokenIds = tokenizer.Encode(corpus);
        if (tokenIds.Length < 2)
        {
            AnsiConsole.MarkupLine(
                $"[red]Corpus tokenized to {tokenIds.Length} token(s); need at least 2 to score perplexity.[/]");
            return 1;
        }

        AnsiConsole.Write(new Rule(
            $"[grey]dotllm eval perplexity | {config.Architecture} {config.NumLayers}L | "
            + $"ctx={config.MaxSequenceLength} | {tokenIds.Length} tokens[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var evalSw = Stopwatch.StartNew();
        PerplexityEvaluator.PerplexityResult result = default;
        AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .Start("Computing perplexity...", _ =>
            {
                result = PerplexityEvaluator.Evaluate(model, tokenIds, settings.MaxWindow);
            });
        evalSw.Stop();

        var table = new Table().Border(TableBorder.Rounded);
        table.AddColumn("Metric");
        table.AddColumn("Value");
        table.AddRow("Perplexity", $"{result.Perplexity:F4}");
        table.AddRow("Mean NLL (nats)", $"{result.MeanNll:F6}");
        table.AddRow("Scored tokens", result.ScoredTokenCount.ToString("N0"));
        table.AddRow("Total tokens", result.TotalTokenCount.ToString("N0"));
        table.AddRow("Load time", $"{loadSw.Elapsed.TotalMilliseconds:F0} ms");
        table.AddRow("Eval time", $"{evalSw.Elapsed.TotalSeconds:F2} s");
        AnsiConsole.Write(table);

        return 0;
    }

    /// <summary>
    /// Resolves the corpus text from either <c>--text</c> (file) or <c>--text-inline</c>.
    /// Returns null when neither or both are supplied (caller treats null as a usage error).
    /// </summary>
    private static string? ResolveCorpus(Settings settings)
    {
        bool hasFile = !string.IsNullOrEmpty(settings.TextFile);
        bool hasInline = !string.IsNullOrEmpty(settings.TextInline);
        if (hasFile == hasInline)
            return null; // neither or both

        if (hasFile)
        {
            if (!File.Exists(settings.TextFile))
            {
                AnsiConsole.MarkupLine($"[red]Text file not found:[/] {settings.TextFile.EscapeMarkup()}");
                return null;
            }
            return File.ReadAllText(settings.TextFile!);
        }

        return settings.TextInline!;
    }
}
