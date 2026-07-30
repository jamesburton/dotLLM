using System.ComponentModel;
using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.Core.Evaluation;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Engine.Evaluation;
using DotLLM.Models.Architectures;
using DotLLM.Models.Evaluation;
using DotLLM.Models.Gguf;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands;

/// <summary>
/// Computes perplexity over a text corpus: load → stream-tokenize → score.
/// </summary>
/// <remarks>
/// Defaults to <see cref="PerplexityMode.SlidingWindow"/> with <c>stride = context / 2</c>, which
/// reproduces llama.cpp's <c>--perplexity</c> methodology, so the reported figure is directly
/// comparable to published numbers for the same model, corpus, context and stride.
/// </remarks>
internal sealed class PerplexityCommand : AsyncCommand<PerplexityCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<model>")]
        [Description("Path to a GGUF file or HuggingFace repo ID (e.g., QuantFactory/SmolLM-135M-GGUF).")]
        public string Model { get; set; } = string.Empty;

        [CommandOption("--corpus|-f")]
        [Description("Path to a UTF-8 text corpus (e.g. wiki.test.raw).")]
        public string Corpus { get; set; } = string.Empty;

        [CommandOption("--context|-c")]
        [Description("Context window in tokens. Clamped to the model's maximum sequence length.")]
        [DefaultValue(512)]
        public int Context { get; set; } = 512;

        [CommandOption("--stride")]
        [Description("Tokens advanced between windows. 0 selects context/2, matching llama.cpp's default.")]
        [DefaultValue(0)]
        public int Stride { get; set; }

        [CommandOption("--max-tokens|-n")]
        [Description("Cap on corpus tokens consumed. 0 = unbounded.")]
        [DefaultValue(0)]
        public int MaxTokens { get; set; }

        [CommandOption("--mode")]
        [Description("Scoring mode: sliding-window (default, llama.cpp-comparable) or teacher-forced.")]
        [DefaultValue("sliding-window")]
        public string Mode { get; set; } = "sliding-window";

        [CommandOption("--quant")]
        [Description("Quantization to select when resolving a HuggingFace repo ID.")]
        public string? Quant { get; set; }

        [CommandOption("--threads")]
        [Description("Compute threads. 0 = auto.")]
        [DefaultValue(0)]
        public int Threads { get; set; }
    }

    public override async Task<int> ExecuteAsync(CommandContext context, Settings settings)
    {
        if (string.IsNullOrWhiteSpace(settings.Corpus))
        {
            AnsiConsole.MarkupLine("[red]--corpus is required.[/]");
            return 1;
        }

        if (!File.Exists(settings.Corpus))
        {
            AnsiConsole.MarkupLine($"[red]Corpus not found: {Markup.Escape(settings.Corpus)}[/]");
            return 1;
        }

        if (!TryParseMode(settings.Mode, out PerplexityMode mode))
        {
            AnsiConsole.MarkupLine(
                $"[red]Unknown --mode '{Markup.Escape(settings.Mode)}'. Expected 'sliding-window' or 'teacher-forced'.[/]");
            return 1;
        }

        string? resolvedPath = GgufFileResolver.Resolve(settings.Model, settings.Quant);
        if (resolvedPath is null)
            return 1;

        using GgufFile gguf = GgufFile.Open(resolvedPath);
        ModelConfig config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using TransformerModel model = TransformerModel.LoadFromGguf(
            gguf, config, new ThreadingConfig(settings.Threads));

        int effectiveContext = Math.Min(settings.Context, config.MaxSequenceLength);
        int effectiveStride = settings.Stride > 0 ? settings.Stride : Math.Max(1, effectiveContext / 2);

        // Streamed, then buffered once: scoring needs random access across windows, but the file
        // itself is never held in memory and the token list is bounded by --max-tokens.
        var tokens = new List<int>();
        using (var reader = new StreamReader(settings.Corpus))
        {
            foreach (int id in CorpusReader.StreamTokens(reader, tokenizer, settings.MaxTokens))
                tokens.Add(id);
        }

        if (tokens.Count < 2)
        {
            AnsiConsole.MarkupLine($"[red]Corpus tokenized to {tokens.Count} tokens; at least 2 are required.[/]");
            return 1;
        }

        var perplexityModel = new TransformerPerplexityModel(model, deviceId: -1);
        var options = new PerplexityOptions(mode, effectiveContext, effectiveStride, settings.MaxTokens);

        var sw = Stopwatch.StartNew();
        PerplexityResult result;
        try
        {
            result = PerplexityEvaluator.Evaluate(perplexityModel, System.Runtime.InteropServices.CollectionsMarshal.AsSpan(tokens), options);
        }
        catch (ArgumentException ex)
        {
            AnsiConsole.MarkupLine($"[red]{Markup.Escape(ex.Message)}[/]");
            return 1;
        }
        sw.Stop();

        // Window geometry and scored-token count are reported alongside the figure deliberately:
        // a perplexity without them is not comparable to anything.
        var table = new Table().Border(TableBorder.Rounded);
        table.AddColumn("Metric");
        table.AddColumn(new TableColumn("Value").RightAligned());
        table.AddRow("Perplexity", $"{result.Perplexity:F4}");
        table.AddRow("Mean NLL (nats)", $"{result.MeanNegativeLogLikelihood:F6}");
        table.AddRow("Scored tokens", $"{result.ScoredTokens:N0}");
        table.AddRow("Windows", $"{result.WindowCount:N0}");
        table.AddRow("Mode", mode == PerplexityMode.SlidingWindow ? "sliding-window" : "teacher-forced");
        table.AddRow("Context", $"{effectiveContext:N0}");
        table.AddRow("Stride", $"{effectiveStride:N0}");
        table.AddRow("Corpus tokens", $"{tokens.Count:N0}");
        table.AddRow("Elapsed", $"{sw.Elapsed.TotalSeconds:F2} s");
        AnsiConsole.Write(table);

        await Task.CompletedTask;
        return 0;
    }

    private static bool TryParseMode(string value, out PerplexityMode mode)
    {
        switch (value.Trim().ToLowerInvariant())
        {
            case "sliding-window":
            case "sliding":
                mode = PerplexityMode.SlidingWindow;
                return true;
            case "teacher-forced":
            case "teacher":
                mode = PerplexityMode.TeacherForced;
                return true;
            default:
                mode = default;
                return false;
        }
    }
}
