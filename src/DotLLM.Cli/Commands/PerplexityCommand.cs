using System.ComponentModel;
using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.Core.Evaluation;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Engine.Evaluation;
using DotLLM.Models;
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
    /// <summary>
    /// Delimiters accepted in a <c>--tokens-file</c>: whitespace plus the punctuation of a
    /// JSON array, so a reference implementation's dump parses as-is.
    /// </summary>
    private static readonly char[] TokenIdSeparators = [' ', '\t', '\r', '\n', ',', '[', ']'];

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
        [Description("Tokens advanced between window starts. 0 selects the context length (non-overlapping chunks, llama.cpp's default).")]
        [DefaultValue(0)]
        public int Stride { get; set; }

        [CommandOption("--unscored-prefix")]
        [Description("Leading tokens of each window used as context only. -1 selects context/2 (llama.cpp's default).")]
        [DefaultValue(-1)]
        public int UnscoredPrefix { get; set; } = -1;

        [CommandOption("--max-tokens|-n")]
        [Description("Cap on corpus tokens consumed. 0 = unbounded.")]
        [DefaultValue(0)]
        public int MaxTokens { get; set; }

        [CommandOption("--mode")]
        [Description("Scoring mode: sliding-window (default, llama.cpp-comparable) or teacher-forced.")]
        [DefaultValue("sliding-window")]
        public string Mode { get; set; } = "sliding-window";

        [CommandOption("--tokens-file")]
        [Description("Read pre-tokenized whitespace-separated ids instead of tokenizing --corpus. Isolates scoring from tokenization when comparing against another implementation.")]
        public string? TokensFile { get; set; }

        [CommandOption("--dump-tokens")]
        [Description("Write the tokenized corpus ids to this path, whitespace-separated, then continue. Diagnostic.")]
        public string? DumpTokens { get; set; }

        [CommandOption("--per-window")]
        [Description("Print each window's perplexity. Use to localize a disagreement with another implementation to specific corpus content.")]
        [DefaultValue(false)]
        public bool PerWindow { get; set; }

        [CommandOption("--bos")]
        [Description("Substitute BOS at the start of each window. Match the model's add_bos setting: llama.cpp only does this when the tokenizer requests it.")]
        [DefaultValue(false)]
        public bool Bos { get; set; }

        [CommandOption("--quant")]
        [Description("Quantization to select when resolving a HuggingFace repo ID.")]
        public string? Quant { get; set; }

        [CommandOption("--device|-d")]
        [Description("Compute device: 'cpu' (default), 'vulkan', 'cuda' / 'cuda:1' ('gpu' is an alias for cuda).")]
        [DefaultValue("cpu")]
        public string Device { get; set; } = "cpu";

        [CommandOption("--threads")]
        [Description("Compute threads. 0 = auto.")]
        [DefaultValue(0)]
        public int Threads { get; set; }
    }

    public override async Task<int> ExecuteAsync(CommandContext context, Settings settings)
    {
        // --tokens-file supplies the token stream directly, so it replaces --corpus rather than
        // supplementing it. Requiring both would defeat the flag's purpose: scoring a reference
        // implementation's exact ids to separate a tokenizer difference from a scoring one.
        if (settings.TokensFile is not null)
        {
            if (!File.Exists(settings.TokensFile))
            {
                AnsiConsole.MarkupLine(
                    $"[red]Tokens file not found: {Markup.Escape(settings.TokensFile)}[/]");
                return 1;
            }
        }
        else if (string.IsNullOrWhiteSpace(settings.Corpus))
        {
            AnsiConsole.MarkupLine("[red]--corpus is required (or --tokens-file).[/]");
            return 1;
        }
        else if (!File.Exists(settings.Corpus))
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

        if (!TryParseDevice(settings.Device, out string backend, out int gpuId))
        {
            AnsiConsole.MarkupLine(
                $"[red]Unknown --device '{Markup.Escape(settings.Device)}'. Expected 'cpu', 'vulkan', 'cuda' or 'cuda:N'.[/]");
            return 1;
        }

        // Both are owned here and released in the finally below. The previous CPU-only code got
        // this from `using TransformerModel`; with a backend switch the model's static type is
        // IModel (itself IDisposable) and Vulkan additionally owns a device handle.
        IModel model;
        IDisposable? ownedDevice = null;
        int forwardDeviceId;
        string deviceLabel;
        switch (backend)
        {
            case "cuda":
            {
                // Shared per-architecture CUDA dispatch — routes hybrid architectures
                // (Qwen3MoeHybrid, Qwen3HybridDense) to their dedicated loaders. The plain
                // CudaTransformerModel loader fails on them with
                // "blk.0.attn_output.weight not present" (issue #259).
                (model, _) = DotLLM.Cuda.CudaModelLoader.CreateFromGguf(gguf, config, gpuId);
                forwardDeviceId = gpuId;
                deviceLabel = DotLLM.Cuda.CudaDevice.GetDevice(gpuId).Name;
                break;
            }

            case "vulkan":
            {
                var vkDevice = DotLLM.Vulkan.VulkanDevice.Create();
                ownedDevice = vkDevice;
                // Shared per-architecture Vulkan dispatch — see the CPU/CUDA branches (#259).
                (model, _) = DotLLM.Vulkan.VulkanModelLoader.CreateFromGguf(
                    vkDevice, gguf, config, ResolveSpvDir());
                forwardDeviceId = 0;
                deviceLabel = vkDevice.DeviceName;
                break;
            }

            default:
            {
                // Shared per-architecture CPU dispatch — routes hybrid architectures
                // (Nemotron-H, Qwen3MoeHybrid, Qwen3HybridDense) to their dedicated loaders
                // rather than the plain TransformerModel, whose tensor naming they do not
                // follow (a GDN layer has no attn_output.weight — issue #259).
                model = ModelLoader.CreateCpuModelFromGguf(
                    gguf, config, new ThreadingConfig(settings.Threads));
                forwardDeviceId = -1;
                deviceLabel = $"cpu-{new ThreadingConfig(settings.Threads).EffectiveThreadCount}t";
                break;
            }
        }

        int effectiveContext = Math.Min(settings.Context, config.MaxSequenceLength);
        // Defaults reproduce llama.cpp: non-overlapping chunks, scoring the second half of each.
        int effectiveStride = settings.Stride > 0 ? settings.Stride : effectiveContext;
        // context/2 + 1, not context/2: llama.cpp scores targets (n_ctx/2, n_ctx), leaving the token
        // at n_ctx/2 as context only. See PerplexityOptions.LlamaCppDefault.
        int effectivePrefix = settings.UnscoredPrefix >= 0
            ? settings.UnscoredPrefix
            : Math.Min(effectiveContext - 1, Math.Max(1, effectiveContext / 2 + 1));

        // Streamed, then buffered once: scoring needs random access across windows, but the file
        // itself is never held in memory and the token list is bounded by --max-tokens.
        var tokens = new List<int>();
        if (settings.TokensFile is not null)
        {
            // Accept both bare whitespace-separated ids and the JSON-array form that reference
            // tools print, so a dump can be pasted in without reformatting.
            foreach (string part in File.ReadAllText(settings.TokensFile)
                         .Split(TokenIdSeparators, StringSplitOptions.RemoveEmptyEntries))
            {
                tokens.Add(int.Parse(part));
                if (settings.MaxTokens > 0 && tokens.Count >= settings.MaxTokens) break;
            }
        }
        else
        {
            using var reader = new StreamReader(settings.Corpus);
            foreach (int id in CorpusReader.StreamTokens(reader, tokenizer, settings.MaxTokens))
                tokens.Add(id);
        }

        if (tokens.Count < 2)
        {
            AnsiConsole.MarkupLine($"[red]Corpus tokenized to {tokens.Count} tokens; at least 2 are required.[/]");
            return 1;
        }

        if (settings.DumpTokens is not null)
            File.WriteAllText(settings.DumpTokens, string.Join(' ', tokens));

        try
        {
        // Probed, not assumed: the CPU transformer returns [seqLen, vocab] but the CUDA model
        // returns only the final row, and that difference silently changes the perplexity rather
        // than raising. See BackendPerplexityModel.Probe.
        bool returnsAllRows = BackendPerplexityModel.Probe(model, forwardDeviceId);
        var perplexityModel = new BackendPerplexityModel(model, forwardDeviceId, returnsAllRows);
        AnsiConsole.MarkupLine(
            $"[grey]device: {Markup.Escape(deviceLabel)}  all-rows logits: {returnsAllRows} "
            + $"({(returnsAllRows ? "single-pass O(n)" : "growing-prefix O(n^2)")})[/]");
        int bosTokenId = settings.Bos ? tokenizer.BosTokenId : -1;
        var options = new PerplexityOptions(
            mode, effectiveContext, effectiveStride, settings.MaxTokens, effectivePrefix, bosTokenId);

        var sw = Stopwatch.StartNew();
        PerplexityResult result;
        try
        {
            PerplexityEvaluator.WindowObserver? observer = settings.PerWindow
                ? (i, ppl, n) => Console.WriteLine($"window {i}: ppl={ppl:F6} scored={n}")
                : null;
            result = PerplexityEvaluator.Evaluate(
                perplexityModel,
                System.Runtime.InteropServices.CollectionsMarshal.AsSpan(tokens),
                options,
                observer);
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
        // Printed as "PPL +/- err" in llama.cpp's own format so the two can be compared by eye.
        // Without the error bar a reader has no way to tell a regression from sampling noise.
        table.AddRow("Perplexity", $"{result.Perplexity:F4} +/- {result.StandardError:F5}");
        table.AddRow("Mean NLL (nats)", $"{result.MeanNegativeLogLikelihood:F6}");
        table.AddRow("Scored tokens", $"{result.ScoredTokens:N0}");
        table.AddRow("Windows", $"{result.WindowCount:N0}");
        table.AddRow("Mode", mode == PerplexityMode.SlidingWindow ? "sliding-window" : "teacher-forced");
        table.AddRow("Context", $"{effectiveContext:N0}");
        table.AddRow("Stride", $"{effectiveStride:N0}");
        table.AddRow("Unscored prefix", $"{effectivePrefix:N0}");
        table.AddRow("Corpus tokens", $"{tokens.Count:N0}");
        table.AddRow("Elapsed", $"{sw.Elapsed.TotalSeconds:F2} s");
        AnsiConsole.Write(table);

        await Task.CompletedTask;
        return 0;
        }
        finally
        {
            model.Dispose();
            ownedDevice?.Dispose();
        }
    }

    /// <summary>
    /// Parses <c>--device</c> into a backend name and GPU ordinal, mirroring <c>bench</c>'s syntax so
    /// the two commands accept the same strings (<c>cpu</c>, <c>vulkan</c>, <c>cuda</c>, <c>cuda:1</c>,
    /// and <c>gpu</c> as an alias for cuda).
    /// </summary>
    /// <param name="device">Raw option value.</param>
    /// <param name="backend">Normalized backend name.</param>
    /// <param name="gpuId">Device ordinal; 0 unless an explicit <c>:N</c> suffix is given.</param>
    /// <returns><see langword="true"/> when the value names a supported backend.</returns>
    private static bool TryParseDevice(string device, out string backend, out int gpuId)
    {
        backend = (device ?? "cpu").Split(':')[0].ToLowerInvariant();
        if (backend.Length == 0) backend = "cpu";
        if (backend == "gpu") backend = "cuda";

        gpuId = 0;
        string[] parts = (device ?? string.Empty).Split(':');
        if (parts.Length > 1 && int.TryParse(parts[1], out int ordinal) && ordinal >= 0)
            gpuId = ordinal;

        return backend is "cpu" or "cuda" or "vulkan";
    }

    /// <summary>
    /// Resolves the SPIR-V blob directory for Vulkan: <c>spv/</c> next to the running assembly,
    /// falling back to the in-repo <c>native/vulkan/spv</c> when running from the source tree.
    /// </summary>
    /// <returns>Absolute path to a directory containing compiled SPIR-V.</returns>
    private static string ResolveSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (string c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        throw new InvalidOperationException(
            "Vulkan SPIR-V directory not found. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");
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
