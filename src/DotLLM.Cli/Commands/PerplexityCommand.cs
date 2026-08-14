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

        [CommandOption("--gpu-layers")]
        [Description("Number of transformer layers to offload to GPU. 0 = CPU only. " +
                     "Omit for default (0 with --device cpu, all with a GPU device).")]
        public int? GpuLayers { get; set; }

        [CommandOption("--first-layer")]
        [Description("First global layer index of the GPU window, so the offloaded block is an arbitrary " +
                     "contiguous range rather than a prefix. Requires --gpu-layers; incompatible with --cycle.")]
        [DefaultValue(0)]
        public int FirstLayer { get; set; }

        [CommandOption("--cycle")]
        [Description("Layer cycling: slide a --gpu-layers-wide GPU window across the whole trunk within one " +
                     "corpus pass, checkpointing boundary activations, so every layer is GPU-executed even " +
                     "when the model does not fit on the device.")]
        [DefaultValue(false)]
        public bool Cycle { get; set; }
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

        // Layer placement is resolved before anything is loaded, because it decides WHICH loader
        // runs: a windowed or cycled run never builds a whole-device model at all.
        int numLayers = config.NumLayers;
        int gpuLayers = settings.GpuLayers.HasValue
            ? Math.Clamp(settings.GpuLayers.Value, 0, numLayers)
            // Same default as run/chat/serve: CPU device means no offload, a GPU device means all of it.
            : (backend == "cpu" ? 0 : numLayers);

        // run/chat treat an explicit --gpu-layers as the request that decides the backend, not a
        // modifier on --device: `--device cpu --gpu-layers 8` offloads 8 layers there too. Matching
        // that here is what "semantics matching run/chat/serve" means.
        if (backend == "cpu" && gpuLayers > 0)
            backend = "cuda";

        if (!TryResolvePlacement(settings, backend, gpuLayers, numLayers, out LayerPlacement placement))
            return 1;

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

        int bosTokenId = settings.Bos ? tokenizer.BosTokenId : -1;
        var options = new PerplexityOptions(
            mode, effectiveContext, effectiveStride, settings.MaxTokens, effectivePrefix, bosTokenId);

        if (placement.UsesLayerWindows)
        {
            return RunLayerWindowed(
                gguf, config, settings, options, tokens, placement, gpuId,
                effectiveContext, effectiveStride, effectivePrefix, mode);
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
            case "cuda" when gpuLayers > 0 && gpuLayers < numLayers:
            {
                // Prefix offload — GPU runs [0..gpuLayers), CPU runs the rest. Same dispatch and
                // same semantics as run/chat/serve, including the per-architecture routing that
                // issue #291 added for Qwen3HybridDense's interleaved GDN / attention layers (the
                // generic splitter assumes a uniform per-layer tensor-name set and throws on them).
                var hybridThreading = new ThreadingConfig(settings.Threads);
                model = config.Architecture == DotLLM.Core.Configuration.Architecture.Qwen3HybridDense
                    ? DotLLM.Cuda.Architectures.HybridQwen3HybridDenseTransformerModel.LoadFromGguf(
                        gguf, config, gpuLayers, gpuId, hybridThreading)
                    : DotLLM.Cuda.HybridTransformerModel.LoadFromGguf(
                        gguf, config, gpuLayers, gpuId, hybridThreading);
                forwardDeviceId = gpuId;
                deviceLabel =
                    $"{DotLLM.Cuda.CudaDevice.GetDevice(gpuId).Name} [0..{gpuLayers}) + cpu [{gpuLayers}..{numLayers})";
                break;
            }

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

        WriteResultTable(result, mode, placement.Describe(deviceLabel, numLayers),
            effectiveContext, effectiveStride, effectivePrefix, tokens.Count, sw.Elapsed);

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
    /// Where the trunk's layers run, and by what mechanism.
    /// </summary>
    /// <param name="Cycle">The GPU window slides across the whole trunk within one corpus pass.</param>
    /// <param name="FirstLayer">First global layer index of the GPU window.</param>
    /// <param name="GpuLayers">Layers resident on the GPU at any one time.</param>
    /// <param name="Backend">Normalized backend name (<c>cpu</c>, <c>cuda</c>, <c>vulkan</c>).</param>
    private readonly record struct LayerPlacement(bool Cycle, int FirstLayer, int GpuLayers, string Backend)
    {
        /// <summary>
        /// True when the run goes through the layer-window engine rather than a whole-device or
        /// prefix-offload model.
        /// </summary>
        public bool UsesLayerWindows => Cycle || FirstLayer > 0;

        /// <summary>Renders the placement for the result table.</summary>
        /// <param name="deviceLabel">Human-readable device name for the GPU half.</param>
        /// <param name="numLayers">Total transformer layers.</param>
        /// <returns>A one-line description naming every device and its layer range.</returns>
        public string Describe(string deviceLabel, int numLayers)
        {
            if (Cycle)
            {
                int phases = (numLayers + GpuLayers - 1) / GpuLayers;
                return $"cycled: {phases} x <={GpuLayers} layers on {deviceLabel}, covering [0..{numLayers}); "
                     + "output head on cpu";
            }

            if (FirstLayer > 0)
            {
                int end = FirstLayer + GpuLayers;
                string tail = end < numLayers ? $" + cpu [{end}..{numLayers})" : string.Empty;
                return $"cpu [0..{FirstLayer}) + {deviceLabel} [{FirstLayer}..{end}){tail}";
            }

            if (GpuLayers <= 0)
                return $"cpu [0..{numLayers}) (whole model)";

            return GpuLayers >= numLayers
                ? $"{deviceLabel} [0..{numLayers}) (whole model)"
                : deviceLabel;
        }
    }

    /// <summary>
    /// Validates the requested layer placement and rejects combinations the harness cannot honour.
    /// </summary>
    /// <param name="settings">Parsed command settings.</param>
    /// <param name="backend">Normalized backend name.</param>
    /// <param name="gpuLayers">Resolved GPU layer count.</param>
    /// <param name="numLayers">Total transformer layers.</param>
    /// <param name="placement">The validated placement.</param>
    /// <returns><see langword="true"/> when the placement is usable; otherwise an error was printed.</returns>
    /// <remarks>
    /// Every rejection here is a case where proceeding would produce a plausible-looking number that
    /// is not the measurement the user asked for — so they fail loudly rather than degrading.
    /// </remarks>
    private static bool TryResolvePlacement(
        Settings settings, string backend, int gpuLayers, int numLayers, out LayerPlacement placement)
    {
        placement = new LayerPlacement(settings.Cycle, settings.FirstLayer, gpuLayers, backend);

        if (settings.FirstLayer < 0 || settings.FirstLayer >= numLayers)
        {
            AnsiConsole.MarkupLine(
                $"[red]--first-layer must be in [0, {numLayers - 1}]; got {settings.FirstLayer}.[/]");
            return false;
        }

        // A partial offload has no Vulkan implementation. Silently loading the whole model on the
        // device instead would report a figure under a placement the user did not ask for.
        if (backend == "vulkan" && gpuLayers > 0 && gpuLayers < numLayers)
        {
            AnsiConsole.MarkupLine(
                "[red]Partial layer offload is not implemented for the Vulkan backend. "
                + "Use --device cuda, or omit --gpu-layers to score the whole model on Vulkan.[/]");
            return false;
        }

        if (!placement.UsesLayerWindows)
            return true;

        if (settings.Cycle && settings.FirstLayer > 0)
        {
            AnsiConsole.MarkupLine(
                "[red]--cycle covers every layer, so --first-layer has no meaning with it. Use one or the other.[/]");
            return false;
        }

        if (backend != "cuda")
        {
            AnsiConsole.MarkupLine(
                $"[red]--cycle / --first-layer need a CUDA device; --device is '{Markup.Escape(settings.Device)}'.[/]");
            return false;
        }

        if (gpuLayers <= 0)
        {
            AnsiConsole.MarkupLine("[red]--cycle / --first-layer require --gpu-layers greater than 0.[/]");
            return false;
        }

        if (!settings.Cycle && settings.FirstLayer + gpuLayers > numLayers)
        {
            AnsiConsole.MarkupLine(
                $"[red]GPU window [{settings.FirstLayer}..{settings.FirstLayer + gpuLayers}) runs past the "
                + $"model's {numLayers} layers.[/]");
            return false;
        }

        return true;
    }

    /// <summary>
    /// Scores through the layer-window engine: an arbitrary contiguous GPU window
    /// (<c>--first-layer</c>) or a GPU window cycled across the whole trunk (<c>--cycle</c>).
    /// </summary>
    /// <remarks>
    /// <para>Both modes go through <see cref="CyclingPerplexityEvaluator"/>, which checkpoints the
    /// hidden state at each layer cut and replays the next window from it. The final scoring pass is
    /// the ordinary <see cref="PerplexityEvaluator"/> against the saved boundaries, so the reported
    /// figure, its error bar, <c>--per-window</c> and the whole window geometry are produced by
    /// exactly the code a whole-device run uses.</para>
    /// <para>The CPU model supplies the output head (final norm + LM head): the CUDA layer windows
    /// return only a hidden state, and sliding-window scoring needs logits for every row rather than
    /// the last row the device forward paths are optimised for. Every transformer <em>layer</em>
    /// still runs on the GPU.</para>
    /// </remarks>
    private static int RunLayerWindowed(
        GgufFile gguf, ModelConfig config, Settings settings, PerplexityOptions options,
        List<int> tokens, LayerPlacement placement, int gpuId,
        int effectiveContext, int effectiveStride, int effectivePrefix, PerplexityMode mode)
    {
        int numLayers = config.NumLayers;
        string deviceLabel = DotLLM.Cuda.CudaDevice.GetDevice(gpuId).Name;

        // The CPU model is the head provider and, for a non-cycled window, also executes the layers
        // outside the GPU window. Its weights are mmap-backed, so holding it resident alongside the
        // device window costs page cache rather than committed memory.
        using IModel cpuModel = ModelLoader.CreateCpuModelFromGguf(
            gguf, config, new ThreadingConfig(settings.Threads));
        using var cpuWindows = new CpuLayerWindowModel(cpuModel, config);
        using var cudaWindows = DotLLM.Cuda.Evaluation.CudaLayerWindowModel.LoadFromGguf(
            gguf, config, gpuId, cpuWindows);

        var assignments = new List<CompositeLayerWindowModel.LayerAssignment>();
        if (placement.Cycle)
        {
            foreach (LayerWindow w in CyclingPerplexityEvaluator.PartitionLayers(numLayers, placement.GpuLayers))
                assignments.Add(new CompositeLayerWindowModel.LayerAssignment(w, cudaWindows));
        }
        else
        {
            int first = placement.FirstLayer;
            int end = first + placement.GpuLayers;
            if (first > 0)
                assignments.Add(new CompositeLayerWindowModel.LayerAssignment(new LayerWindow(0, first), cpuWindows));
            assignments.Add(new CompositeLayerWindowModel.LayerAssignment(
                new LayerWindow(first, placement.GpuLayers), cudaWindows));
            if (end < numLayers)
                assignments.Add(new CompositeLayerWindowModel.LayerAssignment(
                    new LayerWindow(end, numLayers - end), cpuWindows));
        }

        using var composite = new CompositeLayerWindowModel(assignments, cpuWindows);

        AnsiConsole.MarkupLine(
            $"[grey]placement: {Markup.Escape(placement.Describe(deviceLabel, numLayers))}[/]");

        var sw = Stopwatch.StartNew();
        PerplexityResult result;
        try
        {
            PerplexityEvaluator.WindowObserver? observer = settings.PerWindow
                ? (i, ppl, n) => Console.WriteLine($"window {i}: ppl={ppl:F6} scored={n}")
                : null;
            CyclingPerplexityEvaluator.PhaseObserver onPhase = (i, count, w) =>
                AnsiConsole.MarkupLine($"[grey]  layer window {i + 1}/{count}: {w}[/]");

            result = CyclingPerplexityEvaluator.Evaluate(
                composite,
                System.Runtime.InteropServices.CollectionsMarshal.AsSpan(tokens),
                options,
                composite.Windows(),
                observer,
                onPhase);
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException)
        {
            AnsiConsole.MarkupLine($"[red]{Markup.Escape(ex.Message)}[/]");
            return 1;
        }
        sw.Stop();

        WriteResultTable(result, mode, placement.Describe(deviceLabel, numLayers),
            effectiveContext, effectiveStride, effectivePrefix, tokens.Count, sw.Elapsed);
        return 0;
    }

    /// <summary>
    /// Prints the result table shared by every scoring path.
    /// </summary>
    /// <param name="result">The scored result.</param>
    /// <param name="mode">Scoring mode.</param>
    /// <param name="placement">Rendered layer placement.</param>
    /// <param name="context">Effective context window.</param>
    /// <param name="stride">Effective stride.</param>
    /// <param name="unscoredPrefix">Effective unscored prefix.</param>
    /// <param name="corpusTokens">Corpus token count.</param>
    /// <param name="elapsed">Wall-clock scoring time.</param>
    /// <remarks>
    /// Window geometry and scored-token count sit next to the figure deliberately: a perplexity
    /// without them is not comparable to anything. <c>Layer placement</c> is printed on every run,
    /// whole-device included, so a scraped table can never pass a split or cycled run off as a
    /// single-device one (issue #395).
    /// </remarks>
    private static void WriteResultTable(
        PerplexityResult result, PerplexityMode mode, string placement,
        int context, int stride, int unscoredPrefix, int corpusTokens, TimeSpan elapsed)
    {
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
        table.AddRow("Layer placement", placement);
        table.AddRow("Context", $"{context:N0}");
        table.AddRow("Stride", $"{stride:N0}");
        table.AddRow("Unscored prefix", $"{unscoredPrefix:N0}");
        table.AddRow("Corpus tokens", $"{corpusTokens:N0}");
        table.AddRow("Elapsed", $"{elapsed.TotalSeconds:F2} s");
        AnsiConsole.Write(table);
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
