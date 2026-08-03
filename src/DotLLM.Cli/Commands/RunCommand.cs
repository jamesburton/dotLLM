using System.ComponentModel;
using System.Diagnostics;
using System.Text.Json;
using System.Text.RegularExpressions;
using DotLLM.Cli.Helpers;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Engine.Constraints;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using DotLLM.Tokenizers.ToolCallParsers;
using Spectre.Console;
using Spectre.Console.Cli;
using Spectre.Console.Rendering;

namespace DotLLM.Cli.Commands;

/// <summary>
/// Runs text generation on a GGUF model: load → encode prompt → stream tokens via TextGenerator.
/// Supports greedy (default) and sampled decoding via composable sampling pipeline.
/// </summary>
internal sealed class RunCommand : AsyncCommand<RunCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<model>")]
        [Description("Path to a GGUF file or HuggingFace repo ID (e.g., QuantFactory/SmolLM-135M-GGUF).")]
        public string Model { get; set; } = string.Empty;

        [CommandOption("--prompt|-p")]
        [Description("Input prompt for generation. Required unless --prompt-file is given.")]
        public string? Prompt { get; set; }

        [CommandOption("--prompt-file")]
        [Description("Read the prompt from a file instead of --prompt. " +
                     "A single trailing newline is stripped. Mutually exclusive with --prompt.")]
        public string? PromptFile { get; set; }

        [CommandOption("--max-tokens|-n")]
        [Description("Maximum number of tokens to generate.")]
        [DefaultValue(128)]
        public int MaxTokens { get; set; } = 128;

        [CommandOption("--temp|-t")]
        [Description("Sampling temperature. 0 = greedy (default).")]
        [DefaultValue(0f)]
        public float Temperature { get; set; }

        [CommandOption("--top-k")]
        [Description("Top-K sampling. 0 = disabled.")]
        [DefaultValue(0)]
        public int TopK { get; set; }

        [CommandOption("--top-p")]
        [Description("Top-P (nucleus) sampling threshold.")]
        [DefaultValue(1.0f)]
        public float TopP { get; set; } = 1.0f;

        [CommandOption("--min-p")]
        [Description("Min-P sampling threshold. 0 = disabled.")]
        [DefaultValue(0f)]
        public float MinP { get; set; }

        [CommandOption("--repeat-penalty")]
        [Description("Repetition penalty factor. 1.0 = disabled.")]
        [DefaultValue(1.0f)]
        public float RepeatPenalty { get; set; } = 1.0f;

        [CommandOption("--repeat-last-n")]
        [Description("Number of recent tokens for repetition penalty lookback. 0 = full history.")]
        [DefaultValue(0)]
        public int RepeatLastN { get; set; }

        [CommandOption("--frequency-penalty")]
        [Description("OpenAI-style frequency penalty: subtracted proportionally to occurrence count. 0 = disabled.")]
        [DefaultValue(0f)]
        public float FrequencyPenalty { get; set; }

        [CommandOption("--presence-penalty")]
        [Description("OpenAI-style presence penalty: subtracted once for any token already seen. 0 = disabled.")]
        [DefaultValue(0f)]
        public float PresencePenalty { get; set; }

        [CommandOption("--logit-bias|-l")]
        [Description("Per-token additive logit bias 'token_id=bias', repeatable (e.g. '-l 15043=-100').")]
        public string[] LogitBias { get; set; } = Array.Empty<string>();

        [CommandOption("--top-nsigma|--top-n-sigma")]
        [Description("Top-nσ sampling threshold. Negative = disabled (default).")]
        [DefaultValue(-1f)]
        public float TopNSigma { get; set; } = -1f;

        [CommandOption("--dry-multiplier")]
        [Description("DRY (Don't Repeat Yourself) repetition penalty multiplier. 0 = disabled (default).")]
        [DefaultValue(0f)]
        public float DryMultiplier { get; set; }

        [CommandOption("--dry-base")]
        [Description("DRY exponential base for the match-length penalty curve.")]
        [DefaultValue(1.75f)]
        public float DryBase { get; set; } = 1.75f;

        [CommandOption("--dry-allowed-length")]
        [Description("Minimum matched n-gram length before DRY starts penalizing.")]
        [DefaultValue(2)]
        public int DryAllowedLength { get; set; } = 2;

        [CommandOption("--dry-penalty-last-n")]
        [Description("Number of recent tokens considered for DRY matching. 0 = full history.")]
        [DefaultValue(0)]
        public int DryPenaltyLastN { get; set; }

        [CommandOption("--dry-sequence-breaker")]
        [Description("Token string that resets DRY n-gram matching, repeatable. Default: newline, ':', '\"', '*'.")]
        public string[]? DrySequenceBreakers { get; set; }

        [CommandOption("--rope-scaling")]
        [Description("RoPE scaling override: 'none', 'linear', 'yarn', 'ntk', 'dynamic'. Overrides the GGUF-derived value.")]
        public string? RopeScaling { get; set; }

        [CommandOption("--rope-freq-base")]
        [Description("RoPE base frequency (theta) override. Overrides the GGUF-derived value.")]
        public float? RopeFreqBase { get; set; }

        [CommandOption("--rope-scale")]
        [Description("RoPE scaling factor override (linear/YaRN/NTK). Overrides the GGUF-derived value.")]
        public float? RopeScale { get; set; }

        [CommandOption("--yarn-orig-ctx")]
        [Description("YaRN original context length override.")]
        public int? YarnOrigCtx { get; set; }

        [CommandOption("--yarn-attn-factor")]
        [Description("YaRN attention factor override.")]
        public float? YarnAttnFactor { get; set; }

        [CommandOption("--yarn-beta-fast")]
        [Description("YaRN beta-fast parameter override.")]
        public float? YarnBetaFast { get; set; }

        [CommandOption("--yarn-beta-slow")]
        [Description("YaRN beta-slow parameter override.")]
        public float? YarnBetaSlow { get; set; }

        [CommandOption("--seed|-s")]
        [Description("Random seed for reproducible sampling. Omit for non-deterministic.")]
        public int? Seed { get; set; }

        [CommandOption("--threads")]
        [Description("Number of CPU threads for inference. 0 = auto/all cores (default), 1 = single-threaded.")]
        [DefaultValue(0)]
        public int Threads { get; set; }

        [CommandOption("--decode-threads")]
        [Description("Number of threads for decode. 0 = auto (caps at memory channel count).")]
        [DefaultValue(0)]
        public int DecodeThreads { get; set; }

        [CommandOption("--numa-pin")]
        [Description("Pin workers to NUMA-local cores on multi-socket systems.")]
        [DefaultValue(false)]
        public bool NumaPin { get; set; }

        [CommandOption("--pcore-only")]
        [Description("Pin workers to P-cores only (Intel hybrid architectures).")]
        [DefaultValue(false)]
        public bool PCoreOnly { get; set; }

        [CommandOption("--device|-d")]
        [Description("Compute device: 'cpu' (default), 'gpu', 'gpu:0', 'gpu:1'.")]
        [DefaultValue("cpu")]
        public string Device { get; set; } = "cpu";

        [CommandOption("--gpu-layers")]
        [Description("Number of transformer layers to offload to GPU. 0 = CPU only. " +
                     "Omit for default (0 with --device cpu, all with --device gpu).")]
        public int? GpuLayers { get; set; }

        [CommandOption("--quant|-q")]
        [Description("Quantization filter when multiple GGUF files exist (e.g., Q4_K_M, Q8_0).")]
        public string? Quant { get; set; }

        [CommandOption("--json")]
        [Description("Output result as a single JSON object (suppresses all formatted output).")]
        [DefaultValue(false)]
        public bool Json { get; set; }

        [CommandOption("--response-format")]
        [Description("Constrain model output format: 'text' (default), 'json_object', 'json_schema', 'regex', or 'grammar'.")]
        [DefaultValue("text")]
        public string ResponseFormat { get; set; } = "text";

        [CommandOption("--schema")]
        [Description("JSON Schema string or file path (prefixed with @) for json_schema response format.")]
        public string? Schema { get; set; }

        [CommandOption("--pattern")]
        [Description("Regex pattern for regex response format. Entire output must match.")]
        public string? Pattern { get; set; }

        [CommandOption("--grammar")]
        [Description("GBNF grammar string or file path (prefixed with @) for grammar response format.")]
        public string? Grammar { get; set; }

        /// <summary>Use paged KV-cache (block-based allocation).</summary>
        [CommandOption("--paged")]
        [Description("Use paged KV-cache with block-based allocation instead of pre-allocated simple cache.")]
        [DefaultValue(false)]
        public bool Paged { get; set; }

        /// <summary>KV-cache key quantization type.</summary>
        [CommandOption("--cache-type-k")]
        [Description("KV-cache key quantization: f32 (default), q8_0, q4_0.")]
        [DefaultValue("f32")]
        public string CacheTypeK { get; set; } = "f32";

        /// <summary>KV-cache value quantization type.</summary>
        [CommandOption("--cache-type-v")]
        [Description("KV-cache value quantization: f32 (default), q8_0, q4_0.")]
        [DefaultValue("f32")]
        public string CacheTypeV { get; set; } = "f32";

        /// <summary>Mixed-precision window size for KV-cache quantization.</summary>
        [CommandOption("--cache-window")]
        [Description("Mixed-precision window: recent N tokens in full precision (0 = all quantized). Only used when --cache-type-k or --cache-type-v is set.")]
        [DefaultValue(0)]
        public int CacheWindow { get; set; }

        /// <summary>Tool definitions.</summary>
        [CommandOption("--tools")]
        [Description("Tool definitions: JSON array string or file path (prefixed with @). " +
                     "When provided, the prompt is formatted via the model's chat template with tool definitions.")]
        public string? Tools { get; set; }

        /// <summary>Tool selection mode.</summary>
        [CommandOption("--tool-choice")]
        [Description("Tool selection: 'auto' (default), 'none', 'required' (constrain output to a valid tool call), or a function name.")]
        public string ToolChoiceStr { get; set; } = "auto";

        /// <summary>Draft model for speculative decoding.</summary>
        [CommandOption("--speculative-model|--draft-model")]
        [Description("Path or HuggingFace repo ID for a draft model. Enables speculative decoding for faster generation. Must share vocabulary with the main model.")]
        public string? SpeculativeModel { get; set; }

        /// <summary>Number of draft candidates per speculative step.</summary>
        [CommandOption("--speculative-k|--draft-tokens")]
        [Description("Number of draft tokens per speculative step (K). Default 5.")]
        [DefaultValue(5)]
        public int SpeculativeK { get; set; } = 5;

        /// <summary>Maximum prompt tokens per prefill forward pass (llama.cpp -ub analog).</summary>
        [CommandOption("--prefill-chunk-size|--ubatch-size")]
        [Description("Maximum prompt tokens per prefill forward pass (llama.cpp -ub analog). 0 = whole prompt in one pass (default).")]
        [DefaultValue(0)]
        public int PrefillChunkSize { get; set; }

        /// <summary>
        /// Paths to HuggingFace PEFT LoRA adapter directories. Repeatable — each occurrence adds one
        /// adapter. Optionally suffix with <c>=weight</c> (e.g. <c>path/to/lora=0.7</c>) to blend.
        /// Two or more adapters are rank-concatenated into a single composite via <see cref="LoraComposer"/>.
        /// </summary>
        [CommandOption("--lora")]
        [Description("Path to a PEFT LoRA adapter dir; repeatable to stack adapters, optional weight 'path=0.7'. Omit for base.")]
        public string[] LoraPaths { get; set; } = Array.Empty<string>();
    }

    public override async Task<int> ExecuteAsync(CommandContext context, Settings settings)
    {
        if (!TextArgument.TryResolve(settings.Prompt, settings.PromptFile,
                "--prompt|-p", "--prompt-file", required: true,
                out string? resolvedPrompt, out string? promptError))
        {
            if (settings.Json)
                Console.Error.WriteLine($"Error: {promptError}");
            else
                AnsiConsole.MarkupLine($"[red]{Markup.Escape(promptError!)}[/]");
            return 1;
        }

        string prompt = resolvedPrompt!;

        // HuggingFace safetensors directory? (config.json + *.safetensors /
        // model.safetensors.index.json). Auto-detected; loads via the
        // safetensors path instead of GGUF. The GGUF path is unchanged.
        string? hfDir = TryResolveHfDirectory(settings.Model);

        string resolvedPath;
        if (hfDir is not null)
        {
            resolvedPath = hfDir;
        }
        else
        {
            var ggufPath = GgufFileResolver.Resolve(settings.Model, settings.Quant);
            if (ggufPath is null)
                return 1;
            resolvedPath = ggufPath;
        }

        GgufFile? gguf = null;
        IDisposable? safetensorsSource = null;
        ModelConfig config = null!;
        ITokenizer tokenizer = null!;
        IModel model = null!;

        void LoadModel()
        {
            if (hfDir is not null)
            {
                // HuggingFace safetensors checkpoint (e.g. BitNet b1.58 bf16).
                // CPU load via the shared safetensors loader; GPU offload for
                // this path is not wired through the CLI yet.
                var threadingCfg = new ThreadingConfig(
                    settings.Threads, settings.DecodeThreads, settings.NumaPin, settings.PCoreOnly);
                var (m, src, cfg) = ModelLoader.LoadFromSafetensors(hfDir, threadingCfg);
                model = m;
                safetensorsSource = src;
                config = cfg;
                tokenizer = ModelLoader.LoadTokenizerFromHfDirectory(hfDir)
                    ?? throw new InvalidOperationException(
                        $"No tokenizer.json (or legacy vocab.json/merges.txt) found in '{hfDir}'.");
                return;
            }

            gguf = GgufFile.Open(resolvedPath);
            config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            config = GgufModelConfigExtractor.ApplyRoPEOverride(config, BuildRoPEOverride(settings));
            tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

            int gpuLayers = ResolveGpuLayers(settings, config);
            if (gpuLayers <= 0)
            {
                // Shared per-architecture CPU dispatch — routes hybrid architectures
                // (Nemotron-H, Qwen3MoeHybrid) to their dedicated loaders.
                model = ModelLoader.CreateCpuModelFromGguf(gguf, config,
                    new ThreadingConfig(settings.Threads, settings.DecodeThreads, settings.NumaPin, settings.PCoreOnly));
            }
            else if (gpuLayers >= config.NumLayers)
            {
                int gpuId = ParseGpuId(settings.Device);
                model = DotLLM.Cuda.CudaTransformerModel.LoadFromGguf(gguf, config, gpuId);
            }
            else
            {
                int gpuId = ParseGpuId(settings.Device);
                model = DotLLM.Cuda.HybridTransformerModel.LoadFromGguf(gguf, config, gpuLayers, gpuId,
                    new ThreadingConfig(settings.Threads, settings.DecodeThreads, settings.NumaPin, settings.PCoreOnly));
            }
        }

        var loadSw = Stopwatch.StartNew();
        if (settings.Json)
        {
            LoadModel();
        }
        else
        {
            AnsiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .Start("Loading model...", _ => LoadModel());
        }
        loadSw.Stop();

        // Display VRAM warning after spinner completes (so it stays visible).
        // In JSON mode, write to stderr so it doesn't corrupt the JSON output.
        string? vramWarning = (model as DotLLM.Cuda.CudaTransformerModel)?.VramWarning
                           ?? (model as DotLLM.Cuda.HybridTransformerModel)?.VramWarning;
        if (vramWarning is not null)
        {
            if (settings.Json)
                Console.Error.WriteLine($"WARNING: {vramWarning}");
            else
                AnsiConsole.MarkupLine($"[yellow]WARNING: {Markup.Escape(vramWarning)}[/]");
        }

        // Load LoRA adapter(s) if requested.
        // innerAdapters: every per-spec adapter (must be disposed in finally).
        // compositeAdapter: rank-concat result when >1 adapter (also dispose in finally).
        // loraAdapter: the single adapter passed to the generator (either one inner or the composite).
        var innerAdapters = new List<ILoraAdapter>();
        ILoraAdapter? compositeAdapter = null;
        ILoraAdapter? loraAdapter = null;

        if (settings.LoraPaths.Length > 0)
        {
            var stack = new List<(ILoraAdapter adapter, float weight)>(settings.LoraPaths.Length);
            for (int i = 0; i < settings.LoraPaths.Length; i++)
            {
                var (path, weight) = LoraSpec.Parse(settings.LoraPaths[i]);
                string adapterName = $"cli[{i}]";
                var inner = PeftAdapterLoader.LoadFromDirectory(adapterName, path, config);
                innerAdapters.Add(inner);
                if (!inner.IsCompatible(config))
                    throw new InvalidOperationException(
                        $"LoRA adapter at '{path}' is incompatible with this base model.");
                stack.Add((inner, weight));
            }

            if (stack.Count == 1)
            {
                // Single adapter: pass directly; avoids the F32-only composer constraint.
                loraAdapter = stack[0].adapter;
            }
            else
            {
                // Multiple adapters: compose into a single rank-concatenated adapter.
                compositeAdapter = LoraComposer.Compose(stack, config);
                loraAdapter = compositeAdapter;
            }
        }

        var threadingInfo = new ThreadingConfig(settings.Threads, settings.DecodeThreads, settings.NumaPin, settings.PCoreOnly);

        // Parse tool definitions and format prompt via chat template when tools are provided
        ToolDefinition[]? tools = ChatCommand.ParseToolDefinitions(settings.Tools);
        IToolCallParser? toolCallParser = null;
        string effectivePrompt = prompt;
        if (tools is { Length: > 0 } && gguf is null)
        {
            // Tool calling relies on the GGUF-embedded chat template; the HF
            // safetensors path doesn't surface one here. Fall back to the raw
            // prompt so generation still runs.
            if (settings.Json)
                Console.Error.WriteLine("WARNING: --tools is not supported for HuggingFace safetensors models; ignoring.");
            else
                AnsiConsole.MarkupLine("[yellow]WARNING: --tools is not supported for HuggingFace safetensors models; ignoring.[/]");
            tools = null;
        }
        if (tools is { Length: > 0 })
        {
            string bosToken = tokenizer.DecodeToken(tokenizer.BosTokenId);
            string eosToken = tokenizer.DecodeToken(tokenizer.EosTokenId);
            var chatTemplate = GgufChatTemplateFactory.TryCreate(gguf!.Metadata, tokenizer, config.Architecture)
                ?? GgufChatTemplateFactory.CreatePlainFallback(tokenizer);

            var messages = new List<ChatMessage>
            {
                new() { Role = "user", Content = prompt }
            };
            effectivePrompt = chatTemplate.Apply(messages, new ChatTemplateOptions
            {
                AddGenerationPrompt = true,
                Tools = tools
            });
            toolCallParser = GgufChatTemplateFactory.CreateToolCallParser(gguf!.Metadata, config.Architecture);
        }
        var toolChoice = ChatCommand.ParseToolChoice(settings.ToolChoiceStr, tools);

        // Build inference options from CLI flags
        var responseFormat = settings.ResponseFormat.ToLowerInvariant() switch
        {
            "json_object" => (Core.Configuration.ResponseFormat)new Core.Configuration.ResponseFormat.JsonObject(),
            "json_schema" => BuildJsonSchemaFormat(settings.Schema),
            "regex" => BuildRegexFormat(settings.Pattern),
            "grammar" => BuildGrammarFormat(settings.Grammar),
            _ => null
        };

        // When required/function, constrain output to valid tool-call JSON (bare JSON object).
        if (responseFormat is null && tools is { Length: > 0 } && (toolChoice is ToolChoice.Required or ToolChoice.Function))
        {
            string argumentsKey = toolCallParser is LlamaToolCallParser ? "parameters" : "arguments";
            responseFormat = toolChoice switch
            {
                ToolChoice.Required => new Core.Configuration.ResponseFormat.JsonSchema
                    { Schema = ToolCallSchemaBuilder.BuildForRequired(tools, argumentsKey), Name = "tool_call" },
                ToolChoice.Function fn => new Core.Configuration.ResponseFormat.JsonSchema
                    { Schema = ToolCallSchemaBuilder.BuildForFunction(tools.First(t => t.Name == fn.Name), argumentsKey), Name = "tool_call" },
                _ => responseFormat
            };
            // Constrained output is bare JSON → parse with the generic (markerless) parser.
            toolCallParser = new GenericToolCallParser();
        }
        var inferenceOptions = new InferenceOptions
        {
            Temperature = settings.Temperature,
            TopK = settings.TopK,
            TopP = settings.TopP,
            MinP = settings.MinP,
            RepetitionPenalty = settings.RepeatPenalty,
            RepetitionPenaltyWindow = settings.RepeatLastN,
            FrequencyPenalty = settings.FrequencyPenalty,
            PresencePenalty = settings.PresencePenalty,
            LogitBias = ParseLogitBias(settings.LogitBias),
            TopNSigma = settings.TopNSigma,
            DryMultiplier = settings.DryMultiplier,
            DryBase = settings.DryBase,
            DryAllowedLength = settings.DryAllowedLength,
            DryPenaltyLastN = settings.DryPenaltyLastN,
            DrySequenceBreakers = settings.DrySequenceBreakers is { Length: > 0 }
                ? settings.DrySequenceBreakers
                : new InferenceOptions().DrySequenceBreakers,
            MaxTokens = settings.MaxTokens,
            Seed = settings.Seed,
            ResponseFormat = responseFormat,
            Threading = threadingInfo
        };

        if (!settings.Json)
        {
            // Build compact pre-gen header rule
            var quantLabel = InferQuantLabel(resolvedPath, settings.Quant);
            var samplingLabel = BuildSamplingLabel(settings);
            var deviceLabel = model switch
            {
                DotLLM.Cuda.CudaTransformerModel => DotLLM.Cuda.CudaDevice.GetDevice(ParseGpuId(settings.Device)).ToString(),
                DotLLM.Cuda.HybridTransformerModel h => $"hybrid {h.NumGpuLayers}gpu/{config.NumLayers - h.NumGpuLayers}cpu",
                _ => $"{threadingInfo.EffectiveThreadCount} threads"
            };
            var segments = $"{config.Architecture} {config.NumLayers}L/{config.HiddenSize}H | {quantLabel} | {deviceLabel} | {samplingLabel}";
            AnsiConsole.Write(new Rule($"[grey]dotllm | {Markup.Escape(segments)}[/]").LeftJustified());
            AnsiConsole.WriteLine();
        }

        DotLLM.Engine.KvCache.PagedKvCacheFactory? pagedFactory = null;
        DotLLM.Cuda.CudaPagedKvCacheFactory? cudaPagedFactory = null;
        try
        {
            // Add stop sequences for tool calling end-of-turn tokens
            if (tools is { Length: > 0 })
            {
                string eosTokenStr = tokenizer.DecodeToken(tokenizer.EosTokenId);
                var toolStopSeqs = new List<string>();
                foreach (var marker in new[] { "<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<|end|>", "</s>", "</tool_call>" })
                {
                    if (marker != eosTokenStr)
                        toolStopSeqs.Add(marker);
                }
                inferenceOptions = inferenceOptions with { StopSequences = toolStopSeqs };
            }

            if (!settings.Json)
                Console.Write(tools is { Length: > 0 } ? "" : prompt);

            var kvConfig = new KvCacheConfig(
                KvCacheConfig.ParseDType(settings.CacheTypeK),
                KvCacheConfig.ParseDType(settings.CacheTypeV),
                settings.CacheWindow);

            Func<ModelConfig, int, DotLLM.Core.Attention.IKvCache>? kvFactory = null;
            if (model is DotLLM.Cuda.CudaTransformerModel cudaModel)
            {
                if (settings.Paged && kvConfig.IsQuantized)
                {
                    Console.Error.WriteLine("WARNING: Paged KV-cache does not support quantization yet, using quantized GPU cache.");
                    kvFactory = (cfg, size) => cudaModel.CreateKvCache(size, kvConfig);
                }
                else if (settings.Paged)
                {
                    // Issue #252: block-scattered device storage + gather-into-scratch attention
                    // dispatch. Mirrors the CPU `--paged` branch below.
                    cudaPagedFactory = new DotLLM.Cuda.CudaPagedKvCacheFactory(
                        DotLLM.Core.Attention.KvGeometry.FromConfig(config));
                    var factory = cudaPagedFactory;
                    kvFactory = (cfg, size) => cudaModel.CreatePagedKvCache(factory.Pool, size);
                }
                else
                {
                    kvFactory = kvConfig.IsQuantized
                        ? (cfg, size) => cudaModel.CreateKvCache(size, kvConfig)
                        : (cfg, size) => cudaModel.CreateKvCache(size);
                }
            }
            else if (model is DotLLM.Cuda.HybridTransformerModel hybridModel)
            {
                if (settings.Paged)
                    Console.Error.WriteLine("WARNING: Paged KV-cache not supported with hybrid GPU, using hybrid cache.");
                kvFactory = (cfg, size) => hybridModel.CreateKvCache(size);
            }
            else if (settings.Paged && !kvConfig.IsQuantized)
            {
                pagedFactory = new DotLLM.Engine.KvCache.PagedKvCacheFactory(
                    config.NumLayers, config.NumKvHeads, config.HeadDim);
                kvFactory = (cfg, size) => pagedFactory.Create(size);
            }
            else if (settings.Paged && kvConfig.IsQuantized)
            {
                Console.Error.WriteLine("WARNING: Paged KV-cache does not support quantization yet, using quantized simple cache.");
                kvFactory = (cfg, size) => new DotLLM.Engine.KvCache.QuantizedKvCache(
                    cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, size,
                    kvConfig.KeyDType, kvConfig.ValueDType, kvConfig.MixedPrecisionWindowSize);
            }
            else if (kvConfig.IsQuantized)
            {
                kvFactory = (cfg, size) => new DotLLM.Engine.KvCache.QuantizedKvCache(
                    cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, size,
                    kvConfig.KeyDType, kvConfig.ValueDType, kvConfig.MixedPrecisionWindowSize);
            }

            // Load speculative draft model if requested
            IModel? draftModel = null;
            GgufFile? draftGguf = null;
            if (!string.IsNullOrEmpty(settings.SpeculativeModel))
            {
                var draftPath = GgufFileResolver.Resolve(settings.SpeculativeModel, null);
                if (draftPath is null)
                {
                    if (!settings.Json)
                        AnsiConsole.MarkupLine("[red]Speculative draft model not found.[/]");
                    else
                        Console.Error.WriteLine("Error: Speculative draft model not found.");
                    return 1;
                }

                draftGguf = GgufFile.Open(draftPath);
                var draftConfig = GgufModelConfigExtractor.Extract(draftGguf.Metadata);
                if (!DotLLM.Engine.SpeculativeConstants.AreVocabsCompatible(config.VocabSize, draftConfig.VocabSize))
                {
                    var msg = $"Draft model vocab size ({draftConfig.VocabSize}) differs from target ({config.VocabSize}) by more than {DotLLM.Engine.SpeculativeConstants.MaxVocabSizeDifference} tokens.";
                    if (!settings.Json)
                        AnsiConsole.MarkupLine($"[red]{Markup.Escape(msg)}[/]");
                    else
                        Console.Error.WriteLine($"Error: {msg}");
                    return 1;
                }

                draftModel = TransformerModel.LoadFromGguf(draftGguf, draftConfig,
                    new ThreadingConfig(settings.Threads, settings.DecodeThreads));

                if (!settings.Json)
                    AnsiConsole.MarkupLine($"[dim]Speculative decoding: K={settings.SpeculativeK}, draft={System.IO.Path.GetFileName(draftPath)}[/]");

                if (settings.LoraPaths.Length > 0)
                {
                    const string specLoraWarning = "WARNING: --lora with speculative decoding does not adapt the draft model; acceptance rates may degrade.";
                    if (settings.Json) Console.Error.WriteLine(specLoraWarning);
                    else AnsiConsole.MarkupLine($"[yellow]{Markup.Escape(specLoraWarning)}[/]");
                }
            }

            var generator = new TextGenerator(model, tokenizer, kvFactory,
                draftModel: draftModel, speculativeCandidates: settings.SpeculativeK,
                prefillChunkSize: settings.PrefillChunkSize);
            var totalSw = Stopwatch.StartNew();
            int generated = 0;
            InferenceTimings timings = default;
            FinishReason finishReason = FinishReason.Length;
            var generatedText = new System.Text.StringBuilder();

            await foreach (var token in generator.GenerateStreamingTokensAsync(effectivePrompt, inferenceOptions, adapter: loraAdapter))
            {
                generatedText.Append(token.Text);
                if (!settings.Json)
                    Console.Write(token.Text);

                if (token.FinishReason is null || token.Text.Length > 0)
                    generated++;
                if (token.FinishReason.HasValue)
                {
                    finishReason = token.FinishReason.Value;
                    timings = token.Timings ?? default;
                }
            }

            totalSw.Stop();

            // Read timings from streaming result
            double loadMs = loadSw.Elapsed.TotalMilliseconds;
            double promptEvalMs = timings.PrefillTimeMs;
            double evalMs = timings.DecodeTimeMs;
            double samplerMs = timings.SamplingTimeMs;
            double totalMs = totalSw.Elapsed.TotalMilliseconds;
            int promptLen = timings.PrefillTokenCount;
            int evalSteps = timings.DecodeTokenCount;

            // Compute metrics
            int totalTokens = promptLen + generated;
            double decodeTokPerSec = evalSteps > 0 ? evalSteps / (evalMs / 1000.0) : 0;
            double prefillTokPerSec = promptLen > 0 ? promptLen / (promptEvalMs / 1000.0) : 0;
            double totalTokPerSec = totalTokens > 0 ? totalTokens / (totalMs / 1000.0) : 0;

            // Memory metrics
            long modelWeightsBytes;
            if (gguf is not null)
            {
                long fileSize = new FileInfo(resolvedPath).Length;
                modelWeightsBytes = fileSize - gguf.DataSectionOffset;
            }
            else
            {
                // HF safetensors: sum the on-disk shard sizes in the directory.
                modelWeightsBytes = Directory
                    .EnumerateFiles(resolvedPath, "*.safetensors", SearchOption.TopDirectoryOnly)
                    .Sum(f => new FileInfo(f).Length);
            }
            long computeBytes = model.ComputeMemoryBytes;
            int cacheSize = Math.Min(promptLen + settings.MaxTokens, config.MaxSequenceLength);
            // Use actual KV-cache bytes from engine timings (reflects quantization compression).
            // Fall back to computed estimate for GPU caches (based on config).
            long kvCacheBytes;
            if (timings.KvCacheBytes > 0)
                kvCacheBytes = timings.KvCacheBytes;
            else if (kvConfig.IsQuantized)
                kvCacheBytes = ComputeQuantizedKvBytes(config, cacheSize, kvConfig);
            else
                kvCacheBytes = (long)config.NumLayers * 2 * cacheSize
                    * config.NumKvHeads * config.HeadDim
                    * (model is DotLLM.Cuda.CudaTransformerModel ? sizeof(ushort) : sizeof(float));
            // R4-interleaved buffers are a second, committed copy of the weights held alongside
            // the mapped file — counting only the mapping understates the footprint by ~2x.
            long repackedBytes = model.RepackedWeightBytes;
            long totalMemory = modelWeightsBytes + repackedBytes + computeBytes + kvCacheBytes;

            // Backend/decode-path diagnostics
            string samplerPath = BuildSamplerPath(settings);
            string? decodeGraph = (model as DotLLM.Cuda.CudaTransformerModel)?.DecodeGraphState
                .ToString().ToLowerInvariant();

            // Detect tool calls in generated output
            string outputText = generatedText.ToString();
            ToolCall[]? detectedToolCalls = null;
            if (toolCallParser is not null && outputText.Length > 0)
            {
                // Strip stop sequence suffixes before parsing
                foreach (var seq in inferenceOptions.StopSequences)
                {
                    if (outputText.EndsWith(seq, StringComparison.Ordinal))
                    {
                        outputText = outputText[..^seq.Length];
                        break;
                    }
                }
                detectedToolCalls = toolCallParser.TryParse(outputText);
                if (detectedToolCalls is { Length: > 0 })
                    finishReason = FinishReason.ToolCalls;
            }

            // For constrained tool-choice paths, display the resolved tool call in non-JSON mode.
            if (!settings.Json && detectedToolCalls is { Length: > 0 }
                && (toolChoice is ToolChoice.Required or ToolChoice.Function))
            {
                foreach (var tc in detectedToolCalls)
                    AnsiConsole.MarkupLine($"[dim]tool call:[/] [green]{Markup.Escape(tc.FunctionName)}[/]({Markup.Escape(tc.Arguments)})");
            }

            if (settings.Json)
            {
                var result = new RunJsonResult
                {
                    Text = outputText,
                    Prompt = prompt,
                    Model = Path.GetFileName(resolvedPath),
                    Architecture = config.Architecture.ToString(),
                    FinishReason = finishReason.ToString().ToLowerInvariant(),
                    ToolCalls = detectedToolCalls?.Select(tc => new RunToolCallDto
                    {
                        Id = tc.Id,
                        FunctionName = tc.FunctionName,
                        Arguments = tc.Arguments,
                    }).ToArray(),
                    Usage = new RunUsageDto
                    {
                        PromptTokens = promptLen,
                        GeneratedTokens = generated,
                    },
                    Timings = new RunTimingsDto
                    {
                        LoadMs = Math.Round(loadMs, 1),
                        PrefillMs = Math.Round(promptEvalMs, 1),
                        DecodeMs = Math.Round(evalMs, 1),
                        SamplingMs = Math.Round(samplerMs, 1),
                        TotalMs = Math.Round(totalMs, 1),
                        PrefillTokS = Math.Round(prefillTokPerSec, 2),
                        DecodeTokS = Math.Round(decodeTokPerSec, 2),
                        GeneratedTokens = generated,
                        SpeculativeDraftTokens = timings.SpeculativeDraftTokens,
                        SpeculativeAcceptedTokens = timings.SpeculativeAcceptedTokens,
                        SpeculativeAcceptanceRate = timings.SpeculativeAcceptanceRate,
                    },
                    Memory = new RunMemoryDto
                    {
                        WeightsBytes = modelWeightsBytes,
                        RepackedBytes = repackedBytes,
                        ComputeBytes = computeBytes,
                        KvCacheBytes = kvCacheBytes,
                        TotalBytes = totalMemory,
                    },
                    Backend = new RunBackendDto
                    {
                        SamplerPath = samplerPath,
                        DecodeGraph = decodeGraph,
                    },
                };
                Console.WriteLine(JsonSerializer.Serialize(result, CliJsonContext.Default.RunJsonResult));
            }
            else
            {
                Console.WriteLine();
                AnsiConsole.WriteLine();

                // Header grid: title left, hero metric right
                var headerGrid = new Grid();
                headerGrid.AddColumn(new GridColumn().NoWrap());
                headerGrid.AddColumn(new GridColumn().NoWrap().RightAligned());
                headerGrid.AddRow(
                    new Markup("[bold]Generation Complete[/]"),
                    new Markup($"[bold green]{decodeTokPerSec:F2} tok/s[/]"));

                // Build body lines
                var bodyLines = new List<IRenderable>();
                bodyLines.Add(new Markup("  [bold]Performance[/]"));
                bodyLines.Add(new Markup(PerfLine("Prefill", promptEvalMs, promptLen, prefillTokPerSec)));
                bodyLines.Add(new Markup(PerfLine("Decode", evalMs, evalSteps, decodeTokPerSec)));
                bodyLines.Add(new Markup(PerfLine("Sampling", samplerMs, generated, null)));
                bodyLines.Add(new Markup("  [dim]──────────────────────────────────────────────────────[/]"));
                bodyLines.Add(new Markup(PerfLine("Total", totalMs, totalTokens, totalTokPerSec)));
                bodyLines.Add(new Markup(PerfLine("Load", loadMs, null, null)));
                bodyLines.Add(new Text(""));
                bodyLines.Add(new Markup("  [bold]Memory[/]"));
                bodyLines.Add(new Markup(MemLine("Weights", modelWeightsBytes,
                    repackedBytes > 0 ? "(memory-mapped, +repacked below)" : "(memory-mapped)")));
                if (repackedBytes > 0)
                    bodyLines.Add(new Markup(MemLine("Repacked (R4)", repackedBytes, "(committed)")));
                bodyLines.Add(new Markup(MemLine("Compute", computeBytes, null)));
                string kvLabel = kvConfig.IsQuantized
                    ? $"({cacheSize} slots, K:{settings.CacheTypeK} V:{settings.CacheTypeV})"
                    : $"({cacheSize} slots)";
                bodyLines.Add(new Markup(MemLine("KV Cache", kvCacheBytes, kvLabel)));
                bodyLines.Add(new Markup("  [dim]──────────────────────────────────────────────────────[/]"));
                bodyLines.Add(new Markup(MemLine("Total", totalMemory, null)));
                bodyLines.Add(new Text(""));

                var finishReasonStr = finishReason.ToString().ToLowerInvariant();
                bodyLines.Add(new Markup($"  [dim]{Markup.Escape(finishReasonStr)} | {promptLen} prompt, {generated} generated[/]"));

                var backendBits = $"sampler: {samplerPath}";
                if (decodeGraph is not null)
                    backendBits += $" | cuda graph: {decodeGraph}";
                bodyLines.Add(new Markup($"  [dim]{Markup.Escape(backendBits)}[/]"));

                // Assemble panel
                var panelContent = new Rows(
                    new Text(""),
                    headerGrid,
                    new Text(""),
                    new Rows(bodyLines),
                    new Text(""));

                var panel = new Panel(panelContent)
                    .Border(BoxBorder.Rounded)
                    .Padding(2, 0);

                AnsiConsole.Write(panel);
            }
        }
        finally
        {
            // Dispose in reverse dependency order: composite first, then each inner adapter.
            compositeAdapter?.Dispose();
            foreach (var inner in innerAdapters)
                inner.Dispose();
            pagedFactory?.Dispose();
            cudaPagedFactory?.Dispose();
            model.Dispose();
            gguf?.Dispose();
            safetensorsSource?.Dispose();
        }

        return 0;
    }

    private static string PerfLine(string label, double ms, int? tokens, double? tokPerSec)
    {
        var labelPart = $"[dim]{label,-14}[/]";
        var msPart = $"{ms,10:N1} ms";
        var tokensPart = tokens.HasValue ? $"{tokens.Value,6:N0} tokens" : "              ";
        var toksPart = tokPerSec.HasValue ? $"{tokPerSec.Value,10:F2} tok/s" : "";
        return $"  {labelPart} {msPart}   {tokensPart}   {toksPart}";
    }

    private static string MemLine(string label, long bytes, string? annotation)
    {
        var labelPart = $"[dim]{label,-14}[/]";
        var sizePart = $"{FormatHelpers.FormatMiB(bytes),12}";
        var annPart = annotation != null ? $"   [dim]{Markup.Escape(annotation)}[/]" : "";
        return $"  {labelPart} {sizePart}{annPart}";
    }

    private static Core.Configuration.ResponseFormat BuildJsonSchemaFormat(string? schema)
    {
        if (string.IsNullOrEmpty(schema))
            throw new InvalidOperationException("--schema is required when --response-format is json_schema");

        string schemaJson = schema.StartsWith('@')
            ? File.ReadAllText(schema[1..])
            : schema;

        return new Core.Configuration.ResponseFormat.JsonSchema { Schema = schemaJson };
    }

    private static Core.Configuration.ResponseFormat BuildRegexFormat(string? pattern)
    {
        if (string.IsNullOrEmpty(pattern))
            throw new InvalidOperationException("--pattern is required when --response-format is regex");

        return new Core.Configuration.ResponseFormat.Regex { Pattern = pattern };
    }

    private static Core.Configuration.ResponseFormat BuildGrammarFormat(string? grammar)
    {
        if (string.IsNullOrEmpty(grammar))
            throw new InvalidOperationException("--grammar is required when --response-format is grammar");

        string grammarText = grammar.StartsWith('@')
            ? File.ReadAllText(grammar[1..])
            : grammar;

        return new Core.Configuration.ResponseFormat.Grammar { GbnfGrammar = grammarText };
    }

    private static string InferQuantLabel(string resolvedPath, string? quantFlag)
    {
        if (!string.IsNullOrEmpty(quantFlag))
            return quantFlag;

        var match = Regex.Match(Path.GetFileName(resolvedPath), @"\.(Q[\w]+)\.gguf$", RegexOptions.IgnoreCase);
        return match.Success ? match.Groups[1].Value : "unknown";
    }

    /// <summary>
    /// Detects whether <paramref name="modelArg"/> points at a HuggingFace
    /// safetensors checkpoint directory: an existing directory containing a
    /// <c>config.json</c> alongside either a <c>*.safetensors</c> file or a
    /// <c>model.safetensors.index.json</c> shard index. Returns the resolved
    /// directory path, or <see langword="null"/> when it is not such a
    /// directory (leaving the GGUF resolution path unchanged).
    /// </summary>
    internal static string? TryResolveHfDirectory(string modelArg)
    {
        if (string.IsNullOrEmpty(modelArg) || !Directory.Exists(modelArg))
            return null;

        if (!File.Exists(Path.Combine(modelArg, "config.json")))
            return null;

        bool hasWeights =
            File.Exists(Path.Combine(modelArg, "model.safetensors.index.json"))
            || Directory.EnumerateFiles(modelArg, "*.safetensors", SearchOption.TopDirectoryOnly).Any();

        return hasWeights ? Path.GetFullPath(modelArg) : null;
    }

    private static int ResolveGpuLayers(Settings settings, ModelConfig config)
    {
        if (settings.GpuLayers.HasValue)
            return Math.Clamp(settings.GpuLayers.Value, 0, config.NumLayers);
        // Default: 0 for cpu device, all layers for gpu device
        return settings.Device.StartsWith("gpu", StringComparison.OrdinalIgnoreCase)
            ? config.NumLayers : 0;
    }

    private static int ParseGpuId(string device)
    {
        // "gpu" → 0, "gpu:0" → 0, "gpu:1" → 1
        int colonIdx = device.IndexOf(':');
        if (colonIdx < 0) return 0;
        return int.Parse(device.AsSpan(colonIdx + 1));
    }

    /// <summary>
    /// Builds a <see cref="RoPEOverrideOptions"/> from CLI flags. Returns null (no-op) when none
    /// of the RoPE override flags were set.
    /// </summary>
    private static RoPEOverrideOptions? BuildRoPEOverride(Settings settings)
    {
        RoPEScalingType? scalingType = settings.RopeScaling is null ? null : ParseRopeScalingType(settings.RopeScaling);

        var overrides = new RoPEOverrideOptions
        {
            ScalingType = scalingType,
            FreqBase = settings.RopeFreqBase,
            ScalingFactor = settings.RopeScale,
            OrigMaxSeqLen = settings.YarnOrigCtx,
            AttnFactor = settings.YarnAttnFactor,
            BetaFast = settings.YarnBetaFast,
            BetaSlow = settings.YarnBetaSlow,
        };
        return overrides.HasAnyOverride ? overrides : null;
    }

    private static RoPEScalingType ParseRopeScalingType(string value) => value.ToLowerInvariant() switch
    {
        "none" => RoPEScalingType.None,
        "linear" => RoPEScalingType.Linear,
        "yarn" => RoPEScalingType.YaRN,
        "ntk" => RoPEScalingType.NTK,
        "dynamic" or "dynamic_ntk" or "dynamic-ntk" => RoPEScalingType.DynamicNTK,
        "su" or "longrope" => RoPEScalingType.Su,
        _ => throw new InvalidOperationException(
            $"Unknown --rope-scaling value '{value}'. Expected: none, linear, yarn, ntk, dynamic."),
    };

    /// <summary>
    /// Parses <c>--logit-bias</c> entries of the form <c>token_id=bias</c> (e.g. <c>15043=-100</c>)
    /// into a token-id-keyed dictionary. Returns null when no entries are given.
    /// </summary>
    private static IReadOnlyDictionary<int, float>? ParseLogitBias(string[] entries)
    {
        if (entries.Length == 0)
            return null;

        var result = new Dictionary<int, float>(entries.Length);
        foreach (var entry in entries)
        {
            int eq = entry.IndexOf('=');
            if (eq <= 0 || eq == entry.Length - 1)
                throw new InvalidOperationException(
                    $"Invalid --logit-bias entry '{entry}'. Expected 'token_id=bias' (e.g. '15043=-100').");

            if (!int.TryParse(entry.AsSpan(0, eq), out int tokenId))
                throw new InvalidOperationException($"Invalid token id in --logit-bias entry '{entry}'.");
            if (!float.TryParse(entry.AsSpan(eq + 1), System.Globalization.CultureInfo.InvariantCulture, out float bias))
                throw new InvalidOperationException($"Invalid bias value in --logit-bias entry '{entry}'.");

            result[tokenId] = bias;
        }
        return result;
    }

    /// <summary>
    /// Human-readable label for the sampler path that will run, mirroring the auto-build logic in
    /// <c>SamplerPipeline</c> (greedy when temp ≤ 0; bounded "fast top-k" when top-k is set and
    /// top-p/min-p are disabled; otherwise the full step pipeline).
    /// </summary>
    private static string BuildSamplerPath(Settings settings)
    {
        if (settings.Temperature <= 0f)
            return "greedy (argmax)";
        if (settings.TopK > 0 && settings.TopP >= 1.0f && settings.MinP <= 0f)
            return "fast top-k";
        return "full pipeline";
    }

    private static string BuildSamplingLabel(Settings settings)
    {
        if (settings.Temperature <= 0)
            return "greedy";

        var parts = new List<string> { $"temp={settings.Temperature:F1}" };
        if (settings.TopK > 0) parts.Add($"top-k={settings.TopK}");
        if (settings.TopP < 1.0f) parts.Add($"top-p={settings.TopP:F2}");
        if (settings.MinP > 0f) parts.Add($"min-p={settings.MinP:F2}");
        if (settings.RepeatPenalty != 1.0f) parts.Add($"rep={settings.RepeatPenalty:F2}");
        if (settings.Seed.HasValue) parts.Add($"seed={settings.Seed.Value}");
        return string.Join(", ", parts);
    }

    private static long ComputeQuantizedKvBytes(ModelConfig config, int cacheSize, KvCacheConfig kvConfig)
    {
        int kvStride = config.NumKvHeads * config.HeadDim;
        int window = Math.Min(kvConfig.MixedPrecisionWindowSize, cacheSize);
        int quantSlots = Math.Max(0, cacheSize - window);
        int fpBytesPerRow = kvStride * sizeof(float); // FP32 on CPU, FP16 on GPU (close enough for estimate)

        int kQuantRowBytes = kvConfig.KeyDType switch
        {
            KvCacheDType.Q8_0 => kvStride / 32 * 34,
            KvCacheDType.Q4_0 => kvStride / 32 * 18,
            _ => fpBytesPerRow
        };
        int vQuantRowBytes = kvConfig.ValueDType switch
        {
            KvCacheDType.Q8_0 => kvStride / 32 * 34,
            KvCacheDType.Q4_0 => kvStride / 32 * 18,
            _ => fpBytesPerRow
        };

        // Quantized region + full-precision window
        long quantBytes = (long)config.NumLayers * quantSlots * (kQuantRowBytes + vQuantRowBytes);
        long windowBytes = (long)config.NumLayers * window * fpBytesPerRow * 2; // K + V
        return quantBytes + windowBytes;
    }

}
