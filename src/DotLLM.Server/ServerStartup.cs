using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Engine.Scheduler;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Server.RateLimiting;
using DotLLM.Telemetry;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;

namespace DotLLM.Server;

/// <summary>
/// Shared server startup logic used by both the standalone DotLLM.Server exe
/// and the CLI <c>dotllm serve</c> command.
/// </summary>
public static class ServerStartup
{
    /// <summary>
    /// Resolves a model argument (file path or HuggingFace repo ID) to a local GGUF path.
    /// </summary>
    public static string? ResolveModelPath(string modelArg, string? quant)
    {
        // Direct .gguf file path
        if (modelArg.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) && File.Exists(modelArg))
            return modelArg;

        // HuggingFace repo ID — check cached models directory
        var modelsDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models");

        var repoDir = Path.Combine(modelsDir, modelArg.Replace('/', Path.DirectorySeparatorChar));
        if (!Directory.Exists(repoDir))
            return null;

        var ggufFiles = Directory.GetFiles(repoDir, "*.gguf");
        if (quant is not null)
        {
            ggufFiles = ggufFiles.Where(f =>
                Path.GetFileName(f).Contains(quant, StringComparison.OrdinalIgnoreCase)).ToArray();
        }

        return ggufFiles.Length switch
        {
            1 => ggufFiles[0],
            > 1 => ggufFiles.OrderByDescending(f => new FileInfo(f).Length).First(),
            _ => null,
        };
    }

    /// <summary>
    /// Creates a bare <see cref="ServerState"/> with no model loaded.
    /// The server starts and serves UI, but inference requests return 503 until a model is loaded.
    /// </summary>
    public static ServerState CreateBareState(ServerOptions options) => new()
    {
        Options = options,
        IsReady = false,
        LoraRegistry = CreateLoraRegistry(),
        Residency = CreateResidencyManager(options),
    };

    /// <summary>
    /// Builds a <see cref="ModelResidencyManager"/> from the residency-related fields of
    /// <see cref="ServerOptions"/> (#369).
    /// </summary>
    public static ModelResidencyManager CreateResidencyManager(ServerOptions options) => new()
    {
        MaxResidentModels = Math.Max(1, options.MaxResidentModels),
        MemoryBudgetBytes = options.ResidentMemoryBudgetBytes,
        DefaultKeepAliveSeconds = options.KeepAliveSeconds,
    };

    /// <summary>
    /// Builds the process-wide LoRA adapter registry. The factory delegate
    /// uses <see cref="PeftAdapterLoader.LoadFromDirectory"/> so adapters can
    /// be loaded from disk via <c>POST /v1/lora/load</c>.
    /// </summary>
    public static ILoraAdapterRegistry CreateLoraRegistry()
        => new LoraAdapterRegistry(
            (name, path) => PeftAdapterLoader.LoadFromDirectory(name, path, baseConfig: null));

    /// <summary>
    /// Loads a model from the given GGUF path and returns a fully populated <see cref="ServerState"/>.
    /// </summary>
    public static ServerState LoadModel(string resolvedPath, ServerOptions options)
    {
        Console.WriteLine($"[dotllm] Loading model from {resolvedPath}...");
        var gguf = GgufFile.Open(resolvedPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        var threading = new ThreadingConfig(options.Threads, options.DecodeThreads);

        int gpuLayers = options.GpuLayers.HasValue
            ? Math.Clamp(options.GpuLayers.Value, 0, config.NumLayers)
            : options.Device.StartsWith("gpu", StringComparison.OrdinalIgnoreCase) ? config.NumLayers : 0;

        IModel model;
        if (gpuLayers <= 0)
        {
            Console.WriteLine($"[dotllm] CPU inference ({threading.EffectiveThreadCount} threads)");
            // Shared per-architecture CPU dispatch — routes hybrid architectures
            // (Nemotron-H, Qwen3MoeHybrid) to their dedicated loaders.
            model = ModelLoader.CreateCpuModelFromGguf(gguf, config, threading);
        }
        else if (gpuLayers >= config.NumLayers)
        {
            int gpuId = ParseGpuId(options.Device);
            Console.WriteLine($"[dotllm] GPU {gpuId} inference");
            model = DotLLM.Cuda.CudaTransformerModel.LoadFromGguf(gguf, config, gpuId);
        }
        else
        {
            int gpuId = ParseGpuId(options.Device);
            Console.WriteLine($"[dotllm] Hybrid inference ({gpuLayers} GPU + {config.NumLayers - gpuLayers} CPU layers)");
            model = DotLLM.Cuda.HybridTransformerModel.LoadFromGguf(gguf, config, gpuLayers, gpuId, threading);
        }

        // Create chat template
        var declaredChatTemplate = GgufChatTemplateFactory.TryCreate(gguf.Metadata, tokenizer, config.Architecture);
        if (declaredChatTemplate is null)
            Console.WriteLine("[dotllm] Model has no GGUF chat template; using a plain completion-style transcript.");
        IChatTemplate chatTemplate = declaredChatTemplate ?? GgufChatTemplateFactory.CreatePlainFallback(tokenizer);

        // Tool call parser
        var toolCallParser = GgufChatTemplateFactory.CreateToolCallParser(gguf.Metadata, config.Architecture);

        // KV-cache configuration. TurboQuant applies uniformly to K and V, so its bit-width and
        // QJL flag are taken from the key dtype string (e.g. "tq4", "tq4q").
        var keyDType = KvCacheConfig.ParseDType(options.CacheTypeK, out int tqBits, out bool tqUseQjl);
        var kvConfig = new KvCacheConfig(
            keyDType,
            KvCacheConfig.ParseDType(options.CacheTypeV),
            TurboQuantBits: tqBits > 0 ? tqBits : 4,
            TurboQuantUseQjl: tqUseQjl);

        Func<ModelConfig, int, IKvCache>? kvFactory = null;
        PagedKvCacheFactory? pagedFactory = null;
        PrefixTrieManager? prefixTrieManager = null;
        if (model is DotLLM.Cuda.CudaTransformerModel cudaModel)
        {
            if (options.UsePaged)
                Console.WriteLine("[dotllm] Paged KV-cache not supported with CUDA, using GPU cache.");
            kvFactory = kvConfig.IsQuantized
                ? (cfg, size) => cudaModel.CreateKvCache(size, kvConfig)
                : (cfg, size) => cudaModel.CreateKvCache(size);
        }
        else if (model is DotLLM.Cuda.HybridTransformerModel hybridModel)
        {
            if (options.UsePaged)
                Console.WriteLine("[dotllm] Paged KV-cache not supported with hybrid GPU, using hybrid cache.");
            kvFactory = (cfg, size) => hybridModel.CreateKvCache(size);
        }
        else if (options.UsePaged && !kvConfig.IsQuantized)
        {
            pagedFactory = new PagedKvCacheFactory(
                config.NumLayers, config.NumKvHeads, config.HeadDim);
            kvFactory = (cfg, size) => pagedFactory.Create(size);
            // Cross-request prefix cache (Step 37). Enabled in tandem with paged.
            prefixTrieManager = new PrefixTrieManager(pagedFactory);
            Console.WriteLine("[dotllm] Using paged KV-cache (block-based allocation) with cross-request prefix cache");
        }
        else if (options.UsePaged && kvConfig.IsQuantized)
        {
            Console.WriteLine("[dotllm] Paged KV-cache does not support quantization yet, using quantized simple cache.");
            kvFactory = (cfg, size) => new QuantizedKvCache(
                DotLLM.Core.Attention.KvGeometry.FromConfig(cfg), size,
                kvConfig.KeyDType, kvConfig.ValueDType, kvConfig.MixedPrecisionWindowSize);
        }
        else if (kvConfig.IsQuantized)
        {
            kvFactory = (cfg, size) => new QuantizedKvCache(
                DotLLM.Core.Attention.KvGeometry.FromConfig(cfg), size,
                kvConfig.KeyDType, kvConfig.ValueDType, kvConfig.MixedPrecisionWindowSize);
        }
        else if (kvConfig.IsTurboQuant)
        {
            Console.WriteLine($"[dotllm] Using TurboQuant KV-cache ({kvConfig.TurboQuantBits}-bit, data-oblivious" +
                (kvConfig.TurboQuantUseQjl ? ", QJL unbiased scores)." : ")."));
            kvFactory = (cfg, size) => new TurboQuantKvCache(
                cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, size,
                kvConfig.TurboQuantBits, kvConfig.TurboQuantSeed, kvConfig.TurboQuantUseQjl);
        }

        PrefixCache? prefixCache = options.PromptCacheEnabled
            ? new PrefixCache(options.PromptCacheSize)
            : null;

        // Load speculative draft model if configured
        IModel? draftModel = null;
        GgufFile? draftGguf = null;
        string draftModelPath = "";
        if (!string.IsNullOrEmpty(options.SpeculativeModel))
        {
            var draftPath = ResolveModelPath(options.SpeculativeModel, null);
            if (draftPath is null)
                throw new InvalidOperationException($"Speculative draft model not found: {options.SpeculativeModel}");

            draftGguf = GgufFile.Open(draftPath);
            var draftConfig = GgufModelConfigExtractor.Extract(draftGguf.Metadata);
            if (!SpeculativeConstants.AreVocabsCompatible(config.VocabSize, draftConfig.VocabSize))
            {
                draftGguf.Dispose();
                throw new InvalidOperationException(
                    $"Draft model vocab size ({draftConfig.VocabSize}) differs from target ({config.VocabSize}) " +
                    $"by more than {SpeculativeConstants.MaxVocabSizeDifference} tokens. " +
                    "Models must share the same base tokenizer.");
            }
            if (draftConfig.VocabSize != config.VocabSize)
                Console.WriteLine($"[dotllm] Note: vocab sizes differ slightly ({draftConfig.VocabSize} vs {config.VocabSize}) — using shared range for speculative comparison.");

            var draftThreading = new ThreadingConfig(options.Threads, options.DecodeThreads);
            draftModel = TransformerModel.LoadFromGguf(draftGguf, draftConfig, draftThreading);
            draftModelPath = draftPath;
            Console.WriteLine($"[dotllm] Speculative decoding: draft={Path.GetFileName(draftPath)}, K={options.SpeculativeCandidates}");
        }

        var generator = new TextGenerator(model, tokenizer, kvFactory, prefixCache,
            draftModel: draftModel, speculativeCandidates: options.SpeculativeCandidates,
            prefixTrieManager: prefixTrieManager,
            prefillChunkSize: options.PrefillChunkSize);
        if (options.PrefillChunkSize > 0)
            Console.WriteLine($"[dotllm] Prefill chunk size: {options.PrefillChunkSize} tokens per forward pass");

        // Diffusion models (DiffusionGemma) route chat completions through a masked-canvas
        // diffusion generator instead of the autoregressive TextGenerator. Built only when the
        // checkpoint carries a DiffusionConfig; null for every AR architecture (path unchanged).
        DiffusionTextGenerator? diffusionGenerator = config.DiffusionConfig is not null
            ? new DiffusionTextGenerator(model, tokenizer, sampler: null, config.DiffusionConfig)
            : null;

        // Warm-up: JIT pre-compilation + CUDA kernel loading. Diffusion models exercise the
        // cacheless hybrid forward + denoise loop; AR models exercise prefill/decode.
        if (diffusionGenerator is not null)
            WarmupRunner.RunDiffusion(diffusionGenerator, tokenizer, options.Warmup);
        else
            WarmupRunner.Run(generator, tokenizer, options.Warmup);
        prefixCache?.Clear(); // Discard warm-up KV-cache entries

        // Continuous-batch scheduler. Enabled when a paged factory is available and speculative
        // decoding is off — the scheduler doesn't support draft models in this iteration, and
        // GPU/hybrid models keep their existing single-request path until the IModel.ForwardBatch
        // override lands in those backends.
        ContinuousBatchSchedulerService? scheduler = null;
        if (pagedFactory is not null && kvFactory is not null && draftModel is null)
        {
            var schedulerOptions = ResolveSchedulerOptions(options);

            scheduler = new ContinuousBatchSchedulerService(
                model,
                tokenizer,
                kvFactory,
                options: schedulerOptions,
                pagedPool: pagedFactory.Pool);
            Console.WriteLine(options.Scheduler?.EnableFairness == true
                ? "[dotllm] Continuous-batch scheduler active (per-API-key fairness on)"
                : "[dotllm] Continuous-batch scheduler active");
        }

        long estimatedBytes = SafeFileLength(resolvedPath);

        return new ServerState
        {
            Options = options,
            Config = config,
            ToolCallParser = toolCallParser,
            KvCacheConfig = kvConfig,
            KvCacheFactory = kvFactory,
            PagedFactory = pagedFactory,
            PrefixCache = prefixCache,
            PrefixTrieManager = prefixTrieManager,
            IsReady = true,
            Model = model,
            Tokenizer = tokenizer,
            ChatTemplate = chatTemplate,
            Generator = generator,
            DiffusionGenerator = diffusionGenerator,
            Scheduler = scheduler,
            LoadedModelPath = resolvedPath,
            CurrentGguf = gguf,
            DraftModel = draftModel,
            DraftModelPath = draftModelPath,
            DraftGguf = draftGguf,
            LoraRegistry = CreateLoraRegistry(),
            Residency = CreateResidencyManager(options),
            EstimatedBytes = estimatedBytes,
            LastUsedUtc = DateTimeOffset.UtcNow,
        };
    }

    /// <summary>
    /// Best-effort file size lookup used for eviction budget accounting (#369). Never
    /// blocks/throws — returns 0 when the size can't be determined. Resolves symlinks first:
    /// <see cref="FileInfo.Length"/> reports <c>0</c> for the reparse point itself on Windows
    /// (verified against a Hugging Face hub cache NTFS symlink, e.g. <c>hf download</c>'s default
    /// layout) rather than the target's actual size, which would otherwise silently defeat the
    /// memory-budget accounting for every model resolved through that cache.
    /// </summary>
    public static long SafeFileLength(string path)
    {
        try
        {
            if (File.ResolveLinkTarget(path, returnFinalTarget: true) is FileInfo target)
                return target.Length;
            return new FileInfo(path).Length;
        }
        catch { return 0; }
    }

    /// <summary>
    /// Derives the effective <see cref="ContinuousBatchSchedulerOptions"/> for a server config:
    /// applies <see cref="ServerOptions.PrefillChunkSize"/> as the scheduler's per-step prefill
    /// admission cap (<see cref="ContinuousBatchSchedulerOptions.MaxPrefillTokensPerStep"/>) when
    /// the <see cref="ServerOptions.Scheduler"/> section doesn't already set one, and wires
    /// per-API-key fairness weights from the rate-limit policy table when fairness is enabled.
    /// Returns <see langword="null"/> when nothing is configured (scheduler defaults apply).
    /// </summary>
    /// <remarks>
    /// Honest semantics note: on the scheduler path this is an <b>admission-level</b> cap — a
    /// single prompt longer than the cap still prefills in one forward pass once admitted
    /// (see <see cref="ContinuousBatchSchedulerOptions.MaxPrefillTokensPerStep"/> remarks). True
    /// intra-prompt chunking applies on the single-request <see cref="TextGenerator"/> path.
    /// </remarks>
    public static ContinuousBatchSchedulerOptions? ResolveSchedulerOptions(ServerOptions options)
    {
        var schedulerOptions = options.Scheduler;

        if (options.PrefillChunkSize > 0 && (schedulerOptions?.MaxPrefillTokensPerStep ?? 0) <= 0)
        {
            schedulerOptions = (schedulerOptions ?? new ContinuousBatchSchedulerOptions()) with
            {
                MaxPrefillTokensPerStep = options.PrefillChunkSize,
            };
        }

        // When fairness is enabled and a rate-limit config is present, source per-API-key
        // fairness weights from each key's RateLimitPolicy.Weight (default 1.0 ⇒ equal share).
        if (schedulerOptions?.EnableFairness == true && options.RateLimit is { } rateLimit)
        {
            schedulerOptions = schedulerOptions with
            {
                FairnessWeightProvider = apiKey => rateLimit.PolicyFor(apiKey)?.Weight ?? 1.0,
            };
        }

        return schedulerOptions;
    }

    /// <summary>
    /// Builds and configures a <see cref="WebApplication"/> with all dotLLM endpoints.
    /// </summary>
    /// <param name="state">Populated server state with loaded model.</param>
    /// <param name="args">Raw command-line arguments for ASP.NET configuration.</param>
    /// <param name="serveUi">When true, also serves the embedded web chat UI.</param>
    public static WebApplication BuildApp(ServerState state, string[] args, bool serveUi = false)
    {
        var builder = WebApplication.CreateSlimBuilder(args);
        builder.Services.AddSingleton(state);

        // Wire source-generated JSON context for AOT-compatible serialization
        builder.Services.ConfigureHttpJsonOptions(options =>
            options.SerializerOptions.TypeInfoResolverChain.Insert(0, ServerJsonContext.Default));

        // CORS — permissive for development and Chat UI
        builder.Services.AddCors(o => o.AddDefaultPolicy(p =>
            p.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader()));

        // Rate limiting — only wired when enabled in config. The manager and
        // resolver are singletons; the middleware is registered later in the
        // pipeline (below). When disabled this is a pure no-op.
        var rateLimitConfig = state.Options.RateLimit;
        if (rateLimitConfig is { Enabled: true })
        {
            var manager = new RateLimitManager(rateLimitConfig);
            state.RateLimitManager = manager;
            builder.Services.AddSingleton(manager);
            builder.Services.AddSingleton<IApiKeyResolver, HeaderApiKeyResolver>();
        }

        // Keep only warning+ logging to avoid noisy request logs
        builder.Logging.SetMinimumLevel(LogLevel.Warning);

        // OpenTelemetry — opt-in via the standard OTEL_EXPORTER_OTLP_ENDPOINT env var.
        // No listener means EngineTelemetry stays zero-overhead.
        if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("OTEL_EXPORTER_OTLP_ENDPOINT")))
        {
            builder.Services.AddDotLLMOpenTelemetry(state.Options.ModelId);
        }

        var app = builder.Build();
        app.UseDeveloperExceptionPage();
        app.UseCors();

        if (state.RateLimitManager is { } rlm)
        {
            var resolver = app.Services.GetRequiredService<IApiKeyResolver>();
            app.UseDotLLMRateLimiting(rlm, resolver);
        }

        app.MapDotLLMEndpoints(serveUi);

        var lifetime = app.Services.GetRequiredService<IHostApplicationLifetime>();
        state.ShutdownToken = lifetime.ApplicationStopping;

        // Start the continuous-batch scheduler's run loop on a background task, cancelled when
        // the host shuts down. Stopped/rebuilt on model swap, activation, and idle-unload (#369) —
        // see ServerState.StartSchedulerLoop, which every one of those paths funnels through.
        state.StartSchedulerLoop();

        // Idle-unload sweep (#369): periodically evicts models past their keep-alive, including
        // the active one (never interrupting an in-flight request — see ServerState.SweepIdleAsync).
        // Runs even with the default single-model configuration, so idle-unload works out of the
        // box for every server, not just multi-model setups.
        _ = Task.Run(() => state.RunIdleSweepLoopAsync(state.Options.IdleSweepInterval, state.ShutdownToken));

        return app;
    }

    private static int ParseGpuId(string device) =>
        device.IndexOf(':') is int ci and > 0
            ? int.Parse(device.AsSpan(ci + 1))
            : 0;
}
