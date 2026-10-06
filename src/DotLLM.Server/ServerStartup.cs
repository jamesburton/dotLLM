using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Engine.Scheduler;
using DotLLM.HuggingFace;
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
        // A profile name resolves to its base model (#716); profiles shadow a model that happens to share the name.
        if (ModelProfileStore.Resolve(modelArg) is { } resolved)
            return ModelResolver.ResolveLocal(resolved.BaseReference, quant, includeOllama: true);
        return ModelResolver.ResolveLocal(modelArg, quant, includeOllama: true);
    }

    /// <summary>
    /// The id a loaded model is known by: the profile name when <paramref name="requested"/> names a profile (so a request for that alias
    /// matches the active model and residency keys by it), otherwise the file stem.
    /// </summary>
    public static string ModelIdFor(string requested, string resolvedPath) =>
        ModelProfileStore.NormalizeName(requested) is { } n && ModelProfileStore.TryGet(n) is not null
            ? n
            // An ollama blob is named sha256-<hex>: key the model by the ollama name it was requested under.
            : OllamaStore.IsBlobPath(resolvedPath) && OllamaRef.TryParse(requested) is { } o ? o.ToString()
            : Path.GetFileNameWithoutExtension(resolvedPath);

    /// <summary>
    /// <see cref="ResolveModelPath"/>, falling back to a Hub download when <paramref name="autoPull"/> is on and the reference names a Hub repo
    /// (<c>--auto-pull</c>, issue #714). Off by default: a remote client must not be able to start multi-gigabyte downloads by naming a repo.
    /// </summary>
    public static async Task<string?> ResolveOrPullAsync(string modelArg, string? quant, bool autoPull, CancellationToken ct)
    {
        string? path = ResolveModelPath(modelArg, quant);
        if (path is not null || !autoPull) return path;

        string baseRef = ModelProfileStore.Resolve(modelArg)?.BaseReference ?? modelArg;
        if (OllamaRef.TryParse(baseRef) is { } ollamaRef && !ModelResolver.Parse(baseRef).IsRepo)
        {
            using var registry = new OllamaRegistry();
            Console.WriteLine($"[dotllm] --auto-pull: downloading {ollamaRef} from the ollama registry");
            return await ModelResolver.PullOllamaAsync(ollamaRef, registry, progress: null, ct).ConfigureAwait(false);
        }
        var reference = ModelResolver.Parse(baseRef);
        if (!reference.IsRepo) return null;
        using var client = new HuggingFaceClient();
        using var downloader = new HuggingFaceDownloader();
        Console.WriteLine($"[dotllm] --auto-pull: downloading {modelArg}");
        return await ModelResolver.PullAsync(reference, quant, client, downloader, progress: null, ct).ConfigureAwait(false);
    }

    /// <summary>The message for an unresolvable model name, with the way out.</summary>
    public static string NotFoundMessage(string modelArg)
    {
        string baseRef = ModelProfileStore.Resolve(modelArg)?.BaseReference ?? modelArg;
        bool hf = ModelResolver.Parse(baseRef).IsRepo;
        string pull = !hf && OllamaRef.TryParse(baseRef) is { } o ? $"dotllm model pull ollama:{o}" : $"dotllm model pull {modelArg}";
        return $"Model not found: {modelArg}. Download it with `{pull}` or POST /v1/models/pull" +
               (hf || OllamaRef.TryParse(baseRef) is not null ? ", or start the server with --auto-pull." : ".");
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
        if (!DeviceSelector.IsAuto(options.Device))
            return LoadModelCore(resolvedPath, options);

        // --device auto (#722): try the best device first and fall through on a failed load. The server keeps "auto" as its configured device
        // (so later on-demand loads choose again, per model size); the device actually used is recorded in ResolvedDevice.
        // CPU is only reached here by the user's own choice of "auto"; when a GPU candidate failed first, that is surfaced as a prominent
        // WARNING (console + ServerState.DeviceFallbackWarning -> /props) rather than a quiet log line (#733).
        Exception? last = null;
        var gpuFailures = new List<string>();
        foreach (string device in DeviceSelector.Candidates(new FileInfo(resolvedPath).Length))
        {
            try
            {
                Console.WriteLine($"[dotllm] --device auto: trying {device}");
                var loaded = LoadModelCore(resolvedPath, options with { Device = device });
                loaded.Options = loaded.Options with { Device = options.Device, ResolvedDevice = device };
                if (device == "cpu" && gpuFailures.Count > 0)
                {
                    loaded.DeviceFallbackWarning = DeviceSelector.FallbackWarning(ModelIdFor(options.Model, resolvedPath), gpuFailures);
                    Console.WriteLine($"[dotllm] WARNING: {loaded.DeviceFallbackWarning}");
                }
                return loaded;
            }
            catch (Exception ex) when (device != "cpu")
            {
                last = ex;
                gpuFailures.Add($"{device}: {ex.Message}");
                Console.WriteLine($"[dotllm] {device} could not load this model ({ex.Message}); trying the next device");
            }
        }
        throw last ?? new InvalidOperationException("No device could load the model.");
    }

    private static ServerState LoadModelCore(string resolvedPath, ServerOptions options)
    {
        Console.WriteLine($"[dotllm] Loading model from {resolvedPath}...");
        var gguf = GgufFile.Open(resolvedPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        config = GgufModelConfigExtractor.ApplyRoPEOverride(config, options.RopeOverride);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        var threading = new ThreadingConfig(options.Threads, options.DecodeThreads);

        int gpuLayers = options.GpuLayers.HasValue
            ? Math.Clamp(options.GpuLayers.Value, 0, config.NumLayers)
            : options.Device.StartsWith("gpu", StringComparison.OrdinalIgnoreCase) ? config.NumLayers : 0;

        IModel model;
        Func<int, IKvCache>? vulkanKvFactory = null;
        if (DotLLM.Vulkan.VulkanModelLoader.IsVulkanDeviceString(options.Device))
        {
            // Shared per-architecture Vulkan dispatch (#259). Before this branch `--device vulkan` fell
            // through to the CPU path silently (the string does not start with "gpu").
            Console.WriteLine($"[dotllm] Vulkan inference ({DotLLM.Vulkan.VulkanModelLoader.SharedDevice.DeviceName})");
            (model, vulkanKvFactory) = DotLLM.Vulkan.VulkanModelLoader.CreateSharedFromGguf(gguf, config);
        }
        else
        {
            // One dispatch for CPU / all-GPU / partial (#729). Architectures that cannot split layers
            // (Nemotron-H, Qwen3MoeHybrid, Mamba-3) are loaded all-GPU, or the load FAILS with an actionable error (never a silent CPU fallback) —
            // they must never reach HybridTransformerModel (dense Llama-style tensor naming only).
            int gpuId = ParseGpuId(options.Device);
            var plan = GpuOffloadPlanner.Plan(config.Architecture, gpuLayers, config.NumLayers);
            Console.WriteLine(plan.Mode switch
            {
                GpuOffloadMode.Cpu => $"[dotllm] CPU inference ({threading.EffectiveThreadCount} threads)",
                GpuOffloadMode.FullGpu => $"[dotllm] GPU {gpuId} inference",
                GpuOffloadMode.Partial => $"[dotllm] Hybrid inference ({gpuLayers} GPU + {config.NumLayers - gpuLayers} CPU layers)",
                _ => $"[dotllm] GPU {gpuId} inference (partial offload unsupported for {config.Architecture}; all layers must fit)",
            });
            (model, _) = DotLLM.Cuda.CudaModelLoader.CreateForGpuLayers(
                gguf, config, gpuLayers, gpuId, threading,
                w => Console.WriteLine($"[dotllm] WARNING: {w}"));
        }

        // Create chat template. The declared template is untrusted input from the GGUF's
        // tokenizer.chat_template metadata — a parse failure (unsupported Jinja construct,
        // malformed template, etc.) must not take down the whole server process (#273). Fall
        // back to the plain completion-style transcript and keep loading; chat formatting will
        // look wrong for this model, but the model still loads and serves.
        JinjaChatTemplate? declaredChatTemplate;
        try
        {
            declaredChatTemplate = GgufChatTemplateFactory.TryCreate(gguf.Metadata, tokenizer, config.Architecture);
        }
        catch (JinjaException ex)
        {
            Console.WriteLine(
                $"[dotllm] WARNING: model's declared chat_template failed to parse ({ex.Message}); " +
                "falling back to a plain completion-style transcript. Chat formatting will not match " +
                "this model's expected format until the template issue is fixed.");
            declaredChatTemplate = null;
        }

        if (declaredChatTemplate is null)
            Console.WriteLine("[dotllm] Model has no usable GGUF chat template; using a plain completion-style transcript.");
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
        DotLLM.Cuda.CudaPagedKvCacheFactory? cudaPagedFactory = null;
        PrefixTrieManager? prefixTrieManager = null;
        if (vulkanKvFactory is not null)
        {
            // Vulkan models own their device-resident KV storage. Like the CUDA paths, requests run
            // one at a time through the per-request TextGenerator (no ForwardBatch scheduler, no
            // paged/quantized KV, no cross-request prefix reuse yet).
            if (options.UsePaged)
                Console.WriteLine("[dotllm] Paged KV-cache not supported on Vulkan; using the model's own KV-cache.");
            var vkFactory = vulkanKvFactory;
            kvFactory = (cfg, size) => vkFactory(size);
        }
        else if (model is DotLLM.Cuda.CudaTransformerModel cudaModel)
        {
            if (options.UsePaged && kvConfig.IsQuantized)
            {
                Console.WriteLine("[dotllm] Paged KV-cache does not support quantization yet, using quantized GPU cache.");
                kvFactory = (cfg, size) => cudaModel.CreateKvCache(size, kvConfig);
            }
            else if (options.UsePaged)
            {
                // Issue #252: GPU-resident block pool + gather-into-scratch attention dispatch.
                // NOT wired into ContinuousBatchScheduler (pagedFactory stays null below) — the
                // scheduler runs through IModel.ForwardBatch, which CudaTransformerModel doesn't
                // yet override, so CUDA requests continue to serve one at a time through the
                // per-request TextGenerator path exactly as before this cache existed. Only the
                // per-request KV-cache backing changes (block-based instead of one flat buffer).
                cudaPagedFactory = new DotLLM.Cuda.CudaPagedKvCacheFactory(
                    DotLLM.Core.Attention.KvGeometry.FromConfig(config));
                var factory = cudaPagedFactory;
                kvFactory = (cfg, size) => cudaModel.CreatePagedKvCache(factory.Pool, size);
                Console.WriteLine("[dotllm] Using GPU paged KV-cache (block-based allocation, single-request dispatch)");
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
            if (options.UsePaged)
                Console.WriteLine("[dotllm] Paged KV-cache not supported with hybrid GPU, using hybrid cache.");
            kvFactory = (cfg, size) => hybridModel.CreateKvCache(size);
        }
        else if (model is DotLLM.Cuda.Architectures.HybridQwen3HybridDenseTransformerModel qwen3SplitModel)
        {
            // #729: partial-offload Qwen3HybridDense owns a split (GPU head / CPU tail) cache.
            if (options.UsePaged)
                Console.WriteLine("[dotllm] Paged KV-cache not supported with hybrid GPU, using the model's split cache.");
            kvFactory = (cfg, size) => qwen3SplitModel.CreateKvCache(size);
        }
        else if (model is DotLLM.Cuda.Architectures.CudaNemotronHTransformerModel nemotronHModel)
        {
            // #729: the all-GPU fallback for a partial request on Nemotron-H; sparse attention-only KV.
            if (options.UsePaged)
                Console.WriteLine("[dotllm] Paged KV-cache not supported for Nemotron-H on GPU; using the model's own cache.");
            kvFactory = (cfg, size) => nemotronHModel.CreateKvCache(size);
        }
        else if (model is DotLLM.Cuda.Architectures.CudaQwen3HybridDenseTransformerModel qwen3HybridDenseModel)
        {
            // Issue #274: this architecture (e.g. Bonsai-27B) owns its K/V storage internally
            // (a per-attention-layer F16 device cache sized to AttentionLayerCount, not
            // NumLayers — see CreateKvCache's doc). Without this branch the model matched
            // neither CudaTransformerModel nor HybridTransformerModel above and fell through to
            // the generic PagedKvCacheFactory/SimpleKvCache paths, both of which assume a model
            // that does NOT own its KV storage and size a host-RAM pool/buffer against the
            // model's full NumLayers and MaxSequenceLength — tens of GB for a 27B hybrid model,
            // causing an unhandled startup OOM (paged) or per-request OOM under load (simple).
            // The returned handle is length-only (a single int) — no host or device allocation
            // happens here; the real per-layer GPU buffers are sized correctly inside
            // EnsureF16KvCache using AttentionLayerCount.
            if (options.UsePaged)
                Console.WriteLine("[dotllm] Paged KV-cache not supported for Qwen3HybridDense hybrid GPU model (#274); using the model's own internal KV-cache.");
            else if (kvConfig.IsQuantized)
                Console.WriteLine("[dotllm] KV-cache quantization not supported for Qwen3HybridDense hybrid GPU model (#274); using the model's own internal KV-cache.");
            kvFactory = (cfg, size) => qwen3HybridDenseModel.CreateKvCache(size);
        }
        else if (model is DotLLM.Cuda.Architectures.CudaQwen3MoeHybridTransformerModel qwen3MoeHybridModel)
        {
            // Issue #274: same rationale as the Qwen3HybridDense branch above — this
            // architecture also owns its K/V storage internally (identical
            // EnsureF16KvCache/AttentionLayerCount pattern) and must not be routed through the
            // generic paged/simple KV-cache paths.
            if (options.UsePaged)
                Console.WriteLine("[dotllm] Paged KV-cache not supported for Qwen3MoeHybrid hybrid GPU model (#274); using the model's own internal KV-cache.");
            else if (kvConfig.IsQuantized)
                Console.WriteLine("[dotllm] KV-cache quantization not supported for Qwen3MoeHybrid hybrid GPU model (#274); using the model's own internal KV-cache.");
            kvFactory = (cfg, size) => qwen3MoeHybridModel.CreateKvCache(size);
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

        PrefixCache? prefixCache = options.PromptCacheEnabled && vulkanKvFactory is null
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

        // MTP self-speculative decoding (issue #253) — opt-in for serve (unlike run/chat's
        // auto-detect default), since engaging it also takes the continuous-batch scheduler
        // offline for this model (see below). Only actually engages requests when the loaded
        // checkpoint carries an MTP head; otherwise this is a no-op even with --mtp set.
        // An explicit concurrency hint (>= 5, Vulkan) wins over --mtp: the continuous-batch scheduler measured +73% aggregate decode at 8
        // concurrent streams and cannot run speculative decoding, so MTP is dropped for this model with a log line.
        bool concurrencyOverMtp = options.MtpEnabled && options.ExpectedConcurrency >= 5 && vulkanKvFactory is not null
            && !string.Equals(Environment.GetEnvironmentVariable("DOTLLM_VK_SCHEDULER"), "0", StringComparison.Ordinal);
        if (concurrencyOverMtp && model.SupportsMtp)
            Console.WriteLine("[dotllm] --expected-concurrency >= 5: serving through the continuous-batch scheduler; MTP self-speculation is disabled for this model.");
        bool mtpActive = options.MtpEnabled && draftModel is null && model.SupportsMtp && !concurrencyOverMtp;
        if (options.MtpEnabled && draftModel is null && model.SupportsMtp && !concurrencyOverMtp)
            Console.WriteLine($"[dotllm] MTP self-speculative decoding: K={options.SpeculativeCandidates} (model carries an MTP head)");
        else if (options.MtpEnabled && draftModel is null && !model.SupportsMtp)
            Console.WriteLine("[dotllm] --mtp was set but this checkpoint has no MTP head (nextn.* tensors) — ignoring.");

        var generator = new TextGenerator(model, tokenizer, kvFactory, prefixCache,
            draftModel: draftModel, speculativeCandidates: options.SpeculativeCandidates,
            mtpEnabled: options.MtpEnabled, mtpAdaptive: true, recurrentPrefixCache: true,
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
        // decoding is off — the scheduler doesn't support draft models (or MTP self-speculative
        // decoding, same restriction) in this iteration, and GPU/hybrid models keep their existing
        // single-request path until the IModel.ForwardBatch override lands in those backends.
        ContinuousBatchSchedulerService? scheduler = null;
        // Vulkan hybrid models (per-sequence GDN state + serial ForwardBatch) can run through the scheduler, but the
        // serial TextGenerator path is faster until recurrent prefix restore + fused batching land, so this is opt-in:
        // DOTLLM_VK_SCHEDULER=1. No paged pool and no prefix trie (KV-only, paged-only) on this path.
        string? vkSchedEnv = Environment.GetEnvironmentVariable("DOTLLM_VK_SCHEDULER");
        bool vulkanScheduler = vulkanKvFactory is not null && model.SupportsThreadedSequenceState
            && (string.Equals(vkSchedEnv, "1", StringComparison.Ordinal)
                || (options.ExpectedConcurrency >= 5 && !string.Equals(vkSchedEnv, "0", StringComparison.Ordinal)));
        if ((pagedFactory is not null || vulkanScheduler) && kvFactory is not null && draftModel is null && !mtpActive)
        {
            var schedulerOptions = ResolveSchedulerOptions(options);
            if (vulkanScheduler)
            {
                // Every active sequence owns a device KV cache + a GDN state slot on a (usually unified) GPU heap, and there is
                // no byte-budget admission without a paged pool, so bound concurrency conservatively. Override with
                // DOTLLM_VK_SCHEDULER_MAX_SEQS.
                int maxSeqs = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_VK_SCHEDULER_MAX_SEQS"), out int m) && m > 0 ? m : 8;
                schedulerOptions = (schedulerOptions ?? new ContinuousBatchSchedulerOptions()) with
                {
                    MaxActiveSequences = Math.Min(schedulerOptions?.MaxActiveSequences ?? maxSeqs, maxSeqs),
                    // Recurrent prefix snapshots (KV prefix + GDN state copies; ~tens of MB each on a 4B hybrid). Opt out with
                    // DOTLLM_SCHED_RECURRENT_PREFIX=0.
                    RecurrentPrefixCacheEntries = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_SCHED_RECURRENT_PREFIX"), out int rp) && rp >= 0 ? rp : 4,
                };
            }

            scheduler = new ContinuousBatchSchedulerService(
                model,
                tokenizer,
                kvFactory,
                options: schedulerOptions,
                pagedPool: pagedFactory?.Pool);
            Console.WriteLine(options.Scheduler?.EnableFairness == true
                ? "[dotllm] Continuous-batch scheduler active (per-API-key fairness on)"
                : vulkanScheduler
                    ? "[dotllm] Continuous-batch scheduler active (Vulkan; DOTLLM_VK_SCHEDULER=1 or --expected-concurrency >= 5)"
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
            CudaPagedFactory = cudaPagedFactory,
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

        // SDK observability headers (#452) — x-request-id, openai-processing-ms and the
        // x-ratelimit-* trio. Registered unconditionally and OUTSIDE the limiter so the headers
        // also land on its 429 short-circuit; the manager is null when limiting is off, in which
        // case only the id/timing headers are emitted.
        // The headers middleware must partition on the SAME resolver as the limiter, or the
        // x-ratelimit-* values it reports come from a different bucket under a host-supplied
        // IApiKeyResolver.
        var apiKeyResolver = state.RateLimitManager is null
            ? null
            : app.Services.GetRequiredService<IApiKeyResolver>();

        app.UseDotLLMResponseHeaders(state.RateLimitManager, apiKeyResolver);

        if (state.RateLimitManager is { } rlm)
            app.UseDotLLMRateLimiting(rlm, apiKeyResolver!);

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
