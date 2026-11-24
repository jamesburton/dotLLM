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
    /// Whether a q8_0 KV request can use Q8_0 paged blocks: both K and V q8_0 (no mixed-precision window), a uniform
    /// non-hybrid model, and a head dim on whole Q8_0 blocks. Otherwise the quantized simple cache serves it.
    /// </summary>
    public static bool IsPagedQ8Eligible(KvCacheConfig kvConfig, ModelConfig config, IModel? model)
        => kvConfig.KeyDType == KvCacheDType.Q8_0 && kvConfig.ValueDType == KvCacheDType.Q8_0
           && kvConfig.MixedPrecisionWindowSize == 0 && config.HeadDim % 32 == 0
           && !IsHybridOrRecurrentCpuModel(config, model)
           && DotLLM.Core.Attention.KvGeometry.FromConfig(config).IsUniform;

    /// <summary>
    /// Picks the CPU paged block type (#1284). Quantized (q8_0 K+V) requests get Q8_0 blocks; an explicit f32/f16 request
    /// gets F16; when neither K nor V was specified the default is Q8_0 where eligible (faster and half the bytes of F16,
    /// #1268) and <paramref name="info"/> carries the one-time startup message. Mixed (only K or only V given) is unchanged:
    /// a lone q8_0 side is not eligible and falls back to the quantized simple cache, a lone f16 side stays F16.
    /// </summary>
    /// <param name="kvTypeSpecified">True when the user gave a K or V cache type.</param>
    /// <param name="kvConfig">The parsed KV configuration.</param>
    /// <param name="config">Model configuration.</param>
    /// <param name="model">The loaded model, or <c>null</c>.</param>
    /// <param name="info">Information message to log, or <c>null</c> when nothing was defaulted.</param>
    /// <returns>The block element type.</returns>
    public static DotLLM.Core.Tensors.DType ResolvePagedBlockType(
        bool kvTypeSpecified, KvCacheConfig kvConfig, ModelConfig config, IModel? model, out string? info)
    {
        info = null;
        if (kvConfig.IsQuantized)
            return DotLLM.Core.Tensors.DType.Q8_0;
        if (!kvTypeSpecified && IsPagedQ8Eligible(
                new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q8_0, kvConfig.MixedPrecisionWindowSize), config, model))
        {
            info = "KV cache: no type specified; using Q8_0 paged blocks (faster, half the memory of F16). " +
                   "Use --cache-type-k f16 --cache-type-v f16 for F16.";
            return DotLLM.Core.Tensors.DType.Q8_0;
        }
        return DotLLM.Core.Tensors.DType.Float16;
    }

    /// <summary>
    /// True for CPU hybrid/recurrent models (GDN, SSM or hybrid layer layout, or one of the known hybrid model classes) that must
    /// NOT get the generic paged KV pool: it is sized for <c>NumLayers x MaxSequenceLength</c> although only the attention layers
    /// hold KV, so e.g. Bonsai-2-27B (64 layers, 16 KV layers, 262144 ctx) died at startup with "Insufficient memory" (#760).
    /// </summary>
    internal static bool IsHybridOrRecurrentCpuModel(ModelConfig config, IModel? model)
        => config.HybridLayout is not null || config.SsmConfig is not null || config.GdnConfig is not null
           || model is Qwen3HybridDenseTransformerModel or Qwen3MoeHybridTransformerModel
                or NemotronHTransformerModel or Mamba3TransformerModel;

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
    /// matches the active model and residency keys by it), otherwise the repo-qualified <see cref="ModelResolver.IdForPath"/>
    /// (<c>owner/repo/file.gguf</c>; the bare file stem only for a file outside the model stores).
    /// </summary>
    public static string ModelIdFor(string requested, string resolvedPath) =>
        ModelProfileStore.NormalizeName(requested) is { } n && ModelProfileStore.TryGet(n) is not null
            ? n
            // An ollama blob is named sha256-<hex>: key the model by the ollama name it was requested under.
            : OllamaStore.IsBlobPath(resolvedPath) && OllamaRef.TryParse(requested) is { } o ? o.ToString()
            : ModelResolver.IdForPath(resolvedPath);

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
        // One parser for every entry point (#790): "cuda", a typo or "vulkan:x" must never silently mean the CPU.
        if (!DeviceSpec.TryParse(options.Device, out var spec, out string? deviceError))
            throw new ArgumentException(deviceError);

        if (spec.Kind != DeviceKind.Auto)
        {
            // Explicit device: preflight against the machine, then load on exactly that device. An explicit GPU that
            // cannot be honoured is an error (what/why/opt-in), never a CPU run. ResolvedDevice is always recorded so
            // /props and `model ps` report the same thing as an auto load does.
            string canonical = spec.Canonical;
            long bytes = ModelResolver.FileLength(resolvedPath);
            string name = ModelIdFor(options.Model, resolvedPath);
            if (spec.IsGpu)
            {
                // -1 is "all layers"; auto (#1851) is sized by the loader, like an explicit partial count.
                var plan = DeviceSelector.Plan(spec, bytes, Endpoints.DeviceEndpoint.Describe(),
                    fullOffload: options.GpuLayers is null or -1 && !ExpertOffloadPossible(resolvedPath, options.MoeOffload),
                    autoSplit: spec.Kind == DeviceKind.Cuda && options.GpuLayers is null && AutoSplitBlocker(resolvedPath, options) is null);
                if (plan.Error is not null)
                    throw new DeviceUnavailableException(DeviceSelector.ExplicitFailureMessage(options.Device, name, bytes, plan.Error));
            }
            try
            {
                var explicitState = LoadModelCore(resolvedPath, options with { Device = canonical });
                explicitState.Options = explicitState.Options with { ResolvedDevice = canonical, DeviceFallbackWarning = null };   // (#1822) never a stale warning
                return explicitState;
            }
            catch (Exception ex) when (spec.IsGpu && ex is not (DeviceUnavailableException or InvalidServerOptionException))
            {
                throw new DeviceUnavailableException(
                    DeviceSelector.ExplicitFailureMessage(options.Device, name, bytes, $"loading the model failed: {ex.Message}"), ex);
            }
        }

        // --device auto (#722): try the best device first and fall through on a failed load. The server keeps "auto" as its configured device
        // (so later on-demand loads choose again, per model size); the device actually used is recorded in ResolvedDevice.
        // CPU is only reached here by the user's own choice of "auto"; when a GPU candidate failed first, that is surfaced as a prominent
        // WARNING (console + ServerState.DeviceFallbackWarning -> /props) rather than a quiet log line (#733).
        Exception? last = null;
        var gpuFailures = new List<string>();
        string? splitBlocker = options.GpuLayers is null or GpuOffloadPlanner.AutoGpuLayers ? AutoSplitBlocker(resolvedPath, options) : "explicit --gpu-layers";
        var autoDevices = Endpoints.DeviceEndpoint.Describe();
        var autoPlan = DeviceSelector.Plan(spec, ModelResolver.FileLength(resolvedPath), autoDevices, autoSplit: splitBlocker is null);
        if (options.GpuLayers is null)
            DeviceModelLoader.LogSkippedSplit(autoPlan, autoDevices, splitBlocker, m => Console.WriteLine($"[dotllm] {m}"));
        foreach (string device in autoPlan.Candidates)
        {
            if (options.TernaryAdapterPaths is { Count: > 0 } && TernaryAdapterAttach.BackendHasNoHook(device)) continue;   // no CUDA model has the residual hook (#1711): never load a GPU model the adapters cannot attach to
            try
            {
                Console.WriteLine($"[dotllm] --device auto: trying {device}");
                // #1851: a CUDA device the whole model does not fit is loaded as an automatic GPU/CPU layer split.
                int? layers = options.GpuLayers is null && device == autoPlan.SplitCandidate ? GpuOffloadPlanner.AutoGpuLayers : options.GpuLayers;
                var loaded = LoadModelCore(resolvedPath, options with { Device = device, GpuLayers = layers });
                loaded.Options = loaded.Options with { Device = options.Device, GpuLayers = options.GpuLayers, ResolvedDevice = device, DeviceFallbackWarning = null };   // (#1822) set below when it fell back
                if (device == "cpu" && (gpuFailures.Count > 0 || autoPlan.CpuWarning is not null))
                {
                    // Auto on the CPU always warns: after a failed GPU load (reason + consequence), or because no GPU was servable at all.
                    loaded.DeviceFallbackWarning = gpuFailures.Count > 0
                        ? DeviceSelector.FallbackWarning(ModelIdFor(options.Model, resolvedPath), gpuFailures)
                        : autoPlan.CpuWarning;
                    Console.WriteLine($"[dotllm] WARNING: {loaded.DeviceFallbackWarning}");
                }
                return loaded;
            }
            // With adapters requested, a GPU model that has no residual hook (or whose geometry the adapters do not fit) surfaces as an
            // InvalidServerOptionException: fall through to the next device instead of aborting auto (the CPU attach then decides).
            catch (Exception ex) when (device != "cpu" && (ex is not InvalidServerOptionException || options.TernaryAdapterPaths is { Count: > 0 }))
            {
                last = ex;
                gpuFailures.Add($"{device}: {ex.Message}");
                Console.WriteLine($"[dotllm] {device} could not load this model ({ex.Message}); trying the next device");
            }
        }
        throw last ?? new InvalidOperationException("No device could load the model.");
    }

    /// <summary>
    /// Resolves the requested GPU layer count against the layer count the loader actually sees (<paramref name="numLayers"/>).
    /// A negative request (<see cref="AllGpuLayers"/>) or an unset request on a <c>gpu</c> device means "all layers"; otherwise the
    /// request is clamped to <c>[0, numLayers]</c>; unset on a non-GPU device is 0.
    /// </summary>
    public static int ResolveGpuLayers(int? requested, string device, int numLayers) =>
        GpuOffloadPlanner.IsAuto(requested)
            ? throw new ArgumentException("--gpu-layers auto is resolved by the loader (CudaAutoSplit), not by a layer count.", nameof(requested))
        : requested is < 0 ? numLayers
        : requested.HasValue ? Math.Clamp(requested.Value, 0, numLayers)
        : DeviceSpec.TryParse(device, out var spec, out _) && spec.Kind == DeviceKind.Cuda ? numLayers : 0;

    /// <summary>Sentinel <c>gpu_layers</c> value meaning "every layer, as the loader counts them".</summary>
    public const int AllGpuLayers = -1;

    /// <summary>
    /// Embedding-only encoder checkpoints (BERT / nomic-bert, #739): no generator, KV-cache, chat template
    /// or scheduler; only <c>/v1/embeddings</c> (and the ollama embed routes) serve them. CPU only -- a GPU
    /// device request fails loudly rather than silently running on the CPU.
    /// </summary>
    private static ServerState LoadEncoderModel(
        string resolvedPath, ServerOptions options, GgufFile gguf, ModelConfig config, ITokenizer tokenizer)
    {
        var spec = DeviceSpec.TryParse(options.Device, out var parsed, out _) ? parsed : new DeviceSpec(DeviceKind.Auto, 0);
        // #741: architecture 'bert' (F16 weights) has a Vulkan encoder. Explicit vulkan must succeed or fail loudly; auto uses it when
        // a Vulkan device exists and the weights are F16, else the CPU encoder (with the usual warning); CUDA and the other encoder
        // architectures are refused (no silent CPU run for an explicit GPU request).
        bool vulkanCapable = config.Architecture == Architecture.Bert;   // auto-selects Vulkan only for bert (F16); nomic-bert / nomic-bert-moe (#1471) are explicit --device vulkan
        bool vulkanExplicit = vulkanCapable || config.Architecture is Architecture.NomicBert or Architecture.NomicBertMoe;
        IModel? model = null;
        if (spec.Kind == DeviceKind.Vulkan || (spec.Kind == DeviceKind.Auto && vulkanCapable && DotLLM.Vulkan.VulkanDevice.IsAvailable()))
        {
            if (!vulkanExplicit)
            {
                gguf.Dispose();
                throw new NotSupportedException(
                    $"{config.Architecture} embedding models have no Vulkan backend yet (only architecture 'bert' does, #741). Use --device cpu; there is no silent CPU fallback.");
            }
            try
            {
                var (vkModel, _) = DeviceModelLoader.LoadExact(gguf, config, spec.Kind == DeviceKind.Vulkan ? spec.Canonical : "vulkan", null, new ThreadingConfig(options.Threads, options.DecodeThreads),
                    m => Console.WriteLine($"[dotllm] {m}; embeddings only"), m => Console.WriteLine($"[dotllm] WARNING: {m}"));
                model = vkModel;
            }
            catch (NotSupportedException) when (spec.Kind == DeviceKind.Auto)
            {
                Console.WriteLine("[dotllm] WARNING: the Vulkan encoder cannot serve this file (F16 weights are required); falling back to the CPU encoder.");
            }
        }
        else if (spec.Kind != DeviceKind.Cpu && spec.Kind != DeviceKind.Auto)
        {
            gguf.Dispose();
            throw new NotSupportedException(
                $"{config.Architecture} embedding models run on --device cpu or vulkan (architecture 'bert' only), not '{options.Device}'. There is no silent CPU fallback; Decoder embedders (Qwen3-Embedding) run on --device vulkan.");
        }
        if (model is null)
        {
            var threading = new ThreadingConfig(options.Threads, options.DecodeThreads);
            Console.WriteLine($"[dotllm] CPU inference, {config.Architecture} encoder ({threading.EffectiveThreadCount} threads); embeddings only");
            model = BertEncoderModel.LoadFromGguf(gguf, config, threading);
        }
        return new ServerState
        {
            Options = options,
            Config = config,
            KvCacheConfig = new KvCacheConfig(KvCacheConfig.ParseDType(options.CacheTypeK ?? "f32"), KvCacheConfig.ParseDType(options.CacheTypeV ?? "f32")),
            IsReady = true,
            Model = model,
            Tokenizer = tokenizer,
            LoadedModelPath = resolvedPath,
            CurrentGguf = gguf,
            LoraRegistry = CreateLoraRegistry(),
            Residency = CreateResidencyManager(options),
            EstimatedBytes = SafeFileLength(resolvedPath),
            LastUsedUtc = DateTimeOffset.UtcNow,
        };
    }

    /// <summary>Fitted windows below this many tokens get a loud warning: agent clients cannot work in less.</summary>
    internal const int SmallContextWarningTokens = 8192;

    /// <summary>
    /// Issue #1068: chooses the served context for the CUDA hybrid and applies it to the model. Without this the server
    /// advertised the checkpoint's 262,144 tokens on a 12 GB card that fits a fraction of that, so clients never compacted,
    /// the KV cache outgrew VRAM, and WDDM spilled it into system RAM (requests went from minutes to hours).
    /// </summary>
    /// <param name="model">The loaded, warmed-up model.</param>
    /// <param name="options">Server options (<see cref="ServerOptions.ContextSize"/>).</param>
    /// <param name="mtpActive">Whether MTP self-speculation will run.</param>
    /// <param name="mtpK">MTP draft tokens per round.</param>
    /// <returns>The model's config after the limit.</returns>
    internal static ModelConfig ApplyHybridContextFit(
        DotLLM.Cuda.Architectures.CudaQwen3HybridDenseTransformerModel model, ServerOptions options, bool mtpActive, int mtpK)
    {
        var fit = model.MeasureServingContextFit(mtpActive, mtpK + 1);
        if (fit.SizingError is { } sizingError)
            Console.WriteLine($"[dotllm] WARNING: the context sizing pass was refused: {sizingError}");
        int native = model.NativeMaxSequenceLength;
        int chosen = ChooseServedContext(options.ContextSize, fit.MaxContext, native, out string? warning);
        if (warning is not null)
            Console.WriteLine($"[dotllm] WARNING: {warning}");
        if (chosen < native)
            model.LimitContextLength(chosen);
        string kv = model.KvStorageLabel + (mtpActive ? " KV + MTP" : " KV");
        Console.WriteLine(options.ContextSize is > 0
            ? $"[dotllm] Context window: {model.Config.MaxSequenceLength} tokens (--ctx-size; {fit.MaxContext} fit in free VRAM: {fit.Describe()}, {kv})."
            : $"[dotllm] Context window: {model.Config.MaxSequenceLength} of {native} tokens, fitted to free VRAM ({fit.Describe()}, {kv}). " +
              "Set --ctx-size to override.");
        return model.Config;
    }

    /// <summary>
    /// Issues #1841 / #1836: chooses the served context for the dense CUDA host and applies it. With the GPU paged pool the window
    /// is capped to the pool (a request past it cannot run: the pool was sized to free VRAM at startup), so the prompt-length
    /// check refuses an over-long request at admission instead of the pool failing part-way through prefill. Without it the
    /// window is what fits in free VRAM after warm-up (<see cref="DotLLM.Cuda.CudaTransformerModel.MeasureServingContextFit"/>).
    /// Either way <c>--ctx-size</c> is honoured, and <c>/props</c>, <c>/v1/status</c> and <c>/v1/models</c> report the result.
    /// </summary>
    /// <param name="model">The loaded, warmed-up model.</param>
    /// <param name="options">Server options.</param>
    /// <param name="kv">KV storage types of the non-paged cache.</param>
    /// <param name="paged">The GPU paged-KV factory, or null on the non-paged path.</param>
    /// <param name="prefillChunk">The prefill chunk the generator uses (<see cref="ResolvePrefillChunk"/>), 0 for one pass.</param>
    /// <returns>The model's config after the limit.</returns>
    internal static ModelConfig ApplyCudaDenseContextFit(
        DotLLM.Cuda.CudaTransformerModel model, ServerOptions options, KvCacheConfig kv, DotLLM.Cuda.CudaPagedKvCacheFactory? paged,
        int prefillChunk)
    {
        int native = model.NativeMaxSequenceLength;
        if (paged is not null)
        {
            int pool = paged.Pool.TotalBlocks * paged.Pool.BlockSize;
            int chosen = ChoosePagedServedContext(options.ContextSize, pool, native, out string? pagedWarning);
            if (pagedWarning is not null)
                Console.WriteLine($"[dotllm] WARNING: {pagedWarning}");
            if (chosen < model.Config.MaxSequenceLength)
                model.LimitContextLength(chosen);
            Console.WriteLine($"[dotllm] Context window: {model.Config.MaxSequenceLength} of {native} tokens, capped to the GPU paged KV pool " +
                              $"({pool} tokens, sized to free VRAM){(options.ContextSize is > 0 ? "; --ctx-size " + options.ContextSize : "")}.");
            return model.Config;
        }

        var fit = model.MeasureServingContextFit(kv, prefillChunk);
        if (fit.SizingError is { } sizingError)
            Console.WriteLine($"[dotllm] WARNING: the context fit could not measure free VRAM ({sizingError}); serving the smallest window.");
        int served = ChooseServedContext(options.ContextSize, fit.MaxContext, native, out string? warning,
            smallWindowHint: DenseSmallWindowHint);
        if (warning is not null)
            Console.WriteLine($"[dotllm] WARNING: {warning}");
        if (served < native)
            model.LimitContextLength(served);
        string kvLabel = kv.IsTurboQuantAny ? $"tq{kv.TurboQuantBits}" : kv.IsQuantized ? $"{kv.KeyDType}/{kv.ValueDType}".ToLowerInvariant() : "f16";
        string prefill = prefillChunk > 0 ? $", prefill chunk {prefillChunk}" : ", unchunked prefill";
        Console.WriteLine(options.ContextSize is > 0
            ? $"[dotllm] Context window: {model.Config.MaxSequenceLength} tokens (--ctx-size; {fit.MaxContext} fit in free VRAM: {fit.Describe()}, {kvLabel} KV{prefill})."
            : $"[dotllm] Context window: {model.Config.MaxSequenceLength} of {native} tokens, fitted to free VRAM ({fit.Describe()}, {kvLabel} KV{prefill}). " +
              "Set --ctx-size to override.");
        return model.Config;
    }

    /// <summary>The small-window remedy for the dense CUDA host (#1848): it has no MTP head, and an unchunked prefill's scratch is per token.</summary>
    internal const string DenseSmallWindowHint =
        "Use a smaller KV type (--kv-quant-k/-v q8_0 or q4_0), keep the prefill chunked (--prefill-chunk-size, 1024 by default on --no-paged), " +
        "or free GPU memory.";

    /// <summary>
    /// Prefill chunk the dense CUDA host uses when <c>--prefill-chunk-size</c> is not given (#1862), on its non-paged cache.
    /// Chunked, the forward scratch is a fixed cost instead of about as much per token as f16 KV, so the VRAM-fitted window roughly
    /// doubles; cancellation is checked between chunks, so a client disconnect stops a long prefill. On f16 KV the first chunk's
    /// square G3 score plane (numHeads x chunk^2 x 6 B, 192 MiB for 32 heads) stays inside the plane the fit already retains, while
    /// every later chunk runs G3's offset path (#1847), bounded to the same plane. On q8_0 / q4_0 KV every chunk runs the
    /// tensor-core in-place kernel (<c>attention_flash_mma_prefill_kvq</c>, #1850), later chunks with their position offset, and
    /// needs no plane, so it takes the larger <see cref="DefaultCudaDenseQuantizedPrefillChunkTokens"/>. Measured on framework: see
    /// the PRs for #1862 and #1850.
    /// </summary>
    internal const int DefaultCudaDensePrefillChunkTokens = 1024;

    /// <summary>
    /// Prefill chunk the dense CUDA host uses on q8_0 / q4_0 KV when <c>--prefill-chunk-size</c> is not given (#1850). Quantized KV
    /// never runs G3, so the f16 bound (the first chunk's square plane inside the retained one) does not apply and the chunk is a
    /// pure speed / window trade. TwIL-8B on the RTX 3060, 8.4K-token cold prefill: chunk 1024 = 1,325-1,341 tok/s at a 67,584
    /// window, 2048 = 1,508-1,533 at 65,536, 4096 = 1,501-1,511 at 62,464, one pass = ~1,680 at 25,600. 2048 takes most of the
    /// speed back for a 3% smaller window (user decision, PR #1875).
    /// </summary>
    internal const int DefaultCudaDenseQuantizedPrefillChunkTokens = 2048;

    /// <summary>
    /// The prefill chunk a server uses (#1862): an explicit <c>--prefill-chunk-size</c> always wins, 0 (one pass) included; with none,
    /// where <see cref="TakesCudaDensePrefillChunkDefault"/> holds, <see cref="DefaultCudaDenseQuantizedPrefillChunkTokens"/> on
    /// q8_0 / q4_0 KV and <see cref="DefaultCudaDensePrefillChunkTokens"/> otherwise; one pass everywhere else.
    /// </summary>
    /// <param name="requested">The <c>--prefill-chunk-size</c> value, or null when it was not given.</param>
    /// <param name="cudaDenseDefault">Whether the model takes the dense CUDA default (<see cref="TakesCudaDensePrefillChunkDefault"/>).</param>
    /// <param name="quantizedKv">Whether the KV cache is q8_0 / q4_0 (<see cref="KvCacheConfig.IsQuantized"/>).</param>
    /// <returns>Tokens per prefill forward pass, 0 for one pass.</returns>
    internal static int ResolvePrefillChunk(int? requested, bool cudaDenseDefault, bool quantizedKv = false)
        => requested is int r ? Math.Max(0, r)
            : !cudaDenseDefault ? 0
            : quantizedKv ? DefaultCudaDenseQuantizedPrefillChunkTokens : DefaultCudaDensePrefillChunkTokens;

    /// <summary>
    /// Whether <paramref name="model"/> takes the dense CUDA prefill default (#1862, #1850): the dense CUDA host
    /// (<see cref="DotLLM.Cuda.CudaTransformerModel"/>, not the CUDA hybrids, which chunk themselves) on its non-paged cache, f16
    /// (later chunks run G3's offset path) or q8_0 / q4_0 read in place (later chunks run the tensor-core kvq prefill at their
    /// offset). A quantized cache with an FP16 window ring keeps one pass (serve builds none today; the guard is for when it does):
    /// every chunk would run the scalar kernel, and quantize-on-evict across chunk boundaries has no parity coverage. CPU, Vulkan, HIP and the paged pool never do.
    /// </summary>
    /// <param name="model">The loaded model.</param>
    /// <param name="cudaPagedPool">Whether the GPU paged KV pool serves it.</param>
    /// <param name="kv">KV storage types.</param>
    /// <returns>Whether the default applies.</returns>
    internal static bool TakesCudaDensePrefillChunkDefault(IModel model, bool cudaPagedPool, KvCacheConfig kv)
        => model is DotLLM.Cuda.CudaTransformerModel && !cudaPagedPool && !(kv.IsQuantized && kv.MixedPrecisionWindowSize > 0);

    /// <summary>
    /// The served context on the CUDA paged pool (#1836): <c>--ctx-size</c> or the model's window, never more than the pool holds.
    /// </summary>
    /// <param name="requested">The <c>--ctx-size</c> value, or null.</param>
    /// <param name="poolTokens">Tokens the pool holds.</param>
    /// <param name="native">The model's window.</param>
    /// <param name="warning">Set when <c>--ctx-size</c> asked for more than the pool holds.</param>
    /// <returns>The window to serve.</returns>
    internal static int ChoosePagedServedContext(int? requested, int poolTokens, int native, out string? warning)
    {
        warning = null;
        int want = requested is > 0 and int r ? Math.Min(r, native) : native;
        if (want <= poolTokens) return want;
        if (requested is > 0)
            warning = $"--ctx-size {want} exceeds the {poolTokens} tokens the GPU paged KV pool holds in free VRAM; serving {poolTokens}. " +
                      "Use --no-paged with a smaller KV type, or free GPU memory, for a longer window.";
        return Math.Max(1, poolTokens);
    }

    /// <summary>
    /// Issue #1101: applies <c>--ctx-size</c> to a model that can lower its window but has no VRAM fit (the Vulkan hybrid). With
    /// YaRN its window is factor × original (about 1M for Bonsai-2); without this the flag was ignored on Vulkan and the server
    /// advertised the whole window.
    /// </summary>
    /// <param name="model">The loaded model.</param>
    /// <param name="limitable">The same model, as <see cref="IContextLimitable"/>.</param>
    /// <param name="options">Server options (<see cref="ServerOptions.ContextSize"/>).</param>
    /// <returns>The model's config after the limit.</returns>
    internal static ModelConfig ApplyRequestedContextSize(IModel model, IContextLimitable limitable, ServerOptions options)
    {
        int native = limitable.NativeMaxSequenceLength;
        if (options.ContextSize is > 0 and int requested && requested < native)
            limitable.LimitContextLength(requested);
        Console.WriteLine(options.ContextSize is > 0
            ? $"[dotllm] Context window: {model.Config.MaxSequenceLength} of {native} tokens (--ctx-size)."
            : $"[dotllm] Context window: {native} tokens. Set --ctx-size to lower it.");
        return model.Config;
    }

    /// <summary>
    /// Issue #1324: qwen4exp on Vulkan sizes its K/V + indexer caches at load, so the serve options that decide the window are handed to it first:
    /// <c>--ctx-size</c> becomes the requested capacity (reduced to what fits the device) and the K/V is stored as F16 unless
    /// <c>DOTLLM_VK_QWEN4EXP_KV</c> says otherwise (24 instead of 48 KiB per token; measured against F32 in docs/ATTENTION.md).
    /// </summary>
    /// <param name="config">The checkpoint's configuration.</param>
    /// <param name="options">Server options.</param>
    internal static void PrepareQwen4ExpVulkan(ModelConfig config, ServerOptions options)
    {
        if (config.Architecture != Architecture.Qwen4Exp) return;
        DotLLM.Vulkan.VulkanQwen4ExpTransformerModel.RequestedContextTokens = options.ContextSize;
        if (Environment.GetEnvironmentVariable("DOTLLM_VK_QWEN4EXP_KV") is null && Environment.GetEnvironmentVariable("DOTLLM_VK_QWEN4EXP_QSA") != "0")
            DotLLM.Vulkan.VulkanQwen4ExpTransformerModel.KvStorage = DotLLM.Vulkan.VulkanKvStorage.F16;
    }

    /// <summary>
    /// The served context: an explicit request (capped at the model's window) or the fit; never below one 1024-token granule.
    /// </summary>
    /// <param name="requested">The <c>--ctx-size</c> value, or null.</param>
    /// <param name="fit">Tokens that fit in free VRAM.</param>
    /// <param name="native">The checkpoint's window.</param>
    /// <param name="warning">A warning to log, or null.</param>
    /// <param name="smallWindowHint">Remedies for a small fitted window; null for the hybrid default (KV type, --no-mtp).</param>
    /// <returns>The window to serve.</returns>
    internal static int ChooseServedContext(int? requested, int fit, int native, out string? warning, string? smallWindowHint = null)
    {
        warning = null;
        if (requested is > 0 and int r)
        {
            int chosen = Math.Min(r, native);
            if (chosen > fit)
                warning = $"--ctx-size {chosen} exceeds the {fit} tokens that fit in free VRAM. Requests that grow past the fit will be " +
                          "refused with an out-of-memory error (they are not spilled into system RAM).";
            return chosen;
        }

        int served = Math.Clamp(fit, Math.Min(DeviceMemoryBudget.ContextGranularity, native), native);
        if (fit < SmallContextWarningTokens)
            warning = $"only {fit} tokens of context fit in free VRAM (serving {served}). " +
                      (smallWindowHint ?? "Use a smaller KV type (--kv-quant-k/-v q8_0 or q4_0), --no-mtp, or free GPU memory.");
        return served;
    }

    /// <summary>
    /// Whether the GGUF at <paramref name="path"/> is a <c>nemotron_h_moe</c> model whose routed experts may be offloaded
    /// (#1070), read from its metadata; false when the file cannot be read.
    /// </summary>
    /// <summary>Why the GPU/CPU split host cannot run the model at <paramref name="path"/> (#1851), or null when it can.</summary>
    private static string? AutoSplitBlocker(string path, ServerOptions options)
    {
        try
        {
            using var gguf = GgufFile.Open(path);
            // The config the load will use: --rope-scaling and friends can turn YaRN on (or off), which decides the split (#432).
            var config = GgufModelConfigExtractor.ApplyRoPEOverride(GgufModelConfigExtractor.Extract(gguf.Metadata), options.RopeOverride);
            return DotLLM.Cuda.CudaAutoSplit.Blocker(gguf, config);
        }
        catch (Exception ex)
        {
            return $"the model could not be inspected ({ex.Message})";
        }
    }

    private static bool ExpertOffloadPossible(string path, DotLLM.Core.Moe.MoeOffloadOptions? moeOffload)
    {
        try
        {
            using var gguf = GgufFile.Open(path);
            return DeviceModelLoader.ExpertOffloadPossible(GgufModelConfigExtractor.Extract(gguf.Metadata), moeOffload);
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// The KV storage <paramref name="options"/> asks for. K and V are parsed independently (#1888): a TurboQuant side carries its own
    /// bit width and QJL flag, and <paramref name="formatMismatch"/> reports two TurboQuant sides that disagree (a mixed format, #1887 step 6).
    /// </summary>
    internal static KvCacheConfig BuildKvConfig(ServerOptions options, out string? formatMismatch)
        => TurboQuantKv.Parse(options.CacheTypeK, options.CacheTypeV, cacheWindow: 0, out formatMismatch);

    /// <summary>
    /// The authoritative TurboQuant refusal once <paramref name="model"/> has loaded (#1888): every pre-load rule, plus the backend an
    /// <c>auto</c> device landed on, whether the host is the plain CPU transformer, and whether an MTP head will attach.
    /// </summary>
    internal static string? PostLoadTurboQuantRefusal(ServerOptions options, ModelConfig config, IModel model)
    {
        var kv = BuildKvConfig(options, out string? mismatch);
        return TurboQuantKv.PostLoadRefusal(kv, mismatch, KvCommand.Serve, options.UsePaged, config, model,
            hasDraftModel: !string.IsNullOrEmpty(options.SpeculativeModel),
            mtpWillAttach: options.MtpEnabled && model.SupportsMtp);
    }

    /// <summary>
    /// The TurboQuant refusal decidable before the model loads (#1888): mixed formats, a GPU device, the paged cache, the geometry, a draft
    /// model. Null when nothing is wrong yet; the post-load pass then checks the backend an <c>auto</c> device landed on, the host and MTP.
    /// </summary>
    internal static string? PreLoadTurboQuantRefusal(ServerOptions options, ModelConfig config)
    {
        var kv = BuildKvConfig(options, out string? mismatch);
        var backend = DeviceSpec.Parse(options.Device).Kind switch
        {
            DeviceKind.Cpu => KvBackendKind.Cpu,
            DeviceKind.Cuda => KvBackendKind.Cuda,
            DeviceKind.Vulkan => KvBackendKind.Vulkan,
            _ => KvBackendKind.Unknown,
        };
        return TurboQuantKv.PreLoadRefusal(kv, mismatch, KvCommand.Serve, backend, options.UsePaged, config,
            hasDraftModel: !string.IsNullOrEmpty(options.SpeculativeModel));
    }

    /// <summary>
    /// Opens only the metadata of the <c>--speculative-model</c> draft (the file is mapped, no tensor is read) and rejects architectures the dense draft
    /// loader cannot take. A draft that is missing or unreadable is left to the real load below, which reports it as before.
    /// </summary>
    private static void ValidateSpeculativeDraftEarly(ServerOptions options, ModelConfig target)
    {
        if (string.IsNullOrEmpty(options.SpeculativeModel)) return;
        string? draftPath = ResolveModelPath(options.SpeculativeModel, null);
        if (draftPath is null) return;
        ModelConfig draftConfig;
        try
        {
            using var draftGguf = GgufFile.Open(draftPath);
            draftConfig = GgufModelConfigExtractor.Extract(draftGguf.Metadata);
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or NotSupportedException or ArgumentException)
        {
            return;
        }
        SpeculativeDraftValidator.Validate(draftConfig, draftPath, target);
    }

    /// <summary>Most device KV caches the Vulkan prefix cache retains (each is a whole allocation sized prompt + max tokens + headroom).</summary>
    internal const int VulkanPrefixCacheMaxEntries = 2;

    /// <summary>Extra rows a retained Vulkan KV cache is allocated with, so the next turn of the conversation still fits it.</summary>
    internal const int VulkanPrefixCacheHeadroomTokens = 1024;

    /// <summary>
    /// Attaches <see cref="ServerOptions.TernaryAdapterPaths"/> to the loaded model through <see cref="TernaryAdapterAttach"/> (#1619, #1711): CPU
    /// models with a residual hook get a <c>TernaryResidualHook</c>, Vulkan models with one a device-resident modifier. A model without a hook, or
    /// an adapter that does not fit, fails the load (the model and GGUF are disposed): never a silent skip. Returns the handle that owns the
    /// attachment (null when no adapters were requested); <see cref="ServerState"/> disposes it with the model.
    /// </summary>
    internal static IDisposable? AttachTernaryAdapters(IModel model, ModelConfig config, GgufFile gguf, ServerOptions options)
    {
        if (options.TernaryAdapterPaths is not { Count: > 0 } paths) return null;
        try
        {
            var handle = TernaryAdapterAttach.Attach(model, config, paths);
            Console.WriteLine($"[dotllm] Ternary adapters attached ({handle.Where}): {handle.CircuitCount} circuit(s) from {paths.Count} file(s).");
            return handle;
        }
        catch (Exception ex) when (ex is InvalidDataException or InvalidOperationException or NotSupportedException or IOException or ArgumentException)
        {
            model.Dispose();
            gguf.Dispose();
            throw new InvalidServerOptionException(ex.Message.StartsWith("--ternary-adapter", StringComparison.Ordinal) ? ex.Message : $"--ternary-adapter: {ex.Message}");
        }
    }

    /// <summary>The load-time modifier set of <see cref="ServerState.WeightModifiers"/>.</summary>
    internal static IReadOnlyList<(string Kind, string Id, string ContentDigest)> WeightModifiersOf(string? overlayDigest, IDisposable? ternaryAdapters)
    {
        var list = new List<(string, string, string)>();
        if (overlayDigest is not null)
            list.Add((DotLLM.Core.Identity.ModifierSetDigest.KindOverlay, "", overlayDigest));
        if (ternaryAdapters is TernaryAdapterAttach.Handle handle)
            foreach (string d in handle.ContentDigests)
                list.Add((DotLLM.Core.Identity.ModifierSetDigest.KindTernaryAdapter, "", d));
        return list;
    }

    private static ServerState LoadModelCore(string resolvedPath, ServerOptions options)
    {
        Console.WriteLine($"[dotllm] Loading model from {resolvedPath}...");
        var gguf = GgufFile.Open(resolvedPath);
        string? overlayDigest = null;
        if (options.TensorOverlayPath is { } overlayPath)
        {
            try
            {
                var ov = TensorOverlay.Attach(gguf, resolvedPath, overlayPath, verifyBaseHash: !options.OverlaySkipBaseHash);
                overlayDigest = ov.ContentDigest();   // from the mapped bytes the model will run on, before anything else touches them (#1915)
                foreach (string note in ov.Notes) Console.WriteLine($"[dotllm] {note}");
                Console.WriteLine($"[dotllm] Tensor overlay '{overlayPath}': {ov.Count} tensor(s)" + (options.OverlaySkipBaseHash ? " (base hash NOT verified)." : "."));
            }
            catch (Exception ex) when (ex is InvalidDataException or IOException or NotSupportedException)
            {
                gguf.Dispose();
                throw new InvalidServerOptionException($"--tensor-overlay: {ex.Message}");
            }
        }
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        config = GgufModelConfigExtractor.ApplyRoPEOverride(config, options.RopeOverride);
        bool isEncoder = config.Architecture is Architecture.Bert or Architecture.NomicBert or Architecture.NomicBertMoe;
        var tokenizer = isEncoder ? GgufTokenizerFactory.Load(gguf.Metadata) : GgufBpeTokenizerFactory.Load(gguf.Metadata);

        if (isEncoder)
            return LoadEncoderModel(resolvedPath, options, gguf, config, tokenizer);

        var threading = new ThreadingConfig(options.Threads, options.DecodeThreads);

        // A draft that cannot load must fail NOW, not after the target (up to tens of GiB) is resident, and not as a "device" failure (#1319).
        ValidateSpeculativeDraftEarly(options, config);

        // #1888: a TurboQuant KV request that cannot run for real fails now, before the weights load.
        if (PreLoadTurboQuantRefusal(options, config) is { } tqRefusal)
        {
            gguf.Dispose();
            throw new InvalidServerOptionException(tqRefusal);
        }

        // #1892: a TurboQuant request plans its real KV bytes (codes + norms) in the automatic GPU split, not the F16 figure.
        KvCacheConfig? tqPlanKv = BuildKvConfig(options, out _) is { IsTurboQuantAny: true } planKv ? planKv : null;

        if (options.GpuLayers is { } requestedLayers && requestedLayers > config.NumLayers)
            Console.WriteLine($"[dotllm] Requested {requestedLayers} GPU layers but {Path.GetFileName(resolvedPath)} has {config.NumLayers}; using all {config.NumLayers}.");

        PrepareQwen4ExpVulkan(config, options);

        // One shared dispatch for CPU / all-GPU / partial CUDA / Vulkan (#259, #729, #790) - the same call run and chat make.
        // Architectures that cannot split layers load all-GPU or FAIL with an actionable error (never a silent CPU fallback).
        var (model, deviceKvFactory) = DeviceModelLoader.LoadExact(
            gguf, config, options.Device, options.GpuLayers, threading,
            msg => Console.WriteLine($"[dotllm] {msg}"),
            w => Console.WriteLine($"[dotllm] WARNING: {w}"),
            options.MoeOffload, options.PrefillChunkSize ?? 0, tqPlanKv);
        // Only the Vulkan models' KV factory is wired here (CUDA/hybrid models keep their own per-request caches).
        Func<int, IKvCache>? vulkanKvFactory = DeviceSpec.Parse(options.Device).Kind == DeviceKind.Vulkan ? deviceKvFactory : null;

        var ternaryAdapters = AttachTernaryAdapters(model, config, gguf, options);

        if (gguf.Overlay is { } appliedOverlay)
        {
            try { appliedOverlay.VerifyAllApplied(); }
            catch (NotSupportedException ex)
            {
                model.Dispose();
                gguf.Dispose();
                throw new InvalidServerOptionException(ex.Message);
            }
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

        // KV-cache configuration. K and V are parsed independently; a TurboQuant side carries its bit width and QJL flag (#1888).
        var kvConfig = BuildKvConfig(options, out _);   // a mismatch was refused before load
        // #1888: the post-load pass - the backend an auto device landed on, the host model, an attaching MTP head.
        if (PostLoadTurboQuantRefusal(options, config, model) is { } tqPostRefusal)
        {
            model.Dispose();
            gguf.Dispose();
            throw new InvalidServerOptionException(tqPostRefusal);
        }

        Func<ModelConfig, int, IKvCache>? kvFactory = null;
        PagedKvCacheFactory? pagedFactory = null;
        DotLLM.Cuda.CudaPagedKvCacheFactory? cudaPagedFactory = null;
        PrefixTrieManager? prefixTrieManager = null;
        if (kvConfig.IsTurboQuantAny)
        {
            // #1888: first, so no later branch can take a TurboQuant request (IsQuantized excludes it, so every branch below used to). The
            // post-load pass has refused every context that cannot run it. #1892: the MSE stage is read in place (codes + norms, no
            // decoded history) on CPU, CUDA and Vulkan; the shared helper builds the backend's cache, never another one.
            Console.WriteLine($"[dotllm] Using TurboQuant KV-cache ({kvConfig.TurboQuantBits}-bit, data-oblivious" +
                (kvConfig.TurboQuantUseQjl ? ", QJL unbiased scores, dequant-scratch CPU path)." : ", read in place in code space)."));
            kvFactory = TurboQuantKv.CreateFactory(model, kvConfig);
        }
        else if (vulkanKvFactory is not null)
        {
            // Vulkan models own their device-resident KV storage. Requests run one at a time through the
            // per-request TextGenerator unless the opt-in Vulkan scheduler below engages (hybrid models,
            // DOTLLM_VK_SCHEDULER); there is no paged KV pool and no cross-request prefix trie yet.
            if (options.UsePaged)
                Console.WriteLine("[dotllm] Paged KV-cache not supported on Vulkan; using the model's own KV-cache.");
            // #1884: q8_0 K and V on an audited dense host are read in place from the model's own cache (no F32 dequant scratch);
            // other quantized KV keeps VulkanQuantizedKvCache.
            if (kvConfig.IsQuantized && model is DotLLM.Vulkan.VulkanTransformerModel vkModel
                && !vkModel.TryUseInPlaceQuantizedKv(kvConfig.KeyDType, kvConfig.ValueDType, DotLLM.Vulkan.VulkanModelLoader.ResolveSpvDir()))
            {
                string spvDir = DotLLM.Vulkan.VulkanModelLoader.ResolveSpvDir();
                kvFactory = (cfg, size) => vkModel.CreateQuantizedKvCache(
                    spvDir, size, kvConfig.KeyDType, kvConfig.ValueDType);
            }
            else
            {
                if (model is DotLLM.Vulkan.VulkanQwen3HybridDenseTransformerModel vkDense)
                {
                    // #1102: the dense hybrid (Bonsai 2) stores its KV as f16 (the default) and reads it in place. "f32" is the
                    // option default, so it means "not given" here; DOTLLM_VK_KV=f32 selects the F32 A/B baseline. q8_0 / q4_0 /
                    // mixed K and V fail startup.
                    static string? Given(string? v) => string.Equals(v?.Trim(), "f32", StringComparison.OrdinalIgnoreCase) ? null : v;
                    vkDense.ConfigureKvStorage(DotLLM.Vulkan.VulkanKvStorageExtensions.ParseOptions(
                        Given(options.CacheTypeK), Given(options.CacheTypeV)));
                    Console.WriteLine(
                        $"[dotllm] KV storage: {vkDense.KvStorage.ToString().ToLowerInvariant()} ({vkDense.KvBytesPerPosition / 1024.0:F0} KiB/token), " +
                        $"~{vkDense.KvBytesPerPosition * (long)config.MaxSequenceLength / 1048576.0:F0} MiB at {config.MaxSequenceLength} tokens.");
                }

                var vkFactory = vulkanKvFactory;
                kvFactory = (cfg, size) => vkFactory(size);
            }
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
                // NOT wired into ContinuousBatchScheduler (pagedFactory stays null below): although
                // CudaTransformerModel implements IModel.ForwardBatch (#251), the GPU pool is not yet
                // part of the scheduler's admission, so CUDA requests still serve one at a time through
                // the per-request TextGenerator path. Only the per-request KV-cache backing changes
                // (block-based instead of one flat buffer).
                // #1833: sized to free VRAM, not a fixed 65536 tokens (10 GiB for TwIL-8B on a 12 GB card, spilled by WDDM).
                // #1848: --ctx-size bounds the pool request too; the served context is then capped to the pool after warm-up.
                cudaPagedFactory = cudaModel.CreateFittedPagedKvCacheFactory(
                    Math.Min(options.ContextSize is > 0 and int ctx ? Math.Min(ctx, config.MaxSequenceLength) : config.MaxSequenceLength,
                        DotLLM.Cuda.CudaPagedKvCacheFactory.DefaultMaxTotalTokens), out string poolSummary);
                var factory = cudaPagedFactory;
                kvFactory = (cfg, size) => cudaModel.CreatePagedKvCache(factory.Pool, size);
                Console.WriteLine("[dotllm] Using GPU paged KV-cache (block-based allocation, single-request dispatch)");
                Console.WriteLine($"[dotllm] {poolSummary}");
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
            if (kvConfig.IsQuantized)
                Console.WriteLine($"[dotllm] --cache-type-k/-v (K:{kvConfig.KeyDType} V:{kvConfig.ValueDType}) does not apply to the GPU/CPU split: its KV is " +
                                  $"F16 on the GPU layers and {(DotLLM.Cuda.HybridTransformerModel.TailUsesQ8_0(config.HeadDim) ? "q8_0" : "f16")} on the CPU tail.");
            kvFactory = (cfg, size) => hybridModel.CreateKvCache(size);
        }
        else if (model is DotLLM.Cuda.Architectures.HybridQwen3HybridDenseTransformerModel qwen3SplitModel)
        {
            // #729: partial-offload Qwen3HybridDense owns a split (GPU head / CPU tail) cache.
            if (options.UsePaged)
                Console.WriteLine("[dotllm] Paged KV-cache not supported with hybrid GPU, using the model's split cache.");
            if (kvConfig.IsQuantized)
                Console.WriteLine($"[dotllm] WARNING: --cache-type-k/-v (K:{kvConfig.KeyDType} V:{kvConfig.ValueDType}) is ignored: the split GPU/CPU model owns its KV cache and has no quantized storage (model {config.Architecture}).");
            kvFactory = (cfg, size) => qwen3SplitModel.CreateKvCache(size);
        }
        else if (model is DotLLM.Cuda.Architectures.CudaLfm2MoeTransformerModel lfmModel)
        {
            DotLLM.Cuda.Architectures.CudaLfm2MoeTransformerModel.ValidateKvStorage(kvConfig.KeyDType, kvConfig.ValueDType);
            if (options.UsePaged)
                Console.WriteLine("[dotllm] Paged KV-cache not supported for CUDA LFM2; using the model's own sparse cache.");
            kvFactory = (cfg, size) => lfmModel.CreateKvCache(size);
        }
        else if (model is DotLLM.Cuda.Architectures.CudaNemotronHTransformerModel nemotronHModel)
        {
            // #729: the all-GPU fallback for a partial request on Nemotron-H; sparse attention-only KV.
            if (options.UsePaged)
                Console.WriteLine("[dotllm] Paged KV-cache not supported for Nemotron-H on GPU; using the model's own cache.");
            if (kvConfig.IsQuantized)
                Console.WriteLine($"[dotllm] WARNING: --cache-type-k/-v (K:{kvConfig.KeyDType} V:{kvConfig.ValueDType}) is ignored: Nemotron-H owns its KV cache and has no quantized storage (model {config.Architecture}).");
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
            // #967: the model owns quantized K/V storage; --cache-type-k/-v / --kv-quant-k/-v override DOTLLM_HYBRID_KV.
            // "f32" (the option default) means "not given"; an unsupported pair (TurboQuant, row width % 32) throws.
            qwen3HybridDenseModel.ConfigureKvStorage(
                HybridKvStorage.FromServerOption(options.CacheTypeK),
                HybridKvStorage.FromServerOption(options.CacheTypeV));
            Console.WriteLine(
                $"[dotllm] KV storage: {qwen3HybridDenseModel.KvStorageLabel} (K:{qwen3HybridDenseModel.KvKeyStorage} V:{qwen3HybridDenseModel.KvValueStorage}), " +
                $"~{qwen3HybridDenseModel.ComputeKvBytes(config.MaxSequenceLength) / 1048576.0:F0} MiB at {config.MaxSequenceLength} tokens.");
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
        else if (options.UsePaged && !kvConfig.IsQuantized && IsHybridOrRecurrentCpuModel(config, model))
        {
            // CPU hybrid/recurrent models: the paged pool is sized for config.NumLayers x MaxSequenceLength
            // (e.g. Bonsai-2-27B: 64 layers x 262144 ctx, though only 16 layers hold KV), which dies at startup
            // with "Insufficient memory to continue the execution of the program" under default flags. They also
            // carry recurrent state the paged scheduler does not manage, so use the simple cache.
            Console.WriteLine("[dotllm] Paged KV-cache not supported for hybrid/recurrent CPU models; using the simple KV-cache.");
        }
        else if (options.UsePaged && !kvConfig.IsQuantized && !DotLLM.Core.Attention.KvGeometry.FromConfig(config).IsUniform)
        {
            // Per-layer KV row widths: Gemma-4 (dual head dim 256/512) and MiMo-V2 (8 sliding / 4 global KV heads). Per-layer
            // KV strides exist only on the contiguous F32 SimpleKvCache; the CPU forward's GuardKvCacheHeadDim rejects
            // PagedKvCache with a 500 on the first request (#917, #943). Fall back to the simple cache instead of failing
            // every request.
            Console.WriteLine("[dotllm] Paged KV-cache does not support models with per-layer KV widths (Gemma-4, MiMo-V2); using the simple KV-cache.");
        }
        else if (options.UsePaged && (!kvConfig.IsQuantized || IsPagedQ8Eligible(kvConfig, config, model)))
        {
            // Blocks are F16 by default; q8_0 K and V select Q8_0 blocks (#1268, #1142 C4). Attention reads them in place.
            var blockType = ResolvePagedBlockType(
                options.CacheTypeK is not null || options.CacheTypeV is not null, kvConfig, config, model, out string? kvInfo);
            if (kvInfo is not null)
                Console.WriteLine($"[dotllm] INFO: {kvInfo}");
            pagedFactory = new PagedKvCacheFactory(
                config.NumLayers, config.NumKvHeads, config.HeadDim, elementType: blockType);
            kvFactory = (cfg, size) => pagedFactory.Create(size);
            // Cross-request prefix cache (Step 37). Enabled in tandem with paged.
            prefixTrieManager = new PrefixTrieManager(pagedFactory, new PrefixCacheConfig { EvictionPolicy = options.PromptCachePolicy });
            Console.WriteLine($"[dotllm] Using paged KV-cache ({blockType} blocks) with cross-request prefix cache");
        }
        else if (options.UsePaged && kvConfig.IsQuantized)
        {
            Console.WriteLine("[dotllm] Paged KV-cache supports q8_0 K+V only (uniform, non-hybrid models, no mixed-precision window); using quantized simple cache.");
            kvFactory = (cfg, size) => new QuantizedKvCache(
                DotLLM.Core.Attention.KvGeometry.FromConfig(cfg, model?.SupportsCompactSlidingKv == true), size,
                kvConfig.KeyDType, kvConfig.ValueDType, kvConfig.MixedPrecisionWindowSize);
        }
        else if (kvConfig.IsQuantized)
        {
            kvFactory = (cfg, size) => new QuantizedKvCache(
                DotLLM.Core.Attention.KvGeometry.FromConfig(cfg, model?.SupportsCompactSlidingKv == true), size,
                kvConfig.KeyDType, kvConfig.ValueDType, kvConfig.MixedPrecisionWindowSize);
        }

        // Vulkan (single-request path): the cache a request filled stays resident and a later request adopts its longest common prefix
        // (#1591). Each entry holds a whole device KV allocation, so keep few; the headroom lets the next turn of a conversation fit it.
        // CUDA dense (--no-paged) caches are retained the same way since #1594, so they get the same cap: under WDDM a card that runs
        // out of VRAM spills into shared memory instead of failing (#1068). No headroom: a CUDA cache grows by a device-to-device copy
        // of its rows (TryGrowTo, admitted against free VRAM), which costs milliseconds, so it is sized for the request alone.
        // CUDA paged (#1843) the same: each retained entry holds pool blocks and a gather scratch sized to its window, and the
        // paged pool fit charges a few live gather scratches (CudaPagedKvCacheFactory.LiveCachesCharged), not one per entry.
        bool deviceOwnedPrefixKv = vulkanKvFactory is not null
            || model is DotLLM.Cuda.CudaTransformerModel;

        // #1862: the prefill chunk this server uses. An explicit --prefill-chunk-size (0 included) wins; otherwise the dense CUDA
        // non-paged path (f16, and q8_0 / q4_0 since #1850) chunks by default, so its context fit charges a fixed forward scratch instead of a per-token one.
        // #1851: a GPU/CPU split always prefills in chunks (1024 unless a positive --prefill-chunk-size), the scratch it was planned for.
        bool splitModel = model is DotLLM.Cuda.HybridTransformerModel;
        int prefillChunk = splitModel
            ? DotLLM.Cuda.CudaAutoSplit.SplitPrefillChunk(options.PrefillChunkSize ?? 0)
            : ResolvePrefillChunk(options.PrefillChunkSize, TakesCudaDensePrefillChunkDefault(model, cudaPagedFactory is not null, kvConfig),
                kvConfig.IsQuantized);
        PrefixCache? prefixCache = !options.PromptCacheEnabled ? null
            : !deviceOwnedPrefixKv ? new PrefixCache(options.PromptCacheSize, options.PromptCachePolicy)
            : new PrefixCache(Math.Min(options.PromptCacheSize, VulkanPrefixCacheMaxEntries), options.PromptCachePolicy);
        int prefixCacheHeadroom = vulkanKvFactory is not null && prefixCache is not null ? VulkanPrefixCacheHeadroomTokens : 0;
        if (options.PromptCacheEnabled && options.PromptCachePolicy == PrefixCachePolicy.Agent)
            Console.WriteLine("[dotllm] Prompt-cache policy: agent (own-length eviction, protected conversations, save-point and end-of-reply checkpoints; #1654; default since #1778, --prompt-cache-policy lru for LRU)");

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

        // MTP self-speculative decoding (issue #253) - DEFAULT ON for serve when the checkpoint carries an embedded head
        // (--no-mtp opts out). Engaging it takes the continuous-batch scheduler offline for this model, so an explicit
        // concurrency hint (>= 5) wins and drops MTP (the scheduler measured +73% aggregate decode at 8 streams on Vulkan);
        // an external draft model also wins (the two speculation modes are mutually exclusive).
        string? vkSchedEnvForMtp = Environment.GetEnvironmentVariable("DOTLLM_VK_SCHEDULER");
        bool schedulerAvailable = (vulkanKvFactory is not null && !string.Equals(vkSchedEnvForMtp, "0", StringComparison.Ordinal))
            || pagedFactory is not null;
        bool concurrencyOverMtp = options.MtpEnabled && options.ExpectedConcurrency >= 5 && schedulerAvailable;
        // qwen4exp ships its draft head as a separate mtp-*.gguf (#820): attach it when MTP can engage (same rules as DecideMtp) so the default-on
        // serve path actually speculates; a head that does not fit the device is declined with a message, never fatal.
        if (options.MtpEnabled && !concurrencyOverMtp && draftModel is null && (!model.SupportsMtp || options.DFlashPath is not null))
            TryAttachSeparateMtpHead(model, resolvedPath, options);
        int mtpK = options.SpeculativeCandidates == TextGenerator.DefaultSpeculativeCandidates && model.MtpPreferredDraftTokens > 0
            ? model.MtpPreferredDraftTokens : options.SpeculativeCandidates;
        var mtpDecision = DecideMtp(options, model.SupportsMtp, draftModel is not null, concurrencyOverMtp, schedulerAvailable, mtpK);
        bool mtpActive = mtpDecision.Active;
        if (mtpDecision.Message is { } mtpMsg)
            Console.WriteLine(mtpMsg);

        var generator = new TextGenerator(model, tokenizer, kvFactory, prefixCache,
            draftModel: draftModel, speculativeCandidates: options.SpeculativeCandidates,
            mtpEnabled: options.MtpEnabled, mtpAdaptive: true, recurrentPrefixCache: options.PromptCacheEnabled,
            promptLookup: options.PromptLookup,
            prefixTrieManager: prefixTrieManager,
            prefixCacheHeadroomTokens: prefixCacheHeadroom,
            prefillChunkSize: prefillChunk,
            recurrentPrefixCacheOptions: options.GetPrefixSnapshotOptions());
        if (prefillChunk > 0)
            Console.WriteLine($"[dotllm] Prefill chunk size: {prefillChunk} tokens per forward pass" +
                              (splitModel && prefillChunk != options.PrefillChunkSize ? " (GPU/CPU split default; --prefill-chunk-size overrides)"
                               : options.PrefillChunkSize is null ? " (dense CUDA default; --prefill-chunk-size 0 for one pass)" : ""));
        if (options.PromptLookup is { } lookup)
            Console.WriteLine($"[dotllm] Prompt lookup: up to {lookup.DraftTokens} drafts, n-gram {lookup.MinNgram}..{lookup.MaxNgram} (greedy requests; continuous batching off)");
        if (generator.RecurrentStepCheckpointBytes > 0)
            Console.WriteLine($"[dotllm] Prompt-cache steps: a {generator.RecurrentStepCheckpointBytes / 1048576.0:F1} MiB recurrent checkpoint every {options.PromptCacheStep} prefill tokens, keeping {options.PromptCacheStepKeep ?? 8}");
        if (options.PrefixSnapshotDiskDirectory is not null)
            Console.WriteLine($"[dotllm] Prefix snapshot disk transport: {generator.RecurrentPrefixStatistics.DiskTransport ?? "unavailable (provider or directory)"} (restart persistence disabled)");

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

        // Issue #1068: size the CUDA hybrid's served context to free VRAM (after warm-up, so its buffers are counted).
        if (model is DotLLM.Cuda.Architectures.CudaQwen3HybridDenseTransformerModel hybridDense)
            config = ApplyHybridContextFit(hybridDense, options, mtpActive, mtpK);
        // Issues #1841 / #1836: dense CUDA honours --ctx-size and serves no more than fits: the paged pool's capacity, or the
        // non-paged KV + forward scratch that fit in free VRAM after warm-up.
        else if (model is DotLLM.Cuda.CudaTransformerModel cudaDense)
            config = ApplyCudaDenseContextFit(cudaDense, options, kvConfig, cudaPagedFactory, prefillChunk);
        else if (model is DotLLM.Vulkan.VulkanQwen4ExpTransformerModel q4Vulkan)
        {
            // #1324: the window is the capacity the caches were sized for at load (--ctx-size, reduced to fit), already in Config.
            config = q4Vulkan.Config;
            long perToken = DotLLM.Vulkan.VulkanQwen4ExpTransformerModel.KvBytesPerToken(config);
            Console.WriteLine($"[dotllm] Context window: {config.MaxSequenceLength} tokens" +
                (options.ContextSize is > 0 ? " (--ctx-size" + (options.ContextSize > config.MaxSequenceLength ? $" {options.ContextSize} reduced to fit the device" : "") + ")" : " (default)") +
                $"; K/V + indexer {perToken / 1024.0:F1} KiB/token ({DotLLM.Vulkan.VulkanQwen4ExpTransformerModel.KvStorage.ToString().ToLowerInvariant()} K/V), " +
                $"{perToken * (long)config.MaxSequenceLength / 1048576.0:F0} MiB if the window fills (allocated lazily). Raise with --ctx-size.");
        }
        else if (model is DotLLM.Cuda.HybridTransformerModel split)
        {
            // #1851: an automatic split was sized for --ctx-size (default 4096) of KV, and its Config carries that window; serve
            // and report it, so a longer request is refused at admission instead of failing on a position past the window.
            config = split.Config;
            Console.WriteLine($"[dotllm] Context window: {config.MaxSequenceLength} tokens ({split.NumGpuLayers} GPU + " +
                              $"{config.NumLayers - split.NumGpuLayers} CPU layers).");
        }
        else if (model is IContextLimitable limitable)
            config = ApplyRequestedContextSize(model, limitable, options);

        // Continuous-batch scheduler. Enabled when a (CPU) paged factory is available, or on the opt-in
        // Vulkan hybrid path below, and speculative decoding is off: the scheduler doesn't support draft
        // models (or MTP self-speculative decoding, same restriction). CUDA keeps the single-request path
        // until its GPU paged pool is wired into scheduler admission (see above). Streaming requests run
        // on the scheduler too (#1266).
        ContinuousBatchSchedulerService? scheduler = null;
        // Vulkan hybrid models (per-sequence GDN state + serial ForwardBatch) can run through the scheduler, but the
        // serial TextGenerator path is faster until recurrent prefix restore + fused batching land, so this is opt-in:
        // DOTLLM_VK_SCHEDULER=1. No paged pool and no prefix trie (KV-only, paged-only) on this path.
        string? vkSchedEnv = Environment.GetEnvironmentVariable("DOTLLM_VK_SCHEDULER");
        bool vulkanScheduler = vulkanKvFactory is not null && model.SupportsThreadedSequenceState
            && (string.Equals(vkSchedEnv, "1", StringComparison.Ordinal)
                || (options.ExpectedConcurrency >= 5 && !string.Equals(vkSchedEnv, "0", StringComparison.Ordinal)));
        // Clover block diffusion (#1828) is a TextGenerator path: the scheduler would decode those requests causally.
        bool cloverBlock = model.Config.Clover is not null
            && TextGenerator.ParseCloverDecodeOption(Environment.GetEnvironmentVariable("DOTLLM_CLOVER_DECODE") ?? "causal") == CloverDecodeMode.Block;
        if ((pagedFactory is not null || vulkanScheduler) && kvFactory is not null && draftModel is null && !mtpActive && options.PromptLookup is null
            && !cloverBlock)
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
            if (options.PromptCachePolicy != PrefixCachePolicy.Lru)
                schedulerOptions = (schedulerOptions ?? new ContinuousBatchSchedulerOptions()) with { RecurrentPrefixPolicy = options.PromptCachePolicy };

            scheduler = new ContinuousBatchSchedulerService(
                model,
                tokenizer,
                kvFactory,
                options: schedulerOptions,
                pagedPool: pagedFactory?.Pool,
                prefixCache: prefixTrieManager);
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
            TernaryAdapters = ternaryAdapters,
            WeightModifiers = WeightModifiersOf(overlayDigest, ternaryAdapters),
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
            MtpActive = mtpActive,
            MtpStatus = mtpDecision.Status,
            LoraRegistry = CreateLoraRegistry(),
            Residency = CreateResidencyManager(options),
            EstimatedBytes = estimatedBytes,
            LastUsedUtc = DateTimeOffset.UtcNow,
        };
    }

    /// <summary>
    /// Attaches a separate MTP head GGUF (qwen4exp) to <paramref name="model"/>: an explicit <c>--mtp-head</c> always, an auto-detected sibling only when
    /// the trunk is big enough for self-speculation to beat plain decode (measured per quantisation, see <see cref="Qwen4ExpMtpHeadResolver"/>).
    /// Failure to attach (no room on the device, bad tensor table) is reported and serving continues without speculation.
    /// </summary>
    internal static void TryAttachSeparateMtpHead(IModel model, string resolvedPath, ServerOptions options)
    {
        if (options.DFlashPath is { } dflashPath)
        {
            // An explicit drafter is the user's choice: failing to attach it fails startup rather than silently decoding plain.
            if (model is not IDFlashDraftAttachable dflash)
                throw new NotSupportedException($"--dflash: {model.GetType().Name} cannot drive a DFlash drafter (qwen35 targets on CPU, CUDA or Vulkan only).");
            dflash.AttachDFlashDrafter(dflashPath);
            Console.WriteLine($"[dotllm] DFlash: attached drafter {Path.GetFileName(dflashPath)}.");
            return;
        }
        if (model is not IMtpHeadAttachable) return;
        try
        {
            string? headPath = options.MtpHeadPath;
            if (headPath is null)
            {
                headPath = Qwen4ExpMtpHeadResolver.Find(resolvedPath);
                if (headPath is null) return;
                long trunk = ModelResolver.FileLength(resolvedPath);
                if (!Qwen4ExpMtpHeadResolver.ShouldAutoAttach(trunk))
                {
                    Console.WriteLine($"[dotllm] MTP: head {Path.GetFileName(headPath)} found but not attached - a {trunk / (double)(1L << 30):F0} GiB trunk measures slower with " +
                                      "speculation than without (--mtp-head <path> forces it).");
                    return;
                }
            }
            Qwen4ExpMtpHeadResolver.TryAttach(model, resolvedPath, headPath);
            Console.WriteLine($"[dotllm] MTP: attached head {Path.GetFileName(headPath)}.");
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            Console.WriteLine($"[dotllm] MTP: head not attached ({ex.GetType().Name}: {ex.Message}); decoding without speculation.");
        }
    }

    /// <summary>Outcome of the serve-time MTP policy: whether it engages, the startup log line (null = stay quiet), and a short /props status.</summary>
    internal readonly record struct MtpDecision(bool Active, string? Message, string Status);

    /// <summary>
    /// Pure MTP policy (#757). Active only when enabled, the model has an embedded head, no external draft model is given, and the
    /// expected concurrency does not route to the continuous-batch scheduler. Always explains why when MTP is not engaged for a model
    /// that has a head (or when the user explicitly asked with <c>--mtp</c>).
    /// </summary>
    internal static MtpDecision DecideMtp(ServerOptions options, bool supportsMtp, bool hasDraftModel, bool concurrencyOverMtp, bool schedulerAvailable = true, int? draftDepth = null)
    {
        if (!options.MtpEnabled)
            return new(false, supportsMtp ? "[dotllm] MTP: skipped - disabled by --no-mtp (this model carries an MTP head)." : null,
                "off (--no-mtp)");
        if (!supportsMtp)
            return new(false,
                options.MtpExplicit ? "[dotllm] MTP: --mtp was set but this checkpoint has no MTP head (nextn.* tensors) - ignoring." : null,
                "unavailable (no MTP head)");
        if (hasDraftModel)
            return new(false, "[dotllm] MTP: skipped - an external draft model (--speculative-model) was given; it takes precedence over the built-in MTP head.",
                "skipped (external draft model)");
        if (concurrencyOverMtp)
            return new(false,
                $"[dotllm] MTP: skipped - --expected-concurrency {options.ExpectedConcurrency} >= 5: serving through the continuous-batch scheduler, which cannot run MTP self-speculation.",
                "skipped (expected concurrency >= 5)");
        string how = options.MtpExplicit ? "enabled (--mtp)" : "enabled automatically (model carries an MTP head)";
        string tradeoff = schedulerAvailable
            ? " Tradeoff: continuous batching is off for this model; for many concurrent clients pass --no-mtp or --expected-concurrency 5."
            : "";
        return new(true, $"[dotllm] MTP: {how}, K={draftDepth ?? options.SpeculativeCandidates}, adaptive gate on.{tradeoff}", "active");
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
            return ModelResolver.FileLength(path);   // follows symlinks; sums every shard of a split GGUF (#756)
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

        if (options.PrefillChunkSize is > 0 and int chunk && (schedulerOptions?.MaxPrefillTokensPerStep ?? 0) <= 0)
        {
            schedulerOptions = (schedulerOptions ?? new ContinuousBatchSchedulerOptions()) with
            {
                MaxPrefillTokensPerStep = chunk,
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

        // (#1021) "[dotllm] req#N POST /v1/chat/completions" ... "done 200 in 3.2s" / "cancelled after 600.0s".
        app.UseRequestLifecycleLog();

        // (#1822) X-DotLLM-Device (+ X-DotLLM-Device-Fallback) on /v1/* and /api/* responses: a CPU fallback is visible over HTTP.
        DeviceHeaders.Use(app, state);

        // (#1023) A request whose admitted model was swapped out while it queued, and could not be re-activated: 503 / in-stream error.
        ModelUnavailableException.UseModelUnavailableHandler(app);

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

        // (#1021) The startup model's id is one the server hands out too.
        state.RememberIssuedId(state.Options.ModelId, state.LoadedModelPath);

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
}
