using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Moe;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Server.Models;

namespace DotLLM.Server;

/// <summary>The backend families a <c>--device</c> value can name.</summary>
public enum DeviceKind
{
    /// <summary><c>auto</c>: best servable device for the model, CPU last (with a warning).</summary>
    Auto,
    /// <summary><c>cpu</c>: only ever chosen by the user.</summary>
    Cpu,
    /// <summary><c>gpu[:N]</c> / <c>cuda[:N]</c>.</summary>
    Cuda,
    /// <summary><c>vulkan</c> (the default GPU) or <c>vulkan:N</c> (physical device N, #1704).</summary>
    Vulkan,
}

/// <summary>
/// The one parser for <c>--device</c> (#790). Before it, <c>run</c>/<c>chat</c>/<c>serve</c> each tested <c>StartsWith("gpu")</c>, so
/// <c>cuda</c>, <c>auto</c> and any typo silently meant the CPU.
/// </summary>
/// <param name="Kind">Backend family.</param>
/// <param name="Ordinal">CUDA device ordinal, or the Vulkan physical-device index when <paramref name="ExplicitIndex"/>; 0 otherwise.</param>
/// <param name="ExplicitIndex">Vulkan only: the value named an index (<c>vulkan:N</c>). Plain <c>vulkan</c> is the process default device
/// (scoring plus <c>DOTLLM_VULKAN_DEVICE_INDEX</c>), which is not necessarily index 0 on a multi-GPU box.</param>
public readonly record struct DeviceSpec(DeviceKind Kind, int Ordinal, bool ExplicitIndex = false)
{
    /// <summary>Human-readable list of accepted values, used in every validation error.</summary>
    public const string Accepted = "auto, cpu, vulkan, vulkan:N, gpu, gpu:N, cuda, cuda:N";

    /// <summary>Parses a <c>--device</c> value (case-insensitive; <c>gpu</c> is an alias of <c>cuda</c>).</summary>
    public static bool TryParse(string? value, out DeviceSpec spec, out string? error)
    {
        spec = default;
        error = null;
        string v = (value ?? "").Trim().ToLowerInvariant();
        if (v.Length == 0) { spec = new(DeviceKind.Auto, 0); return true; }

        int colon = v.IndexOf(':');
        string head = colon < 0 ? v : v[..colon];
        string? tail = colon < 0 ? null : v[(colon + 1)..];

        switch (head)
        {
            case "auto" or "cpu" when tail is null:
                spec = new(head == "auto" ? DeviceKind.Auto : DeviceKind.Cpu, 0);
                return true;
            case "gpu" or "cuda":
                if (tail is null) { spec = new(DeviceKind.Cuda, 0); return true; }
                if (int.TryParse(tail, out int n) && n >= 0) { spec = new(DeviceKind.Cuda, n); return true; }
                break;
            case "vulkan":
                if (tail is null) { spec = new(DeviceKind.Vulkan, 0); return true; }
                if (int.TryParse(tail, System.Globalization.NumberStyles.None, null, out int vi)) { spec = new(DeviceKind.Vulkan, vi, true); return true; }
                break;
        }
        error = $"Unknown --device '{value}'. Expected one of: {Accepted}.";
        return false;
    }

    /// <summary>Parses or throws <see cref="ArgumentException"/> with the validation message.</summary>
    public static DeviceSpec Parse(string? value) =>
        TryParse(value, out var s, out string? err) ? s : throw new ArgumentException(err, nameof(value));

    /// <summary>The device string loaders understand: <c>auto</c>, <c>cpu</c>, <c>gpu:N</c>, <c>vulkan</c>, <c>vulkan:N</c>.</summary>
    public string Canonical => Kind switch
    {
        DeviceKind.Auto => "auto",
        DeviceKind.Cpu => "cpu",
        DeviceKind.Cuda => $"gpu:{Ordinal}",
        _ => ExplicitIndex ? $"vulkan:{Ordinal}" : "vulkan",
    };

    /// <summary>The Vulkan physical-device index to load on, or null for the process default device (plain <c>vulkan</c>).</summary>
    public int? VulkanIndex => Kind == DeviceKind.Vulkan && ExplicitIndex ? Ordinal : null;

    /// <summary>True for CUDA and Vulkan.</summary>
    public bool IsGpu => Kind is DeviceKind.Cuda or DeviceKind.Vulkan;
}

/// <summary>An explicitly requested GPU device that cannot be honoured. Carries the full actionable message; callers print it, not a stack.</summary>
public sealed class DeviceUnavailableException : Exception
{
    /// <inheritdoc cref="DeviceUnavailableException"/>
    public DeviceUnavailableException(string message, Exception? inner = null) : base(message, inner) { }
}

/// <summary>What <see cref="DeviceSelector.Plan"/> decided for a request, before any load is attempted.</summary>
/// <param name="Spec">The parsed request.</param>
/// <param name="Candidates">Canonical device strings to try, best first. Empty when <see cref="Error"/> is set.</param>
/// <param name="Error">Non-null when an explicitly requested device cannot be honoured (never resolved to CPU).</param>
/// <param name="CpuWarning">Non-null when <c>auto</c> will land on the CPU without any GPU having been tried.</param>
public sealed record DevicePlan(DeviceSpec Spec, IReadOnlyList<string> Candidates, string? Error, string? CpuWarning)
{
    /// <summary>
    /// Under <c>auto</c>: the CUDA candidate the whole model does not fit, to be loaded as an automatic GPU/CPU layer split
    /// (<c>--gpu-layers auto</c>, #1851); null when there is none.
    /// </summary>
    public string? SplitCandidate { get; init; }
}

/// <summary>Result of <see cref="DeviceModelLoader"/>.</summary>
/// <param name="Model">The loaded model.</param>
/// <param name="KvCacheFactory">KV factory the model expects (GPU models); null for the CPU model.</param>
/// <param name="ResolvedDevice">Canonical device actually used.</param>
/// <param name="Warning">Prominent warning (auto fell back to CPU), else null.</param>
public sealed record DeviceLoadResult(IModel Model, Func<int, IKvCache>? KvCacheFactory, string ResolvedDevice, string? Warning)
{
    /// <summary>True when <see cref="ResolvedDevice"/> is the Vulkan backend.</summary>
    public bool IsVulkan => ResolvedDevice.StartsWith("vulkan", StringComparison.Ordinal);
}

/// <summary>
/// The single load dispatch behind <c>serve</c>, <c>run</c> and <c>chat</c> (#790): resolves the request with
/// <see cref="DeviceSelector.Plan"/>, tries candidates in order, and refuses to turn an explicit GPU request into a CPU run.
/// </summary>
public static class DeviceModelLoader
{
    /// <summary>
    /// Loads <paramref name="gguf"/> on exactly <paramref name="canonicalDevice"/> (<c>cpu</c>, <c>gpu:N</c> or <c>vulkan</c>); no fallback.
    /// </summary>
    /// <param name="gguf">Opened GGUF file; must outlive the model.</param>
    /// <param name="config">Model configuration.</param>
    /// <param name="canonicalDevice"><c>cpu</c>, <c>gpu:N</c> or <c>vulkan</c>.</param>
    /// <param name="requestedGpuLayers">Requested GPU layers (CUDA), or null.</param>
    /// <param name="threading">CPU threading.</param>
    /// <param name="log">Informational messages.</param>
    /// <param name="warn">Warnings.</param>
    /// <param name="moeOffload">Routed-expert offload (CUDA <c>nemotron_h_moe</c>, #1070); null = defaults plus environment.</param>
    /// <param name="prefillChunkTokens">The <c>--prefill-chunk-size</c> value (0 = the split default) an automatic split plans for (#1851).</param>
    /// <returns>The model and its KV-cache factory (null for the CPU model).</returns>
    public static (IModel Model, Func<int, IKvCache>? KvCacheFactory) LoadExact(
        GgufFile gguf, ModelConfig config, string canonicalDevice, int? requestedGpuLayers,
        ThreadingConfig threading, Action<string> log, Action<string> warn, MoeOffloadOptions? moeOffload = null, int prefillChunkTokens = 0)
    {
        var spec = DeviceSpec.Parse(canonicalDevice);
        if (spec.Kind == DeviceKind.Auto)
            throw new ArgumentException("LoadExact needs a concrete device, not 'auto'.", nameof(canonicalDevice));

        // Bidirectional embedding towers (EmbeddingGemma) have no GPU implementation: the CUDA/Vulkan decoders are causal and
        // would return wrong vectors without any error. Fail loudly (auto falls through to the CPU with a warning).
        // BERT encoders (bge / e5 / gte / MiniLM / mxbai, F16 weights) have a dedicated Vulkan implementation (#741).
        if (spec.Kind == DeviceKind.Vulkan && config.Architecture is Architecture.Bert or Architecture.NomicBert or Architecture.NomicBertMoe)
        {
            var vkDev = VulkanDeviceFor(spec);
            log($"Vulkan inference ({vkDev.DeviceName}), {config.Architecture} encoder");
            return (DotLLM.Vulkan.VulkanBertEncoderModel.LoadFromGguf(
                vkDev, gguf, config, DotLLM.Vulkan.VulkanModelLoader.ResolveSpvDir()), null);
        }
        if (spec.Kind == DeviceKind.Vulkan && config.Architecture is Architecture.GemmaEmbedding)
        {
            var vkDev = VulkanDeviceFor(spec);
            log($"Vulkan inference ({vkDev.DeviceName}), {config.Architecture} encoder");
            return (DotLLM.Vulkan.VulkanGemmaEmbeddingModel.LoadFromGguf(
                vkDev, gguf, config, DotLLM.Vulkan.VulkanModelLoader.ResolveSpvDir()), null);
        }
        if ((config.BidirectionalAttention || config.Architecture is Architecture.Bert or Architecture.NomicBert or Architecture.NomicBertMoe) && spec.Kind != DeviceKind.Cpu)
            throw new NotSupportedException(
                $"{config.Architecture} is a bidirectional embedding model and runs on the CPU only (requested '{canonicalDevice}'; the Vulkan encoders cover 'bert', 'nomic-bert', 'nomic-bert-moe' and 'gemma-embedding' only, #741 / #1471 / #1472); use --device cpu.");

        if (spec.Kind == DeviceKind.Vulkan)
        {
            if (requestedGpuLayers is > 0)
                warn("--gpu-layers is ignored on vulkan: the whole model is device-resident (there is no partial offload).");
            var vkDev = VulkanDeviceFor(spec);
            log($"Vulkan inference ({vkDev.DeviceName})");
            var (m, kv) = DotLLM.Vulkan.VulkanModelLoader.CreateOnDeviceFromGguf(gguf, config, spec.VulkanIndex);
            return (m, kv);
        }

        if (GpuOffloadPlanner.IsAuto(requestedGpuLayers))
        {
            if (spec.Kind == DeviceKind.Cuda)
            {
                // #1851: the planner chooses N and logs the placement and why.
                var autoMoe = moeOffload ?? MoeOffloadOptions.WithEnvironmentFallback(null);
                return DotLLM.Cuda.CudaModelLoader.CreateForGpuLayers(gguf, config, GpuOffloadPlanner.AutoGpuLayers, spec.Ordinal, threading,
                    warn, autoMoe with { Log = autoMoe.Log ?? log }, ProbeIgpu, log, prefillChunkTokens);
            }
            requestedGpuLayers = 0;   // --device cpu --gpu-layers auto: nothing to place on a GPU
        }

        int gpuLayers = ServerStartup.ResolveGpuLayers(requestedGpuLayers, spec.Canonical, config.NumLayers);
        var plan = GpuOffloadPlanner.Plan(config.Architecture, gpuLayers, config.NumLayers);
        log(plan.Mode switch
        {
            GpuOffloadMode.Cpu => $"CPU inference ({threading.EffectiveThreadCount} threads)",
            GpuOffloadMode.FullGpu => $"GPU {spec.Ordinal} inference",
            GpuOffloadMode.Partial => $"Hybrid inference ({gpuLayers} GPU + {config.NumLayers - gpuLayers} CPU layers)",
            _ => $"GPU {spec.Ordinal} inference (partial offload unsupported for {config.Architecture}; all layers must fit)",
        });
        return DotLLM.Cuda.CudaModelLoader.CreateForGpuLayers(gguf, config, gpuLayers, spec.Ordinal, threading, warn,
            (moeOffload ?? MoeOffloadOptions.WithEnvironmentFallback(null)) is var moe ? moe with { Log = moe.Log ?? log } : null);
    }

    /// <summary>
    /// The iGPU an automatic split could place its tail on (#1851): the first integrated Vulkan device, with its device-local heap
    /// (clamped by <c>DOTLLM_VRAM_BUDGET_MB</c>) as its budget. <see cref="IgpuTier.Supported"/> is false until the dGPU + iGPU host
    /// exists (#1863), so the planner reports it and keeps the tail on the CPU. Null when there is none.
    /// </summary>
    /// <returns>The tier, or null.</returns>
    public static IgpuTier? ProbeIgpu()
    {
        try
        {
            foreach (var d in DotLLM.Vulkan.VulkanDevice.EnumeratePhysicalDevices())
            {
                if (d.DeviceType != VkIntegratedGpu || d.DeviceLocalHeapBytes <= 0) continue;
                return new IgpuTier($"vulkan:{d.Index}", d.Name, DeviceMemoryBudget.ClampSimulatedCapacity(d.DeviceLocalHeapBytes),
                    DeviceMemoryBudget.DefaultReserveBytes, Supported: false);
            }
        }
        catch (Exception)
        {
            // No Vulkan here: no iGPU tier.
        }
        return null;
    }

    /// <summary><c>VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU</c>.</summary>
    private const int VkIntegratedGpu = 1;

    /// <summary>
    /// Says why <c>--device auto</c> passed over a CUDA device (#1851): the whole model does not fit it, and the split host cannot
    /// run this model. Silent when a CUDA device is a candidate or there is none.
    /// </summary>
    /// <param name="plan">The auto plan.</param>
    /// <param name="devices">The machine's devices.</param>
    /// <param name="blocker">Why no split, or null.</param>
    /// <param name="log">Receives the line.</param>
    public static void LogSkippedSplit(DevicePlan plan, DeviceListResponse devices, string? blocker, Action<string> log)
    {
        if (blocker is null || plan.Candidates.Any(c => c.StartsWith("gpu:", StringComparison.Ordinal))) return;
        if (devices.Backends?.FirstOrDefault(b => b.Name == "cuda") is not { Servable: true, Devices.Length: > 0 }) return;
        log($"--device auto: the model does not fit the CUDA device whole, and it cannot be split between GPU and CPU ({blocker}); " +
            "trying the next device");
    }

    /// <summary>The Vulkan device a spec names: physical index N for <c>vulkan:N</c>, else the process default (#1704).</summary>
    public static DotLLM.Vulkan.VulkanDevice VulkanDeviceFor(DeviceSpec spec) =>
        spec.VulkanIndex is int i ? DotLLM.Vulkan.VulkanModelLoader.GetDevice(i) : DotLLM.Vulkan.VulkanModelLoader.SharedDevice;

    /// <summary>
    /// Resolves <paramref name="requestedDevice"/> and loads. <c>auto</c> tries CUDA, then Vulkan, then CPU and warns whenever it ends on
    /// the CPU; an explicit device that cannot be honoured throws <see cref="DeviceUnavailableException"/> (never CPU).
    /// </summary>
    /// <exception cref="ArgumentException">The device string is not recognised.</exception>
    /// <param name="gguf">Opened GGUF file; must outlive the model.</param>
    /// <param name="config">Model configuration.</param>
    /// <param name="requestedDevice">The raw <c>--device</c> value.</param>
    /// <param name="requestedGpuLayers">Requested GPU layers (CUDA), or null.</param>
    /// <param name="threading">CPU threading.</param>
    /// <param name="modelBytes">Size of the model's weights.</param>
    /// <param name="modelName">Model name for messages.</param>
    /// <param name="log">Informational messages.</param>
    /// <param name="warn">Warnings.</param>
    /// <param name="moeOffload">Routed-expert offload (CUDA <c>nemotron_h_moe</c>, #1070); null = defaults plus environment.</param>
    /// <param name="prefillChunkTokens">The <c>--prefill-chunk-size</c> value (0 = the split default) an automatic split plans for (#1851).</param>
    /// <returns>The load result.</returns>
    public static DeviceLoadResult Load(
        GgufFile gguf, ModelConfig config, string requestedDevice, int? requestedGpuLayers,
        ThreadingConfig threading, long modelBytes, string modelName, Action<string> log, Action<string> warn,
        MoeOffloadOptions? moeOffload = null, int prefillChunkTokens = 0)
        => LoadWith(gguf, config, requestedDevice, requestedGpuLayers, threading, modelBytes, modelName, log, warn,
            Endpoints.DeviceEndpoint.Describe(),
            (device, layers) => LoadExact(gguf, config, device, layers, threading, log, warn, moeOffload, prefillChunkTokens), moeOffload,
            c => DotLLM.Cuda.CudaAutoSplit.Blocker(gguf, c));

    /// <summary>
    /// True when a CUDA load of this model may keep its routed experts on the host (#1070: <c>nemotron_h_moe</c> with
    /// <c>--moe-offload</c> not <c>off</c>), so weights larger than free VRAM are not by themselves a reason to refuse it.
    /// </summary>
    /// <param name="config">Model configuration (null = unknown, false).</param>
    /// <param name="moeOffload">Offload options, or null for the defaults plus environment.</param>
    /// <returns>True when expert offload can apply.</returns>
    public static bool ExpertOffloadPossible(ModelConfig? config, MoeOffloadOptions? moeOffload)
        => config?.Architecture == Architecture.NemotronHMoe
           && (moeOffload ?? MoeOffloadOptions.WithEnvironmentFallback(null)).Mode != MoeOffloadMode.Off;

    /// <summary>Same, with the device inventory and the per-device loader injected (testable without hardware).</summary>
    internal static DeviceLoadResult LoadWith(
        GgufFile gguf, ModelConfig config, string requestedDevice, int? requestedGpuLayers,
        ThreadingConfig threading, long modelBytes, string modelName, Action<string> log, Action<string> warn,
        DeviceListResponse devices, Func<string, int?, (IModel Model, Func<int, IKvCache>? Kv)> loadExact,
        MoeOffloadOptions? moeOffload = null, Func<ModelConfig, string?>? autoSplitBlocker = null)
    {
        // #1851: null = the split host can run this model. Without a probe (tests) no split is offered.
        string? blocker = autoSplitBlocker is null ? "no split probe" : autoSplitBlocker(config);
        var spec = DeviceSpec.Parse(requestedDevice);

        // --gpu-layers 0 is the user choosing the CPU, under auto as under cuda.
        if (spec.Kind == DeviceKind.Auto && requestedGpuLayers == 0)
            spec = new(DeviceKind.Cpu, 0);

        // An offloadable MoE (#1070) is exempt from the weights-vs-free-VRAM refusal, like a partial --gpu-layers offload.
        // -1 is "all layers"; auto (-2, #1851) is a partial request the loader sizes, so it is not judged on the whole model here.
        var plan = DeviceSelector.Plan(spec, modelBytes, devices,
            fullOffload: requestedGpuLayers is null or -1 && !ExpertOffloadPossible(config, moeOffload),
            autoSplit: blocker is null && requestedGpuLayers is null or GpuOffloadPlanner.AutoGpuLayers);
        if (spec.Kind == DeviceKind.Auto && requestedGpuLayers is null && autoSplitBlocker is not null)
            LogSkippedSplit(plan, devices, blocker, log);
        if (plan.Error is not null)
            throw new DeviceUnavailableException(DeviceSelector.ExplicitFailureMessage(requestedDevice, modelName, modelBytes, plan.Error));

        var failures = new List<string>();
        Exception? last = null;
        foreach (string device in plan.Candidates)
        {
            try
            {
                // #1851: under auto, a CUDA device the whole model does not fit is loaded as an automatic layer split.
                int? layers = requestedGpuLayers is null && device == plan.SplitCandidate ? GpuOffloadPlanner.AutoGpuLayers : requestedGpuLayers;
                var (model, kv) = loadExact(device, layers);
                string? warning = null;
                if (device == "cpu" && spec.Kind == DeviceKind.Auto)
                    warning = failures.Count > 0 ? DeviceSelector.FallbackWarning(modelName, failures) : plan.CpuWarning;
                return new DeviceLoadResult(model, kv, device, warning);
            }
            catch (Exception ex) when (device != "cpu")
            {
                last = ex;
                string why = ex.Message;
                if (spec.Kind != DeviceKind.Auto)
                    throw new DeviceUnavailableException(DeviceSelector.ExplicitFailureMessage(
                        requestedDevice, modelName, modelBytes, $"loading the model on {device} failed: {why}"), ex);
                failures.Add($"{device}: {why}");
                log($"--device auto: {device} could not load this model ({why}); trying the next device");
            }
        }
        throw last ?? new InvalidOperationException("No device could load the model.");
    }
}
