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
    /// <summary><c>vulkan</c>.</summary>
    Vulkan,
}

/// <summary>
/// The one parser for <c>--device</c> (#790). Before it, <c>run</c>/<c>chat</c>/<c>serve</c> each tested <c>StartsWith("gpu")</c>, so
/// <c>cuda</c>, <c>auto</c> and any typo silently meant the CPU.
/// </summary>
/// <param name="Kind">Backend family.</param>
/// <param name="Ordinal">CUDA device ordinal; 0 for everything else.</param>
public readonly record struct DeviceSpec(DeviceKind Kind, int Ordinal)
{
    /// <summary>Human-readable list of accepted values, used in every validation error.</summary>
    public const string Accepted = "auto, cpu, vulkan, gpu, gpu:N, cuda, cuda:N";

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
                if (tail is null || tail == "0") { spec = new(DeviceKind.Vulkan, 0); return true; }
                error = $"Unknown --device '{value}': only the first Vulkan device can be selected ('vulkan'). Expected one of: {Accepted}.";
                return false;
        }
        error = $"Unknown --device '{value}'. Expected one of: {Accepted}.";
        return false;
    }

    /// <summary>Parses or throws <see cref="ArgumentException"/> with the validation message.</summary>
    public static DeviceSpec Parse(string? value) =>
        TryParse(value, out var s, out string? err) ? s : throw new ArgumentException(err, nameof(value));

    /// <summary>The device string loaders understand: <c>auto</c>, <c>cpu</c>, <c>gpu:N</c>, <c>vulkan</c>.</summary>
    public string Canonical => Kind switch
    {
        DeviceKind.Auto => "auto",
        DeviceKind.Cpu => "cpu",
        DeviceKind.Cuda => $"gpu:{Ordinal}",
        _ => "vulkan",
    };

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
public sealed record DevicePlan(DeviceSpec Spec, IReadOnlyList<string> Candidates, string? Error, string? CpuWarning);

/// <summary>Result of <see cref="DeviceModelLoader"/>.</summary>
/// <param name="Model">The loaded model.</param>
/// <param name="KvCacheFactory">KV factory the model expects (GPU models); null for the CPU model.</param>
/// <param name="ResolvedDevice">Canonical device actually used.</param>
/// <param name="Warning">Prominent warning (auto fell back to CPU), else null.</param>
public sealed record DeviceLoadResult(IModel Model, Func<int, IKvCache>? KvCacheFactory, string ResolvedDevice, string? Warning)
{
    /// <summary>True when <see cref="ResolvedDevice"/> is the Vulkan backend.</summary>
    public bool IsVulkan => ResolvedDevice == "vulkan";
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
    /// <returns>The model and its KV-cache factory (null for the CPU model).</returns>
    public static (IModel Model, Func<int, IKvCache>? KvCacheFactory) LoadExact(
        GgufFile gguf, ModelConfig config, string canonicalDevice, int? requestedGpuLayers,
        ThreadingConfig threading, Action<string> log, Action<string> warn, MoeOffloadOptions? moeOffload = null)
    {
        var spec = DeviceSpec.Parse(canonicalDevice);
        if (spec.Kind == DeviceKind.Auto)
            throw new ArgumentException("LoadExact needs a concrete device, not 'auto'.", nameof(canonicalDevice));

        // Bidirectional embedding towers (EmbeddingGemma) have no GPU implementation: the CUDA/Vulkan decoders are causal and
        // would return wrong vectors without any error. Fail loudly (auto falls through to the CPU with a warning).
        // BERT encoders (bge / e5 / gte / MiniLM / mxbai, F16 weights) have a dedicated Vulkan implementation (#741).
        if (spec.Kind == DeviceKind.Vulkan && config.Architecture == Architecture.Bert)
        {
            log($"Vulkan inference ({DotLLM.Vulkan.VulkanModelLoader.SharedDevice.DeviceName}), {config.Architecture} encoder");
            return (DotLLM.Vulkan.VulkanBertEncoderModel.LoadFromGguf(
                DotLLM.Vulkan.VulkanModelLoader.SharedDevice, gguf, config, DotLLM.Vulkan.VulkanModelLoader.ResolveSpvDir()), null);
        }
        if ((config.BidirectionalAttention || config.Architecture is Architecture.Bert or Architecture.NomicBert or Architecture.NomicBertMoe) && spec.Kind != DeviceKind.Cpu)
            throw new NotSupportedException(
                $"{config.Architecture} is a bidirectional embedding model and runs on the CPU only (requested '{canonicalDevice}'; the Vulkan BERT encoder covers architecture 'bert' only, #741); use --device cpu.");

        if (spec.Kind == DeviceKind.Vulkan)
        {
            if (requestedGpuLayers is > 0)
                warn("--gpu-layers is ignored on vulkan: the whole model is device-resident (there is no partial offload).");
            log($"Vulkan inference ({DotLLM.Vulkan.VulkanModelLoader.SharedDevice.DeviceName})");
            var (m, kv) = DotLLM.Vulkan.VulkanModelLoader.CreateSharedFromGguf(gguf, config);
            return (m, kv);
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
    /// <returns>The load result.</returns>
    public static DeviceLoadResult Load(
        GgufFile gguf, ModelConfig config, string requestedDevice, int? requestedGpuLayers,
        ThreadingConfig threading, long modelBytes, string modelName, Action<string> log, Action<string> warn,
        MoeOffloadOptions? moeOffload = null)
        => LoadWith(gguf, config, requestedDevice, requestedGpuLayers, threading, modelBytes, modelName, log, warn,
            Endpoints.DeviceEndpoint.Describe(),
            (device, layers) => LoadExact(gguf, config, device, layers, threading, log, warn, moeOffload), moeOffload);

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
        MoeOffloadOptions? moeOffload = null)
    {
        var spec = DeviceSpec.Parse(requestedDevice);

        // --gpu-layers 0 is the user choosing the CPU, under auto as under cuda.
        if (spec.Kind == DeviceKind.Auto && requestedGpuLayers == 0)
            spec = new(DeviceKind.Cpu, 0);

        // An offloadable MoE (#1070) is exempt from the weights-vs-free-VRAM refusal, like a partial --gpu-layers offload.
        var plan = DeviceSelector.Plan(spec, modelBytes, devices,
            fullOffload: requestedGpuLayers is null or < 0 && !ExpertOffloadPossible(config, moeOffload));
        if (plan.Error is not null)
            throw new DeviceUnavailableException(DeviceSelector.ExplicitFailureMessage(requestedDevice, modelName, modelBytes, plan.Error));

        var failures = new List<string>();
        Exception? last = null;
        foreach (string device in plan.Candidates)
        {
            try
            {
                var (model, kv) = loadExact(device, requestedGpuLayers);
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
