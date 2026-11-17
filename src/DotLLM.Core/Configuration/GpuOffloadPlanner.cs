using DotLLM.Core.Models;

namespace DotLLM.Core.Configuration;

/// <summary>How a requested <c>--gpu-layers</c> count is to be honoured for a given architecture.</summary>
public enum GpuOffloadMode
{
    /// <summary>Run every layer on the CPU.</summary>
    Cpu,

    /// <summary>Run every layer on the GPU.</summary>
    FullGpu,

    /// <summary>Split: the first N layers on the GPU, the rest on the CPU.</summary>
    Partial,

    /// <summary>
    /// A partial split was requested but the architecture cannot do one. Load ALL layers on the GPU if
    /// they fit; otherwise FAIL with an actionable error. Never fall back to the CPU silently.
    /// </summary>
    FullGpuOrFail,
}

/// <summary>The outcome of <see cref="GpuOffloadPlanner.Plan"/>.</summary>
/// <param name="Mode">How to load.</param>
/// <param name="GpuLayers">GPU layer count for <see cref="GpuOffloadMode.Partial"/>; otherwise 0 or NumLayers.</param>
/// <param name="Warning">Text to show the user, or <see langword="null"/>.</param>
public readonly record struct GpuOffloadPlan(GpuOffloadMode Mode, int GpuLayers, string? Warning);

/// <summary>
/// The single authority on which architectures support a partial CPU/GPU layer split (#729).
/// </summary>
/// <remarks>
/// The generic <c>HybridTransformerModel</c> assumes every layer shares one dense Llama-style tensor
/// set. Architectures whose layers are SSM / Gated-DeltaNet (Nemotron-H, Qwen3MoeHybrid) or that have
/// no GGUF mapping (Mamba-3) cannot use it. <see cref="Architecture.Qwen3HybridDense"/> is the one
/// exception: it has its own architecture-aware split loader (#291).
/// </remarks>
public static class GpuOffloadPlanner
{
    /// <summary>True when a 0 &lt; gpuLayers &lt; NumLayers split is implemented for <paramref name="architecture"/>.</summary>
    public static bool SupportsPartialOffload(Architecture architecture) => architecture switch
    {
        Architecture.NemotronH or Architecture.NemotronHMoe
            or Architecture.Qwen3MoeHybrid or Architecture.Qwen4Exp or Architecture.Mamba3 or Architecture.Lfm2Moe or Architecture.Lfm3
            or Architecture.Glm4Moe or Architecture.Ricursive => false,
        _ => true,
    };

    /// <summary>Decides how to honour <paramref name="requestedGpuLayers"/> for the model.</summary>
    /// <param name="architecture">Model architecture.</param>
    /// <param name="requestedGpuLayers">Requested GPU layer count (clamped to [0, numLayers]).</param>
    /// <param name="numLayers">Total layer count.</param>
    public static GpuOffloadPlan Plan(Architecture architecture, int requestedGpuLayers, int numLayers)
    {
        int n = Math.Clamp(requestedGpuLayers, 0, numLayers);
        if (n <= 0) return new(GpuOffloadMode.Cpu, 0, null);
        if (n >= numLayers) return new(GpuOffloadMode.FullGpu, numLayers, null);
        if (SupportsPartialOffload(architecture)) return new(GpuOffloadMode.Partial, n, null);

        return new(GpuOffloadMode.FullGpuOrFail, numLayers,
            $"Partial GPU offload ({n}/{numLayers} layers) is not supported for architecture {architecture} "
            + $"(tracked in {PriorityIssueUrl}); loading all {numLayers} layers on the GPU, and failing if they do not fit.");
    }

    /// <summary>
    /// Sentinel <c>--gpu-layers</c> value for <c>auto</c> (issue #1851): the loader chooses N with <see cref="LayerSplitPlanner"/>.
    /// Distinct from <c>-1</c> ("all layers"), so code that treats any negative value as "all" must test for this first.
    /// </summary>
    public const int AutoGpuLayers = -2;

    /// <summary>True when <paramref name="requestedGpuLayers"/> is <see cref="AutoGpuLayers"/>.</summary>
    /// <param name="requestedGpuLayers">The requested value.</param>
    /// <returns>Whether the loader is to choose N.</returns>
    public static bool IsAuto(int? requestedGpuLayers) => requestedGpuLayers == AutoGpuLayers;

    /// <summary>
    /// Parses a <c>--gpu-layers</c> value: <c>auto</c> (<see cref="AutoGpuLayers"/>), <c>all</c> or <c>-1</c> (every layer), or a
    /// non-negative count. Empty means unset.
    /// </summary>
    /// <param name="value">The raw value.</param>
    /// <param name="layers">The parsed value, or <see langword="null"/> when unset.</param>
    /// <param name="error">The validation message when parsing fails.</param>
    /// <returns>Whether the value is valid.</returns>
    public static bool TryParseGpuLayers(string? value, out int? layers, out string? error)
    {
        layers = null;
        error = null;
        string v = (value ?? "").Trim();
        if (v.Length == 0) return true;
        if (v.Equals("auto", StringComparison.OrdinalIgnoreCase)) { layers = AutoGpuLayers; return true; }
        if (v.Equals("all", StringComparison.OrdinalIgnoreCase)) { layers = -1; return true; }
        if (int.TryParse(v, System.Globalization.NumberStyles.AllowLeadingSign, System.Globalization.CultureInfo.InvariantCulture, out int n) && n >= -1)
        {
            layers = n;
            return true;
        }
        error = $"Invalid --gpu-layers '{value}'. Expected 'auto', 'all', -1 or a non-negative layer count.";
        return false;
    }

    /// <summary>
    /// Why <c>--gpu-layers auto</c> cannot split this model across the dGPU and the CPU (issue #1851), or <see langword="null"/>
    /// when it can. The split host (<c>HybridTransformerModel</c>) runs dense Llama-style layers; its CPU tail has no per-layer
    /// sliding window or dense-YaRN RoPE yet (#432), no MoE or MLA, and no attention sinks. A model it cannot run correctly is
    /// never split automatically: auto then loads it whole on the GPU or moves on to the next device.
    /// </summary>
    /// <param name="config">Model configuration.</param>
    /// <param name="hasAttentionSinks">Whether the GGUF carries <c>attn_sinks</c> tensors.</param>
    /// <returns>The reason, or null.</returns>
    public static string? AutoSplitBlocker(ModelConfig config, bool hasAttentionSinks = false)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (!SupportsPartialOffload(config.Architecture))
            return $"{config.Architecture} cannot be split between GPU and CPU ({PriorityIssueUrl})";
        if (config.Architecture == Architecture.Qwen3HybridDense)
            return "the Qwen3.5/3.6 hybrid has its own split host, which the automatic planner does not size yet (use --gpu-layers N)";
        if (config.NumLayers < 2)
            return "the model has fewer than two layers";
        if (config.Moe is not null)
            return "MoE layers are not supported by the GPU/CPU layer split (expert offload covers MoE: --moe-offload)";
        if (config.MlaConfig is not null)
            return "MLA attention is not supported by the GPU/CPU layer split";
        if (config.SlidingWindowPattern > 0 || config.PerLayerSlidingWindow is { Count: > 0 })
            return "per-layer sliding-window attention is not implemented on the CPU tail (#432)";
        if (config.RoPEConfig is { IsDenseYarnActive: true })
            return "YaRN RoPE scaling is not implemented on the CPU tail (#432)";
        if (hasAttentionSinks)
            return "attention sinks are not implemented on the CPU tail";
        return null;
    }

    /// <summary>Tracking issue for true split offload of the hybrid SSM architectures.</summary>
    public const string PriorityIssueUrl = "https://github.com/jamesburton/dotLLM/issues/735";

    /// <summary>
    /// The error for a partial request that cannot be met: names what was asked, why it cannot be
    /// honoured, how much VRAM is needed versus available, and the explicit opt-in to run on the CPU.
    /// </summary>
    /// <param name="architecture">Model architecture.</param>
    /// <param name="requestedGpuLayers">Layers the user asked to offload.</param>
    /// <param name="numLayers">Total layers.</param>
    /// <param name="modelBytes">Approximate weight bytes that must be resident (GGUF size).</param>
    /// <param name="gpuTotalBytes">Total VRAM of the requested device, or null if unknown.</param>
    /// <param name="gpuFreeBytes">Free VRAM, or null if unknown.</param>
    /// <param name="cause">Message of the underlying load failure.</param>
    public static string BuildUnsatisfiableMessage(Architecture architecture, int requestedGpuLayers, int numLayers,
        long modelBytes, long? gpuTotalBytes, long? gpuFreeBytes, string cause)
    {
        static string Gib(long b) => $"{b / (1024.0 * 1024 * 1024):F1} GiB";
        string vram = gpuFreeBytes is { } f
            ? $"{Gib(modelBytes)} needed, {Gib(f)} free" + (gpuTotalBytes is { } t ? $" of {Gib(t)}" : "")
            : gpuTotalBytes is { } t2 ? $"{Gib(modelBytes)} needed, {Gib(t2)} total (free unknown)" : $"{Gib(modelBytes)} needed, device memory unknown";
        return $"Cannot honour the GPU request for this {architecture} model: you asked for {requestedGpuLayers}/{numLayers} layers on the GPU, "
            + $"but {architecture} cannot be split between GPU and CPU, so all {numLayers} layers must fit on the GPU and loading them failed "
            + $"({vram}; cause: {cause}). Nothing was run on the CPU. To run it on the CPU anyway, pass `--device cpu` (it will be much slower). "
            + $"Split offload for this architecture is tracked in {PriorityIssueUrl}.";
    }
}
