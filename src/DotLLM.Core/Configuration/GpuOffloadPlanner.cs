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
            or Architecture.Qwen3MoeHybrid or Architecture.Mamba3 => false,
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
