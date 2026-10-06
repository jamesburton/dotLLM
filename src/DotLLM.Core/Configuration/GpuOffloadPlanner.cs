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
    /// A partial split was requested but the architecture cannot do one. Try all-on-GPU; if
    /// that fails (no device, out of VRAM) run on the CPU. A warning is attached.
    /// </summary>
    FullGpuElseCpu,
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

        return new(GpuOffloadMode.FullGpuElseCpu, numLayers,
            $"Partial GPU offload ({n}/{numLayers} layers) is not supported for architecture {architecture}; "
            + "loading it entirely on the GPU instead, or on the CPU if it does not fit.");
    }
}
