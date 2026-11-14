namespace DotLLM.Vulkan;

/// <summary>
/// Default prefill chunk (tokens per forward) for Vulkan hosts that run a prompt as one pass unless the engine splits it
/// (issue #1723). Chunking gives the engine a boundary to report real progress and to observe cancellation; without it a
/// 9.2K-token prompt on an Arc iGPU was one 337 s forward that printed <c>0/9200</c> throughout and could not be cancelled.
/// </summary>
/// <remarks>
/// Resolution order: <c>--prefill-chunk-size</c> (engine; overrides this entirely), then <c>DOTLLM_VULKAN_PREFILL_CHUNK</c>
/// (0 = single pass), then the per-device default of <see cref="ForDeviceType"/>. This is NOT the submit-sizing governor
/// (#1760): it bounds rows per forward for progress/cancel, not GPU time per submit.
/// </remarks>
internal static class VulkanPrefillChunk
{
    /// <summary>Integrated-GPU default. Measured on Arc (#1723): 512 costs nothing and the text is identical.</summary>
    internal const int IntegratedDefault = 512;

    /// <summary>
    /// Discrete-GPU default. Equal to the integrated one for now: the RTX 3060 device-loss limit (#1760) is below this
    /// and is the governor's job; 512 is no worse than the unchunked pass it replaces.
    /// </summary>
    internal const int DiscreteDefault = 512;

    /// <summary>Environment override: tokens per chunk, 0 = single pass.</summary>
    internal const string EnvVar = "DOTLLM_VULKAN_PREFILL_CHUNK";

    /// <summary>Default chunk for a <c>VkPhysicalDeviceType</c> value (1 integrated, 2 discrete; anything else gets the conservative integrated value).</summary>
    internal static int ForDeviceType(int deviceType) => deviceType == 2 ? DiscreteDefault : IntegratedDefault;

    /// <summary>Parses <paramref name="envValue"/> (a non-negative int) else falls back to the per-device default.</summary>
    internal static int Resolve(int deviceType, string? envValue)
        => int.TryParse(envValue, out int v) && v >= 0 ? v : ForDeviceType(deviceType);

    /// <summary>Resolves for a live device, reading <see cref="EnvVar"/>.</summary>
    internal static int Resolve(VulkanDevice device)
        => Resolve(device.PhysicalDeviceTypeValue, Environment.GetEnvironmentVariable(EnvVar));
}
