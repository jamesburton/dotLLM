using System.Collections.Concurrent;

namespace DotLLM.Vulkan;

/// <summary>
/// Per-tensor NVFP4 second-level scales (<c>X.scale</c>, #1579) of the weights the generic Vulkan transformer model uploads.
/// The scale multiplies the matmul OUTPUT (llama.cpp <c>build_lora_mm</c>: <c>mul_mat(w, x) * w_s</c>), so it cannot be folded into
/// the packed blocks. The CPU loader hands the upload a host-pointer-keyed table (
/// <c>TransformerWeights.WeightScales</c>); the upload registers each NVFP4 device buffer's scale here by VkBuffer handle, and
/// <c>VulkanTransformerModel.RecordMatmul</c> reads it back when it dispatches the NVFP4 kernel. Every NVFP4 upload registers (1 when the
/// GGUF has no scale) so a recycled VkBuffer handle can never inherit a stale scale.
/// </summary>
internal static class VulkanNvfp4Scales
{
    [ThreadStatic] private static IReadOnlyDictionary<nint, float>? s_hostSource;
    private static readonly ConcurrentDictionary<nint, float> s_device = new();

    /// <summary>Sets (or clears) the host-pointer scale table the calling thread's uploads consult.</summary>
    internal static void SetHostSource(IReadOnlyDictionary<nint, float>? table) => s_hostSource = table;

    /// <summary>Registers the scale of an NVFP4 device buffer uploaded from host pointer <paramref name="srcPtr"/>.</summary>
    internal static void Register(VulkanDevice.Buffer buffer, nint srcPtr)
    {
        float scale = 1f;
        if (s_hostSource is { } host && host.TryGetValue(srcPtr, out float s)) scale = s;
        s_device[buffer.Handle] = scale;
    }

    /// <summary>Registers an explicit scale (tests).</summary>
    internal static void Register(VulkanDevice.Buffer buffer, float scale) => s_device[buffer.Handle] = scale;

    /// <summary>The scale registered for <paramref name="buffer"/>, or 1 when none.</summary>
    internal static float Get(VulkanDevice.Buffer buffer) => s_device.TryGetValue(buffer.Handle, out float s) ? s : 1f;
}
