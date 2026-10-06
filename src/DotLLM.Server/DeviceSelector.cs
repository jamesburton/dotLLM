using DotLLM.Server.Endpoints;

namespace DotLLM.Server;

/// <summary>
/// <c>--device auto</c> (issue #722): the ordered devices to try for a model - CUDA when a GPU is present and the model fits in its memory, then
/// Vulkan when it is servable, then the CPU. A load that fails on one falls through to the next (see <see cref="ServerStartup.LoadModel"/>).
/// </summary>
public static class DeviceSelector
{
    /// <summary>Fraction of a CUDA device's memory a model file may take and still be tried there (the rest is KV cache and scratch).</summary>
    public const double CudaFitFraction = 0.85;

    /// <summary>True when <paramref name="device"/> is the <c>auto</c> pseudo-device.</summary>
    public static bool IsAuto(string? device) => string.Equals(device?.Trim(), "auto", StringComparison.OrdinalIgnoreCase);

    /// <summary>Candidate device strings, best first, always ending in <c>cpu</c>.</summary>
    public static IReadOnlyList<string> Candidates(long modelBytes) => Candidates(modelBytes, DeviceEndpoint.Describe());

    /// <summary>Same, from an explicit device description (testable).</summary>
    public static IReadOnlyList<string> Candidates(long modelBytes, Models.DeviceListResponse devices)
    {
        var order = new List<string>();
        foreach (var backend in devices.Backends ?? [])
        {
            if (!backend.Servable) continue;
            if (backend.Name == "cuda")
            {
                var best = (backend.Devices ?? []).Where(d => d.DeviceString is not null && (d.TotalMemoryBytes is not { } t || modelBytes <= t * CudaFitFraction))
                    .OrderByDescending(d => d.TotalMemoryBytes ?? 0).FirstOrDefault();
                if (best is not null) order.Insert(0, best.DeviceString!);   // CUDA first
            }
            else if (backend.Name == "vulkan" && backend.Devices is { Length: > 0 } vk && vk[0].DeviceString is { } v)
                order.Add(v);
        }
        order.Add("cpu");
        return order;
    }

    /// <summary>The prominent warning for <c>--device auto</c> ending up on the CPU after a GPU load failed (#733).</summary>
    public static string FallbackWarning(string modelName, IReadOnlyList<string> gpuFailures) =>
        $"Model '{modelName}' is running on the CPU because every GPU device failed to load it under --device auto "
        + $"({string.Join("; ", gpuFailures)}). Expect much lower throughput (typically several times slower). "
        + "Pass an explicit --device (cpu to silence this, or gpu/vulkan to get the real error).";
}
