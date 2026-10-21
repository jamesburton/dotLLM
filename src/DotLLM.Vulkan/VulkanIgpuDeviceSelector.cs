using System.Globalization;

namespace DotLLM.Vulkan;

/// <summary>
/// Picks the Vulkan device of the MoE iGPU cold leg (#1083, design #1063 section 8c′). Pure, so the rule is testable without a GPU.
/// <list type="bullet">
///   <item>No override: the first <b>integrated</b> GPU in enumeration order; none means there is no iGPU leg.</item>
///   <item><c>--moe-igpu-device N</c>: device N; <c>--moe-igpu-device name</c>: the first device whose name contains it
///   (case-insensitive). An override may pick a non-integrated device (with a warning), e.g. a second discrete card.</item>
///   <item>An NVIDIA device is never chosen: on the boxes this targets it is the CUDA GPU the host already uses, and computing its
///   cold experts there through Vulkan would only add a second copy of the banks and a host round trip.</item>
/// </list>
/// </summary>
internal static class VulkanIgpuDeviceSelector
{
    /// <summary>
    /// The device to use, or null with the reason in <paramref name="reason"/> (also set on success: what was picked and why).
    /// </summary>
    /// <param name="devices">The enumerated devices (<see cref="VulkanDevice.EnumeratePhysicalDevices"/>).</param>
    /// <param name="spec">The override (an index or a name substring), or null / empty for the first integrated GPU.</param>
    /// <param name="reason">Receives a one-line explanation for the log.</param>
    /// <returns>The chosen device, or null.</returns>
    public static VulkanPhysicalDeviceInfo? Select(IReadOnlyList<VulkanPhysicalDeviceInfo> devices, string? spec, out string reason)
    {
        ArgumentNullException.ThrowIfNull(devices);
        if (devices.Count == 0)
        {
            reason = "no Vulkan device is present";
            return null;
        }

        if (string.IsNullOrWhiteSpace(spec))
        {
            foreach (var d in devices)
            {
                if (!d.IsIntegrated || d.IsNvidia) continue;
                reason = $"integrated GPU #{d.Index} '{d.Name}'";
                return d;
            }
            reason = "no integrated Vulkan GPU (devices: " + string.Join(", ", devices.Select(Describe)) + ")";
            return null;
        }

        string s = spec.Trim();
        VulkanPhysicalDeviceInfo? match = null;
        if (int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out int index))
        {
            foreach (var d in devices)
                if (d.Index == index) { match = d; break; }
        }
        else
        {
            foreach (var d in devices)
                if (d.Name.Contains(s, StringComparison.OrdinalIgnoreCase)) { match = d; break; }
        }

        if (match is not { } m)
        {
            reason = $"--moe-igpu-device '{s}' matches no Vulkan device (devices: " + string.Join(", ", devices.Select(Describe)) + ")";
            return null;
        }
        if (m.IsNvidia)
        {
            reason = $"--moe-igpu-device '{s}' is the NVIDIA device '{m.Name}', which the iGPU leg never uses (it is the CUDA GPU)";
            return null;
        }
        reason = m.IsIntegrated
            ? $"integrated GPU #{m.Index} '{m.Name}' (--moe-igpu-device {s})"
            : $"#{m.Index} '{m.Name}' (--moe-igpu-device {s}; NOT an integrated GPU: its banks are read across its bus, not zero-copy)";
        return m;
    }

    private static string Describe(VulkanPhysicalDeviceInfo d)
        => $"#{d.Index} '{d.Name}' {(d.IsIntegrated ? "integrated" : d.DeviceType == 2 ? "discrete" : "type " + d.DeviceType)}";
}
