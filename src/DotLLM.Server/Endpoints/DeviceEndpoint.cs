using DotLLM.Server.Models;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// <c>GET /v1/devices</c> (#454) — which backends and devices this server can see, and the exact
/// <c>device</c> string <c>POST /v1/models/load</c> accepts for each. Read-only, so ungated.
/// </summary>
/// <remarks>
/// <para>
/// <b>Probing is done once and cached for the process lifetime.</b>
/// <c>VulkanDevice.IsAvailable</c> takes the Vulkan lifecycle write lock and creates/destroys a
/// <c>VkInstance</c>; <c>CudaDevice.IsAvailable</c> runs <c>cuInit</c>. Neither belongs on a
/// per-request path while a model is live on that device.
/// </para>
/// <para>
/// <b>No device-local heap usage is reported.</b> The Vulkan allocator's per-heap live-bytes
/// ledger does not exist on this branch (it lives on <c>feature/bonsai-2-amd</c>), and the server
/// has no Vulkan load path anyway. The field is omitted rather than reported as a fabricated zero.
/// </para>
/// <para>
/// <b>Vulkan is reported as available-but-not-servable.</b> <c>ServerStartup.LoadModel</c>
/// dispatches to the CPU loader or the CUDA loader only — there is no Vulkan branch — so a tray
/// must not offer a Vulkan device as a load target.
/// </para>
/// </remarks>
public static class DeviceEndpoint
{
    private static DeviceListResponse? s_cached;
    private static readonly object s_lock = new();

    public static void Map(WebApplication app) =>
        app.MapGet("/v1/devices", () => Results.Ok(Describe()));

    /// <summary>Cached backend/device description. Separated from the route so tests can call it.</summary>
    public static DeviceListResponse Describe()
    {
        lock (s_lock)
        {
            return s_cached ??= Probe();
        }
    }

    private static DeviceListResponse Probe() => new()
    {
        Backends = [DescribeCpu(), DescribeCuda(), DescribeVulkan()],
    };

    private static BackendInfoDto DescribeCpu() => new()
    {
        Name = "cpu",
        Available = true,
        DeviceCount = 1,
        Servable = true,
        Devices =
        [
            new DeviceInfoDto
            {
                Index = 0,
                Name = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture
                    + " CPU (" + Environment.ProcessorCount + " logical cores)",
                DeviceString = "cpu",
            }
        ],
    };

    private static BackendInfoDto DescribeCuda()
    {
        if (!SafeProbe(DotLLM.Cuda.CudaDevice.IsAvailable))
            return new BackendInfoDto
            {
                Name = "cuda",
                Available = false,
                DeviceCount = 0,
                Servable = false,
                Note = "No CUDA driver / no CUDA-capable GPU detected.",
            };

        var devices = new List<DeviceInfoDto>();
        try
        {
            int count = DotLLM.Cuda.CudaDevice.GetDeviceCount();
            for (int i = 0; i < count; i++)
            {
                try
                {
                    var d = DotLLM.Cuda.CudaDevice.GetDevice(i);
                    devices.Add(new DeviceInfoDto
                    {
                        Index = i,
                        Name = d.Name,
                        DeviceString = $"gpu:{i}",
                        TotalMemoryBytes = d.TotalMemoryBytes,
                        ComputeCapability = d.ComputeCapability,
                    });
                }
                catch
                {
                    devices.Add(new DeviceInfoDto { Index = i, Name = $"CUDA device {i}", DeviceString = $"gpu:{i}" });
                }
            }
        }
        catch
        {
            // Driver present but enumeration failed - report the backend as unavailable rather
            // than half-populated.
            return new BackendInfoDto
            {
                Name = "cuda",
                Available = false,
                DeviceCount = 0,
                Servable = false,
                Note = "CUDA driver present but device enumeration failed.",
            };
        }

        return new BackendInfoDto
        {
            Name = "cuda",
            Available = devices.Count > 0,
            DeviceCount = devices.Count,
            Servable = devices.Count > 0,
            Devices = devices.ToArray(),
        };
    }

    private static BackendInfoDto DescribeVulkan()
    {
        int count = 0;
        try { count = DotLLM.Vulkan.VulkanDevice.PhysicalDeviceCount(); }
        catch { count = 0; }

        return new BackendInfoDto
        {
            Name = "vulkan",
            Available = count > 0,
            DeviceCount = count,
            Servable = false,
            Note = count > 0
                ? "Vulkan devices are present, but the server's model-load path dispatches to the "
                  + "CPU or CUDA loader only — there is no Vulkan device string for POST /v1/models/load."
                : "No Vulkan loader / no Vulkan-capable device detected.",
            Devices = [],
        };
    }

    private static bool SafeProbe(Func<bool> probe)
    {
        try { return probe(); }
        catch { return false; }
    }
}
