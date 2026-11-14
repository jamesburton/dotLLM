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
/// <b>No device-local heap usage is reported.</b> The field is omitted rather than reported as a
/// fabricated zero.
/// </para>
/// <para>
/// <b>Every Vulkan physical device is listed</b> (#1704) as <c>vulkan:N</c>; the one plain
/// <c>vulkan</c> selects carries <c>is_default</c>.
/// </para>
/// </remarks>
public static class DeviceEndpoint
{
    private static DeviceListResponse? s_cached;
    private static readonly object s_lock = new();

    public static void Map(WebApplication app) =>
        app.MapGet("/v1/devices", (long? model_bytes) =>
        {
            var d = Describe();
            return Results.Ok(d with { RecommendedDevice = DeviceSelector.Candidates(Math.Max(0, model_bytes ?? 0), d)[0] });
        });

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
        Backends = [DescribeCpu(), DescribeCuda(), DescribeVulkan(), DescribeHip()],
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
                        FreeMemoryBytes = DotLLM.Cuda.CudaMemoryInfo.TryGet(i, out long free, out _) ? free : null,
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
        DotLLM.Vulkan.VulkanDevice.PhysicalDeviceInfo[] infos = [];
        try { infos = DotLLM.Vulkan.VulkanDevice.EnumeratePhysicalDevices(); }
        catch { infos = []; }

        // Servable needs the SPIR-V blobs as well as a device: the loader reads them from spv/ beside the binary (#712).
        bool shaders = false;
        if (infos.Length > 0)
        {
            try { DotLLM.Vulkan.VulkanModelLoader.ResolveSpvDir(); shaders = true; }
            catch { shaders = false; }
        }
        return DescribeVulkan(infos, shaders);
    }

    /// <summary>
    /// Builds the Vulkan backend entry from an enumerated device list: one entry per physical device, selectable as <c>vulkan:N</c>
    /// (#1704). Memory is the device-local heap size (total only): Vulkan has no device-wide free figure obtainable without a logical
    /// device, and a fabricated one would mislead the fit check.
    /// </summary>
    internal static BackendInfoDto DescribeVulkan(DotLLM.Vulkan.VulkanDevice.PhysicalDeviceInfo[] infos, bool shaders)
    {
        int count = infos.Length;
        return new BackendInfoDto
        {
            Name = "vulkan",
            Available = count > 0,
            DeviceCount = count,
            Servable = count > 0 && shaders,
            Note = count == 0
                ? "No Vulkan loader / no Vulkan-capable device detected."
                : shaders ? null : "Vulkan devices are present, but the SPIR-V shader blobs (spv/ beside the binary) were not found.",
            // Every physical device is selectable as vulkan:N; plain "vulkan" is the one flagged is_default.
            Devices = count > 0 && shaders
                ? infos.Select(d => new DeviceInfoDto
                {
                    Index = d.Index,
                    Name = d.Name,
                    DeviceString = $"vulkan:{d.Index}",
                    TotalMemoryBytes = d.DeviceLocalHeapBytes > 0 ? d.DeviceLocalHeapBytes : null,
                    IsDefault = d.IsDefault ? true : null,
                }).ToArray()
                : [],
        };
    }

    private static BackendInfoDto DescribeHip() => DescribeHip(
        DotLLM.Hip.HipDevice.IsAvailable,
        DotLLM.Hip.HipDevice.GetDeviceCount,
        i =>
        {
            var device = DotLLM.Hip.HipDevice.GetDevice(i);
            return new DeviceInfoDto { Index = i, Name = device.Name, TotalMemoryBytes = device.TotalMemoryBytes };
        },
        i =>
        {
            // Probe the same native manifest and loading path used by the model. Enumeration alone is insufficient:
            // e.g. gfx1036 can enumerate even though the bundled code objects have no compatible image for it.
            using var context = DotLLM.Hip.HipContext.Create(i);
            using var stream = DotLLM.Hip.HipStream.Create();
            using var kernels = new DotLLM.Hip.Kernels.DenseF16Kernels(context, stream);
            using var storage = new DotLLM.Hip.DirectStorage.HipDirectWeightStore(context);
        });

    /// <summary>Separates runtime detection from per-device inference readiness; injectable for hosts without HIP.</summary>
    internal static BackendInfoDto DescribeHip(Func<bool> available, Func<int> count,
        Func<int, DeviceInfoDto> describe, Action<int> validate)
    {
        const string boundary = "Experimental, explicit --device hip:N only; dense Llama GGUF (F16/Q4_0/Q8_0) or F16/BF16/AWQ4/GPTQ4 SafeTensors, F16/Q8_0 KV. Windows x64 DirectStorage; Linux x64 requires AIS-capable driver/storage and remains hardware-unverified. Hardware qualification: Windows RX 6900 XT / HIP 6.4.2.";
        if (!SafeProbe(available))
            return new BackendInfoDto { Name = "hip", Available = false, Servable = false,
                Note = "No HIP runtime / no HIP-capable AMD GPU detected." };
        var devices = new List<DeviceInfoDto>();
        try
        {
            int n = count();
            for (int i = 0; i < n; i++)
            {
                DeviceInfoDto device = new() { Index = i, Name = $"HIP device {i}" };
                try
                {
                    device = describe(i);
                    validate(i);
                    device = device with { DeviceString = $"hip:{i}", Servable = true, Note = boundary };
                }
                catch (Exception ex)
                {
                    device = device with { DeviceString = null, Servable = false, Note = ex.Message };
                }
                devices.Add(device);
            }
        }
        catch (Exception ex)
        {
            return new BackendInfoDto { Name = "hip", Available = true, Servable = false,
                Note = $"HIP runtime detected, but device enumeration failed: {ex.Message}" };
        }
        return new BackendInfoDto { Name = "hip", Available = true, DeviceCount = devices.Count,
            Servable = devices.Any(d => d.Servable == true), Devices = devices.ToArray(), Note = boundary };
    }

    private static bool SafeProbe(Func<bool> probe)
    {
        try { return probe(); }
        catch { return false; }
    }
}
