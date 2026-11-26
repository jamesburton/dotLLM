using DotLLM.Server;
using DotLLM.Server.Models;
using DotLLM.Server.Tuning;
using Xunit;

namespace DotLLM.Tests.Unit.Server.Tuning;

/// <summary>
/// Apply-time admission of a saved placement (#1902, the no-spill rule): <see cref="AutoEntry.Admit"/> admits only a peak that was recorded,
/// against free memory that could be read. Also the GPU fingerprint fields the key is built from.
/// </summary>
public sealed class AutoAdmissionTests
{
    private const long GiB = 1L << 30;

    private static DeviceListResponse Devices(long? cudaFree, long vulkanHeap = 64 * GiB, bool vulkanIntegrated = true, long? vulkanFree = null) => new()
    {
        Backends =
        [
            new BackendInfoDto
            {
                Name = "cuda", Available = true, DeviceCount = 1, Servable = true,
                Devices = [new DeviceInfoDto { Index = 0, Name = "NVIDIA RTX 3060", TotalMemoryBytes = 12 * GiB, FreeMemoryBytes = cudaFree }],
            },
            new BackendInfoDto
            {
                Name = "vulkan", Available = true, DeviceCount = 1, Servable = true,
                Devices =
                [
                    new DeviceInfoDto
                    {
                        Index = 0, Name = "AMD Radeon 8060S", TotalMemoryBytes = vulkanHeap, FreeMemoryBytes = vulkanFree,
                        IsIntegrated = vulkanIntegrated ? true : null, IsDefault = true,
                    },
                ],
            },
        ],
    };

    private static string? Admit(TunedSettings c, string device, DeviceListResponse devices, long hostFree = 32 * GiB) =>
        AutoEntry.Admit(c, DeviceSpec.Parse(device), devices, () => hostFree);

    [Fact]
    public void Cpu_RecordedHostPeakThatFits_IsAdmitted_AndOneThatDoesNot_IsRefusedWithBytes()
    {
        Assert.Null(Admit(new TunedSettings { PeakHostBytes = 4 * GiB }, "cpu", Devices(null)));
        string? r = Admit(new TunedSettings { PeakHostBytes = 40 * GiB }, "cpu", Devices(null));
        Assert.NotNull(r);
        Assert.Contains("40.00 GiB", r);
        Assert.Contains("32.00 GiB", r);
    }

    [Fact]
    public void UnrecordedHostPeak_IsRefused()
    {
        string? r = Admit(new TunedSettings { Threads = 8 }, "cpu", Devices(null));
        Assert.NotNull(r);
        Assert.Contains("does not record its peak host memory", r);
    }

    [Fact]
    public void Gpu_UnrecordedDevicePeak_IsRefused_EvenWhenTheHostPeakFits()
    {
        string? r = Admit(new TunedSettings { PeakHostBytes = 1 * GiB }, "gpu:0", Devices(cudaFree: 11 * GiB));
        Assert.NotNull(r);
        Assert.Contains("does not record its peak device memory", r);
    }

    [Fact]
    public void Discrete_DevicePeakAgainstFreeVram()
    {
        var c = new TunedSettings { PeakHostBytes = 1 * GiB, PeakDeviceBytes = 10 * GiB };
        Assert.Null(Admit(c, "gpu:0", Devices(cudaFree: 11 * GiB)));
        string? r = Admit(c, "gpu:0", Devices(cudaFree: 9 * GiB));
        Assert.NotNull(r);
        Assert.Contains("10.00 GiB", r);
        Assert.Contains("9.00 GiB", r);
    }

    [Fact]
    public void Discrete_UnknownFreeMemory_IsRefused_NotAdmittedAgainstTheTotal()
    {
        // 10 GiB fits the 12 GiB total, so an implementation that falls back to the total would admit it.
        var c = new TunedSettings { PeakHostBytes = 1 * GiB, PeakDeviceBytes = 10 * GiB };
        string? r = Admit(c, "gpu:0", Devices(cudaFree: null));
        Assert.NotNull(r);
        Assert.Contains("cannot be read", r);
        // the same for a discrete Vulkan device whose free memory is not reported (it fits the 16 GiB heap, and the host has 32 GiB free)
        Assert.NotNull(Admit(c, "vulkan", Devices(null, vulkanHeap: 16 * GiB, vulkanIntegrated: false)));
    }

    [Fact]
    public void Integrated_DeviceAndHostPeaksShareTheHostPool()
    {
        // 20 + 10 = 30 GiB fits 32 GiB free; 20 + 16 = 36 GiB does not, although each alone would.
        Assert.Null(Admit(new TunedSettings { PeakHostBytes = 10 * GiB, PeakDeviceBytes = 20 * GiB }, "vulkan", Devices(null)));
        string? r = Admit(new TunedSettings { PeakHostBytes = 16 * GiB, PeakDeviceBytes = 20 * GiB }, "vulkan", Devices(null));
        Assert.NotNull(r);
        Assert.Contains("shared", r);
    }

    [Fact]
    public void Integrated_DevicePeakAboveTheDeviceLocalHeap_IsRefused()
    {
        // 20 GiB fits the 32 GiB of free host RAM, but not a 16 GiB device-local heap.
        string? r = Admit(new TunedSettings { PeakHostBytes = 1 * GiB, PeakDeviceBytes = 20 * GiB }, "vulkan", Devices(null, vulkanHeap: 16 * GiB));
        Assert.NotNull(r);
        Assert.Contains("device-local heap", r);
    }

    [Fact]
    public void NotEnumeratedDevice_IsRefused()
    {
        Assert.NotNull(Admit(new TunedSettings { PeakHostBytes = 1 * GiB, PeakDeviceBytes = 1 * GiB }, "gpu:3", Devices(cudaFree: 11 * GiB)));
    }

    [Fact]
    public void SavedDevice_IsWhatIsAdmitted_NotTheRequestedOne()
    {
        // --device auto, the saved result chose gpu:0 (CUDA): its free VRAM decides, not the host.
        var c = new TunedSettings { Device = "gpu:0", PeakHostBytes = 1 * GiB, PeakDeviceBytes = 10 * GiB };
        Assert.NotNull(AutoEntry.Admit(c, DeviceSpec.Parse("auto"), Devices(cudaFree: 2 * GiB), () => 64 * GiB));
    }

    [Fact]
    public void GpuFingerprint_CarriesDeviceIdDriverAndVram()
    {
        var list = new DeviceListResponse
        {
            Backends =
            [
                new BackendInfoDto
                {
                    Name = "cuda", Available = true, DeviceCount = 1, Servable = true,
                    Devices =
                    [
                        new DeviceInfoDto
                        {
                            Index = 0, Name = "NVIDIA RTX 3060", PciVendorId = 0x10DE, PciDeviceId = 0x2487, TotalMemoryBytes = 12 * GiB,
                            DriverVersion = "566.36", ApiVersion = "12070",
                        },
                    ],
                },
                new BackendInfoDto
                {
                    Name = "vulkan", Available = true, DeviceCount = 1, Servable = true,
                    Devices =
                    [
                        new DeviceInfoDto
                        {
                            Index = 0, Name = "AMD Radeon 8060S", PciVendorId = 0x1002, PciDeviceId = 0x1586, TotalMemoryBytes = 64 * GiB,
                            IsIntegrated = true, DriverVersion = "8388910", DriverId = "1", DriverInfo = "25.10.1", ApiVersion = "1.4.315",
                        },
                    ],
                },
            ],
        };
        var cuda = DeviceFingerprints.For("gpu:0", list)!;
        Assert.Equal((0x2487u, 12 * GiB, "566.36", "12070"), (cuda.PciDeviceId!.Value, cuda.DeviceMemoryBytes!.Value, cuda.DriverVersion, cuda.ApiVersion));
        var vk = DeviceFingerprints.For("vulkan:0", list)!;
        Assert.Equal((0x1586u, "8388910", "1", "25.10.1", "1.4.315"), (vk.PciDeviceId!.Value, vk.DriverVersion, vk.DriverId, vk.DriverInfo, vk.ApiVersion));
        Assert.Null(vk.DeviceMemoryBytes);   // UMA: kept out of the key
    }
}
