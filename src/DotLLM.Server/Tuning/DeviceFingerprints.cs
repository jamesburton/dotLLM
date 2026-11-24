using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;

namespace DotLLM.Server.Tuning;

/// <summary>Device identity for the tuning key (#1902): CPU, CUDA and Vulkan together, built from the same enumeration as <c>/v1/devices</c>.</summary>
public static class DeviceFingerprints
{
    /// <summary>The fingerprint of the one device a canonical device string names (<c>cpu</c>, <c>gpu:N</c>/<c>cuda:N</c>, <c>vulkan</c>, <c>vulkan:N</c>).</summary>
    /// <param name="resolvedDevice">The device actually used, not device 0 by default.</param>
    /// <param name="devices">Enumeration to read; null probes this machine (cached).</param>
    /// <returns>The fingerprint, or null when the device is not enumerated.</returns>
    public static DeviceFingerprint? For(string resolvedDevice, DeviceListResponse? devices = null)
    {
        string d = resolvedDevice.Trim().ToLowerInvariant();
        if (d == "cpu") return Cpu();
        devices ??= DeviceEndpoint.Describe();
        if (!DeviceSpec.TryParse(d.StartsWith("cuda", StringComparison.Ordinal) ? "gpu" + d[4..] : d, out var spec, out _)) return null;
        switch (spec.Kind)
        {
            case DeviceKind.Cuda:
                return Backend(devices, "cuda").Devices.Where(x => x.Index == spec.Ordinal).Select(x => Gpu("cuda", x)).FirstOrDefault();
            case DeviceKind.Vulkan:
            {
                var vk = Backend(devices, "vulkan").Devices;
                var pick = spec.ExplicitIndex ? vk.FirstOrDefault(x => x.Index == spec.Ordinal)
                                              : vk.FirstOrDefault(x => x.IsDefault == true) ?? vk.FirstOrDefault();
                return pick is null ? null : Gpu("vulkan", pick);
            }
            case DeviceKind.Cpu: return Cpu();
            default: return null;
        }
    }

    /// <summary>Every enumerated device plus the CPU, in a stable order (CPU, CUDA by index, Vulkan by index).</summary>
    public static IReadOnlyList<DeviceFingerprint> Machine(DeviceListResponse? devices = null)
    {
        devices ??= DeviceEndpoint.Describe();
        var list = new List<DeviceFingerprint> { Cpu() };
        foreach (var x in Backend(devices, "cuda").Devices.OrderBy(x => x.Index)) list.Add(Gpu("cuda", x));
        foreach (var x in Backend(devices, "vulkan").Devices.OrderBy(x => x.Index)) list.Add(Gpu("vulkan", x));
        return list;
    }

    private static BackendInfoDto Backend(DeviceListResponse r, string name) =>
        r.Backends.FirstOrDefault(b => b.Name == name) ?? new BackendInfoDto { Name = name };

    private static DeviceFingerprint Gpu(string backend, DeviceInfoDto d) => new()
    {
        Backend = backend,
        GpuName = d.Name,
        PciVendorId = d.PciVendorId,
        PciDeviceId = d.PciDeviceId,
        // UMA devices report a heap whose size drifts with the system; it must not enter the key.
        DeviceMemoryBytes = d.IsIntegrated == true ? null : d.TotalMemoryBytes,
        DriverVersion = d.DriverVersion,
        ApiVersion = d.ApiVersion,
        OsBuild = Environment.OSVersion.VersionString,
    };

    private static DeviceFingerprint? s_cpu;

    /// <summary>The CPU fingerprint of this machine.</summary>
    public static DeviceFingerprint Cpu() => s_cpu ??= new DeviceFingerprint
    {
        Backend = "cpu",
        CpuModel = CpuModel(),
        LogicalCores = Environment.ProcessorCount,
        RamBytes = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes,
        OsBuild = Environment.OSVersion.VersionString,
    };

    private static string CpuModel()
    {
        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                using var k = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(@"HARDWARE\DESCRIPTION\System\CentralProcessor\0");
                if (k?.GetValue("ProcessorNameString") is string n && n.Length > 0) return n.Trim();
            }
            else if (File.Exists("/proc/cpuinfo"))
            {
                foreach (string line in File.ReadLines("/proc/cpuinfo"))
                    if (line.StartsWith("model name", StringComparison.Ordinal) && line.IndexOf(':') is var i and > 0)
                        return line[(i + 1)..].Trim();
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or System.Security.SecurityException) { /* fall through */ }
        return Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER") ?? RuntimeInformation.ProcessArchitecture.ToString();
    }

    /// <summary>
    /// Filename-safe key for one device: <c>{backend}-{vendor:x4}{device:x4}-{name}</c>, or <c>cpu-{model}</c>. Driver fields are excluded:
    /// a driver change makes the result stale, not a different file.
    /// </summary>
    public static string Key(DeviceFingerprint f)
    {
        string name = Sanitize(f.Backend == "cpu" ? f.CpuModel : f.GpuName);
        if (f.Backend == "cpu") return "cpu-" + name;
        return $"{f.Backend}-{f.PciVendorId.GetValueOrDefault():x4}{f.PciDeviceId.GetValueOrDefault():x4}-{name}";
    }

    internal static string Sanitize(string? s)
    {
        string t = Regex.Replace((s ?? "unknown").ToLowerInvariant(), "[^a-z0-9]+", "-", RegexOptions.None, TimeSpan.FromSeconds(1)).Trim('-');
        return t.Length == 0 ? "unknown" : t.Length > 48 ? t[..48] : t;
    }
}

/// <summary>Scope and key helpers for <see cref="TuningKey"/>.</summary>
public static class TuningKeys
{
    /// <summary>
    /// The scope (file-name prefix) of a result: the device key for an explicit device, or <c>auto-{machine key}</c> for <c>--device auto</c>,
    /// where the device is only known after the load.
    /// </summary>
    /// <param name="spec">The parsed <c>--device</c>.</param>
    /// <param name="devices">The fingerprints that scope the result: one device, or every device for <c>auto</c>.</param>
    public static string ScopeFor(DeviceSpec spec, IReadOnlyList<DeviceFingerprint> devices)
    {
        if (spec.Kind == DeviceKind.Auto)
        {
            string joined = string.Join('\n', devices.Select(DeviceFingerprints.Key));
            return "auto-" + ModelHash.Sha256Hex(joined)[..12];
        }
        return DeviceFingerprints.Key(devices[0]);
    }
}
