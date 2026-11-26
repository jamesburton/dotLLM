using System.Security.Cryptography;
using System.Text;
using DotLLM.Server.Models;

namespace DotLLM.Server.Tuning;

/// <summary>Everything <see cref="AutoEntry.Resolve"/> needs from a command (#1902).</summary>
/// <param name="RawArgs">The raw command-line arguments, for <see cref="AutoSettings.ExplicitFlags"/>.</param>
/// <param name="ModelArg">The model as typed, for the printed <c>dotllm tune</c> command.</param>
/// <param name="ModelPath">The resolved model file or directory.</param>
/// <param name="Device">The parsed <c>--device</c>.</param>
/// <param name="Profile">The profile name (default <c>chat</c>).</param>
/// <param name="ServedContext">The context the command would serve (0 when unknown).</param>
/// <param name="Policy">The command.</param>
/// <param name="Supported">Settings the command accepts.</param>
/// <param name="DefaultPaged">Whether the command's default KV cache is paged (true for serve, false for run, chat and bench).</param>
/// <param name="OverlayPath"><c>--tensor-overlay</c>.</param>
/// <param name="TernaryAdapters"><c>--ternary-adapter</c> paths.</param>
/// <param name="MtpHeadPath"><c>--mtp-head</c>.</param>
/// <param name="DFlashPath"><c>--dflash</c>.</param>
public sealed record AutoRequest(
    IReadOnlyList<string> RawArgs, string ModelArg, string ModelPath, DeviceSpec Device, string Profile, int ServedContext,
    AutoPolicy Policy, IReadOnlySet<string> Supported, bool DefaultPaged,
    string? OverlayPath = null, IReadOnlyList<string>? TernaryAdapters = null, string? MtpHeadPath = null, string? DFlashPath = null);

/// <summary>
/// The one entry point <c>serve</c>, <c>run</c>, <c>chat</c> and <c>bench</c> call before the model loads when <c>--auto</c> is given (#1902).
/// It only reuses saved results: it never starts a search, and a miss runs today's defaults with the exact <c>dotllm tune</c> command to run.
/// </summary>
public static class AutoEntry
{
    /// <summary>The default profile name.</summary>
    public const string DefaultProfile = "chat";

    /// <summary>The default KV quality budget until slice 3 measures a band: the floor of max(2 x band, 0.01).</summary>
    public const double DefaultKvBudget = 0.01;

    /// <summary>Resolves the saved result for a request and applies it under the precedence rules.</summary>
    /// <param name="request">The request.</param>
    /// <param name="store">The store; null uses <c>DOTLLM_TUNING_DIR</c> or <c>~/.dotllm/tuning</c>.</param>
    /// <param name="devices">Device enumeration (tests inject one); null probes this machine.</param>
    /// <param name="hostFreeBytes">Free host memory now (tests inject one); null reads the system.</param>
    /// <param name="currentKey">Overrides the computed key (tests); null computes it.</param>
    public static AutoApplied Resolve(AutoRequest request, TuningStore? store = null, DeviceListResponse? devices = null,
                                      Func<long>? hostFreeBytes = null, TuningKey? currentKey = null)
    {
        store ??= new TuningStore();
        hostFreeBytes ??= HostFreeBytes;
        var key = currentKey ?? BuildKey(request, devices);
        var match = store.Find(key);

        var explicitFlags = AutoSettings.ExplicitFlags(request.RawArgs);
        string tune = TuneCommand(request, explicitFlags);
        return AutoSettings.Apply(explicitFlags, match, tune, request.ServedContext,
            (c, _) => Admit(c, request.Device, devices, hostFreeBytes),
            c => Refusal(c, request), request.Policy, request.Supported);
    }

    /// <summary>The current key for a request: model, modifiers, device scope and fingerprints, profile, build and environment.</summary>
    public static TuningKey BuildKey(AutoRequest r, DeviceListResponse? devices = null)
    {
        IReadOnlyList<DeviceFingerprint> fps = r.Device.Kind == DeviceKind.Auto
            ? DeviceFingerprints.Machine(devices)
            : [DeviceFingerprints.For(r.Device.Canonical, devices) ?? new DeviceFingerprint { Backend = r.Device.Canonical }];
        var (envHash, envPairs) = EnvOverrides.Current();
        return new TuningKey
        {
            SchemaVersion = TuningStore.SchemaVersion,
            ModelHash = ModelHash.Of(r.ModelPath),
            ModelModifiers = ModelHash.Modifiers(r.OverlayPath, r.TernaryAdapters, r.MtpHeadPath, r.DFlashPath),
            Scope = TuningKeys.ScopeFor(r.Device, fps),
            Devices = fps,
            Profile = r.Profile,
            // Slice 4 makes profiles real; until then the built-in profile is identified by its name and version.
            ProfileHash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes("builtin:" + r.Profile + ":1"))).ToLowerInvariant(),
            KvQualityBudget = DefaultKvBudget,
            Perf = PerfIdentities.Current(),
            EnvOverridesHash = envHash,
            EnvOverrides = envPairs,
            ProtocolVersion = TuningStore.ProtocolVersion,
        };
    }

    private static string TuneCommand(AutoRequest r, IReadOnlySet<string> flagged)
    {
        string cmd = $"dotllm tune {Quote(r.ModelArg)} --profile {r.Profile}";
        if (flagged.Contains("device") || r.Device.Kind != DeviceKind.Auto) cmd += $" --device {r.Device.Canonical}";
        return cmd;
    }

    private static string Quote(string s) => s.Contains(' ') ? $"\"{s}\"" : s;

    /// <summary>Asks this build whether it runs the candidate's KV settings (<see cref="TurboQuantKvSupport.RefusalReason"/>, #1933).</summary>
    private static string? Refusal(TunedSettings c, AutoRequest r)
    {
        if (!AutoSettings.IsTurboQuant(c.CacheTypeK) && !AutoSettings.IsTurboQuant(c.CacheTypeV)) return null;
        KvBackendKind backend = BackendOf(c.Device is { } d && DeviceSpec.TryParse(d, out var cs, out _) ? cs : r.Device);
        try
        {
            var kv = TurboQuantKv.Parse(c.CacheTypeK, c.CacheTypeV, 0, out string? mismatch);
            var command = r.Policy.Command switch
            {
                AutoCommand.Serve => KvCommand.Serve,
                AutoCommand.Run => KvCommand.Run,
                AutoCommand.Chat => KvCommand.Chat,
                _ => KvCommand.Bench,
            };
            return TurboQuantKv.PreLoadRefusal(kv, mismatch, command, backend, c.Paged ?? r.DefaultPaged, config: null, hasDraftModel: false);
        }
        catch (ArgumentException ex)
        {
            return $"saved KV format is not valid in this build: {ex.Message}";
        }
    }

    private static KvBackendKind BackendOf(DeviceSpec s) => s.Kind switch
    {
        DeviceKind.Cpu => KvBackendKind.Cpu,
        DeviceKind.Cuda => KvBackendKind.Cuda,
        DeviceKind.Vulkan => KvBackendKind.Vulkan,
        _ => KvBackendKind.Unknown,
    };

    /// <summary>
    /// Re-admits the saved placement against free memory now (the no-spill rule). It admits only what it can bound: the recorded peak host
    /// bytes against free host RAM and, for a GPU placement, the recorded peak device bytes against the device's free memory. A peak that
    /// was not recorded, or a dedicated device whose free memory cannot be read, is refused: a heap's total size is not free memory. On an
    /// integrated GPU, device and host memory are one pool, so device plus host peaks must fit free host RAM (slice 2 records the two as
    /// disjoint figures) and the device peak must fit the device-local heap. <see cref="AutoSettings.Apply"/> bounds the served context
    /// against the tuned context before asking.
    /// </summary>
    /// <returns>Null when admitted, else the reason.</returns>
    internal static string? Admit(TunedSettings c, DeviceSpec requested, DeviceListResponse? devices, Func<long> hostFree)
    {
        DeviceSpec spec = c.Device is { } d && DeviceSpec.TryParse(d, out var s, out _) ? s : requested;
        if (c.PeakHostBytes is not long host) return "the saved result does not record its peak host memory";
        long hostFreeNow = hostFree();

        if (spec.IsGpu)
        {
            if (c.PeakDeviceBytes is not long dev) return "the saved result does not record its peak device memory";
            var (found, free, total, integrated) = DeviceMemory(spec, devices);
            if (!found) return $"device {spec.Canonical} is not enumerated, so its free memory cannot be read";
            if (integrated)
            {
                if (total is long heap && dev > heap)
                    return $"needs {AutoSettings.Format(dev)} of device memory, the device-local heap is {AutoSettings.Format(heap)}";
                if (dev + host > hostFreeNow)
                    return $"needs {AutoSettings.Format(dev + host)} of shared device and host memory, {AutoSettings.Format(hostFreeNow)} free";
                return null;
            }
            if (free is not long devFree)
                return $"free memory on {spec.Canonical} cannot be read, so the recorded device peak ({AutoSettings.Format(dev)}) cannot be admitted";
            if (dev > devFree) return $"needs {AutoSettings.Format(dev)} of device memory, {AutoSettings.Format(devFree)} free";
        }
        if (host > hostFreeNow) return $"needs {AutoSettings.Format(host)} of host memory, {AutoSettings.Format(hostFreeNow)} free";
        return null;
    }

    private static (bool Found, long? Free, long? Total, bool Integrated) DeviceMemory(DeviceSpec spec, DeviceListResponse? devices)
    {
        try
        {
            devices ??= DotLLM.Server.Endpoints.DeviceEndpoint.Describe();
            string backend = spec.Kind == DeviceKind.Cuda ? "cuda" : "vulkan";
            var list = devices.Backends.FirstOrDefault(b => b.Name == backend)?.Devices ?? [];
            var dev = spec.Kind == DeviceKind.Vulkan && !spec.ExplicitIndex
                ? list.FirstOrDefault(x => x.IsDefault == true) ?? list.FirstOrDefault()
                : list.FirstOrDefault(x => x.Index == spec.Ordinal);
            if (dev is null) return (false, null, null, false);
            return (true, dev.FreeMemoryBytes, dev.TotalMemoryBytes, dev.IsIntegrated == true);
        }
        catch (Exception ex) when (ex is InvalidOperationException or DllNotFoundException)
        {
            return (false, null, null, false);
        }
    }

    /// <summary>Free physical memory now: total available to the process minus the system-wide memory load.</summary>
    public static long HostFreeBytes()
    {
        var info = GC.GetGCMemoryInfo();
        return Math.Max(0, info.TotalAvailableMemoryBytes - info.MemoryLoadBytes);
    }
}
