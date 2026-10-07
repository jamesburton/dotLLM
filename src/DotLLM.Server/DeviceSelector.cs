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

    /// <summary>Tracking issue linked from every "GPU requested but unavailable" message.</summary>
    public const string PolicyIssueUrl = "https://github.com/jamesburton/dotLLM/issues/790";

    /// <summary>
    /// Resolves a parsed <c>--device</c> request against the machine's devices without loading anything (#790). An explicit GPU request that no
    /// servable device can satisfy yields <see cref="DevicePlan.Error"/> and no candidates - it never resolves to the CPU.
    /// <c>auto</c> that has no servable GPU resolves to the CPU with <see cref="DevicePlan.CpuWarning"/> set.
    /// </summary>
    public static DevicePlan Plan(DeviceSpec spec, long modelBytes, Models.DeviceListResponse devices)
    {
        var backends = devices.Backends ?? [];
        switch (spec.Kind)
        {
            case DeviceKind.Cpu:
                return new DevicePlan(spec, ["cpu"], null, null);

            case DeviceKind.Auto:
            {
                var candidates = Candidates(modelBytes, devices);
                string? warn = candidates[0] == "cpu" ? NoGpuWarning(backends, modelBytes) : null;
                return new DevicePlan(spec, candidates, null, warn);
            }

            case DeviceKind.Cuda:
            {
                var cuda = backends.FirstOrDefault(b => b.Name == "cuda");
                string want = $"gpu:{spec.Ordinal}";
                if (cuda is { Servable: true } && (cuda.Devices ?? []).Any(d => d.DeviceString == want))
                    return new DevicePlan(spec, [want], null, null);
                // Distinguish "no CUDA here" (this branch: nothing was attempted) from "CUDA present but the load failed" (a load-time error
                // with real numbers, produced by the loader - see DeviceModelLoader.LoadWith).
                string why = cuda is null || !cuda.Servable
                    ? "CUDA requested but no NVIDIA/CUDA device is available on this machine" + (cuda?.Note is { } n ? $" ({n})" : "")
                      + "; use --device vulkan on AMD/Intel GPUs, or run on a CUDA host"
                    : $"CUDA device {spec.Ordinal} does not exist (this machine has {cuda.DeviceCount} CUDA device(s): gpu:0..gpu:{cuda.DeviceCount - 1})";
                return new DevicePlan(spec, [], why + AlternativeHint(backends, except: "cuda"), null);
            }

            default: // Vulkan
            {
                var vk = backends.FirstOrDefault(b => b.Name == "vulkan");
                if (vk is { Servable: true } && vk.Devices is { Length: > 0 })
                    return new DevicePlan(spec, ["vulkan"], null, null);
                string why = "Vulkan is not usable on this machine" + (vk?.Note is { } n ? $" ({n})" : "");
                return new DevicePlan(spec, [], why + AlternativeHint(backends, except: "vulkan"), null);
            }
        }
    }

    private static string AlternativeHint(Models.BackendInfoDto[] backends, string except)
    {
        var other = backends.Where(b => b.Servable && b.Name != except && b.Name != "cpu").Select(b => b.Name).ToArray();
        return other.Length == 0 ? "" : $"; {string.Join(" / ", other)} is available - try --device {(other[0] == "cuda" ? "gpu" : other[0])}";
    }

    /// <summary>Why <c>--device auto</c> had no GPU to try (per-backend notes) and what that costs.</summary>
    public static string NoGpuWarning(Models.BackendInfoDto[] backends, long modelBytes)
    {
        var reasons = backends.Where(b => b.Name != "cpu")
            .Select(b => $"{b.Name}: " + (b.Servable ? $"does not fit a {modelBytes / (1024.0 * 1024 * 1024):F1} GiB model" : b.Note ?? "unavailable"))
            .ToArray();
        return "Running on the CPU: --device auto found no GPU that can load this model"
            + (reasons.Length > 0 ? $" ({string.Join("; ", reasons)})" : "")
            + ". Expect much lower throughput (typically several times slower). Pass --device cpu to silence this.";
    }

    /// <summary>
    /// The clean error for an explicitly requested GPU that cannot be honoured: what, why, model size vs device memory, the opt-in to the CPU, and the policy issue.
    /// </summary>
    public static string ExplicitFailureMessage(string requested, string modelName, long modelBytes, string reason)
    {
        string size = $"{modelBytes / (1024.0 * 1024 * 1024):F1} GiB";
        string mem = "";
        try
        {
            if (DeviceSpec.TryParse(requested, out var s, out _) && s.Kind == DeviceKind.Cuda)
            {
                var d = DeviceEndpoint.Describe().Backends?.FirstOrDefault(b => b.Name == "cuda")?.Devices?.FirstOrDefault(x => x.Index == s.Ordinal);
                if (d?.TotalMemoryBytes is { } t) mem = $" on a {t / (1024.0 * 1024 * 1024):F1} GiB device";
            }
        }
        catch { /* inventory is best-effort */ }
        return $"--device {requested} was requested for '{modelName}' ({size} of weights{mem}) but cannot be honoured: {reason}. "
            + "dotllm will not silently fall back to the CPU (typically several times slower). "
            + $"Pass --device cpu to run on the CPU deliberately, or --device auto to let dotllm choose (with a warning). See {PolicyIssueUrl}.";
    }

    /// <summary>The prominent warning for <c>--device auto</c> ending up on the CPU after a GPU load failed (#733).</summary>
    public static string FallbackWarning(string modelName, IReadOnlyList<string> gpuFailures) =>
        $"Model '{modelName}' is running on the CPU because every GPU device failed to load it under --device auto "
        + $"({string.Join("; ", gpuFailures)}). Expect much lower throughput (typically several times slower). "
        + "Pass an explicit --device (cpu to silence this, or gpu/vulkan to get the real error).";
}
