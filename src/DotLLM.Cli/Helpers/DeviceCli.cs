using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Server;
using Spectre.Console;

namespace DotLLM.Cli.Helpers;

/// <summary>
/// CLI-side glue over the shared device resolution (<see cref="DeviceSpec"/>, <see cref="DeviceSelector"/>, <see cref="DeviceModelLoader"/>)
/// used by <c>run</c> and <c>chat</c> (#790): validation, the load with deferred messages (the Spectre spinner would overwrite console
/// writes made during the load), and the "device resolved to X" report with its prominent CPU-fallback warning.
/// </summary>
internal static class DeviceCli
{
    /// <summary>Value of <c>--device</c> when the option is omitted.</summary>
    public const string DefaultDevice = "auto";

    /// <summary>The <c>--device</c> option help shared by <c>run</c>, <c>chat</c> and <c>bench</c>.</summary>
    public const string OptionHelp =
        "Compute device: 'auto' (default: CUDA if the model fits, else Vulkan, else CPU - with a warning if it ends on the CPU), " +
        "'cpu', 'vulkan', 'gpu'/'cuda', 'gpu:1'/'cuda:1'. An explicit GPU that cannot be honoured is an error, never a silent CPU run.";

    /// <summary>Returns a user-facing error for an unrecognised <c>--device</c> value, else null.</summary>
    public static string? Validate(string? device) =>
        DeviceSpec.TryParse(device, out _, out string? error) ? null : error;

    /// <summary>What a device-aware load produced, for the report line.</summary>
    /// <param name="Requested">The raw <c>--device</c> value.</param>
    /// <param name="Resolved">Canonical device used (<c>cpu</c>, <c>vulkan</c>, <c>gpu:N</c>).</param>
    /// <param name="Warning">Prominent warning to show, or null.</param>
    /// <param name="Notes">Informational lines produced during the load.</param>
    /// <param name="Label">Device name (GPU name) for the report line, or null.</param>
    internal sealed record Outcome(string Requested, string Resolved, string? Warning, IReadOnlyList<string> Notes, string? Label = null)
    {
        private string LabelSuffix => Label is null ? "" : $" ({Label})";

        /// <summary>The single "device: requested -> resolved" line.</summary>
        public string Line => string.Equals(Requested, Resolved, StringComparison.OrdinalIgnoreCase)
            ? $"device: {Resolved}{LabelSuffix}"
            : $"device: {Requested} -> {Resolved}{LabelSuffix}";
    }

    /// <summary>
    /// Loads a GGUF through the shared dispatch. Messages are collected in the returned <see cref="Outcome"/> so they can be printed after
    /// the spinner. Throws <see cref="DeviceUnavailableException"/> for an explicit device that cannot be honoured.
    /// </summary>
    public static (DeviceLoadResult Load, Outcome Outcome) Load(
        GgufFile gguf, ModelConfig config, string device, int? gpuLayers, ThreadingConfig threading,
        string modelPath)
    {
        var notes = new List<string>();
        var warnings = new List<string>();
        var res = DeviceModelLoader.Load(
            gguf, config, device, gpuLayers, threading,
            new FileInfo(modelPath).Length, Path.GetFileName(modelPath),
            notes.Add, warnings.Add);
        string? warning = res.Warning;
        if (warnings.Count > 0)
            warning = warning is null ? string.Join(" ", warnings) : warning + " " + string.Join(" ", warnings);
        return (res, new Outcome(device, res.ResolvedDevice, warning, notes, DeviceLabel(res.ResolvedDevice)));
    }

    /// <summary>Human-readable name of the resolved device (GPU name, or the CPU thread count), best effort.</summary>
    public static string? DeviceLabel(string resolved)
    {
        try
        {
            if (!DeviceSpec.TryParse(resolved, out var s, out _)) return null;
            return s.Kind switch
            {
                DeviceKind.Vulkan => DotLLM.Vulkan.VulkanModelLoader.SharedDevice.DeviceName,
                DeviceKind.Cuda => DotLLM.Cuda.CudaDevice.GetDevice(s.Ordinal).Name,
                _ => null,
            };
        }
        catch { return null; }
    }

    /// <summary>
    /// Device resolution for checkpoints the CLI can only load on the CPU (HuggingFace safetensors). An explicit GPU request is an error;
    /// <c>auto</c> resolves to the CPU with a prominent warning.
    /// </summary>
    public static Outcome ResolveCpuOnly(string device, string what, string modelName)
    {
        var spec = DeviceSpec.Parse(device);
        if (spec.IsGpu)
            throw new DeviceUnavailableException(DeviceSelector.ExplicitFailureMessage(
                device, modelName, 0, $"{what} can only be loaded on the CPU by the CLI (no GPU loader for them yet)"));
        string? warning = spec.Kind == DeviceKind.Auto
            ? $"Running on the CPU: --device auto cannot use a GPU for {what} (CPU-only loader). Expect much lower throughput than a GPU run. Pass --device cpu to silence this."
            : null;
        return new Outcome(device, "cpu", warning, []);
    }

    /// <summary>
    /// Device resolution for commands that load through their own backend switch (<c>bench</c>, <c>perplexity</c>): parses, preflights an
    /// explicit GPU against the machine, resolves <c>auto</c> to the best servable device, and reports it. Returns null (after printing the
    /// error) when the request cannot be honoured - never a CPU fallback.
    /// </summary>
    /// <param name="device">Raw <c>--device</c>.</param>
    /// <param name="modelPath">GGUF path (size feeds the auto fit check and the error message).</param>
    /// <param name="json">Write diagnostics to stderr.</param>
    /// <param name="backend">Resolved backend: <c>cpu</c>, <c>cuda</c> or <c>vulkan</c>.</param>
    /// <param name="ordinal">CUDA ordinal (0 otherwise).</param>
    public static bool TryResolveForTool(string device, string modelPath, bool json, out string backend, out int ordinal)
    {
        backend = "cpu"; ordinal = 0;
        if (!DeviceSpec.TryParse(device, out var spec, out string? error)) { PrintError(error!, json); return false; }
        long bytes = new FileInfo(modelPath).Length;
        var plan = DeviceSelector.Plan(spec, bytes, DotLLM.Server.Endpoints.DeviceEndpoint.Describe());
        if (plan.Error is not null)
        {
            PrintError(DeviceSelector.ExplicitFailureMessage(device, Path.GetFileName(modelPath), bytes, plan.Error), json);
            return false;
        }
        string resolved = plan.Candidates[0];
        DeviceSpec.TryParse(resolved, out var rs, out _);
        backend = rs.Kind switch { DeviceKind.Cuda => "cuda", DeviceKind.Vulkan => "vulkan", _ => "cpu" };
        ordinal = rs.Ordinal;
        if (spec.Kind == DeviceKind.Auto || plan.CpuWarning is not null)
            Report(new Outcome(device, resolved, plan.CpuWarning, [], DeviceLabel(resolved)), json);
        return true;
    }

    /// <summary>Prints the notes, the resolved-device line and any warning (stderr when <paramref name="json"/>, so stdout stays pure JSON).</summary>
    public static void Report(Outcome o, bool json)
    {
        if (json)
        {
            foreach (var n in o.Notes) Console.Error.WriteLine(n);
            Console.Error.WriteLine(o.Line);
            if (o.Warning is not null) Console.Error.WriteLine($"WARNING: {o.Warning}");
            return;
        }
        foreach (var n in o.Notes) AnsiConsole.MarkupLine($"[dim]{Markup.Escape(n)}[/]");
        AnsiConsole.MarkupLine($"[dim]{Markup.Escape(o.Line)}[/]");
        if (o.Warning is not null) AnsiConsole.MarkupLine($"[bold yellow]WARNING: {Markup.Escape(o.Warning)}[/]");
    }

    /// <summary>Prints a <see cref="DeviceUnavailableException"/> / validation message the way each command prints errors.</summary>
    public static void PrintError(string message, bool json)
    {
        if (json) Console.Error.WriteLine($"Error: {message}");
        else AnsiConsole.MarkupLine($"[red]{Markup.Escape(message)}[/]");
    }

    /// <summary>CUDA ordinal of a resolved device string (<c>gpu:N</c>); 0 otherwise.</summary>
    public static int CudaOrdinal(string resolved) =>
        DeviceSpec.TryParse(resolved, out var s, out _) && s.Kind == DeviceKind.Cuda ? s.Ordinal : 0;
}
