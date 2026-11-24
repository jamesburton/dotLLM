using DotLLM.Server;
using DotLLM.Server.Tuning;
using Spectre.Console;

namespace DotLLM.Cli.Helpers;

/// <summary>
/// Shared <c>--auto</c> glue (#1902) for <c>serve</c>, <c>run</c>, <c>chat</c> and <c>bench</c>: option text, one entry point that looks up
/// a saved tuning result and reports what it did. It only REUSES saved results; <c>dotllm tune</c> (slice 2) is what writes them.
/// </summary>
internal static class AutoCli
{
    /// <summary>Option name.</summary>
    public const string Option = "--auto";

    /// <summary>Profile option name.</summary>
    public const string ProfileOption = "--auto-profile";

    /// <summary>Help text of <see cref="Option"/>.</summary>
    public const string Help =
        "Apply the saved tuning result for this model, device and build (device, GPU layers, KV format, prefill chunk, MTP, threads...). Explicit flags always win. " +
        "On a miss it runs the defaults and prints the exact 'dotllm tune' command to run; a result that is stale (driver, build, runtime or DOTLLM_* override changed) " +
        "is still applied, with a warning naming what changed and the 'dotllm tune --recheck' command. The saved placement must still fit free memory, or the defaults run (never spills). " +
        "Never starts a search. See docs/CLI.md#auto-tuned-settings---auto.";

    /// <summary>Help text of <see cref="ProfileOption"/>.</summary>
    public const string ProfileHelp = "With --auto: the tuning profile to look up (default chat). Only the built-in chat profile exists in this slice.";

    /// <summary>Raw arguments of the command line; tests set it because they drive the command tree in-process.</summary>
    internal static IReadOnlyList<string>? RawArgsOverride { get; set; }

    private static IReadOnlyList<string> RawArgs => RawArgsOverride ?? Environment.GetCommandLineArgs().Skip(1).ToArray();

    /// <summary>Settings supported by each command.</summary>
    internal static readonly IReadOnlySet<string> RunChatSupported = new HashSet<string>
    {
        "device", "gpu_layers", "moe_offload", "cache_type_k", "cache_type_v", "paged", "prefill_chunk_size", "mtp", "speculative_k",
        "threads", "decode_threads",
    };

    /// <summary>Settings <c>serve</c> accepts.</summary>
    internal static readonly IReadOnlySet<string> ServeSupported = new HashSet<string>(RunChatSupported)
    {
        "expected_concurrency", "ctx_size",
    };

    /// <summary>Settings <c>bench</c> accepts.</summary>
    internal static readonly IReadOnlySet<string> BenchSupported = new HashSet<string>
    {
        "device", "cache_type_k", "cache_type_v", "threads", "decode_threads",
    };

    /// <summary>
    /// Looks up and applies the saved result for a command that was given <c>--auto</c>, and logs what happened. Never throws for a store
    /// problem: a failure here runs the defaults.
    /// </summary>
    /// <param name="command">The command.</param>
    /// <param name="modelArg">The model as typed.</param>
    /// <param name="modelPath">The resolved model file or directory.</param>
    /// <param name="device">The <c>--device</c> value (already validated).</param>
    /// <param name="profile">The <c>--auto-profile</c> value or null.</param>
    /// <param name="servedContext">The context that would be served (0 = unknown).</param>
    /// <param name="supported">Settings the command takes.</param>
    /// <param name="json">The command writes JSON to stdout; the report goes to stderr.</param>
    /// <param name="overlay"><c>--tensor-overlay</c>.</param>
    /// <param name="ternaryAdapters"><c>--ternary-adapter</c> paths.</param>
    /// <param name="mtpHead"><c>--mtp-head</c>.</param>
    /// <param name="dflash"><c>--dflash</c>.</param>
    /// <returns>The outcome; <see cref="AutoApplied.Applied"/> holds what to set.</returns>
    public static AutoApplied? Resolve(AutoCommand command, string modelArg, string modelPath, string device, string? profile, int servedContext,
        IReadOnlySet<string> supported, bool json, string? overlay = null, IReadOnlyList<string>? ternaryAdapters = null,
        string? mtpHead = null, string? dflash = null)
    {
        try
        {
            var spec = DeviceSpec.Parse(device);
            var request = new AutoRequest(RawArgs, modelArg, modelPath, spec, string.IsNullOrWhiteSpace(profile) ? AutoEntry.DefaultProfile : profile.Trim(),
                servedContext, new AutoPolicy(command), supported, DefaultPaged: command == AutoCommand.Serve,
                overlay, ternaryAdapters, mtpHead, dflash);
            var applied = AutoEntry.Resolve(request);
            Report(applied, json);
            return applied;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException or InvalidOperationException)
        {
            Write($"auto: could not look up a tuning result ({ex.Message}); running the defaults", json, warn: true);
            return null;
        }
    }

    /// <summary>Logs the outcome: where the settings came from, each warning, and the advice line.</summary>
    internal static void Report(AutoApplied a, bool json)
    {
        if (a.Match.Result is { } r && a.Match.Kind != TuningMatchKind.Miss)
            Write($"auto: settings from {a.Match.Path} (tuned {r.TunedAtUtc ?? "unknown"}, {a.Match.Kind.ToString().ToLowerInvariant()})", json, warn: false);
        foreach (string w in a.Warnings) Write("auto: warning: " + w, json, warn: true);
        if (a.Advice is not null) Write("auto: " + a.Advice, json, warn: true);
    }

    private static void Write(string line, bool json, bool warn)
    {
        if (json) Console.Error.WriteLine(line);
        else AnsiConsole.MarkupLine(warn ? $"[yellow]{Markup.Escape(line)}[/]" : $"[dim]{Markup.Escape(line)}[/]");
    }

    /// <summary>The wire form of <see cref="AutoApplied.Sources"/> for <c>/v1/status</c>.</summary>
    public static IReadOnlyDictionary<string, string> SourceMap(AutoApplied a) =>
        a.Sources.ToDictionary(kv => kv.Key, kv => kv.Value.ToString().ToLowerInvariant());
}
