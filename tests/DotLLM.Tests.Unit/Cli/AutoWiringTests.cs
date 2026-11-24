using System.Text.Json;
using DotLLM.Cli.Commands;
using DotLLM.Cli.Helpers;
using DotLLM.Server;
using DotLLM.Server.Tuning;
using DotLLM.Tests.Unit.Engine;
using Xunit;

namespace DotLLM.Tests.Unit.Cli;

/// <summary>
/// <c>--auto</c> wired into serve, run, chat and bench (#1902): the saved result is looked up before the model loads, explicit flags win,
/// a miss runs the defaults and says what to run, and nothing changes without <c>--auto</c>. CPU only, on the tiny Qwen fixture.
/// </summary>
[Collection(CliConsoleCollection.Name)]
public sealed class AutoWiringTests : IClassFixture<PrefixCacheFailedRequestTests.TinyQwen>, IDisposable
{
    private readonly PrefixCacheFailedRequestTests.TinyQwen _tiny;
    private readonly string _store = Path.Combine(Path.GetTempPath(), "dotllm-autowiring-" + Guid.NewGuid().ToString("N"));
    private readonly string? _oldTuning = Environment.GetEnvironmentVariable("DOTLLM_TUNING_DIR");
    private readonly string? _oldHash = Environment.GetEnvironmentVariable("DOTLLM_HASH_CACHE_DIR");

    /// <summary>The head-dim-32 model the fixture also writes: its KV rows fill whole 32-element blocks, which the quantised caches require.</summary>
    private string Model => Path.Combine(Path.GetDirectoryName(_tiny.GgufPath)!, "qwen3-hd32.gguf");

    public AutoWiringTests(PrefixCacheFailedRequestTests.TinyQwen tiny)
    {
        _tiny = tiny;
        Environment.SetEnvironmentVariable("DOTLLM_TUNING_DIR", _store);
        Environment.SetEnvironmentVariable("DOTLLM_HASH_CACHE_DIR", Path.Combine(_store, "hash-cache"));
    }

    public void Dispose()
    {
        AutoCli.RawArgsOverride = null;
        Environment.SetEnvironmentVariable("DOTLLM_TUNING_DIR", _oldTuning);
        Environment.SetEnvironmentVariable("DOTLLM_HASH_CACHE_DIR", _oldHash);
        try { Directory.Delete(_store, recursive: true); } catch (IOException) { }
    }

    private AutoRequest Request(AutoCommand command, string[] args) => new(args, Model, Model, DeviceSpec.Parse("cpu"),
        AutoEntry.DefaultProfile, 0, new AutoPolicy(command), AutoCli.ServeSupported, DefaultPaged: command == AutoCommand.Serve);

    /// <summary>Saves a result for the tiny model on the CPU, keyed exactly as the commands will key their lookup.</summary>
    private void SaveFor(TunedSettings settings, bool partial = false)
    {
        var key = AutoEntry.BuildKey(Request(AutoCommand.Run, []));
        new TuningStore().Save(new TuningResult
        {
            Key = key, Settings = settings, Partial = partial ? true : null, TunedAtUtc = "2026-11-24T00:00:00Z", Report = "test",
        });
    }

    private ServeCommand.Settings ServeSettings(params string[] args)
    {
        AutoCli.RawArgsOverride = args;
        return new ServeCommand.Settings { Model = Model, Device = "cpu", Auto = true };
    }

    [Fact]
    public void Serve_AutoHit_AppliesSavedKv_AndStatusReportsSources()
    {
        SaveFor(new TunedSettings { CacheTypeK = "q8_0", CacheTypeV = "q8_0", Threads = 2 });
        var s = ServeSettings("serve", Model, "--auto", "--device", "cpu");

        var options = ServeCommand.BuildServerOptionsAuto(s, null, Model);

        Assert.Equal("q8_0", options.CacheTypeK);
        Assert.Equal("q8_0", options.CacheTypeV);
        Assert.Equal(2, options.Threads);
        Assert.Equal("auto", options.SettingSources!["cache_type_k"]);
        Assert.Equal("flag", options.SettingSources["device"]);
        Assert.Equal("default", options.SettingSources["gpu_layers"]);

        using var state = ServerStartup.LoadModel(Model, options with { Warmup = new DotLLM.Engine.WarmupOptions { Enabled = false } });
        var status = state.GetStatus();
        Assert.Equal("auto", status.SettingsSource!["cache_type_k"]);
        string json = JsonSerializer.Serialize(status, ServerJsonContext.Default.ServerStatusResponse);
        Assert.Contains("\"settings_source\"", json);
    }

    [Fact]
    public void Serve_AutoMiss_RunsDefaults_AndAdvisesTheTuneCommand()
    {
        var s = ServeSettings("serve", Model, "--auto");
        var options = ServeCommand.BuildServerOptionsAuto(s, null, Model);
        Assert.Null(options.CacheTypeK);
        Assert.All(options.SettingSources!.Values, v => Assert.Equal("default", v));

        var applied = AutoEntry.Resolve(Request(AutoCommand.Serve, ["serve", "--auto"]), new TuningStore());
        Assert.Equal(TuningMatchKind.Miss, applied.Match.Kind);
        Assert.Contains("dotllm tune", applied.Advice);
        Assert.Contains("--profile chat", applied.Advice);
        Assert.Contains(Model, applied.Advice);
    }

    [Fact]
    public void Serve_WithoutAuto_StatusHasNoSettingsSource_AndTheStoreIsNotTouched()
    {
        SaveFor(new TunedSettings { CacheTypeK = "q8_0", CacheTypeV = "q8_0" });
        AutoCli.RawArgsOverride = ["serve", Model];
        var s = new ServeCommand.Settings { Model = Model, Device = "cpu" };   // no --auto
        var options = ServeCommand.BuildServerOptionsAuto(s, null, Model);
        Assert.Null(options.CacheTypeK);               // the saved result was NOT applied
        Assert.Null(options.SettingSources);
        Assert.Null(options.AutoProfile);

        using var state = ServerStartup.LoadModel(Model, options with { Warmup = new DotLLM.Engine.WarmupOptions { Enabled = false } });
        var status = state.GetStatus();
        Assert.Null(status.SettingsSource);
        Assert.DoesNotContain("settings_source", JsonSerializer.Serialize(status, ServerJsonContext.Default.ServerStatusResponse));
    }

    [Fact]
    public void Auto_SavedTurboQuant_OnAPagedServe_IsNeverSilentlySubstituted()
    {
        // A saved tq pair on a multi-session (paged) serve is dropped with its reason, and the server starts on its default cache.
        SaveFor(new TunedSettings { CacheTypeK = "tq4", CacheTypeV = "tq4", Paged = false });
        AutoCli.RawArgsOverride = ["serve", Model, "--auto"];
        var applied = AutoEntry.Resolve(Request(AutoCommand.Serve, ["serve", "--auto"]), new TuningStore());
        Assert.Null(applied.Applied.CacheTypeK);
        Assert.Null(applied.Applied.CacheTypeV);
        Assert.Null(applied.Applied.Paged);
        Assert.Contains(applied.Warnings, w => w.Contains("--no-paged", StringComparison.Ordinal));

        var s = new ServeCommand.Settings { Model = Model, Device = "cpu", Auto = true };
        var options = ServeCommand.BuildServerOptionsAuto(s, null, Model);
        Assert.Null(options.CacheTypeK);
        Assert.True(options.UsePaged);
    }

    [Fact]
    public void Auto_SavedTurboQuant_WithPagedTrue_IsRefusedByThisBuild()
    {
        // run defaults to the contiguous cache, but a saved paged=true with tq is something this build refuses: dropped, not substituted.
        SaveFor(new TunedSettings { CacheTypeK = "tq4", CacheTypeV = "tq4", Paged = true, Threads = 2 });
        var applied = AutoEntry.Resolve(Request(AutoCommand.Run, ["run", "--auto"]), new TuningStore());
        Assert.Null(applied.Applied.CacheTypeK);
        Assert.Equal(2, applied.Applied.Threads);
        Assert.Contains(applied.Warnings, w => w.Contains("paged", StringComparison.OrdinalIgnoreCase));
    }

    private (int Code, string Output) RunCli(params string[] args)
    {
        AutoCli.RawArgsOverride = args;
        try { return CliTurboQuantCommandTests.Run(args); }
        finally { AutoCli.RawArgsOverride = null; }
    }

    [Fact]
    public void Run_Auto_ThenModelArgument_ParsesTheModel()
    {
        SaveFor(new TunedSettings { CacheTypeK = "q8_0", CacheTypeV = "q8_0" });
        // --auto is a plain flag, so the model argument after it is still the model (Review Focus 8).
        var (code, output) = RunCli("run", "--auto", Model, "-p", "ab", "--max-tokens", "2", "--device", "cpu");
        Assert.True(code == 0, output);
        Assert.Contains("auto: settings from", output);
        Assert.Contains("K:Q8_0 V:Q8_0", output);                 // the saved KV pair reached the cache
    }

    [Fact]
    public void Run_Auto_WithExplicitKvFlag_FlagWins()
    {
        SaveFor(new TunedSettings { CacheTypeK = "q8_0", CacheTypeV = "q8_0" });
        var (code, output) = RunCli("run", Model, "--auto", "-p", "ab", "--max-tokens", "2", "--device", "cpu", "--cache-type-k", "q4_0");
        Assert.True(code == 0, output);
        Assert.Contains("K:Q4_0 V:Q8_0", output);                 // the flag for K, the saved value for V
    }

    [Fact]
    public void Run_Auto_Miss_RunsAndPrintsTheTuneCommand()
    {
        var (code, output) = RunCli("run", Model, "--auto", "-p", "ab", "--max-tokens", "2", "--device", "cpu");
        Assert.True(code == 0, output);
        Assert.Contains("dotllm tune", output);
        Assert.DoesNotContain("K:Q", output);                     // default cache, nothing applied
    }

    [Fact]
    public void Run_Auto_StaleResult_AppliesAndNamesTheMismatch()
    {
        SaveFor(new TunedSettings { CacheTypeK = "q8_0", CacheTypeV = "q8_0" });
        Environment.SetEnvironmentVariable("DOTLLM_CUDA_STEP_GRAPH", "0");   // an A/B variable that was not set at tune time
        try
        {
            var (code, output) = RunCli("run", Model, "--auto", "-p", "ab", "--max-tokens", "2", "--device", "cpu");
            Assert.True(code == 0, output);
            Assert.Contains("DOTLLM_CUDA_STEP_GRAPH", output);
            Assert.Contains("--recheck", output);
            Assert.Contains("K:Q8_0 V:Q8_0", output);              // stale still applies
        }
        finally { Environment.SetEnvironmentVariable("DOTLLM_CUDA_STEP_GRAPH", null); }
    }

    [Fact]
    public void Bench_Auto_AppliesSavedKv()
    {
        SaveFor(new TunedSettings { CacheTypeK = "q8_0", CacheTypeV = "q8_0" });
        var (code, output) = RunCli("bench", Model, "--auto", "--device", "cpu", "--prompt-tokens", "8", "--gen-tokens", "2", "--reps", "1");
        Assert.True(code == 0, output);
        Assert.Contains("auto: settings from", output);
    }

    [Fact]
    public void Chat_Auto_AppliesSavedResult()
    {
        SaveFor(new TunedSettings { CacheTypeK = "q8_0", CacheTypeV = "q8_0" });
        string[] args = ["chat", Model, "--auto", "--device", "cpu", "--max-tokens", "2"];
        AutoCli.RawArgsOverride = args;
        var (code, output) = CliTurboQuantCommandTests.RunWithInput("ab\n/exit\n", args);
        Assert.True(code == 0, output);
        Assert.Contains("auto: settings from", output);
    }
}
