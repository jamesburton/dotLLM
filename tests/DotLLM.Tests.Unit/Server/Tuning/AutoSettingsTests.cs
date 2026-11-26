using DotLLM.Server.Tuning;
using Xunit;

namespace DotLLM.Tests.Unit.Server.Tuning;

/// <summary>Precedence, admission and provenance of <see cref="AutoSettings.Apply"/> (#1902).</summary>
public sealed class AutoSettingsTests
{
    private const long GiB = 1L << 30;
    private const string Tune = "dotllm tune m.gguf --profile chat";

    private static readonly PlacementAdmission Fits = (_, _) => null;
    private static readonly SettingRefusal NoRefusal = _ => null;
    private static readonly AutoPolicy Run = new(AutoCommand.Run);

    private static TuningMatch Match(TuningMatchKind kind, TunedSettings s, string reason = "") =>
        // A Miss deliberately still carries a result: the applier must go by the kind, not by whether a result happens to be present.
        new(kind, new TuningResult { Settings = s, TunedAtUtc = "2026-11-24T00:00:00Z" }, "p.json", reason);

    private static AutoApplied Apply(TuningMatch m, string[]? flags = null, AutoPolicy? policy = null, PlacementAdmission? admit = null,
                                     SettingRefusal? refuse = null, int ctx = 4096) =>
        AutoSettings.Apply(new HashSet<string>(flags ?? []), m, Tune, ctx, admit ?? Fits, refuse ?? NoRefusal, policy ?? Run);

    private static readonly TunedSettings Saved = new()
    {
        CacheTypeK = "q8_0", CacheTypeV = "q4_0", PrefillChunkSize = 512, Threads = 8, Device = "vulkan:0", TunedContext = 8192,
    };

    [Fact]
    public void Hit_AppliesEverythingSaved_WithAutoSources_AndNoAdvice()
    {
        var a = Apply(Match(TuningMatchKind.Hit, Saved));
        Assert.Equal("q8_0", a.Applied.CacheTypeK);
        Assert.Equal(512, a.Applied.PrefillChunkSize);
        Assert.Equal(SettingSource.Auto, a.Sources["cache_type_k"]);
        Assert.Equal(SettingSource.Auto, a.Sources["device"]);
        Assert.Equal(SettingSource.Default, a.Sources["gpu_layers"]);   // not tuned: stays default
        Assert.Null(a.Advice);
        Assert.Empty(a.Warnings);
    }

    [Fact]
    public void ExplicitFlagWins_PerSide()
    {
        var a = Apply(Match(TuningMatchKind.Hit, Saved), flags: ["cache_type_k"]);
        Assert.Null(a.Applied.CacheTypeK);                      // the command keeps the flag's value
        Assert.Equal(SettingSource.Flag, a.Sources["cache_type_k"]);
        Assert.Equal("q4_0", a.Applied.CacheTypeV);            // V was not flagged: it takes the saved value
        Assert.Equal(SettingSource.Auto, a.Sources["cache_type_v"]);
    }

    [Fact]
    public void TurboQuantPair_WithOneSideFlagged_IsDroppedWhole()
    {
        var tq = new TunedSettings { CacheTypeK = "tq4", CacheTypeV = "tq4", Paged = false, Threads = 8, TunedContext = 8192 };
        var a = Apply(Match(TuningMatchKind.Hit, tq), flags: ["cache_type_k"]);
        Assert.Null(a.Applied.CacheTypeV);
        Assert.Null(a.Applied.Paged);
        Assert.Equal(SettingSource.Default, a.Sources["cache_type_v"]);
        Assert.Equal(SettingSource.Default, a.Sources["paged"]);
        Assert.Equal(SettingSource.Flag, a.Sources["cache_type_k"]);
        Assert.Equal(SettingSource.Auto, a.Sources["threads"]);   // only the KV group is dropped
        Assert.Contains(a.Warnings, w => w.Contains("TurboQuant", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData(AutoCommand.Run, false, true)]
    [InlineData(AutoCommand.Chat, false, true)]
    [InlineData(AutoCommand.Bench, false, true)]
    [InlineData(AutoCommand.Serve, false, false)]   // multi-session serve: TurboQuant is ineligible (James 2026-11-24, #1893 lifts it)
    [InlineData(AutoCommand.Serve, true, true)]      // single-user serve may use it
    public void TurboQuant_IsAppliedOnlyForRunChatBenchOrSingleUserServe(AutoCommand command, bool singleUser, bool applied)
    {
        var tq = new TunedSettings { CacheTypeK = "tq4", CacheTypeV = "tq4", Paged = false, TunedContext = 8192 };
        var a = Apply(Match(TuningMatchKind.Hit, tq), policy: new AutoPolicy(command, singleUser));
        Assert.Equal(applied, a.Applied.CacheTypeK == "tq4" && a.Applied.CacheTypeV == "tq4" && a.Applied.Paged == false);
        if (!applied) Assert.Contains(a.Warnings, w => w.Contains("--no-paged", StringComparison.Ordinal));
    }

    [Fact]
    public void Miss_LeavesDefaults_AndAdvises()
    {
        var a = Apply(Match(TuningMatchKind.Miss, Saved, "no saved result"), flags: ["threads"]);
        Assert.All(a.Sources.Where(kv => kv.Key != "threads"), kv => Assert.Equal(SettingSource.Default, kv.Value));
        Assert.Equal(SettingSource.Flag, a.Sources["threads"]);
        Assert.Equal(new TunedSettings(), a.Applied);
        Assert.Contains(Tune, a.Advice);
        Assert.Contains("no saved result", a.Advice);
    }

    [Fact]
    public void Stale_Applies_AndAdvisesRecheck()
    {
        var a = Apply(Match(TuningMatchKind.Stale, Saved, "driver version changed (tuned with 1, now 2)"));
        Assert.Equal("q8_0", a.Applied.CacheTypeK);
        Assert.Equal(SettingSource.Auto, a.Sources["cache_type_k"]);
        Assert.Contains("driver version changed", a.Advice);
        Assert.Contains(Tune + " --recheck", a.Advice);
    }

    [Fact]
    public void Partial_Applies_AndAdvisesFinish()
    {
        var a = Apply(Match(TuningMatchKind.Partial, Saved, "the tuning run did not finish"));
        Assert.Equal(512, a.Applied.PrefillChunkSize);
        Assert.Contains("incomplete", a.Advice);
        Assert.DoesNotContain("--recheck", a.Advice);
        Assert.Contains(Tune, a.Advice);
    }

    [Fact]
    public void RefusedSetting_IsDropped_NotSubstituted()
    {
        var a = Apply(Match(TuningMatchKind.Hit, Saved), refuse: _ => "this build does not run that KV pair");
        Assert.Null(a.Applied.CacheTypeK);
        Assert.Null(a.Applied.CacheTypeV);                       // dropped, and nothing took its place
        Assert.Equal(SettingSource.Default, a.Sources["cache_type_k"]);
        Assert.Equal(SettingSource.Auto, a.Sources["threads"]);  // everything else still applies
        Assert.Contains(a.Warnings, w => w.Contains("does not run that KV pair", StringComparison.Ordinal));
    }

    [Fact]
    public void Refusal_IsNotAskedAboutAKvPairThatWasNotSaved()
    {
        int asked = 0;
        Apply(Match(TuningMatchKind.Hit, new TunedSettings { Threads = 4 }), refuse: _ => { asked++; return "x"; });
        Assert.Equal(0, asked);
    }

    [Fact]
    public void PlacementThatNoLongerFits_FallsBackToDefaults_WithBytes()
    {
        var a = Apply(Match(TuningMatchKind.Hit, Saved), flags: ["threads"],
                      admit: (_, _) => $"needs {AutoSettings.Format(10 * GiB)} of device memory, {AutoSettings.Format(6 * GiB)} free");
        Assert.All(a.Sources.Where(kv => kv.Key != "threads"), kv => Assert.Equal(SettingSource.Default, kv.Value));
        Assert.Equal(SettingSource.Flag, a.Sources["threads"]);
        Assert.Equal(new TunedSettings(), a.Applied);
        string w = Assert.Single(a.Warnings);
        Assert.Contains("10.00 GiB", w);
        Assert.Contains("6.00 GiB", w);
    }

    [Fact]
    public void Admission_IsAskedAtTheServedContext_FlagBeatingSaved()
    {
        var saved = Saved with { ContextSize = 16384, TunedContext = 16384 };
        var seen = new List<int>();
        PlacementAdmission spy = (_, ctx) => { seen.Add(ctx); return null; };
        AutoSettings.Apply(new HashSet<string>(), Match(TuningMatchKind.Hit, saved), Tune, 4096, spy, NoRefusal, Run);
        AutoSettings.Apply(new HashSet<string> { "ctx_size" }, Match(TuningMatchKind.Hit, saved), Tune, 4096, spy, NoRefusal, Run);
        Assert.Equal([16384, 4096], seen);
    }

    [Fact]
    public void ServedContextAboveTuned_IsNotAdmitted_DefaultsRun()
    {
        // The recorded peak was measured at the tuned context (8192); it does not bound 32768, so nothing may be admitted.
        int asked = 0;
        var a = Apply(Match(TuningMatchKind.Hit, Saved), ctx: 32768, admit: (_, _) => { asked++; return null; });
        Assert.Equal(new TunedSettings(), a.Applied);
        Assert.All(a.Sources, kv => Assert.Equal(SettingSource.Default, kv.Value));
        Assert.Contains(a.Warnings, w => w.Contains("above the tuned context 8192", StringComparison.Ordinal));
        Assert.Equal(0, asked);
        // At or below the tuned context it is admitted (the boundary is inclusive).
        Assert.Equal("q8_0", Apply(Match(TuningMatchKind.Hit, Saved), ctx: 8192).Applied.CacheTypeK);
        Assert.Empty(Apply(Match(TuningMatchKind.Hit, Saved), ctx: 4096).Warnings);
    }

    [Fact]
    public void Serve_WithUnknownServedContext_IsNotAdmitted_UnlessTheResultSavedOne()
    {
        // serve without --ctx-size serves the checkpoint window, which is not known here: the peak at 8192 does not bound it.
        var serve = new AutoPolicy(AutoCommand.Serve);
        var a = Apply(Match(TuningMatchKind.Hit, Saved), policy: serve, ctx: 0);
        Assert.Equal(new TunedSettings(), a.Applied);
        Assert.Contains(a.Warnings, w => w.Contains("not known before load", StringComparison.Ordinal));
        // a saved context within the tuned one makes the served context known, and it is admitted
        Assert.Equal("q8_0", Apply(Match(TuningMatchKind.Hit, Saved with { ContextSize = 8192 }), policy: serve, ctx: 0).Applied.CacheTypeK);
        // run sizes its own cache from prompt + max tokens: an unknown (0) context is admitted there
        Assert.Equal("q8_0", Apply(Match(TuningMatchKind.Hit, Saved), ctx: 0).Applied.CacheTypeK);
    }

    [Fact]
    public void ResultWithoutTunedContext_IsNotAdmitted()
    {
        var a = Apply(Match(TuningMatchKind.Hit, Saved with { TunedContext = null }), ctx: 0);
        Assert.Equal(new TunedSettings(), a.Applied);
        Assert.Contains(a.Warnings, w => w.Contains("does not record the context", StringComparison.Ordinal));
    }

    [Fact]
    public void NothingToApply_DoesNotAskAdmission()
    {
        int asked = 0;
        var a = Apply(Match(TuningMatchKind.Hit, new TunedSettings { Threads = 8 }), flags: ["threads"], admit: (_, _) => { asked++; return "x"; });
        Assert.Equal(0, asked);
        Assert.Empty(a.Warnings);
    }

    [Fact]
    public void UnsupportedSettings_AreIgnored_NotReportedAsAuto()
    {
        var supported = new HashSet<string> { "threads" };
        var a = AutoSettings.Apply(new HashSet<string>(), Match(TuningMatchKind.Hit, Saved), Tune, 0, Fits, NoRefusal, Run, supported);
        Assert.Equal(8, a.Applied.Threads);
        Assert.Null(a.Applied.CacheTypeK);
        Assert.Equal(SettingSource.Default, a.Sources["cache_type_k"]);
    }

    [Theory]
    [InlineData("--kv-quant-k", "cache_type_k")]
    [InlineData("--cache-type-v=q8_0", "cache_type_v")]
    [InlineData("--no-mtp", "mtp")]
    [InlineData("--mtp", "mtp")]
    [InlineData("--ubatch-size", "prefill_chunk_size")]
    [InlineData("-d", "device")]
    [InlineData("--n-cpu-moe", "moe_offload")]
    [InlineData("--no-paged", "paged")]
    [InlineData("--draft-tokens", "speculative_k")]
    public void ExplicitFlags_NormalisesSynonyms(string arg, string name)
    {
        Assert.Equal([name], AutoSettings.ExplicitFlags(["run", "m.gguf", arg, "x"]));
    }

    [Fact]
    public void ExplicitFlags_AutoPullAndAutoMapToNothing()
    {
        Assert.Empty(AutoSettings.ExplicitFlags(["serve", "--auto-pull", "--auto", "--auto-profile", "chat", "model.gguf"]));
    }
}
