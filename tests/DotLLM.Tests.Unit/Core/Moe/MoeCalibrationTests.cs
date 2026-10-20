using DotLLM.Core.Moe;
using Xunit;

namespace DotLLM.Tests.Unit.Moe;

/// <summary>
/// Issue #1071: the calibration cache key and store (<see cref="MoeCalibrationKey"/>, <see cref="MoeCalibrationStore"/>), the
/// record-to-cost-model fit, and the cold-expert CLI options. CPU only (temp files).
/// </summary>
public sealed class MoeCalibrationTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"dotllm-moe-cal-{Guid.NewGuid():N}");

    public MoeCalibrationTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort */ }
    }

    private static MoeCalibrationRecord Record(double h2d = 5.4e9) => new(h2d, 51,
        [new MoeLayoutTiming(1_000_000, 1_200_000, 800, 2200), new MoeLayoutTiming(400_000, 410_000, 300, 650)],
        new DateTime(2026, 10, 20, 12, 0, 0, DateTimeKind.Utc));

    [Fact]
    public void Key_ChangesWithEveryComponent_AndWithTheModelFilesSizeAndMtime()
    {
        string model = Path.Combine(_dir, "m.gguf");
        File.WriteAllBytes(model, new byte[100]);
        var k = MoeCalibrationKey.ForFile("RTX 3060", "0000:03:00.0", model, 15);
        Assert.Equal(Path.GetFullPath(model), k.ModelPath);
        Assert.Equal(100, k.ModelBytes);
        Assert.NotEqual(0, k.ModelMtimeUtcTicks);
        Assert.False(string.IsNullOrWhiteSpace(k.Cpu));

        Assert.NotEqual(k.ToString(), (k with { Gpu = "RTX 4090" }).ToString());
        Assert.NotEqual(k.ToString(), (k with { PciLink = "0000:04:00.0" }).ToString());
        Assert.NotEqual(k.ToString(), (k with { CpuThreads = 7 }).ToString());
        Assert.NotEqual(k.ToString(), (k with { Cpu = "other" }).ToString());

        File.WriteAllBytes(model, new byte[101]);
        Assert.NotEqual(k.ToString(), MoeCalibrationKey.ForFile("RTX 3060", "0000:03:00.0", model, 15).ToString());
        File.SetLastWriteTimeUtc(model, new DateTime(2020, 1, 1, 0, 0, 0, DateTimeKind.Utc));
        var k2 = MoeCalibrationKey.ForFile("RTX 3060", "0000:03:00.0", model, 15);
        File.SetLastWriteTimeUtc(model, new DateTime(2021, 1, 1, 0, 0, 0, DateTimeKind.Utc));
        Assert.NotEqual(k2.ToString(), MoeCalibrationKey.ForFile("RTX 3060", "0000:03:00.0", model, 15).ToString());

        // An unknown model path still makes a key (no file part).
        Assert.Equal(0, MoeCalibrationKey.ForFile("g", "p", null, 1).ModelBytes);
    }

    [Fact]
    public void Store_RoundTrips_KeepsOtherEntries_AndMissesCleanly()
    {
        string path = Path.Combine(_dir, "sub", "moe-calibration.json");
        var a = new MoeCalibrationKey("gpuA", "bus", "cpu", "a.gguf", 1, 2, 3);
        var b = a with { ModelPath = "b.gguf" };
        Assert.Null(MoeCalibrationStore.TryLoad(path, a));

        Assert.True(MoeCalibrationStore.TrySave(path, a, Record()));
        Assert.True(MoeCalibrationStore.TrySave(path, b, Record(h2d: 3.1e9)));
        var ra = MoeCalibrationStore.TryLoad(path, a)!;
        var rb = MoeCalibrationStore.TryLoad(path, b)!;
        Assert.Equal(5.4e9, ra.H2DBytesPerSecond);
        Assert.Equal(3.1e9, rb.H2DBytesPerSecond);
        Assert.Equal(Record().Layouts, ra.Layouts);
        Assert.Equal(Record().MeasuredUtc, ra.MeasuredUtc);
        Assert.Null(MoeCalibrationStore.TryLoad(path, a with { CpuThreads = 4 }));

        // A corrupt file is a cache miss on read and is replaced on write.
        File.WriteAllText(path, "{ not json");
        Assert.Null(MoeCalibrationStore.TryLoad(path, a));
        Assert.True(MoeCalibrationStore.TrySave(path, a, Record()));
        Assert.NotNull(MoeCalibrationStore.TryLoad(path, a));
    }

    [Fact]
    public void Record_FitsTheCostModel_AndReportsCoverage()
    {
        var r = Record();
        var m = r.ToCostModel();
        Assert.Equal(5.4e9, m.H2DBytesPerSecond);
        Assert.Equal(51, m.SyncRoundTripUs);
        var big = m.For(new ExpertLayout(1_000_000, 1_200_000))!;
        Assert.Equal(800, big.Cpu.Us(1), 9);
        Assert.Equal(2200, big.Cpu.Us(8), 9);
        Assert.Equal(MoeCostModel.DefaultGpuCost(2_200_000), big.Gpu);

        Assert.True(r.Covers([new ExpertLayout(400_000, 410_000)]));
        Assert.False(r.Covers([new ExpertLayout(400_000, 410_000), new ExpertLayout(1, 2)]));
        Assert.Contains("5.40 GB/s", r.Describe(), StringComparison.Ordinal);
    }

    [Fact]
    public void Options_ColdModesThreadsStaticLayersAndContext_FromCliAndEnvironment()
    {
        var env = new Dictionary<string, string>(StringComparer.Ordinal);
        string? Get(string k) => env.TryGetValue(k, out var v) ? v : null;

        var d = MoeOffloadOptions.FromCli(new MoeCliValues(), getEnv: Get);
        Assert.Equal(MoeColdMode.Auto, d.Cold);
        Assert.Equal(0, d.CpuMoeLayers);
        Assert.Null(d.CpuThreads);
        Assert.Null(d.ContextTokens);
        Assert.False(d.Calibrate);
        Assert.True(d.MayUseCpuLeg);
        Assert.False((d with { Cold = MoeColdMode.Copy }).MayUseCpuLeg);
        Assert.True((d with { Cold = MoeColdMode.Copy, CpuMoeLayers = 2 }).MayUseCpuLeg);

        var cli = MoeOffloadOptions.FromCli(new MoeCliValues(Cold: "cpu", CpuThreads: 6, NCpuMoe: 3, Calibrate: true, ContextTokens: 8192),
            getEnv: Get);
        Assert.Equal(MoeColdMode.Cpu, cli.Cold);
        Assert.Equal(6, cli.EffectiveCpuThreads);
        Assert.Equal(3, cli.CpuMoeLayers);
        Assert.True(cli.IsStaticCpuLayer(2));
        Assert.False(cli.IsStaticCpuLayer(3));
        Assert.True(cli.Calibrate);
        Assert.Equal(8192, cli.ContextTokens);

        Assert.Equal(MoeOffloadOptions.AllLayers, MoeOffloadOptions.FromCli(new MoeCliValues(CpuMoe: true, NCpuMoe: 2), getEnv: Get).CpuMoeLayers);

        env[MoeOffloadOptions.ColdEnvVar] = "copy";
        env[MoeOffloadOptions.CpuThreadsEnvVar] = "3";
        env[MoeOffloadOptions.NCpuMoeEnvVar] = "5";
        env[MoeOffloadOptions.CalibrateEnvVar] = "1";
        env[MoeOffloadOptions.ContextEnvVar] = "4096";
        var e = MoeOffloadOptions.FromCli(new MoeCliValues(), getEnv: Get);
        Assert.Equal(MoeColdMode.Copy, e.Cold);
        Assert.Equal(3, e.CpuThreads);
        Assert.Equal(5, e.CpuMoeLayers);
        Assert.True(e.Calibrate);
        Assert.Equal(4096, e.ContextTokens);
        Assert.Equal(MoeColdMode.Cpu, MoeOffloadOptions.FromCli(new MoeCliValues(Cold: "CPU"), getEnv: Get).Cold);   // CLI wins
        Assert.Equal(0, MoeOffloadOptions.FromCli(new MoeCliValues(NCpuMoe: 0), getEnv: Get).CpuMoeLayers);
        env[MoeOffloadOptions.CpuMoeEnvVar] = "1";
        Assert.Equal(MoeOffloadOptions.AllLayers, MoeOffloadOptions.ResolveCpuMoeLayers(false, null, Get));

        var lib = MoeOffloadOptions.WithEnvironmentFallback(null, Get);
        Assert.Equal(MoeColdMode.Copy, lib.Cold);
        Assert.Equal(MoeOffloadOptions.AllLayers, lib.CpuMoeLayers);
        Assert.Equal(4096, lib.ContextTokens);
    }

    [Theory]
    [InlineData("warm", null, null, false, null)]
    [InlineData(null, 0, null, false, null)]
    [InlineData(null, null, -1, false, null)]
    [InlineData(null, null, null, false, 0)]
    [InlineData("off", null, null, true, null)]   // --cpu-moe with --moe-offload off
    public void Options_Validate_RejectsBadColdValues(string? coldOrMode, int? threads, int? nCpuMoe, bool cpuMoe, int? ctx)
    {
        var v = string.Equals(coldOrMode, "off", StringComparison.Ordinal)
            ? new MoeCliValues(Mode: "off", CpuMoe: cpuMoe)
            : new MoeCliValues(Cold: coldOrMode, CpuThreads: threads, NCpuMoe: nCpuMoe, CpuMoe: cpuMoe, ContextTokens: ctx);
        Assert.NotNull(MoeOffloadOptions.Validate(v, _ => null));
    }

    [Fact]
    public void Options_Validate_FlagsABadEnvironmentFallback()
    {
        Assert.NotNull(MoeOffloadOptions.Validate(new MoeCliValues(), k => string.Equals(k, MoeOffloadOptions.ColdEnvVar, StringComparison.Ordinal) ? "igpu" : null));
        Assert.NotNull(MoeOffloadOptions.Validate(new MoeCliValues(), k => string.Equals(k, MoeOffloadOptions.NCpuMoeEnvVar, StringComparison.Ordinal) ? "x" : null));
        Assert.Null(MoeOffloadOptions.Validate(new MoeCliValues(Cold: "auto", CpuThreads: 1, NCpuMoe: 0, ContextTokens: 1), _ => null));
    }

    [Fact]
    public void TelemetrySummary_AppendsTheColdLeg_WhenTheCpuRan()
    {
        var s = new MoeTelemetrySnapshot(Hits: 50, Misses: 50, Evictions: 0, BytesH2D: 0, CopyWaitMicros: 0, Tokens: 10,
            CacheSlots: 10, CacheBytes: 1L << 30, Policy: ExpertCachePolicy.Slru, Routed: 100, ColdExpertsCpu: 41,
            CpuComputeMicros: 50_000, CpuWaitMicros: 32_000, Promotions: 3);
        Assert.EndsWith("; cold: CPU 41% of routed, CPU compute 5.0 ms/token, CPU wait 3.2 ms/token, promotions 3", s.FormatSummary(),
            StringComparison.Ordinal);
        Assert.DoesNotContain("cold:", (s with { ColdExpertsCpu = 0, Promotions = 0 }).FormatSummary(), StringComparison.Ordinal);
        Assert.Equal(1, s.Since(s with { ColdExpertsCpu = 40 }).ColdExpertsCpu);
    }
}
