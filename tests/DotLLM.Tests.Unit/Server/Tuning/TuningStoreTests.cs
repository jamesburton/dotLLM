using DotLLM.Server;
using DotLLM.Server.Tuning;
using Xunit;

namespace DotLLM.Tests.Unit.Server.Tuning;

/// <summary>The store: save, checksum, match (#1902).</summary>
public sealed class TuningStoreTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "dotllm-tuningstore-" + Guid.NewGuid().ToString("N"));
    private TuningStore Store => new(_root);

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
    }

    internal static TuningKey Key(string scope = "vulkan-10021586-amd-radeon-8060s", string profile = "chat") => new()
    {
        SchemaVersion = TuningStore.SchemaVersion,
        ModelHash = "a1b2c3",
        ModelModifiers = null,
        Scope = scope,
        Devices =
        [
            new DeviceFingerprint
            {
                Backend = "vulkan", GpuName = "AMD Radeon 8060S", PciVendorId = 0x1002, PciDeviceId = 0x1586, DeviceMemoryBytes = null,
                DriverVersion = "100", DriverId = "amd", DriverInfo = "info", ApiVersion = "1.3.0", OsBuild = "win11",
            },
        ],
        Profile = profile,
        ProfileHash = "ph",
        KvQualityBudget = 0.01,
        Perf = new PerfIdentity { ReleaseVersion = "0.3.0", KernelFilesHash = "kf", KernelAssembliesHash = "ka", RuntimeVersion = "10.0.1" },
        EnvOverridesHash = null,
        EnvOverrides = null,
        ProtocolVersion = TuningStore.ProtocolVersion,
    };

    internal static TuningResult Result(TuningKey key, bool partial = false) => new()
    {
        Key = key,
        Settings = new TunedSettings { CacheTypeK = "q8_0", CacheTypeV = "q8_0", PrefillChunkSize = 512, Threads = 8 },
        Partial = partial ? true : null,
        TunedAtUtc = "2026-11-24T00:00:00Z",
        Report = "r",
    };

    [Fact]
    public void SaveThenFind_IsHit()
    {
        var key = Key();
        Store.Save(Result(key));
        var m = Store.Find(key);
        Assert.Equal(TuningMatchKind.Hit, m.Kind);
        Assert.Equal("q8_0", m.Result!.Settings!.CacheTypeK);
        Assert.Equal(512, m.Result.Settings.PrefillChunkSize);
    }

    [Fact]
    public void NoFile_IsMiss()
    {
        Assert.Equal(TuningMatchKind.Miss, Store.Find(Key()).Kind);
    }

    public static IEnumerable<object[]> ContentFieldChanges()
    {
        // Fields that do NOT change the file path: the saved result is read back and the comparison alone decides.
        yield return Row("schema_version", TuningMatchKind.Miss, k => k with { SchemaVersion = 99 });
        yield return Row("model_modifiers", TuningMatchKind.Miss, k => k with { ModelModifiers = "overlay" });
        yield return Row("device.backend", TuningMatchKind.Miss, k => k with { Devices = [k.Devices![0] with { Backend = "cuda" }] });
        yield return Row("device.gpu_name", TuningMatchKind.Miss, k => k with { Devices = [k.Devices![0] with { GpuName = "x" }] });
        yield return Row("device.vendor", TuningMatchKind.Miss, k => k with { Devices = [k.Devices![0] with { PciVendorId = 1 }] });
        yield return Row("device.device_id", TuningMatchKind.Miss, k => k with { Devices = [k.Devices![0] with { PciDeviceId = 1 }] });
        yield return Row("device.memory", TuningMatchKind.Miss, k => k with { Devices = [k.Devices![0] with { DeviceMemoryBytes = 5 }] });
        yield return Row("device.cpu_model", TuningMatchKind.Miss, k => k with { Devices = [k.Devices![0] with { CpuModel = "x" }] });
        yield return Row("device.ram", TuningMatchKind.Miss, k => k with { Devices = [k.Devices![0] with { RamBytes = 7 }] });
        yield return Row("device.count", TuningMatchKind.Miss, k => k with { Devices = [k.Devices![0], k.Devices[0]] });
        yield return Row("profile_hash", TuningMatchKind.Miss, k => k with { ProfileHash = "changed" });
        yield return Row("kv_budget", TuningMatchKind.Miss, k => k with { KvQualityBudget = 0.02 });
        yield return Row("driver_version", TuningMatchKind.Stale, k => k with { Devices = [k.Devices![0] with { DriverVersion = "101" }] });
        yield return Row("driver_id", TuningMatchKind.Stale, k => k with { Devices = [k.Devices![0] with { DriverId = "mesa" }] });
        yield return Row("driver_info", TuningMatchKind.Stale, k => k with { Devices = [k.Devices![0] with { DriverInfo = "other" }] });
        yield return Row("api_version", TuningMatchKind.Stale, k => k with { Devices = [k.Devices![0] with { ApiVersion = "1.4.0" }] });
        yield return Row("os_build", TuningMatchKind.Stale, k => k with { Devices = [k.Devices![0] with { OsBuild = "win12" }] });
        yield return Row("release", TuningMatchKind.Stale, k => k with { Perf = k.Perf! with { ReleaseVersion = "0.4.0" } });
        yield return Row("kernel_files", TuningMatchKind.Stale, k => k with { Perf = k.Perf! with { KernelFilesHash = "x" } });
        yield return Row("kernel_assemblies", TuningMatchKind.Stale, k => k with { Perf = k.Perf! with { KernelAssembliesHash = "x" } });
        yield return Row("runtime", TuningMatchKind.Stale, k => k with { Perf = k.Perf! with { RuntimeVersion = "11.0.0" } });
        yield return Row("env", TuningMatchKind.Stale, k => k with { EnvOverridesHash = "h", EnvOverrides = ["DOTLLM_X=1"] });
        yield return Row("protocol", TuningMatchKind.Stale, k => k with { ProtocolVersion = 2 });
    }

    private static object[] Row(string name, TuningMatchKind kind, Func<TuningKey, TuningKey> mutate) => [name, kind, mutate];

    [Theory]
    [MemberData(nameof(ContentFieldChanges))]
    public void EachKeyFieldChange_GivesItsOutcome(string name, TuningMatchKind expected, Func<TuningKey, TuningKey> mutate)
    {
        var saved = Key();
        Store.Save(Result(saved));
        var current = mutate(saved);
        Assert.Equal(Store.PathFor(saved), Store.PathFor(current));   // the file is found, so only the comparison can classify the change
        var m = Store.Find(current);
        Assert.True(expected == m.Kind, $"{name}: expected {expected}, got {m.Kind} ({m.Reason})");
    }

    [Fact]
    public void PathFields_AddressAnotherFile_SoModelScopeAndProfileMiss()
    {
        var saved = Key();
        Store.Save(Result(saved));
        foreach (var other in new[] { saved with { ModelHash = "other" }, saved with { Scope = "other-scope" }, saved with { Profile = "agent" } })
        {
            Assert.NotEqual(Store.PathFor(saved), Store.PathFor(other));
            Assert.Equal(TuningMatchKind.Miss, Store.Find(other).Kind);
        }
    }

    [Fact]
    public void ModelScopeAndProfileAreAlsoComparedInContent_NotOnlyInThePath()
    {
        // A file copied or renamed into another key's path must not be accepted: the content comparison is the second line of defence.
        var saved = Key();
        var other = saved with { ModelHash = "other" };
        Store.Save(Result(saved));
        string dest = Store.PathFor(other);
        Directory.CreateDirectory(Path.GetDirectoryName(dest)!);
        File.Copy(Store.PathFor(saved), dest);
        Assert.Equal(TuningMatchKind.Miss, Store.Find(other).Kind);
        string destScope = Store.PathFor(saved with { Scope = "other-scope" });
        File.Copy(Store.PathFor(saved), destScope);
        Assert.Equal(TuningMatchKind.Miss, Store.Find(saved with { Scope = "other-scope" }).Kind);
    }

    [Fact]
    public void ExplicitDeviceResult_IsNotFoundUnderAutoScope_AndReverse()
    {
        var explicitKey = Key(scope: "vulkan-10021586-amd-radeon-8060s");
        var autoKey = Key(scope: "auto-0123456789ab");
        Store.Save(Result(explicitKey));
        Assert.Equal(TuningMatchKind.Miss, Store.Find(autoKey).Kind);
        Store.Save(Result(autoKey));
        Assert.Equal(TuningMatchKind.Hit, Store.Find(autoKey).Kind);
        Assert.Equal(TuningMatchKind.Hit, Store.Find(explicitKey).Kind);   // each in its own file: neither overwrote the other
        Assert.NotEqual(Store.PathFor(explicitKey), Store.PathFor(autoKey));
    }

    [Fact]
    public void EnvOverrideChange_IsStale_AndReasonNamesTheVariable()
    {
        var saved = Key() with { EnvOverridesHash = "h1", EnvOverrides = ["DOTLLM_CUDA_STEP_GRAPH=1"] };
        Store.Save(Result(saved));
        var now = saved with { EnvOverridesHash = "h2", EnvOverrides = ["DOTLLM_CUDA_STEP_GRAPH=0"] };
        var m = Store.Find(now);
        Assert.Equal(TuningMatchKind.Stale, m.Kind);
        Assert.Contains("DOTLLM_CUDA_STEP_GRAPH", m.Reason);

        // set now, unset at tune time
        var unsetThen = Key();
        Store.Save(Result(unsetThen));
        var m2 = Store.Find(unsetThen with { EnvOverridesHash = "h", EnvOverrides = ["DOTLLM_CUDA_STEP_GRAPH=0"] });
        Assert.Equal(TuningMatchKind.Stale, m2.Kind);
        Assert.Contains("DOTLLM_CUDA_STEP_GRAPH", m2.Reason);
        Assert.Contains("unset", m2.Reason);
    }

    [Fact]
    public void TamperedSettings_ChecksumFails_IsMiss_NoThrow()
    {
        var key = Key();
        Store.Save(Result(key));
        string path = Store.PathFor(key);
        string text = File.ReadAllText(path);
        Assert.Contains("\"q8_0\"", text);
        File.WriteAllText(path, text.Replace("\"q8_0\"", "\"q4_0\""));   // valid shape, wrong checksum
        var m = Store.Find(key);
        Assert.Equal(TuningMatchKind.Miss, m.Kind);
        Assert.Contains("checksum", m.Reason);
        Assert.Null(m.Result);
    }

    [Fact]
    public void CorruptJson_IsMiss_NoThrow()
    {
        var key = Key();
        Store.Save(Result(key));
        foreach (string junk in new[] { "{ this is not json", "", "null", "[]" })
        {
            File.WriteAllText(Store.PathFor(key), junk);
            Assert.Equal(TuningMatchKind.Miss, Store.Find(key).Kind);
        }
    }

    [Fact]
    public void PartialFlag_IsPartial_AndPartialPlusStale_IsStaleNamingBoth()
    {
        var key = Key();
        Store.Save(Result(key, partial: true));
        Assert.Equal(TuningMatchKind.Partial, Store.Find(key).Kind);
        var m = Store.Find(key with { ProtocolVersion = 5 });
        Assert.Equal(TuningMatchKind.Stale, m.Kind);
        Assert.Contains("incomplete", m.Reason);
    }

    [Fact]
    public void Checksum_CoversEverythingButItself()
    {
        var r = Result(Key());
        string c = TuningStore.ComputeChecksum(r);
        Assert.Equal(c, TuningStore.ComputeChecksum(r with { Checksum = "anything" }));
        Assert.NotEqual(c, TuningStore.ComputeChecksum(r with { Report = "other" }));
        Assert.NotEqual(c, TuningStore.ComputeChecksum(r with { Settings = r.Settings! with { Threads = 9 } }));
    }
}
