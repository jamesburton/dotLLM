using System.Security.Cryptography;
using DotLLM.Server;
using DotLLM.Server.Models;
using DotLLM.Server.Tuning;
using Xunit;

namespace DotLLM.Tests.Unit.Server.Tuning;

/// <summary>Serialises tests that set <c>DOTLLM_HASH_CACHE_DIR</c> or <c>DOTLLM_TUNING_DIR</c> (process-wide).</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class TuningEnvCollection
{
    /// <summary>Collection name.</summary>
    public const string Name = "TuningEnv";
}

/// <summary>The key components of a saved tuning result (#1902): model hash, device fingerprint and scope, perf identity, environment overrides.</summary>
[Collection(TuningEnvCollection.Name)]
public sealed class TuningKeyTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-tuningkey-" + Guid.NewGuid().ToString("N"));
    private readonly string? _oldHashDir = Environment.GetEnvironmentVariable("DOTLLM_HASH_CACHE_DIR");

    public TuningKeyTests()
    {
        Directory.CreateDirectory(_dir);
        Environment.SetEnvironmentVariable("DOTLLM_HASH_CACHE_DIR", Path.Combine(_dir, "hash-cache"));
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable("DOTLLM_HASH_CACHE_DIR", _oldHashDir);
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static string Hex(byte[] b) => Convert.ToHexString(SHA256.HashData(b)).ToLowerInvariant();

    [Fact]
    public void GgufHash_IsContentSha256_AndFollowsReplacement()
    {
        string f = Path.Combine(_dir, "m.gguf");
        byte[] a = [1, 2, 3, 4, 5];
        File.WriteAllBytes(f, a);
        Assert.Equal(Hex(a), ModelHash.Of(f));

        byte[] b = [9, 9, 9, 9, 9, 9];   // replaced in place: different content, different length, new mtime
        File.WriteAllBytes(f, b);
        File.SetLastWriteTimeUtc(f, DateTime.UtcNow.AddMinutes(5));
        Assert.Equal(Hex(b), ModelHash.Of(f));
    }

    [Fact]
    public void HubBlob_HashesAsBlobId_WithoutReadingTheFile()
    {
        // A fake hub cache: blobs/<sha> plus a snapshot copy that is a "hardlink" in everything the layout can tell (same length, same mtime).
        // The content does NOT hash to the blob name, so a content hash would give a different answer: the test discriminates.
        string repo = Path.Combine(_dir, "hub", "models--org--name");
        string blobs = Path.Combine(repo, "blobs");
        string snap = Path.Combine(repo, "snapshots", "abc123");
        Directory.CreateDirectory(blobs);
        Directory.CreateDirectory(snap);
        string id = new string('a', 64);
        byte[] content = [7, 7, 7, 7];
        string blob = Path.Combine(blobs, id);
        File.WriteAllBytes(blob, content);
        string entry = Path.Combine(snap, "model.gguf");
        File.WriteAllBytes(entry, content);
        File.SetLastWriteTimeUtc(entry, File.GetLastWriteTimeUtc(blob));

        Assert.NotEqual("hf:" + id, Hex(content));
        Assert.Equal("hf:" + id, ModelHash.Of(blob));
        Assert.Equal("hf:" + id, ModelHash.Of(entry));
    }

    [Fact]
    public void SafetensorsDirectory_HashIsOrderIndependentAndContentSensitive()
    {
        string d = Path.Combine(_dir, "st");
        Directory.CreateDirectory(d);
        File.WriteAllBytes(Path.Combine(d, "model-00001.safetensors"), [1, 1]);
        File.WriteAllBytes(Path.Combine(d, "model-00002.safetensors"), [2, 2]);
        File.WriteAllText(Path.Combine(d, "config.json"), "{}");
        File.WriteAllText(Path.Combine(d, "README.md"), "ignored");
        string h1 = ModelHash.Of(d);

        File.WriteAllText(Path.Combine(d, "README.md"), "still ignored");
        Assert.Equal(h1, ModelHash.Of(d));

        File.WriteAllBytes(Path.Combine(d, "model-00002.safetensors"), [2, 3]);
        File.SetLastWriteTimeUtc(Path.Combine(d, "model-00002.safetensors"), DateTime.UtcNow.AddMinutes(9));
        Assert.NotEqual(h1, ModelHash.Of(d));
    }

    [Fact]
    public void Modifiers_NullWhenNone_AndSensitiveToEachKind()
    {
        Assert.Null(ModelHash.Modifiers(null, null, null, null));
        Assert.Null(ModelHash.Modifiers(null, [], null, null));

        string f = Path.Combine(_dir, "x.bin");
        File.WriteAllBytes(f, [1, 2, 3]);
        string? overlay = ModelHash.Modifiers(f, null, null, null);
        string? ternary = ModelHash.Modifiers(null, [f], null, null);
        string? mtp = ModelHash.Modifiers(null, null, f, null);
        string? dflash = ModelHash.Modifiers(null, null, null, f);
        Assert.All(new[] { overlay, ternary, mtp, dflash }, m => Assert.NotNull(m));
        Assert.Equal(4, new[] { overlay, ternary, mtp, dflash }.Distinct().Count());   // the kind is part of the digest, not only the content
    }

    private static DeviceFingerprint Gpu(uint deviceId = 0x1586, string? driver = "1.0") => new()
    {
        Backend = "vulkan", GpuName = "AMD Radeon 8060S", PciVendorId = 0x1002, PciDeviceId = deviceId, DriverVersion = driver,
    };

    [Fact]
    public void Scope_ExcludesDriverVersion()
    {
        var spec = DeviceSpec.Parse("vulkan:0");
        Assert.Equal(TuningKeys.ScopeFor(spec, [Gpu(driver: "1.0")]), TuningKeys.ScopeFor(spec, [Gpu(driver: "2.0")]));
    }

    [Fact]
    public void Scope_DistinguishesTwoGpus()
    {
        var spec = DeviceSpec.Parse("vulkan:0");
        Assert.NotEqual(TuningKeys.ScopeFor(spec, [Gpu(deviceId: 0x1586)]), TuningKeys.ScopeFor(spec, [Gpu(deviceId: 0x1587)]));
    }

    [Fact]
    public void Scope_AutoDiffersFromExplicit()
    {
        var one = Gpu();
        string explicitScope = TuningKeys.ScopeFor(DeviceSpec.Parse("vulkan:0"), [one]);
        string autoScope = TuningKeys.ScopeFor(DeviceSpec.Parse("auto"), [DeviceFingerprints.Cpu(), one]);
        Assert.StartsWith("auto-", autoScope);
        Assert.NotEqual(explicitScope, autoScope);
        // and a different machine (a device missing) is a different auto scope
        Assert.NotEqual(autoScope, TuningKeys.ScopeFor(DeviceSpec.Parse("auto"), [DeviceFingerprints.Cpu()]));
    }

    [Fact]
    public void For_FingerprintsTheDeviceActuallyUsed_NotDeviceZero()
    {
        var list = new DeviceListResponse
        {
            Backends =
            [
                new BackendInfoDto
                {
                    Name = "vulkan", Available = true, DeviceCount = 2, Servable = true,
                    Devices =
                    [
                        new DeviceInfoDto { Index = 0, Name = "NVIDIA RTX 3060", PciVendorId = 0x10DE, PciDeviceId = 0x2487, TotalMemoryBytes = 12L << 30 },
                        new DeviceInfoDto { Index = 1, Name = "AMD Radeon 8060S", PciVendorId = 0x1002, PciDeviceId = 0x1586, TotalMemoryBytes = 64L << 30,
                                            IsIntegrated = true, IsDefault = true },
                    ],
                },
            ],
        };
        var one = DeviceFingerprints.For("vulkan:1", list)!;
        Assert.Equal("AMD Radeon 8060S", one.GpuName);
        Assert.Null(one.DeviceMemoryBytes);                       // UMA: the limit drifts, so it stays out of the key
        Assert.Equal("NVIDIA RTX 3060", DeviceFingerprints.For("vulkan:0", list)!.GpuName);
        Assert.Equal(12L << 30, DeviceFingerprints.For("vulkan:0", list)!.DeviceMemoryBytes);
        Assert.Equal("AMD Radeon 8060S", DeviceFingerprints.For("vulkan", list)!.GpuName);   // plain vulkan = the default device
    }

    [Fact]
    public void KernelFilesHash_ChangesWhenAShaderChanges()
    {
        string root = Path.Combine(_dir, "bin");
        Directory.CreateDirectory(Path.Combine(root, "spv"));
        File.WriteAllBytes(Path.Combine(root, "spv", "gemv.spv"), [1, 2, 3]);
        File.WriteAllText(Path.Combine(root, "notes.txt"), "not a kernel");
        string h1 = PerfIdentities.Current(root).KernelFilesHash!;
        File.WriteAllText(Path.Combine(root, "notes.txt"), "still not a kernel");
        Assert.Equal(h1, PerfIdentities.Current(root).KernelFilesHash);
        File.WriteAllBytes(Path.Combine(root, "spv", "gemv.spv"), [1, 2, 4]);
        Assert.NotEqual(h1, PerfIdentities.Current(root).KernelFilesHash);
    }

    [Fact]
    public void PerfIdentity_HasReleaseRuntimeAndAssemblyIdentity()
    {
        var p = PerfIdentities.Current();
        Assert.False(string.IsNullOrEmpty(p.ReleaseVersion));
        Assert.DoesNotContain('+', p.ReleaseVersion!);
        Assert.Equal(Environment.Version.ToString(), p.RuntimeVersion);
        Assert.Equal(64, p.KernelAssembliesHash!.Length);
    }

    [Fact]
    public void EnvOverrides_IgnoresAllowListed_AndIsOrderIndependent()
    {
        KeyValuePair<string, string>[] vars =
        [
            new("DOTLLM_TUNING_DIR", "C:\\x"), new("DOTLLM_CUDA_STEP_GRAPH", "0"), new("PATH", "p"), new("DOTLLM_QWEN_GGUF", "m.gguf"),
        ];
        var (h1, pairs) = EnvOverrides.Current(() => vars);
        Assert.Equal(["DOTLLM_CUDA_STEP_GRAPH=0"], pairs);
        var (h2, _) = EnvOverrides.Current(() => vars.Reverse());
        Assert.Equal(h1, h2);
        var (h3, _) = EnvOverrides.Current(() => [new("DOTLLM_CUDA_STEP_GRAPH", "1")]);
        Assert.NotEqual(h1, h3);
        Assert.Equal((null, null), EnvOverrides.Current(() => [new("DOTLLM_TUNING_DIR", "x")]));
    }
}
