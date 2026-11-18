using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Models.Gguf;
using DotLLM.Server;
using DotLLM.Tests.Unit.Cli;
using DotLLM.Tests.Unit.Engine;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>#1888 final-review fixes: MLA hosts, the GPU hint, default-bit equivalence, the post-load call site and bench.</summary>
public sealed class ServerTurboQuantReviewFixTests : IClassFixture<PrefixCacheFailedRequestTests.TinyQwen>, IDisposable
{
    private readonly PrefixCacheFailedRequestTests.TinyQwen _tiny;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-1888-" + Guid.NewGuid().ToString("N"));

    public ServerTurboQuantReviewFixTests(PrefixCacheFailedRequestTests.TinyQwen tiny)
    {
        _tiny = tiny;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    private static readonly KvCacheConfig Tq4 = new(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, TurboQuantBits: 4);

    private static ServerOptions Options(string path, string? k, string? v, string device = "cpu", bool paged = false) => new()
    {
        Model = path, ModelId = "x-1888", Device = device, UsePaged = paged, Warmup = WarmupOptions.Disabled, CacheTypeK = k, CacheTypeV = v,
    };

    [Fact]
    public void MlaModel_Refuses_EvenWithAPowerOfTwoHeadDim()
    {
        // C1: TransformerModel ignores the caller's IKvCache on MLA layers, so a TurboQuant cache would be allocated and never used.
        // GLM-4.7-Flash-shaped: qk_nope 192 + qk_rope 64 = head dim 256, a power of two the geometry rule accepts.
        var config = new ModelConfig
        {
            Architecture = Architecture.DeepSeekV2, VocabSize = 1000, HiddenSize = 1024, IntermediateSize = 2048,
            NumLayers = 4, NumAttentionHeads = 16, NumKvHeads = 16, HeadDim = 256, MaxSequenceLength = 2048,
            MlaConfig = new MlaConfig { KvLoraRank = 512, QkNopeHeadDim = 192, QkRopeHeadDim = 64, VHeadDim = 256 },
        };
        string? reason = TurboQuantKvSupport.RefusalReason(Tq4, new TurboQuantContext(
            KvCommand.Serve, KvBackendKind.Cpu, Paged: false, config, HostIsPlainCpuTransformer: true, HasDraftModel: false, MtpWillAttach: false));
        Assert.NotNull(reason);
        Assert.Contains("MLA", reason);
        Assert.Contains("#1899", reason);
    }

    [Theory]
    [InlineData("vulkan")]
    [InlineData("cuda")]
    public void Serve_GpuRefusal_PointsAtTheCpuPath(string device)
    {
        // I1: serve defaults to --device auto, so the user may never have named the GPU; say how to reach the real path.
        string? reason = ServerStartup.PreLoadTurboQuantRefusal(Options(_tiny.GgufPath, "tq4", "tq4", device), _tiny.Config);
        Assert.Contains("--device cpu --no-paged", reason);
    }

    [Theory]
    [InlineData("tq", "tq4")]
    [InlineData("turboquant", "tq4")]
    [InlineData("tq4", "tq")]
    public void DefaultBits_EqualFour_NotAMixedFormat(string k, string v)
    {
        // M1: "tq" / "turboquant" mean the default 4 bits; pairing one with tq4 is the same format.
        Assert.Null(ServerStartup.PreLoadTurboQuantRefusal(Options(_tiny.GgufPath, k, v), _tiny.Config));
        Assert.DoesNotContain("same format", DotLLM.Cli.Helpers.KvStorageCli.TurboQuantRefusal(KvCommand.Run, k, v));
    }

    [Fact]
    public void LoadModel_HybridCpuHost_RefusesAfterLoad()
    {
        // I2: a non-plain CPU host is refused only by the post-load pass (geometry and backend pass before load).
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_dir, "hybrid.gguf"), withMtp: false);
        using (var gguf = GgufFile.Open(path))
            Assert.Null(ServerStartup.PreLoadTurboQuantRefusal(Options(path, "tq4", "tq4"), GgufModelConfigExtractor.Extract(gguf.Metadata)));
        var e = Assert.Throws<InvalidServerOptionException>(() => ServerStartup.LoadModel(path, Options(path, "tq4", "tq4")));
        Assert.Contains("CPU model", e.Message);
    }

    [Fact]
    public void LoadModel_HostWithMtpHead_RefusesAfterLoad()
    {
        // The MTP head is known only once the model has loaded.
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_dir, "hybrid-mtp.gguf"), withMtp: true);
        var e = Assert.Throws<InvalidServerOptionException>(() => ServerStartup.LoadModel(path, Options(path, "tq4", "tq4")));
        Assert.Contains("#1887", e.Message);
    }
}

/// <summary>I4: bench measured the default cache while its reproduce line said tq4.</summary>
[Collection(CliConsoleCollection.Name)]
public sealed class CliBenchTurboQuantTests
{
    [Fact]
    public void Bench_Tq4_ExitsNonZero_WithTheRefusal()
    {
        var (code, output) = CliTurboQuantCommandTests.Run("bench", "no-such-model-1888", "--kv-quant-k", "tq4", "--kv-quant-v", "tq4");
        Assert.NotEqual(0, code);
        Assert.Contains("#1887", output);
    }
}
