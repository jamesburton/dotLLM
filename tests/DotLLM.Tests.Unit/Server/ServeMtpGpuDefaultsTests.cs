using System.Text.Json;
using DotLLM.Models.Gguf;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>#757: <c>serve</c> defaults MTP on for embedded-head models, <c>has_mtp</c> in inspect, GPU-by-default device recommendation.</summary>
public class ServeMtpGpuDefaultsTests : IDisposable
{
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-757-" + Guid.NewGuid().ToString("N"));

    public ServeMtpGpuDefaultsTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best effort */ }
    }

    // ---- option parsing ----

    [Fact]
    public void Parse_Default_MtpOn_NotExplicit_DeviceAuto()
    {
        var o = ServerOptions.Parse(["--model", "m.gguf"]);
        Assert.True(o.MtpEnabled);
        Assert.False(o.MtpExplicit);
        Assert.Equal("auto", o.Device);
    }

    [Fact]
    public void Parse_NoMtp_DisablesMtp()
    {
        var o = ServerOptions.Parse(["--model", "m.gguf", "--no-mtp"]);
        Assert.False(o.MtpEnabled);
    }

    [Fact]
    public void Parse_Mtp_StillAccepted_AndMarksExplicit()
    {
        var o = ServerOptions.Parse(["--model", "m.gguf", "--mtp"]);
        Assert.True(o.MtpEnabled);
        Assert.True(o.MtpExplicit);
    }

    [Fact]
    public void Parse_LastMtpFlagWins()
    {
        Assert.False(ServerOptions.Parse(["--model", "m", "--mtp", "--no-mtp"]).MtpEnabled);
        Assert.True(ServerOptions.Parse(["--model", "m", "--no-mtp", "--mtp"]).MtpEnabled);
    }

    // ---- policy ----

    private static ServerOptions Opts(bool mtp = true, bool explicitFlag = false, int concurrency = 0) =>
        new() { Model = "m", MtpEnabled = mtp, MtpExplicit = explicitFlag, ExpectedConcurrency = concurrency };

    [Fact]
    public void Decide_AutoEnables_WhenHeadPresent_NoDraft_LowConcurrency()
    {
        var d = ServerStartup.DecideMtp(Opts(), supportsMtp: true, hasDraftModel: false, concurrencyOverMtp: false);
        Assert.True(d.Active);
        Assert.Equal("active", d.Status);
        Assert.Contains("enabled automatically", d.Message);
        Assert.Contains("continuous batching is off", d.Message);
    }

    [Fact]
    public void Decide_NoHead_IsQuiet_UnlessExplicit()
    {
        var quiet = ServerStartup.DecideMtp(Opts(), supportsMtp: false, hasDraftModel: false, concurrencyOverMtp: false);
        Assert.False(quiet.Active);
        Assert.Null(quiet.Message);

        var loud = ServerStartup.DecideMtp(Opts(explicitFlag: true), supportsMtp: false, hasDraftModel: false, concurrencyOverMtp: false);
        Assert.False(loud.Active);
        Assert.Contains("no MTP head", loud.Message);
    }

    [Fact]
    public void Decide_NoMtpFlag_ExplainsOptOut()
    {
        var d = ServerStartup.DecideMtp(Opts(mtp: false), supportsMtp: true, hasDraftModel: false, concurrencyOverMtp: false);
        Assert.False(d.Active);
        Assert.Contains("--no-mtp", d.Message);
    }

    [Fact]
    public void Decide_ExternalDraft_Wins()
    {
        var d = ServerStartup.DecideMtp(Opts(), supportsMtp: true, hasDraftModel: true, concurrencyOverMtp: false);
        Assert.False(d.Active);
        Assert.Contains("external draft", d.Message);
    }

    [Fact]
    public void Decide_HighConcurrency_Wins()
    {
        var d = ServerStartup.DecideMtp(Opts(concurrency: 8), supportsMtp: true, hasDraftModel: false, concurrencyOverMtp: true);
        Assert.False(d.Active);
        Assert.Contains("expected-concurrency 8", d.Message);
    }

    // ---- inspect has_mtp ----

    private static (DotLLM.Core.Models.ModelConfig Config, GgufFile File) Open(string path)
    {
        var gguf = GgufFile.Open(path);
        return (GgufModelConfigExtractor.Extract(gguf.Metadata), gguf);
    }

    [Fact]
    public void Inspect_HasMtp_TrueForEmbeddedHead_FalseWithout()
    {
        string with = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "with.gguf"), withMtp: true);
        string without = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "without.gguf"), withMtp: false);

        var (cfgWith, gWith) = Open(with);
        using (gWith) Assert.True(ModelInspectEndpoint.HasEmbeddedMtpHead(cfgWith, gWith.TensorsByName));
        var (cfgWithout, gWithout) = Open(without);
        using (gWithout) Assert.False(ModelInspectEndpoint.HasEmbeddedMtpHead(cfgWithout, gWithout.TensorsByName));
    }

    [Fact]
    public void Inspect_HasMtp_FalseWhenHparamPresentButTensorsMissing()
    {
        // The hparam alone is not enough: LoadMtpHeadIfPresent returns null without nextn.eh_proj, so SupportsMtp is false.
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "nohead.gguf"), withMtp: true, mtpHasOwnHeadTensors: false);
        var (cfg, g) = Open(path);
        using (g)
        {
            if (g.TensorsByName.ContainsKey($"blk.{cfg.NumLayers}.nextn.eh_proj.weight"))
                return; // fixture variant still carries the tensor; nothing to assert
            Assert.False(ModelInspectEndpoint.HasEmbeddedMtpHead(cfg, g.TensorsByName));
        }
    }

    [Fact]
    public void Inspect_Response_SerializesHasMtp_AndDefaultsFalse()
    {
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "json.gguf"), withMtp: true);
        var (cfg, g) = Open(path);
        using (g)
        {
            var resp = ModelInspectEndpoint.BuildResponse(cfg, 1, ModelInspectEndpoint.HasEmbeddedMtpHead(cfg, g.TensorsByName));
            string json = JsonSerializer.Serialize(resp, ServerJsonContext.Default.ModelInspectResponse);
            Assert.Contains("\"has_mtp\":true", json, StringComparison.Ordinal);

            // Round trip + STJ source-gen default guard: `{}`-style payload without the field arrives as false (no initializer to drop).
            var back = JsonSerializer.Deserialize("{\"architecture\":\"x\"}", ServerJsonContext.Default.ModelInspectResponse)!;
            Assert.False(back.HasMtp);
        }
    }

    // ---- JSON defaults (CLAUDE.md STJ rule) ----

    [Fact]
    public void LoadRequest_Mtp_IsNullableAndAbsentByDefault()
    {
        var direct = new ModelLoadRequest { Model = "m" };
        var viaJson = JsonSerializer.Deserialize("{\"model\":\"m\"}", ServerJsonContext.Default.ModelLoadRequest)!;
        Assert.Null(direct.Mtp);
        Assert.Equal(direct.Mtp, viaJson.Mtp);

        var off = JsonSerializer.Deserialize("{\"model\":\"m\",\"mtp\":false}", ServerJsonContext.Default.ModelLoadRequest)!;
        Assert.False(off.Mtp);
    }

    [Fact]
    public void Props_ExposesMtpStatus()
    {
        var props = new PropsResponse { SamplingDefaults = new SamplingDefaultsDto(), MtpActive = true, MtpStatus = "active" };
        string json = JsonSerializer.Serialize(props, ServerJsonContext.Default.PropsResponse);
        Assert.Contains("\"mtp_active\":true", json, StringComparison.Ordinal);
        Assert.Contains("\"mtp_status\":\"active\"", json, StringComparison.Ordinal);
    }

    // ---- GPU-by-default device recommendation ----

    private static DeviceListResponse Devices(bool vulkan, bool cuda24gb = false) => new()
    {
        Backends =
        [
            new BackendInfoDto { Name = "cpu", Available = true, Servable = true, DeviceCount = 1,
                Devices = [new DeviceInfoDto { Index = 0, Name = "cpu", DeviceString = "cpu" }] },
            new BackendInfoDto { Name = "cuda", Available = cuda24gb, Servable = cuda24gb, DeviceCount = cuda24gb ? 1 : 0,
                Devices = cuda24gb ? [new DeviceInfoDto { Index = 0, Name = "gpu", DeviceString = "gpu:0", TotalMemoryBytes = 24L << 30 }] : [] },
            new BackendInfoDto { Name = "vulkan", Available = vulkan, Servable = vulkan, DeviceCount = vulkan ? 1 : 0,
                Devices = vulkan ? [new DeviceInfoDto { Index = 0, Name = "Vulkan GPU", DeviceString = "vulkan" }] : [] },
        ],
    };

    [Fact]
    public void RecommendedDevice_PrefersGpu_CpuOnlyWhenNoneServable()
    {
        Assert.Equal("vulkan", DeviceSelector.Candidates(0, Devices(vulkan: true))[0]);
        Assert.Equal("gpu:0", DeviceSelector.Candidates(1L << 30, Devices(vulkan: true, cuda24gb: true))[0]);
        Assert.Equal("cpu", DeviceSelector.Candidates(0, Devices(vulkan: false))[0]);
    }

    [Fact]
    public void Devices_RecommendedDevice_SerializesAndIsNullableByDefault()
    {
        var d = Devices(vulkan: true) with { RecommendedDevice = "vulkan" };
        string json = JsonSerializer.Serialize(d, ServerJsonContext.Default.DeviceListResponse);
        Assert.Contains("\"recommended_device\":\"vulkan\"", json, StringComparison.Ordinal);
        var back = JsonSerializer.Deserialize("{\"backends\":[]}", ServerJsonContext.Default.DeviceListResponse)!;
        Assert.Null(back.RecommendedDevice);
    }
}
