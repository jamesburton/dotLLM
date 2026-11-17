using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.HuggingFace;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Issue #1851 review: the <c>auto</c> sentinel (-2) must not leak into <c>/props</c> or a stored profile, and the HTTP load and
/// profile paths must validate <c>gpu_layers</c> like the CLI (a stray negative was read as "all layers").
/// </summary>
public sealed class GpuLayersJsonTests
{
    [Theory]
    [InlineData("""{"model":"m","gpu_layers":"auto"}""", GpuOffloadPlanner.AutoGpuLayers)]
    [InlineData("""{"model":"m","gpu_layers":"all"}""", -1)]
    [InlineData("""{"model":"m","gpu_layers":-1}""", -1)]
    [InlineData("""{"model":"m","gpu_layers":12}""", 12)]
    [InlineData("""{"model":"m","gpu_layers":"12"}""", 12)]
    public void LoadRequest_AcceptsWhatTheCliAccepts(string json, int expected)
        => Assert.Equal(expected, JsonSerializer.Deserialize(json, ServerJsonContext.Default.ModelLoadRequest)!.GpuLayers);

    [Fact]
    public void LoadRequest_OmittedGpuLayers_IsUnset()
        => Assert.Null(JsonSerializer.Deserialize("""{"model":"m"}""", ServerJsonContext.Default.ModelLoadRequest)!.GpuLayers);

    [Theory]
    [InlineData("""{"model":"m","gpu_layers":-5}""")]
    [InlineData("""{"model":"m","gpu_layers":-2}""")]
    [InlineData("""{"model":"m","gpu_layers":"half"}""")]
    [InlineData("""{"model":"m","gpu_layers":1.5}""")]
    public void LoadRequest_RefusesAnythingElse(string json)
        => Assert.Throws<JsonException>(() => JsonSerializer.Deserialize(json, ServerJsonContext.Default.ModelLoadRequest));

    [Fact]
    public void Props_ReportAuto_NotTheSentinel()
    {
        string json = JsonSerializer.Serialize(new PropsResponse { SamplingDefaults = new(), GpuLayers = GpuOffloadPlanner.AutoGpuLayers }, ServerJsonContext.Default.PropsResponse);
        Assert.Contains("\"gpu_layers\":\"auto\"", json.Replace(" ", ""));
        Assert.DoesNotContain("-2", json);
    }

    [Fact]
    public void Profile_RoundTripsAuto_AsAString()
    {
        string dir = Path.Combine(Path.GetTempPath(), "dotllm-profile-" + Guid.NewGuid().ToString("N"));
        try
        {
            ModelProfileStore.Save("splitme", new ModelProfile { From = "acme/x-GGUF", GpuLayers = GpuOffloadPlanner.AutoGpuLayers }, dir);
            string text = File.ReadAllText(Directory.GetFiles(dir).Single());
            Assert.Contains("\"auto\"", text);
            Assert.DoesNotContain("-2", text);
            Assert.Equal(GpuOffloadPlanner.AutoGpuLayers, ModelProfileStore.TryGet("splitme", dir)!.GpuLayers);
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch { /* best-effort */ }
        }
    }

    [Fact]
    public void RequestedAllLayers_ClearsProfileAndStartup_ButAutoPassesThrough()
    {
        Assert.Null(ModelManagementEndpoint.ResolveRequestedGpuLayers(-1, 10, 20));
        Assert.Equal(GpuOffloadPlanner.AutoGpuLayers, ModelManagementEndpoint.ResolveRequestedGpuLayers(GpuOffloadPlanner.AutoGpuLayers, 10, 20));
        Assert.Equal(GpuOffloadPlanner.AutoGpuLayers, ModelManagementEndpoint.ResolveRequestedGpuLayers(null, GpuOffloadPlanner.AutoGpuLayers, 20));
    }
}
