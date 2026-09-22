using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Vulkan twin of <c>CudaUnsupportedArchitectureGuardTests</c> (issue #480): gpt-oss must be refused
/// with a specific <see cref="NotSupportedException"/> naming what Vulkan lacks (attention sinks,
/// dense YaRN) instead of loading through the generic <see cref="VulkanTransformerModel"/> and
/// silently producing wrong logits.
/// </summary>
/// <remarks>
/// <para>
/// The guard fires purely on <c>config.Architecture</c>, before any GGUF tensor is read, so any
/// real GGUF works as the <c>gguf</c> argument — only the <see cref="ModelConfig"/> passed alongside
/// it claims gpt-oss. The control (<see cref="CreateFromGguf_SameGgufAsLlama_Loads"/>) proves the
/// refusal comes from the architecture, not from the fixture.
/// </para>
/// <para>
/// Besides the <see cref="VulkanModelLoader.CreateFromGguf"/> dispatch point, the side doors that
/// bypass it are covered: direct <see cref="VulkanTransformerModel.LoadFromGguf(GgufFile, ModelConfig, string?)"/>
/// (the benchmarks call it), <see cref="VulkanPipelineTransformerModel.LoadFromGguf"/>, and
/// <see cref="HybridVulkanCudaTransformerModel.LoadFromGguf"/>. None of those reach a Vulkan device
/// or CUDA before refusing, so they run on any host that has the fixture.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class VulkanUnsupportedArchitectureGuardTests
{
    private static FixtureLocation ResolveAnyGguf() => TestFixtureResolver.ResolveFile(
        "DOTLLM_SMOLLM2_135M_INSTRUCT_Q8_GGUF", "bartowski", "SmolLM2-135M-Instruct-GGUF",
        "SmolLM2-135M-Instruct-Q8_0.gguf");

    private static void AssertGptOssRefusal(NotSupportedException ex)
    {
        Assert.Contains(nameof(Architecture.GptOss), ex.Message, StringComparison.Ordinal);
        Assert.Contains("attention sinks", ex.Message, StringComparison.Ordinal);
        Assert.Contains("YaRN", ex.Message, StringComparison.Ordinal);
        // Not the confusing deep-loader failure mode.
        Assert.DoesNotContain("attn_output.weight", ex.Message, StringComparison.Ordinal);
    }

    private static (GgufFile Gguf, ModelConfig GptOssConfig) OpenAsGptOss(FixtureLocation fixture)
    {
        var gguf = GgufFile.Open(fixture.Path!);
        var realConfig = GgufModelConfigExtractor.Extract(gguf.Metadata);
        // gpt-oss shape on the config the loader sees: architecture + alternating SWA.
        return (gguf, realConfig with
        {
            Architecture = Architecture.GptOss,
            SlidingWindowSize = 128,
            SlidingWindowPattern = 2,
        });
    }

    [SkippableFact]
    public void CreateFromGguf_GptOss_ThrowsNotSupportedNamingMissingFeatures()
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available.");
        var fixture = ResolveAnyGguf();
        Skip.If(!fixture.Found, fixture.SkipMessage("any real GGUF (content unused by this test)"));

        var (gguf, config) = OpenAsGptOss(fixture);
        using var _ = gguf;
        using var device = VulkanDevice.Create();

        var ex = Assert.Throws<NotSupportedException>(
            () => VulkanModelLoader.CreateFromGguf(device, gguf, config, spvDir: AppContext.BaseDirectory));
        AssertGptOssRefusal(ex);
    }

    /// <summary>Control: the identical GGUF + config minus the architecture flip loads fine.</summary>
    [SkippableFact]
    public void CreateFromGguf_SameGgufAsLlama_Loads()
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available.");
        var fixture = ResolveAnyGguf();
        Skip.If(!fixture.Found, fixture.SkipMessage("any real GGUF"));
        string? spvDir = ResolveSpvDir();
        Skip.If(spvDir is null, "Vulkan SPV directory not found.");

        using var gguf = GgufFile.Open(fixture.Path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();

        var (model, _) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spvDir!);
        model.Dispose();
    }

    [SkippableFact]
    public void DirectVulkanTransformerModelLoadFromGguf_GptOss_Throws()
    {
        var fixture = ResolveAnyGguf();
        Skip.If(!fixture.Found, fixture.SkipMessage("any real GGUF (content unused by this test)"));
        var (gguf, config) = OpenAsGptOss(fixture);
        using var _ = gguf;

        AssertGptOssRefusal(Assert.Throws<NotSupportedException>(
            () => VulkanTransformerModel.LoadFromGguf(gguf, config)));
    }

    [SkippableFact]
    public void VulkanPipelineLoadFromGguf_GptOss_Throws()
    {
        var fixture = ResolveAnyGguf();
        Skip.If(!fixture.Found, fixture.SkipMessage("any real GGUF (content unused by this test)"));
        var (gguf, config) = OpenAsGptOss(fixture);
        using var _ = gguf;

        AssertGptOssRefusal(Assert.Throws<NotSupportedException>(
            () => VulkanPipelineTransformerModel.LoadFromGguf(
                gguf, config, splitLayer: 1, device0Index: 0, device1Index: 0)));
    }

    [SkippableFact]
    public void HybridVulkanCudaLoadFromGguf_GptOss_Throws()
    {
        var fixture = ResolveAnyGguf();
        Skip.If(!fixture.Found, fixture.SkipMessage("any real GGUF (content unused by this test)"));
        var (gguf, config) = OpenAsGptOss(fixture);
        using var _ = gguf;

        AssertGptOssRefusal(Assert.Throws<NotSupportedException>(
            () => HybridVulkanCudaTransformerModel.LoadFromGguf(gguf, config, numVulkanLayers: 1)));
    }

    private static string? ResolveSpvDir()
    {
        string? probe = AppContext.BaseDirectory;
        for (int i = 0; i < 8 && probe is not null; i++)
        {
            string candidate = Path.Combine(probe, "native", "vulkan", "spv");
            if (Directory.Exists(candidate)) return candidate;
            probe = Path.GetDirectoryName(probe);
        }
        return null;
    }
}
