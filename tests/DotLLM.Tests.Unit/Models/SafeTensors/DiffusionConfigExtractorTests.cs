using System.Text.Json;
using DotLLM.Core.Models;
using DotLLM.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

/// <summary>
/// Unit tests for <see cref="DiffusionConfigExtractor"/> — the DiffusionGemma
/// <c>config.json</c> + <c>generation_config.json</c> + tokenizer-metadata →
/// <see cref="DiffusionConfig"/> parser, including mask-token resolution and its
/// fail-loud path.
/// </summary>
public sealed class DiffusionConfigExtractorTests
{
    private const string ConfigJson = """
        { "model_type": "diffusion_gemma", "canvas_length": 256 }
        """;

    private const string GenerationConfigJson = """
        {
            "max_denoising_steps": 48,
            "entropy_bound": 0.1,
            "confidence_threshold": 0.005,
            "stability_threshold": 1,
            "t_max": 0.8,
            "t_min": 0.4
        }
        """;

    [Fact]
    public void Extract_FullFixture_PopulatesEveryFieldAndResolvedMaskId()
    {
        using var cfg = JsonDocument.Parse(ConfigJson);
        using var gen = JsonDocument.Parse(GenerationConfigJson);

        DiffusionConfig dc = DiffusionConfigExtractor.Extract(
            cfg.RootElement, gen.RootElement, maskTokenId: 262144);

        Assert.Equal(256, dc.CanvasLength);
        Assert.Equal(48, dc.MaxDenoisingSteps);
        Assert.Equal(0.1f, dc.EntropyBound);
        Assert.Equal(0.005f, dc.ConfidenceThreshold);
        Assert.Equal(1, dc.StabilityThreshold);
        Assert.Equal(0.8f, dc.TemperatureMax);
        Assert.Equal(0.4f, dc.TemperatureMin);
        Assert.Equal(262144, dc.MaskTokenId);
    }

    [Fact]
    public void Extract_NestedSamplerConfig_ReadsScheduleFields()
    {
        const string nested = """
            {
                "max_denoising_steps": 48,
                "entropy_bound_sampler_config": {
                    "entropy_bound": 0.1,
                    "confidence_threshold": 0.005,
                    "stability_threshold": 1,
                    "t_max": 0.8,
                    "t_min": 0.4
                }
            }
            """;
        using var cfg = JsonDocument.Parse(ConfigJson);
        using var gen = JsonDocument.Parse(nested);

        DiffusionConfig dc = DiffusionConfigExtractor.Extract(cfg.RootElement, gen.RootElement, 7);

        Assert.Equal(0.1f, dc.EntropyBound);
        Assert.Equal(0.005f, dc.ConfidenceThreshold);
        Assert.Equal(1, dc.StabilityThreshold);
        Assert.Equal(0.8f, dc.TemperatureMax);
        Assert.Equal(0.4f, dc.TemperatureMin);
    }

    [Fact]
    public void Extract_MissingGenerationConfig_FallsBackToVerifiedDefaults()
    {
        const string canvasOnly = """ { "canvas_length": 128 } """;
        using var cfg = JsonDocument.Parse(canvasOnly);

        DiffusionConfig dc = DiffusionConfigExtractor.Extract(cfg.RootElement, generationConfig: null, maskTokenId: 5);

        Assert.Equal(128, dc.CanvasLength); // from config.json
        Assert.Equal(48, dc.MaxDenoisingSteps);
        Assert.Equal(0.1f, dc.EntropyBound);
        Assert.Equal(0.005f, dc.ConfidenceThreshold);
        Assert.Equal(1, dc.StabilityThreshold);
        Assert.Equal(0.8f, dc.TemperatureMax);
        Assert.Equal(0.4f, dc.TemperatureMin);
        Assert.Equal(5, dc.MaskTokenId);
    }

    [Fact]
    public void Extract_NegativeMaskTokenId_Throws()
    {
        using var cfg = JsonDocument.Parse(ConfigJson);
        using var gen = JsonDocument.Parse(GenerationConfigJson);
        Assert.Throws<ArgumentOutOfRangeException>(
            () => DiffusionConfigExtractor.Extract(cfg.RootElement, gen.RootElement, maskTokenId: -1));
    }

    [Fact]
    public void ExtractFromDirectory_ResolvesMaskFromTokenizerConfig_AndPopulatesEveryField()
    {
        string dir = CreateTempDir();
        try
        {
            File.WriteAllText(Path.Combine(dir, "config.json"), ConfigJson);
            File.WriteAllText(Path.Combine(dir, "generation_config.json"), GenerationConfigJson);
            // special_tokens_map declares the mask token CONTENT; tokenizer_config
            // carries the id in its added_tokens_decoder map.
            File.WriteAllText(Path.Combine(dir, "special_tokens_map.json"),
                """ { "mask_token": "[MASK]" } """);
            File.WriteAllText(Path.Combine(dir, "tokenizer_config.json"),
                """
                {
                    "mask_token": "[MASK]",
                    "added_tokens_decoder": {
                        "0": { "content": "<pad>" },
                        "262144": { "content": "[MASK]", "special": true }
                    }
                }
                """);

            using var cfg = JsonDocument.Parse(ConfigJson);
            DiffusionConfig dc = DiffusionConfigExtractor.ExtractFromDirectory(dir, cfg.RootElement);

            Assert.Equal(256, dc.CanvasLength);
            Assert.Equal(48, dc.MaxDenoisingSteps);
            Assert.Equal(0.1f, dc.EntropyBound);
            Assert.Equal(0.005f, dc.ConfidenceThreshold);
            Assert.Equal(1, dc.StabilityThreshold);
            Assert.Equal(0.8f, dc.TemperatureMax);
            Assert.Equal(0.4f, dc.TemperatureMin);
            Assert.Equal(262144, dc.MaskTokenId);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void ResolveMaskTokenId_FromSpecialTokensMapObjectWithId()
    {
        string dir = CreateTempDir();
        try
        {
            // AddedToken-object form carries the id directly.
            File.WriteAllText(Path.Combine(dir, "special_tokens_map.json"),
                """ { "mask_token": { "content": "[MASK]", "id": 99 } } """);

            int id = DiffusionConfigExtractor.ResolveMaskTokenId(dir);
            Assert.Equal(99, id);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void ResolveMaskTokenId_FromTokenizerJsonAddedTokens()
    {
        string dir = CreateTempDir();
        try
        {
            // Only tokenizer.json present — added_tokens array fallback path.
            File.WriteAllText(Path.Combine(dir, "tokenizer.json"),
                """
                {
                    "added_tokens": [
                        { "id": 1, "content": "<bos>" },
                        { "id": 262144, "content": "[MASK]", "special": true }
                    ]
                }
                """);

            int id = DiffusionConfigExtractor.ResolveMaskTokenId(dir);
            Assert.Equal(262144, id);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void ResolveMaskTokenId_NoMaskToken_FailsLoudly()
    {
        string dir = CreateTempDir();
        try
        {
            // Tokenizer files exist but declare no mask token at all.
            File.WriteAllText(Path.Combine(dir, "special_tokens_map.json"),
                """ { "bos_token": "<bos>", "eos_token": "<eos>" } """);
            File.WriteAllText(Path.Combine(dir, "tokenizer.json"),
                """ { "added_tokens": [ { "id": 1, "content": "<bos>" } ] } """);

            var ex = Assert.Throws<InvalidDataException>(
                () => DiffusionConfigExtractor.ResolveMaskTokenId(dir));
            Assert.Contains("mask token", ex.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void ExtractFromDirectory_UnresolvableMask_FailsLoudly()
    {
        string dir = CreateTempDir();
        try
        {
            File.WriteAllText(Path.Combine(dir, "config.json"), ConfigJson);
            File.WriteAllText(Path.Combine(dir, "generation_config.json"), GenerationConfigJson);
            // No tokenizer metadata at all.
            using var cfg = JsonDocument.Parse(ConfigJson);

            Assert.Throws<InvalidDataException>(
                () => DiffusionConfigExtractor.ExtractFromDirectory(dir, cfg.RootElement));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    private static string CreateTempDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), "dotllm-diffcfg-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }
}
