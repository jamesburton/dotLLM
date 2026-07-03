using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Backends;

/// <summary>
/// Real-weight <b>DiffusionGemma-26B</b> end-to-end masked-diffusion GENERATION on the
/// <b>Vulkan</b> backend — the #40 deliverable that the CPU-only
/// <c>DiffusionGemmaGgufForwardTests</c> cannot provide at usable speed (a 26B MoE denoise
/// loop on CPU is minutes per step; on the iGPU it is sub-second per step).
/// Drives the full pipeline: real GGUF → <see cref="VulkanTransformerModel"/> (region embed,
/// region scalar, self-conditioning, opt-in PKV) → <see cref="DiffusionTextGenerator"/>.
/// Reports the #41/#33 headline numbers: denoise-steps/sec, per-step canvas latency and
/// effective tokens/sec, alongside the decoded text for the coherence check.
/// </summary>
/// <remarks>
/// Gated on <c>DOTLLM_DIFFUSIONGEMMA_GGUF</c> (path to the .gguf) and a Vulkan device; skips
/// otherwise so the default sweep never depends on the multi-gig checkpoint. Canvas/steps/
/// prompt are overridable (<c>DOTLLM_DG_CANVAS</c> / <c>DOTLLM_DG_STEPS</c> /
/// <c>DOTLLM_DG_PROMPT</c> / <c>DOTLLM_DG_PKV</c>) so the same harness serves the quick
/// correctness run (small canvas) and the full-canvas throughput measurement.
/// </remarks>
[Trait("Category", "GPU")]
public sealed class DiffusionGemmaVulkanRealGenerationTests
{
    private const string ModelPathEnvVar = "DOTLLM_DIFFUSIONGEMMA_GGUF";

    private readonly ITestOutputHelper _out;

    public DiffusionGemmaVulkanRealGenerationTests(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void DiffusionGemma_26B_Vulkan_DenoiseLoop_RealWeights()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available on this host.");
        string? path = Environment.GetEnvironmentVariable(ModelPathEnvVar);
        Skip.If(string.IsNullOrWhiteSpace(path) || !File.Exists(path),
            $"Set {ModelPathEnvVar} to a diffusion-gemma GGUF (e.g. diffusiongemma-26B-A4B-it-Q4_K_M.gguf) to run this generation test.");

        string spvDir = ResolveSpvDir();
        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.DiffusionGemma, config.Architecture);
        Assert.NotNull(config.DiffusionConfig);

        var loadSw = Stopwatch.StartNew();
        using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);
        loadSw.Stop();
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        // Small-canvas default keeps the quick run cheap; the throughput run overrides to the
        // checkpoint's full canvas (256) and step budget (48, adaptive early-stop).
        int canvas = EnvInt("DOTLLM_DG_CANVAS", 32);
        int steps = EnvInt("DOTLLM_DG_STEPS", 16);
        bool pkv = Environment.GetEnvironmentVariable("DOTLLM_DG_PKV") != "0";
        var diff = config.DiffusionConfig! with
        {
            CanvasLength = canvas,
            MaxDenoisingSteps = steps,
            TemperatureMax = 0.6f,
            TemperatureMin = 0.2f,
        };

        string prompt = Environment.GetEnvironmentVariable("DOTLLM_DG_PROMPT") ?? "The Eiffel Tower is located in";
        int[] enc = tokenizer.Encode(prompt);
        int[] promptIds = new int[enc.Length + 1];
        promptIds[0] = tokenizer.BosTokenId;
        Array.Copy(enc, 0, promptIds, 1, enc.Length);

        var generator = new DiffusionTextGenerator(model, tokenizer, diffusionConfig: diff, enablePromptKv: pkv);
        var sw = Stopwatch.StartNew();
        DiffusionResult result = generator.Generate(promptIds);
        sw.Stop();

        double stepMs = sw.Elapsed.TotalMilliseconds / Math.Max(1, result.TotalDenoisingSteps);
        _out.WriteLine($"[dg-26B vulkan] model={Path.GetFileName(path)}  pkv={pkv}");
        _out.WriteLine($"  load          : {loadSw.Elapsed.TotalSeconds:F1} s");
        _out.WriteLine($"  prompt        : '{prompt}'  promptLen(incl BOS)={promptIds.Length}  canvas={canvas}  maxSteps={steps}");
        _out.WriteLine($"  gen wall      : {sw.Elapsed.TotalSeconds:F1} s  steps={result.TotalDenoisingSteps}  finish={result.FinishReason}");
        _out.WriteLine($"  step latency  : {stepMs:F0} ms/step  denoise-steps/s={result.TotalDenoisingSteps / sw.Elapsed.TotalSeconds:F2}");
        _out.WriteLine($"  effective     : {result.GeneratedTokenCount / sw.Elapsed.TotalSeconds:F2} tok/s  genToks={result.GeneratedTokenCount}");
        _out.WriteLine($"  distinct toks : {result.GeneratedTokenIds.Distinct().Count()}/{result.GeneratedTokenCount}");
        _out.WriteLine($"  token ids     : [{string.Join(",", result.GeneratedTokenIds)}]");
        _out.WriteLine($"  text          : {result.Text}");

        Assert.True(result.GeneratedTokenCount > 0, "Expected at least one generated token.");
        Assert.DoesNotContain(diff.MaskTokenId, result.GeneratedTokenIds); // canvas fully materialised
        foreach (int id in result.GeneratedTokenIds)
            Assert.InRange(id, 0, config.VocabSize - 1);
        Assert.False(string.IsNullOrWhiteSpace(result.Text), "Decoded text should be non-empty.");
        // Same coherence bar as the CPU denoise-loop test: the self-conditioned canvas must
        // not collapse to a single repeated low-information token.
        Assert.True(result.GeneratedTokenIds.Distinct().Count() > 2,
            $"Self-conditioned canvas must be non-degenerate; got text: '{result.Text}'.");
    }

    private static int EnvInt(string key, int fallback)
        => int.TryParse(Environment.GetEnvironmentVariable(key), out int n) && n > 0 ? n : fallback;

    private static string ResolveSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (var cand in candidates)
        {
            string full = Path.GetFullPath(cand);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        throw new InvalidOperationException(
            "SPIR-V blobs not found. Run native/vulkan/build.sh (or build.ps1) with the Vulkan SDK installed.");
    }
}
