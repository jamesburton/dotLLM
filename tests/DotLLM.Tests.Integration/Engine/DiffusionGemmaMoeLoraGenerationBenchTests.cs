using System.Diagnostics;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Baseline-vs-adapted BENCHMARK for a real trained LoRA adapter on the real DiffusionGemma
/// 26B GGUF (CPU path): (a) steady-state single-forward latency with and without the adapter
/// (the per-forward LoRA-application overhead), and (b) a short end-to-end denoise-loop
/// generation for the same prompt with and without the adapter, printing both decoded texts
/// for qualitative comparison. Companion to
/// <see cref="DiffusionGemmaLoraGgufForwardTests"/> (which validates the adapter *changes*
/// the forward pass; this test measures *cost* and shows *output*).
/// </summary>
/// <remarks>
/// Gated on <c>DOTLLM_DIFFUSIONGEMMA_GGUF</c> + <c>DOTLLM_DIFFUSIONGEMMA_LORA_DIR</c> like the
/// companion test. Optional <c>DOTLLM_DG_BENCH_PROMPT</c> overrides the prompt. Wall-clock on
/// the 26B CPU path is minutes — run explicitly, never in CI. Timings are same-process A/B
/// (identical page-cache state) but still subject to machine noise; treat small deltas as noise.
/// </remarks>
public sealed class DiffusionGemmaMoeLoraGenerationBenchTests
{
    private const string ModelPathEnvVar = "DOTLLM_DIFFUSIONGEMMA_GGUF";
    private const string LoraDirEnvVar = "DOTLLM_DIFFUSIONGEMMA_LORA_DIR";

    private readonly ITestOutputHelper _output;

    public DiffusionGemmaMoeLoraGenerationBenchTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void DiffusionGemma_26B_RealLoraAdapter_ForwardLatency_And_Generation_BaselineVsAdapted()
    {
        string? modelPath = Environment.GetEnvironmentVariable(ModelPathEnvVar);
        Skip.If(string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath),
            $"Set {ModelPathEnvVar} to a diffusion-gemma GGUF to run this benchmark.");
        string? loraDir = Environment.GetEnvironmentVariable(LoraDirEnvVar);
        Skip.If(string.IsNullOrWhiteSpace(loraDir) || !Directory.Exists(loraDir),
            $"Set {LoraDirEnvVar} to a real PEFT adapter directory to run this benchmark.");

        var (model, gguf, config) = ModelLoader.LoadFromGguf(modelPath!);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var _ = gguf;
        using var __ = model;
        Assert.Equal(Architecture.DiffusionGemma, config.Architecture);
        Assert.NotNull(config.DiffusionConfig);
        int maskTokenId = config.DiffusionConfig!.MaskTokenId;

        using var adapter = PeftAdapterLoader.LoadFromDirectory("bench-adapter", loraDir!, config);
        Assert.True(adapter.IsCompatible(config));

        // Gemma chat-format prompt (matches the adapter's training distribution).
        string question = Environment.GetEnvironmentVariable("DOTLLM_DG_BENCH_PROMPT")
            ?? "Write a C# extension method that returns the median of a List<decimal>.";
        string prompt = $"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n";
        int[] encoded = tokenizer.Encode(prompt);
        Assert.NotEmpty(encoded);

        // ---- (a) forward latency A/B: warm-up, then interleaved timed forwards ----
        const int nMask = 8;
        int promptLen = encoded.Length + 1;
        int seqLen = promptLen + nMask;
        int[] seq = new int[seqLen];
        seq[0] = tokenizer.BosTokenId;
        Array.Copy(encoded, 0, seq, 1, encoded.Length);
        for (int i = promptLen; i < seqLen; i++) seq[i] = maskTokenId;
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) positions[i] = i;

        using (ITensor warm = model.Forward(seq, positions, deviceId: -1, kvCache: null,
                   adapter: null, AttentionMaskSpec.Hybrid(promptLen)))
        {
            _output.WriteLine($"  warm-up forward     : done ({warm.Shape[0]}x{warm.Shape[1]})");
        }

        const int reps = 2;
        double baselineMs = 0, adaptedMs = 0;
        for (int r = 0; r < reps; r++)
        {
            var swB = Stopwatch.StartNew();
            using (model.Forward(seq, positions, deviceId: -1, kvCache: null,
                       adapter: null, AttentionMaskSpec.Hybrid(promptLen))) { }
            swB.Stop();
            baselineMs += swB.Elapsed.TotalMilliseconds;

            var swA = Stopwatch.StartNew();
            using (model.Forward(seq, positions, deviceId: -1, kvCache: null,
                       adapter, AttentionMaskSpec.Hybrid(promptLen))) { }
            swA.Stop();
            adaptedMs += swA.Elapsed.TotalMilliseconds;
        }
        baselineMs /= reps;
        adaptedMs /= reps;
        _output.WriteLine($"  baseline forward    : {baselineMs / 1000:F2} s (mean of {reps})");
        _output.WriteLine($"  adapted forward     : {adaptedMs / 1000:F2} s (mean of {reps})");
        _output.WriteLine($"  adapter overhead    : {(adaptedMs - baselineMs) / 1000:F2} s " +
                          $"({(adaptedMs / baselineMs - 1) * 100:F1} %)");

        // ---- (b) short generation A/B (same sampler settings, same prompt) ----
        var diff = config.DiffusionConfig! with { CanvasLength = 16, MaxDenoisingSteps = 12 };
        var generator = new DiffusionTextGenerator(model, tokenizer, diffusionConfig: diff);

        var genBaseSw = Stopwatch.StartNew();
        DiffusionResult baseGen = generator.Generate(prompt, adapter: null);
        genBaseSw.Stop();
        var genAdaptSw = Stopwatch.StartNew();
        DiffusionResult adaptGen = generator.Generate(prompt, adapter: adapter);
        genAdaptSw.Stop();

        _output.WriteLine($"  baseline generation : {genBaseSw.Elapsed.TotalSeconds:F1} s, " +
                          $"{baseGen.TotalDenoisingSteps} steps, {baseGen.GeneratedTokenCount} tokens");
        _output.WriteLine($"    text: {baseGen.Text}");
        _output.WriteLine($"  adapted generation  : {genAdaptSw.Elapsed.TotalSeconds:F1} s, " +
                          $"{adaptGen.TotalDenoisingSteps} steps, {adaptGen.GeneratedTokenCount} tokens");
        _output.WriteLine($"    text: {adaptGen.Text}");

        Assert.True(baseGen.GeneratedTokenCount > 0, "Baseline generation produced no tokens.");
        Assert.True(adaptGen.GeneratedTokenCount > 0, "Adapted generation produced no tokens.");
    }
}
