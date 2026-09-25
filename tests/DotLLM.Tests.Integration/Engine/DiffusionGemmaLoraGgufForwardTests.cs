using System.Diagnostics;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Real-weight, real-adapter validation: loads one of the actual PEFT-trained LoRA
/// adapters for <c>diffusiongemma-26B-A4B-it</c> (e.g. <c>diffusiongemma_instruct_selfcond</c>
/// from the sibling <c>DiffusionGemmaTests</c> benchmarks) against the real 26B GGUF and
/// confirms it loads (region-aware <c>decoder.layers.*</c> / <c>encoder.language_model.
/// layers.*</c> tensor naming, <c>decoder.self_conditioning.*</c> gracefully skipped) and
/// measurably changes a real forward pass — closing the loop on the session's
/// LoRA-over-diffusion investigation with an actual trained adapter, not just synthetic
/// fixtures.
/// </summary>
/// <remarks>
/// Gated on <c>DOTLLM_DIFFUSIONGEMMA_GGUF</c> (path to the .gguf) AND
/// <c>DOTLLM_DIFFUSIONGEMMA_LORA_DIR</c> (path to a PEFT adapter directory containing
/// <c>adapter_config.json</c> + <c>adapter_model.safetensors</c>). Either unset/missing
/// self-skips so the build never depends on the multi-gig checkpoint or a sibling repo.
/// CPU-only, ONE forward pass (not a full denoise loop) — tractable in seconds-to-low-minutes
/// even on the 26B MoE (see <see cref="DiffusionGemmaGgufForwardTests"/> for the equivalent
/// no-adapter single-forward timing).
/// </remarks>
public sealed class DiffusionGemmaLoraGgufForwardTests
{
    private const string ModelPathEnvVar = "DOTLLM_DIFFUSIONGEMMA_GGUF";
    private const string LoraDirEnvVar = "DOTLLM_DIFFUSIONGEMMA_LORA_DIR";

    private readonly ITestOutputHelper _output;

    public DiffusionGemmaLoraGgufForwardTests(ITestOutputHelper output) => _output = output;

    private static string? TryResolveModelPath()
    {
        string? path = Environment.GetEnvironmentVariable(ModelPathEnvVar);
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return null;
        return path;
    }

    private static string? TryResolveLoraDir()
    {
        string? dir = Environment.GetEnvironmentVariable(LoraDirEnvVar);
        if (string.IsNullOrWhiteSpace(dir) || !Directory.Exists(dir))
            return null;
        return dir;
    }

    [SkippableFact]
    public unsafe void DiffusionGemma_26B_RealLoraAdapter_SingleForward_ChangesOutput()
    {
        string? modelPath = TryResolveModelPath();
        Skip.If(modelPath is null, $"Set {ModelPathEnvVar} to a diffusion-gemma GGUF to run this validation.");
        string? loraDir = TryResolveLoraDir();
        Skip.If(loraDir is null, $"Set {LoraDirEnvVar} to a real PEFT adapter directory to run this validation.");

        var loadSw = Stopwatch.StartNew();
        var (model, gguf, config) = ModelLoader.LoadFromGguf(modelPath!);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        loadSw.Stop();
        using var _ = gguf;
        using var __ = model;

        Assert.Equal(Architecture.DiffusionGemma, config.Architecture);
        Assert.NotNull(config.DiffusionConfig);
        int maskTokenId = config.DiffusionConfig!.MaskTokenId;

        var adapterLoadSw = Stopwatch.StartNew();
        using var adapter = PeftAdapterLoader.LoadFromDirectory("real-diffusiongemma-adapter", loraDir!, config);
        adapterLoadSw.Stop();
        Assert.True(adapter.IsCompatible(config));

        _output.WriteLine($"[diffusion-gemma GGUF] {modelPath}");
        _output.WriteLine($"[real adapter dir]     {loraDir}");
        _output.WriteLine($"  model load wall   : {loadSw.Elapsed.TotalSeconds:F2} s");
        _output.WriteLine($"  adapter load wall : {adapterLoadSw.Elapsed.TotalSeconds:F2} s  rank={adapter.Rank} alpha={adapter.Alpha}");

        const string prompt = "The Eiffel Tower is located in";
        int[] encoded = tokenizer.Encode(prompt);
        Assert.NotEmpty(encoded);

        const int nMask = 8;
        int promptLen = encoded.Length + 1; // BOS + prompt
        int seqLen = promptLen + nMask;
        int[] seq = new int[seqLen];
        seq[0] = tokenizer.BosTokenId;
        Array.Copy(encoded, 0, seq, 1, encoded.Length);
        for (int i = promptLen; i < seqLen; i++) seq[i] = maskTokenId;
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) positions[i] = i;

        var baselineSw = Stopwatch.StartNew();
        using ITensor baselineLogits = model.Forward(
            seq, positions, deviceId: -1, kvCache: null, adapter: null,
            AttentionMaskSpec.Hybrid(promptLen));
        baselineSw.Stop();

        var adaptedSw = Stopwatch.StartNew();
        using ITensor adaptedLogits = model.Forward(
            seq, positions, deviceId: -1, kvCache: null, adapter,
            AttentionMaskSpec.Hybrid(promptLen));
        adaptedSw.Stop();

        int vocab = baselineLogits.Shape[1];
        Assert.Equal(config.VocabSize, vocab);
        float* basePtr = (float*)baselineLogits.DataPointer;
        float* adaptPtr = (float*)adaptedLogits.DataPointer;

        float promptMaxDiff = 0f, canvasMaxDiff = 0f;
        bool allFinite = true;
        for (int row = 0; row < seqLen; row++)
        {
            long off = (long)row * vocab;
            float rowMax = 0f;
            for (int v = 0; v < vocab; v++)
            {
                float b = basePtr[off + v];
                float a = adaptPtr[off + v];
                if (!float.IsFinite(a)) allFinite = false;
                float d = MathF.Abs(a - b);
                if (d > rowMax) rowMax = d;
            }
            if (row < promptLen) { if (rowMax > promptMaxDiff) promptMaxDiff = rowMax; }
            else { if (rowMax > canvasMaxDiff) canvasMaxDiff = rowMax; }
        }

        _output.WriteLine($"  baseline forward  : {baselineSw.Elapsed.TotalSeconds:F2} s");
        _output.WriteLine($"  adapted forward   : {adaptedSw.Elapsed.TotalSeconds:F2} s");
        _output.WriteLine($"  prompt maxDiff    : {promptMaxDiff}");
        _output.WriteLine($"  canvas maxDiff    : {canvasMaxDiff}");

        Assert.True(allFinite, "Adapted logits must be finite (no NaN/Inf from the LoRA delta).");
        // This adapter's target_modules (self_attn q_proj/v_proj) are trained for BOTH the
        // decoder (canvas) and encoder (prompt) module trees — see adapter_config.json's
        // target_modules regex — so both regions should show a measurable delta.
        Assert.True(promptMaxDiff > 1e-4f, $"Real adapter had no measurable effect on prompt rows (diff={promptMaxDiff}).");
        Assert.True(canvasMaxDiff > 1e-4f, $"Real adapter had no measurable effect on canvas rows (diff={canvasMaxDiff}).");
    }
}
