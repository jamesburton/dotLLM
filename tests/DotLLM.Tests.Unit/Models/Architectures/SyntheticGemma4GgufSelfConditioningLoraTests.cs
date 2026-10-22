using System.Text.Json;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// DiffusionGemma's <c>decoder.self_conditioning.{gate,up,down}_proj</c> LoRA tensors used to
/// be recognised-but-SKIPPED by <see cref="PeftAdapterLoader"/> (no per-layer index, no
/// application hook). This proves the wired-up path end to end: a real GGUF-loaded
/// DiffusionGemma forward, with self-conditioning active (<c>SetDiffusionSelfCond</c>, non-empty
/// previous-step canvas logits, <c>scUse=1</c>), and an adapter that ONLY carries
/// self-conditioning deltas (no per-layer q/k/v/o/gate/up/down entries at all) measurably
/// changes the canvas-row logits relative to the no-adapter baseline — while leaving the
/// prompt-row logits BIT-IDENTICAL, since self-conditioning is a model-level, canvas-only
/// computation with no encoder/prompt equivalent.
/// </summary>
public sealed class SyntheticGemma4GgufSelfConditioningLoraTests : IDisposable
{
    private readonly string _scratch;

    public SyntheticGemma4GgufSelfConditioningLoraTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-gemma4-sc-lora-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private const int PromptLen = 3;
    private const int CanvasLen = 4;

    [Fact]
    public void Forward_SelfConditioningOnlyAdapter_ChangesCanvasLogitsOnly()
    {
        var cfg = SyntheticGemma4Gguf.Tiny with { BlockCount = 1 };
        string ggufPath = Path.Combine(_scratch, "diffusiongemma-sc-tiny.gguf");
        SyntheticGemma4Gguf.WriteDiffusionGemma(ggufPath, cfg, seed: 0xC0FFEEu);

        string adapterDir = BuildSelfConditioningOnlyPeftFixture(
            rank: 4, alpha: 8f, hidden: cfg.HiddenSize, ff: cfg.DenseFeedForward);

        int seqLen = PromptLen + CanvasLen;
        int[] tokenIds = new int[seqLen];
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) { tokenIds[i] = 3 + i; positions[i] = i; }
        var maskSpec = AttentionMaskSpec.Hybrid(PromptLen);

        // Non-empty previous-step canvas logits + scUse=1 so ApplySelfConditioning's
        // soft-embed + gated GeGLU MLP path actually runs (scUse=0 ⇒ zero-SC no-op).
        var rng = new Random(1234);
        float[] prevLogits = new float[CanvasLen * cfg.VocabSize];
        for (int i = 0; i < prevLogits.Length; i++)
            prevLogits[i] = (float)(rng.NextDouble() * 6.0 - 3.0);

        var (baseline, gguf1, config1) = ModelLoader.LoadFromGguf(ggufPath, ThreadingConfig.SingleThreaded);
        using var _g1 = gguf1;
        using var _m1 = baseline;
        Assert.NotNull(config1.DiffusionConfig);
        baseline.SetDiffusionSelfCond(prevLogits, CanvasLen, scUse: 1f);
        using ITensor baselineLogits = baseline.Forward(
            tokenIds, positions, deviceId: -1, kvCache: null, adapter: null, maskSpec);
        var (baselinePrompt, baselineCanvas) = SplitRows(baselineLogits, PromptLen, CanvasLen);

        var (adapted, gguf2, config2) = ModelLoader.LoadFromGguf(ggufPath, ThreadingConfig.SingleThreaded);
        using var _g2 = gguf2;
        using var _m2 = adapted;
        using var adapter = PeftAdapterLoader.LoadFromDirectory("sc-probe", adapterDir, config2);
        adapted.SetDiffusionSelfCond(prevLogits, CanvasLen, scUse: 1f);
        using ITensor adaptedLogits = adapted.Forward(
            tokenIds, positions, deviceId: -1, kvCache: null, adapter, maskSpec);
        var (adaptedPrompt, adaptedCanvas) = SplitRows(adaptedLogits, PromptLen, CanvasLen);

        float promptDiff = MaxAbsDiff(baselinePrompt, adaptedPrompt);
        float canvasDiff = MaxAbsDiff(baselineCanvas, adaptedCanvas);

        Assert.True(canvasDiff > 1e-4f,
            $"Self-conditioning-only LoRA adapter had no measurable effect on canvas logits (diff={canvasDiff}).");
        Assert.True(promptDiff == 0f,
            $"Self-conditioning-only LoRA adapter leaked into prompt rows (diff={promptDiff}) — "
            + "self-conditioning is a canvas-only, model-level computation with no encoder equivalent.");
    }

    private string BuildSelfConditioningOnlyPeftFixture(int rank, float alpha, int hidden, int ff)
    {
        string dir = Path.Combine(_scratch, $"adapter-sc-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);

        var cfgObj = new
        {
            r = rank,
            lora_alpha = alpha,
            target_modules = new[] { "gate_proj", "up_proj", "down_proj" },
            lora_dropout = 0.0,
            bias = "none",
            task_type = "CAUSAL_LM",
            use_rslora = false,
            use_dora = false,
        };
        File.WriteAllText(Path.Combine(dir, "adapter_config.json"), JsonSerializer.Serialize(cfgObj));

        var rng = new Random(555);
        string p = "base_model.model.model.decoder.self_conditioning";
        var b = new SafetensorsFixtureBuilder();
        b.AddFloat32($"{p}.gate_proj.lora_A.weight", [rank, hidden], RandomVec(rng, rank * hidden, 0.1f));
        b.AddFloat32($"{p}.gate_proj.lora_B.weight", [ff, rank], RandomVec(rng, ff * rank, 0.1f));
        b.AddFloat32($"{p}.up_proj.lora_A.weight", [rank, hidden], RandomVec(rng, rank * hidden, 0.1f));
        b.AddFloat32($"{p}.up_proj.lora_B.weight", [ff, rank], RandomVec(rng, ff * rank, 0.1f));
        b.AddFloat32($"{p}.down_proj.lora_A.weight", [rank, ff], RandomVec(rng, rank * ff, 0.1f));
        b.AddFloat32($"{p}.down_proj.lora_B.weight", [hidden, rank], RandomVec(rng, hidden * rank, 0.1f));
        b.WriteTo(Path.Combine(dir, "adapter_model.safetensors"));
        return dir;
    }

    private static float[] RandomVec(Random rng, int n, float scale)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++)
            v[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        return v;
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        float maxDiff = 0f;
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++)
        {
            float d = MathF.Abs(a[i] - b[i]);
            if (d > maxDiff) maxDiff = d;
        }
        return maxDiff;
    }

    private static unsafe (float[] promptRows, float[] canvasRows) SplitRows(ITensor logits, int promptLen, int canvasLen)
    {
        int vocab = logits.Shape[1];
        float* p = (float*)logits.DataPointer;
        var prompt = new float[promptLen * vocab];
        var canvas = new float[canvasLen * vocab];
        new ReadOnlySpan<float>(p, promptLen * vocab).CopyTo(prompt);
        new ReadOnlySpan<float>(p + (long)promptLen * vocab, canvasLen * vocab).CopyTo(canvas);
        return (prompt, canvas);
    }
}
