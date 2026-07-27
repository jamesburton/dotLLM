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
/// Region-aware LoRA: real DiffusionGemma PEFT adapters train INDEPENDENT deltas for the
/// prompt (encoder) and canvas (decoder) rows of the same unified <c>[prompt|canvas]</c>
/// forward (see <see cref="LoraRegion"/>). This proves the split is REAL isolation, not just
/// "some delta landed somewhere": an adapter with ONLY an Encoder-region entry must change
/// prompt-row logits while leaving canvas-row logits BIT-IDENTICAL to the no-adapter baseline
/// (and vice versa for a Decoder-only adapter) — there is no unregioned ("Any") entry for
/// either adapter to fall back to, so any bleed into the wrong region would be a bug.
/// </summary>
public sealed class SyntheticGemma4GgufRegionAwareLoraTests : IDisposable
{
    private readonly string _scratch;

    public SyntheticGemma4GgufRegionAwareLoraTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-gemma4-region-lora-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private const int PromptLen = 3;
    private const int CanvasLen = 4;

    [Theory]
    [InlineData(LoraRegion.Encoder)]
    [InlineData(LoraRegion.Decoder)]
    public void Forward_RegionTaggedAdapter_OnlyAffectsItsOwnRegion(LoraRegion region)
    {
        // A SINGLE layer, deliberately: with >1 layer, a region-only delta at layer 0 still
        // changes that region's hidden state, which subsequent layers' cross-attention (canvas
        // queries attend prompt K/V, and vice versa) legitimately propagates into the OTHER
        // region's output — that's correct DiffusionGemma behaviour, not a region-split bug.
        // One layer isolates the claim this test actually makes: a q_proj delta that only
        // touches Q on its own region's rows cannot affect the other region's output AT ALL
        // (K/V come from k_proj/v_proj, untouched; FFN is row-local) inside a single layer.
        var cfg = SyntheticGemma4Gguf.Tiny with { BlockCount = 1 };
        string ggufPath = Path.Combine(_scratch, "diffusiongemma-tiny.gguf");
        SyntheticGemma4Gguf.WriteDiffusionGemma(ggufPath, cfg, seed: 0xC0FFEEu);

        int qOut = cfg.HeadCount * cfg.SlidingHeadDim; // layer 0 is sliding
        string adapterDir = BuildRegionOnlyPeftFixture(rank: 4, alpha: 8f, hidden: cfg.HiddenSize,
            qOut: qOut, targetLayer: 0, region);

        int seqLen = PromptLen + CanvasLen;
        int[] tokenIds = new int[seqLen];
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) { tokenIds[i] = 3 + i; positions[i] = i; }
        var maskSpec = AttentionMaskSpec.Hybrid(PromptLen);

        var (baseline, gguf1, config1) = ModelLoader.LoadFromGguf(ggufPath, ThreadingConfig.SingleThreaded);
        using var _g1 = gguf1;
        using var _m1 = baseline;
        Assert.NotNull(config1.DiffusionConfig);
        using ITensor baselineLogits = baseline.Forward(
            tokenIds, positions, deviceId: -1, kvCache: null, adapter: null, maskSpec);
        var (baselinePrompt, baselineCanvas) = SplitRows(baselineLogits, PromptLen, CanvasLen);

        var (adapted, gguf2, config2) = ModelLoader.LoadFromGguf(ggufPath, ThreadingConfig.SingleThreaded);
        using var _g2 = gguf2;
        using var _m2 = adapted;
        using var adapter = PeftAdapterLoader.LoadFromDirectory("region-probe", adapterDir, config2);
        using ITensor adaptedLogits = adapted.Forward(
            tokenIds, positions, deviceId: -1, kvCache: null, adapter, maskSpec);
        var (adaptedPrompt, adaptedCanvas) = SplitRows(adaptedLogits, PromptLen, CanvasLen);

        float promptDiff = MaxAbsDiff(baselinePrompt, adaptedPrompt);
        float canvasDiff = MaxAbsDiff(baselineCanvas, adaptedCanvas);

        if (region == LoraRegion.Encoder)
        {
            Assert.True(promptDiff > 1e-4f, $"Encoder-region adapter had no effect on prompt rows (diff={promptDiff}).");
            Assert.True(canvasDiff == 0f, $"Encoder-region adapter leaked into canvas rows (diff={canvasDiff}).");
        }
        else
        {
            Assert.True(canvasDiff > 1e-4f, $"Decoder-region adapter had no effect on canvas rows (diff={canvasDiff}).");
            Assert.True(promptDiff == 0f, $"Decoder-region adapter leaked into prompt rows (diff={promptDiff}).");
        }
    }

    private string BuildRegionOnlyPeftFixture(int rank, float alpha, int hidden, int qOut,
                                              int targetLayer, LoraRegion region)
    {
        string dir = Path.Combine(_scratch, $"adapter-{region}-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);

        var cfgObj = new
        {
            r = rank,
            lora_alpha = alpha,
            target_modules = new[] { "q_proj" },
            lora_dropout = 0.0,
            bias = "none",
            task_type = "CAUSAL_LM",
            use_rslora = false,
            use_dora = false,
        };
        File.WriteAllText(Path.Combine(dir, "adapter_config.json"), JsonSerializer.Serialize(cfgObj));

        string regionInfix = region == LoraRegion.Encoder ? "encoder.language_model."
            : region == LoraRegion.Decoder ? "decoder."
            : "";
        string p = $"base_model.model.model.{regionInfix}layers.{targetLayer}.self_attn";

        var b = new SafetensorsFixtureBuilder();
        var rng = new Random(321);
        b.AddFloat32($"{p}.q_proj.lora_A.weight", [rank, hidden], RandomVec(rng, rank * hidden, 0.1f));
        b.AddFloat32($"{p}.q_proj.lora_B.weight", [qOut, rank], RandomVec(rng, qOut * rank, 0.1f));
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
