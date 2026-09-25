using System.Text.Json;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Regression coverage for a real, data-confirmed gap: <c>RunGemma4Layer</c> (the Gemma-4 /
/// DiffusionGemma text-tower layer path, reached whenever <c>ModelConfig.Gemma4DualFfn</c> is
/// true — i.e. any GGUF-loaded gemma4 / diffusion-gemma model, including the real 26B) used to
/// have NO <c>ApplyLoraDelta</c> call sites at all, making an active <see cref="ILoraAdapter"/>
/// a silent no-op. Earlier safetensors-loaded synthetic fixtures
/// (<see cref="TransformerModelGemma4MoeForwardTests"/>,
/// <see cref="TransformerModelGemma4SafetensorsLoraTests"/>) never exercised this path — the
/// safetensors loader never sets <c>Gemma4LayerWeights</c> regardless of
/// <c>ModelConfig.Architecture</c>; only <c>TransformerWeights.LoadFromGguf</c> does, when
/// <c>Gemma4DualFfn</c> is set. This uses <see cref="SyntheticGemma4Gguf"/> (the same tiny
/// fixture the synthetic Gemma4 regression/harness tests use) to load through the REAL
/// GGUF → <c>Gemma4DualFfn</c> → <c>RunGemma4Layer</c> path and confirms an adapter targeting
/// q_proj/v_proj now changes the forward output (q/k/v/o + dense FFN LoRA wired in
/// <c>RunGemma4Layer</c>/<c>Gemma4DenseFfn</c>). MoE experts and self-conditioning remain
/// unadapted — separate, documented follow-ups.
/// </summary>
public sealed class SyntheticGemma4GgufLoraTests : IDisposable
{
    private readonly string _scratch;

    public SyntheticGemma4GgufLoraTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-gemma4gguf-lora-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void Forward_Gemma4Gguf_WithLoraAdapter_ChangesOutput()
    {
        var cfg = SyntheticGemma4Gguf.Tiny;
        string ggufPath = Path.Combine(_scratch, "gemma4-tiny.gguf");
        SyntheticGemma4Gguf.WriteGemma4(ggufPath, cfg, seed: 0xC0FFEEu);

        // Layer 0 is a SLIDING layer (BlockCount=6, GlobalLayerStride=6 ⇒ only layer 5 is
        // global) — sliding head dim/kv-heads give qOut/kvOut for the LoRA fixture below.
        int qOut = cfg.HeadCount * cfg.SlidingHeadDim;
        int kvOut = cfg.SlidingKvHeads * cfg.SlidingHeadDim;
        string adapterDir = BuildPeftFixture(rank: 4, alpha: 8f, hidden: cfg.HiddenSize,
            qOut: qOut, kvOut: kvOut, targetLayer: 0);

        int[] tokenIds = [2, 5, 9, 13];
        int[] positions = [0, 1, 2, 3];

        var (baseline, gguf1, config1) = ModelLoader.LoadFromGguf(ggufPath, ThreadingConfig.SingleThreaded);
        using var _g1 = gguf1;
        using var _m1 = baseline;
        Assert.True(config1.Gemma4DualFfn, "fixture must route through RunGemma4Layer for this test to be meaningful.");
        using ITensor baselineLogits = baseline.Forward(
            tokenIds, positions, deviceId: -1, kvCache: null, adapter: null, AttentionMaskSpec.Causal);
        float[] withoutAdapter = CopyLogits(baselineLogits);

        var (adapted, gguf2, config2) = ModelLoader.LoadFromGguf(ggufPath, ThreadingConfig.SingleThreaded);
        using var _g2 = gguf2;
        using var _m2 = adapted;
        using var adapter = PeftAdapterLoader.LoadFromDirectory("gap-probe", adapterDir, config2);
        using ITensor adaptedLogits = adapted.Forward(
            tokenIds, positions, deviceId: -1, kvCache: null, adapter, AttentionMaskSpec.Causal);
        float[] withAdapter = CopyLogits(adaptedLogits);

        float maxDiff = MaxAbsDiff(withoutAdapter, withAdapter);

        // RunGemma4Layer now applies LoRA for q/k/v/o + the dense FFN branch (gate/up/down).
        // MoE experts and self-conditioning remain unadapted (documented, separate gaps).
        Assert.True(maxDiff > 1e-4f,
            $"LoRA adapter had no measurable effect on the Gemma-4/DiffusionGemma layer path (maxDiff={maxDiff}).");
    }

    private string BuildPeftFixture(int rank, float alpha, int hidden, int qOut, int kvOut, int targetLayer)
    {
        string dir = Path.Combine(_scratch, $"adapter-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);

        var cfgObj = new
        {
            r = rank,
            lora_alpha = alpha,
            target_modules = new[] { "q_proj", "v_proj" },
            lora_dropout = 0.0,
            bias = "none",
            task_type = "CAUSAL_LM",
            use_rslora = false,
            use_dora = false,
        };
        File.WriteAllText(Path.Combine(dir, "adapter_config.json"), JsonSerializer.Serialize(cfgObj));

        var b = new SafetensorsFixtureBuilder();
        var rng = new Random(123);
        string p = $"base_model.model.model.layers.{targetLayer}.self_attn";
        b.AddFloat32($"{p}.q_proj.lora_A.weight", [rank, hidden], RandomVec(rng, rank * hidden, 0.05f));
        b.AddFloat32($"{p}.q_proj.lora_B.weight", [qOut, rank], RandomVec(rng, qOut * rank, 0.05f));
        b.AddFloat32($"{p}.v_proj.lora_A.weight", [rank, hidden], RandomVec(rng, rank * hidden, 0.05f));
        b.AddFloat32($"{p}.v_proj.lora_B.weight", [kvOut, rank], RandomVec(rng, kvOut * rank, 0.05f));
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

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }
}
