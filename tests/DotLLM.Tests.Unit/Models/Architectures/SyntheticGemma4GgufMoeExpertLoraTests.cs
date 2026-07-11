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
/// Regression coverage for the routed-MoE LoRA gap in <c>Gemma4Moe</c> (the Gemma-4 /
/// DiffusionGemma custom-router MoE branch): it delegates expert compute to the shared
/// <c>MoeSwiGluMlp.ExecuteRoutedFromAssignments</c> kernel, which already applies per-expert
/// LoRA deltas via <c>ApplyLoraDelta(..., ExpertProjectionName(e, proj), ...)</c> — but the
/// Gemma4-specific call site used to hardcode <c>loraAdapter: null, loraLayer: -1</c>, making an
/// active per-expert adapter a silent no-op on this path even though the SAME kernel already
/// worked for the non-Gemma4 MoE architectures (Mixtral/Qwen-MoE call sites in this file, which
/// pass <c>_currentAdapter</c>/<c>layer</c> through for real). This uses
/// <see cref="SyntheticGemma4Gguf"/>'s MoE-enabled <c>Tiny</c> fixture (ExpertCount=8,
/// ExpertUsedCount=2) loaded through the real GGUF → <c>Gemma4DualFfn</c> → <c>RunGemma4Layer</c>
/// → <c>Gemma4Moe</c> path, with a synthetic per-expert PEFT adapter (no real local adapter
/// targets experts — checked: none of the sampled diffusiongemma_* adapter_config.json files
/// have "experts" in target_modules).
/// </summary>
public sealed class SyntheticGemma4GgufMoeExpertLoraTests : IDisposable
{
    private readonly string _scratch;

    public SyntheticGemma4GgufMoeExpertLoraTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-gemma4gguf-moe-lora-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void Forward_Gemma4Gguf_WithPerExpertMoeLoraAdapter_ChangesOutput()
    {
        var cfg = SyntheticGemma4Gguf.Tiny;
        string ggufPath = Path.Combine(_scratch, "gemma4-moe-lora-tiny.gguf");
        SyntheticGemma4Gguf.WriteGemma4(ggufPath, cfg, seed: 0xBADC0DEu);

        // Target EVERY expert on layer 0 — mirrors how a real PEFT MoE export actually
        // looks (target_modules="experts.gate_proj" etc. matches EVERY expert's nn.Linear
        // submodule, producing one lora_A/lora_B pair per expert, not just one). This also
        // sidesteps depending on which specific expert(s) the synthetic router happens to
        // pick for the test tokens — whichever experts get routed to, all of them carry an
        // adapter delta, so the top-k routed set is guaranteed to include an adapted expert.
        // Per-expert index correctness (that "expert 2" and "expert 0" resolve to
        // INDEPENDENT entries, not one collapsed entry) is separately covered by
        // PeftAdapterLoaderTests.LoadFromDirectory_ParsesPerExpertMoeNaming.
        const int targetLayer = 0;
        string adapterDir = BuildPerExpertMoeFixture(rank: 4, alpha: 8f,
            hidden: cfg.HiddenSize, interm: cfg.ExpertFeedForward,
            targetLayer: targetLayer, numExperts: cfg.ExpertCount);

        int[] tokenIds = [2, 5, 9, 13, 20, 21];
        int[] positions = [0, 1, 2, 3, 4, 5];

        var (baseline, gguf1, config1) = ModelLoader.LoadFromGguf(ggufPath, ThreadingConfig.SingleThreaded);
        using var _g1 = gguf1;
        using var _m1 = baseline;
        Assert.True(config1.Gemma4DualFfn, "fixture must route through RunGemma4Layer/Gemma4Moe for this test to be meaningful.");
        Assert.True(config1.Moe is { NumExperts: > 1 }, "fixture must have a routed MoE branch to exercise per-expert LoRA.");
        using ITensor baselineLogits = baseline.Forward(
            tokenIds, positions, deviceId: -1, kvCache: null, adapter: null, AttentionMaskSpec.Causal);
        float[] withoutAdapter = CopyLogits(baselineLogits);

        var (adapted, gguf2, config2) = ModelLoader.LoadFromGguf(ggufPath, ThreadingConfig.SingleThreaded);
        using var _g2 = gguf2;
        using var _m2 = adapted;
        using var adapter = PeftAdapterLoader.LoadFromDirectory("moe-expert-gap-probe", adapterDir, config2);
        using ITensor adaptedLogits = adapted.Forward(
            tokenIds, positions, deviceId: -1, kvCache: null, adapter, AttentionMaskSpec.Causal);
        float[] withAdapter = CopyLogits(adaptedLogits);

        float maxDiff = MaxAbsDiff(withoutAdapter, withAdapter);

        // Gemma4Moe now passes the active adapter + layer index through to
        // MoeSwiGluMlp.ExecuteRoutedFromAssignments instead of hardcoding null/-1.
        Assert.True(maxDiff > 1e-5f,
            $"Per-expert LoRA adapter had no measurable effect on the Gemma-4 routed-MoE branch (maxDiff={maxDiff}).");
    }

    /// <summary>
    /// Builds a synthetic PEFT directory targeting every expert's gate/up/down
    /// projections on one layer, using the standard HF PEFT MoE tensor-name convention
    /// (<c>mlp.experts.{j}.{proj}.lora_{A,B}.weight</c>).
    /// </summary>
    private string BuildPerExpertMoeFixture(int rank, float alpha, int hidden, int interm,
                                            int targetLayer, int numExperts)
    {
        string dir = Path.Combine(_scratch, $"moe-adapter-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);

        var cfgObj = new
        {
            r = rank,
            lora_alpha = alpha,
            target_modules = new[] { "experts.gate_proj", "experts.up_proj", "experts.down_proj" },
            lora_dropout = 0.0,
            bias = "none",
            task_type = "CAUSAL_LM",
            use_rslora = false,
            use_dora = false,
        };
        File.WriteAllText(Path.Combine(dir, "adapter_config.json"), JsonSerializer.Serialize(cfgObj));

        var b = new SafetensorsFixtureBuilder();
        var rng = new Random(321);
        for (int e = 0; e < numExperts; e++)
        {
            string p = $"base_model.model.model.layers.{targetLayer}.mlp.experts.{e}";
            b.AddFloat32($"{p}.gate_proj.lora_A.weight", [rank, hidden], RandomVec(rng, rank * hidden, 0.05f));
            b.AddFloat32($"{p}.gate_proj.lora_B.weight", [interm, rank], RandomVec(rng, interm * rank, 0.05f));
            b.AddFloat32($"{p}.up_proj.lora_A.weight", [rank, hidden], RandomVec(rng, rank * hidden, 0.05f));
            b.AddFloat32($"{p}.up_proj.lora_B.weight", [interm, rank], RandomVec(rng, interm * rank, 0.05f));
            b.AddFloat32($"{p}.down_proj.lora_A.weight", [rank, interm], RandomVec(rng, rank * interm, 0.05f));
            b.AddFloat32($"{p}.down_proj.lora_B.weight", [hidden, rank], RandomVec(rng, hidden * rank, 0.05f));
        }
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
