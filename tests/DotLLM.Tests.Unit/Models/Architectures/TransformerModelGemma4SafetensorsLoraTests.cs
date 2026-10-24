using System.Text.Json;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// CORRECTED from an earlier mistaken hypothesis: safetensors-loaded Gemma-4 /
/// DiffusionGemma-architecture models NEVER populate <c>Gemma4LayerWeights</c>
/// (<c>TransformerWeightsSafetensorsLoader.LoadLayer</c> has no Gemma4-specific
/// branch at all — only <c>TransformerWeights.LoadFromGguf</c> does, gated on
/// <c>ModelConfig.Gemma4DualFfn</c>). So a safetensors-loaded "Gemma4" fixture
/// always runs the generic Llama-style layer path, which DOES call
/// <c>ApplyLoraDelta</c> — this test confirms that. The REAL Gemma4/DiffusionGemma
/// layer path (<c>RunGemma4Layer</c>, reached only via GGUF loading) had a genuine,
/// now-fixed LoRA gap — see <see cref="SyntheticGemma4GgufLoraTests"/>.
/// </summary>
public sealed class TransformerModelGemma4SafetensorsLoraTests : IDisposable
{
    private const int HiddenSize = 16;
    private const int NumLayers = 2;
    private const int NumHeads = 4;
    private const int VocabSize = 8;
    private const int HeadDim = HiddenSize / NumHeads; // 4
    private const int NumKvHeads = 4;                  // uniform GQA (no dual full/sliding split needed here)
    private const int IntermediateSize = 12;

    private readonly string _scratch;

    public TransformerModelGemma4SafetensorsLoraTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-gemma4-lora-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void Forward_Gemma4SafetensorsLoaded_WithLoraAdapter_ChangesOutput()
    {
        // Safetensors-loaded Architecture.Gemma4 never populates Gemma4LayerWeights, so this
        // exercises the generic Llama-style layer path (ApplyLoraDelta IS called there).
        string modelPath = Path.Combine(_scratch, "gemma4-lora.safetensors");
        WriteFixture(modelPath, seed: 7);
        ModelConfig config = BuildConfig();

        int qOut = NumHeads * HeadDim;
        int kvOut = NumKvHeads * HeadDim;
        string adapterDir = BuildPeftFixture(rank: 4, alpha: 8f, hidden: HiddenSize,
            qOut: qOut, kvOut: kvOut, numLayers: NumLayers);

        int[] tokenIds = [0, 1, 2, 3];
        int[] positions = [0, 1, 2, 3];

        using var sf1 = SafetensorsFile.Open(modelPath);
        using var baseline = TransformerModel.LoadFromSafetensors(sf1, config);
        using ITensor baselineLogits = baseline.Forward(tokenIds, positions, deviceId: -1);
        float[] withoutAdapter = CopyLogits(baselineLogits);

        using var sf2 = SafetensorsFile.Open(modelPath);
        using var adapted = TransformerModel.LoadFromSafetensors(sf2, config);
        using var adapter = PeftAdapterLoader.LoadFromDirectory("gap-probe", adapterDir, config);
        using ITensor adaptedLogits = adapted.Forward(
            tokenIds, positions, deviceId: -1, kvCache: null, adapter, AttentionMaskSpec.Causal);
        float[] withAdapter = CopyLogits(adaptedLogits);

        float maxDiff = MaxAbsDiff(withoutAdapter, withAdapter);

        Assert.True(maxDiff > 1e-4f,
            $"LoRA adapter had no measurable effect via the generic layer path (maxDiff={maxDiff}).");
    }

    // ───────────────────────── helpers ─────────────────────────

    private static ModelConfig BuildConfig()
    {
        var rope = new RoPEConfig(Theta: 10_000.0f, DimensionCount: HeadDim, Type: RoPEType.NeoX);
        return new ModelConfig
        {
            Architecture = Architecture.Gemma4,
            VocabSize = VocabSize,
            HiddenSize = HiddenSize,
            IntermediateSize = IntermediateSize,
            NumLayers = NumLayers,
            NumAttentionHeads = NumHeads,
            NumKvHeads = NumKvHeads,
            HeadDim = HeadDim,
            MaxSequenceLength = 16,
            AttentionType = AttentionType.GQA,
            PositionEncodingType = PositionEncodingType.RoPE,
            RoPEConfig = rope,
            ActivationFunction = ActivationFunction.GELUTanh,
            NormType = NormType.RMSNorm,
            NormEpsilon = 1e-6f,
            TiedEmbeddings = false,
            SlidingWindowSize = null,
            PerLayerSlidingWindow = null,
            FinalLogitSoftcap = 30.0f,
            EmbeddingScale = MathF.Sqrt(HiddenSize),
            ChatTemplate = null,
        };
    }

    /// <summary>Minimal dense (non-MoE, non-PLE) Gemma-4 fixture: enough tensors for
    /// <c>lw.Gemma4</c> to be populated so the forward routes through RunGemma4Layer.</summary>
    private static void WriteFixture(string path, int seed)
    {
        var b = new SafetensorsFixtureBuilder();
        int qStride = NumHeads * HeadDim;
        int kvStride = NumKvHeads * HeadDim;

        AddRand(b, "model.embed_tokens.weight", [VocabSize, HiddenSize], 0.1f, seed + 0);
        AddRand(b, "model.norm.weight", [HiddenSize], 0.05f, seed + 1);
        AddRand(b, "lm_head.weight", [VocabSize, HiddenSize], 0.1f, seed + 2);

        for (int i = 0; i < NumLayers; i++)
        {
            int s = seed + 20 * (i + 1);
            string prefix = $"model.layers.{i}";

            AddRand(b, $"{prefix}.input_layernorm.weight", [HiddenSize], 0.05f, s + 0);
            AddRand(b, $"{prefix}.post_attention_layernorm.weight", [HiddenSize], 0.10f, s + 1);
            AddRand(b, $"{prefix}.pre_feedforward_layernorm.weight", [HiddenSize], 0.05f, s + 2);
            AddRand(b, $"{prefix}.post_feedforward_layernorm.weight", [HiddenSize], 0.10f, s + 3);

            AddRand(b, $"{prefix}.self_attn.q_proj.weight", [qStride, HiddenSize], 0.1f, s + 4);
            AddRand(b, $"{prefix}.self_attn.k_proj.weight", [kvStride, HiddenSize], 0.1f, s + 5);
            AddRand(b, $"{prefix}.self_attn.v_proj.weight", [kvStride, HiddenSize], 0.1f, s + 6);
            AddRand(b, $"{prefix}.self_attn.o_proj.weight", [HiddenSize, qStride], 0.1f, s + 7);
            AddRand(b, $"{prefix}.self_attn.q_norm.weight", [HeadDim], 0.05f, s + 8);
            AddRand(b, $"{prefix}.self_attn.k_norm.weight", [HeadDim], 0.05f, s + 9);

            AddRand(b, $"{prefix}.mlp.gate_proj.weight", [IntermediateSize, HiddenSize], 0.1f, s + 10);
            AddRand(b, $"{prefix}.mlp.up_proj.weight", [IntermediateSize, HiddenSize], 0.1f, s + 11);
            AddRand(b, $"{prefix}.mlp.down_proj.weight", [HiddenSize, IntermediateSize], 0.05f, s + 12);
        }

        b.WriteTo(path);
    }

    /// <summary>Fabricates a byte-accurate PEFT adapter directory targeting q_proj + v_proj
    /// on every layer (mirrors PeftAdapterLoaderTests' fixture builder).</summary>
    private string BuildPeftFixture(int rank, float alpha, int hidden, int qOut, int kvOut, int numLayers)
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
        for (int i = 0; i < numLayers; i++)
        {
            string p = $"base_model.model.model.layers.{i}.self_attn";
            b.AddFloat32($"{p}.q_proj.lora_A.weight", [rank, hidden], RandomVec(rng, rank * hidden, 0.05f));
            b.AddFloat32($"{p}.q_proj.lora_B.weight", [qOut, rank], RandomVec(rng, qOut * rank, 0.05f));
            b.AddFloat32($"{p}.v_proj.lora_A.weight", [rank, hidden], RandomVec(rng, rank * hidden, 0.05f));
            b.AddFloat32($"{p}.v_proj.lora_B.weight", [kvOut, rank], RandomVec(rng, kvOut * rank, 0.05f));
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

    private static void AddRand(SafetensorsFixtureBuilder b, string name, int[] shape, float amplitude, int seed)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        float[] values = new float[n];
        for (long i = 0; i < n; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            values[i] = amplitude * MathF.Cos(phi);
        }
        b.AddFloat32(name, shape, values);
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
