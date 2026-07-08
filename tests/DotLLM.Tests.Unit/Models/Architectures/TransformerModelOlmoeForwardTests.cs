using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// End-to-end coverage for AllenAI OLMoE (<c>allenai/OLMoE-1B-7B-0924</c>,
/// <c>model_type=olmoe</c> / <c>OlmoeForCausalLM</c>) on the CPU forward path.
/// OLMoE reuses the Qwen-MoE tensor layout (<c>mlp.gate</c> router +
/// <c>mlp.experts.{j}.{gate,up,down}_proj</c>) and the Llama-style GQA
/// attention path, so it dispatches to <see cref="Architecture.QwenMoe"/>.
/// <para>
/// The one genuine OLMoE-specific quirk exercised here is the QK-norm: OLMoE
/// applies a SINGLE RMSNorm over the WHOLE Q/K projection
/// (<c>num_heads*head_dim</c> / <c>num_kv_heads*head_dim</c>) before the head
/// split, whereas Qwen3/Gemma apply a per-head RMSNorm. A tiny synthetic
/// checkpoint (config.json + safetensors) is loaded through the full
/// <see cref="ModelLoader.LoadFromSafetensors(string, ThreadingConfig?)"/>
/// path so architecture detection, MoE loading (routed-only, no shared
/// expert), whole-projection QK-norm resolution, and the forward pass are all
/// validated together — no network, no real 13 GB checkpoint.
/// </para>
/// </summary>
public sealed class TransformerModelOlmoeForwardTests : IDisposable
{
    private const int HiddenSize = 16;
    private const int NumLayers = 2;
    private const int NumHeads = 4;
    private const int NumKvHeads = 4;          // OLMoE-0924 uses MHA (kv == q heads)
    private const int HeadDim = HiddenSize / NumHeads; // 4
    private const int VocabSize = 8;
    private const int NumExperts = 6;
    private const int TopK = 2;
    private const int IntermediateSize = 12;   // OLMoE has no moe_intermediate_size

    private readonly string _scratch;

    public TransformerModelOlmoeForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-olmoe-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void LoadFromSafetensors_DetectsQwenMoe_WithRoutedOnlyMoe()
    {
        string dir = WriteFixture("olmoe-detect", seed: 42, qkNormFullProjection: true);

        var (model, source, config) = ModelLoader.LoadFromSafetensors(
            Path.Combine(dir, "model.safetensors"));
        using (model)
        using (source)
        {
            Assert.Equal(Architecture.QwenMoe, config.Architecture);
            Assert.NotNull(config.Moe);
            Assert.Equal(NumExperts, config.Moe!.NumExperts);
            Assert.Equal(TopK, config.Moe.NumExpertsPerTok);
            Assert.Equal(IntermediateSize, config.Moe.MoeIntermediateSize);
            Assert.False(config.Moe.NormTopKProb);
            // Routed-only: no shared expert (the OLMoE MoE→MoTE prerequisite).
            Assert.Null(config.Moe.SharedExpertIntermediateSize);
            Assert.False(config.Moe.HasSharedExpertGate);
        }
    }

    [Fact]
    public void Forward_Olmoe_WholeProjectionQkNorm_FiniteNonDegenerate()
    {
        // Full detect+load+forward. The whole-projection q_norm/k_norm tensors
        // (length num_heads*head_dim == 16) would FAIL to load under the
        // per-head convention (expected head_dim == 4), so a successful finite
        // forward is itself evidence the OLMoE QK-norm quirk is handled.
        string dir = WriteFixture("olmoe-fwd", seed: 7, qkNormFullProjection: true);

        var (model, source, config) = ModelLoader.LoadFromSafetensors(
            Path.Combine(dir, "model.safetensors"));
        using (model)
        using (source)
        {
            Assert.Equal(Architecture.QwenMoe, config.Architecture);

            int[] tokenIds = [0, 1, 2, 3, 4];
            int[] positions = [0, 1, 2, 3, 4];
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);

            Assert.Equal(2, logits.Shape.Rank);
            Assert.Equal(tokenIds.Length, logits.Shape[0]);
            Assert.Equal(VocabSize, logits.Shape[1]);

            var stats = ComputeStats(logits);
            Assert.Equal(stats.TotalCount, stats.FiniteCount);
            Assert.True(stats.StdDev > 0.0f, $"Logits degenerate: std={stats.StdDev}");
        }
    }

    [Fact]
    public void Forward_Olmoe_WholeProjectionQkNorm_DiffersFromPerHeadQkNorm()
    {
        // Discriminative: build two IDENTICAL checkpoints differing ONLY in the
        // q_norm/k_norm tensor length — one whole-projection (OLMoE, length 16)
        // and one per-head (Qwen3-style, length 4) — with the same underlying
        // norm values on their shared leading head. The two must produce
        // measurably different logits, proving the whole-projection RMSNorm is a
        // distinct computation actually taken for OLMoE (not silently collapsed
        // to per-head).
        string full = WriteFixture("olmoe-full", seed: 313, qkNormFullProjection: true);
        string perHead = WriteFixture("olmoe-perhead", seed: 313, qkNormFullProjection: false);

        int[] tokenIds = [0, 1, 2, 3, 4];
        int[] positions = [0, 1, 2, 3, 4];

        float[] fullLogits = RunLogits(Path.Combine(full, "model.safetensors"), tokenIds, positions);
        float[] perHeadLogits = RunLogits(Path.Combine(perHead, "model.safetensors"), tokenIds, positions);

        Assert.All(fullLogits, v => Assert.True(float.IsFinite(v)));
        Assert.All(perHeadLogits, v => Assert.True(float.IsFinite(v)));

        float maxDiff = MaxAbsDiff(fullLogits, perHeadLogits);
        Assert.True(maxDiff > 1e-4f,
            $"Whole-projection vs per-head QK-norm produced no measurable difference (maxDiff={maxDiff}).");
    }

    // ───────────────────────── helpers ─────────────────────────

    private static float[] RunLogits(string modelPath, int[] tokenIds, int[] positions)
    {
        var (model, source, _) = ModelLoader.LoadFromSafetensors(modelPath);
        using (model)
        using (source)
        {
            using ITensor l = model.Forward(tokenIds, positions, deviceId: -1);
            return CopyLogits(l);
        }
    }

    /// <summary>
    /// Writes a synthetic OLMoE checkpoint (config.json + model.safetensors) to
    /// a fresh subdirectory and returns its path. When
    /// <paramref name="qkNormFullProjection"/> is true the q_norm/k_norm tensors
    /// are OLMoE-shaped (length num_heads*head_dim); when false they are
    /// Qwen3-shaped (length head_dim) for the discriminative comparison.
    /// </summary>
    private string WriteFixture(string name, int seed, bool qkNormFullProjection)
    {
        string dir = Path.Combine(_scratch, name);
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "config.json"), BuildConfigJson());

        var b = new SafetensorsFixtureBuilder();
        int qStride = NumHeads * HeadDim;   // = HiddenSize (16)
        int kvStride = NumKvHeads * HeadDim; // = 16
        int qkNormDim = qkNormFullProjection ? qStride : HeadDim;

        AddRand(b, "model.embed_tokens.weight", [VocabSize, HiddenSize], 0.1f, seed + 0);
        AddRand(b, "model.norm.weight", [HiddenSize], amplitude: 0.05f, seed: seed + 1);
        AddRand(b, "lm_head.weight", [VocabSize, HiddenSize], 0.1f, seed + 2);

        for (int i = 0; i < NumLayers; i++)
        {
            int s = seed + 10 * (i + 1);
            string prefix = $"model.layers.{i}";

            // Two-norm layout (input + post-attention/pre-FFN) — OLMoE is not Gemma.
            AddRand(b, $"{prefix}.input_layernorm.weight", [HiddenSize], amplitude: 0.05f, seed: s + 0);
            AddRand(b, $"{prefix}.post_attention_layernorm.weight", [HiddenSize], amplitude: 0.05f, seed: s + 1);

            AddRand(b, $"{prefix}.self_attn.q_proj.weight", [qStride, HiddenSize], 0.4f, s + 2);
            AddRand(b, $"{prefix}.self_attn.k_proj.weight", [kvStride, HiddenSize], 0.4f, s + 3);
            AddRand(b, $"{prefix}.self_attn.v_proj.weight", [kvStride, HiddenSize], 0.4f, s + 4);
            AddRand(b, $"{prefix}.self_attn.o_proj.weight", [HiddenSize, qStride], 0.4f, s + 5);

            // OLMoE QK-norm — whole projection (length 16) vs per-head (length 4).
            // Centered at 1.0 with a strong per-position spread: per-head mode
            // reuses the leading head's 4 weights on EVERY head, whereas
            // whole-projection mode gives each head-position its own weight —
            // so the two normalisation granularities diverge measurably.
            AddRandCentered(b, $"{prefix}.self_attn.q_norm.weight", [qkNormDim], center: 1.0f, amplitude: 0.6f, seed: s + 11);
            AddRandCentered(b, $"{prefix}.self_attn.k_norm.weight", [qkNormDim], center: 1.0f, amplitude: 0.6f, seed: s + 12);

            AddRand(b, $"{prefix}.mlp.gate.weight", [NumExperts, HiddenSize], 0.2f, s + 6);
            for (int e = 0; e < NumExperts; e++)
            {
                int es = s + 100 + 7 * e;
                AddRand(b, $"{prefix}.mlp.experts.{e}.gate_proj.weight", [IntermediateSize, HiddenSize], 0.10f, es + 0);
                AddRand(b, $"{prefix}.mlp.experts.{e}.up_proj.weight", [IntermediateSize, HiddenSize], 0.10f, es + 1);
                AddRand(b, $"{prefix}.mlp.experts.{e}.down_proj.weight", [HiddenSize, IntermediateSize], 0.05f, es + 2);
            }
        }

        b.WriteTo(Path.Combine(dir, "model.safetensors"));
        return dir;
    }

    private static string BuildConfigJson() => $$"""
        {
            "architectures": ["OlmoeForCausalLM"],
            "model_type": "olmoe",
            "hidden_size": {{HiddenSize}},
            "num_hidden_layers": {{NumLayers}},
            "num_attention_heads": {{NumHeads}},
            "num_key_value_heads": {{NumKvHeads}},
            "intermediate_size": {{IntermediateSize}},
            "vocab_size": {{VocabSize}},
            "max_position_embeddings": 64,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
            "clip_qkv": null,
            "num_experts": {{NumExperts}},
            "num_experts_per_tok": {{TopK}},
            "norm_topk_prob": false,
            "tie_word_embeddings": false
        }
        """;

    private static void AddRand(SafetensorsFixtureBuilder b, string name, int[] shape,
                                float amplitude, int seed)
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

    private static void AddRandCentered(SafetensorsFixtureBuilder b, string name, int[] shape,
                                        float center, float amplitude, int seed)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        float[] values = new float[n];
        for (long i = 0; i < n; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            values[i] = center + amplitude * MathF.Cos(phi);
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

    private static unsafe LogitStats ComputeStats(ITensor logits)
    {
        int total = 1;
        for (int i = 0; i < logits.Shape.Rank; i++) total *= logits.Shape[i];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, total);

        int finite = 0;
        double sum = 0, sumSq = 0;
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        foreach (float v in span)
        {
            if (float.IsFinite(v))
            {
                finite++;
                sum += v;
                sumSq += (double)v * v;
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }
        double mean = finite > 0 ? sum / finite : 0.0;
        double variance = finite > 0 ? (sumSq / finite) - (mean * mean) : 0.0;
        double stddev = Math.Sqrt(Math.Max(0.0, variance));
        return new LogitStats(total, finite, (float)mean, (float)stddev, min, max);
    }

    private readonly record struct LogitStats(
        int TotalCount, int FiniteCount, float Mean, float StdDev, float Min, float Max);
}
