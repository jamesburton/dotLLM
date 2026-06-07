using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// End-to-end parity test for the Qwen1.5-MoE forward path on Vulkan —
/// routed top-k experts plus a sigmoid-gated singular shared expert. Built
/// from a synthetic safetensors fixture matching the HF tensor naming
/// (<c>mlp.shared_expert.{gate,up,down}_proj.weight</c> +
/// <c>mlp.shared_expert_gate.weight</c>) and compared against the CPU
/// oracle on the last-token logits row.
/// </summary>
/// <remarks>
/// Qwen1.5-MoE convention specifics: <see cref="MoeConfig.NormTopKProb"/>
/// is <c>false</c> (raw softmax probs as routing weights), exactly one
/// shared expert with the singular tensor name, and a sigmoid gate
/// <c>mlp.shared_expert_gate.weight</c> that scales the shared output by
/// <c>sigmoid(hidden · gate)</c> per token before merging into the routed
/// sum. Tolerance abs 5e-3 / rel 1e-3 — same bar as the other Vulkan MoE
/// forward parity tests; the additional sigmoid stage adds at most a few
/// ULPs over the existing add-only fold.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanTransformerModelMoeQwen15GatedForwardTests : IDisposable
{
    private const int HiddenSize = 16;
    private const int NumLayers = 2;
    private const int NumHeads = 2;
    private const int NumKvHeads = 2;
    private const int HeadDim = 8;
    private const int VocabSize = 8;
    private const int IntermediateSize = 24;
    private const int NumExperts = 4;
    private const int TopK = 2;
    private const int SharedIntermediateSize = 28;

    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly string _scratch;

    public VulkanTransformerModelMoeQwen15GatedForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-moe-qwen15-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void Forward_SingleToken_GatedShared_MatchesCpuReference()
    {
        AssertVulkanMatchesCpu(seqLen: 1, seed: 42);
    }

    [SkippableFact]
    public void Forward_ThreeTokenPrefill_GatedShared_MatchesCpuReference()
    {
        AssertVulkanMatchesCpu(seqLen: 3, seed: 7);
    }

    private void AssertVulkanMatchesCpu(int seqLen, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = Path.Combine(_scratch, $"qwen15-s{seqLen}.safetensors");
        WriteFixture(path, seed);

        ModelConfig config = BuildConfig();

        int[] tokenIds = new int[seqLen];
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++)
        {
            tokenIds[i] = i % VocabSize;
            positions[i] = i;
        }

        float[] cpuLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            using var model = TransformerModel.LoadFromSafetensors(sf, config);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            cpuLogits = CopyLogits(logits);
        }

        float[] vkLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            using var model = VulkanTransformerModel.LoadFromSafetensors(sf, config, spvDir);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            Assert.Equal(1, logits.Shape[0]);
            Assert.Equal(VocabSize, logits.Shape[1]);
            vkLogits = CopyLogits(logits);
        }

        int lastRow = seqLen - 1;
        for (int c = 0; c < VocabSize; c++)
        {
            float cpu = cpuLogits[lastRow * VocabSize + c];
            float vk = vkLogits[c];
            float diff = MathF.Abs(cpu - vk);
            float bar = AbsTol + RelTol * MathF.Abs(cpu);
            Assert.True(diff <= bar,
                $"seqLen={seqLen}, col={c}: cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }

    private static ModelConfig BuildConfig()
    {
        var rope = new RoPEConfig(Theta: 1_000_000.0f, DimensionCount: HeadDim, Type: RoPEType.Norm);
        return new ModelConfig
        {
            Architecture = Architecture.QwenMoe,
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
            ActivationFunction = ActivationFunction.SiLU,
            NormType = NormType.RMSNorm,
            NormEpsilon = 1e-5f,
            TiedEmbeddings = false,
            Moe = new MoeConfig
            {
                NumExperts = NumExperts,
                NumExpertsPerTok = TopK,
                MoeIntermediateSize = IntermediateSize,
                // Qwen1.5-MoE specific: raw softmax probs (no top-k renorm),
                // singular shared expert with sigmoid gate.
                NormTopKProb = false,
                SharedExpertIntermediateSize = SharedIntermediateSize,
                NumSharedExperts = 1,
                HasSharedExpertGate = true,
                DecoderSparseStep = 1,
            },
            ChatTemplate = null,
        };
    }

    private static void WriteFixture(string path, int seed)
    {
        var b = new SafetensorsFixtureBuilder();

        AddRand(b, "model.embed_tokens.weight", [VocabSize, HiddenSize], 0.05f, seed + 0);
        AddRand(b, "model.norm.weight", [HiddenSize], 1.0f, seed + 1, center: 1.0f, jitter: 0.05f);
        AddRand(b, "lm_head.weight", [VocabSize, HiddenSize], 0.05f, seed + 2);

        for (int i = 0; i < NumLayers; i++)
        {
            int s = seed + 100 * (i + 1);
            string prefix = $"model.layers.{i}";

            AddRand(b, $"{prefix}.input_layernorm.weight", [HiddenSize],
                    amplitude: 0.05f, seed: s + 0, center: 1.0f, jitter: 0.05f);
            AddRand(b, $"{prefix}.post_attention_layernorm.weight", [HiddenSize],
                    amplitude: 0.05f, seed: s + 1, center: 1.0f, jitter: 0.05f);

            AddRand(b, $"{prefix}.self_attn.q_proj.weight",
                [NumHeads * HeadDim, HiddenSize], 0.05f, s + 2);
            AddRand(b, $"{prefix}.self_attn.k_proj.weight",
                [NumKvHeads * HeadDim, HiddenSize], 0.05f, s + 3);
            AddRand(b, $"{prefix}.self_attn.v_proj.weight",
                [NumKvHeads * HeadDim, HiddenSize], 0.05f, s + 4);
            AddRand(b, $"{prefix}.self_attn.o_proj.weight",
                [HiddenSize, NumHeads * HeadDim], 0.05f, s + 5);

            // Qwen-MoE routed experts.
            AddRand(b, $"{prefix}.mlp.gate.weight",
                [NumExperts, HiddenSize], 0.1f, s + 6);
            for (int e = 0; e < NumExperts; e++)
            {
                AddRand(b, $"{prefix}.mlp.experts.{e}.gate_proj.weight",
                    [IntermediateSize, HiddenSize], 0.1f, s + 10 + e * 3);
                AddRand(b, $"{prefix}.mlp.experts.{e}.up_proj.weight",
                    [IntermediateSize, HiddenSize], 0.1f, s + 11 + e * 3);
                AddRand(b, $"{prefix}.mlp.experts.{e}.down_proj.weight",
                    [HiddenSize, IntermediateSize], 0.1f, s + 12 + e * 3);
            }

            // Qwen1.5-MoE singular shared expert (no `s`, no index).
            AddRand(b, $"{prefix}.mlp.shared_expert.gate_proj.weight",
                [SharedIntermediateSize, HiddenSize], 0.1f, s + 60);
            AddRand(b, $"{prefix}.mlp.shared_expert.up_proj.weight",
                [SharedIntermediateSize, HiddenSize], 0.1f, s + 61);
            AddRand(b, $"{prefix}.mlp.shared_expert.down_proj.weight",
                [HiddenSize, SharedIntermediateSize], 0.1f, s + 62);

            // Sigmoid gate weight: [1, hidden] in HF, loaded as [hidden].
            // Use small amplitudes so sigmoid(logit) stays in (0, 1) instead
            // of saturating at 0 or 1 — exercises the kernel meaningfully.
            AddRand(b, $"{prefix}.mlp.shared_expert_gate.weight",
                [HiddenSize], 0.1f, s + 70);
        }

        b.WriteTo(path);
    }

    private static void AddRand(SafetensorsFixtureBuilder b, string name, int[] shape,
                                float amplitude, int seed,
                                float center = 0.0f, float jitter = 0.0f)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        float[] values = new float[n];
        for (long i = 0; i < n; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            float cos = MathF.Cos(phi);
            if (jitter > 0f)
                values[i] = center + jitter * cos;
            else
                values[i] = amplitude * cos;
        }
        b.AddFloat32(name, shape, values);
    }
}
