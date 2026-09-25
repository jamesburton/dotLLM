using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
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
/// End-to-end parity test for the Vulkan MoE (Mixtral-convention) forward
/// path against the CPU reference. Builds a synthetic tiny Mixtral
/// checkpoint matching the HF tensor naming
/// (<c>block_sparse_moe.gate</c> + <c>experts.{j}.w1/w2/w3</c>), loads it
/// on both backends, runs <c>Forward</c>, and asserts the last-token
/// logits row matches.
/// </summary>
/// <remarks>
/// All MoE projections in the fixture are F32, so both backends consume
/// identical weight bytes. The drift comes from F32 reduction-order
/// differences in the per-layer matmul / softmax pipeline; the kernel-level
/// parity tests pin each step at abs 1e-4 / rel 1e-3 individually but the
/// end-to-end logit drift compounds through the layers + LM head, hence
/// the looser 5e-3 absolute / 1e-3 relative bar — same as the MLA forward
/// parity test.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanTransformerModelMoeForwardTests : IDisposable
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

    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly string _scratch;

    public VulkanTransformerModelMoeForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-moe-vk-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void Forward_SingleToken_MatchesCpuReference()
    {
        AssertVulkanMatchesCpu(seqLen: 1, seed: 42);
    }

    [SkippableFact]
    public void Forward_ThreeTokenPrefill_MatchesCpuReference()
    {
        AssertVulkanMatchesCpu(seqLen: 3, seed: 7);
    }

    [SkippableFact]
    public void Forward_PerExpertLora_SingleToken_MatchesCpuReference()
    {
        AssertVulkanMatchesCpu(seqLen: 1, seed: 142, adapterSeed: 19);
    }

    [SkippableFact]
    public void Forward_PerExpertLora_PrefillTiledRows_MatchesCpuReference()
    {
        AssertVulkanMatchesCpu(seqLen: 16, seed: 207, adapterSeed: 23);
    }

    private void AssertVulkanMatchesCpu(int seqLen, int seed, int? adapterSeed = null)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = Path.Combine(_scratch, $"moe-s{seqLen}.safetensors");
        WriteFixture(path, seed);

        ModelConfig config = BuildConfig();

        int[] tokenIds = new int[seqLen];
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++)
        {
            tokenIds[i] = i % VocabSize;
            positions[i] = i;
        }

        // ── CPU oracle ────────────────────────────────────────────────
        float[] cpuLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            using var model = TransformerModel.LoadFromSafetensors(sf, config);
            using var adapter = adapterSeed is int s ? BuildPerExpertAdapter(config, s) : null;
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1,
                kvCache: null, adapter: adapter);
            cpuLogits = CopyLogits(logits);
        }

        // ── Vulkan under test ─────────────────────────────────────────
        float[] vkLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            using var model = VulkanTransformerModel.LoadFromSafetensors(sf, config, spvDir);
            using var adapter = adapterSeed is int s ? BuildPerExpertAdapter(config, s) : null;
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1,
                kvCache: null, adapter: adapter);
            // Vulkan returns single-row [1, vocab] of last-token logits.
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
            Architecture = Architecture.Mixtral,
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

            // GQA self-attention (standard Llama-style names).
            AddRand(b, $"{prefix}.self_attn.q_proj.weight",
                [NumHeads * HeadDim, HiddenSize], 0.05f, s + 2);
            AddRand(b, $"{prefix}.self_attn.k_proj.weight",
                [NumKvHeads * HeadDim, HiddenSize], 0.05f, s + 3);
            AddRand(b, $"{prefix}.self_attn.v_proj.weight",
                [NumKvHeads * HeadDim, HiddenSize], 0.05f, s + 4);
            AddRand(b, $"{prefix}.self_attn.o_proj.weight",
                [HiddenSize, NumHeads * HeadDim], 0.05f, s + 5);

            // Mixtral MoE FFN: router gate + (w1, w2, w3) per expert.
            AddRand(b, $"{prefix}.block_sparse_moe.gate.weight",
                [NumExperts, HiddenSize], 0.1f, s + 6);
            for (int e = 0; e < NumExperts; e++)
            {
                AddRand(b, $"{prefix}.block_sparse_moe.experts.{e}.w1.weight",
                    [IntermediateSize, HiddenSize], 0.1f, s + 10 + e * 3);
                AddRand(b, $"{prefix}.block_sparse_moe.experts.{e}.w2.weight",
                    [HiddenSize, IntermediateSize], 0.1f, s + 11 + e * 3);
                AddRand(b, $"{prefix}.block_sparse_moe.experts.{e}.w3.weight",
                    [IntermediateSize, HiddenSize], 0.1f, s + 12 + e * 3);
            }
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

    private static unsafe LoraAdapter BuildPerExpertAdapter(ModelConfig config, int seed)
    {
        const int rank = 2;
        var targets = new List<string>();
        for (int e = 0; e < NumExperts; e++)
        {
            targets.Add($"mlp.experts.{e}.gate_proj");
            targets.Add($"mlp.experts.{e}.up_proj");
            targets.Add($"mlp.experts.{e}.down_proj");
        }

        var adapter = new LoraAdapter("vk-moe-lora", rank, alpha: 4f, targetModules: targets);
        for (int layer = 0; layer < config.NumLayers; layer++)
        {
            for (int e = 0; e < NumExperts; e++)
            {
                AddLora(adapter, layer, $"mlp.experts.{e}.gate_proj",
                    inputDim: HiddenSize, outputDim: IntermediateSize, rank, seed + layer * 100 + e * 10 + 1);
                AddLora(adapter, layer, $"mlp.experts.{e}.up_proj",
                    inputDim: HiddenSize, outputDim: IntermediateSize, rank, seed + layer * 100 + e * 10 + 2);
                AddLora(adapter, layer, $"mlp.experts.{e}.down_proj",
                    inputDim: IntermediateSize, outputDim: HiddenSize, rank, seed + layer * 100 + e * 10 + 3);
            }
        }

        Assert.True(adapter.IsCompatible(config));
        return adapter;
    }

    private static unsafe void AddLora(
        LoraAdapter adapter,
        int layer,
        string projection,
        int inputDim,
        int outputDim,
        int rank,
        int seed)
    {
        nint b = LoraAdapter.AllocAligned((long)rank * inputDim);
        nint a = LoraAdapter.AllocAligned((long)outputDim * rank);
        var bSpan = new Span<float>((void*)b, rank * inputDim);
        var aSpan = new Span<float>((void*)a, outputDim * rank);
        FillSmall(bSpan, seed, 0.002f);
        FillSmall(aSpan, seed + 17, 0.002f);
        adapter.AddLayerWeights(layer, projection, new LoraLayerWeights(
            AHandle: a,
            BHandle: b,
            InputDim: inputDim,
            OutputDim: outputDim));
    }

    private static void FillSmall(Span<float> dst, int seed, float amplitude)
    {
        for (int i = 0; i < dst.Length; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            dst[i] = amplitude * MathF.Sin(phi);
        }
    }
}
