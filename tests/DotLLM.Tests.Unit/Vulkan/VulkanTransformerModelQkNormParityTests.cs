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
/// Issue #594: the dense <see cref="VulkanTransformerModel"/> must apply the Qwen3 per-head Q/K RMSNorm
/// (after projection, before RoPE) exactly as CPU <c>TransformerModel</c> and CUDA do. It uploaded the
/// weights but never applied them, so Qwen3-4B scored perplexity ~1e5 on Vulkan against 15 on CPU.
/// </summary>
/// <remarks>
/// The fixture is deliberately discriminating: the Q/K norm gains are far from 1 (so skipping the norm, or
/// applying it with the wrong row count, changes the logits by O(1)), and <c>numHeads * headDim (32) != hidden (16)</c>
/// with GQA (4 Q heads / 2 KV heads), the shape family Qwen3-4B has and Qwen2.5/Llama do not.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanTransformerModelQkNormParityTests : IDisposable
{
    private const int HiddenSize = 16;
    private const int NumLayers = 3;
    private const int NumHeads = 4;
    private const int NumKvHeads = 2;
    private const int VocabSize = 8;
    private const int HeadDim = 8;
    private const int IntermediateSize = 24;
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly string _scratch;

    public VulkanTransformerModelQkNormParityTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-qknorm-vk-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableTheory]
    [InlineData(1, 11)]
    [InlineData(5, 42)]
    [InlineData(7, 271)]
    public void Forward_QwenWithQkNorm_MatchesCpuReference(int seqLen, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = Path.Combine(_scratch, $"qknorm-{seed}.safetensors");
        WriteFixture(path, seed);
        ModelConfig config = BuildConfig();

        int[] tokenIds = new int[seqLen];
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) { tokenIds[i] = i % VocabSize; positions[i] = i; }

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
            vkLogits = CopyLogits(logits);
        }

        int lastRow = seqLen - 1;
        for (int c = 0; c < VocabSize; c++)
        {
            float cpu = cpuLogits[lastRow * VocabSize + c];
            float vk = vkLogits[c];
            Assert.True(MathF.Abs(cpu - vk) <= AbsTol + RelTol * MathF.Abs(cpu),
                $"seed={seed}, seqLen={seqLen}, col={c}: cpu={cpu:F6} vs vulkan={vk:F6}");
        }
    }

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = 1;
        for (int i = 0; i < logits.Shape.Rank; i++) total *= logits.Shape[i];
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }

    private static ModelConfig BuildConfig() => new()
    {
        Architecture = Architecture.Qwen,
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
        RoPEConfig = new RoPEConfig(Theta: 10000.0f, DimensionCount: HeadDim, Type: RoPEType.NeoX),
        ActivationFunction = ActivationFunction.SiLU,
        NormType = NormType.RMSNorm,
        NormEpsilon = 1e-6f,
        TiedEmbeddings = false,
        MlaConfig = null,
        Moe = null,
        ChatTemplate = null,
    };

    private static void WriteFixture(string path, int seed)
    {
        var b = new SafetensorsFixtureBuilder();
        int qStride = NumHeads * HeadDim;
        int kvStride = NumKvHeads * HeadDim;

        AddRand(b, "model.embed_tokens.weight", [VocabSize, HiddenSize], 0.3f, seed + 0);
        AddRand(b, "model.norm.weight", [HiddenSize], 0.05f, seed + 1, center: 1.0f, jitter: 0.05f);
        AddRand(b, "lm_head.weight", [VocabSize, HiddenSize], 0.3f, seed + 2);

        for (int i = 0; i < NumLayers; i++)
        {
            int s = seed + 10 * (i + 1);
            string p = $"model.layers.{i}";
            AddRand(b, $"{p}.input_layernorm.weight", [HiddenSize], 0.05f, s + 0, center: 1.0f, jitter: 0.05f);
            AddRand(b, $"{p}.post_attention_layernorm.weight", [HiddenSize], 0.05f, s + 1, center: 1.0f, jitter: 0.05f);
            AddRand(b, $"{p}.self_attn.q_proj.weight", [qStride, HiddenSize], 0.3f, s + 2);
            AddRand(b, $"{p}.self_attn.k_proj.weight", [kvStride, HiddenSize], 0.3f, s + 3);
            AddRand(b, $"{p}.self_attn.v_proj.weight", [kvStride, HiddenSize], 0.3f, s + 4);
            AddRand(b, $"{p}.self_attn.o_proj.weight", [HiddenSize, qStride], 0.3f, s + 5);
            // Per-head Q/K norm gains, far from 1 so a skipped / mis-shaped norm is an O(1) logit error.
            AddRand(b, $"{p}.self_attn.q_norm.weight", [HeadDim], 0.5f, s + 11, center: 1.5f, jitter: 0.5f);
            AddRand(b, $"{p}.self_attn.k_norm.weight", [HeadDim], 0.5f, s + 12, center: 0.6f, jitter: 0.4f);
            AddRand(b, $"{p}.mlp.gate_proj.weight", [IntermediateSize, HiddenSize], 0.3f, s + 6);
            AddRand(b, $"{p}.mlp.up_proj.weight", [IntermediateSize, HiddenSize], 0.3f, s + 7);
            AddRand(b, $"{p}.mlp.down_proj.weight", [HiddenSize, IntermediateSize], 0.3f, s + 8);
        }

        b.WriteTo(path);
    }

    private static void AddRand(SafetensorsFixtureBuilder b, string name, int[] shape,
                                float amplitude, int seed, float center = 0.0f, float jitter = 0.0f)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        float[] values = new float[n];
        for (long i = 0; i < n; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            float cos = MathF.Cos(phi);
            values[i] = jitter > 0f ? center + jitter * cos : amplitude * cos;
        }
        b.AddFloat32(name, shape, values);
    }
}
