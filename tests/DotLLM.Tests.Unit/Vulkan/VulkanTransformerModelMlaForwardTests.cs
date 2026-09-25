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
/// End-to-end parity test for the Vulkan MLA (DeepSeek-V2/V3) forward path
/// against the CPU reference. Builds a synthetic tiny MLA checkpoint
/// matching the HF DeepSeek-V2 tensor naming, loads it on both backends,
/// runs <c>Forward</c> on the same tokens / positions, and asserts the
/// logits match within an F32 tolerance.
/// </summary>
/// <remarks>
/// <para>
/// All MLA projections in the fixture are F32, so both backends consume
/// identical weight bytes. The only numerical drift comes from reduction
/// order in the matmul / rmsnorm / attention dot products and from RoPE
/// (CPU uses a precomputed cos/sin table, Vulkan computes per-element with
/// the same theta + frequency formula). The kernel-level parity tests
/// already pin those at abs 1e-4 / rel 1e-3 each; the end-to-end logit
/// drift compounds slightly through the layers + LM head, hence the looser
/// 5e-3 absolute / 1e-3 relative bar here.
/// </para>
/// <para>
/// Mirrors the fixture from
/// <c>TransformerModelMlaForwardTests.WriteFixture</c> — same shapes,
/// same deterministic value generator. The CPU oracle is the existing
/// MLA forward (already pinned by its own parity tests against an HF
/// PyTorch reference), so divergence here means the Vulkan plumbing
/// drifted, not the reference.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanTransformerModelMlaForwardTests : IDisposable
{
    private const int HiddenSize = 16;
    private const int NumLayers = 2;
    private const int NumHeads = 2;
    private const int VocabSize = 8;
    private const int QkNope = 4;
    private const int QkRope = 4;
    private const int VHead = 4;
    private const int KvLoraRank = 8;
    private const int IntermediateSize = 24;

    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly string _scratch;

    public VulkanTransformerModelMlaForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-mla-vk-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void Forward_LoRAQ_SingleToken_MatchesCpuReference()
    {
        AssertVulkanMatchesCpu(qLoraRank: 8, seqLen: 1, seed: 42);
    }

    [SkippableFact]
    public void Forward_MonolithicQ_SingleToken_MatchesCpuReference()
    {
        // V2-Lite shape (q_lora_rank = 0).
        AssertVulkanMatchesCpu(qLoraRank: 0, seqLen: 1, seed: 7);
    }

    [SkippableFact]
    public void Forward_LoRAQ_Prefill_MatchesCpuReference()
    {
        AssertVulkanMatchesCpu(qLoraRank: 8, seqLen: 3, seed: 123);
    }

    private void AssertVulkanMatchesCpu(int qLoraRank, int seqLen, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = Path.Combine(_scratch, $"mla-q{qLoraRank}-s{seqLen}.safetensors");
        WriteFixture(path, qLoraRank, seed);

        ModelConfig config = BuildConfig(qLoraRank);

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
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            cpuLogits = CopyLogits(logits);
        }

        // ── Vulkan under test ─────────────────────────────────────────
        float[] vkLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            using var model = VulkanTransformerModel.LoadFromSafetensors(sf, config, spvDir);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            // Vulkan returns a single-row [1, vocab] of last-token logits.
            // Compare against the matching last-token row of the CPU output.
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
                $"qLoraRank={qLoraRank}, seqLen={seqLen}, col={c}: cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }

    private static ModelConfig BuildConfig(int qLoraRank)
    {
        var mla = new MlaConfig
        {
            KvLoraRank = KvLoraRank,
            QLoraRank = qLoraRank,
            QkNopeHeadDim = QkNope,
            QkRopeHeadDim = QkRope,
            VHeadDim = VHead,
            RopeTheta = 10000.0f,
        };
        var rope = new RoPEConfig(Theta: 10000.0f, DimensionCount: QkNope + QkRope, Type: RoPEType.Norm);

        return new ModelConfig
        {
            Architecture = Architecture.DeepSeekV2,
            VocabSize = VocabSize,
            HiddenSize = HiddenSize,
            IntermediateSize = IntermediateSize,
            NumLayers = NumLayers,
            NumAttentionHeads = NumHeads,
            NumKvHeads = NumHeads,   // MLA is head-parallel on the expanded side
            HeadDim = QkNope + QkRope,
            MaxSequenceLength = 16,
            AttentionType = AttentionType.MLA,
            PositionEncodingType = PositionEncodingType.RoPE,
            RoPEConfig = rope,
            ActivationFunction = ActivationFunction.SiLU,
            NormType = NormType.RMSNorm,
            NormEpsilon = 1e-6f,
            TiedEmbeddings = false,
            MlaConfig = mla,
            Moe = null,
            ChatTemplate = null,
        };
    }

    private static void WriteFixture(string path, int qLoraRank, int seed)
    {
        var b = new SafetensorsFixtureBuilder();
        int qkHead = QkNope + QkRope;
        int qTotal = NumHeads * qkHead;
        int kvADim = KvLoraRank + QkRope;
        int kvBOut = NumHeads * (QkNope + VHead);
        int oInput = NumHeads * VHead;

        AddRand(b, "model.embed_tokens.weight", [VocabSize, HiddenSize], 0.05f, seed + 0);
        AddRand(b, "model.norm.weight", [HiddenSize], 1.0f, seed + 1, center: 1.0f, jitter: 0.05f);
        AddRand(b, "lm_head.weight", [VocabSize, HiddenSize], 0.05f, seed + 2);

        for (int i = 0; i < NumLayers; i++)
        {
            int s = seed + 10 * (i + 1);
            string prefix = $"model.layers.{i}";

            AddRand(b, $"{prefix}.input_layernorm.weight", [HiddenSize],
                    amplitude: 0.05f, seed: s + 0, center: 1.0f, jitter: 0.05f);
            AddRand(b, $"{prefix}.post_attention_layernorm.weight", [HiddenSize],
                    amplitude: 0.05f, seed: s + 1, center: 1.0f, jitter: 0.05f);

            if (qLoraRank > 0)
            {
                AddRand(b, $"{prefix}.self_attn.q_a_proj.weight", [qLoraRank, HiddenSize], 0.1f, s + 2);
                AddRand(b, $"{prefix}.self_attn.q_a_layernorm.weight", [qLoraRank],
                        amplitude: 0.05f, seed: s + 3, center: 1.0f, jitter: 0.05f);
                AddRand(b, $"{prefix}.self_attn.q_b_proj.weight", [qTotal, qLoraRank], 0.1f, s + 4);
            }
            else
            {
                AddRand(b, $"{prefix}.self_attn.q_proj.weight", [qTotal, HiddenSize], 0.1f, s + 2);
            }
            AddRand(b, $"{prefix}.self_attn.kv_a_proj_with_mqa.weight", [kvADim, HiddenSize], 0.1f, s + 5);
            AddRand(b, $"{prefix}.self_attn.kv_a_layernorm.weight", [KvLoraRank],
                    amplitude: 0.05f, seed: s + 6, center: 1.0f, jitter: 0.05f);
            AddRand(b, $"{prefix}.self_attn.kv_b_proj.weight", [kvBOut, KvLoraRank], 0.1f, s + 7);
            AddRand(b, $"{prefix}.self_attn.o_proj.weight", [HiddenSize, oInput], 0.1f, s + 8);

            AddRand(b, $"{prefix}.mlp.gate_proj.weight", [IntermediateSize, HiddenSize], 0.05f, s + 9);
            AddRand(b, $"{prefix}.mlp.up_proj.weight", [IntermediateSize, HiddenSize], 0.05f, s + 10);
            AddRand(b, $"{prefix}.mlp.down_proj.weight", [HiddenSize, IntermediateSize], 0.05f, s + 11);
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
