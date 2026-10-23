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
/// End-to-end parity test for the Vulkan MoE shared-expert (DeepSeek-V2/V3
/// ungated) forward path against the CPU reference. Builds a synthetic tiny
/// MoE checkpoint with the Qwen-MoE routed-expert tensor layout
/// (<c>mlp.gate.weight</c> + <c>mlp.experts.{e}.{gate,up,down}_proj.weight</c>)
/// AND an indexed-plural shared-expert layout
/// (<c>mlp.shared_experts.{k}.{gate,up,down}_proj.weight</c>) with no
/// sigmoid gate, loads it on both backends, runs <c>Forward</c>, and asserts
/// the last-token logits row matches.
/// </summary>
/// <remarks>
/// <para>
/// Driven through <see cref="Architecture.QwenMoe"/> as a stand-in for
/// DeepSeek-V2/V3 — the DeepSeek loader on the CPU side requires MLA
/// attention which is wired separately. The MoE plumbing (routed top-k +
/// shared-expert dense SwiGLU + sum) is identical between the two so this
/// fixture exercises the shared-expert math the Vulkan port targets.
/// </para>
/// <para>
/// Tolerance: abs 5e-3 / rel 1e-3 — same bar as
/// <c>VulkanTransformerModelMoeForwardTests</c>. The shared-expert branch
/// adds a small handful of additional matmuls + adds; the per-step parity
/// is unchanged but the compounded F32-reduction-order drift is the same
/// shape as the routed-only path.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanTransformerModelMoeSharedExpertForwardTests : IDisposable
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
    private const int SharedIntermediateSize = 20; // != IntermediateSize on purpose

    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly string _scratch;

    public VulkanTransformerModelMoeSharedExpertForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-moe-shared-vk-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void Forward_SingleSharedExpert_SingleToken_MatchesCpuReference()
    {
        AssertVulkanMatchesCpu(seqLen: 1, numSharedExperts: 1, seed: 42);
    }

    [SkippableFact]
    public void Forward_SingleSharedExpert_ThreeTokenPrefill_MatchesCpuReference()
    {
        AssertVulkanMatchesCpu(seqLen: 3, numSharedExperts: 1, seed: 7);
    }

    [SkippableFact]
    public void Forward_TwoSharedExperts_SingleToken_MatchesCpuReference()
    {
        AssertVulkanMatchesCpu(seqLen: 1, numSharedExperts: 2, seed: 101);
    }

    [SkippableFact]
    public void Forward_TwoSharedExperts_ThreeTokenPrefill_MatchesCpuReference()
    {
        AssertVulkanMatchesCpu(seqLen: 3, numSharedExperts: 2, seed: 17);
    }

    private void AssertVulkanMatchesCpu(int seqLen, int numSharedExperts, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = Path.Combine(_scratch, $"moe-shared-s{seqLen}-n{numSharedExperts}.safetensors");
        WriteFixture(path, seed, numSharedExperts);

        ModelConfig config = BuildConfig(numSharedExperts);

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
                $"seqLen={seqLen}, shared={numSharedExperts}, col={c}: cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }

    private static ModelConfig BuildConfig(int numSharedExperts)
    {
        var rope = new RoPEConfig(Theta: 1_000_000.0f, DimensionCount: HeadDim, Type: RoPEType.Norm);
        return new ModelConfig
        {
            // QwenMoe carries the routed top-k logic + shared-expert branch
            // with the loader naming the Vulkan path consumes. DeepSeek-V2/V3
            // would route through the same MoE math but with MLA attention,
            // which is independently exercised in VulkanTransformerModelMlaForwardTests.
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
                NormTopKProb = true, // Mixtral / Qwen3-MoE convention
                SharedExpertIntermediateSize = SharedIntermediateSize,
                NumSharedExperts = numSharedExperts,
                HasSharedExpertGate = false, // DeepSeek-V2/V3: no sigmoid gate
                DecoderSparseStep = 1,
            },
            ChatTemplate = null,
        };
    }

    private static void WriteFixture(string path, int seed, int numSharedExperts)
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

            // Qwen-MoE FFN: router gate + (gate_proj, up_proj, down_proj) per expert.
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

            // Indexed-plural shared experts (DeepSeek-V2/V3 layout).
            // numSharedExperts >= 2 → the loader takes the indexed-plural branch
            // (`mlp.shared_experts.{k}.*`). For numSharedExperts == 1 the same
            // tensor naming is also accepted by the indexed branch when the
            // fused-plural variant is absent — see LoadQwenMoeLayer.
            for (int k = 0; k < numSharedExperts; k++)
            {
                AddRand(b, $"{prefix}.mlp.shared_experts.{k}.gate_proj.weight",
                    [SharedIntermediateSize, HiddenSize], 0.1f, s + 50 + k * 3);
                AddRand(b, $"{prefix}.mlp.shared_experts.{k}.up_proj.weight",
                    [SharedIntermediateSize, HiddenSize], 0.1f, s + 51 + k * 3);
                AddRand(b, $"{prefix}.mlp.shared_experts.{k}.down_proj.weight",
                    [HiddenSize, SharedIntermediateSize], 0.1f, s + 52 + k * 3);
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
}
