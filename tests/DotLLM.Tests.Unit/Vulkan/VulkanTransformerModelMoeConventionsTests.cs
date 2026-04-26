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
/// End-to-end Vulkan parity tests for the two MoE tensor-naming conventions
/// not already covered by <see cref="VulkanTransformerModelMoeForwardTests"/>:
/// the IBM Granite-3.x fused-experts layout and the Qwen-MoE / Phi-3.5-MoE
/// per-expert layout with a distinct <c>moe_intermediate_size</c>. Both
/// conventions are unpacked by the CPU loader (<c>LoadGraniteMoeLayer</c>
/// and <c>LoadQwenMoeLayer</c>) into the standard
/// <c>MoeLayerWeights.W1[] / W2[] / W3[]</c> per-expert pointer arrays
/// before they reach the Vulkan upload code, so the Vulkan path SHOULD be
/// agnostic to the source naming. These tests prove that.
/// </summary>
/// <remarks>
/// <para>
/// All MoE projections in the fixtures are F32, so both backends consume
/// identical weight bytes. The drift comes from F32 reduction-order
/// differences in the per-layer matmul / softmax pipeline; the kernel-level
/// parity tests pin each step at abs 1e-4 / rel 1e-3 individually but the
/// end-to-end logit drift compounds through the layers + LM head, hence
/// the looser 5e-3 absolute / 1e-3 relative bar — same as the Mixtral MoE
/// and MLA forward parity tests.
/// </para>
/// <para>
/// The Qwen-MoE-style fixture intentionally uses
/// <c>MoeIntermediateSize != IntermediateSize</c> (the Phi-3.5-MoE / Qwen-MoE
/// scenario where the per-expert FFN width differs from the dense
/// intermediate). This validates that both <see cref="VulkanForwardState"/>
/// MoE scratch sizing and <see cref="VulkanWeights.MoeLayerBuffers"/> bank
/// packing use the per-expert width carried on
/// <see cref="MoeLayerWeights.IntermediateSize"/>, not the dense
/// <see cref="ModelConfig.IntermediateSize"/>.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanTransformerModelMoeConventionsTests : IDisposable
{
    private const int HiddenSize = 16;
    private const int NumLayers = 2;
    private const int NumHeads = 2;
    private const int NumKvHeads = 2;
    private const int HeadDim = 8;
    private const int VocabSize = 8;
    private const int NumExperts = 4;
    private const int TopK = 2;

    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly string _scratch;

    public VulkanTransformerModelMoeConventionsTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-moe-conv-vk-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <summary>
    /// Granite-MoE convention: fused per-layer router + input_linear + output_linear
    /// rank-3 tensors. <c>MoeIntermediateSize == IntermediateSize</c> (Granite
    /// reuses the dense intermediate width). 2 layers, 4 experts, top-2.
    /// </summary>
    [SkippableFact]
    public void GraniteMoe_Forward_SingleToken_MatchesCpuReference()
    {
        AssertGraniteMatchesCpu(seqLen: 1, seed: 42);
    }

    [SkippableFact]
    public void GraniteMoe_Forward_ThreeTokenPrefill_MatchesCpuReference()
    {
        AssertGraniteMatchesCpu(seqLen: 3, seed: 7);
    }

    /// <summary>
    /// Qwen-MoE / Phi-3.5-MoE convention: per-expert tensors with
    /// <c>moe_intermediate_size != intermediate_size</c>. Exercises the
    /// loader's <c>LoadQwenMoeLayer</c> path AND validates the Vulkan MoE
    /// scratch / bank-pack sizing uses the per-expert width.
    /// </summary>
    [SkippableFact]
    public void QwenMoe_DistinctMoeIntermediate_Forward_SingleToken_MatchesCpuReference()
    {
        AssertQwenMatchesCpu(seqLen: 1, seed: 11);
    }

    [SkippableFact]
    public void QwenMoe_DistinctMoeIntermediate_Forward_ThreeTokenPrefill_MatchesCpuReference()
    {
        AssertQwenMatchesCpu(seqLen: 3, seed: 23);
    }

    private void AssertGraniteMatchesCpu(int seqLen, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int intermediate = 24; // dense; equals MoeIntermediateSize for Granite

        string path = Path.Combine(_scratch, $"granite-moe-s{seqLen}.safetensors");
        WriteGraniteFixture(path, intermediate, seed);

        ModelConfig config = BuildConfig(
            architecture: Architecture.GraniteMoe,
            denseIntermediate: intermediate,
            moeIntermediate: intermediate);

        AssertParity(path, config, spvDir, seqLen);
    }

    private void AssertQwenMatchesCpu(int seqLen, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        // Distinct per-expert vs dense intermediate widths — Phi-3.5-MoE /
        // Qwen-MoE territory. Both must round-trip cleanly through the
        // VulkanForwardState scratch sizing and the MoE bank pack.
        const int denseIntermediate = 32;
        const int moeIntermediate = 20;

        string path = Path.Combine(_scratch, $"qwen-moe-s{seqLen}.safetensors");
        WriteQwenFixture(path, denseIntermediate, moeIntermediate, seed);

        ModelConfig config = BuildConfig(
            architecture: Architecture.QwenMoe,
            denseIntermediate: denseIntermediate,
            moeIntermediate: moeIntermediate);

        AssertParity(path, config, spvDir, seqLen);
    }

    private static void AssertParity(string path, ModelConfig config, string spvDir, int seqLen)
    {
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

    private static ModelConfig BuildConfig(Architecture architecture, int denseIntermediate, int moeIntermediate)
    {
        var rope = new RoPEConfig(Theta: 1_000_000.0f, DimensionCount: HeadDim, Type: RoPEType.Norm);
        return new ModelConfig
        {
            Architecture = architecture,
            VocabSize = VocabSize,
            HiddenSize = HiddenSize,
            // Top-level intermediate is the dense MLP width. Phi-3.5-MoE / Qwen-MoE
            // can differ from MoeIntermediateSize; for this test path every layer
            // is MoE so the dense IntermediateSize is unused but still required
            // by ModelConfig.
            IntermediateSize = denseIntermediate,
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
                MoeIntermediateSize = moeIntermediate,
            },
            ChatTemplate = null,
        };
    }

    /// <summary>
    /// Writes a synthetic Granite-MoE-shaped checkpoint: standard Llama-style
    /// GQA self-attention plus the fused per-layer MoE tensors:
    /// <c>block_sparse_moe.router.layer.weight [E, H]</c>,
    /// <c>block_sparse_moe.input_linear.weight [E, 2*I, H]</c> (gate rows
    /// [0..I), up rows [I..2*I)), and
    /// <c>block_sparse_moe.output_linear.weight [E, H, I]</c>.
    /// </summary>
    private static void WriteGraniteFixture(string path, int intermediate, int seed)
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

            // Granite MoE: fused per-layer rank-3 tensors.
            // 1. Router gate [E, H]
            AddRand(b, $"{prefix}.block_sparse_moe.router.layer.weight",
                [NumExperts, HiddenSize], 0.1f, s + 6);

            // 2. Fused input_linear [E, 2*I, H] — per expert e, rows [0..I) are
            //    w1 (gate_proj) and rows [I..2*I) are w3 (up_proj). We build
            //    the dense [E, 2*I, H] block whose per-expert slabs contain
            //    distinct gate-vs-up bytes seeded so each per-expert w1/w3
            //    pair yields a non-trivial matmul.
            int twoI = 2 * intermediate;
            float[] inputLinear = new float[(long)NumExperts * twoI * HiddenSize];
            for (int e = 0; e < NumExperts; e++)
            {
                int expertSeed = s + 100 + e * 11;
                long expertOffset = (long)e * twoI * HiddenSize;
                // w1 rows
                FillRand(inputLinear.AsSpan((int)expertOffset, intermediate * HiddenSize),
                    amplitude: 0.1f, seed: expertSeed + 1);
                // w3 rows (offset by intermediate * HiddenSize)
                FillRand(inputLinear.AsSpan((int)(expertOffset + (long)intermediate * HiddenSize),
                    intermediate * HiddenSize),
                    amplitude: 0.1f, seed: expertSeed + 2);
            }
            b.AddFloat32($"{prefix}.block_sparse_moe.input_linear.weight",
                [NumExperts, twoI, HiddenSize], inputLinear);

            // 3. Fused output_linear [E, H, I] — per expert e, the [H, I]
            //    slab is w2 (down_proj). Seeded so each expert slab differs.
            float[] outputLinear = new float[(long)NumExperts * HiddenSize * intermediate];
            for (int e = 0; e < NumExperts; e++)
            {
                int expertSeed = s + 200 + e * 13;
                long expertOffset = (long)e * HiddenSize * intermediate;
                FillRand(outputLinear.AsSpan((int)expertOffset, HiddenSize * intermediate),
                    amplitude: 0.1f, seed: expertSeed);
            }
            b.AddFloat32($"{prefix}.block_sparse_moe.output_linear.weight",
                [NumExperts, HiddenSize, intermediate], outputLinear);
        }

        b.WriteTo(path);
    }

    /// <summary>
    /// Writes a synthetic Qwen-MoE-shaped checkpoint with a distinct
    /// <c>moe_intermediate_size</c> from the dense <c>intermediate_size</c>.
    /// Per-expert tensors live at
    /// <c>mlp.gate.weight</c> + <c>mlp.experts.{e}.{gate_proj,up_proj,down_proj}</c>,
    /// each shaped against <paramref name="moeIntermediate"/> (NOT the dense
    /// <paramref name="denseIntermediate"/>). The dense intermediate is
    /// surfaced through <see cref="ModelConfig.IntermediateSize"/> only so the
    /// CPU/Vulkan path that allocates MoE-layer scratch must consult
    /// <see cref="MoeLayerWeights.IntermediateSize"/>.
    /// </summary>
    private static void WriteQwenFixture(string path, int denseIntermediate, int moeIntermediate, int seed)
    {
        // denseIntermediate is intentionally unused by tensor shapes — every
        // layer in this fixture is MoE so the dense MLP weights are absent.
        // It propagates through ModelConfig.IntermediateSize so the test
        // exercises the "moe_intermediate_size != intermediate_size" wiring.
        _ = denseIntermediate;

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

            // Qwen-MoE FFN: mlp.gate + per-expert {gate,up,down}_proj at the
            // moe_intermediate width.
            AddRand(b, $"{prefix}.mlp.gate.weight",
                [NumExperts, HiddenSize], 0.1f, s + 6);
            for (int e = 0; e < NumExperts; e++)
            {
                AddRand(b, $"{prefix}.mlp.experts.{e}.gate_proj.weight",
                    [moeIntermediate, HiddenSize], 0.1f, s + 10 + e * 3);
                AddRand(b, $"{prefix}.mlp.experts.{e}.down_proj.weight",
                    [HiddenSize, moeIntermediate], 0.1f, s + 11 + e * 3);
                AddRand(b, $"{prefix}.mlp.experts.{e}.up_proj.weight",
                    [moeIntermediate, HiddenSize], 0.1f, s + 12 + e * 3);
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

    /// <summary>
    /// Fills <paramref name="dest"/> with the same deterministic seeded-cosine
    /// pattern as <see cref="AddRand"/>, but in-place so callers can stitch a
    /// fused multi-expert tensor block out of independently-seeded slabs.
    /// </summary>
    private static void FillRand(Span<float> dest, float amplitude, int seed)
    {
        for (int i = 0; i < dest.Length; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            dest[i] = amplitude * MathF.Cos(phi);
        }
    }
}
