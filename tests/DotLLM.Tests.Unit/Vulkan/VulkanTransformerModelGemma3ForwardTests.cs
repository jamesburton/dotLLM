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
/// End-to-end Vulkan-vs-CPU parity for the four Gemma-family mechanisms wired
/// into the dense <see cref="VulkanTransformerModel"/>:
/// <list type="bullet">
///   <item>Attention soft-cap (<see cref="ModelConfig.AttnLogitSoftcap"/>).</item>
///   <item>Per-layer sliding window (<see cref="ModelConfig.PerLayerSlidingWindow"/>).</item>
///   <item>Query-pre-attn scalar override (<see cref="ModelConfig.QueryPreAttnScalar"/>).</item>
///   <item>Final-logit soft-cap (<see cref="ModelConfig.FinalLogitSoftcap"/>).</item>
/// </list>
/// Drives the SAME synthetic Gemma 3 fixture used by the CPU test
/// (<c>TransformerModelGemma3ForwardTests</c>) through both backends; asserts
/// per-element logit parity within the standard Vulkan envelope (abs 5e-3 /
/// rel 1e-3). A failure here means the Vulkan plumbing diverged from CPU on
/// at least one mechanism — see the per-mechanism CPU tests for which
/// mechanism is broken in isolation.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanTransformerModelGemma3ForwardTests : IDisposable
{
    // Fixture geometry mirrors TransformerModelGemma3ForwardTests so the
    // same WriteFixture / BuildConfig pattern produces byte-identical weights.
    private const int HiddenSize = 16;
    private const int NumLayers = 4;
    private const int NumHeads = 2;
    private const int VocabSize = 8;
    private const int HeadDim = HiddenSize / NumHeads; // 8
    private const int IntermediateSize = 24;
    private const int SlidingWindow = 2;

    // Vulkan envelope matching VulkanTransformerModelMlaForwardTests — the
    // attention reductions, RMSNorm tree reduces, and host-side
    // FinalLogitSoftcap together drift around 1e-4 per layer, so the
    // four-layer Gemma-3 fixture lands inside abs 5e-3.
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly string _scratch;

    public VulkanTransformerModelGemma3ForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-gemma3-vk-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void Forward_Gemma3_AllMechanisms_MatchesCpuReference()
    {
        // The canonical Gemma 3 forward: per-layer sliding/full, attn soft-cap,
        // final soft-cap, and query-pre-attn-scalar all active simultaneously.
        AssertVulkanMatchesCpu(
            withAttnSoftcap: 50.0f,
            withFinalSoftcap: 30.0f,
            withQueryPreAttnScalar: HeadDim,
            seqLen: 5, seed: 42);
    }

    [SkippableFact]
    public void Forward_Gemma3_AttnSoftcap_MatchesCpuReference()
    {
        // Discriminative: only AttnLogitSoftcap is set; QPAS and final-cap
        // are null. Pinpoints regressions in the FA-shader softCap path.
        AssertVulkanMatchesCpu(
            withAttnSoftcap: 50.0f,
            withFinalSoftcap: null,
            withQueryPreAttnScalar: null,
            seqLen: 5, seed: 13);
    }

    [SkippableFact]
    public void Forward_Gemma3_PerLayerSlidingWindow_MatchesCpuReference()
    {
        // Discriminative: only the per-layer pattern is set (layers 0,2
        // sliding, 1,3 full). seqLen > SlidingWindow ensures masking fires.
        AssertVulkanMatchesCpu(
            withAttnSoftcap: null,
            withFinalSoftcap: null,
            withQueryPreAttnScalar: null,
            seqLen: 5, seed: 271);
    }

    [SkippableFact]
    public void Forward_Gemma3_QueryPreAttnScalar_MatchesCpuReference()
    {
        // Discriminative: only QPAS is set. Verifies the shader's
        // scaleOverride push-constant routes through end-to-end.
        AssertVulkanMatchesCpu(
            withAttnSoftcap: null,
            withFinalSoftcap: null,
            withQueryPreAttnScalar: 4f * HeadDim,
            seqLen: 5, seed: 7);
    }

    [SkippableFact]
    public void Forward_Gemma3_FinalSoftcap_MatchesCpuReference()
    {
        // Discriminative: only the final-logit soft-cap is set. Verifies
        // the host-side TensorPrimitives.Tanh path matches the CPU's.
        AssertVulkanMatchesCpu(
            withAttnSoftcap: null,
            withFinalSoftcap: 5.0f,
            withQueryPreAttnScalar: null,
            seqLen: 5, seed: 314);
    }

    private void AssertVulkanMatchesCpu(
        float? withAttnSoftcap, float? withFinalSoftcap, float? withQueryPreAttnScalar,
        int seqLen, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = Path.Combine(_scratch,
            $"gemma3-vk-{seed}-asc{withAttnSoftcap}-fsc{withFinalSoftcap}-qpas{withQueryPreAttnScalar}.safetensors");
        WriteFixture(path, seed);

        ModelConfig config = BuildConfig(
            withAttnSoftcap: withAttnSoftcap,
            withFinalSoftcap: withFinalSoftcap,
            withQueryPreAttnScalar: withQueryPreAttnScalar);

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
            // CPU Forward returns [seqLen, vocab]; we compare the last-token row.
            cpuLogits = CopyLogits(logits);
        }

        // ── Vulkan under test ─────────────────────────────────────────
        float[] vkLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            using var model = VulkanTransformerModel.LoadFromSafetensors(sf, config, spvDir);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            Assert.Equal(1, logits.Shape[0]);
            Assert.Equal(VocabSize, logits.Shape[1]);
            vkLogits = CopyLogits(logits);
        }

        // Vulkan returns last-token-only logits ([1, vocab]); CPU returns all
        // rows ([seqLen, vocab]). Compare against the last row of the CPU
        // output, which is what an inference engine would consume for the
        // next-token sampler.
        int lastRow = seqLen - 1;
        for (int c = 0; c < VocabSize; c++)
        {
            float cpu = cpuLogits[lastRow * VocabSize + c];
            float vk = vkLogits[c];
            float diff = MathF.Abs(cpu - vk);
            float bar = AbsTol + RelTol * MathF.Abs(cpu);
            Assert.True(diff <= bar,
                $"seed={seed}, seqLen={seqLen}, asc={withAttnSoftcap}, fsc={withFinalSoftcap}, "
                + $"qpas={withQueryPreAttnScalar}, col={c}: "
                + $"cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    // ───────────────────────── helpers ─────────────────────────

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = 1;
        for (int i = 0; i < logits.Shape.Rank; i++) total *= logits.Shape[i];
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }

    private static ModelConfig BuildConfig(
        float? withAttnSoftcap, float? withFinalSoftcap, float? withQueryPreAttnScalar)
    {
        var rope = new RoPEConfig(
            Theta: 10000.0f,
            DimensionCount: HeadDim,
            Type: RoPEType.NeoX);

        // Per-layer pattern: layers 0 and 2 are sliding (window=2); layers 1
        // and 3 are full. Matches Gemma 3 sliding_window_pattern=2.
        var perLayer = new int?[NumLayers]
        {
            SlidingWindow,
            null,
            SlidingWindow,
            null,
        };

        return new ModelConfig
        {
            Architecture = Architecture.Gemma3,
            VocabSize = VocabSize,
            HiddenSize = HiddenSize,
            IntermediateSize = IntermediateSize,
            NumLayers = NumLayers,
            NumAttentionHeads = NumHeads,
            NumKvHeads = NumHeads,
            HeadDim = HeadDim,
            MaxSequenceLength = 16,
            AttentionType = AttentionType.GQA,
            PositionEncodingType = PositionEncodingType.RoPE,
            RoPEConfig = rope,
            ActivationFunction = ActivationFunction.GELUTanh,
            NormType = NormType.RMSNorm,
            NormEpsilon = 1e-6f,
            TiedEmbeddings = false,
            SlidingWindowSize = SlidingWindow,
            PerLayerSlidingWindow = perLayer,
            AttnLogitSoftcap = withAttnSoftcap,
            FinalLogitSoftcap = withFinalSoftcap,
            QueryPreAttnScalar = withQueryPreAttnScalar,
            MlaConfig = null,
            Moe = null,
            ChatTemplate = null,
        };
    }

    /// <summary>
    /// Synthetic safetensors fixture — same shape generator the CPU Gemma 3
    /// test uses (<see cref="TransformerModelGemma3ForwardTests.WriteFixture"/>).
    /// Byte-identical weights on both backends → any logit divergence comes
    /// from the Vulkan compute path, not weight loading.
    /// </summary>
    private static void WriteFixture(string path, int seed, float lmHeadAmplitude = 0.1f)
    {
        var b = new SafetensorsFixtureBuilder();
        int qStride = NumHeads * HeadDim;
        int kvStride = NumHeads * HeadDim;

        AddRand(b, "model.embed_tokens.weight", [VocabSize, HiddenSize], 0.1f, seed + 0);
        AddRand(b, "model.norm.weight", [HiddenSize], 0.05f, seed + 1, center: 1.0f, jitter: 0.05f);
        AddRand(b, "lm_head.weight", [VocabSize, HiddenSize], lmHeadAmplitude, seed + 2);

        for (int i = 0; i < NumLayers; i++)
        {
            int s = seed + 10 * (i + 1);
            string prefix = $"model.layers.{i}";

            AddRand(b, $"{prefix}.input_layernorm.weight", [HiddenSize],
                    amplitude: 0.05f, seed: s + 0, center: 1.0f, jitter: 0.05f);
            AddRand(b, $"{prefix}.post_attention_layernorm.weight", [HiddenSize],
                    amplitude: 0.05f, seed: s + 1, center: 1.0f, jitter: 0.05f);

            AddRand(b, $"{prefix}.self_attn.q_proj.weight", [qStride, HiddenSize], 0.1f, s + 2);
            AddRand(b, $"{prefix}.self_attn.k_proj.weight", [kvStride, HiddenSize], 0.1f, s + 3);
            AddRand(b, $"{prefix}.self_attn.v_proj.weight", [kvStride, HiddenSize], 0.1f, s + 4);
            AddRand(b, $"{prefix}.self_attn.o_proj.weight", [HiddenSize, qStride], 0.1f, s + 5);

            AddRand(b, $"{prefix}.mlp.gate_proj.weight", [IntermediateSize, HiddenSize], 0.05f, s + 6);
            AddRand(b, $"{prefix}.mlp.up_proj.weight", [IntermediateSize, HiddenSize], 0.05f, s + 7);
            AddRand(b, $"{prefix}.mlp.down_proj.weight", [HiddenSize, IntermediateSize], 0.05f, s + 8);
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
            values[i] = jitter > 0f ? center + jitter * cos : amplitude * cos;
        }
        b.AddFloat32(name, shape, values);
    }
}
