using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Covers the Vulkan GGUF *loader* path for hybrid recurrent architectures — the layer the
/// synthetic-fixture forward tests skip entirely because they hand pre-built
/// <c>NemotronHLayerWeights</c> straight to <c>BuildFromPrebuiltWeights</c>.
/// </summary>
/// <remarks>
/// <para>
/// Before this, <see cref="VulkanModelLoader.CreateFromGguf"/> threw
/// <see cref="NotSupportedException"/> for <see cref="Architecture.NemotronH"/>, so every Vulkan
/// entry point (<c>bench</c>, <c>perplexity</c>) refused a real Nemotron-H GGUF even though the
/// kernels, weight-upload, forward pass and KV-cache were all complete and tested. Nothing caught
/// it because kernel coverage and loader coverage were disjoint. This test closes the seam:
/// it builds a tiny but structurally faithful <c>nemotron_h</c> GGUF in memory (SSM + attention +
/// FFN layer, exercising all three <see cref="HybridLayerKind"/> branches and the per-layer
/// <c>head_count_kv</c>/<c>feed_forward_length</c> array convention) and drives it through the
/// real dispatch point, asserting parity with the CPU loader on the same file.
/// </para>
/// <para>
/// All tensors are F32 so this validates tensor-name mapping and device upload, not quant
/// handling — the quant matrix is already covered by the forward-parity tests.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanNemotronHGgufLoaderTests
{
    private const int HiddenSize = 16;
    private const int VocabSize = 8;
    private const int HeadDim = 8;
    private const int NumHeads = 2;
    private const int NumKvHeads = 2;
    private const int IntermediateSize = 24;

    // SSM shape — mirrors the F32 case in VulkanNemotronHTransformerModelForwardTests.
    private const int DInner = 16;
    private const int DConv = 4;
    private const int DState = 8;
    private const int NGroup = 2;
    private const int NHead = 2;

    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    /// <summary>
    /// The regression that matters: the shared Vulkan dispatch point must not reject a
    /// Nemotron-H GGUF. Deliberately asserts on the type too — falling through to
    /// <c>VulkanTransformerModel</c> would "load" nothing useful and then die on
    /// <c>blk.0.attn_output.weight</c>.
    /// </summary>
    [SkippableFact]
    public void CreateFromGguf_NemotronH_ReturnsVulkanNemotronHModel()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = WriteSyntheticNemotronHGguf();
        try
        {
            using var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            Assert.Equal(Architecture.NemotronH, config.Architecture);

            using var device = VulkanDevice.Create();
            var (model, kvFactory) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spvDir);
            using (model)
            {
                Assert.IsType<VulkanNemotronHTransformerModel>(model);
                // The KV-cache factory must be the sparse Nemotron-H one (1 attention layer
                // of 3), not a dense cache sized for every block.
                using var kv = kvFactory(8);
                Assert.IsType<VulkanNemotronHKvCache>(kv);
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    /// <summary>
    /// End-to-end: the Vulkan model built from a GGUF produces the same logits as the CPU model
    /// built from the same GGUF. Guards against the loader wiring up the right *type* but the
    /// wrong weights (transposed dims, wrong tensor for a slot, missed upload).
    /// </summary>
    [SkippableFact]
    public void CreateFromGguf_NemotronH_ForwardMatchesCpuLoader()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = WriteSyntheticNemotronHGguf();
        try
        {
            int[] tokenIds = [1, 2, 3];
            int[] positions = [0, 1, 2];

            float[] cpuLogits;
            using (var gguf = GgufFile.Open(path))
            {
                var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
                using var cpu = ModelLoader.CreateCpuModelFromGguf(gguf, config);
                using ITensor logits = cpu.Forward(tokenIds, positions, deviceId: -1);
                cpuLogits = LastRow(logits, VocabSize);
            }

            float[] vkLogits;
            using (var gguf = GgufFile.Open(path))
            {
                var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
                using var device = VulkanDevice.Create();
                var (model, _) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spvDir);
                using (model)
                {
                    using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
                    vkLogits = LastRow(logits, VocabSize);
                }
            }

            for (int c = 0; c < VocabSize; c++)
            {
                float diff = MathF.Abs(cpuLogits[c] - vkLogits[c]);
                float bar = AbsTol + RelTol * MathF.Abs(cpuLogits[c]);
                Assert.True(diff <= bar,
                    $"col={c}: cpu={cpuLogits[c]:F6} vs vulkan={vkLogits[c]:F6} (|diff|={diff:E3} > {bar:E3})");
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    /// <summary>
    /// Mamba-3 has no GGUF representation at all (no <c>general.architecture</c> value, no tensor
    /// naming convention) — it is safetensors-first on every backend. Pin that this is reported as
    /// such rather than by falling into the dense-transformer default and failing on a missing
    /// attention tensor.
    /// </summary>
    [Fact]
    public void ParseArchitecture_Mamba3_IsNotAGgufArchitecture()
    {
        using var data = new GgufTestData();
        data.AddString("general.architecture", "mamba3");
        string path = data.WriteToTempFile();
        try
        {
            using var gguf = GgufFile.Open(path);
            var ex = Assert.Throws<InvalidDataException>(
                () => GgufModelConfigExtractor.Extract(gguf.Metadata));
            Assert.Contains("mamba3", ex.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static float[] LastRow(ITensor logits, int vocabSize)
    {
        int rows = logits.Shape[0];
        var copy = new float[vocabSize];
        unsafe
        {
            var all = new ReadOnlySpan<float>((void*)logits.DataPointer, rows * vocabSize);
            all.Slice((rows - 1) * vocabSize, vocabSize).CopyTo(copy);
        }
        return copy;
    }

    /// <summary>
    /// Builds a 3-layer <c>nemotron_h</c> GGUF — one SSM, one attention, one FFN block — with
    /// deterministic pseudo-random F32 weights, and writes it to a temp file.
    /// </summary>
    private static string WriteSyntheticNemotronHGguf()
    {
        var ssm = new MambaSsmConfig(DConv, DInner, DState, NGroup, NHead);
        var data = new GgufTestData();
        using var _ = data;

        // Layer kinds are encoded exactly as real Nemotron-H GGUFs do: per-layer Int32 arrays
        // where a non-zero head_count_kv means attention, a non-zero feed_forward_length means
        // FFN, and both-zero means SSM.
        int[] headCountKv = [0, NumKvHeads, 0];
        int[] feedForward = [0, 0, IntermediateSize];

        data.AddString("general.architecture", "nemotron_h")
            .AddUInt32("nemotron_h.embedding_length", HiddenSize)
            .AddUInt32("nemotron_h.block_count", 3)
            .AddUInt32("nemotron_h.attention.head_count", NumHeads)
            .AddInt32Array("nemotron_h.attention.head_count_kv", headCountKv)
            .AddInt32Array("nemotron_h.feed_forward_length", feedForward)
            .AddUInt32("nemotron_h.attention.key_length", HeadDim)
            .AddUInt32("nemotron_h.context_length", 32)
            .AddUInt32("nemotron_h.vocab_size", VocabSize)
            .AddFloat32("nemotron_h.attention.layer_norm_rms_epsilon", 1e-5f)
            .AddUInt32("nemotron_h.rope.dimension_count", HeadDim)
            .AddFloat32("nemotron_h.rope.freq_base", 10000.0f)
            .AddUInt32("nemotron_h.ssm.conv_kernel", DConv)
            .AddUInt32("nemotron_h.ssm.inner_size", DInner)
            .AddUInt32("nemotron_h.ssm.state_size", DState)
            .AddUInt32("nemotron_h.ssm.group_count", NGroup)
            .AddUInt32("nemotron_h.ssm.time_step_rank", NHead);

        int seed = 1234;

        // GGUF dim order is [input, output] for 2-D projections.
        AddF32(data, "token_embd.weight", [HiddenSize, VocabSize], ref seed);
        AddF32(data, "output_norm.weight", [HiddenSize], ref seed, positive: true);
        AddF32(data, "output.weight", [HiddenSize, VocabSize], ref seed);

        for (int i = 0; i < 3; i++)
        {
            // Pre-sublayer norm is named attn_norm on every layer kind, SSM and FFN included.
            AddF32(data, $"blk.{i}.attn_norm.weight", [HiddenSize], ref seed, positive: true);
        }

        // Layer 0 — SSM.
        AddF32(data, "blk.0.ssm_in.weight", [HiddenSize, ssm.InputProjectionDim], ref seed);
        AddF32(data, "blk.0.ssm_conv1d.weight", [DConv, ssm.ConvDim], ref seed);
        AddF32(data, "blk.0.ssm_conv1d.bias", [ssm.ConvDim], ref seed);
        // A is exponentiated as -exp(A) in the scan; keep it in a sane range.
        AddF32(data, "blk.0.ssm_a", [NHead], ref seed, positive: true);
        AddF32(data, "blk.0.ssm_d", [NHead], ref seed);
        AddF32(data, "blk.0.ssm_dt.bias", [NHead], ref seed);
        AddF32(data, "blk.0.ssm_norm.weight", [DInner], ref seed, positive: true);
        AddF32(data, "blk.0.ssm_out.weight", [DInner, HiddenSize], ref seed);

        // Layer 1 — attention.
        AddF32(data, "blk.1.attn_q.weight", [HiddenSize, NumHeads * HeadDim], ref seed);
        AddF32(data, "blk.1.attn_k.weight", [HiddenSize, NumKvHeads * HeadDim], ref seed);
        AddF32(data, "blk.1.attn_v.weight", [HiddenSize, NumKvHeads * HeadDim], ref seed);
        AddF32(data, "blk.1.attn_output.weight", [NumHeads * HeadDim, HiddenSize], ref seed);

        // Layer 2 — non-gated (squared-ReLU) FFN: up + down only, deliberately no ffn_gate.
        AddF32(data, "blk.2.ffn_up.weight", [HiddenSize, IntermediateSize], ref seed);
        AddF32(data, "blk.2.ffn_down.weight", [IntermediateSize, HiddenSize], ref seed);

        return data.WriteToTempFile();
    }

    private static void AddF32(
        GgufTestData data, string name, int[] dims, ref int seed, bool positive = false)
    {
        int count = 1;
        foreach (int d in dims) count *= d;

        var bytes = new byte[count * sizeof(float)];
        Span<float> floats = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(bytes.AsSpan());
        var rng = new Random(seed++);
        for (int i = 0; i < count; i++)
        {
            float v = (float)(rng.NextDouble() - 0.5) * 0.5f;
            floats[i] = positive ? MathF.Abs(v) + 0.5f : v;
        }

        data.AddTensor(name, dims, quantType: 0 /* F32 */, bytes);
    }
}
