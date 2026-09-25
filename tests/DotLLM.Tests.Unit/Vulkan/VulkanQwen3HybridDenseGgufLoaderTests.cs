using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Covers the Vulkan GGUF loader path for <see cref="Architecture.Qwen3HybridDense"/>
/// (<c>qwen35</c> — Gated-DeltaNet interleaved with full GQA attention, dense SwiGLU FFN).
/// </summary>
/// <remarks>
/// <para>
/// Before this, <see cref="VulkanModelLoader.CreateFromGguf"/> threw
/// <see cref="NotSupportedException"/> for this architecture even though every kernel it needs
/// — the six GDN kernels, conv1d-causal, GQA attention, SwiGLU, and the whole quantized-matmul
/// family — was already present and parity-tested for the sibling
/// <see cref="VulkanQwen3MoeHybridTransformerModel"/>. Only the host that joins them was
/// missing. Same shape as the Nemotron-H gap in #310; see
/// <see cref="VulkanNemotronHGgufLoaderTests"/>.
/// </para>
/// <para>
/// The fixture's default layout puts a GDN layer at index 0 and a full-attention layer at
/// index 1, so both <see cref="HybridLayerKind"/> branches are exercised — a trunk that was
/// all one kind could not discriminate a mis-wired layer dispatch. Parity is asserted at
/// <c>seqLen &gt; 1</c> <b>and</b> <c>seqLen == 1</c> because
/// <c>RecordMatmul</c> selects a different kernel for each (GEMM vs GEMV), as does the
/// attention path (flash vs per-token).
/// </para>
/// <para>
/// All fixture tensors are F32, so this validates tensor-name mapping, device upload and the
/// forward graph — not quant handling.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen3HybridDenseGgufLoaderTests
{
    private const int VocabSize = SyntheticQwen35HybridDenseMtpGguf.VocabSize;

    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    /// <summary>
    /// The regression that matters: the shared Vulkan dispatch point must not reject a
    /// <c>qwen35</c> GGUF. Asserts on the concrete type too — falling through to
    /// <c>VulkanTransformerModel</c> would die on <c>blk.0.attn_output.weight</c>, which a
    /// GDN layer does not have.
    /// </summary>
    [SkippableFact]
    public void CreateFromGguf_Qwen3HybridDense_ReturnsVulkanQwen3HybridDenseModel()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = WriteFixture(withMtp: true);
        try
        {
            using var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            Assert.Equal(Architecture.Qwen3HybridDense, config.Architecture);

            using var device = VulkanDevice.Create();
            var (model, kvFactory) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spvDir);
            using (model)
            {
                var dense = Assert.IsType<VulkanQwen3HybridDenseTransformerModel>(model);

                // The KV-cache must be the sparse hybrid one, sized for attention layers
                // only (1 of 2 in the default fixture layout) — not one slot per block.
                Assert.Equal(1, dense.AttentionLayerCount);
                using var kv = kvFactory(8);
                Assert.IsType<VulkanNemotronHKvCache>(kv);

                // GDN layers make this a recurrent model; the scheduler relies on this
                // flag to allocate per-sequence state.
                Assert.True(dense.RequiresPerSequenceState);
                using var gdnState = dense.CreateSequenceState();
                Assert.NotNull(gdnState);
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    /// <summary>
    /// End-to-end prefill: the Vulkan model built from a GGUF produces the same last-token
    /// logits as the CPU model built from the same file. Guards against the loader wiring up
    /// the right type but the wrong weights (transposed dims, wrong tensor in a slot, a missed
    /// upload) — in particular the dense <c>ffn_gate</c>/<c>ffn_up</c>/<c>ffn_down</c> triple,
    /// which is the only sublayer this host does not share with the MoE hybrid.
    /// </summary>
    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void CreateFromGguf_Qwen3HybridDense_PrefillMatchesCpuLoader(bool withMtp)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = WriteFixture(withMtp);
        try
        {
            int[] tokenIds = [1, 2, 3];
            int[] positions = [0, 1, 2];
            AssertVulkanMatchesCpu(path, spvDir, tokenIds, positions);
        }
        finally
        {
            File.Delete(path);
        }
    }

    /// <summary>
    /// Single-token (decode-shaped) parity. Distinct from the prefill case: <c>seqLen == 1</c>
    /// routes every projection through the GEMV kernels rather than the GEMM ones, and skips
    /// the flash-attention branch. A prefill-only test cannot catch a GEMV/GEMM mismatch.
    /// </summary>
    [SkippableFact]
    public void CreateFromGguf_Qwen3HybridDense_SingleTokenMatchesCpuLoader()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = WriteFixture(withMtp: true);
        try
        {
            AssertVulkanMatchesCpu(path, spvDir, [5], [0]);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static void AssertVulkanMatchesCpu(
        string path, string spvDir, int[] tokenIds, int[] positions)
    {
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

    private static string WriteFixture(bool withMtp)
    {
        string path = Path.Combine(Path.GetTempPath(), $"qwen35-dense-{Guid.NewGuid():N}.gguf");
        return SyntheticQwen35HybridDenseMtpGguf.Write(path, withMtp: withMtp);
    }
}
