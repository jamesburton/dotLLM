using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The on-device quantization policy shared by both hybrid Vulkan models
/// (<see cref="VulkanQwen3MoeHybridTransformerModel"/> and the dense
/// <see cref="VulkanQwen3HybridDenseTransformerModel"/>).
/// </summary>
/// <remarks>
/// These are pure CPU tests — no Vulkan device is created. They exist because the
/// policy is a hand-maintained table of per-format predicates, and a format missing
/// from it fails silently and expensively: the matrix is widened to F32 at upload
/// instead of throwing. PQ2_0 was absent, which made Bonsai 2 27B ask for ~108 GB of
/// device-local memory for a 7.21 GB checkpoint and die with
/// VK_ERROR_OUT_OF_DEVICE_MEMORY on a heap that looked empty. Nothing in the suite
/// noticed, because every existing test either used a format that was already in the
/// table or was gated on a fixture nobody had.
/// <para>
/// Known gap: <c>VulkanNemotronHWeights</c> carries its own copy of this table (the
/// <c>KeepNative</c> comment says "same matrix as"), and these tests do not reach it. No
/// PQ2_0 NemotronH checkpoint exists, so the two cannot currently disagree in a way that
/// matters — but they can drift silently, and a shared policy type would be the real fix.
/// </para>
/// </remarks>
public class VulkanHybridProjectionUploadPolicyTests
{
    // Bonsai 2 27B's real dense-FFN shape: hidden 5120 -> intermediate 6144.
    // 5120 is a multiple of the 128-weight PQ2_0 group, so the packed path applies.
    private const int InputDim = 5120;
    private const int OutputDim = 6144;

    /// <summary>
    /// The discriminating assertion: a PQ2_0 projection must be sized from the PACKED
    /// row layout, not from <c>outputDim * inputDim * sizeof(float)</c>. Reverting the
    /// <c>KeepPQ2_0</c> arm of <c>KeepNative</c> makes this fail with the F32 size —
    /// which is the exact bug, and is ~15x larger.
    /// </summary>
    [Fact]
    public void Pq2_0Projection_IsSizedPacked_NotWidenedToF32()
    {
        long actual = VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(
            OutputDim, InputDim, QuantizationType.PQ2_0);

        long packed = Dequantize.RowByteSize(InputDim, QuantizationType.PQ2_0) * OutputDim;
        long widened = (long)OutputDim * InputDim * sizeof(float);

        Assert.Equal(packed, actual);
        Assert.NotEqual(widened, actual);

        // Guard the magnitude too, so a future change that keeps it "packed" in some
        // much fatter representation still trips: PQ2_0 is ~2.125 bpw, so the widened
        // form is more than an order of magnitude bigger.
        Assert.True(widened > actual * 10,
            $"expected the F32 form ({widened} B) to dwarf the packed form ({actual} B)");
    }

    /// <summary>
    /// The group-alignment guard is real, not decorative: the packed GEMV/GEMM shaders
    /// read whole 128-element groups, so a contraction axis that is not a multiple of
    /// 128 must still be widened rather than silently truncated.
    /// </summary>
    [Fact]
    public void Pq2_0Projection_WithUnalignedContractionAxis_FallsBackToF32()
    {
        const int unaligned = 5120 + 1;
        long actual = VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(
            OutputDim, unaligned, QuantizationType.PQ2_0);

        Assert.Equal((long)OutputDim * unaligned * sizeof(float), actual);
    }

    /// <summary>
    /// Every format the policy claims to keep packed must actually be sized packed.
    /// A whole-table sweep, so the next format added to the dispatch switch without a
    /// matching upload arm (or vice versa) is caught here rather than by an OOM on a
    /// 27B model.
    /// </summary>
    [Theory]
    [InlineData(QuantizationType.Q8_0, 32)]
    [InlineData(QuantizationType.Q4_K, 256)]
    [InlineData(QuantizationType.Q5_K, 256)]
    [InlineData(QuantizationType.Q6_K, 256)]
    [InlineData(QuantizationType.IQ2_XXS, 256)]
    [InlineData(QuantizationType.IQ2_XS, 256)]
    [InlineData(QuantizationType.IQ2_S, 256)]
    [InlineData(QuantizationType.IQ3_XXS, 256)]
    [InlineData(QuantizationType.IQ3_S, 256)]
    [InlineData(QuantizationType.PQ2_0, 128)]
    public void KeptFormats_AreSizedPacked_OnAnAlignedAxis(QuantizationType qt, int groupSize)
    {
        int k = groupSize * 40; // aligned by construction
        long actual = VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(OutputDim, k, qt);

        Assert.Equal(Dequantize.RowByteSize(k, qt) * OutputDim, actual);
        Assert.True(actual < (long)OutputDim * k * sizeof(float),
            $"{qt} was widened to F32 — it is missing from KeepNative");
    }
}
