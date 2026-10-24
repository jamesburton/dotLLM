using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Regression coverage for the <c>InputQ8Scratch</c> sizing in
/// <see cref="TransformerForwardState"/> (issue #260).
/// </summary>
/// <remarks>
/// <para><c>TransformerModel</c> pre-quantizes each GEMM's activation rows into the single
/// shared <c>InputQ8Scratch</c> buffer. The attention output projection's input is the
/// attention block stride (<c>numHeads × headDim</c>), which for gpt-oss (64 × 64 = 4096)
/// is WIDER than the residual stream (hidden 2880). Sizing the buffer from
/// <c>max(hidden, intermediate)</c> alone therefore under-allocated it and the o_proj
/// pre-quantization wrote past the end of a <c>NativeMemory.AlignedAlloc</c> block —
/// heap corruption (<c>STATUS_HEAP_CORRUPTION</c>) that surfaced at the next allocation.</para>
/// <para>Because <see cref="TransformerForwardState.EnsureCapacity"/> rounds the capacity up
/// to a power of two, the shortfall only materialised at token counts whose rounded capacity
/// did not absorb the ~34% row-size excess — 1..4 tokens crashed while 5 was accidentally
/// safe. The 1..8 sweep below pins every one of those cases.</para>
/// </remarks>
public sealed class TransformerForwardStateScratchTests
{
    // gpt-oss-20b geometry: the discriminating property is numHeads*headDim (4096) > hidden (2880).
    private const int HiddenSize = 2880;
    private const int NumHeads = 64;
    private const int NumKvHeads = 8;
    private const int HeadDim = 64;
    private const int IntermediateSize = 2880;
    private const int VocabSize = 201088;

    /// <summary>Q8_0 packs 32 elements into 34 bytes (Half scale + 32 int8 quants).</summary>
    private static long Q8_0RowBytes(int dim) => (dim / 32) * 34L;

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    public void InputQ8Scratch_CoversAttentionOutputProjectionWidth(int seqLen)
    {
        using var state = NewState();
        state.EnsureCapacity(seqLen);

        // The o_proj pre-quantization writes seqLen rows of (numHeads*headDim) elements.
        long required = seqLen * Q8_0RowBytes(NumHeads * HeadDim);
        Assert.True(state.InputQ8ScratchBytes >= required,
            $"seqLen={seqLen}: InputQ8Scratch is {state.InputQ8ScratchBytes} bytes but the " +
            $"o_proj pre-quantization writes {required} bytes ({NumHeads}×{HeadDim} per token).");
    }

    [Fact]
    public void InputQ8Scratch_StillCoversHiddenAndIntermediateWidths()
    {
        // Guards the fix against regressing the ordinary (hidden/intermediate-widest) case:
        // a model whose FFN intermediate dwarfs both hidden and the attention block.
        using var state = new TransformerForwardState(
            hiddenSize: 512, numHeads: 8, numKvHeads: 8, headDim: 64,
            intermediateSize: 8192, vocabSize: 1024, maxSeqLen: 64,
            ropeDim: 64, ropeTheta: 10000f);
        state.EnsureCapacity(8);

        Assert.True(state.InputQ8ScratchBytes >= 8 * Q8_0RowBytes(8192));
    }

    private static TransformerForwardState NewState() => new(
        hiddenSize: HiddenSize, numHeads: NumHeads, numKvHeads: NumKvHeads, headDim: HeadDim,
        intermediateSize: IntermediateSize, vocabSize: VocabSize, maxSeqLen: 64,
        ropeDim: HeadDim, ropeTheta: 10000f);
}
