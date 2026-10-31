using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>Issue #1137: the CPU logits buffer follows the rows the LM head computes, not the batch's token capacity.</summary>
public sealed unsafe class TransformerForwardStateLogitsTests
{
    private static TransformerForwardState New() => new(
        hiddenSize: 64, numHeads: 4, numKvHeads: 2, headDim: 16, intermediateSize: 128,
        vocabSize: 1000, maxSeqLen: 4096, ropeDim: 16, ropeTheta: 10000f);

    [Fact]
    public void LargeBatchCapacity_ReservesNoLogits_UntilTheHeadAsksForRows()
    {
        using var s = New();
        Assert.Equal(0, s.LogitsRowCapacity);
        s.EnsureCapacity(2048);                       // a 2048-token prefill
        Assert.Equal(0, s.LogitsRowCapacity);
        Assert.Equal(0, s.Logits);

        s.EnsureLogitsCapacity(1);                    // last-row-only head
        Assert.Equal(1, s.LogitsRowCapacity);
        Assert.NotEqual(0, s.Logits);
        new Span<float>((void*)s.Logits, 1000).Fill(3f);   // the single row is fully writable

        s.EnsureLogitsCapacity(5);                    // MTP verify, k+1 rows
        Assert.Equal(8, s.LogitsRowCapacity);
        new Span<float>((void*)s.Logits, 5 * 1000).Fill(1f);

        s.EnsureLogitsCapacity(3);                    // never shrinks or reallocates needlessly
        Assert.Equal(8, s.LogitsRowCapacity);
    }
}
