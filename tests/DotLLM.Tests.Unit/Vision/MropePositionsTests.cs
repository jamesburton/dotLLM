using DotLLM.Core.Configuration;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Vision;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vision;

/// <summary>M-RoPE position bookkeeping and the interleaved axis mapping (issue #2073); expected values worked by hand from llama.cpp mtmd / ggml IMROPE.</summary>
public sealed class MropePositionsTests
{
    private static ImageSpan Span(int start, int gw, int gh) => new(start, new ImageEmbeddings(gw, gh, 4, "h"));

    [Fact]
    public void TextOnly_AllAxesEqualSlot()
    {
        int[] slots = [0, 1, 2, 3];
        var t = new int[4]; var y = new int[4]; var x = new int[4];
        MropePositions.Fill(slots, [], t, y, x);
        Assert.Equal(slots, t); Assert.Equal(slots, y); Assert.Equal(slots, x);
    }

    [Fact]
    public void Image_UsesGridAxes_AndTextAfterIsShiftedBack()
    {
        // 3 text tokens (slots 0..2), a 3x2 image (6 tokens, slots 3..8), then 2 text tokens (slots 9, 10).
        var img = Span(3, gw: 3, gh: 2);
        int[] slots = [2, 3, 4, 5, 6, 7, 8, 9, 10];
        var t = new int[9]; var y = new int[9]; var x = new int[9];
        MropePositions.Fill(slots, [img], t, y, x);
        // slot 2: text.  Image base b = 3: i=0..5 -> y = 3 + i/3, x = 3 + i%3, t = 3.
        Assert.Equal([2, 3, 3, 3, 3, 3, 3, 6, 7], t);
        Assert.Equal([2, 3, 3, 3, 4, 4, 4, 6, 7], y);
        Assert.Equal([2, 3, 4, 5, 3, 4, 5, 6, 7], x);
        // the text after resumes at b + max(gw, gh) = 6; delta = 6 tokens - 3 = 3
        Assert.Equal(3, MropePositions.TotalDelta([img]));
    }

    [Fact]
    public void TwoImages_AccumulateTheDelta()
    {
        // text(2) img A 2x2 (4 tokens: slots 2..5) text(1: slot 6) img B 2x3 (gw 2, gh 3; slots 7..12) text (slot 13)
        var a = Span(2, 2, 2); var b = Span(7, 2, 3);
        int[] slots = [6, 7, 8, 9, 12, 13];
        var t = new int[6]; var y = new int[6]; var x = new int[6];
        MropePositions.Fill(slots, [a, b], t, y, x);
        // after A (delta 4 - 2 = 2): slot 6 -> 4.  B base = 7 - 2 = 5 (gw 2): slot 8 = i1 (y5, x6), slot 9 = i2 (y6, x5), slot 12 = i5 (y7, x6).
        // after B (delta 6 - 3 = 3, total 5): slot 13 -> 8
        Assert.Equal([4, 5, 5, 5, 5, 8], t);
        Assert.Equal([4, 5, 5, 6, 7, 8], y);
        Assert.Equal([4, 5, 6, 5, 6, 8], x);
        Assert.Equal(5, MropePositions.TotalDelta([a, b]));
    }

    [Fact]
    public void Chunking_DoesNotChangeThePositions()
    {
        var img = Span(3, 4, 4);
        var all = Enumerable.Range(0, 30).ToArray();
        var t = new int[30]; var y = new int[30]; var x = new int[30];
        MropePositions.Fill(all, [img], t, y, x);
        for (int c = 0; c < 30; c += 7)
        {
            int n = Math.Min(7, 30 - c);
            var ct = new int[n]; var cy = new int[n]; var cx = new int[n];
            MropePositions.Fill(all.AsSpan(c, n), [img], ct, cy, cx);
            Assert.Equal(t.AsSpan(c, n).ToArray(), ct); Assert.Equal(y.AsSpan(c, n).ToArray(), cy); Assert.Equal(x.AsSpan(c, n).ToArray(), cx);
        }
    }

    [Fact]
    public void AxisPerPair_IsInterleaved_ForTheQwen3Sections()
    {
        var axis = MropePositions.AxisPerPair([11, 11, 10, 0], 32);
        // pairs cycle t, y, x ... until x runs out (10 x pairs: sectors 2..29), then t, y for sectors 30, 31
        for (int j = 0; j < 30; j++) Assert.Equal((byte)(j % 3), axis[j]);
        Assert.Equal(0, axis[30]); Assert.Equal(1, axis[31]);
        Assert.Equal(11, axis.Count(a => a == 0)); Assert.Equal(11, axis.Count(a => a == 1)); Assert.Equal(10, axis.Count(a => a == 2));
    }

    [Fact]
    public void ApplyMrope_WithEqualAxes_EqualsPlainNeoxRope()
    {
        const int heads = 3, kvHeads = 1, headDim = 16, ropeDim = 8, n = 5;
        var rope = new RoPERotation(new RoPEConfig(Theta: 1e6f, DimensionCount: ropeDim), Architecture.Qwen3HybridDense, ropeDim);
        var rng = new Random(7);
        float[] q0 = Enumerable.Range(0, n * heads * headDim).Select(_ => (float)rng.NextDouble() - 0.5f).ToArray();
        float[] k0 = Enumerable.Range(0, n * kvHeads * headDim).Select(_ => (float)rng.NextDouble() - 0.5f).ToArray();
        int[] pos = [4, 5, 9, 10, 11];
        var q1 = (float[])q0.Clone(); var k1 = (float[])k0.Clone();
        var q2 = (float[])q0.Clone(); var k2 = (float[])k0.Clone();
        rope.Apply(q1, k1, pos, heads, kvHeads, headDim, RoPEType.NeoX);
        rope.ApplyMrope(q2, k2, pos, pos, pos, MropePositions.AxisPerPair([2, 1, 1, 0], ropeDim / 2), heads, kvHeads, headDim);
        Assert.Equal(q1, q2); Assert.Equal(k1, k2);
        // and distinct axes must change the result (the test can fail)
        var q3 = (float[])q0.Clone(); var k3 = (float[])k0.Clone();
        rope.ApplyMrope(q3, k3, pos, pos.Select(p => p + 3).ToArray(), pos, MropePositions.AxisPerPair([2, 1, 1, 0], ropeDim / 2), heads, kvHeads, headDim);
        Assert.NotEqual(q1, q3);
    }
}
