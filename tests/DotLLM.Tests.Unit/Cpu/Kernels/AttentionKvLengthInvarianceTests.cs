using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #525: a query position's attention output must depend only on the keys that position can
/// actually see — never on how many further entries happen to be resident in the KV cache.
/// <para>
/// The broken form reduced each softmax row (and the weighted-value accumulation) over the padded
/// <c>seqKv</c>. <c>-inf</c> padding contributes exactly <c>0.0</c> mathematically, but changing the
/// span length changes SIMD lane assignment and the remainder tail, so the *real* elements
/// accumulate in a different order and the last ULP of the softmax denominator moves with the cache
/// length. On a quantized model that ULP is digitized by Q8_0 activation quantization and amplified
/// by <c>o_proj</c> until greedy decoding emits a different token.
/// </para>
/// <para>
/// A second, independent form of the same defect lived in the dense-vs-tiled dispatch, which was
/// taken on <c>seqQ * seqKv</c>: the same visible row was computed by a one-shot softmax at one
/// cache length and by an online (tiled) softmax at another — different arithmetic, not just a
/// different reduction order.
/// </para>
/// <para>
/// These assertions are <b>bit-exact</b> on purpose. Padding-invariance is exact by construction
/// once the padding is never touched, and a tolerance would let the defect back in: the whole bug
/// is a ULP that survives to become a token flip. Deltas measured on the broken form were
/// ~1.19E-07 (padding) and ~1.27E-07 (dispatch).
/// </para>
/// </summary>
public sealed unsafe class AttentionKvLengthInvarianceTests : IDisposable
{
    private const int NumHeads = 9;
    private const int NumKvHeads = 3;
    private const int HeadDim = 64;

    private readonly ComputeThreadPool _pool = new(4);

    public void Dispose() => _pool.Dispose();

    private static float[] Random(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    /// <summary>
    /// Runs attention over the first <paramref name="seqKv"/> rows of a shared K/V buffer. Only the
    /// declared cache length varies between arms — the visible keys are byte-identical.
    /// </summary>
    private static float[] Run(float[] q, float[] kv, int seqQ, int seqKv, int positionOffset,
                               int? slidingWindowSize = null, float softCap = 0f,
                               float[]? alibiSlopes = null, float[]? sinks = null,
                               AttentionMaskMode maskMode = AttentionMaskMode.Causal)
    {
        var output = new float[seqQ * NumHeads * HeadDim];
        var kvSlice = kv.AsSpan(0, seqKv * NumKvHeads * HeadDim);
        Attention.Execute(q, kvSlice, kvSlice, output,
                          seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, positionOffset,
                          1.0f / MathF.Sqrt(HeadDim),
                          alibiSlopes ?? [],
                          slidingWindowSize, softCap, maskMode, 0,
                          sinks ?? []);
        return output;
    }

    private static void AssertBitIdentical(float[] expected, float[] actual, string what)
    {
        int differing = 0;
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                differing++;
            worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]));
        }

        Assert.True(differing == 0,
            $"{what}: {differing}/{expected.Length} outputs changed with KV-cache length " +
            $"(worst |delta| = {worst:E3}). Attention must reduce over the visible prefix only (#525).");
    }

    /// <summary>
    /// Prefill shape: a 3-row chunk at offset 0 must give the same rows whether the cache declares
    /// 3 entries or 3+pad. This is the exact shape the chunked-prefill divergence was traced to
    /// (a 5-token prompt split [3,2] attends position 2 with a 3-entry cache, [5] with a 5-entry one).
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void Prefill_RowsAreInvariantToTrailingKvPadding(int pad)
    {
        const int seqQ = 3;
        float[] q = Random(seqQ * NumHeads * HeadDim, 1);
        float[] kv = Random(64 * NumKvHeads * HeadDim, 2);

        AssertBitIdentical(Run(q, kv, seqQ, seqKv: 3, positionOffset: 0),
                           Run(q, kv, seqQ, seqKv: 3 + pad, positionOffset: 0),
                           $"causal prefill, pad={pad}");
    }

    /// <summary>
    /// Chunk-boundary shape: the same absolute query positions, reached either as one 5-row pass or
    /// as a 2-row pass resuming at offset 3 with a 3-entry cache. The second arm is what the
    /// <c>[3,2]</c> chunked prefill actually issues.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void ResumedChunk_RowsAreInvariantToTrailingKvPadding(int pad)
    {
        const int positionOffset = 3;
        const int seqQ = 2;
        float[] q = Random(seqQ * NumHeads * HeadDim, 7);
        float[] kv = Random(64 * NumKvHeads * HeadDim, 8);

        AssertBitIdentical(Run(q, kv, seqQ, seqKv: 5, positionOffset),
                           Run(q, kv, seqQ, seqKv: 5 + pad, positionOffset),
                           $"resumed chunk at offset 3, pad={pad}");
    }

    /// <summary>
    /// Decode shape: one query at a fixed position must be invariant to how much cache sits beyond
    /// it (speculative verify windows and prefix-cache hits both over-declare the cache).
    /// </summary>
    [Theory]
    [InlineData(17, 3)]
    [InlineData(64, 9)]
    [InlineData(255, 64)]
    public void Decode_RowIsInvariantToTrailingKvPadding(int positionOffset, int pad)
    {
        float[] q = Random(NumHeads * HeadDim, 11);
        float[] kv = Random(512 * NumKvHeads * HeadDim, 12);

        AssertBitIdentical(Run(q, kv, seqQ: 1, seqKv: positionOffset + 1, positionOffset),
                           Run(q, kv, seqQ: 1, seqKv: positionOffset + 1 + pad, positionOffset),
                           $"decode at position {positionOffset}, pad={pad}");
    }

    /// <summary>
    /// The dense/tiled dispatch used to key off <c>seqQ * seqKv * sizeof(float) &lt;= 8192</c>, so a
    /// three-row prefill was computed by one-shot softmax at <c>seqKv = 682</c> and by online
    /// (tiled) softmax at <c>seqKv = 683</c> — same visible keys, different formula. Every real
    /// prompt crosses that boundary, so this arm is not a corner case.
    /// </summary>
    [Fact]
    public void Prefill_RowsAreInvariantAcrossTheDenseTiledDispatchBoundary()
    {
        const int seqQ = 3;
        float[] q = Random(seqQ * NumHeads * HeadDim, 3);
        float[] kv = Random(700 * NumKvHeads * HeadDim, 4);

        AssertBitIdentical(Run(q, kv, seqQ, seqKv: 682, positionOffset: 0),
                           Run(q, kv, seqQ, seqKv: 683, positionOffset: 0),
                           "dense(682) vs tiled(683)");
    }

    /// <summary>Same boundary, decode shape (<c>seqQ = 1</c> flips at <c>seqKv = 2048</c>).</summary>
    [Fact]
    public void Decode_RowIsInvariantAcrossTheDenseTiledDispatchBoundary()
    {
        float[] q = Random(NumHeads * HeadDim, 5);
        float[] kv = Random(2200 * NumKvHeads * HeadDim, 6);

        AssertBitIdentical(Run(q, kv, seqQ: 1, seqKv: 2001, positionOffset: 2000),
                           Run(q, kv, seqQ: 1, seqKv: 2100, positionOffset: 2000),
                           "decode dense(2001) vs tiled(2100)");
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void SlidingWindow_RowsAreInvariantToTrailingKvPadding(int window)
    {
        const int seqQ = 3;
        float[] q = Random(seqQ * NumHeads * HeadDim, 13);
        float[] kv = Random(64 * NumKvHeads * HeadDim, 14);

        AssertBitIdentical(Run(q, kv, seqQ, seqKv: 6, positionOffset: 3, slidingWindowSize: window),
                           Run(q, kv, seqQ, seqKv: 12, positionOffset: 3, slidingWindowSize: window),
                           $"sliding window {window}");
    }

    [Fact]
    public void SoftCap_RowsAreInvariantToTrailingKvPadding()
    {
        const int seqQ = 3;
        float[] q = Random(seqQ * NumHeads * HeadDim, 15);
        float[] kv = Random(64 * NumKvHeads * HeadDim, 16);

        AssertBitIdentical(Run(q, kv, seqQ, seqKv: 3, positionOffset: 0, softCap: 30f),
                           Run(q, kv, seqQ, seqKv: 9, positionOffset: 0, softCap: 30f),
                           "soft-cap");
    }

    [Fact]
    public void Alibi_RowsAreInvariantToTrailingKvPadding()
    {
        const int seqQ = 3;
        float[] q = Random(seqQ * NumHeads * HeadDim, 17);
        float[] kv = Random(64 * NumKvHeads * HeadDim, 18);
        float[] slopes = Random(NumHeads, 19);

        AssertBitIdentical(Run(q, kv, seqQ, seqKv: 3, positionOffset: 0, alibiSlopes: slopes),
                           Run(q, kv, seqQ, seqKv: 8, positionOffset: 0, alibiSlopes: slopes),
                           "ALiBi");
    }

    [Fact]
    public void AttentionSinks_RowsAreInvariantToTrailingKvPadding()
    {
        const int seqQ = 3;
        float[] q = Random(seqQ * NumHeads * HeadDim, 21);
        float[] kv = Random(64 * NumKvHeads * HeadDim, 22);
        float[] sinks = Random(NumHeads, 23);

        AssertBitIdentical(Run(q, kv, seqQ, seqKv: 3, positionOffset: 0, sinks: sinks),
                           Run(q, kv, seqQ, seqKv: 7, positionOffset: 0, sinks: sinks),
                           "attention sinks");
    }

    /// <summary>
    /// The head-parallel worker carries its own copy of the row loop, so it needs its own arm —
    /// the span path passing proves nothing about it.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(5)]
    public void Parallel_PrefillRowsAreInvariantToTrailingKvPadding(int pad)
    {
        const int seqQ = 3;
        float[] q = Random(seqQ * NumHeads * HeadDim, 31);
        float[] kv = Random(64 * NumKvHeads * HeadDim, 32);

        AssertBitIdentical(RunPooled(q, kv, seqQ, seqKv: 3, positionOffset: 0),
                           RunPooled(q, kv, seqQ, seqKv: 3 + pad, positionOffset: 0),
                           $"parallel causal prefill, pad={pad}");
    }

    /// <summary>Same dense/tiled dispatch boundary, through the head-parallel worker.</summary>
    [Fact]
    public void Parallel_PrefillRowsAreInvariantAcrossTheDenseTiledDispatchBoundary()
    {
        const int seqQ = 3;
        float[] q = Random(seqQ * NumHeads * HeadDim, 33);
        float[] kv = Random(700 * NumKvHeads * HeadDim, 34);

        AssertBitIdentical(RunPooled(q, kv, seqQ, seqKv: 682, positionOffset: 0),
                           RunPooled(q, kv, seqQ, seqKv: 683, positionOffset: 0),
                           "parallel dense(682) vs tiled(683)");
    }

    private float[] RunPooled(float[] q, float[] kv, int seqQ, int seqKv, int positionOffset)
    {
        var output = new float[seqQ * NumHeads * HeadDim];
        fixed (float* qp = q, kvp = kv, op = output)
        {
            Attention.Execute(qp, kvp, kvp, op, seqQ, seqKv, NumHeads, NumKvHeads, HeadDim,
                              positionOffset, 1.0f / MathF.Sqrt(HeadDim), _pool);
        }
        return output;
    }

    /// <summary>
    /// Sensitivity control for the harness itself: the comparison must be able to fail. Two
    /// genuinely different cache lengths that change what the query <i>can see</i> (a longer causal
    /// prefix) must produce different output — otherwise a passing invariance arm would prove
    /// nothing.
    /// </summary>
    [Fact]
    public void Harness_DetectsAGenuineChangeInVisibleContext()
    {
        float[] q = Random(NumHeads * HeadDim, 41);
        float[] kv = Random(64 * NumKvHeads * HeadDim, 42);

        // Same seqKv, different query position => a different visible prefix => must differ.
        float[] atPos3 = Run(q, kv, seqQ: 1, seqKv: 8, positionOffset: 3);
        float[] atPos4 = Run(q, kv, seqQ: 1, seqKv: 8, positionOffset: 4);

        bool anyDifferent = false;
        for (int i = 0; i < atPos3.Length && !anyDifferent; i++)
            anyDifferent = BitConverter.SingleToInt32Bits(atPos3[i]) != BitConverter.SingleToInt32Bits(atPos4[i]);

        Assert.True(anyDifferent, "Control failed: the bit-exact comparison cannot detect a real change.");
    }
}
