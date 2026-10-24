using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.KvCache;

/// <summary>
/// Compact sliding-window KV (#1169): a cache built with <c>FromConfig(config, compactSlidingWindows: true)</c> keeps only
/// the recent rows of windowed layers. Attention over it must match a full-history cache exactly, across compactions,
/// chunks larger than the window buffer, rollback inside the margin and prefix truncation.
/// </summary>
public sealed unsafe class SlidingWindowKvCacheTests
{
    private const int Window = 17;          // buffer = 2 * (16 + 64) = 160 rows, so the schedule below compacts often
    private const int NumHeads = 4;
    private const int NumKvHeads = 2;
    private const int HeadDim = 32;
    private const int Stride = NumKvHeads * HeadDim;
    private const int MaxSeq = 2048;

    // Layer 0 windowed, layer 1 full (pattern 2, gpt-oss style).
    private static ModelConfig Config() => new()
    {
        Architecture = Architecture.Llama,
        VocabSize = 32,
        HiddenSize = 128,
        IntermediateSize = 256,
        NumLayers = 2,
        NumAttentionHeads = NumHeads,
        NumKvHeads = NumKvHeads,
        HeadDim = HeadDim,
        MaxSequenceLength = MaxSeq,
        SlidingWindowSize = Window,
        SlidingWindowPattern = 2,
    };

    // Prefill, decode, a chunk bigger than the 160-row buffer, more decode, mid chunks.
    private static readonly int[] Schedule = [5, .. Enumerable.Repeat(1, 200), 300, .. Enumerable.Repeat(1, 90), 50, 7, 129, .. Enumerable.Repeat(1, 40)];

    [Fact]
    public void Geometry_MarksOnlySlidingLayers_AndOnlyOnOptIn()
    {
        var cfg = Config();
        Assert.False(KvGeometry.FromConfig(cfg).HasSlidingWindows);
        var g = KvGeometry.FromConfig(cfg, compactSlidingWindows: true);
        Assert.True(g.HasSlidingWindows);
        Assert.Equal(Window, g.SlidingWindowOf(0));
        Assert.Equal(0, g.SlidingWindowOf(1));
    }

    [Fact]
    public void SimpleCache_CompactMatchesFull_AndAllocatesLess()
    {
        var cfg = Config();
        using var full = new SimpleKvCache(KvGeometry.FromConfig(cfg), MaxSeq);
        using var compact = new SimpleKvCache(KvGeometry.FromConfig(cfg, compactSlidingWindows: true), MaxSeq);
        RunSchedule(full, compact);
        Assert.True(compact.RowBaseOf(0) > 0, "the schedule must have compacted the windowed layer");
        Assert.Equal(0, compact.RowBaseOf(1));
        Assert.True(compact.AllocatedBytes < full.AllocatedBytes * 6 / 10);
        Assert.False(compact.SupportsExactF32PrefixCopy);
    }

    [Theory]
    [InlineData(KvCacheDType.Q8_0)]
    [InlineData(KvCacheDType.Q4_0)]
    public void QuantizedCache_CompactMatchesFull(KvCacheDType dtype)
    {
        var cfg = Config();
        using var full = new QuantizedKvCache(KvGeometry.FromConfig(cfg), MaxSeq, dtype, dtype, windowSize: 0);
        using var compact = new QuantizedKvCache(KvGeometry.FromConfig(cfg, compactSlidingWindows: true), MaxSeq, dtype, dtype, windowSize: 0);
        RunSchedule(full, compact);
        Assert.True(compact.RowBaseOf(0) > 0);
        Assert.True(compact.AllocatedBytes < full.AllocatedBytes * 6 / 10);
    }

    [Fact]
    public void QuantizedCache_Fp32WindowMode_KeepsFullRows()
    {
        var cfg = Config();
        using var cache = new QuantizedKvCache(KvGeometry.FromConfig(cfg, compactSlidingWindows: true), MaxSeq,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize: 8);
        using var full = new QuantizedKvCache(KvGeometry.FromConfig(cfg), MaxSeq, KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize: 8);
        RunSchedule(full, cache);
        Assert.Equal(0, cache.RowBaseOf(0));
    }

    [Fact]
    public void Rollback_WithinMargin_StillMatches_BeyondMargin_Throws()
    {
        var cfg = Config();
        using var full = new SimpleKvCache(KvGeometry.FromConfig(cfg), MaxSeq);
        using var compact = new SimpleKvCache(KvGeometry.FromConfig(cfg, compactSlidingWindows: true), MaxSeq);
        var rng = new Random(7);
        int pos = 0;
        foreach (int n in Schedule) pos = Step(full, compact, rng, pos, n);

        // A speculative-decode style rollback of a few tokens, then continue.
        int back = pos - 10;
        Assert.True(compact.CanTruncateTo(back));
        full.Rollback(back);
        compact.Rollback(back);
        pos = back;
        for (int i = 0; i < 30; i++) pos = Step(full, compact, rng, pos, 1);

        // Far back: the windowed layer no longer holds those rows.
        Assert.False(compact.CanTruncateTo(Window + 1));
        Assert.Throws<InvalidOperationException>(() => compact.Rollback(Window + 1));

        // Emptying always works and resets the base.
        compact.SetCurrentLength(0);
        full.SetCurrentLength(0);
        Assert.Equal(0, compact.RowBaseOf(0));
        pos = 0;
        foreach (int n in Schedule.Take(20)) pos = Step(full, compact, rng, pos, n);
    }

    [Fact]
    public void Span_OnWindowedLayer_Throws()
    {
        using var compact = new SimpleKvCache(KvGeometry.FromConfig(Config(), compactSlidingWindows: true), MaxSeq);
        Assert.Throws<NotSupportedException>(() => compact.KeysSpan(0).Length);
        Assert.Equal(0, compact.KeysSpan(1).Length);
    }

    private static void RunSchedule(IKvCache full, IKvCache compact)
    {
        var rng = new Random(1169);
        int pos = 0;
        foreach (int n in Schedule) pos = Step(full, compact, rng, pos, n);
        Assert.Equal(full.CurrentLength, compact.CurrentLength);
    }

    /// <summary>Appends <paramref name="n"/> random K/V rows at <paramref name="pos"/> to both caches and compares attention per layer.</summary>
    private static int Step(IKvCache full, IKvCache compact, Random rng, int pos, int n)
    {
        float[] k = Random(rng, n * Stride), v = Random(rng, n * Stride), q = Random(rng, n * NumHeads * HeadDim);
        int[] positions = Enumerable.Range(pos, n).ToArray();
        float[] outFull = new float[n * NumHeads * HeadDim], outCompact = new float[outFull.Length];
        fixed (float* kp = k, vp = v, qp = q, of = outFull, oc = outCompact)
        {
            for (int layer = 0; layer < 2; layer++)
            {
                var kRef = new TensorRef(n, Stride, DType.Float32, -1, (nint)kp);
                var vRef = new TensorRef(n, Stride, DType.Float32, -1, (nint)vp);
                full.Update(kRef, vRef, positions, layer);
                compact.Update(kRef, vRef, positions, layer);
                int? window = layer == 0 ? Window : null;
                Attend(full, layer, qp, of, n, pos, window);
                Attend(compact, layer, qp, oc, n, pos, window);
                Assert.Equal(outFull, outCompact);
            }
        }
        return pos + n;
    }

    // Mirrors TransformerModel's cached reads: the quantized kernel applies the row base itself; the F32 caller shifts.
    private static void Attend(IKvCache cache, int layer, float* q, float* output, int seqQ, int pos0, int? window)
    {
        if (cache is IQuantizedKvCache qkv)
        {
            Attention.Execute(q, qkv, layer, output, seqQ, cache.CurrentLength, NumHeads, NumKvHeads, HeadDim, pos0,
                pool: null, slidingWindowSize: window);
            return;
        }
        int rowBase = (cache as IWindowedKvCache)?.RowBaseOf(layer) ?? 0;
        var kc = cache.GetKeysRef(layer);
        var vc = cache.GetValuesRef(layer);
        Assert.Equal(cache.CurrentLength - rowBase, kc.Dim0);
        Attention.Execute(q, (float*)kc.DataPointer, (float*)vc.DataPointer, output, seqQ, cache.CurrentLength - rowBase,
            NumHeads, NumKvHeads, HeadDim, pos0 - rowBase, 1f / MathF.Sqrt(HeadDim), null, window, 0f,
            AttentionMaskMode.Causal, 0, null);
    }

    private static float[] Random(Random rng, int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }
}
