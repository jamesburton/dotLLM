using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Diagnostics;

/// <summary>#1920: the forward-entry KV gate for abortable taps, on the real cache types (no model needed).</summary>
public sealed class AbortableTapKvGuardTests
{
    private const int Window = 8;
    private const int MaxSeq = 2048;

    private static ModelConfig Config() => new()
    {
        Architecture = Architecture.Llama, VocabSize = 32, HiddenSize = 128, IntermediateSize = 256, NumLayers = 2,
        NumAttentionHeads = 4, NumKvHeads = 2, HeadDim = 32, MaxSequenceLength = MaxSeq,
        SlidingWindowSize = Window, SlidingWindowPattern = 2,
    };

    private static unsafe void Fill(IKvCache kv, int rows)
    {
        float[] data = new float[rows * 64];
        var pos = Enumerable.Range(0, rows).ToArray();
        fixed (float* p = data)
        {
            var r = new DotLLM.Core.Tensors.TensorRef(rows, 64, DotLLM.Core.Tensors.DType.Float32, -1, (nint)p);
            for (int l = 0; l < 2; l++) kv.Update(r, r, pos, l);
        }
    }

    [Fact]
    public void Simple_FullCache_AlwaysFeasible()
    {
        using var kv = new SimpleKvCache(KvGeometry.FromConfig(Config()), MaxSeq);
        Fill(kv, 8);
        Assert.True(kv.CanRollbackAfterAppend(8, 700));
    }

    [Fact]
    public void Simple_Compact_ContiguousAppendFeasible_FarAppendNot()
    {
        using var kv = new SimpleKvCache(KvGeometry.FromConfig(Config(), compactSlidingWindows: true), MaxSeq);
        Fill(kv, 8);
        Assert.True(kv.CanRollbackAfterAppend(8, 8));
        Assert.True(kv.CanRollbackAfterAppend(8, 100));   // fits the default ring: no compaction
        Assert.False(kv.CanRollbackAfterAppend(700, 700)); // compacts past the rows the entry length needs
        Assert.False(kv.SupportsAbortRollback);
        // Ruling 3 (#1920): refused at the first abortable forward, even for a contiguous append that would not compact yet.
        foreach (int pos in new[] { 8, 700 })
        {
            var ex = Assert.Throws<NotSupportedException>(() => AbortableTapKvGuard.Checkpoint(kv, [pos]));
            Assert.Contains("SimpleKvCache", ex.Message);
            Assert.Contains("sliding-window", ex.Message);
        }
        Assert.Equal(8, kv.CurrentLength);
    }

    [Fact]
    public void Simple_FullCache_SupportsAbortRollback_AndCheckpointReturnsEntryLength()
    {
        using var kv = new SimpleKvCache(KvGeometry.FromConfig(Config()), MaxSeq);
        Fill(kv, 8);
        Assert.True(kv.SupportsAbortRollback);
        Assert.Equal(8, AbortableTapKvGuard.Checkpoint(kv, [8]));
    }

    /// <summary>The per-forward post-append check stays as a backstop for a cache that claims abort support but cannot undo this append.</summary>
    [Fact]
    public void Checkpoint_BackstopStillRefusesWhenCanRollbackAfterAppendIsFalse()
    {
        using var inner = new SimpleKvCache(KvGeometry.FromConfig(Config()), MaxSeq);
        Fill(inner, 8);
        var kv = new BackstopKv(inner);
        Assert.True(kv.SupportsAbortRollback);
        Assert.Equal(8, AbortableTapKvGuard.Checkpoint(kv, [8]));
        Assert.Throws<NotSupportedException>(() => AbortableTapKvGuard.Checkpoint(kv, [700]));
    }

    private sealed class BackstopKv(IKvCache inner) : IKvCache
    {
        public int CurrentLength => inner.CurrentLength;
        public int MaxLength => inner.MaxLength;
        public void Update(DotLLM.Core.Tensors.ITensor k, DotLLM.Core.Tensors.ITensor v, ReadOnlySpan<int> p, int l) => inner.Update(k, v, p, l);
        public DotLLM.Core.Tensors.ITensor GetKeys(int l) => inner.GetKeys(l);
        public DotLLM.Core.Tensors.ITensor GetValues(int l) => inner.GetValues(l);
        public void Update(DotLLM.Core.Tensors.TensorRef k, DotLLM.Core.Tensors.TensorRef v, ReadOnlySpan<int> p, int l) => inner.Update(k, v, p, l);
        public DotLLM.Core.Tensors.TensorRef GetKeysRef(int l) => inner.GetKeysRef(l);
        public DotLLM.Core.Tensors.TensorRef GetValuesRef(int l) => inner.GetValuesRef(l);
        public void Rollback(int length) => inner.Rollback(length);
        public bool CanRollbackAfterAppend(int minPos, int maxPos) => maxPos < 100;
        public void Dispose() { }
    }

    [Fact]
    public void ThrowIfWindowedConfig_RefusesSlidingWindowConfig_NotFullAttention()
    {
        Assert.Throws<NotSupportedException>(() => AbortableTapKvGuard.ThrowIfWindowedConfig(Config()));
        AbortableTapKvGuard.ThrowIfWindowedConfig(Config() with { SlidingWindowSize = null, SlidingWindowPattern = 0 });
    }

    [Fact]
    public void Quantized_FpWindow_AppendPastWindowIsRefused_Contiguous_NotWhenWithinWindow()
    {
        using var kv = new QuantizedKvCache(KvGeometry.FromConfig(Config()), MaxSeq, KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize: 16);
        Fill(kv, 4);
        Assert.True(kv.CanRollbackAfterAppend(4, 10));    // stays inside the 16-row FP window: nothing evicted
        Assert.False(kv.CanRollbackAfterAppend(4, 40));   // evicts rows into quantized storage
        Assert.False(kv.SupportsAbortRollback);           // ruling 3: refused regardless of length
        Assert.Throws<NotSupportedException>(() => AbortableTapKvGuard.Checkpoint(kv, [4]));
    }

    [Fact]
    public void Checkpoint_NullCache_ReturnsMinusOne() => Assert.Equal(-1, AbortableTapKvGuard.Checkpoint(null, [0]));
}
