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
        Assert.Equal(8, AbortableTapKvGuard.Checkpoint(kv, [8]));
        var ex = Assert.Throws<NotSupportedException>(() => AbortableTapKvGuard.Checkpoint(kv, [700]));
        Assert.Contains("SimpleKvCache", ex.Message);
        Assert.Equal(8, kv.CurrentLength);
    }

    [Fact]
    public void Quantized_FpWindow_AppendPastWindowIsRefused_Contiguous_NotWhenWithinWindow()
    {
        using var kv = new QuantizedKvCache(KvGeometry.FromConfig(Config()), MaxSeq, KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize: 16);
        Fill(kv, 4);
        Assert.True(kv.CanRollbackAfterAppend(4, 10));    // stays inside the 16-row FP window: nothing evicted
        Assert.False(kv.CanRollbackAfterAppend(4, 40));   // evicts rows into quantized storage
    }

    [Fact]
    public void Checkpoint_NullCache_ReturnsMinusOne() => Assert.Equal(-1, AbortableTapKvGuard.Checkpoint(null, [0]));
}
