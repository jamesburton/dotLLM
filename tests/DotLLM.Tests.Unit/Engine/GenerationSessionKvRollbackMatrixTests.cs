using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// #1919 T5: which KV-caches a session can roll back, per backend. The gate is <see cref="GenerationSession.KvTruncationRefusal"/>:
/// a cache is cut only if it declares an exact length cut (<see cref="IPrefixReusableKvCache"/> / <see cref="IKvLengthRollback"/>,
/// or is the CPU <see cref="SimpleKvCache"/>); every other type refuses loudly, naming itself. No GPU needed: the matrix is by type.
/// </summary>
public sealed class GenerationSessionKvRollbackMatrixTests
{
    [Theory]
    // Supported (exact length cut)
    [InlineData(typeof(CudaKvCache), true)]
    [InlineData(typeof(CudaPagedKvCache), true)]
    [InlineData(typeof(CudaQuantizedKvCache), true)]       // windowed instance may still refuse a given cut at runtime (CanTruncateTo)
    [InlineData(typeof(CudaHybridKvCacheHandle), true)]    // Qwen3HybridDense / Qwen3MoeHybrid / Lfm2Moe on CUDA
    [InlineData(typeof(CudaNemotronHKvCache), true)]
    [InlineData(typeof(VulkanKvCache), true)]
    [InlineData(typeof(VulkanQuantizedKvCache), true)]
    [InlineData(typeof(VulkanNemotronHKvCache), true)]     // Qwen3HybridDense / Qwen3MoeHybrid on Vulkan
    [InlineData(typeof(QuantizedKvCache), true)]
    // Refused loudly (no exact-cut declaration)
    [InlineData(typeof(CudaTurboQuantKvCache), false)]
    [InlineData(typeof(VulkanTurboQuantKvCache), false)]
    [InlineData(typeof(TurboQuantKvCache), false)]
    [InlineData(typeof(PagedKvCache), false)]
    [InlineData(typeof(CudaPipelineKvCache), false)]
    [InlineData(typeof(VulkanPipelineKvCache), false)]
    [InlineData(typeof(HybridKvCache), false)]
    [InlineData(typeof(HybridVulkanCudaKvCache), false)]
    [InlineData(typeof(Qwen3HybridDenseSplitKvCache), false)]
    public void CacheType_DeclaresOrRefusesAnExactCut(Type cache, bool supported)
    {
        bool declares = typeof(IPrefixReusableKvCache).IsAssignableFrom(cache)
            || typeof(IKvLengthRollback).IsAssignableFrom(cache)
            || cache == typeof(SimpleKvCache);
        Assert.Equal(supported, declares);
    }

    [Fact]
    public void Refusal_NamesTheCacheType()
    {
        var kv = new NoRollbackCache();
        string? msg = GenerationSession.KvTruncationRefusal(kv, 1);
        Assert.NotNull(msg);
        Assert.Contains(nameof(NoRollbackCache), msg);
    }

    [Fact]
    public void LengthRollbackCache_IsAcceptedWithinItsLength_AndRefusedBeyond()
    {
        var kv = new LengthOnlyCache(8);
        Assert.Null(GenerationSession.KvTruncationRefusal(kv, 8));
        Assert.Null(GenerationSession.KvTruncationRefusal(kv, 3));
        Assert.NotNull(GenerationSession.KvTruncationRefusal(kv, 9));
    }

    /// <summary>The real CUDA hybrid handle: a cut is exact length bookkeeping (the model re-syncs its F16 cursor on the next forward).</summary>
    [Fact]
    public void CudaHybridHandle_CutsToAnyLengthWithinWhatItHolds()
    {
        using var h = new CudaHybridKvCacheHandle(32);
        h.Advance(10);
        Assert.Null(GenerationSession.KvTruncationRefusal(h, 10));
        Assert.Null(GenerationSession.KvTruncationRefusal(h, 0));
        Assert.NotNull(GenerationSession.KvTruncationRefusal(h, 11));
        h.Rollback(4);
        Assert.Equal(4, h.CurrentLength);
        Assert.False(h.CanTruncateTo(5));
        Assert.True(h.CanTruncateTo(4));
        h.Advance(7);   // the next forward writes rows 4..6 again
        Assert.Equal(7, h.CurrentLength);
    }

    private class NoRollbackCache : IKvCache
    {
        public virtual int CurrentLength => 0;
        public int MaxLength => 8;
        public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex) => throw new NotSupportedException();
        public ITensor GetKeys(int layerIndex) => throw new NotSupportedException();
        public ITensor GetValues(int layerIndex) => throw new NotSupportedException();
        public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex) => throw new NotSupportedException();
        public TensorRef GetKeysRef(int layerIndex) => throw new NotSupportedException();
        public TensorRef GetValuesRef(int layerIndex) => throw new NotSupportedException();
        public void Rollback(int length) { }
        public void Dispose() { }
    }

    private sealed class LengthOnlyCache(int length) : NoRollbackCache, IKvLengthRollback
    {
        public override int CurrentLength => length;
        public bool CanTruncateTo(int l) => (uint)l <= (uint)length;
    }
}
