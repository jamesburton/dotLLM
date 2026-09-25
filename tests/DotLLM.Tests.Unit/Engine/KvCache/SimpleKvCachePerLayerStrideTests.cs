using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.KvCache;

/// <summary>
/// Discriminating round-trip coverage for the per-layer-strided
/// <see cref="SimpleKvCache"/> (KV Phase 0). The cache is built from a
/// <see cref="KvGeometry"/> whose layers have DIFFERENT KV row widths (Gemma-4
/// sliding 2×16=32 vs global 2×32=64); a cache that mis-addressed using a single
/// scalar stride would read/write layer 1 at the layer-0 width and corrupt the
/// round-trip. A degenerate equal-stride control proves the per-layer path collapses
/// to correct uniform behaviour.
/// </summary>
public sealed unsafe class SimpleKvCachePerLayerStrideTests
{
    private const int MaxSeqLen = 16;

    [Fact]
    public void PerLayer_DistinctStrides_RoundTripsEachLayerIndependently()
    {
        int[] strides = { 32, 64 }; // layer 0 (sliding) vs layer 1 (global)
        using var cache = new SimpleKvCache(KvGeometry.PerLayer(strides), MaxSeqLen);

        // Guard: the fixture must actually exercise differing strides, else it
        // cannot catch a uniform-stride mis-addressing bug.
        Assert.NotEqual(cache.KvStrideOf(0), cache.KvStrideOf(1));
        Assert.Equal(32, cache.KvStrideOf(0));
        Assert.Equal(64, cache.KvStrideOf(1));

        RoundTripAllLayers(cache, strides);
    }

    [Fact]
    public void PerLayer_DegenerateEqualStrides_CollapsesToUniform()
    {
        int[] strides = { 32, 32 };
        using var cache = new SimpleKvCache(KvGeometry.PerLayer(strides), MaxSeqLen);

        Assert.Equal(cache.KvStrideOf(0), cache.KvStrideOf(1));
        RoundTripAllLayers(cache, strides);
    }

    [Fact]
    public void ScalarCtor_IsByteIdenticalToUniformGeometry()
    {
        const int numLayers = 3, numKvHeads = 4, headDim = 8;
        using var scalar = new SimpleKvCache(numLayers, numKvHeads, headDim, MaxSeqLen);
        using var geom = new SimpleKvCache(
            KvGeometry.Uniform(numLayers, numKvHeads, headDim), MaxSeqLen);

        Assert.Equal(scalar.AllocatedBytes, geom.AllocatedBytes);
        for (int l = 0; l < numLayers; l++)
            Assert.Equal(scalar.KvStrideOf(l), geom.KvStrideOf(l));
        Assert.Equal(numKvHeads * headDim, scalar.KvStride);
    }

    [Fact]
    public void AllocatedBytes_SumsPerLayerStrides()
    {
        int[] strides = { 32, 64 };
        using var cache = new SimpleKvCache(KvGeometry.PerLayer(strides), MaxSeqLen);

        // (32 + 64) rows × 2 (K+V) × MaxSeqLen × 4 bytes
        long expected = (long)(32 + 64) * 2 * MaxSeqLen * sizeof(float);
        Assert.Equal(expected, cache.AllocatedBytes);
    }

    /// <summary>
    /// Writes a per-(layer, position, element) signature into each layer's K/V at
    /// three positions, then reads it back via <see cref="SimpleKvCache.GetKeysRef"/>
    /// / <see cref="SimpleKvCache.GetValuesRef"/> and asserts exact recovery at each
    /// layer's own stride. A scalar cache would mis-address the wider layer.
    /// </summary>
    private static void RoundTripAllLayers(SimpleKvCache cache, int[] strides)
    {
        int[] positions = { 0, 3, 7 };

        for (int layer = 0; layer < strides.Length; layer++)
        {
            int stride = strides[layer];
            foreach (int pos in positions)
            {
                nint kPtr = (nint)NativeMemory.AlignedAlloc((nuint)(stride * sizeof(float)), 64);
                nint vPtr = (nint)NativeMemory.AlignedAlloc((nuint)(stride * sizeof(float)), 64);
                try
                {
                    for (int d = 0; d < stride; d++)
                    {
                        ((float*)kPtr)[d] = Sig(layer, pos, d, isValue: false);
                        ((float*)vPtr)[d] = Sig(layer, pos, d, isValue: true);
                    }
                    var kRef = new TensorRef(1, stride, DType.Float32, -1, kPtr);
                    var vRef = new TensorRef(1, stride, DType.Float32, -1, vPtr);
                    cache.Update(kRef, vRef, [pos], layer);
                }
                finally
                {
                    NativeMemory.AlignedFree((void*)kPtr);
                    NativeMemory.AlignedFree((void*)vPtr);
                }
            }
        }

        int maxPos = 0;
        foreach (int p in positions) maxPos = Math.Max(maxPos, p);

        for (int layer = 0; layer < strides.Length; layer++)
        {
            int stride = strides[layer];
            var kCached = cache.GetKeysRef(layer);
            var vCached = cache.GetValuesRef(layer);
            Assert.Equal(stride, kCached.Dim1);
            Assert.Equal(stride, vCached.Dim1);

            var kSpan = new ReadOnlySpan<float>((void*)kCached.DataPointer, (maxPos + 1) * stride);
            var vSpan = new ReadOnlySpan<float>((void*)vCached.DataPointer, (maxPos + 1) * stride);

            foreach (int pos in positions)
            {
                for (int d = 0; d < stride; d++)
                {
                    Assert.Equal(Sig(layer, pos, d, isValue: false), kSpan[pos * stride + d]);
                    Assert.Equal(Sig(layer, pos, d, isValue: true), vSpan[pos * stride + d]);
                }
            }
        }
    }

    // A unique, recoverable value per (layer, position, element, K/V) — chosen so a
    // wrong-stride read lands on a different signature and the assert fails.
    private static float Sig(int layer, int pos, int d, bool isValue)
        => (isValue ? 100000f : 0f) + layer * 10000f + pos * 100f + d + 0.5f;
}
