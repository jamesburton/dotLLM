using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.KvCache;

/// <summary>
/// Discriminating per-layer-stride coverage for the quantized KV cache (KV Phase 0).
/// The cache is built from a <see cref="KvGeometry"/> whose layers have DIFFERENT KV row
/// widths (Gemma-4 sliding 2×16=32 vs global 2×32=64); a cache that sized or addressed
/// the quantized rows with a single scalar stride would mis-quantize the wider layer. A
/// degenerate equal-stride control proves the per-layer path collapses to correct uniform
/// behaviour, and a scalar-ctor equivalence check proves the migration is byte-identical
/// for uniform models.
/// </summary>
public sealed unsafe class QuantizedKvCachePerLayerStrideTests
{
    private const int BlockSize = 32;
    private const int Q8_0BlockBytes = 34;
    private const int MaxSeqLen = 16;

    [Fact]
    public void PerLayer_DistinctStrides_QuantRowBytesAndRoundTripPerLayer()
    {
        int[] strides = { 32, 64 }; // layer 0 (sliding) vs layer 1 (global)
        using var cache = new QuantizedKvCache(
            KvGeometry.PerLayer(strides), MaxSeqLen,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize: 0);

        IQuantizedKvCache q = cache;

        // Guard: distinct per-layer strides AND distinct quantized row bytes — else the
        // fixture cannot catch a uniform-stride mis-sizing bug.
        Assert.NotEqual(cache.KvStrideOf(0), cache.KvStrideOf(1));
        Assert.Equal(32 / BlockSize * Q8_0BlockBytes, q.KeyQuantizedRowBytesOf(0)); // 34
        Assert.Equal(64 / BlockSize * Q8_0BlockBytes, q.KeyQuantizedRowBytesOf(1)); // 68
        Assert.NotEqual(q.KeyQuantizedRowBytesOf(0), q.KeyQuantizedRowBytesOf(1));

        RoundTripAllLayers(cache, strides);
    }

    [Fact]
    public void PerLayer_DegenerateEqualStrides_CollapsesToUniform()
    {
        int[] strides = { 32, 32 };
        using var cache = new QuantizedKvCache(
            KvGeometry.PerLayer(strides), MaxSeqLen,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize: 0);

        IQuantizedKvCache q = cache;
        Assert.Equal(q.KeyQuantizedRowBytesOf(0), q.KeyQuantizedRowBytesOf(1));
        RoundTripAllLayers(cache, strides);
    }

    [Fact]
    public void ScalarCtor_IsByteIdenticalToUniformGeometry()
    {
        const int numLayers = 3, numKvHeads = 2, headDim = 32;
        using var scalar = new QuantizedKvCache(
            numLayers, numKvHeads, headDim, MaxSeqLen,
            KvCacheDType.Q8_0, KvCacheDType.Q4_0, windowSize: 4);
        using var geom = new QuantizedKvCache(
            KvGeometry.Uniform(numLayers, numKvHeads, headDim), MaxSeqLen,
            KvCacheDType.Q8_0, KvCacheDType.Q4_0, windowSize: 4);

        Assert.Equal(scalar.AllocatedBytes, geom.AllocatedBytes);
        Assert.Equal(scalar.KeyQuantizedRowBytes, geom.KeyQuantizedRowBytes);
        Assert.Equal(scalar.ValueQuantizedRowBytes, geom.ValueQuantizedRowBytes);
        IQuantizedKvCache qs = scalar, qg = geom;
        for (int l = 0; l < numLayers; l++)
        {
            Assert.Equal(qs.KeyQuantizedRowBytesOf(l), qg.KeyQuantizedRowBytesOf(l));
            Assert.Equal(scalar.KvStrideOf(l), geom.KvStrideOf(l));
        }
    }

    /// <summary>
    /// Quantizes a per-(layer, position) constant row into each layer's quantized buffer,
    /// reads it back via <see cref="IQuantizedKvCache.GetQuantizedKeysPtr"/> at that layer's
    /// own row-byte stride, dequantizes the Q8_0 blocks and asserts recovery. A scalar cache
    /// would mis-size / mis-address the wider layer, corrupting the round-trip.
    /// </summary>
    private static void RoundTripAllLayers(QuantizedKvCache cache, int[] strides)
    {
        IQuantizedKvCache q = cache;
        int[] positions = { 0, 3, 7 };

        for (int layer = 0; layer < strides.Length; layer++)
        {
            int stride = strides[layer];
            foreach (int pos in positions)
            {
                float kv = Sig(layer, pos, isValue: false);
                float vv = Sig(layer, pos, isValue: true);
                nint kPtr = AllocConst(stride, kv);
                nint vPtr = AllocConst(stride, vv);
                try
                {
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

        for (int layer = 0; layer < strides.Length; layer++)
        {
            int stride = strides[layer];
            int keyRowBytes = q.KeyQuantizedRowBytesOf(layer);
            int valRowBytes = q.ValueQuantizedRowBytesOf(layer);
            byte* kQuant = (byte*)q.GetQuantizedKeysPtr(layer);
            byte* vQuant = (byte*)q.GetQuantizedValuesPtr(layer);

            foreach (int pos in positions)
            {
                float expectedK = Sig(layer, pos, isValue: false);
                float expectedV = Sig(layer, pos, isValue: true);

                // Dequantize a few representative lanes of the row at this layer's stride.
                AssertDequantConstant(kQuant + (long)pos * keyRowBytes, stride, expectedK, layer, pos, "K");
                AssertDequantConstant(vQuant + (long)pos * valRowBytes, stride, expectedV, layer, pos, "V");
            }
        }
    }

    // Q8_0: each 32-element block = Half scale (2 bytes) + 32 sbytes; value = scale * qs[i].
    // A constant-fill row reconstructs the fill value within Half(scale) rounding (~1e-3 rel).
    private static void AssertDequantConstant(byte* rowQuant, int stride, float expected,
        int layer, int pos, string which)
    {
        int blockCount = stride / BlockSize;
        // Check lane 0 and last lane of each block.
        ReadOnlySpan<int> lanes = [0, BlockSize - 1];
        for (int b = 0; b < blockCount; b++)
        {
            byte* block = rowQuant + b * Q8_0BlockBytes;
            float scale = (float)Unsafe.ReadUnaligned<Half>(block);
            sbyte* qs = (sbyte*)(block + 2);
            foreach (int lane in lanes)
            {
                float got = scale * qs[lane];
                float bar = 1e-2f + 1e-2f * MathF.Abs(expected);
                Assert.True(MathF.Abs(got - expected) <= bar,
                    $"layer {layer} {which} pos {pos} block {b} lane {lane}: got {got:F5} expected {expected:F5} "
                    + $"(|diff|={MathF.Abs(got - expected):E3} > {bar:E3}); stride={stride}");
            }
        }
    }

    private static nint AllocConst(int stride, float value)
    {
        nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(stride * sizeof(float)), 64);
        new Span<float>((void*)ptr, stride).Fill(value);
        return ptr;
    }

    // Distinct, recoverable per-(layer, position, K/V) value so a wrong-stride read lands on
    // a different signature and the assert fails.
    private static float Sig(int layer, int pos, bool isValue)
        => (isValue ? 50f : 0f) + layer * 10f + pos + 1.5f;
}
