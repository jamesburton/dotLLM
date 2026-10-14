using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.KvCache;

public sealed unsafe class QuantizedKvCacheTests
{
    private const int NumLayers = 2;
    private const int NumKvHeads = 2;
    private const int HeadDim = 32; // Must be multiple of 32 (quant block size)
    private const int MaxSeqLen = 16;
    private const int KvStride = NumKvHeads * HeadDim; // 64

    [Fact]
    public void Constructor_AllocatesCorrectly_Q8_0()
    {
        using var cache = new QuantizedKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize: 0);

        Assert.Equal(0, cache.CurrentLength);
        Assert.Equal(MaxSeqLen, cache.MaxLength);
        Assert.Equal(KvCacheDType.Q8_0, cache.KeyDType);
        Assert.Equal(KvCacheDType.Q8_0, cache.ValueDType);
        Assert.True(cache.AllocatedBytes > 0);
    }

    [Fact]
    public void Update_PureQuantized_QuantizesImmediately()
    {
        using var cache = new QuantizedKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize: 0);

        nint kPtr = AllocAndFill(1, 1.5f);
        nint vPtr = AllocAndFill(1, 2.5f);
        try
        {
            var kRef = new TensorRef(1, KvStride, DType.Float32, -1, kPtr);
            var vRef = new TensorRef(1, KvStride, DType.Float32, -1, vPtr);

            cache.Update(kRef, vRef, [0], layerIndex: 0);

            Assert.Equal(1, cache.CurrentLength);
            Assert.Equal(1, cache.QuantizedLength);
            Assert.Equal(0, cache.WindowLength);

            // Quantized data pointer should be non-zero
            Assert.NotEqual((nint)0, cache.GetQuantizedKeysPtr(0));
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kPtr);
            NativeMemory.AlignedFree((void*)vPtr);
        }
    }

    [Fact]
    public void Update_WithWindow_EvictsOnlyWhenWindowFull()
    {
        int windowSize = 4;
        using var cache = new QuantizedKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize);

        // Add tokens 0..3 — should all be in window, no eviction
        for (int i = 0; i < windowSize; i++)
        {
            nint kPtr = AllocAndFill(1, 1.0f + i);
            nint vPtr = AllocAndFill(1, 10.0f + i);
            try
            {
                var kRef = new TensorRef(1, KvStride, DType.Float32, -1, kPtr);
                var vRef = new TensorRef(1, KvStride, DType.Float32, -1, vPtr);
                cache.Update(kRef, vRef, [i], layerIndex: 0);
            }
            finally
            {
                NativeMemory.AlignedFree((void*)kPtr);
                NativeMemory.AlignedFree((void*)vPtr);
            }
        }

        Assert.Equal(windowSize, cache.CurrentLength);
        Assert.Equal(0, cache.QuantizedLength);
        Assert.Equal(windowSize, cache.WindowLength);

        // Add token 4 — should evict token 0 to quantized buffer
        nint kPtr5 = AllocAndFill(1, 5.0f);
        nint vPtr5 = AllocAndFill(1, 15.0f);
        try
        {
            var kRef = new TensorRef(1, KvStride, DType.Float32, -1, kPtr5);
            var vRef = new TensorRef(1, KvStride, DType.Float32, -1, vPtr5);
            cache.Update(kRef, vRef, [windowSize], layerIndex: 0);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kPtr5);
            NativeMemory.AlignedFree((void*)vPtr5);
        }

        Assert.Equal(windowSize + 1, cache.CurrentLength);
        Assert.Equal(1, cache.QuantizedLength);
        Assert.Equal(windowSize, cache.WindowLength);
    }

    [Fact]
    public void Update_SeparateKeyValueTypes_Works()
    {
        using var cache = new QuantizedKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen,
            KvCacheDType.Q8_0, KvCacheDType.Q4_0, windowSize: 0);

        Assert.Equal(KvCacheDType.Q8_0, cache.KeyDType);
        Assert.Equal(KvCacheDType.Q4_0, cache.ValueDType);

        // Key rows should be larger than value rows
        Assert.True(cache.KeyQuantizedRowBytes > cache.ValueQuantizedRowBytes);

        nint kPtr = AllocAndFill(1, 1.0f);
        nint vPtr = AllocAndFill(1, 2.0f);
        try
        {
            var kRef = new TensorRef(1, KvStride, DType.Float32, -1, kPtr);
            var vRef = new TensorRef(1, KvStride, DType.Float32, -1, vPtr);
            cache.Update(kRef, vRef, [0], layerIndex: 0);

            Assert.Equal(1, cache.CurrentLength);
            Assert.Equal(1, cache.QuantizedLength);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kPtr);
            NativeMemory.AlignedFree((void*)vPtr);
        }
    }

    [Fact]
    public void AllocatedBytes_ReflectsCompression()
    {
        using var fpCache = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);
        using var q8Cache = new QuantizedKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize: 0);

        // Quantized should use significantly less memory
        Assert.True(q8Cache.AllocatedBytes < fpCache.AllocatedBytes,
            $"Q8_0 cache ({q8Cache.AllocatedBytes}) should be smaller than FP32 cache ({fpCache.AllocatedBytes})");
    }

    [Fact]
    public void Dispose_IsSafeToCallMultipleTimes()
    {
        var cache = new QuantizedKvCache(1, NumKvHeads, HeadDim, 8,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize: 0);
        cache.Dispose();
        cache.Dispose(); // Should not throw
    }

    [Fact]
    public void IQuantizedKvCache_InterfaceMethods_Work()
    {
        using var cache = new QuantizedKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen,
            KvCacheDType.Q8_0, KvCacheDType.Q4_0, windowSize: 4);

        IQuantizedKvCache qkv = cache;

        Assert.Equal(KvCacheDType.Q8_0, qkv.KeyDType);
        Assert.Equal(KvCacheDType.Q4_0, qkv.ValueDType);
        Assert.Equal(0, qkv.QuantizedLength);
        Assert.Equal(0, qkv.WindowLength);
    }

    [Fact]
    public void PrefillThenDecode_Q8_0_TracksLengthsCorrectly()
    {
        int windowSize = 4;
        using var cache = new QuantizedKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize);

        // Prefill: 3 tokens at once
        nint kPtr = AllocAndFill(3, 1.0f);
        nint vPtr = AllocAndFill(3, 2.0f);
        try
        {
            var kRef = new TensorRef(3, KvStride, DType.Float32, -1, kPtr);
            var vRef = new TensorRef(3, KvStride, DType.Float32, -1, vPtr);
            cache.Update(kRef, vRef, [0, 1, 2], layerIndex: 0);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kPtr);
            NativeMemory.AlignedFree((void*)vPtr);
        }

        Assert.Equal(3, cache.CurrentLength);
        Assert.Equal(0, cache.QuantizedLength); // all within window
        Assert.Equal(3, cache.WindowLength);

        // Decode: add tokens one by one until we trigger eviction
        for (int i = 3; i < 8; i++)
        {
            kPtr = AllocAndFill(1, 1.0f + i);
            vPtr = AllocAndFill(1, 2.0f + i);
            try
            {
                var kRef = new TensorRef(1, KvStride, DType.Float32, -1, kPtr);
                var vRef = new TensorRef(1, KvStride, DType.Float32, -1, vPtr);
                cache.Update(kRef, vRef, [i], layerIndex: 0);
            }
            finally
            {
                NativeMemory.AlignedFree((void*)kPtr);
                NativeMemory.AlignedFree((void*)vPtr);
            }
        }

        Assert.Equal(8, cache.CurrentLength);
        Assert.Equal(4, cache.QuantizedLength); // tokens 0..3 evicted
        Assert.Equal(4, cache.WindowLength);     // tokens 4..7 in window
    }

    /// <summary>
    /// Regression test for #969: Update with more rows than windowSize evicts garbage because
    /// the eviction loop reads from ring slots that haven't been written yet by the current call.
    /// Verifies quantized data matches a reference (pure quantized, no window) for all batch sizes.
    /// </summary>
    [Theory]
    [InlineData(1)]    // single row — no eviction
    [InlineData(8)]    // exactly windowSize — fills window, no eviction yet
    [InlineData(9)]    // windowSize+1 — evicts 1 row from current batch
    [InlineData(24)]   // 3*windowSize — large batch, ring wraps multiple times
    public void Update_LargeBatch_EvictsCorrectData(int totalRows)
    {
        const int window = 8;
        const int maxSeq = 64;
        var rng = new Random(42);

        // Reference: pure quantized (windowSize=0) — every row quantized directly.
        using var refCache = new QuantizedKvCache(NumLayers, NumKvHeads, HeadDim, maxSeq,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize: 0);

        // Under test: windowed cache.
        using var winCache = new QuantizedKvCache(NumLayers, NumKvHeads, HeadDim, maxSeq,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0, window);

        // Fill with random data (distinct per row so we can detect swaps).
        nint kPtr = (nint)NativeMemory.AlignedAlloc((nuint)(totalRows * KvStride * sizeof(float)), 64);
        nint vPtr = (nint)NativeMemory.AlignedAlloc((nuint)(totalRows * KvStride * sizeof(float)), 64);
        try
        {
            var kSpan = new Span<float>((void*)kPtr, totalRows * KvStride);
            var vSpan = new Span<float>((void*)vPtr, totalRows * KvStride);
            for (int i = 0; i < kSpan.Length; i++) kSpan[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < vSpan.Length; i++) vSpan[i] = (float)(rng.NextDouble() * 2 - 1);

            int[] positions = new int[totalRows];
            for (int i = 0; i < totalRows; i++) positions[i] = i;

            var kRef = new TensorRef(totalRows, KvStride, DType.Float32, -1, kPtr);
            var vRef = new TensorRef(totalRows, KvStride, DType.Float32, -1, vPtr);

            // Update both caches for layer 0.
            refCache.Update(kRef, vRef, positions, layerIndex: 0);
            winCache.Update(kRef, vRef, positions, layerIndex: 0);

            Assert.Equal(totalRows, refCache.CurrentLength);
            Assert.Equal(totalRows, winCache.CurrentLength);

            // Compare quantized regions.
            int expectedQuantLen = Math.Max(0, totalRows - window);
            Assert.Equal(expectedQuantLen, winCache.QuantizedLength);

            if (expectedQuantLen > 0)
            {
                int quantBytes = expectedQuantLen * refCache.KeyQuantizedRowBytes;
                var refK = new ReadOnlySpan<byte>((void*)refCache.GetQuantizedKeysPtr(0), quantBytes);
                var winK = new ReadOnlySpan<byte>((void*)winCache.GetQuantizedKeysPtr(0), quantBytes);
                Assert.True(refK.SequenceEqual(winK),
                    $"Quantized keys mismatch for {totalRows} rows (window={window}): eviction read garbage.");

                var refV = new ReadOnlySpan<byte>((void*)refCache.GetQuantizedValuesPtr(0), quantBytes);
                var winV = new ReadOnlySpan<byte>((void*)winCache.GetQuantizedValuesPtr(0), quantBytes);
                Assert.True(refV.SequenceEqual(winV),
                    $"Quantized values mismatch for {totalRows} rows (window={window}): eviction read garbage.");
            }
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kPtr);
            NativeMemory.AlignedFree((void*)vPtr);
        }
    }

    /// <summary>
    /// Verifies that a prefill followed by decode produces the same quantized data regardless
    /// of whether the prefill was chunked to fit the window or sent in one large batch.
    /// </summary>
    [Fact]
    public void Update_LargePrefillThenDecode_MatchesChunkedPrefill()
    {
        const int window = 8;
        const int maxSeq = 64;
        const int prefillLen = 20;
        const int decodeLen = 5;
        var rng = new Random(123);

        nint kPtr = (nint)NativeMemory.AlignedAlloc((nuint)((prefillLen + decodeLen) * KvStride * sizeof(float)), 64);
        nint vPtr = (nint)NativeMemory.AlignedAlloc((nuint)((prefillLen + decodeLen) * KvStride * sizeof(float)), 64);
        try
        {
            var kSpan = new Span<float>((void*)kPtr, (prefillLen + decodeLen) * KvStride);
            var vSpan = new Span<float>((void*)vPtr, (prefillLen + decodeLen) * KvStride);
            for (int i = 0; i < kSpan.Length; i++) kSpan[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < vSpan.Length; i++) vSpan[i] = (float)(rng.NextDouble() * 2 - 1);

            // Approach A: one big prefill + decode.
            using var bigCache = new QuantizedKvCache(1, NumKvHeads, HeadDim, maxSeq,
                KvCacheDType.Q8_0, KvCacheDType.Q8_0, window);
            {
                int[] pos = new int[prefillLen];
                for (int i = 0; i < prefillLen; i++) pos[i] = i;
                var kRef = new TensorRef(prefillLen, KvStride, DType.Float32, -1, kPtr);
                var vRef = new TensorRef(prefillLen, KvStride, DType.Float32, -1, vPtr);
                bigCache.Update(kRef, vRef, pos, layerIndex: 0);
            }
            for (int d = 0; d < decodeLen; d++)
            {
                int pos = prefillLen + d;
                var kRef = new TensorRef(1, KvStride, DType.Float32, -1, kPtr + pos * KvStride * sizeof(float));
                var vRef = new TensorRef(1, KvStride, DType.Float32, -1, vPtr + pos * KvStride * sizeof(float));
                bigCache.Update(kRef, vRef, [pos], layerIndex: 0);
            }

            // Approach B: chunked prefill (chunk ≤ window) + decode.
            using var chunkedCache = new QuantizedKvCache(1, NumKvHeads, HeadDim, maxSeq,
                KvCacheDType.Q8_0, KvCacheDType.Q8_0, window);
            for (int start = 0; start < prefillLen; start += window)
            {
                int chunk = Math.Min(window, prefillLen - start);
                int[] pos = new int[chunk];
                for (int i = 0; i < chunk; i++) pos[i] = start + i;
                var kRef = new TensorRef(chunk, KvStride, DType.Float32, -1, kPtr + start * KvStride * sizeof(float));
                var vRef = new TensorRef(chunk, KvStride, DType.Float32, -1, vPtr + start * KvStride * sizeof(float));
                chunkedCache.Update(kRef, vRef, pos, layerIndex: 0);
            }
            for (int d = 0; d < decodeLen; d++)
            {
                int pos = prefillLen + d;
                var kRef = new TensorRef(1, KvStride, DType.Float32, -1, kPtr + pos * KvStride * sizeof(float));
                var vRef = new TensorRef(1, KvStride, DType.Float32, -1, vPtr + pos * KvStride * sizeof(float));
                chunkedCache.Update(kRef, vRef, [pos], layerIndex: 0);
            }

            // Both must have identical quantized data.
            Assert.Equal(bigCache.CurrentLength, chunkedCache.CurrentLength);
            Assert.Equal(bigCache.QuantizedLength, chunkedCache.QuantizedLength);

            int quantBytes = bigCache.QuantizedLength * bigCache.KeyQuantizedRowBytes;
            if (quantBytes > 0)
            {
                var bigK = new ReadOnlySpan<byte>((void*)bigCache.GetQuantizedKeysPtr(0), quantBytes);
                var chunkK = new ReadOnlySpan<byte>((void*)chunkedCache.GetQuantizedKeysPtr(0), quantBytes);
                Assert.True(bigK.SequenceEqual(chunkK), "Quantized keys differ between big-batch and chunked prefill.");

                var bigV = new ReadOnlySpan<byte>((void*)bigCache.GetQuantizedValuesPtr(0), quantBytes);
                var chunkV = new ReadOnlySpan<byte>((void*)chunkedCache.GetQuantizedValuesPtr(0), quantBytes);
                Assert.True(bigV.SequenceEqual(chunkV), "Quantized values differ between big-batch and chunked prefill.");
            }
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kPtr);
            NativeMemory.AlignedFree((void*)vPtr);
        }
    }

    private static nint AllocAndFill(int seqLen, float value)
    {
        nuint bytes = (nuint)(seqLen * KvStride * sizeof(float));
        nint ptr = (nint)NativeMemory.AlignedAlloc(bytes, 64);
        new Span<float>((void*)ptr, seqLen * KvStride).Fill(value);
        return ptr;
    }
}
