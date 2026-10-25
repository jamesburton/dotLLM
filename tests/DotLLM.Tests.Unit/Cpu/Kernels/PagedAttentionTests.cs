using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Direct paged-KV attention (#1268): the paged kernel reading F32/F16/Q8_0 blocks in place must match the contiguous
/// kernel over the same (decoded) rows. Pages are interleaved with another live sequence's pages so logical and physical
/// order differ; lengths cross both block and tile boundaries. Adapted from Cheongroach's PagedAttention tests.
/// </summary>
public sealed unsafe class PagedAttentionTests
{
    private const int BlockSize = 16;
    private const int Heads = 4;
    private const int KvHeads = 2; // GQA group 2, two KV heads
    private const int HeadDim = 32; // Q8_0 rows need whole 32-element blocks per head
    private const int KvStride = KvHeads * HeadDim;

    public static TheoryData<string, string, int, int> Cases()
    {
        var data = new TheoryData<string, string, int, int>();
        foreach (string type in new[] { "f32", "f16", "q8_0" })
        {
            foreach (string variant in new[] { "default", "scale", "softcap", "sinks", "window" })
            {
                data.Add(type, variant, 600, 599); // decode: one query, crosses tiles (256) and blocks
                data.Add(type, variant, 300, 283); // prefill chunk of 17 queries
            }
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void PagedMatchesContiguous(string type, string variant, int length, int offset)
    {
        DType elementType = ParseType(type);
        float scale = 1f / MathF.Sqrt(HeadDim);
        float softCap = 0f;
        float[]? sinks = null;
        int? window = null;
        switch (variant)
        {
            case "scale": scale = 0.0625f; break; // Gemma query_pre_attn_scalar-style
            case "softcap": softCap = 2.5f; break;
            case "sinks": sinks = [0.5f, -1f, 2f, 0f]; break;
            case "window": window = 37; break;
        }

        using var pool = new KvBlockPool(1, KvHeads, HeadDim, BlockSize, totalBlocks: 128, elementType);
        using var cache = new PagedKvCache(pool, 1, KvStride, length);
        using var other = new PagedKvCache(pool, 1, KvStride, length);
        var rng = new Random(42);
        FillInterleaved(cache, other, rng, length);
        Assert.NotEqual(Enumerable.Range(0, cache.BlockIds.Length), cache.BlockIds.ToArray());

        // Reference rows: what the blocks hold, decoded to F32 (decode is exact for F32/F16, the stored dequant for Q8_0).
        float[] keys = new float[length * KvStride];
        float[] values = new float[length * KvStride];
        cache.DownloadLayer(0, keys, values);

        int queries = length - offset;
        float[] q = RandomValues(rng, queries * Heads * HeadDim);
        float[] expected = new float[q.Length];
        float[] actual = new float[q.Length];
        fixed (float* qp = q, kp = keys, vp = values, ep = expected, ap = actual)
        {
            Attention.Execute(qp, kp, vp, ep, queries, length, Heads, KvHeads, HeadDim, offset, scale,
                pool: null, window, softCap, sinks: sinks);
            Attention.Execute(qp, cache, 0, ap, queries, Heads, KvHeads, HeadDim, offset, scale,
                pool: null, window, softCap, sinks);
        }

        // F32 rows run the contiguous arithmetic exactly (#530); F16/Q8_0 rows are decoded per head slice, also exactly
        // what DownloadLayer decodes, so the only slack is the decode-in-registers vs decode-to-buffer order (none).
        if (elementType == DType.Float32)
            Assert.Equal(expected, actual);
        for (int i = 0; i < actual.Length; i++)
            Assert.True(MathF.Abs(actual[i] - expected[i]) <= 2e-5f, $"{type}/{variant} [{i}]: {actual[i]} vs {expected[i]}");
        Assert.Equal(0, cache.StagingBytes);
    }

    /// <summary>
    /// Past the one-shot range both kernels run the same tiled online softmax from the row's visible start, so an F32
    /// pool must reproduce the contiguous path bit for bit (#530), whatever the page layout or worker count.
    /// </summary>
    [Fact]
    public void F32PagesAreBitIdenticalToContiguousTiledPath()
    {
        const int length = 2300;
        using var pool = new KvBlockPool(1, KvHeads, HeadDim, BlockSize, totalBlocks: 400, DType.Float32);
        using var cache = new PagedKvCache(pool, 1, KvStride, length);
        using var other = new PagedKvCache(pool, 1, KvStride, length);
        var rng = new Random(7);
        FillInterleaved(cache, other, rng, length);
        float[] keys = new float[length * KvStride];
        float[] values = new float[length * KvStride];
        cache.DownloadLayer(0, keys, values);

        float[] q = RandomValues(rng, Heads * HeadDim);
        float[] expected = new float[q.Length];
        float[] serial = new float[q.Length];
        float[] parallel = new float[q.Length];
        using var workers = new ComputeThreadPool(3);
        float scale = 1f / MathF.Sqrt(HeadDim);
        fixed (float* qp = q, kp = keys, vp = values, ep = expected, sp = serial, pp = parallel)
        {
            Attention.Execute(qp, kp, vp, ep, 1, length, Heads, KvHeads, HeadDim, length - 1, scale, workers);
            Attention.Execute(qp, cache, 0, sp, 1, Heads, KvHeads, HeadDim, length - 1, scale, null);
            Attention.Execute(qp, cache, 0, pp, 1, Heads, KvHeads, HeadDim, length - 1, scale, workers);
        }

        Assert.Equal(expected, serial);
        Assert.Equal(serial, parallel);
    }

    [Theory]
    [InlineData("f16")]
    [InlineData("q8_0")]
    public void RollbackAndOverwriteUseLogicalPages(string type)
    {
        using var pool = new KvBlockPool(1, 1, HeadDim, 2, 8, ParseType(type));
        using var cache = new PagedKvCache(pool, 1, HeadDim, 8);
        float[] data = Enumerable.Range(1, 4 * HeadDim).Select(i => (i % 7) * 0.25f).ToArray();
        float[] replacement = Enumerable.Range(0, HeadDim).Select(i => -1f - (i % 5) * 0.5f).ToArray();
        fixed (float* p = data, r = replacement)
        {
            cache.Update(new TensorRef(4, HeadDim, DType.Float32, -1, (nint)p),
                new TensorRef(4, HeadDim, DType.Float32, -1, (nint)p), [0, 1, 2, 3], 0);
            cache.Rollback(1);
            cache.Update(new TensorRef(1, HeadDim, DType.Float32, -1, (nint)r),
                new TensorRef(1, HeadDim, DType.Float32, -1, (nint)r), [1], 0);
            Assert.Equal(2, cache.CurrentLength);

            // A query aligned with the replacement key puts almost all weight on position 1, so the output is that row.
            float* query = stackalloc float[HeadDim];
            float* output = stackalloc float[HeadDim];
            for (int i = 0; i < HeadDim; i++) query[i] = r[i] * 100f;
            Attention.Execute(query, cache, 0, output, 1, 1, 1, HeadDim, 1, 1f / MathF.Sqrt(HeadDim), null);
            for (int i = 0; i < HeadDim; i++)
                Assert.InRange(output[i] - replacement[i], -0.05f, 0.05f);
        }

        Assert.Equal(0, cache.StagingBytes);
    }

    [Fact]
    public void RejectsGeometryMismatch()
    {
        using var pool = new KvBlockPool(1, KvHeads, HeadDim, BlockSize, 4);
        using var cache = new PagedKvCache(pool, 1, KvStride, 16);
        float[] kv = new float[KvStride];
        fixed (float* p = kv)
            cache.Update(new TensorRef(1, KvStride, DType.Float32, -1, (nint)p), new TensorRef(1, KvStride, DType.Float32, -1, (nint)p), [0], 0);
        float[] buf = new float[Heads * HeadDim];
        fixed (float* p = buf)
        {
            float* bp = p;
            Assert.Throws<ArgumentException>(() => Attention.Execute(bp, cache, 0, bp, 1, Heads, 1, HeadDim, 0, 1f, null));
            Assert.Throws<ArgumentException>(() => Attention.Execute(bp, cache, 0, bp, 1, Heads, KvHeads, HeadDim, 1, 1f, null));
        }
    }

    private static DType ParseType(string type) => type switch
    {
        "f32" => DType.Float32,
        "f16" => DType.Float16,
        _ => DType.Q8_0,
    };

    /// <summary>Writes <paramref name="length"/> random rows into <paramref name="cache"/>, a block at a time, interleaving a block of <paramref name="other"/>.</summary>
    private static void FillInterleaved(PagedKvCache cache, PagedKvCache other, Random rng, int length)
    {
        float[] keys = RandomValues(rng, length * KvStride);
        float[] values = RandomValues(rng, length * KvStride);
        fixed (float* kp = keys, vp = values)
        {
            for (int pos = 0; pos < length; pos += BlockSize)
            {
                int count = Math.Min(BlockSize, length - pos);
                int[] positions = Enumerable.Range(pos, count).ToArray();
                cache.Update(new TensorRef(count, KvStride, DType.Float32, -1, (nint)(kp + pos * KvStride)),
                    new TensorRef(count, KvStride, DType.Float32, -1, (nint)(vp + pos * KvStride)), positions, 0);
                if ((pos / BlockSize) % 2 == 0)
                {
                    // Claim the next physical block for another sequence so this cache's pages are not consecutive.
                    other.Update(new TensorRef(count, KvStride, DType.Float32, -1, (nint)kp),
                        new TensorRef(count, KvStride, DType.Float32, -1, (nint)vp),
                        Enumerable.Range(pos, count).ToArray(), 0);
                }
            }
        }
    }

    private static float[] RandomValues(Random rng, int count)
        => Enumerable.Range(0, count).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
}
