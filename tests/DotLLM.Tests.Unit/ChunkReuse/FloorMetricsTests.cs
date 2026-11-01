using ChunkKvLab;
using DotLLM.ChunkReuse.Backends;
using DotLLM.ChunkReuse.Deviation;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Phase G Task 2 (numerical floor): the metric helpers of <c>ChunkKvLab floor</c> (<see cref="FloorMetrics"/>,
/// <see cref="FloorAccumulator"/>) and the reader-based <see cref="KvDeviation"/> overload they use to compare caches of any
/// backend through <see cref="IChunkBackend.ReadRows"/>. Arithmetic tests use planted values with known answers; the
/// read-back test runs the synthetic qwen3 model on the CPU.
/// </summary>
public sealed class FloorMetricsTests : IDisposable
{
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-floor-" + Guid.NewGuid().ToString("N"));

    public FloorMetricsTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); }
        catch (IOException) { /* best effort: a still-mapped file on Windows */ }
        catch (UnauthorizedAccessException) { }
    }

    // ---- percentile and summary ---------------------------------------------------------------------------------------------

    [Fact]
    public void Percentile_IsNearestRank_AndSummaryKeepsMeanP99MaxApart()
    {
        // 1..200 shuffled: nearest rank p99 = value at rank ceil(0.99 * 200) = 198; mean 100.5; max 200. All three differ, so a
        // "p99 = max", "p99 = mean" or interpolating implementation fails.
        float[] values = Enumerable.Range(1, 200).Select(i => (float)i).ToArray();
        new Random(7).Shuffle(values);
        Assert.Equal(198.0, FloorMetrics.Percentile(values, 99));
        Assert.Equal(100.0, FloorMetrics.Percentile(values, 50));
        Assert.Equal(200.0, FloorMetrics.Percentile(values, 100));
        Assert.Equal(1.0, FloorMetrics.Percentile(values, 0.1));

        // Small n: rank ceil(0.99 * 10) = 10 -> the maximum; rank ceil(0.5 * 10) = 5.
        float[] ten = [9, 1, 8, 2, 7, 3, 6, 4, 5, 10];
        Assert.Equal(10.0, FloorMetrics.Percentile(ten, 99));
        Assert.Equal(5.0, FloorMetrics.Percentile(ten, 50));

        // The input is not reordered (the helper sorts a copy).
        float[] before = (float[])values.Clone();
        FloorMetrics.Percentile(values, 99);
        Assert.Equal(before, values);

        FloorMetrics.Summary s = FloorMetrics.Summarize(values);
        Assert.Equal(200, s.Count);
        Assert.Equal(100.5, s.Mean, 9);
        Assert.Equal(198.0, s.P99);
        Assert.Equal(200.0, s.Max);

        Assert.Throws<ArgumentException>(() => FloorMetrics.Percentile([], 99));
        Assert.Throws<ArgumentOutOfRangeException>(() => FloorMetrics.Percentile(values, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => FloorMetrics.Percentile(values, 100.5));
    }

    [Fact]
    public void Accumulator_PoolsRowsAcrossAdds_PerKey()
    {
        var acc = new FloorAccumulator();
        acc.Add("a", [1f, 2f, 3f]);
        acc.Add("b", [100f]);
        acc.Add("a", [10f]);

        // Pooled over the rows of both adds (not a mean of per-add means: that would be (2 + 10) / 2 = 6).
        FloorMetrics.Summary a = acc.Summary("a");
        Assert.Equal(4, a.Count);
        Assert.Equal(4.0, a.Mean, 12);
        Assert.Equal(10.0, a.P99);
        Assert.Equal(10.0, a.Max);
        Assert.Equal(3.0, FloorMetrics.Percentile([1f, 2f, 3f, 10f], 75));

        FloorMetrics.Summary b = acc.Summary("b");
        Assert.Equal(1, b.Count);
        Assert.Equal(100.0, b.Mean);
        Assert.Equal(["a", "b"], acc.Keys.Order(StringComparer.Ordinal).ToArray());
        Assert.Throws<KeyNotFoundException>(() => acc.Summary("c"));
    }

    // ---- per-layer relative L2 through a row reader ---------------------------------------------------------------------------

    [Fact]
    public void Deviation_ReaderOverload_PerLayerRelativeL2_KAndVSeparately()
    {
        // Two layers, stride 2, three rows. Reference rows have norm 5 (3,4); the other cache adds a planted delta per row.
        float[][] refK = [[3, 4, 3, 4, 3, 4], [0, 5, 0, 5, 0, 5]];
        float[][] refV = [[4, 3, 4, 3, 4, 3], [5, 0, 5, 0, 5, 0]];
        float[][] othK = [[3, 4, 3, 4.5f, 3, 6], [0, 5, 0, 5, 0, 5]];   // layer 0 K: deltas 0, 0.5, 2 -> rel 0, 0.1, 0.4
        float[][] othV = [[4, 3, 4, 3, 4, 3], [5, 0, 6, 0, 5, 0]];       // layer 1 V: delta 1 at row 1 -> rel 0.2

        KvDeviation d = KvDeviation.Compute(2, 3, _ => 2, Reader(refK, refV), Reader(othK, othV));

        Assert.Equal(2, d.Layers);
        Assert.Equal(3, d.Tokens);
        Assert.Equal([0f, 0.1f, 0.4f], FloorMetrics.KeyRelativeRows(d, 0), new FloatTol(1e-6f));
        Assert.Equal([0f, 0f, 0f], FloorMetrics.ValueRelativeRows(d, 0));
        Assert.Equal([0f, 0f, 0f], FloorMetrics.KeyRelativeRows(d, 1));
        Assert.Equal([0f, 0.2f, 0f], FloorMetrics.ValueRelativeRows(d, 1), new FloatTol(1e-6f));

        // Only the first `rows` rows of a longer buffer are compared (a reader filling more must not leak into the result).
        KvDeviation two = KvDeviation.Compute(2, 2, _ => 2, Reader(refK, refV), Reader(othK, othV));
        Assert.Equal([0f, 0.1f], FloorMetrics.KeyRelativeRows(two, 0), new FloatTol(1e-6f));
    }

    [Fact]
    public void Deviation_ThroughCpuBackendReadRows_BatchInvariantPrefillIsExact_PlantedRowIsLocated()
    {
        string path = Path.Combine(_scratch, "qwen3.gguf");
        File.WriteAllBytes(path, SyntheticDenseGguf.Build("qwen3"));
        using var gguf = GgufFile.Open(path);
        ModelConfig config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);
        var backend = new CpuChunkBackend(config);
        KvGeometry geometry = KvGeometry.FromConfig(config, compactSlidingWindows: false);
        int[] ids = [3, 17, 5, 29, 11, 40, 8, 22, 1, 35, 12, 7];

        using IKvCache whole = backend.CreateCache(ids.Length);
        using IKvCache blocks = backend.CreateCache(ids.Length);
        Prefill(model, whole, ids, ids.Length);
        Prefill(model, blocks, ids, 5); // 5 + 5 + 2 rows

        // The CPU kernels are batch-invariant (E2): the read-back comparison must give exactly 0 on every layer and row.
        KvDeviation same = FloorMetrics.Deviation(backend, whole, backend, blocks, ids.Length, geometry);
        Assert.Equal(config.NumLayers, same.Layers);
        Assert.Equal(ids.Length, same.Tokens);
        for (int l = 0; l < same.Layers; l++)
        {
            Assert.All(FloorMetrics.KeyRelativeRows(same, l), x => Assert.Equal(0f, x));
            Assert.All(FloorMetrics.ValueRelativeRows(same, l), x => Assert.Equal(0f, x));
            for (int t = 0; t < ids.Length; t++)
                Assert.True(same.KeyReferenceNorm(l, t) > 0f, "fixture: reference rows must be non-zero");
        }

        // Plant one K delta at layer 1, row 7 of the second cache: exactly that cell is non-zero, with the right value.
        int stride = geometry.KvStrideOf(1);
        float[] k = new float[stride], v = new float[stride];
        backend.ReadRows(blocks, 1, 7, 1, k, v);
        float norm = MathF.Sqrt(k.Sum(x => x * x));
        k[stride - 1] += 0.25f * norm;
        backend.WriteRows(blocks, 1, 7, k, v);

        KvDeviation planted = FloorMetrics.Deviation(backend, whole, backend, blocks, ids.Length, geometry);
        for (int l = 0; l < planted.Layers; l++)
        for (int t = 0; t < ids.Length; t++)
        {
            float expected = l == 1 && t == 7 ? 0.25f : 0f;
            Assert.Equal(expected, planted.KeyRelative(l, t), 4);
            Assert.Equal(0f, planted.ValueRelative(l, t));
        }
    }

    // ---- logits: KL, top-1 and the masked-call NLL -------------------------------------------------------------------------

    [Fact]
    public void CompareLogits_KlTop1AndMaskNll_KnownAnswers()
    {
        // vocab 2, three rows. Row 0: P = softmax([0, ln 3]) = [1/4, 3/4] vs Q = [1/2, 1/2]: KL = 1/4 ln(1/2) + 3/4 ln(3/2), argmax
        // ref 1 vs 0 (Q ties go to index 0) -> disagree. Row 1: identical -> KL 0, agree. Row 2: ref [2, 0] vs [5, 1] -> agree,
        // KL(softmax[2,0] || softmax[5,1]) computed below.
        float ln3 = MathF.Log(3f);
        float[] reference = [0f, ln3, 1f, 0f, 2f, 0f];
        float[] rows = [0f, 0f, 1f, 0f, 5f, 1f];
        int[] targets = [1, 0, 1];

        FloorMetrics.LogitsComparison c = FloorMetrics.CompareLogits(reference, rows, vocab: 2, targets, maskCount: 2);

        double kl0 = 0.25 * Math.Log(0.5) + 0.75 * Math.Log(1.5);
        double p2 = 1 / (1 + Math.Exp(-2)), q2 = 1 / (1 + Math.Exp(-4));
        double kl2 = p2 * Math.Log(p2 / q2) + (1 - p2) * Math.Log((1 - p2) / (1 - q2));
        Assert.Equal(3, c.Kl.Length);
        Assert.Equal(kl0, c.Kl[0], 6);
        Assert.Equal(0.0, c.Kl[1], 9);
        Assert.Equal(kl2, c.Kl[2], 6);
        Assert.Equal([false, true, true], c.Top1Agree);

        // Masked-call NLL = sum over the first maskCount rows of -log softmax(row)[target], for each side.
        double refNll = -Math.Log(0.75) - Math.Log(Math.E / (Math.E + 1));
        double nll = -Math.Log(0.5) - Math.Log(Math.E / (Math.E + 1));
        Assert.Equal(refNll, c.ReferenceMaskNll, 6);
        Assert.Equal(nll, c.MaskNll, 6);

        // Greedy exact match of the masked span on each side: ref argmax [1, 0] = targets -> true; other [0, 0] -> false.
        Assert.True(c.ReferenceMaskExact);
        Assert.False(c.MaskExact);

        Assert.Throws<ArgumentException>(() => FloorMetrics.CompareLogits(reference, rows.AsSpan(0, 4), 2, targets, 2));
        Assert.Throws<ArgumentException>(() => FloorMetrics.CompareLogits(reference, rows, 2, targets.AsSpan(0, 2), 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => FloorMetrics.CompareLogits(reference, rows, 2, targets, 4));
    }

    // ---- helpers ---------------------------------------------------------------------------------------------------------

    private static KvDeviation.KvRowsReader Reader(float[][] k, float[][] v) => (layer, rows, keys, values) =>
    {
        // Deliberately fill the whole planted layer when the buffer allows, so a Compute that reads past `rows` would see it.
        int n = Math.Min(k[layer].Length, keys.Length);
        k[layer].AsSpan(0, n).CopyTo(keys);
        v[layer].AsSpan(0, n).CopyTo(values);
    };

    private static void Prefill(TransformerModel model, IKvCache cache, int[] ids, int block)
    {
        for (int off = 0; off < ids.Length; off += block)
        {
            int n = Math.Min(block, ids.Length - off);
            int[] positions = Enumerable.Range(off, n).ToArray();
            using ITensor _ = model.Forward(ids.AsSpan(off, n), positions, -1, cache, lastTokenLogitsOnly: true);
        }
    }

    private sealed class FloatTol(float tol) : IEqualityComparer<float>
    {
        public bool Equals(float x, float y) => MathF.Abs(x - y) <= tol;

        public int GetHashCode(float obj) => 0;
    }
}
