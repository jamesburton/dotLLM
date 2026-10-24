using System;
using DotLLM.Core.Configuration;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Round-trip quality tests for <see cref="VulkanQuantizedKvCache"/>. Uploads FP32 K/V
/// activations via <see cref="VulkanQuantizedKvCache.UpdateSync"/>, dequantizes via
/// <see cref="VulkanQuantizedKvCache.DequantKeysToHost"/>/<see cref="VulkanQuantizedKvCache.DequantValuesToHost"/>,
/// and verifies the reconstructed values against the original using cosine similarity and
/// element-wise error bounds. Multi-layer and incremental-write scenarios are covered.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQuantizedKvCacheTests
{
    private const int BlockSize = 32;

    private readonly ITestOutputHelper _output;
    public VulkanQuantizedKvCacheTests(ITestOutputHelper output) => _output = output;

    [SkippableTheory]
    [InlineData(KvCacheDType.Q8_0, KvCacheDType.Q8_0, 128, 8, 2, 4)]
    [InlineData(KvCacheDType.Q4_0, KvCacheDType.Q4_0, 128, 8, 2, 4)]
    [InlineData(KvCacheDType.Q8_0, KvCacheDType.Q4_0, 128, 4, 1, 6)]
    [InlineData(KvCacheDType.Q8_0, KvCacheDType.Q8_0, 64, 2, 3, 5)]
    public void RoundTrip_CosineSimilarity(
        KvCacheDType keyDType, KvCacheDType valueDType,
        int headDim, int numKvHeads, int numLayers, int seqLen)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int stride = numKvHeads * headDim;
        int maxSeqLen = seqLen + 4;
        var rng = new Random(0xFACE ^ (headDim * 7 + numKvHeads * 3 + numLayers + (int)keyDType * 11));

        using var device = VulkanDevice.Create();
        using var cache = new VulkanQuantizedKvCache(device, spvDir, numLayers, numKvHeads, headDim,
            maxSeqLen, keyDType, valueDType);

        var positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) positions[i] = i;

        double minCosSim = double.MaxValue;
        double maxRelErr = 0;

        for (int layer = 0; layer < numLayers; layer++)
        {
            float[] hostK = RandomRows(rng, seqLen, stride, numKvHeads, headDim, salt: layer * 2);
            float[] hostV = RandomRows(rng, seqLen, stride, numKvHeads, headDim, salt: layer * 2 + 1);

            using var kDev = device.Allocate((long)seqLen * stride * sizeof(float));
            using var vDev = device.Allocate((long)seqLen * stride * sizeof(float));
            device.Upload(new ReadOnlySpan<float>(hostK), kDev);
            device.Upload(new ReadOnlySpan<float>(hostV), vDev);
            cache.UpdateSync(kDev, vDev, positions, seqLen, layer);

            var gotK = new float[seqLen * stride];
            var gotV = new float[seqLen * stride];
            cache.DequantKeysToHost(layer, gotK);
            cache.DequantValuesToHost(layer, gotV);

            double cosK = CosineSimilarity(hostK, gotK);
            double cosV = CosineSimilarity(hostV, gotV);
            minCosSim = Math.Min(minCosSim, Math.Min(cosK, cosV));

            double relK = MaxRelativeError(hostK, gotK);
            double relV = MaxRelativeError(hostV, gotV);
            maxRelErr = Math.Max(maxRelErr, Math.Max(relK, relV));

            _output.WriteLine($"  layer {layer}: cosK={cosK:F6} cosV={cosV:F6} relK={relK:E3} relV={relV:E3}");
        }

        _output.WriteLine($"K={keyDType} V={valueDType} d={headDim} heads={numKvHeads} layers={numLayers} seq={seqLen}: " +
                          $"minCos={minCosSim:F6} maxRelErr={maxRelErr:E3}");

        double cosFloor = keyDType == KvCacheDType.Q8_0 && valueDType == KvCacheDType.Q8_0
            ? 0.9999 : 0.995;
        Assert.True(minCosSim >= cosFloor,
            $"Cosine similarity {minCosSim:F6} below floor {cosFloor:F4}");
    }

    [SkippableTheory]
    [InlineData(KvCacheDType.Q8_0)]
    [InlineData(KvCacheDType.Q4_0)]
    public void IncrementalWrite_MultipleUpdates(KvCacheDType dtype)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int headDim = 64;
        int numKvHeads = 2;
        int stride = numKvHeads * headDim;
        int maxSeqLen = 16;
        int numLayers = 1;
        var rng = new Random(0xDEAD + (int)dtype);

        using var device = VulkanDevice.Create();
        using var cache = new VulkanQuantizedKvCache(device, spvDir, numLayers, numKvHeads, headDim,
            maxSeqLen, dtype, dtype);

        float[] allK = RandomRows(rng, 8, stride, numKvHeads, headDim, salt: 0);
        float[] allV = RandomRows(rng, 8, stride, numKvHeads, headDim, salt: 1);

        // Write first 4 tokens (prefill).
        var pos1 = new int[] { 0, 1, 2, 3 };
        using var k1 = device.Allocate((long)4 * stride * sizeof(float));
        using var v1 = device.Allocate((long)4 * stride * sizeof(float));
        device.Upload(new ReadOnlySpan<float>(allK, 0, 4 * stride), k1);
        device.Upload(new ReadOnlySpan<float>(allV, 0, 4 * stride), v1);
        cache.UpdateSync(k1, v1, pos1, 4, layerIndex: 0);

        Assert.Equal(4, cache.CurrentLength);

        // Write next 4 tokens (decode).
        var pos2 = new int[] { 4, 5, 6, 7 };
        using var k2 = device.Allocate((long)4 * stride * sizeof(float));
        using var v2 = device.Allocate((long)4 * stride * sizeof(float));
        device.Upload(new ReadOnlySpan<float>(allK, 4 * stride, 4 * stride), k2);
        device.Upload(new ReadOnlySpan<float>(allV, 4 * stride, 4 * stride), v2);
        cache.UpdateSync(k2, v2, pos2, 4, layerIndex: 0);

        Assert.Equal(8, cache.CurrentLength);

        // Dequant all 8 tokens and verify against expected.
        var gotK = new float[8 * stride];
        var gotV = new float[8 * stride];
        cache.DequantKeysToHost(0, gotK);
        cache.DequantValuesToHost(0, gotV);

        double cosK = CosineSimilarity(allK, gotK);
        double cosV = CosineSimilarity(allV, gotV);
        _output.WriteLine($"Incremental {dtype}: cosK={cosK:F6} cosV={cosV:F6}");

        double cosFloor = dtype == KvCacheDType.Q8_0 ? 0.9999 : 0.995;
        Assert.True(cosK >= cosFloor, $"Keys cosine {cosK:F6} below {cosFloor}");
        Assert.True(cosV >= cosFloor, $"Values cosine {cosV:F6} below {cosFloor}");
    }

    [SkippableTheory]
    [InlineData(KvCacheDType.Q8_0)]
    [InlineData(KvCacheDType.Q4_0)]
    public void Rollback_TruncatesLength(KvCacheDType dtype)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int headDim = 64;
        int numKvHeads = 2;
        int stride = numKvHeads * headDim;
        int maxSeqLen = 16;
        var rng = new Random(0xB0B0 + (int)dtype);

        using var device = VulkanDevice.Create();
        using var cache = new VulkanQuantizedKvCache(device, spvDir, 1, numKvHeads, headDim,
            maxSeqLen, dtype, dtype);

        float[] hostK = RandomRows(rng, 6, stride, numKvHeads, headDim, salt: 0);
        float[] hostV = RandomRows(rng, 6, stride, numKvHeads, headDim, salt: 1);

        var positions = new int[] { 0, 1, 2, 3, 4, 5 };
        using var kDev = device.Allocate((long)6 * stride * sizeof(float));
        using var vDev = device.Allocate((long)6 * stride * sizeof(float));
        device.Upload(new ReadOnlySpan<float>(hostK), kDev);
        device.Upload(new ReadOnlySpan<float>(hostV), vDev);
        cache.UpdateSync(kDev, vDev, positions, 6, layerIndex: 0);

        Assert.Equal(6, cache.CurrentLength);

        cache.Rollback(3);
        Assert.Equal(3, cache.CurrentLength);

        // Dequant should only return the first 3 tokens' worth.
        var gotK = new float[3 * stride];
        var gotV = new float[3 * stride];
        cache.DequantKeysToHost(0, gotK);
        cache.DequantValuesToHost(0, gotV);

        double cosK = CosineSimilarity(hostK.AsSpan(0, 3 * stride), gotK);
        _output.WriteLine($"Rollback {dtype}: cosK(first 3)={cosK:F6}");

        double cosFloor = dtype == KvCacheDType.Q8_0 ? 0.9999 : 0.995;
        Assert.True(cosK >= cosFloor, $"Keys cosine after rollback {cosK:F6} below {cosFloor}");
    }

    [SkippableTheory]
    [InlineData(KvCacheDType.Q8_0)]
    [InlineData(KvCacheDType.Q4_0)]
    public void Reset_ClearsLength(KvCacheDType dtype)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        using var cache = new VulkanQuantizedKvCache(device, spvDir, 1, 2, 64, 16, dtype, dtype);

        int stride = 2 * 64;
        var rng = new Random(42);
        float[] hostK = RandomRows(rng, 3, stride, 2, 64, salt: 0);
        float[] hostV = RandomRows(rng, 3, stride, 2, 64, salt: 1);

        var positions = new int[] { 0, 1, 2 };
        using var kDev = device.Allocate((long)3 * stride * sizeof(float));
        using var vDev = device.Allocate((long)3 * stride * sizeof(float));
        device.Upload(new ReadOnlySpan<float>(hostK), kDev);
        device.Upload(new ReadOnlySpan<float>(hostV), vDev);
        cache.UpdateSync(kDev, vDev, positions, 3, layerIndex: 0);

        Assert.Equal(3, cache.CurrentLength);
        cache.Reset();
        Assert.Equal(0, cache.CurrentLength);
    }

    // ── Helpers ──

    private static double CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int n = Math.Min(a.Length, b.Length);
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < n; i++)
        {
            dot += (double)a[i] * b[i];
            normA += (double)a[i] * a[i];
            normB += (double)b[i] * b[i];
        }
        double denom = Math.Sqrt(normA) * Math.Sqrt(normB);
        return denom > 0 ? dot / denom : 1.0;
    }

    private static double MaxRelativeError(float[] expected, float[] actual)
    {
        double maxRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = Math.Abs((double)expected[i] - actual[i]);
            double denom = Math.Max(Math.Abs(expected[i]), 1e-6);
            maxRel = Math.Max(maxRel, d / denom);
        }
        return maxRel;
    }

    private static float[] RandomRows(Random rng, int seqLen, int stride, int numKvHeads, int headDim, int salt)
    {
        var buf = new float[seqLen * stride];
        for (int pos = 0; pos < seqLen; pos++)
            for (int h = 0; h < numKvHeads; h++)
            {
                float target = (pos + 1) * 9f + h * 2.5f + 1f + salt;
                int off = pos * stride + h * headDim;
                double sq = 0;
                for (int i = 0; i < headDim; i++)
                {
                    double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
                    float g = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
                    buf[off + i] = g; sq += (double)g * g;
                }
                float scale = target / (float)Math.Sqrt(sq);
                for (int i = 0; i < headDim; i++) buf[off + i] *= scale;
            }
        return buf;
    }
}
