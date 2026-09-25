using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Smoke tests for <see cref="MlaVulkanKvCache"/> — the per-layer
/// expanded-form MLA KV cache.
/// </summary>
/// <remarks>
/// Verifies the construction / dispose lifecycle, the
/// <see cref="MlaVulkanKvCache.AllocatedBytes"/> accounting, and that
/// <c>RecordUpdate</c> correctly slabs K_nope / V / K_pe rows into the
/// cache at the right offsets. Does NOT exercise the full attention
/// path — that's covered by <c>VulkanAttentionMlaF32KernelTests</c> at
/// the kernel level and (eventually) by the end-to-end DeepSeek-V2-Lite
/// argmax-parity integration test.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMlaKvCacheTests
{
    [SkippableFact]
    public void Construction_AllocatesExpectedFootprint()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);

        // DeepSeek-V2-Lite-ish parameters at a tiny scale.
        const int numLayers = 2;
        const int maxSeqLen = 8;
        const int numHeads = 4;
        const int qkNopeHeadDim = 16;
        const int vHeadDim = 16;
        const int qkRopeHeadDim = 8;

        using var device = VulkanDevice.Create();
        using var cache = new MlaVulkanKvCache(
            device, numLayers, maxSeqLen, numHeads, qkNopeHeadDim, vHeadDim, qkRopeHeadDim);

        Assert.Equal(numLayers, cache.NumLayers);
        Assert.Equal(maxSeqLen, cache.MaxLength);
        Assert.Equal(0, cache.CurrentLength);

        long expectedKNopeRow = (long)numHeads * qkNopeHeadDim * sizeof(float);
        long expectedVRow = (long)numHeads * vHeadDim * sizeof(float);
        long expectedKPeRow = (long)qkRopeHeadDim * sizeof(float);
        Assert.Equal(expectedKNopeRow, cache.KNopeRowBytes);
        Assert.Equal(expectedVRow, cache.VRowBytes);
        Assert.Equal(expectedKPeRow, cache.KPeRowBytes);

        long expected = numLayers * maxSeqLen * (expectedKNopeRow + expectedVRow + expectedKPeRow);
        Assert.Equal(expected, cache.AllocatedBytes);
    }

    [SkippableFact]
    public void RecordUpdate_AppendsContiguousRows()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);

        const int numLayers = 1;
        const int maxSeqLen = 16;
        const int numHeads = 2;
        const int qkNopeHeadDim = 8;
        const int vHeadDim = 8;
        const int qkRopeHeadDim = 4;

        int kNopeRow = numHeads * qkNopeHeadDim;
        int vRow = numHeads * vHeadDim;
        int kPeRow = qkRopeHeadDim;

        using var device = VulkanDevice.Create();
        using var cache = new MlaVulkanKvCache(
            device, numLayers, maxSeqLen, numHeads, qkNopeHeadDim, vHeadDim, qkRopeHeadDim);

        // Build a 3-token batch starting at position 0; then a 1-token batch at 3.
        const int firstSeqLen = 3;
        var first = MakeBatch(firstSeqLen, kNopeRow, vRow, kPeRow, seed: 0);
        var firstPositions = new int[] { 0, 1, 2 };
        UploadAndRecord(device, cache, first, firstPositions, firstSeqLen, layerIndex: 0);
        Assert.Equal(3, cache.CurrentLength);

        const int secondSeqLen = 1;
        var second = MakeBatch(secondSeqLen, kNopeRow, vRow, kPeRow, seed: 1);
        var secondPositions = new int[] { 3 };
        UploadAndRecord(device, cache, second, secondPositions, secondSeqLen, layerIndex: 0);
        Assert.Equal(4, cache.CurrentLength);

        // The cache buffers are device-local (can't host-map directly) — stage
        // through a host-visible buffer to read back.
        var kNopeOut = new float[4 * kNopeRow];
        var vOut = new float[4 * vRow];
        var kPeOut = new float[4 * kPeRow];
        DownloadDeviceLocal(device, cache.GetKNopeBuffer(0), kNopeOut, 4 * kNopeRow);
        DownloadDeviceLocal(device, cache.GetVBuffer(0), vOut, 4 * vRow);
        DownloadDeviceLocal(device, cache.GetKPeBuffer(0), kPeOut, 4 * kPeRow);

        for (int i = 0; i < firstSeqLen * kNopeRow; i++) Assert.Equal(first.kNope[i], kNopeOut[i]);
        for (int i = 0; i < firstSeqLen * vRow; i++) Assert.Equal(first.v[i], vOut[i]);
        for (int i = 0; i < firstSeqLen * kPeRow; i++) Assert.Equal(first.kPe[i], kPeOut[i]);

        for (int i = 0; i < secondSeqLen * kNopeRow; i++)
            Assert.Equal(second.kNope[i], kNopeOut[firstSeqLen * kNopeRow + i]);
        for (int i = 0; i < secondSeqLen * vRow; i++)
            Assert.Equal(second.v[i], vOut[firstSeqLen * vRow + i]);
        for (int i = 0; i < secondSeqLen * kPeRow; i++)
            Assert.Equal(second.kPe[i], kPeOut[firstSeqLen * kPeRow + i]);
    }

    [SkippableFact]
    public void RecordUpdate_NonContiguousPositions_Throws()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);

        using var device = VulkanDevice.Create();
        using var cache = new MlaVulkanKvCache(
            device, numLayers: 1, maxSeqLen: 8,
            numHeads: 1, qkNopeHeadDim: 4, vHeadDim: 4, qkRopeHeadDim: 4);

        var batch = MakeBatch(2, 4, 4, 4, seed: 0);
        var positions = new int[] { 0, 2 }; // non-contiguous

        using var bufKNope = device.Allocate((long)batch.kNope.Length * sizeof(float));
        using var bufV = device.Allocate((long)batch.v.Length * sizeof(float));
        using var bufKPe = device.Allocate((long)batch.kPe.Length * sizeof(float));
        device.Upload(batch.kNope, bufKNope);
        device.Upload(batch.v, bufV);
        device.Upload(batch.kPe, bufKPe);

        using var ctx = device.CreateSubmitContext();
        ctx.Begin();
        Assert.Throws<NotSupportedException>(() =>
            cache.RecordUpdate(ctx.CommandBuffer, bufKNope, bufV, bufKPe, positions, 2, 0));
    }

    [SkippableFact]
    public void Rollback_LowersCurrentLength()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);

        using var device = VulkanDevice.Create();
        using var cache = new MlaVulkanKvCache(
            device, numLayers: 1, maxSeqLen: 8,
            numHeads: 1, qkNopeHeadDim: 4, vHeadDim: 4, qkRopeHeadDim: 4);

        var batch = MakeBatch(4, 4, 4, 4, seed: 0);
        UploadAndRecord(device, cache, batch, new int[] { 0, 1, 2, 3 }, 4, layerIndex: 0);
        Assert.Equal(4, cache.CurrentLength);

        cache.Rollback(2);
        Assert.Equal(2, cache.CurrentLength);

        cache.Reset();
        Assert.Equal(0, cache.CurrentLength);
    }

    private static (float[] kNope, float[] v, float[] kPe) MakeBatch(
        int seqLen, int kNopeRow, int vRow, int kPeRow, int seed)
    {
        var rng = new Random(seed * 31 + seqLen * 13 + kNopeRow);
        return (
            RandomFloats(rng, seqLen * kNopeRow),
            RandomFloats(rng, seqLen * vRow),
            RandomFloats(rng, seqLen * kPeRow));
    }

    private static void UploadAndRecord(
        VulkanDevice device, MlaVulkanKvCache cache,
        (float[] kNope, float[] v, float[] kPe) batch,
        int[] positions, int seqLen, int layerIndex)
    {
        using var bufKNope = device.Allocate((long)batch.kNope.Length * sizeof(float));
        using var bufV = device.Allocate((long)batch.v.Length * sizeof(float));
        using var bufKPe = device.Allocate((long)batch.kPe.Length * sizeof(float));
        device.Upload(batch.kNope, bufKNope);
        device.Upload(batch.v, bufV);
        device.Upload(batch.kPe, bufKPe);

        using var ctx = device.CreateSubmitContext();
        ctx.Begin();
        cache.RecordUpdate(ctx.CommandBuffer, bufKNope, bufV, bufKPe, positions, seqLen, layerIndex);
        ctx.SubmitAndWait();
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++) arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return arr;
    }

    /// <summary>
    /// Downloads from a device-local buffer by staging through a host-visible
    /// buffer first (device-local memory cannot be host-mapped directly).
    /// </summary>
    private static void DownloadDeviceLocal(
        VulkanDevice device, VulkanDevice.Buffer src, float[] dst, int count)
    {
        long byteCount = (long)count * sizeof(float);
        using var staging = device.Allocate(byteCount);
        device.CopyBufferRangeSynchronous(src, staging, srcOffset: 0, dstOffset: 0, size: (ulong)byteCount);
        device.Download(staging, dst.AsSpan(0, count));
    }
}
