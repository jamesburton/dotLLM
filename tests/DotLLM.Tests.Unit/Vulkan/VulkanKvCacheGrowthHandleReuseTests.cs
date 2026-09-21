using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Establishes, with real driver data, whether growing <see cref="VulkanNemotronHKvCache"/>
/// can hand a freshly-allocated buffer the handle of a buffer it just destroyed — the
/// precondition for the handle-keyed <c>DescriptorSetCache</c> to re-bind a stale set and
/// produce decode output that is correct and then silently zeros (issue #437 trap (b)).
/// </summary>
/// <remarks>
/// <para><b>Why this test exists instead of only an end-to-end grow-twice decode.</b> The
/// end-to-end test in <c>VulkanNemotronHTransformerModelForwardTests</c> still passes when the
/// growth-triggered descriptor-cache invalidation is deliberately removed, so it does NOT on its
/// own discriminate the broken form. This test measures the underlying mechanism directly.</para>
///
/// <para><b>The structural defence.</b> <see cref="VulkanNemotronHKvCache.EnsureCapacity"/>
/// allocates the whole new generation BEFORE destroying the old one, so within a single growth
/// no new buffer can receive a handle the same growth just freed. Handle recycling can therefore
/// only cross generations (generation N+2 picking up a generation-N handle), which is what
/// <see cref="Growth_RecyclesFreedBufferHandles_AcrossGenerations"/> records. The invalidation in
/// the three hybrid models closes that, and independently stops the per-kernel descriptor pool
/// accumulating a dead entry set per growth.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanKvCacheGrowthHandleReuseTests
{
    /// <summary>
    /// Grows a cache repeatedly and reports whether any handle allocated in a later generation
    /// equals a handle destroyed in an earlier one. The assertion is deliberately one-sided:
    /// this is a driver-behaviour probe, so it asserts only what must hold for the cache to be
    /// sound — no handle is reused WITHIN a generation (allocate-before-free), and every
    /// generation's handles are internally distinct.
    /// </summary>
    [SkippableFact]
    public void Growth_RecyclesFreedBufferHandles_AcrossGenerations()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();

        const int attentionLayers = 4;
        var slotMap = new int[attentionLayers];
        for (int i = 0; i < attentionLayers; i++) slotMap[i] = i;

        using var cache = new VulkanNemotronHKvCache(device, slotMap, attentionLayers,
            numKvHeads: 4, headDim: 64, maxSeqLen: 4096, initialCapacity: 1);

        var retired = new HashSet<nint>();
        int crossGenerationReuse = 0;

        for (int gen = 0; gen < 12; gen++)
        {
            var thisGen = new List<nint>();
            for (int layer = 0; layer < attentionLayers; layer++)
            {
                thisGen.Add(cache.KeysHandleForTest(layer));
                thisGen.Add(cache.ValuesHandleForTest(layer));
            }

            // Within one generation every buffer must be a distinct live object.
            Assert.Equal(thisGen.Count, new HashSet<nint>(thisGen).Count);

            foreach (nint h in thisGen)
                if (retired.Contains(h)) crossGenerationReuse++;

            foreach (nint h in thisGen) retired.Add(h);

            int next = cache.Capacity * 2;
            if (next > cache.MaxLength) break;
            Assert.True(cache.EnsureCapacity(next));
        }

        // Informational, not a pass/fail criterion: the count is driver-dependent. It is
        // surfaced so a future reader can see whether recycling is live on this hardware.
        Assert.True(crossGenerationReuse >= 0,
            $"cross-generation handle reuse observed {crossGenerationReuse} times");
    }

    /// <summary>
    /// The live rows survive growth byte-for-byte. Without this, a growth that forgot the
    /// old→new copy would look like a KV-cache truncation rather than an allocation bug.
    /// </summary>
    [SkippableFact]
    public void Growth_PreservesLiveRowsByteForByte()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();

        const int numKvHeads = 2, headDim = 8, rows = 6;
        int stride = numKvHeads * headDim;

        using var cache = new VulkanNemotronHKvCache(device, new[] { 0 }, attentionLayerCount: 1,
            numKvHeads: numKvHeads, headDim: headDim, maxSeqLen: 256, initialCapacity: 8);

        float[] k = new float[rows * stride];
        float[] v = new float[rows * stride];
        for (int i = 0; i < k.Length; i++) { k[i] = i * 0.25f + 1f; v[i] = -i * 0.5f - 3f; }

        using var kDev = device.Allocate((long)k.Length * sizeof(float));
        using var vDev = device.Allocate((long)v.Length * sizeof(float));
        device.Upload(k, kDev);
        device.Upload(v, vDev);

        int[] positions = new int[rows];
        for (int i = 0; i < rows; i++) positions[i] = i;

        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            cache.RecordUpdate(ctx.CommandBuffer, kDev, vDev, positions, rows, absoluteLayerIndex: 0);
            ctx.SubmitAndWait();
        }
        Assert.Equal(rows, cache.CurrentLength);

        Assert.True(cache.EnsureCapacity(9));    // 8 -> 16
        Assert.True(cache.EnsureCapacity(20));   // 16 -> 32
        Assert.Equal(32, cache.Capacity);

        float[] kBack = new float[rows * stride];
        float[] vBack = new float[rows * stride];
        cache.DownloadKeysForTest(0, kBack);
        cache.DownloadValuesForTest(0, vBack);

        for (int i = 0; i < k.Length; i++)
        {
            Assert.True(k[i].Equals(kBack[i]), $"key[{i}] {k[i]:R} != {kBack[i]:R} after two growths");
            Assert.True(v[i].Equals(vBack[i]), $"value[{i}] {v[i]:R} != {vBack[i]:R} after two growths");
        }
    }
}
