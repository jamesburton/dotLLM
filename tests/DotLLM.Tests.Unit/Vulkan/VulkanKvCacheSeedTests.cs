using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1780: <see cref="VulkanKvCache.TrySeedPrefixFrom"/> copies a protected entry's leading rows into a fresh cache, device to
/// device in the shared element format, and leaves the source untouched. Incompatible sources are refused with nothing written.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanKvCacheSeedTests
{
    private static readonly int[] Strides = [8, 16];

    private static float Val(int layer, int row, int ch, bool value)
        => (layer * 1000 + row * 10 + ch) * 0.0173f * (value ? -1f : 1f) + 0.00031f;

    private static void Fill(VulkanKvCache cache, int start, int count, int salt)
    {
        for (int l = 0; l < Strides.Length; l++)
        {
            int s = Strides[l];
            var k = new float[count * s];
            var v = new float[count * s];
            for (int r = 0; r < count; r++)
                for (int c = 0; c < s; c++)
                {
                    k[r * s + c] = Val(l, start + r, c, false) + salt;
                    v[r * s + c] = Val(l, start + r, c, true) + salt;
                }
            cache.WriteRows(l, start, count, k, v);
        }
    }

    private static (float[] K, float[] V)[] Read(VulkanKvCache cache, int rows)
    {
        var result = new (float[], float[])[Strides.Length];
        for (int l = 0; l < Strides.Length; l++)
        {
            var k = new float[rows * Strides[l]];
            var v = new float[rows * Strides[l]];
            cache.ReadRows(l, 0, rows, k, v);
            result[l] = (k, v);
        }
        return result;
    }

    [SkippableTheory]
    [InlineData(VulkanKvStorage.F32)]
    [InlineData(VulkanKvStorage.F16)]
    public void TrySeedPrefixFrom_CopiesTheLeadingRowsExactly_AndLeavesTheSource(VulkanKvStorage storage)
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device");
        using var dev = VulkanDevice.Create();
        using var source = new VulkanKvCache(dev, Strides, 40, storage);
        using var target = new VulkanKvCache(dev, Strides, 64, storage);
        Fill(source, 0, 40, salt: 0);
        Fill(target, 0, 64, salt: 7);                                    // stale rows a short copy would leave behind
        target.Rollback(0);
        var before = Read(source, 40);

        Assert.True(target.TrySeedPrefixFrom(source, 29));

        Assert.Equal(29, target.CurrentLength);
        var seeded = Read(target, 29);
        var after = Read(source, 40);
        for (int l = 0; l < Strides.Length; l++)
        {
            Assert.Equal(before[l].K.AsSpan(0, 29 * Strides[l]).ToArray(), seeded[l].K);
            Assert.Equal(before[l].V.AsSpan(0, 29 * Strides[l]).ToArray(), seeded[l].V);
            Assert.Equal(before[l].K, after[l].K);
            Assert.Equal(before[l].V, after[l].V);
        }
        Assert.Equal(40, source.CurrentLength);
    }

    [SkippableFact]
    public void TrySeedPrefixFrom_RefusesIncompatibleSources()
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device");
        using var dev = VulkanDevice.Create();
        using var source = new VulkanKvCache(dev, Strides, 40, VulkanKvStorage.F32);
        Fill(source, 0, 40, salt: 0);
        using var f16 = new VulkanKvCache(dev, Strides, 64, VulkanKvStorage.F16);
        using var otherWidth = new VulkanKvCache(dev, new[] { 8, 8 }, 64, VulkanKvStorage.F32);
        using var small = new VulkanKvCache(dev, Strides, 16, VulkanKvStorage.F32);

        Assert.False(f16.TrySeedPrefixFrom(source, 20));                 // no widening or narrowing across formats
        Assert.False(otherWidth.TrySeedPrefixFrom(source, 20));
        Assert.False(small.TrySeedPrefixFrom(source, 20));               // cannot hold the rows
        Assert.False(f16.TrySeedPrefixFrom(f16, 1));
        using var target = new VulkanKvCache(dev, Strides, 64, VulkanKvStorage.F32);
        Assert.False(target.TrySeedPrefixFrom(source, 41));              // more rows than the source holds
        Assert.Equal(0, target.CurrentLength);
    }
}
