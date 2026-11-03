using DotLLM.Core.Attention;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Positioned per-layer K/V row IO on <see cref="VulkanKvCache"/> (#1531 step 1): write/read at arbitrary start
/// positions (F32 and F16 storage) and device-to-device row copy. Values are distinct per (layer, row, channel) so a
/// wrong offset, layer or K/V swap changes the result; F16 values are chosen NOT representable in half so a missing
/// convert cannot pass.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanKvCachePositionedRowsTests
{
    private const int Layers = 2, Stride = 8, MaxLen = 48;

    private static float Val(int layer, int row, int ch, bool value)
        => (layer * 1000 + row * 10 + ch) * 0.0173f * (value ? -1f : 1f) + 0.00031f;

    private static (float[] k, float[] v) Rows(int layer, int start, int count)
    {
        var k = new float[count * Stride];
        var v = new float[count * Stride];
        for (int r = 0; r < count; r++)
            for (int c = 0; c < Stride; c++)
            {
                k[r * Stride + c] = Val(layer, start + r, c, false);
                v[r * Stride + c] = Val(layer, start + r, c, true);
            }
        return (k, v);
    }

    private static float Round(float x, VulkanKvStorage s) => s == VulkanKvStorage.F16 ? (float)(Half)x : x;

    [SkippableTheory]
    [InlineData(VulkanKvStorage.F32)]
    [InlineData(VulkanKvStorage.F16)]
    public void WriteRows_AtOffset_RoundTripsAndLeavesNeighboursUntouched(VulkanKvStorage storage)
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device");
        using var dev = VulkanDevice.Create();
        using var cache = new VulkanKvCache(dev, new[] { Stride, Stride }, MaxLen, storage);

        // Two disjoint chunks at non-zero, non-adjacent offsets; the gap rows must stay as written (zero-filled by us first).
        for (int l = 0; l < Layers; l++)
        {
            var zero = new float[MaxLen * Stride];
            cache.WriteRows(l, 0, MaxLen, zero, zero);
            var (k1, v1) = Rows(l, 5, 6);
            cache.WriteRows(l, 5, 6, k1, v1);
            var (k2, v2) = Rows(l, 20, 3);
            cache.WriteRows(l, 20, 3, k2, v2);
        }
        Assert.Equal(MaxLen, cache.CurrentLength);

        for (int l = 0; l < Layers; l++)
        {
            var rk = new float[MaxLen * Stride];
            var rv = new float[MaxLen * Stride];
            cache.ReadRows(l, 0, MaxLen, rk, rv);
            for (int r = 0; r < MaxLen; r++)
                for (int c = 0; c < Stride; c++)
                {
                    bool in1 = r >= 5 && r < 11, in2 = r >= 20 && r < 23;
                    float ek = in1 || in2 ? Round(Val(l, r, c, false), storage) : 0f;
                    float ev = in1 || in2 ? Round(Val(l, r, c, true), storage) : 0f;
                    Assert.True(ek == rk[r * Stride + c], $"K l={l} r={r} c={c}: expected {ek}, got {rk[r * Stride + c]}");
                    Assert.True(ev == rv[r * Stride + c], $"V l={l} r={r} c={c}: expected {ev}, got {rv[r * Stride + c]}");
                }

            // Partial read of just the second chunk.
            var pk = new float[3 * Stride];
            var pv = new float[3 * Stride];
            cache.ReadRows(l, 20, 3, pk, pv);
            for (int r = 0; r < 3; r++)
                Assert.Equal(Round(Val(l, 20 + r, 0, false), storage), pk[r * Stride]);
        }
    }

    [SkippableFact]
    public void WriteRows_PastMaxLength_AndReadPastCurrentLength_Throw()
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device");
        using var dev = VulkanDevice.Create();
        using var cache = new VulkanKvCache(dev, new[] { Stride }, MaxLen);
        var (k, v) = Rows(0, 0, 4);
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.WriteRows(0, MaxLen - 3, 4, k, v));
        cache.WriteRows(0, 0, 4, k, v);
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.ReadRows(0, 2, 4, new float[4 * Stride], new float[4 * Stride]));
        Assert.Throws<ArgumentException>(() => cache.WriteRows(0, 0, 4, k[..^1], v));
    }

    [SkippableTheory]
    [InlineData(VulkanKvStorage.F32)]
    [InlineData(VulkanKvStorage.F16)]
    public void CopyRowsFrom_DeviceToDevice_PlacesRowsAtDestinationOffset(VulkanKvStorage storage)
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device");
        using var dev = VulkanDevice.Create();
        using var src = new VulkanKvCache(dev, new[] { Stride, Stride }, MaxLen, storage);
        using var dst = new VulkanKvCache(dev, new[] { Stride, Stride }, MaxLen, storage);
        for (int l = 0; l < Layers; l++)
        {
            var (k, v) = Rows(l, 0, 12);
            src.WriteRows(l, 0, 12, k, v);
            dst.CopyRowsFrom(src, l, srcStart: 4, dstStart: 30, count: 5);
        }
        Assert.Equal(35, dst.CurrentLength);
        for (int l = 0; l < Layers; l++)
        {
            var rk = new float[5 * Stride];
            var rv = new float[5 * Stride];
            dst.ReadRows(l, 30, 5, rk, rv);
            for (int r = 0; r < 5; r++)
                for (int c = 0; c < Stride; c++)
                {
                    Assert.Equal(Round(Val(l, 4 + r, c, false), storage), rk[r * Stride + c]);
                    Assert.Equal(Round(Val(l, 4 + r, c, true), storage), rv[r * Stride + c]);
                }
        }
    }

    [SkippableFact]
    public void Cache_ImplementsPositionedSeam()
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device");
        using var dev = VulkanDevice.Create();
        using var cache = new VulkanKvCache(dev, new[] { Stride }, MaxLen);
        Assert.IsAssignableFrom<IPositionedHostKvCache>(cache);
    }
}
