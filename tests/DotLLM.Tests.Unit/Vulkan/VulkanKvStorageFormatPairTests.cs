using System;
using DotLLM.Core.Attention;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>#1892: <see cref="VulkanKvStorage"/> maps onto the per-side <see cref="KvFormatPair"/> and back. No GPU needed.</summary>
public sealed class VulkanKvStorageFormatPairTests
{
    [Fact]
    public void EveryStorage_RoundTripsThroughThePair()
    {
        foreach (var st in Enum.GetValues<VulkanKvStorage>())
            Assert.Equal(st, VulkanKvStorageExtensions.FromFormatPair(st.ToFormatPair()));
    }

    [Fact]
    public void Tq_IsTqOnBothSides_AndMixedTqHasNoStorage()
    {
        Assert.Equal(KvFormatPair.Both(KvSideFormat.Tq), VulkanKvStorage.Tq.ToFormatPair());
        Assert.Throws<NotSupportedException>(() =>
            VulkanKvStorageExtensions.FromFormatPair(new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Tq)));
    }

    [Fact]
    public void Tq_RefusesElementAndBlockSizing()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => VulkanKvStorage.Tq.ElementBytes());
        Assert.Throws<ArgumentOutOfRangeException>(() => VulkanKvStorage.Tq.ShaderName("attention_f32"));
    }
}
