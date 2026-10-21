using System.Reflection;
using DotLLM.Core.Configuration;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Issue #1083: <see cref="NemotronHMoeWeights.WithExpertBanks"/> points the CUDA offload's CPU and iGPU cold legs at its pinned copy of
/// the routed banks, so the banks are resident once instead of being paged in a second time from the mmap. The copy must change the two
/// bank pointers and nothing else. The check is reflective, so a property added to the class later and forgotten in the copy fails here
/// instead of silently reaching the cold legs as its default.
/// </summary>
public sealed class NemotronHMoeWeightsRebaseTests
{
    private static readonly string[] Banks = [nameof(NemotronHMoeWeights.UpExpsWeight), nameof(NemotronHMoeWeights.DownExpsWeight)];

    [Fact]
    public void WithExpertBanks_ChangesOnlyTheTwoBankPointers()
    {
        var original = Fixture();

        var rebased = original.WithExpertBanks(0xA000, 0xB000);

        Assert.NotSame(original, rebased);
        Assert.Equal((nint)0xA000, rebased.UpExpsWeight);
        Assert.Equal((nint)0xB000, rebased.DownExpsWeight);
        Assert.Equal((nint)0x2000, original.UpExpsWeight);   // the original is untouched
        Assert.Equal((nint)0x3000, original.DownExpsWeight);
        foreach (var p in Properties())
        {
            if (Banks.Contains(p.Name)) continue;
            Assert.True(Equals(p.GetValue(original), p.GetValue(rebased)), $"{p.Name} was not carried over by WithExpertBanks");
        }
    }

    [Fact]
    public void Fixture_SetsEveryPropertyToANonDefault()
    {
        // Guards the test above: a property left at its default compares equal even when WithExpertBanks forgot to copy it.
        var w = Fixture();
        foreach (var p in Properties())
        {
            if (!p.CanWrite) continue;   // computed (HasLatent)
            object? d = p.PropertyType.IsValueType ? Activator.CreateInstance(p.PropertyType) : null;
            Assert.False(Equals(p.GetValue(w), d), $"{p.Name} is at its default in Fixture(); give it a distinct value");
        }
    }

    private static IEnumerable<PropertyInfo> Properties() =>
        typeof(NemotronHMoeWeights).GetProperties(BindingFlags.Public | BindingFlags.Instance);

    /// <summary>Every settable property at a distinct non-default value (F32 is QuantizationType 0, so it is not used).</summary>
    private static NemotronHMoeWeights Fixture() => new()
    {
        GateInpWeight = 0x1000,
        GateInpQuantType = QuantizationType.Q8_0,
        SelectionBias = [0.5f, -0.25f],
        UpExpsWeight = 0x2000,
        UpExpsQuantType = QuantizationType.IQ2_XXS,
        UpPerExpertBytes = 1234,
        DownExpsWeight = 0x3000,
        DownExpsQuantType = QuantizationType.Q4_K,
        DownPerExpertBytes = 5678,
        UpShexpWeight = 0x4000,
        UpShexpQuantType = QuantizationType.Q5_K,
        DownShexpWeight = 0x5000,
        DownShexpQuantType = QuantizationType.Q6_K,
        NumExperts = 7,
        NumExpertsPerTok = 3,
        MoeIntermediateSize = 96,
        SharedIntermediateSize = 160,
        NormalizeWeights = true,
        WeightsScale = 2.5f,
        LatentSize = 48,
        LatentDownWeight = 0x6000,
        LatentDownQuantType = QuantizationType.F16,
        LatentUpWeight = 0x7000,
        LatentUpQuantType = QuantizationType.Q4_0,
    };
}
