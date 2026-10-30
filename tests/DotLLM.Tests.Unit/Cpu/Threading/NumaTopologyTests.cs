using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Threading;

public sealed class NumaTopologyTests
{
    /// <summary>
    /// Windows EfficiencyClass: HIGHER = higher-performing. Meteor Lake (Core Ultra 7 155H) reports three classes
    /// (LP-E, E, P); only the max is a P-core. The inverted MIN rule pinned --pcore-only to the slow cores (~6x).
    /// </summary>
    [Theory]
    [InlineData(2, CoreType.Performance)]
    [InlineData(1, CoreType.Efficiency)]
    [InlineData(0, CoreType.Efficiency)]
    public void Hybrid_classification_treats_the_highest_efficiency_class_as_P_cores(byte efficiencyClass, CoreType expected)
    {
        byte[] meteorLake = [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0];
        Assert.Equal(expected, NumaTopology.ClassifyHybrid(efficiencyClass, meteorLake));
    }

    [Fact]
    public void Detect_ReturnsNonNull()
    {
        var topology = NumaTopology.Detect();
        Assert.NotNull(topology);
    }

    [Fact]
    public void Detect_HasAtLeastOneNode()
    {
        var topology = NumaTopology.Detect();
        Assert.True(topology.NumaNodeCount >= 1);
    }

    [Fact]
    public void Detect_ProcessorCount_Positive()
    {
        var topology = NumaTopology.Detect();
        Assert.True(topology.Processors.Count > 0);
    }

    [Fact]
    public void PerformanceCoreIds_NonEmpty()
    {
        var topology = NumaTopology.Detect();
        Assert.NotEmpty(topology.PerformanceCoreIds);
    }

    [Fact]
    public void MemoryChannelEstimate_Positive()
    {
        var topology = NumaTopology.Detect();
        Assert.True(topology.MemoryChannelEstimate > 0);
    }

    [Fact]
    public void ProcessorsByNumaNode_ContainsAllProcessors()
    {
        var topology = NumaTopology.Detect();
        int total = topology.ProcessorsByNumaNode.Values.Sum(list => list.Count);
        Assert.Equal(topology.Processors.Count, total);
    }

    [Fact]
    public void ParseCpuList_SingleValues()
    {
        var result = NumaTopology.ParseCpuList("0,3,5").ToList();
        Assert.Equal(new[] { 0, 3, 5 }, result);
    }

    [Fact]
    public void ParseCpuList_Ranges()
    {
        var result = NumaTopology.ParseCpuList("0-3").ToList();
        Assert.Equal(new[] { 0, 1, 2, 3 }, result);
    }

    [Fact]
    public void ParseCpuList_Mixed()
    {
        var result = NumaTopology.ParseCpuList("0-2,5,7-9").ToList();
        Assert.Equal(new[] { 0, 1, 2, 5, 7, 8, 9 }, result);
    }

    [Fact]
    public void ParseCpuList_Empty()
    {
        var result = NumaTopology.ParseCpuList("").ToList();
        Assert.Empty(result);
    }
}
