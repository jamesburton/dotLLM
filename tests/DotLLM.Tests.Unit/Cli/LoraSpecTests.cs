using DotLLM.Cli.Commands;
using Xunit;

namespace DotLLM.Tests.Unit.Cli;

public sealed class LoraSpecTests
{
    [Fact]
    public void Bare_Path_Has_Weight_One()
    {
        var s = LoraSpec.Parse("C:/x/adapter");
        Assert.Equal("C:/x/adapter", s.Path);
        Assert.Equal(1f, s.Weight);
    }

    [Fact]
    public void Weighted_Path_Parses()
    {
        var s = LoraSpec.Parse("C:/x/adapter=0.7");
        Assert.Equal("C:/x/adapter", s.Path);
        Assert.Equal(0.7f, s.Weight, 5);
    }

    [Fact]
    public void Drive_Colon_Not_Treated_As_Weight()
    {
        var s = LoraSpec.Parse("C:/x");
        Assert.Equal("C:/x", s.Path);
        Assert.Equal(1f, s.Weight);
    }
}
