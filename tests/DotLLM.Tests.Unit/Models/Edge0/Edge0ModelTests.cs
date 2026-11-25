using DotLLM.Models.Edge0;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Edge0;

/// <summary>The diagnostic column-scale switch (<see cref="Edge0Model.ColumnScaleEnv"/>) must never select a wrong arm silently.</summary>
public sealed class Edge0ModelTests
{
    [Theory]
    [InlineData(null, Edge0ColumnScaleMode.Inverse)]
    [InlineData("", Edge0ColumnScaleMode.Inverse)]
    [InlineData("inverse", Edge0ColumnScaleMode.Inverse)]
    [InlineData("Direct", Edge0ColumnScaleMode.Direct)]
    [InlineData(" none ", Edge0ColumnScaleMode.None)]
    public void ColumnScaleMode_AcceptsTheThreeNames(string? value, Edge0ColumnScaleMode expected)
        => Assert.Equal(expected, Edge0Model.ColumnScaleModeFromEnvironment(value));

    [Theory]
    [InlineData("1")]          // Enum.TryParse would read this as Direct
    [InlineData("2")]          // ... and this as None
    [InlineData("7")]          // ... and this as an undefined value that the pack treats as Inverse
    [InlineData("invrese")]    // a typo must not silently run the default
    public void ColumnScaleMode_RefusesAnythingElse(string value)
        => Assert.Throws<ArgumentException>(() => Edge0Model.ColumnScaleModeFromEnvironment(value));
}
