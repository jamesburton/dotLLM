using System.Text.Json;
using System.Text.Json.Serialization.Metadata;
using DotLLM.Server.Tuning;
using Xunit;

namespace DotLLM.Tests.Unit.Server.Tuning;

/// <summary>
/// Guards <see cref="TuningJsonContext"/> against the source-generation trap in CLAUDE.md: an initializer on an <c>init</c> property is
/// silently dropped on deserialization. Direct construction is the only oracle, so each DTO is constructed and compared with <c>{}</c>.
/// </summary>
public sealed class TuningJsonContextDefaultsTests
{
    [Theory]
    [InlineData(typeof(TuningResult))]
    [InlineData(typeof(TuningKey))]
    [InlineData(typeof(TunedSettings))]
    [InlineData(typeof(DeviceFingerprint))]
    [InlineData(typeof(PerfIdentity))]
    public void EveryDtoRoundTripsEmptyJsonToDirectConstruction(Type type)
    {
        object direct = Activator.CreateInstance(type)!;
        JsonTypeInfo info = TuningJsonContext.Default.GetTypeInfo(type)!;
        object? fromJson = JsonSerializer.Deserialize("{}", info);
        Assert.NotNull(fromJson);
        Assert.Equal(direct, fromJson);
    }

    [Fact]
    public void EveryDtoMemberIsNullable_SoNoDefaultIsExpressedByAnInitializer()
    {
        foreach (var type in new[] { typeof(TuningResult), typeof(TuningKey), typeof(TunedSettings), typeof(DeviceFingerprint), typeof(PerfIdentity) })
        {
            object direct = Activator.CreateInstance(type)!;
            foreach (var p in type.GetProperties())
                Assert.True(p.GetValue(direct) is null, $"{type.Name}.{p.Name} has a non-null default; make it nullable and resolve the default in code");
        }
    }
}
