using System.ComponentModel;
using DotLLM.Cli.Commands;
using DotLLM.Cli.Helpers;
using DotLLM.Server;
using Xunit;

namespace DotLLM.Tests.Unit.Cli;

/// <summary>Issue #790: the CLI's <c>--device</c> option - defaults, validation, and the CPU-only checkpoint rule.</summary>
public sealed class DeviceOptionTests
{
    private static string? AttributeDefault<T>() =>
        (string?)typeof(T).GetProperty("Device")!.GetCustomAttributes(typeof(DefaultValueAttribute), false)
            .Cast<DefaultValueAttribute>().Single().Value;

    [Fact]
    public void RunChatBench_DefaultToAuto_BothTheInitializerAndTheSpectreAttribute()
    {
        Assert.Equal("auto", new RunCommand.Settings().Device);
        Assert.Equal("auto", new ChatCommand.Settings().Device);
        Assert.Equal("auto", new BenchCommand.Settings().Device);
        Assert.Equal("auto", AttributeDefault<RunCommand.Settings>());
        Assert.Equal("auto", AttributeDefault<ChatCommand.Settings>());
        Assert.Equal("auto", AttributeDefault<BenchCommand.Settings>());
    }

    [Fact]
    public void Bench_Validate_RejectsUnknownDevices_AndAcceptsEverySpelling()
    {
        Assert.False(new BenchCommand.Settings { Device = "vulkn" }.Validate().Successful);
        foreach (string ok in new[] { "auto", "cpu", "vulkan", "gpu", "cuda", "cuda:1", "gpu:0" })
            Assert.True(new BenchCommand.Settings { Device = ok }.Validate().Successful, ok);
    }

    [Theory]
    [InlineData("auto", true)]
    [InlineData("vulkan", true)]
    [InlineData("cuda", true)]
    [InlineData("gpu:1", true)]
    [InlineData("cpu", true)]
    [InlineData("vulkan:3", false)]
    [InlineData("metal", false)]
    public void Validate_MatchesTheSharedParser(string device, bool valid) =>
        Assert.Equal(valid, DeviceCli.Validate(device) is null);

    [Theory]
    [InlineData("vulkan")]
    [InlineData("gpu")]
    [InlineData("cuda")]
    public void CpuOnlyCheckpoint_WithExplicitGpu_IsAnError(string device)
    {
        var ex = Assert.Throws<DeviceUnavailableException>(() => DeviceCli.ResolveCpuOnly(device, "HuggingFace safetensors checkpoints", "m"));
        Assert.Contains("--device cpu", ex.Message);
    }

    [Fact]
    public void CpuOnlyCheckpoint_UnderAuto_IsCpuWithAWarning_AndUnderCpuIsSilent()
    {
        var auto = DeviceCli.ResolveCpuOnly("auto", "HuggingFace safetensors checkpoints", "m");
        Assert.Equal("cpu", auto.Resolved);
        Assert.NotNull(auto.Warning);
        Assert.Contains("auto -> cpu", auto.Line);

        var cpu = DeviceCli.ResolveCpuOnly("cpu", "HuggingFace safetensors checkpoints", "m");
        Assert.Null(cpu.Warning);
        Assert.Equal("device: cpu", cpu.Line);
    }
}
