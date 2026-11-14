using System.ComponentModel;
using System.Diagnostics;
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
    [InlineData("vulkan:3", true)]
    [InlineData("hip", true)]
    [InlineData("hip:1", true)]
    [InlineData("hip:-1", false)]
    [InlineData("vulkan:x", false)]
    [InlineData("metal", false)]
    public void Validate_MatchesTheSharedParser(string device, bool valid) =>
        Assert.Equal(valid, DeviceCli.Validate(device) is null);

    [Theory]
    [InlineData("vulkan")]
    [InlineData("gpu")]
    [InlineData("cuda")]
    [InlineData("hip")]
    [InlineData("hip:1")]
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

    [Theory]
    [InlineData("run", "--cache-type-k", "f32")]
    [InlineData("run", "--kv-quant-v", "q4_0")]
    [InlineData("chat", "--cache-type-v", "q4_0")]
    public async Task HipInvalidCacheFlags_FailBeforeResolvingTheModel(string command, string option, string value)
    {
        string[] args = command == "run"
            ? [command, "does-not-exist.gguf", "--device", "hip:1", "-p", "hello", option, value]
            : [command, "does-not-exist.gguf", "--device", "hip:1", option, value];
        // Commands write through the static Spectre console. A child process captures the actual CLI entry point
        // without replacing a process-wide console shared by other tests.
        var start = new ProcessStartInfo("dotnet") { RedirectStandardOutput = true, RedirectStandardError = true,
            UseShellExecute = false, CreateNoWindow = true };
        start.ArgumentList.Add(typeof(RunCommand).Assembly.Location);
        foreach (string arg in args) start.ArgumentList.Add(arg);
        start.Environment["DOTLLM_NO_PROMPT"] = "1";
        using var process = Process.Start(start)!;
        Task<string> stdout = process.StandardOutput.ReadToEndAsync();
        Task<string> stderr = process.StandardError.ReadToEndAsync();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        try { await process.WaitForExitAsync(timeout.Token); }
        catch { process.Kill(entireProcessTree: true); throw; }
        Assert.Equal(1, process.ExitCode);
        Assert.Contains("only F16", await stdout + await stderr);
    }
}
