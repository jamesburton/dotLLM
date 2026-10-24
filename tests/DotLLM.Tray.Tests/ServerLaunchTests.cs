using DotLLM.Tray.Config;
using DotLLM.Tray.Hosting;
using Xunit;

namespace DotLLM.Tray.Tests;

/// <summary>
/// Covers the command line the tray builds for a server it owns, and how it finds the executable.
/// </summary>
/// <remarks>
/// <b>Discrimination (#417), by mutation.</b> Removing the unconditional <c>--allow-model-admin</c>
/// from <see cref="ServerLaunchSpecBuilder.BuildArguments"/> failed
/// <see cref="Arguments_AlwaysEnableTheAdminApiOnAServerTheTrayOwns"/> ("Assert.Contains() Failure:
/// Item not found in collection"); reordering the positional model after the options failed
/// <see cref="Arguments_PutThePositionalModelBeforeTheOptions"/>. Both reverted.
/// </remarks>
public sealed class ServerLaunchTests
{
    [Fact]
    public void Arguments_AlwaysEnableTheAdminApiOnAServerTheTrayOwns()
    {
        // Without this flag every write route answers 403 and the tray's own controls are dead
        // buttons against a server it started itself. It is deliberately not configurable.
        var args = ServerLaunchSpecBuilder.BuildArguments(new ServerLaunchOptions());

        Assert.Contains("--allow-model-admin", args);
        Assert.Equal("serve", args[0]);
    }

    [Fact]
    public void Arguments_SuppressTheBrowserSoLoginDoesNotOpenAWindow()
    {
        Assert.Contains("--no-browser", ServerLaunchSpecBuilder.BuildArguments(new ServerLaunchOptions()));
    }

    [Fact]
    public void Arguments_PutThePositionalModelBeforeTheOptions()
    {
        // `dotllm serve [model]` takes the model positionally. Emitting it after --host would
        // make Spectre.Console.Cli bind it as the value of the preceding option.
        var args = ServerLaunchSpecBuilder.BuildArguments(
            new ServerLaunchOptions { Model = "qwen2.5-3b", Host = "127.0.0.1" });

        Assert.Equal("serve", args[0]);
        Assert.Equal("qwen2.5-3b", args[1]);
        var list = args.ToList();
        Assert.True(list.IndexOf("qwen2.5-3b") < list.IndexOf("--host"));
    }

    [Fact]
    public void Arguments_OmitEveryUnsetOptionSoTheServerKeepsItsOwnDefaults()
    {
        var args = ServerLaunchSpecBuilder.BuildArguments(new ServerLaunchOptions());

        Assert.DoesNotContain("--device", args);
        Assert.DoesNotContain("--gpu-layers", args);
        Assert.DoesNotContain("--cache-type-k", args);
        Assert.DoesNotContain("--keep-alive", args);
        Assert.DoesNotContain("--max-resident-models", args);
        Assert.DoesNotContain("--resident-memory-budget", args);
    }

    [Fact]
    public void Arguments_RenderEverySupportedOption()
    {
        // Every flag name here is copied from ServeCommand.Settings. A typo produces a server
        // that refuses to start, which is exactly what this asserts against.
        var args = ServerLaunchSpecBuilder.BuildArguments(new ServerLaunchOptions
        {
            Host = "0.0.0.0",
            Port = 9001,
            Device = "gpu:0",
            GpuLayers = 32,
            CacheTypeK = "q8_0",
            CacheTypeV = "q8_0",
            KeepAliveSeconds = -1,
            MaxResidentModels = 3,
            ResidentMemoryBudgetBytes = 8_589_934_592,
            ExtraArguments = ["--no-paged"],
        });

        AssertPair(args, "--host", "0.0.0.0");
        AssertPair(args, "--port", "9001");
        AssertPair(args, "--device", "gpu:0");
        AssertPair(args, "--gpu-layers", "32");
        AssertPair(args, "--cache-type-k", "q8_0");
        AssertPair(args, "--cache-type-v", "q8_0");
        AssertPair(args, "--keep-alive", "-1");
        AssertPair(args, "--max-resident-models", "3");
        AssertPair(args, "--resident-memory-budget", "8589934592");
        Assert.Contains("--no-paged", args);

        static void AssertPair(IReadOnlyList<string> args, string flag, string value)
        {
            var index = args.ToList().IndexOf(flag);
            Assert.True(index >= 0, $"missing {flag}");
            Assert.Equal(value, args[index + 1]);
        }
    }

    [Fact]
    public void Arguments_UseInvariantNumberFormatting()
    {
        // A da-DK or de-DE machine would otherwise emit "-1,5" for a keep-alive and the server
        // would reject the whole command line.
        var original = Thread.CurrentThread.CurrentCulture;
        Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("de-DE");
        try
        {
            var args = ServerLaunchSpecBuilder.BuildArguments(
                new ServerLaunchOptions { KeepAliveSeconds = 1.5 });
            Assert.Contains("1.5", args);
            Assert.DoesNotContain("1,5", args);
        }
        finally
        {
            Thread.CurrentThread.CurrentCulture = original;
        }
    }

    [Theory]
    [InlineData("localhost", 8080, "http://localhost:8080/")]
    [InlineData("127.0.0.1", 9000, "http://127.0.0.1:9000/")]
    // A wildcard bind is not an address a client can connect to; the tray must dial loopback.
    [InlineData("0.0.0.0", 8080, "http://localhost:8080/")]
    [InlineData("*", 8080, "http://localhost:8080/")]
    public void BaseAddress_IsSomethingAClientCanActuallyDial(string host, int port, string expected)
    {
        var uri = ServerLaunchSpecBuilder.BuildBaseAddress(new ServerLaunchOptions { Host = host, Port = port });
        Assert.Equal(expected, uri.ToString());
    }

    [Fact]
    public void Locate_PrefersAConfiguredPath()
    {
        var located = DotLlmExecutableLocator.Locate(
            @"C:\custom\dotllm.exe",
            fileExists: path => path == @"C:\custom\dotllm.exe" || path == @"C:\tray\dotllm.exe",
            processPath: @"C:\tray\dotllm-tray.exe",
            pathVariable: "");

        Assert.Equal(@"C:\custom\dotllm.exe", located);
    }

    [Fact]
    public void Locate_FallsBackToASiblingOfTheTrayExecutable()
    {
        // The release archive extracts dotllm.exe and the tray side by side. ProcessPath is used
        // rather than Assembly.Location, which is empty in the single-file publish the tray ships.
        var located = DotLlmExecutableLocator.Locate(
            configuredPath: null,
            fileExists: path => path == @"C:\tray\dotllm.exe",
            processPath: @"C:\tray\dotllm-tray.exe",
            pathVariable: "");

        Assert.Equal(@"C:\tray\dotllm.exe", located);
    }

    [Fact]
    public void Locate_FallsBackToPath()
    {
        var located = DotLlmExecutableLocator.Locate(
            configuredPath: null,
            fileExists: path => path == @"C:\tools\dotllm.exe",
            processPath: @"C:\tray\dotllm-tray.exe",
            pathVariable: @"C:\windows;C:\tools");

        Assert.Equal(@"C:\tools\dotllm.exe", located);
    }

    [Fact]
    public void Locate_ReturnsNullWhenNothingIsFound()
    {
        Assert.Null(DotLlmExecutableLocator.Locate(
            configuredPath: @"C:\missing\dotllm.exe",
            fileExists: _ => false,
            processPath: @"C:\tray\dotllm-tray.exe",
            pathVariable: @"C:\windows"));
    }

    [Fact]
    public void ConfiguredPathThatDoesNotExist_DoesNotBlockTheOtherProbes()
    {
        // A stale configured path from an old install must degrade to discovery, not to failure.
        var located = DotLlmExecutableLocator.Locate(
            configuredPath: @"C:\old\dotllm.exe",
            fileExists: path => path == @"C:\tray\dotllm.exe",
            processPath: @"C:\tray\dotllm-tray.exe",
            pathVariable: "");

        Assert.Equal(@"C:\tray\dotllm.exe", located);
    }

    [Fact]
    public void TraySettings_ProjectOntoLaunchOptionsAndABaseAddress()
    {
        var settings = new TraySettings
        {
            Host = "localhost",
            Port = 8123,
            StartupModel = "m",
            Device = "cpu",
            GpuLayers = 0,
            CacheTypeK = "f32",
            CacheTypeV = "f32",
        };

        var options = settings.ToLaunchOptions();

        Assert.Equal("m", options.Model);
        Assert.Equal(8123, options.Port);
        Assert.Equal("cpu", options.Device);
        Assert.Equal("http://localhost:8123/", settings.BaseAddress.ToString());
    }
}
