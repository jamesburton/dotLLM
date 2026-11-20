using DotLLM.Server;
using DotLLM.Tests.Unit.Engine;
using Xunit;

namespace DotLLM.Tests.Unit.Cli;

/// <summary>
/// #1892: <c>run</c>, <c>chat</c> and <c>bench</c> build the same in-place TurboQuant cache <c>serve</c> does (CPU, the tiny Qwen fixture). They
/// used to refuse before loading (#1888), and before that to drop the bit width and run the default cache while the summary said tq.
/// The counter is process-wide and other tests run beside this one, so each test asserts the counter rose and the command succeeded, with
/// the cache type itself asserted in <c>TurboQuantInPlaceCapabilityTests</c>.
/// </summary>
[Collection(CliConsoleCollection.Name)]
public sealed class CliTurboQuantInPlaceCommandTests : IClassFixture<PrefixCacheFailedRequestTests.TinyQwen>
{
    private readonly PrefixCacheFailedRequestTests.TinyQwen _tiny;
    public CliTurboQuantInPlaceCommandTests(PrefixCacheFailedRequestTests.TinyQwen tiny) => _tiny = tiny;

    [Fact]
    public void Run_Tq3_BuildsTheInPlaceCache_AndReportsItsBytes()
    {
        long before = TurboQuantKv.InPlaceCachesBuilt, scratch = TurboQuantKv.ScratchCachesBuilt;
        var (code, output) = CliTurboQuantCommandTests.Run("run", _tiny.GgufPath, "-p", "ab", "--max-tokens", "2",
            "--device", "cpu", "--cache-type-k", "tq3", "--cache-type-v", "tq3");
        Assert.True(code == 0, output);
        Assert.True(TurboQuantKv.InPlaceCachesBuilt >= before + 1, "run did not build the in-place cache");
        Assert.Contains("tq3", output);   // the KV line of the summary names the format: the bit width reached the cache
        Assert.DoesNotContain("TurboQuant KV does not run", output);
        Assert.True(TurboQuantKv.ScratchCachesBuilt >= scratch);
    }

    [Fact]
    public void Run_Tq4q_StaysOnTheScratchCache_OnTheCpu()
    {
        long before = TurboQuantKv.ScratchCachesBuilt;
        var (code, output) = CliTurboQuantCommandTests.Run("run", _tiny.GgufPath, "-p", "ab", "--max-tokens", "2",
            "--device", "cpu", "--cache-type-k", "tq4q", "--cache-type-v", "tq4q");
        Assert.True(code == 0, output);
        Assert.True(TurboQuantKv.ScratchCachesBuilt >= before + 1, "tq*q should still use the dequant-scratch CPU cache until step 5");
    }

    [Fact]
    public void Bench_Tq3_BuildsTheInPlaceCache()
    {
        long before = TurboQuantKv.InPlaceCachesBuilt;
        var (code, output) = CliTurboQuantCommandTests.Run("bench", _tiny.GgufPath, "--device", "cpu", "--kv-quant-k", "tq3", "--kv-quant-v", "tq3",
            "--prompt-tokens", "8", "--gen-tokens", "2", "--reps", "1");
        Assert.True(code == 0, output);
        Assert.True(TurboQuantKv.InPlaceCachesBuilt >= before + 1, "bench did not build the in-place cache");
        Assert.DoesNotContain("is ignored", output);   // the CPU bench cache used to warn that the KV flags were ignored
    }

    [Fact]
    public void Chat_Tq3_BuildsTheInPlaceCache()
    {
        long before = TurboQuantKv.InPlaceCachesBuilt;
        var stdin0 = Console.In;
        Console.SetIn(new StringReader("/exit\n"));
        try
        {
            var (code, output) = CliTurboQuantCommandTests.Run("chat", _tiny.GgufPath, "--device", "cpu", "--cache-type-k", "tq3", "--cache-type-v", "tq3");
            Assert.True(code == 0, output);
            Assert.DoesNotContain("TurboQuant KV does not run", output);
            Assert.Contains("TurboQuant 3-bit", output);
        }
        finally { Console.SetIn(stdin0); }
        // chat builds its cache per request: with no turn taken no cache exists yet, so the line above is the proof the in-place path was chosen.
        Assert.True(TurboQuantKv.InPlaceCachesBuilt >= before);
    }

    [Fact]
    public void Run_Tq3_OnAHybridHost_IsRefusedUpFront_CitingTheHostGap()
    {
        // The synthetic Qwen3.5 hybrid-dense model: a CPU host without the TurboQuant twins. Refused after the load with #1900, before any
        // token: not a NotSupportedException from the first attention layer.
        string path = Path.Combine(Path.GetTempPath(), "dotllm-1892-" + Guid.NewGuid().ToString("N") + ".gguf");
        try
        {
            DotLLM.Models.Gguf.SyntheticQwen35HybridDenseMtpGguf.Write(path, withMtp: false);
            var (code, output) = CliTurboQuantCommandTests.Run("run", path, "-p", "ab", "--max-tokens", "2", "--device", "cpu",
                "--cache-type-k", "tq3", "--cache-type-v", "tq3");
            Assert.NotEqual(0, code);
            Assert.Contains("#1900", output);
        }
        finally { try { File.Delete(path); } catch { /* best effort */ } }
    }
}
