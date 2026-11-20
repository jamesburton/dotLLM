using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Server;
using DotLLM.Tests.Unit.Engine;
using Xunit;

namespace DotLLM.Tests.Unit.Cli;

/// <summary>
/// #1892: <c>run</c>, <c>chat</c> and <c>bench</c> build the same in-place TurboQuant cache <c>serve</c> does (CPU, the tiny Qwen fixture). The
/// evidence is scoped to each test: <see cref="TurboQuantKv.Observer"/> is async-local, so it sees only the caches THIS command built, and the
/// tests assert their type and exact bytes (the process-wide counters rise for any parallel test and prove nothing).
/// </summary>
[Collection(CliConsoleCollection.Name)]
public sealed class CliTurboQuantInPlaceCommandTests : IClassFixture<PrefixCacheFailedRequestTests.TinyQwen>
{
    private readonly PrefixCacheFailedRequestTests.TinyQwen _tiny;
    public CliTurboQuantInPlaceCommandTests(PrefixCacheFailedRequestTests.TinyQwen tiny) => _tiny = tiny;

    /// <summary>Runs a command with the observer attached; returns what it built.</summary>
    private static (int Code, string Output, List<IKvCache> Built) RunObserved(string stdin, params string[] args)
    {
        var built = new List<IKvCache>();
        TurboQuantKv.Observer = c => { lock (built) built.Add(c); };
        try
        {
            var (code, output) = CliTurboQuantCommandTests.RunWithInput(stdin.Length == 0 ? null : stdin, args);
            return (code, output, built);
        }
        finally { TurboQuantKv.Observer = null; }
    }

    private void AssertInPlace(IReadOnlyList<IKvCache> built, int bits)
    {
        Assert.NotEmpty(built);
        foreach (var c in built)
        {
            var tq = Assert.IsType<TurboQuantInPlaceKvCache>(c);
            var layout = new TurboQuantLayout(_tiny.Config.HeadDim, _tiny.Config.NumKvHeads, bits, TurboQuantLayout.DefaultNorm);
            long expected = (long)KvGeometry.SlotCount(_tiny.Config) * 2 * (layout.CodeBytesPerLayer(tq.MaxLength) + layout.NormBytesPerLayer(tq.MaxLength));
            Assert.Equal(expected, tq.AllocatedBytes);   // codes + norms at the requested width: nothing wider
        }
    }

    [Fact]
    public void Run_Tq3_BuildsTheInPlaceCache_OfTheRequestedWidth()
    {
        var (code, output, built) = RunObserved("", "run", _tiny.GgufPath, "-p", "ab", "--max-tokens", "2",
            "--device", "cpu", "--cache-type-k", "tq3", "--cache-type-v", "tq3");
        Assert.True(code == 0, output);
        AssertInPlace(built, bits: 3);
        Assert.Contains("tq3", output);
    }

    [Fact]
    public void Run_Tq4q_StaysOnTheScratchCache_OnTheCpu()
    {
        var (code, output, built) = RunObserved("", "run", _tiny.GgufPath, "-p", "ab", "--max-tokens", "2",
            "--device", "cpu", "--cache-type-k", "tq4q", "--cache-type-v", "tq4q");
        Assert.True(code == 0, output);
        Assert.NotEmpty(built);
        Assert.All(built, c => Assert.IsType<TurboQuantKvCache>(c));
    }

    [Fact]
    public void Bench_Tq3_BuildsTheInPlaceCache()
    {
        var (code, output, built) = RunObserved("", "bench", _tiny.GgufPath, "--device", "cpu", "--kv-quant-k", "tq3", "--kv-quant-v", "tq3",
            "--prompt-tokens", "8", "--gen-tokens", "2", "--reps", "1");
        Assert.True(code == 0, output);
        AssertInPlace(built, bits: 3);
        Assert.DoesNotContain("is ignored", output);   // the CPU bench cache used to warn that the KV flags were ignored
    }

    [Fact]
    public void Chat_Tq3_BuildsTheInPlaceCache_ForItsTurn()
    {
        // One real turn: the generator asks the factory chat chose for a cache, which is the evidence it chose the in-place one.
        var (code, output, built) = RunObserved("ab\n/exit\n", "chat", _tiny.GgufPath, "--device", "cpu",
            "--cache-type-k", "tq3", "--cache-type-v", "tq3", "--max-tokens", "2");
        Assert.True(code == 0, output);
        Assert.True(built.Count > 0, output);
        AssertInPlace(built, bits: 3);
    }

    // ---------------------------------------------------------------- a host without the twins is refused up front

    /// <summary>
    /// The synthetic Qwen3.5 hybrid-dense model is a CPU host without the TurboQuant twins. It must be refused by the post-load pass
    /// (wording only that pass produces), before any cache is built and before any generation output - never by the
    /// <c>InvalidOperationException</c> the plain attention kernel throws mid-forward.
    /// </summary>
    private static void AssertRefusedUpFront(int code, string output, IReadOnlyList<IKvCache> built)
    {
        Assert.NotEqual(0, code);
        Assert.Contains("no TurboQuant twin", output);                 // the resolver's wording: only the up-front pass says it
        Assert.Contains("#1900", output);
        Assert.DoesNotContain("A TurboQuant cache stores rotated codes", output);   // the mid-forward throw of Attention.Execute
        Assert.DoesNotContain("there is no other TurboQuant cache to build", output);   // nor the factory's own last-resort throw
        Assert.Empty(built);                                            // nothing was built, so nothing could have been forwarded
    }

    private string HybridGguf()
    {
        string path = Path.Combine(Path.GetTempPath(), "dotllm-1892-" + Guid.NewGuid().ToString("N") + ".gguf");
        DotLLM.Models.Gguf.SyntheticQwen35HybridDenseMtpGguf.Write(path, withMtp: false);
        return path;
    }

    [Fact]
    public void Run_OnAHybridHost_IsRefusedUpFront()
    {
        string path = HybridGguf();
        try
        {
            var (code, output, built) = RunObserved("", "run", path, "-p", "ab", "--max-tokens", "2", "--device", "cpu",
                "--cache-type-k", "tq3", "--cache-type-v", "tq3");
            AssertRefusedUpFront(code, output, built);
        }
        finally { try { File.Delete(path); } catch { /* best effort */ } }
    }

    [Fact]
    public void Chat_OnAHybridHost_IsRefusedUpFront()
    {
        string path = HybridGguf();
        try
        {
            var (code, output, built) = RunObserved("ab\n/exit\n", "chat", path, "--device", "cpu", "--cache-type-k", "tq3", "--cache-type-v", "tq3");
            AssertRefusedUpFront(code, output, built);
        }
        finally { try { File.Delete(path); } catch { /* best effort */ } }
    }

    [Fact]
    public void Bench_OnAHybridHost_IsRefusedUpFront()
    {
        string path = HybridGguf();
        try
        {
            var (code, output, built) = RunObserved("", "bench", path, "--device", "cpu", "--kv-quant-k", "tq3", "--kv-quant-v", "tq3",
                "--prompt-tokens", "8", "--gen-tokens", "2", "--reps", "1");
            AssertRefusedUpFront(code, output, built);
            Assert.DoesNotContain("tok/s", output);   // no measurement was taken
        }
        finally { try { File.Delete(path); } catch { /* best effort */ } }
    }

    /// <summary>A host that counts Forward calls: the refusal and the factory refusal happen with zero forwards.</summary>
    private sealed class RecordingHost(ModelConfig config) : IModel
    {
        public int Forwards;
        public ModelConfig Config { get; } = config;
        public long ComputeMemoryBytes => 0;
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId) { Forwards++; throw new InvalidOperationException("forwarded"); }
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache) { Forwards++; throw new InvalidOperationException("forwarded"); }
        public void Dispose() { }
    }

    [Fact]
    public void ARefusedHost_IsRefusedWithZeroForwardCalls()
    {
        var host = new RecordingHost(_tiny.Config);
        var kv = new DotLLM.Core.Configuration.KvCacheConfig(DotLLM.Core.Configuration.KvCacheDType.TurboQuant,
            DotLLM.Core.Configuration.KvCacheDType.TurboQuant, TurboQuantBits: 3);
        foreach (var command in new[] { KvCommand.Run, KvCommand.Chat, KvCommand.Bench, KvCommand.Serve })
        {
            string? reason = TurboQuantKv.PostLoadRefusal(kv, null, command, paged: false, host.Config, host, false, false);
            Assert.Contains("no TurboQuant twin", reason);
        }
        Assert.Throws<NotSupportedException>(() => TurboQuantKv.CreateFactory(host, kv));
        Assert.Equal(0, host.Forwards);
    }
}
