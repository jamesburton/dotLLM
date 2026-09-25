using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Regression test for issue #185, Finding 2: <see cref="CudaQwen3HybridDenseTransformerModel"/>'s
/// internal F16 KV-cache "current length" cursor must reset when <c>Forward</c> is called with a
/// DIFFERENT (but same-<see cref="DotLLM.Core.Attention.IKvCache.MaxLength"/>) <see
/// cref="DotLLM.Core.Attention.IKvCache"/> instance -- i.e. a new logical sequence reusing an
/// identically-sized cache slot, exactly the pattern <c>BenchRunner</c>'s <c>-r N&gt;1</c> repeats
/// and <c>ContinuousBatchScheduler</c>'s session-rebuild path can also produce.
///
/// Before the fix, <c>EnsureF16KvCache</c>'s "already allocated and big enough" guard
/// (<c>maxSeqLen &lt;= _f16CacheMaxSeqLen</c>) skipped resetting <c>_f16CacheCurrentLength</c>
/// whenever the new cache's capacity did not force a reallocation, so a second, unrelated sequence
/// sharing that capacity would see the model's cursor "stuck" at the first sequence's final depth.
/// This does NOT corrupt attention logits (causal masking is position-based, not
/// cursor-based -- see the issue) so a black-box logits comparison cannot discriminate broken vs.
/// fixed; this test asserts directly on the internal cursor via
/// <see cref="CudaQwen3HybridDenseTransformerModel.DebugF16CacheCurrentLengthForTest"/>.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public class CudaQwen3HybridDenseKvCacheReuseResetTest
{
    private const string ModelPathEnvVar = "DOTLLM_BONSAI_PQ2_0_GGUF";
    private const string FileName = "Ternary-Bonsai-27B-Q2_0.gguf";

    private readonly ITestOutputHelper _out;
    public CudaQwen3HybridDenseKvCacheReuseResetTest(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    [SkippableFact]
    public void FreshSameSizeKvCache_ResetsCursor_InsteadOfStickingAtPriorSequenceDepth()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? path = ResolveFixturePath();
        Skip.If(path is null,
            $"Bonsai PQ2_0 GGUF fixture not found. Set {ModelPathEnvVar}, or place {FileName} under "
            + "~/.dotllm/models/PrismML/Ternary-Bonsai-27B-GGUF/ or ~/.dotllm/test-cache/PrismML/Ternary-Bonsai-27B-GGUF/.");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.Qwen3HybridDense, config.Architecture);

        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!);

        const int cacheSize = 256;
        int[] promptTokens = [1, 100, 2000, 30000, 5, 777, 42, 9001];
        int[] promptPositions = [0, 1, 2, 3, 4, 5, 6, 7];
        const int decodeSteps = 32;

        // Sequence A: fresh cache, prefill + decode to a real final depth well past promptLen.
        using (var cacheA = model.CreateKvCache(cacheSize))
        {
            using (model.Forward(promptTokens, promptPositions, deviceId: -1, cacheA)) { }

            int nextPos = promptTokens.Length;
            for (int i = 0; i < decodeSteps; i++)
            {
                int[] single = [(7 + i * 13) % 40000];
                int[] singlePos = [nextPos];
                using (model.Forward(single, singlePos, deviceId: -1, cacheA)) { }
                nextPos++;
            }

            int expectedDepthA = promptTokens.Length + decodeSteps;
            Assert.Equal(expectedDepthA, model.DebugF16CacheCurrentLengthForTest);
            _out.WriteLine($"Sequence A: cursor == {model.DebugF16CacheCurrentLengthForTest} (expected {expectedDepthA}).");
        }

        // Sequence B: a BRAND NEW IKvCache instance, same MaxLength (cacheSize) as sequence A's --
        // exactly the shape BenchRunner's rep loop and a same-size scheduler rebuild produce. Its
        // own prefill is shallower than sequence A's final depth: if the cursor is stuck (pre-fix
        // bug), it will still read as sequence A's final depth instead of this prefill's own length.
        using (var cacheB = model.CreateKvCache(cacheSize))
        {
            using (model.Forward(promptTokens, promptPositions, deviceId: -1, cacheB)) { }

            int expectedDepthB = promptTokens.Length;
            int actualDepthB = model.DebugF16CacheCurrentLengthForTest;
            _out.WriteLine($"Sequence B: cursor == {actualDepthB} (expected {expectedDepthB}).");
            Assert.True(actualDepthB == expectedDepthB,
                $"Sequence B's KV-cache cursor is {actualDepthB}, expected {expectedDepthB} -- " +
                "the cursor did not reset for the new same-size IKvCache instance and is still " +
                "reflecting sequence A's prior final depth (issue #185, Finding 2).");
        }
    }

    private static string? ResolveFixturePath()
    {
        string? envPath = Environment.GetEnvironmentVariable(ModelPathEnvVar);
        if (!string.IsNullOrWhiteSpace(envPath) && File.Exists(envPath))
            return envPath;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string[] candidates =
        [
            Path.Combine(home, ".dotllm", "models", "PrismML", "Ternary-Bonsai-27B-GGUF", FileName),
            Path.Combine(home, ".dotllm", "test-cache", "PrismML", "Ternary-Bonsai-27B-GGUF", FileName),
        ];
        foreach (string candidate in candidates)
            if (File.Exists(candidate))
                return candidate;

        return null;
    }

    private static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }
}
