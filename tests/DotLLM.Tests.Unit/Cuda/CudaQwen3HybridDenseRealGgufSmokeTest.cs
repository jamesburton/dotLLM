using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Real-weight CUDA smoke test for <see cref="CudaQwen3HybridDenseTransformerModel"/> against
/// PrismML's actual <c>Ternary-Bonsai-27B-Q2_0.gguf</c> (issue #157). Runs a short deterministic
/// prompt through prefill + a few greedy decode steps on the RTX 3060 and asserts every logits
/// row is finite (no NaN/Inf) — the GPU-side correctness gate before the baseline benchmark.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public class CudaQwen3HybridDenseRealGgufSmokeTest
{
    private const string ModelPathEnvVar = "DOTLLM_BONSAI_PQ2_0_GGUF";
    private const string FileName = "Ternary-Bonsai-27B-Q2_0.gguf";

    private readonly ITestOutputHelper _out;
    public CudaQwen3HybridDenseRealGgufSmokeTest(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    [SkippableFact]
    public void Forward_RealBonsai27B_PrefillAndDecode_ProducesFiniteLogits()
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

        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        sw.Stop();
        _out.WriteLine($"Load: {sw.Elapsed.TotalSeconds:F1}s, AttentionLayerCount={model.AttentionLayerCount}");

        // Deterministic short prompt (arbitrary in-vocab token ids — real BPE tokenization
        // is exercised elsewhere; this test only cares about numerical finiteness end-to-end
        // through every GDN / attention / PQ2_0 dense-FFN layer).
        int[] promptTokens = [1, 100, 2000, 30000, 5];
        int[] positions = [0, 1, 2, 3, 4];

        using var kvCache = model.CreateKvCache(maxSeqLen: 16);

        var prefillSw = System.Diagnostics.Stopwatch.StartNew();
        using (var prefillLogits = model.Forward(promptTokens, positions, deviceId: -1, kvCache))
        {
            prefillSw.Stop();
            AssertFiniteLastRow(prefillLogits, "prefill");
        }
        _out.WriteLine($"Prefill ({promptTokens.Length} tok): {prefillSw.Elapsed.TotalMilliseconds:F0}ms");

        int nextPos = promptTokens.Length;
        var decodeSw = System.Diagnostics.Stopwatch.StartNew();
        for (int step = 0; step < 4; step++)
        {
            int[] single = [7 + step * 13];
            int[] singlePos = [nextPos];
            using var logits = model.Forward(single, singlePos, deviceId: -1, kvCache);
            AssertFiniteLastRow(logits, $"decode step {step}");
            nextPos++;
        }
        decodeSw.Stop();
        _out.WriteLine($"Decode (4 steps): {decodeSw.Elapsed.TotalMilliseconds:F0}ms " +
                       $"({4000.0 / decodeSw.Elapsed.TotalMilliseconds:F2} tok/s)");

        // No-op unless DOTLLM_HYBRID_PROFILE=1 — prints accumulated per-category decode timing
        // (see CudaQwen3HybridDenseTransformerModel.ProfStart/ProfMark) to stderr.
        CudaQwen3HybridDenseTransformerModel.ProfileReportAndReset();
    }

    private static unsafe void AssertFiniteLastRow(DotLLM.Core.Tensors.ITensor logits, string label)
    {
        int rank = logits.Shape.Rank;
        int vocab = logits.Shape[rank - 1];
        long rows = 1;
        for (int d = 0; d < rank - 1; d++) rows *= logits.Shape[d];

        float* basePtr = (float*)logits.DataPointer + (rows - 1) * vocab;
        for (int i = 0; i < vocab; i++)
        {
            float v = basePtr[i];
            Assert.True(float.IsFinite(v), $"{label}: logits[{i}] = {v} is not finite.");
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
