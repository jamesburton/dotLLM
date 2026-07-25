using System.Runtime.InteropServices;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Bit-exact regression test for issue #182: incremental KV F16-&gt;F32 read-staging in
/// <see cref="CudaQwen3HybridDenseTransformerModel"/>'s <c>ForwardFullAttnBody</c>. Before the
/// fix, every decode step reconverted the ENTIRE cached KV range from F16 to F32 from scratch
/// (an O(depth) cost paid every step, even though ordinary prefill/decode only ever appends
/// rows); the fix converts only the newly-appended row(s) each call, keeping previously-converted
/// values in place across calls (per attention-layer slot).
///
/// Since F16-&gt;F32 conversion is a pure per-element function of immutable source bytes,
/// converting a row now vs. later from the same F16 content must produce IDENTICAL F32 values --
/// so the fast (incremental) and reference (forced full-reconversion, mirroring pre-fix behavior
/// via <see cref="CudaQwen3HybridDenseTransformerModel.ForceFullKvReconvertForTest"/>) paths must
/// agree bit-for-bit at every step of a real prefill + untimed depth-extension + many-decode-step
/// run against the real Bonsai-27B weights, including across at least one F32 staging buffer
/// growth event triggered mid-run (growth resets that slot's valid-length bookkeeping -- see
/// <c>EnsureF32KvReadStaging</c> -- and this test's final depth of 152 is chosen to make at least
/// one such growth event very likely for any realistic per-layer kvElems size).
///
/// The two model instances are run SEQUENTIALLY, not concurrently: Bonsai's ~7.2GB PQ2_0 weight
/// set would not fit twice on a 12GB RTX 3060 (loading both at once risks the WDDM-silent-paging
/// hang this investigation has hit before -- see the prismml-bonsai-model project memory).
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public class CudaQwen3HybridDenseIncrementalKvDequantTest
{
    private const string ModelPathEnvVar = "DOTLLM_BONSAI_PQ2_0_GGUF";
    private const string FileName = "Ternary-Bonsai-27B-Q2_0.gguf";
    private const int SliceLen = 48;

    private readonly ITestOutputHelper _out;
    public CudaQwen3HybridDenseIncrementalKvDequantTest(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    [SkippableFact]
    public void IncrementalKvDequant_MatchesForcedFullReconversion_AcrossManyDecodeSteps()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? path = ResolveFixturePath();
        Skip.If(path is null,
            $"Bonsai PQ2_0 GGUF fixture not found. Set {ModelPathEnvVar}, or place {FileName} under "
            + "~/.dotllm/models/PrismML/Ternary-Bonsai-27B-GGUF/ or ~/.dotllm/test-cache/PrismML/Ternary-Bonsai-27B-GGUF/.");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        // Sequential, not concurrent -- see class doc. Reference (pre-fix-equivalent) run first,
        // then the fast (incremental) run; the two model instances never coexist on the GPU.
        _out.WriteLine("Running reference (forced full KV reconversion, pre-fix behavior)...");
        float[][] refLogits = RunSequenceCaptureLogits(path!, ptxDir!, forceFullReconvert: true);

        _out.WriteLine("Running fast (incremental KV reconversion, this fix)...");
        float[][] fastLogits = RunSequenceCaptureLogits(path!, ptxDir!, forceFullReconvert: false);

        Assert.Equal(refLogits.Length, fastLogits.Length);
        for (int step = 0; step < refLogits.Length; step++)
        {
            for (int i = 0; i < SliceLen; i++)
            {
                float a = refLogits[step][i];
                float b = fastLogits[step][i];
                Assert.True(BitConverter.SingleToInt32Bits(a) == BitConverter.SingleToInt32Bits(b),
                    $"step {step}, logits[{i}]: reference(full-reconvert)={a}, fast(incremental)={b} -- " +
                    "not bit-identical; the incremental KV F16->F32 staging path has diverged from " +
                    "the old full-reconversion behavior.");
            }
        }

        _out.WriteLine($"{refLogits.Length} steps compared, {SliceLen} logits each -- all bit-identical.");
    }

    /// <summary>
    /// Deterministic prefill (8 tokens) + untimed depth-extension (120 tokens, re-tiling the
    /// prompt, mirroring <c>BenchRunner</c>'s <c>--depth</c> handling) + 24 single-token decode
    /// steps. Final depth 152.
    /// </summary>
    private static (int[][] tokens, int[][] positions) BuildDeterministicSequence()
    {
        int[] promptTokens = [1, 100, 2000, 30000, 5, 777, 42, 9001];
        int[] promptPositions = [0, 1, 2, 3, 4, 5, 6, 7];

        const int depth = 120;
        int[] extraTokens = new int[depth];
        int[] extraPositions = new int[depth];
        for (int i = 0; i < depth; i++)
        {
            extraTokens[i] = promptTokens[i % promptTokens.Length];
            extraPositions[i] = promptTokens.Length + i;
        }

        const int decodeSteps = 24;
        var tokens = new int[2 + decodeSteps][];
        var positions = new int[2 + decodeSteps][];
        tokens[0] = promptTokens;
        positions[0] = promptPositions;
        tokens[1] = extraTokens;
        positions[1] = extraPositions;

        int nextPos = promptTokens.Length + depth;
        for (int step = 0; step < decodeSteps; step++)
        {
            tokens[2 + step] = [(7 + step * 13) % 40000];
            positions[2 + step] = [nextPos];
            nextPos++;
        }

        return (tokens, positions);
    }

    private static float[][] RunSequenceCaptureLogits(string path, string ptxDir, bool forceFullReconvert)
    {
        var (tokenSteps, positionSteps) = BuildDeterministicSequence();

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.Qwen3HybridDense, config.Architecture);

        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        model.ForceFullKvReconvertForTest = forceFullReconvert;

        using var kvCache = model.CreateKvCache(maxSeqLen: 256);

        var results = new float[tokenSteps.Length][];
        for (int step = 0; step < tokenSteps.Length; step++)
        {
            using var logits = model.Forward(tokenSteps[step], positionSteps[step], deviceId: -1, kvCache);
            results[step] = ExtractLastRowSlice(logits, SliceLen);
        }
        return results;
    }

    private static unsafe float[] ExtractLastRowSlice(ITensor logits, int sliceLen)
    {
        int rank = logits.Shape.Rank;
        int vocab = logits.Shape[rank - 1];
        long rows = 1;
        for (int d = 0; d < rank - 1; d++) rows *= logits.Shape[d];

        float* basePtr = (float*)logits.DataPointer + (rows - 1) * vocab;
        var slice = new float[sliceLen];
        for (int i = 0; i < sliceLen; i++) slice[i] = basePtr[i];
        return slice;
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
