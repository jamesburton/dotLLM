using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Regression test for issue #185, Finding 1: the <c>lastTokenLogitsOnly</c> hint on
/// <see cref="CudaQwen3HybridDenseTransformerModel"/>'s <c>Forward</c> overload must (a) return a
/// <c>[1, vocabSize]</c> tensor instead of <c>[seqLen, vocabSize]</c>, and (b) that one row must
/// closely match (within the tolerance below -- NOT bit-identical, see remark) the last row a full
/// (non-hinted) call over the same tokens/positions would produce.
///
/// This matters because the old code always computed a full <c>[seqLen, vocabSize]</c> Logits
/// tensor even when a caller (e.g. <c>BenchRunner</c>'s untimed prefill / <c>--depth</c> context
/// extension) only ever read the last row -- at this model's 248,320-token vocabulary, that
/// scratch buffer alone was ~970 MiB at seqLen=1024, the dominant term in the VRAM-ceiling hang
/// `dotllm bench --depth` hit beyond ~650-768 on a 12GB card. lastTokenLogitsOnly lets an opted-in
/// caller skip that waste; this test guards against a future regression grossly corrupting the one
/// row that IS still computed.
/// </summary>
/// <remarks>
/// NOT bit-exact by design: lastTokenLogitsOnly's seqLen=1 output routes the lm_head GEMM through
/// <c>Gemm</c>'s dedicated GEMV fast path (an F32-native kernel -- see <c>pq2_0_gemv.cu</c>'s
/// "F32-native activations" section), while the seqLen&gt;1 output routes through the general
/// path (dequant-to-F16 + a cuBLAS F16 GEMM). These are two independently-implemented numerical
/// kernels for the same logical projection -- exactly the class of benign kernel-routing
/// floating-point drift this project already documented and tolerated in the
/// <c>CudaGraphCaptureEquivalenceTest</c> fix (issues #174/#175), not a correctness bug. This
/// exact same drift already exists, unrelated to this fix, between any ordinary decode step
/// (always seqLen=1, always the GEMV path) and any prefill call (seqLen&gt;1, the general path)
/// for a shared position -- lastTokenLogitsOnly just lets --depth's context-extension call also
/// take the GEMV path decode already uses, instead of always taking the general path.
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public class CudaQwen3HybridDenseLastTokenLogitsOnlyTest
{
    private const string ModelPathEnvVar = "DOTLLM_BONSAI_PQ2_0_GGUF";
    private const string FileName = "Ternary-Bonsai-27B-Q2_0.gguf";
    private const int SliceLen = 64;

    private readonly ITestOutputHelper _out;
    public CudaQwen3HybridDenseLastTokenLogitsOnlyTest(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    [SkippableFact]
    public void LastTokenLogitsOnly_MatchesLastRowOfFullCall_AndReturnsSingleRowShape()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? path = ResolveFixturePath();
        Skip.If(path is null,
            $"Bonsai PQ2_0 GGUF fixture not found. Set {ModelPathEnvVar}, or place {FileName} under "
            + "~/.dotllm/models/PrismML/Ternary-Bonsai-27B-GGUF/ or ~/.dotllm/test-cache/PrismML/Ternary-Bonsai-27B-GGUF/.");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        var config = GgufModelConfigExtractor.Extract(GgufFile.Open(path!).Metadata);
        Assert.Equal(Architecture.Qwen3HybridDense, config.Architecture);

        // A multi-token, multi-position call (mirrors BenchRunner's --depth context extension) --
        // large enough that a full [seqLen, vocab] tensor would be sizeable, small enough to keep
        // the test fast.
        const int seqLen = 40;
        int[] tokens = new int[seqLen];
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++)
        {
            tokens[i] = (11 + i * 37) % 40000;
            positions[i] = i;
        }

        // Two SEPARATE model instances, run SEQUENTIALLY (not concurrently -- see class doc on
        // the #182 sibling test for why: Bonsai's ~7.2GB weight set would not fit twice on a 12GB
        // RTX 3060). This model has GatedDeltaNet (recurrent-state) layers whose state is owned by
        // the model instance itself, not the IKvCache -- reusing ONE model instance for both calls
        // would carry GDN state over from the first call into the second regardless of which
        // IKvCache is passed, making the two calls' outputs incomparable. Fresh model instances
        // give each call its own zero-initialized recurrent state, isolating the comparison to
        // exactly what lastTokenLogitsOnly changes: the LM-head row count.
        _out.WriteLine("Running full (lastTokenLogitsOnly: false) on a fresh model instance...");
        float[] fullLastRow;
        using (var gguf1 = GgufFile.Open(path!))
        using (var model1 = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf1, config, deviceId: 0, ptxDir!))
        using (var cacheFull = model1.CreateKvCache(maxSeqLen: 256))
        using (var fullLogits = model1.Forward(tokens, positions, deviceId: -1, cacheFull, lastTokenLogitsOnly: false))
        {
            Assert.Equal(seqLen, fullLogits.Shape[0]);
            fullLastRow = ExtractRow(fullLogits, seqLen - 1, SliceLen);
        }

        _out.WriteLine("Running lastTokenLogitsOnly: true on a second fresh model instance...");
        float[] hintedRow;
        using (var gguf2 = GgufFile.Open(path!))
        using (var model2 = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf2, config, deviceId: 0, ptxDir!))
        using (var cacheHinted = model2.CreateKvCache(maxSeqLen: 256))
        using (var hintedLogits = model2.Forward(tokens, positions, deviceId: -1, cacheHinted, lastTokenLogitsOnly: true))
        {
            Assert.Equal(1, hintedLogits.Shape[0]);
            hintedRow = ExtractRow(hintedLogits, 0, SliceLen);
        }

        // AbsTol(1e-4)+RelTol(1e-3) matches this codebase's standard CUDA fixture-parity bar
        // (e.g. CudaQwen3MoeHybridParityTests) -- appropriate here because the two rows come from
        // genuinely different kernels (GEMV fast path vs. dequant+cuBLAS-F16 general path, see the
        // class remark above), not because either computation is imprecise on its own.
        const float AbsTol = 1e-4f;
        const float RelTol = 1e-3f;
        for (int i = 0; i < SliceLen; i++)
        {
            float a = fullLastRow[i];
            float b = hintedRow[i];
            float diff = MathF.Abs(a - b);
            float tol = AbsTol + RelTol * MathF.Abs(a);
            Assert.True(diff <= tol,
                $"logits[{i}]: full-call last row={a}, lastTokenLogitsOnly row={b}, diff={diff} " +
                $"exceeds tolerance {tol} -- shrinking the LM-head to one row changed the computed " +
                "value by more than the expected GEMV-vs-general-path kernel drift.");
        }

        _out.WriteLine($"{SliceLen} logits compared -- full-call last row vs. lastTokenLogitsOnly row: within tolerance.");
    }

    private static unsafe float[] ExtractRow(DotLLM.Core.Tensors.ITensor logits, int row, int sliceLen)
    {
        int vocab = logits.Shape[logits.Shape.Rank - 1];
        float* basePtr = (float*)logits.DataPointer + (long)row * vocab;
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
