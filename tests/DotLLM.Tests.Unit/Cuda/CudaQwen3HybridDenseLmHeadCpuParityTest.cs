using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Regression test for issue #162: the CUDA <c>Qwen3HybridDense</c> loader uploaded the LM head
/// (<c>output.weight</c>, and its tied-embedding fallback, <c>token_embd.weight</c>) via a
/// hand-rolled alloc+copy that skipped the load-time interleaved-&gt;split-layout PQ2_0 repack
/// every other PQ2_0 weight tensor gets via <c>UploadRawTensor</c>. Every PQ2_0 GEMV/dequant
/// kernel reachable from the LM head projection unconditionally assumes split layout, so the raw
/// interleaved bytes decoded as garbage — logits differed from the CPU F32 reference by roughly
/// 5 orders of magnitude (observed: real value ~-0.75, GPU value ~-79585 for one index), landing
/// outside FP16's finite range for a large fraction of the vocab and surfacing as prefill
/// <c>-Infinity</c> logits.
///
/// The existing <see cref="CudaQwen3HybridDenseRealGgufSmokeTest"/> only asserts finiteness —
/// insufficient on its own, since a garbage decode can produce a finite-but-wrong value just as
/// easily as an infinite one (confirmed during root-cause investigation: most of the corrupted
/// logits were finite, just wildly wrong). This test instead asserts the CUDA LM head logits are
/// numerically CLOSE to the CPU oracle for the same prompt, which the interleaved/split layout
/// mismatch could not have passed even when it happened not to overflow FP16.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public class CudaQwen3HybridDenseLmHeadCpuParityTest
{
    private const string ModelPathEnvVar = "DOTLLM_BONSAI_PQ2_0_GGUF";
    private const string FileName = "Ternary-Bonsai-27B-Q2_0.gguf";

    private readonly ITestOutputHelper _out;
    public CudaQwen3HybridDenseLmHeadCpuParityTest(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    [SkippableFact]
    public unsafe void Forward_RealBonsai27B_CudaLmHeadLogits_MatchCpuReferenceMagnitude()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? path = ResolveFixturePath();
        Skip.If(path is null,
            $"Bonsai PQ2_0 GGUF fixture not found. Set {ModelPathEnvVar}, or place {FileName} under "
            + "~/.dotllm/models/PrismML/Ternary-Bonsai-27B-GGUF/ or ~/.dotllm/test-cache/PrismML/Ternary-Bonsai-27B-GGUF/.");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        int[] promptTokens = [1, 100, 2000, 30000, 5];
        int[] positions = [0, 1, 2, 3, 4];

        // CPU oracle (per this codebase's convention: CPU is the correctness reference for
        // quantized-kernel numerics — see CLAUDE.md).
        float[] cpuLogits;
        using (var gguf = GgufFile.Open(path!))
        {
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var cpuModel = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(
                gguf, config, ThreadingConfig.SingleThreaded);
            using var cpuResult = cpuModel.Forward(promptTokens, positions, deviceId: -1);

            int rank = cpuResult.Shape.Rank;
            int vocab = cpuResult.Shape[rank - 1];
            long rows = 1;
            for (int d = 0; d < rank - 1; d++) rows *= cpuResult.Shape[d];
            float* basePtr = (float*)cpuResult.DataPointer + (rows - 1) * vocab;

            cpuLogits = new float[32];
            for (int i = 0; i < cpuLogits.Length; i++) cpuLogits[i] = basePtr[i];
        }

        // CUDA path under test.
        float[] gpuLogits;
        using (var gguf = GgufFile.Open(path!))
        {
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            Assert.Equal(Architecture.Qwen3HybridDense, config.Architecture);

            using var gpuModel = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
            using var kvCache = gpuModel.CreateKvCache(maxSeqLen: 16);
            using var gpuResult = gpuModel.Forward(promptTokens, positions, deviceId: -1, kvCache);

            int rank = gpuResult.Shape.Rank;
            int vocab = gpuResult.Shape[rank - 1];
            long rows = 1;
            for (int d = 0; d < rank - 1; d++) rows *= gpuResult.Shape[d];
            float* basePtr = (float*)gpuResult.DataPointer + (rows - 1) * vocab;

            gpuLogits = new float[32];
            for (int i = 0; i < gpuLogits.Length; i++) gpuLogits[i] = basePtr[i];
        }

        _out.WriteLine($"CPU logits[0..7] = [{string.Join(", ", cpuLogits[..8])}]");
        _out.WriteLine($"GPU logits[0..7] = [{string.Join(", ", gpuLogits[..8])}]");

        // The interleaved/split layout mismatch produced errors of ~10^4-10^5 in magnitude. A
        // generous absolute+relative tolerance (well beyond ordinary PQ2_0/FP16 quantization
        // noise, which this codebase's other GEMV correctness tests bound at <= 5e-2) is more
        // than enough to catch a regression of that mechanism while not being sensitive to
        // legitimate small numerical differences between the CPU F32 and CUDA F32-accumulate/
        // FP16-intermediate paths.
        for (int i = 0; i < cpuLogits.Length; i++)
        {
            float diff = Math.Abs(cpuLogits[i] - gpuLogits[i]);
            float tolerance = 2.0f + 0.25f * Math.Abs(cpuLogits[i]);
            Assert.True(diff <= tolerance,
                $"logits[{i}]: cpu={cpuLogits[i]}, gpu={gpuLogits[i]}, diff={diff} exceeds tolerance={tolerance} " +
                "(the CUDA LM head appears to be decoding garbage — check that output.weight/token_embd.weight " +
                "went through the PQ2_0 interleaved->split repack, see issue #162).");
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
