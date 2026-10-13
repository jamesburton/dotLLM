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
/// Issue #956: the F32 K/V read-staging in <see cref="CudaQwen3HybridDenseTransformerModel"/> is
/// now ONE shared pair, filled only for the F32-consuming attention tiers. Contract under test:
/// <list type="bullet">
/// <item>prefill (S&gt;1) goes through <c>attention_f32</c> and stages once per attention layer;</item>
/// <item>default mma decode at seqKv &gt;= <see cref="CudaKernels.AttentionGqaSplitMinSeqKv"/> reads the
/// F16 cache directly and stages NOTHING (the counter stays flat);</item>
/// <item>that mma decode agrees with the exact F32 tier (forced by raising the min-seqKv gate
/// in-process) on real Bonsai-2 weights.</item>
/// </list>
/// Model instances run sequentially (a second 7+ GB weight set does not fit a 12 GB card).
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public class CudaQwen3HybridDenseKvStagingTest
{
    private const string ModelPathEnvVar = "DOTLLM_BONSAI_PQ2_0_GGUF";
    private const string FileName = "Ternary-Bonsai-2-27B-PQ2_0.gguf";
    private const int PrefillLen = 300;      // > AttentionGqaSplitMinSeqKv (256) so decode is mma-eligible
    private const int DecodeSteps = 6;

    private readonly ITestOutputHelper _out;
    public CudaQwen3HybridDenseKvStagingTest(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    private sealed record RunResult(int PrefillStaged, int DecodeStaged, int Layers, float[][] DecodeLogits);

    [SkippableFact]
    public void MmaDecode_StagesNothing_AndMatchesF32Tier()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_ATTN_MMA_DECODE_GQA_SPLIT") == "0",
            "mma decode opted out via env; the contract only applies to the default path");
        string? path = ResolveFixturePath();
        Skip.If(path is null, $"Bonsai PQ2_0 GGUF fixture not found. Set {ModelPathEnvVar}.");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        int savedMin = CudaKernels.AttentionGqaSplitMinSeqKv;
        RunResult mma, f32;
        try
        {
            mma = Execute(path!, ptxDir!);
            CudaKernels.AttentionGqaSplitMinSeqKv = int.MaxValue;   // force plain attention_f32 (staged)
            f32 = Execute(path!, ptxDir!);
        }
        finally { CudaKernels.AttentionGqaSplitMinSeqKv = savedMin; }

        _out.WriteLine($"mma: prefill staged {mma.PrefillStaged}, decode staged {mma.DecodeStaged}; " +
                       $"f32: prefill staged {f32.PrefillStaged}, decode staged {f32.DecodeStaged}; layers {mma.Layers}");

        Assert.Equal(mma.Layers, mma.PrefillStaged);                // prefill stages once per attention layer
        Assert.Equal(0, mma.DecodeStaged);                          // mma decode reads the F16 cache directly
        Assert.Equal(f32.Layers * DecodeSteps, f32.DecodeStaged);   // control: the counter really counts

        for (int s = 0; s < DecodeSteps; s++)
        {
            double cos = Cosine(mma.DecodeLogits[s], f32.DecodeLogits[s]);
            _out.WriteLine($"step {s}: cosine {cos:F6}");
            Assert.True(cos > 0.999, $"step {s}: mma vs F32-tier logits cosine {cos}");
        }
    }

    private static RunResult Execute(string path, string ptxDir)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.Qwen3HybridDense, config.Architecture);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        using var kv = model.CreateKvCache(maxSeqLen: 512);

        int[] tokens = new int[PrefillLen];
        int[] pos = new int[PrefillLen];
        for (int i = 0; i < PrefillLen; i++) { tokens[i] = 1 + (i * 7919) % 30000; pos[i] = i; }

        int c0 = model.DebugF32StagedAttentionCallsForTest;
        using (model.Forward(tokens, pos, deviceId: -1, kv, lastTokenLogitsOnly: true)) { }
        int prefillStaged = model.DebugF32StagedAttentionCallsForTest - c0;

        int c1 = model.DebugF32StagedAttentionCallsForTest;
        var logits = new float[DecodeSteps][];
        for (int s = 0; s < DecodeSteps; s++)
        {
            using ITensor t = model.Forward([(7 + s * 13) % 30000], [PrefillLen + s], deviceId: -1, kv);
            logits[s] = Copy(t, config.VocabSize);
        }
        int decodeStaged = model.DebugF32StagedAttentionCallsForTest - c1;
        return new RunResult(prefillStaged, decodeStaged, model.AttentionLayerCount, logits);
    }

    private static unsafe float[] Copy(ITensor t, int count)
        => new ReadOnlySpan<float>((void*)t.DataPointer, count).ToArray();

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        return dot / Math.Sqrt(na * nb);
    }

    private static string? ResolveFixturePath()
    {
        string? envPath = Environment.GetEnvironmentVariable(ModelPathEnvVar);
        if (!string.IsNullOrWhiteSpace(envPath) && File.Exists(envPath))
            return envPath;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string[] candidates =
        [
            Path.Combine(home, ".dotllm", "models", "prism-ml", "Ternary-Bonsai-2-27B-gguf", FileName),
            Path.Combine(home, ".dotllm", "test-cache", "prism-ml", "Ternary-Bonsai-2-27B-gguf", FileName),
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
