using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #956 Phase 2: quantized KV storage (Q8_0 / Q4_0 / mixed K8V4) in
/// <see cref="CudaQwen3HybridDenseTransformerModel"/>. Quantized rows are dequantized into a shared
/// F16 scratch per attention call and fed to the unchanged mma decode kernel, so the contract is:
/// decode still stages nothing in F32, and logits stay close to the F16-KV reference on real
/// Bonsai-2 weights (prefill 300 + 6 decode steps at the real 4-KV-head x 256-dim GQA shape).
/// Thresholds are deliberately loose floors, not accuracy claims: long-context quality is gated
/// separately (needle / perplexity, Phase 3).
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public class CudaQwen3HybridDenseQuantKvTest
{
    private const string ModelPathEnvVar = "DOTLLM_BONSAI_PQ2_0_GGUF";
    private const string FileName = "Ternary-Bonsai-2-27B-PQ2_0.gguf";
    private const int PrefillLen = 300;
    private const int DecodeSteps = 6;

    private readonly ITestOutputHelper _out;
    public CudaQwen3HybridDenseQuantKvTest(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    [SkippableFact]
    public void QuantizedKv_DecodeStaysCloseToF16_AndStagesNothing()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_ATTN_MMA_DECODE_GQA_SPLIT") == "0",
            "mma decode opted out via env; the staging contract only applies to the default path");
        string? path = ResolveFixturePath();
        Skip.If(path is null, $"Bonsai PQ2_0 GGUF fixture not found. Set {ModelPathEnvVar}.");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        var reference = Execute(path!, ptxDir!, KvCacheDType.F32, KvCacheDType.F32);

        var cases = new (string name, KvCacheDType k, KvCacheDType v, double minCos)[]
        {
            ("q8_0", KvCacheDType.Q8_0, KvCacheDType.Q8_0, 0.9999),
            ("k8v4", KvCacheDType.Q8_0, KvCacheDType.Q4_0, 0.998),
            ("q4_0", KvCacheDType.Q4_0, KvCacheDType.Q4_0, 0.99),
        };

        foreach (var c in cases)
        {
            var r = Execute(path!, ptxDir!, c.k, c.v);
            Assert.Equal(0, r.DecodeStaged);
            double worst = 1.0;
            for (int s = 0; s < DecodeSteps; s++)
                worst = Math.Min(worst, Cosine(reference.DecodeLogits[s], r.DecodeLogits[s]));
            _out.WriteLine($"{c.name}: worst decode cosine vs F16 = {worst:F6} (floor {c.minCos})");
            Assert.True(worst > c.minCos, $"{c.name}: worst cosine {worst} <= {c.minCos}");
        }
    }

    /// <summary>
    /// The fused dequant-in-load kernel must produce BIT-IDENTICAL logits to dequantize-into-F16 +
    /// the F16 kernel (same element expression, same accumulation), and must actually be the
    /// kernel that ran (counter), for every K/V format pair.
    /// </summary>
    [SkippableFact]
    public void FusedQuantKvDecode_IsBitIdenticalToDequantScratchPath()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_ATTN_MMA_DECODE_GQA_SPLIT") == "0",
            "mma decode opted out via env");
        string? path = ResolveFixturePath();
        Skip.If(path is null, $"Bonsai PQ2_0 GGUF fixture not found. Set {ModelPathEnvVar}.");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        var pairs = new (string name, KvCacheDType k, KvCacheDType v)[]
        {
            ("q8q8", KvCacheDType.Q8_0, KvCacheDType.Q8_0),
            ("q4q4", KvCacheDType.Q4_0, KvCacheDType.Q4_0),
            ("q8q4", KvCacheDType.Q8_0, KvCacheDType.Q4_0),
            ("q4q8", KvCacheDType.Q4_0, KvCacheDType.Q8_0),
        };
        foreach (var p in pairs)
        {
            var scratch = Execute(path!, ptxDir!, p.k, p.v, fused: false);
            var fusedRun = Execute(path!, ptxDir!, p.k, p.v, fused: true);
            Assert.Equal(0, scratch.FusedCalls);
            Assert.Equal(DecodeSteps * 16, fusedRun.FusedCalls);   // one per attention layer per decode step
            for (int s = 0; s < DecodeSteps; s++)
                for (int i = 0; i < scratch.DecodeLogits[s].Length; i++)
                    Assert.True(BitConverter.SingleToInt32Bits(scratch.DecodeLogits[s][i]) ==
                                BitConverter.SingleToInt32Bits(fusedRun.DecodeLogits[s][i]),
                        $"{p.name} step {s} logit {i}: fused {fusedRun.DecodeLogits[s][i]} != scratch {scratch.DecodeLogits[s][i]}");
            _out.WriteLine($"{p.name}: fused == scratch bit-identical over {DecodeSteps} steps");
        }
    }

    private sealed record RunResult(int DecodeStaged, int FusedCalls, float[][] DecodeLogits);

    private static RunResult Execute(string path, string ptxDir, KvCacheDType key, KvCacheDType value, bool fused = true)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        model.SetKvStorageForTest(key, value);
        model.FuseQuantKvDecodeForTest = fused;
        using var kv = model.CreateKvCache(maxSeqLen: 512);

        int[] tokens = new int[PrefillLen];
        int[] pos = new int[PrefillLen];
        for (int i = 0; i < PrefillLen; i++) { tokens[i] = 1 + (i * 7919) % 30000; pos[i] = i; }
        using (model.Forward(tokens, pos, deviceId: -1, kv, lastTokenLogitsOnly: true)) { }

        int c1 = model.DebugF32StagedAttentionCallsForTest;
        var logits = new float[DecodeSteps][];
        for (int s = 0; s < DecodeSteps; s++)
        {
            using ITensor t = model.Forward([(7 + s * 13) % 30000], [PrefillLen + s], deviceId: -1, kv);
            logits[s] = Copy(t, config.VocabSize);
        }
        return new RunResult(model.DebugF32StagedAttentionCallsForTest - c1,
                             model.DebugFusedKvqAttentionCallsForTest, logits);
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
