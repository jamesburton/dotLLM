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
/// Issue #1068: a short multi-row forward over quantized KV (an MTP verify is K+1 = 4 rows every round) runs the fused
/// dequant-in-load decode kernel once per row instead of dequantizing the whole K and V history into F16 scratch in every
/// attention layer. The per-row result must match the scratch path (same causal attention, different kernels, so to a
/// cosine floor), and must really dequantize nothing.
/// </summary>
/// <remarks>
/// Gated on the real Bonsai 2 checkpoint (<c>DOTLLM_BONSAI2_MTP_GGUF</c>): the synthetic model's KV row is 16 elements,
/// below the 32-element quantization block. Mutants: forcing <c>perRowKvq</c> false fails the counter assertions;
/// attending every row to the full <c>seqKv</c> (dropping the causal <c>+ r + 1</c>) breaks the cosine floor on rows 0..2.
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaQwen3HybridDensePerRowKvqVerifyTests
{
    private const int Prefix = 600;
    private const int Rows = 4;

    private readonly ITestOutputHelper _out;

    public CudaQwen3HybridDensePerRowKvqVerifyTests(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void VerifyRows_OverQ8Kv_MatchTheDequantScratchPath_WithoutDequantizing()
    {
        string? path = FindCheckpoint();
        Skip.If(path is null, "Bonsai 2 MTP checkpoint not found (set DOTLLM_BONSAI2_MTP_GGUF).");
        string? ptxDir = SkipUnlessCuda();

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!);
        model.SetKvStorageForTest(KvCacheDType.Q8_0, KvCacheDType.Q8_0);
        Skip.IfNot(model.FuseQuantKvDecodeForTest, "fused quantized-KV decode kernel not loaded on this device");
        using var kv = model.CreateKvCache(1024);

        int[] prefix = new int[Prefix];
        for (int i = 0; i < Prefix; i++) prefix[i] = 1 + (i * 7919) % 30000;
        using (model.Forward(prefix, Enumerable.Range(0, Prefix).ToArray(), deviceId: -1, kv, lastTokenLogitsOnly: true)) { }
        object? afterPrefix = model.CheckpointRecurrentState();

        int[] rows = [1042, 77, 9001, 314];
        int[] rowPositions = Enumerable.Range(Prefix, Rows).ToArray();

        int perRowBefore = model.DebugPerRowKvqAttentionCallsForTest;
        int dequantBefore = model.DebugKvDequantCallsForTest;
        float[][] perRow = RunRows(model, kv, rows, rowPositions, config.VocabSize);
        int attentionLayers = model.AttentionLayerCount;
        Assert.Equal(attentionLayers * Rows, model.DebugPerRowKvqAttentionCallsForTest - perRowBefore);
        Assert.Equal(0, model.DebugKvDequantCallsForTest - dequantBefore);

        kv.Rollback(Prefix);
        model.RestoreRecurrentState(afterPrefix);
        model.PerRowKvqForTest = false;
        dequantBefore = model.DebugKvDequantCallsForTest;
        float[][] scratch = RunRows(model, kv, rows, rowPositions, config.VocabSize);
        Assert.Equal(2 * attentionLayers, model.DebugKvDequantCallsForTest - dequantBefore);
        (afterPrefix as IDisposable)?.Dispose();

        for (int r = 0; r < Rows; r++)
        {
            double cos = Cosine(perRow[r], scratch[r]);
            _out.WriteLine($"row {r}: cosine(per-row, scratch) = {cos:F7}, argmax {ArgMax(perRow[r])} vs {ArgMax(scratch[r])}");
            Assert.True(cos > 0.9999, $"row {r}: cosine {cos}");
            Assert.Equal(ArgMax(scratch[r]), ArgMax(perRow[r]));
        }
    }

    private static unsafe float[][] RunRows(CudaQwen3HybridDenseTransformerModel model, CudaHybridKvCacheHandle kv,
                                             int[] tokens, int[] positions, int vocab)
    {
        using ITensor t = model.Forward(tokens, positions, deviceId: -1, kv, lastTokenLogitsOnly: false);
        var all = new ReadOnlySpan<float>((void*)t.DataPointer, tokens.Length * vocab);
        var result = new float[tokens.Length][];
        for (int r = 0; r < tokens.Length; r++) result[r] = all.Slice(r * vocab, vocab).ToArray();
        return result;
    }

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        return dot / Math.Sqrt(na * nb);
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    private static string? FindCheckpoint()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_BONSAI2_MTP_GGUF");
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;
        string repo = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface", "hub",
            "models--ProCreations--Ternary-Bonsai-2-27B-MTP", "snapshots");
        if (!Directory.Exists(repo)) return null;
        foreach (string snapshot in Directory.EnumerateDirectories(repo))
        {
            string[] hits = Directory.GetFiles(snapshot, "*MTP*.gguf");
            if (hits.Length > 0) return hits[0];
        }
        return null;
    }

    private static string? SkipUnlessCuda()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        bool driver = NativeLibrary.TryLoad(lib, out nint h);
        if (driver) NativeLibrary.Free(h);
        Skip.IfNot(driver && CudaDevice.IsAvailable(), "No CUDA GPU available");
        foreach (var dir in new[]
                 {
                     Path.Combine(AppContext.BaseDirectory, "ptx"),
                     Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
                 })
        {
            string full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        Skip.If(true, "PTX files not found");
        return null;
    }
}
