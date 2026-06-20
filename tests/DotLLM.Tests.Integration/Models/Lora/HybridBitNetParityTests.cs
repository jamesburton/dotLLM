using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using System.Numerics.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models.Lora;

/// <summary>
/// Hybrid CPU+GPU BASE-path parity vs full-CPU, on a BitNet b1.58 (I2_S) model (#83).
/// Verifies that <see cref="HybridTransformerModel.Forward"/> with NO adapter produces
/// logits numerically consistent with the full-CPU reference path for a BitNet (I2_S)
/// model. With a half-layer split (numGpuLayers = cfg.NumLayers / 2) both GPU-resident
/// and CPU-resident layers run in the same forward pass.
/// Correctness criteria: argmax(hybrid) == argmax(cpu) AND cosine(hybrid, cpu) &gt; 0.999.
/// </summary>
/// <remarks>
/// Plain [Fact] with early-return no-op when DOTLLM_BITNET_GGUF is unset/missing or when
/// CUDA is unavailable — no SkippableFact dependency needed. This is the BASE path (no
/// LoRA adapter): it pins the four hybrid+BitNet root causes (CPU decode I2_S guard,
/// GPU-phase FP32 residual, FP32 boundary transfer, CPU prefill I2_S Gemm/Gemv dispatch).
/// Kept in its own class to avoid cross-class GPU parallelism.
/// </remarks>
public sealed class HybridBitNetParityTests
{
    private readonly ITestOutputHelper _output;

    public HybridBitNetParityTests(ITestOutputHelper output) => _output = output;

    private static string? ModelPath =>
        Environment.GetEnvironmentVariable("DOTLLM_BITNET_GGUF");

    [Fact]
    public unsafe void HybridAndCpu_BitNetBaseLogits_AreNumericallyConsistent()
    {
        // ── Guard 1: model file must be present ──────────────────────────────────
        if (ModelPath is null || !File.Exists(ModelPath))
        {
            _output.WriteLine("SKIP: BitNet GGUF not available (set DOTLLM_BITNET_GGUF).");
            return;
        }

        // ── Guard 2: CUDA device must be available ────────────────────────────────
        if (!CudaDevice.IsAvailable())
        {
            _output.WriteLine("SKIP: No CUDA device available.");
            return;
        }

        // ── Setup ─────────────────────────────────────────────────────────────────
        using var gguf = GgufFile.Open(ModelPath!);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);

        int[] tok = [1, 2, 3];
        int[] pos = [0, 1, 2];

        // ── Hybrid model (system under test) — BASE path, no adapter ───────────────
        // Half-split: ensures BOTH GPU layers and CPU layers are exercised.
        int numGpuLayers = Math.Max(1, cfg.NumLayers / 2);
        using var hybridModel = HybridTransformerModel.LoadFromGguf(
            gguf, cfg, numGpuLayers, deviceId: 0, threading: new ThreadingConfig(0, 0));

        using ITensor hybridLogits = hybridModel.Forward(tok, pos, deviceId: 0, kvCache: null);

        int vocabSize = cfg.VocabSize;
        // hybridLogits shape is [1, vocabSize] (last-token D2H).
        float[] hybridVec = new float[vocabSize];
        fixed (float* dst = hybridVec)
        {
            float* src = (float*)hybridLogits.DataPointer;
            new ReadOnlySpan<float>(src, vocabSize).CopyTo(new Span<float>(dst, vocabSize));
        }
        int hybridArgmax = ArgMax(hybridVec);

        // ── Full-CPU forward (reference) ──────────────────────────────────────────
        using var cpuModel = TransformerModel.LoadFromGguf(gguf, cfg, new ThreadingConfig(0, 0));
        using ITensor cpuLogits = cpuModel.Forward(tok, pos, deviceId: -1, kvCache: null);

        // cpuLogits shape is [seqLen, vocabSize]; take the last-token row.
        long lastRowOffset = (long)(tok.Length - 1) * vocabSize;
        float[] cpuVec = new float[vocabSize];
        fixed (float* dst = cpuVec)
        {
            float* src = (float*)cpuLogits.DataPointer + lastRowOffset;
            new ReadOnlySpan<float>(src, vocabSize).CopyTo(new Span<float>(dst, vocabSize));
        }
        int cpuArgmax = ArgMax(cpuVec);

        // ── Metrics ───────────────────────────────────────────────────────────────
        double cosine = CosineSimilarity(hybridVec, cpuVec);

        _output.WriteLine($"GPU layers: {numGpuLayers}/{cfg.NumLayers}  CPU layers: {cfg.NumLayers - numGpuLayers}");
        _output.WriteLine($"Hybrid argmax: {hybridArgmax}  CPU argmax: {cpuArgmax}");
        _output.WriteLine($"Hybrid top logit: {hybridVec[hybridArgmax]:F4}  CPU top logit: {cpuVec[cpuArgmax]:F4}");
        _output.WriteLine($"Cosine(hybrid, cpu): {cosine:F6}");

        // ── Assertions (the gate) ─────────────────────────────────────────────────
        // 1. Argmax must match (hard correctness gate).
        Assert.True(hybridArgmax == cpuArgmax,
            $"Argmax mismatch: Hybrid={hybridArgmax} CPU={cpuArgmax}. " +
            $"Cosine={cosine:F6}. " +
            $"Hybrid top logit={hybridVec[hybridArgmax]:F4}  CPU top logit={cpuVec[cpuArgmax]:F4}. " +
            $"This indicates the hybrid+BitNet (I2_S) base path is broken " +
            $"(CPU decode I2_S guard, GPU-phase FP32 residual, FP32 boundary transfer, or CPU prefill I2_S dispatch).");

        // 2. Cosine similarity > 0.999 (FP16 GPU layers → FP16 tolerance).
        Assert.True(cosine > 0.999,
            $"Cosine similarity {cosine:F6} is below 0.999. " +
            $"Hybrid argmax={hybridArgmax}  CPU argmax={cpuArgmax}. " +
            $"This indicates significant numerical divergence between hybrid and CPU BitNet paths.");
    }

    // ── Helpers ───────────────────────────────────────────────────────────────────

    private static int ArgMax(float[] vec)
        => TensorPrimitives.IndexOfMax(new ReadOnlySpan<float>(vec));

    private static double CosineSimilarity(float[] a, float[] b)
    {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            normA += (double)a[i] * a[i];
            normB += (double)b[i] * b[i];
        }
        double denom = Math.Sqrt(normA) * Math.Sqrt(normB);
        return denom < 1e-12 ? 0.0 : dot / denom;
    }
}
