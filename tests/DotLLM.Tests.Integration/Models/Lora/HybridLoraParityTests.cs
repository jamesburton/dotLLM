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
/// Hybrid CPU+GPU LoRA parity vs full-CPU, on SmolLM-135M (#82).
/// Verifies that <see cref="HybridTransformerModel.Forward"/> with a synthetic LoRA adapter
/// produces logits that are numerically consistent with the full-CPU reference path.
/// With a half-layer split (numGpuLayers = cfg.NumLayers / 2) both GPU-resident and
/// CPU-resident layers are exercised in the same forward pass.
/// Correctness criteria: argmax(hybrid) == argmax(cpu) AND cosine(hybrid, cpu) &gt; 0.999.
/// Also asserts that hybrid+adapter ≠ hybrid-without-adapter (delta actually fires).
/// </summary>
/// <remarks>
/// Plain [Fact] with early-return no-op when the SmolLM-135M Q8_0 GGUF is absent
/// or when CUDA is unavailable — no SkippableFact dependency needed.
/// Model path is hardcoded to the cached location; test skips gracefully if not present.
/// Kept in its own class to avoid cross-class GPU parallelism.
/// </remarks>
public sealed class HybridLoraParityTests
{
    private readonly ITestOutputHelper _output;

    public HybridLoraParityTests(ITestOutputHelper output) => _output = output;

    private static string ModelPath =>
        @"C:/Users/james/.dotllm/models/QuantFactory/SmolLM-135M-GGUF/SmolLM-135M.Q8_0.gguf";

    [SkippableFact]
    public unsafe void HybridAndCpu_LoraLogits_AreNumericallyConsistent()
    {
        // ── Guard 1: model file must be present ──────────────────────────────────
        Skip.If(!File.Exists(ModelPath),
            "SmolLM-135M Q8_0 GGUF not found.");

        // ── Guard 2: CUDA device must be available ────────────────────────────────
        Skip.If(!CudaDevice.IsAvailable(),
            "No CUDA device available.");

        // ── Setup ─────────────────────────────────────────────────────────────────
        using var gguf = GgufFile.Open(ModelPath);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);

        int[] tok = [1, 2, 3];
        int[] pos = [0, 1, 2];

        // Single adapter instance shared by both hybrid and CPU forward passes.
        // SyntheticLoraAdapterFactory targets q_proj/v_proj on every layer with
        // small deterministic deltas (±0.01, seed=42).
        using var adapter = SyntheticLoraAdapterFactory.ForConfig(cfg, rank: 8, alpha: 16f, seed: 42);

        // ── Hybrid model (system under test) ──────────────────────────────────────
        // Half-split: ensures BOTH GPU layers and CPU layers are exercised.
        int numGpuLayers = Math.Max(1, cfg.NumLayers / 2);
        using var hybridModel = HybridTransformerModel.LoadFromGguf(
            gguf, cfg, numGpuLayers, deviceId: 0, threading: new ThreadingConfig(0, 0));

        // Hybrid + adapter (system under test)
        using ITensor hybridLoraLogits = hybridModel.Forward(tok, pos, deviceId: 0, kvCache: null, adapter: adapter);

        int vocabSize = cfg.VocabSize;
        // hybridLoraLogits shape mirrors CudaTransformerModel: [1, vocabSize] (last-token D2H).
        float[] hybridVec = new float[vocabSize];
        fixed (float* dst = hybridVec)
        {
            float* src = (float*)hybridLoraLogits.DataPointer;
            new ReadOnlySpan<float>(src, vocabSize).CopyTo(new Span<float>(dst, vocabSize));
        }
        int hybridArgmax = ArgMax(hybridVec);

        // Hybrid without adapter (sanity: delta must actually fire)
        using ITensor hybridBaseLogits = hybridModel.Forward(tok, pos, deviceId: 0, kvCache: null, adapter: null);
        float[] hybridBaseVec = new float[vocabSize];
        fixed (float* dst = hybridBaseVec)
        {
            float* src = (float*)hybridBaseLogits.DataPointer;
            new ReadOnlySpan<float>(src, vocabSize).CopyTo(new Span<float>(dst, vocabSize));
        }
        int hybridBaseArgmax = ArgMax(hybridBaseVec);

        // ── Full-CPU forward (reference) ──────────────────────────────────────────
        using var cpuModel = TransformerModel.LoadFromGguf(gguf, cfg, new ThreadingConfig(0, 0));
        using ITensor cpuLogits = cpuModel.Forward(tok, pos, deviceId: -1, kvCache: null, adapter: adapter);

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
        bool adapterChangedHybrid = VectorsAreDifferent(hybridVec, hybridBaseVec, threshold: 1e-4f);

        // Base-vs-base sanity: hybrid-base should match cpu-base for SmolLM (non-BitNet model)
        using ITensor cpuBaseLogitsRaw = cpuModel.Forward(tok, pos, deviceId: -1, kvCache: null, adapter: null);
        float[] cpuBaseVec = new float[vocabSize];
        fixed (float* dst = cpuBaseVec)
        {
            float* src = (float*)cpuBaseLogitsRaw.DataPointer + lastRowOffset;
            new ReadOnlySpan<float>(src, vocabSize).CopyTo(new Span<float>(dst, vocabSize));
        }
        int cpuBaseArgmax = ArgMax(cpuBaseVec);
        double cosineBaseVsBase = CosineSimilarity(hybridBaseVec, cpuBaseVec);

        _output.WriteLine($"GPU layers: {numGpuLayers}/{cfg.NumLayers}  CPU layers: {cfg.NumLayers - numGpuLayers}");
        _output.WriteLine($"Hybrid-adapted argmax: {hybridArgmax}  CPU-adapted argmax: {cpuArgmax}");
        _output.WriteLine($"Cosine(hybrid_lora, cpu_lora): {cosine:F6}");
        _output.WriteLine($"Hybrid+adapter differs from hybrid-base: {adapterChangedHybrid}");
        _output.WriteLine($"[Diagnostic] Hybrid-base argmax: {hybridBaseArgmax}  CPU-base argmax: {cpuBaseArgmax}");
        _output.WriteLine($"[Diagnostic] Cosine(hybrid_base, cpu_base): {cosineBaseVsBase:F6}");

        // ── Assertions (the gate) ─────────────────────────────────────────────────

        // Diagnostic: base-vs-base — helps isolate pre-existing hybrid divergence.
        // Not the primary gate, but logged so failures can be attributed correctly.
        _output.WriteLine(cosineBaseVsBase > 0.999
            ? $"[Diagnostic] Base parity OK (cosine={cosineBaseVsBase:F6} > 0.999)"
            : $"[Diagnostic] Base parity WARN: hybrid-base diverges from cpu-base (cosine={cosineBaseVsBase:F6}). " +
              $"Unexpected for SmolLM-135M — investigate hybrid forward-pass before attributing to LoRA wiring.");

        // GATE A: adapter delta must actually fire (hybrid+adapter ≠ hybrid-base).
        Assert.True(adapterChangedHybrid,
            "Hybrid+adapter logits are identical to hybrid-base logits. " +
            "The LoRA delta did not fire on the hybrid path (no-op bug in GPU or CPU phase).");

        // GATE B1: argmax must match (hard correctness gate).
        Assert.True(hybridArgmax == cpuArgmax,
            $"Argmax mismatch: Hybrid={hybridArgmax} CPU={cpuArgmax}. " +
            $"Cosine={cosine:F6}. " +
            $"Hybrid top logit={hybridVec[hybridArgmax]:F4}  CPU top logit={cpuVec[cpuArgmax]:F4}. " +
            $"[Diagnostic] base cosine={cosineBaseVsBase:F6}. " +
            $"This indicates a LoRA delta placement or layout bug in the hybrid path.");

        // GATE B2: cosine similarity > 0.999 (FP16-vs-FP32 magnitude guard).
        Assert.True(cosine > 0.999,
            $"Cosine similarity {cosine:F6} is below 0.999. " +
            $"Hybrid argmax={hybridArgmax}  CPU argmax={cpuArgmax}. " +
            $"[Diagnostic] base cosine={cosineBaseVsBase:F6}. " +
            $"This indicates significant numerical divergence between hybrid and CPU LoRA paths.");
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

    private static bool VectorsAreDifferent(float[] a, float[] b, float threshold)
    {
        for (int i = 0; i < a.Length; i++)
            if (MathF.Abs(a[i] - b[i]) > threshold)
                return true;
        return false;
    }
}
