using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using System.Numerics.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models.Lora;

/// <summary>
/// Validates that <see cref="LoraComposer.Compose"/> produces a composite adapter whose GPU
/// output matches the CPU reference path, for a two-adapter stack applied to SmolLM-135M (#92).
/// Both adapters are synthetic F32 adapters built via <see cref="SyntheticLoraAdapterFactory"/>.
/// They are composed with equal weights (1.0 each) before forwarding through the CUDA model.
/// Correctness criteria: argmax(gpu_stack) == argmax(cpu_stack) AND cosine &gt; 0.999.
/// Also asserts that cpu_stack ≠ cpu_base (the composed delta actually fires on CPU),
/// and that gpu_stack ≠ gpu_base (the composed delta actually fires on GPU).
/// </summary>
/// <remarks>
/// Plain [Fact] with early-return no-op when the SmolLM-135M Q8_0 GGUF is absent or when
/// CUDA is unavailable — no SkippableFact dependency needed. Safe to run in CI tonight without
/// a GPU; the test passes trivially via early return. Schedule it for the RTX 3060 run tomorrow.
/// Kept in its own class to avoid cross-class GPU parallelism.
/// </remarks>
public sealed class LoraStackCudaParityTests
{
    private readonly ITestOutputHelper _output;

    public LoraStackCudaParityTests(ITestOutputHelper output) => _output = output;

    private const string ModelPath =
        @"C:\Users\james\.dotllm\models\QuantFactory\SmolLM-135M-GGUF\SmolLM-135M.Q8_0.gguf";

    [SkippableFact]
    public unsafe void Composed_Stack_Cuda_Matches_Cpu()
    {
        // ── Guard 1: model file must be present ──────────────────────────────────
        Skip.If(!File.Exists(ModelPath),
            $"SmolLM-135M Q8_0 GGUF not found at {ModelPath}.");

        // ── Guard 2: CUDA device must be available ────────────────────────────────
        Skip.If(!CudaDevice.IsAvailable(),
            "No CUDA device available.");

        // ── Setup ─────────────────────────────────────────────────────────────────
        using var gguf = GgufFile.Open(ModelPath);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);

        int[] tok = [1, 2, 3];
        int[] pos = [0, 1, 2];

        // Two independent synthetic F32 adapters with different seeds.
        // SyntheticLoraAdapterFactory targets q_proj/v_proj on every layer.
        using var adapter1 = SyntheticLoraAdapterFactory.ForConfig(cfg, rank: 8, alpha: 16f, seed: 42);
        using var adapter2 = SyntheticLoraAdapterFactory.ForConfig(cfg, rank: 8, alpha: 16f, seed: 99);

        // Compose both adapters with equal weight 1.0 into a single composite adapter.
        // LoraComposer.Compose does NOT take ownership of adapter1/adapter2 — we dispose them above.
        using var composite = LoraComposer.Compose(
            [(adapter1, 1f), (adapter2, 1f)],
            cfg);

        int vocabSize = cfg.VocabSize;

        // ── Full-CPU forward (reference) ──────────────────────────────────────────
        using var cpuModel = TransformerModel.LoadFromGguf(gguf, cfg, new ThreadingConfig(0, 0));

        // CPU + composite (reference output)
        using ITensor cpuStackLogits = cpuModel.Forward(tok, pos, deviceId: -1, kvCache: null, adapter: composite);
        long lastRowOffset = (long)(tok.Length - 1) * vocabSize;
        float[] cpuVec = new float[vocabSize];
        fixed (float* dst = cpuVec)
        {
            float* src = (float*)cpuStackLogits.DataPointer + lastRowOffset;
            new ReadOnlySpan<float>(src, vocabSize).CopyTo(new Span<float>(dst, vocabSize));
        }
        int cpuArgmax = ArgMax(cpuVec);

        // CPU without adapter (sanity: composite delta must fire on CPU path)
        using ITensor cpuBaseLogits = cpuModel.Forward(tok, pos, deviceId: -1, kvCache: null, adapter: null);
        float[] cpuBaseVec = new float[vocabSize];
        fixed (float* dst = cpuBaseVec)
        {
            float* src = (float*)cpuBaseLogits.DataPointer + lastRowOffset;
            new ReadOnlySpan<float>(src, vocabSize).CopyTo(new Span<float>(dst, vocabSize));
        }
        bool adapterChangedCpu = VectorsAreDifferent(cpuVec, cpuBaseVec, threshold: 1e-4f);

        // ── CUDA forward (system under test) ──────────────────────────────────────
        using var gpuModel = CudaTransformerModel.LoadFromGguf(gguf, cfg, deviceId: 0);

        // GPU + composite (system under test)
        using ITensor gpuStackLogits = gpuModel.Forward(tok, pos, deviceId: 0, kvCache: null, adapter: composite);
        float[] gpuVec = new float[vocabSize];
        fixed (float* dst = gpuVec)
        {
            float* src = (float*)gpuStackLogits.DataPointer;
            new ReadOnlySpan<float>(src, vocabSize).CopyTo(new Span<float>(dst, vocabSize));
        }
        int gpuArgmax = ArgMax(gpuVec);

        // GPU without adapter (sanity: composite delta must fire on GPU path)
        using ITensor gpuBaseLogits = gpuModel.Forward(tok, pos, deviceId: 0, kvCache: null, adapter: null);
        float[] gpuBaseVec = new float[vocabSize];
        fixed (float* dst = gpuBaseVec)
        {
            float* src = (float*)gpuBaseLogits.DataPointer;
            new ReadOnlySpan<float>(src, vocabSize).CopyTo(new Span<float>(dst, vocabSize));
        }
        bool adapterChangedGpu = VectorsAreDifferent(gpuVec, gpuBaseVec, threshold: 1e-4f);

        // ── Metrics ───────────────────────────────────────────────────────────────
        double cosine = CosineSimilarity(cpuVec, gpuVec);

        _output.WriteLine($"Composite rank: {composite.Rank}  (2 × rank-8 adapters)");
        _output.WriteLine($"CPU+stack argmax: {cpuArgmax}  GPU+stack argmax: {gpuArgmax}");
        _output.WriteLine($"Cosine(cpu_stack, gpu_stack): {cosine:F6}");
        _output.WriteLine($"CPU+adapter differs from CPU-base: {adapterChangedCpu}");
        _output.WriteLine($"GPU+adapter differs from GPU-base: {adapterChangedGpu}");

        // ── Assertions ────────────────────────────────────────────────────────────

        // GATE A: composed delta must fire on the CPU path.
        Assert.True(adapterChangedCpu,
            "CPU+stack logits are identical to CPU-base logits. " +
            "LoraComposer.Compose produced a no-op composite (zero delta on CPU path).");

        // GATE B: composed delta must fire on the GPU path.
        Assert.True(adapterChangedGpu,
            "GPU+stack logits are identical to GPU-base logits. " +
            "The composed LoRA delta did not fire on the CUDA path.");

        // GATE C: argmax must match between CPU and GPU (hard correctness gate).
        Assert.True(cpuArgmax == gpuArgmax,
            $"Argmax mismatch: CPU={cpuArgmax} GPU={gpuArgmax}. " +
            $"Cosine={cosine:F6}. " +
            $"CPU top logit={cpuVec[cpuArgmax]:F4}  GPU top logit={gpuVec[gpuArgmax]:F4}. " +
            "This indicates a stacked-LoRA delta placement or rank-concatenation bug in the CUDA path.");

        // GATE D: cosine similarity > 0.999 (mirrors the sibling parity-test threshold;
        // FP16-vs-FP32 and rank-concat both introduce small rounding errors).
        Assert.True(cosine > 0.999,
            $"Cosine similarity {cosine:F6} is below 0.999. " +
            $"CPU argmax={cpuArgmax}  GPU argmax={gpuArgmax}. " +
            "Significant numerical divergence in the composed LoRA GPU path.");
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
