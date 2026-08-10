using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using System.Numerics.Tensors;
using Xunit.Abstractions;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Lora;

/// <summary>
/// CPU↔GPU parity gate for BitNet LoRA (M2).
/// Verifies that <see cref="CudaTransformerModel.Forward"/> with a synthetic LoRA adapter
/// produces logits that are numerically consistent with the CPU reference path.
/// Correctness criteria: argmax(cpu) == argmax(gpu) AND cosine(cpu, gpu) &gt; 0.999.
/// Also asserts that GPU+adapter ≠ GPU-without-adapter (delta actually fires on device).
/// </summary>
/// <remarks>
/// Plain [Fact] with early-return no-op when DOTLLM_BITNET_GGUF is unset/missing
/// or when CUDA is unavailable — no SkippableFact dependency needed.
/// Kept in its own class to avoid cross-class GPU parallelism.
/// </remarks>
public sealed class BitNetCudaLoraParityTests
{
    private readonly ITestOutputHelper _output;

    public BitNetCudaLoraParityTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// BitNet I2_S fixture, resolved via <see cref="KnownTestFixtures.BitNetI2S"/>:
    /// <c>$DOTLLM_BITNET_GGUF</c>, then the dotLLM test cache, then the HF hub cache (#308).
    /// </summary>
    private static FixtureLocation BitNetFixture => KnownTestFixtures.BitNetI2S;

    private static string? ModelPath => BitNetFixture.Path;

    [SkippableFact]
    public unsafe void CpuGpuLoraLogits_AreNumericallyConsistent()
    {
        // ── Guard 1: model file must be present ──────────────────────────────────
        Skip.If(!BitNetFixture.Found, BitNetFixture.SkipMessage(KnownTestFixtures.BitNetI2SDescription));

        // ── Guard 2: CUDA device must be available ────────────────────────────────
        Skip.If(!CudaDevice.IsAvailable(),
            "No CUDA device available.");

        // ── Setup ─────────────────────────────────────────────────────────────────
        using var gguf = GgufFile.Open(ModelPath!);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);

        int[] tok = [1, 2, 3];
        int[] pos = [0, 1, 2];

        // Single adapter instance shared by both CPU and GPU forward passes.
        // SyntheticLoraAdapterFactory targets q_proj/v_proj on every layer with
        // small deterministic deltas (±0.01, seed=42).
        using var adapter = SyntheticLoraAdapterFactory.ForConfig(cfg, rank: 8, alpha: 16f, seed: 42);

        // ── CPU forward (reference) ───────────────────────────────────────────────
        using var cpuModel = TransformerModel.LoadFromGguf(gguf, cfg, new ThreadingConfig(0, 0));
        using ITensor cpuLogits = cpuModel.Forward(tok, pos, deviceId: -1, kvCache: null, adapter: adapter);

        int vocabSize = cfg.VocabSize;
        // cpuLogits shape is [seqLen, vocabSize]; take the last-token row.
        long lastRowOffset = (long)(tok.Length - 1) * vocabSize;

        float[] cpuVec = new float[vocabSize];
        fixed (float* dst = cpuVec)
        {
            float* src = (float*)cpuLogits.DataPointer + lastRowOffset;
            new ReadOnlySpan<float>(src, vocabSize).CopyTo(new Span<float>(dst, vocabSize));
        }
        int cpuArgmax = ArgMax(cpuVec);

        // ── GPU forward (system under test) ───────────────────────────────────────
        using var gpuModel = CudaTransformerModel.LoadFromGguf(gguf, cfg, deviceId: 0);

        // GPU+adapter
        using ITensor gpuLoraLogits = gpuModel.Forward(tok, pos, deviceId: 0, kvCache: null, adapter: adapter);
        // gpuLogits shape is [1, vocabSize] (last-token only; see CudaTransformerModel.Forward D2H).
        float[] gpuVec = new float[vocabSize];
        fixed (float* dst = gpuVec)
        {
            float* src = (float*)gpuLoraLogits.DataPointer;
            new ReadOnlySpan<float>(src, vocabSize).CopyTo(new Span<float>(dst, vocabSize));
        }
        int gpuArgmax = ArgMax(gpuVec);

        // GPU without adapter (sanity: delta must actually fire)
        using ITensor gpuBaseLogits = gpuModel.Forward(tok, pos, deviceId: 0, kvCache: null, adapter: null);
        float[] gpuBaseVec = new float[vocabSize];
        fixed (float* dst = gpuBaseVec)
        {
            float* src = (float*)gpuBaseLogits.DataPointer;
            new ReadOnlySpan<float>(src, vocabSize).CopyTo(new Span<float>(dst, vocabSize));
        }

        // ── Metrics ───────────────────────────────────────────────────────────────
        double cosine = CosineSimilarity(cpuVec, gpuVec);
        bool adapterChangedGpu = VectorsAreDifferent(gpuVec, gpuBaseVec, threshold: 1e-4f);

        _output.WriteLine($"CPU argmax: {cpuArgmax}  GPU argmax: {gpuArgmax}");
        _output.WriteLine($"Cosine(cpu_lora, gpu_lora): {cosine:F6}");
        _output.WriteLine($"GPU+adapter differs from GPU-base: {adapterChangedGpu}");

        // ── Assertions (the gate) ─────────────────────────────────────────────────
        // 1. Argmax must match (hard correctness gate).
        Assert.True(cpuArgmax == gpuArgmax,
            $"Argmax mismatch: CPU={cpuArgmax} GPU={gpuArgmax}. " +
            $"Cosine={cosine:F6}. " +
            $"CPU top logit={cpuVec[cpuArgmax]:F4}  GPU top logit={gpuVec[gpuArgmax]:F4}. " +
            $"This indicates a LoRA delta placement or layout bug on the GPU path.");

        // 2. Cosine similarity > 0.999 (FP16-vs-FP32 magnitude guard).
        Assert.True(cosine > 0.999,
            $"Cosine similarity {cosine:F6} is below 0.999. " +
            $"CPU argmax={cpuArgmax}  GPU argmax={gpuArgmax}. " +
            $"This indicates significant numerical divergence between CPU and GPU LoRA paths.");

        // 3. Sanity: GPU+adapter must differ from GPU-base (delta actually fired).
        Assert.True(adapterChangedGpu,
            "GPU+adapter logits are identical to GPU-base logits. " +
            "The LoRA delta did not fire on the GPU path (no-op bug).");
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
