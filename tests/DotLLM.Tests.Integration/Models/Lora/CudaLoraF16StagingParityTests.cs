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
/// Regression gate for #89: GPU LoRA staging of an <b>F16</b> adapter.
/// Before the fix, <see cref="CudaLoraWeights.Stage"/> assumed F32 host buffers
/// and over-read F16 buffers (4 bytes/elem from a 2-byte/elem buffer) → access
/// violation. This proves an F16 adapter now stages without an AV and matches a
/// CPU run of the SAME F16 adapter.
/// </summary>
/// <remarks>
/// Plain [Fact], env-gated on a cached GGUF + CUDA availability; no-op when
/// absent. Mirrors <see cref="BitNetCudaLoraParityTests"/>.
/// </remarks>
public sealed class CudaLoraF16StagingParityTests
{
    private readonly ITestOutputHelper _output;

    public CudaLoraF16StagingParityTests(ITestOutputHelper output) => _output = output;

    private const string ModelPath =
        @"C:\Users\james\.dotllm\models\QuantFactory\SmolLM-135M-GGUF\SmolLM-135M.Q8_0.gguf";

    [Fact]
    public unsafe void F16Adapter_StagesOnGpu_AndMatchesCpu()
    {
        if (!File.Exists(ModelPath))
        {
            _output.WriteLine($"SKIP: SmolLM-135M GGUF not found at {ModelPath}.");
            return;
        }

        if (!CudaDevice.IsAvailable())
        {
            _output.WriteLine("SKIP: No CUDA device available.");
            return;
        }

        using var gguf = GgufFile.Open(ModelPath);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);

        int[] tok = [1, 2, 3];
        int[] pos = [0, 1, 2];

        // F16 adapter — host buffers are 2 bytes/elem. This is the buffer that
        // crashed the GPU path before the #89 fix.
        using var adapter = SyntheticLoraAdapterFactory.ForConfigF16(cfg, rank: 8, alpha: 16f, seed: 42);

        // ── CPU forward (reference; handles F16 via shared LoraProjection.Apply) ──
        using var cpuModel = TransformerModel.LoadFromGguf(gguf, cfg, new ThreadingConfig(0, 0));
        using ITensor cpuLogits = cpuModel.Forward(tok, pos, deviceId: -1, kvCache: null, adapter: adapter);

        int vocabSize = cfg.VocabSize;
        long lastRowOffset = (long)(tok.Length - 1) * vocabSize;
        float[] cpuVec = new float[vocabSize];
        fixed (float* dst = cpuVec)
        {
            float* src = (float*)cpuLogits.DataPointer + lastRowOffset;
            new ReadOnlySpan<float>(src, vocabSize).CopyTo(new Span<float>(dst, vocabSize));
        }
        int cpuArgmax = ArgMax(cpuVec);

        // ── GPU forward (system under test; calls CudaLoraWeights.Stage) ──────────
        using var gpuModel = CudaTransformerModel.LoadFromGguf(gguf, cfg, deviceId: 0);

        // (a) must not throw (previously AV'd inside Stage).
        using ITensor gpuLoraLogits = gpuModel.Forward(tok, pos, deviceId: 0, kvCache: null, adapter: adapter);
        float[] gpuVec = new float[vocabSize];
        fixed (float* dst = gpuVec)
        {
            float* src = (float*)gpuLoraLogits.DataPointer;
            new ReadOnlySpan<float>(src, vocabSize).CopyTo(new Span<float>(dst, vocabSize));
        }
        int gpuArgmax = ArgMax(gpuVec);

        using ITensor gpuBaseLogits = gpuModel.Forward(tok, pos, deviceId: 0, kvCache: null, adapter: null);
        float[] gpuBaseVec = new float[vocabSize];
        fixed (float* dst = gpuBaseVec)
        {
            float* src = (float*)gpuBaseLogits.DataPointer;
            new ReadOnlySpan<float>(src, vocabSize).CopyTo(new Span<float>(dst, vocabSize));
        }

        double cosine = CosineSimilarity(cpuVec, gpuVec);
        bool adapterChangedGpu = VectorsAreDifferent(gpuVec, gpuBaseVec, threshold: 1e-4f);

        _output.WriteLine($"CPU argmax: {cpuArgmax}  GPU argmax: {gpuArgmax}");
        _output.WriteLine($"Cosine(cpu_lora, gpu_lora): {cosine:F6}");
        _output.WriteLine($"GPU+adapter differs from GPU-base: {adapterChangedGpu}");

        // (b) delta fired on GPU.
        Assert.True(adapterChangedGpu,
            "GPU+F16-adapter logits are identical to GPU-base logits — the LoRA delta did not fire.");

        // (c) GPU matches CPU run of the same F16 adapter.
        Assert.True(cpuArgmax == gpuArgmax,
            $"Argmax mismatch: CPU={cpuArgmax} GPU={gpuArgmax}, cosine={cosine:F6}.");
        Assert.True(cosine > 0.999,
            $"Cosine similarity {cosine:F6} is below 0.999 for the F16 staging path.");
    }

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
